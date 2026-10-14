# MTP (multi-token prediction) 投機的デコードの実装と実測

対象: Qwen3.8-Flash-Next (qwen4exp) / Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
実装: `src/qwen38_mtp.cuh` (新規) + `src/warodecode.cu` (CLI/JSON) +
`src/qwen38_forward.cuh` (アクセサ2つのみ)

## 0. 結論 (先に)

- MTP ブロック (blk.48) の forward と draft→verify ループを実装した。
- 受理判定は **trunk の argmax との厳密一致**なので、greedy 出力列は MTP の有無・
  draft の質によらず構造的に不変 (64/64 を維持)。
- **batch=1 のままだと高速化しない**。理由は §5 の会計で、verify の trunk ステップ数
  = 出力トークン数 が常に成り立つため。実測でも ON は OFF より遅い (§4)。
- 高速化には「M トークン同時 verify」の forward が必要で、必要なカーネルを §6 に列挙した。

## 1. MTP ブロックの構造 (GGUF 実データで確認)

GGUF のメタデータ: `qwen4exp.nextn_predict_layers = 1`,
`qwen4exp.block_count = 49` → trunk 48 層 + MTP 1 ブロック (blk.48)。

blk.48 は **通常の QSA 層と同じテンソル一式** (attn_q/k/v/output, q/k_norm, indexer.*,
ffn_*_exps, ffn_*_shexp, ffn_gate_inp, hc_attn_*, hc_ffn_*) を持ち、
`qwen4exp.attention.compress_ratios[48] = 4` から QSA (ratio 4) 層だと分かる。
差分 (nextn) は次の 6 本:

| tensor | shape | type | 役割 |
|---|---|---|---|
| blk.48.nextn.enorm.weight | [2560] | F32 | 次トークン埋め込みの RMSNorm gamma |
| blk.48.nextn.hnorm.weight | [10240] | F32 | trunk 広域残差 (hc×hidden) の RMSNorm gamma |
| blk.48.nextn.eh_proj.weight | [5120, 2560] | Q8_0 | concat(e_norm, h_norm) → block 入力 |
| blk.48.nextn.hc_head_norm.weight | [10240] | F32 | block 出力の HC head 用 norm |
| blk.48.nextn.hc_head_down.weight | [10240, 320] | Q8_0 | 〃 low-rank down |
| blk.48.nextn.hc_head_up.weight | [320, 10240] | Q8_0 | 〃 low-rank up |
| blk.48.nextn.shared_head_head.weight | [2560, 248320] | Q4_0 | MTP 専用 LM head |

forward (llama.cpp `src/models/qwen4exp.cpp` の graph_mtp と同一):

    h_wide = trunk 最終層の広域残差 (hc×hidden, F32)     ← 位置 p-1 の hidden
    e      = 位置 p のトークン埋め込み (hidden)
    h_norm = rms_norm_stream(h_wide) * hnorm            [hc][hidden]
    e_norm = rms_norm(e) * enorm                        [hidden]
    concat = [ e_norm ; h_norm_stream ] (stream ごと)   [2*hidden]
    res_hc = eh_proj(concat)                            [hc][hidden]
    res_hc = blk.48 の attention + MoE (位置 p で KV を書く)
    h_nextn = res_hc                                     ← 次の draft step へ鎖状に渡す
    logits  = shared_head_head( hc_head_mix(res_hc) )

つまり MTP は「trunk の hidden (p-1) と次トークン埋め込み (p) から p+1 を予測する」小型の
自己回帰 drafter で、自分の KV キャッシュと自分の hidden 鎖を持つ。

## 2. 実装

### 2.1 ファイル

- `src/qwen38_mtp.cuh` (新規, 約 700 行)
  - `Qwen38MtpWeights` / `load_qwen38_mtp_weights()`: blk.48 + nextn のロード。
  - Q4_0 GEMV カーネル (`q38_mtp_q4_0_gemv_kernel`): MTP head は Q4_0 で
    共有 lowbit GEMV セット (Q8_0/Q6_K/IQ4_NL/IQ2_S/IQ3_S) に入らないため専用実装。
  - `q38_mtp_hnorm_kernel` / `q38_mtp_enorm_kernel`: stream ごとの RMSNorm。
  - `Qwen38Mtp`: 重み・専用 QSA 状態・専用 expert cache・draft 1 ステップ forward。
  - `qwen38_run_speculative()`: draft→verify ループ。
- `src/warodecode.cu`: `--mtp` / `--mtp-draft K`、env 既定値、
  JSON 出力 (`mtp`, `mtp_calls`, `mtp_accepts`, `mtp_accept_rate`, `mtp_draft_ms` など)。
- `src/qwen38_forward.cuh`: `runtime()` / `state()` の 2 アクセサのみ追加 (step の挙動は不変)。

### 2.2 既存カーネルの再利用

MTP ブロックは trunk の QSA 層と同一構造なので、既存カーネルをそのまま使う:

| 段 | 使う関数 |
|---|---|
| HC mix / combine | `q38_mtp_hc_mix()` → `run_qwen38_hc_mix_f32` / `run_qwen38_hc_mix`, `run_qwen38_hc_combine` |
| QSA attention | `run_qwen38_qsa` (専用 `Qwen38QsaState`, ratio 4) |
| MoE | `run_qwen38_routed_moe_separate` + `run_qwen38_shared_expert` |
| head | `q38_mtp_hc_mix` + `q38_mtp_q4_0_gemv_kernel` + `q38_argmax_kernel` |

QSA の入口は `il < layout.trunk_layers` で MTP 層を弾くが、
`qwen38_qsa.cuh` (他担当が編集中) を触らずに済むよう、MTP 側で
`trunk_layers = total_blocks` に広げたレイアウト写しを渡している
(幾何フィールドは本体と同一、ガードだけを満たす)。

MTP の expert は Q8_0、trunk は IQ2_S/IQ4_NL で、`GpuExpertCache` は
初期化時にテンソル集合を固定するため、MTP 専用の slot pool をもう 1 つ持つ
(`WARODECODE_MTP_EXPERT_CACHE_MIB`, 既定 1024 MiB)。

### 2.3 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_MTP` | 0 | 1 で投機デコード有効 (`--mtp` と同じ) |
| `WARODECODE_MTP_DRAFT` | 2 | 1 ラウンドで提案する draft 数 k (1..8) |
| `WARODECODE_MTP_EXPERT_CACHE_MIB` | 1024 | MTP 専用 expert slot pool の大きさ |

### 2.4 draft→verify ループ

    pos = 0, fed = start_token, h_prev = nullptr
    while 出力数 < steps:
        k = min(MTP_DRAFT, 残り-1)
        drafts = MTP.draft(k, h_prev, fed, pos)   # 位置 pos..pos+k-1 で draft (KV もここで書く)
        for j in 0..k:
            out = trunk.step(j==0 ? fed : drafts[j-1], pos+j)
            emit(out)
            if j < k and out == drafts[j]: continue   # 受理
            break                                     # 不一致: out が本体の答え
        pos += 実行したステップ数; fed = 最後の出力; h_prev = trunk の広帯域残差

- 受理判定は **argmax の int 完全一致**。draft を trunk に食わせるのは「一致したとき」だけ
  なので、trunk が辿る (位置, 入力トークン) 列は通常デコードと完全に同じになる。
- MTP の KV は位置インデックスのキャッシュ。不一致で捨てる位置は次ラウンドが
  昇順に上書きし、QSA の block pool は「その block の最後の位置を書いたとき」だけ
  再計算されるので、明示的な rollback は不要 (詳細はコード内コメント)。
- 1 ラウンドの trunk ステップ数 = 出力トークン数。**これが batch=1 で高速化しない理由**。

## 3. parity

注意: ランナーの `matched_prefix` は参照 64 トークンと zip する実装で、食い違いが
無い場合は生成列の長さを返す。したがって 192 トークン実行の `matched_prefix = 192` は
「参照列の先頭 64 が一致し、その後も 192 まで生成した」を意味する。

MTP OFF / ON (k=2,3,4) のすべてで、参照 greedy 列 **64/64 一致**
(`token_hash = 0x6684298df825c691`)。
L4 / A100 の両方で一致。受理判定が argmax の厳密比較で、trunk が辿る
(位置, 入力トークン) 列が通常デコードと同一になるため、これは偶然ではなく構造的な結果。

## 4. 実測 (A100-SXM4-40GB)

条件: `--ctx 128 --generate 64`, trunk expert cache 20000 MiB,
MTP expert cache 3072 MiB (512 expert 全部が resident)。3 回中 1 回の測定。

| run | tok_s | seconds | parity | draft forwards | accepts | accept率 | draft合計 | draft 1回 |
|---|---|---|---|---|---|---|---|---|
| OFF | **3.722** | 17.195 | 64/64 | — | — | — | — | — |
| ON k=2 | 3.581 | 17.871 | 64/64 | 64 | 32 | 0.500 | 0.678 s | 10.6 ms |
| ON k=3 | 3.563 | 17.964 | 64/64 | 88 | 34 | 0.386 | 0.806 s | 9.2 ms |
| ON k=4 | 3.558 | 17.989 | 64/64 | 112 | 35 | 0.313 | 0.908 s | 8.1 ms |

- **OFF → ON k=2 は -3.8%、k=4 で -4.4%** (遅くなる)。差分はほぼ draft 時間そのもの
  (0.676 s ≈ 0.678 s) で、trunk 64 step の時間は OFF と同一。§5 の会計どおり、
  トークン数と trunk step 数が 1:1 のままであることが実測で確認できた。
- draft 1 回あたり 8〜11 ms (A100)。trunk 1 step (269 ms) の **3〜4%** と軽い。
- draft の質は悪くない: k=2 の内訳 (full_rounds=12, 受理合計 32/32 ラウンド) から
  **1 個目の受理率 62.5%、2 個目 60%**。k を増やすと 1 ラウンドあたりの
  平均受理数が伸びない (k=4 では 35/29 ラウンド) ため、k は 2〜3 が上限。

### L4 24GB

| run | tok_s | parity | draft forwards | accepts | accept率 | draft合計 |
|---|---|---|---|---|---|---|
| OFF (8000 MiB trunk cache) | **4.105** | 64/64 | — | — | — | — |
| ON k=2 | 3.941 | 64/64 | 64 | 32 | 0.500 | 0.704 s |
| ON k=3 | 3.851 | 64/64 | 88 | 34 | 0.386 | 1.071 s |
| ON k=4 | 3.888 | 64/64 | 112 | 35 | 0.313 | 0.972 s |

- **OFF → ON k=2 は -4.0%、k=3 で -6.2%、k=4 で -5.3%**。A100 と同じく遅くなる。
- 同一実行内の会計 (OFF 15.593 s + draft 0.704 s = 16.297 s vs ON 実測 16.241 s) が
  §5 の「trunk step 数は変わらない」を裏づける。
- 192 トークン (ctx 512) の長期測定: OFF 3.622 tok/s / ON k=2 4.093 tok/s と
  **逆向きに出た**。共有 VM のばらつき (他担当の計測でも同条件で ±30%) が
  MTP の数 % を覆い隠すため、この 1 回は判定に使えない。draft の定常コストは
  2.43 s / 180 forward = 13.5 ms/forward で、トークン数が増えると QSA draft の
  位置が増えて少し重くなる。

#### draft 1 回の中身 (L4, `WARODECODE_MTP_PROFILE=1`, k=2, 64 forwards)

| 段 | 合計 | 1 forward |
|---|---|---|
| RMSNorm + eh_proj | 0.0119 s (1.7%) | 0.19 ms |
| HC mix + QSA + combine | 0.0513 s (7.3%) | 0.80 ms |
| HC mix + **MoE** + shared + combine | **0.4935 s (70%)** | **7.71 ms** |
| hc_head + Q4_0 head GEMV + argmax | 0.1462 s (21%) | 2.28 ms |

MTP の MoE は expert 10 個を **1 個ずつ** (gate/up/silu/down/accum の 5 カーネル × 10)
回すため、転送量 (10 × 5.2 MiB = 52 MiB) に対して時間が 2 桁過大。
**律速は帯域ではなく 1 expert あたりのカーネル起動・同期レイテンシ**である。

注: L4 の初回 k=2 測定 (2.821 tok/s) は MTP expert 2.7 GB の初回ディスク読み込みを
含むため過大評価だった (draft 55 ms/step)。ページキャッシュが温まった再測定を上の表に使う。

## 5. なぜ batch=1 verify では速くならないか

通常デコードでは 1 trunk step = 1 トークン。投機デコードでも 1 ラウンドあたり

    trunk step 数 = 受理数 + 1 = そのラウンドで出すトークン数

なので、トークンあたりの trunk step 数は常に 1 で変わらない。draft の分だけ
(1 step あたり MTP forward × k) が純増する。llama.cpp の MTP 実装が速いのは、
verify を **M=k+1 トークンのバッチ** で 1 回の `llama_decode` に流し、
重み読み出しを M トークンで割り勘にするから。

A100 実測がこの会計をそのまま裏づけている:

    OFF     : 64 trunk step           = 17.195 s  (1 step 268.7 ms)
    ON k=2  : 64 trunk step + 64 draft = 17.195 + 0.678 = 17.873 s (実測 17.871 s)

受理率が 50% でも 60% でも、trunk step 数は 64 のまま動かない。
つまり現状の MTP は「正しいが必ず遅い」実装である。

## 6. 高速化に必要なもの (M トークン verify)

同じ層構造のまま M トークンを 1 パスで処理するには、各段の M 対応が要る:

1. HC mix/combine: 活性 M 行の GEMV→GEMM 化 (重み読み出しは 1 回)。
2. QSA: M 個の KV をまとめて store、M クエリの attention (block pool は M 個分の
   位置が揃った時点で 1 回)。indexer top-k も M 行分。
3. GDN (trunk 36 層): projection は M 行でまとめ、recurrence は位置順に逐次
   (state 更新だけ逐次なら重み読み出しの割り勘は成立)。
4. MoE: ルータ M 行 → 使用 expert の**和集合**を 1 回ロード → M 行の expert GEMM。
   ここが最大の利得 (現状 1 トークンあたり 10 expert を層ごとにロード)。
5. PLE: M 個の履歴に対する row index をまとめて pread (レイテンシの割り勘)。
6. head: M 行分の logits (受理判定に必要)。

加えて verify 側に「不一致位置での KV/状態 rollback」(llama.cpp の seq_rm 相当) が要る。

### trunk 1 step の内訳 (A100, `WARODECODE_PROFILE=1`, OFF, 64 tokens, ms/token)

| 段 | ms | 比率 |
|---|---|---|
| embed | 0.02 | 0% |
| PLE | 0.59 | 0% |
| **hc_mix (attn)** | **40.0** | 15% |
| **attention (QSA + GDN)** | **43.8** | 16% |
| hc_combine (attn) | 0.38 | 0% |
| **hc_mix (ffn)** | **40.1** | 15% |
| **MoE** | **141.1** | 52% |
| hc_combine (ffn) | 0.38 | 0% |
| head | 1.30 | 0% |
| 合計 | 268.9 | 100% |

重み転送量は 1 トークンあたり ~4.4 GB なので、269 ms は **有効 16 GB/s**
(A100 HBM は 1.9 TB/s) にしかならない。つまりこのエンジンは帯域ではなく
**カーネル起動・同期レイテンシ律速**である:

- HC mix 1 回の転送量は ~7 MB (0.03 ms 相当) なのに実測 0.83 ms/layer。
- MoE は 1 layer あたり expert 10 個 × 5 カーネル = 約 50 launch、実測 2.94 ms/layer。
- 1 トークンで 48 層 × 6 段 = 288 段、おおよそ 2,000〜2,500 カーネル。
  269 ms / 2,500 ≈ **107 µs/kernel** で、健全なカーネル (5〜20 µs) の 5〜20 倍。

この事実は MTP の見通しを両方向に効かせる:

- 悪い方向: draft (7.7 ms/step) の 70% は MoE の 50 launch で、MTP を足すと純増になる。
- 良い方向: **M トークンを 1 パスにするとカーネル数がほぼ M 分の 1** になる
  (HC mix は M 行 GEMM 1 回、MoE は expert 和集合で 1 回、attention も M 行分 1 回)。
  帯域律速なら利得は「M 分の 1」までだが、起動律速なのでそれに近い利得が期待できる。

### 実測値からの見込み (A100, k=2 → M=3 verify)

- trunk 1 step T1 = 269 ms、draft 1 step D = 8〜11 ms。
- k=2 の実測受理率から 1 verify バッチ (M=3) の期待出力 = 1 + 0.625 + 0.625*0.60 = 2.0 トークン。
- verify バッチのコストを c*T1 (c = 3 トークン分を 1 パスで処理したときの相対コスト) とすると

| c | 1 トークンあたり | 速度比 |
|---|---|---|
| 1.0 (重み読み出し律速) | 146 ms | 1.85x |
| 1.2 | 172 ms | 1.56x |
| 1.5 | 213 ms | 1.26x |
| 2.0 | 269 ms | 1.00x (利得なし) |

つまり **c < 2 が達成できれば 1.3〜1.9x** が見込める。課題は c を決めているものが
何かの特定で、A100/L4 の実測は「HBM 帯域律速ではない」(有効 17 GB/s, HBM は 300 GB/s 級)
ことを示している。したがって c を下げるには帯域の割り勘だけでなく、
**per-token のホスト側レイテンシ (expert staging の同期、PLE の pread、launch 数) の
削減**も必要になる可能性が高い。まず `WARODECODE_PROFILE=1` (他担当のステージ計測)
で 269 ms の内訳を確定させるのが次の一手。

## 7. 未検証リスク

1. **MTP の attention は QSA (学習時と同じ ratio 4) を使っている**。参照実装
   (llama.cpp `graph_mtp`) は「draft は dense で十分」として dense attention を
   使っており、acceptance を最大化する構成は未検証。本実装は既存 QSA カーネルを
   そのまま使う方が安全なので QSA を選んだ (受理率 62.5%/60% はこの構成の値)。
2. **高速化は未達**。batch=1 verify では原理的にプラスにならない (§5)。M トークン
   verify を実装しない限り「MTP で速くなる」とは言えない。
3. **VRAM 増**: MTP ON で +約 3.7 GB (head 0.64 GB + expert pool 3072 MiB)。
   L4 24GB で trunk cache 8000 MiB との併用は動くが、A100 40GB で cache 30000 MiB と
   併用すると足りない可能性が高い (今回は 20000 MiB で測定)。
4. **MTP expert cache のサイズ依存**: 初回測定 (3072 MiB, コールド) は 55 ms/step、
   ページキャッシュが温まった再測定は 8.7〜12 ms/step。64 トークンの短いベンチでは
   初回の 2.7 GB 読み込みが平均に乗るため、tok_s の比較は「同じ状態で連続実行」が前提。
5. **強制入力 / trace 経路では MTP は無効** (`--forced-input-tokens` や
   `--state-trace` と併用すると JSON の `"mtp"` は false になる)。
   64/64 parity の検証は通常 greedy 経路で行っている。
6. **未実施の測定**: L4 の k=3 は 1 回ビルド失敗 (他担当の一時的な破損) の後に再実行。
   長期 (192 トークン) の定常値は §4 の表を参照。MTP の dense-attention 版との
   A/B、expert 和集合を使う M=3 verify の試作は未着手。
7. **`mtp_calls` の二重計上を修正済み** (修正前の JSON は 2 倍の値。
   `mtp_draft_tokens` と `mtp_draft_seconds` は正しい)。

## 8. 再現手順

```bash
# MTP OFF / ON の A/B (Colab ランナー)。ラベルごとに engine JSON を
# issue1_artifacts/mtp_engine_<label>.json に保存する。
export PATH=/home/sahen/.local/bin:$PATH
bash tools/mtp_colab_bench.sh warpdecode-l4 89 \
  "off::WARODECODE_EXPERT_CACHE_MIB=8000" \
  "k2::WARODECODE_EXPERT_CACHE_MIB=8000,WARODECODE_MTP=1,WARODECODE_MTP_DRAFT=2,WARODECODE_MTP_EXPERT_CACHE_MIB=3072"

# A100 は WARODECODE_EXPERT_CACHE_MIB=20000 と arch 80 に読み替える。

# MTP ブロックのテンソル構成 (GGUF ヘッダのみ読む)
python3 tools/mtp_gguf_probe.py   # Colab VM 上で実行
```

測定済みの生データ:

- 64 トークン: issue1_artifacts/mtp_engine_mtpl_{off,k2,k3,k4}.json (L4),
  issue1_artifacts/mtp_engine_mtp2_off.json ほか (A100)
- trunk stage: issue1_artifacts/mtp_engine_mtpa_off_prof.json + /tmp の run_bg.log
- parity: issue1_artifacts/colab_run_mtpl_*.json (matched_prefix = 64)
