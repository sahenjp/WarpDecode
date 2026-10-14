# Task G: MoE expert 単位カーネル統合 (WARODECODE_MOE_FUSED_GRID)

対象: `src/qwen38_moe.cuh` のルーティング済み expert 実行と、その下請けの
`src/lowbit.cuh` の量子化 GEMV。
検証: device ハーネス (L4) + CPU ホストチェック + L4 実機 A/B。parity は 64/64 を維持。

---

## 0. 結論

1. **1 層あたり 60 launch → 5 launch** に統合した (48 層で 2880 → 240 launch/token)。
   カーネル単体 (L4, top-k=10, hidden 2560 / expert FFN 640) では layer あたり
   **360 us → 180 us (約 2.0 倍)**。
2. **数値は完全に同一**。IQ2_S / IQ3_S / IQ4_NL の GEMV、SiLU、Q8_0 活性量子化、
   加重和のすべてで batched と per-expert の出力が bit 一致することを device 上で
   確認した (`tests/moe_fused_grid_bench.cu`)。production の helper 経由
   (`q38_moe_upload_expert_ptr_table` + `q38_moe_run_batched_experts`) でも
   per-expert ループと bit 一致する。
3. 実機 A/B (L4, expert cache 既定 1536 MiB, `--bench-repeat 3`):
   **parity は ON/OFF 両方で 64/64**、tok_s は **3.428 → 3.447 (+0.55%)**。
   起動ログで batched 経路が実際に選択されていることも確認した。
4. **端到端の伸びが小さい理由も計測で判明した**。既存の stage profiler
   (`WARODECODE_PROFILE=1`) の内訳は 1 token あたり
   **moe 192.6 ms / attn 43.9 / hc_mix_attn 29.0 / hc_mix_ffn 28.9 / head 3.0 / 合計 299.4 ms**
   で、**MoE 区間だけでトークンの 64%** を占める。しかしそのうちルーティング済み
   expert の GEMV は 17.3 ms (MoE 区間の 9%、トークン全体の 5.8%) しかない。
   残り約 175 ms は expert cache のステージング転送 (host → device) が支配している。
   **launch 削減で動かせるのは GEMV 17.3 ms とその起動分だけ**なので、端到端 +0.5% は
   上限に近い。以前の「次に効くのは expert 単位の統合」は GEMV カーネル時間だけを
   見た見積もりで、トークン全体の支配要因ではなかった。

---

## 1. 統合の形

### 1.1 現状 (WARODECODE_MOE_FUSED_GRID=0)

1 層 = 選択された 10 expert について、expert ごとに以下を順に launch する。

| # | カーネル | 形状 | L4 実測 (1 launch) |
|---|---|---|---|
| 1 | IQ2_S gate GEMV | 640x2560 | 7.25 us |
| 2 | IQ2_S up GEMV | 640x2560 | 7.25 us |
| 3 | SiLU(gate) (in-place) | 640 要素 | < 2 us |
| 4 | SiLU*up (mul) | 640 要素 | < 2 us |
| 5 | Q8_0 活性量子化 | 640 要素 | < 2 us |
| 6 | IQ4_NL down GEMV | 2560x640 | 14.29 us |
| 7 | 加重和 (F32) | 2560 要素 | < 2 us |

-> **60 launch / 層、2880 launch / token**。実行区間は約 360 us / 層 = 17.3 ms/token。

### 1.2 統合後 (WARODECODE_MOE_FUSED_GRID=1)

| # | カーネル | グリッド | 実測 (1 launch) |
|---|---|---|---|
| 1 | gate+up GEMV (IQ2_S or IQ3_S) | `grid.x` = 行ブロック, `grid.y` = 2k (=20) | 28.9 us |
| 2 | SiLU*up (expert ごと) | `grid.y` = k | < 3 us |
| 3 | Q8_0 活性量子化 (expert ごと) | `grid.y` = k | < 3 us |
| 4 | down GEMV (IQ4_NL) | `grid.x` = 行ブロック, `grid.y` = k | 114.1 us |
| 5 | 加重和 (j 昇順ループ) | `grid.x` = 行ブロック | < 3 us |

-> **5 launch / 層、240 launch / token**。

設計のポイント:

- **grid.y = expert**。expert の重みポインタは 768 B の `Q38MoeExpertPtrTable` に
  まとめ、層ごとに 1 回だけ pinned host から device へコピーする
  (`q38_moe_upload_expert_ptr_table`)。カーネルは `w_experts[blockIdx.y]` を読む。
- **活性の共有範囲が projection で違う**。gate/up は全 expert が同じ Q8_K 活性を
  共有するので 1 本のポインタでよい。down は expert ごとの SiLU*up 出力を食うので、
  kernel が `blockIdx.y` で活性のスライスも選ぶ
  (`x_qs[e * cols]`, `x_scales[e * (cols / 32)]`)。
- 中間バッファは `[2k][ffn]` / `[k][ffn]` / `[k][hidden]` の連結配列にして、
  separate レイアウト (gate を 2j、up を 2j+1) と fused gate_up レイアウト
  (`[j][2*ffn]`) が同じ平坦 index になるようにした。この不変条件は
  `tests/moe_fused_grid_host_check.sh` が全数検査する。

### 1.3 数値が変わらない根拠

- **GEMV**: 3 種の DP4A カーネルを「1 行をデコードする `__device__` ルーチン」+
  薄いカーネルに分割し、単一 expert 版と batched 版が **同じルーチンを呼ぶ** ようにした。
  行の担当が変わるだけで命令列も加算順も同一。
- **SiLU / Q8_0 量子化**: 既存カーネルの本体を逐語コピーし、expert オフセットだけを足した。
  SiLU は「`fdiv_rn` で丸めた値をレジスタに保持してから積む」という既存 2 カーネルと
  同じ順序を保つ。
- **加重和**: 逐次 launch は `accum[i] = add_rn(accum[i], mul_rn(x_j[i], w_j))` を
  j 昇順に k 回適用する。batched 版は 1 スレッドが同じ式を j 昇順にループするだけで、
  per-element の系列は完全に同じ (FMA 収縮もしない)。
- ルータ・gate GEMV・共有 expert・HC/attention には一切触れていない。

---

## 2. 実装

| ファイル | 追加内容 |
|---|---|
| `src/lowbit.cuh` | 行ルーチン `iq2_s_q8_k_row_dp4a` / `iq3_s_q8_k_row_dp4a` / `iq4_nl_q8_0_act_row_dp4a` と、その batched カーネル 3 種 + launcher 3 種。既存の single-expert カーネルは同じルーチンを呼ぶ薄いラッパーになった (外部 I/F は不変)。 |
| `src/qwen38_moe.cuh` | `Q38MoeExpertPtrTable`、MoE 用 batched カーネル 4 種 (SiLU*up / Q8_0 量子化 / 加重和 / trace コピー)、batched ランナー 2 種 (separate / fused)、scratch の作業バッファ、`WARODECODE_MOE_FUSED_GRID` スイッチ。 |
| `tests/moe_fused_grid_bench.cu` | device ハーネス (bit 比較 + 速度計測)。 |
| `tests/moe_fused_grid_host_test.cpp` / `.sh` | CPU のみの bit 比較 (加算順序・index 対応)。 |
| `tests/qwen38_cuda_syntax_check.sh` | ハーネスも g++ で構文検査するよう拡張、`dim3` 2 引数コンストラクタを stub に追加。 |
| `tests/host_checks.sh` | 新しいホストチェックを suite に追加。 |

スイッチ:

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_MOE_FUSED_GRID` | **OFF** | 1 層分の expert を 5 launch の batched grid で実行する |

既定を OFF にしたのは、A/B を回す前の状態を変えないため。ON にすると起動時に
stderr へ 1 行 (`MoE expert-batched grid taken ...`) を出し、条件が揃わず
フォールバックした場合も理由を 1 行だけ出す。フォールバック条件は次の 4 つ:

1. `WARODECODE_MOE_FUSED_GRID` が OFF、
2. DP4A カーネルが使えない (sm < 80)。batched GEMV には参照実装のフォールバックが無い、
3. `WARODECODE_MOE_F32_MID=0` のレガシー F16 経路 (d_down() 経由で加算するため)、
4. expert cache のスロット数が `64 * top-k` 未満 (下記 §3 の排他条件)。

---

## 3. expert cache のスロット排他 (安全性)

batched 経路は **launch の前に k expert 分のスロットをまとめて取得**する。逐次ループなら
「取得 → カーネル投入 → 次の取得」の間にスロットが retire されるが、batched では
取得だけが先に並ぶため、後続の acquire が先行分のスロットを上書きしないことが前提になる。

`GpuExpertCache::choose_slot()` は「未使用スロット → 最も古い last_use」の順に選ぶので、
**同じ層で連続して取得した k スロットは last_use が最大の k 個**になり、層あたりの
スロット数が k 以上なら eviction の対象にならない。そこで
`cache.slot_count() >= 64 * k` を条件にした (49 層モデルでは層あたり 64 本を確保すれば
十分。実測構成は 3600 スロット)。条件を満たさない小さな cache では逐次ループに戻る。

補足: 転送は copy stream 側 (`stage_.submit`) で行われ、`stage_.finish()` が
decode stream を done イベントで待たせるので、batched カーネルは全スロットの
ステージング完了後に走る。1 層分のポインタテーブルの H2D コピーも全 acquire の後に
投入されるので順序は保たれる。

---

## 4. 検証

### 4.1 device ハーネス (`tests/moe_fused_grid_bench.cu`, L4 / sm_89)

```
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 -Isrc tests/moe_fused_grid_bench.cu -o /content/moe_fused_bench
```

bit 比較 (memcmp):

| 比較 | 結果 |
|---|---|
| IQ2_S gate/up GEMV 10 expert (batched vs 10 launch) | bit-identical (25600 B) |
| IQ4_NL down GEMV 10 expert (expert ごとの活性) | bit-identical (102400 B) |
| SiLU(gate)*up 10 expert | bit-identical (25600 B) |
| Q8_0 活性量子化 (values / scales) | bit-identical (6400 B / 400 B) |
| 加重和 10 expert | bit-identical (10240 B) |
| production helper 経由 (`q38_moe_upload_expert_ptr_table` + `q38_moe_run_batched_experts`) vs per-expert ランナー列 | bit-identical (10240 B) |

PASS (0 mismatches)。

### 4.2 CPU ホストチェック (`bash tests/moe_fused_grid_host_check.sh`)

nvcc 無しで、加算順序と index 対応だけを検査する。

- 加重和: 逐次 k launch と j 昇順ループが bit 一致。
- 負のコントロール: FMA 収縮・expert 順序の反転を検出できる (テスト自体の感度確認)。
- separate / fused の gate/up index 式が一致し、batched バッファ内でスライスが衝突しない。

PASS (0 failures)。

### 4.3 実機 A/B (warpdecode-l4b, L4 24GB, expert cache 8000 MiB)

```
env WARODECODE_MOE_FUSED_GRID=<0|1> build-sm89/warodecode --model ... --ctx 128 \
    --device 0 --start-token 1 --no-graph --no-autotune --generate 64 --bench-repeat 3
```

| 実行 | tok_s | 秒 (64 tok) | parity |
|---|---|---|---|
| OFF 1 回目 | 3.425 | 18.684 | 64/64 |
| ON 1 回目 | 3.447 | 18.568 | 64/64 |
| ON 2 回目 | 3.447 | 18.568 | 64/64 |
| OFF 2 回目 | 3.431 | 18.654 | 64/64 |
| prof OFF (`WARODECODE_PROFILE=1`) | 3.385 | 18.906 | 64/64 |
| prof ON (`WARODECODE_PROFILE=1`) | 3.433 | 18.645 | 64/64 |

- **tok_s: OFF 平均 3.428 → ON 3.447 = +0.55%** (parity 64/64 は全 6 実行で維持)。
- ON 実行の stderr に
  `warodecode: MoE expert-batched grid taken (top-k=10, cache slots=691, 5 launches/layer)`
  が出ており、batched 経路が実際に使われている (フォールバックしていない)。
- stage profiler の内訳 (ms/token):

  | stage | OFF | ON |
  |---|---|---|
  | embed | 0.016 | 0.016 |
  | ple | 0.745 | 0.746 |
  | hc_mix_attn | 29.00 | 28.96 |
  | attn | 43.86 | 43.87 |
  | hc_combine_attn | 0.282 | 0.282 |
  | hc_mix_ffn | 28.95 | 28.91 |
  | **moe** | **192.57** | **188.85** |
  | hc_combine_ffn | 0.255 | 0.255 |
  | head | 3.00 | 3.01 |
  | total | 299.42 | 295.65 |

  MoE 区間は **-1.9%**。カーネル単体で測った 9 ms/token の削減のうち、端到端では
  約 3.7 ms しか現れない。残りは expert cache の転送と重なって隠れている。
- 参考: この A/B は expert cache 既定 (1536 MiB, 691 スロット) で回した。
  8000 MiB 構成 (phases ベンチの 3.95 tok/s) では転送が減るぶん GEMV の比率が
  上がるため、統合の相対効果はもう少し大きくなる見込み (未計測)。

---

## 5. 途中で見つけて直した不具合 (記録)

1. **pointer table の host/device 取り違え**。最初の実装は host 側
   `Q38MoeExpertPtrTable` の配列をカーネルに渡していた (device 上の構造体を読む設計
   なのに host アドレスを渡していた)。エンジンは最初の実行で落ちた。
   §4.1 の "production helper" 比較を追加したのはこの再発防止のため。
2. **down 射影の活性共有**。batched down GEMV が全 expert で 1 本の活性
   (expert 0 の SiLU*up) を読んでいた。gate/up は全 expert 共有で正しいが、down は
   expert ごとに違う。エンジンの parity が 0/64 になったことで発覚し、expert ごとの
   スライスに修正した。初版のハーネスは down も「共有活性」で比較していたため
   このバグを見逃していた (現在は expert ごとの活性で比較する)。

---

## 6. 未検証リスクと次の一手

1. **絶対値の比較は汚染されている**。作業ツリーの `src/expert_cache.cuh` は他タスク
   (Task H: pread / pinned ring) が同時編集しており、A/B の実行中もビルドが 1 回
   失敗した (呼び出し引数の不一致)。tok_s の絶対値は他タスクの変更に左右されるので、
   同一実行内の ON/OFF 相対比較だけを根拠にしている。
2. **fused gate_up 経路 (rt.fused_experts=true) は実機未検証**。今回のモデルは
   separate レイアウト (`fused=false`) なので、fused 用の batched ランナーは
   コード上の対称性のみで、device ハーネスの対象外。
3. **CaptureSelectedOutput (trace) 経路**は batched 版を実装してあるが、
   `--state-trace` / `--extended-parity-trace` 付きの実行は今回の A/B では回していない。
4. **sm < 80 では常に逐次ループ**。batched GEMV に参照実装フォールバックを用意していない。
5. MTP (投機デコード) 経路との組み合わせは未検討。MTP 側が同じ
   `run_qwen38_routed_moe_*` を呼ぶならそのまま恩恵を受けるはずだが、未確認。
6. **既定は OFF のまま**にした。+0.55% は再現性のある正の効果で、parity も
   bit 一致も確認済みなので、ON にしても安全である (1 行の変更)。ただし現時点の
   効果は小さく、他タスクの作業ツリーに影響を与えないことを優先した。
   次のいずれかで ON に切り替えるのがよい:
   (a) expert cache の転送経路 (Task C/H) が速くなり launch の相対重みが上がったとき、
   (b) 既定 cache を 8000 MiB にして GEMV 比率が上がったとき。
7. **次の一手は MoE 区間の内訳そのもの**。profile で moe = 192.6 ms/token と分かったが、
   ルーティング済み GEMV は 17.3 ms で、残り約 175 ms は expert cache の
   ステージング転送が支配している (1 token あたり 480 expert、大半が miss)。
   ここを削るのが最大の効果を持つ (pinned ring / pread 経路 = Task H の担当)。
   なお `WARODECODE_PROFILE=1` の出力 JSON は機械可読でない
   (`json.loads` が 1336 文字目で失敗する) ため、今回の数値は正規表現で
   抜き出した。profile の JSON を直すと今後の解析が楽になる。
