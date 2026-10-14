# Task I: M token 同時 verify（MoE バッチ化）の実装と実測

対象: Qwen3.8-Flash-Next (qwen4exp) / Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
実装: `src/qwen38_moe.cuh` (M-token バッチ MoE の入口), `src/qwen38_mtp.cuh`
(ブロック MoE を k ドラフト位置でまとめる), `tests/moe_batch_verify_*.{cu,cpp}`,
`tools/mtp_batch_ab_remote.sh`

## 0. 結論（先に）

- MoE を **M トークン分まとめて 1 パス**で実行するプリミティブを実装した
  (`Qwen38MoeTokenBatch`, `WARODECODE_MTP_BATCH_VERIFY`, `WARODECODE_MOE_BATCH_K`)。
- **ビット一致は構造的**。pair（(token, expert) 選択）ごとに単一 expert カーネルと同じ
  行デコードを通り、各トークンの重み付き和は自分の top-k 順に畳まれる。
  ホスト検証 (`bash tests/moe_batch_verify_host_check.sh`) は PASS。
- **MTP の trunk verify 側は、この構成では MoE をまとめられない**。理由は §3:
  1 トークンの trunk step は 48 層の逐次依存チェーンで、層 L の MoE 出力を
  層 L+1 の入力が消費する。M トークン分の同一層を同時に持つには
  「M トークンを流すバッチ forward」が要り、それは禁止ファイル
  (`src/qwen38_forward.cuh` / `src/warodecode.cu`) の中にある。
  したがって `WARODECODE_MTP_BATCH_VERIFY` が実際に効くのは
  **MTP ブロック自身の MoE**（ドラフト k 位置）になる。
- trunk MoE の M トークン和集合を 1 回にする設計は §6 に「何が必要か」として
  具体的に書き、プリミティブはそのまま使える形にしてある。

## 1. 実装した設計

### 1.1 M-token バッチ MoE（`src/qwen38_moe.cuh`）

1 回のバッチ pass を **pair**（= (token, expert, activation) の三つ組）の列にする。
token t の j 番目の選択は pair `t * k + j`。バッチバッファのレイアウト:

| バッファ | 形 | 中身 |
|---|---|---|
| PtrTable | [pairs] | `Q38MoeBatchPtrEntry`（gate/up/down ポインタと 2 バイト境界マスク） |
| gate_up | [2*pairs][ffn] | pair p の gate は 2p、up は 2p+1 |
| Act | [pairs][ffn] | SiLU(gate) * up |
| Q8Values / Q8Scales | [pairs][ffn] / [ffn/32] | down 用 Q8_0 活性 |
| Down | [pairs][hidden] | expert 出力（重み付き和の前） |
| Accum | [tokens][hidden] | トークンごとの routed 重み付き和 |
| Chart | pass 記述 | トークンごとの first_pair / pair_count / ids / weights |
| Ids / Weights / Logits | [tokens][k] / [tokens][n_experts] | ルータ選択の結果 |

カーネルは Task G の expert バッチ版の twin で、`grid.y` が expert ではなく
pair を選ぶ。重みの行デコードは同じ `iq2_s_q8_k_row_dp4a` /
`iq3_s_q8_k_row_dp4a` / `iq4_nl_q8_0_act_row_dp4a` を呼ぶので、行の値は
単一 expert カーネルとビット同一。重み付き和は
`a = add_rn(a, mul_rn(down[p][i], weights[p]))` を pair 順に回すだけで、
これは直列版の `q38_weighted_accum_f32_kernel` を k 回並べた順序と一致する。

ルータ選択はトークンごとに enqueue し（共有 scratch を書くため同時には走らせられない）、
ids/weights/logits の D2H をまとめて **1 回の cudaStreamSynchronize** で待つ。
直列版はトークンごとに同期するので、ここでも同期回数が M 分の 1 になる。

### 1.2 MTP ブロックの MoE を k 位置でまとめる（`src/qwen38_mtp.cuh`）

`forward_one()` を「prepare（norm/eh_proj/attn mix/QSA）→ MoE → finish（ffn combine/head）」
の 3 段に割り、ドラフト 1 ラウンドの k 位置を次の順で通す:

    for i in 0..k-1:
        prepare(i)            # 位置 i の QSA KV を書き、attn まで終える
        mix(i)                # FFN 入力（バッチ活性）
    block_moe(slots, k)       # ★ ルータ選択は位置ごと、expert 実行は 1 パス
    for i in 0..k-1:
        finish(i)             # ffn combine + head

- 各スロットは自分の広域残差を退避する。k 位置分を保持するため、combine の前に
  自分の残差を復元する（直列版は 1 位置ずつ combine していた）。
- 直列経路 `forward_one()` は同じ 3 段を 1 スロットで回す形で残してあり、
  既定では従来どおりの経路を通る（ビット同一）。
- バッチ経路は `WARODECODE_MTP_BATCH_VERIFY=1` かつ k>1 のときだけ選ばれる。
  選ばれた/選ばれなかった理由を stderr に 1 行だけ出す（A/B ログで判別できる）。

### 1.3 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_MTP_BATCH_VERIFY` | 0 | 1 で MTP ブロックの MoE を k 位置まとめて実行 |
| `WARODECODE_MOE_BATCH_K` | 4 | 1 パスで扱う expert バッチ上限（1..32） |
| `WARODECODE_MTP_DRAFT` | 2 | ドラフト長 k（= バッチのトークン数） |
| `WARODECODE_MTP_EXPERT_CACHE_MIB` | 1024 | MTP 専用 expert プール（バッチ経路は 64 slot/top-k 以上必要） |

## 2. bit 一致の検証

### 2.1 ホスト（no nvcc）

    bash tests/moe_batch_verify_host_check.sh

検証するのは pair 写像（pair = token * k + j）と重み付き和の順序。
負のコントロールとして (a) FMA 収縮、(b) pair→token 写像の取り違え を入れ、
どちらも比較で検出できることを確認する。

結果:

    Task I host check (tokens=4, top-k=10, hidden=2560)
      pair index decomposes to (token, top-k slot)               ok
      ordered accumulation is bit-identical to M serial passes   ok
      negative control: an FMA contraction is visible            ok
      negative control: a wrong pair -> token map is visible     ok
    PASS (0 failures)

### 2.2 デバイス（実 GPU）

    nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 -Isrc \
         tests/moe_batch_verify_bench.cu -o /content/moe_batch_verify_bench
    /content/moe_batch_verify_bench

M = 1..4 について、直列（トークンごとに `run_qwen38_routed_moe_separate`）の出力と
バッチの出力を **バイト比較** する。負のコントロールは活性の入れ替え。

（実測は §4 に追記）

## 3. なぜ trunk の MoE はこの構成でまとめられないか

- `Qwen38Decoder::step(token, pos)` は 1 トークン分の 48 層を逐次に回す。
  層 L の MoE 出力は HC combine で残差に混ざり、層 L+1 の入力になる。
  つまり **同一層を M トークン分同時に持つには、層ループの外側にトークン次元が要る**。
- それはバッチ forward そのもので、実装は禁止ファイル
  `src/qwen38_forward.cuh`（約 1500 行の step 本体）の中にある。
  ここを触らずに外から M トークン分の層 L 入力を作る方法は無い
  （1 トークンずつ step を呼ぶと、層 L の MoE はトークンごとに 1 回発生する）。
- A/B 用に `WARODECODE_MTP_BATCH_VERIFY` を verify ループに張る案も検討したが、
  verify の M トークンは step() の出力が次トークンの入力になるため同時に存在しない。
- したがって本タスクで実際に MoE をバッチできるのは **MTP ブロックの MoE**
  （ドラフト k 位置）だけになる。プリミティブ側は M トークン一般で動くので、
  forward 側に「M トークンを流すバッチ層ループ」が入れば、そのまま呼び出せる。

## 4. 実測

### 4.1 デバイス bit 一致 + 1 層のコスト

（`tests/moe_batch_verify_bench.cu` の出力を追記）

### 4.2 実機 A/B（L4 24GB, `tools/mtp_batch_ab_remote.sh`）

条件: `--ctx 128 --generate 64`, trunk expert cache 8000 MiB, MTP expert cache 3072 MiB。
parity は参照 64 トークン列との一致。

（結果を追記）

## 5. 未検証リスク

1. **trunk MoE の M トークン化は未実装**（§3）。本タスクの測定は MTP ブロックの
   MoE バッチ化に閉じている。trunk 側の利得は forward のバッチ化待ち。
2. `Qwen38MoeTokenBatch` の対応形は separate gate/up + IQ4_NL down（MTP ブロックの形）。
   fused gate_up は未対応（`compatible()` が false を返す）。
3. バッチ経路の expert 取得は pair 数分の slot を先に確保する。expert cache が
   64 slot/top-k を下回ると直列に落ちる（Task G と同じ不変条件）。
4. MTP ブロックの QSA は位置ごとに逐次のまま。バッチ化するなら QSA/KV の
   M 位置対応が別途必要。
5. バッチ経路の expert 取得は「pair 数分の slot を先に確保 → 起動前に 1 回検証」。
   `resolved(true)` が、同じ slot を別の expert が使っている pair を見つけたら
   直列に落ちる（slot 境界の同一性チェック）。
6. バッチ経路の活性は __half バッファを F32 として読む alias を使う
   （F16 表現可能な値なので widening は起きない、という前提）。F32-acitvation を
   本当に F32 で持つ経路（`_f32_x`）に切り替えると、この前提は不要になる。
7. プロファイルの `draft_steps` はバッチ経路では k を加算する（1 forward = k 位置）。
   既存の `mtp_draft_ms` の解釈が直列版とずれるので、比較時は注意。
