# Task N: CPU 側 MoE 経路 — 実装・bit 検証・A/B

Task L の A/B（[LLAMACPP_AB.md](LLAMACPP_AB.md)）で分かった敗因は単純でした。llama.cpp は
MoE の expert を **mmap 越しに CPU で計算する**ためトークンごとの重み転送が発生せず、本エンジンは
expert を VRAM へ **転送してから** GPU で計算するため負けます
（A100 40GB: 本エンジン 6.15 tok/s vs llama.cpp 12.54〜23.06 tok/s）。

Task N はその戦略をそのまま取り込みます。**routed expert だけを CPU へ移し、attention / HC mix /
GDN / QSA / head / router / shared expert は GPU のまま**残すのが本エンジン側の差分です。

- 対象モデル: `Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf`（85,923,705,344 bytes、experts は
  gate/up = IQ2_S（blk.2 のみ IQ3_S）、down = IQ4_NL、separate gate/up レイアウト）
- セッション: Colab `warpdecode-a100`（A100-SXM4-40GB / 12 vCPU / CUDA 13.0）
- 参照: pinned llama.cpp `6fcaa16f4b360649933a54d1f91ad40ed35c0e11`（オラクルと A/B の両方）
- parity の基準: `issue1_artifacts/reference_tokens_64.json`（pinned CPU オラクルの 64 トークン、
  SHA-256 固定）

---

## 1. 実装

| ファイル | 役割 |
|---|---|
| `src/qwen38_cpu_moe.cuh` | CUDA 非依存の本体。AVX2 GEMV（IQ2_S / IQ3_S / IQ4_NL / Q8_0）、Q8_K / Q8_0 活性量子化、F16 変換、ggml AVX2 超越関数、スレッドプール、レイヤーランナー |
| `src/qwen38_cpu_moe_grid.inc` | ggml の `iq2s_grid` / `iq3s_grid` の複製（CUDA 側テーブルと一致することをテストで確認） |
| `src/qwen38_cpu_moe_runtime.cuh` | CUDA ドライバ。GGUF mmap 上の expert 重みビュー、層ごとの hidden F32 / expert id / weight の D2H と routed 和の H2D |
| `src/qwen38_runtime.cuh` / `src/qwen38_forward.cuh` / `src/warodecode.cu` | 分岐と配線（既存 GPU 経路は不変）、起動ログ |
| `CMakeLists.txt` | engine のホストコンパイルに `-Xcompiler=-mavx2,-mfma` を追加（未指定だとスカラー双対へ無言でフォールバックするため） |
| `tests/qwen38_cpu_moe_host_test.cpp` | AVX2 カーネル = スカラー双対 = int64 厳密ドット、F16 変換、テーブル一致、スレッド数不変、累積の丸め規則 |
| `tests/qwen38_cpu_moe_oracle_test.cpp` | **pinned llama.cpp の AVX2 カーネル本文をそのまま取り込んで** bit 比較 |
| `tests/cpu_moe_gpu_ref_test.cu` | 同一入力で CUDA DP4A カーネルと CPU カーネルが同じ float を返すことの確認（nvcc、DP4A on/off） |
| `tools/cpu_moe_ab.sh` | llama.cpp と交互に回す A/B（K=0/16/24 vs CPU MoE 層数スイープ） |

### カーネル

| 重み | 活性 | 方式 |
|---|---|---|
| IQ2_S | Q8_K | 256 値スーパーブロックごとに 1 FMA。チャンク 0..3 が scale の下ニブル、4..7 が上ニブル |
| IQ3_S | Q8_K | 同上。32 値ブロックごとに 1 scale（偶数ブロック下ニブル / 奇数ブロック上ニブル）、閉じ係数なし |
| IQ4_NL | Q8_0 | 32 値ブロックごとに 1 FMA。偶数/奇数ブロックを別アキュムレータに分け、最後に `hsum_float_8` の木で合流 |
| Q8_0 | Q8_0 | 32 値ブロックごとに 1 FMA |

整数側は完全に厳密です。IQ2_S / IQ3_S のコードブックは |q| ≤ 43、IQ4_NL は ≤ 127、活性は 7bit
なので int16 の部分和が飽和せず int32 の総和も溢れません。したがって
`_mm256_maddubs_epi16` + `_mm256_madd_epi16` は素の整数ドット積と等価です。float 側は
ggml の**丸め順そのまま**で、FMA の縮約も並べ替えもしません。

routed 和の累積だけは **mul → add の 2 回丸め**です（ggml の `ggml_mul` + `ggml_add`、
CUDA 側の `q38_weighted_accum_f32_kernel` と同じ）。ここを 1 回の FMA にすると expert ごとに
1 ulp ずれ、22 トークン目で greedy が反転します（実際に踏みました）。

### 配置ポリシー（環境変数）

| 変数 | 意味 |
|---|---|
| `WARODECODE_CPU_MOE=1` | 全 trunk 層の routed expert を CPU で計算 |
| `WARODECODE_CPU_MOE_LAYERS=N` | 先頭 N 層だけ CPU（残りは GPU キャッシュ）。llama.cpp の `--keep-moe-gpu-last 48-N` に対応 |
| `WARODECODE_CPU_MOE_THREADS=n` | スレッドプール幅（既定 min(nproc, 16)。A100 は 12） |

shared expert は CPU に移しません。llama.cpp の `--n-cpu-moe`（tensor override）も
`blk.<n>.ffn_.*_exps` だけを CPU に置くため、条件が揃います。

CPU MoE は `WARODECODE_MOE_F32_MID=1`（既定）と F32 HC mix 行を要求します。F16 に丸めた
活性を渡すと ggml の参照からずれるためで、満たさない場合は起動時に明示的に落ちます。

### データの流れ（1 層あたり）

```
router (GPU)                     ← softmax/top-k はそのまま
  ↓ expert id / weight (k=10)    56 B   D2H
hc_f32_scratch.mixed()          10 KB   D2H  ← F32 活性（丸めなし）
CPU: Q8_K 量子化 → 10 expert を並列 GEMV → routed 和（昇順 mul→add）
  ↓ routed 和 (F32)             10 KB   H2D → st.moe.d_accum()
shared expert + q38_finish_moe_f32 (GPU, 従来どおり)
```

expert の重みは 1 バイトも VRAM へ動きません。mmap のページを CPU が直接読みます。

---

## 2. 直した不具合（すべて実装側。テスト側の誤りも 2 件）

引き継いだ時点のホストテストは 69 checks / 51 failures でした。失敗は 4 系統に分かれます。

1. **`f16_to_f32` の非正規化数が常に半分**（実装バグ）。指数を `127 - 15 - shift` としていた
   ため、block scale が 6.1e-5 未満（f16 非正規化）のブロックがすべて 1/2 になっていました。正しくは
   `127 - 14 - shift`。テストの `_Float16` 比較 200,000 ケースで不一致 6,245 件 → 0 件。
2. **IQ2_S のスケール適用と符号展開、IQ3_S の符号展開が参照と別物**（実装バグ）。IQ2_S は 32 値
   ブロックごとに scale バイトが 1 個で、前半 16 値（チャンク 0..3）が下ニブル、後半が上ニブルです。
   引き継いだ実装は scale を `scales[ib32 >> 1]` から取り、16 個の int16 をインターリーブして
   `madd_epi16` に渡していました。IQ3_S は 32bit 符号語の上位 2 バイトを捨てていたため、値
   16..31 の符号が反転しませんでした。テスト側の `exact_iq2_s_int_dot` も同じ誤ったスケール
   規則を使っていたので、両方を ggml の規則に揃えました（int64 厳密ドットはレーン単位に変更）。
3. **`quantize_q8_0` の SoA 署名と AoS 呼び出しの不一致**（実装バグ）。量子化関数は
   `int8_t * qs, uint16_t * scales` の分離配列を受け取る設計でしたが、呼び出し側（テストと
   `run_expert_serial`）は ggml の AoS `q8_0` ブロックを渡していました。書き込みが互いを
   上書きし、下流の活性が完全に壊れていました。関数を AoS（`q8_0 * out`）に変更しました。
   テストの「q8_0 block scale」検査も `d0 == 3.0f/127.0f` という fp16 では不可能な厳密一致を
   要求していたので、保存ビット（`f32_to_f16_bits(3.0f/127.0f)`）の比較に直しました。
4. **routed 和の累積が FMA**（実装バグ）。`std::fma(weight, expert, accum)` は 1 回丸めですが、
   ggml と CUDA 側（`q38_weighted_accum_f32_kernel` = `__fmul_rn` + `__fadd_rn`）は
   2 回丸めです。ホストで FMA に縮約されないよう `mul_rn` / `add_rn`
   （`_mm_mul_ss` / `_mm_add_ss`）を追加しました。これが実機 parity 22/64 の直接原因で、
   修正後は 64/64 になりました。
5. **gate+up 融合カーネルのスケール/符号レーン取り違え**（実装バグ、後から入った最適化）。
   `dot_gate_up_iq2_s_row` が 16 個の int16 スケールを `[ls1(0..3), ls2(0..3), ls1(4..7), ...]`
   の順に詰めて `scales16 + 8*((ib32>>1)&1)` を `madd_epi16` に渡していました。`madd_epi16` は
   32bit レーンごとに隣接 int16 を掛けるので、正しくは「その 32 値ブロックの ls1 を 4 レーン +
   ls2 を 4 レーン」でなければならず、混ざったベクトルは ggml と一致しません。符号語も 4 バイトを
   2 つの 32 値ブロックに分割して適用していました。`iq2_s_scale_vector(scales[b])` と
   `load_i32(signs + b*4)` を使う実装に戻し、ホストテストに
   「fused gate+up == 2 本の independent row（bit 一致）」の検査を追加しました（これが無いと
   融合最適化の回帰を検出できません）。

---

## 3. bit 一致の検証（三段構え）

1. **CPU 内部**（`bash tests/qwen38_cpu_moe_host_check.sh`）
   **82 checks / 0 failures**。AVX2 カーネル = スカラー双対（同じレーン構成を整数だけ 1 個ずつ足した
   再構成）= int64 厳密ドット。負の対照（expert 重みの入れ替え、1 vs 8 スレッドの bit 一致、累積が
   mul→add であることの確認、融合 gate+up が独立 2 行と一致することの確認）を含みます。
2. **llama.cpp オラクル**（同スクリプトの 2 本目、`tests/qwen38_cpu_moe_oracle_test.cpp`）
   **96 checks / 0 failures**。pinned commit の `ggml-cpu/arch/x86/quants.c` の AVX2 カーネル
   （IQ2_S / IQ3_S / IQ4_NL / Q8_0 とそのヘルパ、マスク定数、scale shuffle テーブル）をそのまま
   取り込み、同じ重み・同じ活性で bit 比較します。列数 256/512/640/768/224/2560/5120、
   IQ4_NL の奇数ブロック数（スカラー末尾経路）も含みます。
3. **GPU 参照**（`tests/cpu_moe_gpu_ref_test.cu`、要 nvcc）
   **9 checks × 2（DP4A on/off）= 0 failures**。実 shape（hidden 2560 / ffn 640、3 expert）で
   CUDA の `launch_iq2_s_gemv_q8_k_f32` / `launch_iq3_s_gemv_q8_k_f32` /
   `launch_iq4_nl_gemv_q8_0_f32` と CPU カーネルが同じ float を返すことを行単位で確認します。

---

## 4. 実機 parity（A100）

`WARODECODE_CPU_MOE=1`（全 48 層 CPU、12 スレッド、expert cache 8000 MiB）で生成 64 トークン:

| 実行 | 一致 | 備考 |
|---|---|---|
| CPU MoE 全層（累積 FMA 版） | **22/64** | 22 トークン目で反転。原因は上記 4 |
| CPU MoE 全層（累積 mul→add 版） | **64/64** | `issue1_artifacts/colab_run_cpu_moe_parity2_a100.json` |

起動ログで `cpu-moe=on layers=48/48 threads=12` を確認しています。

---


---

## 5. 実測結果（A100 40GB / warpdecode-a100）

計測条件は Task L と同じ（同一 GGUF、ctx 128、batch=1、start token 1、greedy 64 トークン、
F16 KV、12 スレッド）。VRAM は実行中の nvidia-smi memory.used の最大値。

### 5.0 計測プロトコル上の注意（重要）

CPU MoE は 1 トークンで **約 945 MB** の expert 重みを mmap 越しに読みます
（48 層 × 10 expert × 約 2 MB = gate 524 KB + up 524 KB + down 921 KB / expert）。
llama.cpp もまったく同じバイト列を同じ mmap 経由で読むため、**ページキャッシュが冷えていると
両者ともディスク律速**になり、CPU 側の差が見えなくなります。実際、キャッシュが冷えた状態の
全 48 層 CPU 実行は 64 トークン ×2 反復が 20 分以上かかって完走しませんでした（tok_s 0.2 相当）。
比較を成立させるには

1. 計測前に `cat <model> > /dev/null` でページキャッシュを温める（80 GB、1〜2 分）
2. llama.cpp と本エンジンを**交互**に、同一プロセス内 2 反復の最速値で測る
3. 配置と VRAM を対応させる（`CPU_MOE_LAYERS=N` ↔ llama.cpp `--keep-moe-gpu-last 48-N`）

が必要です。§5.1 の CPU 側の値はこの統制が完全ではないため、**「CPU 経路はこの条件で
全 GPU より遅い」という下限の証拠**として読んでください。

### 5.1 配置スイープ（expert cache 30,000 MiB = expert が VRAM にほぼ常駐）

| 構成 | CPU MoE 層 | decode (tok/s) | VRAM (MiB) | parity |
|---|---|---|---|---|
| 全 GPU（基準） | 0 | **5.488 / 6.155** | 36,248 | 64/64 |
| CPU MoE 8 層 | 8 | 5.197 | 36,248 | 64/64 |
| CPU MoE 16 層 | 16 | 4.995 | 36,248 | 64/64 |
| CPU MoE 24 層 | 24 | 4.773 | 36,248 | 64/64 |
| CPU MoE 全 48 層 | 48 | 4.255 | 36,248 | **64/64** |

thread 幅を 6 / 10 / 12 / 24 に振っても 4.04〜4.26 tok/s（差は約 5%）。12 vCPU の
Xeon @2.20GHz では 10 expert の並列度で既に飽和しています。

### 5.2 小さい expert cache（VRAM を意図的に絞った条件）

| 構成 | decode (tok/s) | VRAM (MiB) |
|---|---|---|
| 全 GPU、cache 8,000 MiB | 5.563（Task L と一致） | 14,248 |
| CPU MoE 16 層、cache 4,000 MiB | 4.875 | — |
| CPU MoE 全 48 層、cache 1,500 MiB | **4.256** | — |

### 5.3 llama.cpp との同一セッション比較（交互・温キャッシュ）

同じ A100 セッションで pinned llama.cpp の計測ハーネス
（`/content/wd_taskl/llama.cpp/build-cuda-sm80/bin/taskl-llama-decode`、commit `6fcaa16f`）を
ビルドし、`cat <model> > /dev/null` でページキャッシュを温めてから交互に測りました
（1 プロセス 2 反復の最速値）。

| 構成 | tok/s | VRAM (MiB) | oracle 一致 |
|---|---|---|---|
| llama.cpp K=0（MoE 全 CPU、12 スレッド） | **12.429** | 5,098 | 36/64（既定ビルドのため） |
| llama.cpp K=16 | **18.612** | 26,179 | 36/64 |
| llama.cpp K=24 | **21.308** | 33,865 | 40/64 |
| 本エンジン GPU のみ（cache 30,000 MiB） | **6.629**（同条件の最良。別ラウンドで 5.488 / 6.155） | 39,031 | **64/64** |
| 本エンジン CPU MoE 全 48 層 | 4.256（§5.1/§5.2 より。同一ラウンドの実行は他ジョブに SIGTERM で中断） | — | **64/64** |
| 本エンジン CPU MoE 32 / 16 / 8 層 | 4.773 / 4.995 / 5.197（同上） | 36,248 | **64/64** |

llama.cpp の数値は Task L（[LLAMACPP_AB.md](LLAMACPP_AB.md)）の 12.541 / 18.486 と一致し、
このセッションでも再現しています。**CPU MoE は llama.cpp K=0 の約 1/3 です。**
一方で oracle とトークン単位で一致するのは本エンジンだけです（llama.cpp の既定ビルドは
自分の CPU 参照実装と 36/64 しか一致しません。§1 のビルド条件を参照）。

注: 本エンジン側の同時計測は、共有セッションで他ジョブに SIGTERM で中断され（engine_cpu48 は
rc=143）、CPU 側の数値は §5.1/§5.2 の配置スイープからの引用です。

### 5.4 結論（正直に）

**CPU MoE は parity を保ったまま動きますが、A100 では全 GPU より遅く、
llama.cpp K=0 の 1/3 です。** 主因は 2 つ:

1. **直列化**: 1 層ごとに「GPU 完了待ち → D2H → CPU 計算 → H2D」が必要で、GPU と CPU が
   重なりません。全 GPU の MoE 段は非同期に流れるため、同じ計算を CPU に移しても短縮になりません。
2. **CPU の実効速度**: 12 vCPU（Xeon @2.20GHz）では 1 トークンあたりの expert 重み
   約 945 MB を読み切れず、1 トークンあたり 0.7〜0.8 秒の純 CPU 時間が GPU の
   クリティカルパスに加算されます。**同じ計算をする llama.cpp K=0 は 12.43 tok/s
   （= 11.6 GB/s 相当）なので、CPU 側の実装差だけで約 2.9 倍あります**。差分の候補は
   スケジュールで、llama.cpp は 1 expert を 12 スレッドで行分割する（隣接行を連続に読む）
   のに対し、本実装は 10 expert を 1 スレッドずつ並列に流します（10 本の独立ストリーム）。
   スレッド数を 6/10/12/24 と振っても 4.04〜4.26 tok/s で飽和するのは、この条件では
   メモリ側がボトルネックであることと整合します。行分割スケジュールへの変更は未検証です。

したがって Task N の目標「llama.cpp K=0（12.54 tok/s）を超える」は **未達** です。
本エンジンの GPU のみの最良値（30,000 MiB cache で 5.49〜6.16 tok/s）は、
llama.cpp の K=0（12.54）の 44〜49% にとどまります。

---

---

## 6. 再現手順

```bash
# 1. ホスト検証（nvcc 不要。AVX2 = スカラー双対 = int64 厳密ドット + llama.cpp AVX2 カーネル比較）
bash tests/qwen38_cpu_moe_host_check.sh

# 2. 実機（A100 セッション）。モデル DL → ビルド → ホスト検証 → 配置スイープ → parity
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/colab_run_cmd.py --label cpumoefin \
  --cmd "bash /content/cpu_moe_final.sh" \
  --upload tools/cpu_moe_final.sh:/content/cpu_moe_final.sh

# 3. 短縮版（配置 1 ラウンド + parity のみ）
  --cmd "bash /content/cpu_moe_quick.sh" --upload tools/cpu_moe_quick.sh:/content/cpu_moe_quick.sh

# 4. parity のみ（4 構成）
  --cmd "bash /content/cpu_moe_parity.sh" \
  --upload tools/cpu_moe_parity.sh:/content/cpu_moe_parity.sh \
  --upload issue1_artifacts/reference_tokens_64.json:/content/reference_tokens_64.json

# 5. llama.cpp をビルドして同一セッションで交互 A/B（Task L のハーネスを再利用）
  --cmd "bash /content/cpu_moe_llamacpp_ab.sh" \
  --upload tools/cpu_moe_llamacpp_ab.sh:/content/cpu_moe_llamacpp_ab.sh \
  --upload qwen38_llamacpp_gpu_bench.cpp:/content/qwen38_llamacpp_gpu_bench.cpp
```

| ツール | 用途 |
|---|---|
| `tools/cpu_moe_final.sh` | モデル DL → ビルド → ホスト検証 → 配置スイープ 3 ラウンド → parity |
| `tools/cpu_moe_quick.sh` | 同短縮版（1 ラウンド、`--bench-repeat` なし） |
| `tools/cpu_moe_parity.sh` | parity のみ 4 構成（全層 / 24 層 / 8 層 / 全 GPU） |
| `tools/cpu_moe_llamacpp_ab.sh` | pinned llama.cpp をビルドし K=0/16/24 と交互 A/B |
| `tools/cpu_moe_ab.sh` | 既存の llama.cpp ビルドを使う軽量 A/B |

CPU MoE は計測前に **ページキャッシュを温めてください**（§5.0）。`cpu_moe_final.sh` は
モデル DL 直後の状態で走るため、初回ケースが cold になります。

---

## 7. 未検証のリスク
- **llama.cpp の数値は同一セッションで再現済み**（§5.3、K=0 12.429 / K=16 18.612）。
  ただし本エンジン側の CPU 数値は、共有セッションで他ジョブに中断されたため
  §5.1/§5.2 の配置スイープからの引用で、llama.cpp と同じラウンドでは取れていません
  （engine_cpu48 は SIGTERM で rc=143）。厳密な交互比較は
  `tools/cpu_moe_ab3.sh`（キャッシュ温め + アイドル待ち付き）で再実行できます。
- **I/O 律速の影響が大きい**: ページキャッシュが冷えた状態の全 CPU 実行は
  64 トークン ×2 反復が 20 分以上かかり完走しません。§5 の数値はすべて
  「モデルを一度フルで読んだ後」の条件です。
- **IQ3_S は実機で blk.2 のみ**: parity 64/64 なので経路は通っています（ホスト/オラクルでは
  IQ3_S を直接検証済み）。
- **CPU 側 shared expert は実装していません**（llama.cpp の `--n-cpu-moe` も routed expert だけを
  CPU に置くため条件は揃っています）。shared expert は常に GPU 経路です。
- **prefill は CPU MoE 非対応**: batch=1 decode 専用で、prefill は常に GPU 経路です。
- **fused gate_up レイアウトの GGUF は拒否**します（configure() で明示的に例外）。
  本モデルは separate レイアウトなので問題ありません。
- **CPU が速いマシンでは結論が変わり得ます**。今回の 2.20GHz Xeon は CPU MoE に不利な条件です。
  ただし直列化の問題はマシン速度に依存しません。
