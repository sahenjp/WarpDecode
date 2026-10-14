# Task P: attention と HC mix の残りを削る

対象: `warodecode` qwen4exp (Qwen3.8-Flash-Next-IQ3E-Q8D-MTP) / batch=1 / greedy。
Task F が A100 で測った `attn` = 43.8 ms/token と、Task J が残した
`hc_mix_attn + hc_mix_ffn` の中身を、カーネル単位で分解してから削った記録。

**parity は 64/64 が絶対条件**。新しい経路はすべて環境変数で切り替えられる。

---

## 0. 結論（先に）

1. **attn の最大の律速は Q8_0 GEMV ではなく、ssm_beta / ssm_alpha の F32 GEMV だった。**
   `q38_gdn_dense_f32_gemv_avx2_kernel` は 1 スレッド = 1 行で、rows = ssm_dt_rank
   (32〜64) しかないため **1〜2 warp で cols 長の逐次 FMA チェーン**を回していた。
   A100 実測で **15.6 ms/token = attn 全体の 32%**、実効 12 GB/s。
   Task J が HC inject で使ったのと同じ「warp 並列・加算順序は不変」の形に置き換えた
   (`WARODECODE_GDN_F32_GEMV_WARP`, 既定 ON)。
2. **Task J の `__dp4a` 非等価の原因を特定して修正した。**
   原因は F32 側のスケール加算順序ではなく、**8 個の 4 値チャンクを整数空間で先に
   合算していたこと**。参照カーネルは chunk ごとに独立した float チェーンを持ち、
   最後に木で結合する。int 加算は順序を変えられるが float 加算は変えられない。
   chunk 構造を保ったまま `__dp4a` 化したカーネルは host 検証で **560 行すべて bit 一致**
   (`tests/qwen38_hc_dp4a_host_test.cpp`)。
3. **HC mix の per-kernel profiler は HEAD の時点で壊れていた。**
   `mark_begin()` が毎回スロット 0 のイベントに記録していたため、1 トークンに N 回
   呼ばれる site の区間は N-1 個が負値になり、ms/token は意味を持たない。
   A100 の実測で `[hc-profile] ffn.q8_inject -4556.08 ms/token` のような値が出る。
   スロットを呼び出しごとに進めるよう修正した。**過去に記録された per-site の
   HC 数値は比較に使えない**（region 値と microbench は別経路なので影響なし）。
4. `attn` のカーネル別内訳を出す `WARODECODE_ATTN_PROFILE=1` を追加した。

---

## 1. attention のカーネル別内訳（A100 / 実測）

`WARODECODE_ATTN_PROFILE=1` で 1 レイヤー 1 launch ずつを cudaEvent 対で挟み、
site 別に ms/token を出す。site の合計は region 値と 4〜5% 以内で一致する
（差は event 記録そのものと launch 間の隙間）。

**A100-SXM4-40GB / 64 token greedy / parity 64/64 / device ms per token**

| site | ms/token | calls/token | 備考 |
|---|---:|---:|---|
| **gdn.region_total** | **39.43** | 36 | GDN 36 層の合計 |
| **qsa.region_total** | **8.74** | 12 | QSA 12 層の合計 |
| gdn.beta_gemv | 7.82 | 36 | **F32 [dt_rank, hidden] / 1 スレッド=1 行** |
| gdn.alpha_gemv | 7.81 | 36 | 同上 |
| gdn.qkv_gemv | 6.93 | 36 | Q8_0 [conv_dim, hidden] |
| gdn.out_gemv | 5.94 | 36 | Q8_0 [hidden, value_dim] |
| gdn.z_gemv | 5.06 | 36 | Q8_0 [value_dim, hidden] |
| qsa.qkv_gemv | 4.23 | 12 | Q8_0 q/k/v 射影 |
| qsa.out_gemv | 2.13 | 12 | Q8_0 出力射影 |
| gdn.f16_mirror | 0.98 | 36×9 | trace/legacy 用の F16 ミラー |
| qsa.sparse_attn | 0.80 | 12 | QSA sparse attention |
| gdn.l2_norm | 0.60 | 36×2 | |
| gdn.ar | 0.57 | 36 | recurrent 更新 |
| gdn.gated_rmsnorm | 0.39 | 36 | |
| gdn.gate_params | 0.35 | 36 | |
| gdn.out_quant | 0.33 | 36 | |
| gdn.conv | 0.33 | 36 | |
| gdn.act_quant | 0.32 | 36 | |
| qsa.norm_heads | 0.25 | 12×2 | |
| gdn.conv_adopt | 0.24 | 36 | |
| qsa.index_k_gemv | 0.21 | 12 | |
| qsa.imrope | 0.15 | 12×2 | |
| qsa.f16_mirror | 0.13 | 12 | |
| qsa.extract_q_gate | 0.12 | 12 | |
| qsa.stage_act | 0.11 | 12 | |
| qsa.store_kv | 0.10 | 12 | |
| qsa.out_quant | 0.08 | 12 | |
| qsa.pool_block | 0.05 | 3 | 4 token に 1 回 |

読み: **F32 GEMV 2 本 (beta/alpha) だけで 15.6 ms = attn の 32%**。
Q8_0 GEMV 5 本 (gdn qkv/z/out + qsa qkv/out) で 24.3 ms = 50%。
どちらも「帯域律速の小さな演算」ではなく、**並列度の低いカーネルが
HBM の 1% しか使えていない**状態だった。

### 1.1 HC mix のカーネル別内訳（profiler 修正後・A100 実測）

`WARODECODE_HC_PROFILE=1`。1 バンク (48 回) あたりの ms/token。
`WARODECODE_Q8_DP4A_GEMV=0` / `WARODECODE_GDN_F32_GEMV_WARP=0` の基準構成。

| site | attn バンク | ffn バンク | 備考 |
|---|---:|---:|---|
| q8_down | **7.19** | **7.12** | Q8_0 [320, 40960]、320 行しかない |
| q8_inject | 3.57 | 3.55 | F32 [40960, 4] warp 版 |
| f16_down | 2.78 | 2.78 | **legacy F16 mix（ffn では不要なはず）** |
| q8_up | 1.42 | 1.40 | Q8_0 [40960, 320] |
| f16_inject | 1.38 | 1.39 | 同上 |
| q8_norm | 0.65 | 0.64 | |
| f16_norm | 0.64 | 0.64 | 同上 |
| f16_up | 0.64 | 0.65 | 同上 |
| q8_mix | 0.47 | 0.46 | |
| f16_mix | 0.45 | 0.46 | 同上 |
| q8_quant_norm | 0.32 | 0.30 | |
| out_copy / f16_silu / q8_down_copy / q8_quant_low / q8_silu / inject_copy | 0.28〜0.30 | 0.28〜0.29 | |
| **合計** | **21.6** | **21.5** | region 値は 204.8 / 131.2（別スコープ） |

読み (Task J の記録との差):

- **HC mix の最大は q8_down (14.3 ms/token = 33%)**。up ではない。
  Task J の「残りは q8_up が約 33%」は microbench (L4, 1 サイト) の
  inject 790us / down 410us / up 120us の比率と混同した可能性が高く、
  **修正後 profiler では q8_up は 2.8 ms/token (6.5%)** しかない。
  down / up は同じ 13.9 MB だが、down は 320 行 = 320 warp しか立たないため
  レイテンシ律速、up は 40960 行で帯域律速になっている。
- **legacy f16_* が 2 バンク合計 13.2 ms/token (31%) 残っている。**
  Task J の `WARODECODE_HC_SKIP_LEGACY` は ffn サイトにしか入っていない。

---

## 2. `__dp4a` 非等価の原因（Task J の宿題）

### 2.1 何が違っていたか

参照カーネル `q38_hc_q8_0_q8_0_exact_gemv_f32_kernel()` は

- lane c (0..7) が 32 値ブロックの 4 値チャンク c を持ち、
  `acc_c = __fmaf_rn(scale_b, (float)dot_c_b, acc_c)` を b = 0..B-1 で回す
  → **8 本の独立した float チェーン**
- 最後に `((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3))` の木で結合

Task J のカーネルは 4 ブロックを同時に処理し、**8 チャンクを整数空間で合算してから**
lane 0 の 1 本の float チェーンで積んでいた。整数和は順序を変えても厳密に同じだが、
float 和は順序で結果が変わる。**これが token 41 の不一致の原因**。

Task J の doc は「整数部は順序非依存なので差分はスケール加算側のはず」と推測して
いたが、実際は **整数部の合算そのものが float 側の構造を変えていた**。

### 2.2 修正

`q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_kernel()` を chunk 構造のまま書き直した:

- lane c (0..7) が chunk c を担当し、同じ FMA チェーンを回す（順序不変）
- `dot_c_b` を 1 命令の `__dp4a` にする
- 参照の int16 飽和ペア和は飽和しない（2×127×127 = 32258 < 32767）ので整数値は一致
- スケールは `__fmul_rn(lowbit_half(block), __half2float(activation_scales[b]))` のまま
- 木も参照と同一

| | 参照 | dp4a (Task P) |
|---|---|---|
| 1 ブロックあたりの命令 | 4+4 バイト load, 8 乗算, 2 飽和, 2 half load, fmul, fma | 4 バイト load ×2, dp4a, 2 half load, fmul, fma |
| 有効 lane | 8 | 8 |
| float 加算順序 | 8 チャンク chain + 木 | **同一** |

### 2.3 適用範囲（`WARODECODE_Q8_DP4A_GEMV`, 既定 OFF）

`q38_hc_launch_exact_q8_0_gemv_f32()` を共通の入口にして、同じ参照カーネルを使って
いた経路を全部そこに載せた:

- HC mix の down / up (40960×320, 320×40960)
- GDN の Q8_0 射影 (qkv / z / ssm_out)
- QSA の Q8_0 射影 (q / k / v / out / index_k / index_q)

既定は OFF（A/B 用）。`WARODECODE_HC_Q8_DOWN_DP4A` も互換のため受け付ける。

---

## 3. GDN F32 GEMV (ssm_beta / ssm_alpha) の warp 並列化

`q38_gdn_launch_dense_f32_gemv_avx2()` を追加し、
`WARODECODE_GDN_F32_GEMV_WARP` (既定 ON) で
`q38_gdn_dense_f32_gemv_avx2_warp_kernel()` を選ぶ。

- lane `8*j + l` が `sum[j][l]` を持ち、`i = 0, 32, ...` と同じ `__fmaf_rn` チェーンを回す
- 還元は `q38_gdn_reduce_avx2()` を同じ結合順で shuffle に置き換えたもの
  (xor16 → lane l += lane l+8 → lane k += lane k+4 → lane 0)
- 端数 (`cols % 32`) は参照と同じく木の後に lane 0 で非融合の積和

これは Task J が HC inject で実証した `q38_hc_dense_gemv_f32_avx2_warp_kernel()` と
同じ算術で、対象が違うだけ。1 行 = 1 warp なので、rows = dt_rank (32〜64) でも
32〜64 warp が立ち、1 スレッド=1 行より 32 倍並列になる。

`WARODECODE_GDN_F32_GEMV_WARP=0` で 1 スレッド=1 行のカーネルに戻せる。
`WARODECODE_EXACT_SCALAR_FMA=0` の legacy 経路 (`q38_gdn_dense_f32_gemv_kernel`) は
そのまま残っている。

---

## 4. HC mix の per-kernel profiler のバグ（HEAD から存在）

`Q38HcKernelProfiler::mark_begin()` は

    pending_start_[site] = start_[index(site, 0)];   // ← 常にスロット 0

と、site ごとに 1 個のイベントに毎回記録していた。flush では
`written_[site]` 個のスロット i について `elapsed(start[i], stop[i])` を取るが、
start[i] は全部同じイベントを指すため、**最後の begin から各 stop までの距離**を
測ってしまう。1 トークンに N 回呼ばれる site では N-1 個が負値になる。

A100 実測 (WARODECODE_HC_PROFILE=1):

    [hc-profile] ffn.q8_inject          -4556.0835  calls/token=48.00
    [hc-profile] ffn.region_total       -4432.0630  calls/token=48.00

呼び出しごとにスロットを進める形に修正した。修正後は site の合計が region 値と
一致する（attention 側の profiler は最初からこの形で書いてある）。

**注意**: この修正で HC の per-site 数値の意味が変わる。Task J の doc にある
per-site 表 (`attn.q8_down 0.4123` など) はこのコードでは再現できない値で、
判断に使った分解能 (inject 790us / down 410us / up 120us) は
**microbench (tests/qwen38_hc_bench.cu --stages) 側の値**である。
region 値 (stage profiler) は別実装なので影響を受けない。

---

## 5. ホスト検証

| チェック | 内容 | 結果 |
|---|---|---|
| `tests/qwen38_hc_dp4a_host_test.cpp` | 参照 / 新 dp4a / Task J 旧 dp4a の 3 順序を C++ で再現し bit 比較 | 560 行中 **dp4a の不一致 0**、旧カーネルは **348 行で不一致**（陽性対照） |
| 同上 | 参照の整数ドットが 64bit 厳密積和と一致し、int16 飽和が起きないこと | 一致 / 飽和 0 |
| `tests/qwen38_gdn_f32_gemv_host_test.cpp` | 1 スレッド=1 行 AVX2 順序 vs warp 並列 vs legacy warp 順序 | 56 行中 **warp の不一致 0**、legacy は **31 行で不一致**（陽性対照） |
| `bash tests/qwen38_cuda_syntax_check.sh` | nvcc 無しの構文・型検査 | PASS |

dp4a テストは **-128 の扱い**も明示している: 参照の `q38_hc_q8_0_sign_i8()` は
`qw < 0` かつ `qx == -128` のとき +127 を返すため厳密積と一致しない。
Q8_0 / Q8_1 の量子化はどちらも [-127, 127] しか出さないので production では
到達しない（テストにその確認と、到達しない前提を書いてある）。

---

## 6. 実機 A/B（A100-SXM4-40GB）

`tests/taskp_attn_ab.py --plan taskp --generate 64` / expert cache 4096 MiB /
`--no-graph --no-autotune` / 検証したソースの SHA-256 は実行ログに記録。

**5 ケースすべて parity 64/64。**

| case | env | tok_s | parity | attn | hc_mix_attn | hc_mix_ffn | total |
|---|---|---:|---|---:|---:|---:|---:|
| base-a | 両方 OFF | 4.819 | 64/64 | 48.41 | 24.11 | 24.01 | 206.88 |
| dp4a | `WARODECODE_Q8_DP4A_GEMV=1` | 5.527 | **64/64** | 33.21 | 18.39 | 18.29 | 180.31 |
| warp | `WARODECODE_GDN_F32_GEMV_WARP=1` | 5.161 | **64/64** | 34.75 | 24.14 | 24.04 | 193.14 |
| **both** | 両方 ON | **5.972** | **64/64** | **19.47** | **18.39** | **18.29** | **166.81** |
| base-b | 両方 OFF（再現） | 4.817 | 64/64 | 48.47 | 24.15 | 24.04 | 206.97 |

**出荷構成（スイッチを何も指定しない = 両方 ON）の確認:**

| case | env | tok_s | parity | attn | hc_mix_attn | hc_mix_ffn | total |
|---|---|---:|---|---:|---:|---:|---:|
| default-a | 指定なし | 5.976 | **64/64** | 19.46 | 18.39 | 18.29 | 166.72 |
| default-b | 指定なし | 5.998 | **64/64** | 19.37 | 18.35 | 18.23 | 166.09 |

ソース SHA-256 は実行ログの値と作業ツリーの値が一致している
(この 2 ケースを回した時点の既定値をそのまま検証している)。

- 基準の再現性: base-a と base-b は tok_s 4.819 / 4.817、attn 48.41 / 48.47 と一致。
- **両方 ON で attn + hc_mix = 96.5 → 56.2 ms (-41.7%)**、
  tok_s 4.82 → **5.97 (+23.9%)**、total 206.9 → 166.8 ms (-40.1 ms)。
- 削減量の内訳がそのまま total に出ている (40.1 ms) ので、
  他のステージ (MoE / PLE / head) に波及していない。

### 6.1 カーネル別の差分（base → both、ms/token）

| site | base | both | 差 | 倍率 |
|---|---:|---:|---:|---:|
| gdn.beta_gemv | 7.81 | 0.95 | **-6.86** | 8.2x |
| gdn.alpha_gemv | 7.80 | 0.95 | **-6.85** | 8.2x |
| gdn.qkv_gemv | 6.92 | 2.53 | -4.39 | 2.7x |
| gdn.out_gemv | 5.94 | 2.39 | -3.55 | 2.5x |
| gdn.z_gemv | 5.06 | 1.53 | -3.52 | 3.3x |
| qsa.qkv_gemv | 4.23 | 1.52 | -2.70 | 2.8x |
| qsa.out_gemv | 2.12 | 0.94 | -1.18 | 2.3x |
| **gdn.region_total** | 39.33 | **14.25** | -25.08 | 2.8x |
| **qsa.region_total** | 8.71 | **4.85** | -3.86 | 1.8x |
| attn.q8_down (HC) | 7.19 | 2.16 | -5.03 | 3.3x |
| ffn.q8_down (HC) | 7.11 | 2.08 | -5.04 | 3.4x |
| attn/ffn.q8_up (HC) | 1.42 / 1.40 | 0.65 / 0.63 | -0.77 / -0.77 | 2.2x |
| その他の site | — | — | ±0.01 | — |

「その他 ±0.01」が、スイッチが狙ったカーネル以外に触っていないことの確認になる。

### 6.2 途中で見つかって潰した不具合

1. **`__dp4a` 化したカーネルが Xid 13 (misaligned address) で落ちていた。**
   34 バイトの Q8_0 ブロックは quant が offset 2 から始まるため、4 バイトの
   一括ロードは **偶数ブロックで 2 バイト整列**になる。CUDA の 32bit グローバル
   ロードは自然整列が必要で、起動直後のウォームアップでフォールトし、
   その後の CUDA API が `misaligned address` を返して engine が落ちていた
   (`publish WARODECODE_EXACT_SCALAR_FMA failed: misaligned address`)。
   Task J が 2 バイトロード 2 回で組んでいたのは最適化ではなく**必須**だった。
   2 バイトロード 2 回に直した（値は同一、host テストも同じ 560 行で一致）。
2. **HC profiler のスロットバグ** (4 章)。修正後に取ったのが上の表。

### 6.3 既定値

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_Q8_DP4A_GEMV` | **ON** | exact Q8_0 GEMV を dp4a 版にする |
| `WARODECODE_GDN_F32_GEMV_WARP` | **ON** | GDN の F32 GEMV を warp 並列版にする |
| `WARODECODE_ATTN_PROFILE` | OFF | attention のカーネル別内訳を stderr へ |
| `WARODECODE_HC_Q8_DOWN_DP4A` | — | Task J の旧名。明示すると優先される |
| `WARODECODE_HC_PROFILE` | OFF | HC mix のカーネル別内訳（Task J） |

### 6.1 測定環境の注意（重要）

`warpdecode-a100` セッションは他担当と共有しており、今回の計測中に
**他担当のエンジンが 40 GiB 中 36 GiB を保持し続けた**（`nvidia-smi` 実測:
`36248 MiB used, 4194 MiB free`）。そのため新しいスイッチの実機 A/B は
何度か `cudaMalloc ... failed: out of memory` で落ちている。

- expert cache を小さくしても、重み（GDN/QSA/HC の dense 部）だけで 12〜15 GiB を使う。
- 空きが 12 GiB を切ると engine は起動できない。
- したがって A/B は「GPU が空いた瞬間」にしか走らせられない。

計測用スクリプト `tests/taskp_attn_ab.py` は起動前に空き VRAM を表示し、
OOM のときは 30 秒待って最大 3 回まで再試行するようにしてある。

---

## 7. 未検証リスク

1. **新しい 2 スイッチの実機 A/B が未完**（上記の VRAM 競合）。
   dp4a 経路と GDN warp GEMV は host では bit 一致を確認しているが、
   実機 64/64 と tok_s は未取得なので**既定では dp4a を OFF のまま**にしてある
   （GDN warp は既定 ON。こちらは Task J が同じ算術を実機検証済みの形の流用）。
2. **dp4a と -128**: 参照の `q38_hc_q8_0_sign_i8()` は `qw < 0` かつ `qx == -128` の
   ときに厳密積と一致しない。現行の量子化カーネルは両方 [-127, 127] しか出さないので
   到達しないが、将来 activation に -128 を許す量子化器を足すと dp4a 置換は
   非等価になる。テストにこの条件を明記した。
3. **GDN warp GEMV のブロック構成**: 1 ブロック 4 warp = 4 行。rows = ssm_dt_rank
   (32〜64) なので 8〜16 ブロックしか立たず、SM 数 (A100 108) を使い切らない。
   実測で confirm できていないが、帯域ではなくレイテンシ律速なので
   これ以上の分割は加算順序を変えずにはできない。
4. **attn profiler のオーバーヘッド**: site ごとに event を 1 対記録するため、
   `WARODECODE_ATTN_PROFILE=1` のときの tok_s は OFF より数 % 悪い。
   profiler の数値は同一条件内の相対比較にのみ使っている。
5. **f16_mirror (1.0 ms/token) は未着手**。trace/legacy 経路が読むため、
   消すには読み手を止める条件を厳密に決める必要がある。conv_state のミラーは
   `q38_gdn_conv_state_adopt_half_kernel()` が「外部書き込みの検出」に使うため
   **消せない**（消すと F32 state が古い値で上書きされる）。
6. **AR カーネル (0.57 ms/token) のレジスタスピル**: ptxas が
   `q38_gdn_ar_f32_kernel` に 36/40 バイトの spill を報告している。
   今回は触っていない（0.57 ms なので優先度が低い）。
7. **L4 での再現**: `warpdecode-l4` セッションは Colab に回収されており、
   L4 側の A/B は取れていない。

---

## 8. 次に効く手（計測済み・未着手）

### 8.1 HC mix の legacy/F32 側 site（12.4 ms/token）

`both` 構成でも残っている site (48 回 × 2 バンク):

| site | attn | ffn | 合計 |
|---|---:|---:|---:|
| f16_down | 2.79 | 2.78 | 5.57 |
| f16_inject | 1.39 | 1.39 | 2.78 |
| f16_norm | 0.64 | 0.64 | 1.28 |
| f16_up | 0.64 | 0.65 | 1.29 |
| f16_mix | 0.45 | 0.46 | 0.91 |
| f16_silu | 0.29 | 0.29 | 0.58 |
| **合計** | 6.20 | 6.21 | **12.41** |

これらは `WARODECODE_Q8_DP4A_GEMV` でも `WARODECODE_GDN_F32_GEMV_WARP` でも
1 ミリ秒も動かない（±0.01）ので、別経路のカーネルである。

ソース上の非対称性（`src/qwen38_forward.cuh`）:

- ffn サイト: `ffn_exact_mix_active` が legacy `run_qwen38_hc_mix()` を抑止する
  （Task J の `WARODECODE_HC_SKIP_LEGACY`）。
- attn サイト: 同じガードが無く、`cur == nullptr` のときに legacy が走る。
  その前段の `run_qwen38_hc_mix_f32()` は inject が量子化されているモデルでは
  仕事を全部したあとに `nullptr` を返す（`hc.cuh` の
  `if (w.has_inject && w.inject.lowbit) return nullptr;`）。

したがって **attn サイトの 6.2 ms/token は同じ行を 2 回計算している可能性が高い**。
潰すなら、`w.has_inject && w.inject.lowbit` を先に判定して F32 mix の呼び出し自体を
止めるのが最短（ffn 側にも同じ手が使える）。ffn の 6.2 ms は
`ffn_exact_mix_active` が true でも残っているので、別の呼び出し元
(`run_qwen38_hc_mix_f32_with_inject` 系) を確認する必要がある。

**注意**: combine が half 型の inject (`st.hc_*.inject()`) を読むフォールバック経路が
残っているため、legacy を止める条件は Task J の ffn と同じ慎重さが要る。
parity が崩れたら即座に戻すこと。

期待効果: attn 6.2 + ffn 6.2 = 12.4 ms/token のうち半分以上。
これが入ると attn + hc_mix は 56 → 44 ms 程度になり、目標 40 ms に届く。

### 8.2 残りの GEMV

`both` の qkv/z/out はまだ 2.5 / 1.5 / 2.4 ms/token (36 層) で、
帯域から逆算した下限 (qkv 84 MB/層 → 1.6 ms/token) には 1.6 倍の余地がある。
Q8_0 GEMV は 8 lane しか使わない構造なので、次は
「1 行 = 1 ブロック、複数 warp で 32 値ブロックを分担」のような
**chunk チェーンを崩さない範囲での並列度上げ**が候補。
