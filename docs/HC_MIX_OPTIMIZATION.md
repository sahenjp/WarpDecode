# Task J: HC mix の最適化（80 ms = トークンの30%）

対象: `warodecode` qwen4exp (Qwen3.8-Flash-Next-IQ3E-Q8D-MTP) / batch=1 / greedy。
Task F が A100 で計測した「hc_mix_attn + hc_mix_ffn = 80 ms/token」を
カーネル単位に分解し、原因を特定して潰した記録。**parity は 64/64 を維持**。

---

## 0. 結論（先に）

**A100-SXM4-40GB / expert cache 20000 MiB / --generate 64 / profile ON / 実測2回ずつ**

| | pre-J（全スイッチOFF） | Task J 既定 | 差 |
|---|---:|---:|---:|
| `hc_mix_attn` | 40.24 ms | **18.55 ms** | -21.7 |
| `hc_mix_ffn` | 40.23 ms | **18.54 ms** | -21.7 |
| **HC mix 合計** | **80.47 ms** | **37.09 ms** | **-43.4 ms（2.17x）** |
| `attn` | 43.96 | 44.04 | ±0（無関係） |
| `moe` | 102.6 | 100.4 | ±0（無関係） |
| `total_ms` | 231.7 | **185.5** | **-46.2 ms** |
| `tok_s` | 4.29〜4.33 | **5.39** | **+25%** |
| **parity** | **64/64** | **64/64** | 維持 |

- 削減量 43.4 ms（mix 区間）と 46.2 ms（トークン全体）がほぼ一致し、
  **HC mix 以外に波及していない**ことを確認できる。
- pre-J の 40.24 / 40.23 ms は Task F の 40.14 / 40.19 ms をほぼ完全に再現しており、
  Task F の計測値と本タスクの A/B が同じものを測っていることの裏取りになる。

### 「80 ms は本当に GPU が働いていたのか」への答え
**本当に GPU が働いていた。artifact ではない。**
L4 で HC mix 1サイトを単体実行すると 1366 us、1 launch ごとに同期しても 1366 us で、
**連続 launch（キュー投入）と 2% 以内で一致**する。48層×2サイトに換算すると
L4 131 ms / A100 80 ms となり、Task F の区間計測と閉じる。
ただし中身は「帯域律速の小さな演算」ではなく、**極端に並列度の低い GEMV 2本**だった。

---

## 1. 計測方法（2つ足した）

### 1.1 カーネル単位の event 分解（エンジン内）
`src/qwen38_hc.cuh` に `Q38HcKernelProfiler` / `Q38HcKernelScope` を追加。

- HC mix の中の**各カーネル起動を個別の cudaEvent 対で挟み**、GPU タイムライン上の
  所要時間を site ごとに積算する。
- `WARODECODE_HC_PROFILE=1` で有効。OFF のときは分岐1つで、既定経路は不変。
- token 末尾（`cudaStreamSynchronize` 済み）で `flush_token()` して集計し、
  終了時に stderr へ site 別 ms/token を出す:

```
[hc-profile] tokens=64 unit=ms_per_token
[hc-profile] attn.q8_down            0.4123  calls/token=48.00
[hc-profile] ffn.region_total        0.9012  calls/token=48.00
```

site は attn / ffn の 2 バンクに分けてある（共有カーネル本体を呼び分けるため
`base_site` 引数を追加）。

有効化されたスイッチは起動時に1行だけ stderr に出す（どの経路が動いたかログで判別するため）:

```
warodecode: HC mix Task J switches: skip_legacy=1 gamma_resident=1 inject_warp=1 down_dp4a=0
```

### 1.2 スタンドアロン・マイクロベンチ
`tests/qwen38_hc_bench.cu`（新規）。合成データで実際の HC mix 本体を回す:

```bash
nvcc -arch=sm_89 -O3 -I src -o hc_bench tests/qwen38_hc_bench.cu
./hc_bench --hidden 10240 --hc 4 --low-rank 320 --iterations 200 --stages --json out.json
```

- `--stages`: 1カーネルずつ launch + sync して、どのカーネルが何 us かを列挙する。
- `[sync]` 列 = 1 launch ごとに同期（GPU が空のときのコスト）、
  `[batch]` 列 = 連続 launch（エンジンと同じキュー状態）。
  **両者がほぼ一致**するので、エンジン内の区間計測は GPU 実行時間を捉えていると確認できた。
- 実行ランナー: `tools/hc_mix_colab.py --mode bench`（専用 log/pid、他タスクを kill しない）。

> 落とし穴の記録: `--stages` が最初「warmup sync: an illegal memory access」で落ちたが、
> 原因はベンチ側が**ホストポインタを inject GEMV に渡していた**こと。エンジンの不具合ではない。

---

## 2. 内訳（L4 / hidden=10240, hc=4, low_rank=320 / 1サイトあたり us）

```
既定構成（exact Q8_0 + F32 inject）1365.7 us
  attn|ffn.q8_inject     790.5   58%   F32 inject GEMV (4 x 40960, F32 655 KB)
  attn|ffn.q8_down       410.6   30%   Q8_0 down GEMV (320 x 40960, 13.9 MB)
  attn|ffn.q8_up         120.6    9%   Q8_0 up GEMV   (40960 x 320, 13.9 MB)
  attn|ffn.q8_gamma_h2d   33.2    2%   40 KB の host->device コピー
  attn|ffn.q8_norm        14.6    1%   grouped RMSNorm (40960)
  attn|ffn.q8_mix          7.5
  attn|ffn.down_copy       3.3         low_raw -> low の D2D コピー
  attn|ffn.q8_quant_low    2.9
  attn|ffn.q8_quant_norm   2.8
  attn|ffn.silu            2.7
```

**読み**: カーネルは小さいのに GEMV 2本が支配的。どちらも「重みを読む帯域」ではなく
**並列度の低さによるレイテンシ律速**だった。

- `q8_inject` は **1スレッド = 1行**。inject 重みは `[40960, 4]` なので
  **4スレッドしか立たず**、各スレッドが 1280 反復を回る。実効 0.8 GB/s。
- `q8_down` は 1 warp = 1 行で、1ブロック(32値)ごとに 5段の `__shfl_down` 還元。
  1行あたり 1280 回の還元 = 直列依存が長く、実効 34 GB/s（L4 の実効帯域 ~150 GB/s の 1/4）。
- `q8_up` は 120 us / 13.9 MB = 117 GB/s で**ほぼ帯域限界**。ここは触らない。

---

## 3. 実装した改善

| # | スイッチ | 既定 | 内容 | 効果（L4, 1サイト） |
|---|---|---|---|---|
| 1 | `WARODECODE_HC_SKIP_LEGACY` | **1** | ffn サイトの重複 F16 mix を削除 | -151 us |
| 2 | `WARODECODE_HC_GAMMA_RESIDENT` | **1** | RMSNorm 重みを device 常駐化 | -33 us |
| 3 | `WARODECODE_HC_INJECT_F32_WARP` | **1** | F32 inject GEMV を warp 並列化 | -729 us |
| 4 | `WARODECODE_HC_Q8_DOWN_DP4A` | **0**（否定的結果） | down GEMV を `__dp4a` 4ブロック同時化 | -344 us だが parity 非維持 |

### 3.1 ffn サイトの重複 mix を削除（`qwen38_forward.cuh`）
`WARODECODE_HC_Q8_0` が有効なとき、ffn サイトは `run_qwen38_hc_mix`（legacy F16 一式 =
6カーネル）を実行した直後に `run_qwen38_hc_mix_f32_with_inject`（exact Q8_0 一式）で
**同じ値をもう一度**計算していた。attn サイトには Task F の指摘で
`attn_f32_mix_active` ガードが入っていたが、ffn サイトには入っていなかった。

- 追加したのは `ffn_exact_mix_active` ガード 1 つ（attn 側と同じ形）。
- 正しさ: legacy F16 が書く `st.hc_ffn.{xn,gate,lo,inject,mixed}` は exact 経路では
  (a) F32->half コピーで上書きされるか、(b) combine が F32 scratch を読むため使われない。
  実行時に読む消費者は `run_qwen38_hc_combine`（f16 inject）だけで、
  それは `ffn_f32_inject` が false のときだけ走る。
- さらに**ガード条件に `qwen38_hc_f32_inject_enabled()` を追加**した。
  inject テンソルが量子化されているモデル（blk.48 MTP）や `WARODECODE_HC_INJECT_F32=0` では
  exact 経路が `st.hc_*.inject()` を埋めないため、legacy を残さないと
  **古い inject を combine が読む**（潜在バグ。attn 側にも同じ形で入れた）。

### 3.2 RMSNorm 重みの device 常駐（`qwen38_hc.cuh`）
exact 経路は呼ばれるたびに `w.debug_norm_f32`（ホスト F32 40 KB）を
`cudaMemcpyAsync` で scratch に送っていた。**96回/token、33 us/回 = 3.2 ms/token**。

- `Qwen38HcWeights::norm_f32_device` を追加し、`load_qwen38_hc_weights()` /
  `load_qwen38_hc_head()` で 1 回だけ upload。
- ミックス本体は `q38_hc_norm_f32_device()` が返す device ポインタを直接読む。
  手組み weights（probe/ベンチ）では nullptr が返り、従来どおりコピーする。

### 3.3 F32 inject GEMV の warp 並列化（bit-identical、既定ON）
`q38_hc_dense_gemv_f32_avx2_kernel` は 1スレッド = 1行なので 4 スレッドしか立たない。
`ggml_vec_dot_f32` の順序を保ったまま、**1 lane = 1 accumulator の 32 本並列**に組み替えた
（`q38_hc_dense_gemv_f32_avx2_warp_kernel`）:

- lane `8*j + l` が `sum[j][l]` を担当し、`i = 0, 32, 64, ...` を
  同じ `__fmaf_rn` 連鎖で回す（**各 accumulator の順序は不変**）。
- 還元 `GGML_F32x8_REDUCE` を同じ結合順の shuffle に置き換え
  （`shfl_xor(16)` → lane l += lane l+8 → lane k += lane k+4 → lane0 で `h0+h1`）。
- ループを `#pragma unroll 4` してメモリレベル並列度を確保。
- **790.5 us → 61.1 us（12.9x）**。`WARODECODE_HC_INJECT_F32_WARP=0` で従来カーネルに戻せる。

### 3.4 down GEMV の `__dp4a` 4ブロック同時化 → **否定的結果（既定OFF）**
`q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_kernel` を実装した（広い方 = cols >= 1024 のみ）。

- 32値の整数ドットを `__dp4a` 1命令にし、**4ブロックを同時**処理して
  還元を 4ブロックあたり 3 shuffle（従来 20 shuffle）に削減。
- 速度は出た: **410.6 us → 約 66 us（6.2x）**、A100 で 11.2 ms/token の削減。
- **しかし 64/64 にならない**。`tests/hc_mix_ab.py --plan=bisect`（A100）で
  `down_dp4a=0` のときだけ 64/64 に戻り、他の3スイッチを個別に落としても 41/64 のまま
  → **このカーネル単独の非等価**が原因と確定。
- 整数ドットは和の順序に依存しない（int 加算は結合的・厳密）ので、差分は
  4ブロック分の `scale * dot` を lane 0 で足す F32 側に残っているはず。**未解決**。
- 既定は 0。`WARODECODE_HC_Q8_DOWN_DP4A=1` で実験的に有効化できるが、
  **parity を壊すので既定にしない**。

---

## 4. 実機 A/B（A100 / warpdecode-a100 / expert cache 20000 MiB）

`tests/hc_mix_ab.py --plan=parity --repeat 2 --generate 64`

| case | tok_s | parity | hc_mix_attn | hc_mix_ffn | total_ms |
|---|---:|---|---:|---:|---:|
| warmup（既定） | 5.302 | 64/64 | 18.55 | 18.54 | 188.5 |
| base-0（既定） | 5.385 | **64/64** | 18.55 | 18.54 | 185.6 |
| base-1（既定） | 5.391 | **64/64** | 18.54 | 18.54 | 185.4 |
| all_off-0（pre-J） | 4.328 | **64/64** | 40.24 | 40.23 | 231.0 |
| all_off-1（pre-J） | 4.293 | **64/64** | 40.30 | 40.63 | 232.9 |

`all_off` = `WARODECODE_HC_SKIP_LEGACY=0` + `WARODECODE_HC_GAMMA_RESIDENT=0` +
`WARODECODE_HC_INJECT_F32_WARP=0` + `WARODECODE_HC_Q8_DOWN_DP4A=0`（= Task J 以前の経路）。

マイクロベンチ（L4、1サイト・200反復）での段階的な効果:

| 構成 | per-call | 対 初期 |
|---|---:|---:|
| 初期（exact + F32 inject） | 1365.7 us | 1.0x |
| + inject warp 化 | 653.8 us | 2.1x |
| + gamma 常駐 | — | — |
| + down `__dp4a`（現在は既定OFF） | 280.3 us | 4.9x |
| **既定（dp4a OFF）** | **約 386 us** | **3.5x** |

---

## 5. 未検証リスク

1. **`__dp4a` down GEMV の非等価は未解決**。整数部ではなく F32 側の
   4ブロック分のスケール加算にあると推測しているが、特定していない。
   既定OFFなので parity への影響はない。
2. **L4 でのエンジン A/B は既定スイッチの組み合わせでは未取得**。
   L4 の測定はマイクロベンチ（合成データ）と、dp4a を有効にした状態の
   parity/bisect プランのみ。A100 の最終確認は 64/64。
3. `q8_up`（A100 で mix の約 33% を占める）は未着手。
   L4 では帯域限界近傍だが、A100 ではここが次の的になる。
4. `WARODECODE_HC_SKIP_LEGACY=1`（既定）で legacy F16 の副作用
   （`st.hc_*.xn/gate/lo/inject` の中身）に依存する**デバッグ用 trace 経路**は、
   その値を読む分岐（`trace->state_trace` かつ debug override 有効時）で意味が変わりうる。
   通常の decode 経路では読まれない。
5. マイクロベンチの重みは乱数合成で、モデル実重みの値分布とは異なる。
   ただし計測した4本はいずれも値に依存しない（命令数・メモリ配置で決まる）律速。
6. 統合ツリーは他タスク（Task H の expert I/O 等）が同時に編集しているため、
   本計測は**その時点の固定スナップショット**で取得した。絶対 `tok_s` は
   統合ツリーの最新状態と一致しない可能性がある（相対比較は同一スナップショット内で有効）。
7. 計測中に **warpdecode-l4 が Colab に回収され**、A100 セッションが空いていたため
   A100 で最終確認を行った。l4b は Task K が使用中で、Task K のアップロードが
   一時期 /content/warodecode-source.zip を上書きしていた（その間の1回の計測は破棄した）。

---

## 6. 再現手順

```bash
# 1. 構文確認（CUDA なしで通る）
bash tests/qwen38_cuda_syntax_check.sh

# 2. マイクロベンチ（L4）
export PATH=/home/sahen/.local/bin:$PATH
WARODECODE_COLAB_SESSION=warpdecode-l4 python3 tools/hc_mix_colab.py \
    --label taskJ_bench --mode bench --iterations 200 --bench-args=--stages

# 3. エンジン parity A/B（A100。既定 vs 全スイッチOFF を各2回）
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  WD_EXPERT_CACHE_MIB=20000 python3 tools/hc_mix_colab.py \
    --label taskJ_parity --mode ab --plan=parity --repeat 2 --generate 64

# 4. どのスイッチが parity を壊すかの切り分け
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/hc_mix_colab.py --label taskJ_bisect --mode ab --plan=bisect

# 5. 個別スイッチの A/B（--switches で1つずつ）
#    WARODECODE_HC_SKIP_LEGACY=0 / WARODECODE_HC_GAMMA_RESIDENT=0 /
#    WARODECODE_HC_INJECT_F32_WARP=0 / WARODECODE_HC_Q8_DOWN_DP4A=0
```
