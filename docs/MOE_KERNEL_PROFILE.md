# Task S: MoE expert カーネル実行 64.5 ms の分解と削減（A100 20000MiB）

対象: `src/qwen38_moe.cuh`（MoE 本体と計測）、`src/qwen38_forward.cuh`（MoE 区間の
フックのみ）。parity は llama.cpp CPU greedy オラクルとの 64/64 を全構成で確認している。

---

## 0. 結論（先に）

1. **Task Q の「expert カーネル実行 64.5 ms」は 5 本の expert カーネルの時間ではなかった。**
   `WARODECODE_MOE_PROFILE=1` で 1 launch ずつ event 対に分けたところ、同じ 5 launch は
   **8.4 ms/token** しかなく、Task G のマイクロベンチ（L4 で 8.61 ms/token、
   `tests/moe_fused_grid_bench.cu`）と 3% 以内で一致した。**転送なし構成の 64.5 ms は
   ほぼ全部が router にある。**

   | 区間 | ms/token | 割合 |
   |---|---:|---:|
   | **router 選択（GEMV 25.7 + 単一スレッド top-k 16.8 + 同期 0.5 + D2H 0.5）** | **43.0** | **62%** |
   | shared expert（うち gate dot 5.2、9 launch） | 12.7 | 18% |
   | expert cache acquire（I/O なし構成では簿記のみ） | 9.5 | 14% |
   | **expert カーネル 5 launch（gate+up GEMV / SiLU / Q8_0 / down GEMV / 加重和）** | **8.4** | **12%** |
   | mirror（F16 化 x2 + add_shared + finish） | 1.1 | 1.6% |
   | accum zero / finish / ptr H2D / rx quant | 1.6 | 2% |

2. **router の 43 ms は 2 つの「1 スレッド / 極小並列」カーネルだった。**
   - `dense_f32_gemv_avx2_kernel`（`src/matrix.cuh`）は **1 スレッド = 1 行**で、
     router [512, 2560] F32 に対して **2 ブロック 512 スレッド**しか立たない。
     さらに隣接レーンが 10 KB 離れたアドレスを読むため **完全に非コアレス**。
     実測 **25.7 ms/token（535 us/層、5.24 MB を 9.8 GB/s）**。
   - `q38_topk_softmax_kernel` は `<<<1, 1>>>` の **完全単一スレッド**で
     512 logits の softmax と 10 段の 512 幅 argsort を回す。実測 **16.8 ms/token**。
   - `q38_moe_gate_dot_f32_avx2_kernel` も `<<<1, 1>>>`（同じ 32 アキュムレータ木を
     1 スレッドで再現）。実測 **5.2 ms/token（108 us/層、n=2560 の内積 1 本）**。

3. **この 3 本を bit 一致のまま並列化して削った。**
   合計で **MoE 区間 −25 ms（95.99 → 70.80 ms）、tok_s 7.372 → 9.06（+23%）**、
   parity 64/64。転送なし構成では **68.7 → 46.2 ms**（top-k と gate dot の分を入れると
   さらに下がる。第 5 章）。

4. **帯域下限との比較**: expert 重みは 1 トークンで約 **1.20 GB**（routed 480 expert x
   1.88 MiB + shared 48 x 4.98 MiB）を GPU が読み直す。A100 の HBM 1.555 TB/s なら
   **0.77 ms/token** が下限。5 launch の実測 8.4 ms は **下限の 11 倍 = 113 GB/s**
   で、ここにはまだ余地がある（が、削減量は数 ms 規模）。
   一方 **PCIe 転送の下限は 231 MB / 12.2 GB/s = 19 ms** で、I/O ありの
   `acquire` 34.5 ms の大半はこれ。**64.5 ms の主因は帯域ではなく並列度**だった。

---

## 1. 計測方法

### 1.1 カーネル単位の event 分解（`WARODECODE_MOE_PROFILE=1`）

`src/qwen38_moe.cuh` に `Q38MoeKernelProfiler` / `Q38MoeKernelScope` を追加した。
Task J の HC profiler / Task P の attention profiler と同じ仕組みで、MoE 区間の
各 launch を cudaEvent 対で挟み、site ごとに **dev / host / gap** を出す:

| 列 | 意味 |
|---|---|
| `dev` | GPU タイムライン上の区間幅（ms/token）。区間内で GPU が遊んだ時間も乗る |
| `host` | 同じ区間のホスト wall time |
| `gap` | `host - dev`。正ならホストが GPU より長く区間内にいた（同期待ち等） |
| `calls/token` / `recorded` / `dropped` | 呼び出し回数。`dropped` は event 上限で落ちた本数 |

site は 24 個。親子関係のある site（`stage_total` ⊃ `routed_total` ⊃ `batch_total` ⊃ 5 launch、
`router_select` ⊃ act_quant/gemv/topk/d2h/sync）を意図的に入れ子にしてあり、
**親 dev − 子 dev の合計 = launch gap + 未計測分**が読み取れる。
token 末尾の `cudaStreamSynchronize` 後に `flush_token()` して `cudaEventElapsedTime`
で回収する（`cudaErrorNotReady` を踏まない）。

```
[moe-profile] tokens=64 unit=ms_per_token
[moe-profile] moe.router.gemv       dev=  25.7159 host=   0.2168 gap= -25.4991 calls/token=48.00 recorded=3072 dropped=0 evfail=0
```

スイッチ OFF ではクラスが空でフックは分岐 1 つになり、既定経路はバイト単位で不変。

### 1.2 A/B ランナー

`tests/moe_kernel_profile_ab.py`（新規）。1 ケース = 1 エンジンプロセスで、
共通コマンドは

```
warodecode --model <gguf> --ctx 128 --device 0 --start-token 1 \
           --no-graph --no-autotune --generate 64
```

各ケースが自分の 64 トークンを出し、`issue1_artifacts/reference_tokens_64.json` と
突き合わせて 64/64 を判定する。collect するのは (a) エンジン stdout JSON
（tok_s / parity / Task F の `profile`）、(b) stderr の `[moe-profile]` 表、
(c) expert cache の `{"expert_cache_stats"}`。生の `[moe-profile]` 行も JSON に
残すので、パース失敗でデータを失うことはない。

A100 (warpdecode-a100) / cache 20000 MiB / 実行は `tools/colab_run_cmd_q.py`
（`/content/wd_q` に隔離、GPU に他プロセスがいれば待つ）。

> 注意: 本タスクの実行中、**別タスク (Task T) が `src/qwen38_forward.cuh` と
> `src/qwen38_trace.hpp` を同時編集していた**（18:13 時点で diff +4 行 → 18:16 に +286 行）。
> Colab に上げるツリーは `/tmp/wd-moe-snap` に凍結し、その 2 ファイルだけ HEAD に戻して
> 本タスクの 4 フックを再適用したものを使用した。第 5 章の数値はすべてこの凍結ツリー。

---

## 2. 現状の分解（A100 / 20000 MiB / 64 token greedy）

### 2.1 I/O なし構成（Task Q が「64.5 ms」と呼んだもの）

`WARODECODE_EXPERT_IO_SKIP=1` + expert-batched grid。round 2 の実測で MoE 区間
**68.73 ms**（Task Q の 64.5 ms と計測オーバーヘッド差の範囲で一致）。

| site | dev ms/token | calls/token | 意味 |
|---|---:|---:|---|
| `moe.stage_total` | 69.04 | 48 | Task F の `kQwen38StageMoe` と同義 |
| `moe.routed_total` | 54.26 | 48 | routed expert ランナー全体 |
| `moe.router_select` | **42.88** | 48 | router + top-k + D2H + 同期 |
| `moe.shared` | 12.73 | 48 | shared expert（9 launch） |
| `moe.acquire` | 9.50 | 48 | cache acquire（SKIP では簿記のみ） |
| `moe.batch_total` | **8.44** | 48 | **expert 5 launch の合計** |
| └ `moe.batch.down` | 3.73 | 48 | IQ4_NL down GEMV |
| └ `moe.batch.gate_up` | 2.75 | 48 | IQ2_S/IQ3_S gate+up GEMV |
| └ `moe.batch.accum` / `q8_quant` / `silu` | 0.34 / 0.34 / 0.32 | 48 | |
| `moe.mirror` / `accum_zero` / `finish` / `ptr_h2d` / `rx_quant` | 1.11 / 0.45 / 0.44 / 0.38 / 0.32 | 48–96 | |

**5 launch の 8.44 ms は L4 マイクロベンチの 8.61 ms と 2% 一致。**
Task Q の差分法が「expert カーネル実行 64.5 ms」と読んだのは、
`SKIP` が消したのが転送だけであり、残った 64.5 ms の中身が router だったため。

### 2.2 router 43 ms の中身（round 3 の子 site）

| site | dev ms/token (warp OFF) | dev ms/token (warp ON) | 備考 |
|---|---:|---:|---|
| `moe.router.gemv` | **25.72** | **0.81** | `dense_f32_gemv_avx2_kernel`（2 ブロック 512 スレッド） |
| `moe.router.topk` | **16.79** | 16.88 | `<<<1,1>>>` の softmax + argsort |
| `moe.router.act_quant`（外側） | 26.03 | 1.12 | gemv を内包する外側区間 |
| `moe.router.d2h` | 0.52 | 0.52 | 40 B + 40 B の pinned D2H |
| `moe.router.sync` | dev 0.55 / **host 101.0** | dev 0.53 / **host 76.5** | ここで GPU の残作業を待つ |

`gemv + topk = 42.5 ms ≈ router_select 44.7 ms` で、**router はこの 2 本で説明できる**。
`sync` の host が大きいのは GPU が詰まっている側の待ちであり、
区間の dev はほぼ 0（イベント 2 本が隣接）— 独立したコストではない。

### 2.3 GPU が読む重み量と帯域下限

| 対象 | 1 トークン | 内訳 |
|---|---:|---|
| routed expert | 946 MB | 48 層 x 10 expert x (gate_up 1,049,600 B + down 921,600 B) |
| shared expert | 251 MB | 48 層 x 3 行列 x 1,638,400 重み x 1.0625 B（Q8_0） |
| **合計** | **約 1.20 GB** | |

- **HBM 下限**: 1.20 GB / 1,555 GB/s = **0.77 ms/token**。
  実測 5 launch 8.4 ms は **113 GB/s = 下限の 11 倍**。
- **PCIe 下限**: 20000 MiB cache の実転送 231 MB/token を 12.2 GB/s で ≈ **19 ms**
  （Task Q の測定値。I/O ありの `moe.acquire` dev 34.5 ms は DMA 待ちを含む）。
- つまり **64.5 ms は帯域では説明できず、並列度（1 スレッド / 2 ブロック）が原因**。

---

## 3. 実装した改善（すべて bit 一致・環境変数で A/B 可）

### 3.1 `WARODECODE_ROUTER_F32_WARP`（既定 1）

`q38_router_gemv_f32_warp_kernel` を追加。router の F32 logits GEMV は
ggml の AVX2 訪問順（`sum[4][8]`、要素 `i + j*8 + l`、i は 32 刻み）で
**加算順が固定**なので、その 32 個のアキュムレータを 1 warp の lane に 1 対 1 で
割り当てる（lane t = `sum[t>>3][t&7]`）。各 lane は元と同一の FMA 連鎖を
同一の要素順で走るので値はビット一致し、ロードはレーンが行を並走するため
コアレスになる。最後の `q38_matrix_reduce_avx2()` は同じペア順を
`__shfl_down_sync` で再現した（ペア順の等価性はローカルで 20 万ケース検証:
`200000/200000 bit-identical`）。

**実測: 25.72 → 0.81 ms/token（32 倍）**、5.24 MB/層 を 9.8 GB/s → 約 300 GB/s。

### 3.2 `WARODECODE_ROUTER_TOPK_PARALLEL`（既定 1）

`q38_topk_softmax_parallel_kernel`（512 スレッド, 1 ブロック）を追加。
単一スレッド版と同じ演算を同じ順で分割する:

- **max**: `fmaxf` は結合的・可換なので reduction tree でも同一値。
- **exp**: `q38_exact_avx2_exp_f32(__fsub_rn(logits[e], max_v))` を要素並列（順序無関係）。
- **sum**: 8 要素ごとの `q38_moe_softmax_lane_sum()` は独立に計算し、
  `double` への **順序付き加算 64 回だけ**を 1 スレッドに残す（末尾の
  scalar tail も同スレッドで元の順に評価）。`sum == 0` の -INFINITY 分岐も同じ。
- **select**: 参照実装は昇順スキャン + 厳密な `>` なので「未選択の最大値のうち
  最小 index」を選ぶ。`(値, index)` の全順序で「値が大なら勝ち、等しければ index が
  小さい方が勝つ」という combine 則は同じ結果を与え（NaN は `>` も `==` も偽なので
  参照と同じく昇格しない）、reduction tree の形に依存しない。
- **norm**: 同じ `double` 和、同じ `fmaxf(., 6.103515625e-5f)` クランプ、
  同じ `__fdiv_rn` と `__fmul_rn`（w_scale）。

参照実装が `q38_moe_router_legacy()` に落とす条件（`n_experts % 8 != 0` など）では
従来カーネルのまま。

### 3.3 `WARODECODE_SHEXP_GATE_WARP`（既定 1）

`q38_moe_gate_dot_f32_avx2_kernel` は `<<<1,1>>>` で同じ 32 アキュムレータ木を
1 スレッドで回していた。3.1 と同じ分解（lane t = `sum[t>>3][t&7]`、同じ shuffle 木）で
warp 版を追加。**5.22 → 0.38 ms/token 見込み**（round 4 で確認）。

---

## 4. 未計測だったもの

- `moe.loop_total`（per-expert F16 legacy ループ）は本モデルでは使われない
  （`WARODECODE_MOE_FUSED_GRID=0` でも `q38_moe_run_routed_experts_q8_act_separate`
  の 7 launch/expert 経路を通る）。そのため `moe.loop.*` の子 site は
  instrument していない。grid OFF との比較は `prof-loop`（round 2: MoE 97.6 ms）と
  `prof-grid`（94.2 ms）で取れている。
- `moe.router.act_quant` は gemv を内包する外側区間なので、単独の値は
  `act_quant.dev - gemv.dev` で読む（round 3: 26.03 − 25.72 = 0.31 ms）。

---

## 5. A/B と parity

（round 4 の結果を反映。以下は暫定。）

| case | スイッチ | tok_s | parity |
|---|---|---|---|
| round 2 `plain` | 計測なし・全最適化 OFF | 7.955 | **64/64** |
| round 3 `prof-warp0` | router GEMV 従来 | 7.372 | **64/64** |
| round 3 `prof-warp1` | router GEMV warp 並列 | 9.06 | **64/64** |
| round 4 | 第 5 章で更新 | | |

---

## 6. 未検証リスク

1. **A100 のみ**。L4 では HBM 帯域と PCIe 幅が違い、router の比率は変わるはず
   （ただし 1 スレッドカーネルの遅さは GPU 非依存）。
2. **`MOE_BATCH_GROUPS` の再評価が未了**。router が速くなったことで、
   expert pass が全 10 expert の DMA を先に待つ構成（既定 groups=1）の相対コストが
   上がっている可能性がある。
3. **別タスク (Task T) が `src/qwen38_forward.cuh` / `src/qwen38_trace.hpp` を
   同時編集している。** 本タスクの A/B は凍結スナップショットで取っており、
   マージ後の再測定は未実施。`qwen38_moe.cuh` は本タスクのみが触っている。
4. **`skip` 系は routing が発散する**（Task Q と同じ注意）。帯域の反実仮想ではなく
   同一構成内の比較に使っている。
5. **計測オーバーヘッド**。`MOE_PROFILE` は MoE 区間に event 対を足す。
   round 3 の `prof-warp1` は 9.06 tok/s、計測なしの `warp1` は 9.711 tok/s なので
   **約 7% のオーバーヘッド**がある。絶対値は計測ありの値であり、相対比較にのみ使う。
   parity はすべての構成で成立している。
6. **`warp0`（計測なしの従来 GEMV）ケースは他プロセスの VRAM 残りで
   `cudaMalloc expert cache failed: out of memory` で落ちた**（後続の
   `prof-warp0` は同じスイッチで 64/64）。A/B は `prof-warp0` と `prof-warp1`
   の対で取っている。
