# Task S: MoE expert カーネル実行 64.5 ms の分解と削減（A100 20000MiB）

対象: `src/qwen38_moe.cuh`（MoE 本体と計測）、`src/qwen38_forward.cuh`（MoE 区間の
フックのみ）。parity は llama.cpp CPU greedy オラクルとの 64/64 を全構成で確認している。
実行は `tests/moe_kernel_profile_ab.py` + `tools/colab_run_cmd_q.py`（A100 / cache 20000 MiB /
64 token greedy）。

---

## 0. 結論（先に）

### 0.1 64.5 ms は expert カーネルではなかった

Task Q は差分法で「expert カーネル実行（I/O なし・5 launch/層）= 64.5 ms」と結論したが、
**それは 5 本の expert カーネルの時間ではない。** `WARODECODE_MOE_PROFILE=1` で
1 launch ずつ cudaEvent 対に分けると、同じ 5 launch は **8.44 ms/token** しかなく、
Task G のマイクロベンチ（L4 で 8.61 ms/token、`tests/moe_fused_grid_bench.cu`）と
2% 以内で一致する。残りは **router の選択処理**だった。

| 区間（転送なし構成、round 2・全スイッチ OFF） | ms/token | 割合 |
|---|---:|---:|
| **router 選択（GEMV 25.7 + 単一スレッド top-k 16.8 + D2H/同期 1.0）** | **42.9** | **62%** |
| shared expert（うち gate dot 5.2、9 launch） | 12.7 | 18% |
| expert cache acquire（I/O なしでは簿記のみ） | 9.5 | 14% |
| **expert カーネル 5 launch（gate+up GEMV / SiLU / Q8_0 / down GEMV / 加重和）** | **8.4** | **12%** |
| mirror（F16 化 x2 + add_shared + finish） | 1.1 | 1.6% |
| accum zero / finish / ptr H2D / rx quant | 1.6 | 2% |
| 合計 | 68.7 | 100% |

区間の親子関係（`router_select` ⊃ `act_quant` ⊃ `gemv` など）から、
`gemv + topk = 42.5 ms ≈ router_select 44.7 ms` と確認できた。

### 0.2 原因は「1 スレッド / 2 ブロック」の 3 カーネル

| カーネル | 実装 | 実測 |
|---|---|---:|
| `dense_f32_gemv_avx2_kernel`（`src/matrix.cuh`） | **1 スレッド = 1 行**。router [512, 2560] F32 で **2 ブロック 512 スレッド**、隣接レーンが 10 KB 離れた非コアレス読み | **25.7 ms/token**（5.24 MB を 9.8 GB/s） |
| `q38_topk_softmax_kernel` | `<<<1, 1>>>` の完全単一スレッド。512 logits の softmax + 10 段 × 512 幅 argsort | **16.8 ms/token** |
| `q38_moe_gate_dot_f32_avx2_kernel` | `<<<1, 1>>>`。n=2560 の内積 1 本を 1 スレッドで | **5.2 ms/token**（108 us/層） |

### 0.3 削減結果（A100 / 20000 MiB / 64 token greedy / 計測なし）

| case | tok_s | parity |
|---|---:|---|
| 3 スイッチ全 OFF（従来） | 7.982 | **64/64** |
| **3 スイッチ全 ON（新既定）** | **12.359** | **64/64** |

**+54.8%（7.982 → 12.359 tok/s）、parity 64/64 維持。**

MoE 区間（1 トークン、48 層）:

| 構成 | 変更前 | 変更後 | 差 |
|---|---:|---:|---:|
| 転送なし・Task Q が 64.5 ms と呼んだ構成 | 68.73 | **26.35** | **−62%** |
| I/O あり（実運用の既定） | 94.23 | **51.50** | −45% |

**目標「64.5 → 40 ms 以下」は達成（26.35 ms）。**

### 0.4 最終状態の MoE 内訳（転送なし、26.35 ms）

| site | ms/token | 備考 |
|---|---:|---|
| `moe.acquire` | 9.69 | I/O なしでは 48x10 回のホスト簿記 |
| `moe.batch_total` | 8.67 | expert 5 launch（もう最大ではない） |
| `moe.shared` | 8.27 | gate 0.67 + Q8_0 GEMV 3 本（`q38_hc_q8_0_q8_0_exact_gemv_f32_kernel`、実効 `34 GB/s） |
| `moe.router_select` | **4.44** | gemv 0.81 + topk 1.47 + quant/D2H/同期 |
| `moe.mirror` | 1.11 | |
| finish / accum_zero / ptr_h2d / rx_quant | 1.4 | |

---

## 1. 計測方法

### 1.1 カーネル単位の event 分解（`WARODECODE_MOE_PROFILE=1`）

`src/qwen38_moe.cuh` に `Q38MoeKernelProfiler` / `Q38MoeKernelScope` を追加した
（Task J の HC profiler / Task P の attention profiler と同じ仕組み）。各 launch を
cudaEvent 対で挟み、site ごとに出す:

| 列 | 意味 |
|---|---|
| `dev` | GPU タイムライン上の区間幅（ms/token）。区間内で GPU が遊んだ時間も乗る |
| `host` | 同じ区間のホスト wall time |
| `gap` | `host - dev`。正ならホストが GPU より長く区間内にいた（同期待ち等） |
| `calls/token` / `recorded` / `dropped` / `evfail` | 呼び出し回数と取りこぼし。`recorded>0, calls=0` は event 読み出し失敗を意味するので、「動かなかった」と「読めなかった」を区別できる |

site は 24 個で、親子（`stage_total` ⊃ `routed_total` ⊃ `batch_total` ⊃ 5 launch、
`router_select` ⊃ act_quant/gemv/topk/d2h/sync）を意図的に入れ子にしてある。
**親 dev − 子 dev の合計 = launch gap + 未計測分**。
token 末尾の `cudaStreamSynchronize` 後に `flush_token()` し `cudaEventElapsedTime` で回収。

```
[moe-profile] tokens=64 unit=ms_per_token
[moe-profile] moe.router.gemv  dev=25.7159 host=0.2168 gap=-25.4991 calls/token=48.00 recorded=3072 dropped=0 evfail=0
```

スイッチ OFF ではクラスが空でフックは分岐 1 つ。既定経路はバイト単位で不変。

### 1.2 A/B ランナー

`tests/moe_kernel_profile_ab.py`（新規）。1 ケース = 1 エンジンプロセス、共通コマンドは

```
warodecode --model <gguf> --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64
```

各ケースが自分の 64 トークンを出し `issue1_artifacts/reference_tokens_64.json` と
突き合わせて 64/64 を判定する。エンジン stdout JSON（tok_s / parity / Task F profile）、
stderr の `[moe-profile]` 表、expert cache 統計を集める。生の `[moe-profile]` 行も
JSON に残すのでパース失敗でデータを失わない。

> **並行編集への注意**: 本タスクの実行中、別タスク (Task T / sync-other 担当) が
> `src/qwen38_forward.cuh` と `src/qwen38_trace.hpp` を同時編集し、さらに
> commit `9573e93` で本タスクの作業中変更も一緒に取り込まれた。本タスクの A/B は
> **/tmp/wd-moe-snap に凍結したツリー**で取得している（第 4 章の数値はすべてこの凍結ツリー）。

---

## 2. 実装した改善（すべて bit 一致・環境変数で A/B 可）

### 2.1 `WARODECODE_ROUTER_F32_WARP`（既定 1）: router GEMV 25.7 → 0.81 ms

router の F32 logits GEMV は ggml の AVX2 訪問順（`sum[4][8]`、要素 `i + j*8 + l`、
i は 32 刻み）で **加算順が固定**されている。そこで 32 個のアキュムレータを 1 warp の
lane に 1 対 1 で割り当て（lane t = `sum[t>>3][t&7]`）、各 lane が元と同一の FMA 連鎖を
同一の要素順で走る `q38_router_gemv_f32_warp_kernel` を追加した。配置だけが変わるので
値はビット一致し、ロードはレーンが行を並走するためコアレスになる。
最後の `q38_matrix_reduce_avx2()` は同じペア順を `__shfl_down_sync` で再現した
（ペア順の等価性はローカルで 20 万ケース検証: `200000/200000 bit-identical`）。
`WARODECODE_ROUTER_F32_WARP=0` で従来カーネル（`src/matrix.cuh` は無変更）。

### 2.2 `WARODECODE_ROUTER_TOPK_PARALLEL`（既定 1）: router top-k 16.8 → 1.47 ms

`q38_topk_softmax_parallel_kernel`（512 スレッド 1 ブロック）を追加。単一スレッド版と
同じ演算を同じ順で分割する:

- **max**: `fmaxf` は結合的・可換なので reduction tree でも同一値。
- **exp**: `q38_exact_avx2_exp_f32(__fsub_rn(logits[e], max_v))` を要素並列（順序無関係）。
- **sum**: 8 要素ごとの `q38_moe_softmax_lane_sum()` は独立に計算し、`double` への
  **順序付き加算 64 回だけ**を 1 スレッドに残す（scalar tail も元の順に同スレッドで評価）。
- **select**: 参照実装は昇順スキャン + 厳密な `>` なので「未選択の最大値のうち最小 index」を選ぶ。
  `(値, index)` の全順序で「値が大なら勝ち、等しければ index が小さい方が勝つ」combine 則は
  同じ結果を与え（NaN は `>` も `==` も偽なので参照と同じく昇格しない）、reduction tree の
  形に依存しない。
- **norm**: 同じ `double` 和、同じ `fmaxf(., 6.103515625e-5f)` クランプ、同じ `__fdiv_rn` / `__fmul_rn`。

参照実装が `q38_moe_router_legacy()` に落とす条件（`n_experts % 8 != 0` 等）では従来カーネル。
`WARODECODE_ROUTER_TOPK_PARALLEL=0` で従来カーネル。

### 2.3 `WARODECODE_SHEXP_GATE_WARP`（既定 1）: shared gate dot 5.2 → 0.67 ms

`q38_moe_gate_dot_f32_avx2_kernel` は `<<<1,1>>>` で同じ 32 アキュムレータ木を
1 スレッドで回していた。2.1 と同じ分解（lane t = `sum[t>>3][t&7]`、同じ shuffle 木）の
warp 版を追加。`WARODECODE_SHEXP_GATE_WARP=0` で従来カーネル。

---

## 3. 帯域下限との比較

expert 重みは 1 トークンで約 **1.20 GB** を GPU が読み直す:

| 対象 | 1 トークン | 内訳 |
|---|---:|---|
| routed expert | 946 MB | 48 層 x 10 expert x (gate_up 1,049,600 B + down 921,600 B) |
| shared expert | 251 MB | 48 層 x 3 行列 x 1,638,400 重み x 1.0625 B（Q8_0） |

- **HBM 下限**: 1.20 GB / 1,555 GB/s = **0.77 ms/token**。
  実測の expert 5 launch 8.4 ms は実効 **113 GB/s = 下限の 11 倍**で、
  まだ余地はある（が削減量は数 ms 規模）。
- **PCIe 下限**: 20000 MiB cache の実転送 231 MB/token を 12.2 GB/s で ≈ **19 ms**
  （Task Q の測定値）。I/O ありの `moe.acquire` dev 34.8 ms は DMA 待ちを含み、
  ここはリンク上限付近（Task Q の結論を再確認）。
- **つまり 64.5 ms は帯域では説明できず、並列度（1 スレッド / 2 ブロック）だった。**

---

## 4. A/B と parity（A100 / 20000 MiB / 計測なし）

| case | 内容 | tok_s | parity |
|---|---|---:|---|
| `base` | 3 スイッチ全 OFF | 7.982 | **64/64** |
| `opt` | 3 スイッチ全 ON（新既定） | **12.359** | **64/64** |
| `opt-g2` | ON + `MOE_BATCH_GROUPS=2` | 12.330 | **64/64** |
| `opt-g5` | ON + `MOE_BATCH_GROUPS=5` | 11.747 | **64/64** |

計測ありの内訳（`WARODECODE_MOE_PROFILE=1` + `PROFILE=1` + `EXPERT_STATS=1`）:

| case | tok_s | total_ms | moe_ms | acquire | batch_total | shared | router_select | gemv | topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prof-opt` | 10.985 | 90.68 | **51.50** | 34.81 | 8.73 | 8.26 | 4.48 | 0.81 | 1.47 |
| `prof-opt-skip`（I/O なし・parity は設計上 0/64） | 15.211 | 65.44 | **26.35** | 9.69 | 8.67 | 8.27 | 4.44 | 0.81 | 1.47 |

段階ごとの効果（すべて parity 64/64）:

| round | 追加した改善 | tok_s | MoE ms（転送なし） |
|---|---|---:|---:|
| 2 | なし（ベースライン計測） | 7.955 | 68.73 |
| 3 | router GEMV warp 化 | 9.711 | 46.18 |
| 4 | + shared gate dot warp 化（top-k は配線漏れで未発動） | 10.433 | 41.54 |
| 5 | + router top-k 並列化（配線修正） | **12.359** | **26.35** |

**`MOE_BATCH_GROUPS` は既定 1 のままが最良**（g2/g5 はわずかに負け）。
router が速くなって DMA 露出の相対コストが上がるという懸念は、実測では現れなかった。

---

## 5. 実装中に見つけた不具合（記録）

- **top-k の配線漏れ**: 並列カーネル自体は正しかったが、このモデルが実際に通る
  `q38_moe_select_experts_f32_x()` の呼び出しは、計測スコープを入れた際にインデントが
  変わっており、一括置換の対象から外れていた。round 4 の A/B で
  「`WARODECODE_ROUTER_TOPK_PARALLEL=0` と `=1` の top-k が 16.9 ms で同一」という形で
  検出できた（**A/B が無ければ「効果なし」と誤判定していた**）。
- **計測が動かない原因の切り分け**: 最初の版は「site 行が 1 本も出ない」という症状で、
  原因は summary の出力条件（`calls==0 かつ overflow==0` をスキップ）だった。
  `recorded` / `evfail` / `host` を常に出すようにして解決した。**「フックが動いていない」
  と「event が読めていない」を区別できる列を必ず残すべき**という教訓。

---

## 6. 未検証リスク

1. **A100 のみ**。L4/T4 では HBM 帯域と PCIe 幅が違うため router の比率は変わる
   （ただし 1 スレッドカーネルの遅さは GPU 非依存で、同じ 3 本が効くはず）。
2. **別タスクとの並行編集**。`src/qwen38_forward.cuh` / `src/qwen38_trace.hpp` は
   Task T が同時編集しており、commit `9573e93` に本タスクの作業中変更も混ざっている。
   本タスクの数値は凍結スナップショットのもので、**マージ後の再測定は未実施**。
   `src/qwen38_moe.cuh` は本タスクのみが触っている。
3. **`skip` 系（`EXPERT_IO_SKIP`）は routing が発散する**（parity 0/64）。帯域の
   反実仮想ではなく同一構成内の比較にのみ使っている。
4. **計測オーバーヘッド**。`MOE_PROFILE` + `PROFILE` + `STATS` 同時で
   `opt` 12.359 → `prof-opt` 10.985（**約 11% 低下**）。表の絶対値は計測ありであり、
   改善量の比較には計測なしの `base` / `opt` を使うこと。
5. **残っている最大の改善候補**（本タスクでは未着手）:
   - `moe.acquire` 9.7 ms（I/O なし）。48x10 回のホスト簿記で、`src/expert_cache.cuh`
     側の担当領域。
   - `moe.shared` の Q8_0 GEMV 3 本で約 7.6 ms（実効 34 GB/s）。
     `q38_hc_q8_0_q8_0_exact_gemv_f32_kernel` は `src/qwen38_hc.cuh` 側の担当領域。
   - expert 5 launch 8.67 ms は HBM 下限の 11 倍。DP4A の行ルーチン改善余地。
6. **OOM 1 件**。`warp0`（計測なし・従来 GEMV）ケースが他プロセスの VRAM 残りで
   `cudaMalloc expert cache failed: out of memory` になった。同一スイッチの後続ケースは
   64/64 で通っており、スイッチの影響ではない。
7. **`moe.loop_total`（per-expert F16 legacy ループ）は本モデルでは未使用**。
   `WARODECODE_MOE_FUSED_GRID=0` でも `q38_moe_run_routed_experts_q8_act_separate`
   の経路を通るため、`moe.loop.*` の子 site は instrument していない。
