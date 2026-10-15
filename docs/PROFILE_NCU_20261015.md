# ncu による 1 トークン全カーネル計測（2026-10-15, A100 80GB）

対象: Qwen3.8-Flash-Next（IQ3E-Q8D-MTP）/ batch=1 decode / ctx 128 /
engine の既定構成（expert cache 20,000 MiB, --no-graph）。

計測: `ncu --metrics gpu__time_duration.sum --launch-count 4000`
（3000 本超の launch を 1 トークン分そのまま取得し、カーネル名ごとに合計）。

---

## 0. 結論

1. **1 トークン = 3,183 launch、カーネル時間の合計 41.0 ms**（実測のトークン時間は 37.6 ms。
   ncu は既定で毎回 L2 をフラッシュするため、キャッシュに載っている小カーネルは
   実測より重く出る。合計が 9% 大きいのはこの分）。
2. **最大の費目は Q8_0 の exact GEMV ファミリで 16.0 ms（全体の 39%）。**
   内訳は非パイプ版 10.20 ms（276 呼び出し / 36.95 us）、PIPE=32 3.90 ms（96 / 40.6 us）、
   PIPE=16 1.92 ms（120 / 16.0 us）。1 層あたり約 10 本の Q8_0 GEMV が走っている。
3. **層ごとの host 往復（cudaStreamSynchronize + ids の D2H）は 3.5% しかなかった。**
   同期・読み戻し・acquire を丸ごと外した診断ビルド（WARODECODE_MOE_SELECT_NOSYNC=1,
   出力は不正になる）で 26.60→27.60 / 26.76→27.85 / 26.60→27.60 tok/s。
   したがって「トークン時間の半分はホスト」という仮説は否定され、**残りは全部カーネル**。
4. argmax（248,320 要素）は ncu 上 532 us で、マルチブロック + 単調キー化に置き換えたが
   **実測の tok/s は変わらなかった**（26.65→26.59 等、ノイズ内）。実運用では logits が
   L2 に載っているため、ncu のコールド値が過大だった。正しさはホスト検証済みで、
   置き換え自体は残してある（単一ブロック 1 SM 依存をやめた）。

---

## 1. カーネル別の内訳（上位 25、ms/token）

| カーネル | 合計 ms | 平均 us | 呼び出し |
|---|---:|---:|---:|
| q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_kernel | **10.198** | 36.95 | 276 |
| q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_pipe_kernel<32> | 3.898 | 40.61 | 96 |
| iq2_s_q8_k_gemv_experts_dp4a_ilp_kernel<1,4> | 2.886 | 61.39 | 47 |
| q38_hc_dense_gemv_f32_avx2_warp_pipe_kernel<16> | 2.803 | 29.20 | 96 |
| q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_pipe_kernel<16> | 1.916 | 15.96 | 120 |
| q38_gdn_dense_f32_gemv_avx2_warp_kernel | 1.842 | 25.59 | 72 |
| q38_topk_softmax_parallel_kernel | 1.582 | 32.96 | 48 |
| q38_hc_q8_0_quantize_f32_exact_kernel | 1.335 | 3.47 | 385 |
| q38_hc_group_rmsnorm_f32_kernel | 1.329 | 13.70 | 97 |
| iq4_nl_q8_0_act_gemv_experts_dp4a_smem_kernel<8> | 1.104 | 23.00 | 48 |
| q38_gdn_float_to_half_kernel | 1.059 | 3.27 | 324 |
| q38_hc_copy_f32_to_half_kernel | 0.905 | 3.12 | 290 |
| q38_hc_mix_f32_cpu_order_kernel | 0.896 | 9.33 | 96 |
| q6_k_q8_k_gemv_kernel<1>（出力ヘッド） | 0.854 | 854.27 | 1 |
| q38_gdn_l2_norm_f32_kernel | 0.698 | 9.69 | 72 |
| q38_router_gemv_f32_warp_kernel | 0.641 | 13.36 | 48 |
| q38_gdn_ar_f32_kernel（ptxas が spill 警告） | 0.558 | 15.49 | 36 |
| q38_argmax_kernel | 0.532 | 532.35 | 1 |
| q38_hc_combine_f32_residual_f32_block_f32_inject_kernel | 0.531 | 5.53 | 96 |
| q38_hc_silu_scale_ggml_f32_kernel | 0.457 | 3.15 | 145 |
| q38_moe_gate_dot_f32_avx2_warp_kernel | 0.424 | 8.83 | 48 |
| q38_gdn_gated_rmsnorm_f32_kernel<float> | 0.419 | 11.64 | 36 |
| q38_rmsnorm_heads_exact_kernel<float,float> | 0.355 | 14.77 | 24 |
| q38_ple_rmsnorm_mul_exact_kernel | 0.328 | 109.34 | 3 |
| q38_hc_q8_0_q8_0_exact_gemv_f32_kernel（非 dp4a 経路） | 0.312 | 155.86 | 2 |

（上位 25 の合計 ≈ 39.4 ms / 全体 41.0 ms）

---

## 2. ここから分かる次の一手

- **Q8_0 exact GEMV を速くする以外に大きな伸びしろは無い。** 16 ms（39%）がここに集中しており、
  単体ベンチでは大きい形（[8192,2560] など）が 543 GB/s = HBM ピークの 35% しか出ていない。
  **（2026-10-15 追記・訂正）** 当初「34 バイトブロックによるセクター増幅が原因」と
  書いたが、ncu の DRAM カウンタで測ると実増幅は **1.26 倍**しかなく、dram 4〜28%・
  sm 14〜46%・l1tex 11〜46% で**どのパイプも 50% 未満**だった。律速は
  **帯域ではなく遅延を隠す並列度（warp 数 × 先行ロード数）**である。
  `docs/GEMV_VARIANT_SWEEP.md` の実測で、PIPE を深くする案も chunk を複数 warp に
  分割する案も**全形で劣化**することを確認した（現行のランチャー選択が全 11 形で最速）。
  残る候補は cp.async で行を shared に先読みする方式のみ。
- 小カーネル群（F32→F16 変換が 711 launch / 2.3 ms、量子化が 385 launch / 1.3 ms）は
  1 本 3 us 前後で**launch 律速**。融合して本数を減らすのが唯一の効き方。
- argmax / topk のような選択カーネルは ncu のコールド値が過大に出るので、
  **ncu の数字だけで最適化対象を決めない**こと（今回の実例）。

---

## 3. 注意

- ncu は `--cache-control all`（既定）で毎カーネル L2 を流すため、入力が L2 に載っている
  カーネルの時間は実測より大きく出る。合計がトークン実測より 9% 大きいのはこのため。
  DRAM から読む大きな GEMV については影響が小さく、順位は信頼できる。
- この計測は既定構成・cache 20,000 MiB・warm（2 反復目以降）で取得した。
