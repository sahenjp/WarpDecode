# Issue #1 検証結果 (2026-10-12, Colab L4 / SM89)

検証環境: Google Colab L4 (SM89, 23GB, driver 580.82.07, CUDA 13.0, cmake 3.31)。
86GB の GGUF は VM のローカルディスクに常駐させ、1 ラン約 12 秒で反復できる。
成果物は `issue1_artifacts/colab_gpu_verify.json`, `position_sweep.json`,
`position10_depth.json`, `position10_inject.json`。

## 1. ビルドと合成 CUDA テストは全て PASS

engine sha256 `21f1ea305f7f635fc926a987342ddfe0eae7f2db134d30fb30bf1418989906a8`。

- lowbit / qwen38-moe / qwen38-qsa / issue1-moe-precision … PASS
- `tests/qwen38_gdn_cuda_test.cu` と `tests/qwen38_hc_cuda_test.cu` は **CMake 未登録の孤立テスト**で、
  F32 残差化により HC テストはコンパイル不能だった。両方を CMake に登録し、テストを新 API に追従させて PASS にした。

## 2. 精度 ablation はアーキ依存（前回結論の撤回）

pos36 期待 56212 / pos39 期待 264。

| variant | A100 pos36 | A100 pos39 | L4 pos36 | L4 pos39 |
|---|---|---|---|---|
| baseline | 8206 | 449 | 8206 | 449 |
| residual_only | 8206 | 449 | 8206 | 449 |
| moe_output_only | 7650 | 449 | **56212** | 449 |
| hc_only | 9493 | 449 | 9493 | 449 |
| res_hc | **56212** | 449 | 9493 | **264** |
| full | 4246 | 279 | 4246 | **264** |

A100 で効いた res_hc は L4 では pos36 を落とす。逆に L4 では moe_output_only が pos36 を通す。
**単一スイッチで両アーキ・両位置を満たす構成は無い**ため、精度スイッチは症状を動かしているだけ。

## 3. アーキ非依存の失敗点は「生成 index 10」

64トークン parity は A100 でも L4 でも first_mismatch=10。正規前置を強制しても再現し、
position 10 でエンジンは 1083（正解 4613）を出力する。

## 4. position sweep: 誤差は最初のトークン・最初の層から存在する

CPU 参照（`position39_cpu_layer<N>_output_all_positions.f32`、WDPTRC1 と bit 一致で検証済み）に対する層別 NRMSE:

| pos | L0 | L4 | L8 | L47 |
|---|---|---|---|---|
| 0 | 0.0141 | 0.0139 | 0.0126 | 0.0637 |
| 1 | 0.0100 | 0.0314 | 0.0281 | 0.2797 |
| 2 | 0.0194 | 0.0525 | 0.0472 | 0.2478 |
| 3 | 0.0159 | 0.0321 | 0.0385 | 0.3032 |
| 5 | 0.0147 | 0.0314 | 0.0423 | 0.3325 |
| 10 | 0.0119 | 0.0308 | 0.0403 | 0.4249 |

**position 0（履歴なし）でも layer 0 が既に 1.4% ずれ、1 回の forward で L47 で 6.4% まで増幅する。**
つまり状態ドリフトだけでなく、層内の数値不一致そのものが主因。履歴が伸びるほど増幅する。

## 5. position-10 CPU 注入スイープは非単調（局所化に使えない）

| 注入層 L | token10 |
|---|---|
| baseline | 1083 |
| 1 / 3 / 4 / 9 / 17 | 1083 |
| 5 / 33 / 41 | **4613** |

単調でない = position 10 の argmax は僅差で決まっており、
途中の値を 1 つ変えるだけで反転する。トークン一致は崖っぷちの指標。

## 6. router 重みの「6.8 倍差」は比較ミスだった（容疑者解除）

CPU オラクルは `ffn_moe_weights`（raw softmax, 合計 0.148）と
`ffn_moe_weights_norm`（正規化, 合計 1.0）の両方を記録している。

position39 layer8 での実測比 warp / cpu_norm = 0.987〜1.016（≈1%）。
エンジンの重み計算は CPU の正規化済み重みと一致しており、
以前報告された 6.8 倍差は **raw と normalized の取り違え**による見かけの値だった。

## 7. 含意

- 残る差は特定のロジックバグではなく、**F16 中間値中心のパイプラインと F32 CPU 参照の忠実度差**が
  48 層で複利成長したもの、という説明が最も整合する。
- したがって「トークン完全一致」を目標にする限り、現構成では到達が難しい。
  選択肢は (a) 活性パスを F32 へ寄せて CPU と同条件にする、(b) parity 目標を
  logit 許容差ベースへ変更する、のいずれか。

## 8. 次の一手（候補）

1. Colab VM 上で pinned llama.cpp + `qwen38_forced_cpu_trace.cpp` をビルドし、
   **position 0 の CPU 内部ベクトル**（hc_* / gdn_* / ffn_*）を取得して layer-0 の誤差を区画別に帰属する。
   VM は 12 vCPU / 52GB RAM / 空き 114GB で、mmap 実行なら現実的。
2. layer-0 の誤差がどのブロック（HC mix / GDN / MoE / 埋め込み）で生じるかを確定させ、
   そこだけ F32 化する（速度影響を最小化）。
3. 並行して parity 目標そのもの（完全一致 vs 許容差）を決める。


---

# 追記 (2026-10-12 後半): 根本原因の特定

## 9.1 CPU オラクルを Colab VM 上で取得

- `tools/issue1_cpu_oracle_colab.py`（llama.cpp pinned PR #28243 head を CPU ビルドし、
  `qwen38_forced_cpu_trace.cpp` をリンク）で position 0 / 10 の CPU 内部ベクトルを取得し、
  `issue1_artifacts/cpu_pos0_internals.bin` (33MB) に保存。
- 途中で判明した不具合: harness は stdout に JSON、stderr に llama.cpp の CR 進行ログを出すため、
  ストリームを統合すると JSON が壊れる。stderr を分離して解決（`run_harness()`）。
- 比較は `tools/issue1_compare_internals.py`（WDPTRC1 対応表 49 ルール、self-test 付き）。

**注意**: warp の state trace は `embedding.hc_init` を埋め込み書き込み**前**に取得しており常にゼロ。
正しい対応は CPU `hc_init` ↔ warp `layer0.input_residual`。

## 9.2 position 0（履歴ゼロ）の層内誤差予算

CPU オラクルと直接比較できるペアだけを使った NRMSE:

| stage | NRMSE |
|---|---|
| 埋め込み → 入力残差 | 0.02% |
| HC RMSNorm | 0.03% |
| HC gate / mix | 0.84% / 1.35% |
| GDN qkv / z / alpha / gate | 1.43% / 1.43% / 1.08% / 0.73% |
| GDN norm 出力 | 2.06% |
| attention 出力 | 2.35% |
| layer 0 出力 | 1.41% |
| layer 8 router logits | 0.34% |
| layer 8 MoE 出力 | 1.42% |

→ 誤差は **すべて低bit GEMV（行列積）で発生**。埋め込み・RMSNorm・要素演算は一致している。

## 9.3 原因: 活性の量子化方式が CPU と違う

同一の重み・活性で厳密値と比較する数値実験:

| 方式 | 厳密値に対する相対誤差 |
|---|---|
| F16 活性でドット積（warp の既定パス） | 0.017% |
| Q8_0 活性でドット積（llama.cpp CPU） | 0.583% |
| F16 と Q8_0 の差 | 0.583% |

llama.cpp CPU は量子化重みに対して **活性を Q8_0（32要素ブロック、整数ドット、ブロック別スケール）に量子化**する。
warp の既定パスは **F16 活性**。したがって観測された ~1% の差は
「warp が壊れている」のではなく「CPU 側の量子化方式を再現していない」ことが原因。

## 9.4 実機検証（L4, position 0, layer 0）

warp の ggml 互換 Q8_0 リプレイ経路を有効化して比較:

| stage | 既定 (F16 活性) | Q8_0 exact replay |
|---|---|---|
| hc_attn gate | 0.844% | **0.022%** |
| hc_attn mix | 1.346% | **0.037%** |
| gdn qkv | 1.427% | **0.201%** |
| gdn z | 1.429% | **0.196%** |
| attention 出力 | 2.347% | **0.871%** |
| layer0 出力 | 1.412% | **0.992%** |

## 9.5 エンドツーエンド確認

`--debug-hc-q8_0-ffn-from-layer 0 --debug-hc-q8_0-ffn-through-layer 47`
（HC FFN 全47層を ggml 互換 Q8_0 活性化へ置換）で:

- position 10 のトークン: **1083（既定・誤り） → 4613（正解）**
- 層 NRMSE はわずかな改善（L0 1.19% → 0.96%、L8 4.03% → 3.79%）だが、
  僅差で決まっていた argmax が正解へ戻る。

## 9.6 結論と修正方針

- 真因は **活性の量子化方式が ggml と不一致**であること。
  これまで試した F32 残差スイッチ（RESIDUAL_F32 / HC_F32 / MOE_F32）が
  効かなかった理由も説明できる（残差の表現を変えても GEMV の活性量子化は変わらない）。
- 修正方針: 量子化重みに対する GEMV を
  **ggml 互換の Q8_0 活性量子化 + 整数ドット + F32 スケール**へ統一する。
  エンジンには既に HC / QSA の exact カーネルがあり、Q8_0 量子化は CPU と byte 一致の実績がある。
  速度面でも int8 ドットは有利で、性能目標とも整合する。
- 対象範囲: HC down/up、GDN qkv/z/beta/alpha/out、QSA q/k/v/out、router（いずれも Q8_0 重み）、
  MoE 専門家（IQ2_S / IQ4_NL は型別カーネルが必要）、出力ヘッド（Q6_K）。


---

# 追記 2: 実装（既定パスの ggml 互換 Q8_0 活性化）

## 10.1 実装内容

`src/qwen38_forward.cuh`:

- `q38_hc_q8_0_enabled()` … 環境変数 `WARODECODE_HC_Q8_0`（**既定 ON**）
- `q38_hc_q8_0_compatible(w)` … 重みが Q8_0 かつ F32 norm 重みがあるときだけ適用
- HC **FFN** 分岐: ggml 互換の F32 RMSNorm → Q8_0 活性量子化 → 整数ドット GEMV（既存の exact カーネル）→
  ggml 互換 SiLU → Q8_0 → 整数ドット GEMV → F32 mix を既定化。
- HC **attn** 分岐も同じ経路を実装。ただし `WARODECODE_HC_Q8_0_ATTN`（既定 OFF）で opt-in。

## 10.2 効果（L4, 64トークン greedy parity）

| 構成 | 一致プレフィックス |
|---|---|
| 変更前（F16 活性） | 10 / 64 |
| HC FFN のみ Q8_0（既定） | **22 / 64** |
| HC attn + FFN Q8_0 | 10 / 64 |

## 10.3 HC attn を opt-in にした理由

attn 側も段階誤差は明確に改善する:

| stage | FFN のみ | attn + FFN |
|---|---|---|
| hc_attn mix | 1.346% | **0.030%** |
| gdn qkv | 1.427% | **0.202%** |
| gdn z | 1.429% | **0.198%** |
| attention 出力 | 2.347% | **0.871%** |

しかし end-to-end では 22 → 10 と悪化する。GDN / QSA / MoE / 出力ヘッドがまだ F16 活性のままで、
部分的に忠実なパイプラインは argmax の崖をどちら側にも越えうるため。
**残りを同じ方式へ揃えてから既定化する**のが正しい順序と判断した。

## 10.4 残作業（同じ方式へ揃える対象）

1. GDN: qkv / z / beta / alpha / out（36層、いずれも Q8_0 重み）＋ conv・再帰 state の F32 化
2. QSA: q / k / v / out / index_q / index_k
3. MoE: router（Q8_0）、shared expert（Q8_0）、専門家（IQ2_S / IQ4_NL は**型別カーネルが必要**）
4. 出力ヘッド（Q6_K）と PLE
5. 速度: exact カーネルはデバッグ品質（lane 0–7 のみ使用）なので、
   正しさを維持したまま DP4A / Tensor Core 版へ置き換える

## 10.5 実装メモ

- `tests/qwen38_hc_cuda_test.cu` は F32 残差 API に追従済み（CMake 登録も追加）。
- Colab CLI は長時間 `exec` で接続が切れることがあるため、
  **VM 側で nohup 実行 → 短い exec でポーリング**する方式が安定する。
- エンジンの再ビルドは `cmake --build build-sm89 --target warodecode` を毎回呼ぶ
  （キャッシュがあれば数十秒〜).


---

# 追記 3 (2026-10-13): GDN / QSA / MoE も ggml 互換へ — 一致 36/64

## 11.1 追加実装（subagent 3 本 + レビュー）

| 対象 | ファイル | 環境変数（既定） |
|---|---|---|
| GDN 投影（qkv/z/beta/alpha/out） | `src/qwen38_gdn.cuh` | `WARODECODE_GDN_Q8_0` (ON) |
| QSA 投影（q/k/v/out/index） | `src/qwen38_qsa.cuh` | `WARODECODE_QSA_Q8_0` (ON) |
| router / shared expert | `src/qwen38_moe.cuh` | `WARODECODE_ROUTER_Q8_0`, `WARODECODE_SHEXP_Q8_0` (ON) |
| HC attn | `src/qwen38_forward.cuh` | `WARODECODE_HC_Q8_0_ATTN` (OFF, 理由は §11.3) |

いずれも「F32 化 → ggml 互換の Q8_0 活性量子化 → 整数ドット GEMV →
（必要なら F32 のまま中間計算）→ half へ書き戻し」という同じ形。
条件を満たさない重み・形状では従来の F16 経路へフォールバックする。

## 11.2 64トークン greedy parity の推移（L4, 同一プロンプト）

| 構成 | 一致プレフィックス |
|---|---|
| 変更前（すべて F16 活性） | 10 / 64 |
| HC FFN のみ Q8_0 | 22 / 64 |
| **＋ GDN / QSA / router / shared expert** | **36 / 64** |
| ＋ HC attn も Q8_0（opt-in を有効化） | 22 / 64（後退） |

次の不一致は index 36（参照 `56212` に対しエンジン `8206`）。
これは以前から「pos36 ゲート」として知られていた箇所で、残る誤差源がここに集約されている。

## 11.3 HC attn を依然として opt-in にしている理由

GHZ: attn 側を有効化すると段階誤差は改善する（HC mix 1.346% → 0.030%）が、
end-to-end では 36 → 22 に後退する。理由は未解明だが、
attn 分岐の inject 重みが F16 経路から供給されており（Q8_0 経路は inject を計算しない）、
混合状態が ggml と一致しないためと考えられる。**inject も含めて忠実化してから既定化する。**

## 11.4 残る誤差源（優先順）

1. **routed experts（IQ2_S / IQ4_NL）** … 実装着手済み。
   llama.cpp CPU では IQ2_S の活性は **Q8_K**（Q8_0 ではない）、IQ4_NL は Q8_0。
   warp の unpack・テーブルは**既に ggml と byte 一致**しており、差は活性表現だけ。
2. **出力ヘッド（Q6_K）** … CPU は Q8_K 活性。最終 logits を直接決めるため argmax への影響が最大。
3. PLE 投影、ブロック境界の F32 保持（HC mix → GDN/QSA 入力が half に丸められている）。
4. 速度（exact カーネルはデバッグ品質で 0.87 tok/s。DP4A / Tensor Core 版へ置換が必要）。

## 11.5 調査で判明した重要な訂正

- **IQ2_S の活性は Q8_0 ではなく Q8_K**（256要素ブロック、符号付き max、F32 スケール、bsums）。
- llama.cpp の **CUDA** 実装は Q8_1 を使い整数除算で切り捨てるが、
  **CPU オラクルに合わせるには真似してはいけない**。
- Q6_K の vec_dot_type も Q8_K。
- warp の IQ2_S/IQ4_NL unpack は ggml と完全一致（grid テーブル・signs・index 式・ls 分割・nibble 対応を検証済み）。
  したがって専門家カーネルの改修は「活性側だけ」で足りる。


---

# 追記 4 (2026-10-13 夜): per-type 活性の原則と F32 重み経路

## 12.1 モデルの実テンソル型（GGUF から実測）

| テンソル | 型 | 備考 |
|---|---|---|
| blk.N.attn_qkv / attn_gate(z) / ssm_out | Q8_0 | GDN の主投影 |
| **blk.N.ssm_beta / ssm_alpha** | **F32** | 量子化されていない |
| blk.N.attn_q/k/v/output | Q8_0 | QSA の主投影 |
| **blk.N.indexer.q_proj / k_proj** | **BF16** | 量子化されていない |
| **blk.N.ffn_gate_inp（router）** | **F32** | 量子化されていない |
| blk.N.ffn_gate/up_exps | IQ2_S | → 活性 Q8_K |
| blk.N.ffn_down_exps | IQ4_NL | → 活性 Q8_0 |
| blk.N.ffn_*_shexp | Q8_0 | |
| blk.N.hc_*_down / up | Q8_0 | |
| **blk.N.hc_*_inject** | **F32** | |
| blk.1.ple_key / ple_value | Q8_0 | |
| token_embd | Q8_0 | |
| output（ヘッド） | Q6_K | → 活性 Q8_K |

## 12.2 原則（ggml）

**活性を量子化するのは量子化重み（Q8_0 / IQ* / Q6_K）のときだけ。**
F32 / F16 / BF16 の重みは **F32 活性のまま**行列積する（vec_dot_type = F32）。

したがって「全投影が Q8_0 でなければフォールバック」という実装は誤りで、
**投影ごとに型を見て切り替える**必要がある。実際、
`q38_gdn_q8_0_compatible()` が beta/alpha まで Q8_0 を要求していたため、
**GDN の ggml 互換経路が丸ごとフォールバックしていた**（gdn qkv が 1.43% のまま）。

## 12.3 実装済み（本追記時点）

- `src/half_tensor.cuh`: `DeviceFloatArray` + `upload_float_array()`（F32 テンソルを F32 のまま保持）
- `src/matrix.cuh`: `GpuMatrix::dense_f32`、`dense_f32_gemv_kernel`、
  `launch_matrix_gemv_f32_weights()`、`matrix_has_f32_weights()`
- `src/qwen38_moe.cuh`: router が F32 のとき F32 活性 × F32 重みの GEMV を使う
  （`WARODECODE_ROUTER_F32` 既定 ON）

## 12.4 分かったこと: router の誤差は入力由来

router を F32 化しても logits の CPU 比誤差は 0.29% のままだった（値の差は 5e-7）。
これは **誤差が重み・活性の丸めではなく、router への入力（HC FFN mix）のずれで決まっている**ことを意味する。
= 残作業の優先度は「入力チェーン（HC attn / GDN / QSA / ブロック境界）を忠実化する」ことにある。

## 12.5 進行中の作業（subagent 並列）

| 担当 | ファイル | 内容 |
|---|---|---|
| A | lowbit.cuh + warodecode.cu | 出力ヘッド Q6_K × Q8_K |
| B | qwen38_hc.cuh | inject を F32 で計算 |
| C | qwen38_ple.cuh | PLE 射影の ggml 互換化 |
| D | qwen38_gdn.cuh | GDN の per-type 化（beta/alpha は F32 GEMV） |
| E | qwen38_qsa.cuh | QSA の per-type 化 + indexer(BF16) の F32 化 |

**運用上の注意**: subagent が書き込み中のツリーでビルドすると破損スナップショットを拾う
（実際に 1 回、第一トークンから崩れた出力を得た）。ビルド前に全ファイルの mtime が安定していることを確認する。


---

# 追記 5 (2026-10-13 深夜): 全コンポーネントの ggml 互換化と F32 化へ

## 13.1 実装済み（subagent 5本 + 統合）

| 対象 | ファイル | 環境変数（既定） | 検証 |
|---|---|---|---|
| 出力ヘッド Q6_K × Q8_K | lowbit.cuh + warodecode.cu | `WARODECODE_HEAD_Q8_K` (ON) | ggml AVX2 と **2,598,531 dot で不一致 0** |
| HC inject F32 | qwen38_hc.cuh | `WARODECODE_HC_INJECT_F32` (ON) | ggml と **384/384 bit 一致** |
| HC attn Q8_0 + F32 inject 配線 | qwen38_forward.cuh | `WARODECODE_HC_Q8_0_ATTN` (**既定 ON へ変更**) | HC mix 誤差 1.35% → 0.030% |
| GDN per-type（beta/alpha は F32） | qwen38_gdn.cuh | `WARODECODE_GDN_Q8_0` (ON) | 以前は条件不一致で**丸ごとフォールバックしていた**のを修正 |
| QSA per-type + indexer F32 | qwen38_qsa.cuh | `WARODECODE_QSA_Q8_0` (ON) | — |
| indexer の **BF16 活性丸め** | qwen38_qsa.cuh | `WARODECODE_QSA_INDEX_BF16` (ON) | ggml の traits は BF16 重みで活性を BF16 に丸める |
| router F32 重み経路 | matrix.cuh, half_tensor.cuh, qwen38_moe.cuh | `WARODECODE_ROUTER_F32` (ON) | F32 テンソルを F16 に落としていたのを修正 |
| PLE ggml 互換化 | qwen38_ple.cuh | `WARODECODE_PLE_Q8_0` (ON) | 9 項目のホスト bit 比較で不一致 0 |
| routed experts IQ2_S×Q8_K / IQ4_NL×Q8_0 | lowbit.cuh, qwen38_moe.cuh | `WARODECODE_EXPERT_Q8_ACT` (ON) | **81,280 / 10,240 dot で不一致 0** |

## 13.2 効果（L4 実機）

position 0, layer 0 の段差（CPU オラクル比 NRMSE）:

| stage | 変更前 | 現在（attn ON） |
|---|---|---|
| hc_attn mix | 1.346% | **0.030%** |
| gdn qkv | 1.427% | **0.085%** |
| attention 出力 | 2.347% | **0.386%** |
| hc_ffn mix | 1.55% | 0.826% |
| layer0 出力 | 1.481% | **0.366%** |

position 10 の層別（attn ON）: L0 0.36% / L4 2.73% / L8 2.35% / L16 5.0% / L32 14.7% / L47 29.0%

速度: **4.21 tok/s**（F16 高速パス 2.38 tok/s を大きく上回る。int8 化は速度にも有利）

## 13.3 残る課題

- 64トークン parity は依然 10/64。**position 10 の logits マージンは -0.16（相対 0.9%）**しかなく、
  現状の累積誤差（L47 で 29%）では届かない。
- 誤差の主因は「**中間バッファが __half のため 1 演算ごとに丸めが入る**」こと。
  ggml は中間値を F32 のまま扱う。
- 対策として GDN / QSA / MoE の中間バッファを **F32 に統一**する作業を開始（subagent 3本）。
  丸めてよいのは ggml も丸める箇所（量子化重みの活性量子化、F16 KV キャッシュ、最終出力）だけ。

## 13.4 運用メモ

- subagent が書き込み中のツリーでビルドすると壊れたバイナリを作る。**ビルド前に mtime の安定を確認**する。
- Colab CLI の exec は詰まることがあるため、**HTTP ダウンロードでのポーリング**が安定。
- L4 セッションは 21 時間で回収される。再作成時はモデル 86GB の再取得（~5 分, 300MB/s）が必要。


---

# 追記 6 (2026-10-14 未明): F32 化の統合と最良構成 39/64

## 14.1 追加実装（subagent 3本 + 統合）

| 対象 | ファイル | 環境変数（既定） |
|---|---|---|
| GDN 中間バッファ F32 + conv 履歴 F32 | qwen38_gdn.cuh | `WARODECODE_GDN_F32_MID` (ON) |
| QSA 中間 F32（index_raw / pooled / attention 含む） | qwen38_qsa.cuh | `WARODECODE_QSA_F32_MID` (ON) |
| MoE 中間 F32（per-expert の丸め除去） | qwen38_moe.cuh | `WARODECODE_MOE_F32_MID` (ON) |
| F32 block_out 用 HC combine | qwen38_hc.cuh | （内部 API） |
| GDN/QSA/MoE の F32 出力配線 | qwen38_forward.cuh | `WARODECODE_GDN_F32_OUT` / `QSA_F32_OUT` (ON), `MOE_F32_OUT` (opt-in) |

**検証**: GDN は AR の state が half 版と bit 一致、MoE は 100,000 サンプルで不一致 0、
QSA は CPU トレースから indexer の score / pooled / k_norm の演算順を逆算して一致を確認。

## 14.2 64トークン parity の推移（L4, 同一プロンプト）

| 構成 | 一致 |
|---|---|
| 変更前（すべて F16 活性） | 10 / 64 |
| HC FFN の Q8_0 化 | 22 / 64 |
| ＋GDN/QSA/router/shared expert | 36 / 64 |
| ＋専門家 Q8_K（F16 出力配線） | 10 / 64 |
| ＋出力ヘッド Q6_K / PLE / BF16 indexer / attn 既定 ON | 10 / 64 |
| ＋F32 中間バッファ（配線なし） | 11 / 64 |
| **＋GDN/QSA の F32 出力配線（MoE 配線は opt-in）** | **39 / 64** |

次の不一致は index 39（参照 `264` に対し `449`）＝ **当初の Issue #1 のターゲット位置**。
速度は 4.11 tok/s（F16 高速パス 2.38 tok/s の 1.7 倍）。

## 14.3 F32 出力配線の A/B（トークン一致は崖っぷち指標）

| 構成 | 一致 |
|---|---|
| 全配線 ON | 6 |
| QSA 配線 OFF | 36 |
| GDN 配線 OFF | 22 |
| **MoE 配線 OFF** | **39** |
| 全配線 OFF | 10 |
| MoE+GDN OFF | 10 |

非単調であり、**MoE の F32 出力配線には未発見の不整合がある**（カーネル自体は有効）。
既定は「MoE 配線 OFF・GDN/QSA 配線 ON」。MoE 配線は `WARODECODE_MOE_F32_OUT=1` で opt-in。

## 14.4 残る課題

1. **MoE F32 出力配線のデバッグ**（39 → さらに上を狙える）
2. ブロック境界の F32 化（HC mix 出力 `cur` が half）
3. QSA の indexer RoPE base 仮説（CPU は **base=1e7** の可能性。position ≥ 2048 でしか効かないため未検証）
   → `WARODECODE_QSA_INDEX_ROPE_BASE=1e7` で長文脈時に検証する
4. 残差の蓄積そのものを 1 桁下げる（margin は logits で相対 0.9%）

## 14.5 現在の構成まとめ（既定）

- 量子化重み（Q8_0/IQ2_S/IQ4_NL/Q6_K）→ ggml 互換の活性量子化 + 整数ドット（多くのカーネルで bit 一致を確認）
- 非量子化重み（F32/BF16）→ F32 活性（BF16 重みは活性を BF16 に丸める）
- 中間バッファ → F32（GDN/QSA/MoE カーネル、HC の exact 経路）
- F32 出力配線 → GDN/QSA は ON、MoE は opt-in
- KV キャッシュ → F16（CPU と同一条件）


---

# 追記 7 (2026-10-14): position 39 まで到達 — 残り 0.245

## 15.1 現在のベスト構成

- 既定（全 ggml 互換カーネル ON + F32 中間 ON + GDN/QSA の F32 出力配線 ON + MoE 配線 OFF）
- **一致 39 / 64**、次の不一致は **index 39**（参照 `264` に対し `449`）
- 速度 **4.11 tok/s**（F16 高速パス 2.38 の 1.7 倍）
- **index 36 のゲート（56212）は正解** — 以前の最難関を通過

## 15.2 残距離の定量（position 39 の logits）

| | margin(264 − 449) |
|---|---|
| CPU オラクル | **+4.088** |
| warp（現在） | **−0.245** |
| 差分 | −4.333 |

warp の margin が正になれば index 39 は正解になる。**必要な改善は 0.245（logits の 1.3%）**。
当初の baseline は −8.68 だったので、誤差は 35 倍改善している。

## 15.3 position 39 の層別プロファイル（CPU 比 NRMSE）

```
L0 0.003  L8 0.022  L16 0.048  L24 0.084  L32 0.140  L40 0.201  L47 0.299
```

単一の支配的な段差はなく、**全層で滑らかに増大**する（1 層あたり ~1.05〜1.08 倍）。

## 15.4 なぜ一致が難しいのか（本質）

**活性の量子化はステップ関数**である。warp は今も多くの活性を __half で持ち回っており、
その丸め（相対 2.6e-4）が CPU の F32 活性との差を生む。量子化のしきい値をまたぐと
1 ブロックの量子化値が 1/127 だけ飛ぶ（ブロック最大値の ~0.8%）。
これが層ごとに複利で増幅し、48 層で ~30% に達する。

つまり **64/64 の一致には「活性チェーンの全段を bit 一致」にすることが必要**。
行列積そのものは既に ggml と bit 一致している（Q8_0/Q8_K/IQ2_S/IQ4_NL/Q6_K で数百万 dot の不一致 0）ので、
残るのは以下:

1. **__half を経由する活性**（HC mix 出力 → GDN/QSA 入力、各ミラー、トレース用バッファ）
   → HC mix 出力を F32 のまま GDN/QSA へ渡す改修を進行中
2. RMSNorm の縮約順、sigmoid/SiLU の expf 実装差（CUDA vs glibc、~2 ulp）
3. attention softmax の縮約順
4. 残る F16 キャッシュ境界（KV は CPU も F16 なので一致 ✓）

## 15.5 実装済みの環境変数（既定）

`WARODECODE_{HC_Q8_0, HC_Q8_0_ATTN, HC_INJECT_F32, GDN_Q8_0, GDN_F32_MID, GDN_F32_OUT, QSA_Q8_0,
QSA_F32_MID, QSA_F32_OUT, QSA_INDEX_BF16, ROUTER_Q8_0, ROUTER_F32, SHEXP_Q8_0, EXPERT_Q8_ACT,
MOE_F32_MID, PLE_Q8_0, HEAD_Q8_K}` = ON、
`WARODECODE_MOE_F32_OUT` = opt-in（配線に未解決の不一致があり、ON にすると一致が 39 → 6 に落ちる）。


---

# 追記 8 (2026-10-14 未明): 段差は同等・トークン一致は崖っぷち

## 16.1 本ラウンドでやったこと

1. **F32 入力版エントリ**を GDN/QSA に追加（`run_qwen38_gdn_f32_x` / `run_qwen38_qsa_f32_x`）
   — HC mix 出力を half を経由せず射影へ渡す。配線は `attn_f32_inject` を条件に有効化。
2. **出力ヘッドの F32 入力版**（`launch_q6_k_head_gemv_q8_k_from_f32`）を追加
   — logits 直前の half 丸めを除去。`WARODECODE_HEAD_F32_IN`（既定 ON）。

## 16.2 測定結果（いずれも L4 実機）

| 構成 | 一致 | position39 margin |
|---|---|---|
| 変更前（39/64 を記録したビルド） | 39 | −0.245 |
| F32 入力配線 ON | 22 | −0.742 |
| F32 入力配線 OFF（現状）+ ヘッド F32 ON | 22 | −0.742 |
| ヘッド F32 OFF（A/B） | 22 | — |

**ヘッド F32 入力は数値に影響なし**（`head_f32` は F32 HC head mix が走らない既定構成では null）。
**F32 入力配線は結果を変えたが、マージンは僅かに悪化**した。

## 16.3 段差（position 0, layer 0）の比較

| stage | 旧ビルド | 現ビルド |
|---|---|---|
| gdn qkv | 0.085% | **0.061%** |
| gdn norm_out | 0.113% | **0.083%** |
| attention 出力 | 0.386% | **0.335%** |
| hc_combine | 0.397% | **0.347%** |
| hc_ffn mix | 0.826% | 0.949% |
| moe out | 0.682% | 0.756% |
| layer0 出力 | 0.366% | 0.403% |

GDN 系は改善、MoE/FFN 系は同等〜微悪。**トークン一致 39 → 22 の差は段差の大きさでは説明できない**
（層出力の誤差プロファイルは L47 で 29.9% vs 28.7% とほぼ同一）。
= 量子化コード反転の「どちら側に転ぶか」の差であり、**一致数は崖っぷち指標**であることが再確認された。

## 16.4 結論（本ラウンド時点）

- カーネル単位の忠実度は既に ggml と同水準（主要な量子化ドット積は bit 一致）。
- 残差は「活性チェーンの丸め」に起因する**離散的な量子化コード反転**の累積で、層ごとに ~1.05〜1.08 倍に増える。
- 64/64 には活性チェーン全体を bit 一致（差 < 1e-6）にする必要があり、
  1e-4 レベルの差ではコード反転がランダムに起こる。
- **速度は 4.1 tok/s**（F16 高速パス 2.38 の 1.7 倍）で、正しさと速度は両立している。

## 16.5 次にやるべきこと（優先順）

1. **活性チェーンの bit 一致化を体系的に進める**:
   - half を経由する経路の洗い出し（`cur`、router/shexp/MoE 入力、各ミラー、トレース用バッファ）
   - RMSNorm の縮約順を ggml の AVX2 と一致させる
   - sigmoid/SiLU の `expf` を glibc と bit 一致させる
2. MoE F32 出力配線の再検証（旧コードでは 39 → 6 に低下。配線の前提を再確認）
3. 長文脈での indexer RoPE base 検証（`WARODECODE_QSA_INDEX_ROPE_BASE=1e7`、position ≥ 2048）
4. 速度: DP4A/Tensor Core 化と expert cache チューニング


---

# 追記 9 (2026-10-14 早朝): stale バイナリ問題の発見と修正

## 17.1 重大なプロセス不具合（発見・修正済み）

**症状**: 出力ヘッドの F32 入力化を追加した時点でコンパイルエラーが発生し、
`tools/colab_run_bg.py` がビルド失敗を検知せず**古いバイナリ**で実行を続けていた。
その結果、01:05〜02:10 の測定（p14b / p15 / p16 / 切り分け r_* / diag / pos39_cur）は
すべて同一の古いバイナリによるもので、**コード変更の効果を測っていなかった**。

**原因**: `rt_.output` は `LowBitMatrix` なのに `GpuMatrix` 前提のコードを書いた（`.lowbit`/`.low` 参照）。
加えて `launch_q6_k_head_gemv_q8_k_from_f32` に `threads` 引数を渡し忘れ。

**再発防止**: ランナーを修正し、`BUILD_EXIT != 0` のときエンジンを実行せず
`BUILD_FAILED` を報告するようにした（`tools/colab_run_bg.py`）。

## 17.2 実ビルドでの再測定（修正後）

| 構成 | 一致 | 速度 |
|---|---|---|
| 既定（F32-x 有効・ヘッド F32・MoE 配線 off） | **22 / 64** | 4.19 tok/s |
| MoE F32 出力配線 ON | 22 / 64 | — |

**実ビルドでは MoE 配線の有無でトークン列が完全に一致**した。
以前「39 → 6 に悪化」と報告したのは、ビルドをまたいだ比較であり、
stale バイナリの影響を受けていた可能性がある（当時の A/B 自体は同一バイナリ内だったが、
その後のコード変更で崖っぷちの位置が移動している）。

## 17.3 F32 入力（HC mix → GDN/QSA）の検証

- 有効条件を「F32 inject が使えたか」から「**F32 mix が走ったか**」に修正
  （`attn_f32_row_valid`）。既定構成では F32 inject がロードされないため、
  従来の条件では F32 入力経路が一度も使われていなかった。
- 診断プリントで確認: `f32row[0]=-0.226163357` / `halfmirror[0]=-0.226196289`（差 3.3e-5）
  → F32 行は half ミラーと異なり、**経路は実際に有効**。
- トークン列は F32 入力の有無で同一。段差トレースは half ミラーを読むため差が現れない。

## 17.4 現在の確定状態

- 実ビルドで **22 / 64**、速度 **4.19 tok/s**（F16 高速パス 2.38 の 1.76 倍）
- 全 ggml 互換カーネル（HC FFN/attn・GDN・QSA・router・shexp・experts・head・PLE）が有効
- F32 中間バッファ + F32 出力配線（GDN/QSA）+ F32 入力（GDN/QSA）が有効
- 最良記録は 39/64（旧ビルド）。現在のビルドでは再現しない（崖っぷち指標のため）

## 17.5 教訓（今後に効く）

1. **ビルド失敗を必ず検知する**（今回の最大の学び）。
2. **トークン一致数は単独の判断指標にしない**。段差（NRMSE）と併記する。
3. 段差トレースが half ミラーを読む項目は、F32 経路の検証には使えない。


---

# 追記 10 (2026-10-14 未明): 種誤差の発見 — attention 分岐が CPU と bit 一致

## 18.1 発見: 埋め込みの half 丸めが全誤差の seed だった

**症状の手がかり**: layer 0 の出力誤差が position 10 でも 39 でも **0.394% と完全に一定**だった。
位置に依存しない = 各トークンの先頭で注入される誤差がある、と気づいた。

**原因**: トークン埋め込み（Q8_0）を `__half` バッファに gather してから F32 の残差へ展開していた。
ggml は dequantize した F32 値をそのまま残差に入れる。half 丸め（相対 ~5e-4）が
**毎トークンの残差の種**になっていた。

**修正**: `launch_q8_0_gather_row_f32` + `q38_repeat_embedding_from_f32_kernel` を追加し、
F32 の埋め込み行から残差を構築（`WARODECODE_EMBED_F32`、既定 ON）。

## 18.2 効果（L4 実機）

| 指標 | 修正前 | 修正後 |
|---|---|---|
| **64トークン一致** | 22 / 64 | **36 / 64** |
| layer 0 出力（pos 0, NRMSE） | 0.394% | **0.091%** |
| HC mix（pos 0, F32 比較） | — | **0.00000（bit 一致）** |
| GDN qkv / z / norm_out / final（pos 0） | — | **すべて 0.00000（bit 一致）** |
| **attention 分岐出力（pos 0）** | — | **0.00000（bit 一致）** |

**layer 0 の attention 分岐全体（HC mix → GDN → 活性量子化 → 整数ドット → 出力）が
CPU オラクルと bit 一致した。** これは「量子化重み + ggml 互換活性量子化」の方針が
完全に正しいことの決定的な証拠である。

## 18.3 残る seed: MoE 分岐

layer 0（pos 0）の MoE 分岐の段差:

| stage | 誤差 |
|---|---|
| router logits | 0.0044% |
| expert weights | 0.0069% |
| **ffn out** | **0.133%** |
| layer out（MoE 込み） | 0.091% |

MoE への入力 `cur` が __half のため、専門家の活性量子化（Q8_K / Q8_0）が
CPU の F32 活性と食い違い、コード反転で ~0.13% の誤差を生む。
→ MoE の F32 入力版を実装中（`run_qwen38_routed_moe_*_f32_x` / `run_qwen38_shared_expert_f32_x`）。

## 18.4 A/B（同一ビルド・実測）

| 構成 | 一致 |
|---|---|
| F32 埋め込み（既定） | **36** |
| F16 埋め込み（`WARODECODE_EMBED_F32=0`） | 22 |
| ＋MoE F32 出力配線（`WARODECODE_MOE_F32_OUT=1`） | 10 |

MoE F32 出力配線は layer 0 の誤差をほぼ変えない（0.000913 → 0.000911）のに
トークンは大きく振れる = 依然として崖っぷち指標である。

## 18.5 方法論の修正

これまでの段差測定は `copy_half` 経由のトレースを読んでおり、
**half 丸め（~5e-4）が下限**になっていた。今回 `*_f32` のトレース項目を追加して
真の誤差（1e-6 台）を測れるようにした。これが上記の発見につながった。

教訓: **「half ミラーを読むトレースで F32 経路の改善は測れない」**。


---

# 追記 11 (2026-10-14 早朝): F32 入力化の完遂 — layer 0/1 が bit 一致

## 19.1 MoE の F32 入力化（実装・配線・検証）

`src/qwen38_moe.cuh` に F32 入力版を追加し、forward.cuh で配線した:
- `run_qwen38_routed_moe_separate_f32_x` / `_fused_f32_x` / `run_qwen38_shared_expert_f32_x`
- F32 入力スイッチ `moe_f32_in` は出力契約と独立（`WARODECODE_MOE_F32_OUT` とは別）
- ルータ・専門家の活性量子化・shared expert の gate 内積まで **F32 のまま**流す

## 19.2 検証結果（position 0, layer 0 — F32 vs F32 比較）

| stage | MoE 入力 half | MoE 入力 F32 |
|---|---|---|
| router logits | 0.000044 | **0.000000** |
| expert weights | 0.000069 | **0.000000** |
| ffn out | （未計測） | **0.000000** |
| **layer 0 出力** | 0.000913 | **0.000000** |

**layer 0 が完全に bit 一致した。**

## 19.3 層別の bit 一致状況（position 0, F32 埋め込み + MoE F32 入力）

```
L0  = 0.00000005   ← bit 一致
L1  = 0.00000008   ← bit 一致
L2  = 0.00137790   ← ここで崩れる
L3  = 0.00180143
L4  = 0.00269526
L5..L11 = 0.0016〜0.0026
```

**exactness は layer 2 で崩れる。** layer 0 と layer 2 はどちらも GDN+MoE 層で構造は同一、
入力（L1 出力）も bit 一致しているのに、L2 だけ 0.1378% の誤差が出る。
layer 2 固有の差分を次に特定する必要がある（CPU オラクルの内部値が layer 0/7/8 しか無いため、
layer 2 用の追加キャプチャが有効）。

## 19.4 トークン一致（崖っぷち指標の実例）

| 構成 | 一致 |
|---|---|
| F16 埋め込み | 22 |
| **F32 埋め込み（+F16 MoE 入力）** | **36** |
| F32 埋め込み + **MoE F32 入力**（現状・忠実度は最良） | 22 |

MoE F32 入力は **layer 0 を bit 一致にする**（＝原理的に正しい）が、トークン一致は 36 → 22 に下がる。
これは崖っぷちの反転であり、**忠実度（段差）とトークン一致は別指標**であることの決定的な実例。
今後は段差を主指標にし、トークン一致は最終確認に使う。

## 19.5 position 39 の層別誤差（F32 埋め込み + MoE F32 入力）

| layer | 旧 | 現在 |
|---|---|---|
| L0 | 0.394% | 0.225% |
| L8 | 2.08% | 2.24% |
| L24 | 9.17% | 8.24% |
| L32 | 14.3% | 13.1% |
| L47 | 28.7% | 26.6% |

position 39 では L0 が 0.225%（position 0 では bit 一致）— これは **GDN/conv の状態が
過去位置のドリフトを運ぶ**ため。per-position の seed ではなく状態経由の累積である。

## 19.6 次の一手

1. **layer 2 の 0.1378% の原因特定**（layer 0 と構造が同じなのに崩れる理由）
   - CPU オラクルに layer 2 の内部値を追加キャプチャするのが最短
2. 状態経由の累積（position 依存）の遮断
3. 段差を主指標にした反復（トークン一致は最終確認）


---

# 追記 12 (2026-10-14 早朝): 根本原因 = blk.2 の専門家が IQ3_S だった

## 20.1 発見の経緯

1. position 0 で layer 0/1 は **bit 一致**、layer 2 から 0.1378% の誤差が出ることを確認
2. CPU オラクルに layer 0〜3 の内部値を追加キャプチャ（`WARODECODE_CPU_ORACLE_LAYERS=0,1,2,3`）
3. layer 2 の段差を測ると **GDN・HC・router・expert weights はすべて bit 一致**、
   しかし **選択エキスパート出力（10エキスパート全部）が 1.7% ずれ**ていた
4. エキスパートキャッシュ量を変えると誤差の出る層が移動 → データ/経路の問題と判断
5. **GGUF のテンソル型を全層ダンプ**した結果:

```
blk.0..1   gate=IQ2_S  up=IQ2_S  down=IQ4_NL
blk.2      gate=IQ3_S  up=IQ3_S  down=IQ4_NL   ← ここだけ IQ3_S
blk.3..47  gate=IQ2_S  up=IQ2_S  down=IQ4_NL
blk.48     gate=Q8_0   up=Q8_0   down=Q8_0     （MTP）
```

**47 層は IQ2_S、blk.2 の 1 層だけ IQ3_S。**
warp の IQ3_S は F16 活性の旧経路（`iq3_s_gemv_kernel`）しか無く、
ggml 互換の **IQ3_S × Q8_K** 整数ドットが未実装だった。
→ layer 2 の専門家出力が 1.7% ずれ、それが以降の全層に伝播していた。

## 20.2 意味

- layer 0/1 が bit 一致していたのは「IQ2_S 経路が完全に正しい」ことの証明 ✓
- 誤差の蓄積は「1 層の未実装パス」が原因であり、**設計の誤りではない**
- IQ3_S × Q8_K を実装すれば、48 層すべてが bit 一致する可能性が高い
  → **64/64 トークン一致の射程に入った**

## 20.3 実装中

- `src/lowbit.cuh`: `iq3_s_q8_k_gemv_kernel`（ggml の `ggml_vec_dot_iq3_s_q8_K` AVX2 と bit 一致を目標）
- `src/qwen38_moe.cuh`: 専門家パスで IQ3_S を IQ2_S と同じ活性型（Q8_K）として扱う

## 20.4 診断用に追加したもの

- `WARODECODE_EXPERT_CACHE_MIB`（`src/qwen38_tuning.hpp`）: 専門家キャッシュ量の上書き。
  閾値依存の事象を切り分けるのに有効。
- 段差トレースの F32 版（`*_f32`）: half ミラー（~5e-4）に埋もれない真の誤差が見える。
- CPU オラクルの層指定キャプチャ（`WARODECODE_CPU_ORACLE_LAYERS`）。

## 20.5 教訓

**「1 層だけ bit 一致しない」ときは、まずモデルのその層のテンソル型を疑う。**
量子化形式が他層と違うと、実装済みの型の経路から外れて旧経路に落ちる。


---

# 追記 13 (2026-10-14 朝): IQ3_S 修正の効果 — position 0 で全48層が bit 一致

## 21.1 IQ3_S × Q8_K の実装結果

`src/lowbit.cuh` に `iq3_s_q8_k_gemv_kernel` を追加（ggml `ggml_vec_dot_iq3_s_q8_K` AVX2 の忠実な転写）。
ホスト検証: **51,520 dot / bit 不一致 0**（integer bsum 照合 14,336 superblock / 0、負の対照で検出力も確認）。
`src/qwen38_moe.cuh` は gate/up の型述語を IQ2_S|IQ3_S に拡張し、down は IQ4_NL のまま。

## 21.2 効果（実機 L4）

| 指標 | 修正前 | 修正後 |
|---|---|---|
| layer 2 の専門家出力（pos 0, CPU比） | 1.73% | **0.021%**（half ミラー相当 = bit 一致） |
| layer 2 の ffn out | 0.87% | **0.00000015** |
| **全 48 層の layer-out（pos 0）** | L2 から 0.14%〜 | **全部 1e-7（bit 一致）** |
| 64トークン一致 | 22 | 22（不一致位置は index 10 → **index 22** に前進） |

**position 0 において、48 層すべてが CPU オラクルと bit 一致した。**
これは「量子化重み + ggml 互換活性量子化 + F32 中間 + F32 入力」の設計が
完全に正しいことの最終的な証明である。

## 21.3 残る誤差（position ≥ 1）

position 1 の層別誤差:

```
L0 = 0.0000092%   ← ほぼ bit 一致
L1 = 0.444%       ← ここで跳ねる（PLE 層）
L2 = 0.535%
...
L5 = 0.827%
L6 = 6.06%        ← さらに大きな跳躍
L7 = 5.94%
```

- **L1（PLE 層）**: CPU オラクルで PLE 内部を取得し比較したところ
  `ple_gate` 4.4e-5 / `ple_conv_out` 1.3e-4 と小さいが、
  L1 の入力（PLE 加算後）は 0.208% ずれる。PLE の残差加算経路に
  まだ丸めが残っている可能性が高い（`gated_value` に warp 側トレースが無く未確認）。
- **L6 の 6% 跳躍**: 離散的な事象（専門家選択の反転など）の可能性。

position 22 では L0 が 0.298%（状態経由のドリフト）、L47 は 26% 台。

## 21.4 次の一手

1. **PLE の残差加算経路**を F32 に（`ple_gated_value` の F32 トレースを追加して確認）
2. **L6 の 6% 跳躍**の原因（expert 選択 or 何か）
3. position ごとの誤差成長の追跡（position 1 → 22 でどう増えるか）


---

# 追記 14 (2026-10-14 朝): position 1 の誤差分解 — 量子化反転の増幅領域

## 22.1 position 1 の段差（実測）

| layer | hc_norm | hc_mix | layer out |
|---|---|---|---|
| L0 | — | — | 0.0000092% |
| L1 | 0.0219%（ミラー下限） | **0.209%** | 0.444% |
| L5 | 0.80% | 0.93% | 0.83% |
| L6 | 0.96% | 0.98% | **6.06%** |

- **L1 の norm はミラー下限（~5e-4）に張り付いており実質一致**、しかし mix は 0.209%（下限の 10 倍 = 実誤差）
- L6 の 6% 跳躍は **ルータ誤差（0.16%）による専門家選択の反転**と推定（上流誤差の症状）

## 22.2 PLE は原因ではない（検証済み）

position 1 の PLE を F32 同士で比較:

| 項目 | 誤差 |
|---|---|
| `ple_gated_value` | 4.4e-5 |
| `ple_conv_out` | 1.3e-5 |
| **PLE 寄与合計** | **1.4e-5**（寄与 rms 0.0060 に対し絶対 8.4e-8） |
| 残差 rms | 0.0115 |

PLE の寄与誤差は残差に対して **7e-6 相対** — 無視できる水準。

## 22.3 残る機構: 量子化のステップ反転による増幅

- L0 出力の誤差 9.2e-6% ＋ PLE の 1e-5 → 層 1 の入力残差は ~1e-5 相対でしか違わない
- しかし **Q8_0 活性量子化はステップ関数**なので、1e-5 の入力差でも一部のコードが反転し、
  **mix で 0.2%（200 倍）に増幅**される
- 以降の層でも同じ増幅が繰り返され、position 22 では L0 が 0.3%、L47 が ~26% になる

**したがって 64/64 には 1e-5 レベルの差を消す必要がある。** 残る発生源は:
1. L0 出力の 9.2e-6%（GDN の状態更新の丸め/演算順）
2. PLE の 1.4e-5（gate の 4.4e-5 が支配的）

## 22.4 現在の達成状況

| 項目 | 状態 |
|---|---|
| **position 0 の全 48 層** | **bit 一致** ✓ |
| position 1 の L0 | 9.2e-6%（ほぼ一致） |
| 64トークン一致 | 22 / 64（不一致は index 22） |
| 速度 | 4.2 tok/s（F16 高速パスの 1.8 倍） |

## 22.5 次の一手

1. **L0 の 9.2e-6%** の発生源（GDN の状態更新・conv・AR の演算順を ggml と厳密一致させる）
2. **PLE の gate の 4.4e-5**（query/key norm の経路に half が残っていないか）
3. 上記が消えれば position 1 以降も bit 一致に近づき、64/64 が射程に入る


---

# 追記 15 (2026-10-14 朝): 一致 22 → 41 / 64（fast-math と超越関数の修正）

## 23.1 決定的な発見: `--use_fast_math` が効いていた

CMAKE の CUDA フラグに `--use_fast_math` が入っており:
- device の `expf` は **`__expf`（MUFU.EX2、相対 ~2^-21 = 5e-7）** に置換されていた
- `-prec-div=false` により **素の `/` も近似除算**になっていた

つまり GDN/PLE の silu・sigmoid・decay は「1 ulp ずれ」ではなく **数 ulp ずれ**ており、
これが 1e-7〜1e-6 の差 → 量子化コード反転 → 層ごとの増幅を生んでいた。

## 23.2 修正内容（`src/qwen38_gdn.cuh`、10 カーネル 13 箇所）

オラクルは演算ごとに実装が違うことを llama.cpp の実ソースから確定:
- **SiLU は AVX2 多項式エミュレーション**（`ggml_v_silu` → `ggml_v_expf`、libm 不使用）
- **sigmoid / exp / softplus は scalar の glibc**（`op_sigmoid` / `op_exp` / `op_softplus`、GDN スキャン内の `expf` も scalar）

そこで:
- SiLU → 既存の ggml AVX2 エミュレーションヘルパを再利用
- sigmoid / decay / softplus → **glibc `e_expf.c` / `e_logf.c` の移植**（表も同梱、全演算を `__f*_rn` で明示して FMA 融合と近似除算を排除）
- softplus も `(x>20) ? x : logf(1+expf(x))` の形に修正（旧実装は log1pf 形式で不一致）

環境変数 `WARODECODE_EXACT_EXP`（既定 ON）で切替可能。

## 23.3 検証（ホスト、llama.cpp 実コードとの bit 比較）

| 比較 | サンプル数 | 不一致 |
|---|---|---|
| exp エミュ vs `ggml_v_expf`(AVX2 実物) | 180,002,690 | **0** |
| SiLU エミュ vs `ggml_v_silu`(AVX2 実物) | 180,002,690 | **0** |
| glibc 移植 `expf` vs glibc `expf` | 180,002,690 | **0** |
| glibc 移植 `logf` vs glibc `logf` | 180,003,048 | **0** |
| exact sigmoid / exp / softplus vs glibc | 各 180,002,690 | **0** |

合計 **約 17.8 億比較・不一致 0**。
（参考: `(float)exp((double)x)` は glibc `expf` と 0.0169% 不一致 → glibc は correctly rounded ではないため移植が必要だった）

## 23.4 効果（実機 L4）

| 構成 | 64トークン一致 |
|---|---|
| 修正前 | 22 / 64 |
| **exact exp 適用後** | **41 / 64** |
| ＋exact rsqrt（検証の結果、効果なし・悪化） | 22 / 64 |
| rsqrt 差し戻し | **41 / 64**（確定） |

次の不一致は index 41（ref `2404` に対し `303`）。

## 23.5 rsqrt の検証（差し戻し済み）

`rsqrtf`（2 ulp）→ `__fdiv_rn(1.0f, __fsqrt_rn(v))` の置換を 25 箇所で試したが、
position 1 の段差は不変、parity は 41 → 22 に悪化したため**差し戻した**。
（CPU 側の rms_norm も近似を使っている可能性があり、単純な置換では合わない）

## 23.6 残る課題

position 1 の層別プロファイルは修正前後でほぼ不変:
```
L0 = 9.2e-6%   L1 = 0.444%   L6 = 6.09%
```
- L0 の 9.2e-6% は GDN の状態更新に残る 1e-7 級の差（exp 以外の要因）
- L1 の 0.444% は PLE ではなく **mix の量子化反転による増幅**（入力差は ~1e-5）
- L6 の 6% はルータ誤差による専門家選択の反転


---

# 追記 16 (2026-10-14 朝): GDN の実装バグ 2 件と前提の訂正

## 24.1 前提の訂正（重要）

1. **`-DGGML_NATIVE=OFF` は FMA を無効化しない** — ggml はこのオプションで `INS_ENB=ON` にし、
   `-mavx2 -mfma -mf16c` を付ける。オラクル自身の runtime JSON にも記録されていた。
   → オラクルは AVX2+FMA を持つ。私の以前の「非融合」前提は誤り。
2. **GDN の AR/scan はスカラではなく `GGML_OP_GATED_DELTA_NET`** で、`ggml_vec_dot_f32`（AVX2+FMA intrinsics）を使う。
   warp の既存 AVX2 順序エミュレーションは **既に bit 一致**していた（実トレースで 0/792576）。
   スカラ非融合に変えると 785,798/792,576 が不一致になるため変更していない。

## 24.2 発見した実装バグ（`src/qwen38_gdn.cuh`）

| カーネル | oracle の実体 | warp の旧実装 | 結果 |
|---|---|---|---|
| **l2norm** | `build_gdn_l2_norm` = `scale(rms_norm(x, eps/n), 1/sqrtf(n))`、**double 逐次和 + 1.0f/sqrtf** | float FMA 木 + rsqrtf、**1/sqrtf(n) の乗算が欠落** | 実トレース **3008/4096 不一致** → 修正後 **0/4096** |
| conv | `sumf += s*c` 昇順（gcc は vmulps + 順序付き vaddss = 非融合） | fmaf 融合 | 合成 1,000,000 行で旧 310,534 不一致 → 新 **0** |
| AR/GDN の scale | `1.0f/sqrtf((float)S_v)` | `rsqrtf` | 修正（既定 ON） |
| gated RMSNorm | double 和 + `1/sqrtf(mean+eps)` | float FMA 木 | 修正 |

新スイッチ `WARODECODE_EXACT_SCALAR_FMA`（既定 ON）。

**検証（不一致 0）**: 実 oracle トレース `cpu_pos1_l0.bin` で l2norm 0/4096・GDN 0/792,576、
合成テストで conv 0/1,000,000・l2norm 0/12,800,000・GDN 0/8,454,144。

## 24.3 PLE/QSA/MoE/HC の exact-exp（既定 ON、softmax のみ既定 OFF）

30 箇所を CPU 実装（SiLU = AVX2 多項式、sigmoid/exp = glibc）に置換し、
**6,012,051 点で bit 不一致 0**（実 GGML_OP_SILU/SIGMOID/SOFT_MAX を含む）。
ただし softmax は **CPU が「max 1パス＋全要素 exp・再スケール無し」**、warp はオンライン再スケールで
**構造自体が違う**ため、exp だけ合わせても丸め順が一致しない。
実測: 全部 ON → 10/64、softmax のみ OFF → 22/64、ブロック全体 OFF → 22/64。

## 24.4 現在の parity

| 構成 | 一致 |
|---|---|
| GDN exact-exp のみ | **41 / 64** |
| GDN exact-exp + l2norm 修正 + ブロック exact-exp | 22 / 64 |

**41 が最高記録**。ただし position 1 の層別プロファイルはこれらの修正で**変化していない**
（L0 = 9.2e-6%、L1 = 0.444%、L6 = 6.06%）。
→ position 1 の残差は l2norm/exp とは別の要因。
サブエージェントの指摘: `q38_gdn_core_first_f32_kernel` の warp-shuffle ドットが AVX2 順序でない（次の候補）。

## 24.5 次の一手

1. `q38_gdn_core_first_f32_kernel` の順序を oracle の AVX2 実装に合わせる（position 1 の L0 残差の候補）
2. **QSA softmax の構造**を CPU と同じ「max 先行 → 行順 exp 蓄積」に組み替える（exp を合わせる前提）
3. MoE ルータ重みの正規化チェーン（CPU は 3 丸め、warp は 2 丸め）
4. L1 の mix 増幅（入力 1e-5 → 0.2%）の元を潰す


---

# 追記 17 (2026-10-14 午前): ルータ・first-token・AVX2 GEMV の bit 一致で段差が激減

## 25.1 実装した修正（subagent 2本）

| 対象 | 内容 | ホスト検証 |
|---|---|---|
| **MoE ルータ重み** | oracle の `build_moe_ffn` の演算列（全512 softmax → 8レーン double 和 → 逆数乗算 → argsort top-k → sum_rows(double) → clamp → **真の除算** → w_scale）を忠実再現 | **95行＋110,000行で id 差 0・weight bit 差 0** |
| **first-token GDN** | position 0 経路を `ggml_compute_forward_gated_delta_net_one_chunk` の AVX2 順序（`ggml_vec_dot_f32` 4×8 アキュムレータ + F32x8 還元・非融合末尾）に | 0/6144、0/25600、旧実装は 1542/6144 ✗ |
| **本線 beta/alpha GEMV** | `q38_gdn_dense_f32_gemv_avx2_kernel` を適用（warp-shuffle 還元 → AVX2 順序） | 0/48 行 |

新スイッチ: `WARODECODE_EXACT_ROUTER` / `WARODECODE_EXACT_FIRST_TOKEN`（いずれも既定 ON）。

**重要な訂正**: `llama-state-positions-0-2.bin`（10/1 取得）は **非 AVX2（double 累積）ビルド**の産物だった。
bit 基準には `cpu_pos1_l0.bin`（10/14 取得の AVX2 版）を使うこと。

## 25.2 効果（position 1 の層別誤差）

| layer | 修正前 | 修正後 |
|---|---|---|
| L0 | 9.2e-6% | 9.2e-6% |
| L1 | 0.444% | 0.444% |
| L5 | 0.85% | 0.85% |
| **L6** | **6.06%** | **0.97%** |
| L11 | 6.27% | 1.16% |
| L46 | （計測せず） | 22.8% |

**L6 の 6% 跳躍（専門家選択の反転）が消えた** ✓ — ルータの bit 一致により選択が一致するようになった。
残る段差は L14（1.2% → 3.3%）と L19（QSA 層、→ 5.0%）に移った。

## 25.3 現在の状態

| 指標 | 値 |
|---|---|
| 64トークン一致 | 22 / 64（最高記録は 41） |
| position 1 の L0 | 9.2e-6%（ほぼ一致） |
| position 1 の L6 | 0.97%（以前 6.06%） |
| position 0 の全 48 層 | bit 一致 |

**トークン一致（22）と段差（大幅改善）が乖離**している。index 22 で 1 トークンだけ外れて
その後再同期する形（ref 498 → got 318）で、崖っぷちの性質が強い。

## 25.4 次の一手

1. **L14 の段差**（GDN 層で 2.7 倍の跳躍）
2. **QSA のオンライン softmax 構造**を CPU と同じ「max 先行 → 全要素 exp → 再スケール無し」に組み替える
   （L19 以降の段差の主因と推定）
3. 速度の再測定（exact 化のコスト確認）


---

# 追記 18 (2026-10-14 午前): L1 の mix 誤差は実在（測定は正常）

## 26.1 検証したこと

1. **F32 mix バッファが上書きされていないか**を診断（`layer*.hc_attn_mix_f32_early` を追加して
   attention 実行前に取得し、既存の late アイテムと比較）:
   - L0: early = late = 0.00000000（bit 一致）
   - L1: early = late（完全一致、max|diff| = 0）
   → **測定は正常**。トレース項目は正しい値を捉えている。
2. **HC の F32 γ（`debug_norm_f32`）は全トランク層でロードされている**（`load_qwen38_hc_weights` が
   norm が F32 かつ down/up が Q8_0 のとき必ず設定）→ F32 mix 経路は全層で有効。

## 26.2 現状の数値（position 1）

| layer | 誤差 |
|---|---|
| L0 | 9.2e-6% |
| L1 | 0.444%（mix 単体で 0.208%） |
| L6 | 0.97% |
| L14 | 3.29% |
| L19（QSA） | 5.0% |
| L46 | 22.8% |

**未解決の謎**: L0 の出力誤差は 9.2e-8 相対しかないのに、L1 の mix は 0.208% ずれる。
量子化コード反転の見積り（1e-7 入力 → 1e-5 出力）では説明できない **200 倍の増幅**。
L1 の入力（L0 出力＋PLE）以外の要因、たとえば mix 自体の演算順・γ の一致を疑う必要がある。

## 26.3 進行中

- **QSA のオンライン softmax 構造**を CPU と同じ「max 先行 → 全要素 exp → 再スケール無し」に
  組み替える作業を subagent に依頼中（L19 以降の段差の主因と推定）。

## 26.4 次の一手

1. QSA softmax の構造一致（進行中）
2. L1 の mix 増幅の原因（mix の演算順を CPU と 1 演算ずつ突き合わせる）
3. L14 の段差（専門家選択の反転か、他の離散事象か）


---

# 追記 19 (2026-10-14 午前): L1 増幅の原因を局所化 — PLE gate の 4.4e-5

## 27.1 誤差の連鎖（position 1）

HC mix の F32 内部値を計測（`layer*.hc_attn_norm_f32` / `hc_attn_gate_f32` のトレースを追加）:

| stage | 誤差 | 増幅 |
|---|---|---|
| L1 の入力残差（L0 出力＋PLE） | ~7e-6 | — |
| HC norm | 6.6e-5 | ×9 |
| **HC gate（up 射影の出力）** | **1.77e-3** | **×27** |
| HC mix | 2.08e-3 | ×1.2 |

HC mix は「norm → Q8_0 量子化 → down 射影 → SiLU → Q8_0 量子化 → up 射影 → sigmoid 重み付き和」で、
**量子化 2 段**が入力の微差を増幅する。したがって **入力の 7e-6 を消せば mix の 0.2% も消える**。

## 27.2 L1 入力の 7e-6 の出所 = PLE の寄与

position 1 の PLE（F32 同士）:

| stage | 誤差 |
|---|---|
| `ple_gate` | **4.371e-5** |
| `ple_conv_out` | 1.341e-5 |
| `ple_gated_value` | 4.371e-5 |

PLE の寄与の誤差は **gate が支配的**。gate の式自体は以前の subagent が ggml 互換にしたが、
**入力（正規化後の q/k、s、|s|）の一致は未検証**だった。
→ `src/qwen38_ple.cuh` の gate 経路を oracle と bit 一致させる作業を subagent に依頼中。

## 27.3 診断用に追加したトレース

- `layer*.hc_attn_mix_f32_early`（attention 実行前の mix。late 項目との一致で「バッファ上書き無し」を確認）
- `layer*.hc_attn_norm_f32` / `hc_attn_gate_f32` / `hc_attn_low_raw_f32` / `hc_attn_low_f32`

これらは half ミラー（~5e-4 の下限）では見えなかった真の誤差を可視化する。

## 27.4 進行中の作業

| 担当 | 内容 |
|---|---|
| A | `src/qwen38_qsa.cuh`: QSA のオンライン softmax を CPU の「max 先行 → 全要素 exp → 再スケール無し」へ |
| B | `src/qwen38_ple.cuh`: gate 経路の bit 一致（4.4e-5 の除去） |


---

# 追記 20 (2026-10-14 午前): 桁の読み違いの訂正と、L0 内部の矛盾

## 28.1 重要な訂正（私の誤りの修正）

position 1 の L0 出力誤差は **0.00009224 = 9.2e-5（0.0092%）** であり、
私が以前記述した「9.2e-6%（0.0000092%）」は**桁の読み違い**だった。
9.2e-5 であれば L1 の入力ドリフト 7e-6〜9e-5 が自然に説明でき、
「200倍の不可解な増幅」という謎は**存在しなかった**。

## 28.2 PLE は原因ではない（subagent の検証）

- PLE の gate 経路（Q8_0 GEMV / rms_norm / mul / sum_rows / scale / abs / clamp / sqrt / sgn / sigmoid）は
  **既に CPU と bit 一致**（実カーネル本文を機械抽出し、402 ケースで不一致 0）。
  → `src/qwen38_ple.cuh` は**無変更**（修正すべきものが無かった）。
- 4.371e-5 という gate 誤差は **L0 出力の 9.224e-5 ドリフトが入力に来ている**ことで説明できる
  （感度実験: 乱数重み 32 セットで gate relL2 の平均 2.85e-5、観測値は分布内）。
- 因果は「gate の誤差 → L1 入力」ではなく **「L0 の FFN ドリフト → PLE 入力 → gate の誤差」**。

## 28.3 L0 内部の矛盾（次のターゲット）

position 1 の layer 0 を F32 同士で計測（今回 F32 トレースを追加）:

| stage | 誤差 |
|---|---|
| hc_ffn_inject | 5e-8（bit 一致） |
| hc combine（attention 直後） | 3e-8（bit 一致） |
| routed MoE sum（d_accum） | 1.2e-7（bit 一致） |
| shared expert（down_f32） | 0.00000000（bit 一致） |
| **layer 0 出力** | **9.224e-5** |

**入力・分岐・inject がすべて bit 一致なのに、層出力だけが 9.2e-5 ずれる。**
これは combine の残差入力（または combine 自体の演算）に問題があることを示す。
次は「ffn combine が読む残差」を直接 F32 でトレースして比較する。

## 28.4 QSA exact カーネル（subagent 実装）

- CPU の非 flash attention tail を**忠実再現**（990 ケース＋カーネル本文シミュレーション＋実トレース 0/6144）。
- 重要な発見: softmax の行長は `pos+1` ではなく `min(cells.size(), max(256, pad256(used+1)))`。
  oracle は n_ctx=128 なので行長 128 で、これが F16 ドットの 32 境界を決める
  （`pos+1` を 8 切り上げにすると 3089/6144 不一致）。
- **ただし engine で有効化すると層別誤差が悪化**（L12 1.19% → 7.6%）。
  → 既定 OFF（`WARODECODE_EXACT_QSA_SOFTMAX=0`）。engine 側の可視セル／行長の配線に課題が残る。
- 残差の最有力は **スコアのドット積の丸め順**（warp のツリー還元 vs CPU の 4 アキュムレータ F16 ドット）。

## 28.5 測定ツールの既知の誤り（subagent 指摘）

- `tools/issue1_compare_internals.py` の `hc_init` ↔ `embedding.hc_init` は誤対応。
  正しくは **`layer0.input_residual`**（bit 一致）。放置すると「最初の発散 = embedding」と誤診する。
- `ffn_moe_out` の対応も 100% 不一致（バッファ再利用のため）。
- engine トレースは `_f32` が付かない項目は half ミラーであり、1e-4 台の議論には使えない。


---

# 追記 21 (2026-10-14 午前): L0〜L2 が bit 一致、次の障壁は RoPE

## 29.1 共有エキスパートの half ミラーが L0 ドリフトの原因だった

position 1 の layer 0 では inject・combine・分岐がすべて bit 一致なのに層出力だけ 9.2e-5 ずれる、
という矛盾を追った結果:

- 旧経路は combine に **共有エキスパートの half ミラー（`d_down`）** を渡していた
  （`run_qwen38_hc_combine_moe_f32_inject` の `shared` 引数）。
- `WARODECODE_MOE_F32_OUT=1`（`moe_out_f32` 経由）にすると **L0 出力誤差が 9.224e-5 → 1.1e-7**（800 倍改善）。
- → **既定を ON に変更**（`q38_moe_f32_out_opt_in`、env で A/B 可）。

## 29.2 効果（position 1 の層別誤差）

| layer | 旧既定 | MoE F32 出力 ON |
|---|---|---|
| L0 | 9.2e-5 | **1.1e-7** |
| L1 | 4.44e-3 | **1.6e-7** |
| L2 | 5.35e-3 | **1.8e-7** |
| L11 | 1.16e-2 | 4.57e-2 |
| L47 | — | 0.317 |

**L0〜L2 が bit 一致になった**。これまで L1 の 0.444% は PLE だと考えていたが、
実際は L0 の FFN ドリフトが入力に来ていたものだった（PLE 自体は完全に正しい）。

## 29.3 次の障壁: QSA 層の IMRoPE

layer 3（最初の QSA 層）の内部比較（position 1）:

| stage | 誤差 |
|---|---|
| hc_attn mix | 1.9e-7（bit 一致） |
| **QSA query** | **3.41e-5** |
| **QSA key** | **5.05e-5** |
| **QSA value** | **0.00000000（bit 一致）** |
| layer out | 1.23e-3 |

**value だけが完全一致で q/k は RoPE を通る** — これが唯一の違い。
`src/qwen38_rope.cuh:65` の `sincosf` は `--use_fast_math` により
MUFU 近似（`__sincosf`、絶対誤差 ~2^-21）に置換される。CPU は glibc の `sinf`/`cosf` を使う。
→ expf と同じクラスの問題。修正を subagent に依頼中。

## 29.4 トークン一致（参考）

| 構成 | 一致 |
|---|---|
| MoE F32 出力 OFF（旧既定） | 22 |
| MoE F32 出力 ON（新既定） | 10 |
| GDN exact-exp のみ（参考・最高記録） | 41 |

忠実度は新既定が圧倒的に上（L0〜L2 が bit 一致）だが、トークン一致は崖っぷちで振れる。


---

# 追記 22 (2026-10-14 昼): QSA の F32 正規化重みが half 化されていた

## 30.1 RoPE の修正（実装済み）

`src/qwen38_rope.cuh` と `src/qwen38_qsa.cuh` の IMRoPE を、CPU の
`ggml_rope_cache_init()` と同じ計算に直した:
- `theta_scale = powf(freq_base, -2/n)` を double で計算（fast-math の `__powf` を回避）
- `theta = p * powf(theta_scale, pair)` → **CPU と同じ逐次乗算**（`theta *= theta_scale`）
- `sincosf` → `(float)sin((double)theta)` / `(float)cos((double)theta)`（glibc と一致）
- スイッチ `WARODECODE_EXACT_ROPE`（既定 ON）

**ただし QSA の q/k 誤差は変わらなかった**（3.414e-5 → 3.415e-5）＝ RoPE は主因ではなかった。
（それでも CPU と同じ計算に揃えたので、より正しい実装になっている）

## 30.2 真の原因: q/k 正規化重みの half 丸め

GGUF の実テンソル型を確認:

| テンソル | 型 |
|---|---|
| `blk.N.attn_q_norm.weight` | **F32 [256]** |
| `blk.N.attn_k_norm.weight` | **F32 [256]** |
| `blk.N.indexer.q_norm.weight` / `k_norm.weight` | **F32 [128]** |

一方 `src/qwen38_qsa.cuh` のローダは `upload_half_array()` で **half に丸めていた**。
half 丸めは相対 ~5e-4 → q/k の正規化で ~3.4e-5 の差 → 観測値と一致する。
value は正規化重みを持たないため bit 一致のまま、という観測とも整合する。

同じパターンの過去の例: HC の γ（`debug_norm_f32` ✓ 修正済み）、GDN の `ssm_norm`（✓ 修正済み）、
PLE の norm（✓ 修正済み）、β/α（✓ 修正済み）、HC inject（✓ 修正済み）。
**「F32 テンソルを half に落とすと必ず 5e-4 の系統誤差が入る」** という一貫したパターンで、
QSA の q/k/indexer norm が最後の残りだった。

→ F32 の重みを F32 のまま使う修正を subagent に依頼中（`WARODECODE_QSA_F32_NORM` 既定 ON）。

## 30.3 速度（実測）

| 構成 | 速度 |
|---|---|
| F16 高速パス（基準） | 2.38 tok/s |
| 現構成（exact 化後） | **3.782 tok/s（1.59倍）** |

exact 化のコストは約 10% だが、int8 経路の利得が上回り要件を満たす。


---

# 追記 23 (2026-10-14 午後): QSA の q/k/v が bit 一致、残るは attention 演算

## 31.1 RoPE 移植のコンパイル修正

RoPE subagent の新実装（`qwen38_rope.cuh` + `warodecode_q38_glibc_trig.hpp` + `qwen38_imrope_math.cuh`）に
コンパイルエラーが 2 件あり、修正した:
1. include の拡張子ミス（`.hpp` → `.cuh`）
2. device 用 `__constant__` テーブルの定義が使用箇所より後ろにあり未定義エラー →
   定義を `sincos_table()` の前へ移動（前方宣言では host pass で二重定義になるため）

## 31.2 成果（実機 L4・position 1 / layer 3）

| stage | 修正前 | 修正後 |
|---|---|---|
| QSA query | 3.41e-5 | **0.00000000** |
| QSA key | 5.05e-5 | **0.00000000** |
| QSA value | 0 | 0.00000000 |
| **attention 出力** | 1.23e-3 | **1.48e-3（まだ残る）** |
| layer 出力 | 1.23e-3 | 1.46e-3 |

**q/k/v が bit 一致になった**（IMRoPE の忠実移植＋F32 norm 重み＋射影の効果）。

## 31.3 残る誤差の在処

q/k/v が bit 一致なので、誤差は **attention の演算そのもの**で発生している。候補:
1. **ブロック選択の不一致**（indexer の可視ブロックが違えば離散的に大きくずれる）← 最有力
2. スコアのドット積の丸め順（warp のツリー還元 vs CPU の `ggml_vec_dot_f16` 4 アキュムレータ + FMA、KV は F16）
3. 値の重み付き和（CPU は確率を F16 に丸めてからドット）

exact QSA カーネル（`WARODECODE_EXACT_QSA_SOFTMAX=1`）を使うと attention 出力の誤差は
1.48e-3 → 1.05e-3 に減るが決定的ではない（layer 出力はむしろ微増）。

→ ブロック選択を最有力として調査を subagent に依頼中。


---

# 追記 24 (2026-10-14 夕): **Issue #1 完全解決 — 64/64 トークン一致達成**

## 32.1 決定打: スコアドットの q の F16 丸め ＋ exact attention カーネル

CPU は KV キャッシュが F16 なので `ggml_mul_mat(k_f16, q)` の **vec_dot_type = F16** となり、
`ggml_cpu_fp32_to_fp16()` で **q を F16 に丸めてから**内積する。
warp は q を F32 のまま使っていたため、スコアに ~2.4e-4 の相対差が出て、
softmax → 値和 → **Q8_0 の出力射影**で増幅されていた。

修正:
1. スコアドットで q を `__half2float(__float2half_rn(qd))` に（3 カーネル）。
2. **CPU 順の exact attention カーネル**（`WARODECODE_EXACT_QSA_SOFTMAX`、3 パス・double 行和・
   確率の F16 丸め・4×8 アキュムレータ）を**既定 ON** に変更。

## 32.2 結果（実機 L4）

### 層別誤差（position 1）

| layer | 誤差 |
|---|---|
| **L0〜L7** | **1.1e-7 〜 1.8e-7（bit 一致）** |
| L8 | 1.27e-3（次の障壁） |

QSA 層（L3, L7）も含めて **L0〜L7 が bit 一致**した。

### attention の段別（position 1, layer 3）

| 比較 | 誤差 |
|---|---|
| `attn_f32` vs CPU `attn_gated` | **0.00000000** |
| `attention_out_f32` vs CPU `attn_output` | **0.00000000** |

## 32.3 最終検証（正規ツール）

`python3 tools/issue1_parity_check.py`（参照 SHA-256 ガード付き・独立実行）:

```
/match64        True
/first_mismatch None
/tok_s          3.195
```

**64/64 完全一致**（ランナー経由の実行でも要素ごとに 64/64 一致を確認 = 2 回再現）。

## 32.4 速度（要件の後半）

| 構成 | 速度 |
|---|---|
| F16 高速パス（基準） | 2.38 tok/s |
| **最終構成（int8 経路 + 忠実化）** | **3.195 tok/s（1.34倍）** |

exact attention カーネルの追加コストで 3.78 → 3.20 tok/s になったが、**F16 基準は上回っている**。

## 32.5 効いた修正の総括（すべてホストまたは実機で bit 一致検証済み）

1. 量子化ドット（IQ2_S/IQ3_S/IQ4_NL/Q6_K/Q8_0 × 適切な活性量子化）
2. `--use_fast_math` の超越関数（device expf → glibc 移植、SiLU は AVX2 多項式、softmax）
3. GDN: l2norm（double 逐次和 + `1/sqrtf(n)`）、conv の非融合順、gate の AVX2 順
4. IMRoPE: glibc sinf/cosf 移植 + 逐次乗算 + FMA 回転形
5. F32 テンソルの half 丸め排除（HC γ/inject、GDN β/α/norm、PLE norms、QSA norms、共有エキスパート gate）
6. F32 GEMV の AVX2 順序（router / inject / β/α）
7. MoE ルータ重みの CPU 演算列（全 512 softmax → double 和 → 真の除算）
8. MoE 出力の F32 経路（共有エキスパートの half ミラー排除）
9. QSA attention の CPU 順（max 先行 → 全要素 exp → double 行和 → 確率の F16 丸め → 4×8 アキュムレータ）
10. **スコアドットの q の F16 丸め** ← 最後の決定打
