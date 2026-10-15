# Task T: sync/other の分解（A100 / 20000 MiB / parity 64/64）

対象: batch=1 decode、64 トークン、expert cache 20000 MiB、WARODECODE_PROFILE=1。
計測は区間プロファイラ（CPU 時間つき）と、その中に足した MoE 3 分割・select 5 分割、
および VM 上でビルドする独立プローブ 2 本で行った。

---

## 0. 結論

1. **「sync/other 48 ms」は 1 本の同期ではなく、MoE 区間の中の [expert 選択] と
   [shared expert] と [転送] の合計だった。** 現行既定（スイッチなし）の実測内訳は
   トークン 99.4 ms（10.05 tok/s、parity 64/64）に対して:

   | 区間 | ms/token | 備考 |
   |---|---:|---|
   | pre-MoE（attn + HC mix + combine + PLE + head + embed） | 39.0 | 区間プロファイルの device 合計 |
   | MoE: **expert 選択** | **18.3** | router GEMV + top-k + D2H 読み戻し + 待ち |
   | MoE: routed（expert 取得 + expert GEMV） | 33.9 | うち**転送 25.6**、カーネル実行 8.4 |
   | MoE: shared expert + 残り | 8.5 | |
   | sync / other（プロファイルの残差） | 0.02 / 1.3 | |

2. **48 層 × 1 回の `cudaStreamSynchronize` は主因ではない。** 実測で
   空ストリーム 0.62 µs、kernel 込み 10.1 µs。48 層で 0.5 ms/token にしかならない。
   待ち方を `cudaEventQuery` スピンに変えても差はゼロだった（t1: 127.90 vs 127.91）。

3. **host はプリエンプトされていない**（wall − CPU = 0.013〜0.027 ms/token）。
   どの区間も「CPU を回して待っている」わけではない。

4. **1 スレッド exact router の並列化は、この環境では改善にならなかった**（下記 5 節）。
   当初 t3 の計測で -15 ms と見えたが、t4 のクリーンな連続計測では **+9.2 ms の悪化**。
   t3 は VM が遅い時間帯に当たっており、比較対象が汚染されていた。

5. 転送 25.6 ms は依然として最大級。既存のコピーストリーム数（2/4/8）では動かない
   （別担当の sweep r1〜r4）。

---

## 1. 計測手法

| 道具 | 何を分けるか |
|---|---|
| `cpu_ms`（stage ごと、`CLOCK_THREAD_CPUTIME_ID`） | wall と CPU の差 = 「実行中」か「待機/プリエンプト」か |
| `moe_region_begin/cut/end` | MoE 区間を [select][routed][shared] に 3 分割 |
| `WARODECODE_SELECT_PROBE=1` | select の中を 5 分割（device イベント、4 record/層） |
| `tests/moe_sync_cost_probe.cu` | 同期・起動・D2H の単価を VM 上で直接測る |
| `tests/router_parallel_exact_test.cu` | 並列 router と現行 router の id / weight の bit 比較 |

ランナは `tests/sync_overhead_sweep.py` + `tests/sync_overhead_sweep.sh`（1 ジョブで複数構成を連続実行）。

---

## 2. 実測: ホスト/ドライバのプリミティブ単価（A100 Colab VM、3000 回平均）

| プリミティブ | µs/回 |
|---|---:|
| cudaStreamSynchronize（空ストリーム） | 0.62 |
| 1 スレッド kernel + cudaStreamSynchronize | 10.07 |
| 1 スレッド kernel + cudaEventSynchronize | 10.21 |
| 1 スレッド kernel + cudaEventQuery スピン | 10.16 |
| 1 スレッド kernel + 40 B pinned D2H + cudaStreamSynchronize | 20.33 |
| kernel 起動のみ（back-to-back） | 2.70 |

48 層ぶんに換算すると、同期 0.5 ms、読み戻し 1.0 ms、起動 3 ms。
**どれを足しても 48 ms にならない**——これが「同期が主因ではない」ことの根拠である。

---

## 3. 実測: MoE 区間の 3 分割（t4、同一ジョブ内で連続計測、全ケース parity 64/64）

| ケース | total | MoE | select(dev) | routed(dev) | shared(dev) |
|---|---:|---:|---:|---:|---:|
| base-a（スイッチなし） | 99.94 | 61.15 | 0.16* | 52.46* | 8.54 |
| base-b（再現） | 99.36 | 60.41 | 0.16* | 51.68* | 8.57 |
| split-sync（選択を hoist） | 99.53 | 60.80 | **18.35** | 33.95 | 8.51 |
| split-sync-b（再現） | 99.15 | 60.38 | **18.28** | 33.59 | 8.52 |
| router-sync（+並列 router） | 108.66 | 69.89 | **27.66** | 33.71 | 8.53 |
| probe-sync（+5 分割計測） | 125.11 | 86.14 | 43.70 | 33.85 | 8.59 |
| skip（転送ゼロ、parity 対象外） | 84.66 | 44.77 | 27.80 | 8.38 | 8.60 |

\* base は選択が runner の内側にあるため select 区間が 0 で routed に含まれる（52.46 = 選択 18.3 + routed 33.9 とほぼ一致）。

読み取れること:

- **選択 18.3 ms と shared 8.5 ms は転送の有無で変わらない**（skip でも 27.8 / 8.6。
  skip は並列 router 入りなので、直列なら 18.3 / 8.5）。
- **転送 = routed(33.9) − routed(skip 8.4) = 25.6 ms/token**（210 MiB/token を pread 13.4 ms +
  DMA 12 ms で運ぶ。PCIe Gen4 x16 の実効帯域相当で、これ以上は帯域の壁）。
- この 2 つで 43.8 ms。**親タスクの「other/sync 48 ms」とほぼ同じ量**で、中身は
  「1 スレッド GPU カーネル + 小さなカーネル列 + 転送」だった。
- `WARODECODE_SELECT_PROBE=1` の 4 イベント/層は **+25 ms/token** の観測オーバーヘッドがある
  （probe-sync 125.1 vs split-sync 99.5）。5 分割の数値（t3）はこの影響下の値である。

---

## 4. select 区間の 5 分割（t3、参考値・VM が遅い時間帯の計測）

| piece | 内容 | 直列 top-k | 並列 top-k |
|---|---|---:|---:|
| router | 活性化 Q8_0 量子化 + router GEMV(512x2560) | 25.72 | 25.83 |
| top-k | exact router の top-k / softmax | 16.77 | 1.36 |
| readback | pinned D2H 40 B × 2 | 0.51 | 0.53 |
| wait | 読み戻し待ち | 0.56 | 0.52 |
| tail | | 0.17 | 0.18 |

**注意**: この表は t3（VM が遅い時間帯、total 129.6 ms）の値で、t4 のクリーンな値
（select 全体 18.3 ms）と整合しない。t3 では同じ select が 43.7 ms かかっており、
区間内訳の絶対値は信用できない。**相対比較（top-k だけが 12 倍速くなる）は
カーネル単体の性質として正しいが、トークン全体の改善にはつながらなかった**（5 節）。

---

## 5. A/B: 1 スレッド exact router の並列化（t4、同一ジョブ内）

| ケース | env | tok/s | parity | total | select(dev) |
|---|---|---:|---|---:|---:|
| base-a | （なし） | 9.994 | 64/64 | 99.94 | — |
| base-b | （なし） | 10.053 | 64/64 | 99.36 | — |
| split-sync | `WARODECODE_MOE_SELECT_SPLIT=1` | 10.036 | 64/64 | 99.53 | 18.35 |
| split-sync-b | 同上（再現） | 10.075 | 64/64 | 99.15 | 18.28 |
| **router-sync** | `+WARODECODE_ROUTER_PARALLEL=1` | **9.194** | 64/64 | **108.66** | **27.66** |
| probe-sync | `+WARODECODE_SELECT_PROBE=1` | 7.983 | 64/64 | 125.11 | 43.70 |

- **並列 router は −9.2 ms/token の悪化**（4 ケースが 99.1〜99.9 で安定していた直後の計測なので、
  同一ジョブ内の比較として意味がある）。原因は 1 ブロック内の 10 段 × バリア + 共有メモリ上の
  64 bit atomicMax 競合と見られる（256 スレッドが同一アドレスに 10 回ぶつける）。
  **採用しない。既定 OFF のまま。**
- 選択の hoist 自体は中立（99.53 / 99.15 vs 99.94 / 99.36）。
- 5 分割計測は観測オーバーヘッドが大きい（+25 ms）ので、常用の計測器にはできない。

### 5.1 実装したスイッチ（すべて既定 OFF）

| 環境変数 | 内容 |
|---|---|
| `WARODECODE_MOE_SELECT_SPLIT` | expert 選択を MoE runner から step() へ移す（カーネル・順序・引数は同一） |
| `WARODECODE_ROUTER_PARALLEL` | 1 スレッド exact router の並列 bit 一致版（**t4 で悪化を確認**） |
| `WARODECODE_SELECT_POLL` | 読み戻し待ちを event query スピンに（**差ゼロ**） |
| `WARODECODE_SELECT_PROBE` | select の 5 分割計測（+25 ms の観測コスト） |

並列 router が bit 一致する根拠（実装コメントより）: max は fmaxf で順序不変、
exp は要素ごと、chunk 和はチャンク順のまま 1 スレッドで double 加算、正規化は要素ごと、
top-k の各段は「最大値・同値なら小さい index・既出は除外」で、順序保存 float ビットと
~index をパックした 64 bit atomicMax と厳密に同値。全アンダーフロー行と全 -INFINITY 行は
参照実装へそのまま委譲する。VM 上で動かす bit 比較テスト（`tests/router_parallel_exact_test.cu`）
は用意したが、**今回のジョブではコンパイルに失敗して未実行**（`qwen38_cpu_moe.cuh` の
SSE intrinsic が素の nvcc で解決できない。テスト側に `<immintrin.h>` を足して修正済み、再実行は未了）。

---

## 6. 次に掘るべき順（この計測で残った的）

1. **転送 25.6 ms/token**（pread 13.4 + DMA 12）。帯域の壁に近いので、減らすなら
   「読む量を減らす（キャッシュ拡大・量子化の変更）」か「計算と重ねる」しかない。
   バッチ経路は 10 expert ぶんの pread/DMA を全部発行してからカーネルを起動するため、
   転送が完全に露出する（別担当の `WARODECODE_MOE_BATCH_GROUPS` がこの対策）。
2. **expert 選択 18.3 ms/token**。うち top-k が大半（t3 の 1 スレッド実行 = 0.35 ms/層）。
   並列化は単純なブロック分割では負けたので、次は「1 warp で 512 を分担して
   shfl リダクション」または「CPU 側で計算」を検討する価値がある。
3. **shared expert 8.5 ms/token**（0.18 ms/層、起動 10 個程度）。起動費では説明できない。
4. pre-MoE 39 ms（attn 14.7 + HC mix 20 + head 1.3 + PLE 0.6）は別タスクの領域。

---

## 7. 未検証リスク・注意

- **この VM は実行ごとに 30% 動く。** 同一ビルド・同一構成で total が 99.4 ms の日と
  129.6 ms の日があった（t3 vs t4）。**同一ジョブ内で連続に取った値以外を A/B の根拠にしてはいけない。**
  今回のスイッチ評価はすべて同一ジョブ内の連続計測で行った。
- 並列 router の bit 一致は**テスト未実行**（上記のコンパイル問題）。採用しない判断なので
  実害はないが、将来使うなら先にテストを通すこと。
- 5 分割計測はイベント 4 個/層で +25 ms の観測コストがある。常用しないこと。
- 計測はすべて 20000 MiB キャッシュ・64 トークン・同一参照トークン列での parity 確認つき。
- 触ったファイル: `src/qwen38_forward.cuh`、`src/qwen38_trace.hpp`、`tests/moe_sync_cost_probe.cu`、
  `tests/router_parallel_exact_test.cu`、`tests/sync_overhead_sweep.{sh,py}`、本ドキュメント。
  `src/qwen38_moe.cuh` と `src/expert_cache.cuh` は触っていない（別担当が編集中）。
