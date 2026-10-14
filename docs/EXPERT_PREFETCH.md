# expert 先読みパイプライン (Task K)

対象: `warodecode` qwen4exp (Qwen3.8-Flash-Next-IQ3E-Q8D-MTP) / batch=1 / greedy。
目的は MoE 区間の「次の expert を待つ時間」を消すこと。数値・ルーティングには一切触れず、
expert キャッシュの *読み込みを前倒しするかどうか* だけを変える。

**parity 64/64 は全 9 構成で確認した**（実測値は §3.2）。

---

## 0. 結論 (TL;DR)

1. **MoE 区間の host−device 差 (L4 c1536 で 91 ms/token) は expert の読み込み待ちではない。**
   host は MoE 区間の中で、直前にキュー済みの attention / HC mix の GPU 実行を
   **層ごとの `cudaStreamSynchronize`（expert 選択）で待っている**。expert I/O は
   MoE host の 17% (c1536) / 11% (c8000) にすぎない。
2. **履歴ベースの先読みは、このワークロードでは原理的に仕事がない。**
   予測（前トークンで同じ層が選んだ expert 列）28,549 件は、**28,549 件すべてが
   すでにキャッシュに resident** だった。理由は定量的に説明できる:
   キャッシュのヒット集合＝前トークンの expert 集合そのもの
   (`hit_rate` 0.3878 と `prev_token_reuse` 0.3877 が c1536 で一致)。
   ミスする expert は「前トークンで使わなかった expert」なので、履歴からは予測できない。
3. **先読みリング自体は正しく動く**（診断スイッチ `WARODECODE_EXPERT_PREFETCH_ALL=1` で
   resident でも post させると 28,549 件を実際に読んで 28,541 件をリングへ届け、
   parity 64/64 を維持）。ただしそれは**無駄読み**で、token は −2.7% / −3.5% 遅くなる。
4. expert の読みは **帯域飽和**している（c8000 15.2 GB/s、c1536 17.1 GB/s。4 スレッドの
   実測上限 ≈17.8 GB/s）。読みを非同期化して host をブロックしなくしても、
   資源（ページキャッシュのコピー帯域と CPU）が埋まっているので速くならない。

→ **A/B の結論: 既定構成 (off) に対する先読みの効果は ±0.2% 以内。採用しない。**
   実装は診断・再現のために残し、既定は off のまま。

---

## 1. 現状の待ち時間の分解

### 1.1 測り方

既存の `WARODECODE_EXPERT_STATS` は expert キャッシュの**内側**
(`acquire_ms` / `stage_ms` / `io_pread_ms` / `io_ring_wait_ms`) しか見ていない。
MoE 区間の host 時間のうち stage は 15 ms しかなく、残りがどこにあるか分からなかったので、
**acquire と acquire の間**を測る計器を足した（`src/expert_cache.cuh` の
`ExpertCacheMonitor::note_gap_ms`、stats 有効時のみ）。

| 統計 | 意味 |
|---|---|
| `acquire_ms` | acquire() の中にいた時間（既存） |
| `stage_ms` / `io_pread_ms` | そのうち submit() の中 / pread の実時間（既存） |
| `intra_gap_ms` | 同じ層の中で、ある acquire の終了から次の acquire の開始まで。中身は expert GEMV の発行 + accumulate |
| `layer_gap_ms` | 層が変わるときの同じ区間。中身は残りの MoE + hc_combine + 次層の hc_mix + attention + router + **per-layer の expert 選択 sync** |

**注意**: 最初の acquire には前の acquire が無いので、そのギャップは数えない
（初版はここを `steady_clock` のエポックとの差で計算してしまい、`layer_gap_ms` が
1e11 ms になった。修正済みで、本ドキュメントの値は修正後の実測）。

### 1.2 実測（warpdecode-l4b / arch=89 / 64 token greedy / 他プロセスなし）

profile OFF ではなく `WARODECODE_PROFILE=1` を付けた run の平均値 (ms/token)。

| 区間 | c8000 off | c1536 off | 備考 |
|---|---:|---:|---|
| `acquire` 合計 | 22.59 | 36.23 | expert キャッシュ acquire の内側 |
| 　└ `stage` | 19.61 | 34.09 | `submit()` の中 |
| 　　└ `io_pread` | 16.79 | 29.30 | pread の実時間 |
| 　└ その他 | 2.98 | 2.14 | キー計算 / retire イベント / 統計 |
| `intra_gap` | 9.49 | 9.48 | expert GEMV 発行 + accumulate |
| 層間（MoE 内の待ち） | **144.89** | **149.93** | 下記 |
| **MoE host 合計** | **176.97** | **195.64** | `profile_detail.host_moe_ms` |
| MoE device | 85.96 | 104.88 | `profile.moe_ms` |
| token 合計 (device) | 193.22 | 212.06 | |

層間（MoE 内の待ち）の正体: その区間で host が自分でやる仕事は
attention 4.5 + hc_mix 6.8 ≈ 11 ms/token しかない(`host_attn_ms` + `host_hc_mix_*_ms`)。
残り **約 134〜139 ms/token は GPU の完了待ち**である。中身は
`q38_select_experts()` の中の `cudaStreamSynchronize(stream)`（expert 選択の D2H + 同期）で、
この時点で GPU は**同じ層の attention / HC mix** を実行している。

### 1.3 「MoE で GPU が 114 ms 遊んでいる」の再解釈

MoE host (195.6) − MoE device (104.9) = 90.7 ms の差は、GPU が遊んでいる時間ではない。
その 195.6 ms は

    host_moe = acquire (36.2) + intra_gap (9.5) + 層間 (149.9)

に分かれ、**層間の 149.9 ms のうち host 自身の仕事は 11 ms 程度**（attention 4.5 + hc_mix 6.8）、
残りは待ちである。その待ちの間に GPU が実行しているのは、**MoE の begin イベントより前に
キュー済みの** attention (device 43.8) と HC mix (device 29.2 + 29.1 = 58.3) である。
これらの時間は profiler 上は `attn_ms` / `hc_mix_*_ms` として別の区間に計上されるので、
**MoE の host−device 差として現れる**。したがって:

    host_moe ≈ device_moe + (MoE begin より前にキューされた attn/hc_mix の GPU 実行) + MoE 内の host 仕事

と読むのが正しい。c1536 では 104.9 + 102.1 = 207 が「GPU が MoE 区間で実行した総量」で、
host_moe 195.6 はむしろそれより小さい（前段の一部は MoE begin より前に消化される）。

**expert I/O を消しても MoE 区間は最大 34 ms (c1536) / 20 ms (c8000) しか縮まない。**
token 全体で見た上限も同じ（+16% / +10%）で、しかも §0-4 のとおり読みは帯域飽和なので
実際にはその上限にも届かない。

---

## 2. 実装した先読みの設計

### 2.1 構造

オンデマンド経路は 1 expert を埋めるのに

1. `pread()` で pinned リングへ読む（**呼び出しスレッドがここでブロックする**）
2. その**あと**でないと DMA を発行できない

という順序なので、host は原理的に「GPU より 1 expert 分」しか先に進めない。
先読みは 1 を別スレッドへ移し、`acquire()` 側は「もう読めている pinned バッファから
DMA を発行するだけ」にする。実装は `src/expert_cache.cuh` の `ExpertReadAhead` と
`ExpertStagePath::submit()` の read-ahead 分岐。

### 2.2 予測（どの expert を先読みするか）

ルーティング結果は層の MoE の直前にしか分からず、expert キャッシュからは見えない
（`acquire(layer, expert)` が 1 件ずつ呼ばれる）。そこで
**直前のトークンで同じ層が選んだ expert 列**を予測に使う。

- decode は 1 トークンごとに層 0..N-1 を 1 回ずつ歩くので、**層番号が減ったらトークン境界**。
  その時点で「今のトークンの層別 expert 列」を「次のトークンの予測」へ差し替える。
- ある層の**最初の acquire** のときにだけ予測を評価する。
  `expert_prefetch_depth` は何層先まで読むか (1 = その層の残り、2 = その層 + 次の層)。
- **すでにキャッシュに載っている expert は post しない**（resident フィルタ）。
- 予測が外れた場合は何もしない。外れた予測はリングに残り、次の post で再利用される。

この resident フィルタが本 task の分岐点になった（§3.2）。

### 2.3 リングの状態機械

| 状態 | 意味 | 遷移させる主体 |
|---|---|---|
| `kFree` | 再利用可。最後に buf を読んだ DMA は完了済み | 呼び出し側 |
| `kQueued` | 読み込みジョブを publish 済み。リーダーが buf に書いてよい | 呼び出し側 |
| `kReady` | `pread()` 完了。まだ誰も使っていない | リーダースレッド |
| `kFailed` | `pread()` 失敗。`kReady` と同じく再利用対象 | リーダースレッド |
| `kBorrowed` | `submit()` が取り、DMA を**発行する前** | 呼び出し側 |
| `kInFlight` | DMA を発行し、完了イベントを記録済み | 呼び出し側 |

**`kInFlight` を分けているのが安全の要**。エントリごとの完了イベントは「そのイベントを
最後に記録した DMA」の完了を表すので、DMA を発行した直後（イベント記録前）に
`cudaEventQuery` を見ると**前世代の完了**が見える。その窓で再利用すると、DMA が
読んでいる最中のバッファをリーダーが上書きする。`kInFlight` へ移すのは
`cudaEventRecord` の**後**だけにして、その状態のエントリのみ再利用候補にしている。

その他の安全側の性質:

- 予測はあくまで pinned バッファを埋めるだけで、**キャッシュの index / 追い出し順 /
  スロット配置には一切触れない**。`acquire()` のオンデマンド経路もそのまま残る。
- 同じ key は同時に 2 回 queue しない。
- 読めなかったエントリは `kFailed` になり、**絶対に fill に使われない**（オンデマンドへ落ちる）。
- リングが埋まったら post を諦める（`prefetch_dropped`）。ブロックしない。
- ファイルディスクリプタはエントリが持ち、リーダーは共有状態を触らない。
  `ExpertFileMap::resolve()` は GGUF のシャードを跨ぐと fd を開き直すので、
  **解決してから fd を読む**順序にしてある。

### 2.4 parity が不変である理由

先読みが DMA の供給元にするのは、**オンデマンド経路が同じキーで発行するのと
同一の `pread()`・同一のファイルオフセット**で読んだバイト列である。
したがって予測が当たったときにスロットへ入るバイト列は 1 バイトも変わらない。
`tests/expert_prefetch_host_test.cpp` が「先読み fill とオンデマンド fill の
`memcmp` が 0」でこれを直接固定している。

### 2.5 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `WARODECODE_EXPERT_PREFETCH` | **0 (off)** | 1 で先読み有効。off ではリングを確保すらせず、`submit()` は Task K 以前と同一経路 |
| `WARODECODE_EXPERT_PREFETCH_DEPTH` | メモリ帯の `expert_prefetch_depth` | 何層先まで読むか (1〜3)。正の値を渡すと有効化も兼ねる |
| `WARODECODE_EXPERT_PREFETCH_THREADS` | 2 | 先読みリーダーのスレッド数（1 = リーダーのみ） |
| `WARODECODE_EXPERT_PREFETCH_SLOTS` | depth × 16 (上限 64) | pinned エントリ数。1 エントリ = スロット 1 個分 |
| `WARODECODE_EXPERT_PREFETCH_ALL` | **0 (off)** | 診断用。resident でも post する（リングの実機検証と、無駄読みの値付けに使う） |

既定を off にしてあるのは、効果がキャッシュ容量・帯域・vCPU 数に依存するため。
`Qwen38Tuning::expert_prefetch_depth` はこの実装で**実際に消費される**
（`qwen38_tuning_for_gpu` がメモリ帯で決め、`qwen38_runtime.cuh` が
`ExpertCacheOptions` へ渡し、`ExpertStagePath` がリングを駆動する）。

---

## 3. 検証

### 3.1 ホストテスト

`bash tests/expert_prefetch_host_check.sh`（45 チェック、PASS）。CUDA スタブを使って
実 `pread` / 実 mmap 上で駆動する。

- `ExpertReadAhead` 単体: 内容一致 / 同一 key の二重 queue 拒否 / 未登録 key を渡さない /
  EOF を超える読みの失敗が fill に使われない / 失敗後もリングが復帰する / `release()`。
- `ExpertStagePath` 経由: 予測列が前トークンの層別リストであること、現在 acquire 中の
  expert を含まないこと、depth 1 が次層へ手を出さないこと、層ごとに 1 回だけ評価されること。
- **先読み fill とオンデマンド fill のバイト一致**（スロットのギャップが触られないことを含む）。
- `GpuExpertCache` を丸ごと駆動: 2 層 × 4 expert の偽モデルで「1 トークン目は予測しない →
  2 トークン目で post する → リーダーが読む → 予測 expert がリングから供される」。
  スロット 2 個のキャッシュで 1 つ目を意図的に追い出し、**予測が non-resident になる
  唯一の条件**を作っている。

`tests/qwen38_cuda_syntax_check.sh` も PASS（nvcc 無しの型・構文検査）。

### 3.2 実機 A/B（warpdecode-l4b / arch=89 / 64 token greedy）

- 実行: `tools/expert_prefetch_colab.py --plan l4`（`tests/expert_prefetch_ab.py` を VM 上で実行）
- ソース: **凍結ツリー** `git archive HEAD` + Task K の 3 ファイル。
  計測時、共有ツリーは別担当の編集途中で nvcc が通らなかった
  (`src/qwen38_forward.cuh:2175: q38_hc_f32_inject_enabled が未定義`) ため、
  `WARODECODE_SOURCE_DIR` で凍結コピーを指してビルドした。
- 汚染チェック: GPU compute app 数のサンプリングで **max=1**（自分の engine のみ）。

| 構成 | tok_s | parity | moe dev | moe host | stage/tok | pread/tok | pf post | pf cand | pf resid | pf used |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c8000 off | **5.173** | 64/64 | 85.96 | 176.97 | 19.61 | 16.79 | 0 | 0 | 0 | 0 |
| c8000 pf1 | 5.174 | 64/64 | 85.85 | 177.14 | 19.55 | 16.76 | 0 | 28549 | 28549 | 0 |
| c8000 pf2 | 5.151 | 64/64 | 86.65 | 177.76 | 20.07 | 17.00 | 0 | 58159 | 58159 | 0 |
| c8000 pfall | 5.033 | 64/64 | 91.38 | 181.98 | 24.92 | 21.78 | 28549 | 28549 | 0 | 0 |
| c1536 off | **4.714** | 64/64 | 104.88 | 195.64 | 34.09 | 29.30 | 0 | 0 | 0 | 0 |
| c1536 pf1 | 4.706 | 64/64 | 105.36 | 195.62 | 34.60 | 29.18 | 0 | 28549 | 28549 | 0 |
| c1536 pf2 | 4.696 | 64/64 | 105.67 | 195.82 | 35.01 | 29.42 | 0 | 58159 | 58159 | 0 |
| c1536 pfall | 4.548 | 64/64 | 112.50 | 202.84 | 44.72 | 39.35 | 28549 | 28549 | 0 | 0 |

（`pf2` の候補が 2 倍なのは、depth 2 が「その層 + 次の層」を予測するため。）

ベースラインの妥当性: この run の off は c8000 5.173 / c1536 4.714。
Task H の同時期実測は c8000 5.287 / 5.181、c1536 4.795 / 4.787 で、2% 以内で一致する。

### 3.3 A/B の読み方

1. **既定の先読み (pf1/pf2) は 1 件も post していない。**
   予測候補 28,549 件が **28,549 件すべて resident** でフィルタに落ちた。
   予測＝前トークンの expert 列は、キャッシュにとって「直近に使った集合」＝
   ヒット集合そのものなので、**先読みすべき対象（ミスする expert）と交わらない**。
   c1536 の `hit_rate` 0.3878 と `prev_token_reuse` 0.3877 が一致しているのが直接の証拠で、
   このキャッシュは実質「1 トークン分のキャッシュ」であり、ミスは
   「前トークンで選ばれなかった expert」＝履歴から予測できない。
   → pf1/pf2 が off と同じ数値なのは「効果が無い」のではなく「**何も実行していない**」。
2. **pfall（診断）はリングを実際に動かした。**
   28,549 件を post し、28,541 件をリングへ届けた（失敗 0）。parity は 64/64 のまま。
   ただし予測先はすべて resident なので `submit()`（ミス経路）には一度も入らず、
   `pf_used` は 0。**無駄読みの値段は −2.7% (c8000) / −3.5% (c1536)**、
   stage 帯域は 15.2 → 12.0 GB/s (c8000) に落ち、MoE device も 86.0 → 91.4 ms に増えた。
3. これは「先読みを非同期化しても速くならない」ことの傍証でもある。pfall の読みは
   **別スレッドが発行していて呼び出し側をブロックしていない**のに、増えた読み分
   (pread +5.0 ms/token) がほぼそのまま token (+5.4 ms/token) に出た。
   つまり制約は host のブロックではなく、**ページキャッシュのコピー帯域と CPU**。
   実際 c1536 の stage は 34.09 ms/token で 17.1 GiB/s、c8000 は 15.2 GiB/s と、
   4 スレッドの実測上限 (≈17.8 GiB/s) の 85〜96% で回っている。

---

## 4. 未検証リスク

- **layer_gap の中身は直接計器を打っていない。** 「per-layer の expert 選択 sync で
  待っている」という結論は、区間の長さ (145〜150 ms/token) と、その区間で host が
  自分でやる仕事が 11 ms/token しかないことからの推定。`q38_select_experts()` は
  本 task の担当ファイル外 (`src/qwen38_moe.cuh`) なので、sync 自体の所要時間は
  計測していない。次に触るなら、あの sync を外す/減らすのが最大のレバーになる。
- **A100 では計測できていない。** `warpdecode-a100` セッションが計測前に失効しており
  (`colab sessions` から消え、exec が 404)、残っていたのは L4 系のみ。
  数値は L4 (arch=89) のもの。A100 はホスト CPU と PCIe が違うので、
  expert I/O の比率は小さくなる（A100 の既存実測でも 20000 MiB で stage 15 ms/token）。
- **予測精度の評価は「28,549/28,549 resident」の1点のみ。** キャッシュ容量を
  1 トークン分より遥かに小さく (例: 256 MiB) すれば予測の一部は non-resident になり、
  先読みが打てるようになる。ただしその構成自体が実用構成ではない。
- **リング本数・スレッド数は掃引していない。** `prefetch_threads=2`, `prefetch_slots=16`
  (depth 1) 固定。pfall では reader が on-demand プールと帯域を奪い合うことが分かったので、
  実運用で先読みを入れるなら `prefetch_threads=1` 側から掃引する価値はある。
- **`kFailed` エントリの再利用**はホストテストで確認済みだが、実機での自然発生は
  観測していない（pfall の 28,549 件は全て成功）。
- **凍結ツリーでの計測**: 計測時の共有ツリーは他担当の編集中でビルドできなかったため、
  HEAD + Task K の 3 ファイルで計測した。他担当の最新変更（attention / HC / MoE の
  最適化）が入った tree では絶対値が動く。off/on の相対比較は同一 tree 内なので有効。
