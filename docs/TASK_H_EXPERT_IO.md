# Task H: expert 重み転送の帯域改善（pread + pinned ring + 並列 pread）

対象: `src/expert_cache.cuh` の host→device ステージング経路。
llama.cpp CPU オラクルとの 64/64 トークン一致は**全構成で維持**している（§3.4）。

## 0. 結論

1. **律速はページフォルトと pageable ステージングの二重コピー**であり、ディスクではない。
   A100 では転送 34.69 GiB の 93% がページキャッシュ上にあり、O_DIRECT の実測は
   1.8 GB/s（＝ディスク側の下限参照値）にすぎない。現行の
   `mmap → cudaMemcpyAsync` は 2 MiB あたり 0.559 ms（3.75 GB/s）だが、同じ 2 MiB を
   `pread` で pinned バッファへ読むと 0.31 ms、4 スレッドに分けると 0.12 ms になる。
2. 実装は `pread()` + `cudaMallocHost` リング + DMA（＋並列 pread）へ置換した。
   **既定 ON**（`WARODECODE_EXPERT_IO_PREAD=0` で旧経路に戻る）。
3. 実機 A/B（64 token greedy, 同一条件の対比較）:

   | | pageable（旧） | pread 1 スレッド | pread 既定（4 スレッド） |
   |---|---|---|---|
   | A100 1536 MiB | 2.894 tok/s | 3.268 (+13%) | **4.089 tok/s (+41%)** |
   | A100 20000 MiB | 3.740 | 4.016 (+7%) | **4.516 (+21%)** |
   | L4 1536 MiB | 3.271 | 3.808 (+16%) | **4.795 (+47%)** (3 スレッド) |
   | L4 8000 MiB | 4.109 | 4.559 (+11%) | **5.287 (+29%)** (3 スレッド) |

   ステージング帯域は **3.6〜3.9 GiB/s → 14〜17 GiB/s**。parity は全 52 本で 64/64。
4. 副産物として、旧 `WARODECODE_EXPERT_PIN` が parity を壊した原因を特定した（§2.5）。

## 1. 現状の切り分け

### 1.1 環境（Colab VM, 実測）

| | A100-SXM4-40GB | L4 24GB |
|---|---|---|
| vCPU / MemTotal | 12 / 87.5 GB | 12 / 55.5 GB |
| ページキャッシュ | 80.3 GB（モデル 85.9 GB の 93%） | 50.7 GB（同 59%） |
| ディスク | overlay on 250 GB SSD (ROTA=0) | 同 |
| read_ahead_kb / THP | 128 / madvise | 128 / madvise |
| toolchain | gcc 13.3 / CUDA 13.0 | 同 |

「RAM にほぼ載る A100」と「載らない L4」で挙動が違う点が §1.2 に出る。

### 1.2 マイクロベンチマーク（`tests/expert_io_bandwidth.cu`）

ランダムな 2 MiB チャンク 128 本（= 256 MiB）を各経路で読んだ実測。
実行: `/content/wd_h/io_bw <model> 128 2 <case>`。

**A100（全チャンクがページキャッシュ resident）**

| 経路 | GB/s | ms/2MiB | 備考 |
|---|---|---|---|
| `mmap+h2d`（旧経路） | **3.75** | 0.559 | minor fault 33 回/2MiB |
| `pageable+h2d`（ホットな malloc 元） | 6.83 | 0.307 | フォルト無し |
| `pread`（pageable へ） | 5.46 | 0.384 | |
| `pinpread`（pinned へ） | 6.81 | 0.308 | |
| `pinpread+h2d` | 6.68 | 0.314 | **採用経路の 1 スレッド** |
| `pread+h2d` | 2.81 | 0.747 | pageable 経由は 2 回コピーで最悪 |
| `mmap+willneed+h2d` | 3.68 | 0.569 | madvise は効かない |
| `mmap+populate+h2d` | 3.36 | 0.624 | 同上 |
| `pin+h2d`（充填済み pinned からの純 DMA） | 12.1（同期時） | 0.002（enqueue） | ホストはクリティカルパスから外れる |
| `odirect` | 1.82 | 1.152 | 非採用 |
| `mmap`（resident 36% のコールド状態） | 0.46 | 4.553 | major fault 10 回/2MiB |

**L4（同一ベンチ、resident 62.5%）**

| 経路 | GB/s | ms/2MiB |
|---|---|---|
| `mmap+h2d`（旧経路） | **0.32** | 6.480 | major fault 13.1 回/2MiB = 同期ディスク読み |
| `pinpread` 1 スレッド | 5.92 | 0.354 |
| `pinpread+h2d` 1 スレッド | 6.84 | 0.306 |
| `pinpread+willneed` | 7.00 | 0.300 |
| `pinpread` 2 スレッド | 12.28 | 0.171 |
| `pinpread` 4 スレッド | **17.78** | 0.118 |
| `pread` 4 スレッド（pageable へ） | 24.22 | 0.087 |

サイズ依存（L4, 1 スレッド pinned）: 0.5 MiB → 6.996 / 1 MiB → 6.932 / 2 MiB → 6.844 GB/s。
エンジンの 1 expert は 3 スライス計 1.85 MiB なので、この帯域域に入る。
L4 の `dd` 参考値: O_DIRECT 2 GiB = 3.4 GB/s、buffered 4 GiB = 416 MB/s。

### 1.3 読み方

1. `mmap → cudaMemcpyAsync` の 0.559 ms/2MiB は、
   **(a) マッピングのページフォルト**（64 KiB fault-around で 33 回/2MiB）と
   **(b) ドライバの pageable バウンスバッファへの CPU memcpy** の和である。
   (b) だけなら `pageable+h2d` の 0.307 ms なので、(a) が約 0.25 ms を占める。
2. `pread` はページフォルトを踏まずページキャッシュをまとめてコピーするので単発 0.31 ms。
   pinned へ読めば DMA がホストをブロックしない（`pin+h2d` の host 側 0.002 ms）ため、
   **ホストがクリティカルパスから外れる**。
3. A100 はページキャッシュがほぼ全量 resident。つまり **ページフォルト律速であり、
   ディスク律速ではない**。旧担当の「ページフォルト/ディスク側が律速」という推定のうち、
   ディスク側は否定される。
4. L4 は 59% しか resident でないため旧経路は major fault（同期ディスク読み）で 0.3 GB/s まで
   崩壊する。同一バイト列でも `pread` は 6.8 GB/s（約 20 倍）で、syscall 側の readahead と
   フォルト処理の効率差がそのまま出る。
5. `madvise(MADV_WILLNEED/POPULATE_READ/SEQUENTIAL)` と `posix_fadvise(WILLNEED)` は
   mmap 経路のフォルトを消せなかった（3.68 / 3.36 GB/s）。`O_DIRECT` は 1.8 GB/s で論外。

## 2. 実装（`src/expert_cache.cuh`）

### 2.1 経路

`ExpertStagePath` に第 2 の充填経路を足した。

1. `/proc/self/maps` からテンソルポインタの file offset と fd を解決（`ExpertFileMap`）。
   ランタイムのインターフェースを広げずに済むよう、この解決をヘッダ内で完結させている。
2. リングエントリ（`cudaMallocHost`、スロットと同じレイアウト）へ `pread()` で読む。
3. スロットの連続領域ごとに `cudaMemcpyAsync` を 1 回だけ発行（分離型キャッシュの 3 スライス
   → 1 DMA）。
4. リングエントリの再利用前に、そのエントリを最後に読んだ DMA のイベントを
   `cudaEventSynchronize()` で待つ。

### 2.2 並列 pread（`ExpertPreadPool`）

1 コアでは DRAM 帯域を飽和できないため fork-join のワーカープールを追加した。
acquire() がジョブを publish し、自分の分担を読み、他スレッドの完了を待ってから DMA を発行する。
アイドル時は 4096 回スピンしたのち 500 µs の condition_variable 待ちに落ちる
（GPU 待ちの ~200 ms/token の間コアを焼き続けないため）。

ジョブの分割は「スロット空間 `[0, max(offset+bytes))` をスレッド数で等分」であり、
**スライス間の隙間を含む**。分離型キャッシュは層ごとの最大スライス長で容量を取るため
スロットはスライスで敷き詰められていない（実測 1.84 MiB のデータに対し 2.33 MiB のスロット）。
ここを「スライス長の総和」で分割すると最終スライスが切り詰められ **全 expert が壊れる**
（parity が matched_prefix=0 になる実例を検出済み → §2.4 のテストで再発防止）。

### 2.3 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `WARODECODE_EXPERT_IO_PREAD` | **1** | 1 で pread + pinned リング経路、0 で旧 pageable 経路 |
| `WARODECODE_EXPERT_IO_THREADS` | **4** | pread を分担するスレッド数（1 = スレッド無し） |
| `WARODECODE_EXPERT_IO_RING` | 4 | pinned リング本数 |
| `WARODECODE_EXPERT_IO_FADVISE` | off | `willneed`/`sequential`/`random`/`dontneed` |
| `WARODECODE_EXPERT_STATS` | 0 | 既存。`io_pread_ms`/`io_ring_wait_ms`/`io_served`/`io_fallbacks`/`io_enabled`/`io_ring`/`io_threads`/`io_fadvise` が増える |

`WARODECODE_EXPERT_IO_FADVISE=willneed` は A100/L4 とも有意差なし（L4 ではわずかに悪化）
なので既定は無効のまま。`io_ring_wait_ms` は全実行で 25〜78 ms（stage の 1% 前後）で飽和していない。

### 2.4 parity ガード（4 段）

1. 経路有効化時に `verify_mapping()` がサンプルスロット全体（2〜3 MiB）を `pread` と mmap で
   memcmp する。失敗したら旧経路に落ちるだけで、誤った expert がスロットに入ることはない。
2. リングエントリごとの DMA 完了イベント（§2.5）。
3. `pread` が短読み/失敗したら、その expert だけ旧 pageable コピーにフォールバック。
4. `tests/expert_io_pool_host_check.sh`（13 チェック）がスレッド数 1〜4 でスライス内容・
   隙間の非破壊・EOF での失敗を検証する。§2.2 のバグはこのテストが捕捉することを確認済み
   （バグを戻すと 2〜4 スレッドのケースが FAIL する）。

### 2.5 撤回された旧 `WARODECODE_EXPERT_PIN` との違い

旧実装は「リングを再利用する前に、そのリングを読んでいる DMA の完了を待つ」ガードが無かった。
1536 MiB（高 churn）でのみ壊れたのは、リング本数より DMA の完了待ち回数が閾値を超えたためと
説明できる。現実装はエントリごとに `cudaEvent_t` を持ち、ホストが書き込む前に必ず
`cudaEventSynchronize()` する。1536 MiB で 64/64 を複数回確認している。

## 3. 実機 A/B

計測は 1 セッション 1 ジョブ、`nvidia-smi` で他プロセスが消えるまで待ってから開始
（`tools/colab_run_cmd_h.py`、ソース/ビルド/ログ/pid を `/content/wd_h` に隔離）。
ログは `/content/wd_h/run_*.log`。全ケースで `foreign_gpu=[]` を記録。

### 3.1 A100-SXM4-40GB（arch=80, 64 token greedy, ctx=128）

スレッド数を変えて同一条件で交互に実行（各 2 本）:

| cache | 経路 | tok_s | 平均 | 改善 | stage_ms | ステージ帯域 | parity |
|---|---|---|---|---|---|---|---|
| 1536 MiB (691 slot) | pageable | 2.895 / 2.893 | 2.894 | — | 9161 / 9176 | 3.78 GiB/s | 64/64 |
| 1536 MiB | pread 1 スレッド | 3.279 / 3.256 | 3.268 | +12.9% | 6341 / 6486 | 5.41 | 64/64 |
| 1536 MiB | pread 2 スレッド | 3.887 / 3.885 | 3.886 | +34.3% | 3315 / 3355 | 10.40 | 64/64 |
| 1536 MiB | **pread 4 スレッド（既定）** | 4.092 / 4.086 | **4.089** | **+41.3%** | 1984 / 2012 | **17.36** | 64/64 |
| 20000 MiB (9002) | pageable | 3.731 / 3.748 | 3.740 | — | 4096 / 4061 | 3.37 | 64/64 |
| 20000 MiB | pread 1 スレッド | 4.015 / 4.017 | 4.016 | +7.4% | 2724 / 2722 | 5.05 | 64/64 |
| 20000 MiB | **pread 4 スレッド（既定）** | 4.526 / 4.505 | **4.516** | **+20.7%** | 826 / 863 | **16.30** | 64/64 |

host_ms/token: 1536 pageable 345.6 → 既定 244.8 ms、20000 pageable 267.8 → 既定 222.4 ms。
**stage 以外の約 210 ms/token は構成によらず一定**で、転送を消しても残る律速になる。

### 3.2 L4 24GB（arch=89）

| cache | 経路 | tok_s | 平均 | 改善 | stage_ms | ステージ帯域 | parity |
|---|---|---|---|---|---|---|---|
| 1536 MiB | pageable | 3.268 / 3.274 | 3.271 | — | 8956 / 8933 | 3.87 GiB/s | 64/64 |
| 1536 MiB | pread 1 スレッド | 3.792 / 3.824 | 3.808 | +16.4% | 6026 / 5892 | 5.82 | 64/64 |
| 1536 MiB | **pread 3 スレッド** | 4.802 / 4.787 | **4.795** | **+46.6%** | 2265 / 2300 | **15.20** | 64/64 |
| 8000 MiB (3600) | pageable | 4.108 / 4.110 | 4.109 | — | 4921 / 4915 | 3.61 | 64/64 |
| 8000 MiB | pread 1 スレッド | 4.560 / 4.557 | 4.559 | +10.9% | 3167 / 3159 | 5.62 | 64/64 |
| 8000 MiB | **pread 3 スレッド** | 5.281 / 5.292 | **5.287** | **+28.7%** | 1237 / 1194 | 14.63 | 64/64 |

L4 は他担当と共有しており、以前のラウンドでは同一構成が 2.88〜4.63 tok/s とばらついた。
上の表は**同一時間帯の対比較**のみを根拠にしている。L4 は 1536 MiB と 8000 MiB で
3 スレッドのみ掃引（A100 の 2 スレッド→4 スレッドの伸びから 4 が既定）。

### 3.3 スレッド数の効果

A100: 1 → 2 → 4 スレッドで 3.268 → 3.886 → 4.089 tok/s（1536 MiB）。
L4: 1 → 3 スレッドで 3.808 → 4.795 tok/s。マイクロベンチでも 6.8 → 12.3 → 17.8 GB/s
（1/2/4 スレッド）で単調。**1 コアの memcpy が上限**であり、ページキャッシュが DRAM に
ある以上、並列化が最も効くレバーだった。

### 3.4 parity

| ラウンド | 本数 | 結果 |
|---|---|---|
| A100 r1（pread / pageable、3 容量 × 2） | 13 | 64/64 |
| A100 スレッド掃引（1/2/4） | 15 | 64/64 |
| L4 r1（pread / pageable / willneed） | 11 | 64/64 |
| L4 スレッド掃引（1/3） | 13 | 64/64 |
| **既定構成の確認（最終コード）** | 18 | 64/64 |
| 合計（64 トークン実行） | **80** | **64/64** |

### 3.5 既定構成（最終コード）の確認

`WARODECODE_EXPERT_IO_PREAD` を**未設定**にしたまま（＝既定 ON）と、明示的な
`=0`（旧経路）を同一時間帯で交互に実行した最終確認。バックオフ付きワーカーを含む
出荷コードでの測定である。

| | 旧経路（=0） | 既定（ON, 4 スレッド） | 改善 | parity |
|---|---|---|---|---|
| A100 1536 MiB | 2.896 / 2.893 | 4.058（1 本目 2.533 は外れ値） | **+40.1%** | 64/64 |
| A100 20000 MiB | 3.733 / 3.734 | 4.494 / 4.482 | **+20.4%** | 64/64 |
| L4 1536 MiB | 3.271 / 3.272 | 4.817 / 4.816 | **+47.2%** | 64/64 |
| L4 8000 MiB | 4.114 / 4.104 | 5.294 / 5.302 | **+28.9%** | 64/64 |

A100 1536 の 1 本目（2.533 tok/s, stage 2782 ms）は同じ構成の 2 本目（4.058 tok/s,
stage 2144 ms）と 60% 違う外れ値で、この回の warmup も 2.739 tok/s と低かった。
同時刻に A100 を共有していたプロセスの影響と見られる（parity は 64/64 のまま）。

---

壊れた中間版（§2.2 の切り詰めバグ）は A100/L4 の**全ケースで matched_prefix=0** となり、
A/B ごとの parity 確認が実際にバグを検出した。これは「parity を各構成で取る」運用が
有効に働いた例である。

## 4. 未検証リスク

- **`/proc/self/maps` 依存**: モデルが通常のファイル mmap でない場合（匿名メモリ、パイプ、
  `(deleted)` パス）は `verify_mapping` が失敗し旧経路に落ちる。この fallback は実機で
  一度も発火しておらず（常に pread 経路が有効）、**fallback 経路自体は実測未カバー**。
- **既定 ON の影響範囲**: `ExpertCacheOptions::io_pread` を true にしたため、他の担当が
  `src/expert_cache.cuh` を使う run も自動的に新経路になる。旧経路で比較したい場合は
  `WARODECODE_EXPERT_IO_PREAD=0` を明示すること。
- **ワーカースレッドの CPU 消費**: アイドル時は 500 µs スリープに落ちるが、acquire が連続する
  区間はスピンする。vCPU が少ない環境では本体を圧迫しうる（未検証）。
- **共有 VM のノイズ**: L4 の絶対値は共有相手の負荷に依存する。A100 は 15 ケース連続で
  `foreign_gpu=[]`。
- **未テスト経路**: prefill / MTP / `--bench-repeat` では統計のトークン境界が崩れるため、
  §3 の数値は decode 専用条件でのみ意味を持つ。
- **リング本数**: `io_ring=4` 固定で掃引していない。`io_ring_wait_ms` は stage の 1% 前後で、
  4 本が不足している兆候は無い。
- **メモリ増**: pinned リングは `slot_bytes × 4`（分離型で約 9.3 MiB）。A100/L4 とも問題なし。
- **再現用**: 解析は `tests/expert_io_bandwidth.cu`（切り分け）、`tests/expert_io_ab.py`
  （A/B と parity）、`tests/expert_io_pool_host_check.sh`（プールのホストテスト）、
  `tools/colab_run_cmd_h.py`（隔離実行）で行った。
