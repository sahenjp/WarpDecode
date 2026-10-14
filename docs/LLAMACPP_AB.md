# llama.cpp との直接 A/B（A100・同一モデル・同一条件）

Task L。**第三者が同じ手順で再現できる形**で、warp エンジンと llama.cpp の
batch=1 decode を同一 GPU・同一モデル・同一条件で並べて測った記録です。

- セッション: Colab `warpdecode-a100`（NVIDIA A100-SXM4-40GB / CUDA 13.0 / 12 vCPU / RAM 83 GiB）
- モデル: `Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf`（85,923,705,344 bytes、1 ファイル）
- 計測: 2026-10-14 20:35〜21:10 JST（同一セッション内・交互実行）
- 再現: `tools/llamacpp_ab.py`（本文の数値をそのまま出す唯一のコマンド）
- 生データ: `issue1_artifacts/taskl_ab_results.jsonl`（交互 3 ラウンド 16 実行）、
  `issue1_artifacts/taskl_ab_extra_results.jsonl`（大容量 cache 追加 5 実行）

計測に使ったバイナリ（VM 上で `md5sum` を記録）:

| バイナリ | md5 | ビルド元 |
|---|---|---|
| `/content/wd_taskl/warodecode/build-sm80/warodecode` | `61c2ea03c7f30c0e92e142d94a7dfcd9` | 20:22 JST 時点のツリー。`src/` は rev `5933825` と同一（engine コードはその後 `c8390ab` でも不変）。他担当の未コミットな tokenizer/CLI 変更は含まない |
| `/content/wd_taskl/llama.cpp/build-cuda-sm80/bin/taskl-llama-decode` | `dcb055533f989a60053792a157ede108` | pinned commit + `examples/taskl-decode` の計測ハーネス |
| `/content/wd_taskl/llama.cpp/build-cpu-oracle/bin/taskl-llama-decode` | （オラクル再現ビルド） | 同じ commit を `GGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF` でビルド |

## 結論（先に要点）

| 対比 | 本エンジン | llama.cpp | 結果 |
|---|---|---|---|
| VRAM を揃えた decode（約 12.8 GiB 同士） | 5.56 tok/s (14,248 MiB) | **15.00 tok/s** (12,796 MiB) | llama.cpp が **2.7 倍速い** |
| VRAM を揃えた decode（約 20.7 GiB 同士） | 5.88 tok/s (22,248 MiB) | **18.49 tok/s** (20,494 MiB) | llama.cpp が **3.1 倍速い** |
| VRAM を揃えた decode（約 29 GiB 同士） | 6.14 tok/s (36,248 MiB) | **23.06 tok/s** (28,196 MiB) | llama.cpp が **3.8 倍速い** |
| llama.cpp が MoE expert を全部 CPU に置いた場合 | 5.56 tok/s | **12.54 tok/s** (5,098 MiB) | llama.cpp が **2.3 倍速い** |
| llama.cpp 純 CPU（12 スレッド、VRAM 436 MiB） | 6.14 tok/s（最良） | 4.56 tok/s | 本エンジンが **1.35 倍速い** |
| CPU オラクル（pinned llama.cpp CPU 出力）との一致 | **64/64** | GPU パス 36〜40/64、CPU パス 44/64 | 本エンジンのみ一致 |

**正直な結論: A100 40GB では llama.cpp のほうが速い。** 本エンジンの最良値 6.148 tok/s
（expert cache 30,000 MiB、VRAM 36.2 GiB）は、llama.cpp の 12.5〜23.1 tok/s に対して
**2.1〜3.8 倍の劣勢**です（同 VRAM 予算の比較では 2.7〜3.8 倍）。
一方、CPU オラクルとトークン単位で一致するのは本エンジンだけで、llama.cpp は既定ビルドでは
自分の CPU 参照実装すら再現しません（第 5 節）。**速度と数値忠実度は別の軸**として扱うべきです。

## 1. ビルド条件

| 項目 | 本エンジン | llama.cpp（計測用・既定ビルド） | llama.cpp（オラクル再現ビルド） |
|---|---|---|---|
| ソース | このリポジトリ `5933825` | `ggml-org/llama.cpp` `6fcaa16f4b360649933a54d1f91ad40ed35c0e11` | 同左 |
| コミット件名 | — | `Address review comments` | 同左 |
| CUDA | sm80（`-DCMAKE_CUDA_ARCHITECTURES=80`） | sm80、`-DGGML_CUDA=ON` | `-DGGML_CUDA=OFF` |
| CPU カーネル | —（CPU は I/O 専用、計算に使わない） | `-DGGML_NATIVE=ON`（既定）、repack 有効 | `-DGGML_NATIVE=OFF`、`-DGGML_CPU_REPACK=OFF` |
| その他 | `-DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON` | `-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF` | 同左 |
| ビルド | `-j8`、数分 | `-j8`、**11 分 48 秒**（11:23:49→11:35:37 UTC） | `-j8`、約 7 分 |
| Toolkit | CUDA 13.0 (V13.0.88)、nvcc 13.0.88、GNU 13.3.0 | 同左 | 同左 |

- 計測用の既定ビルドは、第三者が `cmake -DGGML_CUDA=ON` だけで作る構成に合わせました。
  この選択がトークン一致に効きます（第 5 節）。
- オラクル再現ビルドは、README の parity 検証に使った CPU オラクルと同じ cmake フラグです
  （`docs/ISSUE1_FINDINGS_20261012.md`、`tools/issue1_cpu_oracle_colab.py`）。

### 計測ハーネス

llama.cpp 側は `qwen38_llamacpp_gpu_bench.cpp` を pinned チェックアウトの
`examples/taskl-decode/` に追加してビルドします。llama.cpp の公開 API だけを使い、

- `n_gpu_layers`（`-ngl` 相当）と、層ごとの MoE expert を CPU に置く tensor override
  （`blk.<n>.ffn_.*_exps` → CPU。llama.cpp の `--n-cpu-moe` と同じ仕組み）
- `n_ctx=128`、`n_batch=1`、`n_ubatch=1`、`n_seq_max=1`、KV = F16、flash attention 無効
- 入力は `--start-token 1` の 1 トークン、以後は **argmax（同点は小さい token id）** を
  そのまま次入力にする自由走行 greedy、64 トークン
- 1 トークンごとに `llama_decode` を壁時計で計測（モデルロードとコンテキスト生成は別会計）

を行います。本エンジン側は
`--start-token 1 --generate 64 --ctx 128 --no-graph --no-autotune` で、同じくモデルロードを除いた
decode 時間を報告します（JSON の `tok_s`）。

## 2. モデルの載せ方（86 GB を 40 GB の VRAM に載せる）

どちらも**重みの一部をホスト側から供給**しますが、供給経路が違います。

| | 本エンジン | llama.cpp |
|---|---|---|
| 常駐 | 非 expert 重み + KV + ワークスペースを VRAM に常駐 | `-ngl 99`（全層）で attention など非 expert を VRAM に |
| expert 重み | VRAM の **expert cache**（`WARODECODE_EXPERT_CACHE_MIB`）にスロットを確保し、トークンごとに必要な expert を `pread` + pinned ring + DMA で転送 | CPU に置いた expert は **mmap 越しに CPU で GEMV**（ページキャッシュ由来）、GPU に置いた expert は VRAM 常駐 |
| ファイルアクセス | `pread`（明示読み） | `mmap`（`LLAMA_LOAD_MODE_MMAP`、既定） |
| CPU スレッド | 計算に使わない | 12（`nproc`） |
| 配置の切り替え | expert cache 8,000 / 16,000 / 24,000 / 30,000 MiB | `--keep-moe-gpu-last K`（末尾 K 層の expert だけ VRAM、残りは CPU） |
| 対応する実測 VRAM | 8,000→14,248 / 16,000→22,248 / 24,000→30,248 / 30,000→36,248 MiB | K=0→5,098 / K=8→12,796 / K=16→20,494 / K=20→24,344 / K=24→28,196 MiB |

**揃えられない点（明記）**: expert の載せ方は「同じ VRAM 予算」以外は同条件ではありません。
本エンジンは必要な expert だけを動的に転送し、llama.cpp は静的に層を割り当てて CPU/GPU で
計算します。カーネル、転送方式、CPU の使い方（本エンジンは 0 スレッド、llama.cpp は 12 スレッド）が
異なるため、**同一条件なのは「同じ GGUF・同じ GPU・ctx 128・batch=1・greedy 64 トークン・
同程度の VRAM 予算」まで**です。

## 3. 速度の対比（3 ラウンド交互・各実行 3 反復）

1 ラウンド = `engine 8000 → llama(MoE 全 CPU) → engine 16000 → llama(K=8) → llama(K=16)` の順で実行し、
3 ラウンド繰り返しました（GPU 状態の偏りを避けるため交互）。各実行でモデルはロードし直し、
**プロセス内 3 反復の最速値**を採用（llama.cpp 側は全反復を生データに記録）。

| 構成 | VRAM 実測 (MiB) | best | median | min | 反復数 | 出力一致 |
|---|---|---|---|---|---|---|
| 本エンジン cache 8,000 MiB | 14,248 | 5.563 | 5.561 | 5.542 | 3 実行×3 反復 | **64/64** |
| llama.cpp MoE 全 CPU（K=0） | 5,098 | 12.541 | 12.533 | 12.407 | 3×3 | 36/64 |
| 本エンジン cache 16,000 MiB | 22,248 | 5.879 | 5.868 | 5.853 | 3×3 | **64/64** |
| llama.cpp K=8（VRAM 12.8 GiB） | 12,796 | 14.999 | 14.581 | 14.579 | 3×3 | 36/64 |
| llama.cpp K=16（VRAM 20.5 GiB） | 20,494 | 18.486 | 18.215 | 17.921 | 3×3 | 36/64 |
| llama.cpp 純 CPU（`-ngl 0`、12 threads） | 436 | 4.560 | — | — | 1×3 | 44/64 |

単位は tok/s。VRAM はプロセス実行中の `nvidia-smi memory.used` 最大値。

### 大容量 cache 側の追加計測（1 ラウンド×3 反復・交互）

| 構成 | VRAM 実測 (MiB) | best | 出力一致 |
|---|---|---|---|
| 本エンジン cache 24,000 MiB | 30,248 | 6.137 | **64/64** |
| llama.cpp K=20（VRAM 23.8 GiB） | 24,344 | 20.390 | 36/64 |
| 本エンジン cache 30,000 MiB | 36,248 | 6.136 | **64/64** |
| llama.cpp K=24（VRAM 27.5 GiB） | 28,196 | 23.056 | 40/64 |
| 本エンジン cache 30,000 MiB（2 回目） | 36,248 | 6.148 | **64/64** |

### VRAM 予算を揃えた対比（これが本タスクの主結果）

| VRAM 予算 | 本エンジン | llama.cpp | 比 |
|---|---|---|---|
| 約 13 GiB | 5.563（cache 8,000 MiB、実測 14,248 MiB） | 14.999（K=8、実測 12,796 MiB） | **llama.cpp が 2.70 倍** |
| 約 21 GiB | 5.879（cache 16,000 MiB、実測 22,248 MiB） | 18.486（K=16、実測 20,494 MiB） | **llama.cpp が 3.14 倍** |
| 約 29 GiB | 6.142（cache 30,000 MiB、実測 36,248 MiB） | 23.056（K=24、実測 28,196 MiB） | **llama.cpp が 3.75 倍** |

llama.cpp は MoE expert を **全部 CPU に置いても** 12.5 tok/s（VRAM 5,098 MiB）で、本エンジンの
最良値 6.148 の 2.0 倍です。本エンジンが expert の供給に使っている PCIe 転送経路より、
12 スレッドの CPU GEMV + ページキャッシュのほうが速い——これがこの GPU での結論です
（本エンジンの内訳分析 `docs/TASK_H_EXPERT_IO.md` が示す「expert 転送が支配的」という
観測と整合します）。

参考: llama.cpp のオラクル再現ビルド（CUDA 無し、純 CPU）は 12 スレッドで 4.892 tok/s、
2 スレッドで 1.844 tok/s。**純 CPU の llama.cpp と本エンジンの差は 1.26 倍**しかありません。

## 4. 速度以外の比較

| 項目 | 本エンジン | llama.cpp |
|---|---|---|
| モデルロード時間（実測） | 約 2.5〜3.6 秒（`wall` − 3 反復ぶんの decode。`--load-only` は本ビルドで rc=1 になり未計測） | 1.00 秒（`-ngl 0`）/ 2.25 秒（K=0）/ 4.46 秒（K=8）/ 6.74 秒（K=16）/ 7.77 秒（K=20）/ **24.37 秒（K=24）** |
| 1 実行の wall time（3 反復込み） | 34.7〜37.3 秒 | 19.4 秒（K=0・VRAM 5 GiB）〜37.5 秒（K=24・VRAM 28 GiB） |
| VRAM 使用量 | cache 指定 + 約 6.2 GiB の固定分 | 約 4.5 GiB の固定分 + 層ごとの expert |
| ホスト側の作業 | `pread` による expert 転送（CPU 計算なし） | 12 スレッドの CPU GEMV |
| 出力（CPU オラクルとの一致） | **64/64**（全 24 実行で維持） | GPU パス 36〜40/64、CPU パス 44/64（既定ビルド） |
| バイナリ | `build-sm80/warodecode` 5,666,192 B | `llama-cli` 1,422,344 B + 共有ライブラリ（`libggml-cuda.so` 66,341,668 B ほか） |
| 実装の汎用性 | qwen4exp 専用（専用 forward） | 汎用（本モデル用カーネルは pinned commit 側に存在） |

## 5. 出力一致（重要な発見）

| 実行 | 一致 |
|---|---|
| 本エンジン（cache 8,000 / 16,000 / 24,000 / 30,000 MiB） | **64/64**（全 24 実行） |
| llama.cpp 既定ビルド、GPU パス（K=0/8/16/20） | 36/64（36 番目で分岐。分岐後のトークン列は 4 構成で完全に同一） |
| llama.cpp 既定ビルド、GPU パス（K=24） | 40/64（配置を変えると分岐点も動く） |
| llama.cpp 既定ビルド、CPU パス（`-ngl 0`、12 スレッド） | 44/64 |
| llama.cpp オラクル再現ビルド、CPU パス（2 / 12 スレッド） | **64/64** |

読み方:

1. **ハーネスの実装は正しい。** オラクルと同じ cmake フラグでビルドすると、自由走行 greedy で
   CPU オラクルと 64/64 一致します（本エンジンが合わせている相手を、別実装でも再現できる）。
2. したがって 36/64・44/64 は**ハーネスのバグではなく、ビルド構成による数値差**です。
   既定ビルド（`GGML_NATIVE=ON` / `GGML_CPU_REPACK=ON`）では CPU 側の縮約順・SIMD 経路が変わり、
   活性の量子化コードが反転して数十トークンで分岐します。これは README の
   「量子化重みは活性も量子化する／演算順と丸めを合わせる」という知見と同じ現象です。
3. GPU パスが CPU パスより早く（36 vs 44）分岐するのは CUDA カーネルの縮約順の違いです。
   GPU 4 構成で分岐後の列が同一なのは、GPU 側の数値経路が配置に依らず同じだからです。
4. つまり「llama.cpp と一致しているか」は **どのビルドの llama.cpp か**を指定しないと意味を持ちません。
   本エンジンの parity 64/64 は、pinned commit + オラクル cmake フラグの CPU 出力に対する一致です。

## 6. 再現手順

### 一括（推奨）

```bash
export PATH=/home/sahen/.local/bin:$PATH
# 事前に colab sessions でセッションの空きを確認し、nvidia-smi で他プロセスが無いことを確認する
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/llamacpp_ab.py --rounds 3 --reps 3 --llama-k 8 --llama-k2 16
```

これで VM 上に

1. pinned llama.cpp の CUDA sm80 ビルド（`/content/wd_taskl/llama.cpp/build-cuda-sm80`）
2. 本エンジンの sm80 ビルド（`/content/wd_taskl/warodecode/build-sm80`）と計測ハーネス
   （`examples/taskl-decode/taskl-llama-decode`）
3. オラクル一致チェックと配置スイープ（phase `probe`）
4. 3 ラウンド交互の A/B（phase `matrix`）

を作り、`issue1_artifacts/taskl_ab_results.jsonl` に全実行の生データを落とします。
段階ごとの再実行は `--phases` で指定します（`--phases matrix` / `--phases none --collect`）。

### 手動で同じことをする場合（VM 上）

```bash
git clone https://github.com/ggml-org/llama.cpp /content/llama.cpp
cd /content/llama.cpp && git checkout 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
cmake -S . -B build-cuda-sm80 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON
cmake --build build-cuda-sm80 -j 8 --target llama-cli llama-bench

# qwen38_llamacpp_gpu_bench.cpp を examples/taskl-decode/taskl-llama-decode.cpp として置き、
# CMakeLists.txt に add_executable(...)+target_link_libraries(... PRIVATE llama) を書いて
cmake -S . -B build-cuda-sm80   # 再構成
cmake --build build-cuda-sm80 -j 8 --target taskl-llama-decode

# llama.cpp: MoE expert を全部 CPU に置いて decode
bin/taskl-llama-decode /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
  128 64 99 --keep-moe-gpu-last 0 --threads 12 --reps 3

# llama.cpp: オラクル再現ビルド（parity の基準）
cmake -S . -B build-cpu-oracle -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DGGML_CPU_REPACK=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cpu-oracle -j 8 --target taskl-llama-decode
build-cpu-oracle/bin/taskl-llama-decode <model> 128 64 0 --threads 2 --reps 1   # -> 64/64

# 本エンジン
WARODECODE_EXPERT_CACHE_MIB=16000 build-sm80/warodecode \
  --model <model> --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune \
  --generate 64 --bench-repeat 3
```

## 7. 未検証リスク

- **40GB VRAM 1 枚での結果**です。モデル 85.9 GB > VRAM 40 GB という条件が両者の載せ方を歪めており、
  モデルが VRAM に全部載る GPU では順位が変わり得ます（未検証）。
- expert の載せ方は「同じ VRAM 予算」以外は同条件ではありません（第 2 節）。
- ページキャッシュの状態は制御していません（RAM 83 GiB に対しファイル 85.9 GB）。両者とも
  プロセス内で 3 反復して warm 側の値を使っており、交互実行で偏りを避けていますが、
  cold 状態だけを切り出した比較は未取得です。
- プリフィル（prefill）は未計測です（本タスクは batch=1 decode の対比）。
- llama.cpp の MTP / speculative decoding、`--n-cpu-moe` 以外の配置指定、
  `LLAMA_LOAD_MODE_DIRECT_IO`、flash attention 有効化、KV Q8 は未計測です
  （ハーネスは `--no-mmap` / `--direct-io` / `--flash 1` に対応済み）。
- 本エンジンの設定最適化は既定（DP4A、HEAD_Q8_K、copy stream、Task J の HC mix 既定）で、
  expert cache 以外は振っていません。6.148 tok/s が本エンジンの上限である証明にはなりません。
- llama.cpp 側は 1 セッション・1 GPU・1 モデルの結果で、llama-bench など公式ツールでの
  追試はしていません（本ハーネスは同じライブラリの公開 API を使っています）。
