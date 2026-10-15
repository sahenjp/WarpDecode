# llama.cpp との直接 A/B（A100・同一モデル・同一条件・同一ジョブ内で交互計測）

Task M。**第三者が同じ手順で再現できる形**で、warp エンジンと llama.cpp の batch=1 decode を
同一 GPU・同一モデル・**同一ジョブ内で交互に**測った記録です。

- セッション: Colab `warpdecode-a100`（NVIDIA A100-SXM4-40GB / CUDA 13.0 / 12 vCPU / RAM 83 GiB）
- モデル: `Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf`（85,923,705,344 bytes、1 ファイル）
- 計測: 2026-10-15 10:08:32〜10:13 UTC（19:08〜19:13 JST）。**1 ジョブ・18 実行・交互**
- 再現: `tools/llamacpp_ab.py`（phase `ab3`）＋ `tests/llamacpp_ab_report.py`
- 生データ: `issue1_artifacts/taskl_ab3_results.jsonl`（18 実行）、集計 `issue1_artifacts/taskl_ab3_summary.json`

## 0. 結論（先に要点）

| 対比（VRAM は実測ピーク） | 本エンジン | llama.cpp | 同一ラウンド内の比 | 判定 |
|---|---|---|---|---|
| cache 8,000 MiB（14,256 MiB） vs K=0（5,098 MiB） | 11.719 tok/s | 12.307 tok/s | **0.954**（3 ラウンド 0.947〜0.954） | 引き分け（5% 以内） |
| cache 20,000 MiB（26,256 MiB） vs K=16（20,494 MiB） | 18.042 tok/s | 17.911 tok/s | **1.012**（1.000〜1.037） | 引き分け（5% 以内） |
| cache 30,000 MiB（36,256 MiB） vs K=24（28,196 MiB） | 18.052 tok/s | 23.006 tok/s | **0.789**（0.784〜0.791） | **llama.cpp の勝ち（約 1.27 倍）** |

**正直な結論: この構成では「llama.cpp に勝った」とは言えません。**
3 組のうち 2 組は統計的に引き分け（±5% 以内）、1 組は明確な負けです。
さらに **VRAM はどの組でも本エンジンのほうが多い**（+9 GiB / +5.8 GiB / +8.1 GiB）ので、
VRAM あたりの効率では全組で劣っています。

ただし前回（2026-10-14・Task L）の「2.7〜3.8 倍の差」は解消し、**0.79〜1.01 倍まで詰まりました**。
本エンジン側が 5.56〜6.14 → 11.7〜18.1 tok/s に伸びたことが主因です。

### 唯一「勝っている」と言える見方（条件付き）

VRAM を無視して「同じモデル・同じ GPU で 1 トークンあたり何秒か」だけを見ると、
cache 20,000 MiB（26.3 GiB）の 18.04 tok/s は llama.cpp K=0（5.1 GiB）の 12.31 tok/s に対して **1.47 倍**です。
ただしこれは **5.2 倍の VRAM を使った比較**であり、性能主張として使うには条件が違いすぎます。
「VRAM 予算を揃えた比較」（本節の表）が本来の比較です。

## 1. なぜ「同一ジョブ内の交互計測」が必要か

この VM は**同一バイナリ・同一構成でも decode の total が 99.4 ↔ 129.6 ms（30%）動きます**
（`docs/SYNC_OVERHEAD.md`）。したがって「別の時間帯に測った 7.6 tok/s と 12.43 tok/s を比べる」
という比較は**無効**です。本タスクでは

1. llama.cpp とエンジンを**同じジョブの中で交互に**実行し、
2. **同じラウンドの中の比だけ**を根拠にし、
3. 各ラウンドの先頭で `nvidia-smi` により**GPU が空であること**を確認し、
4. 実行中も 1 秒ごとに GPU プロセスを記録して**他プロセスの混入を検出**する

という条件で測りました。ラウンド間の絶対値は比較に使いません。

## 2. ビルド条件と載せ方

| 項目 | 本エンジン | llama.cpp（計測ハーネス込み） |
|---|---|---|
| ソース | このリポジトリ rev `78b8f43c9e0b62f74dc988931d205c8137cd3eb8`（作業ツリーは clean） | `ggml-org/llama.cpp` `6fcaa16f4b360649933a54d1f91ad40ed35c0e11`（`Address review comments`） |
| ソース同一性 | `src/` + CMakeLists の SHA-256 `aa58f11b5b239e40f15250680915cc2d82197a53b59aa5c6da534c2e0f40ceba`（ビルド時に `REVISION.txt` として同梱） | `git rev-parse HEAD` をビルドスクリプトで照合 |
| CUDA | sm80（`-DCMAKE_CUDA_ARCHITECTURES=80`） | sm80、`-DGGML_CUDA=ON -DGGML_NATIVE=ON`、`-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF` |
| ビルド | `-DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON`、`-j8` | `-j8`（CUDA オブジェクトが大半、約 12 分） |
| バイナリ md5 | `517f1bcda48d2c9845ded5aab2b5d4e6`（`build-sm80/warodecode`） | `dcb055533f989a60053792a157ede108`（`build-cuda-sm80/bin/taskl-llama-decode`、51,416 B） |
| ツールキット | CUDA 13.0.88、cmake 3.31.10、GNU 13.3.0 | 同左 |

llama.cpp 側のハーネスは `qwen38_llamacpp_gpu_bench.cpp` を pinned チェックアウトの
`examples/taskl-decode/` に追加してビルドします（公開 API のみ使用）。
`tools/colab_build_llamacpp.sh` も同じハーネスまで作るように更新しました。

### モデルの載せ方（86 GB を 40 GB の VRAM に載せる）

| | 本エンジン | llama.cpp |
|---|---|---|
| 常駐 | 非 expert 重み + KV + ワークスペースを VRAM に常駐 | `-ngl 99`（全層）で attention など非 expert を VRAM に |
| expert 重み | VRAM の **expert cache**（`WARODECODE_EXPERT_CACHE_MIB`）にスロットを確保し、必要な expert を `pread` + pinned ring で転送 | CPU に置いた expert は **mmap 越しに CPU で GEMV**（ページキャッシュ由来）、GPU に置いた expert は VRAM 常駐 |
| CPU の使い方 | **計算に使わない**（I/O スレッドのみ） | 12 スレッドで CPU GEMV |
| 配置スイッチ | expert cache 8,000 / 20,000 / 30,000 MiB | `--keep-moe-gpu-last K`（末尾 K 層の expert だけ VRAM） K=0 / 16 / 24 |

**揃えられない点（明記）**: 同一なのは「同じ GGUF・同じ GPU・ctx 128・batch=1・greedy 64 トークン」
までです。カーネル、expert の供給経路、CPU の使い方が違います。
また VRAM は本エンジンのほうが常に多く、**VRAM を揃えた比較にはなっていません**（第 5 節で補足）。

## 3. 計測方法

- 1 ラウンド = `engine 8,000 → llama K=0 → engine 20,000 → llama K=16 → engine 30,000 → llama K=24`。
  偶数ラウンドは**逆順**（`llama K=24 → engine 30,000 → …`）で実行し、単調なドリフトが
  ランタイム間の差に見えないようにしています。
- 各実行の前に `nvidia-smi --query-compute-apps` が空になるまで待機（最大 30 分）。
  全 18 実行で待ち時間 0 秒、実行中に観測された GPU プロセスは**各実行 1 個（自分自身のみ）**でした。
- 本エンジン: `--model <gguf> --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune
  --generate 64 --bench-repeat 3`（プロセス内 3 反復の最速値）。
- llama.cpp: `taskl-llama-decode <gguf> 128 64 99 --keep-moe-gpu-last K --threads 12 --reps 3`
  （`n_ctx=128`、`n_batch=n_ubatch=1`、`n_seq_max=1`、KV=F16、flash attention 無効、
  自由走行 greedy・argmax を次の入力にする）。
- **両者とも「プロセス内 3 反復の最速値」**を採用します。llama.cpp の 1 反復目は毎回遅く
  （ページフォルトと CUDA モジュールロード）、本エンジンはその値を報告しないためです。
- 入力は `--start-token 1` の 1 トークンから 64 トークンを自由走行 greedy（`-p` で 1 トークン与えるのと同じ）。
  生成列は CPU オラクル `reference_tokens_64.json`（SHA-256 `2bb72a88…`）と照合します。

> **計測したリビジョンの確認（重要）**: ビルド後に VM 上の `REVISION.txt` を取得し、
> `78b8f43` の `src/` + `CMakeLists.txt` を `git archive` で取り出して再計算した SHA-256 が
> 同梱値 `aa58f11b…` と一致することを確認しました。計測したエンジンはこのリビジョンそのものです。
> なお `78b8f43` には**この A/B 用のツール変更（`tools/llamacpp_ab.py` /
> `tools/colab_build_llamacpp.sh` / `tests/llamacpp_ab_report.py`）も同梱されています**（別担当が
> 一括でコミットしたため）。エンジンのバイナリを決めるのは `src/` なので、速度の数値には影響しません。

## 4. 結果（3 ラウンド・同一ジョブ内の交互計測）

### 実行ごとの統計（メトリクス = プロセス内最速値）

| 実行 | ラウンド | best | median | worst | VRAM ピーク (MiB) | 64 トークン一致 | 観測 GPU プロセス数 |
|---|---|---|---|---|---|---|---|
| engine cache 8,000 MiB | 3 | 11.742 | 11.719 | 11.704 | 14,256 | **64/64** | 1 |
| llama.cpp K=0（MoE 全 CPU） | 3 | 12.377 | 12.307 | 12.264 | 5,098 | 36/64 | 1 |
| engine cache 20,000 MiB | 3 | 18.124 | 18.042 | 17.987 | 26,256 | **64/64** | 1 |
| llama.cpp K=16 | 3 | 17.989 | 17.911 | 17.397 | 20,494 | 36/64 | 1 |
| engine cache 30,000 MiB | 3 | 18.150 | 18.052 | 17.931 | 36,256 | **64/64** | 1 |
| llama.cpp K=24 | 3 | 23.024 | 23.006 | 22.664 | 28,196 | 40/64 | 1 |

単位は tok/s。**ラウンド間の絶対値は比較に使いません**（同一構成でも 30% 動く VM のため）。

### 同一ラウンド内の比（これが本タスクの主結果）

| ラウンド | engine 8,000 | llama K=0 | 比 | engine 20,000 | llama K=16 | 比 | engine 30,000 | llama K=24 | 比 |
|---|---|---|---|---|---|---|---|---|---|
| 1（順方向） | 11.719 | 12.377 | 0.947 | 17.987 | 17.989 | 1.000 | 18.052 | 23.024 | 0.784 |
| 2（逆方向） | 11.704 | 12.264 | 0.954 | 18.042 | 17.397 | 1.037 | 18.150 | 23.006 | 0.789 |
| 3（順方向） | 11.742 | 12.307 | 0.954 | 18.124 | 17.911 | 1.012 | 17.931 | 22.664 | 0.791 |
| **中央値** | 11.719 | 12.307 | **0.954** | 18.042 | 17.911 | **1.012** | 18.052 | 23.006 | **0.789** |

比のばらつきは 3 ラウンドで **1% 未満**（0.947〜0.954 / 1.000〜1.037 / 0.784〜0.791）。
ラウンド内の相対値は十分に信頼できます。

### 判定

| 対比 | 中央値の比 | 判定（±5% を引き分けとする） |
|---|---|---|
| cache 8,000 vs K=0 | 0.954 | **引き分け**（本エンジンが 4.6% 遅いが 5% 以内） |
| cache 20,000 vs K=16 | 1.012 | **引き分け**（1.2% 差） |
| cache 30,000 vs K=24 | 0.789 | **llama.cpp の勝ち**（本エンジンが 21% 遅い） |

### メトリクス感度（結論が指標に依存しないか）

| 指標 | cache 8,000 vs K=0 | cache 20,000 vs K=16 | cache 30,000 vs K=24 |
|---|---|---|---|
| プロセス内最速（採用） | 0.954（引き分け） | 1.012（引き分け） | 0.789（負け） |
| プロセス内中央値 | 0.967（引き分け） | 1.026（引き分け） | 0.811（負け） |
| 1 反復目のみ（llama.cpp に不利な下限） | 1.047（引き分け） | 1.113（勝ち） | 0.866（負け） |

**指標を変えると 20,000 MiB の組は「勝ち」にも「引き分け」にもなります。**
これは誤差ではなく「1 反復目のコストを誰が払うか」の違いなので、
本エンジンの勝ち主張には使えません（採用は最速値どうしの比較）。
一方 30,000 MiB の組はどの指標でも負けです。

### 読み取れること

1. **本エンジンは cache 20,000 MiB で飽和します。** 18.042 → 18.052 tok/s（+0.05%）。
   30,000 MiB に増やすと VRAM を 10 GiB 多く使って速度は変わりません。
2. **llama.cpp は K に比例して伸びます。** 12.307 → 17.911 → 23.006 tok/s。
   VRAM を増やすほど素直に速くなり、K=24 で本エンジンを 1.27 倍上回ります。
3. **本エンジンは常に VRAM を多く使います**（+9.2 / +5.8 / +8.1 GiB）。
   「引き分け」の 2 組も、より多くの VRAM を使って同速という意味では負けています。

## 5. VRAM を揃えた場合の補足

第 3 節の 3 組は VRAM が揃っていません（要求仕様どおりの組です）。参考として実測 VRAM を並べると:

| 本エンジン | VRAM | llama.cpp | VRAM | 参考: 同じ VRAM 帯の llama.cpp |
|---|---|---|---|---|
| cache 8,000 MiB | 14,256 MiB | K=0 | 5,098 MiB | K=8（旧計測で 12,796 MiB / 15.0 tok/s） |
| cache 20,000 MiB | 26,256 MiB | K=16 | 20,494 MiB | K=20（旧計測で 24,344 MiB / 20.4 tok/s） |
| cache 30,000 MiB | 36,256 MiB | K=24 | 28,196 MiB | — |

同じ VRAM 帯（約 13 GiB と約 24 GiB）で比べると、llama.cpp は本エンジンの 18.0〜20.4 tok/s に対して
15.0〜20.4 tok/s を出しており、**本エンジンが VRAM を多く使って同速かやや速い**という関係です。
（K=8 / K=20 の値は 2026-10-14 の旧計測で、同一ラウンドの比ではないため参考値です。今回の主結果には使いません。）

## 6. 速度以外の比較

| 項目 | 本エンジン | llama.cpp |
|---|---|---|
| 出力（CPU オラクルとの一致） | **64/64（18 実行すべて）** | 36/64（K=0 / K=16）、40/64（K=24） |
| 1 実行の wall time（3 反復込み） | 14.8 秒（cache 20,000 / 30,000）、19.0〜19.3 秒（cache 8,000） | 19.7〜23.3 秒 |
| モデルロード | 約 2.5 秒（`wall` − decode。expert は遅延ロード） | mmap のため数十 ms だが初回トークンでページフォルトを払う |
| ホスト側の作業 | `pread` による expert 転送（CPU 計算なし） | 12 スレッドの CPU GEMV |
| 汎用性 | qwen4exp 専用（専用 forward） | 汎用 |

## 7. 出力一致（重要・前回から変わらない発見）

| 実行 | 一致 |
|---|---|
| 本エンジン（cache 8,000 / 20,000 / 30,000 MiB × 3 ラウンド = 9 実行） | **64/64（全 9 実行）** |
| llama.cpp 既定ビルド GPU パス（K=0 / K=16 = 6 実行） | 36/64（36 番目で分岐、分岐後の列はすべて同一） |
| llama.cpp 既定ビルド GPU パス（K=24 = 3 実行） | 40/64 |

読み方（2026-10-14 の調査の再確認）:

1. ハーネスは正しく、pinned commit を**オラクル再現ビルド**（`GGML_CUDA=OFF -DGGML_NATIVE=OFF
   -DGGML_CPU_REPACK=OFF`、2 スレッド）で作れば CPU オラクルと 64/64 一致します。
2. したがって 36/64・40/64 は**ハーネスのバグではなくビルド構成による数値差**です。
   既定ビルド（`GGML_NATIVE=ON`）では CPU 側の縮約順・SIMD 経路が変わり、量子化活性が反転して
   数十トークンで分岐します。
3. つまり「llama.cpp と一致しているか」は**どのビルドの llama.cpp か**を指定しないと意味を持ちません。
   本エンジンの parity 64/64 は pinned commit + オラクル cmake フラグの CPU 出力に対する一致です。

## 8. 再現手順

```bash
export PATH=/home/sahen/.local/bin:$PATH
# 0) 前提: A100 セッションと 85.9 GB の GGUF が /content/models にあること
colab sessions                     # warpdecode-a100 が空いていることを確認
nvidia-smi                         # 他プロセスがいないことを確認（1 セッション 1 ジョブ）

# 1) ビルド（llama.cpp 約 12 分 + エンジン約 5 分）と交互 A/B を 1 ジョブで流す
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/llamacpp_ab.py --phases build_llamacpp,build_engine,ab3 \
    --rounds 3 --reps 3 --engine-caches 8000,20000,30000 --llama-ks 0,16,24

# 2) 集計（同一ラウンド内の比と勝敗）
python3 tests/llamacpp_ab_report.py issue1_artifacts/taskl_ab3_results.jsonl \
  --pairs engine_8000=llama_k0,engine_20000=llama_k16,engine_30000=llama_k24
```

補足: この環境の `google-colab-cli` には `exec -c` が無いため、`tools/llamacpp_ab.py` は
すべてのスニペットを一時ファイル経由（`exec -f`）で実行します。
`tools/colab_build_llamacpp.sh` は llama.cpp 本体とハーネスだけを作る単体版です。

## 9. 未検証リスク

- **A100 40GB・1 GPU・1 モデル**の結果です。モデル 85.9 GB > VRAM 40 GB という条件が両者の
  載せ方を歪めています。モデルが VRAM に全部載る GPU では順位が変わり得ます（未検証）。
- expert の載せ方は「同じ GGUF・同じ ctx・同じ batch」以外は同条件ではありません（第 2 節）。
  **llama.cpp は 12 スレッドの CPU 計算を使い、本エンジンは CPU を計算に使いません。**
- 本エンジンはどの組でも VRAM を多く使います。**VRAM を完全に揃えた組（例: cache 8,000 vs K=8）**は
  今回の主結果には含んでいません（第 5 節の参考値のみ）。
- ページキャッシュの状態は制御していません（RAM 83 GiB に対しファイル 85.9 GB）。
  交互実行で偏りは避けていますが、cold 状態だけを切り出した比較は未取得です。
- GPU クロックは固定していません（Colab では不可）。実行中の `clocks.sm` は記録していません。
- 本エンジンの `tok_s` は**プロセス内 3 反復の最速値のみ**で、中央値や分散は取得できません
  （llama.cpp 側は全反復を記録しています）。
- プリフィル（prefill）は未計測です（batch=1 decode の対比）。
- llama.cpp の MTP / speculative decoding、flash attention 有効化、KV Q8、`--nproc-moe` は未計測です。
- 「本エンジンが勝つ」構成の探索（cache と K の総当たり、MoE の載せ方の変更）は未実施です。

## 参考: 2026-10-14 の計測（Task L・別時間帯・現在は無効）

同じ 2 ランタイムを A100 で測った最初の記録です。**別の時間帯に測った絶対値どうしの比較**であり、
この VM の 30% ドリフトを考えると現在は根拠として使えません。履歴として残します。

| 構成 | VRAM 実測 (MiB) | best tok/s | 出力一致 |
|---|---|---|---|
| 本エンジン cache 8,000 MiB | 14,248 | 5.563 | 64/64 |
| 本エンジン cache 16,000 MiB | 22,248 | 5.879 | 64/64 |
| 本エンジン cache 24,000 MiB | 30,248 | 6.137 | 64/64 |
| 本エンジン cache 30,000 MiB | 36,248 | 6.148 | 64/64 |
| llama.cpp MoE 全 CPU（K=0） | 5,098 | 12.541 | 36/64 |
| llama.cpp K=8 | 12,796 | 14.999 | 36/64 |
| llama.cpp K=16 | 20,494 | 18.486 | 36/64 |
| llama.cpp K=20 | 24,344 | 20.390 | 36/64 |
| llama.cpp K=24 | 28,196 | 23.056 | 40/64 |
| llama.cpp 純 CPU（`-ngl 0`、12 threads） | 436 | 4.560 | 44/64 |

当時の結論は「A100 40GB では llama.cpp が 2.7〜3.8 倍速い」。
今回の同一ジョブ内交互計測では **0.79〜1.01 倍**まで差が縮んでおり、
**その後の最適化（HC mix、expert I/O、router 並列化）で 2〜3 倍改善した**ことが分かります。

---

## 付録: README 貼り付け用の「比較」節（案）

README.md の `### llama.cpp (CUDA) との直接 A/B（A100 40GB・同一モデル・同一条件）`
節（現行: 5.56〜6.15 tok/s vs 12.54〜23.06 tok/s・「2.7〜3.8 倍」）は**古い値**です。
同一ジョブ内の交互計測に置き換える案を以下に示します。**反映は親が行ってください。**

同じ節の直前にある表の `decode (A100 16000MiB)` の行（5.61〜5.88 tok/s）も
今回の計測では cache 20,000 MiB で 18.0 tok/s なので、あわせて更新すると整合します。

````markdown
### llama.cpp (CUDA) との直接 A/B（A100 40GB・同一ジョブ内で交互計測）

同じセッション・同じ 85.9GB GGUF で、pinned llama.cpp を CUDA sm80 でビルドし、
**同一ジョブの中で交互に** batch=1 decode を実測しました。この VM は同一構成でも
total が 30% 動くため、**同じラウンドの中の比だけ**を根拠にしています
（詳細と再現手順: `docs/LLAMACPP_AB.md`）。

| 構成 | VRAM 実測 | decode | 出力一致 |
|---|---|---|---|
| 本エンジン expert cache 8,000 MiB | 14,256 MiB | 11.72 tok/s | **64/64** |
| llama.cpp `-ngl 99`（MoE expert 全 CPU・12 threads, K=0） | 5,098 MiB | 12.31 tok/s | 36/64 |
| 本エンジン expert cache 20,000 MiB | 26,256 MiB | 18.04 tok/s | **64/64** |
| llama.cpp `-ngl 99`（末尾 16 層の expert を GPU, K=16） | 20,494 MiB | 17.91 tok/s | 36/64 |
| 本エンジン expert cache 30,000 MiB | 36,256 MiB | 18.05 tok/s | **64/64** |
| llama.cpp `-ngl 99`（末尾 24 層の expert を GPU, K=24） | 28,196 MiB | 23.01 tok/s | 40/64 |

**正直な結論: この構成では「llama.cpp より高速」とは言えません。**
3 組のうち 2 組（8,000 MiB vs K=0、20,000 MiB vs K=16）は同一ラウンド内の比が 0.95〜1.01 で
誤差の範囲（±5%）の引き分け、30,000 MiB vs K=24 は比 0.79 で **llama.cpp が 1.27 倍速い**という結果です。
しかも VRAM はどの組でも本エンジンのほうが 5.8〜9.2 GiB 多く、VRAM 効率では全組で劣ります。
3 ラウンドの比は 1% 未満のばらつきで一致しており、この判定は安定しています。

一方で前回計測（2026-10-14・別時間帯の比較）の「2.7〜3.8 倍の差」は
**0.79〜1.01 倍まで縮小**しました。HC mix、expert I/O、router 並列化などの最適化で
本エンジンが 5.6〜6.1 → 11.7〜18.1 tok/s に伸びたためです。
なお expert cache を 20,000 → 30,000 MiB に増やしても 18.04 → 18.05 tok/s で飽和します。

一方で CPU オラクルと 64/64 一致するのは本エンジンだけで、llama.cpp は既定ビルド
（`GGML_NATIVE=ON`）だと自分の CPU 参照実装すら再現しません（36〜40/64）。
速度と数値忠実度は別の軸として読んでください。

再現:

```bash
export PATH=/home/sahen/.local/bin:$PATH
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/llamacpp_ab.py --phases build_llamacpp,build_engine,ab3 \
    --rounds 3 --reps 3 --engine-caches 8000,20000,30000 --llama-ks 0,16,24
python3 tests/llamacpp_ab_report.py issue1_artifacts/taskl_ab3_results.jsonl
```
````
