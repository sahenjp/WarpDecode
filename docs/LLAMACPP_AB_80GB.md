# A100 80GB での llama.cpp 比較（同一ジョブ内・交互実行）

対象: Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf (85.9 GB) / batch=1 decode / ctx 128 /
greedy 64 トークン / oracle は pinned llama.cpp の CPU greedy（SHA-256 で固定）。
VM: **NVIDIA A100-SXM4-80GB（81,920 MiB）** / RAM 167 GB / 12 vCPU（Colab）。
すべて同一セッション・交互実行・3 ラウンド・best-of-3。

---

## 0. 結論

| 条件 | 本エンジン | llama.cpp | 比 | 出力一致 |
|---|---|---|---|---|
| A100 **40GB**・VRAM 28.2 GiB 同等（cache 22,000 vs K=24） | **26.58 tok/s** | 23.62 tok/s | **+12.5%** | 64/64 vs 40/64 |
| A100 **80GB**・本エンジン cache 20,000 MiB | **26.82 tok/s** | 23.18 tok/s（K=24） | **+15.7%** | 64/64 vs 40/64 |
| A100 **80GB**・本エンジン cache 70,000 MiB（VRAM 76 GB） | **26.80 tok/s** | — | — | 64/64 |

**80GB の A100 でも、このモデルで有効な llama.cpp 構成のうち最速は K=24（23.18 tok/s）で、
本エンジンが +15.7% 上回ります。** VRAM は本エンジンのほうが 2 GiB 少ない
（26,256 MiB vs 28,196 MiB）条件での比較です。

---

## 1. 本エンジン: cache を増やしても速度は変わらない（が、遅くもならない）

| expert cache | VRAM 実測 | 3 ラウンド | 中央値 | 出力一致 |
|---|---|---|---|---|
| 20,000 MiB | 26,256 MiB | 26.820 / 26.862 / 26.625 | 26.82 tok/s | **64/64** |
| 50,000 MiB | 56,256 MiB | 26.798 / 26.556 / 26.803 | 26.80 tok/s | **64/64** |
| 70,000 MiB | 76,256 MiB | 26.870 / 26.802 / 26.479 | 26.80 tok/s | **64/64** |

80 GB の VRAM をほぼ使い切る cache 70,000 MiB（総 VRAM 76,256 MiB）でも正常動作し、
parity は 64/64 のままです。ただし **64 トークンの生成では cache 20,000 MiB ですでに
全 working set（7,448 expert = 14.7 GB）が載る**ため、VRAM を増やしても速度は上がりません
（docs/CEILING.md の「compulsory miss の床」）。40GB 機の 26.58 tok/s とほぼ同じ値です。

より長い生成（数千トークン）では distinct expert 集合が 20 GB を超えるため、
80 GB のほうが有利になります。この 64 トークン比較はその効果を含みません。

---

## 2. llama.cpp 側: 80GB では expert を GPU に寄せるほど出力が壊れる

pinned llama.cpp（`refs/pull/28243/head`, commit `6fcaa16f`）の
`--keep-moe-gpu-last K` は「末尾 K 層の expert を GPU に置く」で、K=48 は
**全 expert を GPU に置く＝素の `-ngl 99` と等価**です（CPU 側 override が 0 個になる）。

| K（GPU に置く MoE 層数） | VRAM | 最速反復 tok/s | 64 トークン一致 |
|---|---|---|---|
| 16 | 20,494 MiB | 17.9 | 36/64 |
| **24** | 28,196 MiB | **23.2** | **40/64** |
| 32 | 37,000 MiB 前後 | 36.4 | 10/64 |
| 40 | 45,000 MiB 前後 | 68.5 | 22/64 |
| 44 | 50,000 MiB 前後 | 110.1 | 6/64 |
| 48（全部 GPU） | 51,470 MiB | 561〜586 | **6/64** |

- K=48 は **64 トークンを 0.110 秒**で生成します（1.72 ms/token）。このモデルの
  1 トークンあたりの重みは約 **4.7 GB**（routed expert 950 MB + GDN/QSA 2.2 GB +
  HC/router/shared/head 1.5 GB）なので、1.72 ms で処理するには **2.7 TB/s** 必要で、
  A100-80GB の HBM ピーク 2.04 TB/s を超えます。**計算が飛んでいる**と判断できます。
- 実際、出力は 6/64 で、K=0（全 expert CPU, 36/64）や K=24（40/64）より悪化します。
  つまり K≥32 の高速値は「速いが不正解」で、比較に使えません。
- 同一ジョブ内で engine は全ラウンド 64/64 を維持しています。

---

## 3. 本家 mainline llama.cpp はこのモデルをロードできない

公平のため、PR を含まない mainline（`c0cb53ac4e34a0c16cbb4bdc963f0b5f3e60eacf`,
"sync : ggml"）を CUDA sm80 でビルドして同じ harness で `-ngl 99`（全 GPU）を試しました。

    llama_model_load: error loading model:
      done_getting_tensors: wrong number of tensors; expected 1257, got 1256

**ロード自体が失敗します。** qwen4exp（Qwen3.8-Flash-Next）対応は pinned PR 側にしかなく、
したがって「80GB で全 expert を GPU に載せた llama.cpp」を評価できる唯一のビルドが
上記の PR ビルドであり、その全 GPU 構成は出力が壊れています。

---

## 4. 再現手順

    export PATH=/home/sahen/.local/bin:$PATH
    WARODECODE_COLAB_SESSION=<session> WARODECODE_COLAB_ARCH=80 \
      python3 tools/llamacpp_ab.py --phases build_llamacpp,build_engine,ab3 \
      --rounds 3 --reps 3 --engine-caches 20000,50000,70000 --llama-ks 24,48,48

K のスイープ（32/40/44）は同じバイナリで `--llama-ks 32,40,44` を指定します。
engine の tok/s は自身の JSON、llama.cpp は harness の `LLAMA_AB=` JSON から取り、
どちらも `matched_prefix`（CPU oracle との一致トークン数）を併記しています。

---

## 5. 注意

- 前節のとおり **K≥32 の llama.cpp 数値は不正解**なので、速度表に混ぜてはいけません。
  harness が parity を併記するのはこのためです。
- 40GB 機の比較（+12.5%）と 80GB 機の比較（+15.7%）は、いずれも同じ pinned
  llama.cpp ビルド・同じ GGUF・同一ジョブ内の交互実行です。
- 本エンジンの 80GB 値（26.8 tok/s）は 40GB 値（26.6 tok/s）とほぼ同じで、
  これは 64 トークンでは expert cache 20 GB で足りているためです。
