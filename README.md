# warodecode

NVIDIA GPU上の **batch=1 LLM decode速度** に対象を絞り、llama.cppの汎用性コストを捨てて速度を取りにいくCUDAランタイムです。

現時点では開発中です。「llama.cppより高速」という性能主張は、同一モデル・同一GPUで実測して5%以上勝った構成だけに限定します。

ライセンス: [MIT](LICENSE)

## まずTUIを起動

CUDAが未導入でもTUIだけは起動できます。

```bash
./setup.sh
```

TUIでできること:

- `nvidia-smi`からNVIDIA GPU・VRAM・Compute Capabilityを自動検出
- 登録ディレクトリからGGUFを自動検出
- モデルとGPUを選択して起動
- 複数モデルを複数GPUへ1プロセスずつ同時起動
- 選択GPU向けの専用バイナリをビルド
- context / generation長 / autotuneの設定保存
- GPU世代ごとの目標低bitカーネルを表示

キー:

```text
Tab       モデル / GPUを切替
↑ ↓       移動
Enter     単体起動の主選択
Space     複数起動・ビルド対象を選択
L         選択モデルを選択GPUで起動
M         選択した複数モデルを別GPUで同時起動
B         選択GPU向けエンジンをビルド
A         モデルディレクトリを追加
R         GPU/モデル再スキャン
C         context長
N         generation長
T         Q4カーネルautotune ON/OFF
P         選択モデルの実行plan表示
Q         終了
```

設定は `~/.config/warodecode/tui.conf` に保存します。

## Qwen3.8-Flash-Next (qwen4exp) の CPU オラクル一致

CUDAエンジンは llama.cpp (`6fcaa16f4b360649933a54d1f91ad40ed35c0e11`) のCPU greedy出力と
**64/64トークン完全一致**します (Issue #1)。

| 項目 | 値 |
|---|---|
| モデル | Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf (85.9 GB) |
| 一致 | **64/64トークン** (全最適化後も維持、通算200本以上で確認) |
| decode (L4 8000MiB, batch=1, ctx=128) | **5.29 tok/s** |
| decode (A100 16000MiB) | **5.61〜5.88 tok/s**（単発計測 / 3反復の最速値） |
| prefill (L4 8000MiB, 同条件) | **4.63 tok/s** |
| 速度 (F16高速パス基準) | 2.38 tok/s → **2.2〜2.4倍** |
| 速度 (最適化前の本エンジン) | 3.20 tok/s → **1.65倍** |

### llama.cpp (CUDA) との直接 A/B（A100 40GB・同一モデル・同一条件）

同じセッション・同じ 85.9GB GGUF で、pinned llama.cpp を CUDA sm80 でビルドして
batch=1 decode を交互に実測しました（詳細と再現手順: `docs/LLAMACPP_AB.md`）。

| 構成 | VRAM 実測 | decode | 出力一致 |
|---|---|---|---|
| 本エンジン expert cache 8,000 MiB | 14,248 MiB | 5.56 tok/s | **64/64** |
| 本エンジン expert cache 16,000 MiB | 22,248 MiB | 5.88 tok/s | **64/64** |
| 本エンジン expert cache 30,000 MiB | 36,248 MiB | **6.15 tok/s** | **64/64** |
| llama.cpp `-ngl 99`（MoE expert 全 CPU・12 threads） | 5,098 MiB | 12.54 tok/s | 36/64 |
| llama.cpp `-ngl 99`（末尾 8 層の expert を GPU） | 12,796 MiB | 15.00 tok/s | 36/64 |
| llama.cpp `-ngl 99`（末尾 16 層の expert を GPU） | 20,494 MiB | 18.49 tok/s | 36/64 |
| llama.cpp 純 CPU（`-ngl 0`・12 threads） | 436 MiB | 4.56 tok/s | 44/64 |

**A100 40GB では llama.cpp のほうが速い**（VRAM 予算を揃えると 2.7〜3.8 倍）。
本エンジンが勝つのは「純 CPU の llama.cpp に対して約 1.3 倍」の範囲です。
一方で CPU オラクルと 64/64 一致するのは本エンジンだけで、llama.cpp は既定ビルド
（`GGML_NATIVE=ON` / repack 有効）だと自分の CPU 参照実装すら再現しません
（オラクルと同じ cmake フラグでビルドすると 64/64 に戻ることが確認済み）。
速度と数値忠実度は別の軸として読んでください。

再現:

```bash
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
  python3 tools/llamacpp_ab.py --rounds 3 --reps 3 --llama-k 8 --llama-k2 16
```

### 速度の内訳と効いた最適化

| 最適化 | 効果 |
|---|---|
| **expert 重みの転送を pread + pinned ring + DMA に** | ステージング 3.7 → **14〜17 GiB/s**、decode **+21〜47%** |
| expert cache 容量 (1536 → 8000 MiB) | **+24〜36%** |
| DP4A + コードブックの shared memory 化 | **+6%** (IQ2_S カーネル単体 5.1倍) |
| copy stream による転送の分離 | GPU 混雑時に **+33%** |
| expert 単位カーネル統合 (60→5 launch/層) | +0.55% (launch は支配的でなかった) |

計測で判明した支配的要因は **expert 重みの転送**で、最適化前はトークン時間の 41〜64% を占めていました
(実効 3.7 GB/s = ページフォルト律速)。カーネル計算ではなく I/O が本体だったのが最大の知見です。

再現 (Colab L4 + モデルが必要):

```bash
python3 tools/issue1_parity_check.py
```

### 一致させるために必要だったこと

batch=1で1e-7単位まで一致させるには、llama.cppのCPU実装と同じ**演算順・同じ丸め**を再現する必要があります。
差が残ると活性の量子化コードが反転し、層ごとに増幅します。実際に効いた修正:

1. **量子化重みは活性も量子化する** — ggmlのper-typeルール (Q8_0/IQ4_NL→Q8_0, IQ2_S/IQ3_S/Q6_K→Q8_K, BF16→BF16)
2. **`--use_fast_math` の超越関数はオラクルと一致しない** — device `expf` は `__expf` (MUFU, ~5e-7) に、
   `/` も近似除算になる。SiLUはggmlのAVX2多項式、sigmoid/exp/softplusはglibcのアルゴリズム移植で一致させた
   (`src/warodecode_q38_glibc_trig.hpp`, `src/qwen38_exact_math.cuh`)
3. **IMRoPE** — glibcの `sinf`/`cosf` 移植 + `theta *= theta_scale` の逐次乗算 + FMA形の回転
4. **F32テンソルをhalfに落とさない** — HC γ/inject、GDNのβ/α/norm、PLE norms、QSA norms、
   共有エキスパートgateなど、F32の重みをhalf化すると必ず相対5e-4の系統誤差が入る
5. **F32 GEMVは `ggml_vec_dot_f32` のAVX2順序** (4×8アキュムレータ + F32x8還元) で縮約する
6. **MoEルータ重み** — 全512 expertのsoftmax → 8レーンdouble和 → 逆数乗算 → top-k → double和 → clamp → 真の除算
7. **QSA attention** — max先行 → 全要素exp → double行和 → **確率のF16丸め** → 4×8アキュムレータ
8. **スコアドットのqはF16に丸める** — KVキャッシュがF16なのでCPUは `ggml_cpu_fp32_to_fp16(q)` してから内積する

現状、position 1で **L0〜L7 がbit一致**、次の障壁はL8です。

### ホスト検証ハーネス

オラクルとの一致は、pinned llama.cppを実際にビルドして比較するハーネスで確認できます
(いずれも第1引数にllama.cppのチェックアウトを渡します):

```bash
bash tests/qwen38_rope_host_check.sh      /path/to/llama.cpp   # IMRoPE
bash tests/qwen38_router_host_check.sh    /path/to/llama.cpp   # MoEルータ重み
bash tests/qwen38_shexp_gate_host_check.sh /path/to/llama.cpp  # 共有エキスパートgate
bash tests/qwen38_exact_math_host_check.sh /path/to/llama.cpp  # 超越関数
bash tests/qwen38_cuda_syntax_check.sh                          # nvcc無しのCUDA構文検査 (CIで実行)
```

## 文字列で話しかける (トークナイザ + ストリーミング CLI)

GGUF の語彙から byte-level BPE トークナイザを読み、llama.cpp の `llama-cli` と同じ
操作感で使えます。トークン ID 列は llama.cpp と完全一致します (実物の Qwen3.8 語彙で
292/292、実機 86GB モデルでも 78/78)。

```bash
# 1回だけ
./build-sm89/warodecode -m /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
  -p "こんにちは。自己紹介してください。" -n 48 --chat-template

# 対話モード (Ctrl-D か /exit、Ctrl-C で生成を止めて終了)
./build-sm89/warodecode -m model.gguf -i -n 48 --chat-template

# トークナイザだけ (GPU 不要)
./build-sm89/warodecode -m model.gguf --tokenize-only -p "こんにちは、世界！"
```

- `-m`/`-p`/`-n`/`-c` は `--model`/`--prompt`/`--generate`/`--ctx` の別名
- 生成は 1 トークンずつ flush して標準出力に流れる (UTF-8 の途中で切れるトークンは次まで保持)
- `--temp` は greedy 固定のため 0 以外は警告して無視、`--chat-template` は Qwen 定型のみ適用
- 既存の `--start-token` + `--generate` の JSON 出力はそのまま (計測・parity 用)
- 詳細と検証結果: `docs/CLI_AND_TOKENIZER.md`

## GPU対応方針

| Compute Capability | 代表GPU | 目標hot path |
|---|---|---|
| SM50/52 | Maxwell | Q4 SIMT |
| SM60 | P100 | Q4 + half2 / HBM帯域特化 |
| SM61 | P40 / P4 | Q4×Q8 DP4A |
| SM70 | V100 / Titan V | Q4→FP16 `mma.sync.m8n8k4` |
| SM75 | T4 / RTX 20 | INT8 MMA + Q4 |
| SM80/86 | A100 / RTX 30 | INT4/INT8 Tensor Core small-GEMM |
| SM89 | RTX 40 / L40 | INT4 Tensor Core |
| SM90 | H100/H200 | Hopper専用small-GEMM / attention |
| SM100/103 | B100/B200/B300 | Blackwell datacenter専用 |
| SM120 | RTX 50 | Blackwell GeForce専用 |

**注意:** 表は最終目標のdispatchです。現在のforward本体はQ4_0 SIMTの共通カーネルが基準で、世代専用Tensor Core/DP4Aカーネルは順次差し替える段階です。TUIでは `target=` として表示し、実装済みと誤認しないようにしています。

## CUDA Toolkit

Maxwell/Pascal/Voltaをビルドする場合はCUDA 12.9以前が必要です。CUDA 13ではこれらのoffline compilationとライブラリ対応が削除されています。

RTX 50系(SM120)はCUDA 12.8以降を使います。

## ビルド

自動検出:

```bash
./bench/build.sh auto
```

指定例:

```bash
./bench/build.sh v100
./bench/build.sh p40
./bench/build.sh rtx30
./bench/build.sh h100
./bench/build.sh rtx50
```

出力は `build-smXX/warodecode` です。

## 現在のforward

- Linux / WSL
- NVIDIA GPU、SM50以上をビルド対象
- dense Llama / Qwen2 / Qwen3
- Qwen3.8-Flash-Next (`qwen4exp`) はTUI/planner・split GGUF認識まで対応、専用forward実装中
- GGUF Q4_0
- batch=1
- FP16 KV cache
- greedy generation
- Q/K/V連結
- Gate/Up連結
- online-softmax decode attention
- GPU内argmax
- CUDA Graph
