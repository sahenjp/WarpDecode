# warodecode

NVIDIA GPU上の **batch=1 LLM decode速度** に対象を絞り、llama.cppの汎用性コストを捨てて速度を取りにいくCUDAランタイムです。

現時点では開発中です。「llama.cppより高速」という性能主張は、同一モデル・同一GPUで実測して5%以上勝った構成だけに限定します。

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
- Q4 kernel launch形状の実機autotune
- CPU inference fallbackなし

Q4_0はロード時に量子値を変えずplanar配置へ再パックします。

## 実行

```bash
./build-sm86/warodecode \
  --model /models/model-q4_0.gguf \
  --ctx 4096 \
  --generate 128 \
  --device 0
```

起動時に代表的なQ4行列を64/128/256/512 threadsで短く実測し、そのGPUで速いlaunch形状を採用します。比較のため無効化もできます。

```bash
./build-sm86/warodecode --model model.gguf --no-autotune
```

## llama.cppとのA/B

```bash
./bench/ab.sh \
  ./build-sm86/warodecode \
  /path/to/llama-bench \
  /models/model-q4_0.gguf \
  999
```

5%未満の差は勝ち扱いしません。

## 複数GPU

現在実装済みなのは、**GPUごとに独立したモデルプロセスを同時起動**するモードです。TUIでモデルとGPUをSpace選択して `M` を押します。

1モデルを複数GPUへ分割するtensor parallel / layer splitは未実装です。P2P/NVLinkの有無で最適方式が変わるため、遅い単純層分割を「対応済み」として追加せず、実測できる形で別実装します。

## 「llama.cppより速い」の条件

- 同じGGUF
- 同じGPU / GPU枚数
- 同じKV型
- batch=1
- 同じ生成長
- warmup後5回以上
- GPUクロック・電力状態を記録
- token/logit品質を別ゲートで確認
- 5%未満は同等扱い

現環境にはNVIDIA GPU / nvccがないため、CUDA本体の実機性能は未確認です。TUI、GGUF parser、設定・検出部分はホスト側でテストしています。


## Qwen3.8-Flash-Next

`qwen4exp`を看板最適化対象に追加しました。split GGUFはTUI上で1モデルにまとめ、metadata-only first shardでも認識します。専用方針は `docs/QWEN38_FLASH_NEXT.md`。

狙うhot pathは **persistent QSA block cache + compact KV gather / direct PLE row reader / hot expert slot cache / grouped low-bit MoE / adaptive MTP2-4** です。現時点ではplannerとPLE direct readerが入り、qwen4exp forward本体は未配線です。
