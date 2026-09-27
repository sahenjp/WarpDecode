# V100協力テスト

V100実機を持っている人には、最初はコード改造を頼まずログだけ返してもらう。

推奨環境:

- Linux
- CUDA 12.9以前
- V100 16GBまたは32GB
- 最新llama.cppを同じマシンでビルド
- Q4_0のdense Llama/Qwen GGUF

手順:

```bash
git clone <warodecodeのリポジトリ>
cd warodecode
./bench/build.sh v100

./bench/ab.sh \
  ./build-sm70/warodecode \
  /path/to/llama.cpp/build/bin/llama-bench \
  /models/model-Q4_0.gguf \
  999
```

追加で:

```bash
./bench/profile.sh ./build-sm70/warodecode /models/model-Q4_0.gguf
nvidia-smi -q > bench-out/nvidia-smi-q.txt
nvcc --version > bench-out/nvcc.txt
```

返してほしいもの:

```text
bench-out/
GPU型番(V100 PCIe/SXM2、16/32GB)
CUDA Toolkit版
NVIDIA driver版
モデル名とGGUF量子形式
```

最初の目的は勝敗ではなく、Q4 GEMV / attention / logitsのどこが律速かをNsightで確定すること。そこからSM70専用MMA kernelのtileを決める。
