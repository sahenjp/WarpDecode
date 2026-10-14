#!/usr/bin/env bash
# Build the pinned llama.cpp (CUDA sm80) on the Colab VM for the A/B comparison.
# The Colab CLI has no "exec -c", so this is uploaded as a file and run with
# "colab exec -s <session> -f tools/colab_build_llamacpp.sh".
set -euo pipefail
mkdir -p /content/wd_taskl
cd /content/wd_taskl
if [ ! -d llama.cpp ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp llama.cpp
fi
cd llama.cpp
git fetch --depth 1 origin 6fcaa16f4b360649933a54d1f91ad40ed35c0e11 2>/dev/null || true
git checkout -q 6fcaa16f4b360649933a54d1f91ad40ed35c0e11 2>/dev/null || true
if [ ! -d build-cuda-sm80 ]; then
  cmake -S . -B build-cuda-sm80 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_TESTS=OFF \
    > /content/wd_taskl/cmake.log 2>&1
fi
cmake --build build-cuda-sm80 -j 8 --target llama-cli llama-bench \
  > /content/wd_taskl/build.log 2>&1
echo "LLAMACPP_BUILD_OK $(ls -la build-cuda-sm80/bin/llama-cli | awk '{print $5}') bytes"
