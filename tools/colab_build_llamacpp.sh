#!/usr/bin/env bash
# Build the pinned llama.cpp (CUDA, sm<ARCH>) on the Colab VM together with the
# batch=1 decode harness used by tools/llamacpp_ab.py, so the A/B can be
# reproduced from a single file.
#
# The pinned Colab CLI has no "exec -c", so upload this file and run it with
#   colab exec -s <session> --timeout 300 -f tools/colab_build_llamacpp.sh
# (WARODECODE_COLAB_ARCH selects the CUDA arch; default 80 = A100.)
set -euo pipefail

WORK=${WARODECODE_WORK:-/content/wd_taskl}
ARCH=${WARODECODE_COLAB_ARCH:-80}
COMMIT=6fcaa16f4b360649933a54d1f91ad40ed35c0e11
HARNESS=${WARODECODE_HARNESS:-$(dirname "$0")/../qwen38_llamacpp_gpu_bench.cpp}
mkdir -p "$WORK"
cd "$WORK"

if [ ! -d llama.cpp/.git ]; then
  rm -rf llama.cpp
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp llama.cpp
fi
cd llama.cpp
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch --quiet origin "$COMMIT"
git -c advice.detachedHead=false checkout --quiet "$COMMIT"
HEAD_NOW=$(git rev-parse HEAD)
if [ "$HEAD_NOW" != "$COMMIT" ]; then
  echo "LLAMACPP_BUILD_FAILED pinned commit mismatch: $HEAD_NOW"; exit 1
fi
echo "LLAMACPP_HEAD=$HEAD_NOW $(git log -1 --format=%s)"

BUILD="$WORK/llama.cpp/build-cuda-sm$ARCH"
cmake -S "$WORK/llama.cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  > "$WORK/cmake.log" 2>&1
cmake --build "$BUILD" -j 8 --target llama-cli llama-bench > "$WORK/build.log" 2>&1
echo "LLAMA_CLI_OK $(ls -la "$BUILD/bin/llama-cli" | awk '{print $5}') bytes"

# The harness is an extra example of the pinned checkout: it needs examples/ to
# pick up the directory, so the add_subdirectory line is appended when missing.
if [ -f "$HARNESS" ]; then
  mkdir -p examples/taskl-decode
  sed "s/__COMMIT__/$COMMIT/g" "$HARNESS" > examples/taskl-decode/taskl-llama-decode.cpp
  printf 'add_executable(taskl-llama-decode taskl-llama-decode.cpp)\ntarget_link_libraries(taskl-llama-decode PRIVATE llama)\n' \
    > examples/taskl-decode/CMakeLists.txt
  if ! grep -q 'file(GLOB' examples/CMakeLists.txt && ! grep -q 'taskl-decode' examples/CMakeLists.txt; then
    echo 'add_subdirectory(taskl-decode)' >> examples/CMakeLists.txt
  fi
  cmake -S "$WORK/llama.cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    > "$WORK/harness_cmake.log" 2>&1
  cmake --build "$BUILD" -j 8 --target taskl-llama-decode > "$WORK/harness_build.log" 2>&1
  md5sum "$BUILD/bin/taskl-llama-decode"
  echo "LLAMACPP_HARNESS_OK $(ls -la "$BUILD/bin/taskl-llama-decode" | awk '{print $5}') bytes"
else
  echo "LLAMACPP_HARNESS_SKIPPED (no $HARNESS; upload qwen38_llamacpp_gpu_bench.cpp next to this script)"
fi
echo "LLAMACPP_BUILD_OK arch=sm$ARCH"
