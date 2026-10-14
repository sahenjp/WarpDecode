#!/usr/bin/env bash
# Host bit-comparison of the exact IMRoPE in src/qwen38_rope.cuh against the real
# llama.cpp CPU rope at the pinned commit
# (6fcaa16f4b360649933a54d1f91ad40ed35c0e11).
#
# The checkout must be built with the same instruction set as the CPU oracle:
#
#   LLAMA=~/llama.cpp
#   git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
#   git -C "$LLAMA" checkout 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
#   cmake -S "$LLAMA" -B "$LLAMA/build-host" -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
#     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
#     -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=ON
#   cmake --build "$LLAMA/build-host" --target ggml -j"$(nproc)"
#
# Usage: bash tests/qwen38_rope_host_check.sh [llama.cpp checkout]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA=${1:-${LLAMA_CPP:-}}

if [[ -z "$LLAMA" || ! -f "$LLAMA/ggml/include/ggml.h" ]]; then
  echo "usage: bash tests/qwen38_rope_host_check.sh <llama.cpp checkout>" >&2
  echo "       (or export LLAMA_CPP=...)" >&2
  exit 2
fi
if [[ ! -f "$LLAMA/build-host/bin/libggml.so" ]]; then
  echo "missing $LLAMA/build-host/bin/libggml.so - see the header of this script" >&2
  exit 2
fi

WORK=${Q38_ROPE_WORK:-/tmp/qwen38-rope-host}
mkdir -p "$WORK"

# Static confirmations: the sequence the header transcribes really is the one the
# CPU oracle executes.
grep -q "theta_t \*= theta_scale;" "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q "const float theta_scale = powf(freq_base, -2.0f/n_dims);" "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q "\*cos_theta = cosf(theta) \* mscale;" "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q "case GGML_ROPE_TYPE_IMROPE:" "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q "const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;" "$LLAMA/ggml/src/ggml-cpu/ops.cpp"

# -ffp-contract=off keeps every expression in the transcribed glibc routines a
# single IEEE operation, matching nvcc default contraction behaviour.
g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off \
    -I "$ROOT/src" -I "$LLAMA/ggml/include" \
    "$ROOT/tests/qwen38_rope_host_test.cpp" \
    -L "$LLAMA/build-host/bin" -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LLAMA/build-host/bin" -o "$WORK/rope_host_test"

"$WORK/rope_host_test"
