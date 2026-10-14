#!/usr/bin/env bash
# Host bit-comparison of src/qwen38_exact_math.cuh against the real llama.cpp
# CPU code at the pinned commit (6fcaa16f4b360649933a54d1f91ad40ed35c0e11).
#
# The checkout must be built with the same instruction set as the CPU oracle:
#
#   LLAMA=~/llama.cpp
#   git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
#   git -C "$LLAMA" checkout 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
#   cmake -S "$LLAMA" -B "$LLAMA/build-host" -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF \
#     -DGGML_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
#     -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build "$LLAMA/build-host" --target ggml-cpu ggml-base -j"$(nproc)"
#
# Usage: bash tests/qwen38_exact_math_host_check.sh [llama.cpp checkout]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA=${1:-${LLAMA_CPP:-}}

if [[ -z "$LLAMA" || ! -f "$LLAMA/ggml/src/ggml-cpu/vec.h" ]]; then
  echo "usage: bash tests/qwen38_exact_math_host_check.sh <llama.cpp checkout>" >&2
  echo "       (or export LLAMA_CPP=...)" >&2
  exit 2
fi
if [[ ! -f "$LLAMA/build-host/bin/libggml-cpu.so" ]]; then
  echo "missing $LLAMA/build-host/bin/libggml-cpu.so - see the header of this script" >&2
  exit 2
fi

WORK=${Q38_EXACT_MATH_WORK:-/tmp/qwen38-exact-math-host}
mkdir -p "$WORK"

# The drift guard compiles the qwen38_hc.cuh copy of the AVX2 polynomial from the
# real source text, so the two copies cannot diverge unnoticed.
awk '/^__device__ inline float q38_hc_ggml_avx2_exp_f32/ { f = 1 } f { print } f && /^}/ { exit }' "$ROOT/src/qwen38_hc.cuh" > "$WORK/q38_hc_avx2_extract.h"
if ! grep -q 'return __fmul_rn(__fmaf_rn(s2, j, s2), s1);' "$WORK/q38_hc_avx2_extract.h"; then
  echo "failed to extract q38_hc_ggml_avx2_exp_f32 from src/qwen38_hc.cuh" >&2
  exit 2
fi

# Static confirmations: the ops the warp kernels reproduce are really the ones
# the CPU oracle dispatches.
grep -q 'ggml_vec_silu_f32' "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q 'ggml_vec_soft_max_f32(ne00, dp, wp, max)' "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q 'return 1.f / (1.f + expf(-x));' "$LLAMA/ggml/src/ggml-cpu/unary-ops.cpp"
grep -q 'return (x > 20.0f) ? x : logf(1.0f + expf(x));' "$LLAMA/ggml/src/ggml-cpu/unary-ops.cpp"

# -ffp-contract=off: every rounding-mode intrinsic in the header must stay a
# single IEEE operation on the host too.
g++ -std=c++17 -O2 -ffp-contract=off -mavx -mavx2 -mfma -mf16c -msse4.2 -mbmi2 -D__device__= -D__forceinline__=inline -D__constant__= -I "$ROOT/src" -I "$WORK" -I "$LLAMA/ggml/include" -I "$LLAMA/ggml/src" -I "$LLAMA/ggml/src/ggml-cpu" "$ROOT/tests/qwen38_exact_math_host_test.cpp" -L "$LLAMA/build-host/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$LLAMA/build-host/bin" -o "$WORK/exact_math_host_test"

"$WORK/exact_math_host_test"
