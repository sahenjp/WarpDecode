#!/usr/bin/env bash
# Host bit-equality check of the Task N CPU MoE kernels.
#
#   qwen38_cpu_moe_host_test.cpp    AVX2 kernels vs their scalar twins vs exact
#                                   int64 lane sums, plus the codebooks, the F16
#                                   conversions and the layer runner.
#   qwen38_cpu_moe_oracle_test.cpp  the same kernels vs the *actual* AVX2 code
#                                   of the pinned llama.cpp (copied into the
#                                   test file), so the CPU MoE path cannot drift
#                                   from the CPU oracle by a rounding step.
#
# nvcc is not required: the kernels are plain C++ over AVX2 intrinsics.
#
# Usage: bash tests/qwen38_cpu_moe_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_CPU_MOE_WORK:-/tmp/qwen38-cpu-moe-host}"
mkdir -p "$WORK"

# -mavx2 -mfma: the kernels are AVX2/FMA by construction and both the L4 and the
# A100 hosts are AVX2 machines. -ffp-contract=off keeps every non-intrinsic
# multiply/add a single rounded operation, matching the __fmul_rn/__fadd_rn
# pairs the CUDA twins use.
FLAGS="-std=c++17 -O2 -Wall -Wextra -Wpedantic -mavx2 -mfma -ffp-contract=off"

g++ $FLAGS -I "$ROOT/src" -I "$ROOT/tests" \
    "$ROOT/tests/qwen38_cpu_moe_host_test.cpp" -o "$WORK/qwen38_cpu_moe_host_test"
g++ $FLAGS -I "$ROOT/src" -I "$ROOT/tests" \
    "$ROOT/tests/qwen38_cpu_moe_oracle_test.cpp" -o "$WORK/qwen38_cpu_moe_oracle_test"

"$WORK/qwen38_cpu_moe_host_test"
"$WORK/qwen38_cpu_moe_oracle_test"
