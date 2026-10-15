#!/usr/bin/env bash
# Host bit-comparison of the Task W ILP routed-expert GEMV row routines
# (src/lowbit.cuh). nvcc is not required: the test mirrors both loop structures
# at the real byte offsets and compares the ordered operand sequences and the
# resulting float chains bit for bit.
#
# Usage: bash tests/qwen38_expert_ilp_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${Q38_EXPERT_ILP_WORK:-/tmp/qwen38-expert-ilp-host}
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored fmul/fma a single rounded operation,
# matching the nvcc default the kernels are built with.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
    "$ROOT/tests/qwen38_expert_ilp_host_test.cpp" -o "$WORK/qwen38_expert_ilp_host_test"

"$WORK/qwen38_expert_ilp_host_test"
