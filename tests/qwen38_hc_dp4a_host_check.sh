#!/usr/bin/env bash
# Host bit-comparison of the exact Q8_0 x Q8_0 GEMV orderings (Task P).
#
# nvcc is not required: the test mirrors the parity-verified reference kernel,
# the new __dp4a kernel and the removed Task J ordering from src/qwen38_hc.cuh
# as plain C++, and compares the row results bit for bit.
#
# Usage: bash tests/qwen38_hc_dp4a_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_HC_DP4A_WORK:-/tmp/qwen38-hc-dp4a-host}"
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored fmul/fma a single rounded operation,
# matching the nvcc default the kernels are built with.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
    "$ROOT/tests/qwen38_hc_dp4a_host_test.cpp" -o "$WORK/qwen38_hc_dp4a_host_test"

"$WORK/qwen38_hc_dp4a_host_test"
