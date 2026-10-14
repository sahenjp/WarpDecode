#!/usr/bin/env bash
# Host bit-comparison of the GDN F32 GEMV orderings (Task P).
#
# nvcc is not required: the test mirrors the one-thread AVX2-order reference,
# the new warp-parallel kernel and the legacy lane-strided order from
# src/qwen38_gdn.cuh as plain C++ and compares the row results bit for bit.
#
# Usage: bash tests/qwen38_gdn_f32_gemv_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_GDN_GEMV_WORK:-/tmp/qwen38-gdn-gemv-host}"
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored fmul/fma a single rounded operation,
# matching the nvcc default the kernels are built with.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
    "$ROOT/tests/qwen38_gdn_f32_gemv_host_test.cpp" -o "$WORK/qwen38_gdn_f32_gemv_host_test"

"$WORK/qwen38_gdn_f32_gemv_host_test"
