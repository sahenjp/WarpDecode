#!/usr/bin/env bash
# Host bit-comparison of the __dp4a GEMV variants in src/lowbit.cuh (Task D).
#
# nvcc is not required: the test mirrors the reference (per-element, shuffle
# reduced) and the DP4A (byte packed, one instruction per four values) orderings
# as plain C++ over the real codebooks in src/iq2s_grid.cuh and
# src/iq3s_grid.cuh, and compares the per-chunk and per-row results bit for bit.
# The sign expansion and the IQ4_NL nibble packing are checked exhaustively.
#
# Usage: bash tests/lowbit_dp4a_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_DP4A_WORK:-/tmp/qwen38-dp4a-host}"
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored fmul/fma a single rounded operation,
# matching the nvcc default the kernels are built with.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -D__device__= -D__constant__= -D__forceinline__=inline -I "$ROOT/src" "$ROOT/tests/lowbit_dp4a_host_test.cpp" -o "$WORK/lowbit_dp4a_host_test"

"$WORK/lowbit_dp4a_host_test"
