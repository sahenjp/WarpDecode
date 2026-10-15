#!/usr/bin/env bash
# Host bit-comparison of the Task V pipelined Q8_0 x Q8_0 exact GEMV
# (src/qwen38_hc.cuh). nvcc is not required: the test mirrors the scalar order,
# the __dp4a integer dot and the pipelined loop structure as plain C++.
#
# Usage: bash tests/qwen38_hc_q8_pipe_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${Q38_HC_Q8_PIPE_WORK:-/tmp/qwen38-hc-q8-pipe-host}
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored fmul/fma a single rounded operation,
# matching the nvcc default the kernels are built with.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
    "$ROOT/tests/qwen38_hc_q8_pipe_host_test.cpp" -o "$WORK/qwen38_hc_q8_pipe_host_test"

"$WORK/qwen38_hc_q8_pipe_host_test"
