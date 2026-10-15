#!/usr/bin/env bash
# Host bit-comparison of the multi-block argmax key packing
# (src/qwen38_forward.cuh, q38_argmax_key). nvcc is not required.
#
# Usage: bash tests/qwen38_argmax_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${Q38_ARGMAX_WORK:-/tmp/qwen38-argmax-host}
mkdir -p "$WORK"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    "$ROOT/tests/qwen38_argmax_host_test.cpp" -o "$WORK/qwen38_argmax_host_test"

"$WORK/qwen38_argmax_host_test"
