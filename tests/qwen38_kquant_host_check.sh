#!/usr/bin/env bash
# Host check of the k-quant dequantisers used to requantise dense GGUF tensors to
# Q8_0 (src/ggml_dequant.hpp). nvcc is not required.
#
# Usage: bash tests/qwen38_kquant_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${Q38_KQUANT_WORK:-/tmp/qwen38-kquant-host}
mkdir -p "$WORK"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc \
    "$ROOT/tests/gemv_kquant_dequant_host_test.cpp" -o "$WORK/qwen38_kquant_host_test"

"$WORK/qwen38_kquant_host_test"
