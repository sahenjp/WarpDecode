#!/usr/bin/env bash
# Host (no nvcc) check of the Task G batched-MoE orderings and index mapping.
#
# Usage: bash tests/moe_fused_grid_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_MOE_FUSED_WORK:-/tmp/qwen38-moe-fused-host}"
mkdir -p "$WORK"

# -ffp-contract=off keeps every mirrored multiply/add a single rounded operation,
# matching the __fmul_rn / __fadd_rn pairs the kernels use.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
    "$ROOT/tests/moe_fused_grid_host_test.cpp" -o "$WORK/moe_fused_grid_host_test"

"$WORK/moe_fused_grid_host_test"
