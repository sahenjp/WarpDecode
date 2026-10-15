#!/bin/bash
# Build the host/device read-path bandwidth probe (tests/expert_io_bandwidth.cu).
#
# The probe is standalone nvcc, not a CMake target: it is only used to attribute
# the engine's transfer path and never linked into the engine.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
arch="${WARODECODE_IO_BENCH_ARCH:-80}"
out="${1:-$root/tests/expert_io_bench}"
# nvcc wants the underscore form (sm_80); a bare "sm80" is rejected by the CUDA
# 12.8 toolkit the Colab image ships.
nvcc -O2 -std=c++17 -arch="sm_${arch}" "$root/tests/expert_io_bandwidth.cu" -o "$out"
echo "built $out (sm_${arch})"
