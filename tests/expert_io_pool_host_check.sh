#!/usr/bin/env bash
# Host test for the Task H pread pool. The pool lives in src/expert_cache.cuh,
# which pulls in the CUDA runtime, so the test compiles against the stubs that
# tests/qwen38_cuda_syntax_check.sh installs and reuses its work directory.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${Q38_CUDA_SYNTAX_WORK:-/tmp/qwen38-cuda-syntax}
cd "$ROOT"

# Always regenerate: $WORK/src is the launch-syntax rewritten copy of src/, and a
# stale copy would silently test the previous revision of the pool.
bash tests/qwen38_cuda_syntax_check.sh >/dev/null

# $WORK/src holds the launch-syntax rewritten headers of the syntax check, and
# the CUDA-only spellings come from the stub prelude, so the same include set
# compiles the pool test for the host.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -mavx2 -mfma -mf16c \
  -D__CUDACC__ -include "$WORK/stub/cuda_prelude.h" -I "$WORK/stub" -I "$WORK/src" \
  tests/expert_io_pool_host_test.cpp -o /tmp/warodecode-io-pool-host-test -pthread
/tmp/warodecode-io-pool-host-test
