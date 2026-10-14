#!/usr/bin/env bash
# Host test for the Task K expert read-ahead ring.
#
# The ring lives in src/expert_cache.cuh, which pulls in the CUDA runtime, so the
# test compiles against the stubs that tests/qwen38_cuda_syntax_check.sh installs
# and reuses its work directory. Those stubs are deliberately inert, but the
# read-ahead test has to move real bytes (a pinned allocation and the H2D copy),
# so this script patches the *generated* stub in the work directory: the
# checked-in stub stays inert for every other test.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# printenv keeps set -u happy when the variable was never exported.
WORK=$(printenv Q38_CUDA_SYNTAX_WORK || true)
[ -n "$WORK" ] || WORK=/tmp/qwen38-cuda-syntax
cd "$ROOT"

# The cache reads these itself, so a value exported in the caller's shell would
# silently change what the test measures.
for name in WARODECODE_EXPERT_PREFETCH WARODECODE_EXPERT_PREFETCH_DEPTH \
            WARODECODE_EXPERT_PREFETCH_THREADS WARODECODE_EXPERT_PREFETCH_SLOTS \
            WARODECODE_EXPERT_IO_PREAD WARODECODE_EXPERT_IO_RING \
            WARODECODE_EXPERT_IO_THREADS WARODECODE_EXPERT_IO_FADVISE \
            WARODECODE_EXPERT_STATS; do
  unset "$name"
done

# Always regenerate: $WORK/src is the launch-syntax rewritten copy of src/, and a
# stale copy would silently test the previous revision of the ring.
bash tests/qwen38_cuda_syntax_check.sh >/dev/null

python3 - "$WORK/stub/cuda_runtime.h" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()


def sub(old, new):
    global text
    if text.count(old) != 1:
        raise SystemExit('stub patch anchor not found exactly once: %r' % old[:60])
    text = text.replace(old, new)


sub('#include <cstdint>', '#include <cstdint>\n#include <cstdlib>\n#include <cstring>')
sub('inline cudaError_t cudaMallocHost(void ** p, std::size_t n) { (void) p; (void) n; return cudaSuccess; }',
    'inline cudaError_t cudaMallocHost(void ** p, std::size_t n) '
    '{ *p = std::malloc(n); return *p != nullptr ? cudaSuccess : (cudaError_t) 1; }')
sub('inline cudaError_t cudaFreeHost(void * p) { (void) p; return cudaSuccess; }',
    'inline cudaError_t cudaFreeHost(void * p) { std::free(p); return cudaSuccess; }')
sub('inline cudaError_t cudaMalloc(void ** p, std::size_t n) { (void) p; (void) n; return cudaSuccess; }',
    'inline cudaError_t cudaMalloc(void ** p, std::size_t n) '
    '{ *p = std::malloc(n); return *p != nullptr ? cudaSuccess : (cudaError_t) 1; }')
sub('inline cudaError_t cudaFree(void * p) { (void) p; return cudaSuccess; }',
    'inline cudaError_t cudaFree(void * p) { std::free(p); return cudaSuccess; }')
sub('''inline cudaError_t cudaMemcpyAsync(void * d, const void * s, std::size_t n, cudaMemcpyKind k, cudaStream_t st) {
    (void) d; (void) s; (void) n; (void) k; (void) st; return cudaSuccess;
}''',
    '''inline cudaError_t cudaMemcpyAsync(void * d, const void * s, std::size_t n, cudaMemcpyKind k, cudaStream_t st) {
    (void) k; (void) st;
    if (d != nullptr && s != nullptr && n != 0) std::memcpy(d, s, n);
    return cudaSuccess;
}''')
path.write_text(text)
PY

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -mavx2 -mfma -mf16c \
  -D__CUDACC__ -include "$WORK/stub/cuda_prelude.h" -I "$WORK/stub" -I "$WORK/src" \
  tests/expert_prefetch_host_test.cpp -o /tmp/warodecode-prefetch-host-test -pthread
/tmp/warodecode-prefetch-host-test
