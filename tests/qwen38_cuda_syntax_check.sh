#!/usr/bin/env bash
# Static type/syntax check of the qwen38 CUDA headers without a CUDA toolchain.
#
# nvcc is not required: the headers are mirrored into a work directory, the
# "kernel<<<cfg>>>(args)" launch syntax is rewritten to "kernel(args)" with a
# guard that refuses any match that is not a real launch, and the CUDA-only
# spellings are supplied by small stubs. The result is compiled with g++
# -fsyntax-only, which exercises the template bodies (including the F32 dot
# dispatchers of the MoE, router and HC inject paths) on the host.
#
# What this catches: unbalanced braces, unknown names, wrong argument types,
# broken template instantiation, missing includes.
# What it does NOT check: the launch configurations themselves, device-only
# semantics, PTX/codegen, or anything that needs nvcc.
#
# Usage: bash tests/qwen38_cuda_syntax_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${Q38_CUDA_SYNTAX_WORK:-/tmp/qwen38-cuda-syntax}
STUB="$WORK/stub"
cd "$ROOT"

rm -rf "$WORK"
mkdir -p "$WORK/src" "$WORK/tests" "$STUB"

# ---------------------------------------------------------------------------
# Host stubs for the CUDA-only names the headers use.
# ---------------------------------------------------------------------------
cat > "$STUB/cuda_fp16.h" <<'FP16_EOF'
#pragma once

#include <cstdint>

using __half = _Float16;

struct __half2 {
    __half x;
    __half y;
};

inline __half __float2half(float v) { return static_cast<__half>(v); }
inline __half __float2half_rn(float v) { return static_cast<__half>(v); }
inline float __half2float(__half v) { return static_cast<float>(v); }
inline bool __heq(__half a, __half b) { return static_cast<float>(a) == static_cast<float>(b); }
FP16_EOF

cat > "$STUB/cuda_runtime.h" <<'RUNTIME_EOF'
#pragma once

#include "cuda_fp16.h"

#include <cstddef>
#include <cstdint>

struct q38_stub_dim3 {
    unsigned int x = 1, y = 1, z = 1;
    q38_stub_dim3() = default;
    // nvcc's dim3 takes one, two or three extents; the expert-batched MoE
    // launchers use the two-dimensional form.
    q38_stub_dim3(unsigned int x_, unsigned int y_, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
extern const q38_stub_dim3 threadIdx;
extern const q38_stub_dim3 blockIdx;
extern const q38_stub_dim3 blockDim;
extern const q38_stub_dim3 gridDim;
using dim3 = q38_stub_dim3;

typedef int cudaError_t;
static constexpr cudaError_t cudaSuccess = 0;

typedef struct q38_stub_stream * cudaStream_t;
typedef struct q38_stub_event * cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

inline const char * cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaMalloc(void ** p, std::size_t n) { (void) p; (void) n; return cudaSuccess; }
inline cudaError_t cudaFree(void * p) { (void) p; return cudaSuccess; }
inline cudaError_t cudaMallocHost(void ** p, std::size_t n) { (void) p; (void) n; return cudaSuccess; }
inline cudaError_t cudaFreeHost(void * p) { (void) p; return cudaSuccess; }
inline cudaError_t cudaMemcpy(void * d, const void * s, std::size_t n, cudaMemcpyKind k) {
    (void) d; (void) s; (void) n; (void) k; return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void * d, const void * s, std::size_t n, cudaMemcpyKind k, cudaStream_t st) {
    (void) d; (void) s; (void) n; (void) k; (void) st; return cudaSuccess;
}
template <typename T>
inline cudaError_t cudaMemcpyToSymbol(const T & symbol, const void * s, std::size_t n,
                                      std::size_t off = 0, cudaMemcpyKind k = cudaMemcpyHostToDevice) {
    (void) symbol; (void) s; (void) n; (void) off; (void) k; return cudaSuccess;
}
inline cudaError_t cudaMemset(void * p, int v, std::size_t n) { (void) p; (void) v; (void) n; return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void * p, int v, std::size_t n, cudaStream_t st) {
    (void) p; (void) v; (void) n; (void) st; return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t st) { (void) st; return cudaSuccess; }
// The DP4A launchers pick their kernel from the current device's sm, so the
// host check needs the two runtime queries that supplies.
enum cudaDeviceAttr {
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
};
inline cudaError_t cudaGetDevice(int * device) { if (device) *device = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device) {
    (void) attr; (void) device; if (value) *value = 8; return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t * e) { (void) e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t st = nullptr) { (void) e; (void) st; return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { (void) e; return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t a, cudaEvent_t b) {
    (void) ms; (void) a; (void) b; return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { (void) e; return cudaSuccess; }
// Device properties are read by the benchmark harnesses (GPU name + sm).
struct q38_stub_device_prop {
    char name[64] = "stub-device";
    int major = 8;
    int minor = 9;
};
using cudaDeviceProp = q38_stub_device_prop;
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp * p, int device) {
    (void) device; if (p) *p = cudaDeviceProp(); return cudaSuccess;
}
// Streams/events used by the expert-cache transfer path (Task C).
static constexpr unsigned int cudaStreamNonBlocking = 1;
static constexpr unsigned int cudaEventDisableTiming = 2;
static constexpr cudaError_t cudaErrorNotReady = 34;
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t * s, unsigned int f) {
    (void) f; *s = nullptr; return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t st) { (void) st; return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t st, cudaEvent_t e, unsigned int f) {
    (void) st; (void) e; (void) f; return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t * e, unsigned int f) {
    (void) f; *e = nullptr; return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) { (void) e; return cudaSuccess; }

inline void __syncthreads() {}
inline float __fmaf_rn(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fsqrt_rn(float a) { return __builtin_sqrtf(a); }
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dsub_rn(double a, double b) { return a - b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __ddiv_rn(double a, double b) { return a / b; }
inline float __double2float_rn(double a) { return static_cast<float>(a); }
inline int __float2int_rn(float a) { return __builtin_lrintf(a); }
inline unsigned int __float_as_uint(float a) { unsigned int u; __builtin_memcpy(&u, &a, 4); return u; }
inline float __uint_as_float(unsigned int u) { float a; __builtin_memcpy(&a, &u, 4); return a; }
inline long long __double_as_longlong(double a) { long long u; __builtin_memcpy(&u, &a, 8); return u; }
inline double __longlong_as_double(long long u) { double a; __builtin_memcpy(&a, &u, 8); return a; }

template <typename T>
inline T __shfl_sync(unsigned int mask, T var, int src, int width = 32) {
    (void) mask; (void) src; (void) width; return var;
}
template <typename T>
inline T __shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = 32) {
    (void) mask; (void) delta; (void) width; return var;
}
template <typename T>
inline T __shfl_xor_sync(unsigned int mask, T var, int lane_mask, int width = 32) {
    (void) mask; (void) lane_mask; (void) width; return var;
}
RUNTIME_EOF

cat > "$STUB/cuda_prelude.h" <<'PRELUDE_EOF'
#pragma once

#include <stdlib.h>
#include <string.h>
#include <algorithm>

using std::min;
using std::max;

#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
#define __managed__
#define __forceinline__ inline
#define __restrict__ __restrict

#include "cuda_runtime.h"
PRELUDE_EOF

# ---------------------------------------------------------------------------
# Mirror the headers with the launch syntax rewritten. The guard refuses any
# "<<" whose closing ">>>(" is too far away or crosses a comment, because such a
# match would silently delete code (the check must never report success on a
# mutilated copy).
# ---------------------------------------------------------------------------
python3 - "$ROOT" "$WORK" <<'PY_EOF'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
work = pathlib.Path(sys.argv[2])


def rewrite(text, name):
    out = []
    i = 0
    launches = 0
    while True:
        j = text.find("<<<", i)
        if j < 0:
            out.append(text[i:])
            break
        k = text.find(">>>(", j)
        segment = text[j:k]
        if k < 0 or len(segment) > 400 or "//" in segment or segment.count("\n") > 6:
            raise SystemExit("unsafe launch match in %s near offset %d: %r" % (name, j, segment[:80]))
        out.append(text[i:j])
        out.append("(")
        i = k + 4
        launches += 1
    return "".join(out), launches


total = 0
for pattern in ("src/*.cuh", "src/*.hpp", "src/*.inc"):
    for path in root.glob(pattern):
        text, n = rewrite(path.read_text(), str(path))
        (work / "src" / path.name).write_text(text)
        total += n
for name in ("tests/qwen38_moe_cuda_test.cu", "tests/moe_fused_grid_bench.cu",
             "tests/qwen38_tokenizer_host_test.cpp"):
    path = root / name
    text, n = rewrite(path.read_text(), str(path))
    (work / "tests" / (path.stem + ".cpp")).write_text(text)
    total += n
print("launch sites rewritten: %d" % total)
PY_EOF

# ---------------------------------------------------------------------------
# -D__CUDACC__ turns on the CUDA-side aliases of qwen38_exact_math.cuh (the
# published exact-exp flag and its accessor) that the device code calls.
# ---------------------------------------------------------------------------
cd "$WORK"
g++ -std=c++17 -fsyntax-only -ffp-contract=off -mavx2 -mfma -mf16c -D__CUDACC__ \
  -Wno-unknown-pragmas -Wno-attributes \
  -include "$STUB/cuda_prelude.h" -I "$STUB" -I "$WORK/src" \
  tests/qwen38_moe_cuda_test.cpp

# The Task G harness is compiled the same way so its own launch sites and the
# expert-batched API it calls are checked without a CUDA toolchain.
g++ -std=c++17 -fsyntax-only -ffp-contract=off -mavx2 -mfma -mf16c -D__CUDACC__ \
  -Wno-unknown-pragmas -Wno-attributes \
  -include "$STUB/cuda_prelude.h" -I "$STUB" -I "$WORK/src" \
  tests/moe_fused_grid_bench.cpp

# The Task M tokenizer and its CLI driver are host-only (no CUDA), so the same
# -fsyntax-only pass type checks them without a model file. The unicode tables
# are mirrored above as src/*.inc, and qwen38_tokenizer.cuh includes them.
g++ -std=c++17 -fsyntax-only -Wall -Wextra -I "$WORK/src" \
  tests/qwen38_tokenizer_host_test.cpp

# warodecode.cu itself cannot be compiled without nvcc (it includes CUB), so its
# structural invariants are checked textually: braces and namespace closings must
# balance. This catches the "namespace wd opened twice, closed once" class of
# edit that a header-only syntax pass would miss.
python3 - "$ROOT/src/warodecode.cu" "$ROOT/src/qwen38_tokenizer.cuh" <<'BALANCE_EOF'
import pathlib
import sys

for name in sys.argv[1:]:
    text = pathlib.Path(name).read_text()
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in ("\"", "'"):
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    clean = "".join(out)
    delta = clean.count("{") - clean.count("}")
    if delta != 0:
        raise SystemExit("unbalanced braces in %s (%d open, %d close)"
                         % (name, clean.count("{"), clean.count("}")))
    # A second "namespace wd {" inside the already open one silently creates
    # wd::wd and every helper becomes unreachable (nvcc: "namespace wd has no
    # member ..."). Exactly one such opening line is allowed per file.
    opens = [line for line in text.splitlines() if line.strip().startswith("namespace wd {")]
    if len(opens) != 1:
        raise SystemExit("expected exactly one 'namespace wd {' in %s, found %d"
                         % (name, len(opens)))
    print("balanced: %s (%d braces, 1 namespace)" % (pathlib.Path(name).name, clean.count("{")))
BALANCE_EOF

echo "PASS (g++ -fsyntax-only, launch syntax rewritten)"
