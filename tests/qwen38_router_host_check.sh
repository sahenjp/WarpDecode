#!/usr/bin/env bash
# Host bit-comparison of the MoE router weight chain in src/qwen38_moe.cuh
# against the real llama.cpp CPU code at the pinned oracle commit
# (6fcaa16f4b360649933a54d1f91ad40ed35c0e11).
#
# The reference is build_moe_ffn's router part replayed as a ggml graph and
# evaluated by that checkout's own CPU backend (soft_max -> argsort_top_k ->
# get_rows -> sum_rows -> clamp -> div), so the comparison is op for op against
# the arithmetic the CPU oracle executed. Two case sets are compared:
#
#   * a golden corpus extracted from the oracle traces in issue1_artifacts/
#     (ffn_moe_logits / ffn_moe_topk / ffn_moe_weights_norm recorded by the
#     runner itself), and
#   * synthetic rows aimed at the branches that are hard to hit in the model:
#     exact ties, near-ties inside one ulp, all-equal rows, -inf / denormal /
#     overflowing logits, and maxima at either end of the row.
#
# The checkout must be built the same way as the CPU oracle:
#
#   LLAMA=/tmp/qwen38-pinned
#   git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
#   git -C "$LLAMA" checkout 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
#   cmake -S "$LLAMA" -B "$LLAMA/build-host" -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF \
#     -DGGML_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
#     -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build "$LLAMA/build-host" --target ggml-cpu ggml-base -j"$(nproc)"
#
# Usage: bash tests/qwen38_router_host_check.sh [llama.cpp checkout] [n_synthetic]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA=${1:-${LLAMA_CPP:-}}
PER_KIND=${2:-200}

if [[ -z "$LLAMA" || ! -f "$LLAMA/ggml/src/ggml-cpu/vec.h" ]]; then
  echo "usage: bash tests/qwen38_router_host_check.sh <llama.cpp checkout> [n_synthetic]" >&2
  echo "       (or export LLAMA_CPP=...)" >&2
  exit 2
fi
if [[ ! -f "$LLAMA/build-host/bin/libggml-cpu.so" ]]; then
  echo "missing $LLAMA/build-host/bin/libggml-cpu.so - see the header of this script" >&2
  exit 2
fi

WORK=${Q38_ROUTER_WORK:-/tmp/qwen38-router-host}
mkdir -p "$WORK"

# ---------------------------------------------------------------------------
# Static confirmations: the ops the router kernel reproduces are really the ones
# the CPU oracle dispatches for this model.
# ---------------------------------------------------------------------------
grep -q 'ggml_soft_max(ctx0, logits)' "$LLAMA/src/llama-graph.cpp"
grep -q 'ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)' "$LLAMA/src/llama-graph.cpp"
grep -q 'weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY)' "$LLAMA/src/llama-graph.cpp"
grep -q 'weights = ggml_div(ctx0, weights, weights_sum)' "$LLAMA/src/llama-graph.cpp"
grep -q 'sum = 1.0/sum;' "$LLAMA/ggml/src/ggml-cpu/ops.cpp"
grep -q 'sum += (ggml_float)_mm_cvtss_f32(val2);' "$LLAMA/ggml/src/ggml-cpu/vec.cpp"
grep -q 'return a / b;' "$LLAMA/ggml/src/ggml-cpu/binary-ops.cpp"
grep -q 'sum += (ggml_float)x\[i\];' "$LLAMA/ggml/src/ggml-cpu/vec.h"
grep -q 'max = MAX(max, x\[i\]);' "$LLAMA/ggml/src/ggml-cpu/vec.h"

# ---------------------------------------------------------------------------
# Drift guard: the candidate is the real source text of the router block, so the
# host check cannot pass against a stale copy of the kernel.
# ---------------------------------------------------------------------------
START=$(grep -n 'MoE router weight chain, matching build_moe_ffn' "$ROOT/src/qwen38_moe.cuh" | head -1 | cut -d: -f1)
END=$(grep -n '^#endif  // __CUDACC__' "$ROOT/src/qwen38_moe.cuh" | head -1 | cut -d: -f1)
if [[ -z "$START" || -z "$END" || "$END" -le "$START" ]]; then
  echo "failed to locate the router block in src/qwen38_moe.cuh" >&2
  exit 2
fi
# START is the first line of the block comment; END is the #endif that closes the
# block, so the section is [START - 2, END].
sed -n "$((START - 2)),${END}p" "$ROOT/src/qwen38_moe.cuh" > "$WORK/q38_router_section.h"
if ! grep -q 'q38_moe_router_exact' "$WORK/q38_router_section.h"; then
  echo "extracted section does not define q38_moe_router_exact" >&2
  exit 2
fi
if ! grep -q 'sum += static_cast<double>(q38_moe_softmax_lane_sum(v));' "$WORK/q38_router_section.h"; then
  echo "extracted section does not contain the 8-lane softmax sum tree" >&2
  exit 2
fi

cat > "$WORK/cuda_runtime.h" <<'CUDA_STUB_EOF'
// Host stand-in for <cuda_runtime.h>. qwen38_exact_math.cuh includes the real
// header inside its __CUDACC__ block; the host bit check compiles that block on
// the host to reach the same accessors, so it supplies this stub instead. Only
// the names the header mentions are needed, and the publish has to stay a no-op
// so the two device flags keep their defaults - the exact path, which is what
// the default environment selects.
#pragma once
#include <cstddef>
using cudaError_t = int;
static constexpr int cudaSuccess = 0;
template <typename T>
static inline cudaError_t cudaMemcpyToSymbol(const T &, const void *, std::size_t) { return cudaSuccess; }
CUDA_STUB_EOF

cat > "$WORK/qwen38_moe_router_extract.h" <<'HEADER_EOF'
// Generated by tests/qwen38_router_host_check.sh - do not edit.
#include <cmath>
#include <cstdint>
#include <cstring>

// qwen38_exact_math.cuh keeps its CUDA-side aliases (the two published flags and
// their accessor) behind __CUDACC__. The host check needs the same accessors, so
// define that macro for this header; the <cuda_runtime.h> it then wants is the
// stub in this work directory, which keeps the publish a no-op so the two device
// flags keep their defaults - the exact path, which is what the default
// environment selects.
#define __CUDACC__ 1
#include "qwen38_exact_math.cuh"
#undef __CUDACC__

#define __restrict__
#define __restrict

namespace wd {

// Copied verbatim from the top of src/qwen38_moe.cuh (the router block below
// calls it); the extraction cannot carry the whole header.
inline bool q38_moe_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

HEADER_EOF
cat "$WORK/q38_router_section.h" >> "$WORK/qwen38_moe_router_extract.h"
cat >> "$WORK/qwen38_moe_router_extract.h" <<'FOOTER_EOF'

// Test-only entry point: the kernel gives the selection the same buffer the
// softmax wrote because it never needs the row again; the test does.
inline void q38_test_router_probs_row(
        const float * logits,
        int n_experts,
        int k,
        float w_scale,
        float * probs,
        int * ids,
        float * weights) {
    q38_moe_softmax_row(logits, n_experts, probs);
    q38_moe_router_select(probs, n_experts, k, w_scale, ids, weights);
}

// The candidate under test is the extracted source; this alias keeps the test's
// call sites unqualified.
inline void q38_test_router_row(
        const float * logits, int n_experts, int k, float w_scale,
        float * probs, int * ids, float * weights) {
    q38_moe_router_exact(logits, n_experts, k, w_scale, probs, ids, weights);
}

}  // namespace wd
FOOTER_EOF

# ---------------------------------------------------------------------------
# Golden corpus: every ffn_moe_logits row the CPU oracle ever recorded, with the
# top-k and the normalised weights it produced next to it.
# ---------------------------------------------------------------------------
python3 "$ROOT/tests/qwen38_router_extract_corpus.py" "$ROOT" "$WORK/router_corpus.bin"

# ---------------------------------------------------------------------------
# -ffp-contract=off keeps every rounding-mode intrinsic a single IEEE operation
# on the host too; -D__device__= empties the CUDA qualifier from the extracted
# text, exactly like tests/qwen38_exact_math_host_check.sh does.
# ---------------------------------------------------------------------------
g++ -std=c++17 -O2 -ffp-contract=off -mavx -mavx2 -mfma -mf16c -msse4.2 -mbmi2 \
  -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-function \
  -D__device__= -D__forceinline__=inline -D__constant__= \
  -I "$WORK" -I "$ROOT/src" -I "$LLAMA/ggml/include" -I "$LLAMA/ggml/src" -I "$LLAMA/ggml/src/ggml-cpu" \
  "$ROOT/tests/qwen38_router_host_test.cpp" \
  -L "$LLAMA/build-host/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$LLAMA/build-host/bin" \
  -o "$WORK/router_host_test"

"$WORK/router_host_test" "$WORK/router_corpus.bin" "$PER_KIND"
