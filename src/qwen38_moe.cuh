#pragma once

#include "expert_cache.cuh"
#include "dense_vector.cuh"
#include "gguf.hpp"
#include "lowbit.cuh"
// ggml-exact Q8_0 activation quantisation + integer-dot GEMV. qwen38_hc.cuh
// includes neither gdn/qsa/moe, so pulling it in here stays acyclic.
#include "qwen38_exact_math.cuh"
#include "qwen38_hc.cuh"
// ggml-ordered F32 arithmetic helpers (q38_mul_rn / q38_add_rn /
// q38_moe_branch_f32). Header-only, no project includes.
#include "qwen38_precision.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace wd {

// ggml-compatible activation quantisation for the MoE router and the shared
// expert. llama.cpp's CPU reference quantises the activation to Q8_0 (32-value
// blocks, integer dot, per-block scale) before every matmul against a
// quantised weight; the legacy F16-activation GEMV rounds the activation
// differently and drifts ~1% per projection. Both switches default to ON and
// only apply while the weight is Q8_0, so non-quantised routers keep the
// existing dense path. WARODECODE_ROUTER_Q8_0=0 / WARODECODE_SHEXP_Q8_0=0
// restore the legacy path.
inline bool q38_moe_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_moe_router_f32_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_F32", true);
    return enabled;
}

inline bool q38_moe_router_q8_0_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_Q8_0", true);
    return enabled;
}

inline bool q38_moe_shexp_q8_0_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_Q8_0", true);
    return enabled;
}

// WARODECODE_SHEXP_GATE_F32 (default ON).
//
// blk.N.ffn_gate_inp_shexp.weight is F32 [n_embd] in the pinned GGUF, so
// llama.cpp builds the shared expert's sigmoid gate with
// build_lora_mm(ffn_gate_inp_shexp, cur) on F32 tensors: type_traits_cpu[F32]
// has vec_dot_type == F32 and llamafile_sgemm() declines the single decode
// token (sgemm.cpp:3819, "if (n < 2) return false"), so the gate is one
// ggml_vec_dot_f32 per token. upload_half_vector() rounded that weight to F16
// (~5e-4 systematic) before the sigmoid, which the expert activation
// quantisation then amplifies. With this switch the weight stays F32 and the
// dot runs in the oracle's AVX2+FMA visit order
// (q38_moe_gate_dot_f32_avx2_kernel). WARODECODE_SHEXP_GATE_F32=0 restores the
// legacy F16-weight gate, and a model whose tensor is not F32 falls back to it
// automatically.
inline bool q38_moe_shexp_gate_f32_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_GATE_F32", true);
    return enabled;
}

// Routed experts (512 experts, top-10) dominate the MoE work, so their
// activation also has to match ggml: llama.cpp's CPU reference quantises the
// gate/up input to the Q8_K vec_dot_type shared by IQ2_S and IQ3_S once and the
// SiLU output to the IQ4_NL vec_dot_type (Q8_0) for the down projection. Both
// projections stay in F32 and the quantised gate/up activation is shared by all
// ten experts of a layer. Defaults to ON and only applies to the observed expert
// tensor types, so anything else keeps the legacy F16-activation path.
inline bool q38_moe_expert_q8_act_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_EXPERT_Q8_ACT", true);
    return enabled;
}

// F32 MoE intermediates. ggml never rounds the FFN intermediates: the expert
// gate/up/down projections, the SiLU*up product, the per-expert weighted sum
// and the shared-expert addition all stay F32, and only the activation of a
// *quantised* weight is rounded (to Q8_0 / Q8_K) before the integer dot. The
// legacy MoE stored each expert output and the shared-expert output in __half,
// so every projection added one F16 rounding on top of the quantisation.
// With WARODECODE_MOE_F32_MID=1 (default) the routed experts accumulate their
// unrounded down projection into d_accum, the shared expert keeps its unrounded
// down projection in down_f32(), and q38_add_shared_f32_kernel() /
// q38_finish_moe_f32() build the MoE branch without an F16 round. The F16
// buffers and kernels stay in place for WARODECODE_MOE_F32_MID=0.
inline bool q38_moe_f32_mid_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_MOE_F32_MID", true);
    return enabled;
}

struct Qwen38MoeLayerTensors {
    const TensorInfo * router = nullptr;
    const TensorInfo * gate_up = nullptr;
    const TensorInfo * gate = nullptr;
    const TensorInfo * up = nullptr;
    const TensorInfo * down = nullptr;

    bool fused_gate_up() const { return gate_up != nullptr; }
};

struct Qwen38SharedExpertWeights {
    // Legacy F16 mirror of the shared-expert gate weight
    // (blk.N.ffn_gate_inp_shexp.weight). Only the non-F32 models and the
    // WARODECODE_SHEXP_GATE_F32=0 A/B path consume it.
    DeviceHalfVector gate_input;
    // Unrounded F32 gate weight. The pinned GGUF stores the tensor as F32
    // [n_embd], and ggml's F32 trait has vec_dot_type == F32, so the CPU builds
    // the shared gate from an F32 weight and an F32 activation. Keeping only the
    // half mirror rounds the weight by ~5e-4 before the sigmoid, which the
    // downstream activation quantisation amplifies. Empty for a quantised
    // tensor, in which case the legacy F16 path stays in charge.
    DeviceFloatArray gate_input_f32;
    LowBitMatrix gate;
    LowBitMatrix up;
    LowBitMatrix down;
};

inline Qwen38SharedExpertWeights load_qwen38_shared_expert(
        const GgufFile & g,
        int il,
        int hidden,
        int shared_ffn) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const TensorInfo & gate_input = g.tensor(p + "ffn_gate_inp_shexp.weight");
    const TensorInfo & gate = g.tensor(p + "ffn_gate_shexp.weight");
    const TensorInfo & up = g.tensor(p + "ffn_up_shexp.weight");
    const TensorInfo & down = g.tensor(p + "ffn_down_shexp.weight");

    if (gate_input.dims.size() != 1 || static_cast<int>(gate_input.dims[0]) != hidden) {
        throw std::runtime_error("shared expert gate-input shape mismatch");
    }
    auto check_matrix = [&](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("shared expert ") + what + " shape mismatch: " + t.name);
        }
        if (!lowbit_gemv_supported(t.type)) {
            throw std::runtime_error(std::string("shared expert ") + what + " low-bit type unsupported: " + t.name);
        }
    };
    check_matrix(gate, shared_ffn, hidden, "gate");
    check_matrix(up, shared_ffn, hidden, "up");
    check_matrix(down, hidden, shared_ffn, "down");

    Qwen38SharedExpertWeights out;
    out.gate_input = upload_half_vector(gate_input);
    if (gate_input.type == GgmlType::F32) {
        out.gate_input_f32 = upload_float_array(gate_input);
    }
    out.gate = upload_lowbit_matrix(gate);
    out.up = upload_lowbit_matrix(up);
    out.down = upload_lowbit_matrix(down);
    return out;
}

inline Qwen38MoeLayerTensors qwen38_moe_tensors(const GgufFile & g, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    Qwen38MoeLayerTensors out;
    out.router = &g.tensor(p + "ffn_gate_inp.weight");
    out.down = &g.tensor(p + "ffn_down_exps.weight");
    out.gate_up = g.optional_tensor(p + "ffn_gate_up_exps.weight");
    if (!out.gate_up) {
        out.gate = &g.tensor(p + "ffn_gate_exps.weight");
        out.up = &g.tensor(p + "ffn_up_exps.weight");
    }
    return out;
}

// ---------------------------------------------------------------------------
// MoE router weight chain, matching build_moe_ffn (llama-graph.cpp at the
// pinned oracle commit 6fcaa16f4b360649933a54d1f91ad40ed35c0e11).
//
// The oracle path for this model - gating_op = SOFTMAX, norm_w = true,
// w_scale = qwen4exp.expert_weights_scale, exp_probs_b = null - is:
//
//   logits  = build_lora_mm(ffn_gate_inp, cur)                 [512]
//   probs   = ggml_soft_max(logits)                            [512]  softmax over ALL experts
//   sel     = ggml_argsort_top_k(probs, n_expert_used)         [10]   select by PROB, descending
//   weights = ggml_get_rows(probs, sel)                        [10]
//   wsum    = ggml_sum_rows(weights)                           [ 1]
//   wsum    = ggml_clamp(wsum, 6.103515625e-5, +inf)           [ 1]
//   weights = ggml_div(weights, wsum)                          [10]
//   weights = ggml_scale(weights, w_scale)     skipped when w_scale == 1.0f
//
// Three properties of that chain are load-bearing and easy to get wrong:
//
//   1. softmax runs over all n_expert cells, so every probability carries the
//      full 512-term denominator and the selected cells keep it. The legacy
//      warp kernel instead exponentiated only the top-k logits against their own
//      maximum and folded the expert scale into one reciprocal multiply - a
//      different sequence of roundings with a different number of divisions.
//   2. selection is by probability, not by logit. Softmax is order preserving on
//      the logits but not injective after rounding, so a near-tie can round to
//      two equal probabilities and then be ordered by ggml's std::sort instead
//      of by the raw logit. The selection below therefore compares the
//      probabilities, and where two of them are bit equal it takes the lower
//      expert index: ggml's descending comparator is a strict >, so std::sort
//      leaves equal elements unordered and its choice there is not reproducible
//      at all. That is harmless because equal probabilities carry equal weights,
//      so only the identity of the expert can differ, never the k values fed to
//      the experts.
//   3. ggml normalises inside the softmax op with a reciprocal multiply
//      (dp[i] * (1/sum), ggml-cpu/ops.cpp:5682) and renormalises the top-k with
//      a true division (ggml-cpu/binary-ops.cpp: op_div). The warp kernel must
//      not fuse or reorder those two steps.
//
// ggml_vec_soft_max_f32 (ggml-cpu/vec.cpp:531) never calls libm: its AVX2+FMA
// branch - the one the oracle dispatch picks for this model - evaluates
// dp[i] = ggml_v_expf(x[i] - max) with the polynomial at ggml-cpu/vec.h:1215 and
// accumulates the row sum with the 8-lane horizontal adds
// (_mm256_extractf128_ps / _mm_movehl_ps / _mm_movehdup_ps) into a double. The
// kernel below reproduces that per-8-element tree. q38_exact_avx2_exp_f32
// (qwen38_exact_math.cuh) is the lane-wise copy of the polynomial; n_expert =
// 512 is a multiple of 8, so the scalar tail of ggml_vec_soft_max_f32 - the only
// part the copy does not model - never runs for this model. A row length that is
// not a multiple of 8 falls back to the legacy router below instead, because
// there the oracle's tail uses libm expf, which has no device counterpart.
//
// WARODECODE_EXACT_ROUTER (default ON) selects the chain above. Setting it to 0
// restores the legacy two-rounding warp formula for A/B runs.
// ---------------------------------------------------------------------------
// The kernel below reads the switch through a device symbol (the same pattern
// qwen38_gdn.cuh uses) because q38_topk_softmax_kernel keeps its signature.
__device__ int q38_moe_exact_router_flag = 1;

inline bool q38_moe_exact_router_enabled() {
    static const bool enabled = [] {
        const int value = q38_moe_env_flag("WARODECODE_EXACT_ROUTER", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_moe_exact_router_flag, &value, sizeof(value));
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

__device__ __forceinline__ bool q38_moe_exact_router_on() {
    return q38_moe_exact_router_flag != 0;
}

inline const bool q38_moe_exact_router_published = q38_moe_exact_router_enabled();

constexpr int Q38_MOE_ROUTER_KMAX = 16;
// Scratch for one full softmax row. n_expert = 512 for this model.
constexpr int Q38_MOE_ROUTER_MAX_EXPERTS = 1024;

// One step of ggml_vec_soft_max_f32's AVX2 row sum. For the 8 lanes v[0..7] the
// _mm_movehl_ps / _mm_movehdup_ps sequence computes
// ((v0+v4) + (v2+v6)) + ((v1+v5) + (v3+v7)).
__device__ __forceinline__ float q38_moe_softmax_lane_sum(const float * v) {
    const float lo = __fadd_rn(v[0], v[4]);
    const float lh = __fadd_rn(v[2], v[6]);
    const float hi = __fadd_rn(v[1], v[5]);
    const float hh = __fadd_rn(v[3], v[7]);
    return __fadd_rn(__fadd_rn(lo, lh), __fadd_rn(hi, hh));
}

// ggml_soft_max over the whole row (ggml_compute_forward_soft_max_f32,
// ggml-cpu/ops.cpp:5584). probs receives n_experts probabilities.
//
//   max:  ggml_vec_max_f32 (ggml-cpu/vec.h:1541) is an order-independent scan,
//         so the plain ascending loop reproduces it exactly.
//   exp:  dp[i] = ggml_v_expf(x[i] - max), the subtraction its own IEEE step.
//   sum:  accumulated per 8 lanes into a double (ggml_float), exactly like the
//         AVX2 branch of ggml_vec_soft_max_f32.
//   norm: sum = 1.0/sum, then one multiply per element (ggml_vec_scale_f32). That
//         vector scale starts from -INFINITY so a zero row stays zero.
__device__ __forceinline__ void q38_moe_softmax_row(
        const float * __restrict__ logits,
        int n_experts,
        float * __restrict__ probs) {
    float max_v = -INFINITY;
    for (int e = 0; e < n_experts; ++e) {
        max_v = fmaxf(max_v, logits[e]);
    }

    double sum = 0.0;
    int i = 0;
    for (; i + 8 <= n_experts; i += 8) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = q38_exact_avx2_exp_f32(__fsub_rn(logits[i + j], max_v));
            probs[i + j] = v[j];
        }
        sum += static_cast<double>(q38_moe_softmax_lane_sum(v));
    }
    for (; i < n_experts; ++i) {
        const float val = q38_exact_avx2_exp_f32(__fsub_rn(logits[i], max_v));
        probs[i] = val;
        sum += static_cast<double>(val);
    }

    // ggml_vec_scale_f32 starts from -INFINITY, so a row whose exponents all
    // underflowed stays -INFINITY rather than turning into NaN.
    if (sum == 0.0) {
        for (int e = 0; e < n_experts; ++e) {
            probs[e] = -INFINITY;
        }
        return;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int e = 0; e < n_experts; ++e) {
        probs[e] = __fmul_rn(probs[e], inv);
    }
}

// ggml_argsort_top_k(probs, k) + norm_w + w_scale. probs is read only; the
// candidate ids and the normalised weights are the outputs.
//
//   selection: a descending argsort, i.e. the k largest probabilities with a tie
//              falling to the lower expert index. Taking the rungs in order
//              reproduces that order. The kernel keeps the ids and the weights in
//              ascending expert order for equal probabilities; the oracle's
//              std::sort is free to return any order there (see the block comment
//              above), and the weights are equal either way.
//   norm_w:    ggml_sum_rows, whose ggml_vec_sum_f32 accumulates into ggml_float
//              (double) and rounds once at the end (ggml-cpu/vec.h:1495), then
//              the ggml_clamp against the F16 minimum (a no-op for any row with a
//              positive sum) and the ggml_div renormalisation.
//   w_scale:   skipped by ggml when it is exactly 1.0f.
__device__ __forceinline__ void q38_moe_router_select(
        const float * __restrict__ probs,
        int n_experts,
        int k,
        float w_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    float top[Q38_MOE_ROUTER_KMAX];
    // The k rungs of the descending argsort. Each rung scans the whole row and
    // skips the experts already taken, which is the same sequence ggml's argsort
    // produces: the rungs are non-increasing, so every taken expert is >= the
    // next rung and skipping it cannot change the choice.
    for (int j = 0; j < k; ++j) {
        // Start below every probability so a row that is all -INFINITY (every
        // exponent underflowed) still resolves, and read the value back through
        // the chosen id: ggml_get_rows selects the cell, it does not invent one,
        // so that row yields -INFINITY weights, not zero.
        float best = -INFINITY;
        int best_id = 0;
        for (int e = 0; e < n_experts; ++e) {
            const float p = probs[e];
            if (p > best) {
                bool taken = false;
                for (int m = 0; m < j; ++m) {
                    if (ids[m] == e) {
                        taken = true;
                        break;
                    }
                }
                if (!taken) {
                    best = p;
                    best_id = e;
                }
            }
        }
        ids[j] = best_id;
        top[j] = probs[best_id];
    }

    double wsum = 0.0;
    for (int j = 0; j < k; ++j) {
        wsum += static_cast<double>(top[j]);
    }
    const float wsum_f32 = fmaxf(static_cast<float>(wsum), 6.103515625e-5f);
    for (int j = 0; j < k; ++j) {
        weights[j] = __fdiv_rn(top[j], wsum_f32);
    }

    if (w_scale != 0.0f && w_scale != 1.0f) {
        for (int j = 0; j < k; ++j) {
            weights[j] = __fmul_rn(weights[j], w_scale);
        }
    }
}

// The two halves together, the shape the kernel calls.
__device__ __forceinline__ void q38_moe_router_exact(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float w_scale,
        float * __restrict__ probs,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    q38_moe_softmax_row(logits, n_experts, probs);
    q38_moe_router_select(probs, n_experts, k, w_scale, ids, weights);
}

#if defined(__CUDACC__)
// Legacy two-rounding router, kept for WARODECODE_EXACT_ROUTER=0.
__device__ __forceinline__ void q38_moe_router_legacy(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    float best[Q38_MOE_ROUTER_KMAX];
    int best_id[Q38_MOE_ROUTER_KMAX];
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        best[j] = -3.402823466e+38F;
        best_id[j] = -1;
    }

    for (int e = 0; e < n_experts; ++e) {
        const float v = logits[e];
        if (v <= best[k - 1]) continue;

        int pos = k - 1;
        while (pos > 0 && v > best[pos - 1]) {
            best[pos] = best[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best[pos] = v;
        best_id[pos] = e;
    }

    const float max_v = best[0];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        if (j >= k) break;
        const float w = q38_exact_softmax_exp_f32(best[j], max_v, q38_exact_exp_on());
        weights[j] = w;
        ids[j] = best_id[j];
        sum += w;
    }

    const float inv = sum > 0.0f ? expert_scale / sum : 0.0f;
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        if (j >= k) break;
        weights[j] *= inv;
    }
}

__global__ void q38_topk_softmax_kernel(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    __shared__ float probs[Q38_MOE_ROUTER_MAX_EXPERTS];
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (k <= 0 || k > Q38_MOE_ROUTER_KMAX) return;

    if (q38_moe_exact_router_on() &&
        n_experts > 0 && n_experts <= Q38_MOE_ROUTER_MAX_EXPERTS && n_experts % 8 == 0) {
        q38_moe_router_exact(logits, n_experts, k, expert_scale, probs, ids, weights);
        return;
    }
    q38_moe_router_legacy(logits, n_experts, k, expert_scale, ids, weights);
}
#endif  // __CUDACC__
__global__ void q38_silu_mul_kernel(
        const __half * __restrict__ gate,
        const __half * __restrict__ up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    // ggml_silu -> ggml_vec_silu_f32 (ggml-cpu/vec.cpp:380) -> ggml_v_silu.
    const float s = q38_exact_silu_f32(g, q38_exact_exp_on());
    out[i] = __float2half(s * u);
}

__global__ void q38_silu_mul_fused_kernel(
        const __half * __restrict__ gate_up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate_up[i]);
    const float u = __half2float(gate_up[n + i]);
    const float s = q38_exact_silu_f32(g, q38_exact_exp_on());
    out[i] = __float2half(s * u);
}

__global__ void q38_weighted_accum_kernel(
        const __half * __restrict__ x,
        float weight,
        float * __restrict__ accum,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = fmaf(weight, __half2float(x[i]), accum[i]);
}

// F32 twin of q38_weighted_accum_kernel for the WARODECODE_MOE_F32_MID path.
// The expert output is already F32, so the accumulation is exactly the two
// rounding steps ggml performs per selected expert (ggml_mul by the expert
// weight, then ggml_add into the running sum) instead of a fused multiply-add
// on an F16-rounded value.
__global__ void q38_weighted_accum_f32_kernel(
        const float * __restrict__ x,
        float weight,
        float * __restrict__ accum,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = q38_add_rn(accum[i], q38_mul_rn(x[i], weight));
}

__global__ void q38_float_to_half_kernel(
        const float * __restrict__ x,
        __half * __restrict__ y,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// ggml SiLU(gate) * up product in F32. The shared expert keeps the gated
// activation unrounded so the following Q8_0 quantisation sees the same F32
// values the CPU reference sees.
__global__ void q38_silu_mul_f32_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __fmul_rn(gate[i], up[i]);
}

// Q8_0 activation quantisation in the exact ggml order (amax/127 with
// round-to-nearest-even division, q = round(value * 127/amax)). Byte-identical
// to ggml's quantize_row_q8_0 for elements that are a multiple of 32.
inline void q38_moe_quantize_activation_q8_0(
        const float * input,
        int elements,
        int8_t * quantized,
        __half * scales,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = elements / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        input, elements, quantized, scales);
}

// Q8_0 weight x Q8_0 activation integer dot with an F32 output, in the same
// accumulation order the ggml SIMD kernels use.
inline void q38_moe_gemv_q8_0_exact(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const __half * activation_scales,
        float * output,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = (weights.rows + warps - 1) / warps;
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, output, weights.rows, weights.cols);
}

// F32-activation twin of half_dot_kernel() (dense_vector.cuh) for the shared
// expert's gate_input projection. The F16 weights widen exactly and the F32
// activation is consumed as-is instead of being rounded to F16 first. The thread
// mapping, the fmaf order and the shared-memory reduction are the same, so an F32
// activation that is exactly F16-representable reproduces half_dot_kernel() bit
// for bit.
__global__ void q38_moe_gate_dot_f32_x_kernel(
        const __half * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ out,
        int n) {
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum = fmaf(__half2float(a[i]), b[i], sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = smem[0];
}

inline void q38_moe_gate_dot_f32_x(
        const __half * weights,
        const float * x,
        float * out,
        int n,
        cudaStream_t stream) {
    q38_moe_gate_dot_f32_x_kernel<<<1, 256, 0, stream>>>(weights, x, out, n);
}

// Tree reduction half of ggml_vec_dot_f32 on AVX2+FMA, the instruction set the
// pinned CPU oracle compiles (GGML_NATIVE=OFF still enables AVX2/FMA, see
// qwen38_gdn.cuh): GGML_F32_ARR == 4 accumulators of GGML_F32_EPR == 8 lanes,
// folded (0,2)/(1,3) then (0,1), then the 8-lane GGML_F32x8_REDUCE
// (t[k] = x[k] + x[k+4], h0 = t0 + t1, h1 = t2 + t3, res = h0 + h1). Callers
// must fill sum[4][8] in AVX2 visit order: base in {0, 32, 64, 96}, stream j in
// 0..3, lane l in 0..7, i = base + j*8 + l with FMA accumulation. Identical to
// q38_gdn_reduce_avx2(); kept as a local copy so this file stays independent of
// qwen38_gdn.cuh.
__device__ __forceinline__ float q38_moe_reduce_avx2(float sum[4][8]) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        sum[0][l] = sum[0][l] + sum[2][l];
        sum[1][l] = sum[1][l] + sum[3][l];
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) sum[0][l] = sum[0][l] + sum[1][l];
    float t0[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) t0[k] = sum[0][k] + sum[0][k + 4];
    const float h0 = t0[0] + t0[1];
    const float h1 = t0[2] + t0[3];
    return h0 + h1;
}

// F32 weight vector x F32 activation -> one ggml_vec_dot_f32 (ggml-cpu/vec.cpp:11)
// in its own visit order, for the shared expert's gate projection
// (build_lora_mm(ffn_gate_inp_shexp, cur) with the F32 weight and F32 activation
// the pinned GGUF selects). ggml_compute_forward_mul_mat() sees
// src1->type == vec_dot_type == F32, skips the activation quantisation and - for
// a single decode token - hands the whole row to vec_dot(ne00, ...) with
// nrc == 1 (llamafile_sgemm() returns false for n < 2, sgemm.cpp:3819).
//
// One thread walks the row, so the visit order is exactly the oracle's single
// vec_dot call: 32-value blocks of four 8-lane FMA accumulators, the
// GGML_F32x8_REDUCE tree, then the "sumf += x[i]*y[i]" non-fused tail over
// n & ~31. n == 2560 for the pinned model, so the tail never runs there.
// Launch with a single thread: thread 0 owns the whole row.
__global__ void q38_moe_gate_dot_f32_avx2_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ out,
        int n) {
    float sum[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = n & ~31;
    for (int i = 0; i < np; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int c = i + j * 8 + l;
                sum[j][l] = __fmaf_rn(w[c], x[c], sum[j][l]);
            }
        }
    }
    float acc = q38_moe_reduce_avx2(sum);
    for (int c = np; c < n; ++c) acc = __fadd_rn(acc, __fmul_rn(w[c], x[c]));
    *out = acc;
}

inline void q38_moe_gate_dot_f32_avx2(
        const float * weights,
        const float * x,
        float * out,
        int n,
        cudaStream_t stream) {
    q38_moe_gate_dot_f32_avx2_kernel<<<1, 1, 0, stream>>>(weights, x, out, n);
}

__global__ void q38_add_shared_kernel(
        __half * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ gate,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float y = __half2float(routed[i]) + (*gate) * __half2float(shared[i]);
    routed[i] = __float2half(y);
}

// F32 twin of q38_add_shared_kernel: routed[i] += shared[i] * gate[0] with no
// F16 intermediate. The multiply/add order matches q38_moe_branch_f32(), which
// the HC combine kernel applies to the same two operands, so the F32 MoE branch
// stays bit-identical whichever side fuses the addition.
__global__ void q38_add_shared_f32_kernel(
        float * __restrict__ routed,
        const float * __restrict__ shared,
        const float * __restrict__ gate,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    routed[i] = q38_moe_branch_f32(routed[i], shared[i], gate[0]);
}

class Qwen38MoeScratch {
public:
    Qwen38MoeScratch() = default;
    Qwen38MoeScratch(const Qwen38MoeScratch &) = delete;
    Qwen38MoeScratch & operator=(const Qwen38MoeScratch &) = delete;

    ~Qwen38MoeScratch() { reset(); }

    void initialize(int n_experts, int k, int hidden, int expert_ffn) {
        if (d_logits_) throw std::runtime_error("MoE scratch already initialized");
        if (n_experts <= 0 || n_experts > 4096 || k <= 0 || k > 16 || k > n_experts ||
            hidden <= 0 || expert_ffn <= 0) {
            throw std::runtime_error("invalid MoE scratch geometry");
        }

        n_experts_ = n_experts;
        k_ = k;
        hidden_ = hidden;
        expert_ffn_ = expert_ffn;

        alloc(reinterpret_cast<void **>(&d_logits_), static_cast<size_t>(n_experts_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_ids_), static_cast<size_t>(k_) * sizeof(int));
        alloc(reinterpret_cast<void **>(&d_weights_), static_cast<size_t>(k_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_gate_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_up_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_gate_up_), static_cast<size_t>(2 * expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_act_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_down_), static_cast<size_t>(hidden_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_accum_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_shared_gate_), sizeof(float));

        // ggml-exact Q8_0 activation scratch (router + shared expert). The F32
        // buffers keep the projections unrounded like the CPU reference; the
        // int8/scale pair holds the quantised activation.
        const int activ = std::max(hidden_, expert_ffn_);
        alloc(reinterpret_cast<void **>(&d_act_f32_), static_cast<size_t>(activ) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_gate_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_up_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_down_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_q8_0_values_), static_cast<size_t>(activ) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_q8_0_scales_),
              static_cast<size_t>((activ + 31) / 32) * sizeof(__half));

        // ggml-exact routed expert scratch: Q8_K for the IQ2_S/IQ3_S gate/up pair
        // (one quantisation per layer, shared by every selected expert), Q8_0 for the
        // IQ4_NL down projection and F32 for the projections in between. The
        // Q8_K buffers are rounded up to whole 256-value blocks so the geometry
        // check, not the allocation, is what decides whether the path applies.
        const int q8k_blocks = (hidden_ + 255) / 256;
        alloc(reinterpret_cast<void **>(&d_rx_x_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_values_),
              static_cast<size_t>(q8k_blocks) * 256u * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_scales_), static_cast<size_t>(q8k_blocks) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_bsums_),
              static_cast<size_t>(q8k_blocks) * 16u * sizeof(int16_t));
        alloc(reinterpret_cast<void **>(&d_rx_gate_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_up_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_gate_up_f32_), static_cast<size_t>(2 * expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_act_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_down_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8_0_values_), static_cast<size_t>(expert_ffn_) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_rx_q8_0_scales_),
              static_cast<size_t>((expert_ffn_ + 31) / 32) * sizeof(__half));

        host_alloc(reinterpret_cast<void **>(&h_ids_), static_cast<size_t>(k_) * sizeof(int));
        host_alloc(reinterpret_cast<void **>(&h_weights_), static_cast<size_t>(k_) * sizeof(float));
    }

    __half * selected_output_trace() {
        if (!d_selected_output_trace_) {
            alloc(reinterpret_cast<void **>(&d_selected_output_trace_),
                  static_cast<size_t>(k_) * hidden_ * sizeof(__half));
        }
        return d_selected_output_trace_;
    }

    // F32 MoE branch (routed weighted sum + shared expert output * gate). Only
    // the WARODECODE_MOE_F32_MID path asks for it, so it is allocated on first
    // use instead of growing every legacy session.
    float * moe_out_f32() {
        if (!d_moe_out_f32_) {
            alloc(reinterpret_cast<void **>(&d_moe_out_f32_),
                  static_cast<size_t>(hidden_) * sizeof(float));
        }
        return d_moe_out_f32_;
    }

    // F16 mirror of the MoE activation for the legacy F16-activation fallbacks
    // of the F32-input runners (the ggml-exact Q8/Q8_K path never needs it).
    // Lazily allocated like moe_out_f32() so legacy sessions do not grow.
    __half * half_activation() {
        if (!d_half_activation_) {
            alloc(reinterpret_cast<void **>(&d_half_activation_),
                  static_cast<size_t>(hidden_) * sizeof(__half));
        }
        return d_half_activation_;
    }

    int n_experts() const { return n_experts_; }
    int k() const { return k_; }
    int hidden() const { return hidden_; }
    int expert_ffn() const { return expert_ffn_; }

    float * d_logits() const { return d_logits_; }
    int * d_ids() const { return d_ids_; }
    float * d_weights() const { return d_weights_; }
    __half * d_gate() const { return d_gate_; }
    __half * d_up() const { return d_up_; }
    __half * d_gate_up() const { return d_gate_up_; }
    __half * d_act() const { return d_act_; }
    __half * d_down() const { return d_down_; }
    float * d_accum() const { return d_accum_; }
    float * d_shared_gate() const { return d_shared_gate_; }
    float * act_f32() const { return d_act_f32_; }
    float * gate_f32() const { return d_gate_f32_; }
    float * up_f32() const { return d_up_f32_; }
    float * down_f32() const { return d_down_f32_; }
    int8_t * q8_0_values() const { return d_q8_0_values_; }
    __half * q8_0_scales() const { return d_q8_0_scales_; }
    float * rx_x_f32() const { return d_rx_x_f32_; }
    int8_t * rx_q8k_values() const { return d_rx_q8k_values_; }
    float * rx_q8k_scales() const { return d_rx_q8k_scales_; }
    int16_t * rx_q8k_bsums() const { return d_rx_q8k_bsums_; }
    float * rx_gate_f32() const { return d_rx_gate_f32_; }
    float * rx_up_f32() const { return d_rx_up_f32_; }
    float * rx_gate_up_f32() const { return d_rx_gate_up_f32_; }
    float * rx_act_f32() const { return d_rx_act_f32_; }
    float * rx_down_f32() const { return d_rx_down_f32_; }
    int8_t * rx_q8_0_values() const { return d_rx_q8_0_values_; }
    __half * rx_q8_0_scales() const { return d_rx_q8_0_scales_; }
    int * h_ids() const { return h_ids_; }
    float * h_weights() const { return h_weights_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc MoE scratch failed: ") + cudaGetErrorString(e));
    }

    static void host_alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMallocHost(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMallocHost MoE scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (d_logits_) cudaFree(d_logits_);
        if (d_ids_) cudaFree(d_ids_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_gate_) cudaFree(d_gate_);
        if (d_up_) cudaFree(d_up_);
        if (d_gate_up_) cudaFree(d_gate_up_);
        if (d_act_) cudaFree(d_act_);
        if (d_down_) cudaFree(d_down_);
        if (d_accum_) cudaFree(d_accum_);
        if (d_shared_gate_) cudaFree(d_shared_gate_);
        if (d_act_f32_) cudaFree(d_act_f32_);
        if (d_gate_f32_) cudaFree(d_gate_f32_);
        if (d_up_f32_) cudaFree(d_up_f32_);
        if (d_down_f32_) cudaFree(d_down_f32_);
        if (d_q8_0_values_) cudaFree(d_q8_0_values_);
        if (d_q8_0_scales_) cudaFree(d_q8_0_scales_);
        if (d_rx_x_f32_) cudaFree(d_rx_x_f32_);
        if (d_rx_q8k_values_) cudaFree(d_rx_q8k_values_);
        if (d_rx_q8k_scales_) cudaFree(d_rx_q8k_scales_);
        if (d_rx_q8k_bsums_) cudaFree(d_rx_q8k_bsums_);
        if (d_rx_gate_f32_) cudaFree(d_rx_gate_f32_);
        if (d_rx_up_f32_) cudaFree(d_rx_up_f32_);
        if (d_rx_gate_up_f32_) cudaFree(d_rx_gate_up_f32_);
        if (d_rx_act_f32_) cudaFree(d_rx_act_f32_);
        if (d_rx_down_f32_) cudaFree(d_rx_down_f32_);
        if (d_rx_q8_0_values_) cudaFree(d_rx_q8_0_values_);
        if (d_rx_q8_0_scales_) cudaFree(d_rx_q8_0_scales_);
        if (d_selected_output_trace_) cudaFree(d_selected_output_trace_);
        if (d_moe_out_f32_) cudaFree(d_moe_out_f32_);
        if (d_half_activation_) cudaFree(d_half_activation_);
        if (h_ids_) cudaFreeHost(h_ids_);
        if (h_weights_) cudaFreeHost(h_weights_);
        d_logits_ = nullptr;
        d_ids_ = nullptr;
        d_weights_ = nullptr;
        d_gate_ = d_up_ = d_gate_up_ = d_act_ = d_down_ = nullptr;
        d_accum_ = nullptr;
        d_shared_gate_ = nullptr;
        d_act_f32_ = d_gate_f32_ = d_up_f32_ = d_down_f32_ = nullptr;
        d_q8_0_values_ = nullptr;
        d_q8_0_scales_ = nullptr;
        d_selected_output_trace_ = nullptr;
        d_moe_out_f32_ = nullptr;
        d_half_activation_ = nullptr;
        h_ids_ = nullptr;
        h_weights_ = nullptr;
    }

    int n_experts_ = 0;
    int k_ = 0;
    int hidden_ = 0;
    int expert_ffn_ = 0;

    float * d_logits_ = nullptr;
    int * d_ids_ = nullptr;
    float * d_weights_ = nullptr;
    __half * d_gate_ = nullptr;
    __half * d_up_ = nullptr;
    __half * d_gate_up_ = nullptr;
    __half * d_act_ = nullptr;
    __half * d_down_ = nullptr;
    float * d_accum_ = nullptr;
    float * d_shared_gate_ = nullptr;
    float * d_act_f32_ = nullptr;
    float * d_gate_f32_ = nullptr;
    float * d_up_f32_ = nullptr;
    float * d_down_f32_ = nullptr;
    int8_t * d_q8_0_values_ = nullptr;
    __half * d_q8_0_scales_ = nullptr;
    float * d_rx_x_f32_ = nullptr;
    int8_t * d_rx_q8k_values_ = nullptr;
    float * d_rx_q8k_scales_ = nullptr;
    int16_t * d_rx_q8k_bsums_ = nullptr;
    float * d_rx_gate_f32_ = nullptr;
    float * d_rx_up_f32_ = nullptr;
    float * d_rx_gate_up_f32_ = nullptr;
    float * d_rx_act_f32_ = nullptr;
    float * d_rx_down_f32_ = nullptr;
    int8_t * d_rx_q8_0_values_ = nullptr;
    __half * d_rx_q8_0_scales_ = nullptr;
    __half * d_selected_output_trace_ = nullptr;
    float * d_moe_out_f32_ = nullptr;
    __half * d_half_activation_ = nullptr;
    int * h_ids_ = nullptr;
    float * h_weights_ = nullptr;
};

// The router only needs the Q8_0 activation path when the weight itself is
// Q8_0 (ggml does not quantise the activation for F16/F32 weights) and the
// hidden width is a whole number of 32-value Q8_0 blocks.
inline bool q38_moe_router_q8_0_compatible(
        const GpuMatrix & router,
        const Qwen38MoeScratch & scratch) {
    return router.lowbit && router.type == GgmlType::Q8_0 &&
        router.rows == scratch.n_experts() && router.cols == scratch.hidden() &&
        scratch.hidden() > 0 && scratch.hidden() % 32 == 0;
}

// Same rule for the shared expert: every GEMV input width (hidden for
// gate/up, expert_ffn for down) must be a whole number of Q8_0 blocks.
inline bool q38_moe_shexp_q8_0_compatible(
        const Qwen38SharedExpertWeights & w,
        const Qwen38MoeScratch & scratch) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (hidden <= 0 || expert_ffn <= 0 || hidden % 32 != 0 || expert_ffn % 32 != 0) return false;
    return w.gate.type == GgmlType::Q8_0 && w.gate.rows == expert_ffn && w.gate.cols == hidden &&
        w.up.type == GgmlType::Q8_0 && w.up.rows == expert_ffn && w.up.cols == hidden &&
        w.down.type == GgmlType::Q8_0 && w.down.rows == hidden && w.down.cols == expert_ffn;
}

// ggml-exact routed expert activation path. It needs the gate/up activation to
// be a whole number of Q8_K blocks (256 values) and the down activation to be an
// even number of Q8_0 blocks (64 values), because ggml's AVX2 IQ4_NL kernel
// consumes two 32-value blocks per step and falls back to a scalar tail
// otherwise.
inline bool q38_moe_expert_q8_act_geometry(int hidden, int expert_ffn) {
    return hidden > 0 && hidden % 256 == 0 && expert_ffn > 0 && expert_ffn % 64 == 0;
}

// Both IQ2_S and IQ3_S declare Q8_K as their ggml vec_dot_type, so the packed
// gate/up activation feeds either kernel; only the weight decode differs. The
// model is IQ2_S everywhere except blk.2's routed experts, which are IQ3_S.
inline bool q38_moe_gate_up_q8_k_type(GgmlType type) {
    return type == GgmlType::IQ2_S || type == GgmlType::IQ3_S;
}

inline bool q38_moe_separate_experts_q8_act_compatible(
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        const Qwen38MoeScratch & scratch) {
    if (!q38_moe_expert_q8_act_enabled()) return false;
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (!q38_moe_expert_q8_act_geometry(hidden, expert_ffn)) return false;
    if (gate_exps.dims.size() != 3 || up_exps.dims.size() != 3 || down_exps.dims.size() != 3) return false;
    return q38_moe_gate_up_q8_k_type(gate_exps.type) &&
            static_cast<int>(gate_exps.dims[0]) == hidden &&
            static_cast<int>(gate_exps.dims[1]) == expert_ffn &&
        q38_moe_gate_up_q8_k_type(up_exps.type) &&
            static_cast<int>(up_exps.dims[0]) == hidden &&
            static_cast<int>(up_exps.dims[1]) == expert_ffn &&
        down_exps.type == GgmlType::IQ4_NL &&
            static_cast<int>(down_exps.dims[0]) == expert_ffn &&
            static_cast<int>(down_exps.dims[1]) == hidden;
}

inline bool q38_moe_fused_experts_q8_act_compatible(
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        const Qwen38MoeScratch & scratch) {
    if (!q38_moe_expert_q8_act_enabled()) return false;
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (!q38_moe_expert_q8_act_geometry(hidden, expert_ffn)) return false;
    if (gate_up_exps.dims.size() != 3 || down_exps.dims.size() != 3) return false;
    return q38_moe_gate_up_q8_k_type(gate_up_exps.type) &&
            static_cast<int>(gate_up_exps.dims[0]) == hidden &&
            static_cast<int>(gate_up_exps.dims[1]) == 2 * expert_ffn &&
        down_exps.type == GgmlType::IQ4_NL &&
            static_cast<int>(down_exps.dims[0]) == expert_ffn &&
            static_cast<int>(down_exps.dims[1]) == hidden;
}

inline void q38_select_experts(
        const GpuMatrix & router,
        const __half * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }

    constexpr int threads = 256;
    const int hidden = scratch.hidden();
    if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
        // ggml quantises the activation to Q8_0 before the matmul; reproduce
        // that rounding instead of the legacy F16 activation GEMV.
        q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        q38_moe_quantize_activation_q8_0(
            scratch.act_f32(), hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
        q38_moe_gemv_q8_0_exact(
            router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
    } else if (q38_moe_router_f32_enabled() && matrix_has_f32_weights(router)) {
        // This model's ffn_gate_inp is F32, and ggml keeps non-quantised tensors
        // in F32 on both sides (vec_dot_type = F32). The router decides which
        // experts run, so its rounding is amplified downstream.
        q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        launch_matrix_gemv_f32_weights(router, scratch.act_f32(), scratch.d_logits(), stream);
    } else {
        launch_matrix_gemv_float(router, x, scratch.d_logits(), stream);
    }

    q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights());

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert ids failed: ") + cudaGetErrorString(e));

    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert weights failed: ") + cudaGetErrorString(e));

    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("expert selection sync failed: ") + cudaGetErrorString(e));
}

inline void q38_select_experts_f32_input(
        const GpuMatrix & router,
        const float * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }

    if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
        // The F32 router input can be quantised in place, exactly like ggml.
        q38_moe_quantize_activation_q8_0(
            x, scratch.hidden(), scratch.q8_0_values(), scratch.q8_0_scales(), stream);
        q38_moe_gemv_q8_0_exact(
            router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
    } else {
        launch_matrix_gemv_f32_input(router, x, scratch.d_logits(), stream);
    }
    q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights());

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-router expert ids failed: ") + cudaGetErrorString(e));
    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-router expert weights failed: ") + cudaGetErrorString(e));
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("F32-router selection sync failed: ") + cudaGetErrorString(e));
}

// F32-activation router selection for the *_f32_x runners. ggml quantises the
// activation only when the router weight is quantised; an F32 router keeps both
// sides F32 and never sees the F16 weight copy (launch_matrix_gemv_f32_weights).
// The dense fallback widens the F16 weight copy exactly, so the only difference
// from q38_select_experts_f32_input() is which of the two non-quantised weight
// copies is used. The legacy half-input q38_select_experts() is untouched.
inline void q38_moe_select_experts_f32_x(
        const GpuMatrix & router,
        const float * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }

    if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
        // The F32 router input is quantised in place, exactly like ggml.
        q38_moe_quantize_activation_q8_0(
            x, scratch.hidden(), scratch.q8_0_values(), scratch.q8_0_scales(), stream);
        q38_moe_gemv_q8_0_exact(
            router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
    } else if (q38_moe_router_f32_enabled() && matrix_has_f32_weights(router)) {
        launch_matrix_gemv_f32_weights(router, x, scratch.d_logits(), stream);
    } else {
        launch_matrix_gemv_f32_input(router, x, scratch.d_logits(), stream);
    }

    q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights());

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-x router expert ids failed: ") + cudaGetErrorString(e));
    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-x router expert weights failed: ") + cudaGetErrorString(e));
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("F32-x router selection sync failed: ") + cudaGetErrorString(e));
}

// Activation-type dispatch shared by the routed/expert runners: F16 activations
// take the legacy widening (exact) and F32 activations are consumed directly.
template <typename XT>
inline void q38_moe_select_experts_input(
        const GpuMatrix & router,
        const XT * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_select_experts_f32_x(router, x, scratch, expert_scale, stream);
    } else {
        q38_select_experts(router, x, scratch, expert_scale, stream);
    }
}

// Write the routed weighted sum into the caller's MoE output. The F16 output
// (legacy contract) rounds once, like before; the F32 output keeps the sum
// unrounded for the F32 HC combine (out == d_accum() is allowed and is a no-op).
template <typename OutT>
inline void q38_finish_moe(
        Qwen38MoeScratch & scratch,
        OutT * out,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    const int blocks = (hidden + threads - 1) / threads;
    if constexpr (std::is_same_v<OutT, float>) {
        if (out == scratch.d_accum()) return;
        const cudaError_t e = cudaMemcpyAsync(
            out, scratch.d_accum(), static_cast<size_t>(hidden) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("copy F32 MoE output failed: ") + cudaGetErrorString(e));
        }
    } else {
        q38_float_to_half_kernel<<<blocks, threads, 0, stream>>>(
            scratch.d_accum(), out, hidden);
    }
}

// F32 twin of q38_add_shared_kernel(): routed[i] += shared[i] * gate[0].
inline void q38_moe_add_shared_f32(
        float * routed_in_out,
        const float * shared,
        const float * gate,
        int n,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    q38_add_shared_f32_kernel<<<blocks, threads, 0, stream>>>(
        routed_in_out, shared, gate, n);
}

// Complete F32 MoE branch: out = routed weighted sum + shared expert * gate,
// the F32 twin of q38_finish_moe() followed by q38_add_shared_kernel(). "out"
// must be a scratch buffer distinct from d_accum(): the routed sum stays
// available for tracing (layer*.ffn.*), and the caller typically passes
// Qwen38MoeScratch::moe_out_f32().
inline void q38_finish_moe_f32(
        Qwen38MoeScratch & scratch,
        float * out,
        cudaStream_t stream) {
    if (out == scratch.d_accum()) {
        throw std::runtime_error(
            "F32 MoE output must not alias the routed accumulation buffer");
    }
    q38_finish_moe(scratch, out, stream);
    q38_moe_add_shared_f32(
        out, scratch.down_f32(), scratch.d_shared_gate(), scratch.hidden(), stream);
}

// F16 activation for the legacy F16-activation fallbacks of the F32-input
// runners. Those GEMV kernels are F16-activation by construction, so the F32
// activation is rounded once here; the ggml-exact Q8/Q8_K paths never call this.
// An F16 input is returned unchanged (no conversion, no rounding).
template <typename XT>
inline const __half * q38_moe_legacy_half_activation(
        const XT * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        const int hidden = scratch.hidden();
        constexpr int threads = 256;
        q38_hc_copy_f32_to_half_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.half_activation(), hidden);
        return scratch.half_activation();
    } else {
        return x;
    }
}

// Accumulate one selected expert into the routed F32 sum. The F32-mid path
// reads the unrounded projection (ggml keeps the per-expert product in F32);
// the legacy path reads the F16 buffer the expert wrote. Exactly one of the two
// source pointers is used.
inline void q38_moe_accumulate_expert(
        const float * expert_f32,
        const __half * expert_half,
        float weight,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    const int blocks = (hidden + threads - 1) / threads;
    if (f32_mid) {
        q38_weighted_accum_f32_kernel<<<blocks, threads, 0, stream>>>(
            expert_f32, weight, scratch.d_accum(), hidden);
    } else {
        q38_weighted_accum_kernel<<<blocks, threads, 0, stream>>>(
            expert_half, weight, scratch.d_accum(), hidden);
    }
}

// Debug trace of one selected expert's output. The trace buffer stays F16
// (forward.cuh reads it with copy_half and exposes it as
// layer*.ffn.selected_expert_outputs), so the F32-mid path rounds only this
// copy and never the value that feeds the accumulation.
inline void q38_moe_capture_expert_output(
        const float * expert_f32,
        const __half * expert_half,
        __half * destination,
        int n,
        bool f32_mid,
        cudaStream_t stream) {
    if (f32_mid) {
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        q38_float_to_half_kernel<<<blocks, threads, 0, stream>>>(expert_f32, destination, n);
        return;
    }
    const cudaError_t e = cudaMemcpyAsync(
        destination, expert_half, static_cast<size_t>(n) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("copy selected expert output failed: ") + cudaGetErrorString(e));
    }
}

// Quantise the MoE input to Q8_K once per layer. gate and up consume the same
// activation, so every selected expert reuses this single quantisation instead
// of rounding the activation per projection.
inline void q38_moe_quantize_routed_x_q8_k(
        const __half * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
        x, scratch.rx_x_f32(), hidden);
    launch_q8_k_quantize_f32(
        scratch.rx_x_f32(), hidden, scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
        scratch.rx_q8k_bsums(), stream);
}

// F32-activation twin of q38_moe_quantize_routed_x_q8_k(): the caller's F32
// activation is quantised in place instead of being rounded to F16 first.
// Bit-identical to the F16 version whenever x is exactly F16-representable
// (F16 -> F32 widening is exact, so the quantiser sees the same values).
inline void q38_moe_quantize_routed_x_q8_k_f32(
        const float * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    launch_q8_k_quantize_f32(
        x, scratch.hidden(), scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
        scratch.rx_q8k_bsums(), stream);
}

// Activation-type dispatch for the routed expert gate/up quantisation.
template <typename XT>
inline void q38_moe_quantize_routed_x(
        const XT * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_quantize_routed_x_q8_k_f32(x, scratch, stream);
    } else {
        q38_moe_quantize_routed_x_q8_k(x, scratch, stream);
    }
}

// IQ2_S and IQ3_S both consume the Q8_K activation, so the routed gate/up
// projection only has to pick the matching weight decoder. Anything else is a
// programming error: q38_moe_gate_up_q8_k_type() already gated the dispatch.
inline void q38_moe_gemv_gate_up_q8_k_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream) {
    if (weights.type == GgmlType::IQ2_S) {
        launch_iq2_s_gemv_q8_k_f32(weights, activation, activation_scales, output, stream);
    } else if (weights.type == GgmlType::IQ3_S) {
        launch_iq3_s_gemv_q8_k_f32(weights, activation, activation_scales, output, stream);
    } else {
        throw std::runtime_error("routed expert gate/up weight is neither IQ2_S nor IQ3_S");
    }
}

// Shared second half of the ggml-order routed expert FFN: ggml's vectorised SiLU
// on the F32 gate, the gated product in F32, its Q8_0 re-quantisation and the
// IQ4_NL down projection. The down projection lands in rx_down_f32(); when
// round_half is set it is additionally copied to the F16 d_down() buffer so the
// legacy q38_weighted_accum_kernel() and the legacy outputs stay unchanged.
// The F32-mid path passes round_half = false and accumulates rx_down_f32().
inline void q38_moe_finish_routed_expert_q8_act(
        float * gate_f32,
        const float * up_f32,
        const LowBitMatrix & down,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    constexpr int threads = 256;
    const int ffn_blocks = (expert_ffn + threads - 1) / threads;
    const int hidden_blocks = (hidden + threads - 1) / threads;

    q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        gate_f32, expert_ffn, 1.0f);
    q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        gate_f32, up_f32, scratch.rx_act_f32(), expert_ffn);
    q38_moe_quantize_activation_q8_0(
        scratch.rx_act_f32(), expert_ffn, scratch.rx_q8_0_values(), scratch.rx_q8_0_scales(), stream);
    launch_iq4_nl_gemv_q8_0_f32(
        down, scratch.rx_q8_0_values(), scratch.rx_q8_0_scales(), scratch.rx_down_f32(), stream);
    if (round_half) {
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.rx_down_f32(), scratch.d_down(), hidden);
    }
}

// XT is the activation type: __half keeps the legacy widen-then-quantise step,
// float quantises the caller's F32 activation directly.
template <bool CaptureSelectedOutput, typename XT = __half>
inline void q38_moe_run_routed_experts_q8_act_separate(
        int layer,
        const XT * x,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();

    q38_moe_quantize_routed_x(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);
        if (!q38_moe_gate_up_q8_k_type(w.gate.type) || !q38_moe_gate_up_q8_k_type(w.up.type) ||
            w.down.type != GgmlType::IQ4_NL) {
            throw std::runtime_error("Qwen3.8 routed expert low-bit type changed inside a layer");
        }

        q38_moe_gemv_gate_up_q8_k_f32(
            w.gate, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_gate_f32(), stream);
        q38_moe_gemv_gate_up_q8_k_f32(
            w.up, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_up_f32(), stream);
        // round_half is the F16 fallback: the F32 path keeps rx_down_f32().
        q38_moe_finish_routed_expert_q8_act(
            scratch.rx_gate_f32(), scratch.rx_up_f32(), w.down, scratch, !f32_mid, stream);

        if constexpr (CaptureSelectedOutput) {
            q38_moe_capture_expert_output(
                scratch.rx_down_f32(), scratch.d_down(),
                selected_output_trace + static_cast<size_t>(j) * hidden,
                hidden, f32_mid, stream);
        }
        q38_moe_accumulate_expert(
            scratch.rx_down_f32(), scratch.d_down(), weight, scratch, f32_mid, stream);
    }
}

template <bool CaptureSelectedOutput, typename XT = __half>
inline void q38_moe_run_routed_experts_q8_act_fused(
        int layer,
        const XT * x,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();

    q38_moe_quantize_routed_x(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);
        if (!q38_moe_gate_up_q8_k_type(w.gate_up.type) || w.down.type != GgmlType::IQ4_NL) {
            throw std::runtime_error("Qwen3.8 fused expert low-bit type changed inside a layer");
        }

        q38_moe_gemv_gate_up_q8_k_f32(
            w.gate_up, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_gate_up_f32(), stream);
        // The fused matrix stores gate first and up second, matching
        // q38_silu_mul_fused_kernel on the half path.
        q38_moe_finish_routed_expert_q8_act(
            scratch.rx_gate_up_f32(), scratch.rx_gate_up_f32() + expert_ffn, w.down, scratch,
            !f32_mid, stream);

        if constexpr (CaptureSelectedOutput) {
            q38_moe_capture_expert_output(
                scratch.rx_down_f32(), scratch.d_down(),
                selected_output_trace + static_cast<size_t>(j) * hidden,
                hidden, f32_mid, stream);
        }
        q38_moe_accumulate_expert(
            scratch.rx_down_f32(), scratch.d_down(), weight, scratch, f32_mid, stream);
    }
}

// MoE runner for one layer. OutT is the routed-output element type: the legacy
// call sites pass a __half * (WARODECODE_MOE_F32_MID then only removes the F16
// roundings *inside* the layer and keeps the single final round), while the F32
// call sites pass a float * (see run_qwen38_routed_moe_fused_f32) and consume
// the unrounded routed sum with the F32 HC combine.
template <bool CaptureSelectedOutput = false, typename OutT = __half, typename XT = __half>
inline void run_qwen38_routed_moe_fused(
        int layer,
        const XT * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        OutT * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    if constexpr (CaptureSelectedOutput) {
        if (!selected_output_trace) throw std::runtime_error("selected expert trace buffer is null");
    }
    if (gate_up_exps.dims.size() != 3 || down_exps.dims.size() != 3) {
        throw std::runtime_error("Qwen3.8 fused expert tensors must be 3D");
    }
    if (static_cast<int>(gate_up_exps.dims[1]) != 2 * scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 fused gate-up FFN width mismatch");
    }

    if (!preselected) q38_moe_select_experts_input<XT>(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaMemsetAsync(
        scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const bool f32_mid = q38_moe_f32_mid_enabled();
    if (q38_moe_fused_experts_q8_act_compatible(gate_up_exps, down_exps, scratch)) {
        q38_moe_run_routed_experts_q8_act_fused<CaptureSelectedOutput, XT>(
            layer, x, gate_up_exps, down_exps, cache, scratch, f32_mid, selected_output_trace, stream);
        q38_finish_moe(scratch, out, stream);
        return;
    }

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;
    // Legacy F16-activation fallback: round the F32 input once, exactly what the
    // half-input runner receives.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate_up, legacy_x, scratch.d_gate_up(), stream);
        q38_silu_mul_fused_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        if constexpr (CaptureSelectedOutput) {
            e = cudaMemcpyAsync(
                selected_output_trace + static_cast<size_t>(j) * scratch.hidden(),
                scratch.d_down(), static_cast<size_t>(scratch.hidden()) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) throw std::runtime_error(std::string("copy selected fused expert output failed: ") + cudaGetErrorString(e));
        }
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}

// F32-output twin of run_qwen38_routed_moe_fused(): the routed weighted sum is
// written unrounded so it can be combined with the F32 shared expert through
// q38_finish_moe_f32() or the F32 HC combine.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_fused_f32(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_fused<CaptureSelectedOutput, float>(
        layer, x, router, gate_up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

// F32-activation twin of run_qwen38_routed_moe_fused_f32(): the MoE input is
// consumed as F32, so the router activation, the expert Q8_K activation and the
// F32 intermediates never pass through F16. Whenever x is exactly
// F16-representable the result is bit-identical to the half-input version.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_fused_f32_x(
        int layer,
        const float * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_fused<CaptureSelectedOutput, float, float>(
        layer, x, router, gate_up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

template <bool CaptureSelectedOutput = false, typename OutT = __half, typename XT = __half>
inline void run_qwen38_routed_moe_separate(
        int layer,
        const XT * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        OutT * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    if constexpr (CaptureSelectedOutput) {
        if (!selected_output_trace) throw std::runtime_error("selected expert trace buffer is null");
    }
    if (!preselected) q38_moe_select_experts_input<XT>(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaMemsetAsync(
        scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const bool f32_mid = q38_moe_f32_mid_enabled();
    if (q38_moe_separate_experts_q8_act_compatible(gate_exps, up_exps, down_exps, scratch)) {
        q38_moe_run_routed_experts_q8_act_separate<CaptureSelectedOutput, XT>(
            layer, x, gate_exps, up_exps, down_exps, cache, scratch, f32_mid,
            selected_output_trace, stream);
        q38_finish_moe(scratch, out, stream);
        return;
    }

    // Legacy F16-activation fallback for weight types the ggml-exact path does
    // not cover. WARODECODE_MOE_F32_MID deliberately leaves it alone: its
    // GEMV chain is F16-activation based, the model's experts are IQ2_S or IQ3_S
    // (or Q8_0 in the unit test), and the F32 runners below are the supported
    // way to get an unrounded MoE branch.
    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;
    // The F32-input runner reaches this fallback too: round its activation once
    // (a no-op for the half-input runner), which is exactly what these
    // F16-activation GEMV kernels would have received.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate, legacy_x, scratch.d_gate(), stream);
        launch_lowbit_gemv_half(w.up, legacy_x, scratch.d_up(), stream);
        q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        if constexpr (CaptureSelectedOutput) {
            const cudaError_t copy_error = cudaMemcpyAsync(
                selected_output_trace + static_cast<size_t>(j) * scratch.hidden(),
                scratch.d_down(), static_cast<size_t>(scratch.hidden()) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream);
            if (copy_error != cudaSuccess) throw std::runtime_error(std::string("copy selected expert output failed: ") + cudaGetErrorString(copy_error));
        }
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}

// F32-output twin of run_qwen38_routed_moe_separate(): the routed weighted sum
// is written unrounded so it can be combined with the F32 shared expert through
// q38_finish_moe_f32() or the F32 HC combine.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_separate_f32(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_separate<CaptureSelectedOutput, float>(
        layer, x, router, gate_exps, up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

// F32-activation twin of run_qwen38_routed_moe_separate_f32(): the routed MoE
// input is consumed as F32 (router activation and expert Q8_K activation stay
// unrounded). Bit-identical to the half-input version whenever x is exactly
// F16-representable.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_separate_f32_x(
        int layer,
        const float * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_separate<CaptureSelectedOutput, float, float>(
        layer, x, router, gate_exps, up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}


// True when the shared-expert gate weight is resident in F32 (the pinned GGUF's
// F32 tensor) and the F32 path is enabled, so the gate can be built the way the
// CPU oracle builds it.
inline bool q38_moe_shexp_gate_f32_ready(
        const Qwen38SharedExpertWeights & shared,
        int hidden) {
    return q38_moe_shexp_gate_f32_enabled() &&
        shared.gate_input_f32.data != nullptr &&
        shared.gate_input_f32.size == static_cast<size_t>(hidden);
}

// Shared-expert gate: sigmoid(build_lora_mm(ffn_gate_inp_shexp, cur)).
//
// With the F32 weight resident the dot is one ggml_vec_dot_f32 in AVX2 order; an
// F16 activation widens exactly into act_f32() first (the oracle's activation is
// F32, so widening only removes the path's own rounding). Otherwise the legacy
// F16-weight dot runs: q38_moe_gate_dot_f32_x() for an F32 activation and
// half_dot_kernel() for an F16 one.
template <typename XT>
inline void q38_moe_shared_gate_dot(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    if (q38_moe_shexp_gate_f32_ready(shared, hidden)) {
        if constexpr (std::is_same_v<XT, float>) {
            q38_moe_gate_dot_f32_avx2(
                shared.gate_input_f32.data, x, scratch.d_shared_gate(), hidden, stream);
        } else {
            constexpr int threads = 256;
            const int blocks = (hidden + threads - 1) / threads;
            q38_hc_copy_half_to_f32_kernel<<<blocks, threads, 0, stream>>>(
                x, scratch.act_f32(), hidden);
            q38_moe_gate_dot_f32_avx2(
                shared.gate_input_f32.data, scratch.act_f32(),
                scratch.d_shared_gate(), hidden, stream);
        }
        return;
    }
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_gate_dot_f32_x(
            shared.gate_input.data, x, scratch.d_shared_gate(), hidden, stream);
    } else {
        half_dot_kernel<<<1, 256, 0, stream>>>(
            shared.gate_input.data, x, scratch.d_shared_gate(), hidden);
    }
}

inline void q38_moe_check_shared_expert_geometry(
        const Qwen38SharedExpertWeights & shared,
        const Qwen38MoeScratch & scratch) {
    if (shared.gate_input.size != scratch.hidden() ||
        shared.gate.rows != scratch.expert_ffn() || shared.gate.cols != scratch.hidden() ||
        shared.up.rows != scratch.expert_ffn() || shared.up.cols != scratch.hidden() ||
        shared.down.rows != scratch.hidden() || shared.down.cols != scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 shared expert geometry mismatch");
    }
}

// ggml-order shared expert: F32 gate/up projections with a Q8_0-quantised
// activation, ggml SiLU, then the down projection with the activation
// re-quantised from the unrounded F32 gated value. The unrounded down
// projection stays in down_f32(); round_half additionally copies it into the
// F16 d_down() buffer the legacy accumulation kernel consumes.
//
// Activation type: an F16 input is widened into act_f32() first (exact) and
// quantised from there, an F32 input is quantised in place. Everything after the
// first quantisation is identical.
template <typename XT>
inline void q38_moe_run_shared_expert_q8_0_input(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    constexpr int threads = 256;
    const int hidden_blocks = (hidden + threads - 1) / threads;
    const int ffn_blocks = (expert_ffn + threads - 1) / threads;

    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_quantize_activation_q8_0(
            x, hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    } else {
        q38_hc_copy_half_to_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        q38_moe_quantize_activation_q8_0(
            scratch.act_f32(), hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    }
    q38_moe_gemv_q8_0_exact(
        shared.gate, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.gate_f32(), stream);
    q38_moe_gemv_q8_0_exact(
        shared.up, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.up_f32(), stream);

    q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.gate_f32(), expert_ffn, 1.0f);
    q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.gate_f32(), scratch.up_f32(), scratch.act_f32(), expert_ffn);

    q38_moe_quantize_activation_q8_0(
        scratch.act_f32(), expert_ffn, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    q38_moe_gemv_q8_0_exact(
        shared.down, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.down_f32(), stream);
    if (round_half) {
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.down_f32(), scratch.d_down(), hidden);
    }
}

// F16-activation entry point of the chain above (unchanged signature/contract).
inline void run_qwen38_shared_expert_q8_0(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_q8_0_input<__half>(x, shared, scratch, round_half, stream);
}

// F32 shared expert (WARODECODE_MOE_F32_MID). The unrounded result stays in
// down_f32() and the sigmoid gate in d_shared_gate(), so the caller can build
// the MoE branch with q38_moe_add_shared_f32() / q38_finish_moe_f32() without
// rounding the shared-expert output to F16.
//
// Activation type: only three steps depend on it -- the gate dot product
// (q38_moe_shared_gate_dot() keeps the oracle's F32 weight and widens an F16
// activation into act_f32() before the AVX2 dot), the first Q8_0 quantisation
// (an F16 input is widened exactly, an F32 input is quantised in place) and the
// legacy F16 fallback below. Everything else is shared so the two entry points
// cannot drift apart.
template <typename XT>
inline void q38_moe_run_shared_expert_f32_input(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    q38_moe_check_shared_expert_geometry(shared, scratch);

    q38_moe_shared_gate_dot(x, shared, scratch, stream);
    sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(scratch.d_shared_gate());

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    if (q38_moe_shexp_q8_0_enabled() && q38_moe_shexp_q8_0_compatible(shared, scratch)) {
        q38_moe_run_shared_expert_q8_0_input<XT>(x, shared, scratch, false, stream);
        return;
    }

    // Legacy F16-activation chain for weight types the Q8_0 path does not
    // cover; only its final output is lifted into F32 so the F32 contract holds.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);
    launch_lowbit_gemv_half(shared.gate, legacy_x, scratch.d_gate(), stream);
    launch_lowbit_gemv_half(shared.up, legacy_x, scratch.d_up(), stream);
    q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
    launch_lowbit_gemv_half(shared.down, scratch.d_act(), scratch.d_down(), stream);
    q38_hc_copy_half_to_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
        scratch.d_down(), scratch.down_f32(), scratch.hidden());
}

inline void run_qwen38_shared_expert_f32(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_f32_input<__half>(x, shared, scratch, stream);
}

// F32-activation twin of run_qwen38_shared_expert_f32(): the sigmoid gate dot
// product and the Q8_0 activation quantisation read the caller's F32 activation
// directly, so neither the gate weight widening nor the expert activation loses
// precision to F16. Bit-identical to the half-input version whenever x is
// exactly F16-representable.
inline void run_qwen38_shared_expert_f32_x(
        const float * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_f32_input<float>(x, shared, scratch, stream);
}

// Legacy contract: routes the shared expert into the F16 buffer in place. With
// WARODECODE_MOE_F32_MID the expert is computed unrounded and only this final
// F16 store (and the existing half accumulation) rounds; the F32 path is
// run_qwen38_shared_expert_f32() plus q38_moe_add_shared_f32().
inline void run_qwen38_shared_expert(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        __half * routed_in_out,
        cudaStream_t stream) {
    q38_moe_check_shared_expert_geometry(shared, scratch);

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    if (q38_moe_f32_mid_enabled()) {
        run_qwen38_shared_expert_f32(x, shared, scratch, stream);
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.down_f32(), scratch.d_down(), scratch.hidden());
        q38_add_shared_kernel<<<hidden_blocks, threads, 0, stream>>>(
            routed_in_out, scratch.d_down(), scratch.d_shared_gate(), scratch.hidden());
        return;
    }

    q38_moe_shared_gate_dot(x, shared, scratch, stream);
    sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(scratch.d_shared_gate());

    if (q38_moe_shexp_q8_0_enabled() && q38_moe_shexp_q8_0_compatible(shared, scratch)) {
        run_qwen38_shared_expert_q8_0(x, shared, scratch, true, stream);
    } else {
        launch_lowbit_gemv_half(shared.gate, x, scratch.d_gate(), stream);
        launch_lowbit_gemv_half(shared.up, x, scratch.d_up(), stream);
        q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(shared.down, scratch.d_act(), scratch.d_down(), stream);
    }
    q38_add_shared_kernel<<<hidden_blocks, threads, 0, stream>>>(
        routed_in_out, scratch.d_down(), scratch.d_shared_gate(), scratch.hidden());
}

} // namespace wd
