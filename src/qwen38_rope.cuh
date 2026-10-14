#pragma once

// Qwen3.8 interleaved M-RoPE (IMRoPE) applied in place to an F16 Q or K tensor.
//
// The arithmetic is a transcription of the llama.cpp CPU oracle
// (ggml/src/ggml-cpu/ops.cpp, pinned revision
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11); see
// src/qwen38_imrope_math.cuh for the operation sequence and for why each
// rounding matters. tests/qwen38_rope_host_test.cpp compares this arithmetic
// against the real ggml_rope_multi output bit for bit.
//
// WARODECODE_EXACT_ROPE (default ON) switches to it; WARODECODE_EXACT_ROPE=0
// restores the previous powf/sincosf kernel unchanged.

#include "qwen38_layout.hpp"
#include "qwen38_imrope_math.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace wd {

struct Qwen38Position4 {
    int t = 0;
    int h = 0;
    int w = 0;
    int extra = 0;

    static Qwen38Position4 text(int pos) {
        return {pos, pos, pos, pos};
    }
};

// The former axis helper, kept at this name because src/qwen38_qsa.cuh calls it
// from its own F32 twin kernel. The body lives in qwen38_imrope_math.cuh.
__device__ __forceinline__ int q38_imrope_axis(
        int pair,
        int s0,
        int s1,
        int s2,
        int s3) {
    return q38imrope::axis_for_pair(pair, s0, s1, s2, s3);
}

#if defined(__CUDACC__)

// One read per thread; constant for the process. Published from the dynamic
// initialisation below, i.e. before main(), so no launcher has to remember the
// call. Every engine/test/tool binary here is a single CUDA translation unit
// (CUDA_SEPARABLE_COMPILATION OFF), so the inline variable runs exactly once.
__device__ int q38_exact_rope_flag = 1;

inline bool q38_rope_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_exact_rope_enabled() {
    static const bool enabled = [] {
        const int value = q38_rope_env_flag("WARODECODE_EXACT_ROPE", true) ? 1 : 0;
        // The glibc tables have to reach the device before the first launch.
        q38glibc::q38_glibc_fill_device_tables();
        const cudaError_t e = cudaMemcpyToSymbol(q38_exact_rope_flag, &value, sizeof(value));
        // The device flag starts at 1 (exact), the same selection the default
        // WARODECODE_EXACT_ROPE value asks for, so a publish that cannot run
        // degrades to the default instead of aborting from a static initialiser.
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

inline const bool q38_exact_rope_published = q38_exact_rope_enabled();

__device__ __forceinline__ bool q38_exact_rope_on() {
    return q38_exact_rope_flag != 0;
}

__device__ __forceinline__ void q38_imrope_rotate_f16(
        __half * x, int i0, int i1, float theta_base, float theta_scale_value, int pair) {
    const float a = __half2float(x[i0]);
    const float b = __half2float(x[i1]);
    float out0, out1;
    q38imrope::rotate_pair_exact(theta_base, theta_scale_value, pair, a, b, &out0, &out1);
    x[i0] = __float2half(out0);
    x[i1] = __float2half(out1);
}

__global__ void q38_imrope_neox_inplace_kernel(
        __half * x,
        int head_dim,
        int rotary_dim,
        int n_heads,
        int4 pos,
        int4 sections,
        float freq_base) {
    const int pair_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = rotary_dim / 2;
    const int total_pairs = n_heads * pairs_per_head;
    if (pair_global >= total_pairs) return;

    const int head = pair_global / pairs_per_head;
    const int pair = pair_global - head * pairs_per_head;
    const int axis = q38imrope::axis_for_pair(
        pair, sections.x, sections.y, sections.z, sections.w);

    const int p = axis == 0 ? pos.x :
                  axis == 1 ? pos.y :
                  axis == 2 ? pos.z : pos.w;

    const int base = head * head_dim;
    const int i0 = base + pair;
    const int i1 = i0 + pairs_per_head;

    if (q38_exact_rope_on()) {
        const float theta_scale_value = q38imrope::theta_scale(freq_base, rotary_dim);
        q38_imrope_rotate_f16(x, i0, i1, static_cast<float>(p), theta_scale_value, pair);
    } else {
        const float theta_scale_value = powf(freq_base, -2.0f / static_cast<float>(rotary_dim));
        const float theta = static_cast<float>(p) * powf(theta_scale_value, static_cast<float>(pair));
        float sn, cs;
        sincosf(theta, &sn, &cs);

        const float a = __half2float(x[i0]);
        const float b = __half2float(x[i1]);
        x[i0] = __float2half(a * cs - b * sn);
        x[i1] = __float2half(a * sn + b * cs);
    }
}

#endif // __CUDACC__

inline void launch_qwen38_imrope(
        __half * x,
        int head_dim,
        int n_heads,
        const Qwen38Position4 & pos,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    if (head_dim <= 0 || (head_dim & 1) || n_heads <= 0) {
        throw std::runtime_error("invalid IMRoPE geometry");
    }
    if (q.rope_dimension_count <= 0 || q.rope_dimension_count > head_dim ||
        (q.rope_dimension_count & 1) != 0) {
        throw std::runtime_error("invalid Qwen3.8 rotary dimension count");
    }
    if (q.rope_sections.size() != 4) {
        throw std::runtime_error("Qwen3.8 IMRoPE requires 4 rope.dimension_sections");
    }

    const int4 p = make_int4(pos.t, pos.h, pos.w, pos.extra);
    const int4 s = make_int4(
        static_cast<int>(q.rope_sections[0]),
        static_cast<int>(q.rope_sections[1]),
        static_cast<int>(q.rope_sections[2]),
        static_cast<int>(q.rope_sections[3]));

    const int total_pairs = n_heads * (q.rope_dimension_count / 2);
    const int threads = 256;
    q38_imrope_neox_inplace_kernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
        x, head_dim, q.rope_dimension_count, n_heads, p, s, q.rope_freq_base);
}

} // namespace wd
