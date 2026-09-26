#pragma once

#include "qwen38_layout.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

__device__ __forceinline__ int q38_imrope_axis(
        int pair,
        int s0,
        int s1,
        int s2,
        int s3) {
    const int sect_dims = s0 + s1 + s2 + s3;
    if (sect_dims <= 0) return 0;
    const int sector = pair % sect_dims;

    if (sector % 3 == 1 && sector < 3 * s1) return 1;
    if (sector % 3 == 2 && sector < 3 * s2) return 2;
    if (sector % 3 == 0 && sector < 3 * s0) return 0;
    return 3;
}

__global__ void q38_imrope_neox_inplace_kernel(
        __half * x,
        int head_dim,
        int n_heads,
        int4 pos,
        int4 sections,
        float freq_base) {
    const int pair_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = head_dim / 2;
    const int total_pairs = n_heads * pairs_per_head;
    if (pair_global >= total_pairs) return;

    const int head = pair_global / pairs_per_head;
    const int pair = pair_global - head * pairs_per_head;
    const int axis = q38_imrope_axis(
        pair, sections.x, sections.y, sections.z, sections.w);

    const int p = axis == 0 ? pos.x :
                  axis == 1 ? pos.y :
                  axis == 2 ? pos.z : pos.w;

    const float theta_scale = powf(freq_base, -2.0f / static_cast<float>(head_dim));
    const float theta = static_cast<float>(p) * powf(theta_scale, static_cast<float>(pair));
    float sn, cs;
    sincosf(theta, &sn, &cs);

    const int base = head * head_dim;
    const int i0 = base + pair;
    const int i1 = i0 + pairs_per_head;
    const float a = __half2float(x[i0]);
    const float b = __half2float(x[i1]);
    x[i0] = __float2half(a * cs - b * sn);
    x[i1] = __float2half(a * sn + b * cs);
}

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
    if (q.rope_sections.size() != 4) {
        throw std::runtime_error("Qwen3.8 IMRoPE requires 4 rope.dimension_sections");
    }

    const int4 p = make_int4(pos.t, pos.h, pos.w, pos.extra);
    const int4 s = make_int4(
        static_cast<int>(q.rope_sections[0]),
        static_cast<int>(q.rope_sections[1]),
        static_cast<int>(q.rope_sections[2]),
        static_cast<int>(q.rope_sections[3]));

    const int total_pairs = n_heads * (head_dim / 2);
    const int threads = 256;
    q38_imrope_neox_inplace_kernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
        x, head_dim, n_heads, p, s, q.rope_freq_base);
}

} // namespace wd
