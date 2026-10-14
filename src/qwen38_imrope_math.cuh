#pragma once

// Device-and-host compilable transcription of the operations the llama.cpp CPU
// oracle performs for the Qwen3.8 interleaved M-RoPE (IMRoPE).
//
// This file is included by src/qwen38_rope.cuh (the CUDA kernel) and, verbatim,
// by tests/qwen38_rope_host_test.cpp, so the host bit-comparison exercises the
// same arithmetic the kernel runs.
//
// ---------------------------------------------------------------------------
// The CPU sequence, ggml/src/ggml-cpu/ops.cpp at the pinned revision
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11:
//
//   ggml_compute_forward_rope_flt:
//     const float theta_scale = powf(freq_base, -2.0f/n_dims);
//
//   ggml_mrope_cache_init (is_imrope == true, indep_sects == is_vision == false):
//     float theta_t = theta_base_t, theta_h = ..., theta_w = ..., theta_e = ...;
//     for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
//         int sector = (i0 / 2) % sect_dims;
//         float theta = theta_t;
//         if (is_imrope) {
//             if      (sector % 3 == 1 && sector < 3*sections[1]) theta = theta_h;
//             else if (sector % 3 == 2 && sector < 3*sections[2]) theta = theta_w;
//             else if (sector % 3 == 0 && sector < 3*sections[0]) theta = theta_t;
//             else                                                theta = theta_e;
//         }
//         rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale,
//                   &cache[i0 + 0], &cache[i0 + 1]);
//         cache[i0 + 1] *= sin_sign;
//         theta_t *= theta_scale;  theta_h *= theta_scale;
//         theta_w *= theta_scale;  theta_e *= theta_scale;
//     }
//
//   rope_yarn with the parameters this model uses - no freq_factors (src2 is
//   null), freq_scale = cparams.rope_freq_scale = 1, ext_factor = 0
//   (cparams.yarn_ext_factor = -1 is clamped to 0), attn_factor = mscale = 1 -
//   reduces to
//     theta_interp = 1.0f * theta;  theta = theta_interp;
//     *cos_theta = cosf(theta) * 1.0f;   *sin_theta = sinf(theta) * 1.0f;
//   where cosf/sinf are the glibc float routines (glibc 2.43 on the oracle).
//
//   rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst) then applies, for every
//   pair index j < n_dims/2 and every head, the NEOX rotation of the elements
//   (j, j + n_dims/2):
//     dst[j]            = x0*cos_theta - x1*sin_theta
//     dst[j + n_dims/2] = x0*sin_theta + x1*cos_theta
//   with float operands and one float rounding per stored element.
//
// Consequences the kernel relies on:
//   * theta_j = (float)position * theta_scale stepped by j successive float
//     multiplications, NOT powf(theta_scale, j). The two differ in the last
//     bits and the difference reaches ~1e-5 at the angles where cos(theta) is
//     large, which is the 3.4e-5/5.1e-5 mismatch this file removes.
//   * The four IMRoPE cursors all start at the same position for text input
//     (Qwen38Position4::text) and are stepped identically, so whichever axis the
//     interleaving selects the value is the same. The axis selection is kept so
//     the code stays correct for any position tuple.
//   * sincosf() must not be used: with --use_fast_math nvcc lowers it to the
//     MUFU-based __sincosf, whose absolute error (~2^-21) alone exceeds the
//     whole error budget. The glibc sinf/cosf transcription below is used
//     instead.
//   * The products are evaluated with the fusion the oracle's object code
//     measures. The host comparison in tests/qwen38_rope_host_test.cpp tries the
//     unfused form as well; the unfused form mismatches a few percent of the
//     elements, the form used here matches all of them.

#include <cmath>
#include <cstdint>

#include "warodecode_q38_glibc_trig.hpp"

namespace wd {

namespace q38imrope {

// The axis a pair belongs to, as ggml_mrope_cache_init() (is_imrope branch)
// selects it. Only the index matters: all four cursors hold the same value for
// a text position, see the file header.
__host__ __device__ __forceinline__ int axis_for_pair(
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

// theta_scale = powf(freq_base, -2.0f/n_dims).
__host__ __device__ __forceinline__ float theta_scale(float freq_base, int rotary_dim) {
    return static_cast<float>(std::pow(freq_base, -2.0f / static_cast<float>(rotary_dim)));
}

// theta_j: theta_base stepped by exactly j float multiplications.
__host__ __device__ __forceinline__ float theta_for_pair(
        float theta_base, float theta_scale_value, int pair) {
    float theta = theta_base;
    for (int i = 0; i < pair; ++i) {
        theta *= theta_scale_value;
    }
    return theta;
}

// rotate_pairs<T> of ops.cpp. Measured against the real ggml_rope_multi output:
// with x0 = src[j], x1 = src[j + n_dims/2], sn = sinf(theta), cs = cosf(theta),
// the oracle equals fmaf(x0, cs, -(x1*sn)) / fmaf(x0, sn, x1*cs) on every
// element, and the unfused form differs by one ulp on a few percent of them.
__host__ __device__ __forceinline__ void rotate_pair(
        float x0, float x1, float sn, float cs, float * out0, float * out1) {
#if defined(__CUDA_ARCH__)
    *out0 = __fmaf_rn(x0, cs, -(x1 * sn));
    *out1 = __fmaf_rn(x0, sn,  (x1 * cs));
#else
    *out0 = std::fma(x0, cs, -(x1 * sn));
    *out1 = std::fma(x0, sn,  (x1 * cs));
#endif
}

// The unfused form, kept so the harness can show which variant the oracle uses.
__host__ __device__ __forceinline__ void rotate_pair_unfused(
        float x0, float x1, float sn, float cs, float * out0, float * out1) {
    *out0 = x0 * cs - x1 * sn;
    *out1 = x0 * sn + x1 * cs;
}

// One rotated pair, exactly as the kernel computes it. This is the function the
// host harness drives.
__host__ __device__ __forceinline__ void rotate_pair_exact(
        float theta_base, float theta_scale_value, int pair,
        float x0, float x1, float * out0, float * out1) {
    const float theta = theta_for_pair(theta_base, theta_scale_value, pair);
    const float sn = q38glibc::sinf(theta);
    const float cs = q38glibc::cosf(theta);
    rotate_pair(x0, x1, sn, cs, out0, out1);
}

} // namespace q38imrope
} // namespace wd
