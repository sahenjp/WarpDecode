#pragma once

// Exact transcendental helpers for the warp blocks outside the GDN kernel
// (qwen38_ple.cuh / qwen38_qsa.cuh / qwen38_moe.cuh / qwen38_hc.cuh).
//
// WARODECODE_EXACT_EXP (default ON) selects, per call site, the same
// transcendental the pinned CPU oracle executes for the matching ggml op.
// qwen38_gdn.cuh documents the derivation; this header is a COPY of those
// helpers so that the other blocks can reach them without depending on the GDN
// kernel (qwen38_gdn.cuh is deliberately left untouched).
//
//   ggml_silu
//     -> ggml_compute_forward_silu_f32 -> ggml_vec_silu_f32
//        (ggml-cpu/vec.cpp:380), whose AVX2 branch is ggml_v_silu
//        (ggml-cpu/vec.h:1255). It never calls libm: expf is the polynomial
//        ggml_v_expf (ggml-cpu/vec.h:1215) built from _mm256_fmadd_ps /
//        _mm256_fnmadd_ps / _mm256_mul_ps / _mm256_add_ps / _mm256_div_ps, each
//        a single IEEE operation. q38_exact_avx2_exp_f32() below is the
//        lane-wise copy of that polynomial; it is byte-identical to
//        q38_hc_ggml_avx2_exp_f32() in qwen38_hc.cuh, and the host bit test
//        asserts the two agree on every sampled input.
//   ggml_sigmoid
//     -> op_sigmoid (ggml-cpu/unary-ops.cpp:31): 1.f / (1.f + expf(-x)) with
//        the compiler's scalar expf, i.e. this host's glibc, which is the ARM
//        optimized-routines port in sysdeps/ieee754/flt-32/e_expf.c.
//        q38_exact_expf_glibc() is a port of that routine (0.502 ulp; a
//        correctly rounded exp is one ulp away on 0.017% of inputs, so the
//        routine itself has to be reproduced).
//   ggml_soft_max / ggml_soft_max_ext
//     -> ggml_compute_forward_soft_max_f32 (ggml-cpu/ops.cpp:5584) ->
//        ggml_vec_soft_max_f32 (ggml-cpu/vec.cpp:531): dp[i] =
//        ggml_v_expf(x[i] - max) for the first 8*floor(n/8) elements and
//        dp[i] = expf(x[i] - max) for the remaining n%8. The vector branch is
//        the one reproduced by q38_exact_softmax_exp_f32(); the element index
//        inside the softmax row is not available at the warp call sites, so the
//        n%8 tail uses the vector form as well.
//
// CUDA has no bit-comparable counterpart: device expf() is a ~2 ulp function and
// this build also passes --use_fast_math, which lowers expf() to __expf
// (MUFU.EX2) and plain '/' to an approximate divide. Every exact branch below
// therefore uses rounding-mode intrinsics only.
//
// WARODECODE_EXACT_EXP=0 restores the legacy device formulas.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace wd {

inline bool q38_exact_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

#if defined(__CUDACC__)
// One read per thread; the value is constant within a process. Published from
// the dynamic initialisation of q38_exact_exp_published below, i.e. before
// main(), so no launcher has to remember the call. Each engine/test/tool binary
// here is a single CUDA translation unit (CUDA_SEPARABLE_COMPILATION OFF), so
// the inline variable is initialised exactly once.
__device__ int q38_exact_exp_flag = 1;

inline bool q38_exact_exp_enabled() {
    static const bool enabled = [] {
        const int value = q38_exact_env_flag("WARODECODE_EXACT_EXP", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_exact_exp_flag, &value, sizeof(value));
        // The device flag is initialised to 1 (exact), the same selection the
        // default WARODECODE_EXACT_EXP value asks for, so a publish that cannot
        // run (no usable runtime) degrades to the default instead of aborting
        // the process from a static initialiser.
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

inline const bool q38_exact_exp_published = q38_exact_exp_enabled();

__device__ __forceinline__ bool q38_exact_exp_on() {
    return q38_exact_exp_flag != 0;
}
#endif

// AVX2 ggml_v_expf (ggml-cpu/vec.h:1215), one lane at a time. The vector
// function branches on the whole mask and blends; every lane's result equals the
// per-lane result computed here, including the |n| > 192 flush and the
// negative-infinity case (which returns 0 for a masked softmax cell).
__device__ __forceinline__ float q38_exact_avx2_exp_f32(float x) {
    constexpr float r = 0x1.8p23f;
    const float z = __fmaf_rn(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float inner = __fmaf_rn(-n, 0x1.62e4p-1f, x);
    const float b = __fmaf_rn(-n, 0x1.7f7d1cp-20f, inner);
    const uint32_t e = __float_as_uint(z) << 23;
    const float k = __uint_as_float(e + __float_as_uint(1.0f));
    const bool c = fabsf(n) > 126.0f;
    const float u = __fmul_rn(b, b);
    const float t0 = __fmaf_rn(0x1.0e4020p-7f, b, 0x1.573e2ep-5f);
    const float t1 = __fmaf_rn(t0, u,
        __fmaf_rn(0x1.555e66p-3f, b, 0x1.fffdb6p-2f));
    const float j = __fmaf_rn(t1, u, __fmul_rn(0x1.ffffecp-1f, b));
    if (!c) return __fmaf_rn(j, k, k);

    const uint32_t g = n <= 0.0f ? 0x82000000u : 0u;
    const float s1 = __uint_as_float(g + 0x7f000000u);
    const float s2 = __uint_as_float(e - g);
    if (fabsf(n) > 192.0f) return __fmul_rn(s1, s1);
    return __fmul_rn(__fmaf_rn(s2, j, s2), s1);
}

// glibc's expf, ported from sysdeps/ieee754/flt-32/e_expf.c. The table is
// e_exp2f_data.c's tab[] (tab[i] = uint(2^(i/N)) - (i << (52 - BITS)), N = 32)
// and the constants are that file's shift / invln2_scaled / poly_scaled. Same
// body as q38_gdn_expf_glibc() in qwen38_gdn.cuh; every arithmetic step uses an
// explicit rounding-mode intrinsic so --use_fast_math cannot fuse the double
// multiply-adds (the oracle's x86-64 glibc baseline has no FMA).
constexpr int Q38_EXACT_EXPF_N = 32;

__device__ __constant__ uint64_t k_q38_exact_expf_tab[Q38_EXACT_EXPF_N] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};

__device__ __forceinline__ float q38_exact_expf_glibc(float x) {
    constexpr double shift = 0x1.8p+52;
    constexpr double invln2n = 0x1.71547652b82fep+0 * Q38_EXACT_EXPF_N;
    constexpr double c0 = 0x1.c6af84b912394p-5 / Q38_EXACT_EXPF_N / Q38_EXACT_EXPF_N / Q38_EXACT_EXPF_N;
    constexpr double c1 = 0x1.ebfce50fac4f3p-3 / Q38_EXACT_EXPF_N / Q38_EXACT_EXPF_N;
    constexpr double c2 = 0x1.62e42ff0c52d6p-1 / Q38_EXACT_EXPF_N;

    const uint32_t ix = __float_as_uint(x);
    const uint32_t abstop = (ix >> 20) & 0x7ffu;
    if (abstop >= (0x42b00000u >> 20)) {           // top12(88.0f): huge, inf or nan
        if (ix == 0xff800000u) return 0.0f;        // -inf -> 0
        if (abstop >= 0x7f8u) return x + x;        // +inf / nan
        if (x > 0x1.62e42ep6f) return __fmul_rn(0x1p97f, 0x1p97f);      // overflow
        if (x < -0x1.9fe368p6f) return __fmul_rn(0x1p-95f, 0x1p-95f);   // underflow
    }

    const double z = __dmul_rn(invln2n, static_cast<double>(x));
    const double kd_shift = __dadd_rn(z, shift);
    const uint64_t ki = static_cast<uint64_t>(__double_as_longlong(kd_shift));
    const double kd = __dsub_rn(kd_shift, shift);
    const double r = __dsub_rn(z, kd);
    uint64_t t = k_q38_exact_expf_tab[static_cast<int>(ki % Q38_EXACT_EXPF_N)];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(static_cast<long long>(t));
    const double p = __dadd_rn(__dmul_rn(c0, r), c1);
    const double r2 = __dmul_rn(r, r);
    double y = __dadd_rn(__dmul_rn(c2, r), 1.0);
    y = __dadd_rn(__dmul_rn(p, r2), y);
    y = __dmul_rn(y, s);
    return static_cast<float>(y);
}

// expf() with the switch: the glibc port when the exact path is on, device expf
// otherwise (a plain expf() for the call sites that already ran one).
// Block-wide switch for these call sites (WARODECODE_EXACT_EXP_BLOCKS, default
// OFF). With the GDN kernel the exact transcendentals moved the greedy prefix
// from 22 to 41 tokens, but applying the same helpers to the PLE/QSA/MoE/HC
// sites regressed it (10 with softmax included, 22 without), so they stay
// opt-in until each structural mismatch the host tests cannot see - the online
// softmax rescale, the MoE router normalisation chain - is fixed too.
__device__ int q38_exact_block_flag = 1;

inline bool q38_exact_block_enabled() {
    static const bool enabled = [] {
        const int value = q38_exact_env_flag("WARODECODE_EXACT_EXP_BLOCKS", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_exact_block_flag, &value, sizeof(value));
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

inline const bool q38_exact_block_published = q38_exact_block_enabled();

__device__ __forceinline__ float q38_exact_expf_f32(float x, bool exact_exp) {
    if (!exact_exp || q38_exact_block_flag == 0) return expf(x);
    return q38_exact_expf_glibc(x);
}

// op_sigmoid(): 1 / (1 + expf(-x)).
__device__ __forceinline__ float q38_exact_sigmoid_f32(float x, bool exact_exp) {
    if (!exact_exp || q38_exact_block_flag == 0) return 1.0f / (1.0f + expf(-x));
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, q38_exact_expf_glibc(-x)));
}

// ggml_silu: x / (1 + exp_avx2(-x)), one IEEE add and one IEEE divide.
__device__ __forceinline__ float q38_exact_silu_f32(float x, bool exact_exp) {
    if (!exact_exp || q38_exact_block_flag == 0) return x / (1.0f + expf(-x));
    const float neg_x = __fsub_rn(0.0f, x);
    const float exp_neg_x = q38_exact_avx2_exp_f32(neg_x);
    return __fdiv_rn(x, __fadd_rn(1.0f, exp_neg_x));
}

// ggml_vec_soft_max_f32(): exp(x - max) through the AVX2 ggml_v_expf. The
// subtraction is its own IEEE step (the vector code is _mm256_sub_ps followed by
// ggml_v_expf), and the n%8 scalar-expf tail is not modelled.
// The softmax sites are gated by their own switch (WARODECODE_EXACT_SOFTMAX_EXP,
// default OFF): the CPU oracle evaluates softmax with a single max pass over all
// visible elements, while warp rescales online, so matching exp alone does not
// make the rounding order agree and can move the tokens the wrong way.
__device__ int q38_exact_softmax_flag = 0;

inline bool q38_exact_softmax_enabled() {
    static const bool enabled = [] {
        const int value = q38_exact_env_flag("WARODECODE_EXACT_SOFTMAX_EXP", false) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_exact_softmax_flag, &value, sizeof(value));
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

inline const bool q38_exact_softmax_published = q38_exact_softmax_enabled();

__device__ __forceinline__ float q38_exact_softmax_exp_f32(float x, float max, bool exact_exp) {
    if (!exact_exp || q38_exact_block_flag == 0 || q38_exact_softmax_flag == 0) return expf(x - max);
    return q38_exact_avx2_exp_f32(__fsub_rn(x, max));
}

}  // namespace wd
