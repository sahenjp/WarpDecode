#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

// An F32 weight matrix kept in F32 on the device.
//
// GGML never quantises the activation of an F32 weight: the CPU type traits
// table (ggml-cpu.c, type_traits_cpu[GGML_TYPE_F32]) sets
// vec_dot_type = GGML_TYPE_F32 and vec_dot = ggml_vec_dot_f32, so
// ggml_compute_forward_mul_mat() hands the F32 activation row straight to an
// F32 dot product. GDN's ssm_beta / ssm_alpha are F32 in the pinned GGUF, so
// routing them through upload_half_array() would round the weights to F16 and
// diverge from the CPU oracle before the dot even starts. This type keeps the
// exact F32 bytes from the mmap.
struct Qwen38GdnF32Matrix {
    int rows = 0;
    int cols = 0;
    float * data = nullptr;

    Qwen38GdnF32Matrix() = default;
    Qwen38GdnF32Matrix(const Qwen38GdnF32Matrix &) = delete;
    Qwen38GdnF32Matrix & operator=(const Qwen38GdnF32Matrix &) = delete;

    Qwen38GdnF32Matrix(Qwen38GdnF32Matrix && o) noexcept
        : rows(o.rows), cols(o.cols), data(o.data) {
        o.rows = 0;
        o.cols = 0;
        o.data = nullptr;
    }

    Qwen38GdnF32Matrix & operator=(Qwen38GdnF32Matrix && o) noexcept {
        if (this != &o) {
            reset();
            rows = o.rows;
            cols = o.cols;
            data = o.data;
            o.rows = 0;
            o.cols = 0;
            o.data = nullptr;
        }
        return *this;
    }

    ~Qwen38GdnF32Matrix() { reset(); }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        rows = 0;
        cols = 0;
    }

    bool valid() const { return data != nullptr && rows > 0 && cols > 0; }
};

inline Qwen38GdnF32Matrix upload_f32_matrix(const TensorInfo & t) {
    if (t.type != GgmlType::F32) {
        throw std::runtime_error("F32 matrix upload requires an F32 tensor: " + t.name);
    }
    if (t.dims.size() != 2) {
        throw std::runtime_error("F32 matrix tensor must be 2D: " + t.name);
    }
    const size_t n = static_cast<size_t>(t.elements());
    if (n == 0 || t.data == nullptr) {
        throw std::runtime_error("F32 matrix tensor is empty: " + t.name);
    }

    Qwen38GdnF32Matrix out;
    out.cols = static_cast<int>(t.dims[0]);
    out.rows = static_cast<int>(t.dims[1]);
    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), n * sizeof(float));
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc F32 matrix failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, t.data, n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy F32 matrix failed: ") + cudaGetErrorString(c));
    }
    return out;
}

// Environment switches for the GDN paths. Both are read once per process, the
// same caching the Q8_0 switch already used.
inline bool q38_gdn_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

// WARODECODE_GDN_F32_MID (default ON).
//
// ggml only rounds an activation where it has to (Q8_0/Q8_K/BF16 activation
// quantisation, the F16 KV cache) and keeps every other intermediate in F32.
// The legacy warp path stores the GDN conv history, the projection outputs, the
// gate parameters and the gated-RMSNorm chain in __half, which adds one
// rounding per operation and compounds along the sequence. With this switch on
// those buffers live in F32 and the F16 scratch slots become mirrors that are
// refreshed for the legacy __half* API and for the layer*.gdn.* debug traces.
inline bool q38_gdn_f32_mid_enabled() {
    static const bool enabled = q38_gdn_env_flag("WARODECODE_GDN_F32_MID", true);
    return enabled;
}

inline bool q38_gdn_q8_0_enabled() {
    static const bool enabled = q38_gdn_env_flag("WARODECODE_GDN_Q8_0", true);
    return enabled;
}

// WARODECODE_EXACT_EXP (default ON).
//
// The pinned CPU oracle (llama.cpp 6fcaa16f4b360649933a54d1f91ad40ed35c0e11,
// AVX2 + FMA + glibc) evaluates the GDN transcendentals with two different
// implementations, so warp has to reproduce two different things:
//
//   * ggml_silu: the SILU op -> ggml_compute_forward_silu_f32() (ggml-cpu/ops.cpp
//     :2579) -> ggml_vec_silu_f32() (ggml-cpu/vec.cpp:380), whose AVX2 branch is
//     ggml_v_silu() (ggml-cpu/vec.h:1255). That one never calls libm: expf is the
//     polynomial ggml_v_expf() (vec.h:1215), built from _mm256_fmadd_ps /
//     _mm256_fnmadd_ps / _mm256_mul_ps / _mm256_add_ps / _mm256_div_ps. Each of
//     those is a single IEEE operation, so it is reproducible lane by lane.
//     qwen38_hc.cuh already carries that lane-wise copy as
//     q38_hc_ggml_avx2_exp_f32(), so this file reuses it instead of keeping a
//     second set of coefficients that could drift.
//   * ggml_sigmoid / ggml_exp / ggml_softplus: op_sigmoid() / op_exp() /
//     op_softplus() (ggml-cpu/unary-ops.cpp:31,39,80), plus the expf() the GDN
//     scan itself calls for exp(g) (ggml-cpu/ops.cpp:11005,11012). All of them
//     use the compiler's scalar expf/logf, i.e. this host's glibc, which is the
//     ARM optimized-routines port in sysdeps/ieee754/flt-32 (e_expf.c: 0.502
//     ulp, e_logf.c: 0.818 ulp). Those are faithful but NOT correctly rounded:
//     measured on this host (glibc 2.43) glibc expf misses the correctly rounded
//     result on 0.0585% of inputs, so (float)exp((double)x) - which is correctly
//     rounded - disagrees with the oracle by 1 ulp on 0.017% of inputs and
//     cannot give bit equality. q38_gdn_expf_glibc() / q38_gdn_logf_glibc()
//     below are therefore ports of the two routines themselves.
//     op_softplus() is (x > 20.0f) ? x : logf(1.0f + expf(x)), not the log1pf()
//     form this file used to run.
//
// CUDA has neither: device expf() is a ~2 ulp function, and the build also
// passes --use_fast_math (CMakeLists.txt:55,63), which lowers expf() to __expf
// (MUFU.EX2, ~2^-21 relative error) and turns plain '/' into an approximation
// (--prec-div=false). The exact path below therefore uses only rounding-mode
// intrinsics (and, for the glibc ports, plain non-fused double arithmetic, which
// --use_fast_math would otherwise contract - the oracle's x86-64 glibc build has
// no FMA in its baseline ISA). Host check against this machine's glibc: expf
// 300M samples, logf 260M samples, 0 mismatches (see the task report).
//
// WARODECODE_EXACT_EXP=0 restores the original warp formulas (device expf and
// '/') for A/B runs.
//
// The switch is published into a __device__ symbol instead of a kernel
// parameter because q38_gdn_ar_kernel() and q38_gdn_gated_rmsnorm_kernel() are
// also launched from qwen38_forward.cuh and tests/qwen38_gdn_cuda_test.cu,
// whose signatures this change must not alter.
__device__ int q38_gdn_exact_exp_flag = 1;

inline bool q38_gdn_exact_exp_enabled() {
    static const bool enabled = [] {
        const int value = q38_gdn_env_flag("WARODECODE_EXACT_EXP", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_gdn_exact_exp_flag, &value, sizeof(value));
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("publish WARODECODE_EXACT_EXP failed: ") + cudaGetErrorString(e));
        }
        return value != 0;
    }();
    return enabled;
}

// One read per thread per kernel; the value is constant within a process.
__device__ __forceinline__ bool q38_gdn_exact_exp_on() {
    return q38_gdn_exact_exp_flag != 0;
}

// WARODECODE_EXACT_SCALAR_FMA (default ON).
//
// The pinned CPU oracle is built with -DGGML_NATIVE=OFF. That does NOT disable
// the x86-64 SIMD/ISA extensions: ggml/CMakeLists.txt:141-171 sets INS_ENB=ON
// exactly when GGML_NATIVE is OFF, so GGML_AVX2 / GGML_FMA / GGML_F16C default
// to ON and ggml/src/ggml-cpu/CMakeLists.txt:318-332 adds
// "-mfma -mavx2 -mf16c -DGGML_FMA -DGGML_AVX2". The canonical oracle image
// (issue1_artifacts/canonical_cpu_oracle_runtime.json) records
// "c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0" and the exact CMake flags
// (-DGGML_NATIVE=OFF), so the oracle has AVX2 and FMA available and GCC's
// default -ffp-contract=fast applies (no -ffp-contract flag is added anywhere).
//
// What that means per kernel, verified against the pinned source and against
// the real oracle trace (issue1_artifacts/cpu_pos1_l0.bin):
//
//   * conv: ggml_compute_forward_ssm_conv_f32 (ggml-cpu/ops.cpp:9703) is a
//     scalar chain "sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];" over i0 ascending.
//     For the model's 4-tap kernel the oracle does NOT fuse it: gcc 13.3 and
//     gcc 16.2 at -O3 -mavx2 -mfma both compile the chain into a 4-wide vmulps
//     plus four ordered vaddss, and the same function compiled with
//     -ffp-contract=off produces bit-identical output over 1M random rows (the
//     vfmadd231ss in the object only covers a non-multiple-of-4 tail). The
//     non-fused ascending chain is therefore the oracle arithmetic and is what
//     q38_gdn_conv_accumulate() selects; __fmaf_rn is the OFF/legacy path.
//
//   * AR / scan: the model builds GGML_OP_GATED_DELTA_NET
//     (ggml_compute_forward_gated_delta_net_one_chunk, ops.cpp:10895), not
//     ssm_scan. Its dots are ggml_vec_dot_f32 (vec.cpp:11) - explicit AVX2
//     intrinsics: four 8-lane accumulators, _mm256_fmadd_ps, and the
//     GGML_F32x8_REDUCE pairwise/hadd reduction - and its outer product is
//     ggml_vec_mad_f32 (vec.h:319, _mm256_fmadd_ps). Replaying that order on
//     the oracle trace reproduces all 786432 state values and 6144 attention
//     values bit-exactly (0 / 792576), while the scalar non-fused chain
//     disagrees on 785798 / 792576 values. The AVX2-order emulation in
//     q38_gdn_ar_kernel() / q38_gdn_ar_f32_kernel() is therefore kept as is;
//     only the individual operations are pinned to explicit rounding modes.
//
//   * l2 norm: the model normalises q/k with build_gdn_l2_norm
//     (src/models/models.h:14), i.e. ggml_scale(ggml_rms_norm(x, eps/n),
//     1.0f/sqrtf(n)) - NOT ggml_l2_norm. ggml_compute_forward_rms_norm_f32
//     (ops.cpp:3924) accumulates "sum += (ggml_float)(x[i00]*x[i00])" in
//     DOUBLE, ascending, then uses mean = sum/ne00 and scale = 1.0f/sqrtf(...).
//     ggml_scale is a second, separate multiply (ggml_vec_scale_f32). The
//     legacy warp kernel used a float FMA tree and rsqrtf() and omitted the
//     1/sqrtf(n) factor: 3008 of 4096 values of the same oracle trace differ.
//     This switch selects the oracle formula.
//
// The same double-accumulator / 1.0f/sqrtf divergence exists in the gated
// RMSNorm (build_norm_gated -> build_norm(LLM_NORM_RMS) -> ggml_rms_norm +
// ggml_mul, src/models/qwen4exp.cpp:1217), so that kernel follows the same
// switch.
//
// WARODECODE_EXACT_SCALAR_FMA=0 restores the previous warp arithmetic
// (rsqrtf + float tree norms, plain fmaf everywhere) for A/B runs.
//
// Like WARODECODE_EXACT_EXP the switch is published into a __device__ symbol
// instead of a kernel parameter because these kernels are also launched from
// qwen38_forward.cuh and tests/qwen38_gdn_cuda_test.cu, whose signatures this
// change must not alter.
__device__ int q38_gdn_exact_scalar_fma_flag = 1;

inline bool q38_gdn_exact_scalar_fma_enabled() {
    static const bool enabled = [] {
        const int value = q38_gdn_env_flag("WARODECODE_EXACT_SCALAR_FMA", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_gdn_exact_scalar_fma_flag, &value, sizeof(value));
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("publish WARODECODE_EXACT_SCALAR_FMA failed: ") + cudaGetErrorString(e));
        }
        return value != 0;
    }();
    return enabled;
}

__device__ __forceinline__ bool q38_gdn_scalar_fma_on() {
    return q38_gdn_exact_scalar_fma_flag != 0;
}

// WARODECODE_EXACT_FIRST_TOKEN (default ON).
//
// Position 0 is the only position the model evaluates with an all-zero
// recurrent state, and it reaches the GDN through
// ggml_compute_forward_gated_delta_net_one_chunk() (ggml-cpu/ops.cpp:10895)
// with n_tokens == 1. With state == 0 that function reduces to
//
//   delta[j]     = (v_d[j] - dot(row j of state 0, k_d)) * beta   // sum == +0.0f
//   row j        = ggml_vec_mad_f32(S_v, &s_out[j*S_v], k_d, delta[j])   // -> fmadd(k[i], delta[j], 0.0f)
//   attn[j]      = ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j*S_v], 0, q_d, 0, 1) * (1.0f/sqrtf((float) S_v))
//
// Both dots are ggml_vec_dot_f32 (ggml-cpu/vec.cpp:11) with the AVX2+FMA macro
// set the pinned CPU oracle compiles with (ggml/CMakeLists.txt:141-171 sets
// INS_ENB=ON whenever GGML_NATIVE is OFF, ggml/src/ggml-cpu/CMakeLists.txt
// adds -mavx2 -mfma): GGML_F32_STEP / GGML_F32_EPR == 32 / 8, i.e.
// GGML_F32_ARR == 4 accumulators of 8 lanes, _mm256_fmadd_ps over 32-element
// blocks, then the GGML_F32x8_REDUCE tree
// (sum[0]+=sum[2], sum[1]+=sum[3], sum[0]+=sum[1], then
//  t[k] = s[k]+s[k+4], h0 = t0+t1, h1 = t2+t3, res = h0+h1)
// followed by "sumf += x[i]*y[i]" over the n & ~31 tail. q38_gdn_reduce_avx2()
// is that tree.
//
// The legacy q38_gdn_core_first_f32_kernel() split each dot across the 32 lanes
// of the warp (lane l accumulates i = l, l+32, ...) and reduced with a
// shfl_down tree (offsets 16, 8, 4, 2, 1). The per-lane chains are the same
// terms, but the reduction pairs lanes (l, l+16) where AVX2 pairs (l, 8+l) and
// (16+l, 24+l), so the two orders disagree on a large fraction of random rows.
// The same kernel also fused the outer product into the dot as a rounded
// multiply (k[i]*delta) instead of the oracle's fmadd(k[i], delta, 0.0f); the
// two agree for every non-zero operand but not on the sign of a zero result.
//
// This switch pins the whole position-0 path - the qkv/ssm_out projections are
// untouched, they are already bit exact - to the oracle arithmetic:
//   * q38_gdn_dense_f32_gemv_avx2_kernel()  replaces the warp-order GEMV used
//     for the layer-0 ssm_beta / ssm_alpha F32 debug weights (ggml keeps an F32
//     weight's activation in F32, so the projection is one ggml_vec_dot_f32 per
//     row).
//   * q38_gdn_core_first_f32_avx2_kernel()  replaces the warp-shuffle dot.
//   * q38_gdn_conv_first_f32_kernel()       uses the oracle's ascending
//     non-fused conv chain, whose first d_conv-1 taps read the zero history.
//   * the l2 norm / gate / gated-RMSNorm kernels are asked for their exact
//     variants, so the path can be verified on its own even when
//     WARODECODE_EXACT_SCALAR_FMA or WARODECODE_EXACT_EXP are turned off.
//
// WARODECODE_EXACT_FIRST_TOKEN=0 restores the previous first-token kernels
// (warp-shuffle dots and the fused multiply) for A/B runs.
//
// Like the two switches above this is published into a __device__ symbol
// because q38_gdn_first_token_exact_on() is read inside kernels whose
// signatures the callers in qwen38_forward.cuh must not see change.
__device__ int q38_gdn_first_token_exact_flag = 1;

inline bool q38_gdn_first_token_exact_enabled() {
    static const bool enabled = [] {
        const int value = q38_gdn_env_flag("WARODECODE_EXACT_FIRST_TOKEN", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_gdn_first_token_exact_flag, &value, sizeof(value));
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("publish WARODECODE_EXACT_FIRST_TOKEN failed: ") + cudaGetErrorString(e));
        }
        return value != 0;
    }();
    return enabled;
}

__device__ __forceinline__ bool q38_gdn_first_token_exact_on() {
    return q38_gdn_first_token_exact_flag != 0;
}

// Shared tail of both normalisation kernels: the oracle's
//   ggml_float sum = 0.0; for (i00) sum += (ggml_float)(x[i00]*x[i00]);
//   mean = sum/ne00; scale = 1.0f/sqrtf(mean + eps);
// evaluated by one thread (the reduction is a strictly ascending double chain,
// so it cannot be split across threads) and broadcast through shared memory.
// The elementwise multiply stays parallel and matches ggml_vec_scale_f32.
__device__ __forceinline__ void q38_gdn_oracle_norm_scale(
        const float * __restrict__ x,
        int base,
        int head_dim,
        float eps,
        float * __restrict__ scale_smem) {
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            const float value = x[base + i];
            // (ggml_float)(x[i00]*x[i00]): the product is rounded to float
            // first, then widened (exactly) and accumulated in double.
            sum = __dadd_rn(sum, static_cast<double>(__fmul_rn(value, value)));
        }
        const float mean = __double2float_rn(__ddiv_rn(sum, static_cast<double>(head_dim)));
        const float eps_scaled = __fdiv_rn(eps, static_cast<float>(head_dim));
        scale_smem[0] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps_scaled)));
        scale_smem[1] = __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim)));
    }
    __syncthreads();
}

// glibc's expf, ported from sysdeps/ieee754/flt-32/e_expf.c. The table is
// e_exp2f_data.c's tab[] (tab[i] = uint(2^(i/N)) - (i << (52 - BITS)), N = 32)
// and the constants are that file's shift / invln2_scaled / poly_scaled, spelled
// out with the same per-step rounding as the C source. Every arithmetic step
// uses an explicit rounding-mode intrinsic so that --use_fast_math cannot fuse
// the double multiply-adds.
constexpr int Q38_GDN_EXPF_N = 32;

__device__ __constant__ uint64_t k_q38_gdn_expf_tab[Q38_GDN_EXPF_N] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};

__device__ __forceinline__ float q38_gdn_expf_glibc(float x) {
    constexpr double shift = 0x1.8p+52;
    constexpr double invln2n = 0x1.71547652b82fep+0 * Q38_GDN_EXPF_N;
    constexpr double c0 = 0x1.c6af84b912394p-5 / Q38_GDN_EXPF_N / Q38_GDN_EXPF_N / Q38_GDN_EXPF_N;
    constexpr double c1 = 0x1.ebfce50fac4f3p-3 / Q38_GDN_EXPF_N / Q38_GDN_EXPF_N;
    constexpr double c2 = 0x1.62e42ff0c52d6p-1 / Q38_GDN_EXPF_N;

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
    uint64_t t = k_q38_gdn_expf_tab[static_cast<int>(ki % Q38_GDN_EXPF_N)];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(static_cast<long long>(t));
    const double p = __dadd_rn(__dmul_rn(c0, r), c1);
    const double r2 = __dmul_rn(r, r);
    double y = __dadd_rn(__dmul_rn(c2, r), 1.0);
    y = __dadd_rn(__dmul_rn(p, r2), y);
    y = __dmul_rn(y, s);
    return static_cast<float>(y);
}

// glibc's logf, ported from sysdeps/ieee754/flt-32/e_logf.c. The two tables are
// e_logf_data.c's tab[] (invc / logc per subinterval of [2^-1, 2^1), 16 entries)
// split into two arrays.
__device__ __constant__ double k_q38_gdn_logf_invc[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0, 0x1.3c995b0b80385p+0,
    0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0, 0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0,
    0x1.0953f419900a7p+0, 0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1, 0x1.767dcf5534862p-1,
};

__device__ __constant__ double k_q38_gdn_logf_logc[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2, -0x1.b31d8a68224e9p-3,
    -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,   -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4,
    -0x1.252f438e10c1ep-5,  0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,  0x1.4043057b6ee09p-2,
};

__device__ __forceinline__ float q38_gdn_logf_glibc(float x) {
    constexpr double ln2 = 0x1.62e42fefa39efp-1;
    constexpr double a0 = -0x1.00ea348b88334p-2;
    constexpr double a1 =  0x1.5575b0be00b6ap-2;
    constexpr double a2 = -0x1.ffffef20a4123p-2;

    uint32_t ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;                    // log(1) == +0
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {   // x < 2^-126, inf or nan
        if (ix * 2u == 0u) return -INFINITY;
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) != 0u || ix * 2u >= 0xff000000u) {
            return __fdiv_rn(x - x, x - x);                // nan
        }
        ix = __float_as_uint(x * 0x1p23f);
        ix -= 23u << 23;
    }

    const uint32_t tmp = ix - 0x3f330000u;                 // OFF
    const int i = static_cast<int>((tmp >> (23 - 4)) % 16u);
    const int k = static_cast<int>(tmp) >> 23;             // arithmetic shift
    const uint32_t iz = ix - (tmp & 0xff800000u);
    const double invc = k_q38_gdn_logf_invc[i];
    const double logc = k_q38_gdn_logf_logc[i];
    const double z = static_cast<double>(__uint_as_float(iz));

    const double r = __dsub_rn(__dmul_rn(z, invc), 1.0);
    const double y0 = __dadd_rn(logc, __dmul_rn(static_cast<double>(k), ln2));
    const double r2 = __dmul_rn(r, r);
    double y = __dadd_rn(__dmul_rn(a1, r), a2);
    y = __dadd_rn(__dmul_rn(a0, r2), y);
    y = __dadd_rn(__dmul_rn(y, r2), __dadd_rn(y0, r));
    return static_cast<float>(y);
}

// ggml_silu: x / (1 + exp_avx2(-x)), one IEEE add and one IEEE divide. ggml's
// AVX2 loop covers every element of the conv output because the row length
// (conv_dim) is a multiple of 8 for this model, so its scalar expf() tail never
// runs; a non-multiple would need the tail handled with expf() instead.
__device__ __forceinline__ float q38_gdn_silu_f32(float x, bool exact_exp) {
    if (!exact_exp) return x / (1.0f + expf(-x));
    const float neg_x = __fsub_rn(0.0f, x);
    const float exp_neg_x = q38_hc_ggml_avx2_exp_f32(neg_x);
    return __fdiv_rn(x, __fadd_rn(1.0f, exp_neg_x));
}

// op_sigmoid(): 1 / (1 + expf(-x)).
__device__ __forceinline__ float q38_gdn_sigmoid_f32(float x, bool exact_exp) {
    if (!exact_exp) return 1.0f / (1.0f + expf(-x));
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, q38_gdn_expf_glibc(-x)));
}

// expf() as called by ggml's GDN scan for exp(g).
__device__ __forceinline__ float q38_gdn_gate_exp_f32(float x, bool exact_exp) {
    return exact_exp ? q38_gdn_expf_glibc(x) : expf(x);
}

struct Qwen38GdnWeights {
    GpuMatrix qkv;
    GpuMatrix z;
    // F16 mirrors of ssm_beta / ssm_alpha: the legacy path of an F16/BF16
    // model, and the last-resort fallback when no F32 copy is available.
    GpuMatrix beta;
    GpuMatrix alpha;
    // F32 device copies of ssm_beta / ssm_alpha when the GGUF stores them as
    // F32 (the pinned model). ggml keeps such weights at full precision and
    // leaves the activation in F32 (vec_dot_type == F32).
    Qwen38GdnF32Matrix beta_f32;
    Qwen38GdnF32Matrix alpha_f32;
    DeviceHalfArray conv;
    DeviceHalfArray dt;
    DeviceHalfArray a;
    DeviceHalfArray norm;
    GpuMatrix out;
    // F32 device copies of the F32 parameter tensors. The pinned GGUF stores
    // ssm_conv1d / ssm_dt / ssm_a / ssm_norm as F32 and ggml runs every one of
    // them in F32, so the F16 copies above are only the legacy fallback.
    Qwen38GdnF32Matrix conv_f32;
    DeviceFloatArray dt_f32;
    DeviceFloatArray a_f32;
    DeviceFloatArray norm_f32;
    std::vector<float> debug_beta_f32;
    std::vector<float> debug_alpha_f32;
    std::vector<float> debug_conv_f32;
    std::vector<float> debug_dt_f32;
    std::vector<float> debug_a_f32;
    std::vector<float> debug_norm_f32;
};

inline Qwen38GdnWeights load_qwen38_gdn_weights(
        const GgufFile & g,
        int il,
        const Qwen38Layout & q) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;

    const TensorInfo & qkv = g.tensor(p + "attn_qkv.weight");
    const TensorInfo & z = g.tensor(p + "attn_gate.weight");
    const TensorInfo & beta = g.tensor(p + "ssm_beta.weight");
    const TensorInfo & alpha = g.tensor(p + "ssm_alpha.weight");
    const TensorInfo & conv = g.tensor(p + "ssm_conv1d.weight");
    const TensorInfo & dt = g.tensor(p + "ssm_dt.bias");
    const TensorInfo & a = g.tensor(p + "ssm_a");
    const TensorInfo & norm = g.tensor(p + "ssm_norm.weight");
    const TensorInfo & out_t = g.tensor(p + "ssm_out.weight");

    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("GDN ") + what + " shape mismatch: " + t.name);
        }
    };
    check2(qkv, conv_dim, q.hidden, "qkv");
    check2(z, value_dim, q.hidden, "gate");
    check2(beta, q.ssm_dt_rank, q.hidden, "beta");
    check2(alpha, q.ssm_dt_rank, q.hidden, "alpha");
    check2(out_t, q.hidden, value_dim, "output");

    if (conv.dims.size() != 2 ||
        static_cast<int>(conv.dims[0]) != q.ssm_conv_kernel ||
        static_cast<int>(conv.dims[1]) != conv_dim) {
        throw std::runtime_error("GDN conv shape mismatch: " + conv.name);
    }
    if (static_cast<int>(dt.elements()) != q.ssm_dt_rank ||
        static_cast<int>(a.elements()) != q.ssm_dt_rank ||
        static_cast<int>(norm.elements()) != q.ssm_state_size) {
        throw std::runtime_error("GDN vector shape mismatch");
    }

    Qwen38GdnWeights w;
    w.qkv = upload_matrix(qkv);
    w.z = upload_matrix(z);
    w.beta = upload_matrix(beta);
    w.alpha = upload_matrix(alpha);
    // Keep the exact F32 parameter weights as well: ggml's F32 trait has
    // vec_dot_type == F32, so the beta/alpha projections must not round their
    // weights to F16. upload_matrix() above still provides the F16 fallback.
    if (beta.type == GgmlType::F32) w.beta_f32 = upload_f32_matrix(beta);
    if (alpha.type == GgmlType::F32) w.alpha_f32 = upload_f32_matrix(alpha);
    w.conv = upload_half_array(conv);
    w.dt = upload_half_array(dt);
    w.a = upload_half_array(a);
    w.norm = upload_half_array(norm);
    w.out = upload_matrix(out_t);
    // Keep the exact F32 parameter bytes when the GGUF stores them as F32: a
    // conv weight or a norm scale rounded to F16 shifts every position.
    if (conv.type == GgmlType::F32) w.conv_f32 = upload_f32_matrix(conv);
    if (dt.type == GgmlType::F32) w.dt_f32 = upload_float_array(dt);
    if (a.type == GgmlType::F32) w.a_f32 = upload_float_array(a);
    if (norm.type == GgmlType::F32) w.norm_f32 = upload_float_array(norm);
    if (il == 0) {
        const TensorInfo * f32_tensors[] = {&beta, &alpha, &conv, &dt, &a, &norm};
        for (const TensorInfo * t : f32_tensors) {
            if (t->type != GgmlType::F32) {
                throw std::runtime_error("layer-0 GDN F32 debug requires F32 parameter tensors");
            }
        }
        auto copy_f32 = [](const TensorInfo & t) {
            const float * values = reinterpret_cast<const float *>(t.data);
            return std::vector<float>(values, values + t.elements());
        };
        w.debug_beta_f32 = copy_f32(beta);
        w.debug_alpha_f32 = copy_f32(alpha);
        w.debug_conv_f32 = copy_f32(conv);
        w.debug_dt_f32 = copy_f32(dt);
        w.debug_a_f32 = copy_f32(a);
        w.debug_norm_f32 = copy_f32(norm);
    }
    return w;
}

class Qwen38GdnState {
public:
    Qwen38GdnState() = default;
    Qwen38GdnState(const Qwen38GdnState &) = delete;
    Qwen38GdnState & operator=(const Qwen38GdnState &) = delete;
    ~Qwen38GdnState() { reset(); }

    // f32_mid selects the WARODECODE_GDN_F32_MID layout: conv_state_f32()
    // becomes the F32 convolution history and conv_state() its F16 mirror.
    void initialize(const Qwen38Layout & q, bool f32_mid = q38_gdn_f32_mid_enabled()) {
        if (state_) throw std::runtime_error("GDN state already initialized");
        state_dim_ = q.ssm_state_size;
        value_heads_ = q.ssm_dt_rank;
        conv_kernel_ = q.ssm_conv_kernel;
        conv_channels_ = q.ssm_state_size * q.ssm_group_count * 2
                       + q.ssm_state_size * q.ssm_dt_rank;
        f32_mid_ = f32_mid;

        const size_t state_n = static_cast<size_t>(state_dim_) * state_dim_ * value_heads_;
        const size_t conv_n = static_cast<size_t>(conv_kernel_ - 1) * conv_channels_;

        cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&state_), state_n * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc GDN state failed: ") + cudaGetErrorString(e));
        e = cudaMalloc(reinterpret_cast<void **>(&conv_state_), conv_n * sizeof(__half));
        if (e != cudaSuccess) {
            reset();
            throw std::runtime_error(std::string("cudaMalloc GDN conv state failed: ") + cudaGetErrorString(e));
        }
        cudaMemset(state_, 0, state_n * sizeof(float));
        cudaMemset(conv_state_, 0, conv_n * sizeof(__half));

        if (f32_mid_) {
            // ggml keeps the convolution history in F32. The F16 buffer above
            // stays allocated as the mirror the legacy accessors and the
            // layer*.gdn.* traces read.
            const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&conv_state_f32_), conv_n * sizeof(float));
            if (a == cudaSuccess) {
                cudaMemset(conv_state_f32_, 0, conv_n * sizeof(float));
            } else {
                // Out of device memory is not worth failing the model over:
                // drop back to the F16 history.
                std::fprintf(stderr,
                    "WarpDecode: GDN F32 convolution state unavailable (%s); falling back to the F16 path\n",
                    cudaGetErrorString(a));
                f32_mid_ = false;
            }
        }
    }

    void clear(cudaStream_t stream) {
        if (!state_) return;
        const size_t state_n = static_cast<size_t>(state_dim_) * state_dim_ * value_heads_;
        const size_t conv_n = static_cast<size_t>(conv_kernel_ - 1) * conv_channels_;
        cudaMemsetAsync(state_, 0, state_n * sizeof(float), stream);
        cudaMemsetAsync(conv_state_, 0, conv_n * sizeof(__half), stream);
        if (conv_state_f32_) cudaMemsetAsync(conv_state_f32_, 0, conv_n * sizeof(float), stream);
    }

    float * state() const { return state_; }
    // Legacy accessor: the F16 convolution history. With WARODECODE_GDN_F32_MID
    // this is the mirror of conv_state_f32() and only refreshed after each
    // update, so external writers can still reach the state through it (the
    // F32 history adopts changed elements at the start of the next token).
    __half * conv_state() const { return conv_state_; }
    // F32 convolution history, null on the legacy F16 path.
    float * conv_state_f32() const { return conv_state_f32_; }
    bool conv_state_f32_enabled() const { return conv_state_f32_ != nullptr; }
    size_t conv_state_elements() const {
        return static_cast<size_t>(conv_kernel_ - 1) * static_cast<size_t>(conv_channels_);
    }

private:
    void reset() {
        if (state_) cudaFree(state_);
        if (conv_state_) cudaFree(conv_state_);
        if (conv_state_f32_) cudaFree(conv_state_f32_);
        state_ = nullptr;
        conv_state_ = nullptr;
        conv_state_f32_ = nullptr;
    }

    int state_dim_ = 0;
    int value_heads_ = 0;
    int conv_kernel_ = 0;
    int conv_channels_ = 0;
    bool f32_mid_ = false;
    float * state_ = nullptr;
    __half * conv_state_ = nullptr;
    float * conv_state_f32_ = nullptr;
};

class Qwen38GdnScratch {
public:
    Qwen38GdnScratch() = default;
    Qwen38GdnScratch(const Qwen38GdnScratch &) = delete;
    Qwen38GdnScratch & operator=(const Qwen38GdnScratch &) = delete;
    ~Qwen38GdnScratch() { reset(); }

    // f32_mid selects the WARODECODE_GDN_F32_MID layout: the F32 intermediates
    // are allocated and the F16 slots become refreshable mirrors.
    void initialize(const Qwen38Layout & q, bool f32_mid = q38_gdn_f32_mid_enabled()) {
        if (qkv_) throw std::runtime_error("GDN scratch already initialized");
        const int key_dim = q.ssm_state_size * q.ssm_group_count;
        const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
        conv_dim_ = key_dim * 2 + value_dim;
        key_dim_ = key_dim;
        value_dim_ = value_dim;
        value_heads_ = q.ssm_dt_rank;
        hidden_ = q.hidden;
        // Activation buffers for the ggml-compatible Q8_0 path. attn_qkv /
        // attn_gate / ssm_beta / ssm_alpha all consume the same hidden state
        // (hidden elements, quantised once and reused), while ssm_out consumes
        // the gated-RMSNorm output (value_dim elements). One pair of buffers is
        // sized for the wider of the two and rounded up to a whole number of
        // 32-element Q8_0 blocks.
        const int act_widest = hidden_ > value_dim_ ? hidden_ : value_dim_;
        act_elements_ = (act_widest + 31) & ~31;

        alloc(reinterpret_cast<void **>(&qkv_), static_cast<size_t>(conv_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&conv_out_), static_cast<size_t>(conv_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&z_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&beta_), static_cast<size_t>(value_heads_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&alpha_), static_cast<size_t>(value_heads_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(value_heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&core_out_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&norm_out_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&final_out_), static_cast<size_t>(q.hidden) * sizeof(__half));

        f32_mid_ = f32_mid;
        if (f32_mid_) {
            // F32 intermediates (WARODECODE_GDN_F32_MID). Sized exactly like the
            // F16 slots above; the F16 slots stay allocated because the legacy
            // __half* API, the layer-14 replay path and the layer*.gdn.* traces
            // still read them.
            try {
                alloc(reinterpret_cast<void **>(&qkv_f32_), static_cast<size_t>(conv_dim_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&conv_out_f32_), static_cast<size_t>(conv_dim_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&z_f32_), static_cast<size_t>(value_dim_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&beta_f32_), static_cast<size_t>(value_heads_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&alpha_f32_), static_cast<size_t>(value_heads_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&core_out_f32_), static_cast<size_t>(value_dim_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&norm_out_f32_), static_cast<size_t>(value_dim_) * sizeof(float));
                alloc(reinterpret_cast<void **>(&final_out_f32_), static_cast<size_t>(hidden_) * sizeof(float));
            } catch (const std::exception & error) {
                reset_f32();
                f32_mid_ = false;
                std::fprintf(stderr,
                    "WarpDecode: GDN F32 intermediate buffers unavailable (%s); falling back to the F16 path\n",
                    error.what());
            }
        }

        // Exact Q8_0 projection scratch (see run_qwen38_gdn).
        alloc(reinterpret_cast<void **>(&act_f32_), static_cast<size_t>(act_elements_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&act_q8_), static_cast<size_t>(act_elements_) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&act_scales_), static_cast<size_t>(act_elements_ / 32) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gemv_f32_), static_cast<size_t>(conv_dim_) * sizeof(float));
    }

    __half * qkv() const { return qkv_; }
    __half * conv_out() const { return conv_out_; }
    __half * z() const { return z_; }
    __half * beta() const { return beta_; }
    __half * alpha() const { return alpha_; }
    float * gate() const { return gate_; }
    __half * core_out() const { return core_out_; }
    __half * norm_out() const { return norm_out_; }
    __half * final_out() const { return final_out_; }
    // F32 intermediates (WARODECODE_GDN_F32_MID); null on the legacy F16 path.
    float * qkv_f32() const { return qkv_f32_; }
    float * conv_out_f32() const { return conv_out_f32_; }
    float * z_f32() const { return z_f32_; }
    float * beta_f32() const { return beta_f32_; }
    float * alpha_f32() const { return alpha_f32_; }
    float * core_out_f32() const { return core_out_f32_; }
    float * norm_out_f32() const { return norm_out_f32_; }
    float * final_out_f32() const { return final_out_f32_; }
    bool f32_mid() const { return f32_mid_; }
    // True when every F32 intermediate slot is usable.
    bool f32_ready() const {
        return f32_mid_ && qkv_f32_ != nullptr && conv_out_f32_ != nullptr &&
               z_f32_ != nullptr && beta_f32_ != nullptr && alpha_f32_ != nullptr &&
               core_out_f32_ != nullptr && norm_out_f32_ != nullptr && final_out_f32_ != nullptr;
    }
    float * act_f32() const { return act_f32_; }
    int8_t * act_q8() const { return act_q8_; }
    __half * act_scales() const { return act_scales_; }
    float * gemv_f32() const { return gemv_f32_; }
    int act_elements() const { return act_elements_; }
    // gemv_f32_ is sized for the widest projection (conv_dim), which is also
    // the upper bound for the F32 parameter projections.
    int gemv_elements() const { return conv_dim_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc GDN scratch failed: ") + cudaGetErrorString(e));
    }

    void reset_f32() {
        if (qkv_f32_) cudaFree(qkv_f32_);
        if (conv_out_f32_) cudaFree(conv_out_f32_);
        if (z_f32_) cudaFree(z_f32_);
        if (beta_f32_) cudaFree(beta_f32_);
        if (alpha_f32_) cudaFree(alpha_f32_);
        if (core_out_f32_) cudaFree(core_out_f32_);
        if (norm_out_f32_) cudaFree(norm_out_f32_);
        if (final_out_f32_) cudaFree(final_out_f32_);
        qkv_f32_ = conv_out_f32_ = z_f32_ = beta_f32_ = alpha_f32_ = nullptr;
        core_out_f32_ = norm_out_f32_ = final_out_f32_ = nullptr;
    }

    void reset() {
        reset_f32();
        if (qkv_) cudaFree(qkv_);
        if (conv_out_) cudaFree(conv_out_);
        if (z_) cudaFree(z_);
        if (beta_) cudaFree(beta_);
        if (alpha_) cudaFree(alpha_);
        if (gate_) cudaFree(gate_);
        if (core_out_) cudaFree(core_out_);
        if (norm_out_) cudaFree(norm_out_);
        if (final_out_) cudaFree(final_out_);
        if (act_f32_) cudaFree(act_f32_);
        if (act_q8_) cudaFree(act_q8_);
        if (act_scales_) cudaFree(act_scales_);
        if (gemv_f32_) cudaFree(gemv_f32_);
        qkv_ = conv_out_ = z_ = beta_ = alpha_ = core_out_ = norm_out_ = final_out_ = nullptr;
        gate_ = nullptr;
        act_f32_ = nullptr;
        act_q8_ = nullptr;
        act_scales_ = nullptr;
        gemv_f32_ = nullptr;
    }

    int conv_dim_ = 0;
    int key_dim_ = 0;
    int value_dim_ = 0;
    int value_heads_ = 0;
    __half * qkv_ = nullptr;
    __half * conv_out_ = nullptr;
    __half * z_ = nullptr;
    __half * beta_ = nullptr;
    __half * alpha_ = nullptr;
    float * gate_ = nullptr;
    __half * core_out_ = nullptr;
    __half * norm_out_ = nullptr;
    __half * final_out_ = nullptr;
    int hidden_ = 0;
    bool f32_mid_ = false;
    float * qkv_f32_ = nullptr;
    float * conv_out_f32_ = nullptr;
    float * z_f32_ = nullptr;
    float * beta_f32_ = nullptr;
    float * alpha_f32_ = nullptr;
    float * core_out_f32_ = nullptr;
    float * norm_out_f32_ = nullptr;
    float * final_out_f32_ = nullptr;
    int act_elements_ = 0;
    float * act_f32_ = nullptr;
    int8_t * act_q8_ = nullptr;
    __half * act_scales_ = nullptr;
    float * gemv_f32_ = nullptr;
};

class Qwen38GdnF32DebugScratch {
public:
    Qwen38GdnF32DebugScratch() = default;
    Qwen38GdnF32DebugScratch(const Qwen38GdnF32DebugScratch &) = delete;
    Qwen38GdnF32DebugScratch & operator=(const Qwen38GdnF32DebugScratch &) = delete;
    ~Qwen38GdnF32DebugScratch() { reset(); }

    void initialize(const Qwen38Layout & q) {
        const int key_dim = q.ssm_state_size * q.ssm_group_count;
        const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
        conv_dim_ = key_dim * 2 + value_dim;
        key_dim_ = key_dim;
        value_dim_ = value_dim;
        heads_ = q.ssm_dt_rank;
        hidden_ = q.hidden;
        alloc(reinterpret_cast<void **>(&qkv_), static_cast<size_t>(conv_dim_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&z_), static_cast<size_t>(value_dim_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&beta_), static_cast<size_t>(heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&alpha_), static_cast<size_t>(heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&conv_out_), static_cast<size_t>(conv_dim_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&core_out_), static_cast<size_t>(value_dim_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&norm_out_), static_cast<size_t>(value_dim_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&final_out_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&beta_weights_), static_cast<size_t>(heads_) * hidden_ * sizeof(float));
        alloc(reinterpret_cast<void **>(&alpha_weights_), static_cast<size_t>(heads_) * hidden_ * sizeof(float));
        alloc(reinterpret_cast<void **>(&conv_weights_), static_cast<size_t>(conv_dim_) * q.ssm_conv_kernel * sizeof(float));
        alloc(reinterpret_cast<void **>(&dt_), static_cast<size_t>(heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&a_), static_cast<size_t>(heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&norm_), static_cast<size_t>(q.ssm_state_size) * sizeof(float));
    }

    int conv_dim() const { return conv_dim_; }
    int key_dim() const { return key_dim_; }
    int value_dim() const { return value_dim_; }
    int heads() const { return heads_; }
    int hidden() const { return hidden_; }
    float * qkv() const { return qkv_; }
    float * z() const { return z_; }
    float * beta() const { return beta_; }
    float * alpha() const { return alpha_; }
    float * conv_out() const { return conv_out_; }
    float * gate() const { return gate_; }
    float * core_out() const { return core_out_; }
    float * norm_out() const { return norm_out_; }
    float * final_out() const { return final_out_; }
    float * beta_weights() const { return beta_weights_; }
    float * alpha_weights() const { return alpha_weights_; }
    float * conv_weights() const { return conv_weights_; }
    float * dt() const { return dt_; }
    float * a() const { return a_; }
    float * norm() const { return norm_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc GDN F32 debug scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (qkv_) cudaFree(qkv_);
        if (z_) cudaFree(z_);
        if (beta_) cudaFree(beta_);
        if (alpha_) cudaFree(alpha_);
        if (conv_out_) cudaFree(conv_out_);
        if (gate_) cudaFree(gate_);
        if (core_out_) cudaFree(core_out_);
        if (norm_out_) cudaFree(norm_out_);
        if (final_out_) cudaFree(final_out_);
        if (beta_weights_) cudaFree(beta_weights_);
        if (alpha_weights_) cudaFree(alpha_weights_);
        if (conv_weights_) cudaFree(conv_weights_);
        if (dt_) cudaFree(dt_);
        if (a_) cudaFree(a_);
        if (norm_) cudaFree(norm_);
        qkv_ = z_ = beta_ = alpha_ = conv_out_ = gate_ = core_out_ = norm_out_ = final_out_ = nullptr;
        beta_weights_ = alpha_weights_ = conv_weights_ = dt_ = a_ = norm_ = nullptr;
    }

    int conv_dim_ = 0;
    int key_dim_ = 0;
    int value_dim_ = 0;
    int heads_ = 0;
    int hidden_ = 0;
    float * qkv_ = nullptr;
    float * z_ = nullptr;
    float * beta_ = nullptr;
    float * alpha_ = nullptr;
    float * conv_out_ = nullptr;
    float * gate_ = nullptr;
    float * core_out_ = nullptr;
    float * norm_out_ = nullptr;
    float * final_out_ = nullptr;
    float * beta_weights_ = nullptr;
    float * alpha_weights_ = nullptr;
    float * conv_weights_ = nullptr;
    float * dt_ = nullptr;
    float * a_ = nullptr;
    float * norm_ = nullptr;
};

// ggml_compute_forward_ssm_conv_f32 (ggml-cpu/ops.cpp:9747) accumulates
// "sumf += s[i0 + i1*ncs] * c[i0 + i1*nc]" over i0 ascending. For this model's
// 4-tap kernel the oracle build (-O3 -mavx2 -mfma, gcc 13.3 / 15.2 / 16.2) does
// NOT fuse that chain: gcc emits a 4-wide vmulps followed by four ordered
// vaddss, and compiling the same function with -ffp-contract=off yields
// bit-identical output (measured over 1M rows). The vfmadd231ss in the object
// only covers a non-multiple-of-4 tail. The ascending mul/add chain below is
// therefore the oracle arithmetic; __fmaf_rn stays for
// WARODECODE_EXACT_SCALAR_FMA=0.
__device__ __forceinline__ float q38_gdn_conv_accumulate(float sum, float a, float b, bool exact) {
    return exact ? __fadd_rn(sum, __fmul_rn(a, b)) : __fmaf_rn(a, b, sum);
}

__global__ void q38_gdn_conv_kernel(
        const __half * __restrict__ current,
        __half * __restrict__ history,
        const __half * __restrict__ kernel,
        __half * __restrict__ out,
        int channels,
        int kernel_size) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const bool exact_exp = q38_gdn_exact_exp_on();
    float sum = 0.0f;
    const int hist = kernel_size - 1;
    const size_t hbase = static_cast<size_t>(ch) * hist;
    const size_t kbase = static_cast<size_t>(ch) * kernel_size;

    const bool exact = q38_gdn_scalar_fma_on();
    for (int t = 0; t < hist; ++t) {
        sum = q38_gdn_conv_accumulate(sum, __half2float(history[hbase + t]),
                                      __half2float(kernel[kbase + t]), exact);
    }
    sum = q38_gdn_conv_accumulate(sum, __half2float(current[ch]),
                                  __half2float(kernel[kbase + hist]), exact);

    for (int t = 0; t + 1 < hist; ++t) history[hbase + t] = history[hbase + t + 1];
    if (hist > 0) history[hbase + hist - 1] = current[ch];

    out[ch] = __float2half(q38_gdn_silu_f32(sum, exact_exp));
}

__global__ void q38_l2_norm_heads_kernel(
        __half * x,
        int head_dim,
        int n_heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int base = h * head_dim;
    if (q38_gdn_scalar_fma_on()) {
        // Same build_gdn_l2_norm formula as the F32 kernel below; the values are
        // read out of the F16 mirror, so only the elementwise multiply rounds.
        __shared__ float scale_smem[2];
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int i = 0; i < head_dim; ++i) {
                const float value = __half2float(x[base + i]);
                sum = __dadd_rn(sum, static_cast<double>(__fmul_rn(value, value)));
            }
            const float mean = __double2float_rn(__ddiv_rn(sum, static_cast<double>(head_dim)));
            const float eps_scaled = __fdiv_rn(eps, static_cast<float>(head_dim));
            scale_smem[0] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps_scaled)));
            scale_smem[1] = __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim)));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float scaled = __fmul_rn(__half2float(x[base + i]), scale_smem[0]);
            x[base + i] = __float2half(__fmul_rn(scaled, scale_smem[1]));
        }
        return;
    }
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = __half2float(x[base + i]);
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        x[base + i] = __float2half(__half2float(x[base + i]) * inv);
    }
}

__device__ __forceinline__ float q38_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

// op_softplus() (ggml-cpu/unary-ops.cpp:80):
//     (x > 20.0f) ? x : logf(1.0f + expf(x))
// The legacy q38_softplus() above is the log1pf() form used elsewhere in ggml
// (ggml_compute_softplus_f32 in the SSM scan) and has an extra -20.0f branch
// that op_softplus() does not, so it cannot be reused for the GDN alpha gate.
__device__ __forceinline__ float q38_gdn_softplus_f32(float x, bool exact_exp) {
    if (!exact_exp) return q38_softplus(x);
    if (x > 20.0f) return x;
    return q38_gdn_logf_glibc(__fadd_rn(1.0f, q38_gdn_expf_glibc(x)));
}

__global__ void q38_gdn_dense_f32_gemv_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;
    const float * row_ptr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (int i = lane; i < cols; i += 32) acc = fmaf(row_ptr[i], x[i], acc);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Tree reduction half of ggml_vec_dot_f32 on AVX2+FMA (pinned CPU oracle
// build flags). Verified bit-exact against the pinned trace over all 6144
// layer-0 GDN output dots. Callers must fill sum[4][8] in AVX2 visit order:
// base in {0,32,64,96}, stream j in 0..3, lane l in 0..7,
// i = base + j*8 + l with FMA accumulation.
__device__ __forceinline__ float q38_gdn_reduce_avx2(float sum[4][8]) {
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

// F32 weight row x F32 activation row -> one ggml_vec_dot_f32 (vec.cpp:11) in
// its own visit order, for the position-0 ssm_beta / ssm_alpha projections.
//
// ggml_compute_forward_mul_mat() hands an F32 weight a F32 activation
// (type_traits_cpu[GGML_TYPE_F32].vec_dot_type == GGML_TYPE_F32), so with a
// single decode token the projection is one ggml_vec_dot_f32 per row:
// ggml_compute_forward_mul_mat_one_chunk() calls vec_dot(ne00, ...) with no
// quantisation in between. The llamafile sgemm shortcut is not taken here:
// llamafile_sgemm() returns false for n < 2 on x86 (llamafile/sgemm.cpp:3819),
// and n is the number of src1 columns, i.e. the token count.
//
// One thread per row: 32-element blocks of four 8-lane accumulators, the
// GGML_F32x8_REDUCE tree (q38_gdn_reduce_avx2), then the n & ~31 tail.
// GGML_F32_STEP / GGML_F32_EPR are 32 / 8 for the AVX2+FMA macro set the pinned
// oracle compiles with (simd-mappings.h:587, GGML_F32_ARR == 4).
__global__ void q38_gdn_dense_f32_gemv_avx2_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float * row_ptr = w + static_cast<size_t>(row) * cols;
    float sum[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = cols & ~31;
    for (int i = 0; i < np; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int c = i + j * 8 + l;
                sum[j][l] = __fmaf_rn(row_ptr[c], x[c], sum[j][l]);
            }
        }
    }
    float acc = q38_gdn_reduce_avx2(sum);
    // ggml_vec_dot_f32 leftovers: "sumf += x[i]*y[i]" over the n & ~31 tail. The
    // oracle build does NOT contract it either (measured: 23/48 rows differ from
    // the tail as an FMA and 0/48 as a separate multiply and add), so this is the
    // non-fused chain, like q38_gdn_conv_accumulate(). Not reached for the pinned
    // model: both projections have cols 2560 and 2048, multiples of 32.
    for (int c = np; c < cols; ++c) acc = __fadd_rn(acc, __fmul_rn(row_ptr[c], x[c]));
    y[row] = acc;
}

// Task P (WARODECODE_GDN_F32_GEMV_WARP, default ON).
//
// q38_gdn_dense_f32_gemv_avx2_kernel() puts one *thread* on each row of an F32
// weight, so the ssm_beta / ssm_alpha projections (rows = ssm_dt_rank, 32-64)
// run as one or two warps with a cols-long serial FMA chain each. Measured on
// A100: 7.82 + 7.81 ms/token, i.e. 36% of the whole attention region, at about
// 12 GB/s. The loop is latency bound, not bandwidth bound - the same failure
// Task J fixed for the HC inject projection.
//
// This is the identical fix: lane 8*j + l of a warp owns the ggml_vec_dot_f32
// accumulator sum[j][l] and walks i = 0, 32, 64, ... with the same __fmaf_rn
// chain, so every accumulator sees exactly the operand sequence the one-thread
// kernel produced. The reduce half rebuilds q38_gdn_reduce_avx2() with shuffles
// in the same association order:
//
//   sum[0][l] += sum[2][l] ; sum[1][l] += sum[3][l]    -> one __shfl_xor(16) add
//   sum[0][l] += sum[1][l]                             -> lane l += lane l+8
//   t0[k] = sum[0][k] + sum[0][k+4]                    -> lane k += lane k+4
//   h0 = t0[0] + t0[1] ; h1 = t0[2] + t0[3] ; h0 + h1  -> lane 0
//
// The non-fused tail stays on lane 0 after the tree, which is where
// q38_gdn_dense_f32_gemv_avx2_kernel() adds it as well.
__global__ void q38_gdn_dense_f32_gemv_avx2_warp_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int j = lane >> 3;   // accumulator row of sum[4][8]
    const int l = lane & 7;    // accumulator column
    const float * row_ptr = w + static_cast<size_t>(row) * cols;

    float acc = 0.0f;
    // The accumulator chain must stay sequential (ggml_vec_dot_f32 order), so
    // the only headroom is memory-level parallelism: unrolling lets the loads
    // for several iterations be in flight behind the FMA chain. The loop stops
    // at the last whole 32-value block: everything past it belongs to the
    // non-fused tail below, exactly as in the one-thread kernel.
    const int np = cols & ~31;
#pragma unroll 4
    for (int i = 0; i < np; i += 32) {
        const int c = i + j * 8 + l;
        acc = __fmaf_rn(row_ptr[c], x[c], acc);
    }

    // sum[0][l] += sum[2][l] and sum[1][l] += sum[3][l] in one exchange.
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 16));
    // sum[0][l] += sum[1][l] for l = 0..7 (lanes 0..7 read lanes 8..15).
    const float s1 = __shfl_sync(0xffffffffu, acc, (lane & 7) + 8);
    if (lane < 8) acc = __fadd_rn(acc, s1);
    // t0[k] = sum[0][k] + sum[0][k+4] for k = 0..3.
    const float s4 = __shfl_sync(0xffffffffu, acc, (lane & 3) + 4);
    if (lane < 4) acc = __fadd_rn(acc, s4);

    const float t1 = __shfl_sync(0xffffffffu, acc, 1);
    const float t2 = __shfl_sync(0xffffffffu, acc, 2);
    const float t3 = __shfl_sync(0xffffffffu, acc, 3);
    if (lane == 0) {
        float out = __fadd_rn(__fadd_rn(acc, t1), __fadd_rn(t2, t3));
        for (int c = np; c < cols; ++c) {
            out = __fadd_rn(out, __fmul_rn(row_ptr[c], x[c]));
        }
        y[row] = out;
    }
}

inline bool q38_gdn_f32_gemv_warp_enabled() {
    static const bool enabled = q38_gdn_env_flag("WARODECODE_GDN_F32_GEMV_WARP", true);
    return enabled;
}

// One dispatch for both F32 orderings, so the replay path and the production
// path can never plan a different kernel. "avx2" selects the oracle order
// (WARODECODE_EXACT_SCALAR_FMA, default ON) and is what the warp kernel
// accelerates; the warp form needs a whole 32-value block, so a narrow or
// unaligned width keeps the one-thread-per-row kernel.
inline void q38_gdn_launch_dense_f32_gemv_avx2(
        const float * w,
        const float * x,
        float * y,
        int rows,
        int cols,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    if (q38_gdn_exact_scalar_fma_enabled()) {
        if (q38_gdn_f32_gemv_warp_enabled() && cols >= 32 && (cols % 32) == 0) {
            constexpr int rows_per_block = 4;
            const int blocks = (rows + rows_per_block - 1) / rows_per_block;
            q38_gdn_dense_f32_gemv_avx2_warp_kernel<<<blocks, rows_per_block * 32, 0, stream>>>(
                w, x, y, rows, cols);
            return;
        }
        q38_gdn_dense_f32_gemv_avx2_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
            w, x, y, rows, cols);
        return;
    }
    q38_gdn_dense_f32_gemv_kernel<<<(rows + warps - 1) / warps, threads, 0, stream>>>(
        w, x, y, rows, cols);
}

// Position-0 half of ggml_compute_forward_ssm_conv_f32 (ggml-cpu/ops.cpp:9703).
// The sliding window is s[i0 + i1*ncs] with ncs = d_conv - 1 + n_t = d_conv: the
// first d_conv-1 taps read the conv state, which is all zero for the first token,
// and the last tap reads the current qkv row. The oracle chain is the ascending,
// non-fused "sumf += s[...] * c[...]" (q38_gdn_conv_accumulate), so the zero
// history steps leave sumf at exactly +0.0f and the last one rounds
// current*w once - including the (-0.0f) -> (+0.0f) normalisation that an
// accumulator starting at +0.0f performs. WARODECODE_EXACT_FIRST_TOKEN=0 keeps
// the previous single multiply, which cannot observe the zero history.
__global__ void q38_gdn_conv_first_f32_kernel(
        const float * __restrict__ current,
        const float * __restrict__ kernel,
        float * __restrict__ out,
        int channels,
        int kernel_size) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const bool exact_first = q38_gdn_first_token_exact_on();
    const bool exact_exp = exact_first || q38_gdn_exact_exp_on();
    const float * taps = kernel + static_cast<size_t>(channel) * kernel_size;
    float value;
    if (exact_first) {
        value = 0.0f;
        for (int t = 0; t + 1 < kernel_size; ++t) {
            value = q38_gdn_conv_accumulate(value, 0.0f, taps[t], true);
        }
        value = q38_gdn_conv_accumulate(value, current[channel], taps[kernel_size - 1], true);
    } else {
        value = current[channel] * taps[kernel_size - 1];
    }
    out[channel] = q38_gdn_silu_f32(value, exact_exp);
}

__global__ void q38_gdn_l2_norm_f32_kernel(
        float * x,
        int head_dim,
        int heads,
        float eps) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int base = head * head_dim;
    // The position-0 switch selects the build_gdn_l2_norm arithmetic on its own,
    // so the first-token path stays exact even with WARODECODE_EXACT_SCALAR_FMA
    // turned off.
    if (q38_gdn_scalar_fma_on() || q38_gdn_first_token_exact_on()) {
        // build_gdn_l2_norm (src/models/models.h:14):
        //   ggml_scale(ggml_rms_norm(x, eps/n), 1.0f/sqrtf(n))
        // ggml_rms_norm accumulates in double and applies 1.0f/sqrtf(...);
        // ggml_scale is a second, separate multiply (ggml_vec_scale_f32).
        __shared__ float scale_smem[2];
        q38_gdn_oracle_norm_scale(x, base, head_dim, eps, scale_smem);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float scaled = __fmul_rn(x[base + i], scale_smem[0]);
            x[base + i] = __fmul_rn(scaled, scale_smem[1]);
        }
        return;
    }
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float value = x[base + i];
        sum = fmaf(value, value, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) x[base + i] *= inv;
}

// op_sigmoid() -> 1/(1+expf(-x)) and op_softplus() -> (x > 20.0f) ? x : logf(1.0f + expf(x))
// with this host's glibc (ggml-cpu/unary-ops.cpp:31,80). The position-0 switch
// selects them on its own, so the first-token path cannot be pulled off the
// oracle arithmetic by WARODECODE_EXACT_EXP.
__global__ void q38_gdn_gate_params_f32_kernel(
        float * beta,
        const float * alpha,
        const float * dt,
        const float * a,
        float * gate,
        int heads) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    const bool exact_exp = q38_gdn_exact_exp_on() || q38_gdn_first_token_exact_on();
    beta[h] = q38_gdn_sigmoid_f32(beta[h], exact_exp);
    gate[h] = q38_gdn_softplus_f32(__fadd_rn(alpha[h], dt[h]), exact_exp) * a[h];
}

__global__ void q38_gdn_core_first_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ beta,
        float * __restrict__ out,
        int head_dim,
        int key_heads,
        int value_heads) {
    const int vh = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (vh >= value_heads || col >= head_dim) return;
    // ggml_repeat tiles the key-head axis cyclically when expanding it to value heads.
    const int kh = vh % key_heads;
    const float * qh = q + kh * head_dim;
    const float * khv = k + kh * head_dim;
    const float * vhv = v + vh * head_dim;
    const float delta = beta[vh] * vhv[col];
    float attn_partial = 0.0f;
    for (int r = 0; r < head_dim / 32; ++r) {
        const int i = r * 32 + lane;
        const float state = khv[i] * delta;
        attn_partial = fmaf(state, qh[i], attn_partial);
    }
    const float attn = warp_sum(attn_partial);
    // ops.cpp:10965 "const float scale = 1.0f / sqrtf((float) S_v)": correctly
    // rounded, unlike the 2 ulp rsqrtf().
    const float qscale = q38_gdn_scalar_fma_on()
        ? __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim)))
        : rsqrtf(static_cast<float>(head_dim));
    if (lane == 0) out[vh * head_dim + col] = __fmul_rn(attn, qscale);
}

// Position-0 twin of the kernel above, for WARODECODE_EXACT_FIRST_TOKEN: one
// thread per (value head, value dimension) evaluates the oracle's
// ggml_compute_forward_gated_delta_net_one_chunk() arithmetic for an all-zero
// recurrent state, in the oracle's own visit order.
//
//   delta       = (v_d[col] - <dot of the zero state row with k_d>) * beta
//                 -> v_d[col] - (+0.0f) then * beta
//   state row i = ggml_vec_mad_f32(S_v, &s_out[col*S_v], k_d, delta):
//                 _mm256_fmadd_ps => fmadd(k_d[i], delta, 0.0f)
//   attn[col]   = ggml_vec_dot_f32(S_v, &sum, 0, &s_out[col*S_v], 0, q_d, 0, 1)
//                 * (1.0f / sqrtf((float) S_v))
//
// ggml_vec_dot_f32 (ggml-cpu/vec.cpp:11) with GGML_F32_STEP / GGML_F32_EPR ==
// 32 / 8: one accumulator chain per (stream, lane) over i == stream*8 + lane
// (mod 32), in ascending i, then GGML_F32x8_REDUCE (q38_gdn_reduce_avx2) and the
// n & ~31 tail. The two fp operations that produce the row entries are a plain
// multiply and an FMA with a zero addend: they round identically for every
// non-zero product, and the FMA additionally turns (-0.0f) into (+0.0f) exactly
// as the oracle's accumulator does.
__global__ void q38_gdn_core_first_f32_avx2_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ beta,
        float * __restrict__ out,
        int head_dim,
        int key_heads,
        int value_heads) {
    const int vh = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (vh >= value_heads || col >= head_dim) return;
    // ggml_repeat tiles the key-head axis cyclically when expanding it to value heads.
    const int kh = vh % key_heads;
    const float * qh = q + kh * head_dim;
    const float * khv = k + kh * head_dim;
    const float * vhv = v + vh * head_dim;
    // delta[j] = (v_d[j] - sum) * beta_val, with sum the dot of the zero state
    // row and k_d: ggml_vec_dot_f32 over zeros returns exactly +0.0f.
    const float delta = __fmul_rn(__fsub_rn(vhv[col], 0.0f), beta[vh]);
    float sum[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = head_dim & ~31;
    for (int i = 0; i < np; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r = i + j * 8 + l;
                const float row = __fmaf_rn(khv[r], delta, 0.0f);
                sum[j][l] = __fmaf_rn(row, qh[r], sum[j][l]);
            }
        }
    }
    float attn = q38_gdn_reduce_avx2(sum);
    for (int r = np; r < head_dim; ++r) {
        attn = __fmaf_rn(__fmaf_rn(khv[r], delta, 0.0f), qh[r], attn);
    }
    // ops.cpp:10965 "const float scale = 1.0f / sqrtf((float) S_v)".
    const float qscale = __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim)));
    out[vh * head_dim + col] = __fmul_rn(attn, qscale);
}

// Widening helper shared by the F32 kernels whose companion tensor may still
// be the legacy F16 copy (conv weight, norm scale). Keeping it a template
// parameter avoids duplicating the arithmetic for the two storage types.
__device__ __forceinline__ float q38_gdn_as_f32(float v) { return v; }
__device__ __forceinline__ float q38_gdn_as_f32(__half v) { return __half2float(v); }

// x, z and out are F32; the norm scale is either the F32 weight or its legacy
// F16 copy, so it stays a template parameter.
template <typename N>
__global__ void q38_gdn_gated_rmsnorm_f32_kernel(
        const float * __restrict__ x,
        const N * __restrict__ norm,
        const float * __restrict__ z,
        float * __restrict__ out,
        int head_dim,
        int heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= heads) return;
    const bool exact = q38_gdn_scalar_fma_on() || q38_gdn_first_token_exact_on();
    const bool exact_exp = q38_gdn_exact_exp_on() || exact;
    const int base = h * head_dim;
    if (exact) {
        // build_norm_gated (src/models/qwen4exp.cpp:719):
        //   ggml_mul(ggml_mul(ggml_rms_norm(x, eps), weights), ggml_sigmoid(z))
        // ggml_rms_norm (ops.cpp:3924) accumulates in double and applies
        // 1.0f/sqrtf(mean + eps). qwen4exp uses a sigmoid output gate, not silu.
        __shared__ float scale_smem[1];
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int i = 0; i < head_dim; ++i) {
                const float value = x[base + i];
                sum = __dadd_rn(sum, static_cast<double>(__fmul_rn(value, value)));
            }
            const float mean = __double2float_rn(__ddiv_rn(sum, static_cast<double>(head_dim)));
            scale_smem[0] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const int idx = base + i;
            const float gate = q38_gdn_sigmoid_f32(z[idx], exact_exp);
            const float normalized = __fmul_rn(x[idx], scale_smem[0]);
            const float weighted = __fmul_rn(normalized, q38_gdn_as_f32(norm[i]));
            out[idx] = __fmul_rn(weighted, gate);
        }
        return;
    }
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float value = x[base + i];
        sum = fmaf(value, value, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const int idx = base + i;
        const float gate = q38_gdn_sigmoid_f32(z[idx], exact_exp);
        out[idx] = x[idx] * inv * q38_gdn_as_f32(norm[i]) * gate;
    }
}

inline float * run_qwen38_gdn_f32_first_token(
        const float * input,
        const Qwen38GdnWeights & w,
        Qwen38GdnF32DebugScratch & scratch,
        const Qwen38Layout & q,
        float rms_eps,
        Q8_1Buffer & q8_1,
        cudaStream_t stream) {
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;
    (void)q38_gdn_exact_exp_enabled();
    // WARODECODE_EXACT_FIRST_TOKEN: pin the first-token dots, conv and gate
    // arithmetic to the oracle on this path's own switch. The l2 norm / gated
    // RMSNorm / sigmoid helpers read that switch inside their kernels, so the
    // recurrent path keeps answering to WARODECODE_EXACT_SCALAR_FMA / _EXP only.
    const bool exact = q38_gdn_first_token_exact_enabled();
    const size_t param_count = static_cast<size_t>(q.ssm_dt_rank);
    if (w.debug_beta_f32.size() != param_count * q.hidden ||
        w.debug_alpha_f32.size() != param_count * q.hidden ||
        w.debug_conv_f32.size() != static_cast<size_t>(conv_dim) * q.ssm_conv_kernel ||
        w.debug_dt_f32.size() != param_count || w.debug_a_f32.size() != param_count ||
        w.debug_norm_f32.size() != static_cast<size_t>(q.ssm_state_size) ||
        w.qkv.type != GgmlType::Q8_0 || w.z.type != GgmlType::Q8_0 ||
        w.out.type != GgmlType::Q8_0) {
        throw std::runtime_error("layer-0 GDN F32 debug weights mismatch");
    }

    auto copy_weights = [&](float * dst, const std::vector<float> & src) {
        const cudaError_t e = cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(float),
                                              cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error("copy layer-0 GDN F32 debug weights failed");
    };
    copy_weights(scratch.beta_weights(), w.debug_beta_f32);
    copy_weights(scratch.alpha_weights(), w.debug_alpha_f32);
    copy_weights(scratch.conv_weights(), w.debug_conv_f32);
    copy_weights(scratch.dt(), w.debug_dt_f32);
    copy_weights(scratch.a(), w.debug_a_f32);
    copy_weights(scratch.norm(), w.debug_norm_f32);

    launch_q8_0_gemv_q8_1_f32(w.qkv.low, input, scratch.qkv(), q8_1, stream);
    launch_q8_0_gemv_q8_1_f32(w.z.low, input, scratch.z(), q8_1, stream);

    constexpr int threads = 256;
    if (exact) {
        // ssm_beta / ssm_alpha are F32 weights, so ggml evaluates these two
        // projections as one ggml_vec_dot_f32 per row, in the AVX2 visit order.
        // Task P added the warp-parallel form of that same order behind the
        // shared dispatcher; the arithmetic is unchanged.
        q38_gdn_launch_dense_f32_gemv_avx2(
            scratch.beta_weights(), input, scratch.beta(), q.ssm_dt_rank, q.hidden, stream);
        q38_gdn_launch_dense_f32_gemv_avx2(
            scratch.alpha_weights(), input, scratch.alpha(), q.ssm_dt_rank, q.hidden, stream);
    } else {
        q38_gdn_dense_f32_gemv_kernel<<<(q.ssm_dt_rank + 7) / 8, threads, 0, stream>>>(
            scratch.beta_weights(), input, scratch.beta(), q.ssm_dt_rank, q.hidden);
        q38_gdn_dense_f32_gemv_kernel<<<(q.ssm_dt_rank + 7) / 8, threads, 0, stream>>>(
            scratch.alpha_weights(), input, scratch.alpha(), q.ssm_dt_rank, q.hidden);
    }

    q38_gdn_conv_first_f32_kernel<<<(conv_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.qkv(), scratch.conv_weights(), scratch.conv_out(), conv_dim, q.ssm_conv_kernel);
    float * q_conv = scratch.conv_out();
    float * k_conv = scratch.conv_out() + key_dim;
    float * v_conv = scratch.conv_out() + key_dim * 2;
    q38_gdn_l2_norm_f32_kernel<<<q.ssm_group_count, threads, 0, stream>>>(
        q_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);
    q38_gdn_l2_norm_f32_kernel<<<q.ssm_group_count, threads, 0, stream>>>(
        k_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);

    q38_gdn_gate_params_f32_kernel<<<(q.ssm_dt_rank + threads - 1) / threads, threads, 0, stream>>>(
        scratch.beta(), scratch.alpha(), scratch.dt(), scratch.a(), scratch.gate(), q.ssm_dt_rank);
    if (q.ssm_state_size != 128) {
        throw std::runtime_error("layer-0 GDN F32 debug currently expects state size 128");
    }
    if (exact) {
        // One thread per (value head, value dimension): the AVX2-ordered dot.
        const dim3 exact_grid(q.ssm_dt_rank, (q.ssm_state_size + threads - 1) / threads);
        q38_gdn_core_first_f32_avx2_kernel<<<exact_grid, threads, 0, stream>>>(
            q_conv, k_conv, v_conv, scratch.beta(), scratch.core_out(),
            q.ssm_state_size, q.ssm_group_count, q.ssm_dt_rank);
    } else {
        const dim3 core_grid(q.ssm_dt_rank, (q.ssm_state_size + 3) / 4);
        const dim3 core_block(32, 4);
        q38_gdn_core_first_f32_kernel<<<core_grid, core_block, 0, stream>>>(
            q_conv, k_conv, v_conv, scratch.beta(), scratch.core_out(),
            q.ssm_state_size, q.ssm_group_count, q.ssm_dt_rank);
    }
    q38_gdn_gated_rmsnorm_f32_kernel<<<q.ssm_dt_rank, threads, 0, stream>>>(
        scratch.core_out(), scratch.norm(), scratch.z(), scratch.norm_out(),
        q.ssm_state_size, q.ssm_dt_rank, rms_eps);
    launch_q8_0_gemv_q8_1_f32(w.out.low, scratch.norm_out(), scratch.final_out(), q8_1, stream);
    return scratch.final_out();
}

__global__ void q38_gdn_gate_params_kernel(
        __half * beta,
        const __half * alpha,
        const __half * dt,
        const __half * a,
        float * gate,
        int n_heads) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    const bool exact_exp = q38_gdn_exact_exp_on();
    const float b = __half2float(beta[h]);
    beta[h] = __float2half(q38_gdn_sigmoid_f32(b, exact_exp));
    const float av = __half2float(alpha[h]) + __half2float(dt[h]);
    gate[h] = q38_gdn_softplus_f32(av, exact_exp) * __half2float(a[h]);
}

__global__ void q38_gdn_ar_kernel(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        const float * __restrict__ gate,
        const __half * __restrict__ beta,
        float * __restrict__ state,
        __half * __restrict__ out,
        int head_dim,
        int key_heads,
        int value_heads) {
    const int vh = blockIdx.x;
    const int j = threadIdx.x;
    if (vh >= value_heads || j >= head_dim) return;

    const int kh = vh % key_heads;
    const __half * qh = q + kh * head_dim;
    const __half * khv = k + kh * head_dim;
    const __half * vhv = v + vh * head_dim;
    float * sh = state + static_cast<size_t>(vh) * head_dim * head_dim;

    const bool exact_exp = q38_gdn_exact_exp_on();
    const float decay = q38_gdn_gate_exp_f32(gate[vh], exact_exp);
    // Sequential fallback preserves the legacy behavior for geometries the
    // AVX2-order path below does not cover (it requires head_dim == 128,
    // the pinned model's ssm_state_size, verified bit-exact; other multiples
    // of 32 would also match but are kept on the legacy path).
    if (head_dim != 128) {
        float pred_seq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const size_t idx = static_cast<size_t>(i) * head_dim + j;
            const float sv = __fmul_rn(sh[idx], decay);
            sh[idx] = sv;
            pred_seq = __fmaf_rn(sv, __half2float(khv[i]), pred_seq);
        }
        const float delta_seq = __fmul_rn(__half2float(beta[vh]), __fsub_rn(__half2float(vhv[j]), pred_seq));
        const float qscale_seq = q38_gdn_scalar_fma_on() ? __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim))) : rsqrtf(static_cast<float>(head_dim));
        float y_seq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const size_t idx = static_cast<size_t>(i) * head_dim + j;
            const float sv = __fmaf_rn(__half2float(khv[i]), delta_seq, sh[idx]);
            sh[idx] = sv;
            y_seq = __fmaf_rn(sv, __half2float(qh[i]), y_seq);
        }
        out[vh * head_dim + j] = __float2half(__fmul_rn(y_seq, qscale_seq));
        return;
    }
    // Prediction dot in pinned-CPU AVX2 order (decay writes are per-element
    // and order-free, so they are fused into the AVX2 visit order).
    float psum[4][8];
#pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) psum[j_s][l] = 0.0f;
    }
#pragma unroll
    for (int base = 0; base < head_dim; base += 32) {
#pragma unroll
        for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = base + j_s * 8 + l;
                const size_t idx = static_cast<size_t>(i) * head_dim + j;
                const float sv = __fmul_rn(sh[idx], decay);
                sh[idx] = sv;
                psum[j_s][l] = __fmaf_rn(sv, __half2float(khv[i]), psum[j_s][l]);
            }
        }
    }
    const float pred = q38_gdn_reduce_avx2(psum);

    const float delta = __fmul_rn(__half2float(beta[vh]), __fsub_rn(__half2float(vhv[j]), pred));
    const float qscale = q38_gdn_scalar_fma_on() ? __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim))) : rsqrtf(static_cast<float>(head_dim));
    float ysum[4][8];
#pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) ysum[j_s][l] = 0.0f;
    }
#pragma unroll
    for (int base = 0; base < head_dim; base += 32) {
#pragma unroll
        for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = base + j_s * 8 + l;
                const size_t idx = static_cast<size_t>(i) * head_dim + j;
                const float sv = __fmaf_rn(__half2float(khv[i]), delta, sh[idx]);
                sh[idx] = sv;
                // Match the CPU reference: accumulate the raw dot first, apply the
                // 1/sqrt(head_dim) scale once afterwards instead of rounding
                // q[i]*qscale per element.
                ysum[j_s][l] = __fmaf_rn(sv, __half2float(qh[i]), ysum[j_s][l]);
            }
        }
    }
    const float y = q38_gdn_reduce_avx2(ysum);
    out[vh * head_dim + j] = __float2half(__fmul_rn(y, qscale));
}

__global__ void q38_gdn_gated_rmsnorm_kernel(
        const __half * __restrict__ x,
        const __half * __restrict__ norm,
        const __half * __restrict__ z,
        __half * __restrict__ out,
        int head_dim,
        int n_heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const bool exact_exp = q38_gdn_exact_exp_on();
    const int base = h * head_dim;
    if (q38_gdn_scalar_fma_on()) {
        // Same build_norm_gated formula as the F32 kernel above; the values are
        // read out of the F16 mirror, so only the elementwise multiplies round.
        __shared__ float scale_smem[1];
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int i = 0; i < head_dim; ++i) {
                const float value = __half2float(x[base + i]);
                sum = __dadd_rn(sum, static_cast<double>(__fmul_rn(value, value)));
            }
            const float mean = __double2float_rn(__ddiv_rn(sum, static_cast<double>(head_dim)));
            scale_smem[0] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const int idx = base + i;
            const float gate = q38_gdn_sigmoid_f32(__half2float(z[idx]), exact_exp);
            const float normalized = __fmul_rn(__half2float(x[idx]), scale_smem[0]);
            const float weighted = __fmul_rn(normalized, __half2float(norm[i]));
            out[idx] = __float2half(__fmul_rn(weighted, gate));
        }
        return;
    }
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = __half2float(x[base + i]);
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const int idx = base + i;
        const float gate = q38_gdn_sigmoid_f32(__half2float(z[idx]), exact_exp);
        out[idx] = __float2half(__half2float(x[idx]) * inv * __half2float(norm[i]) * gate);
    }
}

// ---------------------------------------------------------------------------
// F32 GDN intermediates (WARODECODE_GDN_F32_MID, default ON).
//
// ggml's GDN reference keeps the convolution history, the projection outputs,
// the gate parameters and the gated-RMSNorm chain in F32: on the CPU only the
// quantised-weight activations, the F16 KV cache and the final output round.
// The kernels above store every one of those in __half, one rounding per
// operation, which compounds along the sequence. These are the same
// computations on F32 buffers, with the AVX2-order visit order unchanged, so
// the F32 path is also the one that can be bit-compared against the pinned CPU
// oracle.
// ---------------------------------------------------------------------------

// current / history / out are F32; the conv weight keeps whichever storage the
// model provided (F32 in the pinned GGUF, F16 as the legacy fallback).
template <typename K>
__global__ void q38_gdn_conv_f32_state_kernel(
        const float * __restrict__ current,
        float * __restrict__ history,
        const K * __restrict__ kernel,
        float * __restrict__ out,
        int channels,
        int kernel_size) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const bool exact_exp = q38_gdn_exact_exp_on();
    float sum = 0.0f;
    const int hist = kernel_size - 1;
    const size_t hbase = static_cast<size_t>(ch) * hist;
    const size_t kbase = static_cast<size_t>(ch) * kernel_size;

    // Same ggml_compute_forward_ssm_conv_f32 chain as the legacy kernel above:
    // ascending taps, non-fused under WARODECODE_EXACT_SCALAR_FMA.
    const bool exact = q38_gdn_scalar_fma_on();
    for (int t = 0; t < hist; ++t) {
        sum = q38_gdn_conv_accumulate(sum, history[hbase + t],
                                      q38_gdn_as_f32(kernel[kbase + t]), exact);
    }
    sum = q38_gdn_conv_accumulate(sum, current[ch],
                                  q38_gdn_as_f32(kernel[kbase + hist]), exact);

    for (int t = 0; t + 1 < hist; ++t) history[hbase + t] = history[hbase + t + 1];
    if (hist > 0) history[hbase + hist - 1] = current[ch];

    out[ch] = q38_gdn_silu_f32(sum, exact_exp);
}

// The F32 history is the source of truth, so an external writer can only reach
// the F16 mirror (the --debug-layer9-gdn-conv-state-f32 replay path). The mirror
// is refreshed from the F32 history after every update, so an element only
// differs when something outside these kernels wrote it: adopt those.
__global__ void q38_gdn_conv_state_adopt_half_kernel(
        float * __restrict__ history,
        const __half * __restrict__ mirror,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __half value = mirror[i];
    if (__heq(value, __float2half(history[i]))) return;
    history[i] = __half2float(value);
}

// Gate parameters when dt / a are only available as the legacy F16 copies;
// beta and alpha are already F32 by the time this runs.
__global__ void q38_gdn_gate_params_f32_halfparam_kernel(
        float * beta,
        const float * alpha,
        const __half * dt,
        const __half * a,
        float * gate,
        int heads) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    const bool exact_exp = q38_gdn_exact_exp_on();
    beta[h] = q38_gdn_sigmoid_f32(beta[h], exact_exp);
    const float av = alpha[h] + __half2float(dt[h]);
    gate[h] = q38_gdn_softplus_f32(av, exact_exp) * __half2float(a[h]);
}

// F32 twin of q38_gdn_ar_kernel(): same visit order and FMA structure, F32
// inputs, and an F32 output with no per-element rounding into __half.
__global__ void q38_gdn_ar_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ gate,
        const float * __restrict__ beta,
        float * __restrict__ state,
        float * __restrict__ out,
        int head_dim,
        int key_heads,
        int value_heads) {
    const int vh = blockIdx.x;
    const int j = threadIdx.x;
    if (vh >= value_heads || j >= head_dim) return;

    const int kh = vh % key_heads;
    const float * qh = q + kh * head_dim;
    const float * khv = k + kh * head_dim;
    const float * vhv = v + vh * head_dim;
    float * sh = state + static_cast<size_t>(vh) * head_dim * head_dim;

    const bool exact_exp = q38_gdn_exact_exp_on();
    const float decay = q38_gdn_gate_exp_f32(gate[vh], exact_exp);
    // Sequential fallback for geometries the AVX2-order path below does not
    // cover (it requires head_dim == 128, the pinned model's ssm_state_size).
    if (head_dim != 128) {
        float pred_seq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const size_t idx = static_cast<size_t>(i) * head_dim + j;
            const float sv = __fmul_rn(sh[idx], decay);
            sh[idx] = sv;
            pred_seq = __fmaf_rn(sv, khv[i], pred_seq);
        }
        const float delta_seq = __fmul_rn(beta[vh], __fsub_rn(vhv[j], pred_seq));
        const float qscale_seq = q38_gdn_scalar_fma_on() ? __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim))) : rsqrtf(static_cast<float>(head_dim));
        float y_seq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const size_t idx = static_cast<size_t>(i) * head_dim + j;
            const float sv = __fmaf_rn(khv[i], delta_seq, sh[idx]);
            sh[idx] = sv;
            y_seq = __fmaf_rn(sv, qh[i], y_seq);
        }
        out[vh * head_dim + j] = __fmul_rn(y_seq, qscale_seq);
        return;
    }
    // Prediction dot in pinned-CPU AVX2 order (decay writes are per-element and
    // order-free, so they are fused into the AVX2 visit order).
    float psum[4][8];
#pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) psum[j_s][l] = 0.0f;
    }
#pragma unroll
    for (int base = 0; base < head_dim; base += 32) {
#pragma unroll
        for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = base + j_s * 8 + l;
                const size_t idx = static_cast<size_t>(i) * head_dim + j;
                const float sv = __fmul_rn(sh[idx], decay);
                sh[idx] = sv;
                psum[j_s][l] = __fmaf_rn(sv, khv[i], psum[j_s][l]);
            }
        }
    }
    const float pred = q38_gdn_reduce_avx2(psum);

    const float delta = __fmul_rn(beta[vh], __fsub_rn(vhv[j], pred));
    const float qscale = q38_gdn_scalar_fma_on() ? __fdiv_rn(1.0f, __fsqrt_rn(static_cast<float>(head_dim))) : rsqrtf(static_cast<float>(head_dim));
    float ysum[4][8];
#pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) ysum[j_s][l] = 0.0f;
    }
#pragma unroll
    for (int base = 0; base < head_dim; base += 32) {
#pragma unroll
        for (int j_s = 0; j_s < 4; ++j_s) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = base + j_s * 8 + l;
                const size_t idx = static_cast<size_t>(i) * head_dim + j;
                const float sv = __fmaf_rn(khv[i], delta, sh[idx]);
                sh[idx] = sv;
                // Match the CPU reference: accumulate the raw dot first, apply
                // the 1/sqrt(head_dim) scale once afterwards instead of
                // rounding q[i]*qscale per element.
                ysum[j_s][l] = __fmaf_rn(sv, qh[i], ysum[j_s][l]);
            }
        }
    }
    const float y = q38_gdn_reduce_avx2(ysum);
    out[vh * head_dim + j] = __fmul_rn(y, qscale);
}

// ---------------------------------------------------------------------------
// ggml-compatible Q8_0 GDN projections (WARODECODE_GDN_Q8_0, default ON).
//
// llama.cpp's CPU backend does not multiply a Q8_0 weight by an F16 activation:
// the vec_dot_type of Q8_0 is Q8_0, so the activation row is first quantised to
// 32-element Q8_0 blocks (RN scale d = amax/127, q = RN(value/d) saturated) and
// the product is then an integer dot with the per-block scales applied in F32.
// The legacy warp path multiplies a differently rounded F16 activation, which
// drifts ~1% per projection; over the 36 GDN layers that compounds and flips the
// greedy token. This path reuses the byte-verified kernels from qwen38_hc.cuh to
// reproduce the CPU order exactly.
//
// Only the five projections change. The conv / L2-norm / AR / gated-RMSNorm
// kernels stay untouched, and launch_matrix_gemv_half() remains the fallback.
// ---------------------------------------------------------------------------

// q38_gdn_env_flag() / q38_gdn_q8_0_enabled() / q38_gdn_f32_mid_enabled() live
// next to the weights they configure, above Qwen38GdnWeights.

// Every requirement that must hold before the exact Q8_0 path may run. Anything
// that fails here sends run_qwen38_gdn() back to launch_matrix_gemv_half().
//
// Only the three projections whose GGUF type really is Q8_0 are required to be
// Q8_0 here. ssm_beta / ssm_alpha are F32 in the pinned model, and ggml chooses
// the dot product from the *weight* type: an F32 weight has vec_dot_type F32
// and is multiplied by the unquantised F32 activation. Their type is therefore
// irrelevant to this predicate; q38_gdn_param_projection() picks the matching
// dot per projection.
inline bool q38_gdn_q8_0_compatible(const Qwen38GdnWeights & w, const Qwen38Layout & q) {
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;
    if (q.hidden <= 0 || value_dim <= 0 || conv_dim <= 0 || q.ssm_dt_rank <= 0) return false;
    // The quantiser works on whole 32-element blocks and the exact GEMV derives
    // blocks_per_row from cols >> 5, so every quantised length and every GEMV
    // column count must tile exactly.
    if ((q.hidden % 32) != 0 || (value_dim % 32) != 0) return false;
    const GpuMatrix * matrices[] = {&w.qkv, &w.z, &w.out};
    for (const GpuMatrix * m : matrices) {
        if (!m->lowbit || m->type != GgmlType::Q8_0 || m->low.data == nullptr) return false;
        if ((m->cols % 32) != 0) return false;
    }
    if (w.qkv.cols != q.hidden || w.qkv.rows != conv_dim) return false;
    if (w.z.cols != q.hidden || w.z.rows != value_dim) return false;
    if (w.out.cols != value_dim || w.out.rows != q.hidden) return false;
    // Parameter projections: geometry only. They are read through whichever
    // path their own type selects, so a mismatch here is the only fatal case.
    if (w.beta.cols != q.hidden || w.beta.rows != q.ssm_dt_rank) return false;
    if (w.alpha.cols != q.hidden || w.alpha.rows != q.ssm_dt_rank) return false;
    return true;
}

// True when the exact F32 device copy of a parameter projection can be used:
// the tensor really was F32 in the GGUF, its geometry matches the layout, and
// both the F32 activation copy and the F32 GEMV scratch are large enough.
inline bool q38_gdn_f32_projection_usable(
        const Qwen38GdnF32Matrix & m,
        const Qwen38GdnScratch & scratch,
        const Qwen38Layout & q) {
    return m.valid() && m.cols == q.hidden && m.rows == q.ssm_dt_rank &&
           m.cols <= scratch.act_elements() && m.rows <= scratch.gemv_elements();
}

__global__ void q38_gdn_float_to_half_kernel(
        const float * __restrict__ x,
        __half * __restrict__ y,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// F32 activation row -> F16 mirror in an owned staging slot. Only the legacy
// __half GEMV fallbacks read this row; nothing on the F32 chain does, so the
// mirror cannot feed a rounding back into the unrounded activations.
inline void q38_gdn_mirror_activation_half(
        const float * x,
        __half * dst,
        int elements,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_gdn_float_to_half_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        x, dst, elements);
}

// F32 row -> the activation scratch the projections read. The copy is
// bit-exact, never a rounding: the Q8_0 blocks and the F32 dots must see the
// caller's values, only in a buffer the kernels can read. Passing the scratch
// itself is the documented no-op case.
inline void q38_gdn_stage_activation_f32(
        const float * src,
        float * dst,
        int elements,
        cudaStream_t stream) {
    if (src == dst) return;
    const cudaError_t e = cudaMemcpyAsync(dst, src,
        static_cast<size_t>(elements) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("GDN F32 activation staging failed");
}

// x (__half) -> act_f32 -> Q8_0(act_q8, act_scales). One quantisation serves
// every projection that reads the same activation vector.
inline void q38_gdn_q8_0_quantize_activation(
        const __half * x,
        int elements,
        const Qwen38GdnScratch & scratch,
        cudaStream_t stream) {
    // Defence in depth: the compat predicate already guarantees both quantised
    // lengths tile into whole Q8_0 blocks and fit the scratch, but a mismatch
    // here would silently overflow act_q8/act_scales.
    if (elements <= 0 || (elements % 32) != 0 || elements > scratch.act_elements()) {
        throw std::runtime_error("GDN Q8_0 activation scratch mismatch");
    }
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = elements / 32;
    q38_hc_copy_half_to_f32_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        x, scratch.act_f32(), elements);
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        scratch.act_f32(), elements, scratch.act_q8(), scratch.act_scales());
}

// F32-input overload. The activation row already is F32, so no half -> F32
// conversion kernel runs: the row is staged bit-exactly and the Q8_0 blocks are
// built from its own values. This is the quantisation an F32 activation chain
// (a F32 HC mix output) has to feed the projections with.
inline void q38_gdn_q8_0_quantize_activation(
        const float * x,
        int elements,
        const Qwen38GdnScratch & scratch,
        cudaStream_t stream) {
    if (elements <= 0 || (elements % 32) != 0 || elements > scratch.act_elements()) {
        throw std::runtime_error("GDN Q8_0 activation scratch mismatch");
    }
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = elements / 32;
    q38_gdn_stage_activation_f32(x, scratch.act_f32(), elements, stream);
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        scratch.act_f32(), elements, scratch.act_q8(), scratch.act_scales());
}

// Q8_0 weight x Q8_0 activation -> F32 in ggml integer-dot order, then rounded
// into the legacy __half scratch slot so the rest of the pipeline is unchanged.
inline void q38_gdn_q8_0_gemv_to_half(
        const GpuMatrix & w,
        const Qwen38GdnScratch & scratch,
        __half * out,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_launch_exact_q8_0_gemv_f32(
        w.low.data, scratch.act_q8(), scratch.act_scales(), scratch.gemv_f32(),
        w.rows, w.cols, stream);
    q38_gdn_float_to_half_kernel<<<(w.rows + threads - 1) / threads, threads, 0, stream>>>(
        scratch.gemv_f32(), out, w.rows);
}

// F32 weight x F32 activation -> F32 in ggml's F32 dot order, then rounded into
// the legacy __half scratch slot so the rest of the pipeline is unchanged.
//
// ggml-cpu.c's type_traits_cpu[GGML_TYPE_F32] is { vec_dot = ggml_vec_dot_f32,
// vec_dot_type = GGML_TYPE_F32 }, so ggml_compute_forward_mul_mat() keeps the
// F32 activation row and calls the F32 dot: no activation quantisation and no
// dequantised-weight multiply. q38_gdn_dense_f32_gemv_kernel() is exactly that
// dot (FMA accumulation per lane + warp reduction), one row per warp, the same
// structure as the F16 fallbacks.
//
// The caller must already hold the F32 activation in scratch.act_f32(); the
// Q8_0 branch fills it once per token and both projections reuse it.
inline void q38_gdn_f32_gemv_to_half(
        const Qwen38GdnF32Matrix & w,
        const float * x,
        __half * out,
        const Qwen38GdnScratch & scratch,
        cudaStream_t stream) {
    constexpr int threads = 256;
    if (!w.valid()) throw std::runtime_error("GDN F32 GEMV requires an F32 weight matrix");
    // Defence in depth: q38_gdn_f32_projection_usable() already checked the
    // scratch range, but an overflow here would corrupt the next projection.
    if (w.rows > scratch.gemv_elements()) throw std::runtime_error("GDN F32 GEMV output scratch mismatch");
    // The oracle reaches 1.0f/sqrtf-style F32 matmuls through one
    // ggml_vec_dot_f32 per row (AVX2: 4x8 accumulators + F32x8 reduce). The
    // warp-order reduction differs on 12-26 of 48 rows by 1 ulp, which survives
    // into the gate parameters and the written GDN state - hence the AVX2 order,
    // now warp-parallel (Task P) but arithmetically identical.
    q38_gdn_launch_dense_f32_gemv_avx2(w.data, x, scratch.gemv_f32(), w.rows, w.cols, stream);
    q38_gdn_float_to_half_kernel<<<(w.rows + threads - 1) / threads, threads, 0, stream>>>(
        scratch.gemv_f32(), out, w.rows);
}

// ssm_beta / ssm_alpha dispatch: one decision per projection, from the weight
// type, mirroring type_traits_cpu[src0t].vec_dot_type in ggml-cpu.c.
//   Q8_0       -> vec_dot_type Q8_0: reuse the activation already quantised for
//                 attn_qkv / attn_gate (bit-exact integer dot).
//   F32        -> vec_dot_type F32: multiply the F32 activation copy by the F32
//                 weight row with no quantisation (the pinned model).
//   F16 / BF16 -> vec_dot_type F16: the legacy half GEMV, unchanged.
// Precondition: in the Q8_0 branch the activation has already been copied to
// scratch.act_f32() (and quantised), so no extra conversion happens here.
inline void q38_gdn_param_projection(
        const GpuMatrix & m,
        const Qwen38GdnF32Matrix & m_f32,
        const __half * x,
        const Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        __half * out,
        cudaStream_t stream) {
    if (m.type == GgmlType::Q8_0 && m.lowbit && m.low.data != nullptr) {
        q38_gdn_q8_0_gemv_to_half(m, scratch, out, stream);
        return;
    }
    if (q38_gdn_f32_projection_usable(m_f32, scratch, q)) {
        q38_gdn_f32_gemv_to_half(m_f32, scratch.act_f32(), out, scratch, stream);
        return;
    }
    // Last resort: whatever storage upload_matrix() produced (F16/BF16 dense, or
    // any other low-bit type). Failing loudly beats dereferencing null.
    if ((m.lowbit && m.low.data == nullptr) || (!m.lowbit && m.dense.data == nullptr)) {
        throw std::runtime_error("GDN parameter projection has no usable weight storage");
    }
    launch_matrix_gemv_half(m, x, out, stream);
}

// The name the F32 chain has used for "quantise a row that already is F32"
// (the gated-RMSNorm output), now spelled as the F32-input overload above so
// both routes share one implementation. No half -> F32 conversion happens here.
inline void q38_gdn_q8_0_quantize_activation_f32(
        const float * x,
        int elements,
        const Qwen38GdnScratch & scratch,
        cudaStream_t stream) {
    q38_gdn_q8_0_quantize_activation(x, elements, scratch, stream);
}

// Q8_0 weight x Q8_0 activation -> F32 in ggml integer-dot order, left in F32
// for the F32-mid pipeline instead of a rounded __half.
inline void q38_gdn_q8_0_gemv_to_f32(
        const GpuMatrix & w,
        const Qwen38GdnScratch & scratch,
        float * out,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_launch_exact_q8_0_gemv_f32(
        w.low.data, scratch.act_q8(), scratch.act_scales(), out, w.rows, w.cols, stream);
}

// F32 weight x F32 activation -> F32 in ggml's F32 dot order: the same kernel as
// the half variant, without the rounding on the way out.
inline void q38_gdn_f32_gemv_to_f32(
        const Qwen38GdnF32Matrix & w,
        const float * x,
        float * out,
        cudaStream_t stream) {
    if (!w.valid()) throw std::runtime_error("GDN F32 GEMV requires an F32 weight matrix");
    q38_gdn_launch_dense_f32_gemv_avx2(w.data, x, out, w.rows, w.cols, stream);
}

// F32 twin of q38_gdn_param_projection(): same dispatch from the weight type,
// but the result stays F32.
//   q8_0_activation_ready: the caller already quantised the shared activation
//                          into act_q8()/act_scales(), the only state in which a
//                          Q8_0 parameter weight may use its Q8_0 vec_dot_type.
//   half_tmp:              landing slot for the legacy F16 GEMV last resort.
// Precondition: scratch.act_f32() holds the F32 activation row.
inline void q38_gdn_param_projection_f32(
        const GpuMatrix & m,
        const Qwen38GdnF32Matrix & m_f32,
        const __half * x,
        bool q8_0_activation_ready,
        float * out,
        __half * half_tmp,
        const Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    if (q8_0_activation_ready && m.type == GgmlType::Q8_0 && m.lowbit && m.low.data != nullptr) {
        q38_gdn_q8_0_gemv_to_f32(m, scratch, out, stream);
        return;
    }
    if (q38_gdn_f32_projection_usable(m_f32, scratch, q)) {
        q38_gdn_f32_gemv_to_f32(m_f32, scratch.act_f32(), out, stream);
        return;
    }
    if ((m.lowbit && m.low.data == nullptr) || (!m.lowbit && m.dense.data == nullptr)) {
        throw std::runtime_error("GDN parameter projection has no usable weight storage");
    }
    // Last resort: the legacy F16/BF16 GEMV, widened afterwards so the rest of
    // the pipeline still runs in F32.
    launch_matrix_gemv_half(m, x, half_tmp, stream);
    q38_hc_copy_half_to_f32_kernel<<<(m.rows + 255) / 256, 256, 0, stream>>>(
        half_tmp, out, m.rows);
}

// F32-input twin of q38_gdn_param_projection_f32(). The activation row is F32,
// so no path that stays inside the F32 chain rounds it: an exact Q8_0 weight
// reuses the shared Q8_0 activation, an F32 weight multiplies the row directly.
// Only the last-resort legacy F16/BF16 GEMV consumes an F16 activation, so that
// mirror is produced right there, into the caller's dead F16 scratch slot, and
// nowhere else - it can never feed a rounding back into the F32 chain.
inline void q38_gdn_param_projection_f32_x(
        const GpuMatrix & m,
        const Qwen38GdnF32Matrix & m_f32,
        const float * x_f32,
        bool q8_0_activation_ready,
        float * out,
        __half * half_tmp,
        __half * x_half_staging,
        const Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    if (q8_0_activation_ready && m.type == GgmlType::Q8_0 && m.lowbit && m.low.data != nullptr) {
        q38_gdn_q8_0_gemv_to_f32(m, scratch, out, stream);
        return;
    }
    if (q38_gdn_f32_projection_usable(m_f32, scratch, q)) {
        q38_gdn_f32_gemv_to_f32(m_f32, x_f32, out, stream);
        return;
    }
    if ((m.lowbit && m.low.data == nullptr) || (!m.lowbit && m.dense.data == nullptr)) {
        throw std::runtime_error("GDN parameter projection has no usable weight storage");
    }
    // Last resort: the legacy F16/BF16 GEMV. It reads an F16 activation row, so
    // the F32 row is mirrored here and the result is widened back afterwards.
    q38_gdn_mirror_activation_half(x_f32, x_half_staging, q.hidden, stream);
    launch_matrix_gemv_half(m, x_half_staging, half_tmp, stream);
    q38_hc_copy_half_to_f32_kernel<<<(m.rows + 255) / 256, 256, 0, stream>>>(
        half_tmp, out, m.rows);
}

// Everything run_qwen38_gdn_f32() needs before it may run. A false result sends
// run_qwen38_gdn() back to the legacy F16 path.
inline bool q38_gdn_f32_path_available(
        const Qwen38GdnState & state,
        const Qwen38GdnScratch & scratch,
        const Qwen38Layout & q) {
    if (!q38_gdn_f32_mid_enabled()) return false;
    if (!scratch.f32_ready() || state.conv_state_f32() == nullptr) return false;
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;
    if (q.hidden <= 0 || q.ssm_state_size <= 0 || q.ssm_group_count <= 0 ||
        q.ssm_dt_rank <= 0 || conv_dim <= 0) return false;
    return true;
}

// ---------------------------------------------------------------------------
// F32 GDN path (WARODECODE_GDN_F32_MID, default ON).
//
// Same arithmetic as run_qwen38_gdn(), but every intermediate that ggml keeps
// in F32 stays in F32: the projection outputs, the convolution history, the
// gate parameters, the recurrent-update inputs and the gated-RMSNorm chain.
// The projections themselves are unchanged: a quantised weight still
// quantises its activation exactly the way ggml does (Q8_0 blocks, integer
// dot), only its result is no longer rounded to __half on the way out.
//
// The F16 scratch slots are refreshed as mirrors at the end of the token so the
// legacy __half* accessors, the layer-14 replay path and the layer*.gdn.*
// traces keep their meaning; the unrounded rows are read through the *_f32()
// accessors.
//
// run_qwen38_gdn_f32_x() is the F32-input form: the hidden-state row arrives as
// F32 and is never rounded into __half on its way into the layer. That is what
// ggml does - the CPU rounds an activation only where the weight type forces it
// - so it is the entry an F32 producer (the HC mix output) should reach without
// a half mirror in between. run_qwen38_gdn_f32() widens an F16 row once and
// runs the same implementation, which keeps the two entries in step.
// ---------------------------------------------------------------------------
inline float * run_qwen38_gdn_f32_x(
        const float * x,
        const Qwen38GdnWeights & w,
        Qwen38GdnState & state,
        Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        float rms_eps,
        cudaStream_t stream) {
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;
    (void)q38_gdn_exact_exp_enabled();
    const int heads = q.ssm_dt_rank;
    constexpr int threads = 256;

    if (!scratch.f32_ready() || state.conv_state_f32() == nullptr) {
        throw std::runtime_error("GDN F32 path requires the F32 state and scratch buffers");
    }

    // Task P: outer bracket of one GDN layer. With WARODECODE_ATTN_PROFILE unset
    // this is a single predictable branch on the profiler's enabled() flag.
    Q38AttnKernelScope gdn_region_scope(kQ38AttnSiteGdnRegion, stream);

    float * const qkv_f32 = scratch.qkv_f32();
    float * const conv_out_f32 = scratch.conv_out_f32();
    float * const z_f32 = scratch.z_f32();
    float * const beta_f32 = scratch.beta_f32();
    float * const alpha_f32 = scratch.alpha_f32();
    float * const core_out_f32 = scratch.core_out_f32();
    float * const norm_out_f32 = scratch.norm_out_f32();
    float * const final_out_f32 = scratch.final_out_f32();

    // Landing slot for the F16 activation mirror the legacy GEMV fallbacks read.
    // scratch.final_out() is dead until the ssm_out projection writes it at the
    // end of this function, so it can hold the mirror for the whole token.
    __half * const activation_half = scratch.final_out();

    const bool q8_0 = q38_gdn_q8_0_enabled() && q38_gdn_q8_0_compatible(w, q);
    if (q8_0) {
        // attn_qkv / attn_gate and the ssm_beta / ssm_alpha F32 projections all
        // read the same hidden state: stage it once and quantise that copy once,
        // then reuse it for every Q8_0 dot. The row is already F32, so no
        // half -> F32 conversion happens anywhere on this route.
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnActQuant, stream);
            q38_gdn_q8_0_quantize_activation(x, q.hidden, scratch, stream);
        }
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnQkv, stream);
            q38_gdn_q8_0_gemv_to_f32(w.qkv, scratch, qkv_f32, stream);
        }
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnZ, stream);
            q38_gdn_q8_0_gemv_to_f32(w.z, scratch, z_f32, stream);
        }
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnBeta, stream);
            q38_gdn_param_projection_f32_x(w.beta, w.beta_f32, x, true, beta_f32,
                scratch.beta(), activation_half, scratch, q, stream);
        }
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnAlpha, stream);
            q38_gdn_param_projection_f32_x(w.alpha, w.alpha_f32, x, true, alpha_f32,
                scratch.alpha(), activation_half, scratch, q, stream);
        }
    } else {
        // WARODECODE_GDN_Q8_0=0: keep the legacy projections with their own
        // activation rounding, but widen the results so the token still runs
        // the F32 intermediate chain. Those kernels read an F16 activation, so
        // the F32 row is mirrored once for them - on this diagnostic branch only.
        q38_gdn_mirror_activation_half(x, activation_half, q.hidden, stream);
        launch_matrix_gemv_half(w.qkv, activation_half, scratch.qkv(), stream);
        launch_matrix_gemv_half(w.z, activation_half, scratch.z(), stream);
        q38_hc_copy_half_to_f32_kernel<<<(conv_dim + threads - 1) / threads, threads, 0, stream>>>(
            scratch.qkv(), qkv_f32, conv_dim);
        q38_hc_copy_half_to_f32_kernel<<<(value_dim + threads - 1) / threads, threads, 0, stream>>>(
            scratch.z(), z_f32, value_dim);
        q38_gdn_param_projection_f32_x(w.beta, w.beta_f32, x, false, beta_f32,
            scratch.beta(), activation_half, scratch, q, stream);
        q38_gdn_param_projection_f32_x(w.alpha, w.alpha_f32, x, false, alpha_f32,
            scratch.alpha(), activation_half, scratch, q, stream);
    }

    // ggml never rounds the convolution history. An external replay tool can
    // only reach the F16 mirror, so adopt whatever it changed before reading.
    const size_t conv_state_n = state.conv_state_elements();
    if (conv_state_n > 0) {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnConvAdopt, stream);
        q38_gdn_conv_state_adopt_half_kernel<<<
            (static_cast<int>(conv_state_n) + threads - 1) / threads, threads, 0, stream>>>(
            state.conv_state_f32(), state.conv_state(), static_cast<int>(conv_state_n));
    }

    const int conv_blocks = (conv_dim + threads - 1) / threads;
    {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnConv, stream);
        if (w.conv_f32.valid()) {
            q38_gdn_conv_f32_state_kernel<float><<<conv_blocks, threads, 0, stream>>>(
                qkv_f32, state.conv_state_f32(), w.conv_f32.data, conv_out_f32,
                conv_dim, q.ssm_conv_kernel);
        } else {
            q38_gdn_conv_f32_state_kernel<__half><<<conv_blocks, threads, 0, stream>>>(
                qkv_f32, state.conv_state_f32(), w.conv.data, conv_out_f32,
                conv_dim, q.ssm_conv_kernel);
        }
    }

    float * const q_conv = conv_out_f32;
    float * const k_conv = conv_out_f32 + key_dim;
    float * const v_conv = conv_out_f32 + key_dim * 2;
    {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnL2Norm, stream);
        q38_gdn_l2_norm_f32_kernel<<<q.ssm_group_count, threads, 0, stream>>>(
            q_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);
        q38_gdn_l2_norm_f32_kernel<<<q.ssm_group_count, threads, 0, stream>>>(
            k_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);
    }

    const int param_blocks = (heads + threads - 1) / threads;
    {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnGateParams, stream);
        if (w.dt_f32.data != nullptr && w.a_f32.data != nullptr) {
            q38_gdn_gate_params_f32_kernel<<<param_blocks, threads, 0, stream>>>(
                beta_f32, alpha_f32, w.dt_f32.data, w.a_f32.data, scratch.gate(), heads);
        } else {
            q38_gdn_gate_params_f32_halfparam_kernel<<<param_blocks, threads, 0, stream>>>(
                beta_f32, alpha_f32, w.dt.data, w.a.data, scratch.gate(), heads);
        }
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnAr, stream);
        q38_gdn_ar_f32_kernel<<<heads, q.ssm_state_size, 0, stream>>>(
            q_conv, k_conv, v_conv, scratch.gate(), beta_f32,
            state.state(), core_out_f32,
            q.ssm_state_size, q.ssm_group_count, heads);
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnNorm, stream);
        if (w.norm_f32.data != nullptr) {
            q38_gdn_gated_rmsnorm_f32_kernel<<<heads, threads, 0, stream>>>(
                core_out_f32, w.norm_f32.data, z_f32, norm_out_f32,
                q.ssm_state_size, heads, rms_eps);
        } else {
            q38_gdn_gated_rmsnorm_f32_kernel<<<heads, threads, 0, stream>>>(
                core_out_f32, w.norm.data, z_f32, norm_out_f32,
                q.ssm_state_size, heads, rms_eps);
        }
    }

    if (q8_0) {
        // ssm_out reads the gated-RMSNorm output, so it needs its own
        // quantisation pass (value_dim elements).
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnOutQuant, stream);
            q38_gdn_q8_0_quantize_activation_f32(norm_out_f32, value_dim, scratch, stream);
        }
        {
            Q38AttnKernelScope scope(kQ38AttnSiteGdnOut, stream);
            q38_gdn_q8_0_gemv_to_f32(w.out, scratch, final_out_f32, stream);
        }
    } else {
        Q38AttnKernelScope scope(kQ38AttnSiteGdnOut, stream);
        q38_gdn_float_to_half_kernel<<<(value_dim + threads - 1) / threads, threads, 0, stream>>>(
            norm_out_f32, scratch.norm_out(), value_dim);
        launch_matrix_gemv_half(w.out, scratch.norm_out(), scratch.final_out(), stream);
        q38_hc_copy_half_to_f32_kernel<<<(q.hidden + threads - 1) / threads, threads, 0, stream>>>(
            scratch.final_out(), final_out_f32, q.hidden);
    }

    // Refresh the F16 mirrors the legacy accessors and the traces read. They
    // show the rounded view of the F32 values, which is exactly what the old
    // path stored.
    const auto mirror = [&](const float * src, __half * dst, int n) {
        q38_gdn_float_to_half_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(src, dst, n);
    };
    Q38AttnKernelScope mirror_scope(kQ38AttnSiteGdnMirror, stream);
    mirror(qkv_f32, scratch.qkv(), conv_dim);
    mirror(conv_out_f32, scratch.conv_out(), conv_dim);
    mirror(z_f32, scratch.z(), value_dim);
    mirror(beta_f32, scratch.beta(), heads);
    mirror(alpha_f32, scratch.alpha(), heads);
    mirror(core_out_f32, scratch.core_out(), value_dim);
    mirror(norm_out_f32, scratch.norm_out(), value_dim);
    mirror(final_out_f32, scratch.final_out(), q.hidden);
    if (conv_state_n > 0) {
        mirror(state.conv_state_f32(), state.conv_state(), static_cast<int>(conv_state_n));
    }
    return final_out_f32;
}

// F16-input entry. half -> F32 is exact, so the row is widened once here and the
// F32 implementation above runs on it. F16 -> F32 -> F16 is the identity on
// every __half value, so the legacy GEMV fallbacks inside that implementation
// still see exactly the bits this caller passed in.
inline float * run_qwen38_gdn_f32(
        const __half * x,
        const Qwen38GdnWeights & w,
        Qwen38GdnState & state,
        Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        float rms_eps,
        cudaStream_t stream) {
    if (!scratch.f32_ready() || state.conv_state_f32() == nullptr) {
        throw std::runtime_error("GDN F32 path requires the F32 state and scratch buffers");
    }
    constexpr int threads = 256;
    q38_hc_copy_half_to_f32_kernel<<<(q.hidden + threads - 1) / threads, threads, 0, stream>>>(
        x, scratch.act_f32(), q.hidden);
    return run_qwen38_gdn_f32_x(scratch.act_f32(), w, state, scratch, q, rms_eps, stream);
}

// Legacy entry point. It runs the F32 path by default and returns the F16
// mirror of final_out; callers that can consume F32 should call
// run_qwen38_gdn_f32_x() (an F32 row) or run_qwen38_gdn_f32() (an F16 row)
// directly. The F16 path below remains as the fallback.
inline __half * run_qwen38_gdn(
        const __half * x,
        const Qwen38GdnWeights & w,
        Qwen38GdnState & state,
        Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        float rms_eps,
        cudaStream_t stream) {
    (void)q38_gdn_exact_exp_enabled();
    if (q38_gdn_f32_path_available(state, scratch, q)) {
        run_qwen38_gdn_f32(x, w, state, scratch, q, rms_eps, stream);
        return scratch.final_out();
    }

    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;

    const bool q8_0 = q38_gdn_q8_0_enabled() && q38_gdn_q8_0_compatible(w, q);
    if (q8_0) {
        // attn_qkv / attn_gate and the ssm_beta / ssm_alpha F32 projections all
        // read the same hidden state. The half input is expanded to F32 exactly
        // once; Q8_0 weights then quantise that F32 copy once and reuse it.
        q38_gdn_q8_0_quantize_activation(x, q.hidden, scratch, stream);
        q38_gdn_q8_0_gemv_to_half(w.qkv, scratch, scratch.qkv(), stream);
        q38_gdn_q8_0_gemv_to_half(w.z, scratch, scratch.z(), stream);
        // ssm_beta / ssm_alpha are F32 weights, so ggml leaves their activation
        // in F32 (vec_dot_type == F32); a Q8_0 variant still reuses the shared
        // Q8_0 activation. Same output slots as the legacy path.
        q38_gdn_param_projection(w.beta, w.beta_f32, x, scratch, q, scratch.beta(), stream);
        q38_gdn_param_projection(w.alpha, w.alpha_f32, x, scratch, q, scratch.alpha(), stream);
    } else {
        launch_matrix_gemv_half(w.qkv, x, scratch.qkv(), stream);
        launch_matrix_gemv_half(w.z, x, scratch.z(), stream);
        launch_matrix_gemv_half(w.beta, x, scratch.beta(), stream);
        launch_matrix_gemv_half(w.alpha, x, scratch.alpha(), stream);
    }

    const int conv_blocks = (conv_dim + 255) / 256;
    q38_gdn_conv_kernel<<<conv_blocks, 256, 0, stream>>>(
        scratch.qkv(), state.conv_state(), w.conv.data, scratch.conv_out(),
        conv_dim, q.ssm_conv_kernel);

    __half * q_conv = scratch.conv_out();
    __half * k_conv = scratch.conv_out() + key_dim;
    __half * v_conv = scratch.conv_out() + key_dim * 2;

    q38_l2_norm_heads_kernel<<<q.ssm_group_count, 256, 0, stream>>>(
        q_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);
    q38_l2_norm_heads_kernel<<<q.ssm_group_count, 256, 0, stream>>>(
        k_conv, q.ssm_state_size, q.ssm_group_count, rms_eps);

    const int param_blocks = (q.ssm_dt_rank + 255) / 256;
    q38_gdn_gate_params_kernel<<<param_blocks, 256, 0, stream>>>(
        scratch.beta(), scratch.alpha(), w.dt.data, w.a.data,
        scratch.gate(), q.ssm_dt_rank);

    q38_gdn_ar_kernel<<<q.ssm_dt_rank, q.ssm_state_size, 0, stream>>>(
        q_conv, k_conv, v_conv, scratch.gate(), scratch.beta(),
        state.state(), scratch.core_out(),
        q.ssm_state_size, q.ssm_group_count, q.ssm_dt_rank);

    q38_gdn_gated_rmsnorm_kernel<<<q.ssm_dt_rank, 256, 0, stream>>>(
        scratch.core_out(), w.norm.data, scratch.z(), scratch.norm_out(),
        q.ssm_state_size, q.ssm_dt_rank, rms_eps);

    if (q8_0) {
        // ssm_out reads the gated-RMSNorm output, so it needs its own
        // quantisation pass (value_dim elements).
        q38_gdn_q8_0_quantize_activation(scratch.norm_out(), value_dim, scratch, stream);
        q38_gdn_q8_0_gemv_to_half(w.out, scratch, scratch.final_out(), stream);
    } else {
        launch_matrix_gemv_half(w.out, scratch.norm_out(), scratch.final_out(), stream);
    }
    return scratch.final_out();
}

} // namespace wd
