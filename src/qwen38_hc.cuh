#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "qwen38_exact_math.cuh"
#include "qwen38_precision.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38HcWeights {
    DeviceHalfArray norm;
    GpuMatrix down;
    GpuMatrix up;
    GpuMatrix inject;
    // Unrounded F32 inject weights straight from the GGUF, empty when the
    // tensor is quantised (the blk.48 MTP block stores Q8_0 inject weights).
    // GpuMatrix's dense path stores a dense F32 tensor as F16
    // (upload_half_array), which would round the inject weight before the
    // multiply; the CPU reference keeps it F32.
    DeviceFloatArray inject_f32;
    std::vector<float> debug_norm_f32;
    std::vector<float> debug_inject_f32;
    bool has_inject = true;
    bool has_inject_f32 = false;
};

// True when the GGUF inject tensor was F32 and its unrounded values are
// resident on the device, so the F32 inject path can replace the F16 path.
// This has to stay a runtime check: the MTP draft block's inject tensors are
// quantised in the same model.
inline bool qwen38_hc_f32_inject_ready(const Qwen38HcWeights & w) {
    return w.has_inject && w.has_inject_f32 &&
           w.inject_f32.data != nullptr && w.inject_f32.size != 0;
}

// Per-file environment switch, matching the other q38_*_env_flag helpers.
inline bool q38_hc_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

// Default-on switch for the F32 inject path: WARODECODE_HC_INJECT_F32=0
// restores the legacy F16 inject, and a model whose inject tensor is quantised
// falls back automatically through qwen38_hc_f32_inject_ready().
inline bool qwen38_hc_f32_inject_enabled(const Qwen38HcWeights & w) {
    static const bool enabled = q38_hc_env_flag("WARODECODE_HC_INJECT_F32", true);
    return enabled && qwen38_hc_f32_inject_ready(w);
}

inline Qwen38HcWeights load_qwen38_hc_weights(
        const GgufFile & g,
        int il,
        const char * branch,
        int hidden,
        int hc,
        int hc_low_rank) {
    const std::string p = "blk." + std::to_string(il) + ".hc_" + branch + "_";
    const TensorInfo & norm = g.tensor(p + "norm.weight");
    const TensorInfo & down = g.tensor(p + "down.weight");
    const TensorInfo & up = g.tensor(p + "up.weight");
    const TensorInfo & inject = g.tensor(p + "inject.weight");

    const int hc_dim = hidden * hc;
    if (static_cast<int>(norm.elements()) != hc_dim) {
        throw std::runtime_error("HC norm shape mismatch: " + norm.name);
    }
    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("HC ") + what + " shape mismatch: " + t.name);
        }
    };
    check2(down, hc_low_rank, hc_dim, "down");
    check2(up, hc_dim, hc_low_rank, "up");
    check2(inject, hc, hc_dim, "inject");

    Qwen38HcWeights out;
    out.norm = upload_half_array(norm);
    if (norm.type == GgmlType::F32 &&
        down.type == GgmlType::Q8_0 && up.type == GgmlType::Q8_0) {
        const float * values = reinterpret_cast<const float *>(norm.data);
        out.debug_norm_f32.assign(values, values + norm.elements());
    }
    // Keep the unrounded F32 inject weights resident on the device for every
    // layer and both branches. Quantised inject tensors (blk.48 MTP) keep the
    // legacy lowbit/dense path; callers fall back through
    // qwen38_hc_f32_inject_ready().
    if (inject.type == GgmlType::F32) {
        out.inject_f32 = upload_float_array(inject);
        out.has_inject_f32 =
            out.inject_f32.size == static_cast<size_t>(hc) * static_cast<size_t>(hc_dim);
    }
    // Layer-0 attention F32 debug replay additionally keeps a host copy, which
    // is what the trace/override tooling loads.
    if (il == 0 && std::strcmp(branch, "attn") == 0 && inject.type == GgmlType::F32) {
        const float * values = reinterpret_cast<const float *>(inject.data);
        out.debug_inject_f32.assign(values, values + inject.elements());
    }
    out.down = upload_matrix(down);
    out.up = upload_matrix(up);
    out.inject = upload_matrix(inject);
    return out;
}

inline Qwen38HcWeights load_qwen38_hc_head(
        const GgufFile & g,
        int hidden,
        int hc,
        int hc_low_rank) {
    const TensorInfo & norm = g.tensor("output_hc_norm.weight");
    const TensorInfo & down = g.tensor("output_hc_down.weight");
    const TensorInfo & up = g.tensor("output_hc_up.weight");

    const int hc_dim = hidden * hc;
    if (static_cast<int>(norm.elements()) != hc_dim) {
        throw std::runtime_error("HC head norm shape mismatch");
    }
    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("HC head ") + what + " shape mismatch: " + t.name);
        }
    };
    check2(down, hc_low_rank, hc_dim, "down");
    check2(up, hc_dim, hc_low_rank, "up");

    Qwen38HcWeights out;
    out.norm = upload_half_array(norm);
    if (norm.type == GgmlType::F32 && down.type == GgmlType::Q8_0 && up.type == GgmlType::Q8_0) {
        const float * values = reinterpret_cast<const float *>(norm.data);
        out.debug_norm_f32.assign(values, values + norm.elements());
    }
    out.down = upload_matrix(down);
    out.up = upload_matrix(up);
    out.has_inject = false;
    return out;
}

__global__ void q38_hc_group_rmsnorm_kernel(
        const __half * __restrict__ residual,
        const __half * __restrict__ gamma,
        __half * __restrict__ xn,
        int hidden,
        int hc,
        float eps) {
    const int stream = blockIdx.x;
    if (stream >= hc) return;

    __shared__ float smem[256];
    const int base = stream * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __half2float(residual[base + i]);
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const int idx = base + i;
        const float v = __half2float(residual[idx]);
        const float g = __half2float(gamma[idx]);
        xn[idx] = __float2half(v * inv * g);
    }
}

// ggml_silu on the CPU is ggml_vec_silu_f32 (ggml-cpu/vec.cpp:380), whose AVX2
// branch is ggml_v_silu -> the ggml_v_expf polynomial (ggml-cpu/vec.h:1215,
// 1255). q38_exact_silu_f32() reproduces that lane by lane.
__global__ void q38_hc_silu_scale_kernel(__half * x, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool exact_exp = q38_exact_exp_on();
    const float v = __half2float(x[i]) * inv_hc;
    x[i] = __float2half(q38_exact_silu_f32(v, exact_exp));
}

__global__ void q38_hc_mix_kernel(
        const __half * __restrict__ xn,
        const __half * __restrict__ gate,
        __half * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    // ggml_sigmoid -> op_sigmoid (ggml-cpu/unary-ops.cpp:31), scalar glibc expf.
    const bool exact_exp = q38_exact_exp_on();
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float x = __half2float(xn[idx]);
        const float g = __half2float(gate[idx]);
        const float sig = q38_exact_sigmoid_f32(g, exact_exp);
        sum = fmaf(x, sig, sum);
    }
    mixed[i] = __float2half(sum / static_cast<float>(hc));
}

__global__ void q38_hc_sigmoid_gate_f32_trace_kernel(
        const __half * __restrict__ gate,
        float * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = __half2float(gate[i]);
        output[i] = q38_exact_sigmoid_f32(value, q38_exact_exp_on());
    }
}

__global__ void q38_hc_sigmoid_f32_gate_trace_kernel(
        const float * __restrict__ gate,
        float * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = q38_exact_sigmoid_f32(gate[i], q38_exact_exp_on());
}

__global__ void q38_hc_copy_f32_to_half_kernel(
        const float * __restrict__ input,
        __half * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half(input[i]);
}

__global__ void q38_hc_q8_0_quantize_half_kernel(
        const __half * __restrict__ input,
        int elements,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= elements / 32) return;

    const int i = block * 32 + lane;
    const float value = __half2float(input[i]);
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = amax / 127.0f;
    quantized[i] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / d));
    if (lane == 0) scales[block] = __float2half(d);
}

__global__ void q38_hc_q8_0_quantize_f32_kernel(
        const float * __restrict__ input,
        int elements,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= elements / 32) return;

    const int i = block * 32 + lane;
    const float value = input[i];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = amax / 127.0f;
    quantized[i] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / d));
    if (lane == 0) scales[block] = __float2half(d);
}

__global__ void q38_hc_q8_0_q8_0_gemv_f32_kernel(
        const uint8_t * __restrict__ weight,
        const int8_t * __restrict__ activation,
        const __half * __restrict__ activation_scales,
        float * __restrict__ output,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int blocks_per_row = cols >> 5;
    const uint8_t * row_ptr = weight + static_cast<size_t>(row) * blocks_per_row * 34u;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * block = row_ptr + static_cast<size_t>(b) * 34u;
        const int8_t qw = *reinterpret_cast<const int8_t *>(block + 2 + lane);
        int dot = qw * static_cast<int>(activation[(b << 5) + lane]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            const __half weight_scale = *reinterpret_cast<const __half *>(block);
            const float scale = __half2float(weight_scale) * __half2float(activation_scales[b]);
            sum = __fadd_rn(sum, __fmul_rn(scale, static_cast<float>(dot)));
        }
    }
    if (lane == 0) output[row] = sum;
}

__device__ inline int q38_hc_q8_0_abs_i8(int8_t value) {
    if (value == static_cast<int8_t>(-128)) return 128;
    return value < 0 ? -static_cast<int>(value) : static_cast<int>(value);
}

__device__ inline int q38_hc_q8_0_sign_i8(int8_t value, int8_t sign) {
    if (sign == 0) return 0;
    if (sign < 0) return value == static_cast<int8_t>(-128) ? 127 : -static_cast<int>(value);
    return static_cast<int>(value);
}

__device__ inline int q38_hc_q8_0_sat_i16(int value) {
    return max(-32768, min(32767, value));
}

__global__ void q38_hc_q8_0_quantize_f32_exact_kernel(
        const float * __restrict__ input,
        int elements,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= elements / 32) return;

    const int i = block * 32 + lane;
    const float value = input[i];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float inverse = amax == 0.0f ? 0.0f : __fdiv_rn(127.0f, amax);
    const int q = amax == 0.0f ? 0 : __float2int_rn(__fmul_rn(value, inverse));
    quantized[i] = static_cast<int8_t>(max(-128, min(127, q)));
    if (lane == 0) scales[block] = __float2half_rn(d);
}

__global__ void q38_hc_q8_0_q8_0_exact_gemv_f32_kernel(
        const uint8_t * __restrict__ weight,
        const int8_t * __restrict__ activation,
        const __half * __restrict__ activation_scales,
        float * __restrict__ output,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int blocks_per_row = cols >> 5;
    const uint8_t * row_ptr = weight + static_cast<size_t>(row) * blocks_per_row * 34u;
    float accumulator = 0.0f;
    if (lane < 8) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t * block = row_ptr + static_cast<size_t>(b) * 34u;
            const int base = lane * 4;
            const int8_t qw0 = *reinterpret_cast<const int8_t *>(block + 2 + base + 0);
            const int8_t qw1 = *reinterpret_cast<const int8_t *>(block + 2 + base + 1);
            const int8_t qw2 = *reinterpret_cast<const int8_t *>(block + 2 + base + 2);
            const int8_t qw3 = *reinterpret_cast<const int8_t *>(block + 2 + base + 3);
            const int8_t qx0 = activation[(b << 5) + base + 0];
            const int8_t qx1 = activation[(b << 5) + base + 1];
            const int8_t qx2 = activation[(b << 5) + base + 2];
            const int8_t qx3 = activation[(b << 5) + base + 3];

            const int pair0 = q38_hc_q8_0_sat_i16(
                q38_hc_q8_0_abs_i8(qw0) * q38_hc_q8_0_sign_i8(qx0, qw0) +
                q38_hc_q8_0_abs_i8(qw1) * q38_hc_q8_0_sign_i8(qx1, qw1));
            const int pair1 = q38_hc_q8_0_sat_i16(
                q38_hc_q8_0_abs_i8(qw2) * q38_hc_q8_0_sign_i8(qx2, qw2) +
                q38_hc_q8_0_abs_i8(qw3) * q38_hc_q8_0_sign_i8(qx3, qw3));
            const int dot = pair0 + pair1;
            const float weight_scale = lowbit_half(block);
            const float activation_scale = __half2float(activation_scales[b]);
            const float scale = __fmul_rn(weight_scale, activation_scale);
            accumulator = __fmaf_rn(scale, static_cast<float>(dot), accumulator);
        }
    }

    const float a0 = __shfl_sync(0xffffffffu, accumulator, 0);
    const float a1 = __shfl_sync(0xffffffffu, accumulator, 1);
    const float a2 = __shfl_sync(0xffffffffu, accumulator, 2);
    const float a3 = __shfl_sync(0xffffffffu, accumulator, 3);
    const float a4 = __shfl_sync(0xffffffffu, accumulator, 4);
    const float a5 = __shfl_sync(0xffffffffu, accumulator, 5);
    const float a6 = __shfl_sync(0xffffffffu, accumulator, 6);
    const float a7 = __shfl_sync(0xffffffffu, accumulator, 7);
    const float r0 = __fadd_rn(a4, a0);
    const float r1 = __fadd_rn(a5, a1);
    const float r2 = __fadd_rn(a6, a2);
    const float r3 = __fadd_rn(a7, a3);
    const float s0 = __fadd_rn(r0, r2);
    const float s1 = __fadd_rn(r1, r3);
    if (lane == 0) output[row] = __fadd_rn(s0, s1);
}

__device__ inline float q38_hc_ggml_avx2_exp_f32(float x) {
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

__global__ void q38_hc_silu_scale_ggml_f32_kernel(float * x, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __fmul_rn(x[i], inv_hc);
    const float exp_neg = q38_hc_ggml_avx2_exp_f32(-v);
    x[i] = __fdiv_rn(v, __fadd_rn(1.0f, exp_neg));
}

__global__ void q38_hc_mix_f32_cpu_order_kernel(
        const float * __restrict__ xn,
        const float * __restrict__ gate,
        float * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const bool exact_exp = q38_exact_exp_on();
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float sigmoid = q38_exact_sigmoid_f32(gate[idx], exact_exp);
        sum = __fadd_rn(sum, __fmul_rn(xn[idx], sigmoid));
    }
    mixed[i] = __fmul_rn(sum, 1.0f / static_cast<float>(hc));
}

__global__ void q38_hc_mix_f32_gate_replay_kernel(
        const __half * __restrict__ xn,
        const float * __restrict__ gate,
        __half * __restrict__ mixed,
        int hidden,
        int hc,
        bool cpu_order) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const bool exact_exp = q38_exact_exp_on();
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float x = __half2float(xn[idx]);
        const float sig = q38_exact_sigmoid_f32(gate[idx], exact_exp);
        sum = cpu_order ? __fadd_rn(sum, __fmul_rn(x, sig)) : fmaf(x, sig, sum);
    }
    mixed[i] = __float2half(sum / static_cast<float>(hc));
}

__global__ void q38_hc_mix_f32_x_f32_gate_replay_kernel(
        const float * __restrict__ xn,
        const float * __restrict__ gate,
        __half * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const bool exact_exp = q38_exact_exp_on();
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float sig = q38_exact_sigmoid_f32(gate[idx], exact_exp);
        sum = __fadd_rn(sum, __fmul_rn(xn[idx], sig));
    }
    mixed[i] = __float2half(sum / static_cast<float>(hc));
}

struct Qwen38HcF32Trace {
    std::vector<float> init;
    std::vector<float> norm;
    std::vector<float> low_raw;
    std::vector<float> low;
    std::vector<float> gate;
    std::vector<float> mixed;
};

class Qwen38HcF32DebugScratch {
public:
    Qwen38HcF32DebugScratch() = default;
    Qwen38HcF32DebugScratch(const Qwen38HcF32DebugScratch &) = delete;
    Qwen38HcF32DebugScratch & operator=(const Qwen38HcF32DebugScratch &) = delete;
    ~Qwen38HcF32DebugScratch() { reset(); }

    void initialize(int hidden, int hc, int low_rank) {
        hidden_ = hidden;
        hc_ = hc;
        low_rank_ = low_rank;
        const size_t hc_dim = static_cast<size_t>(hidden) * hc;
        alloc(reinterpret_cast<void **>(&embedding_), static_cast<size_t>(hidden) * sizeof(float));
        alloc(reinterpret_cast<void **>(&residual_), hc_dim * sizeof(float));
        alloc(reinterpret_cast<void **>(&norm_), hc_dim * sizeof(float));
        alloc(reinterpret_cast<void **>(&gamma_), hc_dim * sizeof(float));
        alloc(reinterpret_cast<void **>(&low_), static_cast<size_t>(low_rank) * sizeof(float));
        alloc(reinterpret_cast<void **>(&low_raw_), static_cast<size_t>(low_rank) * sizeof(float));
        alloc(reinterpret_cast<void **>(&gate_), hc_dim * sizeof(float));
        alloc(reinterpret_cast<void **>(&mixed_), static_cast<size_t>(hidden) * sizeof(float));
        alloc(reinterpret_cast<void **>(&inject_), static_cast<size_t>(hc) * sizeof(float));
        alloc(reinterpret_cast<void **>(&inject_weights_), hc_dim * hc * sizeof(float));
        alloc(reinterpret_cast<void **>(&q8_0_values_), hc_dim * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&q8_0_scales_), (hc_dim / 32) * sizeof(__half));
        q8_1_.initialize(static_cast<int>(hc_dim));
    }

    int hidden() const { return hidden_; }
    int hc() const { return hc_; }
    int low_rank() const { return low_rank_; }
    float * embedding() const { return embedding_; }
    float * residual() const { return residual_; }
    float * norm() const { return norm_; }
    float * gamma() const { return gamma_; }
    float * low() const { return low_; }
    float * low_raw() const { return low_raw_; }
    float * gate() const { return gate_; }
    float * mixed() const { return mixed_; }
    float * inject() const { return inject_; }
    float * inject_weights() const { return inject_weights_; }
    int8_t * q8_0_values() const { return q8_0_values_; }
    __half * q8_0_scales() const { return q8_0_scales_; }
    Q8_1Buffer & q8_1() { return q8_1_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc HC F32 debug scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (embedding_) cudaFree(embedding_);
        if (residual_) cudaFree(residual_);
        if (norm_) cudaFree(norm_);
        if (gamma_) cudaFree(gamma_);
        if (low_) cudaFree(low_);
        if (low_raw_) cudaFree(low_raw_);
        if (gate_) cudaFree(gate_);
        if (mixed_) cudaFree(mixed_);
        if (inject_) cudaFree(inject_);
        if (inject_weights_) cudaFree(inject_weights_);
        if (q8_0_values_) cudaFree(q8_0_values_);
        if (q8_0_scales_) cudaFree(q8_0_scales_);
        embedding_ = residual_ = norm_ = gamma_ = low_ = low_raw_ = gate_ = mixed_ = nullptr;
        inject_ = inject_weights_ = nullptr;
        q8_0_values_ = nullptr;
        q8_0_scales_ = nullptr;
    }

    int hidden_ = 0;
    int hc_ = 0;
    int low_rank_ = 0;
    float * embedding_ = nullptr;
    float * residual_ = nullptr;
    float * norm_ = nullptr;
    float * gamma_ = nullptr;
    float * low_ = nullptr;
    float * low_raw_ = nullptr;
    float * gate_ = nullptr;
    float * mixed_ = nullptr;
    float * inject_ = nullptr;
    float * inject_weights_ = nullptr;
    int8_t * q8_0_values_ = nullptr;
    __half * q8_0_scales_ = nullptr;
    Q8_1Buffer q8_1_;
};

__global__ void q38_hc_q8_0_gather_row_f32_kernel(
        const uint8_t * __restrict__ w,
        int cols,
        int row,
        float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const int block = i >> 5;
    const int lane = i & 31;
    const uint8_t * p = w + (static_cast<size_t>(row) * (cols >> 5) + block) * 34u;
    const float d = lowbit_half(p);
    const int8_t q = *reinterpret_cast<const int8_t *>(p + 2 + lane);
    out[i] = d * static_cast<float>(q);
}

__global__ void q38_hc_repeat_f32_kernel(
        const float * __restrict__ embedding,
        float * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i < total) residual[i] = embedding[i % hidden];
}

__global__ void q38_hc_group_rmsnorm_f32_kernel(
        const float * __restrict__ residual,
        const float * __restrict__ gamma,
        float * __restrict__ xn,
        int hidden,
        int hc,
        float eps) {
    const int stream = blockIdx.x;
    if (stream >= hc) return;

    __shared__ float smem[256];
    const int base = stream * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = residual[base + i];
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const int idx = base + i;
        xn[idx] = residual[idx] * inv * gamma[idx];
    }
}

// F32-residual variants: the persistent HC residual stream is F32.
__global__ void q38_hc_group_rmsnorm_f32in_kernel(
        const float * __restrict__ residual,
        const __half * __restrict__ gamma,
        __half * __restrict__ xn,
        int hidden,
        int hc,
        float eps) {
    const int stream = blockIdx.x;
    if (stream >= hc) return;
    __shared__ float smem[256];
    const int base = stream * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = residual[base + i];
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const int idx = base + i;
        xn[idx] = __float2half(residual[idx] * inv * __half2float(gamma[idx]));
    }
}

__global__ void q38_hc_silu_scale_f32_kernel(float * x, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool exact_exp = q38_exact_exp_on();
    const float v = x[i] * inv_hc;
    x[i] = q38_exact_silu_f32(v, exact_exp);
}

__global__ void q38_hc_mix_f32_kernel(
        const float * __restrict__ xn,
        const float * __restrict__ gate,
        float * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const bool exact_exp = q38_exact_exp_on();
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float sig = q38_exact_sigmoid_f32(gate[idx], exact_exp);
        sum = fmaf(xn[idx], sig, sum);
    }
    mixed[i] = sum / static_cast<float>(hc);
}

__global__ void q38_hc_dense_gemv_f32_kernel(
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

// WARODECODE_HC_INJECT_F32_AVX2 (default ON).
//
// The HC inject weight (blk.N.hc_*_inject.weight, F32 [hc_dim, hc]) is an F32
// tensor, so build_hc_mix's "build_lora_mm(w_inject, xn)" is one
// ggml_vec_dot_f32 per row on the CPU: 32-value blocks of four 8-lane FMA
// accumulators, the GGML_F32x8_REDUCE tree, then a non-fused leftover tail
// (llamafile_sgemm() declines the single decode token, sgemm.cpp:3819).
// q38_hc_dense_gemv_f32_kernel() sums lane-strided with a shfl_down tree
// instead, which is a different summation order and therefore different bits.
// This switch selects the oracle order; =0 restores the warp order for A/B.
inline bool q38_hc_f32_gemv_avx2_enabled() {
    static const bool enabled = q38_hc_env_flag("WARODECODE_HC_INJECT_F32_AVX2", true);
    return enabled;
}

// Tree reduction half of ggml_vec_dot_f32 on AVX2+FMA, identical to
// q38_gdn_reduce_avx2() / q38_moe_reduce_avx2().
__device__ __forceinline__ float q38_hc_reduce_avx2(float sum[4][8]) {
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

// One ggml_vec_dot_f32 per row of the F32 inject weight, one thread per row.
__global__ void q38_hc_dense_gemv_f32_avx2_kernel(
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
    float acc = q38_hc_reduce_avx2(sum);
    // Non-fused leftover tail, like q38_gdn_dense_f32_gemv_avx2_kernel(); the
    // inject width is hc * n_embd (10240 for the pinned model), a multiple of 32.
    for (int c = np; c < cols; ++c) acc = __fadd_rn(acc, __fmul_rn(row_ptr[c], x[c]));
    y[row] = acc;
}

// F32 inject GEMV entry point used by the layer-0 replay and the production
// path, so both read the same arithmetic.
inline void q38_hc_launch_dense_gemv_f32(
        const float * w,
        const float * x,
        float * y,
        int rows,
        int cols,
        cudaStream_t stream) {
    if (q38_hc_f32_gemv_avx2_enabled()) {
        // One thread per row, so the grid follows the thread count.
        constexpr int threads = 256;
        q38_hc_dense_gemv_f32_avx2_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
            w, x, y, rows, cols);
        return;
    }
    constexpr int warps = 8;
    const int blocks = (rows + warps - 1) / warps;
    q38_hc_dense_gemv_f32_kernel<<<blocks, warps * 32, 0, stream>>>(w, x, y, rows, cols);
}

__global__ void q38_hc_combine_f32_kernel(
        float * __restrict__ residual,
        const float * __restrict__ block_out,
        const float * __restrict__ inject,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    const int c = i / hidden;
    const int h = i - c * hidden;
    // ggml_scale(inject, 1/hc) -> ggml_sigmoid -> ggml_scale(w, 2): an exact
    // reciprocal instead of the --use_fast_math approximate "2.0f / (...)".
    const float z = inject[c] / static_cast<float>(hc);
    const float w = 2.0f / (1.0f + expf(-z));
    residual[i] = fmaf(block_out[h], w, residual[i]);
}

__global__ void q38_hc_copy_half_to_f32_kernel(
        const __half * __restrict__ input,
        float * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __half2float(input[i]);
}

inline void run_qwen38_hc_f32_mix_from_residual(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;
    if (w.debug_norm_f32.size() != static_cast<size_t>(hc_dim) ||
        !w.down.lowbit || w.down.type != GgmlType::Q8_0 ||
        !w.up.lowbit || w.up.type != GgmlType::Q8_0 ||
        w.down.cols != hc_dim || w.down.rows != scratch.low_rank() ||
        w.up.cols != scratch.low_rank() || w.up.rows != hc_dim) {
        throw std::runtime_error("layer-0 HC F32 mix geometry/type mismatch");
    }
    cudaError_t e = cudaMemcpyAsync(
        scratch.gamma(), w.debug_norm_f32.data(), static_cast<size_t>(hc_dim) * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy layer-0 HC F32 norm failed");

    constexpr int threads = 256;
    q38_hc_group_rmsnorm_f32_kernel<<<hc, threads, 0, stream>>>(
        residual, scratch.gamma(), scratch.norm(), hidden, hc, eps);
    launch_q8_0_gemv_q8_1_f32(w.down.low, scratch.norm(), scratch.low_raw(), scratch.q8_1(), stream);
    e = cudaMemcpyAsync(
        scratch.low(), scratch.low_raw(), static_cast<size_t>(scratch.low_rank()) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy layer-0 HC F32 low projection failed");
    q38_hc_silu_scale_f32_kernel<<<(scratch.low_rank() + threads - 1) / threads, threads, 0, stream>>>(
        scratch.low(), scratch.low_rank(), 1.0f / static_cast<float>(hc));
    launch_q8_0_gemv_q8_1_f32(w.up.low, scratch.low(), scratch.gate(), scratch.q8_1(), stream);
    q38_hc_mix_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
        scratch.norm(), scratch.gate(), scratch.mixed(), hidden, hc);
    if (!w.debug_inject_f32.empty()) {
        if (w.debug_inject_f32.size() != static_cast<size_t>(hc_dim * hc)) {
            throw std::runtime_error("layer-0 HC F32 inject geometry mismatch");
        }
        e = cudaMemcpyAsync(scratch.inject_weights(), w.debug_inject_f32.data(),
                            w.debug_inject_f32.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error("copy layer-0 HC F32 inject weights failed");
        q38_hc_launch_dense_gemv_f32(
            scratch.inject_weights(), scratch.norm(), scratch.inject(), hc, hc_dim, stream);
    }
}

inline void run_qwen38_hc_q8_0_f32_mix_from_residual(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;
    const int low_rank = scratch.low_rank();
    if (w.debug_norm_f32.size() != static_cast<size_t>(hc_dim) ||
        !w.down.lowbit || w.down.type != GgmlType::Q8_0 ||
        !w.up.lowbit || w.up.type != GgmlType::Q8_0 ||
        w.down.cols != hc_dim || w.down.rows != low_rank ||
        w.up.cols != low_rank || w.up.rows != hc_dim) {
        throw std::runtime_error("exact Q8_0 HC replay weight geometry/type mismatch");
    }

    cudaError_t e = cudaMemcpyAsync(
        scratch.gamma(), w.debug_norm_f32.data(), static_cast<size_t>(hc_dim) * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy exact Q8_0 HC norm weights failed");
    q38_hc_group_rmsnorm_f32_kernel<<<hc, 256, 0, stream>>>(
        residual, scratch.gamma(), scratch.norm(), hidden, hc, eps);

    constexpr int q8_threads = 256;
    constexpr int warps = q8_threads / 32;
    const int down_blocks = hc_dim / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(down_blocks + warps - 1) / warps,
                                              q8_threads, 0, stream>>>(
        scratch.norm(), hc_dim, scratch.q8_0_values(), scratch.q8_0_scales());
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(low_rank + warps - 1) / warps,
                                               q8_threads, 0, stream>>>(
        w.down.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
        scratch.low_raw(), low_rank, hc_dim);
    e = cudaMemcpyAsync(scratch.low(), scratch.low_raw(),
                        static_cast<size_t>(low_rank) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy exact Q8_0 HC down projection failed");
    q38_hc_silu_scale_ggml_f32_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
        scratch.low(), low_rank, 1.0f / static_cast<float>(hc));

    const int low_blocks = low_rank / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(low_blocks + warps - 1) / warps,
                                              q8_threads, 0, stream>>>(
        scratch.low(), low_rank, scratch.q8_0_values(), scratch.q8_0_scales());
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps,
                                               q8_threads, 0, stream>>>(
        w.up.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
        scratch.gate(), hc_dim, low_rank);
    q38_hc_mix_f32_cpu_order_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        scratch.norm(), scratch.gate(), scratch.mixed(), hidden, hc);
}

// Exact Q8_0 HC mix for the F32 inject path. The grouped RMSNorm, the Q8_0
// activation quantisation, the down/up exact GEMV, the ggml SiLU and the F32
// stream mix are exactly those of run_qwen38_hc_q8_0_f32_mix_from_residual(),
// and the inject projection reuses the same F32 norm with the unrounded F32
// weights that load_qwen38_hc_weights() left resident on the device. The [hc]
// inject result stays F32 in scratch.inject(), which is what
// run_qwen38_hc_combine_f32_inject() consumes; the legacy half-typed
// scratch.inject() is left untouched.
// Throws when the caller ignored qwen38_hc_f32_inject_ready().
inline void run_qwen38_hc_q8_0_f32_mix_with_inject_from_residual(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hc = scratch.hc();
    const int hc_dim = scratch.hidden() * hc;
    if (!qwen38_hc_f32_inject_ready(w)) {
        throw std::runtime_error("HC F32 inject weights unavailable");
    }
    if (w.inject_f32.size != static_cast<size_t>(hc_dim) * static_cast<size_t>(hc)) {
        throw std::runtime_error("HC F32 inject weight geometry mismatch");
    }
    run_qwen38_hc_q8_0_f32_mix_from_residual(residual, w, scratch, eps, stream);

    // Same dense F32 GEMV the legacy dense path uses, but reading the F32
    // weights and the F32 norm, writing the unrounded [hc] result, and - with
    // WARODECODE_HC_INJECT_F32_AVX2 (default ON) - in the ggml_vec_dot_f32 order
    // the CPU oracle uses for the F32 inject weight.
    q38_hc_launch_dense_gemv_f32(
        w.inject_f32.data, scratch.norm(), scratch.inject(), hc, hc_dim, stream);
}

inline void run_qwen38_hc_f32_mix_from_half_residual(
        const __half * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hc_dim = scratch.hidden() * scratch.hc();
    q38_hc_copy_half_to_f32_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
        residual, scratch.residual(), hc_dim);
    run_qwen38_hc_f32_mix_from_residual(scratch.residual(), w, scratch, eps, stream);
}

inline void run_qwen38_hc_q8_0_f32_mix_from_half_residual(
        const __half * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;
    const int low_rank = scratch.low_rank();
    if (w.debug_norm_f32.size() != static_cast<size_t>(hc_dim) ||
        !w.down.lowbit || w.down.type != GgmlType::Q8_0 ||
        !w.up.lowbit || w.up.type != GgmlType::Q8_0 ||
        w.down.cols != hc_dim || w.down.rows != low_rank ||
        w.up.cols != low_rank || w.up.rows != hc_dim) {
        throw std::runtime_error("exact Q8_0 HC replay weight geometry/type mismatch");
    }

    cudaError_t e = cudaMemcpyAsync(
        scratch.gamma(), w.debug_norm_f32.data(), static_cast<size_t>(hc_dim) * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy exact Q8_0 HC norm weights failed");
    q38_hc_copy_half_to_f32_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
        residual, scratch.residual(), hc_dim);
    q38_hc_group_rmsnorm_f32_kernel<<<hc, 256, 0, stream>>>(
        scratch.residual(), scratch.gamma(), scratch.norm(), hidden, hc, eps);

    constexpr int q8_threads = 256;
    constexpr int warps = q8_threads / 32;
    const int down_blocks = hc_dim / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(down_blocks + warps - 1) / warps,
                                              q8_threads, 0, stream>>>(
        scratch.norm(), hc_dim, scratch.q8_0_values(), scratch.q8_0_scales());
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(low_rank + warps - 1) / warps,
                                               q8_threads, 0, stream>>>(
        w.down.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
        scratch.low_raw(), low_rank, hc_dim);
    e = cudaMemcpyAsync(scratch.low(), scratch.low_raw(),
                        static_cast<size_t>(low_rank) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error("copy exact Q8_0 HC down projection failed");
    q38_hc_silu_scale_ggml_f32_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
        scratch.low(), low_rank, 1.0f / static_cast<float>(hc));

    const int low_blocks = low_rank / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(low_blocks + warps - 1) / warps,
                                              q8_threads, 0, stream>>>(
        scratch.low(), low_rank, scratch.q8_0_values(), scratch.q8_0_scales());
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps,
                                               q8_threads, 0, stream>>>(
        w.up.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
        scratch.gate(), hc_dim, low_rank);
    q38_hc_mix_f32_cpu_order_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        scratch.norm(), scratch.gate(), scratch.mixed(), hidden, hc);
}

inline void run_qwen38_hc_f32_debug(
        const LowBitMatrix & token_embd,
        int token,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & scratch,
        float eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    if (token_embd.type != GgmlType::Q8_0 || token_embd.cols != hidden ||
        token < 0 || token >= token_embd.rows) {
        throw std::runtime_error("layer-0 HC F32 debug embedding mismatch");
    }

    constexpr int threads = 256;
    const int hc_dim = hidden * scratch.hc();
    q38_hc_q8_0_gather_row_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
        token_embd.data, hidden, token, scratch.embedding());
    q38_hc_repeat_f32_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.embedding(), scratch.residual(), hidden, scratch.hc());
    run_qwen38_hc_f32_mix_from_residual(scratch.residual(), w, scratch, eps, stream);
}

__global__ void q38_hc_combine_kernel(
        __half * __restrict__ residual,
        const __half * __restrict__ block_out,
        const __half * __restrict__ inject,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;

    const int c = i / hidden;
    const int h = i - c * hidden;
    const float z = __half2float(inject[c]) / static_cast<float>(hc);
    const float w = 2.0f / (1.0f + expf(-z));
    const float r = __half2float(residual[i]);
    const float b = __half2float(block_out[h]);
    residual[i] = __float2half(fmaf(b, w, r));
}

// F32 residual variant: accumulate the branch output into an F32 residual
// stream (matching the pinned CPU reference) instead of rounding to F16 here.
__global__ void q38_hc_combine_f32_residual_kernel(
        float * __restrict__ residual,
        const __half * __restrict__ block_out,
        const __half * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;

    const int c = i / hidden;
    const int h = i - c * hidden;
    const float z = __half2float(inject[c]) / static_cast<float>(hc);
    const float w = 2.0f / (1.0f + expf(-z));
    const float b = __half2float(block_out[h]);
    const float v = fmaf(b, w, residual[i]);
    // round_half reproduces the legacy F16 residual lifetime for ablation.
    residual[i] = round_half ? __half2float(__float2half(v)) : v;
}

// F32 MoE candidate: preserve the routed sum through shared-expert addition,
// then preserve the FFN branch through the HC residual multiply/add. The
// optional F16 write reproduces the legacy residual lifetime for ablation.
__global__ void q38_hc_combine_moe_f32_residual_kernel(
        float * __restrict__ residual,
        const float * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ shared_gate,
        const __half * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;

    const int c = i / hidden;
    const int h = i - c * hidden;
    const float branch = q38_moe_branch_f32(
        routed[h], __half2float(shared[h]), shared_gate[0]);
    const float z = __half2float(inject[c]) / static_cast<float>(hc);
    const float weight = 2.0f / (1.0f + expf(-z));
    const float value = q38_residual_add_f32(residual[i], branch, weight);
    residual[i] = round_half ? q38_round_half(value) : value;
}

// F32-inject variants of the F32-residual combine kernels. The pinned CPU
// reference derives the scatter weight from the unrounded inject projection:
//   w = 2 * sigmoid(inject * (1 / hc))
// (ggml_scale(inject, 1/hc) -> ggml_sigmoid -> ggml_scale(w, 2), each a single
// F32 multiply/divide/add step), then adds the block output scaled by it.
// The legacy kernels above instead read a F16-rounded inject and divide by hc.
// For hc = 4 both orders are bit-identical (scaling by a power of two is
// exact); the ggml order is kept so other hyper-connection counts stay faithful.
__global__ void q38_hc_combine_f32_residual_f32_inject_kernel(
        float * __restrict__ residual,
        const __half * __restrict__ block_out,
        const float * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;

    const int c = i / hidden;
    const int h = i - c * hidden;
    const float z = __fmul_rn(inject[c], 1.0f / static_cast<float>(hc));
    const float sigmoid = q38_exact_sigmoid_f32(z, q38_exact_exp_on());
    const float weight = __fmul_rn(2.0f, sigmoid);
    const float branch = __half2float(block_out[h]);
    const float value = q38_residual_add_f32(residual[i], branch, weight);
    // round_half reproduces the legacy F16 residual lifetime for ablation.
    residual[i] = round_half ? __half2float(__float2half(value)) : value;
}

// MoE branch variant of the F32-inject combine: the routed sum keeps F32
// precision through the shared-expert addition, exactly like
// q38_hc_combine_moe_f32_residual_kernel, but the scatter weight comes from the
// unrounded F32 inject projection.
__global__ void q38_hc_combine_moe_f32_residual_f32_inject_kernel(
        float * __restrict__ residual,
        const float * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ shared_gate,
        const float * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;

    const int c = i / hidden;
    const int h = i - c * hidden;
    const float branch = q38_moe_branch_f32(
        routed[h], __half2float(shared[h]), shared_gate[0]);
    const float z = __fmul_rn(inject[c], 1.0f / static_cast<float>(hc));
    const float sigmoid = q38_exact_sigmoid_f32(z, q38_exact_exp_on());
    const float weight = __fmul_rn(2.0f, sigmoid);
    const float value = q38_residual_add_f32(residual[i], branch, weight);
    residual[i] = round_half ? q38_round_half(value) : value;
}

class Qwen38HcScratch {
public:
    Qwen38HcScratch() = default;
    Qwen38HcScratch(const Qwen38HcScratch &) = delete;
    Qwen38HcScratch & operator=(const Qwen38HcScratch &) = delete;
    ~Qwen38HcScratch() { reset(); }

    void initialize(int hidden, int hc, int low_rank) {
        if (xn_) throw std::runtime_error("HC scratch already initialized");
        if (hidden <= 0 || hc <= 1 || low_rank <= 0) throw std::runtime_error("invalid HC geometry");
        hidden_ = hidden;
        hc_ = hc;
        low_rank_ = low_rank;

        alloc(reinterpret_cast<void **>(&xn_), static_cast<size_t>(hidden_) * hc_ * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(hidden_) * hc_ * sizeof(__half));
        alloc(reinterpret_cast<void **>(&lo_), static_cast<size_t>(low_rank_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&inject_), static_cast<size_t>(hc_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&mixed_), static_cast<size_t>(hidden_) * sizeof(__half));
    }

    int hidden() const { return hidden_; }
    int hc() const { return hc_; }
    int low_rank() const { return low_rank_; }
    __half * xn() const { return xn_; }
    __half * gate() const { return gate_; }
    __half * lo() const { return lo_; }
    __half * inject() const { return inject_; }
    __half * mixed() const { return mixed_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc HC scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (xn_) cudaFree(xn_);
        if (gate_) cudaFree(gate_);
        if (lo_) cudaFree(lo_);
        if (inject_) cudaFree(inject_);
        if (mixed_) cudaFree(mixed_);
        xn_ = gate_ = lo_ = inject_ = mixed_ = nullptr;
    }

    int hidden_ = 0;
    int hc_ = 0;
    int low_rank_ = 0;
    __half * xn_ = nullptr;
    __half * gate_ = nullptr;
    __half * lo_ = nullptr;
    __half * inject_ = nullptr;
    __half * mixed_ = nullptr;
};

inline __half * run_qwen38_hc_mix(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcScratch & scratch,
        float rms_eps,
        cudaStream_t stream,
        float * trace_down_f32 = nullptr,
        float * trace_low_pre_activation = nullptr,
        float * trace_gate_f32 = nullptr,
        const float * norm_override_f32 = nullptr) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;

    if (w.norm.size != static_cast<size_t>(hc_dim) ||
        w.down.cols != hc_dim || w.down.rows != scratch.low_rank() ||
        w.up.cols != scratch.low_rank() || w.up.rows != hc_dim) {
        throw std::runtime_error("HC weight geometry mismatch");
    }

    if (norm_override_f32) {
        q38_hc_copy_f32_to_half_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
            norm_override_f32, scratch.xn(), hc_dim);
    } else {
        q38_hc_group_rmsnorm_f32in_kernel<<<hc, 256, 0, stream>>>(
            residual, w.norm.data, scratch.xn(), hidden, hc, rms_eps);
    }
    launch_matrix_gemv_half(w.down, scratch.xn(), scratch.lo(), stream);
    if (trace_down_f32) launch_matrix_gemv_float(w.down, scratch.xn(), trace_down_f32, stream);
    if (trace_low_pre_activation) {
        q38_hc_copy_half_to_f32_kernel<<<(scratch.low_rank() + 255) / 256, 256, 0, stream>>>(
            scratch.lo(), trace_low_pre_activation, scratch.low_rank());
    }

    const int lo_blocks = (scratch.low_rank() + 255) / 256;
    q38_hc_silu_scale_kernel<<<lo_blocks, 256, 0, stream>>>(
        scratch.lo(), scratch.low_rank(), 1.0f / static_cast<float>(hc));

    launch_matrix_gemv_half(w.up, scratch.lo(), scratch.gate(), stream);
    if (trace_gate_f32) launch_matrix_gemv_float(w.up, scratch.lo(), trace_gate_f32, stream);

    const int hidden_blocks = (hidden + 255) / 256;
    q38_hc_mix_kernel<<<hidden_blocks, 256, 0, stream>>>(
        scratch.xn(), scratch.gate(), scratch.mixed(), hidden, hc);

    if (w.has_inject) {
        if (w.inject.cols != hc_dim || w.inject.rows != hc) {
            throw std::runtime_error("HC inject geometry mismatch");
        }
        launch_matrix_gemv_half(w.inject, scratch.xn(), scratch.inject(), stream);
    }
    return scratch.mixed();
}

// CPU-compatible F32 HC mix: run the grouped RMSNorm, low-rank down/up
// projections, SiLU and stream mixing in F32 (matching the pinned CPU
// reference) instead of F16, then convert the half-typed router/gate inputs
// back for the rest of the pipeline. Returns nullptr when the weights do not
// support the F32 path so the caller can fall back to the F16 path.
inline __half * run_qwen38_hc_mix_f32(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & f32,
        Qwen38HcScratch & scratch,
        float rms_eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;
    if (!w.down.lowbit || w.down.type != GgmlType::Q8_0 ||
        !w.up.lowbit || w.up.type != GgmlType::Q8_0 ||
        w.debug_norm_f32.size() != static_cast<size_t>(hc_dim) ||
        w.down.cols != hc_dim || w.down.rows != scratch.low_rank() ||
        w.up.cols != scratch.low_rank() || w.up.rows != hc_dim) {
        return nullptr;
    }
    run_qwen38_hc_f32_mix_from_residual(residual, w, f32, rms_eps, stream);
    q38_hc_copy_f32_to_half_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        f32.mixed(), scratch.mixed(), hidden);
    if (w.has_inject) {
        if (w.inject.lowbit) {
            return nullptr;
        }
        launch_matrix_gemv_f32_input(w.inject, f32.norm(), f32.inject(), stream);
        q38_hc_copy_f32_to_half_kernel<<<(hc + 255) / 256, 256, 0, stream>>>(
            f32.inject(), scratch.inject(), hc);
    }
    return scratch.mixed();
}

// Drop-in F32-inject sibling of run_qwen38_hc_mix_f32. It runs the exact Q8_0
// replay plus the F32 inject projection and copies the mixed stream back to the
// half scratch the rest of the pipeline consumes; the unrounded [hc] inject
// stays in f32.inject() for run_qwen38_hc_combine_f32_inject(). Returns nullptr
// when WARODECODE_HC_INJECT_F32=0 or the weights do not support the path, so
// callers can keep the existing F16 fallback.
inline __half * run_qwen38_hc_mix_f32_with_inject(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & f32,
        Qwen38HcScratch & scratch,
        float rms_eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;
    if (f32.hidden() != hidden || f32.hc() != hc || f32.low_rank() != scratch.low_rank()) {
        throw std::runtime_error("HC F32 scratch geometry mismatch");
    }
    if (!w.down.lowbit || w.down.type != GgmlType::Q8_0 ||
        !w.up.lowbit || w.up.type != GgmlType::Q8_0 ||
        w.debug_norm_f32.size() != static_cast<size_t>(hc_dim) ||
        w.down.cols != hc_dim || w.down.rows != scratch.low_rank() ||
        w.up.cols != scratch.low_rank() || w.up.rows != hc_dim ||
        !qwen38_hc_f32_inject_enabled(w)) {
        return nullptr;
    }
    run_qwen38_hc_q8_0_f32_mix_with_inject_from_residual(residual, w, f32, rms_eps, stream);
    q38_hc_copy_f32_to_half_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        f32.mixed(), scratch.mixed(), hidden);
    // Keep the legacy half-typed inject buffer consistent too (traces and any
    // leftover half-inject combine read it); the F32 result above is the one
    // run_qwen38_hc_combine_f32_inject() must use.
    q38_hc_copy_f32_to_half_kernel<<<(hc + 255) / 256, 256, 0, stream>>>(
        f32.inject(), scratch.inject(), hc);
    return scratch.mixed();
}

inline void run_qwen38_hc_combine(
        float * residual,
        const __half * block_out,
        const Qwen38HcScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_f32_residual_kernel<<<blocks, 256, 0, stream>>>(
        residual, block_out, scratch.inject(), scratch.hidden(), scratch.hc(), round_half);
}

inline void run_qwen38_hc_combine_moe_f32(
        float * residual,
        const float * routed,
        const __half * shared,
        const float * shared_gate,
        const Qwen38HcScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_moe_f32_residual_kernel<<<blocks, 256, 0, stream>>>(
        residual, routed, shared, shared_gate, scratch.inject(),
        scratch.hidden(), scratch.hc(), round_half);
}

// F32-inject combine: consumes the unrounded [hc] inject that
// run_qwen38_hc_q8_0_f32_mix_with_inject_from_residual() (or
// run_qwen38_hc_mix_f32_with_inject()) wrote into the F32 scratch, and applies
// the ggml-ordered scatter weight in F32. The half-inject wrappers above stay
// available for the legacy paths.
// The GDN / MoE F32 paths produce an unrounded branch output, so the combine
// needs variants that never round it through __half. Both inject flavours are
// provided because the attn branch can fall back to the half inject.
__global__ void q38_hc_combine_f32_residual_f32_block_kernel(
        float * __restrict__ residual,
        const float * __restrict__ block_out,
        const __half * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    const int c = i / hidden;
    const int h = i - c * hidden;
    const float z = __half2float(inject[c]) / static_cast<float>(hc);
    const float w = 2.0f / (1.0f + expf(-z));
    const float v = fmaf(block_out[h], w, residual[i]);
    residual[i] = round_half ? __half2float(__float2half(v)) : v;
}

__global__ void q38_hc_combine_f32_residual_f32_block_f32_inject_kernel(
        float * __restrict__ residual,
        const float * __restrict__ block_out,
        const float * __restrict__ inject,
        int hidden,
        int hc,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    const int c = i / hidden;
    const int h = i - c * hidden;
    const float z = __fmul_rn(inject[c], 1.0f / static_cast<float>(hc));
    const float s = q38_exact_sigmoid_f32(z, q38_exact_exp_on());
    const float w = __fmul_rn(2.0f, s);
    const float v = q38_residual_add_f32(residual[i], block_out[h], w);
    residual[i] = round_half ? q38_round_half(v) : v;
}

inline void run_qwen38_hc_combine_f32_block(
        float * residual,
        const float * block_out,
        const Qwen38HcScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_f32_residual_f32_block_kernel<<<blocks, 256, 0, stream>>>(
        residual, block_out, scratch.inject(), scratch.hidden(), scratch.hc(), round_half);
}

inline void run_qwen38_hc_combine_f32_block_f32_inject(
        float * residual,
        const float * block_out,
        const Qwen38HcF32DebugScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_f32_residual_f32_block_f32_inject_kernel<<<blocks, 256, 0, stream>>>(
        residual, block_out, scratch.inject(), scratch.hidden(), scratch.hc(), round_half);
}

inline void run_qwen38_hc_combine_f32_inject(
        float * residual,
        const __half * block_out,
        const Qwen38HcF32DebugScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_f32_residual_f32_inject_kernel<<<blocks, 256, 0, stream>>>(
        residual, block_out, scratch.inject(), scratch.hidden(), scratch.hc(), round_half);
}

inline void run_qwen38_hc_combine_moe_f32_inject(
        float * residual,
        const float * routed,
        const __half * shared,
        const float * shared_gate,
        const Qwen38HcF32DebugScratch & scratch,
        cudaStream_t stream,
        bool round_half = false) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_moe_f32_residual_f32_inject_kernel<<<blocks, 256, 0, stream>>>(
        residual, routed, shared, shared_gate, scratch.inject(),
        scratch.hidden(), scratch.hc(), round_half);
}

} // namespace wd
