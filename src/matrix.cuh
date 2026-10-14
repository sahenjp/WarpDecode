#pragma once

#include "half_tensor.cuh"
#include "lowbit.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace wd {

struct GpuMatrix {
    int rows = 0;
    int cols = 0;
    GgmlType type = GgmlType::F32;
    bool lowbit = false;
    LowBitMatrix low;
    DeviceHalfArray dense;
    // Filled for F32 tensors only: ggml computes those with F32 weights and F32
    // activations, so the F16 copy above is not accurate enough for parity.
    DeviceFloatArray dense_f32;

    GpuMatrix() = default;
    GpuMatrix(const GpuMatrix &) = delete;
    GpuMatrix & operator=(const GpuMatrix &) = delete;
    GpuMatrix(GpuMatrix &&) noexcept = default;
    GpuMatrix & operator=(GpuMatrix &&) noexcept = default;
};

inline GpuMatrix upload_matrix(const TensorInfo & t) {
    if (t.dims.size() != 2) {
        throw std::runtime_error("matrix tensor must be 2D: " + t.name);
    }

    GpuMatrix out;
    out.cols = static_cast<int>(t.dims[0]);
    out.rows = static_cast<int>(t.dims[1]);
    out.type = t.type;

    if (lowbit_gemv_supported(t.type)) {
        out.lowbit = true;
        out.low = upload_lowbit_matrix(t);
    } else if (t.type == GgmlType::F16 || t.type == GgmlType::BF16 || t.type == GgmlType::F32) {
        out.lowbit = false;
        out.dense = upload_half_array(t);
        if (t.type == GgmlType::F32) {
            out.dense_f32 = upload_float_array(t);
        }
    } else {
        throw std::runtime_error("unsupported matrix tensor type: " + t.name);
    }
    return out;
}

template <bool OUT_F32>
__global__ void dense_half_gemv_kernel(
        const __half * __restrict__ w,
        const __half * __restrict__ x,
        void * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const __half * row_ptr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        acc = fmaf(__half2float(row_ptr[c]), __half2float(x[c]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if constexpr (OUT_F32) static_cast<float *>(y)[row] = acc;
        else static_cast<__half *>(y)[row] = __float2half(acc);
    }
}

__global__ void dense_half_gemv_f32_input_kernel(
        const __half * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const __half * row_ptr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        acc = fmaf(__half2float(row_ptr[c]), x[c], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

inline void launch_matrix_gemv_half(
        const GpuMatrix & w,
        const __half * x,
        __half * y,
        cudaStream_t stream,
        int threads = 256) {
    if (w.lowbit) {
        launch_lowbit_gemv_half(w.low, x, y, stream, threads);
        return;
    }
    if (threads < 32 || threads % 32 != 0) throw std::runtime_error("matrix GEMV threads must be a multiple of 32");
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;
    dense_half_gemv_kernel<false><<<blocks, threads, 0, stream>>>(
        w.dense.data, x, y, w.rows, w.cols);
}

inline void launch_matrix_gemv_float(
        const GpuMatrix & w,
        const __half * x,
        float * y,
        cudaStream_t stream,
        int threads = 256) {
    if (w.lowbit) {
        launch_lowbit_gemv_float(w.low, x, y, stream, threads);
        return;
    }
    if (threads < 32 || threads % 32 != 0) throw std::runtime_error("matrix GEMV threads must be a multiple of 32");
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;
    dense_half_gemv_kernel<true><<<blocks, threads, 0, stream>>>(
        w.dense.data, x, y, w.rows, w.cols);
}

inline void launch_matrix_gemv_f32_input(
        const GpuMatrix & w,
        const float * x,
        float * y,
        cudaStream_t stream,
        int threads = 256) {
    if (w.lowbit) throw std::runtime_error("F32-input GEMV requires a dense matrix");
    if (threads < 32 || threads % 32 != 0) throw std::runtime_error("matrix GEMV threads must be a multiple of 32");
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;
    dense_half_gemv_f32_input_kernel<<<blocks, threads, 0, stream>>>(
        w.dense.data, x, y, w.rows, w.cols);
}

// ggml runs non-quantised tensors with vec_dot_type F32: both the weights and the
// activation stay F32, so neither the F16 weight copy nor an F16 activation may
// be used for those projections (the F32 router, GDN beta/alpha, HC inject ...).
__global__ void dense_f32_gemv_kernel(
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
    for (int c = lane; c < cols; c += 32) acc = fmaf(row_ptr[c], x[c], acc);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// WARODECODE_ROUTER_F32_AVX2 (default ON).
//
// An F32 weight is multiplied by an F32 activation on the CPU
// (type_traits_cpu[GGML_TYPE_F32].vec_dot_type == GGML_TYPE_F32, nrows == 1) and
// a single decode token skips llamafile_sgemm(), so every row of the F32 router
// is one ggml_vec_dot_f32: 32-value blocks of four 8-lane FMA accumulators, the
// GGML_F32x8_REDUCE tree, then a non-fused leftover tail.
//
// dense_f32_gemv_kernel() instead sums lane-strided (i = lane, lane + 32, ...)
// and reduces with a shfl_down tree, which is a different summation order and
// therefore different bits from the oracle. The AVX2 kernel below replaces it
// for the F32 router (WARODECODE_ROUTER_F32); WARODECODE_ROUTER_F32_AVX2=0
// restores the previous warp order for A/B runs.
inline bool q38_matrix_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_matrix_f32_avx2_enabled() {
    static const bool enabled = q38_matrix_env_flag("WARODECODE_ROUTER_F32_AVX2", true);
    return enabled;
}

// Tree reduction half of ggml_vec_dot_f32 on AVX2+FMA, identical to
// q38_gdn_reduce_avx2() / q38_moe_reduce_avx2(). Callers fill sum[4][8] in AVX2
// visit order: base in {0, 32, 64, 96}, stream j in 0..3, lane l in 0..7,
// i = base + j*8 + l with FMA accumulation.
__device__ __forceinline__ float q38_matrix_reduce_avx2(float sum[4][8]) {
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

// One ggml_vec_dot_f32 per row, one thread per row.
__global__ void dense_f32_gemv_avx2_kernel(
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
    float acc = q38_matrix_reduce_avx2(sum);
    // ggml's leftover tail is the non-fused "sumf += x[i]*y[i]" chain; the model
    // router is 2560 wide (a multiple of 32), so it never runs there.
    for (int c = np; c < cols; ++c) acc = __fadd_rn(acc, __fmul_rn(row_ptr[c], x[c]));
    y[row] = acc;
}

inline bool matrix_has_f32_weights(const GpuMatrix & w) {
    return !w.lowbit && w.type == GgmlType::F32 && w.dense_f32.data != nullptr;
}

inline void launch_matrix_gemv_f32_weights(
        const GpuMatrix & w,
        const float * x,
        float * y,
        cudaStream_t stream,
        int threads = 256) {
    if (!matrix_has_f32_weights(w)) {
        throw std::runtime_error("F32-weight GEMV requires an F32 dense matrix");
    }
    if (threads < 32 || threads % 32 != 0) throw std::runtime_error("matrix GEMV threads must be a multiple of 32");
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;
    if (q38_matrix_f32_avx2_enabled()) {
        // One thread per row (not one warp per row), so the grid follows the
        // thread count rather than the warp count.
        dense_f32_gemv_avx2_kernel<<<(w.rows + threads - 1) / threads, threads, 0, stream>>>(
            w.dense_f32.data, x, y, w.rows, w.cols);
        return;
    }
    dense_f32_gemv_kernel<<<blocks, threads, 0, stream>>>(
        w.dense_f32.data, x, y, w.rows, w.cols);
}

} // namespace wd
