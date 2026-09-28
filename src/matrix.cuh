#pragma once

#include "half_tensor.cuh"
#include "lowbit.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

} // namespace wd
