#pragma once

#include "gguf.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace wd {

struct LowBitMatrix {
    int rows = 0;
    int cols = 0;
    GgmlType type = GgmlType::F32;
    uint8_t * data = nullptr;
    size_t bytes = 0;

    LowBitMatrix() = default;
    LowBitMatrix(const LowBitMatrix &) = delete;
    LowBitMatrix & operator=(const LowBitMatrix &) = delete;

    LowBitMatrix(LowBitMatrix && o) noexcept
        : rows(o.rows), cols(o.cols), type(o.type), data(o.data), bytes(o.bytes) {
        o.data = nullptr;
        o.rows = o.cols = 0;
        o.bytes = 0;
    }

    LowBitMatrix & operator=(LowBitMatrix && o) noexcept {
        if (this != &o) {
            reset();
            rows = o.rows;
            cols = o.cols;
            type = o.type;
            data = o.data;
            bytes = o.bytes;
            o.data = nullptr;
            o.rows = o.cols = 0;
            o.bytes = 0;
        }
        return *this;
    }

    ~LowBitMatrix() { reset(); }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        rows = cols = 0;
        bytes = 0;
    }
};

inline bool lowbit_gemv_supported(GgmlType type) {
    return type == GgmlType::Q8_0 || type == GgmlType::IQ4_NL;
}

inline LowBitMatrix upload_lowbit_matrix(const TensorInfo & t) {
    if (t.dims.size() != 2) {
        throw std::runtime_error("low-bit GEMV requires a 2D tensor: " + t.name);
    }
    if (!lowbit_gemv_supported(t.type)) {
        throw std::runtime_error("unsupported low-bit GEMV tensor type: " + t.name);
    }

    LowBitMatrix out;
    out.cols = static_cast<int>(t.dims[0]);
    out.rows = static_cast<int>(t.dims[1]);
    out.type = t.type;
    out.bytes = t.byte_size();

    if (out.cols <= 0 || out.rows <= 0 || out.cols % 32 != 0) {
        throw std::runtime_error("invalid low-bit matrix shape: " + t.name);
    }

    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), out.bytes);
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc low-bit matrix failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, t.data, out.bytes, cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy low-bit matrix failed: ") + cudaGetErrorString(c));
    }
    return out;
}

__device__ __forceinline__ float lowbit_half(const uint8_t * p) {
    const __half h = *reinterpret_cast<const __half *>(p);
    return __half2float(h);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float iq4_nl_value(uint8_t q) {
    // ggml IQ4_NL codebook.
    constexpr int8_t lut[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return static_cast<float>(lut[q & 0x0f]);
}

template <bool OUT_F32>
__global__ void q8_0_gemv_kernel(
        const uint8_t * __restrict__ w,
        const __half * __restrict__ x,
        void * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int blocks_per_row = cols >> 5;
    const size_t row_stride = static_cast<size_t>(blocks_per_row) * 34u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 34u;
        const float d = lowbit_half(blk);
        const int8_t q = *reinterpret_cast<const int8_t *>(blk + 2 + lane);
        acc = fmaf(d * static_cast<float>(q), __half2float(x[(b << 5) + lane]), acc);
    }
    acc = warp_sum(acc);

    if (lane == 0) {
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = acc;
        } else {
            static_cast<__half *>(y)[row] = __float2half(acc);
        }
    }
}

template <bool OUT_F32>
__global__ void iq4_nl_gemv_kernel(
        const uint8_t * __restrict__ w,
        const __half * __restrict__ x,
        void * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int blocks_per_row = cols >> 5;
    const size_t row_stride = static_cast<size_t>(blocks_per_row) * 18u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 18u;
        const float d = lowbit_half(blk);
        const uint8_t packed = blk[2 + (lane & 15)];
        const uint8_t q = lane < 16 ? (packed & 0x0f) : (packed >> 4);
        acc = fmaf(d * iq4_nl_value(q), __half2float(x[(b << 5) + lane]), acc);
    }
    acc = warp_sum(acc);

    if (lane == 0) {
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = acc;
        } else {
            static_cast<__half *>(y)[row] = __float2half(acc);
        }
    }
}

inline void launch_lowbit_gemv_half(
        const LowBitMatrix & w,
        const __half * x,
        __half * y,
        cudaStream_t stream,
        int threads = 256) {
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("low-bit GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;

    if (w.type == GgmlType::Q8_0) {
        q8_0_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ4_NL) {
        iq4_nl_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else {
        throw std::runtime_error("unsupported low-bit GEMV type");
    }
}

inline void launch_lowbit_gemv_float(
        const LowBitMatrix & w,
        const __half * x,
        float * y,
        cudaStream_t stream,
        int threads = 256) {
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("low-bit GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;

    if (w.type == GgmlType::Q8_0) {
        q8_0_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ4_NL) {
        iq4_nl_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else {
        throw std::runtime_error("unsupported low-bit GEMV type");
    }
}

} // namespace wd
