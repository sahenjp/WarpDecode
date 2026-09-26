#pragma once

#include "gguf.hpp"
#include "iq2s_grid.cuh"
#include "iq3s_grid.cuh"

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
    bool owns = true;

    LowBitMatrix() = default;
    LowBitMatrix(const LowBitMatrix &) = delete;
    LowBitMatrix & operator=(const LowBitMatrix &) = delete;

    LowBitMatrix(LowBitMatrix && o) noexcept
        : rows(o.rows), cols(o.cols), type(o.type), data(o.data), bytes(o.bytes), owns(o.owns) {
        o.data = nullptr;
        o.rows = o.cols = 0;
        o.bytes = 0;
        o.owns = true;
    }

    LowBitMatrix & operator=(LowBitMatrix && o) noexcept {
        if (this != &o) {
            reset();
            rows = o.rows;
            cols = o.cols;
            type = o.type;
            data = o.data;
            bytes = o.bytes;
            owns = o.owns;
            o.data = nullptr;
            o.rows = o.cols = 0;
            o.bytes = 0;
            o.owns = true;
        }
        return *this;
    }

    ~LowBitMatrix() { reset(); }

    void reset() {
        if (data && owns) cudaFree(data);
        data = nullptr;
        rows = cols = 0;
        bytes = 0;
        owns = true;
    }
};

inline bool lowbit_gemv_supported(GgmlType type) {
    return type == GgmlType::Q8_0 || type == GgmlType::Q6_K || type == GgmlType::IQ4_NL || type == GgmlType::IQ2_S || type == GgmlType::IQ3_S;
}

inline LowBitMatrix borrow_lowbit_matrix(
        uint8_t * data,
        int rows,
        int cols,
        GgmlType type,
        size_t bytes) {
    LowBitMatrix out;
    out.data = data;
    out.rows = rows;
    out.cols = cols;
    out.type = type;
    out.bytes = bytes;
    out.owns = false;
    return out;
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

    const int quantum = (t.type == GgmlType::Q6_K || t.type == GgmlType::IQ2_S || t.type == GgmlType::IQ3_S) ? 256 : 32;
    if (out.cols <= 0 || out.rows <= 0 || out.cols % quantum != 0) {
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


__device__ __forceinline__ uint8_t iq2s_grid_value(uint16_t index, int lane8) {
    const uint64_t packed = k_iq2s_grid[index];
    return static_cast<uint8_t>(packed >> (lane8 * 8));
}

template <bool OUT_F32>
__global__ void iq2_s_gemv_kernel(
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

    const int superblocks_per_row = cols >> 8; // 256 values per IQ2_S block
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 82u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    const int group = lane >> 3;
    const int lane8 = lane & 7;
    float acc = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        const float d = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;

#pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t scale_byte = scales[ib32];
            const uint8_t scale4 = group < 2 ? (scale_byte & 0x0f) : (scale_byte >> 4);
            const float dl = d * (0.5f + static_cast<float>(scale4)) * 0.25f;

            const uint8_t qlo = qs[ib32 * 4 + group];
            const uint16_t qhi = static_cast<uint16_t>(
                (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u);
            const uint16_t grid_index = static_cast<uint16_t>(qlo) | qhi;
            float qv = static_cast<float>(iq2s_grid_value(grid_index, lane8));

            const uint8_t sign_bits = signs[ib32 * 4 + group];
            if (sign_bits & (1u << lane8)) qv = -qv;

            const int col = (sb << 8) + (ib32 << 5) + lane;
            acc = fmaf(dl * qv, __half2float(x[col]), acc);
        }
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


__device__ __forceinline__ uint8_t iq3s_grid_value(uint16_t index, int lane4) {
    const uint32_t packed = k_iq3s_grid[index];
    return static_cast<uint8_t>(packed >> (lane4 * 8));
}

template <bool OUT_F32>
__global__ void iq3_s_gemv_kernel(
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

    const int superblocks_per_row = cols >> 8; // 256 values per IQ3_S block
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 110u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    const int group = lane >> 3;  // 4 groups of 8
    const int lane8 = lane & 7;
    float acc = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 110u;
        const float d = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;

#pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * qs32 = qs + ib32 * 8;
            const uint8_t qh32 = qh[ib32];
            const uint8_t sign_bits = signs[ib32 * 4 + group];
            const uint8_t scale_byte = scales[ib32 >> 1];
            const uint8_t scale4 = (ib32 & 1) ? (scale_byte >> 4) : (scale_byte & 0x0f);
            const float db = d * static_cast<float>(1 + 2 * scale4);

            uint16_t grid_index;
            int grid_lane;
            if (lane8 < 4) {
                grid_index = static_cast<uint16_t>(qs32[2 * group])
                    | static_cast<uint16_t>((static_cast<uint16_t>(qh32) << (8 - 2 * group)) & 0x100u);
                grid_lane = lane8;
            } else {
                grid_index = static_cast<uint16_t>(qs32[2 * group + 1])
                    | static_cast<uint16_t>((static_cast<uint16_t>(qh32) << (7 - 2 * group)) & 0x100u);
                grid_lane = lane8 - 4;
            }

            float qv = static_cast<float>(iq3s_grid_value(grid_index, grid_lane));
            if (sign_bits & (1u << lane8)) qv = -qv;

            const int col = (sb << 8) + (ib32 << 5) + lane;
            acc = fmaf(db * qv, __half2float(x[col]), acc);
        }
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


__device__ __forceinline__ float q6_k_value(const uint8_t * blk, int i) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = lowbit_half(blk + 208);

    const int half = i >> 7;
    const int r = i & 127;
    const int l = r & 31;
    const int q = r >> 5;

    const uint8_t * ql64 = ql + half * 64;
    const uint8_t * qh32 = qh + half * 32;
    const int8_t * sc8 = sc + half * 8;

    int v = 0;
    int scale_index = 0;
    if (q == 0) {
        v = (ql64[l] & 0x0f) | (((qh32[l] >> 0) & 3) << 4);
        scale_index = (l >> 4) + 0;
    } else if (q == 1) {
        v = (ql64[l + 32] & 0x0f) | (((qh32[l] >> 2) & 3) << 4);
        scale_index = (l >> 4) + 2;
    } else if (q == 2) {
        v = (ql64[l] >> 4) | (((qh32[l] >> 4) & 3) << 4);
        scale_index = (l >> 4) + 4;
    } else {
        v = (ql64[l + 32] >> 4) | (((qh32[l] >> 6) & 3) << 4);
        scale_index = (l >> 4) + 6;
    }
    return d * static_cast<float>(sc8[scale_index]) * static_cast<float>(v - 32);
}

template <bool OUT_F32>
__global__ void q6_k_gemv_kernel(
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

    const int superblocks_per_row = cols >> 8;
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 210u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    float acc = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 210u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = lane + j * 32;
            acc = fmaf(
                q6_k_value(blk, i),
                __half2float(x[(sb << 8) + i]),
                acc);
        }
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

__global__ void q8_0_gather_row_kernel(
        const uint8_t * __restrict__ w,
        int cols,
        int row,
        __half * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const int block = i >> 5;
    const int lane = i & 31;
    const int blocks_per_row = cols >> 5;
    const uint8_t * p = w +
        (static_cast<size_t>(row) * blocks_per_row + block) * 34u;
    const float d = lowbit_half(p);
    const int8_t q = *reinterpret_cast<const int8_t *>(p + 2 + lane);
    out[i] = __float2half(d * static_cast<float>(q));
}

inline void launch_q8_0_gather_row(
        const LowBitMatrix & w,
        int row,
        __half * out,
        cudaStream_t stream) {
    if (w.type != GgmlType::Q8_0 || w.cols <= 0 || w.cols % 32 != 0 ||
        row < 0 || row >= w.rows) {
        throw std::runtime_error("invalid Q8_0 embedding row gather");
    }
    const int threads = 256;
    q8_0_gather_row_kernel<<<(w.cols + threads - 1) / threads, threads, 0, stream>>>(
        w.data, w.cols, row, out);
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
    } else if (w.type == GgmlType::Q6_K) {
        q6_k_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ4_NL) {
        iq4_nl_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ2_S) {
        iq2_s_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ3_S) {
        iq3_s_gemv_kernel<false><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
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
    } else if (w.type == GgmlType::Q6_K) {
        q6_k_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ4_NL) {
        iq4_nl_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ2_S) {
        iq2_s_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else if (w.type == GgmlType::IQ3_S) {
        iq3_s_gemv_kernel<true><<<blocks, threads, 0, stream>>>(w.data, x, y, w.rows, w.cols);
    } else {
        throw std::runtime_error("unsupported low-bit GEMV type");
    }
}

} // namespace wd
