#pragma once

#include "gguf.hpp"
#include "iq2s_grid.cuh"
#include "iq3s_grid.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// DP4A helpers (Task D).
//
// Every quantised GEMV below reproduces ggml's integer dot product by reducing
// each four-value chunk of a 32-value block with three shfl_down adds and by
// multiplying the four per-value products with the block scale one by one. Both
// halves of that reduction are exact integer arithmetic - the codebook
// magnitudes are bounded by 127 and the activations are 7-bit - so one __dp4a()
// over the four (weight, activation) byte pairs yields exactly the same int32 as
// the shuffle reduction, and multiplying that chunk sum by the block scale
// yields exactly the same product: integer addition and multiplication are
// associative and exact, and no product can overflow int32 (the worst case,
// IQ2_S with |q| <= 43 and a 15x group scale, stays below 2^22 per superblock).
//
// The float side is untouched. The DP4A kernels assign one chunk per lane and
// decode four rows per warp (lanes 0-7 own chunks 0-7 of row 0, lanes 8-15 of
// row 1, ...), which is what lets 32 lanes cover four 32-value blocks per step.
// Every float operation - the per-superblock fmaf, the IQ4_NL even/odd block
// accumulators and the closing hsum_float_8 tree - still runs on the same values
// in the same order as the reference kernel, so the results are bit-identical.
// ---------------------------------------------------------------------------

// __dp4a needs sm_61; the DP4A kernels are selected from sm_80 up (L4 = sm_89,
// A100 = sm_80) and fall back to this scalar equivalent everywhere else,
// including the host pass of the nvcc-free syntax check. The fallback adds the
// same four byte products, so it is also bit-identical - only slower.
__device__ __forceinline__ int lowbit_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __dp4a(a, b, c);
#else
    int sum = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int>(static_cast<int8_t>(a >> (8 * i))) *
               static_cast<int>(static_cast<int8_t>(b >> (8 * i)));
    }
    return sum;
#endif
}

// Byte j of the result is 0xff exactly when sign bit j of "bits" (low four bits
// are read) is set. The four bits are moved eight apart first so that a single
// multiply by 0xff expands them into four independent byte masks.
__device__ __forceinline__ unsigned lowbit_sign_mask4(unsigned bits) {
    const unsigned spaced = (bits & 1u) | ((bits & 2u) << 7) |
                            ((bits & 4u) << 14) | ((bits & 8u) << 21);
    return spaced * 0xffu;
}

// Negates byte j of "quad" when sign bit j of "bits" is set. ggml's IQ2_S/IQ3_S
// sign tables negate the magnitude, not the activation, and ~v + 1 == -v in
// two's complement. The addition cannot carry into the next byte because every
// IQ2_S/IQ3_S codebook magnitude is in [1, 127] and therefore never 0.
__device__ __forceinline__ unsigned lowbit_sign_i8x4(unsigned quad, unsigned bits) {
    const unsigned mask = lowbit_sign_mask4(bits);
    return (quad ^ mask) + (mask & 0x01010101u);
}

inline bool lowbit_env_flag(const char * name, bool default_on);

// WARODECODE_DP4A (default ON): route the quantised GEMVs through the __dp4a
// kernels below. WARODECODE_DP4A=0 restores the reference kernels for A/B runs.
inline bool lowbit_dp4a_enabled() {
    static const bool enabled = lowbit_env_flag("WARODECODE_DP4A", true);
    return enabled;
}

// WARODECODE_DP4A_SMEM (default ON): stage the IQ2_S/IQ3_S grid codebooks in
// shared memory instead of reading them from __constant__. Every lane of a warp
// looks up a different grid entry, and the constant cache serializes those
// divergent reads; the shared memory unit serves them in a couple of cycles.
inline bool lowbit_dp4a_smem_enabled() {
    static const bool enabled = lowbit_env_flag("WARODECODE_DP4A_SMEM", true);
    return enabled;
}

inline int lowbit_current_device_sm() {
    static const int sm = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 0;
        int major = 0;
        int minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            return 0;
        }
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
            return 0;
        }
        return major * 10 + minor;
    }();
    return sm;
}

// The DP4A kernels are only selected on sm_80+ where __dp4a is a single
// instruction and both target GPUs (L4, A100) live. Older targets keep the
// reference kernels even when the flag is on. The one-time stderr line makes an
// on-device A/B run state which path it took ("tok_s" alone cannot, because both
// paths are required to be bit-identical).
inline bool lowbit_dp4a_available() {
    static const bool available = [] {
        const int sm = lowbit_current_device_sm();
        const bool requested = lowbit_dp4a_enabled();
        const bool selected = requested && sm >= 80;
        std::fprintf(stderr, "warodecode: DP4A GEMV %s (WARODECODE_DP4A=%d, sm_%d)\n",
                     selected ? "selected" : "not selected", requested ? 1 : 0, sm);
        return selected;
    }();
    return available;
}

class Q8_1Buffer {
public:
    Q8_1Buffer() = default;
    Q8_1Buffer(const Q8_1Buffer &) = delete;
    Q8_1Buffer & operator=(const Q8_1Buffer &) = delete;
    ~Q8_1Buffer() { reset(); }

    void initialize(int elements) {
        if (values_) throw std::runtime_error("Q8_1 buffer already initialized");
        if (elements <= 0 || elements % 32 != 0) throw std::runtime_error("Q8_1 input must be a positive multiple of 32");
        elements_ = elements;
        const int blocks = elements / 32;
        cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&values_), static_cast<size_t>(elements) * sizeof(int8_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc Q8_1 values failed: ") + cudaGetErrorString(e));
        e = cudaMalloc(reinterpret_cast<void **>(&scales_), static_cast<size_t>(blocks) * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc Q8_1 scales failed: ") + cudaGetErrorString(e));
    }

    int elements() const { return elements_; }
    int8_t * values() const { return values_; }
    __half * scales() const { return scales_; }

private:
    void reset() {
        if (values_) cudaFree(values_);
        if (scales_) cudaFree(scales_);
        values_ = nullptr;
        scales_ = nullptr;
        elements_ = 0;
    }

    int elements_ = 0;
    int8_t * values_ = nullptr;
    __half * scales_ = nullptr;
};

__global__ void q8_1_quantize_f32_kernel(
        const float * __restrict__ x,
        int elements,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= elements / 32) return;

    const int i = block * 32 + lane;
    const float value = x[i];
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

__global__ void q8_0_q8_1_gemv_f32_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x,
        const __half * __restrict__ x_scales,
        float * __restrict__ y,
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
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * block = row_ptr + static_cast<size_t>(b) * 34u;
        const int8_t qw = *reinterpret_cast<const int8_t *>(block + 2 + lane);
        int dot = static_cast<int>(qw) * static_cast<int>(x[(b << 5) + lane]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            const float scale = lowbit_half(block) * __half2float(x_scales[b]);
            sum += scale * static_cast<float>(dot);
        }
    }
    if (lane == 0) y[row] = sum;
}

inline void launch_q8_0_gemv_q8_1_f32(
        const LowBitMatrix & weights,
        const float * input,
        float * output,
        Q8_1Buffer & scratch,
        cudaStream_t stream) {
    if (weights.type != GgmlType::Q8_0 || weights.rows <= 0 ||
        weights.cols > scratch.elements() || weights.cols % 32 != 0) {
        throw std::runtime_error("Q8_0 x Q8_1 GEMV geometry mismatch");
    }
    constexpr int threads = 256;
    const int input_blocks = weights.cols / 32;
    q8_1_quantize_f32_kernel<<<(input_blocks + 7) / 8, threads, 0, stream>>>(
        input, weights.cols, scratch.values(), scratch.scales());
    constexpr int warps = threads / 32;
    const int row_blocks = (weights.rows + warps - 1) / warps;
    q8_0_q8_1_gemv_f32_kernel<<<row_blocks, threads, 0, stream>>>(
        weights.data, scratch.values(), scratch.scales(), output, weights.rows, weights.cols);
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

// Same gather, F32 output. ggml dequantises the embedding row into F32 and never
// rounds it, so the half round trip in the F16 form is a real (and, because the
// embedding seeds every position, systematic) error source.
__global__ void q8_0_gather_row_f32_kernel(
        const uint8_t * __restrict__ w,
        int cols,
        int row,
        float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const int block = i >> 5;
    const int lane = i & 31;
    const int blocks_per_row = cols >> 5;
    const uint8_t * p = w +
        (static_cast<size_t>(row) * blocks_per_row + block) * 34u;
    const float d = lowbit_half(p);
    const int8_t q = *reinterpret_cast<const int8_t *>(p + 2 + lane);
    out[i] = d * static_cast<float>(q);
}

inline void launch_q8_0_gather_row_f32(
        const LowBitMatrix & w,
        int row,
        float * out,
        cudaStream_t stream,
        int threads = 256) {
    if (w.type != GgmlType::Q8_0 || w.cols <= 0 || w.cols % 32 != 0 ||
        row < 0 || row >= w.rows || out == nullptr) {
        throw std::runtime_error("invalid Q8_0 embedding row gather (f32)");
    }
    q8_0_gather_row_f32_kernel<<<(w.cols + threads - 1) / threads, threads, 0, stream>>>(
        w.data, w.cols, row, out);
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

// ---------------------------------------------------------------------------
// Q6_K x Q8_K, the vec_dot pair llama.cpp uses for the output head.
//
// Q6_K's vec_dot_type is Q8_K, so the CPU reference widens the F16 activation to
// F32, quantises it exactly like quantize_row_q8_K_ref and only then takes the
// integer dot product:
//   lowbit_copy_half_to_f32_kernel -> launch_q8_k_quantize_f32 -> GEMV
// The F16-activation q6_k_gemv_kernel rounds the activation instead and drifts by
// roughly a percent per projection, which the argmax over the vocab-sized head
// sees directly. The definitions follow the Q8_K activation quantiser below;
// they are declared here because launch_lowbit_gemv_float dispatches to them.
// ---------------------------------------------------------------------------
inline bool lowbit_head_q8_k_enabled();
inline bool lowbit_q6_k_head_q8_k_eligible(const LowBitMatrix & weights);
inline void launch_q6_k_head_gemv_q8_k_f32(
        const LowBitMatrix & weights,
        const __half * activation,
        float * output,
        cudaStream_t stream,
        int threads = 256);

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
    // Q6_K is the Qwen3.8 output head, whose vec_dot_type in ggml is Q8_K.
    // llama.cpp widens the head activation to F32, quantises it to Q8_K and takes
    // the integer dot product, so route the Q6_K case through that pipeline when
    // WARODECODE_HEAD_Q8_K is on and the geometry supports it.
    if (lowbit_q6_k_head_q8_k_eligible(w)) {
        launch_q6_k_head_gemv_q8_k_f32(w, x, y, stream, threads);
        return;
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

// ---------------------------------------------------------------------------
// ggml-compatible activation quantisation for the low-bit GEMV kernels below.
//
// llama.cpp's CPU reference quantises the activation to the weight type's
// vec_dot_type before every matmul, then takes an integer dot product:
//   IQ2_S, IQ3_S, Q6_K, IQ2_XXS -> Q8_K (256-value blocks, int8 quants + F32
//                                  block scale + 16 per-16-value integer sums)
//   IQ4_NL, Q8_0                -> Q8_0 (32-value blocks, int8 quants + F16
//                                  block scale)
// The kernels above take an F16 activation and therefore round the activation
// before multiplying, which drifts from the CPU oracle by roughly a percent per
// projection. The kernels in this section consume the quantised activation
// instead and reproduce ggml's integer dot product and scale application order.
// ---------------------------------------------------------------------------

// ggml quantize_row_q8_K: one warp per 256-value block, each lane owning eight
// values. ggml picks "max" as the signed value at the position of the largest
// magnitude, so its iscale = -127/max may be negative. Reducing |x| instead only
// flips the sign of iscale, of every quantised value and of d = 1/iscale
// together, which leaves every downstream product bit-identical.
__global__ void q8_k_quantize_f32_kernel(
        const float * __restrict__ x,
        int8_t * __restrict__ quantized,
        float * __restrict__ scales,
        int16_t * __restrict__ bsums,
        int blocks) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= blocks) return;

    const int base = lane << 3;
    const float * src = x + static_cast<size_t>(block) * 256u;
    int8_t * dst = quantized + static_cast<size_t>(block) * 256u;

    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        amax = fmaxf(amax, fabsf(src[base + j]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);

    if (amax == 0.0f) {
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[base + j] = 0;
        if (lane == 0) {
            scales[block] = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) bsums[static_cast<size_t>(block) * 16u + j] = 0;
        }
        return;
    }

    // ggml stores MIN(127, nearest_int(iscale*x)). |iscale*x| never exceeds 127
    // so the clamp cannot fire, but it is kept for parity with the reference.
    const float iscale = __fdiv_rn(-127.0f, amax);
    int partial = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = min(127, __float2int_rn(__fmul_rn(src[base + j], iscale)));
        dst[base + j] = static_cast<int8_t>(q);
        partial += q;
    }

    // bsums[j] is the integer sum of values 16j..16j+15, i.e. of lanes 2j, 2j+1.
    const int pair_lo = __shfl_sync(0xffffffffu, partial, (lane & 15) << 1);
    const int pair_hi = __shfl_sync(0xffffffffu, partial, ((lane & 15) << 1) + 1);
    if (lane == 0) scales[block] = __fdiv_rn(1.0f, iscale);
    if (lane < 16) {
        bsums[static_cast<size_t>(block) * 16u + lane] = static_cast<int16_t>(pair_lo + pair_hi);
    }
}

inline void launch_q8_k_quantize_f32(
        const float * input,
        int elements,
        int8_t * quantized,
        float * scales,
        int16_t * bsums,
        cudaStream_t stream,
        int threads = 256) {
    if (elements <= 0 || elements % 256 != 0) {
        throw std::runtime_error("Q8_K quantisation requires a positive multiple of 256 elements");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("Q8_K quantisation threads must be a multiple of 32");
    }
    const int blocks = elements / 256;
    const int warps = threads / 32;
    q8_k_quantize_f32_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        input, quantized, scales, bsums, blocks);
}

// ggml ggml_vec_dot_iq2_s_q8_K in warp form. The IQ2_S unpack (grid index,
// sign bits, 4-bit group scales) is the same as iq2_s_gemv_kernel above; only
// the activation changed. ggml accumulates an int32 bsum per 256-value
// superblock as sum_i ls_i * sumi_i with ls_i = 1 + 2*scale4_i, applies the F32
// scale d = fp16(weight_d) * activation_d once per superblock and multiplies the
// total by 0.125. The AVX2 kernel keeps eight int32 lanes alive, each covering
// four consecutive values, and reduces them with hsum_float_8 at the end, so the
// four-value chunks are reduced separately here too and the closing float tree
// matches hsum_float_8.
template <bool OUT_F32>
__global__ void iq2_s_q8_k_gemv_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
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

    const int group = lane >> 3; // which of the four 8-value grids of a 32-value block
    const int lane8 = lane & 7;
    float accumulator = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        const float d_w = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        const int8_t * a = x_qs + (sb << 8);

        int bsum = 0;
#pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t scale_byte = scales[ib32];
            const int ls = group < 2
                ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                : 1 + 2 * static_cast<int>(scale_byte >> 4);

            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                (static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u)));
            int qv = static_cast<int>(iq2s_grid_value(grid_index, lane8));
            if (signs[ib32 * 4 + group] & (1u << lane8)) qv = -qv;

            bsum += static_cast<int>(a[(ib32 << 5) + lane]) * qv * ls;
        }

        // Sum the four values of every 4-value chunk: lane 4m then holds chunk m,
        // which is exactly the int32 lane the AVX2 kernel feeds to its FMA.
        bsum += __shfl_down_sync(0xffffffffu, bsum, 2);
        bsum += __shfl_down_sync(0xffffffffu, bsum, 1);
        if ((lane & 3) == 0) {
            accumulator = fmaf(d_w * x_d[sb], static_cast<float>(bsum), accumulator);
        }
    }

    const float v0 = __shfl_sync(0xffffffffu, accumulator, 0);
    const float v1 = __shfl_sync(0xffffffffu, accumulator, 4);
    const float v2 = __shfl_sync(0xffffffffu, accumulator, 8);
    const float v3 = __shfl_sync(0xffffffffu, accumulator, 12);
    const float v4 = __shfl_sync(0xffffffffu, accumulator, 16);
    const float v5 = __shfl_sync(0xffffffffu, accumulator, 20);
    const float v6 = __shfl_sync(0xffffffffu, accumulator, 24);
    const float v7 = __shfl_sync(0xffffffffu, accumulator, 28);
    if (lane == 0) {
        // hsum_float_8(accumf): (a4+a0, a5+a1, a6+a2, a7+a3) then a tree.
        const float r0 = __fadd_rn(v4, v0);
        const float r1 = __fadd_rn(v5, v1);
        const float r2 = __fadd_rn(v6, v2);
        const float r3 = __fadd_rn(v7, v3);
        const float s0 = __fadd_rn(r0, r2);
        const float s1 = __fadd_rn(r1, r3);
        const float value = 0.125f * __fadd_rn(s0, s1);
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = value;
        } else {
            static_cast<__half *>(y)[row] = __float2half(value);
        }
    }
}

// ---------------------------------------------------------------------------
// Task G: expert-batched DP4A GEMV.
//
// One routed-MoE layer decodes the same activation against top-k expert
// matrices, and the current path launches one kernel per expert and projection.
// The kernels below take a whole layer in a single launch instead: grid.y
// selects the expert through a small device array of weight pointers, and every
// expert's rows are decoded by exactly the same per-row routine the
// single-expert kernels use. Only the launch geometry changes, so every output
// value stays bit-identical.
// ---------------------------------------------------------------------------
// Longest expert batch one launch addresses: the pointer array has a fixed size
// so one small struct copy per layer can carry every weight pointer.
constexpr int kLowBitExpertBatchMax = 32;

// IQ2_S x Q8_K with one __dp4a per four values, one row per warp.
//
// Chunk c of a 32-value block covers columns 4c..4c+3, which sit in grid group
// g = c >> 1 at grid positions lane8 = 4*(c & 1) .. +3. Both chunks of a group
// therefore live in that group's single 64-bit grid entry: c even reads its low
// half, c odd its high half. The sign byte holds one bit per grid position, and
// the group's 4-bit scale belongs to groups 0/1 for lanes 0-7 and to groups 2/3
// for lanes 8-15, so "ls" depends only on the chunk index, exactly as before.
//
// The warp keeps the reference geometry - all 32 lanes on the same row, 8 warps
// per block, one warp per row - because a 640-row expert projection is only 640
// warps of work: decoding four rows per warp quarters that and leaves most SMs
// idle. Instead the 32 lanes are the 8 chunks times four phases, where phase q
// owns the 32-value blocks q and q+4. The integer partials of the four phases
// are then summed with two shuffles, which is exact (the phases own disjoint
// blocks) and gives every lane the same int32 the reference's three-lane
// shuffle produces before the per-superblock scale.
// Per-row decode shared by the single-expert kernel below and its
// expert-batched twin. Both callers run exactly the same arithmetic, so their
// rows are bit-identical; only lane 0 of the warp carries a meaningful return
// value (the closing tree reduces the eight chunk accumulators onto lane 0).
template <bool USE_SMEM>
__device__ __forceinline__ float iq2_s_q8_k_row_dp4a(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        int row,
        int cols,
        const uint64_t * grid_table) {
    const int lane = threadIdx.x & 31;

    const int chunk = lane & 7;        // 4-value chunk of a 32-value block
    const int phase = lane >> 3;       // which pair of 32-value blocks
    const int group = chunk >> 1;      // 8-value IQ2_S grid group
    const int grid_half = chunk & 1;   // low or high half of the grid entry

    const int superblocks_per_row = cols >> 8; // 256 values per IQ2_S block
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 82u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;
    const bool scale_low_nibble = chunk < 4;

    float accumulator = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        const float d_w = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        const int8_t * a = x_qs + (sb << 8);

        int partial = 0;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ib32 = phase + 4 * p;
            const uint8_t scale_byte = scales[ib32];
            const int ls = scale_low_nibble
                ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                : 1 + 2 * static_cast<int>(scale_byte >> 4);

            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u));
            const unsigned quad = static_cast<unsigned>(
                (USE_SMEM ? grid_table[grid_index] : k_iq2s_grid[grid_index]) >>
                (32 * grid_half));
            const unsigned sign_bits = static_cast<unsigned>(
                signs[ib32 * 4 + group] >> (4 * grid_half));
            const unsigned weights4 = lowbit_sign_i8x4(quad, sign_bits);
            const unsigned activation4 = *reinterpret_cast<const unsigned *>(
                a + static_cast<size_t>(ib32) * 32u + static_cast<size_t>(chunk) * 4u);
            partial += lowbit_dp4a(
                static_cast<int>(activation4), static_cast<int>(weights4), 0) * ls;
        }

        // Phases 0-3 of a chunk own disjoint 32-value blocks, so the integer
        // partials combine exactly, and lanes 0-7 end up with the chunk sums.
        partial += __shfl_down_sync(0xffffffffu, partial, 8);
        partial += __shfl_down_sync(0xffffffffu, partial, 16);
        if (lane < 8) {
            accumulator = fmaf(d_w * x_d[sb], static_cast<float>(partial), accumulator);
        }
    }

    // Chunk accumulators live in lanes 0-7; the closing hsum_float_8 pairing is
    // (v0+v4, v1+v5, v2+v6, v3+v7), then (s0+s2) + (s1+s3).
    const float r = __fadd_rn(accumulator, __shfl_down_sync(0xffffffffu, accumulator, 4));
    const float s = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2));
    return __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 1));
}

template <bool OUT_F32, bool USE_SMEM>
__global__ void iq2_s_q8_k_gemv_dp4a_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        void * __restrict__ y,
        int rows,
        int cols) {
    // One warp reads 16 distinct grid entries per unrolled block, so the
    // constant cache serializes them. A shared copy is filled once per block
    // (before the early return, so every thread reaches the barrier).
    __shared__ uint64_t grid_table[1024];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
            grid_table[i] = k_iq2s_grid[i];
        }
        __syncthreads();
    }

    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float value = iq2_s_q8_k_row_dp4a<USE_SMEM>(w, x_qs, x_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    // ggml applies 0.125 for IQ2_S only (IQ3_S has no closing factor).
    const float scaled = 0.125f * value;
    if constexpr (OUT_F32) {
        static_cast<float *>(y)[row] = scaled;
    } else {
        static_cast<__half *>(y)[row] = __float2half(scaled);
    }
}

// Expert-batched twin of the kernel above: one launch covers every matrix in
// w_experts, grid.y selects the expert, and expert e writes
// y[e * rows .. e * rows + rows). Every row still goes through
// iq2_s_q8_k_row_dp4a(), so the output matches the per-expert launches bit for
// bit.
template <bool USE_SMEM>
__global__ void iq2_s_q8_k_gemv_experts_dp4a_kernel(
        const uint8_t * const * __restrict__ w_experts,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        float * __restrict__ y,
        int rows,
        int cols) {
    __shared__ uint64_t grid_table[1024];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
            grid_table[i] = k_iq2s_grid[i];
        }
        __syncthreads();
    }

    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float value = iq2_s_q8_k_row_dp4a<USE_SMEM>(
        w_experts[blockIdx.y], x_qs, x_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    y[static_cast<size_t>(blockIdx.y) * static_cast<size_t>(rows) + row] = 0.125f * value;
}

// Chooses between the __dp4a and the reference warp kernel. Both decode one row
// per warp, so the launch geometry is shared.
template <bool OUT_F32>
inline void launch_iq2_s_gemv_q8_k_dispatch(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        void * output,
        cudaStream_t stream,
        int threads) {
    const int warps = threads / 32;
    const int blocks = (weights.rows + warps - 1) / warps;
    if (lowbit_dp4a_available()) {
        if (lowbit_dp4a_smem_enabled()) {
            iq2_s_q8_k_gemv_dp4a_kernel<OUT_F32, true><<<blocks, threads, 0, stream>>>(
                weights.data, activation, activation_scales, output, weights.rows, weights.cols);
        } else {
            iq2_s_q8_k_gemv_dp4a_kernel<OUT_F32, false><<<blocks, threads, 0, stream>>>(
                weights.data, activation, activation_scales, output, weights.rows, weights.cols);
        }
        return;
    }
    iq2_s_q8_k_gemv_kernel<OUT_F32><<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, output, weights.rows, weights.cols);
}

inline void launch_iq2_s_gemv_q8_k_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ2_S || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 256 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K GEMV threads must be a multiple of 32");
    }
    launch_iq2_s_gemv_q8_k_dispatch<true>(
        weights, activation, activation_scales, output, stream, threads);
}

inline void launch_iq2_s_gemv_q8_k_half(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        __half * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ2_S || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 256 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K GEMV threads must be a multiple of 32");
    }
    launch_iq2_s_gemv_q8_k_dispatch<false>(
        weights, activation, activation_scales, output, stream, threads);
}

// ggml ggml_vec_dot_iq3_s_q8_K in warp form, consuming the same ggml-exact Q8_K
// activation as iq2_s_q8_k_gemv_kernel above. The IQ3_S unpack (two 4-value
// halves per 32-value block, 9-bit grid index, one sign byte per 8 values and a
// packed pair of 4-bit group scales) is the same as iq3_s_gemv_kernel; only the
// activation changed. ggml accumulates an int32 bsum per 256-value superblock as
// sum_i ls_i * sumi_i with ls_i = 1 + 2*scale4_i, applies the F32 scale
// d = fp16(weight_d) * activation_d once per superblock and returns
// hsum_float_8(accumf) with no further factor (IQ2_S ends with 0.125, IQ3_S does
// not). The AVX2 kernel keeps eight int32 lanes alive, each covering four
// consecutive values, and reduces them with hsum_float_8 at the end, so the
// four-value chunks are reduced separately here too and the closing float tree
// matches hsum_float_8.
template <bool OUT_F32>
__global__ void iq3_s_q8_k_gemv_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
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

    const int group = lane >> 3; // which of the four 8-value grids of a 32-value block
    const int lane8 = lane & 7;
    float accumulator = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 110u;
        const float d_w = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        const int8_t * a = x_qs + (sb << 8);

        int bsum = 0;
#pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * qs32 = qs + ib32 * 8;
            const uint8_t qh32 = qh[ib32];
            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = (ib32 & 1) ? 1 + 2 * static_cast<int>(scale_byte >> 4)
                                      : 1 + 2 * static_cast<int>(scale_byte & 0x0f);

            uint16_t grid_index;
            int grid_lane;
            if (lane8 < 4) {
                grid_index = static_cast<uint16_t>(qs32[2 * group])
                    | static_cast<uint16_t>(
                        (static_cast<uint16_t>(qh32) << (8 - 2 * group)) & 0x100u);
                grid_lane = lane8;
            } else {
                grid_index = static_cast<uint16_t>(qs32[2 * group + 1])
                    | static_cast<uint16_t>(
                        (static_cast<uint16_t>(qh32) << (7 - 2 * group)) & 0x100u);
                grid_lane = lane8 - 4;
            }

            int qv = static_cast<int>(iq3s_grid_value(grid_index, grid_lane));
            if (signs[ib32 * 4 + group] & (1u << lane8)) qv = -qv;

            bsum += static_cast<int>(a[(ib32 << 5) + lane]) * qv * ls;
        }

        // Sum the four values of every 4-value chunk: lane 4m then holds chunk m,
        // which is exactly the int32 lane the AVX2 kernel feeds to its FMA.
        bsum += __shfl_down_sync(0xffffffffu, bsum, 2);
        bsum += __shfl_down_sync(0xffffffffu, bsum, 1);
        if ((lane & 3) == 0) {
            accumulator = fmaf(d_w * x_d[sb], static_cast<float>(bsum), accumulator);
        }
    }

    const float v0 = __shfl_sync(0xffffffffu, accumulator, 0);
    const float v1 = __shfl_sync(0xffffffffu, accumulator, 4);
    const float v2 = __shfl_sync(0xffffffffu, accumulator, 8);
    const float v3 = __shfl_sync(0xffffffffu, accumulator, 12);
    const float v4 = __shfl_sync(0xffffffffu, accumulator, 16);
    const float v5 = __shfl_sync(0xffffffffu, accumulator, 20);
    const float v6 = __shfl_sync(0xffffffffu, accumulator, 24);
    const float v7 = __shfl_sync(0xffffffffu, accumulator, 28);
    if (lane == 0) {
        // hsum_float_8(accumf): (a4+a0, a5+a1, a6+a2, a7+a3) then a tree. ggml
        // applies no closing factor for IQ3_S, unlike IQ2_S' 0.125.
        const float r0 = __fadd_rn(v4, v0);
        const float r1 = __fadd_rn(v5, v1);
        const float r2 = __fadd_rn(v6, v2);
        const float r3 = __fadd_rn(v7, v3);
        const float s0 = __fadd_rn(r0, r2);
        const float s1 = __fadd_rn(r1, r3);
        const float value = __fadd_rn(s0, s1);
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = value;
        } else {
            static_cast<__half *>(y)[row] = __float2half(value);
        }
    }
}

// IQ3_S x Q8_K with one __dp4a per four values, one row per warp.
//
// The two 4-value halves of an 8-value grid group come from two different 9-bit
// grid indices, so a chunk is exactly one 32-bit grid entry: the index is
// qs32[2*group + half] plus the ninth bit taken from qh, and the whole word holds
// the chunk's four magnitudes. The sign byte carries one bit per grid position,
// and the group scale (1 + 2*scale4) is packed as two nibbles for the even and
// odd 32-value blocks, so "ls" depends only on ib32 - and therefore only on the
// phase below. The lane mapping, the phase reduction and the closing tree are the
// same as iq2_s_q8_k_gemv_dp4a_kernel; ggml applies no closing factor here.
// Per-row decode shared by the single-expert kernel below and its
// expert-batched twin (see iq2_s_q8_k_row_dp4a for the contract): the rows are
// produced by identical arithmetic, and only lane 0 receives the result.
template <bool USE_SMEM>
__device__ __forceinline__ float iq3_s_q8_k_row_dp4a(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        int row,
        int cols,
        const uint32_t * grid_table) {
    const int lane = threadIdx.x & 31;

    const int chunk = lane & 7;        // 4-value chunk of a 32-value block
    const int phase = lane >> 3;       // which pair of 32-value blocks
    const int group = chunk >> 1;      // 8-value IQ3_S grid group
    const int grid_half = chunk & 1;   // low or high half of the group

    const int superblocks_per_row = cols >> 8; // 256 values per IQ3_S block
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 110u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;

    float accumulator = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 110u;
        const float d_w = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        const int8_t * a = x_qs + (sb << 8);

        int partial = 0;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ib32 = phase + 4 * p;
            const uint8_t * qs32 = qs + ib32 * 8;
            const uint8_t qh32 = qh[ib32];
            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = (ib32 & 1)
                ? 1 + 2 * static_cast<int>(scale_byte >> 4)
                : 1 + 2 * static_cast<int>(scale_byte & 0x0f);

            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs32[2 * group + grid_half]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh32)
                     << (8 - 2 * group - grid_half)) & 0x100u));
            const unsigned quad = USE_SMEM ? grid_table[grid_index] : k_iq3s_grid[grid_index];
            const unsigned sign_bits = static_cast<unsigned>(
                signs[ib32 * 4 + group] >> (4 * grid_half));
            const unsigned weights4 = lowbit_sign_i8x4(quad, sign_bits);
            const unsigned activation4 = *reinterpret_cast<const unsigned *>(
                a + static_cast<size_t>(ib32) * 32u + static_cast<size_t>(chunk) * 4u);
            partial += lowbit_dp4a(
                static_cast<int>(activation4), static_cast<int>(weights4), 0) * ls;
        }

        partial += __shfl_down_sync(0xffffffffu, partial, 8);
        partial += __shfl_down_sync(0xffffffffu, partial, 16);
        if (lane < 8) {
            accumulator = fmaf(d_w * x_d[sb], static_cast<float>(partial), accumulator);
        }
    }

    const float r = __fadd_rn(accumulator, __shfl_down_sync(0xffffffffu, accumulator, 4));
    const float s = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2));
    return __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 1));
}

template <bool OUT_F32, bool USE_SMEM>
__global__ void iq3_s_q8_k_gemv_dp4a_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        void * __restrict__ y,
        int rows,
        int cols) {
    __shared__ uint32_t grid_table[512];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) {
            grid_table[i] = k_iq3s_grid[i];
        }
        __syncthreads();
    }

    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float value = iq3_s_q8_k_row_dp4a<USE_SMEM>(w, x_qs, x_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    if constexpr (OUT_F32) {
        static_cast<float *>(y)[row] = value;
    } else {
        static_cast<__half *>(y)[row] = __float2half(value);
    }
}

// Expert-batched twin of the kernel above (see
// iq2_s_q8_k_gemv_experts_dp4a_kernel for the grid contract).
template <bool USE_SMEM>
__global__ void iq3_s_q8_k_gemv_experts_dp4a_kernel(
        const uint8_t * const * __restrict__ w_experts,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        float * __restrict__ y,
        int rows,
        int cols) {
    __shared__ uint32_t grid_table[512];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) {
            grid_table[i] = k_iq3s_grid[i];
        }
        __syncthreads();
    }

    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float value = iq3_s_q8_k_row_dp4a<USE_SMEM>(
        w_experts[blockIdx.y], x_qs, x_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    y[static_cast<size_t>(blockIdx.y) * static_cast<size_t>(rows) + row] = value;
}

template <bool OUT_F32>
inline void launch_iq3_s_gemv_q8_k_dispatch(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        void * output,
        cudaStream_t stream,
        int threads) {
    const int warps = threads / 32;
    const int blocks = (weights.rows + warps - 1) / warps;
    if (lowbit_dp4a_available()) {
        if (lowbit_dp4a_smem_enabled()) {
            iq3_s_q8_k_gemv_dp4a_kernel<OUT_F32, true><<<blocks, threads, 0, stream>>>(
                weights.data, activation, activation_scales, output, weights.rows, weights.cols);
        } else {
            iq3_s_q8_k_gemv_dp4a_kernel<OUT_F32, false><<<blocks, threads, 0, stream>>>(
                weights.data, activation, activation_scales, output, weights.rows, weights.cols);
        }
        return;
    }
    iq3_s_q8_k_gemv_kernel<OUT_F32><<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, output, weights.rows, weights.cols);
}

inline void launch_iq3_s_gemv_q8_k_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ3_S || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 256 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K GEMV threads must be a multiple of 32");
    }
    launch_iq3_s_gemv_q8_k_dispatch<true>(
        weights, activation, activation_scales, output, stream, threads);
}

inline void launch_iq3_s_gemv_q8_k_half(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        __half * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ3_S || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 256 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K GEMV threads must be a multiple of 32");
    }
    launch_iq3_s_gemv_q8_k_dispatch<false>(
        weights, activation, activation_scales, output, stream, threads);
}

// ggml kvalues_iq4nl as integers: ggml keeps the table as int8_t and negates the
// Q8_0 activation rather than the codebook, which is the same product.
__device__ __forceinline__ int iq4_nl_lut_value(uint8_t q) {
    constexpr int8_t lut[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return static_cast<int>(lut[q & 0x0f]);
}

// ggml ggml_vec_dot_iq4_nl_q8_0 in warp form, consuming the ggml-exact Q8_0
// activation (int8 quants + F16 block scale) instead of an F16 activation.
// ggml sums the two halves of every 32-value block as integers, applies
// d = fp16(activation_d) * fp16(weight_d) once per block, and the AVX2 kernel
// keeps even and odd blocks in two separate eight-lane float accumulators that
// are added before hsum_float_8. Both details are reproduced here: the 4-value
// chunks stay apart, even and odd blocks use separate accumulators, and the
// closing tree matches hsum_float_8. ggml falls back to a scalar tail for an odd
// block count, so the launchers require a multiple of 64 columns.
template <bool OUT_F32>
__global__ void iq4_nl_q8_0_act_gemv_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
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

    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 18u;
        const float d = lowbit_half(blk) * __half2float(x_scales[b]);
        const uint8_t packed = blk[2 + (lane & 15)];
        const uint8_t q = lane < 16 ? (packed & 0x0f) : (packed >> 4);
        int prod = static_cast<int>(x_qs[(b << 5) + lane]) * iq4_nl_lut_value(q);

        prod += __shfl_down_sync(0xffffffffu, prod, 2);
        prod += __shfl_down_sync(0xffffffffu, prod, 1);
        if ((lane & 3) == 0) {
            if (b & 1) {
                acc_odd = fmaf(d, static_cast<float>(prod), acc_odd);
            } else {
                acc_even = fmaf(d, static_cast<float>(prod), acc_even);
            }
        }
    }

    const float e0 = __shfl_sync(0xffffffffu, acc_even, 0);
    const float e1 = __shfl_sync(0xffffffffu, acc_even, 4);
    const float e2 = __shfl_sync(0xffffffffu, acc_even, 8);
    const float e3 = __shfl_sync(0xffffffffu, acc_even, 12);
    const float e4 = __shfl_sync(0xffffffffu, acc_even, 16);
    const float e5 = __shfl_sync(0xffffffffu, acc_even, 20);
    const float e6 = __shfl_sync(0xffffffffu, acc_even, 24);
    const float e7 = __shfl_sync(0xffffffffu, acc_even, 28);
    const float o0 = __shfl_sync(0xffffffffu, acc_odd, 0);
    const float o1 = __shfl_sync(0xffffffffu, acc_odd, 4);
    const float o2 = __shfl_sync(0xffffffffu, acc_odd, 8);
    const float o3 = __shfl_sync(0xffffffffu, acc_odd, 12);
    const float o4 = __shfl_sync(0xffffffffu, acc_odd, 16);
    const float o5 = __shfl_sync(0xffffffffu, acc_odd, 20);
    const float o6 = __shfl_sync(0xffffffffu, acc_odd, 24);
    const float o7 = __shfl_sync(0xffffffffu, acc_odd, 28);
    if (lane == 0) {
        const float v0 = __fadd_rn(e0, o0);
        const float v1 = __fadd_rn(e1, o1);
        const float v2 = __fadd_rn(e2, o2);
        const float v3 = __fadd_rn(e3, o3);
        const float v4 = __fadd_rn(e4, o4);
        const float v5 = __fadd_rn(e5, o5);
        const float v6 = __fadd_rn(e6, o6);
        const float v7 = __fadd_rn(e7, o7);
        const float r0 = __fadd_rn(v4, v0);
        const float r1 = __fadd_rn(v5, v1);
        const float r2 = __fadd_rn(v6, v2);
        const float r3 = __fadd_rn(v7, v3);
        const float s0 = __fadd_rn(r0, r2);
        const float s1 = __fadd_rn(r1, r3);
        const float value = __fadd_rn(s0, s1);
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = value;
        } else {
            static_cast<__half *>(y)[row] = __float2half(value);
        }
    }
}

// Expands four packed IQ4_NL nibbles into the four signed bytes __dp4a consumes.
// The codebook is the same one iq4_nl_value()/iq4_nl_lut_value() use; nibble j
// becomes byte j, and the two's complement byte is what dp4a reads as int8.
__device__ __forceinline__ unsigned lowbit_iq4_nl_pack4(unsigned nibbles) {
    return static_cast<unsigned>(static_cast<uint8_t>(
               iq4_nl_lut_value(static_cast<uint8_t>(nibbles)))) |
           (static_cast<unsigned>(static_cast<uint8_t>(
                iq4_nl_lut_value(static_cast<uint8_t>(nibbles >> 8)))) << 8) |
           (static_cast<unsigned>(static_cast<uint8_t>(
                iq4_nl_lut_value(static_cast<uint8_t>(nibbles >> 16)))) << 16) |
           (static_cast<unsigned>(static_cast<uint8_t>(
                iq4_nl_lut_value(static_cast<uint8_t>(nibbles >> 24)))) << 24);
}

// IQ4_NL x Q8_0 with one __dp4a per four values, two rows per warp.
//
// A 32-value block is 16 bytes of packed nibbles: values 0-15 take the low
// nibble of byte i and values 16-31 its high nibble, so a 4-value chunk covers
// four consecutive bytes of one nibble plane. Those four bytes are fetched with
// two 16-bit loads (the 18-byte block stride keeps every row start even).
//
// Unlike IQ2_S/IQ3_S the scale is applied per block, so the blocks of a chunk
// cannot be summed as integers before scaling: the float FMA chain has to stay in
// block order. ggml already keeps two chains per chunk - even and odd blocks -
// and adds them before hsum_float_8, so the warp splits those two chains over
// lanes instead: 16 lanes per row are 8 chunks times the even/odd block parity.
// Each lane then walks stride-2 blocks, and the two chains meet again with one
// shuffle. That halves the warps per row (the down projection is 2560 rows, so
// 1280 warps still fill an A100) instead of quartering them.
// Per-row decode shared by the single-expert kernel below and its
// expert-batched twin. "row" is already clamped into [0, rows); the result is
// meaningful on the lanes with (lane & 15) == 0 that own the row.
__device__ __forceinline__ float iq4_nl_q8_0_act_row_dp4a(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        int row,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int lane_in_row = lane & 15;
    const int chunk = lane_in_row & 7;    // 4-value chunk of a 32-value block
    const int parity = lane_in_row >> 3;  // 0 = even blocks, 1 = odd blocks

    const int blocks_per_row = cols >> 5;
    const size_t row_stride = static_cast<size_t>(blocks_per_row) * 18u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;
    const int byte_base = (chunk & 3) * 4;
    const bool high_nibble = chunk >= 4;

    float accumulator = 0.0f;
    for (int b = parity; b < blocks_per_row; b += 2) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 18u;
        const float d = lowbit_half(blk) * __half2float(x_scales[b]);
        const uint8_t * qs = blk + 2 + byte_base;
        const unsigned packed =
            static_cast<unsigned>(*reinterpret_cast<const uint16_t *>(qs)) |
            (static_cast<unsigned>(*reinterpret_cast<const uint16_t *>(qs + 2)) << 16);
        const unsigned nibbles = high_nibble
            ? ((packed >> 4) & 0x0f0f0f0fu)
            : (packed & 0x0f0f0f0fu);
        const unsigned weights4 = lowbit_iq4_nl_pack4(nibbles);
        const unsigned activation4 = *reinterpret_cast<const unsigned *>(
            x_qs + static_cast<size_t>(b) * 32u + static_cast<size_t>(chunk) * 4u);
        accumulator = fmaf(
            d, static_cast<float>(lowbit_dp4a(
                   static_cast<int>(activation4), static_cast<int>(weights4), 0)),
            accumulator);
    }

    // Reunite the even and odd chains, exactly as ggml's
    // "acc_even + acc_odd" before hsum_float_8, then run the closing tree on the
    // eight chunk values in lanes row_in_warp*8 + 0..7.
    const float combined =
        __fadd_rn(accumulator, __shfl_xor_sync(0xffffffffu, accumulator, 8));
    const float r = __fadd_rn(combined, __shfl_down_sync(0xffffffffu, combined, 4));
    const float s = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2));
    return __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 1));
}

template <bool OUT_F32>
__global__ void iq4_nl_q8_0_act_gemv_dp4a_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        void * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int row_in_warp = lane >> 4;    // one of the two rows this warp decodes

    const int output_row =
        (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2 + row_in_warp;
    // Rows beyond "rows" are clamped so they stay in bounds; they are not stored.
    const int row = min(output_row, rows - 1);

    const float value = iq4_nl_q8_0_act_row_dp4a(w, x_qs, x_scales, row, cols);
    if ((lane & 15) != 0 || output_row >= rows) return;
    if constexpr (OUT_F32) {
        static_cast<float *>(y)[output_row] = value;
    } else {
        static_cast<__half *>(y)[output_row] = __float2half(value);
    }
}

// Expert-batched twin: grid.y selects the expert (via w_experts), expert e
// writes y[e * rows .. e * rows + rows), and the rows go through the same
// routine as the single-expert kernel. The nibble reads stay 16-bit aligned
// because the caller passes the expert pointer array; the odd-base guard of the
// dispatcher still applies to every entry (the staging pool is 2-byte aligned).
//
// Unlike the gate/up kernels, the down projection does NOT share one activation
// across experts: every expert consumes its own SiLU(gate) * up product, so
// blockIdx.y also selects a slice of the activation. x_qs is x_qs[e * cols] and
// x_scales is x_scales[e * (cols / 32)], which is exactly the [k][...] layout
// the batched Q8_0 quantisation kernel writes.
__global__ void iq4_nl_q8_0_act_gemv_experts_dp4a_kernel(
        const uint8_t * const * __restrict__ w_experts,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int row_in_warp = lane >> 4;

    const int output_row =
        (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2 + row_in_warp;
    const int row = min(output_row, rows - 1);

    const int8_t * expert_x_qs =
        x_qs + static_cast<size_t>(blockIdx.y) * static_cast<size_t>(cols);
    const __half * expert_x_scales =
        x_scales + static_cast<size_t>(blockIdx.y) * static_cast<size_t>(cols / 32);

    const float value = iq4_nl_q8_0_act_row_dp4a(
        w_experts[blockIdx.y], expert_x_qs, expert_x_scales, row, cols);
    if ((lane & 15) != 0 || output_row >= rows) return;
    y[static_cast<size_t>(blockIdx.y) * static_cast<size_t>(rows) + output_row] = value;
}

// The 16-bit nibble reads need a 2-byte aligned base. Every IQ4_NL block is 18
// bytes and every expert cache slot stride is a multiple of that, so the base is
// the only thing to check; anything odd keeps the reference kernel.
template <bool OUT_F32>
inline void launch_iq4_nl_gemv_q8_0_dispatch(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const __half * activation_scales,
        void * output,
        cudaStream_t stream,
        int threads) {
    const int warps = threads / 32;
    if (lowbit_dp4a_available() &&
        (reinterpret_cast<uintptr_t>(weights.data) & 1u) == 0) {
        constexpr int rows_per_warp = 2;
        const int blocks =
            (weights.rows + warps * rows_per_warp - 1) / (warps * rows_per_warp);
        iq4_nl_q8_0_act_gemv_dp4a_kernel<OUT_F32><<<blocks, threads, 0, stream>>>(
            weights.data, activation, activation_scales, output, weights.rows, weights.cols);
        return;
    }
    const int blocks = (weights.rows + warps - 1) / warps;
    iq4_nl_q8_0_act_gemv_kernel<OUT_F32><<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, output, weights.rows, weights.cols);
}

inline void launch_iq4_nl_gemv_q8_0_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const __half * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ4_NL || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 64 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 GEMV threads must be a multiple of 32");
    }
    launch_iq4_nl_gemv_q8_0_dispatch<true>(
        weights, activation, activation_scales, output, stream, threads);
}

// ---------------------------------------------------------------------------
// Task G: expert-batched launchers.
//
// Each one replaces "experts" single-expert launches with a single launch whose
// grid.y selects the expert (see the *_experts_dp4a_kernel definitions above).
// w_experts is a device array of "experts" weight pointers; the kernel reads its
// block's entry only, and expert e writes output[e * rows .. e * rows + rows).
// The launchers return false when the DP4A kernels are unavailable so the
// caller can keep its per-expert loop. The rows themselves are produced by the
// same per-row routines as the single-expert kernels, so the results are
// bit-identical.
// ---------------------------------------------------------------------------
inline bool launch_iq2_s_gemv_q8_k_experts_f32(
        const uint8_t * const * w_experts,
        int experts,
        int rows,
        int cols,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (!lowbit_dp4a_available()) return false;
    if (experts <= 0 || experts > 65535 || rows <= 0 ||
        cols <= 0 || cols % 256 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K expert-batched GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ2_S x Q8_K expert-batched GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const dim3 grid((rows + warps - 1) / warps, experts);
    if (lowbit_dp4a_smem_enabled()) {
        iq2_s_q8_k_gemv_experts_dp4a_kernel<true><<<grid, threads, 0, stream>>>(
            w_experts, activation, activation_scales, output, rows, cols);
    } else {
        iq2_s_q8_k_gemv_experts_dp4a_kernel<false><<<grid, threads, 0, stream>>>(
            w_experts, activation, activation_scales, output, rows, cols);
    }
    return true;
}

inline bool launch_iq3_s_gemv_q8_k_experts_f32(
        const uint8_t * const * w_experts,
        int experts,
        int rows,
        int cols,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (!lowbit_dp4a_available()) return false;
    if (experts <= 0 || experts > 65535 || rows <= 0 ||
        cols <= 0 || cols % 256 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K expert-batched GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ3_S x Q8_K expert-batched GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const dim3 grid((rows + warps - 1) / warps, experts);
    if (lowbit_dp4a_smem_enabled()) {
        iq3_s_q8_k_gemv_experts_dp4a_kernel<true><<<grid, threads, 0, stream>>>(
            w_experts, activation, activation_scales, output, rows, cols);
    } else {
        iq3_s_q8_k_gemv_experts_dp4a_kernel<false><<<grid, threads, 0, stream>>>(
            w_experts, activation, activation_scales, output, rows, cols);
    }
    return true;
}

// Every expert base must be 2-byte aligned (the kernel reads the packed nibbles
// with 16-bit loads); the caller checks that on the host pointers it fills
// w_experts with, because the DP4A path has no reference fallback here.
//
// activation/activation_scales are the batched buffers this kernel slices per
// expert (k * cols int8 values and k * (cols / 32) halves), NOT a single shared
// activation: each expert's down projection consumes its own activation.
inline bool launch_iq4_nl_gemv_q8_0_experts_f32(
        const uint8_t * const * w_experts,
        int experts,
        int rows,
        int cols,
        const int8_t * activation,
        const __half * activation_scales,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (!lowbit_dp4a_available()) return false;
    if (experts <= 0 || experts > 65535 || rows <= 0 ||
        cols <= 0 || cols % 64 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 expert-batched GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 expert-batched GEMV threads must be a multiple of 32");
    }
    constexpr int rows_per_warp = 2;
    const int warps = threads / 32;
    const dim3 grid((rows + warps * rows_per_warp - 1) / (warps * rows_per_warp), experts);
    iq4_nl_q8_0_act_gemv_experts_dp4a_kernel<<<grid, threads, 0, stream>>>(
        w_experts, activation, activation_scales, output, rows, cols);
    return true;
}

inline void launch_iq4_nl_gemv_q8_0_half(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const __half * activation_scales,
        __half * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::IQ4_NL || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 64 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("IQ4_NL x Q8_0 GEMV threads must be a multiple of 32");
    }
    launch_iq4_nl_gemv_q8_0_dispatch<false>(
        weights, activation, activation_scales, output, stream, threads);
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_K output head (definitions for the forward declarations above).
// ---------------------------------------------------------------------------

inline bool lowbit_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool lowbit_head_q8_k_enabled() {
    static const bool enabled = lowbit_env_flag("WARODECODE_HEAD_Q8_K", true);
    return enabled;
}

// __half2float is exact, so this is bit-identical to q38_hc_copy_half_to_f32_kernel
// in qwen38_hc.cuh. That kernel cannot be referenced from here because
// qwen38_hc.cuh already includes this header.
__global__ void lowbit_copy_half_to_f32_kernel(
        const __half * __restrict__ input,
        float * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __half2float(input[i]);
}

// ggml ggml_vec_dot_q6_K_q8_K (AVX2 branch) in warp form. It unpacks the 6-bit
// codes into four 32-value chunks per 128-value half of the superblock, folds two
// neighbouring products into an int16 with vpmaddubsw, multiplies each 4-value
// group by its 16-value scale with vpmaddwd, subtracts the q6_K -32 offset
// through the Q8_K 16-value integer sums and keeps eight int32 lanes alive until
// hsum_float_8. Those eight int32 lanes become eight float lanes that are only
// combined at the very end, so this kernel keeps the same lane assignment: warp
// lane l owns byte l of every 32-value chunk, the 4-value reduction leaves the
// AVX lanes on lanes 0,4,...,28, and the closing tree is hsum_float_8.
template <bool OUT_F32>
__global__ void q6_k_q8_k_gemv_kernel(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        const int16_t * __restrict__ x_bsums,
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

    // get_scale_shuffle(is) repeats scales[2*is] over the low 16 values of the
    // 32-value chunk and scales[2*is+1] over the high 16, so the scale a lane
    // needs depends only on which half of the chunk it sits in.
    const int scale_select = lane >> 4;
    float accumulator = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 210u;
        const uint8_t * ql = blk;
        const uint8_t * qh = blk + 128;
        const int8_t * scales = reinterpret_cast<const int8_t *>(blk + 192);
        const float d_w = lowbit_half(blk + 208);
        const int8_t * a = x_qs + static_cast<size_t>(sb) * 256u;

        int sumi = 0;
#pragma unroll
        for (int is = 0; is < 8; ++is) {
            const int half = is >> 2;
            const int sub = is & 3;
            // q4_0..q4_3: ql supplies the low nibble of one of the two 32-byte
            // planes, qh the two upper code bits, one bit pair per plane.
            const uint8_t ql_byte = ql[(half << 6) + ((sub & 1) ? 32 : 0) + lane];
            const uint8_t qh_byte = qh[(half << 5) + lane];
            const int v = (sub < 2)
                ? ((static_cast<int>(ql_byte) & 0x0f) |
                   ((static_cast<int>(qh_byte >> (2 * sub)) & 0x03) << 4))
                : ((static_cast<int>(ql_byte) >> 4) |
                   ((static_cast<int>(qh_byte >> (2 * sub)) & 0x03) << 4));
            const int scale = scales[(is << 1) + scale_select];
            sumi += v * static_cast<int>(a[(is << 5) + lane]) * scale;
        }

        // vpmaddwd folds the four products of one AVX lane into a single int32.
        // Integer addition is exact and associative, so only the float split has
        // to line up; lanes 4m..4m+3 all carry the same scale and collapse onto
        // lane 4m.
        sumi += __shfl_down_sync(0xffffffffu, sumi, 2);
        sumi += __shfl_down_sync(0xffffffffu, sumi, 1);
        if ((lane & 3) == 0) {
            const int m = lane >> 2;
            const int16_t * bsums = x_bsums + static_cast<size_t>(sb) * 16u;
            // ggml folds the q6_K -32 offset into the Q8_K sums:
            // q8sclsub[m] = (bsums[2m]*scales[2m] + bsums[2m+1]*scales[2m+1]) << 5,
            // subtracted from the integer dot before the F32 scale is applied.
            const int32_t correction = static_cast<int32_t>(static_cast<uint32_t>(
                static_cast<int32_t>(bsums[m << 1]) * static_cast<int32_t>(scales[m << 1]) +
                static_cast<int32_t>(bsums[(m << 1) + 1]) *
                    static_cast<int32_t>(scales[(m << 1) + 1])) << 5);
            accumulator = fmaf(
                __fmul_rn(d_w, x_d[sb]),
                static_cast<float>(sumi - correction),
                accumulator);
        }
    }

    const float v0 = __shfl_sync(0xffffffffu, accumulator, 0);
    const float v1 = __shfl_sync(0xffffffffu, accumulator, 4);
    const float v2 = __shfl_sync(0xffffffffu, accumulator, 8);
    const float v3 = __shfl_sync(0xffffffffu, accumulator, 12);
    const float v4 = __shfl_sync(0xffffffffu, accumulator, 16);
    const float v5 = __shfl_sync(0xffffffffu, accumulator, 20);
    const float v6 = __shfl_sync(0xffffffffu, accumulator, 24);
    const float v7 = __shfl_sync(0xffffffffu, accumulator, 28);
    if (lane == 0) {
        // hsum_float_8(acc): (a4+a0, a5+a1, a6+a2, a7+a3) then a tree.
        const float r0 = __fadd_rn(v4, v0);
        const float r1 = __fadd_rn(v5, v1);
        const float r2 = __fadd_rn(v6, v2);
        const float r3 = __fadd_rn(v7, v3);
        const float value = __fadd_rn(__fadd_rn(r0, r2), __fadd_rn(r1, r3));
        if constexpr (OUT_F32) {
            static_cast<float *>(y)[row] = value;
        } else {
            static_cast<__half *>(y)[row] = __float2half(value);
        }
    }
}

inline void launch_q6_k_gemv_q8_k_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        const int16_t * activation_sums,
        float * output,
        cudaStream_t stream,
        int threads = 256) {
    if (weights.type != GgmlType::Q6_K || weights.rows <= 0 ||
        weights.cols <= 0 || weights.cols % 256 != 0) {
        throw std::runtime_error("Q6_K x Q8_K GEMV geometry mismatch");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("Q6_K x Q8_K GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const int blocks = (weights.rows + warps - 1) / warps;
    q6_k_q8_k_gemv_kernel<true><<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, activation_sums,
        output, weights.rows, weights.cols);
}

// The three-stage head pipeline needs a few kilobytes of device scratch, but it
// runs once per decoded token, so the buffers are allocated once and kept for the
// process lifetime. Growing them could free memory an earlier launch on this
// stream still reads, so the stream is drained first.
struct LowBitHeadQ8KScratch {
    float * activation = nullptr;
    int8_t * quants = nullptr;
    float * scales = nullptr;
    int16_t * sums = nullptr;
    int cols = 0;
};

inline LowBitHeadQ8KScratch & lowbit_head_q8_k_scratch() {
    static LowBitHeadQ8KScratch scratch;
    return scratch;
}

inline unsigned long long & lowbit_head_q8_k_launch_ref() {
    static unsigned long long launches = 0;
    return launches;
}

// Number of output-head projections that ran through the Q8_K path. Reported by
// the CLI so an on-device run can confirm the new path is live.
inline unsigned long long lowbit_head_q8_k_launch_count() {
    return lowbit_head_q8_k_launch_ref();
}

inline void lowbit_head_q8_k_release(LowBitHeadQ8KScratch & scratch) {
    if (scratch.activation) cudaFree(scratch.activation);
    if (scratch.quants) cudaFree(scratch.quants);
    if (scratch.scales) cudaFree(scratch.scales);
    if (scratch.sums) cudaFree(scratch.sums);
    scratch.activation = nullptr;
    scratch.quants = nullptr;
    scratch.scales = nullptr;
    scratch.sums = nullptr;
    scratch.cols = 0;
}

inline void lowbit_head_q8_k_reserve(int cols, cudaStream_t stream) {
    LowBitHeadQ8KScratch & scratch = lowbit_head_q8_k_scratch();
    if (scratch.cols >= cols) return;
    if (scratch.activation) {
        const cudaError_t e = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("Q8_K head scratch drain failed: ") + cudaGetErrorString(e));
        }
        lowbit_head_q8_k_release(scratch);
    }
    const size_t blocks = static_cast<size_t>(cols) / 256u;
    cudaError_t e = cudaMalloc(
        reinterpret_cast<void **>(&scratch.activation), static_cast<size_t>(cols) * sizeof(float));
    if (e == cudaSuccess) {
        e = cudaMalloc(
            reinterpret_cast<void **>(&scratch.quants), static_cast<size_t>(cols) * sizeof(int8_t));
    }
    if (e == cudaSuccess) {
        e = cudaMalloc(
            reinterpret_cast<void **>(&scratch.scales), blocks * sizeof(float));
    }
    if (e == cudaSuccess) {
        e = cudaMalloc(
            reinterpret_cast<void **>(&scratch.sums), blocks * 16u * sizeof(int16_t));
    }
    if (e != cudaSuccess) {
        lowbit_head_q8_k_release(scratch);
        throw std::runtime_error(
            std::string("cudaMalloc Q8_K head scratch failed: ") + cudaGetErrorString(e));
    }
    scratch.cols = cols;
}

inline bool lowbit_q6_k_head_q8_k_eligible(const LowBitMatrix & weights) {
    if (!lowbit_head_q8_k_enabled()) return false;
    return weights.type == GgmlType::Q6_K && weights.rows > 0 &&
           weights.cols > 0 && weights.cols % 256 == 0;
}

// Widen the F16 head activation to F32, quantise it exactly like
// quantize_row_q8_K_ref and take the Q6_K x Q8_K integer dot product into F32
// logits. WARODECODE_HEAD_Q8_K=0 restores the legacy F16-activation GEMV.
// Same Q6_K x Q8_K head GEMV, but fed the caller's F32 activation: the CPU
// quantises its F32 head input, so avoiding the F16 round trip removes a
// ~2.6e-4 perturbation of every Q8_K block scale (and the code flips it causes).
inline void launch_q6_k_head_gemv_q8_k_from_f32(
        const LowBitMatrix & weights,
        const float * activation,
        float * output,
        cudaStream_t stream,
        int threads) {
    if (!lowbit_q6_k_head_q8_k_eligible(weights)) {
        throw std::runtime_error("Q6_K x Q8_K output head path is not available for this matrix");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("Q6_K x Q8_K GEMV threads must be a multiple of 32");
    }
    lowbit_head_q8_k_reserve(weights.cols, stream);
    LowBitHeadQ8KScratch & scratch = lowbit_head_q8_k_scratch();
    launch_q8_k_quantize_f32(
        activation, weights.cols, scratch.quants, scratch.scales,
        scratch.sums, stream, threads);
    launch_q6_k_gemv_q8_k_f32(
        weights, scratch.quants, scratch.scales, scratch.sums, output, stream, threads);
    ++lowbit_head_q8_k_launch_ref();
}

inline void launch_q6_k_head_gemv_q8_k_f32(
        const LowBitMatrix & weights,
        const __half * activation,
        float * output,
        cudaStream_t stream,
        int threads) {
    if (!lowbit_q6_k_head_q8_k_eligible(weights)) {
        throw std::runtime_error("Q6_K x Q8_K output head path is not available for this matrix");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("Q6_K x Q8_K GEMV threads must be a multiple of 32");
    }
    lowbit_head_q8_k_reserve(weights.cols, stream);
    LowBitHeadQ8KScratch & scratch = lowbit_head_q8_k_scratch();
    lowbit_copy_half_to_f32_kernel<<<(weights.cols + 255) / 256, 256, 0, stream>>>(
        activation, scratch.activation, weights.cols);
    launch_q8_k_quantize_f32(
        scratch.activation, weights.cols, scratch.quants, scratch.scales,
        scratch.sums, stream, threads);
    launch_q6_k_gemv_q8_k_f32(
        weights, scratch.quants, scratch.scales, scratch.sums, output, stream, threads);
    ++lowbit_head_q8_k_launch_ref();
}

} // namespace wd
