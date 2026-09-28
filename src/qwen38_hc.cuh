#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
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
    std::vector<float> debug_norm_f32;
    std::vector<float> debug_inject_f32;
    bool has_inject = true;
};

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
    if (il == 0 && (std::strcmp(branch, "attn") == 0 || std::strcmp(branch, "ffn") == 0)) {
        if (norm.type != GgmlType::F32) {
            throw std::runtime_error("layer-0 HC F32 debug requires an F32 norm tensor");
        }
        const float * values = reinterpret_cast<const float *>(norm.data);
        out.debug_norm_f32.assign(values, values + norm.elements());
    }
    if (il == 0 && std::strcmp(branch, "attn") == 0) {
        if (inject.type != GgmlType::F32) {
            throw std::runtime_error("layer-0 HC F32 debug requires an F32 inject tensor");
        }
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

__global__ void q38_hc_silu_scale_kernel(__half * x, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]) * inv_hc;
    x[i] = __float2half(v / (1.0f + expf(-v)));
}

__global__ void q38_hc_mix_kernel(
        const __half * __restrict__ xn,
        const __half * __restrict__ gate,
        __half * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float x = __half2float(xn[idx]);
        const float g = __half2float(gate[idx]);
        const float sig = 1.0f / (1.0f + expf(-g));
        sum = fmaf(x, sig, sum);
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
        embedding_ = residual_ = norm_ = gamma_ = low_ = low_raw_ = gate_ = mixed_ = nullptr;
        inject_ = inject_weights_ = nullptr;
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

__global__ void q38_hc_silu_scale_f32_kernel(float * x, int n, float inv_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * inv_hc;
    x[i] = v / (1.0f + expf(-v));
}

__global__ void q38_hc_mix_f32_kernel(
        const float * __restrict__ xn,
        const float * __restrict__ gate,
        float * __restrict__ mixed,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * hidden + i;
        const float sig = 1.0f / (1.0f + expf(-gate[idx]));
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
    const float w = 2.0f / (1.0f + expf(-inject[c] / static_cast<float>(hc)));
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
        constexpr int warps = 8;
        q38_hc_dense_gemv_f32_kernel<<<(hc + warps - 1) / warps, warps * 32, 0, stream>>>(
            scratch.inject_weights(), scratch.norm(), scratch.inject(), hc, hc_dim);
    }
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
        const __half * residual,
        const Qwen38HcWeights & w,
        Qwen38HcScratch & scratch,
        float rms_eps,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int hc = scratch.hc();
    const int hc_dim = hidden * hc;

    if (w.norm.size != static_cast<size_t>(hc_dim) ||
        w.down.cols != hc_dim || w.down.rows != scratch.low_rank() ||
        w.up.cols != scratch.low_rank() || w.up.rows != hc_dim) {
        throw std::runtime_error("HC weight geometry mismatch");
    }

    q38_hc_group_rmsnorm_kernel<<<hc, 256, 0, stream>>>(
        residual, w.norm.data, scratch.xn(), hidden, hc, rms_eps);
    launch_matrix_gemv_half(w.down, scratch.xn(), scratch.lo(), stream);

    const int lo_blocks = (scratch.low_rank() + 255) / 256;
    q38_hc_silu_scale_kernel<<<lo_blocks, 256, 0, stream>>>(
        scratch.lo(), scratch.low_rank(), 1.0f / static_cast<float>(hc));

    launch_matrix_gemv_half(w.up, scratch.lo(), scratch.gate(), stream);

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

inline void run_qwen38_hc_combine(
        __half * residual,
        const __half * block_out,
        const Qwen38HcScratch & scratch,
        cudaStream_t stream) {
    const int total = scratch.hidden() * scratch.hc();
    const int blocks = (total + 255) / 256;
    q38_hc_combine_kernel<<<blocks, 256, 0, stream>>>(
        residual, block_out, scratch.inject(), scratch.hidden(), scratch.hc());
}

} // namespace wd
