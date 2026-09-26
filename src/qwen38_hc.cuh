#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace wd {

struct Qwen38HcWeights {
    DeviceHalfArray norm;
    GpuMatrix down;
    GpuMatrix up;
    GpuMatrix inject;
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
