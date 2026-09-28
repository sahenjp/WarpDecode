#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "qwen38_layout.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38GdnWeights {
    GpuMatrix qkv;
    GpuMatrix z;
    GpuMatrix beta;
    GpuMatrix alpha;
    DeviceHalfArray conv;
    DeviceHalfArray dt;
    DeviceHalfArray a;
    DeviceHalfArray norm;
    GpuMatrix out;
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
    w.conv = upload_half_array(conv);
    w.dt = upload_half_array(dt);
    w.a = upload_half_array(a);
    w.norm = upload_half_array(norm);
    w.out = upload_matrix(out_t);
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

    void initialize(const Qwen38Layout & q) {
        if (state_) throw std::runtime_error("GDN state already initialized");
        state_dim_ = q.ssm_state_size;
        value_heads_ = q.ssm_dt_rank;
        conv_kernel_ = q.ssm_conv_kernel;
        conv_channels_ = q.ssm_state_size * q.ssm_group_count * 2
                       + q.ssm_state_size * q.ssm_dt_rank;

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
    }

    void clear(cudaStream_t stream) {
        if (!state_) return;
        const size_t state_n = static_cast<size_t>(state_dim_) * state_dim_ * value_heads_;
        const size_t conv_n = static_cast<size_t>(conv_kernel_ - 1) * conv_channels_;
        cudaMemsetAsync(state_, 0, state_n * sizeof(float), stream);
        cudaMemsetAsync(conv_state_, 0, conv_n * sizeof(__half), stream);
    }

    float * state() const { return state_; }
    __half * conv_state() const { return conv_state_; }

private:
    void reset() {
        if (state_) cudaFree(state_);
        if (conv_state_) cudaFree(conv_state_);
        state_ = nullptr;
        conv_state_ = nullptr;
    }

    int state_dim_ = 0;
    int value_heads_ = 0;
    int conv_kernel_ = 0;
    int conv_channels_ = 0;
    float * state_ = nullptr;
    __half * conv_state_ = nullptr;
};

class Qwen38GdnScratch {
public:
    Qwen38GdnScratch() = default;
    Qwen38GdnScratch(const Qwen38GdnScratch &) = delete;
    Qwen38GdnScratch & operator=(const Qwen38GdnScratch &) = delete;
    ~Qwen38GdnScratch() { reset(); }

    void initialize(const Qwen38Layout & q) {
        if (qkv_) throw std::runtime_error("GDN scratch already initialized");
        const int key_dim = q.ssm_state_size * q.ssm_group_count;
        const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
        conv_dim_ = key_dim * 2 + value_dim;
        key_dim_ = key_dim;
        value_dim_ = value_dim;
        value_heads_ = q.ssm_dt_rank;

        alloc(reinterpret_cast<void **>(&qkv_), static_cast<size_t>(conv_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&conv_out_), static_cast<size_t>(conv_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&z_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&beta_), static_cast<size_t>(value_heads_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&alpha_), static_cast<size_t>(value_heads_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(value_heads_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&core_out_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&norm_out_), static_cast<size_t>(value_dim_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&final_out_), static_cast<size_t>(q.hidden) * sizeof(__half));
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

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc GDN scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (qkv_) cudaFree(qkv_);
        if (conv_out_) cudaFree(conv_out_);
        if (z_) cudaFree(z_);
        if (beta_) cudaFree(beta_);
        if (alpha_) cudaFree(alpha_);
        if (gate_) cudaFree(gate_);
        if (core_out_) cudaFree(core_out_);
        if (norm_out_) cudaFree(norm_out_);
        if (final_out_) cudaFree(final_out_);
        qkv_ = conv_out_ = z_ = beta_ = alpha_ = core_out_ = norm_out_ = final_out_ = nullptr;
        gate_ = nullptr;
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

__global__ void q38_gdn_conv_kernel(
        const __half * __restrict__ current,
        __half * __restrict__ history,
        const __half * __restrict__ kernel,
        __half * __restrict__ out,
        int channels,
        int kernel_size) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    float sum = 0.0f;
    const int hist = kernel_size - 1;
    const size_t hbase = static_cast<size_t>(ch) * hist;
    const size_t kbase = static_cast<size_t>(ch) * kernel_size;

    for (int t = 0; t < hist; ++t) {
        sum = fmaf(__half2float(history[hbase + t]), __half2float(kernel[kbase + t]), sum);
    }
    sum = fmaf(__half2float(current[ch]), __half2float(kernel[kbase + hist]), sum);

    for (int t = 0; t + 1 < hist; ++t) history[hbase + t] = history[hbase + t + 1];
    if (hist > 0) history[hbase + hist - 1] = current[ch];

    out[ch] = __float2half(sum / (1.0f + expf(-sum)));
}

__global__ void q38_l2_norm_heads_kernel(
        __half * x,
        int head_dim,
        int n_heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    __shared__ float smem[256];

    float sum = 0.0f;
    const int base = h * head_dim;
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

__global__ void q38_gdn_conv_first_f32_kernel(
        const float * __restrict__ current,
        const float * __restrict__ kernel,
        float * __restrict__ out,
        int channels,
        int kernel_size) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const float value = current[channel] * kernel[channel * kernel_size + kernel_size - 1];
    out[channel] = value / (1.0f + expf(-value));
}

__global__ void q38_gdn_l2_norm_f32_kernel(
        float * x,
        int head_dim,
        int heads,
        float eps) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float smem[256];
    const int base = head * head_dim;
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

__global__ void q38_gdn_gate_params_f32_kernel(
        float * beta,
        const float * alpha,
        const float * dt,
        const float * a,
        float * gate,
        int heads) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    beta[h] = 1.0f / (1.0f + expf(-beta[h]));
    gate[h] = q38_softplus(alpha[h] + dt[h]) * a[h];
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
    if (lane == 0) out[vh * head_dim + col] = attn * rsqrtf(static_cast<float>(head_dim));
}

__global__ void q38_gdn_gated_rmsnorm_f32_kernel(
        const float * __restrict__ x,
        const float * __restrict__ norm,
        const float * __restrict__ z,
        float * __restrict__ out,
        int head_dim,
        int heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= heads) return;
    __shared__ float smem[256];
    const int base = h * head_dim;
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
        const float gate = 1.0f / (1.0f + expf(-z[idx]));
        out[idx] = x[idx] * inv * norm[i] * gate;
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
    q38_gdn_dense_f32_gemv_kernel<<<(q.ssm_dt_rank + 7) / 8, threads, 0, stream>>>(
        scratch.beta_weights(), input, scratch.beta(), q.ssm_dt_rank, q.hidden);
    q38_gdn_dense_f32_gemv_kernel<<<(q.ssm_dt_rank + 7) / 8, threads, 0, stream>>>(
        scratch.alpha_weights(), input, scratch.alpha(), q.ssm_dt_rank, q.hidden);

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
    const dim3 core_grid(q.ssm_dt_rank, (q.ssm_state_size + 3) / 4);
    const dim3 core_block(32, 4);
    q38_gdn_core_first_f32_kernel<<<core_grid, core_block, 0, stream>>>(
        q_conv, k_conv, v_conv, scratch.beta(), scratch.core_out(),
        q.ssm_state_size, q.ssm_group_count, q.ssm_dt_rank);
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
    const float b = __half2float(beta[h]);
    beta[h] = __float2half(1.0f / (1.0f + expf(-b)));
    const float av = __half2float(alpha[h]) + __half2float(dt[h]);
    gate[h] = q38_softplus(av) * __half2float(a[h]);
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

    const float decay = expf(gate[vh]);
    float pred = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        const size_t idx = static_cast<size_t>(i) * head_dim + j;
        const float sv = sh[idx] * decay;
        sh[idx] = sv;
        pred = fmaf(sv, __half2float(khv[i]), pred);
    }

    const float delta = __half2float(beta[vh]) * (__half2float(vhv[j]) - pred);
    const float qscale = rsqrtf(static_cast<float>(head_dim));
    float y = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        const size_t idx = static_cast<size_t>(i) * head_dim + j;
        const float sv = fmaf(__half2float(khv[i]), delta, sh[idx]);
        sh[idx] = sv;
        y = fmaf(sv, __half2float(qh[i]) * qscale, y);
    }
    out[vh * head_dim + j] = __float2half(y);
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
    __shared__ float smem[256];

    const int base = h * head_dim;
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
        const float gate = 1.0f / (1.0f + expf(-__half2float(z[idx])));
        out[idx] = __float2half(__half2float(x[idx]) * inv * __half2float(norm[i]) * gate);
    }
}

inline __half * run_qwen38_gdn(
        const __half * x,
        const Qwen38GdnWeights & w,
        Qwen38GdnState & state,
        Qwen38GdnScratch & scratch,
        const Qwen38Layout & q,
        float rms_eps,
        cudaStream_t stream) {
    const int key_dim = q.ssm_state_size * q.ssm_group_count;
    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
    const int conv_dim = key_dim * 2 + value_dim;

    launch_matrix_gemv_half(w.qkv, x, scratch.qkv(), stream);
    launch_matrix_gemv_half(w.z, x, scratch.z(), stream);
    launch_matrix_gemv_half(w.beta, x, scratch.beta(), stream);
    launch_matrix_gemv_half(w.alpha, x, scratch.alpha(), stream);

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

    launch_matrix_gemv_half(w.out, scratch.norm_out(), scratch.final_out(), stream);
    return scratch.final_out();
}

} // namespace wd
