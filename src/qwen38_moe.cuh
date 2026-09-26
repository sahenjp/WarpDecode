#pragma once

#include "expert_cache.cuh"
#include "dense_vector.cuh"
#include "gguf.hpp"
#include "lowbit.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38MoeLayerTensors {
    const TensorInfo * router = nullptr;
    const TensorInfo * gate_up = nullptr;
    const TensorInfo * gate = nullptr;
    const TensorInfo * up = nullptr;
    const TensorInfo * down = nullptr;

    bool fused_gate_up() const { return gate_up != nullptr; }
};

struct Qwen38SharedExpertWeights {
    DeviceHalfVector gate_input;
    LowBitMatrix gate;
    LowBitMatrix up;
    LowBitMatrix down;
};

inline Qwen38SharedExpertWeights load_qwen38_shared_expert(
        const GgufFile & g,
        int il,
        int hidden,
        int shared_ffn) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const TensorInfo & gate_input = g.tensor(p + "ffn_gate_inp_shexp.weight");
    const TensorInfo & gate = g.tensor(p + "ffn_gate_shexp.weight");
    const TensorInfo & up = g.tensor(p + "ffn_up_shexp.weight");
    const TensorInfo & down = g.tensor(p + "ffn_down_shexp.weight");

    if (gate_input.dims.size() != 1 || static_cast<int>(gate_input.dims[0]) != hidden) {
        throw std::runtime_error("shared expert gate-input shape mismatch");
    }
    auto check_matrix = [&](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("shared expert ") + what + " shape mismatch: " + t.name);
        }
        if (!lowbit_gemv_supported(t.type)) {
            throw std::runtime_error(std::string("shared expert ") + what + " low-bit type unsupported: " + t.name);
        }
    };
    check_matrix(gate, shared_ffn, hidden, "gate");
    check_matrix(up, shared_ffn, hidden, "up");
    check_matrix(down, hidden, shared_ffn, "down");

    Qwen38SharedExpertWeights out;
    out.gate_input = upload_half_vector(gate_input);
    out.gate = upload_lowbit_matrix(gate);
    out.up = upload_lowbit_matrix(up);
    out.down = upload_lowbit_matrix(down);
    return out;
}

inline Qwen38MoeLayerTensors qwen38_moe_tensors(const GgufFile & g, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    Qwen38MoeLayerTensors out;
    out.router = &g.tensor(p + "ffn_gate_inp.weight");
    out.down = &g.tensor(p + "ffn_down_exps.weight");
    out.gate_up = g.optional_tensor(p + "ffn_gate_up_exps.weight");
    if (!out.gate_up) {
        out.gate = &g.tensor(p + "ffn_gate_exps.weight");
        out.up = &g.tensor(p + "ffn_up_exps.weight");
    }
    return out;
}

__global__ void q38_topk_softmax_kernel(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    constexpr int KMAX = 16;
    float best[KMAX];
    int best_id[KMAX];
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        best[j] = -3.402823466e+38F;
        best_id[j] = -1;
    }

    for (int e = 0; e < n_experts; ++e) {
        const float v = logits[e];
        if (v <= best[k - 1]) continue;

        int pos = k - 1;
        while (pos > 0 && v > best[pos - 1]) {
            best[pos] = best[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best[pos] = v;
        best_id[pos] = e;
    }

    float max_v = best[0];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        if (j >= k) break;
        const float w = expf(best[j] - max_v);
        weights[j] = w;
        ids[j] = best_id[j];
        sum += w;
    }

    const float inv = sum > 0.0f ? expert_scale / sum : 0.0f;
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        if (j >= k) break;
        weights[j] *= inv;
    }
}

__global__ void q38_silu_mul_kernel(
        const __half * __restrict__ gate,
        const __half * __restrict__ up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float s = g / (1.0f + expf(-g));
    out[i] = __float2half(s * u);
}

__global__ void q38_silu_mul_fused_kernel(
        const __half * __restrict__ gate_up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate_up[i]);
    const float u = __half2float(gate_up[n + i]);
    const float s = g / (1.0f + expf(-g));
    out[i] = __float2half(s * u);
}

__global__ void q38_weighted_accum_kernel(
        const __half * __restrict__ x,
        float weight,
        float * __restrict__ accum,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = fmaf(weight, __half2float(x[i]), accum[i]);
}

__global__ void q38_float_to_half_kernel(
        const float * __restrict__ x,
        __half * __restrict__ y,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}


__global__ void q38_add_shared_kernel(
        __half * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ gate,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float y = __half2float(routed[i]) + (*gate) * __half2float(shared[i]);
    routed[i] = __float2half(y);
}

class Qwen38MoeScratch {
public:
    Qwen38MoeScratch() = default;
    Qwen38MoeScratch(const Qwen38MoeScratch &) = delete;
    Qwen38MoeScratch & operator=(const Qwen38MoeScratch &) = delete;

    ~Qwen38MoeScratch() { reset(); }

    void initialize(int n_experts, int k, int hidden, int expert_ffn) {
        if (d_logits_) throw std::runtime_error("MoE scratch already initialized");
        if (n_experts <= 0 || n_experts > 4096 || k <= 0 || k > 16 || k > n_experts ||
            hidden <= 0 || expert_ffn <= 0) {
            throw std::runtime_error("invalid MoE scratch geometry");
        }

        n_experts_ = n_experts;
        k_ = k;
        hidden_ = hidden;
        expert_ffn_ = expert_ffn;

        alloc(reinterpret_cast<void **>(&d_logits_), static_cast<size_t>(n_experts_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_ids_), static_cast<size_t>(k_) * sizeof(int));
        alloc(reinterpret_cast<void **>(&d_weights_), static_cast<size_t>(k_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_gate_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_up_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_gate_up_), static_cast<size_t>(2 * expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_act_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_down_), static_cast<size_t>(hidden_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_accum_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_shared_gate_), sizeof(float));

        host_alloc(reinterpret_cast<void **>(&h_ids_), static_cast<size_t>(k_) * sizeof(int));
        host_alloc(reinterpret_cast<void **>(&h_weights_), static_cast<size_t>(k_) * sizeof(float));
    }

    int n_experts() const { return n_experts_; }
    int k() const { return k_; }
    int hidden() const { return hidden_; }
    int expert_ffn() const { return expert_ffn_; }

    float * d_logits() const { return d_logits_; }
    int * d_ids() const { return d_ids_; }
    float * d_weights() const { return d_weights_; }
    __half * d_gate() const { return d_gate_; }
    __half * d_up() const { return d_up_; }
    __half * d_gate_up() const { return d_gate_up_; }
    __half * d_act() const { return d_act_; }
    __half * d_down() const { return d_down_; }
    float * d_accum() const { return d_accum_; }
    float * d_shared_gate() const { return d_shared_gate_; }
    int * h_ids() const { return h_ids_; }
    float * h_weights() const { return h_weights_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc MoE scratch failed: ") + cudaGetErrorString(e));
    }

    static void host_alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMallocHost(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMallocHost MoE scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (d_logits_) cudaFree(d_logits_);
        if (d_ids_) cudaFree(d_ids_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_gate_) cudaFree(d_gate_);
        if (d_up_) cudaFree(d_up_);
        if (d_gate_up_) cudaFree(d_gate_up_);
        if (d_act_) cudaFree(d_act_);
        if (d_down_) cudaFree(d_down_);
        if (d_accum_) cudaFree(d_accum_);
        if (d_shared_gate_) cudaFree(d_shared_gate_);
        if (h_ids_) cudaFreeHost(h_ids_);
        if (h_weights_) cudaFreeHost(h_weights_);
        d_logits_ = nullptr;
        d_ids_ = nullptr;
        d_weights_ = nullptr;
        d_gate_ = d_up_ = d_gate_up_ = d_act_ = d_down_ = nullptr;
        d_accum_ = nullptr;
        d_shared_gate_ = nullptr;
        h_ids_ = nullptr;
        h_weights_ = nullptr;
    }

    int n_experts_ = 0;
    int k_ = 0;
    int hidden_ = 0;
    int expert_ffn_ = 0;

    float * d_logits_ = nullptr;
    int * d_ids_ = nullptr;
    float * d_weights_ = nullptr;
    __half * d_gate_ = nullptr;
    __half * d_up_ = nullptr;
    __half * d_gate_up_ = nullptr;
    __half * d_act_ = nullptr;
    __half * d_down_ = nullptr;
    float * d_accum_ = nullptr;
    float * d_shared_gate_ = nullptr;
    int * h_ids_ = nullptr;
    float * h_weights_ = nullptr;
};

inline void q38_select_experts(
        const GpuMatrix & router,
        const __half * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }

    launch_matrix_gemv_float(router, x, scratch.d_logits(), stream);

    q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights());

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert ids failed: ") + cudaGetErrorString(e));

    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert weights failed: ") + cudaGetErrorString(e));

    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("expert selection sync failed: ") + cudaGetErrorString(e));
}

inline void q38_finish_moe(
        Qwen38MoeScratch & scratch,
        __half * out,
        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (scratch.hidden() + threads - 1) / threads;
    q38_float_to_half_kernel<<<blocks, threads, 0, stream>>>(
        scratch.d_accum(), out, scratch.hidden());
}

inline void run_qwen38_routed_moe_fused(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        __half * out,
        float expert_scale,
        cudaStream_t stream) {
    if (gate_up_exps.dims.size() != 3 || down_exps.dims.size() != 3) {
        throw std::runtime_error("Qwen3.8 fused expert tensors must be 3D");
    }
    if (static_cast<int>(gate_up_exps.dims[1]) != 2 * scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 fused gate-up FFN width mismatch");
    }

    q38_select_experts(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaMemsetAsync(
        scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate_up, x, scratch.d_gate_up(), stream);
        q38_silu_mul_fused_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}

inline void run_qwen38_routed_moe_separate(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        __half * out,
        float expert_scale,
        cudaStream_t stream) {
    q38_select_experts(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaMemsetAsync(
        scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate, x, scratch.d_gate(), stream);
        launch_lowbit_gemv_half(w.up, x, scratch.d_up(), stream);
        q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}


inline void run_qwen38_shared_expert(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        __half * routed_in_out,
        cudaStream_t stream) {
    if (shared.gate_input.size != scratch.hidden() ||
        shared.gate.rows != scratch.expert_ffn() || shared.gate.cols != scratch.hidden() ||
        shared.up.rows != scratch.expert_ffn() || shared.up.cols != scratch.hidden() ||
        shared.down.rows != scratch.hidden() || shared.down.cols != scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 shared expert geometry mismatch");
    }

    half_dot_kernel<<<1, 256, 0, stream>>>(
        shared.gate_input.data, x, scratch.d_shared_gate(), scratch.hidden());
    sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(scratch.d_shared_gate());

    launch_lowbit_gemv_half(shared.gate, x, scratch.d_gate(), stream);
    launch_lowbit_gemv_half(shared.up, x, scratch.d_up(), stream);

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
    launch_lowbit_gemv_half(shared.down, scratch.d_act(), scratch.d_down(), stream);
    q38_add_shared_kernel<<<hidden_blocks, threads, 0, stream>>>(
        routed_in_out, scratch.d_down(), scratch.d_shared_gate(), scratch.hidden());
}

} // namespace wd
