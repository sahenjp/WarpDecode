#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "qwen38_layout.hpp"
#include "qwen38_rope.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38QsaWeights {
    GpuMatrix q;
    GpuMatrix k;
    GpuMatrix v;
    GpuMatrix out;

    DeviceHalfArray q_norm;
    DeviceHalfArray k_norm;

    GpuMatrix index_q;
    GpuMatrix index_k;
    DeviceHalfArray index_q_norm;
    DeviceHalfArray index_k_norm;
};

inline Qwen38QsaWeights load_qwen38_qsa_weights(
        const GgufFile & g,
        int il,
        const Qwen38Layout & q) {
    const std::string p = "blk." + std::to_string(il) + ".";

    const TensorInfo & wq = g.tensor(p + "attn_q.weight");
    const TensorInfo & wk = g.tensor(p + "attn_k.weight");
    const TensorInfo & wv = g.tensor(p + "attn_v.weight");
    const TensorInfo & wo = g.tensor(p + "attn_output.weight");
    const TensorInfo & qn = g.tensor(p + "attn_q_norm.weight");
    const TensorInfo & kn = g.tensor(p + "attn_k_norm.weight");

    const TensorInfo & iq = g.tensor(p + "indexer.q_proj.weight");
    const TensorInfo & ik = g.tensor(p + "indexer.k_proj.weight");
    const TensorInfo & iqn = g.tensor(p + "indexer.q_norm.weight");
    const TensorInfo & ikn = g.tensor(p + "indexer.k_norm.weight");

    const int q_rows = q.heads * q.key_length * 2;
    const int k_rows = q.kv_heads * q.key_length;
    const int v_rows = q.kv_heads * q.value_length;
    const int attn_rows = q.heads * q.value_length;

    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 ||
            static_cast<int>(t.dims[0]) != cols ||
            static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("QSA ") + what + " shape mismatch: " + t.name);
        }
    };

    check2(wq, q_rows, q.hidden, "Q");
    check2(wk, k_rows, q.hidden, "K");
    check2(wv, v_rows, q.hidden, "V");
    check2(wo, q.hidden, attn_rows, "output");
    check2(iq, q.indexer_heads * q.indexer_key_length, q.hidden, "index Q");
    check2(ik, q.indexer_key_length, q.hidden, "index K");

    if (static_cast<int>(qn.elements()) != q.key_length ||
        static_cast<int>(kn.elements()) != q.key_length ||
        static_cast<int>(iqn.elements()) != q.indexer_key_length ||
        static_cast<int>(ikn.elements()) != q.indexer_key_length) {
        throw std::runtime_error("QSA norm vector shape mismatch");
    }

    if (q.key_length != q.value_length) {
        throw std::runtime_error("Qwen3.8 fast QSA currently requires key_length == value_length");
    }
    Qwen38QsaWeights out;
    out.q = upload_matrix(wq);
    out.k = upload_matrix(wk);
    out.v = upload_matrix(wv);
    out.out = upload_matrix(wo);
    out.q_norm = upload_half_array(qn);
    out.k_norm = upload_half_array(kn);
    out.index_q = upload_matrix(iq);
    out.index_k = upload_matrix(ik);
    out.index_q_norm = upload_half_array(iqn);
    out.index_k_norm = upload_half_array(ikn);
    return out;
}

class Qwen38QsaState {
public:
    Qwen38QsaState() = default;
    Qwen38QsaState(const Qwen38QsaState &) = delete;
    Qwen38QsaState & operator=(const Qwen38QsaState &) = delete;
    ~Qwen38QsaState() { reset(); }

    void initialize(
            int max_ctx,
            int ratio,
            const Qwen38Layout & q) {
        if (k_cache_) throw std::runtime_error("QSA state already initialized");
        if (max_ctx <= 0 || ratio <= 0 || ratio > 64) {
            throw std::runtime_error("invalid QSA cache geometry");
        }
        if (q.key_length != q.value_length) {
            throw std::runtime_error("unsupported QSA fast-path K/V geometry");
        }
        if (q.indexer_key_length <= 0 || q.indexer_key_length > 256) {
            throw std::runtime_error("unsupported QSA indexer key length");
        }

        max_ctx_ = max_ctx;
        ratio_ = ratio;
        max_blocks_ = max_ctx_ / ratio_;
        if (max_blocks_ <= 0) throw std::runtime_error("QSA context is shorter than one block");

        const size_t kv_width = static_cast<size_t>(q.kv_heads) * q.key_length;
        const size_t idx_width = static_cast<size_t>(q.indexer_key_length);

        alloc(reinterpret_cast<void **>(&k_cache_), static_cast<size_t>(max_ctx_) * kv_width * sizeof(__half));
        alloc(reinterpret_cast<void **>(&v_cache_), static_cast<size_t>(max_ctx_) * kv_width * sizeof(__half));
        alloc(reinterpret_cast<void **>(&index_raw_), static_cast<size_t>(max_ctx_) * idx_width * sizeof(__half));
        alloc(reinterpret_cast<void **>(&index_block_), static_cast<size_t>(max_blocks_) * idx_width * sizeof(__half));

        alloc(reinterpret_cast<void **>(&scores_), static_cast<size_t>(max_blocks_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&scores_sorted_), static_cast<size_t>(max_blocks_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&ids_), static_cast<size_t>(max_blocks_) * sizeof(int));
        alloc(reinterpret_cast<void **>(&ids_sorted_), static_cast<size_t>(max_blocks_) * sizeof(int));

        std::vector<int> host_ids(static_cast<size_t>(max_blocks_));
        std::iota(host_ids.begin(), host_ids.end(), 0);
        check(cudaMemcpy(ids_, host_ids.data(), host_ids.size() * sizeof(int), cudaMemcpyHostToDevice),
              "copy QSA block ids");

        sort_temp_bytes_ = 0;
        check(cub::DeviceRadixSort::SortPairsDescending(
                  nullptr, sort_temp_bytes_,
                  scores_, scores_sorted_, ids_, ids_sorted_,
                  max_blocks_, 0, sizeof(float) * 8),
              "query QSA sort temp");
        alloc(&sort_temp_, sort_temp_bytes_);

        clear(nullptr);
    }

    void clear(cudaStream_t stream) {
        if (!k_cache_) return;
        cudaMemsetAsync(k_cache_, 0, k_bytes(), stream);
        cudaMemsetAsync(v_cache_, 0, v_bytes(), stream);
        cudaMemsetAsync(index_raw_, 0, index_raw_bytes(), stream);
        cudaMemsetAsync(index_block_, 0, index_block_bytes(), stream);
    }

    int max_ctx() const { return max_ctx_; }
    int max_blocks() const { return max_blocks_; }
    int ratio() const { return ratio_; }

    __half * k_cache() const { return k_cache_; }
    __half * v_cache() const { return v_cache_; }
    __half * index_raw() const { return index_raw_; }
    __half * index_block() const { return index_block_; }
    float * scores() const { return scores_; }
    float * scores_sorted() const { return scores_sorted_; }
    int * ids() const { return ids_; }
    int * ids_sorted() const { return ids_sorted_; }
    void * sort_temp() const { return sort_temp_; }
    size_t sort_temp_bytes() const { return sort_temp_bytes_; }

private:
    static void check(cudaError_t e, const char * what) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
        }
    }

    static void alloc(void ** p, size_t n) {
        if (n == 0) return;
        check(cudaMalloc(p, n), "cudaMalloc QSA state");
    }

    size_t k_bytes() const { return k_cache_bytes_; }
    size_t v_bytes() const { return v_cache_bytes_; }
    size_t index_raw_bytes() const { return index_raw_bytes_; }
    size_t index_block_bytes() const { return index_block_bytes_; }

public:
    void set_geometry_bytes(
            size_t k,
            size_t v,
            size_t raw,
            size_t block) {
        k_cache_bytes_ = k;
        v_cache_bytes_ = v;
        index_raw_bytes_ = raw;
        index_block_bytes_ = block;
    }

private:
    void reset() {
        if (k_cache_) cudaFree(k_cache_);
        if (v_cache_) cudaFree(v_cache_);
        if (index_raw_) cudaFree(index_raw_);
        if (index_block_) cudaFree(index_block_);
        if (scores_) cudaFree(scores_);
        if (scores_sorted_) cudaFree(scores_sorted_);
        if (ids_) cudaFree(ids_);
        if (ids_sorted_) cudaFree(ids_sorted_);
        if (sort_temp_) cudaFree(sort_temp_);
        k_cache_ = v_cache_ = index_raw_ = index_block_ = nullptr;
        scores_ = scores_sorted_ = nullptr;
        ids_ = ids_sorted_ = nullptr;
        sort_temp_ = nullptr;
    }

    int max_ctx_ = 0;
    int max_blocks_ = 0;
    int ratio_ = 0;

    __half * k_cache_ = nullptr;
    __half * v_cache_ = nullptr;
    __half * index_raw_ = nullptr;
    __half * index_block_ = nullptr;

    float * scores_ = nullptr;
    float * scores_sorted_ = nullptr;
    int * ids_ = nullptr;
    int * ids_sorted_ = nullptr;
    void * sort_temp_ = nullptr;
    size_t sort_temp_bytes_ = 0;

    size_t k_cache_bytes_ = 0;
    size_t v_cache_bytes_ = 0;
    size_t index_raw_bytes_ = 0;
    size_t index_block_bytes_ = 0;
};

class Qwen38QsaScratch {
public:
    Qwen38QsaScratch() = default;
    Qwen38QsaScratch(const Qwen38QsaScratch &) = delete;
    Qwen38QsaScratch & operator=(const Qwen38QsaScratch &) = delete;
    ~Qwen38QsaScratch() { reset(); }

    void initialize(const Qwen38Layout & q) {
        if (q_full_) throw std::runtime_error("QSA scratch already initialized");
        const int head = q.key_length;
        q_full_n_ = q.heads * head * 2;
        q_n_ = q.heads * head;
        kv_n_ = q.kv_heads * head;
        idx_q_n_ = q.indexer_heads * q.indexer_key_length;
        idx_k_n_ = q.indexer_key_length;

        alloc(reinterpret_cast<void **>(&q_full_), static_cast<size_t>(q_full_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&q_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&k_), static_cast<size_t>(kv_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&v_), static_cast<size_t>(kv_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&attn_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&idx_q_), static_cast<size_t>(idx_q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&idx_k_), static_cast<size_t>(idx_k_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&final_), static_cast<size_t>(q.hidden) * sizeof(__half));
    }

    __half * q_full() const { return q_full_; }
    __half * q() const { return q_; }
    __half * gate() const { return gate_; }
    __half * k() const { return k_; }
    __half * v() const { return v_; }
    __half * attn() const { return attn_; }
    __half * idx_q() const { return idx_q_; }
    __half * idx_k() const { return idx_k_; }
    __half * final() const { return final_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc QSA scratch failed: ") + cudaGetErrorString(e));
    }

    void reset() {
        if (q_full_) cudaFree(q_full_);
        if (q_) cudaFree(q_);
        if (gate_) cudaFree(gate_);
        if (k_) cudaFree(k_);
        if (v_) cudaFree(v_);
        if (attn_) cudaFree(attn_);
        if (idx_q_) cudaFree(idx_q_);
        if (idx_k_) cudaFree(idx_k_);
        if (final_) cudaFree(final_);
        q_full_ = q_ = gate_ = k_ = v_ = attn_ = idx_q_ = idx_k_ = final_ = nullptr;
    }

    int q_full_n_ = 0;
    int q_n_ = 0;
    int kv_n_ = 0;
    int idx_q_n_ = 0;
    int idx_k_n_ = 0;

    __half * q_full_ = nullptr;
    __half * q_ = nullptr;
    __half * gate_ = nullptr;
    __half * k_ = nullptr;
    __half * v_ = nullptr;
    __half * attn_ = nullptr;
    __half * idx_q_ = nullptr;
    __half * idx_k_ = nullptr;
    __half * final_ = nullptr;
};

__global__ void q38_qsa_extract_q_gate_kernel(
        const __half * __restrict__ q_full,
        __half * __restrict__ q,
        __half * __restrict__ gate,
        int head_dim,
        int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = head_dim * n_heads;
    if (i >= n) return;
    const int h = i / head_dim;
    const int d = i - h * head_dim;
    const int base = h * (2 * head_dim);
    q[i] = q_full[base + d];
    gate[i] = q_full[base + head_dim + d];
}

__global__ void q38_rmsnorm_heads_inplace_kernel(
        __half * x,
        const __half * gamma,
        int head_dim,
        int n_heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    __shared__ float smem[256];

    const int base = h * head_dim;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = __half2float(x[base + d]);
        sum = fmaf(v, v, sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int idx = base + d;
        x[idx] = __float2half(__half2float(x[idx]) * inv * __half2float(gamma[d]));
    }
}

__global__ void q38_qsa_pool_block_kernel(
        const __half * __restrict__ raw,
        __half * __restrict__ pooled,
        const __half * __restrict__ norm,
        int start_token,
        int ratio,
        int dim,
        float eps) {
    __shared__ float smem[256];
    float v = 0.0f;
    const int d = threadIdx.x;
    if (d < dim) {
        for (int j = 0; j < ratio; ++j) {
            v += __half2float(raw[static_cast<size_t>(start_token + j) * dim + d]);
        }
        v /= static_cast<float>(ratio);
    }

    const float sq = d < dim ? v * v : 0.0f;
    smem[d] = sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) smem[d] += smem[d + stride];
        __syncthreads();
    }

    if (d < dim) {
        const float inv = rsqrtf(smem[0] / static_cast<float>(dim) + eps);
        pooled[d] = __float2half(v * inv * __half2float(norm[d]));
    }
}

__global__ void q38_qsa_score_blocks_kernel(
        const __half * __restrict__ query,
        const __half * __restrict__ block_keys,
        float * __restrict__ scores,
        int n_blocks,
        int dim,
        int n_heads) {
    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    __shared__ float red[256];
    __shared__ float total;

    if (threadIdx.x == 0) total = 0.0f;
    __syncthreads();

    const __half * key = block_keys + static_cast<size_t>(b) * dim;
    for (int h = 0; h < n_heads; ++h) {
        float dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            dot = fmaf(
                __half2float(query[h * dim + d]),
                __half2float(key[d]),
                dot);
        }
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) total += fmaxf(red[0], 0.0f);
        __syncthreads();
    }

    if (threadIdx.x == 0) scores[b] = total;
}

__global__ void q38_qsa_sparse_attention_kernel(
        const __half * __restrict__ q,
        const __half * __restrict__ gate,
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        const int * __restrict__ selected_blocks,
        __half * __restrict__ out,
        int n_selected_blocks,
        int ratio,
        int tail_start,
        int pos,
        int n_heads,
        int kv_heads,
        int head_dim,
        float scale) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= n_heads) return;

    __shared__ float red[256];
    __shared__ float sh_alpha;
    __shared__ float sh_beta;
    __shared__ float sh_l;

    const int kvh = (h * kv_heads) / n_heads;
    const float qd = d < head_dim ? __half2float(q[h * head_dim + d]) : 0.0f;
    float acc = 0.0f;
    float m = -3.402823466e+38F;
    float l = 0.0f;

    auto consume = [&](int token) {
        const size_t base = (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
        const float dot = d < head_dim ? qd * __half2float(k_cache[base + d]) : 0.0f;
        red[d] = dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }

        if (d == 0) {
            const float score = red[0] * scale;
            const float nm = fmaxf(m, score);
            sh_alpha = m > -3.0e38F ? expf(m - nm) : 0.0f;
            sh_beta = expf(score - nm);
            sh_l = l * sh_alpha + sh_beta;
            m = nm;
        }
        __syncthreads();
        // l is thread-local; every feature lane needs the updated softmax denominator.
        l = sh_l;

        if (d < head_dim) {
            acc = acc * sh_alpha + sh_beta * __half2float(v_cache[base + d]);
        }
        __syncthreads();
    };

    for (int sb = 0; sb < n_selected_blocks; ++sb) {
        const int block_id = selected_blocks[sb];
        const int start = block_id * ratio;
        for (int j = 0; j < ratio; ++j) consume(start + j);
    }

    for (int token = tail_start; token <= pos; ++token) consume(token);

    if (d < head_dim) {
        const float attn = l > 0.0f ? acc / l : 0.0f;
        const float g = 1.0f / (1.0f + expf(-__half2float(gate[h * head_dim + d])));
        out[h * head_dim + d] = __float2half(attn * g);
    }
}

inline void q38_qsa_store_kv(
        Qwen38QsaState & state,
        const Qwen38QsaScratch & scratch,
        const Qwen38Layout & q,
        int pos,
        cudaStream_t stream) {
    const size_t kv_n = static_cast<size_t>(q.kv_heads) * q.key_length;
    cudaError_t e = cudaMemcpyAsync(
        state.k_cache() + static_cast<size_t>(pos) * kv_n,
        scratch.k(), kv_n * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("store QSA K failed: ") + cudaGetErrorString(e));

    e = cudaMemcpyAsync(
        state.v_cache() + static_cast<size_t>(pos) * kv_n,
        scratch.v(), kv_n * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("store QSA V failed: ") + cudaGetErrorString(e));
}

inline __half * run_qwen38_qsa(
        const __half * x,
        const Qwen38QsaWeights & w,
        Qwen38QsaState & state,
        Qwen38QsaScratch & scratch,
        const Qwen38Layout & q,
        int il,
        int pos,
        const Qwen38Position4 & position,
        cudaStream_t stream) {
    if (pos < 0 || pos >= state.max_ctx()) throw std::runtime_error("QSA position out of range");
    if (il < 0 || il >= q.trunk_layers || !q.is_qsa(il)) {
        throw std::runtime_error("run_qwen38_qsa called on non-QSA layer");
    }

    const int ratio = state.ratio();
    if (q.compress_ratios.empty() || static_cast<int>(q.compress_ratios[static_cast<size_t>(il)]) != ratio) {
        throw std::runtime_error("QSA state ratio does not match layer metadata");
    }
    if (q.indexer_top_k % ratio != 0) {
        throw std::runtime_error("fast QSA currently requires indexer_top_k divisible by compression ratio");
    }

    const int head_dim = q.key_length;
    const int q_n = q.heads * head_dim;
    const int idx_dim = q.indexer_key_length;
    const int block_budget = q.indexer_top_k / ratio;

    launch_matrix_gemv_half(w.q, x, scratch.q_full(), stream);
    launch_matrix_gemv_half(w.k, x, scratch.k(), stream);
    launch_matrix_gemv_half(w.v, x, scratch.v(), stream);

    const int threads = 256;
    q38_qsa_extract_q_gate_kernel<<<(q_n + threads - 1) / threads, threads, 0, stream>>>(
        scratch.q_full(), scratch.q(), scratch.gate(), head_dim, q.heads);

    q38_rmsnorm_heads_inplace_kernel<<<q.heads, 256, 0, stream>>>(
        scratch.q(), w.q_norm.data, head_dim, q.heads, q.rms_eps);
    q38_rmsnorm_heads_inplace_kernel<<<q.kv_heads, 256, 0, stream>>>(
        scratch.k(), w.k_norm.data, head_dim, q.kv_heads, q.rms_eps);

    launch_qwen38_imrope(scratch.q(), head_dim, q.heads, position, q, stream);
    launch_qwen38_imrope(scratch.k(), head_dim, q.kv_heads, position, q, stream);
    q38_qsa_store_kv(state, scratch, q, pos, stream);

    // Raw index keys are cached exactly as projected. Pooling + norm + IMRoPE
    // happens once when a block becomes complete.
    launch_matrix_gemv_half(w.index_k, x, scratch.idx_k(), stream);
    cudaError_t e = cudaMemcpyAsync(
        state.index_raw() + static_cast<size_t>(pos) * idx_dim,
        scratch.idx_k(), static_cast<size_t>(idx_dim) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("store QSA index K failed: ") + cudaGetErrorString(e));

    if ((pos + 1) % ratio == 0) {
        const int block_id = pos / ratio;
        __half * pooled = state.index_block() + static_cast<size_t>(block_id) * idx_dim;
        q38_qsa_pool_block_kernel<<<1, 256, 0, stream>>>(
            state.index_raw(), pooled, w.index_k_norm.data,
            block_id * ratio, ratio, idx_dim, q.rms_eps);
        launch_qwen38_imrope(
            pooled, idx_dim, 1,
            Qwen38Position4::text(block_id * ratio),
            q, stream);
    }

    const int full_blocks = (pos + 1) / ratio;
    const int tail_start = full_blocks * ratio;
    int selected_blocks = std::min(full_blocks, block_budget);
    const int * selected_ids = state.ids();

    if (full_blocks > block_budget) {
        launch_matrix_gemv_half(w.index_q, x, scratch.idx_q(), stream);
        q38_rmsnorm_heads_inplace_kernel<<<q.indexer_heads, 256, 0, stream>>>(
            scratch.idx_q(), w.index_q_norm.data, idx_dim, q.indexer_heads, q.rms_eps);
        launch_qwen38_imrope(
            scratch.idx_q(), idx_dim, q.indexer_heads,
            position, q, stream);

        q38_qsa_score_blocks_kernel<<<full_blocks, 256, 0, stream>>>(
            scratch.idx_q(), state.index_block(), state.scores(),
            full_blocks, idx_dim, q.indexer_heads);

        size_t sort_temp_bytes = state.sort_temp_bytes();
        e = cub::DeviceRadixSort::SortPairsDescending(
            state.sort_temp(), sort_temp_bytes,
            state.scores(), state.scores_sorted(),
            state.ids(), state.ids_sorted(),
            full_blocks, 0, 32, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("QSA block sort failed: ") + cudaGetErrorString(e));
        selected_ids = state.ids_sorted();
    }

    const float scale = q.attention_scale != 0.0f
        ? q.attention_scale
        : rsqrtf(static_cast<float>(head_dim));

    q38_qsa_sparse_attention_kernel<<<q.heads, 256, 0, stream>>>(
        scratch.q(), scratch.gate(),
        state.k_cache(), state.v_cache(),
        selected_ids, scratch.attn(),
        selected_blocks, ratio, tail_start, pos,
        q.heads, q.kv_heads, head_dim, scale);

    launch_matrix_gemv_half(w.out, scratch.attn(), scratch.final(), stream);
    return scratch.final();
}

inline void initialize_qwen38_qsa_state(
        Qwen38QsaState & state,
        int max_ctx,
        int ratio,
        const Qwen38Layout & q) {
    state.initialize(max_ctx, ratio, q);
    const size_t kv_width = static_cast<size_t>(q.kv_heads) * q.key_length;
    const size_t idx_width = static_cast<size_t>(q.indexer_key_length);
    state.set_geometry_bytes(
        static_cast<size_t>(max_ctx) * kv_width * sizeof(__half),
        static_cast<size_t>(max_ctx) * kv_width * sizeof(__half),
        static_cast<size_t>(max_ctx) * idx_width * sizeof(__half),
        static_cast<size_t>(max_ctx / ratio) * idx_width * sizeof(__half));
}

} // namespace wd
