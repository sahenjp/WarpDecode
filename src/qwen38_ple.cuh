#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "ple_reader.hpp"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38PleWeights {
    GpuMatrix key;
    GpuMatrix value;
    DeviceHalfArray norm_key;
    DeviceHalfArray norm_query;
    DeviceHalfArray norm_conv;
    DeviceHalfArray conv;
};

inline Qwen38PleWeights load_qwen38_ple_weights(
        const GgufFile & g,
        int il,
        const Qwen38Layout & q) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const TensorInfo & key = g.tensor(p + "ple_key.weight");
    const TensorInfo & value = g.tensor(p + "ple_value.weight");
    const TensorInfo & nk = g.tensor(p + "ple_norm_key.weight");
    const TensorInfo & nq = g.tensor(p + "ple_norm_query.weight");
    const TensorInfo & nc = g.tensor(p + "ple_norm_conv.weight");
    const TensorInfo & conv = g.tensor(p + "ple_conv1d.weight");

    const int hc_dim = q.hidden * q.hc_count;
    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 ||
            static_cast<int>(t.dims[0]) != cols ||
            static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("PLE ") + what + " shape mismatch: " + t.name);
        }
    };
    check2(key, hc_dim, q.hidden, "key");
    check2(value, q.hidden, q.hidden, "value");

    if (static_cast<int>(nk.elements()) != hc_dim ||
        static_cast<int>(nq.elements()) != hc_dim ||
        static_cast<int>(nc.elements()) != hc_dim) {
        throw std::runtime_error("PLE norm shape mismatch");
    }
    if (conv.dims.size() != 2 ||
        static_cast<int>(conv.dims[0]) != q.ple_conv_kernel ||
        static_cast<int>(conv.dims[1]) != hc_dim) {
        throw std::runtime_error("PLE conv shape mismatch: " + conv.name);
    }

    Qwen38PleWeights w;
    w.key = upload_matrix(key);
    w.value = upload_matrix(value);
    w.norm_key = upload_half_array(nk);
    w.norm_query = upload_half_array(nq);
    w.norm_conv = upload_half_array(nc);
    w.conv = upload_half_array(conv);
    return w;
}

__global__ void q38_iq4_nl_rows_to_half_kernel(
        const uint8_t * __restrict__ raw,
        __half * __restrict__ out,
        int rows,
        int cols,
        int row_bytes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (i >= total) return;

    const int r = i / cols;
    const int c = i - r * cols;
    const int b = c >> 5;
    const int lane = c & 31;
    const uint8_t * blk = raw + static_cast<size_t>(r) * row_bytes + b * 18;
    const float d = lowbit_half(blk);
    const uint8_t packed = blk[2 + (lane & 15)];
    const uint8_t q = lane < 16 ? (packed & 0x0f) : (packed >> 4);
    out[i] = __float2half(d * iq4_nl_value(q));
}

class Qwen38PleTable {
public:
    Qwen38PleTable() = default;
    Qwen38PleTable(const Qwen38PleTable &) = delete;
    Qwen38PleTable & operator=(const Qwen38PleTable &) = delete;

    ~Qwen38PleTable() {
        if (h_raw_) cudaFreeHost(h_raw_);
        if (d_raw_) cudaFree(d_raw_);
        if (d_emb_) cudaFree(d_emb_);
    }

    void initialize(
            const std::string & path,
            const GgufFile & g,
            const Qwen38Layout & q) {
        if (reader_) throw std::runtime_error("PLE table already initialized");
        const TensorInfo & t = g.tensor("per_layer_token_embd.weight");
        if (t.dims.size() != 2 ||
            static_cast<int>(t.dims[0]) != q.ple_embedding_length) {
            throw std::runtime_error("PLE embedding table shape mismatch");
        }
        if (t.type != GgmlType::IQ4_NL) {
            throw std::runtime_error("PLE table currently requires IQ4_NL");
        }

        rows_ = t.dims[1];
        head_dim_ = q.ple_embedding_length;
        n_heads_ = q.ple_head_count();

        TensorInfo row;
        row.name = t.name + ".row";
        row.dims = {static_cast<uint64_t>(head_dim_)};
        row.type = t.type;
        row_bytes_ = static_cast<uint32_t>(row.byte_size());

        reader_ = std::make_unique<PleDirectReader>(
            path,
            static_cast<uint64_t>(g.data_offset()) + t.relative_offset,
            row_bytes_,
            rows_);

        const size_t raw_bytes = static_cast<size_t>(row_bytes_) * n_heads_;
        cudaError_t e = cudaMallocHost(reinterpret_cast<void **>(&h_raw_), raw_bytes);
        if (e != cudaSuccess) throw std::runtime_error("cudaMallocHost PLE rows failed");
        e = cudaMalloc(reinterpret_cast<void **>(&d_raw_), raw_bytes);
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE rows failed");
        e = cudaMalloc(
            reinterpret_cast<void **>(&d_emb_),
            static_cast<size_t>(head_dim_) * n_heads_ * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE embedding failed");
    }

    __half * gather(
            const std::vector<uint64_t> & rows,
            cudaStream_t stream) {
        if (!reader_ || static_cast<int>(rows.size()) != n_heads_) {
            throw std::runtime_error("invalid PLE gather request");
        }
        reader_->read_rows(rows, h_raw_);
        const size_t raw_bytes = static_cast<size_t>(row_bytes_) * n_heads_;
        cudaError_t e = cudaMemcpyAsync(
            d_raw_, h_raw_, raw_bytes, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error("copy PLE rows failed");

        const int total = head_dim_ * n_heads_;
        const int threads = 256;
        q38_iq4_nl_rows_to_half_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            d_raw_, d_emb_, n_heads_, head_dim_, static_cast<int>(row_bytes_));
        return d_emb_;
    }

    int head_dim() const { return head_dim_; }
    int n_heads() const { return n_heads_; }

private:
    std::unique_ptr<PleDirectReader> reader_;
    uint64_t rows_ = 0;
    uint32_t row_bytes_ = 0;
    int head_dim_ = 0;
    int n_heads_ = 0;
    uint8_t * h_raw_ = nullptr;
    uint8_t * d_raw_ = nullptr;
    __half * d_emb_ = nullptr;
};

inline std::vector<uint64_t> qwen38_ple_row_indices(
        const Qwen38Layout & q,
        int token,
        const std::vector<int> & history) {
    if (q.ple_ngram_size < 2 || q.ple_head_count() <= 0) {
        throw std::runtime_error("PLE metadata is not initialized");
    }

    std::vector<int64_t> ctx(static_cast<size_t>(q.ple_ngram_size), q.ple_eos_token_id);
    ctx[0] = token;
    bool cut = false;
    for (int s = 1; s < q.ple_ngram_size; ++s) {
        int t = q.ple_eos_token_id;
        if (!cut && history.size() >= static_cast<size_t>(s)) {
            t = history[history.size() - static_cast<size_t>(s)];
        } else {
            cut = true;
        }
        if (t < 0 || t == q.ple_eos_token_id) cut = true;
        ctx[static_cast<size_t>(s)] = cut ? q.ple_eos_token_id : t;
    }

    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(q.ple_head_count()));
    const int per_gram = q.ple_heads_per_ngram;
    for (int n = 2; n <= q.ple_ngram_size; ++n) {
        uint64_t mixed = static_cast<uint64_t>(ctx[0]) * q.ple_layer_multipliers[0];
        for (int j = 1; j < n; ++j) {
            mixed ^= static_cast<uint64_t>(ctx[static_cast<size_t>(j)]) *
                     q.ple_layer_multipliers[static_cast<size_t>(j)];
        }
        const int base = (n - 2) * per_gram;
        for (int g = 0; g < per_gram; ++g) {
            const int h = base + g;
            const uint64_t vocab = q.ple_head_vocab_sizes[static_cast<size_t>(h)];
            const uint64_t off = q.ple_head_offsets[static_cast<size_t>(h)];
            out.push_back(mixed % vocab + off);
        }
    }
    return out;
}

class Qwen38PleState {
public:
    Qwen38PleState() = default;
    Qwen38PleState(const Qwen38PleState &) = delete;
    Qwen38PleState & operator=(const Qwen38PleState &) = delete;
    ~Qwen38PleState() { if (history_) cudaFree(history_); }

    void initialize(const Qwen38Layout & q) {
        const int hist = (q.ple_conv_kernel - 1) * q.ple_ngram_size;
        const int channels = q.hidden * q.hc_count;
        if (hist <= 0 || channels <= 0) throw std::runtime_error("invalid PLE state geometry");
        hist_ = hist;
        channels_ = channels;
        const size_t n = static_cast<size_t>(hist_) * channels_;
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&history_), n * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE history failed");
        cudaMemset(history_, 0, n * sizeof(__half));
    }

    void clear(cudaStream_t stream) {
        if (history_) cudaMemsetAsync(
            history_, 0, static_cast<size_t>(hist_) * channels_ * sizeof(__half), stream);
    }

    __half * history() const { return history_; }
    int hist() const { return hist_; }
    int channels() const { return channels_; }

private:
    __half * history_ = nullptr;
    int hist_ = 0;
    int channels_ = 0;
};

class Qwen38PleScratch {
public:
    Qwen38PleScratch() = default;
    Qwen38PleScratch(const Qwen38PleScratch &) = delete;
    Qwen38PleScratch & operator=(const Qwen38PleScratch &) = delete;
    ~Qwen38PleScratch() { reset(); }

    void initialize(const Qwen38Layout & q) {
        const int hc_dim = q.hidden * q.hc_count;
        alloc(&key_, hc_dim);
        alloc(&key_norm_, hc_dim);
        alloc(&query_norm_, hc_dim);
        alloc(&value_, q.hidden);
        alloc(&gated_, hc_dim);
        alloc(&normed_, hc_dim);
        alloc(&conv_out_, hc_dim);
        alloc(&gate_, q.hc_count);
    }

    __half * key() const { return key_; }
    __half * key_norm() const { return key_norm_; }
    __half * query_norm() const { return query_norm_; }
    __half * value() const { return value_; }
    __half * gated() const { return gated_; }
    __half * normed() const { return normed_; }
    __half * conv_out() const { return conv_out_; }
    float * gate() const { return gate_; }

private:
    static void alloc(__half ** p, size_t n) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE scratch failed");
    }
    static void alloc(float ** p, size_t n) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE gate failed");
    }
    void reset() {
        if (key_) cudaFree(key_);
        if (key_norm_) cudaFree(key_norm_);
        if (query_norm_) cudaFree(query_norm_);
        if (value_) cudaFree(value_);
        if (gated_) cudaFree(gated_);
        if (normed_) cudaFree(normed_);
        if (conv_out_) cudaFree(conv_out_);
        if (gate_) cudaFree(gate_);
        key_ = key_norm_ = query_norm_ = value_ = gated_ = normed_ = conv_out_ = nullptr;
        gate_ = nullptr;
    }

    __half * key_ = nullptr;
    __half * key_norm_ = nullptr;
    __half * query_norm_ = nullptr;
    __half * value_ = nullptr;
    __half * gated_ = nullptr;
    __half * normed_ = nullptr;
    __half * conv_out_ = nullptr;
    float * gate_ = nullptr;
};

__global__ void q38_ple_gate_kernel(
        const __half * __restrict__ key,
        const __half * __restrict__ query,
        float * __restrict__ gate,
        int hidden,
        int hc) {
    const int c = blockIdx.x;
    if (c >= hc) return;
    __shared__ float smem[256];
    float dot = 0.0f;
    const int base = c * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        dot = fmaf(__half2float(key[base + i]), __half2float(query[base + i]), dot);
    }
    smem[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float s = smem[0] * rsqrtf(static_cast<float>(hidden));
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float signed_mag = copysignf(mag, s);
        gate[c] = 1.0f / (1.0f + expf(-signed_mag));
    }
}

__global__ void q38_ple_gated_value_kernel(
        const __half * __restrict__ value,
        const float * __restrict__ gate,
        __half * __restrict__ out,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    const int c = i / hidden;
    const int h = i - c * hidden;
    out[i] = __float2half(__half2float(value[h]) * gate[c]);
}

__global__ void q38_ple_conv_step_kernel(
        const __half * __restrict__ current,
        __half * __restrict__ history,
        const __half * __restrict__ kernel,
        __half * __restrict__ out,
        int channels,
        int kernel_size,
        int dilation,
        int hist,
        int pos) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        const int back = (kernel_size - 1 - k) * dilation;
        float x = 0.0f;
        if (back == 0) {
            x = __half2float(current[ch]);
        } else if (pos >= back) {
            const int slot = (pos - back) % hist;
            x = __half2float(history[static_cast<size_t>(slot) * channels + ch]);
        }
        sum = fmaf(x, __half2float(kernel[static_cast<size_t>(ch) * kernel_size + k]), sum);
    }

    out[ch] = __float2half(sum / (1.0f + expf(-sum)));
    history[static_cast<size_t>(pos % hist) * channels + ch] = current[ch];
}

__global__ void q38_ple_add_kernel(
        __half * __restrict__ residual,
        const __half * __restrict__ gated,
        const __half * __restrict__ conv,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] = __float2half(
            __half2float(residual[i]) +
            __half2float(gated[i]) +
            __half2float(conv[i]));
    }
}

inline void run_qwen38_ple(
        __half * residual,
        int token,
        const std::vector<int> & token_history,
        int pos,
        const Qwen38PleWeights & w,
        Qwen38PleTable & table,
        Qwen38PleState & state,
        Qwen38PleScratch & scratch,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    const auto rows = qwen38_ple_row_indices(q, token, token_history);
    __half * emb = table.gather(rows, stream);

    launch_matrix_gemv_half(w.key, emb, scratch.key(), stream);
    launch_matrix_gemv_half(w.value, emb, scratch.value(), stream);

    q38_hc_group_rmsnorm_kernel<<<q.hc_count, 256, 0, stream>>>(
        scratch.key(), w.norm_key.data, scratch.key_norm(),
        q.hidden, q.hc_count, q.rms_eps);
    q38_hc_group_rmsnorm_kernel<<<q.hc_count, 256, 0, stream>>>(
        residual, w.norm_query.data, scratch.query_norm(),
        q.hidden, q.hc_count, q.rms_eps);

    q38_ple_gate_kernel<<<q.hc_count, 256, 0, stream>>>(
        scratch.key_norm(), scratch.query_norm(), scratch.gate(), q.hidden, q.hc_count);

    const int hc_dim = q.hidden * q.hc_count;
    const int threads = 256;
    q38_ple_gated_value_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.value(), scratch.gate(), scratch.gated(), q.hidden, q.hc_count);

    q38_hc_group_rmsnorm_kernel<<<q.hc_count, 256, 0, stream>>>(
        scratch.gated(), w.norm_conv.data, scratch.normed(),
        q.hidden, q.hc_count, q.rms_eps);

    q38_ple_conv_step_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.normed(), state.history(), w.conv.data, scratch.conv_out(),
        hc_dim, q.ple_conv_kernel, q.ple_ngram_size, state.hist(), pos);

    q38_ple_add_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        residual, scratch.gated(), scratch.conv_out(), hc_dim);
}

} // namespace wd
