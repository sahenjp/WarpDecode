#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "ple_reader.hpp"
#include "qwen38_exact_math.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

// F32 mirror of a GGUF tensor that llama.cpp keeps in F32 (the three PLE norm
// weights and ple_conv1d). The legacy path rounds them to F16 through
// upload_half_array(); the ggml-exact path must not. DeviceFloatArray lives in
// half_tensor.cuh; F16/BF16 sources are widened exactly.
inline DeviceFloatArray q38_ple_f32_mirror(const TensorInfo & t) {
    const size_t n = static_cast<size_t>(t.elements());
    std::vector<float> host(n);

    if (t.type == GgmlType::F32) {
        std::memcpy(host.data(), t.data, n * sizeof(float));
    } else if (t.type == GgmlType::F16) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < n; ++i) {
            __half h;
            std::memcpy(&h, &src[i], sizeof(h));
            host[i] = __half2float(h);
        }
    } else if (t.type == GgmlType::BF16) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < n; ++i) host[i] = bf16_to_float(src[i]);
    } else {
        throw std::runtime_error("PLE tensor cannot be uploaded as F32: " + t.name);
    }

    DeviceFloatArray out;
    out.size = n;
    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), n * sizeof(float));
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc PLE F32 mirror failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy PLE F32 mirror failed: ") + cudaGetErrorString(c));
    }
    return out;
}

struct Qwen38PleWeights {
    GpuMatrix key;
    GpuMatrix value;
    DeviceHalfArray norm_key;
    DeviceHalfArray norm_query;
    DeviceHalfArray norm_conv;
    DeviceHalfArray conv;

    // ggml-exact path: the F32 tensors above are consumed in F32 (llama.cpp
    // keeps them F32) and the conv geometry is validated against the layout.
    DeviceFloatArray norm_key_f32;
    DeviceFloatArray norm_query_f32;
    DeviceFloatArray norm_conv_f32;
    DeviceFloatArray conv_f32;
    int conv_rows = 0;
    int conv_cols = 0;
};

// ggml-compatible PLE projections (WARODECODE_PLE_Q8_0, default ON).
//
// llama.cpp's CPU backend never multiplies a Q8_0 weight by an F16 activation:
// the vec_dot_type of Q8_0 is Q8_0, so the 2560-element PLE embedding row is
// first quantised into 32-element Q8_0 blocks (RN scale d = amax/127, q =
// RN(value * 127/amax) saturated) and the product is then an integer dot with
// the two per-block scales applied in F32. The legacy warp path rounds the
// embedding and the projections to F16 instead, which drifts ~1% per
// projection. This path reuses the byte-verified kernels from qwen38_hc.cuh to
// reproduce the CPU order exactly, keeps the whole PLE chain in F32
// (rms_norm * gamma, the gate, the gated value, the dilated conv) and only
// falls back to half for the legacy scratch/trace slots.
// WARODECODE_PLE_Q8_0=0 restores the legacy path.
inline bool q38_ple_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_ple_q8_0_enabled() {
    static const bool enabled = q38_ple_env_flag("WARODECODE_PLE_Q8_0", true);
    return enabled;
}

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
    w.norm_key_f32 = q38_ple_f32_mirror(nk);
    w.norm_query_f32 = q38_ple_f32_mirror(nq);
    w.norm_conv_f32 = q38_ple_f32_mirror(nc);
    w.conv_f32 = q38_ple_f32_mirror(conv);
    w.conv_rows = q.ple_conv_kernel;
    w.conv_cols = hc_dim;
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

// ggml dequantize_row_iq4_nl keeps the F32 product d * kvalues_iq4nl[q]; the
// legacy kernel rounds it to half before any projection reads it.
__global__ void q38_iq4_nl_rows_to_f32_kernel(
        const uint8_t * __restrict__ raw,
        float * __restrict__ out,
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
    out[i] = __fmul_rn(d, iq4_nl_value(q));
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
        if (d_emb_f32_) cudaFree(d_emb_f32_);
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
        e = cudaMalloc(
            reinterpret_cast<void **>(&d_emb_f32_),
            static_cast<size_t>(head_dim_) * n_heads_ * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE F32 embedding failed");
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

    // ggml_get_rows returns F32 for a quantised source, so the exact path
    // quantises this F32 row directly instead of the legacy half mirror.
    const float * gather_f32(
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
        q38_iq4_nl_rows_to_f32_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            d_raw_, d_emb_f32_, n_heads_, head_dim_, static_cast<int>(row_bytes_));
        return d_emb_f32_;
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
    float * d_emb_f32_ = nullptr;
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
    ~Qwen38PleState() {
        if (history_) cudaFree(history_);
        if (history_f32_) cudaFree(history_f32_);
    }

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
        // llama.cpp keeps the PLE conv state in an F32 recurrent row, so the
        // exact path must not round the history to half every step. The half
        // mirror is still updated for the existing state trace.
        const cudaError_t f = cudaMalloc(reinterpret_cast<void **>(&history_f32_), n * sizeof(float));
        if (f != cudaSuccess) throw std::runtime_error("cudaMalloc PLE F32 history failed");
        cudaMemset(history_f32_, 0, n * sizeof(float));
    }

    void clear(cudaStream_t stream) {
        if (history_) cudaMemsetAsync(
            history_, 0, static_cast<size_t>(hist_) * channels_ * sizeof(__half), stream);
        if (history_f32_) cudaMemsetAsync(
            history_f32_, 0, static_cast<size_t>(hist_) * channels_ * sizeof(float), stream);
    }

    __half * history() const { return history_; }
    float * history_f32() const { return history_f32_; }
    int hist() const { return hist_; }
    int channels() const { return channels_; }

private:
    __half * history_ = nullptr;
    float * history_f32_ = nullptr;
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

        // ggml-exact path. hidden_ is the number of activation elements that
        // every Q8_0 quantisation in this file covers.
        hidden_ = q.hidden;
        hc_ = q.hc_count;
        hc_dim_ = hc_dim;
        alloc(&key_f32_, hc_dim);
        alloc(&key_norm_f32_, hc_dim);
        alloc(&query_norm_f32_, hc_dim);
        alloc(&query_f32_, hc_dim);
        alloc(&value_f32_, q.hidden);
        alloc(&gated_f32_, hc_dim);
        alloc(&normed_f32_, hc_dim);
        alloc(&conv_f32_, hc_dim);
        alloc(&act_q8_, static_cast<size_t>((q.hidden + 31) / 32) * 32);
        alloc(&act_scales_, static_cast<size_t>((q.hidden + 31) / 32));
    }

    __half * key() const { return key_; }
    __half * key_norm() const { return key_norm_; }
    __half * query_norm() const { return query_norm_; }
    __half * value() const { return value_; }
    __half * gated() const { return gated_; }
    __half * normed() const { return normed_; }
    __half * conv_out() const { return conv_out_; }
    float * gate() const { return gate_; }

    float * key_f32() const { return key_f32_; }
    float * key_norm_f32() const { return key_norm_f32_; }
    float * query_norm_f32() const { return query_norm_f32_; }
    float * query_f32() const { return query_f32_; }
    float * value_f32() const { return value_f32_; }
    float * gated_f32() const { return gated_f32_; }
    float * normed_f32() const { return normed_f32_; }
    float * conv_f32() const { return conv_f32_; }
    int8_t * act_q8() const { return act_q8_; }
    __half * act_scales() const { return act_scales_; }
    int act_elements() const { return hidden_; }
    int hidden() const { return hidden_; }
    int hc() const { return hc_; }
    int hc_dim() const { return hc_dim_; }

private:
    static void alloc(__half ** p, size_t n) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE scratch failed");
    }
    static void alloc(float ** p, size_t n) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE scratch failed");
    }
    static void alloc(int8_t ** p, size_t n) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(int8_t));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc PLE quantised scratch failed");
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
        if (key_f32_) cudaFree(key_f32_);
        if (key_norm_f32_) cudaFree(key_norm_f32_);
        if (query_norm_f32_) cudaFree(query_norm_f32_);
        if (query_f32_) cudaFree(query_f32_);
        if (value_f32_) cudaFree(value_f32_);
        if (gated_f32_) cudaFree(gated_f32_);
        if (normed_f32_) cudaFree(normed_f32_);
        if (conv_f32_) cudaFree(conv_f32_);
        if (act_q8_) cudaFree(act_q8_);
        if (act_scales_) cudaFree(act_scales_);
        key_ = key_norm_ = query_norm_ = value_ = gated_ = normed_ = conv_out_ = nullptr;
        gate_ = nullptr;
        key_f32_ = key_norm_f32_ = query_norm_f32_ = query_f32_ = nullptr;
        value_f32_ = gated_f32_ = normed_f32_ = conv_f32_ = nullptr;
        act_q8_ = nullptr;
        act_scales_ = nullptr;
    }

    __half * key_ = nullptr;
    __half * key_norm_ = nullptr;
    __half * query_norm_ = nullptr;
    __half * value_ = nullptr;
    __half * gated_ = nullptr;
    __half * normed_ = nullptr;
    __half * conv_out_ = nullptr;
    float * gate_ = nullptr;

    float * key_f32_ = nullptr;
    float * key_norm_f32_ = nullptr;
    float * query_norm_f32_ = nullptr;
    float * query_f32_ = nullptr;
    float * value_f32_ = nullptr;
    float * gated_f32_ = nullptr;
    float * normed_f32_ = nullptr;
    float * conv_f32_ = nullptr;
    int8_t * act_q8_ = nullptr;
    __half * act_scales_ = nullptr;
    int hidden_ = 0;
    int hc_ = 0;
    int hc_dim_ = 0;
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
        // ggml_sigmoid -> op_sigmoid (ggml-cpu/unary-ops.cpp:31): the scalar
        // glibc expf of the CPU build, not the device __expf.
        gate[c] = q38_exact_sigmoid_f32(signed_mag, q38_exact_exp_on());
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

    // ggml_silu -> ggml_vec_silu_f32 (ggml-cpu/vec.cpp:380) -> ggml_v_silu.
    out[ch] = __float2half(q38_exact_silu_f32(sum, q38_exact_exp_on()));
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

__global__ void q38_ple_add_f32_residual_kernel(
        float * __restrict__ residual,
        const __half * __restrict__ gated,
        const __half * __restrict__ conv,
        int n,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = residual[i] + __half2float(gated[i]) + __half2float(conv[i]);
        residual[i] = round_half ? __half2float(__float2half(v)) : v;
    }
}

// ---------------------------------------------------------------------------
// ggml-exact PLE chain (WARODECODE_PLE_Q8_0, default ON).
//
// llama.cpp's CPU reference (src/models/qwen4exp.cpp, build_ple) computes:
//   key   = mul_mat(ple_key,   emb)       Q8_0 weight, F32 emb, F32 out
//   value = mul_mat(ple_value, emb)       Q8_0 weight, F32 emb, F32 out
//   kn    = rms_norm(key,    eps) * ple_norm_key     (fused F32 mul)
//   qn    = rms_norm(hidden, eps) * ple_norm_query   (fused F32 mul)
//   s     = sum_rows(kn * qn) * (1/sqrt(n_embd))
//   gate  = sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))
//   gated = value * gate
//   norm  = rms_norm(gated, eps) * ple_norm_conv     (fused F32 mul)
//   conv  = silu(sum_k shifted_k * conv1d_k)         (dilation = ngram size)
//   out   = hidden + (gated + conv)
// Both matmuls use the Q8_0 vec_dot: the embedding row is quantised to Q8_0
// before the integer dot, and every norm / conv / gate step stays in F32.
// ---------------------------------------------------------------------------

// llama.cpp fuses GGML_OP_RMS_NORM with the following GGML_OP_MUL on the CPU
// (ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>:
//  y[i] = x[i] * scale * w[i]) and sums with ggml_vec_sum_f32, which accumulates
// in double (typedef double ggml_float) and only rounds the mean to F32.
__global__ void q38_ple_rmsnorm_mul_exact_kernel(
        const float * __restrict__ x,
        const float * __restrict__ gamma,
        float * __restrict__ out,
        int hidden,
        int hc,
        float eps) {
    const int c = blockIdx.x;
    if (c >= hc) return;
    const int base = c * hidden;

    __shared__ float scale_smem;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const float v = x[base + i];
            sum += static_cast<double>(__fmul_rn(v, v));
        }
        const float mean = static_cast<float>(sum / static_cast<double>(hidden));
        scale_smem = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();

    const float scale = scale_smem;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const int idx = base + i;
        out[idx] = __fmul_rn(__fmul_rn(x[idx], scale), gamma[idx]);
    }
}

// The gate mirrors the reference graph: the sum_rows product is rounded to F32
// first (ggml_mul), summed with the double accumulator of ggml_vec_sum_f32 and
// scaled by the same host-side factor ggml_scale receives. The unary ops follow
// ops.cpp/unary-ops.cpp: abs, clamp(..., 1e-6, inf), sqrt, sgn, mul, sigmoid,
// and the sigmoid uses the glibc expf port through q38_exact_sigmoid_f32().
__global__ void q38_ple_gate_exact_kernel(
        const float * __restrict__ key,
        const float * __restrict__ query,
        float * __restrict__ gate,
        int hidden,
        int hc,
        float scale) {
    const int c = blockIdx.x;
    if (c >= hc || threadIdx.x != 0) return;
    const int base = c * hidden;

    double sum = 0.0;
    for (int i = 0; i < hidden; ++i) {
        sum += static_cast<double>(__fmul_rn(key[base + i], query[base + i]));
    }
    const float s = __fmul_rn(static_cast<float>(sum), scale);
    const float mag = __fsqrt_rn(fmaxf(fabsf(s), 1e-6f));
    const float sign = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    gate[c] = q38_exact_sigmoid_f32(__fmul_rn(sign, mag), q38_exact_exp_on());
}

__global__ void q38_ple_gated_value_f32_kernel(
        const float * __restrict__ value,
        const float * __restrict__ gate,
        float * __restrict__ out,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    const int c = i / hidden;
    out[i] = __fmul_rn(value[i - c * hidden], gate[c]);
}

// out[c] = sum_k w[k,c] * x[c, t - (K-1-k)*dilation], accumulated in tap order
// with separate F32 mul/add like the reference's ggml_mul + ggml_add chain. The
// state holds the raw (pre-SiLU) normalised values, newest at pos % hist.
__global__ void q38_ple_conv_step_f32_kernel(
        const float * __restrict__ current,
        float * __restrict__ history,
        __half * __restrict__ history_half,
        const float * __restrict__ kernel,
        float * __restrict__ out,
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
            x = current[ch];
        } else if (pos >= back) {
            const int slot = (pos - back) % hist;
            x = history[static_cast<size_t>(slot) * channels + ch];
        }
        const float term = __fmul_rn(x, kernel[static_cast<size_t>(ch) * kernel_size + k]);
        sum = (k == 0) ? term : __fadd_rn(sum, term);
    }

    out[ch] = sum;
    const size_t slot = static_cast<size_t>(pos % hist) * channels + ch;
    history[slot] = current[ch];
    // The half ring only feeds the existing state trace in the exact path.
    if (history_half != nullptr) history_half[slot] = __float2half(current[ch]);
}

// out = hidden + (gated + conv), the same association ggml_add uses
// (ggml_add(ctx, hidden, ggml_add(ctx, gated, conv_out))).
__global__ void q38_ple_add_exact_kernel(
        float * __restrict__ residual,
        const float * __restrict__ gated,
        const float * __restrict__ conv,
        int n,
        bool round_half) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float branch = __fadd_rn(gated[i], conv[i]);
    const float v = __fadd_rn(residual[i], branch);
    residual[i] = round_half ? __half2float(__float2half(v)) : v;
}

__global__ void q38_ple_add_half_exact_kernel(
        __half * __restrict__ residual,
        const float * __restrict__ gated,
        const float * __restrict__ conv,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float branch = __fadd_rn(gated[i], conv[i]);
    residual[i] = __float2half(__fadd_rn(__half2float(residual[i]), branch));
}

// Every condition that must hold before the exact chain may run; anything that
// fails here sends run_qwen38_ple() back to the legacy F16 path.
inline bool q38_ple_q8_0_compatible(const Qwen38PleWeights & w, const Qwen38Layout & q) {
    if (q.hidden <= 0 || q.hc_count <= 0) return false;
    const int hc_dim = q.hidden * q.hc_count;
    // The quantiser works on whole 32-element blocks and the exact GEMV derives
    // blocks_per_row from cols >> 5, so every activation length must tile.
    if ((q.hidden % 32) != 0) return false;
    if (!w.key.lowbit || w.key.type != GgmlType::Q8_0 || w.key.low.data == nullptr) return false;
    if (!w.value.lowbit || w.value.type != GgmlType::Q8_0 || w.value.low.data == nullptr) return false;
    if (w.key.cols != q.hidden || w.key.rows != hc_dim) return false;
    if (w.value.cols != q.hidden || w.value.rows != q.hidden) return false;
    if ((w.key.cols % 32) != 0 || (w.value.cols % 32) != 0) return false;
    if (w.norm_key_f32.data == nullptr || w.norm_query_f32.data == nullptr ||
        w.norm_conv_f32.data == nullptr || w.conv_f32.data == nullptr) return false;
    if (w.norm_key_f32.size < static_cast<size_t>(hc_dim) ||
        w.norm_query_f32.size < static_cast<size_t>(hc_dim) ||
        w.norm_conv_f32.size < static_cast<size_t>(hc_dim)) return false;
    if (w.conv_rows != q.ple_conv_kernel || w.conv_cols != hc_dim) return false;
    if (w.conv_f32.size < static_cast<size_t>(w.conv_rows) * static_cast<size_t>(hc_dim)) return false;
    return true;
}

// Runs the whole PLE branch in F32; the caller adds the result to its residual.
inline void q38_ple_q8_0_compute(
        const float * hidden_f32,
        int token,
        const std::vector<int> & token_history,
        int pos,
        const Qwen38PleWeights & w,
        Qwen38PleTable & table,
        Qwen38PleState & state,
        Qwen38PleScratch & scratch,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    const int hidden = q.hidden;
    const int hc = q.hc_count;
    const int hc_dim = hidden * hc;
    constexpr int threads = 256;
    constexpr int warps = threads / 32;

    if (scratch.act_elements() < hidden) {
        throw std::runtime_error("PLE Q8_0 activation scratch mismatch");
    }

    const std::vector<uint64_t> rows = qwen38_ple_row_indices(q, token, token_history);
    const float * emb = table.gather_f32(rows, stream);

    // One Q8_0 copy of the embedding row serves both projections.
    const int act_blocks = hidden / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(act_blocks + warps - 1) / warps, threads, 0, stream>>>(
        emb, hidden, scratch.act_q8(), scratch.act_scales());

    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps, threads, 0, stream>>>(
        w.key.low.data, scratch.act_q8(), scratch.act_scales(), scratch.key_f32(), hc_dim, hidden);
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(hidden + warps - 1) / warps, threads, 0, stream>>>(
        w.value.low.data, scratch.act_q8(), scratch.act_scales(), scratch.value_f32(), hidden, hidden);

    q38_ple_rmsnorm_mul_exact_kernel<<<hc, threads, 0, stream>>>(
        scratch.key_f32(), w.norm_key_f32.data, scratch.key_norm_f32(), hidden, hc, q.rms_eps);
    q38_ple_rmsnorm_mul_exact_kernel<<<hc, threads, 0, stream>>>(
        hidden_f32, w.norm_query_f32.data, scratch.query_norm_f32(), hidden, hc, q.rms_eps);

    const float gate_scale = 1.0f / std::sqrt(static_cast<float>(hidden));
    q38_ple_gate_exact_kernel<<<hc, 1, 0, stream>>>(
        scratch.key_norm_f32(), scratch.query_norm_f32(), scratch.gate(), hidden, hc, gate_scale);

    q38_ple_gated_value_f32_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.value_f32(), scratch.gate(), scratch.gated_f32(), hidden, hc);

    q38_ple_rmsnorm_mul_exact_kernel<<<hc, threads, 0, stream>>>(
        scratch.gated_f32(), w.norm_conv_f32.data, scratch.normed_f32(), hidden, hc, q.rms_eps);

    q38_ple_conv_step_f32_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.normed_f32(), state.history_f32(), state.history(), w.conv_f32.data,
        scratch.conv_f32(), hc_dim, q.ple_conv_kernel, q.ple_ngram_size, state.hist(), pos);

    // ggml_silu on the CPU is ggml_vec_silu_f32, whose AVX2 branch evaluates the
    // expf approximation emulated by q38_hc_ggml_avx2_exp_f32 (inv_hc == 1).
    q38_hc_silu_scale_ggml_f32_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
        scratch.conv_f32(), hc_dim, 1.0f);

    // Keep the legacy half slot populated so the existing state trace stays
    // meaningful; the residual add below uses the F32 tensors.
    q38_hc_copy_f32_to_half_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        scratch.conv_f32(), scratch.conv_out(), hc_dim);
}

inline void run_qwen38_ple(
        float * residual,
        int token,
        const std::vector<int> & token_history,
        int pos,
        const Qwen38PleWeights & w,
        Qwen38PleTable & table,
        Qwen38PleState & state,
        Qwen38PleScratch & scratch,
        const Qwen38Layout & q,
        cudaStream_t stream,
        bool round_half = false) {
    if (q38_ple_q8_0_enabled() && q38_ple_q8_0_compatible(w, q)) {
        const int exact_dim = q.hidden * q.hc_count;
        q38_ple_q8_0_compute(
            residual, token, token_history, pos, w, table, state, scratch, q, stream);
        q38_ple_add_exact_kernel<<<(exact_dim + 255) / 256, 256, 0, stream>>>(
            residual, scratch.gated_f32(), scratch.conv_f32(), exact_dim, round_half);
        return;
    }

    const auto rows = qwen38_ple_row_indices(q, token, token_history);
    __half * emb = table.gather(rows, stream);

    launch_matrix_gemv_half(w.key, emb, scratch.key(), stream);
    launch_matrix_gemv_half(w.value, emb, scratch.value(), stream);

    q38_hc_group_rmsnorm_kernel<<<q.hc_count, 256, 0, stream>>>(
        scratch.key(), w.norm_key.data, scratch.key_norm(),
        q.hidden, q.hc_count, q.rms_eps);
    q38_hc_group_rmsnorm_f32in_kernel<<<q.hc_count, 256, 0, stream>>>(
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

    q38_ple_add_f32_residual_kernel<<<(hc_dim + threads - 1) / threads, threads, 0, stream>>>(
        residual, scratch.gated(), scratch.conv_out(), hc_dim, round_half);
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
    if (q38_ple_q8_0_enabled() && q38_ple_q8_0_compatible(w, q)) {
        // The legacy half residual is the F32 residual rounded to F16, so widen
        // it before the exact chain and keep the same round-trip semantics.
        const int exact_dim = q.hidden * q.hc_count;
        q38_hc_copy_half_to_f32_kernel<<<(exact_dim + 255) / 256, 256, 0, stream>>>(
            residual, scratch.query_f32(), exact_dim);
        q38_ple_q8_0_compute(
            scratch.query_f32(), token, token_history, pos, w, table, state, scratch, q, stream);
        q38_ple_add_half_exact_kernel<<<(exact_dim + 255) / 256, 256, 0, stream>>>(
            residual, scratch.gated_f32(), scratch.conv_f32(), exact_dim);
        return;
    }

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
