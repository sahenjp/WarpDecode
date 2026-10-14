#pragma once

#include "gguf.hpp"
#include "half_tensor.cuh"
#include "matrix.cuh"
#include "qwen38_exact_math.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"
#include "qwen38_rope.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

// ggml/llama.cpp-compatible QSA projections.
//
// The activation is block-quantised only for quantised weights: a Q8_0 weight
// uses the Q8_0 vec_dot (32-element blocks, integer dot, per-block scale), while
// a float weight (F32/F16/BF16) is a plain float multiply with no integer dot.
// Feeding F16 activations to the Q8_0 weights drifts ~1.4% per projection and
// compounds over the 12 QSA layers.
//
// The real model ships Q8_0 attn_q/attn_k/attn_v/attn_output but BF16
// indexer.q_proj/indexer.k_proj, so the two groups are dispatched separately:
// WARODECODE_QSA_Q8_0=0 restores the legacy F16 kernels for the four Q8_0 core
// projections only, and the indexer follows its own weight type (BF16 -> F32
// activation, never a block quantisation).
inline bool q38_qsa_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_qsa_q8_0_enabled() {
    static const bool enabled = q38_qsa_env_flag("WARODECODE_QSA_Q8_0", true);
    return enabled;
}

// ggml keeps every intermediate activation in F32: only a quantised weight's
// activation (Q8_0/Q8_K/BF16), the F16 KV cache and the layer output are ever
// rounded. The QSA fast path used to hold q/k/v/attn/final and the indexer keys
// as __half, so every kernel rounded its result and the drift compounded over the
// 12 QSA layers (position 10 reached ~29% at L47 with a 0.9% logit margin).
//
// WARODECODE_QSA_F32_MID (default ON) runs the whole middle of the layer in F32:
// the projections keep their F32 accumulators, RMSNorm/IMRoPE/pooling/block
// scoring/sparse attention read and write float, and the only remaining roundings
// are the ones ggml performs too - the Q8_0/BF16 activation of a quantised weight,
// the F16 KV cache store and the __half value returned to the caller.
//
// The legacy F16 buffers stay allocated and are written as mirrors (one extra
// F32 -> F16 conversion per buffer), because the debug trace (layer*.qsa.*) and
// the CPU-override replays read them through their existing half accessors.
// WARODECODE_QSA_F32_MID=0 restores the previous F16 scratch path unchanged.
inline bool q38_qsa_f32_mid_enabled() {
    static const bool enabled = q38_qsa_env_flag("WARODECODE_QSA_F32_MID", true);
    return enabled;
}

// The four QSA norm weights are F32 in the real checkpoint
// (blk.N.attn_q_norm.weight [256], blk.N.attn_k_norm.weight [256],
// blk.N.indexer.q_norm/k_norm.weight [128]) but the loader used to round them
// through upload_half_array(), i.e. __float2half. A half weight is off by up to
// 4.9e-4 relative, and the elementwise multiply turns that into the measured
// 3.4e-5 (q) / 5.0e-5 (k) error of the first QSA layer, while v - which has no
// norm weight - stayed bit-exact and RoPE was not the cause.
//
// WARODECODE_QSA_F32_NORM (default ON) keeps those weights in F32 and runs the
// norm kernels with the CPU's own arithmetic: ggml_compute_forward_rms_norm_f32
// accumulates in double and applies 1.0f/sqrtf(mean + eps), and build_norm is
// ggml_rms_norm followed by a separate ggml_mul, so the result is
// (x*scale)*w in that order. WARODECODE_QSA_F32_NORM=0 restores the half gamma
// path unchanged (it also leaves the F32 arrays unallocated).
inline bool q38_qsa_f32_norm_enabled() {
    static const bool enabled = q38_qsa_env_flag("WARODECODE_QSA_F32_NORM", true);
    return enabled;
}

// Diagnostic override for the IMRoPE frequency base of the *indexer* rope (the
// pooled block key and the indexer query). The model metadata the fast path reads
// (qwen4exp.rope.freq_base) is applied to the main attention and to the indexer
// alike, but the CPU oracle trace says the two do not share a base:
//
//   * inverting the indexer pipeline on the canonical CPU trace
//     (issue1_artifacts/llama-state-position-36-core.bin) and requiring the
//     recovered per-dim indexer.k_norm weight to be identical across every
//     complete block leaves a relative spread of 1e-5 for base = 1e7 and > 6 for
//     1e6, 1e5, 1e8 (and for every other pair split / rotary width). Reproduced on
//     7 QSA layers (3, 7, 11, 15, 23, 35, 47) of the same trace;
//   * the fitted decay is base^(-2p/64) with the neox pair split (p, p + 32) and
//     rope position = block start (block_id * ratio), i.e. exactly what
//     q38_qsa_launch_imrope_f32 reproduces for R = 64;
//   * the implied rotary width is an exact integer only for 1e7:
//     2 * ln(1e7) / 0.5037 = 64.0.
//
// Unset or 0 keeps the metadata value, so the default path is unchanged; setting
// WARODECODE_QSA_INDEX_ROPE_BASE=1e7 tests the reference without a rebuild.
inline float q38_qsa_index_rope_base(float model_base) {
    const char * v = std::getenv("WARODECODE_QSA_INDEX_ROPE_BASE");
    if (v == nullptr || v[0] == '\0') return model_base;
    const float parsed = static_cast<float>(std::atof(v));
    return parsed > 0.0f ? parsed : model_base;
}

// The layout the indexer rope calls use. The main attention keeps the model
// value; only the indexer keys/queries move to the override.
inline Qwen38Layout q38_qsa_index_rope_layout(const Qwen38Layout & q) {
    const float base = q38_qsa_index_rope_base(q.rope_freq_base);
    if (base == q.rope_freq_base) return q;
    Qwen38Layout out = q;
    out.rope_freq_base = base;
    return out;
}

// ---------------------------------------------------------------------------
// CPU-order QSA sparse attention (WARODECODE_EXACT_QSA_SOFTMAX, default ON).
//
// The pinned CPU oracle runs this attention with cp.flash_attn_type =
// LLAMA_FLASH_ATTN_TYPE_DISABLED (issue1_qwen38_local_reference.py,
// qwen38_forced_cpu_trace.cpp), so build_attn_mha takes the non-flash branch
// (llama-graph.cpp:2690) and the whole tail is a plain ggml graph:
//
//   kq  = ggml_mul_mat(k_f16, q)            -> F32 scores, no online rescale
//   kq  = ggml_soft_max_ext(kq, mask, kq_scale, 0)            (ops.cpp:5584)
//         wp    = kq * scale                 (ggml_vec_scale_f32, plain mul)
//         wp[i]+= 1.0f * mask[i]             (0 visible / -inf masked)
//         max   = MAX over the whole row     (ggml_vec_max_f32, ascending)
//         dp[i] = exp(wp[i] - max)           (ggml_vec_soft_max_f32, vec.cpp:531)
//         sum   = double accumulation of the AVX2 horizontal adds
//         sum   = 1.0 / sum                  (double division, then float)
//         dp[i]*= (float) sum                (ggml_vec_scale_f32)
//   kqv = ggml_mul_mat(v_f16, dp)           -> dp is rounded to F16 first
//   cur = kqv * ggml_sigmoid(gate)          -> separate ops, separate roundings
//
// The two properties the online kernel cannot reproduce are:
//   * the row sum is a *double* accumulator fed by one float per 8-element
//     AVX2 group, whose horizontal add is ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7));
//   * the normalisation is a reciprocal multiply, not a final division, and the
//     probabilities are then rounded to F16 by the F16 vec_dot conversion.
//
// The kernel below therefore walks the row in cache order in three passes
// (max -> exp + row sum -> reciprocal + value sum) and reuses the AVX2
// ggml_v_expf lane copy (q38_exact_avx2_exp_f32) and the glibc expf port
// (q38_exact_expf_glibc) that src/qwen38_exact_math.cuh already validates.
//
// WARODECODE_EXACT_QSA_SOFTMAX=0 restores the legacy online-rescale kernel.
// ---------------------------------------------------------------------------
inline bool q38_qsa_exact_softmax_enabled() {
    // Default ON. Earlier this kernel made the layer profile worse, but that was
    // measured while the score dot still used the unrounded F32 q; with the F16
    // rounding in place (see the qd16 sites in the sparse kernels) the whole QSA
    // attention - softmax, value sum and output projection - is bit-identical to
    // the oracle at position 1, measured on L4.
    static const bool enabled = q38_qsa_env_flag("WARODECODE_EXACT_QSA_SOFTMAX", true);
    return enabled;
}

// Visible tokens per head: indexer_top_k whole blocks plus the incomplete tail.
// The real checkpoint has indexer_top_k = 2048 and compress_ratio = 8, i.e. at
// most 2048 + 7 entries; the constant leaves room for a ratio of up to 64.
constexpr int Q38_QSA_EXACT_MAX_VISIBLE = 2112;
constexpr int Q38_QSA_EXACT_MAX_BLOCKS  = 256;

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

    // F32 copies of the four norm weights, allocated only when the checkpoint
    // stores them as F32 and WARODECODE_QSA_F32_NORM is on. The half arrays
    // above stay as the legacy fallback and as the debug-trace mirror.
    DeviceFloatArray q_norm_f32;
    DeviceFloatArray k_norm_f32;
    DeviceFloatArray index_q_norm_f32;
    DeviceFloatArray index_k_norm_f32;
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

    // Keep the exact F32 bytes of the four norm weights when the GGUF stores
    // them as F32 (upload_half_array above still provides the F16 fallback).
    // A non-F32 checkpoint keeps taking the legacy branch of every launcher.
    if (q38_qsa_f32_norm_enabled()) {
        if (qn.type == GgmlType::F32) out.q_norm_f32 = upload_float_array(qn);
        if (kn.type == GgmlType::F32) out.k_norm_f32 = upload_float_array(kn);
        if (iqn.type == GgmlType::F32) out.index_q_norm_f32 = upload_float_array(iqn);
        if (ikn.type == GgmlType::F32) out.index_k_norm_f32 = upload_float_array(ikn);
    }
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

        // F32 index keys. The CPU oracle caches the per-position raw index key and
        // the block-pooled key in F32 and only rounds them at the F16 KV cache, so
        // the two F16 buffers above stay as mirrors for the debug trace and for the
        // legacy path.
        alloc(reinterpret_cast<void **>(&index_raw_f32_), static_cast<size_t>(max_ctx_) * idx_width * sizeof(float));
        alloc(reinterpret_cast<void **>(&index_block_f32_), static_cast<size_t>(max_blocks_) * idx_width * sizeof(float));

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
        cudaMemsetAsync(index_raw_f32_, 0,
            static_cast<size_t>(max_ctx_) * index_width_bytes(), stream);
        cudaMemsetAsync(index_block_f32_, 0,
            static_cast<size_t>(max_blocks_) * index_width_bytes(), stream);
    }

    int max_ctx() const { return max_ctx_; }
    int max_blocks() const { return max_blocks_; }
    int ratio() const { return ratio_; }

    __half * k_cache() const { return k_cache_; }
    __half * v_cache() const { return v_cache_; }
    // F16 mirrors of the F32 index keys (see initialize): the legacy path and the
    // layer*.qsa.index_key_position trace read these.
    __half * index_raw() const { return index_raw_; }
    __half * index_block() const { return index_block_; }
    float * index_raw_f32() const { return index_raw_f32_; }
    float * index_block_f32() const { return index_block_f32_; }
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
    size_t index_width_bytes() const { return index_width_bytes_; }

public:
    void set_geometry_bytes(
            size_t k,
            size_t v,
            size_t raw,
            size_t block,
            size_t index_width) {
        k_cache_bytes_ = k;
        v_cache_bytes_ = v;
        index_raw_bytes_ = raw;
        index_block_bytes_ = block;
        index_width_bytes_ = index_width;
    }

private:
    void reset() {
        if (k_cache_) cudaFree(k_cache_);
        if (v_cache_) cudaFree(v_cache_);
        if (index_raw_) cudaFree(index_raw_);
        if (index_block_) cudaFree(index_block_);
        if (index_raw_f32_) cudaFree(index_raw_f32_);
        if (index_block_f32_) cudaFree(index_block_f32_);
        if (scores_) cudaFree(scores_);
        if (scores_sorted_) cudaFree(scores_sorted_);
        if (ids_) cudaFree(ids_);
        if (ids_sorted_) cudaFree(ids_sorted_);
        if (sort_temp_) cudaFree(sort_temp_);
        k_cache_ = v_cache_ = index_raw_ = index_block_ = nullptr;
        index_raw_f32_ = index_block_f32_ = nullptr;
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
    float * index_raw_f32_ = nullptr;
    float * index_block_f32_ = nullptr;

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
    size_t index_width_bytes_ = 0;
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

        // F32/Q8_0 work buffers shared by every ggml-compatible projection path.
        // act_f32/act_q8/act_scales hold the largest activation the layer ever
        // feeds to a projection (the hidden input or the attention context), and
        // gemv_f32 holds the widest projection output. The Q8_0 core projections
        // use act_q8/act_scales and the F32 indexer projections use act_f32, so
        // both are sized here and never alias the diagnostic replay buffers.
        const int exact_elements = std::max(q.hidden, q_n_);
        const int exact_rows = std::max(
            std::max(q_full_n_, kv_n_),
            std::max(std::max(q_n_, idx_q_n_), std::max(idx_k_n_, q.hidden)));

        alloc(reinterpret_cast<void **>(&q_full_), static_cast<size_t>(q_full_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&q_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gate_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&k_), static_cast<size_t>(kv_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&v_), static_cast<size_t>(kv_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&attn_), static_cast<size_t>(q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&pre_gate_trace_), static_cast<size_t>(q_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&idx_q_), static_cast<size_t>(idx_q_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&idx_k_), static_cast<size_t>(idx_k_n_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&final_), static_cast<size_t>(q.hidden) * sizeof(__half));

        // F32 working set for WARODECODE_QSA_F32_MID. Every entry below is the
        // authoritative value in the F32 path; the matching F16 buffer above stays
        // as the mirror that the debug trace and the legacy path read.
        alloc(reinterpret_cast<void **>(&q_full_f32_), static_cast<size_t>(q_full_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&q_f32_), static_cast<size_t>(q_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&gate_f32_), static_cast<size_t>(q_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&k_f32_), static_cast<size_t>(kv_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&v_f32_), static_cast<size_t>(kv_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&attn_f32_), static_cast<size_t>(q_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&idx_q_f32_), static_cast<size_t>(idx_q_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&idx_k_f32_), static_cast<size_t>(idx_k_n_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&final_f32_), static_cast<size_t>(q.hidden) * sizeof(float));

        // A non block-aligned layout keeps the Q8_0 projection path unavailable,
        // but the indexer still needs the F32 activation buffers, so the scratch
        // is allocated whenever the geometry is usable and each projection
        // reports its own availability.
        if (exact_elements > 0 && exact_rows > 0) {
            ensure_exact_q8_0_scratch(exact_elements, exact_rows);
        }
    }

    // Exact ggml-compatible scratch: separate from the diagnostic Q8_0
    // activation replay buffers above so the two paths can never alias.
    // exact_elements_/exact_rows_ describe the F32 activation and the F32 gemv
    // capacity; the Q8_0 buffers are allocated with a 32-wide block-aligned
    // capacity instead, because only the Q8_0 projections need whole blocks.
    void ensure_exact_q8_0_scratch(int elements, int rows) {
        if (elements <= 0 || rows <= 0) {
            throw std::runtime_error("QSA exact projection geometry must be positive");
        }
        if (act_f32_) {
            if (exact_elements_ < elements || exact_rows_ < rows) {
                throw std::runtime_error("QSA exact projection scratch is too small");
            }
            return;
        }
        const int q8_capacity = ((elements + 31) / 32) * 32;
        alloc(reinterpret_cast<void **>(&act_f32_), static_cast<size_t>(elements) * sizeof(float));
        alloc(reinterpret_cast<void **>(&act_q8_), static_cast<size_t>(q8_capacity) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&act_scales_), static_cast<size_t>(q8_capacity / 32) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&gemv_f32_), static_cast<size_t>(rows) * sizeof(float));
        exact_elements_ = elements;
        exact_rows_ = rows;
    }

    void ensure_q8_0_activation_scratch(int elements) {
        if (elements <= 0 || elements % 32 != 0) {
            throw std::runtime_error("QSA Q8_0 activation geometry must be block-aligned");
        }
        if (q8_0_values_) {
            if (q8_0_capacity_ < elements) throw std::runtime_error("QSA Q8_0 scratch is too small");
            return;
        }
        alloc(reinterpret_cast<void **>(&q8_0_values_), static_cast<size_t>(elements) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&q8_0_scales_), static_cast<size_t>(elements / 32) * sizeof(__half));
        q8_0_capacity_ = elements;
    }

    __half * q_full() const { return q_full_; }
    __half * q() const { return q_; }
    __half * gate() const { return gate_; }
    __half * k() const { return k_; }
    __half * v() const { return v_; }
    __half * attn() const { return attn_; }
    float * pre_gate_trace() const { return pre_gate_trace_; }
    __half * idx_q() const { return idx_q_; }
    __half * idx_k() const { return idx_k_; }
    __half * final() const { return final_; }
    float * q_full_f32() const { return q_full_f32_; }
    float * q_f32() const { return q_f32_; }
    float * gate_f32() const { return gate_f32_; }
    float * k_f32() const { return k_f32_; }
    float * v_f32() const { return v_f32_; }
    float * attn_f32() const { return attn_f32_; }
    float * idx_q_f32() const { return idx_q_f32_; }
    float * idx_k_f32() const { return idx_k_f32_; }
    float * final_f32() const { return final_f32_; }
    int8_t * q8_0_values() const { return q8_0_values_; }
    __half * q8_0_scales() const { return q8_0_scales_; }
    float * act_f32() const { return act_f32_; }
    int8_t * act_q8() const { return act_q8_; }
    __half * act_scales() const { return act_scales_; }
    float * gemv_f32() const { return gemv_f32_; }
    int exact_elements() const { return exact_elements_; }
    int exact_rows() const { return exact_rows_; }

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
        if (pre_gate_trace_) cudaFree(pre_gate_trace_);
        if (idx_q_) cudaFree(idx_q_);
        if (idx_k_) cudaFree(idx_k_);
        if (final_) cudaFree(final_);
        if (q8_0_values_) cudaFree(q8_0_values_);
        if (q8_0_scales_) cudaFree(q8_0_scales_);
        if (act_f32_) cudaFree(act_f32_);
        if (act_q8_) cudaFree(act_q8_);
        if (act_scales_) cudaFree(act_scales_);
        if (gemv_f32_) cudaFree(gemv_f32_);
        if (q_full_f32_) cudaFree(q_full_f32_);
        if (q_f32_) cudaFree(q_f32_);
        if (gate_f32_) cudaFree(gate_f32_);
        if (k_f32_) cudaFree(k_f32_);
        if (v_f32_) cudaFree(v_f32_);
        if (attn_f32_) cudaFree(attn_f32_);
        if (idx_q_f32_) cudaFree(idx_q_f32_);
        if (idx_k_f32_) cudaFree(idx_k_f32_);
        if (final_f32_) cudaFree(final_f32_);
        q_full_ = q_ = gate_ = k_ = v_ = attn_ = idx_q_ = idx_k_ = final_ = nullptr;
        q_full_f32_ = q_f32_ = gate_f32_ = k_f32_ = v_f32_ = nullptr;
        attn_f32_ = idx_q_f32_ = idx_k_f32_ = final_f32_ = nullptr;
        pre_gate_trace_ = nullptr;
        q8_0_values_ = nullptr;
        q8_0_scales_ = nullptr;
        act_f32_ = nullptr;
        act_q8_ = nullptr;
        act_scales_ = nullptr;
        gemv_f32_ = nullptr;
        q8_0_capacity_ = 0;
        exact_elements_ = 0;
        exact_rows_ = 0;
    }

    int q_full_n_ = 0;
    int q_n_ = 0;
    int kv_n_ = 0;
    int idx_q_n_ = 0;
    int idx_k_n_ = 0;
    int q8_0_capacity_ = 0;

    __half * q_full_ = nullptr;
    __half * q_ = nullptr;
    __half * gate_ = nullptr;
    __half * k_ = nullptr;
    __half * v_ = nullptr;
    __half * attn_ = nullptr;
    float * pre_gate_trace_ = nullptr;
    int8_t * q8_0_values_ = nullptr;
    __half * q8_0_scales_ = nullptr;
    float * act_f32_ = nullptr;
    int8_t * act_q8_ = nullptr;
    __half * act_scales_ = nullptr;
    float * gemv_f32_ = nullptr;
    int exact_elements_ = 0;
    int exact_rows_ = 0;
    __half * idx_q_ = nullptr;
    __half * idx_k_ = nullptr;
    __half * final_ = nullptr;

    float * q_full_f32_ = nullptr;
    float * q_f32_ = nullptr;
    float * gate_f32_ = nullptr;
    float * k_f32_ = nullptr;
    float * v_f32_ = nullptr;
    float * attn_f32_ = nullptr;
    float * idx_q_f32_ = nullptr;
    float * idx_k_f32_ = nullptr;
    float * final_f32_ = nullptr;
};

__global__ void q38_qsa_q8_0_quantize_half_kernel(
        const __half * __restrict__ input,
        int elements,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int block = blockIdx.x * warps_per_block + warp_in_block;
    if (block >= elements / 32) return;
    const int i = block * 32 + lane;
    const float value = __half2float(input[i]);
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

__global__ void q38_qsa_q8_0_q8_0_gemv_half_kernel(
        const uint8_t * __restrict__ weight,
        const int8_t * __restrict__ activation,
        const __half * __restrict__ activation_scales,
        __half * __restrict__ output,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;
    const int blocks_per_row = cols >> 5;
    const uint8_t * row_ptr = weight + static_cast<size_t>(row) * blocks_per_row * 34u;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * block = row_ptr + static_cast<size_t>(b) * 34u;
        const int8_t qw = *reinterpret_cast<const int8_t *>(block + 2 + lane);
        int dot = qw * static_cast<int>(activation[(b << 5) + lane]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            const __half wd = *reinterpret_cast<const __half *>(block);
            const float scale = __half2float(wd) * __half2float(activation_scales[b]);
            sum = __fadd_rn(sum, __fmul_rn(scale, static_cast<float>(dot)));
        }
    }
    if (lane == 0) output[row] = __float2half(sum);
}

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

// F32 twin of q38_qsa_extract_q_gate_kernel: same q/gate split out of the
// interleaved projection output, without the F16 store.
__global__ void q38_qsa_extract_q_gate_f32_kernel(
        const float * __restrict__ q_full,
        float * __restrict__ q,
        float * __restrict__ gate,
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

// F32 twin of q38_rmsnorm_heads_inplace_kernel. The reduction order is the same
// (per-head tree sum over head_dim, then rsqrt(mean + eps)) so the only change is
// that the squared and normalised values stay F32. The gamma stays the F16 copy
// of the GGUF weight: the weight type is a separate axis from the activation
// precision and the legacy paths keep reading the same DeviceHalfArray.
__global__ void q38_rmsnorm_heads_f32_kernel(
        float * x,
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
        const float v = x[base + d];
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
        x[idx] = x[idx] * inv * __half2float(gamma[d]);
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

// F32 twin of q38_qsa_pool_block_kernel: the ratio-token mean, the sum of squares
// and the normalised output all stay F32. The sequential order over the pooled
// tokens is unchanged, only the operands are no longer widened from __half.
__global__ void q38_qsa_pool_block_f32_kernel(
        const float * __restrict__ raw,
        float * __restrict__ pooled,
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
            v += raw[static_cast<size_t>(start_token + j) * dim + d];
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
        pooled[d] = v * inv * __half2float(norm[d]);
    }
}

// ---------------------------------------------------------------------------
// F32-gamma norm kernels (WARODECODE_QSA_F32_NORM, default ON).
//
// build_norm(x, w, LLM_NORM_RMS) is ggml_mul(ggml_rms_norm(x, eps), w)
// (src/models/qwen4exp.cpp:1462; used for Qcur:611/1042, Kcur:623/1052 and the
// indexer keys/queries at 869/881) and ggml_compute_forward_rms_norm_f32
// (ggml/src/ggml-cpu/ops.cpp:3924) is a scalar DOUBLE accumulation with
// 1.0f/sqrtf(mean + eps):
//
//   ggml_float sum = 0.0;
//   for (i00 ascending) sum += (ggml_float)(x[i00] * x[i00]);
//   const float mean  = sum/ne00;
//   const float scale = 1.0f/sqrtf(mean + eps);
//   y[i00] = x[i00] * scale * w[i00];        // rms_norm, then a separate mul
//
// The reduction order is observable, so it must stay serial in one thread
// rather than become a block tree, and the multiply order is (x*scale)*w.
// The kernels below replace only the gamma and the reduction; the surrounding
// pipeline, the trace mirrors and every bit-exact helper stay untouched.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q38_qsa_as_f32(float v) { return v; }
__device__ __forceinline__ float q38_qsa_as_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ void q38_qsa_store_f32(float * p, float v) { *p = v; }
__device__ __forceinline__ void q38_qsa_store_f32(__half * p, float v) { *p = __float2half(v); }

// mean = sum/dim in double, then 1.0f/sqrtf(mean + eps): __fdiv_rn/__fsqrt_rn
// are the correctly-rounded pair libm's sqrtf + float division perform.
template <typename X>
__device__ __forceinline__ float q38_qsa_rms_scale(
        const X * __restrict__ x,
        int dim,
        float eps) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        const float value = q38_qsa_as_f32(x[i]);
        sum = __dadd_rn(sum, static_cast<double>(__fmul_rn(value, value)));
    }
    const float mean = __double2float_rn(__ddiv_rn(sum, static_cast<double>(dim)));
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
}

// x and gamma are F32 on the F32 middle path; the legacy F16 path reuses the
// same template with __half operands so both share one arithmetic.
template <typename X, typename G>
__global__ void q38_rmsnorm_heads_exact_kernel(
        X * x,
        const G * __restrict__ gamma,
        int head_dim,
        int n_heads,
        float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    __shared__ float scale_smem[1];

    const int base = h * head_dim;
    if (threadIdx.x == 0) scale_smem[0] = q38_qsa_rms_scale(x + base, head_dim, eps);
    __syncthreads();

    const float scale = scale_smem[0];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int idx = base + d;
        q38_qsa_store_f32(
            x + idx,
            __fmul_rn(__fmul_rn(q38_qsa_as_f32(x[idx]), scale), q38_qsa_as_f32(gamma[d])));
    }
}

// Pooling + norm twin of q38_qsa_pool_block_f32_kernel. The CPU sums the ratio
// slices with a chain of ggml_add (ascending) and scales by 1.0f/(float) r
// (a multiply by the reciprocal, exact for the model's ratio = 8), then
// normalises with the same double accumulator as above.
template <typename X, typename G>
__global__ void q38_qsa_pool_block_exact_kernel(
        const X * __restrict__ raw,
        X * __restrict__ pooled,
        const G * __restrict__ gamma,
        int start_token,
        int ratio,
        int dim,
        float eps) {
    __shared__ float smem[256];
    __shared__ float scale_smem[1];

    const int d = threadIdx.x;
    if (d < dim) {
        float v = 0.0f;
        for (int j = 0; j < ratio; ++j) {
            v = __fadd_rn(v, q38_qsa_as_f32(raw[static_cast<size_t>(start_token + j) * dim + d]));
        }
        v = __fmul_rn(v, __fdiv_rn(1.0f, static_cast<float>(ratio)));
        smem[d] = v;
    }
    __syncthreads();

    if (threadIdx.x == 0) scale_smem[0] = q38_qsa_rms_scale(smem, dim, eps);
    __syncthreads();

    if (d < dim) {
        const float normalized = __fmul_rn(smem[d], scale_smem[0]);
        q38_qsa_store_f32(pooled + d, __fmul_rn(normalized, q38_qsa_as_f32(gamma[d])));
    }
}

// The F32 weight is used when it exists; otherwise these fall back to the
// legacy half-gamma kernels above, whose arithmetic is unchanged.
inline void q38_qsa_launch_norm_heads_f32(
        float * x,
        const float * gamma_f32,
        const __half * gamma_half,
        int head_dim,
        int n_heads,
        float eps,
        cudaStream_t stream) {
    if (gamma_f32 != nullptr) {
        q38_rmsnorm_heads_exact_kernel<float, float><<<n_heads, 256, 0, stream>>>(
            x, gamma_f32, head_dim, n_heads, eps);
    } else {
        q38_rmsnorm_heads_f32_kernel<<<n_heads, 256, 0, stream>>>(
            x, gamma_half, head_dim, n_heads, eps);
    }
}

inline void q38_qsa_launch_norm_heads_half(
        __half * x,
        const float * gamma_f32,
        const __half * gamma_half,
        int head_dim,
        int n_heads,
        float eps,
        cudaStream_t stream) {
    if (gamma_f32 != nullptr) {
        q38_rmsnorm_heads_exact_kernel<__half, float><<<n_heads, 256, 0, stream>>>(
            x, gamma_f32, head_dim, n_heads, eps);
    } else {
        q38_rmsnorm_heads_inplace_kernel<<<n_heads, 256, 0, stream>>>(
            x, gamma_half, head_dim, n_heads, eps);
    }
}

inline void q38_qsa_launch_pool_block_f32(
        const float * raw,
        float * pooled,
        const float * gamma_f32,
        const __half * gamma_half,
        int start_token,
        int ratio,
        int dim,
        float eps,
        cudaStream_t stream) {
    if (gamma_f32 != nullptr) {
        q38_qsa_pool_block_exact_kernel<float, float><<<1, 256, 0, stream>>>(
            raw, pooled, gamma_f32, start_token, ratio, dim, eps);
    } else {
        q38_qsa_pool_block_f32_kernel<<<1, 256, 0, stream>>>(
            raw, pooled, gamma_half, start_token, ratio, dim, eps);
    }
}

inline void q38_qsa_launch_pool_block_half(
        const __half * raw,
        __half * pooled,
        const float * gamma_f32,
        const __half * gamma_half,
        int start_token,
        int ratio,
        int dim,
        float eps,
        cudaStream_t stream) {
    if (gamma_f32 != nullptr) {
        q38_qsa_pool_block_exact_kernel<__half, float><<<1, 256, 0, stream>>>(
            raw, pooled, gamma_f32, start_token, ratio, dim, eps);
    } else {
        q38_qsa_pool_block_kernel<<<1, 256, 0, stream>>>(
            raw, pooled, gamma_half, start_token, ratio, dim, eps);
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

__global__ void q38_qsa_apply_gate_to_pregate_kernel(
        const float * __restrict__ pregate,
        const __half * __restrict__ gate,
        __half * __restrict__ out,
        int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    // ggml_sigmoid on the attention output (build_layer_attn,
    // qwen4exp.cpp:1092) -> op_sigmoid (ggml-cpu/unary-ops.cpp:31): the scalar
    // glibc expf of the CPU build, not the device __expf.
    const float g = q38_exact_sigmoid_f32(__half2float(gate[i]), q38_exact_exp_on());
    out[i] = __float2half(pregate[i] * g);
}

// F32 twin of q38_qsa_score_blocks_kernel: the per-head dot and the ReLU sum are
// the same formula (score = sum_h max(0, q_h . k_h)) with F32 operands, so the
// discrete top-k block choice is made on the same quantity the CPU scores.
// WARODECODE_QSA_F32_MID consumes this into state.scores().
__global__ void q38_qsa_score_blocks_f32_kernel(
        const float * __restrict__ query,
        const float * __restrict__ block_keys,
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

    const float * key = block_keys + static_cast<size_t>(b) * dim;
    for (int h = 0; h < n_heads; ++h) {
        float dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            dot = fmaf(query[h * dim + d], key[d], dot);
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
        float * __restrict__ pre_gate_trace,
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
    const bool exact_exp = q38_exact_exp_on();
    const float qd = d < head_dim ? __half2float(q[h * head_dim + d]) : 0.0f;
    float acc = 0.0f;
    float m = -3.402823466e+38F;
    float l = 0.0f;

    auto consume = [&](int token) {
        const size_t base = (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
        // The oracle multiplies F16 activations: ggml_mul_mat(k_f16, q) uses the
        // F16 vec_dot_type, so ggml_cpu_fp32_to_fp16() rounds q before the dot
        // (ggml-cpu.c). Scoring with the unrounded F32 q left the attention
        // output ~1.5e-4 off, which the Q8_0 output projection then amplified.
        const float qd16 = __half2float(__float2half_rn(qd));
        const float dot = d < head_dim ? qd16 * __half2float(k_cache[base + d]) : 0.0f;
        red[d] = dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }

        if (d == 0) {
            const float score = red[0] * scale;
            const float nm = fmaxf(m, score);
            // ggml_soft_max_ext -> ggml_compute_forward_soft_max_f32
            // (ggml-cpu/ops.cpp:5584) -> ggml_vec_soft_max_f32
            // (ggml-cpu/vec.cpp:531): exp(x - max) through the AVX2 ggml_v_expf
            // for 8*floor(n/8) elements (the scalar tail is not modelled here).
            // The oracle runs with flash attention disabled, so the plain
            // soft_max op above is what it executes - there is no online
            // rescale on the CPU side, hence no CPU counterpart for sh_alpha
            // either; it uses the same softmax exp as sh_beta.
            sh_alpha = m > -3.0e38F ? q38_exact_softmax_exp_f32(m, nm, exact_exp) : 0.0f;
            sh_beta = q38_exact_softmax_exp_f32(score, nm, exact_exp);
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
        if (pre_gate_trace) pre_gate_trace[h * head_dim + d] = attn;
        const float g = q38_exact_sigmoid_f32(
            __half2float(gate[h * head_dim + d]), exact_exp);
        out[h * head_dim + d] = __float2half(attn * g);
    }
}

// F32 twin of q38_qsa_sparse_attention_kernel. The online softmax, the value
// accumulation and the sigmoid gate all stay F32; the K/V cache is still read as
// __half because the CPU runs the same F16 KV cache (type_k/type_v = F16), so the
// cache store is the single rounding point of the QSA key/value chain.
//
// The softmax exponentials use q38_exact_softmax_exp_f32(): the pinned CPU
// oracle builds this attention with cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_
// DISABLED (issue1_qwen38_local_reference.py, qwen38_forced_cpu_trace.cpp), so
// build_attn_mha takes the non-flash branch and runs ggml_soft_max_ext ->
// ggml_compute_forward_soft_max_f32 -> ggml_vec_soft_max_f32, whose AVX2 branch
// is the ggml_v_expf polynomial - not scalar libm.
//
// pre_gate_trace keeps its meaning: the un-gated attention output (the CPU's
// attn_pregate), while out carries the gated value.
__global__ void q38_qsa_sparse_attention_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ gate,
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        const int * __restrict__ selected_blocks,
        float * __restrict__ out,
        float * __restrict__ pre_gate_trace,
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
    const bool exact_exp = q38_exact_exp_on();
    const float qd = d < head_dim ? q[h * head_dim + d] : 0.0f;
    float acc = 0.0f;
    float m = -3.402823466e+38F;
    float l = 0.0f;

    auto consume = [&](int token) {
        const size_t base = (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
        // The oracle multiplies F16 activations: ggml_mul_mat(k_f16, q) uses the
        // F16 vec_dot_type, so ggml_cpu_fp32_to_fp16() rounds q before the dot
        // (ggml-cpu.c). Scoring with the unrounded F32 q left the attention
        // output ~1.5e-4 off, which the Q8_0 output projection then amplified.
        const float qd16 = __half2float(__float2half_rn(qd));
        const float dot = d < head_dim ? qd16 * __half2float(k_cache[base + d]) : 0.0f;
        red[d] = dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }

        if (d == 0) {
            const float score = red[0] * scale;
            const float nm = fmaxf(m, score);
            // ggml_soft_max_ext -> ggml_compute_forward_soft_max_f32
            // (ggml-cpu/ops.cpp:5584) -> ggml_vec_soft_max_f32
            // (ggml-cpu/vec.cpp:531): exp(x - max) through the AVX2 ggml_v_expf
            // for 8*floor(n/8) elements (the scalar tail is not modelled here).
            // The oracle runs with flash attention disabled, so the plain
            // soft_max op above is what it executes - there is no online
            // rescale on the CPU side, hence no CPU counterpart for sh_alpha
            // either; it uses the same softmax exp as sh_beta.
            sh_alpha = m > -3.0e38F ? q38_exact_softmax_exp_f32(m, nm, exact_exp) : 0.0f;
            sh_beta = q38_exact_softmax_exp_f32(score, nm, exact_exp);
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
        if (pre_gate_trace) pre_gate_trace[h * head_dim + d] = attn;
        const float g = q38_exact_sigmoid_f32(gate[h * head_dim + d], exact_exp);
        out[h * head_dim + d] = attn * g;
    }
}

// ---------------------------------------------------------------------------
// CPU-order softmax row reduction and value sum.
//
// Both helpers are pure scalar code so the host bit check can compile them
// verbatim (the same extraction trick tests/qwen38_exact_math_host_check.sh
// uses for q38_hc_ggml_avx2_exp_f32) and compare them against the real
// ggml_vec_soft_max_f32 / ggml_vec_dot_f16 linked out of the pinned build.
// ---------------------------------------------------------------------------

// ggml_vec_soft_max_f32() (ggml-cpu/vec.cpp:531) over the visible cells of a
// sparse row, with the preceding ggml_vec_max_f32 pass already folded in by the
// caller. s_val is in/out: it holds the scaled scores on entry (ascending row
// order, listed by s_pos) and exp(x - max) on exit, which is what the value
// pass needs. The double return is exactly the accumulator
// ggml_compute_forward_soft_max_f32 feeds into "sum = 1.0/sum".
//
// The AVX2 branch walks 8 lanes at a time and folds them with
//     ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))
// adding one float per group into a double. Masked lanes are exact zeros, so a
// group without a visible cell contributes exactly 0.0 and is skipped - that is
// bit-identical to the CPU walking the whole padded row.
// The n%8 tail is the scalar expf() branch of the same function; it is
// unreachable for the padded rows the CPU builds (get_n_kv rounds the row up to
// a multiple of 256) but is kept so the sequence stays complete.
__device__ __forceinline__ double q38_qsa_exact_softmax_exp_sum_f32(
        float * s_val, const int * s_pos, int n_vis, int n_row, float max) {
    double sum = 0.0;
    int idx = 0;
    const int n_vec = n_row & ~7;
    for (int g = 0; g < n_vec; g += 8) {
        if (idx >= n_vis) break;
        if (s_pos[idx] >= g + 8) continue;
        float v[8];
        for (int lane = 0; lane < 8; ++lane) {
            const int j = g + lane;
            float e = 0.0f;
            if (idx < n_vis && s_pos[idx] == j) {
                e = q38_exact_avx2_exp_f32(__fsub_rn(s_val[idx], max));
                s_val[idx] = e;
                ++idx;
            }
            v[lane] = e;
        }
        const float t0 = __fadd_rn(v[0], v[4]);
        const float t1 = __fadd_rn(v[2], v[6]);
        const float t2 = __fadd_rn(v[1], v[5]);
        const float t3 = __fadd_rn(v[3], v[7]);
        sum += static_cast<double>(__fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3)));
    }
    for (int j = n_vec; j < n_row; ++j) {
        float e = 0.0f;
        if (idx < n_vis && s_pos[idx] == j) {
            e = q38_exact_expf_glibc(__fsub_rn(s_val[idx], max));
            s_val[idx] = e;
            ++idx;
        }
        sum += static_cast<double>(e);
    }
    return sum;
}

// ggml_vec_dot_f16() (ggml-cpu/vec.cpp:264) on its AVX2/FMA path: GGML_F16_STEP
// = 32, GGML_F16_ARR = 4 accumulators of 8 lanes, each updated with one
// _mm256_fmadd_ps (GGML_F32x8_FMA(a,b,c) is a + b*c), then GGML_F32x8_REDUCE:
//     s0 += s2, s1 += s3, s0 += s1,
//     t0[k] = s0[k] + s0[k+4], res = (t0[0]+t0[1]) + (t0[2]+t0[3])
// and the n%32 tail as sumf += (double)(x[i]*y[i]) - a plain float multiply, no
// FMA - into the same double accumulator.
//
// src1 is the F32 softmax row while the F16 V cache makes vec_dot_type F16, so
// ggml first converts the probabilities with ggml_cpu_fp32_to_fp16
// (_mm256_cvtps_ph with _MM_FROUND_TO_NEAREST_INT): the probabilities really are
// rounded to F16 before the value sum. v_row[kv*v_stride] is this feature lane's
// F16 cache value.
//
// Accumulator (j, lane) collects kv = i + 8*j + lane for ascending i, so walking
// the visible cells in ascending kv order restores the CPU's per-accumulator FMA
// order exactly.
__device__ __forceinline__ float q38_qsa_exact_f16_dot_f32(
        const __half * v_row, int v_stride,
        const float * s_exp, const int * s_pos, int n_vis, int n_row, float rcp) {
    float s[4][8];
    for (int j = 0; j < 4; ++j) {
        for (int lane = 0; lane < 8; ++lane) s[j][lane] = 0.0f;
    }

    const int n_vec = n_row & ~31;
    int t = 0;
    for (; t < n_vis; ++t) {
        const int kv = s_pos[t];
        if (kv >= n_vec) break;
        const float p = __half2float(__float2half_rn(__fmul_rn(s_exp[t], rcp)));
        const float x = __half2float(v_row[static_cast<size_t>(kv) * v_stride]);
        s[(kv >> 3) & 3][kv & 7] = __fmaf_rn(x, p, s[(kv >> 3) & 3][kv & 7]);
    }

    for (int lane = 0; lane < 8; ++lane) {
        s[0][lane] = __fadd_rn(s[0][lane], s[2][lane]);
        s[1][lane] = __fadd_rn(s[1][lane], s[3][lane]);
    }
    for (int lane = 0; lane < 8; ++lane) {
        s[0][lane] = __fadd_rn(s[0][lane], s[1][lane]);
    }
    float t0[4];
    for (int k = 0; k < 4; ++k) t0[k] = __fadd_rn(s[0][k], s[0][k + 4]);
    double sumf = static_cast<double>(
        __fadd_rn(__fadd_rn(t0[0], t0[1]), __fadd_rn(t0[2], t0[3])));

    for (; t < n_vis; ++t) {
        const int kv = s_pos[t];
        const float p = __half2float(__float2half_rn(__fmul_rn(s_exp[t], rcp)));
        const float x = __half2float(v_row[static_cast<size_t>(kv) * v_stride]);
        sumf += static_cast<double>(__fmul_rn(x, p));
    }
    return static_cast<float>(sumf);
}

// CPU-order twin of q38_qsa_sparse_attention_f32_kernel, selected by
// WARODECODE_EXACT_QSA_SOFTMAX (default ON).
//
// The score chain (dot product, scale, KV cache read) is deliberately unchanged;
// what the CPU order fixes is the softmax and the value sum:
//   pass 1  score every visible cell and keep it in *cache row order*;
//   pass 2  the CPU's single max pass, then ggml_vec_soft_max_f32's grouped
//           exp + double row sum, then sum = 1.0/sum as a double division;
//   pass 3  p = exp * (float)sum, rounded to F16 by the F16 vec_dot conversion,
//           accumulated in ggml_vec_dot_f16's four-accumulator FMA order.
// There is no final "acc / l" division - the CPU multiplies by the reciprocal.
//
// The host dispatch guarantees the visible count fits the shared arrays; the
// clamps below only keep a direct launch memory-safe.
__global__ void q38_qsa_sparse_attention_exact_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ gate,
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        const int * __restrict__ selected_blocks,
        float * __restrict__ out,
        float * __restrict__ pre_gate_trace,
        int n_selected_blocks,
        int ratio,
        int tail_start,
        int pos,
        int n_kv_cap,
        int n_heads,
        int kv_heads,
        int head_dim,
        float scale) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= n_heads) return;

    __shared__ float red[256];
    __shared__ int   s_blocks[Q38_QSA_EXACT_MAX_BLOCKS];
    __shared__ float s_val[Q38_QSA_EXACT_MAX_VISIBLE];
    __shared__ int   s_pos[Q38_QSA_EXACT_MAX_VISIBLE];
    __shared__ float sh_rcp;

    const int kvh = (h * kv_heads) / n_heads;
    const bool exact_exp = q38_exact_exp_on();
    const float qd = d < head_dim ? q[h * head_dim + d] : 0.0f;

    // The CPU walks the softmax row in cache order; the block ids arrive in the
    // top-k order of cub's descending sort, so they are sorted here. The selected
    // set is unchanged, only the reduction order is restored.
    const int nb = n_selected_blocks < Q38_QSA_EXACT_MAX_BLOCKS
        ? n_selected_blocks : Q38_QSA_EXACT_MAX_BLOCKS;
    if (d == 0) {
        for (int i = 0; i < nb; ++i) s_blocks[i] = selected_blocks[i];
        for (int i = 1; i < nb; ++i) {
            const int key = s_blocks[i];
            int j = i - 1;
            for (; j >= 0 && s_blocks[j] > key; --j) s_blocks[j + 1] = s_blocks[j];
            s_blocks[j + 1] = key;
        }
    }
    __syncthreads();

    // Pass 1: score every visible cell and keep it in row order.
    int n_vis = 0;
    auto push_token = [&](int token) {
        const size_t base = (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
        // The oracle multiplies F16 activations: ggml_mul_mat(k_f16, q) uses the
        // F16 vec_dot_type, so ggml_cpu_fp32_to_fp16() rounds q before the dot
        // (ggml-cpu.c). Scoring with the unrounded F32 q left the attention
        // output ~1.5e-4 off, which the Q8_0 output projection then amplified.
        const float qd16 = __half2float(__float2half_rn(qd));
        const float dot = d < head_dim ? qd16 * __half2float(k_cache[base + d]) : 0.0f;
        red[d] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        if (d == 0 && n_vis < Q38_QSA_EXACT_MAX_VISIBLE) {
            s_val[n_vis] = red[0] * scale;
            s_pos[n_vis] = token;
        }
        ++n_vis;
        __syncthreads();
    };

    for (int sb = 0; sb < nb; ++sb) {
        const int start = s_blocks[sb] * ratio;
        for (int j = 0; j < ratio; ++j) push_token(start + j);
    }
    for (int token = tail_start; token <= pos; ++token) push_token(token);

    const int n_use = n_vis < Q38_QSA_EXACT_MAX_VISIBLE ? n_vis : Q38_QSA_EXACT_MAX_VISIBLE;
    // The CPU softmax row is the padded KV view, not the used prefix:
    // llama_kv_cache::get_n_kv() returns
    //     min(cells.size(), max(256, GGML_PAD(used_max_p1, 256)))
    // (llama-kv-cache.cpp:1255, n_pad_cur = max(attn_n_pad = 1, 256) = 256), so a
    // 128-cell context really scores a 128-long row. The length matters twice:
    // it fixes the 8-lane grouping of the row sum *and* the 32-lane split between
    // the vector and the scalar branch of ggml_vec_dot_f16. n_kv_cap is the
    // engine's KV cache capacity, i.e. cells.size().
    const int padded = ((pos + 1 + 255) & ~255) > 256 ? ((pos + 1 + 255) & ~255) : 256;
    const int n_row = padded < n_kv_cap ? padded : n_kv_cap;

    // Pass 2: the CPU's max pass (ggml_vec_max_f32, ascending and order
    // independent for finite scores), then the grouped exp + double row sum, then
    // sum = 1.0/sum as a double division rounded to float.
    if (d == 0) {
        float mx = -INFINITY;
        for (int i = 0; i < n_use; ++i) mx = fmaxf(mx, s_val[i]);
        const double sum = q38_qsa_exact_softmax_exp_sum_f32(s_val, s_pos, n_use, n_row, mx);
        sh_rcp = static_cast<float>(1.0 / sum);
    }
    __syncthreads();

    // Pass 3: normalise with the reciprocal, round to F16, and accumulate the
    // value sum in ggml_vec_dot_f16's order.
    if (d < head_dim) {
        const float acc = q38_qsa_exact_f16_dot_f32(
            v_cache + static_cast<size_t>(kvh) * head_dim + d, kv_heads * head_dim,
            s_val, s_pos, n_use, n_row, sh_rcp);
        if (pre_gate_trace) pre_gate_trace[h * head_dim + d] = acc;
        const float g = q38_exact_sigmoid_f32(gate[h * head_dim + d], exact_exp);
        out[h * head_dim + d] = acc * g;
    }
}

// F32 twin of the IMRoPE kernel in src/qwen38_rope.cuh. That header only exposes
// the F16 variant and this task is scoped to src/qwen38_qsa.cuh, so the twin lives
// here: same axis/section selection (q38_imrope_axis), same theta_scale/theta
// formula, same neox pair split (i0 = base + pair, i1 = i0 + rotary_dim / 2), and
// the two rotated values simply stay F32. Used for the F32 q/k projections, the
// indexer query and the pooled block key.
__global__ void q38_qsa_imrope_neox_f32_kernel(
        float * x,
        int head_dim,
        int rotary_dim,
        int n_heads,
        int4 pos,
        int4 sections,
        float freq_base,
        bool exact_trig) {
    const int pair_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = rotary_dim / 2;
    const int total_pairs = n_heads * pairs_per_head;
    if (pair_global >= total_pairs) return;

    const int head = pair_global / pairs_per_head;
    const int pair = pair_global - head * pairs_per_head;
    const int axis = q38_imrope_axis(
        pair, sections.x, sections.y, sections.z, sections.w);

    const int p = axis == 0 ? pos.x :
                  axis == 1 ? pos.y :
                  axis == 2 ? pos.z : pos.w;

    // Same arithmetic as qwen38_rope.cuh (see qwen38_imrope_math.cuh):
    // theta_scale = powf(freq_base, -2/n) once, theta stepped by |pair| float
    // multiplications, glibc sinf/cosf (not --use_fast_math's MUFU sincosf) and
    // the FMA form of rotate_pairs<>, all verified bit-identical to
    // ggml_rope_multi. The F32 QSA twin was the last path missing this, which is
    // what left q/k ~3-5e-5 off the oracle while the value projection (no RoPE)
    // stayed bit-exact.
    const int base = head * head_dim;
    const int i0 = base + pair;
    const int i1 = i0 + pairs_per_head;
    const float a = x[i0];
    const float b = x[i1];
    if (exact_trig) {
        const float ts = q38imrope::theta_scale(freq_base, rotary_dim);
        float out0 = 0.0f;
        float out1 = 0.0f;
        q38imrope::rotate_pair_exact(
            static_cast<float>(p), ts, pair, a, b, &out0, &out1);
        x[i0] = out0;
        x[i1] = out1;
    } else {
        const float theta_scale = powf(freq_base, -2.0f / static_cast<float>(rotary_dim));
        const float theta = static_cast<float>(p) * powf(theta_scale, static_cast<float>(pair));
        float sn, cs;
        sincosf(theta, &sn, &cs);
        x[i0] = a * cs - b * sn;
        x[i1] = a * sn + b * cs;
    }
}

inline void q38_qsa_launch_imrope_f32(
        float * x,
        int head_dim,
        int n_heads,
        const Qwen38Position4 & pos,
        const Qwen38Layout & q,
        cudaStream_t stream) {
    if (head_dim <= 0 || (head_dim & 1) || n_heads <= 0) {
        throw std::runtime_error("invalid IMRoPE geometry");
    }
    if (q.rope_dimension_count <= 0 || q.rope_dimension_count > head_dim ||
        (q.rope_dimension_count & 1) != 0) {
        throw std::runtime_error("invalid Qwen3.8 rotary dimension count");
    }
    if (q.rope_sections.size() != 4) {
        throw std::runtime_error("Qwen3.8 IMRoPE requires 4 rope.dimension_sections");
    }

    const int4 p = make_int4(pos.t, pos.h, pos.w, pos.extra);
    const int4 s = make_int4(
        static_cast<int>(q.rope_sections[0]),
        static_cast<int>(q.rope_sections[1]),
        static_cast<int>(q.rope_sections[2]),
        static_cast<int>(q.rope_sections[3]));

    const int total_pairs = n_heads * (q.rope_dimension_count / 2);
    const int threads = 256;
    static const bool exact_trig = [] {
        const char * v = std::getenv("WARODECODE_EXACT_ROPE");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    q38_qsa_imrope_neox_f32_kernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
        x, head_dim, q.rope_dimension_count, n_heads, p, s, q.rope_freq_base, exact_trig);
}

// F32 -> F16 mirror of a buffer the F32 path keeps in float. The debug trace
// (layer*.qsa.*) and the CPU-override replays still read the legacy __half
// accessors, so every F32 buffer that they can observe is copied back with a
// single rounding. The F32 buffer stays authoritative for the computation.
inline void q38_qsa_mirror_half(
        const float * src,
        __half * dst,
        int n,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_copy_f32_to_half_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        src, dst, n);
}

// __half -> F32 widening. Exact for every F16 value, used when a projection had
// to fall back to the legacy half kernel and the result still feeds the F32 path.
inline void q38_qsa_widen_half(
        const __half * src,
        float * dst,
        int n,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_copy_half_to_f32_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        src, dst, n);
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

// Exact ggml-compatible projection path. A projection qualifies only when the
// weight is Q8_0 and its geometry is block-aligned; anything else keeps the
// legacy F16 kernels. The compatibility check is deliberately conservative so a
// layout change can never silently run the wrong kernel.
inline bool q38_qsa_exact_q8_0_matrix_ok(
        const GpuMatrix & m,
        int rows,
        int cols) {
    return m.lowbit && m.type == GgmlType::Q8_0 &&
           m.rows == rows && m.cols == cols &&
           cols > 0 && (cols % 32) == 0;
}

// A float weight (F32/F16/BF16) is a float multiply: ggml builds the integer
// vec_dot only for quantised weights, so the activation is never block-quantised
// here. upload_matrix stores every float type as a dense half row and BF16 -> F16
// keeps the 8-bit BF16 mantissa exactly, so the stored weights are the values
// the reference reads and only the activation has to be widened before the dot.
inline bool q38_qsa_dense_f32_matrix_ok(
        const GpuMatrix & m,
        int rows,
        int cols) {
    return !m.lowbit && m.dense.data != nullptr &&
           m.rows == rows && m.cols == cols && cols > 0;
}

// The four Q8_0 core projections only. The indexer is deliberately not part of
// this decision: its weight type picks its own path, so a layout whose indexer
// is not Q8_0 (the real model ships BF16 index_q/index_k) can never hold the
// core Q8_0 projections back.
inline bool q38_qsa_exact_q8_0_ready(
        const Qwen38QsaWeights & w,
        const Qwen38Layout & q,
        const Qwen38QsaScratch & scratch) {
    if (!q38_qsa_q8_0_enabled()) return false;
    if (q.hidden <= 0 || (q.hidden % 32) != 0) return false;

    const int q_rows = q.heads * q.key_length * 2;
    const int k_rows = q.kv_heads * q.key_length;
    const int v_rows = q.kv_heads * q.value_length;
    const int out_rows = q.hidden;
    const int out_cols = q.heads * q.value_length;

    if (!q38_qsa_exact_q8_0_matrix_ok(w.q, q_rows, q.hidden)) return false;
    if (!q38_qsa_exact_q8_0_matrix_ok(w.k, k_rows, q.hidden)) return false;
    if (!q38_qsa_exact_q8_0_matrix_ok(w.v, v_rows, q.hidden)) return false;
    if (!q38_qsa_exact_q8_0_matrix_ok(w.out, out_rows, out_cols)) return false;

    const int elements = std::max(q.hidden, out_cols);
    const int rows = std::max(std::max(q_rows, k_rows), std::max(v_rows, out_rows));
    return scratch.exact_elements() >= elements && scratch.exact_rows() >= rows;
}

// Indexer dispatch. A Q8_0 indexer keeps the Q8_0 activation, exactly like the
// core path (ggml quantises for quantised weights). A float indexer - the BF16
// index_q/index_k of the real model - always uses the F32 activation path and
// never inherits WARODECODE_QSA_Q8_0.
inline bool q38_qsa_index_q8_0_ok(
        const GpuMatrix & m,
        int rows,
        int cols,
        const Qwen38QsaScratch & scratch) {
    return q38_qsa_q8_0_enabled() &&
           q38_qsa_exact_q8_0_matrix_ok(m, rows, cols) &&
           scratch.exact_elements() >= cols && scratch.exact_rows() >= rows;
}

inline bool q38_qsa_index_f32_ok(
        const GpuMatrix & m,
        int rows,
        int cols,
        const Qwen38QsaScratch & scratch) {
    return q38_qsa_dense_f32_matrix_ok(m, rows, cols) &&
           scratch.exact_elements() >= cols && scratch.exact_rows() >= rows;
}

// F16 activation -> F32 -> Q8_0 (one quantisation per input vector, reused by
// every projection fed by that input).
inline void q38_qsa_exact_q8_0_quantize_half(
        const __half * input,
        int elements,
        Qwen38QsaScratch & scratch,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    q38_hc_copy_half_to_f32_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        input, scratch.act_f32(), elements);
    const int blocks = elements / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        scratch.act_f32(), elements, scratch.act_q8(), scratch.act_scales());
}

// Integer-dot GEMV into gemv_f32(), then a rounding F32 -> F16 store so the
// remaining fast path keeps its existing half scratch layout.
inline void q38_qsa_exact_q8_0_project(
        const GpuMatrix & w,
        const Qwen38QsaScratch & scratch,
        __half * output,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_launch_exact_q8_0_gemv_f32(
        w.low.data, scratch.act_q8(), scratch.act_scales(), scratch.gemv_f32(),
        w.rows, w.cols, stream);
    q38_hc_copy_f32_to_half_kernel<<<(w.rows + threads - 1) / threads, threads, 0, stream>>>(
        scratch.gemv_f32(), output, w.rows);
}

// ggml's BF16 traits use vec_dot_type = BF16, so the CPU rounds the activation
// to BF16 (round-to-nearest-even) before the dot. Both model indexer projections
// are BF16, and the indexer picks the sparse-attention blocks (a discrete
// choice), so rounding the same way matters more here than raw precision does.
__global__ void q38_qsa_round_bf16_kernel(float * x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const uint32_t bits = __float_as_uint(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return;
    const uint32_t rounded = bits + 0x00007fffu + ((bits >> 16) & 1u);
    x[i] = __uint_as_float(rounded & 0xffff0000u);
}

inline bool q38_qsa_bf16_activation_enabled() {
    static const bool enabled = q38_qsa_env_flag("WARODECODE_QSA_INDEX_BF16", true);
    return enabled;
}

// F16 activation -> F32. The layer input is a half buffer, so this is the widest
// activation the indexer can be fed without touching the buffers the rest of the
// fast path shares; the widening itself is exact.
inline void q38_qsa_dense_f32_activation(
        const __half * input,
        int elements,
        Qwen38QsaScratch & scratch,
        cudaStream_t stream) {
    constexpr int threads = 256;
    q38_hc_copy_half_to_f32_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        input, scratch.act_f32(), elements);
    if (q38_qsa_bf16_activation_enabled()) {
        q38_qsa_round_bf16_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
            scratch.act_f32(), elements);
    }
}

// F32-input variant: the row is already staged in act_f32(), so the half -> F32
// widening has nothing to do and only the BF16 rounding ggml performs for a BF16
// weight is left. A caller's F32 values are therefore rounded exactly once, from
// their own bits, instead of once after a detour through __half.
inline void q38_qsa_dense_f32_activation_f32(
        Qwen38QsaScratch & scratch,
        int elements,
        cudaStream_t stream) {
    if (!q38_qsa_bf16_activation_enabled()) return;
    constexpr int threads = 256;
    q38_qsa_round_bf16_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        scratch.act_f32(), elements);
}

// F32 activation x dense weight -> F32 accumulation, then a single rounding
// F32 -> F16 store so the indexer keeps the half buffers its downstream kernels
// (pooling, IMRoPE, block scoring) already read.
inline void q38_qsa_dense_f32_project(
        const GpuMatrix & w,
        const Qwen38QsaScratch & scratch,
        __half * output,
        cudaStream_t stream) {
    constexpr int threads = 256;
    launch_matrix_gemv_f32_input(w, scratch.act_f32(), scratch.gemv_f32(), stream, threads);
    q38_hc_copy_f32_to_half_kernel<<<(w.rows + threads - 1) / threads, threads, 0, stream>>>(
        scratch.gemv_f32(), output, w.rows);
}

// F32 variant of the two exact projection helpers below: the activation is
// already F32, so there is no half -> F32 staging copy, and the accumulator is
// stored straight into a float destination.
inline void q38_qsa_exact_q8_0_quantize_f32(
        const float * input,
        int elements,
        Qwen38QsaScratch & scratch,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = elements / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        input, elements, scratch.act_q8(), scratch.act_scales());
}

inline void q38_qsa_exact_q8_0_project_f32(
        const GpuMatrix & w,
        const Qwen38QsaScratch & scratch,
        float * output,
        cudaStream_t stream) {
    q38_hc_launch_exact_q8_0_gemv_f32(
        w.low.data, scratch.act_q8(), scratch.act_scales(), output, w.rows, w.cols, stream);
}

inline void q38_qsa_dense_f32_project_f32(
        const GpuMatrix & w,
        const Qwen38QsaScratch & scratch,
        float * output,
        cudaStream_t stream) {
    constexpr int threads = 256;
    launch_matrix_gemv_f32_input(w, scratch.act_f32(), output, stream, threads);
}

// Projection dispatch plan for one QSA layer call. Both entry points (the F16
// row and the F32 row) plan identically - only the activation source differs -
// so the per-weight vec_dot_type rule of ggml-cpu.c is decided in exactly one
// place and the two entries can never drift apart.
struct Qwen38QsaPlan {
    bool exact_q8_0 = false;     // the four core projections use the Q8_0 dot
    bool exact_index_q = false;  // index_q uses the Q8_0 dot
    bool exact_index_k = false;  // index_k uses the Q8_0 dot
    bool index_q_f32 = false;    // index_q multiplies the F32 row directly
    bool index_k_f32 = false;    // index_k multiplies the F32 row directly
    bool f32_mid = false;        // the F32 intermediate path runs
};

inline Qwen38QsaPlan q38_qsa_plan(
        const Qwen38QsaWeights & w,
        const Qwen38QsaScratch & scratch,
        const Qwen38Layout & q,
        bool q8_0_activation_replay,
        bool v_q8_0_activation_replay) {
    Qwen38QsaPlan plan;
    // The diagnostic replay switches stay authoritative: while either one is on
    // the legacy branch runs unchanged and the F32 path stays entirely off.
    const bool replay = q8_0_activation_replay || v_q8_0_activation_replay;
    plan.exact_q8_0 = !replay && q38_qsa_exact_q8_0_ready(w, q, scratch);

    // The indexer is planned on its own weight type, never on the four Q8_0 core
    // projections: BF16 index_q/index_k always project with a F32 activation.
    // The diagnostic replay switches keep owning the Q8_0 activation buffers, so
    // a Q8_0 indexer stays on the legacy path while they are active.
    const int idx_q_rows = q.indexer_heads * q.indexer_key_length;
    const int idx_k_rows = q.indexer_key_length;
    plan.exact_index_q = !replay &&
        q38_qsa_index_q8_0_ok(w.index_q, idx_q_rows, q.hidden, scratch);
    plan.exact_index_k = !replay &&
        q38_qsa_index_q8_0_ok(w.index_k, idx_k_rows, q.hidden, scratch);
    plan.index_q_f32 = !plan.exact_index_q &&
        q38_qsa_index_f32_ok(w.index_q, idx_q_rows, q.hidden, scratch);
    plan.index_k_f32 = !plan.exact_index_k &&
        q38_qsa_index_f32_ok(w.index_k, idx_k_rows, q.hidden, scratch);

    // F32 middle path: default on, and off whenever the diagnostic replay
    // switches own the Q8_0 activation buffers or the environment disables it.
    // The dispatch is planned before the legacy branch so a F32 run never touches
    // the F16 scratch that the replay paths read back.
    plan.f32_mid = q38_qsa_f32_mid_enabled() && !replay;
    return plan;
}

// F32 middle path (WARODECODE_QSA_F32_MID). Every buffer between the projections
// and the layer output stays F32, which is what ggml does; only the Q8_0/BF16
// activation, the F16 KV cache store and the returned __half keep a rounding.
//
// The projection dispatch is the same per-type rule the legacy path uses, only
// with F32 destinations:
//   Q8_0 weight      -> quantise the F32 activation, integer dot, F32 result
//   float weight     -> F32 activation x dense weight (BF16 -> F16 is exact)
//   anything else    -> legacy __half kernel, widened back to F32 (exact)
//
// x_f32 / x_half carry the layer activation row in the two representations the
// kernels need: the unrounded F32 values, and the exact F16 row when the caller
// already had one. Exactly one of the two is set. The F32 row is authoritative -
// it is what the Q8_0 blocks, the dense dots and the F32 intermediates consume -
// and the F16 row is only ever read by a legacy __half GEMV fallback, which by
// construction needs an F16 activation.
inline __half * q38_qsa_run_f32_mid(
        const float * x_f32,
        const __half * x_half,
        const Qwen38QsaWeights & w,
        Qwen38QsaState & state,
        Qwen38QsaScratch & scratch,
        const Qwen38Layout & q,
        int pos,
        const Qwen38Position4 & position,
        cudaStream_t stream,
        float * pre_gate_trace,
        bool exact_q8_0,
        bool exact_index_q,
        bool exact_index_k,
        bool index_q_f32,
        bool index_k_f32) {
    const int ratio = state.ratio();
    const int head_dim = q.key_length;
    const int q_full_n = q.heads * head_dim * 2;
    const int q_n = q.heads * head_dim;
    const int kv_n = q.kv_heads * head_dim;
    const int idx_dim = q.indexer_key_length;
    const int idx_q_n = q.indexer_heads * idx_dim;
    const int block_budget = q.indexer_top_k / ratio;
    const int threads = 256;
    const Qwen38Layout index_rope = q38_qsa_index_rope_layout(q);

    // Task P: outer bracket of one QSA layer (see the GDN twin above).
    Q38AttnKernelScope qsa_region_scope(kQ38AttnSiteQsaRegion, stream);

    // Same guard as the legacy branch: the Q8_0 activation scratch has to cover
    // the widest activation (the hidden row) and the widest projection output.
    if (exact_q8_0) {
        scratch.ensure_exact_q8_0_scratch(
            std::max(q.hidden, w.out.cols), scratch.exact_rows());
    }

    // Per-weight projection path: 0 = exact Q8_0, 1 = dense F32, 2 = legacy half.
    auto core_path = [&](const GpuMatrix & m, int rows, int cols) {
        if (q38_qsa_q8_0_enabled() && q38_qsa_exact_q8_0_matrix_ok(m, rows, cols) &&
            scratch.exact_elements() >= cols && scratch.exact_rows() >= rows) {
            return 0;
        }
        if (q38_qsa_dense_f32_matrix_ok(m, rows, cols)) return 1;
        return 2;
    };
    const int path_q = core_path(w.q, q_full_n, q.hidden);
    const int path_k = core_path(w.k, kv_n, q.hidden);
    const int path_v = core_path(w.v, q.kv_heads * q.value_length, q.hidden);

    // The activation of the layer is staged once. ggml's vec_dot_type rule then
    // picks the representation per weight: Q8_0 quantises the F32 activation, a
    // float weight multiplies it as is. An F32 caller is copied bit-exactly (no
    // rounding anywhere); the F16 entry hands its own row, which widens exactly.
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaStage, stream);
        if (x_f32 != nullptr) {
            if (x_f32 != scratch.act_f32()) {
                const cudaError_t e = cudaMemcpyAsync(scratch.act_f32(), x_f32,
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("QSA F32 activation staging failed");
            }
        } else {
            q38_qsa_widen_half(x_half, scratch.act_f32(), q.hidden, stream);
        }
        if (path_q == 0 || path_k == 0 || path_v == 0) {
            q38_qsa_exact_q8_0_quantize_f32(scratch.act_f32(), q.hidden, scratch, stream);
        }
    }

    // The legacy __half GEMV fallbacks read an F16 activation row. An F16 caller
    // already owns that exact row; an F32 caller gets it mirrored - once, and
    // only if such a fallback is reached - into scratch.final(), which the
    // output projection overwrites at the end of the layer. Nothing on the F32
    // chain reads this row back.
    const __half * fallback_row = x_half;
    auto activation_half = [&]() -> const __half * {
        if (fallback_row == nullptr) {
            q38_qsa_mirror_half(x_f32, scratch.final(), q.hidden, stream);
            fallback_row = scratch.final();
        }
        return fallback_row;
    };

    auto project_core = [&](const GpuMatrix & m, int path, int rows, float * output, __half * staging) {
        if (path == 0) {
            q38_qsa_exact_q8_0_project_f32(m, scratch, output, stream);
        } else if (path == 1) {
            q38_qsa_dense_f32_project_f32(m, scratch, output, stream);
        } else {
            launch_matrix_gemv_half(m, activation_half(), staging, stream);
            q38_qsa_widen_half(staging, output, rows, stream);
        }
    };
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaQkv, stream);
        project_core(w.q, path_q, q_full_n, scratch.q_full_f32(), scratch.q_full());
        project_core(w.k, path_k, kv_n, scratch.k_f32(), scratch.k());
        project_core(w.v, path_v, q.kv_heads * q.value_length, scratch.v_f32(), scratch.v());
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaExtract, stream);
        q38_qsa_extract_q_gate_f32_kernel<<<(q_n + threads - 1) / threads, threads, 0, stream>>>(
            scratch.q_full_f32(), scratch.q_f32(), scratch.gate_f32(), head_dim, q.heads);
        q38_qsa_mirror_half(scratch.gate_f32(), scratch.gate(), q_n, stream);
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaNorm, stream);
        q38_qsa_launch_norm_heads_f32(
            scratch.q_f32(), w.q_norm_f32.data, w.q_norm.data,
            head_dim, q.heads, q.rms_eps, stream);
        q38_qsa_launch_norm_heads_f32(
            scratch.k_f32(), w.k_norm_f32.data, w.k_norm.data,
            head_dim, q.kv_heads, q.rms_eps, stream);
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaImrope, stream);
        q38_qsa_launch_imrope_f32(scratch.q_f32(), head_dim, q.heads, position, q, stream);
        q38_qsa_launch_imrope_f32(scratch.k_f32(), head_dim, q.kv_heads, position, q, stream);
    }

    // Mirrors are written once a value is final, so the layer*.qsa.* trace keeps
    // reporting the values the attention actually consumes.
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaMirror, stream);
        q38_qsa_mirror_half(scratch.q_f32(), scratch.q(), q_n, stream);
        q38_qsa_mirror_half(scratch.k_f32(), scratch.k(), kv_n, stream);
        q38_qsa_mirror_half(scratch.v_f32(), scratch.v(), kv_n, stream);
    }
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaStoreKv, stream);
        q38_qsa_store_kv(state, scratch, q, pos, stream);
    }

    // Both indexer projections consume the same activation row, so its exact
    // representation (Q8_0 for a quantised indexer, BF16-rounded F32 otherwise) is
    // prepared once and reused by index_k and index_q. act_f32() already holds
    // the staged row, so the Q8_0 blocks are built straight from the F32 values
    // and the BF16 rounding a BF16 weight asks for is applied to it in place.
    if (exact_index_k || exact_index_q) {
        q38_qsa_exact_q8_0_quantize_f32(scratch.act_f32(), q.hidden, scratch, stream);
    }
    if (index_k_f32 || index_q_f32) {
        q38_qsa_dense_f32_activation_f32(scratch, q.hidden, stream);
    }
    auto project_index_f32 = [&](
            const GpuMatrix & m,
            float * output,
            __half * staging,
            bool use_q8_0,
            bool use_f32) {
        if (use_q8_0) {
            q38_qsa_exact_q8_0_project_f32(m, scratch, output, stream);
        } else if (use_f32) {
            q38_qsa_dense_f32_project_f32(m, scratch, output, stream);
        } else {
            launch_matrix_gemv_half(m, activation_half(), staging, stream);
            q38_qsa_widen_half(staging, output, m.rows, stream);
        }
    };

    // Raw index keys are cached in F32 exactly as projected. Pooling + norm +
    // IMRoPE happens once when a block becomes complete.
    float * idx_k_store = state.index_raw_f32() + static_cast<size_t>(pos) * idx_dim;
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaIndexK, stream);
        project_index_f32(w.index_k, idx_k_store, scratch.idx_k(), exact_index_k, index_k_f32);
        q38_qsa_mirror_half(
            idx_k_store, state.index_raw() + static_cast<size_t>(pos) * idx_dim, idx_dim, stream);
        q38_qsa_mirror_half(idx_k_store, scratch.idx_k(), idx_dim, stream);
    }

    if ((pos + 1) % ratio == 0) {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaPool, stream);
        const int block_id = pos / ratio;
        float * pooled = state.index_block_f32() + static_cast<size_t>(block_id) * idx_dim;
        q38_qsa_launch_pool_block_f32(
            state.index_raw_f32(), pooled, w.index_k_norm_f32.data, w.index_k_norm.data,
            block_id * ratio, ratio, idx_dim, q.rms_eps, stream);
        q38_qsa_mirror_half(
            pooled, state.index_block() + static_cast<size_t>(block_id) * idx_dim, idx_dim, stream);
        q38_qsa_launch_imrope_f32(
            pooled, idx_dim, 1,
            Qwen38Position4::text(block_id * ratio),
            index_rope, stream);
    }

    const int full_blocks = (pos + 1) / ratio;
    const int tail_start = full_blocks * ratio;
    int selected_blocks = std::min(full_blocks, block_budget);
    const int * selected_ids = state.ids();

    if (full_blocks > block_budget) {
        {
            Q38AttnKernelScope scope(kQ38AttnSiteQsaIndexQ, stream);
            project_index_f32(w.index_q, scratch.idx_q_f32(), scratch.idx_q(), exact_index_q, index_q_f32);
            q38_qsa_launch_norm_heads_f32(
                scratch.idx_q_f32(), w.index_q_norm_f32.data, w.index_q_norm.data,
                idx_dim, q.indexer_heads, q.rms_eps, stream);
            q38_qsa_launch_imrope_f32(
                scratch.idx_q_f32(), idx_dim, q.indexer_heads,
                position, index_rope, stream);
            q38_qsa_mirror_half(scratch.idx_q_f32(), scratch.idx_q(), idx_q_n, stream);
        }

        {
            Q38AttnKernelScope scope(kQ38AttnSiteQsaScore, stream);
            q38_qsa_score_blocks_f32_kernel<<<full_blocks, 256, 0, stream>>>(
                scratch.idx_q_f32(), state.index_block_f32(), state.scores(),
                full_blocks, idx_dim, q.indexer_heads);
        }

        size_t sort_temp_bytes = state.sort_temp_bytes();
        cudaError_t sort_error = cudaSuccess;
        {
            Q38AttnKernelScope scope(kQ38AttnSiteQsaSort, stream);
            sort_error = cub::DeviceRadixSort::SortPairsDescending(
                state.sort_temp(), sort_temp_bytes,
                state.scores(), state.scores_sorted(),
                state.ids(), state.ids_sorted(),
                full_blocks, 0, 32, stream);
        }
        if (sort_error != cudaSuccess) {
            throw std::runtime_error(std::string("QSA block sort failed: ") + cudaGetErrorString(sort_error));
        }
        selected_ids = state.ids_sorted();
    }

    const float scale = q.attention_scale != 0.0f
        ? q.attention_scale
        : rsqrtf(static_cast<float>(head_dim));

    // CPU-order sparse attention (WARODECODE_EXACT_QSA_SOFTMAX, default ON). The
    // legacy online kernel stays reachable for every geometry the exact kernel's
    // shared arrays cannot hold (indexer_top_k + ratio > Q38_QSA_EXACT_MAX_VISIBLE).
    const int tail_tokens = pos >= tail_start ? pos - tail_start + 1 : 0;
    const int visible_tokens = selected_blocks * ratio + tail_tokens;
    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaSparse, stream);
        if (q38_qsa_exact_softmax_enabled() &&
            visible_tokens <= Q38_QSA_EXACT_MAX_VISIBLE &&
            selected_blocks <= Q38_QSA_EXACT_MAX_BLOCKS) {
            q38_qsa_sparse_attention_exact_f32_kernel<<<q.heads, 256, 0, stream>>>(
                scratch.q_f32(), scratch.gate_f32(),
                state.k_cache(), state.v_cache(),
                selected_ids, scratch.attn_f32(), pre_gate_trace,
                selected_blocks, ratio, tail_start, pos, state.max_ctx(),
                q.heads, q.kv_heads, head_dim, scale);
        } else {
            q38_qsa_sparse_attention_f32_kernel<<<q.heads, 256, 0, stream>>>(
                scratch.q_f32(), scratch.gate_f32(),
                state.k_cache(), state.v_cache(),
                selected_ids, scratch.attn_f32(), pre_gate_trace,
                selected_blocks, ratio, tail_start, pos,
                q.heads, q.kv_heads, head_dim, scale);
        }
        q38_qsa_mirror_half(scratch.attn_f32(), scratch.attn(), q_n, stream);
    }

    {
        Q38AttnKernelScope scope(kQ38AttnSiteQsaOut, stream);
        if (exact_q8_0) {
            // The CPU quantises the F32 attention output for the Q8_0 output
            // projection, so the quantiser reads the F32 buffer directly.
            {
                Q38AttnKernelScope quant_scope(kQ38AttnSiteQsaOutQuant, stream);
                q38_qsa_exact_q8_0_quantize_f32(scratch.attn_f32(), w.out.cols, scratch, stream);
            }
            q38_qsa_exact_q8_0_project_f32(w.out, scratch, scratch.final_f32(), stream);
        } else if (q38_qsa_dense_f32_matrix_ok(w.out, q.hidden, q_n)) {
            launch_matrix_gemv_f32_input(w.out, scratch.attn_f32(), scratch.final_f32(), stream, threads);
        } else {
            launch_matrix_gemv_half(w.out, scratch.attn(), scratch.final(), stream);
            q38_qsa_widen_half(scratch.final(), scratch.final_f32(), q.hidden, stream);
        }
        q38_qsa_mirror_half(scratch.final_f32(), scratch.final(), q.hidden, stream);
    }
    return scratch.final();
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
        cudaStream_t stream,
        float * pre_gate_trace = nullptr,
        bool q8_0_activation_replay = false,
        bool v_q8_0_activation_replay = false) {
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

    // The dispatch plan is shared with the F32-input entry point, so the two
    // entries can never plan a different projection path.
    const Qwen38QsaPlan plan = q38_qsa_plan(
        w, scratch, q, q8_0_activation_replay, v_q8_0_activation_replay);
    if (plan.f32_mid) {
        // The F32 implementation widens this F16 row once (exactly) and mirrors it
        // back to F16 only where a legacy __half GEMV needs it (also exactly), so
        // an F16 caller reaches every kernel with the bits it passed in.
        return q38_qsa_run_f32_mid(
            nullptr, x, w, state, scratch, q, pos, position, stream, pre_gate_trace,
            plan.exact_q8_0, plan.exact_index_q, plan.exact_index_k,
            plan.index_q_f32, plan.index_k_f32);
    }

    if (q8_0_activation_replay) {
        if (!w.q.lowbit || w.q.type != GgmlType::Q8_0 ||
            !w.k.lowbit || w.k.type != GgmlType::Q8_0 ||
            !w.v.lowbit || w.v.type != GgmlType::Q8_0 ||
            !w.out.lowbit || w.out.type != GgmlType::Q8_0) {
            throw std::runtime_error("QSA Q8_0 activation replay requires Q8_0 Q/K/V/output weights");
        }
        const int input_elements = q.hidden;
        const int input_blocks = input_elements / 32;
        constexpr int threads = 256;
        constexpr int warps = threads / 32;
        scratch.ensure_q8_0_activation_scratch(q_n);
        q38_qsa_q8_0_quantize_half_kernel<<<(input_blocks + 7) / 8, threads, 0, stream>>>(
            x, input_elements, scratch.q8_0_values(), scratch.q8_0_scales());
        q38_qsa_q8_0_q8_0_gemv_half_kernel<<<(w.q.rows + warps - 1) / warps, threads, 0, stream>>>(
            w.q.low.data, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.q_full(), w.q.rows, w.q.cols);
        q38_qsa_q8_0_q8_0_gemv_half_kernel<<<(w.k.rows + warps - 1) / warps, threads, 0, stream>>>(
            w.k.low.data, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.k(), w.k.rows, w.k.cols);
        q38_qsa_q8_0_q8_0_gemv_half_kernel<<<(w.v.rows + warps - 1) / warps, threads, 0, stream>>>(
            w.v.low.data, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.v(), w.v.rows, w.v.cols);
    } else if (plan.exact_q8_0) {
        scratch.ensure_exact_q8_0_scratch(
            std::max(q.hidden, w.out.cols), scratch.exact_rows());
        q38_qsa_exact_q8_0_quantize_half(x, q.hidden, scratch, stream);
        q38_qsa_exact_q8_0_project(w.q, scratch, scratch.q_full(), stream);
        q38_qsa_exact_q8_0_project(w.k, scratch, scratch.k(), stream);
        q38_qsa_exact_q8_0_project(w.v, scratch, scratch.v(), stream);
    } else {
        launch_matrix_gemv_half(w.q, x, scratch.q_full(), stream);
        launch_matrix_gemv_half(w.k, x, scratch.k(), stream);
        if (v_q8_0_activation_replay) {
            if (!w.v.lowbit || w.v.type != GgmlType::Q8_0) {
                throw std::runtime_error("QSA V Q8_0 activation replay requires Q8_0 V weights");
            }
            const int input_elements = q.hidden;
            const int input_blocks = input_elements / 32;
            constexpr int q8_threads = 256;
            constexpr int q8_warps = q8_threads / 32;
            scratch.ensure_q8_0_activation_scratch(input_elements);
            q38_qsa_q8_0_quantize_half_kernel<<<
                (input_blocks + q8_warps - 1) / q8_warps,
                q8_threads, 0, stream>>>(
                    x, input_elements, scratch.q8_0_values(), scratch.q8_0_scales());
            q38_qsa_q8_0_q8_0_gemv_half_kernel<<<
                (w.v.rows + q8_warps - 1) / q8_warps,
                q8_threads, 0, stream>>>(
                    w.v.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
                    scratch.v(), w.v.rows, w.v.cols);
        } else {
            launch_matrix_gemv_half(w.v, x, scratch.v(), stream);
        }
    }

    const int threads = 256;
    q38_qsa_extract_q_gate_kernel<<<(q_n + threads - 1) / threads, threads, 0, stream>>>(
        scratch.q_full(), scratch.q(), scratch.gate(), head_dim, q.heads);

    q38_qsa_launch_norm_heads_half(
        scratch.q(), w.q_norm_f32.data, w.q_norm.data,
        head_dim, q.heads, q.rms_eps, stream);
    q38_qsa_launch_norm_heads_half(
        scratch.k(), w.k_norm_f32.data, w.k_norm.data,
        head_dim, q.kv_heads, q.rms_eps, stream);

    launch_qwen38_imrope(scratch.q(), head_dim, q.heads, position, q, stream);
    launch_qwen38_imrope(scratch.k(), head_dim, q.kv_heads, position, q, stream);
    q38_qsa_store_kv(state, scratch, q, pos, stream);

    // Both indexer projections consume the same activation row, so its exact
    // representation (Q8_0 for a quantised indexer, F32 otherwise) is prepared
    // once and reused by index_k and index_q.
    if (plan.exact_index_k || plan.exact_index_q) {
        q38_qsa_exact_q8_0_quantize_half(x, q.hidden, scratch, stream);
    }
    if (plan.index_k_f32 || plan.index_q_f32) {
        q38_qsa_dense_f32_activation(x, q.hidden, scratch, stream);
    }
    auto project_index = [&](const GpuMatrix & m, __half * output, bool use_q8_0, bool use_f32) {
        if (use_q8_0) {
            q38_qsa_exact_q8_0_project(m, scratch, output, stream);
        } else if (use_f32) {
            q38_qsa_dense_f32_project(m, scratch, output, stream);
        } else {
            launch_matrix_gemv_half(m, x, output, stream);
        }
    };

    // Raw index keys are cached exactly as projected. Pooling + norm + IMRoPE
    // happens once when a block becomes complete.
    project_index(w.index_k, scratch.idx_k(), plan.exact_index_k, plan.index_k_f32);
    cudaError_t e = cudaMemcpyAsync(
        state.index_raw() + static_cast<size_t>(pos) * idx_dim,
        scratch.idx_k(), static_cast<size_t>(idx_dim) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("store QSA index K failed: ") + cudaGetErrorString(e));

    if ((pos + 1) % ratio == 0) {
        const int block_id = pos / ratio;
        __half * pooled = state.index_block() + static_cast<size_t>(block_id) * idx_dim;
        q38_qsa_launch_pool_block_half(
            state.index_raw(), pooled, w.index_k_norm_f32.data, w.index_k_norm.data,
            block_id * ratio, ratio, idx_dim, q.rms_eps, stream);
        launch_qwen38_imrope(
            pooled, idx_dim, 1,
            Qwen38Position4::text(block_id * ratio),
            q38_qsa_index_rope_layout(q), stream);
    }

    const int full_blocks = (pos + 1) / ratio;
    const int tail_start = full_blocks * ratio;
    int selected_blocks = std::min(full_blocks, block_budget);
    const int * selected_ids = state.ids();

    if (full_blocks > block_budget) {
        project_index(w.index_q, scratch.idx_q(), plan.exact_index_q, plan.index_q_f32);
        q38_qsa_launch_norm_heads_half(
            scratch.idx_q(), w.index_q_norm_f32.data, w.index_q_norm.data,
            idx_dim, q.indexer_heads, q.rms_eps, stream);
        launch_qwen38_imrope(
            scratch.idx_q(), idx_dim, q.indexer_heads,
            position, q38_qsa_index_rope_layout(q), stream);

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
        selected_ids, scratch.attn(), pre_gate_trace,
        selected_blocks, ratio, tail_start, pos,
        q.heads, q.kv_heads, head_dim, scale);

    if (q8_0_activation_replay) {
        const int output_input_blocks = q_n / 32;
        constexpr int threads = 256;
        constexpr int warps = threads / 32;
        q38_qsa_q8_0_quantize_half_kernel<<<(output_input_blocks + 7) / 8, threads, 0, stream>>>(
            scratch.attn(), q_n, scratch.q8_0_values(), scratch.q8_0_scales());
        q38_qsa_q8_0_q8_0_gemv_half_kernel<<<(w.out.rows + warps - 1) / warps, threads, 0, stream>>>(
            w.out.low.data, scratch.q8_0_values(), scratch.q8_0_scales(),
            scratch.final(), w.out.rows, w.out.cols);
    } else if (plan.exact_q8_0) {
        q38_qsa_exact_q8_0_quantize_half(scratch.attn(), w.out.cols, scratch, stream);
        q38_qsa_exact_q8_0_project(w.out, scratch, scratch.final(), stream);
    } else {
        launch_matrix_gemv_half(w.out, scratch.attn(), scratch.final(), stream);
    }
    return scratch.final();
}

// ---------------------------------------------------------------------------
// F32-input entry point (WARODECODE_QSA_F32_MID chain).
//
// Same layer as run_qwen38_qsa(), with the same arguments except that the
// hidden-state row arrives as F32: the Q/K/V and index projections, the IMRoPE
// input, the indexer block scoring and the sparse-attention input all read the
// caller's unrounded values, which is what ggml does for a layer whose
// activation is not forced through a quantiser. This is the entry an F32
// producer (the HC mix output) should reach without a half mirror in between.
//
// The F16 middle path and the diagnostic Q8_0 activation replay consume an F16
// activation row by construction - that is what those kernels are - so with
// WARODECODE_QSA_F32_MID=0 or a replay switch active they run on a single
// rounding of the F32 row, exactly the value a caller that passed the F16 mirror
// would have produced.
//
// The returned pointer is the F16 mirror of the layer output; the unrounded
// result stays in scratch.final_f32(), so this is a drop-in replacement for
// run_qwen38_qsa() at a call site that already reads final_f32().
// ---------------------------------------------------------------------------
inline __half * run_qwen38_qsa_f32_x(
        const float * x,
        const Qwen38QsaWeights & w,
        Qwen38QsaState & state,
        Qwen38QsaScratch & scratch,
        const Qwen38Layout & q,
        int il,
        int pos,
        const Qwen38Position4 & position,
        cudaStream_t stream,
        float * pre_gate_trace = nullptr,
        bool q8_0_activation_replay = false,
        bool v_q8_0_activation_replay = false) {
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

    const Qwen38QsaPlan plan = q38_qsa_plan(
        w, scratch, q, q8_0_activation_replay, v_q8_0_activation_replay);
    if (plan.f32_mid) {
        return q38_qsa_run_f32_mid(
            x, nullptr, w, state, scratch, q, pos, position, stream, pre_gate_trace,
            plan.exact_q8_0, plan.exact_index_q, plan.exact_index_k,
            plan.index_q_f32, plan.index_k_f32);
    }

    // Legacy F16 middle path / diagnostic Q8_0 activation replay: those kernels
    // read an F16 activation row, so the F32 row is rounded once into the F16
    // staging slot. scratch.final() is dead until the output projection writes it
    // at the end of the layer call below.
    __half * const x_half = scratch.final();
    q38_qsa_mirror_half(x, x_half, q.hidden, stream);
    return run_qwen38_qsa(
        x_half, w, state, scratch, q, il, pos, position, stream, pre_gate_trace,
        q8_0_activation_replay, v_q8_0_activation_replay);
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
        static_cast<size_t>(max_ctx / ratio) * idx_width * sizeof(__half),
        idx_width * sizeof(float));
}

} // namespace wd
