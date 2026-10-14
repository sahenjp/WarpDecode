#pragma once

// Qwen3.8-Flash-Next multi-token-prediction (MTP) draft block and the
// speculative draft/verify loop.
//
// The model ships one MTP block as blk.<trunk_layers> (blk.48 here; GGUF
// qwen4exp.nextn_predict_layers = 1). It is a full decoder block (QSA attention
// + MoE + hyper-connections) plus the nextn.* tensors that pair the trunk
// hidden state with the next token embedding:
//
//   h_wide   = trunk wide residual at position p-1          [hc][hidden] F32
//   e        = token embedding of the token at position p   [hidden]
//   h_norm   = rms_norm_per_stream(h_wide) * nextn.hnorm    [hc][hidden]
//   e_norm   = rms_norm(e) * nextn.enorm                    [hidden]
//   concat   = [ e_norm ; h_norm_stream ] per stream        [2*hidden]
//   res_hc   = nextn.eh_proj(concat)                        [hc][hidden]
//   res_hc   = block(res_hc) at position p            (QSA attention + MoE)
//   h_nextn  = res_hc                            (chained into the next step)
//   logits   = nextn.shared_head_head( hc_head_mix(res_hc) )
//
// Reference: llama.cpp src/models/qwen4exp.cpp graph_mtp at the pinned oracle
// commit 6fcaa16f4b360649933a54d1f91ad40ed35c0e11.
//
// The draft is only a proposal: the speculative loop feeds a drafted token to
// the trunk only when the trunk's own argmax at that position is bit-identical,
// so the emitted sequence is always the plain greedy chain and this file can
// never change parity. A numerically wrong draft only lowers the acceptance
// rate.

#include "qwen38_forward.cuh"
#include "qwen38_runtime.cuh"
#include "qwen38_state.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

// ---------------------------------------------------------------------------
// Environment switches
// ---------------------------------------------------------------------------
inline bool q38_mtp_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline int q38_mtp_env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return std::atoi(v);
}

// WARODECODE_MTP: master switch, default off.
inline bool q38_mtp_enabled() { return q38_mtp_flag("WARODECODE_MTP", false); }

// WARODECODE_MTP_DRAFT: how many tokens one draft round proposes (k = 1..8).
inline int q38_mtp_draft_len() {
    int k = q38_mtp_env_int("WARODECODE_MTP_DRAFT", 2);
    if (k < 1) k = 1;
    if (k > 8) k = 8;
    return k;
}

// WARODECODE_MTP_PROFILE: per-stage CUDA-event breakdown of one draft step
// (norms/eh_proj, QSA branch, MoE branch, head GEMV). Default off; it only adds
// event pairs around the existing launches and never changes any value.
inline bool q38_mtp_profile_enabled() { return q38_mtp_flag("WARODECODE_MTP_PROFILE", false); }

// WARODECODE_MTP_BATCH_VERIFY: batch the MTP block's routed MoE across the k
// draft positions of one round (a layer is then one union-expert pass instead
// of k per-token passes). Every value stays bit-identical to the serial path;
// only the launch count and the weight staging change. Default off.
inline bool q38_mtp_batch_verify_enabled() {
    return q38_mtp_flag("WARODECODE_MTP_BATCH_VERIFY", false);
}

// WARODECODE_MTP_EXPERT_CACHE_MIB: VRAM slot pool for the MTP block's experts.
// The MTP experts are Q8_0 (~5.2 MiB per expert), so this is deliberately much
// smaller than the trunk's pool.
inline int q38_mtp_expert_cache_mib() {
    int mib = q38_mtp_env_int("WARODECODE_MTP_EXPERT_CACHE_MIB", 1024);
    if (mib < 64) mib = 64;
    return mib;
}

// HC mix for the MTP block. The block's inject tensors are Q8_0, and
// run_qwen38_hc_mix_f32() returns nullptr for a quantised inject only after
// doing its work, so the F32 entry is only attempted when it can finish.
inline __half * q38_mtp_hc_mix(
        const float * residual,
        const Qwen38HcWeights & w,
        Qwen38HcF32DebugScratch & f32,
        Qwen38HcScratch & scratch,
        float eps,
        cudaStream_t stream) {
    if (!(w.has_inject && w.inject.lowbit)) {
        __half * out = run_qwen38_hc_mix_f32(residual, w, f32, scratch, eps, stream);
        if (out != nullptr) return out;
    }
    return run_qwen38_hc_mix(residual, w, scratch, eps, stream);
}

inline void q38_mtp_check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen3.8 MTP: ") + what + ": " + cudaGetErrorString(e));
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Wide-residual RMSNorm: one block per residual stream, gamma over the full
// [hc][hidden] width, result written into the F16 second half of each concat row.
__global__ void q38_mtp_hnorm_kernel(
        const float * __restrict__ h_wide,
        const float * __restrict__ gamma,
        __half * __restrict__ out,
        int hidden,
        int hc,
        float eps) {
    const int c = blockIdx.x;
    if (c >= hc) return;
    const float * row = h_wide + static_cast<size_t>(c) * hidden;
    __shared__ float acc[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = row[i];
        sum = fmaf(v, v, sum);
    }
    acc[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) acc[threadIdx.x] += acc[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = q38_exact_rsqrt(acc[0] / static_cast<float>(hidden) + eps);
    __half * dst = out + static_cast<size_t>(c) * 2 * hidden + hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __fmul_rn(row[i], scale);
        dst[i] = __float2half(__fmul_rn(v, gamma[static_cast<size_t>(c) * hidden + i]));
    }
}

// Token-embedding RMSNorm (no gamma) times nextn.enorm, broadcast over the hc
// streams into the F16 first half of every concat row.
__global__ void q38_mtp_enorm_kernel(
        const float * __restrict__ e_row,
        const float * __restrict__ gamma,
        __half * __restrict__ out,
        int hidden,
        int hc,
        float eps) {
    __shared__ float acc[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = e_row[i];
        sum = fmaf(v, v, sum);
    }
    acc[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) acc[threadIdx.x] += acc[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = q38_exact_rsqrt(acc[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __fmul_rn(__fmul_rn(e_row[i], scale), gamma[i]);
        const __half h = __float2half(v);
        for (int c = 0; c < hc; ++c) out[static_cast<size_t>(c) * 2 * hidden + i] = h;
    }
}

// The MTP head (blk.48.nextn.shared_head_head.weight, Q4_0 [hidden, vocab]) is
// not part of the shared low-bit GEMV set, so it gets its own GEMV: one warp per
// row with the activation row staged in shared memory. A Q4_0 block is 18 bytes:
// one fp16 scale plus 16 nibble pairs (lanes 0..15 take the low nibbles, lanes
// 16..31 the high ones).
__global__ void q38_mtp_q4_0_gemv_kernel(
        const uint8_t * __restrict__ w,
        const __half * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    extern __shared__ __half xs[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int row = blockIdx.x * nwarp + warp;
    if (row >= rows) return;
    const int blocks_per_row = cols / 32;
    const uint8_t * rp = w + static_cast<size_t>(row) * blocks_per_row * 18;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t * blk = rp + static_cast<size_t>(b) * 18;
        const __half d = *reinterpret_cast<const __half *>(blk);
        const uint8_t q = blk[2 + (lane & 15)];
        const int v = (lane < 16) ? (q & 0x0f) : (q >> 4);
        acc = fmaf(__half2float(d) * static_cast<float>(v - 8),
                   __half2float(xs[b * 32 + lane]), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, static_cast<unsigned int>(off));
    }
    if (lane == 0) y[row] = acc;
}// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
struct Qwen38MtpWeights {
    int layer = 0;
    int ratio = 0;

    Qwen38HcWeights hc_attn;
    Qwen38HcWeights hc_ffn;
    Qwen38HcWeights hc_head;
    std::unique_ptr<Qwen38QsaWeights> qsa_w;

    GpuMatrix router;
    Qwen38MoeLayerTensors routed;
    Qwen38SharedExpertWeights shared;

    DeviceFloatArray enorm;
    DeviceFloatArray hnorm;
    LowBitMatrix eh_proj;

    uint8_t * shared_head = nullptr;    // raw Q4_0 blocks, device resident
    size_t shared_head_bytes = 0;
    int shared_head_rows = 0;
    int shared_head_cols = 0;

    Qwen38MtpWeights() = default;
    Qwen38MtpWeights(const Qwen38MtpWeights &) = delete;
    Qwen38MtpWeights & operator=(const Qwen38MtpWeights &) = delete;
    ~Qwen38MtpWeights() { if (shared_head) cudaFree(shared_head); }
};

// Same geometry as load_qwen38_hc_head() but for the MTP block's
// nextn.hc_head_* tensors (which have no inject).
inline Qwen38HcWeights q38_mtp_load_hc_head(
        const GgufFile & g,
        const std::string & prefix,
        int hidden,
        int hc,
        int hc_low_rank) {
    const TensorInfo & norm = g.tensor(prefix + "norm.weight");
    const TensorInfo & down = g.tensor(prefix + "down.weight");
    const TensorInfo & up = g.tensor(prefix + "up.weight");

    const int hc_dim = hidden * hc;
    if (static_cast<int>(norm.elements()) != hc_dim) {
        throw std::runtime_error("MTP HC head norm shape mismatch: " + norm.name);
    }
    auto check2 = [](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols ||
            static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("MTP HC head ") + what + " shape mismatch: " + t.name);
        }
    };
    check2(down, hc_low_rank, hc_dim, "down");
    check2(up, hc_dim, hc_low_rank, "up");

    Qwen38HcWeights out;
    out.norm = upload_half_array(norm);
    if (norm.type == GgmlType::F32 && down.type == GgmlType::Q8_0 && up.type == GgmlType::Q8_0) {
        const float * values = reinterpret_cast<const float *>(norm.data);
        out.debug_norm_f32.assign(values, values + norm.elements());
    }
    out.down = upload_matrix(down);
    out.up = upload_matrix(up);
    out.has_inject = false;
    return out;
}

inline void load_qwen38_mtp_weights(
        const GgufFile & g,
        const Qwen38Layout & q,
        Qwen38MtpWeights & out) {
    if (q.mtp_layers <= 0 || q.trunk_layers >= q.total_blocks) {
        throw std::runtime_error("Qwen3.8 MTP: the model has no MTP block to load");
    }
    const int il = q.trunk_layers;
    const std::string p = "blk." + std::to_string(il) + ".";

    out.layer = il;

    if (q.compress_ratios.size() != static_cast<size_t>(q.total_blocks) ||
        q.compress_ratios[static_cast<size_t>(il)] <= 0) {
        throw std::runtime_error("Qwen3.8 MTP: the MTP block is expected to be a QSA layer");
    }
    out.ratio = static_cast<int>(q.compress_ratios[static_cast<size_t>(il)]);

    out.hc_attn = load_qwen38_hc_weights(g, il, "attn", q.hidden, q.hc_count, q.hc_low_rank);
    out.hc_ffn = load_qwen38_hc_weights(g, il, "ffn", q.hidden, q.hc_count, q.hc_low_rank);
    out.hc_head = q38_mtp_load_hc_head(g, p + "nextn.hc_head_", q.hidden, q.hc_count, q.hc_low_rank);
    out.qsa_w = std::make_unique<Qwen38QsaWeights>(load_qwen38_qsa_weights(g, il, q));

    out.routed = qwen38_moe_tensors(g, il);
    qwen38_check_expert_tensors(out.routed, q, il);
    if (out.routed.fused_gate_up()) {
        throw std::runtime_error("Qwen3.8 MTP: fused expert layout is not supported");
    }
    out.router = upload_matrix(*out.routed.router);
    out.shared = load_qwen38_shared_expert(g, il, q.hidden, q.shared_ffn);

    const TensorInfo & enorm = g.tensor(p + "nextn.enorm.weight");
    const TensorInfo & hnorm = g.tensor(p + "nextn.hnorm.weight");
    const TensorInfo & eh_proj = g.tensor(p + "nextn.eh_proj.weight");
    const TensorInfo & head = g.tensor(p + "nextn.shared_head_head.weight");

    if (enorm.type != GgmlType::F32 || static_cast<int>(enorm.elements()) != q.hidden) {
        throw std::runtime_error("Qwen3.8 MTP: nextn.enorm must be F32 [hidden]");
    }
    if (hnorm.type != GgmlType::F32 ||
        static_cast<int>(hnorm.elements()) != q.hidden * q.hc_count) {
        throw std::runtime_error("Qwen3.8 MTP: nextn.hnorm must be F32 [hc*hidden]");
    }
    if (eh_proj.type != GgmlType::Q8_0 || eh_proj.dims.size() != 2 ||
        static_cast<int>(eh_proj.dims[0]) != 2 * q.hidden ||
        static_cast<int>(eh_proj.dims[1]) != q.hidden) {
        throw std::runtime_error("Qwen3.8 MTP: nextn.eh_proj must be Q8_0 [hidden,2*hidden]");
    }
    if (head.type != GgmlType::Q4_0 || head.dims.size() != 2 ||
        static_cast<int>(head.dims[0]) != q.hidden ||
        (static_cast<int>(head.dims[0]) % 32) != 0) {
        throw std::runtime_error("Qwen3.8 MTP: nextn.shared_head_head must be Q4_0 [vocab,hidden]");
    }

    out.enorm = upload_float_array(enorm);
    out.hnorm = upload_float_array(hnorm);
    out.eh_proj = upload_lowbit_matrix(eh_proj);

    out.shared_head_rows = static_cast<int>(head.dims[1]);
    out.shared_head_cols = static_cast<int>(head.dims[0]);
    out.shared_head_bytes = head.byte_size();
    q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&out.shared_head), out.shared_head_bytes),
                  "cudaMalloc shared head");
    q38_mtp_check(cudaMemcpy(out.shared_head, head.data, out.shared_head_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy shared head");
}

// ---------------------------------------------------------------------------
// Draft statistics
// ---------------------------------------------------------------------------
struct Qwen38MtpStats {
    int draft_len = 0;
    long long rounds = 0;
    long long draft_steps = 0;       // MTP block forward passes
    long long draft_tokens = 0;      // proposed draft tokens
    long long accepts = 0;           // draft tokens the trunk confirmed
    long long full_rounds = 0;       // rounds where every draft was accepted
    long long verify_steps = 0;      // trunk steps == emitted tokens
    double draft_seconds = 0.0;
    double seconds = 0.0;
    // WARODECODE_MTP_PROFILE breakdown of the draft steps.
    double stage_norm_seconds = 0.0;
    double stage_attn_seconds = 0.0;
    double stage_moe_seconds = 0.0;
    double stage_head_seconds = 0.0;

    double draft_ms_per_step() const {
        return draft_steps > 0 ? 1000.0 * draft_seconds / static_cast<double>(draft_steps) : 0.0;
    }
    double accept_rate() const {
        return draft_tokens > 0 ? static_cast<double>(accepts) / static_cast<double>(draft_tokens) : 0.0;
    }
};

// ---------------------------------------------------------------------------
// MTP block runtime
// ---------------------------------------------------------------------------
class Qwen38Mtp {
public:
    Qwen38Mtp() = default;
    Qwen38Mtp(const Qwen38Mtp &) = delete;
    Qwen38Mtp & operator=(const Qwen38Mtp &) = delete;

    ~Qwen38Mtp() {
        if (moe_out_) cudaFree(moe_out_);
        if (moe_slots_) cudaFree(moe_slots_);
        if (draft_hidden_dev_) cudaFree(draft_hidden_dev_);
        if (embed_row_) cudaFree(embed_row_);
        if (concat_) cudaFree(concat_);
        if (logits_) cudaFree(logits_);
        if (d_token_) cudaFree(d_token_);
        if (d_nonfinite_) cudaFree(d_nonfinite_);
        if (h_token_) cudaFreeHost(h_token_);
        if (h_nonfinite_) cudaFreeHost(h_nonfinite_);
    }

    void initialize(const GgufFile & g, Qwen38Runtime & rt, int max_ctx) {
        if (max_ctx <= 0) throw std::runtime_error("Qwen3.8 MTP: invalid context");
        const Qwen38Layout & q = rt.layout;
        rt_ = &rt;
        max_ctx_ = max_ctx;

        load_qwen38_mtp_weights(g, q, weights_);

        hidden_ = q.hidden;
        hc_ = q.hc_count;
        hc_dim_ = hidden_ * hc_;
        low_rank_ = q.hc_low_rank;
        vocab_ = rt.vocab;
        layer_ = weights_.layer;
        ratio_ = weights_.ratio;
        draft_len_ = q38_mtp_draft_len();

        // The QSA entry points gate on "il < layout.trunk_layers" and the MTP
        // block is a QSA layer at il == trunk_layers. Widening that single field
        // of a local layout copy keeps the block on the exact trunk kernel path
        // without editing qwen38_qsa.cuh (owned by another workstream); every
        // geometry field the QSA path reads is identical to rt.layout.
        qsa_layout_ = q;
        qsa_layout_.trunk_layers = q.total_blocks;
        if (!qsa_layout_.is_qsa(layer_)) {
            throw std::runtime_error("Qwen3.8 MTP: the MTP layer is not a QSA layer");
        }

        qsa_state_ = std::make_unique<Qwen38QsaState>();
        initialize_qwen38_qsa_state(*qsa_state_, max_ctx_, ratio_, q);
        qsa_scratch_ = std::make_unique<Qwen38QsaScratch>();
        qsa_scratch_->initialize(q);

        hc_attn_.initialize(hidden_, hc_, low_rank_);
        hc_ffn_.initialize(hidden_, hc_, low_rank_);
        hc_head_.initialize(hidden_, hc_, low_rank_);
        f32_.initialize(hidden_, hc_, low_rank_);
        moe_.initialize(q.experts, q.experts_used, hidden_, q.expert_ffn);

        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&moe_out_), sizeof(__half) * hidden_),
                      "cudaMalloc MoE output");
        // One MoE output slice per draft position of a batched round: slot 0 is
        // moe_out_, slots 1..k-1 live here.
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&moe_slots_),
                      sizeof(__half) * static_cast<size_t>(draft_len_ > 1 ? draft_len_ - 1 : 1) * hidden_),
                      "cudaMalloc MoE output slices");
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&embed_row_), sizeof(float) * hidden_),
                      "cudaMalloc embedding row");
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&concat_),
            sizeof(__half) * static_cast<size_t>(hc_) * 2 * hidden_), "cudaMalloc concat");
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&logits_), sizeof(float) * vocab_),
                      "cudaMalloc logits");
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&d_token_), sizeof(int)), "cudaMalloc token");
        q38_mtp_check(cudaMalloc(reinterpret_cast<void **>(&d_nonfinite_), sizeof(int)),
                      "cudaMalloc nonfinite flag");
        q38_mtp_check(cudaMallocHost(reinterpret_cast<void **>(&h_token_), sizeof(int)),
                      "cudaMallocHost token");
        q38_mtp_check(cudaMallocHost(reinterpret_cast<void **>(&h_nonfinite_), sizeof(int)),
                      "cudaMallocHost nonfinite flag");

        // The MTP experts are Q8_0 while the trunk's are IQ2_S/IQ4_NL, and a
        // GpuExpertCache is bound to one tensor set at initialize() time, so the
        // MTP block keeps its own (much smaller) slot pool.
        expert_cache_.initialize(static_cast<size_t>(q38_mtp_expert_cache_mib()),
                                 *weights_.routed.gate, *weights_.routed.up, *weights_.routed.down);
        ready_ = true;
    }

    bool ready() const { return ready_; }
    int draft_len() const { return draft_len_; }
    // --mtp-draft / WARODECODE_MTP_DRAFT override of the per-round draft count.
    void set_draft_len(int k) {
        if (k < 1) k = 1;
        if (k > 8) k = 8;
        draft_len_ = k;
    }
    int layer() const { return layer_; }
    int ratio() const { return ratio_; }
    size_t expert_cache_slots() const { return expert_cache_.slot_count(); }
    Qwen38MtpStats & stats() { return stats_; }

    void set_stream(cudaStream_t stream) { stream_ = stream; }

    // Why a batched round would stay serial: null means it can run, otherwise
    // the reason (printed once per process by q38_mtp_note_batch_decision()).
    const char * batch_blocker() const {
        if (!ready_) return "the MTP block is not initialized";
        if (!lowbit_dp4a_available()) return "the DP4A expert GEMV kernels are unavailable";
        if (!q38_moe_expert_q8_act_geometry(hidden_, moe_.expert_ffn())) {
            return "the expert activation geometry is not a whole number of Q8 blocks";
        }
        if (!(rt_->layout.expert_weights_scale > 0.0f)) return "the expert scale is not positive";
        // The same bound the batch class enforces: one batch reaches at most
        // 2 * k slots, and resolved(true) rejects the batch if a later acquire
        // recycled an earlier pair's slot.
        if (expert_cache_.slot_count() < 8u * static_cast<size_t>(moe_.k())) {
            return "the MTP expert cache holds fewer than 8 slots per top-k selection";
        }
        return nullptr;
    }

    bool batch_ready() const { return batch_blocker() == nullptr; }

    void clear() {
        if (!ready_) return;
        qsa_state_->clear(stream_);
        q38_mtp_check(cudaMemsetAsync(f32_.residual(), 0,
            sizeof(float) * static_cast<size_t>(hc_dim_), stream_), "clear residual");
        q38_mtp_check(cudaStreamSynchronize(stream_), "clear sync");
        drafts_.clear();
    }

    // One draft round: propose up to k tokens for positions pos+1 .. pos+k from
    // the pending token at position pos and the trunk's wide residual at
    // position pos-1 (h_wide = nullptr keeps the MTP block's own hidden state).
    const std::vector<int> & draft(int k, const float * h_wide, int token, int pos) {
        drafts_.clear();
        if (k <= 0) return drafts_;
        if (k > 1 && q38_mtp_batch_verify_enabled() && batch_ready()) {
            // One line per process, so an A/B log can tell "no effect" from
            // "path never taken" and which expert batch cap was used.
            static bool announced = false;
            if (!announced) {
                announced = true;
                std::fprintf(stderr,
                             "warodecode: MTP batched block MoE taken (k=%d, expert batch cap=%d)\n",
                             k, moe_batch_.batch_k());
            }
            // Batched block MoE: the same k positions, one routed expert pass.
            // last_hidden() is left in the same place the serial loop leaves it
            // (the last position's combined wide residual), so the verify loop
            // consumes either path identically.
            const size_t n = static_cast<size_t>(k) * static_cast<size_t>(hc_dim_);
            draft_hidden_dev_ = ensure_draft_hidden(n);
            draft_batched(k, h_wide, token, pos, draft_hidden_dev_);
            // The verify loop reads state.residual() (the MTP block's last
            // hidden) between rounds, so leave it exactly where the serial loop
            // leaves it: the last position's combined residual.
            q38_mtp_check(cudaMemcpyAsync(f32_.residual(), draft_hidden_dev_ +
                static_cast<size_t>(k - 1) * static_cast<size_t>(hc_dim_),
                sizeof(float) * static_cast<size_t>(hc_dim_), cudaMemcpyDeviceToDevice, stream_),
                "restore last draft residual");
            for (size_t i = 0; i < slots_.size(); ++i) drafts_.push_back(slots_[i].token_out);
            return drafts_;
        }
        if (q38_mtp_batch_verify_enabled() && k > 1) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "warodecode: MTP batched block MoE not taken (%s)\n",
                             batch_blocker() != nullptr ? batch_blocker() : "k < 2");
            }
        }
        drafts_.reserve(static_cast<size_t>(k));
        const float * h = h_wide;
        int fed = token;
        for (int i = 0; i < k; ++i) {
            const int p = pos + i;
            if (p < 0 || p >= max_ctx_) {
                throw std::runtime_error("Qwen3.8 MTP: draft position out of range");
            }
            const int next = forward_one(fed, p, h);
            drafts_.push_back(next);
            fed = next;
            h = nullptr;   // later steps consume the MTP block's own hidden state
        }
        return drafts_;
    }

    // The MTP block's own wide residual after the most recent draft step.
    const float * last_hidden() const { return f32_.residual(); }

private:
    // -----------------------------------------------------------------------
    // MTP block forward, split into the three phases the batch path needs.
    //
    //   prepare()   embedding norm, wide-residual norm, eh_proj, attention mix
    //               and the QSA branch: per position, sequential (the QSA draft
    //               cache is written and read position by position).
    //   block_moe() the FFN mix of every prepared position followed by ONE
    //               routed MoE pass over the token batch, then the shared
    //               expert per position.
    //   finish()    the FFN combine, the wide-residual copy, the head mix and
    //               the Q4_0 head GEMV for one position.
    //
    // forward_one() = prepare + block_moe({one slot}) + finish and is the path
    // every run used before this file grew a batch path; draft() calls the three
    // phases directly and folds the k positions of a round into one block_moe().
    // -----------------------------------------------------------------------
    struct DraftSlot {
        int token = 0;
        int pos = 0;
        const float * h_override = nullptr;
        // Device buffer the slot's wide residual is copied into, or null to
        // leave it in f32_.residual() the way the serial path does. "owned"
        // means the caller passed its own device buffer (the batch path's
        // draft_hidden_), so prepare() has to preserve f32_.residual() for the
        // following prepare() and finish() restores it before the combine.
        float * hidden_out = nullptr;
        bool hidden_owned = false;
        int out_slot = 0;                  // slice of moe_out_/moe_slots_ to use
        int token_out = 0;
    };

    // Phase 1: everything up to and including the attention branch at the
    // slot's position. The FFN mix is deliberately left to block_moe().
    void prepare_slot(DraftSlot & slot) {
        cudaStream_t stream = stream_;
        const Qwen38Layout & q = rt_->layout;

        // The first slot of a round seeds the block's wide residual from the
        // trunk; every later slot keeps what the previous finish() left there.
        if (slot.h_override != nullptr) {
            q38_mtp_check(cudaMemcpyAsync(f32_.residual(), slot.h_override,
                sizeof(float) * static_cast<size_t>(hc_dim_), cudaMemcpyDeviceToDevice, stream),
                "seed residual");
        }

        if (slot.hidden_out != nullptr && slot.hidden_owned) {
            // The batch path has to preserve the wide residual of every slot
            // across the following prepare() calls, so it keeps a device copy.
            q38_mtp_check(cudaMemcpyAsync(slot.hidden_out, f32_.residual(),
                sizeof(float) * static_cast<size_t>(hc_dim_), cudaMemcpyDeviceToDevice, stream),
                "copy wide residual");
        }

        // 1) embedding row -> enorm -> concat first half
        launch_q8_0_gather_row_f32(rt_->token_embd, slot.token, embed_row_, stream);
        q38_mtp_enorm_kernel<<<1, 256, 0, stream>>>(
            embed_row_, weights_.enorm.data, concat_, hidden_, hc_, q.rms_eps);

        // 2) wide residual -> hnorm -> concat second half
        q38_mtp_hnorm_kernel<<<hc_, 256, 0, stream>>>(
            f32_.residual(), weights_.hnorm.data, concat_, hidden_, hc_, q.rms_eps);

        // 3) nextn.eh_proj per stream -> the block's residual seed
        for (int c = 0; c < hc_; ++c) {
            launch_lowbit_gemv_float(
                weights_.eh_proj,
                concat_ + static_cast<size_t>(c) * 2 * hidden_,
                f32_.residual() + static_cast<size_t>(c) * hidden_, stream);
        }

        // 4) QSA attention branch
        __half * cur = q38_mtp_hc_mix(
            f32_.residual(), weights_.hc_attn, f32_, hc_attn_, q.rms_eps, stream);
        __half * attn_out = run_qwen38_qsa(
            cur, *weights_.qsa_w, *qsa_state_, *qsa_scratch_,
            qsa_layout_, layer_, slot.pos, Qwen38Position4::text(slot.pos), stream);
        run_qwen38_hc_combine(
            f32_.residual(), attn_out, hc_attn_, stream, !q38_residual_f32_enabled());
    }

    // Phase 2a: the FFN mix of one prepared slot. It reads the block residual,
    // which prepare_slot() left exactly where the serial forward had it, and
    // returns the batch activation the routed experts consume.
    __half * slot_mix(DraftSlot & slot) {
        return q38_mtp_hc_mix(
            f32_.residual(), weights_.hc_ffn, f32_, hc_ffn_, rt_->layout.rms_eps, stream_);
    }

    // Phase 2b: the routed MoE of every slot. The pair kernels run once for the
    // whole batch, and the shared expert runs per slot.
    //
    // The serial routed call site consumes the HC mix output as __half, while
    // the F32-activation batch API wants an F32 activation that is exactly
    // F16-representable (the other F32-x runners are bit-identical under that
    // condition). The batch path therefore feeds the very same half words
    // through a widened alias; the shared expert keeps the __half view, so both
    // call sites stay exactly as they were.
    void block_moe(
            DraftSlot * slots,
            int count,
            const std::vector<__half *> & activations) {
        cudaStream_t stream = stream_;
        const Qwen38Layout & q = rt_->layout;

        const bool batch = count > 1 &&
            moe_batch_.compatible(expert_cache_, moe_, *weights_.routed.gate,
                                  *weights_.routed.up, *weights_.routed.down, count) &&
            q38_mtp_try_block_moe_batch(slots, count, activations, stream);
        if (!batch) {
            for (int i = 0; i < count; ++i) {
                run_qwen38_routed_moe_separate<false>(
                    layer_, activations[static_cast<size_t>(i)], weights_.router,
                    *weights_.routed.gate, *weights_.routed.up, *weights_.routed.down,
                    expert_cache_, moe_, moe_slot(slots[i].out_slot), q.expert_weights_scale, stream);
            }
        }

        for (int i = 0; i < count; ++i) {
            run_qwen38_shared_expert(
                activations[static_cast<size_t>(i)], weights_.shared, moe_,
                moe_slot(slots[i].out_slot), stream);
        }
    }

    // One batch pass, or false to keep the per-slot loop. Every failure here is
    // a geometry/type mismatch, never a numeric fallback: the pair kernels are
    // the same per-row decode the serial path uses.
    bool q38_mtp_try_block_moe_batch(
            DraftSlot * slots,
            int count,
            const std::vector<__half *> & activations,
            cudaStream_t stream) {
        widen_activations(activations);
        moe_batch_.open_all(moe_, weights_.router, batch_activation_ptrs_,
                            rt_->layout.expert_weights_scale, stream, false);
        const int k = moe_.k();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < k; ++j) {
                const int expert = moe_.h_ids()[i * k + j];
                auto w = expert_cache_.acquire(
                    layer_, expert, *weights_.routed.gate, *weights_.routed.up,
                    *weights_.routed.down, stream);
                moe_batch_.set_pair_weights(i * k + j, w.gate, w.up, w.down);
            }
        }
        if (!moe_batch_.resolved(true)) return false;
        moe_batch_.run(moe_, *weights_.routed.gate, *weights_.routed.up, *weights_.routed.down, stream);
        // The layout of moe_slots_ is [k][hidden] and the slots of one pass are
        // always 0..k-1, so the batch output lands exactly on the slices the
        // shared expert and the combine then read.
        moe_batch_.write_output(moe_, nullptr, moe_slots_, stream);
        return true;
    }

    // Phase 3: the FFN combine and the head for one slot. The wide residual of
    // the slot has to be in f32_.residual() while the combine runs, so a batch
    // restores the copy prepare() made.
    void finish_slot(DraftSlot & slot) {
        cudaStream_t stream = stream_;
        const Qwen38Layout & q = rt_->layout;

        if (slot.hidden_out != nullptr && slot.hidden_owned) {
            // The batch path's prepares ran in front of this finish, so the
            // slot's own post-attention residual has to come back before the
            // combine; the serial path never left it in the first place.
            q38_mtp_check(cudaMemcpyAsync(f32_.residual(), slot.hidden_out,
                sizeof(float) * static_cast<size_t>(hc_dim_), cudaMemcpyDeviceToDevice, stream),
                "restore wide residual");
        }

        run_qwen38_hc_combine(
            f32_.residual(), moe_slot(slot.out_slot), hc_ffn_, stream, !q38_residual_f32_enabled());
        if (slot.hidden_out != nullptr && slot.hidden_owned) {
            // The head mix reads the combined residual; the caller wants the
            // pre-head state (last_hidden() of the serial loop), which is
            // exactly the combined residual the line above produced, so copy it
            // before the mix overwrites f32_.
            q38_mtp_check(cudaMemcpyAsync(slot.hidden_out, f32_.residual(),
                sizeof(float) * static_cast<size_t>(hc_dim_), cudaMemcpyDeviceToDevice, stream),
                "copy combined wide residual");
        }

        // 6) nextn head: hc_head mix then the Q4_0 shared head GEMV
        const int threads = 256;
        __half * head = q38_mtp_hc_mix(
            f32_.residual(), weights_.hc_head, f32_, hc_head_, q.rms_eps, stream);
        const int warps = threads / 32;
        const int blocks = (weights_.shared_head_rows + warps - 1) / warps;
        q38_mtp_q4_0_gemv_kernel<<<blocks, threads,
            static_cast<size_t>(weights_.shared_head_cols) * sizeof(__half), stream>>>(
            weights_.shared_head, head, logits_,
            weights_.shared_head_rows, weights_.shared_head_cols);
        q38_argmax_kernel<<<1, 256, 0, stream>>>(logits_, vocab_, d_token_, d_nonfinite_);
        q38_mtp_check(cudaMemcpyAsync(h_token_, d_token_, sizeof(int),
            cudaMemcpyDeviceToHost, stream), "copy draft token");
        q38_mtp_check(cudaMemcpyAsync(h_nonfinite_, d_nonfinite_, sizeof(int),
            cudaMemcpyDeviceToHost, stream), "copy draft flag");
        q38_mtp_check(cudaStreamSynchronize(stream), "draft sync");
        if (*h_nonfinite_) {
            throw std::runtime_error("Qwen3.8 MTP: draft logits contain NaN or infinity");
        }
        slot.token_out = *h_token_;
    }

    // The per-position MoE output slices. The serial path uses slice 0 and
    // alternating with moe_out_, the batch path fills one slice per position.
    __half * moe_slot(int index) {
        if (index <= 0) return moe_out_;
        return moe_slots_ + static_cast<size_t>(index - 1) * static_cast<size_t>(hidden_);
    }

    // One MTP block forward at pos (the serial entry point).
    int forward_one(int token, int pos, const float * h_override) {
        const bool profile = q38_mtp_profile_enabled();
        cudaEvent_t e0{}, e1{};
        q38_mtp_check(cudaEventCreate(&e0), "event create");
        q38_mtp_check(cudaEventCreate(&e1), "event create");
        q38_mtp_check(cudaEventRecord(e0, stream_), "event record");
        cudaEvent_t s0{}, s1{}, s2{}, s3{}, s4{};
        if (profile) {
            q38_mtp_check(cudaEventCreate(&s0), "profile event");
            q38_mtp_check(cudaEventCreate(&s1), "profile event");
            q38_mtp_check(cudaEventCreate(&s2), "profile event");
            q38_mtp_check(cudaEventCreate(&s3), "profile event");
            q38_mtp_check(cudaEventCreate(&s4), "profile event");
            q38_mtp_check(cudaEventRecord(s0, stream_), "profile event record");
        }

        DraftSlot slot;
        slot.token = token;
        slot.pos = pos;
        slot.h_override = h_override;
        slot.hidden_out = nullptr;
        slot.hidden_owned = false;
        slot.out_slot = 0;
        try {
            prepare_slot(slot);
            single_activation_.assign(1, slot_mix(slot));
            if (profile) q38_mtp_check(cudaEventRecord(s1, stream_), "profile event record");
            block_moe(&slot, 1, single_activation_);
            if (profile) q38_mtp_check(cudaEventRecord(s3, stream_), "profile event record");
            finish_slot(slot);
            if (profile) q38_mtp_check(cudaEventRecord(s4, stream_), "profile event record");
        } catch (...) {
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaEventDestroy(s0);
            cudaEventDestroy(s1);
            cudaEventDestroy(s2);
            cudaEventDestroy(s3);
            cudaEventDestroy(s4);
            throw;
        }

        ++stats_.draft_steps;
        q38_mtp_check(cudaEventRecord(e1, stream_), "event record");
        q38_mtp_check(cudaEventSynchronize(e1), "event sync");
        float ms = 0.0f;
        q38_mtp_check(cudaEventElapsedTime(&ms, e0, e1), "event elapsed");
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        stats_.draft_seconds += static_cast<double>(ms) / 1000.0;
        if (profile) {
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            cudaEventElapsedTime(&p[0], s0, s1);
            cudaEventElapsedTime(&p[1], s1, s2);
            cudaEventElapsedTime(&p[2], s2, s3);
            cudaEventElapsedTime(&p[3], s3, s4);
            stats_.stage_norm_seconds += static_cast<double>(p[0]) / 1000.0;
            stats_.stage_attn_seconds += static_cast<double>(p[1]) / 1000.0;
            stats_.stage_moe_seconds += static_cast<double>(p[2]) / 1000.0;
            stats_.stage_head_seconds += static_cast<double>(p[3]) / 1000.0;
            cudaEventDestroy(s0);
            cudaEventDestroy(s1);
            cudaEventDestroy(s2);
            cudaEventDestroy(s3);
            cudaEventDestroy(s4);
        }
        return slot.token_out;
    }

    // One batched draft round: prepare every position, one MoE pass over the
    // whole batch, then finish. The draft chain stays sequential - the token of
    // position i feeds prepare() of position i + 1 - and only the MoE, whose
    // input is a pure function of each position, is folded together.
    void draft_batched(int k, const float * h_wide, int token, int pos, float * hidden_out) {
        // hidden_out is device memory (draft_hidden_dev_); the caller only needs
        // the last slot's residual, which draft() copies back into f32_.
        if (k < 1) return;
        slots_.assign(static_cast<size_t>(k), DraftSlot{});
        batch_activations_.assign(static_cast<size_t>(k), nullptr);

        const bool profile = q38_mtp_profile_enabled();
        cudaEvent_t s0{}, s1{}, s2{}, s3{};
        if (profile) {
            q38_mtp_check(cudaEventCreate(&s0), "profile event");
            q38_mtp_check(cudaEventCreate(&s1), "profile event");
            q38_mtp_check(cudaEventCreate(&s2), "profile event");
            q38_mtp_check(cudaEventCreate(&s3), "profile event");
            q38_mtp_check(cudaEventRecord(s0, stream_), "profile event record");
        }

        // The order below is the serial chain's order: prepare i + 1 runs after
        // the QSA branch of position i wrote its KV entries, and the FFN mix of
        // position i runs while f32_.residual() still holds that position's
        // post-attention residual. Every slot snapshots its wide residual
        // because the next prepare() overwrites f32_.residual().
        for (int i = 0; i < k; ++i) {
            DraftSlot & slot = slots_[static_cast<size_t>(i)];
            slot.token = (i == 0) ? token : slots_[static_cast<size_t>(i - 1)].token_out;
            slot.pos = pos + i;
            slot.h_override = (i == 0) ? h_wide : nullptr;
            slot.hidden_out = hidden_out + static_cast<size_t>(i) * static_cast<size_t>(hc_dim_);
            slot.hidden_owned = true;
            slot.out_slot = i;
            prepare_slot(slot);
            batch_activations_[static_cast<size_t>(i)] = slot_mix(slot);
        }

        if (profile) q38_mtp_check(cudaEventRecord(s1, stream_), "profile event record");

        // One routed MoE pass for the whole round, then the per-position combine
        // and head. finish() restores each slot's residual copy, so the finishes
        // are independent of each other.
        block_moe(slots_.data(), k, batch_activations_);
        if (profile) q38_mtp_check(cudaEventRecord(s2, stream_), "profile event record");
        for (int i = 0; i < k; ++i) finish_slot(slots_[static_cast<size_t>(i)]);
        if (profile) {
            q38_mtp_check(cudaEventRecord(s3, stream_), "profile event record");
            q38_mtp_check(cudaEventSynchronize(s3), "profile event sync");
            float p[3] = {0.0f, 0.0f, 0.0f};
            q38_mtp_check(cudaEventElapsedTime(&p[0], s0, s1), "profile event elapsed");
            q38_mtp_check(cudaEventElapsedTime(&p[1], s1, s2), "profile event elapsed");
            q38_mtp_check(cudaEventElapsedTime(&p[2], s2, s3), "profile event elapsed");
            stats_.stage_norm_seconds += static_cast<double>(p[0]) / 1000.0;
            stats_.stage_attn_seconds += 0.0;
            stats_.stage_moe_seconds += static_cast<double>(p[1]) / 1000.0;
            stats_.stage_head_seconds += static_cast<double>(p[2]) / 1000.0;
            stats_.draft_steps += k;
            cudaEventDestroy(s0);
            cudaEventDestroy(s1);
            cudaEventDestroy(s2);
            cudaEventDestroy(s3);
        }
    }

    Qwen38Runtime * rt_ = nullptr;
    Qwen38MtpWeights weights_;
    cudaStream_t stream_ = nullptr;
    int max_ctx_ = 0;
    int hidden_ = 0;
    int hc_ = 0;
    int hc_dim_ = 0;
    int low_rank_ = 0;
    int vocab_ = 0;
    int layer_ = 0;
    int ratio_ = 0;
    int draft_len_ = 2;
    bool ready_ = false;

    Qwen38Layout qsa_layout_;
    std::unique_ptr<Qwen38QsaState> qsa_state_;
    std::unique_ptr<Qwen38QsaScratch> qsa_scratch_;
    Qwen38HcScratch hc_attn_;
    Qwen38HcScratch hc_ffn_;
    Qwen38HcScratch hc_head_;
    Qwen38HcF32DebugScratch f32_;
    Qwen38MoeScratch moe_;
    GpuExpertCache expert_cache_;

    __half * moe_out_ = nullptr;
    float * embed_row_ = nullptr;
    __half * concat_ = nullptr;
    float * logits_ = nullptr;
    int * d_token_ = nullptr;
    int * d_nonfinite_ = nullptr;
    int * h_token_ = nullptr;
    int * h_nonfinite_ = nullptr;

    std::vector<int> drafts_;
    // Device scratch for the batch path's per-position wide residuals. Sized on
    // demand because k comes from the environment.
    float * ensure_draft_hidden(size_t elements) {
        if (draft_hidden_dev_ == nullptr || draft_hidden_capacity_ < elements) {
            if (draft_hidden_dev_) cudaFree(draft_hidden_dev_);
            draft_hidden_dev_ = nullptr;
            const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&draft_hidden_dev_),
                                             elements * sizeof(float));
            if (e != cudaSuccess) {
                throw std::runtime_error(std::string("Qwen3.8 MTP: cudaMalloc draft hidden: ") +
                                         cudaGetErrorString(e));
            }
            draft_hidden_capacity_ = elements;
        }
        return draft_hidden_dev_;
    }

    // The batch API consumes F32 activations; the __half buffer is exactly
    // F16-representable, so this alias cannot round.
    void widen_activations(const std::vector<__half *> & src) {
        batch_activation_ptrs_.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            batch_activation_ptrs_[i] = reinterpret_cast<const float *>(src[i]);
        }
    }

    std::vector<DraftSlot> slots_;
    std::vector<__half *> batch_activations_;
    std::vector<__half *> single_activation_;
    std::vector<const float *> batch_activation_ptrs_;
    float * draft_hidden_dev_ = nullptr;    // [capacity][hc_dim] batch wide residuals
    size_t draft_hidden_capacity_ = 0;
    __half * moe_slots_ = nullptr;          // [draft_len_ - 1][hidden] extra MoE outputs
    Qwen38MoeTokenBatch moe_batch_;
    Qwen38MtpStats stats_;
};

// ---------------------------------------------------------------------------
// Speculative draft/verify loop
//
// One round drafts k tokens for positions pos+1 .. pos+k from the pending token
// at position pos, then feeds the trunk positions pos .. pos+j. The trunk's own
// argmax at each position is the accept/reject oracle: a draft is consumed only
// while it is bit-identical to that argmax; the first mismatch contributes the
// trunk's token (exactly what plain decoding emits) and ends the round.
//
// Emitted tokens == trunk steps in every case, so with single-token kernels this
// loop cannot reduce the trunk work: it is the scaffolding for an M-position
// verify forward, and it preserves the greedy chain by construction.
// ---------------------------------------------------------------------------
inline Qwen38MtpStats qwen38_run_speculative(
        Qwen38Decoder & decoder,
        Qwen38Mtp & mtp,
        int start_token,
        int steps,
        std::vector<int> * generated) {
    if (steps <= 0) throw std::runtime_error("Qwen3.8 MTP steps must be positive");
    if (!mtp.ready()) throw std::runtime_error("Qwen3.8 MTP is not initialized");
    if (generated == nullptr) throw std::runtime_error("Qwen3.8 MTP needs an output vector");

    Qwen38DecodeState & state = decoder.state();
    cudaStream_t stream = state.stream();
    mtp.set_stream(stream);

    decoder.clear();
    mtp.clear();

    Qwen38MtpStats & stats = mtp.stats();
    stats = Qwen38MtpStats{};
    stats.draft_len = mtp.draft_len();

    cudaEvent_t begin{}, end{};
    q38_mtp_check(cudaEventCreate(&begin), "event create");
    q38_mtp_check(cudaEventCreate(&end), "event create");
    q38_mtp_check(cudaEventRecord(begin, stream), "event record");

    generated->clear();
    generated->reserve(static_cast<size_t>(steps));

    int pos = 0;
    int fed = start_token;
    const float * h_prev = nullptr;

    while (static_cast<int>(generated->size()) < steps) {
        const int remaining = steps - static_cast<int>(generated->size());
        int k = mtp.draft_len();
        if (k > remaining - 1) k = remaining - 1;   // never draft past the horizon

        const std::vector<int> & drafts = mtp.draft(k, h_prev, fed, pos);
        ++stats.rounds;
        stats.draft_tokens += k;
        // stats.draft_steps is counted once per forward inside forward_one().

        int ran = 0;
        int accepted = 0;
        for (int j = 0; j <= k; ++j) {
            if (static_cast<int>(generated->size()) >= steps) break;
            const int in_tok = (j == 0) ? fed : drafts[static_cast<size_t>(j - 1)];
            const int out = decoder.step(in_tok, pos + j);
            generated->push_back(out);
            ++ran;
            ++stats.verify_steps;
            if (j == k) break;                       // bonus token of a full round
            if (out == drafts[static_cast<size_t>(j)]) {
                ++accepted;
                ++stats.accepts;
                continue;
            }
            break;                                   // mismatch: out is the trunk token
        }
        if (k > 0 && accepted == k && ran > k) ++stats.full_rounds;

        // The next round starts one past the last position the trunk processed.
        // Stale draft KV entries can only exist at positions >= that point and
        // the next round rewrites them in increasing order before any attention
        // reads them (a QSA block is only pooled once its last position is
        // written), so no explicit rollback is needed here.
        pos += ran;
        if (ran > 0) {
            fed = generated->back();
            h_prev = state.residual();
        }
    }

    q38_mtp_check(cudaEventRecord(end, stream), "event record");
    q38_mtp_check(cudaEventSynchronize(end), "event sync");
    float ms = 0.0f;
    q38_mtp_check(cudaEventElapsedTime(&ms, begin, end), "event elapsed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    stats.seconds = static_cast<double>(ms) / 1000.0;
    return stats;
}

} // namespace wd
