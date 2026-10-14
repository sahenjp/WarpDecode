#pragma once

#include "qwen38_runtime.cuh"
#include "qwen38_state.cuh"
#include "qwen38_trace.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wd {

// Diagnostic ablation switches. WARODECODE_RESIDUAL_F32=1 (default) keeps the
// persistent HC residual stream in F32; =0 reproduces the legacy F16 lifetime.
// WARODECODE_HC_F32=1 additionally runs the HC mixing in F32.
inline bool q38_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_residual_f32_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_RESIDUAL_F32", true);
    return enabled;
}

inline bool q38_hc_f32_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_HC_F32", false);
    return enabled;
}

inline bool q38_moe_f32_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_MOE_F32", false);
    return enabled;
}

// ggml-compatible activation quantisation. llama.cpp feeds quantised weights
// with activations quantised to Q8_0 (32-element blocks, integer dot, per-block
// scale); the legacy F16-activation path multiplies a differently rounded
// activation and drifts ~1% per projection, which compounds over 48 layers and
// flips the greedy token. Enabled by default when the weights are Q8_0 and the
// F32 norm weights are available; WARODECODE_HC_Q8_0=0 restores the legacy path.
// Wiring switches for the F32 branch outputs. The kernels have their own
// WARODECODE_*_F32_MID switches; these gate whether forward.cuh consumes the
// unrounded result instead of the half mirror, so each wiring can be A/B'd
// without a rebuild.
inline bool q38_f32_out_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return true;
    return v[0] != '0';
}

inline bool q38_gdn_f32_out_enabled() {
    static const bool enabled = q38_f32_out_enabled("WARODECODE_GDN_F32_OUT");
    return enabled;
}

inline bool q38_qsa_f32_out_enabled() {
    static const bool enabled = q38_f32_out_enabled("WARODECODE_QSA_F32_OUT");
    return enabled;
}

// MoE F32 output wiring (WARODECODE_MOE_F32_OUT, default ON). The legacy path
// hands the combine the *half* mirror of the shared expert (d_down), which put a
// 9.2e-5 error into layer 0's output at every position; routing the unrounded
// value through moe_out_f32 drops that to 1.1e-7 (measured on L4). The greedy
// prefix is knife-edge sensitive to this, so the switch stays available for A/B.
inline bool q38_moe_f32_out_opt_in() {
    static const bool enabled = [] {
        const char * v = std::getenv("WARODECODE_MOE_F32_OUT");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return enabled;
}

inline bool q38_hc_q8_0_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_HC_Q8_0", true);
    return enabled;
}

// The attn branch is measurably more faithful to the CPU too (HC mix error
// 1.35% -> 0.03%) but it only helps end to end once the GDN/QSA/MoE paths use
// the same activation quantisation, so it stays opt-in for now.
inline bool q38_hc_q8_0_attn_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_HC_Q8_0_ATTN", true);
    return enabled;
}

inline bool q38_hc_q8_0_compatible(const Qwen38HcWeights & w) {
    return !w.debug_norm_f32.empty() &&
           w.down.lowbit && w.down.type == GgmlType::Q8_0 &&
           w.up.lowbit && w.up.type == GgmlType::Q8_0;
}

// Task J (WARODECODE_HC_SKIP_LEGACY, default ON).
//
// Both HC mix sites used to run the legacy F16 mirror first and the exact Q8_0
// replay afterwards. The F16 pass is pure duplicate work when the Q8_0 path is
// active: it writes st.hc_*.{xn,gate,lo,inject,mixed}, and every one of those
// buffers is either overwritten by the exact pass (the F32 -> half copies) or
// read only by the legacy combines that the exact pass makes unreachable
// (run_qwen38_hc_combine_f32_inject / run_qwen38_hc_combine_moe_f32_inject use
// the F32 scratch copy). The attn site already skipped it through
// attn_f32_mix_active; this is the same guard for the ffn site.
// =0 restores the previous behaviour for A/B runs.
inline bool q38_hc_skip_legacy_enabled() {
    static const bool enabled = q38_env_flag("WARODECODE_HC_SKIP_LEGACY", true);
    return enabled;
}

// One-time line so an on-device A/B log states which HC path actually ran
// (tok_s alone cannot: the fast and the reference paths are required to be
// bit-identical, and a stale binary looks exactly like a working switch).
inline void q38_hc_report_switches_once() {
    static const bool once = [] {
        std::fprintf(stderr,
                     "warodecode: HC mix Task J switches: skip_legacy=%d gamma_resident=%d "
                     "inject_warp=%d down_dp4a=%d\n",
                     q38_hc_skip_legacy_enabled() ? 1 : 0,
                     q38_hc_gamma_resident_enabled() ? 1 : 0,
                     q38_hc_f32_gemv_warp_enabled() ? 1 : 0,
                     q38_hc_q8_down_dp4a_enabled() ? 1 : 0);
        return true;
    }();
    (void)once;
}

__global__ void q38_repeat_embedding_kernel(
        const __half * __restrict__ emb,
        __half * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i < total) residual[i] = emb[i % hidden];
}

// F32 residual variant: the wide HC residual stream is kept in F32 to match the
// pinned CPU reference (which stores the residual in F32).
__global__ void q38_repeat_embedding_f32_kernel(
        const __half * __restrict__ emb,
        float * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i < total) residual[i] = __half2float(emb[i % hidden]);
}

// F32-source twin: ggml dequantises the token embedding into F32 and seeds the
// residual with those exact values. Reading the half mirror instead costs ~5e-4
// on every position (the embedding is the seed of the whole chain), so the F32
// row is what the residual must be built from.
__global__ void q38_repeat_embedding_from_f32_kernel(
        const float * __restrict__ emb,
        float * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i < total) residual[i] = emb[i % hidden];
}

__global__ void q38_float_copy_kernel(
        const float * __restrict__ in,
        float * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

struct Q38MaxPair {
    float v;
    int i;
};

struct Qwen38MoeRouteOverride {
    int position = -1;
    std::vector<int> expert_ids;
    std::vector<float> weights;
};

struct Qwen38Trace {
    std::vector<float> hc_init;
    std::vector<float> hc_attn_norm;
    std::vector<float> hc_attn_low;
    std::vector<float> hc_attn_gate;
    std::vector<float> hc_attn_mix;
    std::vector<float> hc_attn_mix_used;
    Qwen38HcF32Trace hc_f32;
    std::vector<float> gdn_qkv;
    std::vector<float> gdn_z;
    std::vector<float> gdn_beta;
    std::vector<float> gdn_alpha;
    std::vector<float> gdn_gate;
    std::vector<float> gdn_core_out;
    std::vector<float> gdn_norm_out;
    std::vector<float> attn_out;
    std::vector<float> after_attn;
    std::vector<float> hc_ffn_mix;
    std::vector<float> hc_ffn_mix_f32;
    std::vector<float> gdn_f32_out;
    std::vector<float> after_attn_f32;
    std::vector<float> ffn_out;
    std::vector<float> after_ffn;
    std::vector<float> router_logits;
    std::vector<float> router_weights;
    std::vector<int> expert_ids;
    bool use_f32_hc_layer0 = false;
    bool debug_f32_hc_attn_0_8 = false;
    bool debug_f32_hc_ffn_0_8 = false;
    bool debug_f32_hc_ffn_layer8 = false;
    int debug_hc_q8_1_attn_from = 0;
    int debug_hc_q8_1_attn_through = -1;
    int debug_hc_q8_1_ffn_from = 0;
    int debug_hc_q8_1_ffn_through = -1;
    int debug_hc_q8_0_ffn_from = 0;
    int debug_hc_q8_0_ffn_through = -1;
    bool debug_f32_hc_head = false;
    bool debug_layer0_hc_gate_f32_store = false;
    bool debug_layer0_hc_gate_f32_sigmoid = false;
    int debug_hc_gate_q8_0_layer = -1;
    int debug_hc_full_q8_0_layer = -1;
    int debug_qsa_q8_0_layer = -1;
    int debug_qsa_v_q8_0_layer = -1;
    int debug_hc_norm_f32_layer = -1;
    std::vector<float> debug_hc_norm_f32;
    size_t debug_hc_norm_override_count = 0;
    int debug_attention_output_f32_layer = -1;
    std::vector<float> debug_attention_output_f32;
    size_t debug_attention_output_override_count = 0;
    int debug_attention_input_f32_layer = -1;
    std::vector<float> debug_attention_input_f32;
    size_t debug_attention_input_override_count = 0;
    int debug_qsa_context_f32_layer = -1;
    std::vector<float> debug_qsa_context_f32;
    size_t debug_qsa_context_override_count = 0;
    int debug_qsa_pregate_f32_layer = -1;
    std::vector<float> debug_qsa_pregate_f32;
    size_t debug_qsa_pregate_override_count = 0;
    int debug_qsa_query_f32_layer = -1;
    std::vector<float> debug_qsa_query_f32;
    size_t debug_qsa_query_override_count = 0;
    int debug_qsa_key_store_f32_layer = -1;
    std::vector<float> debug_qsa_key_store_f32;
    size_t debug_qsa_key_store_override_count = 0;
    int debug_qsa_value_store_f32_layer = -1;
    std::vector<float> debug_qsa_value_store_f32;
    size_t debug_qsa_value_store_override_count = 0;
    int debug_qsa_key_cache_prefix_f32_layer = -1;
    std::vector<float> debug_qsa_key_cache_prefix_f32;
    size_t debug_qsa_key_cache_prefix_override_count = 0;
    int debug_qsa_value_cache_prefix_f32_layer = -1;
    std::vector<float> debug_qsa_value_cache_prefix_f32;
    size_t debug_qsa_value_cache_prefix_override_count = 0;
    int debug_hc_attn_combine_layer = -1;
    std::vector<float> debug_hc_attn_combine_f32;
    size_t debug_hc_attn_combine_override_count = 0;
    int debug_hc_inject_f32_layer = -1;
    std::vector<float> debug_hc_inject_f32;
    size_t debug_hc_inject_override_count = 0;
    int debug_layer_input_residual_layer = -1;
    std::vector<float> debug_layer_input_residual_f32;
    size_t debug_layer_input_residual_override_count = 0;
    std::vector<float> debug_layer8_after_attention_residual_f32;
    std::vector<float> debug_layer8_hc_ffn_mix_f32;
    std::vector<float> debug_final_residual_f32;
    std::vector<float> debug_layer14_gdn_core_f32;
    std::vector<float> debug_layer14_gdn_norm_output_f32;
    std::vector<float> debug_layer9_gdn_state_f32;
    std::vector<float> debug_layer9_gdn_conv_state_f32;
    int debug_layer9_gdn_state_layer = 9;
    int debug_layer9_gdn_conv_state_layer = 9;
    std::vector<int> debug_moe_ids;
    std::vector<float> debug_moe_weights;
    std::vector<Qwen38MoeRouteOverride> debug_moe_routes;
    std::vector<float> debug_router_cpu_activation_f32;
    std::vector<float> debug_routed_moe_output;
    int debug_routed_moe_output_layer = 8;
    bool debug_routed_moe_output_all_positions = false;
    size_t debug_moe_route_override_count = 0;
    size_t debug_routed_moe_output_override_count = 0;
    size_t debug_hc_output_override_count = 0;
    int debug_ffn_output_layer = 8;
    std::vector<float> debug_ffn_output;
    std::vector<float> debug_hc_residual;
    int debug_hc_residual_layer = 8;
    bool debug_hc_residual_all_positions = false;
    bool extended_parity = false;
    bool state_trace = false;
    int position = -1;
    int state_trace_position = -1;
    bool capture_state_position = false;
    std::vector<Qwen38TraceVector> extended_vectors;
    std::vector<Qwen38TraceSummaryVector> state_summaries;
    std::vector<Qwen38TraceIntegerVector> integer_vectors;

    bool captures_state_position() const {
        return state_trace && capture_state_position;
    }

    void add_extended(const std::string & name, const std::vector<float> & values) {
        if (extended_parity || captures_state_position()) {
            extended_vectors.push_back({state_trace ? "pos" + std::to_string(position) + "." + name : name, values});
        }
    }

    void add_extended_ints(const std::string & name, const std::vector<int> & values) {
        if (!extended_parity) return;
        std::vector<float> converted(values.begin(), values.end());
        extended_vectors.push_back({name, std::move(converted)});
    }

    void add_state_summary(
            const std::string & name,
            const std::vector<float> & values,
            std::vector<uint64_t> dims) {
        if (captures_state_position()) {
            state_summaries.push_back({"pos" + std::to_string(position) + "." + name,
                std::move(dims), qwen38_trace_stats(values)});
        }
    }

    void add_integer_vector(const std::string & name, std::vector<int64_t> values) {
        if (captures_state_position()) {
            integer_vectors.push_back({"pos" + std::to_string(position) + "." + name, std::move(values)});
        }
    }
};

__global__ void q38_argmax_kernel(
        const float * __restrict__ logits,
        int n,
        int * __restrict__ token,
        int * __restrict__ has_nonfinite) {
    __shared__ Q38MaxPair sm[256];
    __shared__ int sm_nonfinite[256];
    Q38MaxPair best{-3.402823466e+38F, 0};
    int nonfinite = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (!::isfinite(v)) {
            nonfinite = 1;
            continue;
        }
        if (v > best.v || (v == best.v && i < best.i)) {
            best.v = v;
            best.i = i;
        }
    }
    sm[threadIdx.x] = best;
    sm_nonfinite[threadIdx.x] = nonfinite;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const Q38MaxPair other = sm[threadIdx.x + stride];
            if (other.v > sm[threadIdx.x].v ||
                (other.v == sm[threadIdx.x].v && other.i < sm[threadIdx.x].i)) {
                sm[threadIdx.x] = other;
            }
            sm_nonfinite[threadIdx.x] |= sm_nonfinite[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *token = sm[0].i;
        *has_nonfinite = sm_nonfinite[0];
    }
}

// Phase split of one Qwen38Decoder::run() call. The numbers come from CUDA
// events recorded on the decode stream, so model load, autotune and host setup
// stay outside them. prefill_tokens == 0 means "pure decode": decode_seconds
// then covers every forward of the call.
struct Qwen38RunTiming {
    double prefill_seconds = 0.0;
    int prefill_tokens = 0;
    double decode_seconds = 0.0;
    int decode_tokens = 0;
};

// Task F: per-stage CUDA-event profiler for one decode token.
//
// The stage ids, the accumulators and the JSON writer live in
// qwen38_trace.hpp (which stays host-only so the plain g++ trace test keeps
// compiling). This class is the CUDA side: it owns the cudaEvent_t pool and
// records the pairs on the decode stream.
//
// Disabled by default: with WARODECODE_PROFILE unset (or "0") no event is ever
// created, the pools stay empty and every hook is a single predictable branch
// on enabled(), so the default decode path is unchanged.
class Qwen38StageProfiler {
public:
    // Upper bound of one stage's repeats inside a single token. The longest
    // stages repeat once per trunk layer, so this only has to beat the layer
    // count of the model with room to spare.
    static constexpr int kStageCap = 96;
    static constexpr int kLayerCap = 128;

    Qwen38StageProfiler() {
        const char * v = std::getenv("WARODECODE_PROFILE");
        enabled_ = v != nullptr && v[0] != '\0' && v[0] != '0';
        if (!enabled_) return;
        const size_t slots = static_cast<size_t>(kQwen38StageCount) * static_cast<size_t>(kStageCap);
        start_.assign(slots, nullptr);
        stop_.assign(slots, nullptr);
        layer_start_.assign(kLayerCap, nullptr);
        layer_stop_.assign(kLayerCap, nullptr);
        for (size_t i = 0; i < slots; ++i) {
            if (cudaEventCreate(&start_[i]) != cudaSuccess || cudaEventCreate(&stop_[i]) != cudaSuccess) {
                enabled_ = false;
                return;
            }
        }
        for (int i = 0; i < kLayerCap; ++i) {
            if (cudaEventCreate(&layer_start_[static_cast<size_t>(i)]) != cudaSuccess ||
                cudaEventCreate(&layer_stop_[static_cast<size_t>(i)]) != cudaSuccess) {
                enabled_ = false;
                return;
            }
        }
        totals_.layer_device_ms.assign(kLayerCap, 0.0);
        totals_.layer_host_ms.assign(kLayerCap, 0.0);
    }

    Qwen38StageProfiler(const Qwen38StageProfiler &) = delete;
    Qwen38StageProfiler & operator=(const Qwen38StageProfiler &) = delete;

    ~Qwen38StageProfiler() {
        for (cudaEvent_t e : start_) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : stop_) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : layer_start_) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : layer_stop_) if (e) cudaEventDestroy(e);
    }

    bool enabled() const { return enabled_; }
    const Qwen38ProfileTotals & totals() const { return totals_; }
    std::string to_json() const { return qwen38_profile_json(totals_); }

    static double host_now() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void mark_begin(int stage, cudaStream_t stream) {
        if (!enabled_) return;
        host_begin_[static_cast<size_t>(stage)] = host_now();
        if (pending_[stage]) {
            // A previous begin never saw its end (aborted run). Drop it instead of
            // pairing it with the wrong stop event.
            ++unmatched_[stage];
            pending_[stage] = false;
        }
        const int slot = written_[stage];
        if (slot >= kStageCap) {
            ++overflow_[stage];
            return;
        }
        if (cudaEventRecord(start_[index(stage, slot)], stream) != cudaSuccess) {
            ++overflow_[stage];
            return;
        }
        pending_[stage] = true;
    }

    void mark_end(int stage, cudaStream_t stream) {
        if (!enabled_) return;
        totals_.stage[static_cast<size_t>(stage)].host_ms +=
            host_now() - host_begin_[static_cast<size_t>(stage)];
        if (!pending_[stage]) {
            ++unmatched_[stage];
            return;
        }
        pending_[stage] = false;
        const int slot = written_[stage];
        if (cudaEventRecord(stop_[index(stage, slot)], stream) != cudaSuccess) {
            ++unmatched_[stage];
            return;
        }
        ++written_[stage];
    }

    void layer_begin(int layer, cudaStream_t stream) {
        if (!enabled_ || layer < 0 || layer >= kLayerCap) return;
        layer_host_begin_[static_cast<size_t>(layer)] = host_now();
        cudaEventRecord(layer_start_[static_cast<size_t>(layer)], stream);
    }

    void layer_end(int layer, cudaStream_t stream) {
        if (!enabled_ || layer < 0 || layer >= kLayerCap) return;
        totals_.layer_host_ms[static_cast<size_t>(layer)] +=
            host_now() - layer_host_begin_[static_cast<size_t>(layer)];
        cudaEventRecord(layer_stop_[static_cast<size_t>(layer)], stream);
        if (layer + 1 > layer_count_) layer_count_ = layer + 1;
    }

    // Marks the profiled token as decode (true) or prefill (false). run() owns
    // the phase, step() only reports its own wall time.
    void set_decode_phase(bool decode) { decode_phase_ = decode; }

    void note_token_host(double host_ms) {
        if (!enabled_) return;
        totals_.token_host_ms += host_ms;
        ++totals_.tokens;
        if (decode_phase_) ++totals_.phases;
    }

    // Reads the recorded event pairs back. Call only once the decode stream has
    // been synchronised (step() already ends with a cudaStreamSynchronize): every
    // recorded event is complete at that point, so cudaEventElapsedTime cannot
    // return cudaErrorNotReady.
    void flush_token() {
        if (!enabled_) return;
        const double flush_start = host_now();
        for (int s = 0; s < kQwen38StageCount; ++s) {
            Qwen38StageSample & sample = totals_.stage[static_cast<size_t>(s)];
            for (int i = 0; i < written_[s]; ++i) {
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, start_[index(s, i)], stop_[index(s, i)]) == cudaSuccess) {
                    sample.device_ms += static_cast<double>(ms);
                    ++sample.calls;
                }
            }
            sample.unmatched += unmatched_[s];
            sample.overflow += overflow_[s];
            written_[s] = 0;
            pending_[s] = false;
            unmatched_[s] = 0;
            overflow_[s] = 0;
        }
        for (int i = 0; i < layer_count_; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(
                    &ms, layer_start_[static_cast<size_t>(i)], layer_stop_[static_cast<size_t>(i)]) == cudaSuccess) {
                totals_.layer_device_ms[static_cast<size_t>(i)] += static_cast<double>(ms);
                ++totals_.layer_calls;
            }
        }
        layer_count_ = 0;
        totals_.flush_host_ms += host_now() - flush_start;
    }

private:
    static size_t index(int stage, int slot) {
        return static_cast<size_t>(stage) * static_cast<size_t>(kStageCap) + static_cast<size_t>(slot);
    }

    bool enabled_ = false;
    bool decode_phase_ = true;
    std::vector<cudaEvent_t> start_;
    std::vector<cudaEvent_t> stop_;
    std::vector<cudaEvent_t> layer_start_;
    std::vector<cudaEvent_t> layer_stop_;
    int written_[kQwen38StageCount] = {};
    int unmatched_[kQwen38StageCount] = {};
    int overflow_[kQwen38StageCount] = {};
    bool pending_[kQwen38StageCount] = {};
    double host_begin_[kQwen38StageCount] = {};
    double layer_host_begin_[kLayerCap] = {};
    int layer_count_ = 0;
    Qwen38ProfileTotals totals_;
};

// RAII form of one profiled region. Holds no CUDA handle itself, so it is safe
// to declare inside the hot path; with the profiler off both calls inline to a
// single branch. Regions that cannot be brace-delimited use mark_begin() /
// mark_end() directly on the profiler.
class Qwen38StageScope {
public:
    Qwen38StageScope(Qwen38StageProfiler * profiler, int stage, cudaStream_t stream)
        : profiler_(profiler), stage_(stage), stream_(stream) {
        if (profiler_ != nullptr) profiler_->mark_begin(stage_, stream_);
    }
    ~Qwen38StageScope() {
        if (profiler_ != nullptr) profiler_->mark_end(stage_, stream_);
    }
    Qwen38StageScope(const Qwen38StageScope &) = delete;
    Qwen38StageScope & operator=(const Qwen38StageScope &) = delete;

private:
    Qwen38StageProfiler * profiler_ = nullptr;
    int stage_ = 0;
    cudaStream_t stream_ = nullptr;
};

class Qwen38Decoder {
public:
    Qwen38Decoder(Qwen38Runtime & rt, Qwen38DecodeState & state)
        : rt_(rt), state_(state) {
        if (rt_.vocab <= 0) throw std::runtime_error("Qwen3.8 vocab is not initialized");
        hc_f32_scratch_.initialize(rt_.layout.hidden, rt_.layout.hc_count, rt_.layout.hc_low_rank);
        cudaError_t e = cudaMalloc(
            reinterpret_cast<void **>(&logits_),
            static_cast<size_t>(rt_.vocab) * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 logits failed");
        e = cudaMalloc(reinterpret_cast<void **>(&next_token_), sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 next token failed");
        e = cudaMalloc(reinterpret_cast<void **>(&has_nonfinite_), sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 validation flag failed");
        e = cudaMallocHost(reinterpret_cast<void **>(&host_next_token_), sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error("cudaMallocHost Qwen3.8 token failed");
        e = cudaMallocHost(reinterpret_cast<void **>(&host_has_nonfinite_), sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error("cudaMallocHost Qwen3.8 validation flag failed");
    }

    Qwen38Decoder(const Qwen38Decoder &) = delete;
    Qwen38Decoder & operator=(const Qwen38Decoder &) = delete;

    ~Qwen38Decoder() {
        if (logits_) cudaFree(logits_);
        if (next_token_) cudaFree(next_token_);
        if (has_nonfinite_) cudaFree(has_nonfinite_);
        if (host_next_token_) cudaFreeHost(host_next_token_);
        if (host_has_nonfinite_) cudaFreeHost(host_has_nonfinite_);
    }

    void clear() {
        state_.clear();
        history_.clear();
    }

    // MTP hook (src/qwen38_mtp.cuh): the speculative draft/verify loop drives
    // step() itself and reads the decode state the decoder already owns. These
    // accessors change no behaviour of step()/run().
    Qwen38Runtime & runtime() { return rt_; }
    Qwen38DecodeState & state() { return state_; }

    // Task F: per-stage profile totals. All zeros unless WARODECODE_PROFILE is
    // set, so the caller can always print the object.
    const Qwen38ProfileTotals & profile_totals() const { return profile_.totals(); }
    bool profile_enabled() const { return profile_.enabled(); }
    // Always a complete JSON object so the caller can splice it unconditionally.
    std::string profile_json() const {
        return profile_.enabled() ? profile_.to_json() : std::string("{\"enabled\":0}");
    }
    // Host-side wall clock, interval counts and the per-layer arrays. Separate
    // from profile_json() so "profile" holds per-token scalars only.
    std::string profile_detail_json() const {
        return profile_.enabled()
            ? qwen38_profile_detail_json(profile_.totals()) : std::string("{}");
    }

    int step(int token, int pos, Qwen38Trace * trace = nullptr,
             bool debug_f32_hc_attn_0_8 = false,
             bool debug_f32_hc_ffn_0_8 = false,
             bool debug_f32_hc_ffn_layer8 = false,
             int debug_hc_q8_1_attn_from = 0,
             int debug_hc_q8_1_attn_through = -1,
             int debug_hc_q8_1_ffn_from = 0,
             int debug_hc_q8_1_ffn_through = -1,
             int debug_hc_q8_0_ffn_from = 0,
             int debug_hc_q8_0_ffn_through = -1) {
        const Qwen38Layout & q = rt_.layout;
        if (token < 0 || token >= rt_.vocab) throw std::runtime_error("Qwen3.8 token out of range");
        if (pos < 0 || pos >= state_.max_ctx()) throw std::runtime_error("Qwen3.8 position out of range");
        if (pos != static_cast<int>(history_.size())) {
            throw std::runtime_error("Qwen3.8 decoder currently requires contiguous single-sequence positions");
        }

        cudaStream_t stream = state_.stream();
        // Task F: one event pair spans the whole token body; the device->host token
        // copy and the closing synchronise are measured as kQwen38StageSync.
        const bool profile_on = profile_.enabled();
        const double profile_token_start = profile_on ? Qwen38StageProfiler::host_now() : 0.0;
        profile_.mark_begin(kQwen38StageToken, stream);
        Qwen38HcF32DebugScratch & hc_f32_scratch = hc_f32_scratch_;
        bool f32_layer0_router_preselected = false;
        if (trace && trace->state_trace) {
            trace->position = pos;
            trace->capture_state_position = trace->state_trace_position < 0 ||
                pos == trace->state_trace_position;
            trace_context_ = "position " + std::to_string(pos) + " embedding";
            trace->add_integer_vector("position", {pos});
            trace->add_integer_vector("input_token", {token});
            trace->add_extended("embedding.hc_init", copy_half(
                stream, state_.residual(), static_cast<size_t>(q.hidden) * q.hc_count));
        }
        const bool legacy_trace = trace &&
            (!trace->state_trace || (trace->use_f32_hc_layer0 && pos == 0));
        // The wide HC residual starts as four identical copies of the token embedding.
        // ggml dequantises that row into F32 and never rounds it; the F16 gather
        // would put a ~5e-4 error into the seed of every position.
        const int hc_dim = q.hidden * q.hc_count;
        const bool embed_f32 = [] {
            const char * v = std::getenv("WARODECODE_EMBED_F32");
            return v == nullptr || v[0] == '\0' || v[0] != '0';
        }();
        profile_.mark_begin(kQwen38StageEmbed, stream);
        if (embed_f32 && state_.head_hidden_f32() != nullptr) {
            launch_q8_0_gather_row_f32(rt_.token_embd, token, state_.head_hidden_f32(), stream);
            q38_repeat_embedding_from_f32_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                state_.head_hidden_f32(), state_.residual(), q.hidden, q.hc_count);
            q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                state_.head_hidden_f32(), state_.head_hidden(), q.hidden);
        } else {
            launch_q8_0_gather_row(rt_.token_embd, token, state_.head_hidden(), stream);
            q38_repeat_embedding_f32_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                state_.head_hidden(), state_.residual(), q.hidden, q.hc_count);
        }
        profile_.mark_end(kQwen38StageEmbed, stream);
        if (legacy_trace) {
            if (trace->state_trace) trace_context_ = "position " + std::to_string(pos) + " F32 layer-0 parity";
            trace->hc_init = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));
            trace->add_extended("layer0.hc_init", trace->hc_init);
        }
        if (legacy_trace) {
            const Qwen38HcWeights & hc_attn = rt_.layers.front().hc_attn;
            run_qwen38_hc_f32_debug(rt_.token_embd, token, hc_attn, hc_f32_scratch, q.rms_eps, stream);
            trace->hc_f32.init = copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim));
            trace->hc_f32.norm = copy_float(stream, hc_f32_scratch.norm(), static_cast<size_t>(hc_dim));
            trace->hc_f32.low_raw = copy_float(stream, hc_f32_scratch.low_raw(), static_cast<size_t>(q.hc_low_rank));
            trace->hc_f32.low = copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(q.hc_low_rank));
            trace->hc_f32.gate = copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim));
            trace->hc_f32.mixed = copy_float(stream, hc_f32_scratch.mixed(), static_cast<size_t>(q.hidden));
            trace->add_extended("layer0.hc_init_f32", trace->hc_f32.init);
            trace->add_extended("layer0.hc_attn_norm_f32", trace->hc_f32.norm);
            trace->add_extended("layer0.hc_attn_low_f32", trace->hc_f32.low);
            trace->add_extended("layer0.hc_attn_gate_f32", trace->hc_f32.gate);
            trace->add_extended("layer0.hc_attn_mixed_f32", trace->hc_f32.mixed);
            if (trace->use_f32_hc_layer0) {
                Qwen38RuntimeLayer & layer0 = rt_.layers.front();
                Qwen38LayerState & state0 = state_.layer(0);
                if (layer0.qsa || !layer0.gdn_w || !state0.gdn_state) {
                    throw std::runtime_error("layer-0 F32 parity path requires GDN attention");
                }
                Qwen38GdnF32DebugScratch gdn_scratch;
                gdn_scratch.initialize(q);
                float * attn_out = run_qwen38_gdn_f32_first_token(
                    hc_f32_scratch.mixed(), *layer0.gdn_w, gdn_scratch, q, q.rms_eps,
                    hc_f32_scratch.q8_1(), stream);
                trace->gdn_f32_out = copy_float(stream, attn_out, static_cast<size_t>(q.hidden));

                const int combine_blocks = (hc_dim + 255) / 256;
                q38_hc_combine_f32_kernel<<<combine_blocks, 256, 0, stream>>>(
                    hc_f32_scratch.residual(), attn_out, hc_f32_scratch.inject(),
                    q.hidden, q.hc_count);
                trace->after_attn_f32 = copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim));

                Qwen38HcF32DebugScratch ffn_scratch;
                ffn_scratch.initialize(q.hidden, q.hc_count, q.hc_low_rank);
                run_qwen38_hc_f32_mix_from_residual(
                    hc_f32_scratch.residual(), layer0.hc_ffn, ffn_scratch, q.rms_eps, stream);
                trace->hc_ffn_mix_f32 = copy_float(stream, ffn_scratch.mixed(), static_cast<size_t>(q.hidden));
                trace->add_extended("layer0.hc_mixed_f32", trace->hc_ffn_mix_f32);
                q38_select_experts_f32_input(
                    layer0.router, ffn_scratch.mixed(), state0.moe,
                    q.expert_weights_scale, stream);
                f32_layer0_router_preselected = true;
            }
        }

        for (int il = 0; il < q.trunk_layers; ++il) {
            Qwen38RuntimeLayer & layer = rt_.layers[static_cast<size_t>(il)];
            Qwen38LayerState & st = state_.layer(il);
            profile_.layer_begin(il, stream);
            if (trace && trace->state_trace) {
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " input residual";
            }

            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_layer_input_residual_layer &&
                !trace->debug_layer_input_residual_f32.empty()) {
                if (trace->debug_layer_input_residual_f32.size() != static_cast<size_t>(hc_dim)) {
                    throw std::runtime_error("layer input residual override is invalid");
                }
                const std::string input_name = "layer" + std::to_string(il) + ".input_residual";
                trace->add_extended(input_name + "_before_cpu_override",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.residual(), trace->debug_layer_input_residual_f32.data(),
                    static_cast<size_t>(hc_dim) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy layer input residual override failed");
                q38_float_copy_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.residual(), state_.residual(), hc_dim);
                ++trace->debug_layer_input_residual_override_count;
                trace->add_extended(input_name + "_cpu_f32", trace->debug_layer_input_residual_f32);
                trace->add_extended(input_name + "_after_cpu_override",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }

            if (trace && trace->state_trace) {
                trace->add_extended(
                    "layer" + std::to_string(il) + ".input_residual",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }

            if (q.is_ple(il)) {
                // Covers the synchronous host-side row read (pread) and the H2D of
                // the PLE rows: the device number shows the GPU waiting for it.
                Qwen38StageScope ple_scope(&profile_, kQwen38StagePle, stream);
                if (!rt_.ple_table || !layer.ple_w || !st.ple_state || !st.ple_scratch) {
                    throw std::runtime_error("Qwen3.8 PLE runtime state is incomplete");
                }
                if (trace && trace->extended_parity && !trace->state_trace && il == 1) {
                    trace->add_extended("layer1.ple_pre_residual", copy_half(
                        stream, state_.residual(), static_cast<size_t>(hc_dim)));
                }
                if (trace && trace->state_trace) {
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " PLE before";
                    const std::vector<uint64_t> rows = qwen38_ple_row_indices(q, token, history_);
                    trace->add_integer_vector("layer" + std::to_string(il) + ".ple_row_indices",
                        std::vector<int64_t>(rows.begin(), rows.end()));
                    trace->add_integer_vector("layer" + std::to_string(il) + ".ple_token_history_before",
                        std::vector<int64_t>(history_.begin(), history_.end()));
                    const size_t n = static_cast<size_t>(st.ple_state->hist()) * st.ple_state->channels();
                    trace->add_extended("layer" + std::to_string(il) + ".ple_history_before",
                        copy_half(stream, st.ple_state->history(), n));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_pre_residual",
                        copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                }
                run_qwen38_ple(
                    state_.residual(), token, history_, pos,
                    *layer.ple_w, *rt_.ple_table,
                    *st.ple_state, *st.ple_scratch,
                    q, stream, !q38_residual_f32_enabled());
                if (trace && trace->extended_parity && !trace->state_trace && il == 1) {
                    trace->add_extended("layer1.ple_post_residual", copy_half(
                        stream, state_.residual(), static_cast<size_t>(hc_dim)));
                }
                if (trace && trace->state_trace) {
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " PLE after";
                    const size_t n = static_cast<size_t>(st.ple_state->hist()) * st.ple_state->channels();
                    trace->add_extended("layer" + std::to_string(il) + ".ple_history_after",
                        copy_half(stream, st.ple_state->history(), n));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_post_residual",
                        copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_conv_out",
                        copy_half(stream, st.ple_scratch->conv_out(), static_cast<size_t>(hc_dim)));
                    // F32 twins: the half mirrors above hide everything below 5e-4,
                    // which is exactly the range the PLE divergence lives in.
                    trace->add_extended("layer" + std::to_string(il) + ".ple_gated_value_f32",
                        copy_float(stream, st.ple_scratch->gated_f32(), static_cast<size_t>(hc_dim)));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_conv_out_f32",
                        copy_float(stream, st.ple_scratch->conv_f32(), static_cast<size_t>(hc_dim)));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_post_residual_f32",
                        copy_float(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_query_norm_f32",
                        copy_float(stream, st.ple_scratch->query_norm_f32(), static_cast<size_t>(hc_dim)));
                    trace->add_extended("layer" + std::to_string(il) + ".ple_gate",
                        copy_float(stream, st.ple_scratch->gate(), static_cast<size_t>(q.hc_count)));
                    std::vector<int64_t> history_after(history_.begin(), history_.end());
                    history_after.push_back(token);
                    trace->add_integer_vector("layer" + std::to_string(il) + ".ple_token_history_after",
                        std::move(history_after));
                }
            }
            if (trace && trace->state_trace) {
                trace->add_extended(
                    "layer" + std::to_string(il) + ".residual_after_ple",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }

            const bool debug_f32_hc_attn =
                (debug_f32_hc_attn_0_8 && il <= 8) ||
                (debug_hc_q8_1_attn_through >= 0 &&
                 il >= debug_hc_q8_1_attn_from && il <= debug_hc_q8_1_attn_through);
            const bool trace_hc_attn_projection = trace && trace->state_trace &&
                (il == 0 || il == 2) && !debug_f32_hc_attn;
            const bool trace_layer0_hc_attn = trace_hc_attn_projection && il == 0;
            const bool debug_hc_norm_override = trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                trace->debug_hc_norm_f32_layer == il && !trace->debug_hc_norm_f32.empty();
            const bool debug_layer0_hc_gate_f32_store = trace_layer0_hc_attn &&
                trace->debug_layer0_hc_gate_f32_store;
            const bool debug_layer0_hc_gate_f32_sigmoid = trace_layer0_hc_attn &&
                trace->debug_layer0_hc_gate_f32_sigmoid;
            const bool debug_hc_gate_q8_0 = trace && trace->state_trace &&
                trace->debug_hc_gate_q8_0_layer == il;
            const bool debug_hc_full_q8_0 = trace && trace->state_trace &&
                trace->debug_hc_full_q8_0_layer == il;
            const bool debug_layer0_hc_gate_replay = debug_layer0_hc_gate_f32_store ||
                debug_layer0_hc_gate_f32_sigmoid ||
                (debug_hc_gate_q8_0 && il == 0) ||
                (debug_hc_full_q8_0 && il == 0);
            const float * hc_norm_override_f32 = nullptr;
            if (debug_hc_norm_override) {
                if (trace->debug_hc_norm_f32.size() != static_cast<size_t>(hc_dim)) {
                    throw std::runtime_error("HC norm override has the wrong size");
                }
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.norm(), trace->debug_hc_norm_f32.data(),
                    static_cast<size_t>(hc_dim) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy HC norm override failed");
                hc_norm_override_f32 = hc_f32_scratch.norm();
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_norm_override_f32",
                    trace->debug_hc_norm_f32);
                ++trace->debug_hc_norm_override_count;
            }
            profile_.mark_begin(kQwen38StageHcMixAttn, stream);
            Q38HcKernelScope hc_attn_region_scope(
                kQ38HcSiteAttnF16Norm, kQ38HcMixRegion, stream);
            // attn_f32_mix_active has to be known *before* the first mix call: when it
            // is set, run_qwen38_hc_mix_f32_with_inject() below recomputes the very
            // same row into the same scratch, so running the plain F32 mix first only
            // buys duplicated work. Task F measured both HC mix sites at ~40 ms/token
            // on A100 (272 ms total), i.e. ~15% of the token was being spent twice.
            const bool attn_f32_mix_active = q38_hc_q8_0_enabled() && q38_hc_q8_0_attn_enabled() &&
                q38_hc_q8_0_compatible(layer.hc_attn) &&
                qwen38_hc_f32_inject_enabled(layer.hc_attn) &&
                !(trace && trace->state_trace && trace->debug_hc_full_q8_0_layer == il);
            __half * cur = nullptr;
            bool attn_f32_inject = false;
            if (!attn_f32_mix_active && q38_hc_f32_enabled()) {
                cur = run_qwen38_hc_mix_f32(
                    state_.residual(), layer.hc_attn, hc_f32_scratch, st.hc_attn,
                    q.rms_eps, stream, kQ38HcSiteAttnF16Norm);
            }
            if (cur == nullptr) {
                // The legacy F16 path still runs first: it fills the F16 inject
                // weights that run_qwen38_hc_combine consumes.
                cur = run_qwen38_hc_mix(
                    state_.residual(), layer.hc_attn, st.hc_attn, q.rms_eps, stream,
                    trace_hc_attn_projection ? hc_f32_scratch.low_raw() : nullptr,
                    trace_hc_attn_projection ? hc_f32_scratch.low() : nullptr,
                    trace_hc_attn_projection ? hc_f32_scratch.gate() : nullptr,
                    hc_norm_override_f32, kQ38HcSiteAttnF16Norm);
            }
            // The unrounded HC mix row is valid whenever one of the exact Q8_0 mix
            // entry points ran (with or without the F32 inject); the debug replay
            // branches recompute the mix through half kernels and invalidate it.
            const bool attn_f32_row_valid = attn_f32_mix_active &&
                !debug_hc_gate_q8_0 && !debug_layer0_hc_gate_replay;
            if (attn_f32_mix_active) {
                __half * inject_cur = run_qwen38_hc_mix_f32_with_inject(
                    state_.residual(), layer.hc_attn, hc_f32_scratch, st.hc_attn, q.rms_eps, stream,
                    kQ38HcSiteAttnF16Norm);
                if (inject_cur != nullptr) {
                    // F32-inject variant: the unrounded inject stays in the F32
                    // scratch and run_qwen38_hc_combine_f32_inject consumes it.
                    cur = inject_cur;
                    attn_f32_inject = true;
                } else {
                    run_qwen38_hc_q8_0_f32_mix_from_residual(
                        state_.residual(), layer.hc_attn, hc_f32_scratch, q.rms_eps, stream,
                        kQ38HcSiteAttnF16Norm);
                    q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                        hc_f32_scratch.mixed(), st.hc_attn.mixed(), q.hidden);
                    cur = st.hc_attn.mixed();
                }
            }
            if (trace_hc_attn_projection && il == 2) {
                const std::string layer = "layer" + std::to_string(il) + ".hc_attn_";
                trace->add_extended(layer + "down_projection_f32",
                    copy_float(stream, hc_f32_scratch.low_raw(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended(layer + "down_preactivation_f16",
                    copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended(layer + "low_post_activation",
                    copy_half(stream, st.hc_attn.lo(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended(layer + "gate_projection_direct_f32",
                    copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                trace->add_extended(layer + "gate_preactivation_f16",
                    copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim)));
                q38_hc_sigmoid_gate_f32_trace_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    st.hc_attn.gate(), hc_f32_scratch.residual(), hc_dim);
                trace->add_extended(layer + "gate_post_sigmoid_f16_input",
                    copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim)));
                q38_hc_sigmoid_f32_gate_trace_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.gate(), hc_f32_scratch.residual(), hc_dim);
                trace->add_extended(layer + "gate_post_sigmoid_direct_f32",
                    copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim)));
            }
            if (debug_hc_full_q8_0 && trace_layer0_hc_attn) {
                trace->add_extended("layer0.hc_attn_down_direct_f32",
                    copy_float(stream, hc_f32_scratch.low_raw(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended("layer0.hc_attn_down_direct_preactivation_f16",
                    copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended("layer0.hc_attn_low_direct_post_activation_f16",
                    copy_half(stream, st.hc_attn.lo(), static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended("layer0.hc_attn_gate_direct_f32",
                    copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer0.hc_attn_gate_direct_preactivation_f16",
                    copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim)));
            }
            if (debug_hc_full_q8_0) {
                if (!layer.hc_attn.down.lowbit || layer.hc_attn.down.type != GgmlType::Q8_0 ||
                    !layer.hc_attn.up.lowbit || layer.hc_attn.up.type != GgmlType::Q8_0) {
                    throw std::runtime_error("full HC Q8_0 replay requires Q8_0 down/up weights");
                }
                constexpr int q8_threads = 256;
                const int low_rank = st.hc_attn.low_rank();
                const int warps = q8_threads / 32;
                const int norm_blocks = hc_dim / 32;
                if (hc_norm_override_f32) {
                    q38_hc_q8_0_quantize_f32_kernel<<<(norm_blocks + 7) / 8, q8_threads, 0, stream>>>(
                        hc_norm_override_f32, hc_dim, hc_f32_scratch.q8_0_values(),
                        hc_f32_scratch.q8_0_scales());
                } else {
                    q38_hc_q8_0_quantize_half_kernel<<<(norm_blocks + 7) / 8, q8_threads, 0, stream>>>(
                        st.hc_attn.xn(), hc_dim, hc_f32_scratch.q8_0_values(),
                        hc_f32_scratch.q8_0_scales());
                }
                q38_hc_q8_0_q8_0_gemv_f32_kernel<<<(low_rank + warps - 1) / warps, q8_threads, 0, stream>>>(
                    layer.hc_attn.down.low.data, hc_f32_scratch.q8_0_values(),
                    hc_f32_scratch.q8_0_scales(), hc_f32_scratch.embedding(), low_rank, hc_dim);
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.low(), hc_f32_scratch.embedding(),
                    static_cast<size_t>(low_rank) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy full Q8_0 HC down output failed");
                q38_hc_silu_scale_f32_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.low(), low_rank, 1.0f / static_cast<float>(q.hc_count));
                const int low_blocks = low_rank / 32;
                q38_hc_q8_0_quantize_f32_kernel<<<(low_blocks + 7) / 8, q8_threads, 0, stream>>>(
                    hc_f32_scratch.low(), low_rank, hc_f32_scratch.q8_0_values(),
                    hc_f32_scratch.q8_0_scales());
                q38_hc_q8_0_q8_0_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps, q8_threads, 0, stream>>>(
                    layer.hc_attn.up.low.data, hc_f32_scratch.q8_0_values(),
                    hc_f32_scratch.q8_0_scales(), hc_f32_scratch.gate(), hc_dim, low_rank);
                q38_hc_copy_f32_to_half_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.low(), st.hc_attn.lo(), low_rank);
                q38_hc_copy_f32_to_half_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.gate(), st.hc_attn.gate(), hc_dim);
                if (hc_norm_override_f32) {
                    q38_hc_mix_f32_x_f32_gate_replay_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                        hc_norm_override_f32, hc_f32_scratch.gate(), st.hc_attn.mixed(),
                        q.hidden, q.hc_count);
                } else {
                    q38_hc_mix_f32_gate_replay_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                        st.hc_attn.xn(), hc_f32_scratch.gate(), st.hc_attn.mixed(),
                        q.hidden, q.hc_count, true);
                }
                if (trace && trace->state_trace && il != 0) {
                    const std::string layer = "layer" + std::to_string(il) + ".hc_attn_";
                    trace->add_extended(layer + "down_full_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.embedding(), static_cast<size_t>(low_rank)));
                    trace->add_extended(layer + "low_full_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(low_rank)));
                    trace->add_extended(layer + "gate_full_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                }
                cur = st.hc_attn.mixed();
            }
            if (debug_hc_gate_q8_0) {
                if (!layer.hc_attn.up.lowbit || layer.hc_attn.up.type != GgmlType::Q8_0) {
                    throw std::runtime_error("HC Q8_0 gate replay requires Q8_0 up weights");
                }
                constexpr int q8_threads = 256;
                const int low_rank = st.hc_attn.low_rank();
                const int input_blocks = low_rank / 32;
                q38_hc_q8_0_quantize_half_kernel<<<(input_blocks + 7) / 8, q8_threads, 0, stream>>>(
                    st.hc_attn.lo(), low_rank, hc_f32_scratch.q8_0_values(),
                    hc_f32_scratch.q8_0_scales());
                const int warps = q8_threads / 32;
                q38_hc_q8_0_q8_0_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps, q8_threads, 0, stream>>>(
                    layer.hc_attn.up.low.data, hc_f32_scratch.q8_0_values(),
                    hc_f32_scratch.q8_0_scales(), hc_f32_scratch.gate(), hc_dim, low_rank);
                q38_hc_copy_f32_to_half_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.gate(), st.hc_attn.gate(), hc_dim);
                q38_hc_mix_f32_gate_replay_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    st.hc_attn.xn(), hc_f32_scratch.gate(), st.hc_attn.mixed(),
                    q.hidden, q.hc_count, true);
                if (trace && trace->state_trace && il != 0) {
                    trace->add_extended("layer" + std::to_string(il) + ".hc_attn_gate_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                }
                cur = st.hc_attn.mixed();
            } else if (debug_layer0_hc_gate_f32_store || debug_layer0_hc_gate_f32_sigmoid) {
                q38_hc_copy_f32_to_half_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.gate(), st.hc_attn.gate(), hc_dim);
                if (debug_layer0_hc_gate_f32_store) {
                    q38_hc_mix_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                        st.hc_attn.xn(), st.hc_attn.gate(), st.hc_attn.mixed(),
                        q.hidden, q.hc_count);
                } else {
                    q38_hc_mix_f32_gate_replay_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                        st.hc_attn.xn(), hc_f32_scratch.gate(), st.hc_attn.mixed(),
                        q.hidden, q.hc_count, false);
                }
                cur = st.hc_attn.mixed();
            }
            if (trace_layer0_hc_attn) {
                q38_hc_sigmoid_gate_f32_trace_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    st.hc_attn.gate(), hc_f32_scratch.residual(), hc_dim);
                trace->add_extended("layer0.hc_attn_input",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer0.hc_attn_down_projection_f32",
                    copy_float(stream, hc_f32_scratch.low_raw(),
                        static_cast<size_t>(st.hc_attn.low_rank())));
                if (debug_hc_full_q8_0) {
                    trace->add_extended("layer0.hc_attn_down_full_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.embedding(), static_cast<size_t>(st.hc_attn.low_rank())));
                    trace->add_extended("layer0.hc_attn_low_full_q8_0_f32",
                        copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(st.hc_attn.low_rank())));
                } else {
                    trace->add_extended("layer0.hc_attn_down_preactivation_f16",
                        copy_float(stream, hc_f32_scratch.low(),
                            static_cast<size_t>(st.hc_attn.low_rank())));
                }
                trace->add_extended("layer0.hc_attn_low_post_activation",
                    copy_half(stream, st.hc_attn.lo(), static_cast<size_t>(st.hc_attn.low_rank())));
                const char * gate_trace_name = debug_hc_full_q8_0
                    ? "layer0.hc_attn_gate_full_q8_0_f32"
                    : (debug_hc_gate_q8_0 ? "layer0.hc_attn_gate_q8_0_f32"
                                          : "layer0.hc_attn_gate_projection_f32");
                trace->add_extended(gate_trace_name,
                    copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                if (debug_layer0_hc_gate_replay && !debug_hc_full_q8_0) {
                    trace->add_extended("layer0.hc_attn_gate_replay_f32",
                        copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                }
                trace->add_extended("layer0.hc_attn_gate_preactivation_f16",
                    copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer0.hc_attn_gate_post_sigmoid_f32",
                    copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim)));
                q38_hc_sigmoid_f32_gate_trace_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.gate(), hc_f32_scratch.residual(), hc_dim);
                const char * sigmoid_trace_name = debug_hc_full_q8_0
                    ? "layer0.hc_attn_gate_post_sigmoid_full_q8_0"
                    : (debug_hc_gate_q8_0 ? "layer0.hc_attn_gate_post_sigmoid_q8_0"
                                          : "layer0.hc_attn_gate_post_sigmoid_f32_projection");
                trace->add_extended(sigmoid_trace_name,
                    copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim)));
            }
            if (debug_f32_hc_attn) {
                if (layer.hc_attn.debug_norm_f32.empty()) {
                    throw std::runtime_error("F32 HC attention replay weights unavailable for requested layer");
                }
                run_qwen38_hc_f32_mix_from_residual(
                    state_.residual(), layer.hc_attn, hc_f32_scratch, q.rms_eps, stream);
                const int blocks = (q.hidden + 255) / 256;
                q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_attn.mixed(), q.hidden);
                cur = st.hc_attn.mixed();
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_attention_input_f32_layer &&
                !trace->debug_attention_input_f32.empty()) {
                if (trace->debug_attention_input_f32.size() != static_cast<size_t>(q.hidden)) {
                    throw std::runtime_error("attention input override is invalid");
                }
                const std::string input_name = "layer" + std::to_string(il) + ".attention_input";
                trace->add_extended(input_name + "_before_cpu_override",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.mixed(), trace->debug_attention_input_f32.data(),
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy attention input override failed");
                q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_attn.mixed(), q.hidden);
                cur = st.hc_attn.mixed();
                ++trace->debug_attention_input_override_count;
                trace->add_extended(input_name + "_cpu_f32", trace->debug_attention_input_f32);
                trace->add_extended(input_name + "_after_cpu_override",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
            }
            if (trace && trace->state_trace) {
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_attn_norm";
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_norm",
                    debug_f32_hc_attn
                        ? copy_float(stream, hc_f32_scratch.norm(), static_cast<size_t>(hc_dim))
                        : copy_half(stream, st.hc_attn.xn(), static_cast<size_t>(hc_dim)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_attn_gate";
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_gate",
                    debug_f32_hc_attn
                        ? copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim))
                        : copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_attn_mix";
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_mix",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                if (attn_f32_row_valid) {
                    trace->add_extended("layer" + std::to_string(il) + ".hc_attn_mix_f32",
                        copy_float(stream, hc_f32_scratch.mixed(), static_cast<size_t>(q.hidden)));
                }
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_attn_inject";
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_inject",
                    copy_half(stream, st.hc_attn.inject(), static_cast<size_t>(q.hc_count)));
            }
            if (legacy_trace && il == 0) {
                trace->hc_attn_norm = copy_half(stream, st.hc_attn.xn(), static_cast<size_t>(hc_dim));
                trace->hc_attn_low = copy_half(
                    stream, st.hc_attn.lo(), static_cast<size_t>(st.hc_attn.low_rank()));
                trace->hc_attn_gate = copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim));
                trace->hc_attn_mix = copy_half(stream, cur, static_cast<size_t>(q.hidden));
            }
            if (trace && trace->extended_parity && il == 1) {
                if (!q.is_ple(il) && !trace->state_trace) {
                    const std::vector<float> unchanged = copy_half(
                        stream, state_.residual(), static_cast<size_t>(hc_dim));
                    trace->add_extended("layer1.ple_pre_residual", unchanged);
                    trace->add_extended("layer1.ple_post_residual", unchanged);
                }
            }
            if (legacy_trace && trace->use_f32_hc_layer0 && il == 0) {
                const int blocks = (q.hidden + 255) / 256;
                q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_attn.mixed(), q.hidden);
                cur = st.hc_attn.mixed();
                trace->hc_attn_mix_used = copy_half(stream, cur, static_cast<size_t>(q.hidden));
            }
            if (trace && trace->extended_parity && il == 0) {
                trace->add_extended("layer0.hc_attn_mixed", trace->hc_attn_mix);
            }
            if (trace && trace->extended_parity && il == 1) {
                trace->add_extended("layer1.hc_attn_norm", copy_half(
                    stream, st.hc_attn.xn(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer1.hc_attn_gate", copy_half(
                    stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer1.hc_attn_mix", copy_half(
                    stream, cur, static_cast<size_t>(q.hidden)));
                trace->add_extended(
                    layer.qsa ? "layer1.qsa_input" : "layer1.gdn_input",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
            }

            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_context_f32_layer &&
                !trace->debug_qsa_context_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA attention context override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_pregate_f32_layer &&
                !trace->debug_qsa_pregate_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA pregate override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_query_f32_layer &&
                !trace->debug_qsa_query_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA query override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_key_store_f32_layer &&
                !trace->debug_qsa_key_store_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA key store override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_value_store_f32_layer &&
                !trace->debug_qsa_value_store_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA value store override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_key_cache_prefix_f32_layer &&
                !trace->debug_qsa_key_cache_prefix_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA key cache prefix override targeted a non-QSA layer");
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_qsa_value_cache_prefix_f32_layer &&
                !trace->debug_qsa_value_cache_prefix_f32.empty() && !layer.qsa) {
                throw std::runtime_error("QSA value cache prefix override targeted a non-QSA layer");
            }

            // Diagnostic twin of layer*.hc_attn_mix_f32 captured *before* the
            // attention runs: a difference between the two proves the F32 mix
            // scratch is being reused by a later stage (and would invalidate any
            // mix error measured from the late item).
            if (trace && trace->state_trace && attn_f32_row_valid) {
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_mix_f32_early",
                    copy_float(stream, hc_f32_scratch.mixed(), static_cast<size_t>(q.hidden)));
                // F32 internals of the exact HC mix: the half mirrors these feed
                // sit at ~5e-4, so a 0.2% mix error cannot be attributed without
                // them. hc_norm is the grouped RMSNorm row, hc_gate is the up
                // projection whose sigmoid weights the mix.
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_norm_f32",
                    copy_float(stream, hc_f32_scratch.norm(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_gate_f32",
                    copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim)));
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_low_raw_f32",
                    copy_float(stream, hc_f32_scratch.low_raw(),
                        static_cast<size_t>(st.hc_attn.low_rank())));
                trace->add_extended("layer" + std::to_string(il) + ".hc_attn_low_f32",
                    copy_float(stream, hc_f32_scratch.low(),
                        static_cast<size_t>(st.hc_attn.low_rank())));
            }
            profile_.mark_end(kQwen38StageHcMixAttn, stream);

            profile_.mark_begin(kQwen38StageAttn, stream);
            __half * attn_out = nullptr;
            float * attn_out_f32 = nullptr;
            if (layer.qsa) {
                if (!layer.qsa_w || !st.qsa_state || !st.qsa_scratch) {
                    throw std::runtime_error("Qwen3.8 QSA state is incomplete");
                }
                // NOTE: run_qwen38_qsa_f32_x() (F32 activation input) is available
                // but measurably *worse* here: it moved the position-39 margin from
                // -0.245 to -0.742. The half round trip apparently compensates for a
                // larger upstream error, so the half input stays the default.
                attn_out = (attn_f32_row_valid && q38_qsa_f32_out_enabled())
                    ? run_qwen38_qsa_f32_x(
                          hc_f32_scratch.mixed(), *layer.qsa_w, *st.qsa_state, *st.qsa_scratch,
                          q, il, pos, Qwen38Position4::text(pos), stream,
                          trace && trace->state_trace &&
                              (il == 7 || (trace->captures_state_position() &&
                               pos == trace->state_trace_position &&
                               il == trace->debug_qsa_pregate_f32_layer &&
                               !trace->debug_qsa_pregate_f32.empty()))
                              ? st.qsa_scratch->pre_gate_trace() : nullptr,
                          trace && trace->state_trace && trace->debug_qsa_q8_0_layer == il,
                          trace && trace->state_trace && trace->debug_qsa_v_q8_0_layer == il)
                    : run_qwen38_qsa(
                    cur, *layer.qsa_w, *st.qsa_state, *st.qsa_scratch,
                    q, il, pos, Qwen38Position4::text(pos), stream,
                    trace && trace->state_trace &&
                        (il == 7 || (trace->captures_state_position() &&
                         pos == trace->state_trace_position &&
                         il == trace->debug_qsa_pregate_f32_layer &&
                         !trace->debug_qsa_pregate_f32.empty()))
                        ? st.qsa_scratch->pre_gate_trace() : nullptr,
                    trace && trace->state_trace && trace->debug_qsa_q8_0_layer == il,
                    trace && trace->state_trace && trace->debug_qsa_v_q8_0_layer == il);
                if (trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_query_f32_layer &&
                    !trace->debug_qsa_query_f32.empty()) {
                    const int qsa_elements = q.heads * q.key_length;
                    if (trace->debug_qsa_query_f32.size() !=
                        static_cast<size_t>(qsa_elements)) {
                        throw std::runtime_error("QSA query override has the wrong size");
                    }
                    const std::string query_name = "layer" + std::to_string(il) + ".qsa.query";
                    trace->add_extended(query_name + "_before_cpu_override",
                        copy_half(stream, st.qsa_scratch->q(),
                            static_cast<size_t>(qsa_elements)));
                    std::vector<__half> query_half(trace->debug_qsa_query_f32.size());
                    for (size_t i = 0; i < query_half.size(); ++i) {
                        query_half[i] = __float2half(trace->debug_qsa_query_f32[i]);
                    }
                    cudaError_t debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync before QSA query override failed");
                    }
                    debug_error = cudaMemcpyAsync(
                        st.qsa_scratch->q(), query_half.data(),
                        query_half.size() * sizeof(__half), cudaMemcpyHostToDevice, stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("copy QSA query override failed");
                    }
                    debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync QSA query override failed");
                    }
                    const int ratio = st.qsa_state->ratio();
                    const int full_blocks = (pos + 1) / ratio;
                    const int tail_start = full_blocks * ratio;
                    const int block_budget = q.indexer_top_k / ratio;
                    const int selected_count = std::min(full_blocks, block_budget);
                    const int * selected_ids = full_blocks > block_budget
                        ? st.qsa_state->ids_sorted() : st.qsa_state->ids();
                    const float attention_scale = q.attention_scale != 0.0f
                        ? q.attention_scale
                        : rsqrtf(static_cast<float>(q.key_length));
                    q38_qsa_sparse_attention_kernel<<<q.heads, 256, 0, stream>>>(
                        st.qsa_scratch->q(), st.qsa_scratch->gate(),
                        st.qsa_state->k_cache(), st.qsa_state->v_cache(),
                        selected_ids, st.qsa_scratch->attn(), nullptr,
                        selected_count, ratio, tail_start, pos,
                        q.heads, q.kv_heads, q.key_length, attention_scale);
                    launch_matrix_gemv_half(
                        layer.qsa_w->out, st.qsa_scratch->attn(),
                        st.qsa_scratch->final(), stream);
                    attn_out = st.qsa_scratch->final();
                    ++trace->debug_qsa_query_override_count;
                    trace->add_extended(query_name + "_cpu_f32",
                        trace->debug_qsa_query_f32);
                }
                const bool override_qsa_key_store = trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_key_store_f32_layer &&
                    !trace->debug_qsa_key_store_f32.empty();
                const bool override_qsa_value_store = trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_value_store_f32_layer &&
                    !trace->debug_qsa_value_store_f32.empty();
                if (override_qsa_key_store || override_qsa_value_store) {
                    const size_t kv_n = static_cast<size_t>(q.kv_heads) * q.key_length;
                    const size_t expected_kv_n = kv_n;
                    if ((override_qsa_key_store &&
                         trace->debug_qsa_key_store_f32.size() != expected_kv_n) ||
                        (override_qsa_value_store &&
                         trace->debug_qsa_value_store_f32.size() != expected_kv_n)) {
                        throw std::runtime_error("QSA key/value store override has the wrong size");
                    }
                    std::vector<__half> key_store_half;
                    std::vector<__half> value_store_half;
                    if (override_qsa_key_store) {
                        key_store_half.resize(kv_n);
                        for (size_t i = 0; i < kv_n; ++i) {
                            key_store_half[i] = __float2half(trace->debug_qsa_key_store_f32[i]);
                        }
                    }
                    if (override_qsa_value_store) {
                        value_store_half.resize(kv_n);
                        for (size_t i = 0; i < kv_n; ++i) {
                            value_store_half[i] = __float2half(trace->debug_qsa_value_store_f32[i]);
                        }
                    }
                    const size_t cache_offset = static_cast<size_t>(pos) * kv_n;
                    if (override_qsa_key_store) {
                        cudaError_t debug_error = cudaMemcpyAsync(
                            st.qsa_state->k_cache() + cache_offset, key_store_half.data(),
                            kv_n * sizeof(__half), cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("copy QSA key store override failed");
                        }
                    }
                    if (override_qsa_value_store) {
                        cudaError_t debug_error = cudaMemcpyAsync(
                            st.qsa_state->v_cache() + cache_offset, value_store_half.data(),
                            kv_n * sizeof(__half), cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("copy QSA value store override failed");
                        }
                    }
                    const std::string qsa_name = "layer" + std::to_string(il) + ".qsa.";
                    if (override_qsa_key_store) {
                        trace->add_extended(qsa_name + "key_cache_position_cpu_store_f32",
                            trace->debug_qsa_key_store_f32);
                        trace->add_extended(qsa_name + "key_cache_position_after_cpu_store_override",
                            copy_half(stream, st.qsa_state->k_cache() + cache_offset, kv_n));
                        trace->add_integer_vector(qsa_name + "key_store_override_position", {pos});
                    }
                    if (override_qsa_value_store) {
                        trace->add_extended(qsa_name + "value_cache_position_cpu_store_f32",
                            trace->debug_qsa_value_store_f32);
                        trace->add_extended(qsa_name + "value_cache_position_after_cpu_store_override",
                            copy_half(stream, st.qsa_state->v_cache() + cache_offset, kv_n));
                        trace->add_integer_vector(qsa_name + "value_store_override_position", {pos});
                    }

                    const int ratio = st.qsa_state->ratio();
                    const int full_blocks = (pos + 1) / ratio;
                    const int tail_start = full_blocks * ratio;
                    const int block_budget = q.indexer_top_k / ratio;
                    const int selected_count = std::min(full_blocks, block_budget);
                    const int * selected_ids = full_blocks > block_budget
                        ? st.qsa_state->ids_sorted() : st.qsa_state->ids();
                    const float attention_scale = q.attention_scale != 0.0f
                        ? q.attention_scale
                        : rsqrtf(static_cast<float>(q.key_length));
                    q38_qsa_sparse_attention_kernel<<<q.heads, 256, 0, stream>>>(
                        st.qsa_scratch->q(), st.qsa_scratch->gate(),
                        st.qsa_state->k_cache(), st.qsa_state->v_cache(),
                        selected_ids, st.qsa_scratch->attn(), nullptr,
                        selected_count, ratio, tail_start, pos,
                        q.heads, q.kv_heads, q.key_length, attention_scale);
                    cudaError_t debug_error = cudaSuccess;
                    if (override_qsa_key_store) {
                        debug_error = cudaMemcpyAsync(
                            st.qsa_state->k_cache() + cache_offset, st.qsa_scratch->k(),
                            kv_n * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("restore Warp QSA key cache position failed");
                        }
                    }
                    if (override_qsa_value_store) {
                        debug_error = cudaMemcpyAsync(
                            st.qsa_state->v_cache() + cache_offset, st.qsa_scratch->v(),
                            kv_n * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("restore Warp QSA value cache position failed");
                        }
                    }
                    launch_matrix_gemv_half(
                        layer.qsa_w->out, st.qsa_scratch->attn(),
                        st.qsa_scratch->final(), stream);
                    attn_out = st.qsa_scratch->final();
                    if (override_qsa_key_store) ++trace->debug_qsa_key_store_override_count;
                    if (override_qsa_value_store) ++trace->debug_qsa_value_store_override_count;
                }
                const bool override_qsa_key_cache_prefix = trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_key_cache_prefix_f32_layer &&
                    !trace->debug_qsa_key_cache_prefix_f32.empty();
                const bool override_qsa_value_cache_prefix = trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_value_cache_prefix_f32_layer &&
                    !trace->debug_qsa_value_cache_prefix_f32.empty();
                if (override_qsa_key_cache_prefix || override_qsa_value_cache_prefix) {
                    const size_t kv_n = static_cast<size_t>(q.kv_heads) * q.key_length;
                    const size_t max_prefix_n = static_cast<size_t>(pos + 1) * kv_n;
                    const size_t key_prefix_n = override_qsa_key_cache_prefix
                        ? trace->debug_qsa_key_cache_prefix_f32.size() : 0;
                    const size_t value_prefix_n = override_qsa_value_cache_prefix
                        ? trace->debug_qsa_value_cache_prefix_f32.size() : 0;
                    if ((override_qsa_key_cache_prefix &&
                         (key_prefix_n == 0 || key_prefix_n > max_prefix_n || key_prefix_n % kv_n != 0)) ||
                        (override_qsa_value_cache_prefix &&
                         (value_prefix_n == 0 || value_prefix_n > max_prefix_n || value_prefix_n % kv_n != 0))) {
                        throw std::runtime_error(
                            "QSA cache prefix override must contain whole position-major cache slots not beyond the current position");
                    }
                    std::vector<__half> key_prefix_half;
                    std::vector<__half> value_prefix_half;
                    std::vector<__half> original_key_prefix;
                    std::vector<__half> original_value_prefix;
                    if (override_qsa_key_cache_prefix) {
                        key_prefix_half.resize(key_prefix_n);
                        original_key_prefix.resize(key_prefix_n);
                        for (size_t i = 0; i < key_prefix_n; ++i) {
                            key_prefix_half[i] = __float2half(
                                trace->debug_qsa_key_cache_prefix_f32[i]);
                        }
                    }
                    if (override_qsa_value_cache_prefix) {
                        value_prefix_half.resize(value_prefix_n);
                        original_value_prefix.resize(value_prefix_n);
                        for (size_t i = 0; i < value_prefix_n; ++i) {
                            value_prefix_half[i] = __float2half(
                                trace->debug_qsa_value_cache_prefix_f32[i]);
                        }
                    }
                    cudaError_t debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync before QSA cache prefix override failed");
                    }
                    const size_t key_prefix_bytes = key_prefix_n * sizeof(__half);
                    const size_t value_prefix_bytes = value_prefix_n * sizeof(__half);
                    if (override_qsa_key_cache_prefix) {
                        debug_error = cudaMemcpy(original_key_prefix.data(),
                            st.qsa_state->k_cache(), key_prefix_bytes, cudaMemcpyDeviceToHost);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("save Warp QSA key cache prefix failed");
                        }
                        debug_error = cudaMemcpyAsync(st.qsa_state->k_cache(),
                            key_prefix_half.data(), key_prefix_bytes,
                            cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("copy CPU QSA key cache prefix failed");
                        }
                    }
                    if (override_qsa_value_cache_prefix) {
                        debug_error = cudaMemcpy(original_value_prefix.data(),
                            st.qsa_state->v_cache(), value_prefix_bytes, cudaMemcpyDeviceToHost);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("save Warp QSA value cache prefix failed");
                        }
                        debug_error = cudaMemcpyAsync(st.qsa_state->v_cache(),
                            value_prefix_half.data(), value_prefix_bytes,
                            cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("copy CPU QSA value cache prefix failed");
                        }
                    }
                    debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync QSA cache prefix override failed");
                    }
                    const std::string qsa_name = "layer" + std::to_string(il) + ".qsa.";
                    if (override_qsa_key_cache_prefix) {
                        trace->add_extended(qsa_name + "key_cache_prefix_cpu_f32",
                            trace->debug_qsa_key_cache_prefix_f32);
                        trace->add_extended(qsa_name + "key_cache_prefix_after_cpu_override",
                            copy_half(stream, st.qsa_state->k_cache(), key_prefix_n));
                    }
                    if (override_qsa_value_cache_prefix) {
                        trace->add_extended(qsa_name + "value_cache_prefix_cpu_f32",
                            trace->debug_qsa_value_cache_prefix_f32);
                        trace->add_extended(qsa_name + "value_cache_prefix_after_cpu_override",
                            copy_half(stream, st.qsa_state->v_cache(), value_prefix_n));
                    }

                    const int ratio = st.qsa_state->ratio();
                    const int full_blocks = (pos + 1) / ratio;
                    const int tail_start = full_blocks * ratio;
                    const int block_budget = q.indexer_top_k / ratio;
                    const int selected_count = std::min(full_blocks, block_budget);
                    const int * selected_ids = full_blocks > block_budget
                        ? st.qsa_state->ids_sorted() : st.qsa_state->ids();
                    const float attention_scale = q.attention_scale != 0.0f
                        ? q.attention_scale
                        : rsqrtf(static_cast<float>(q.key_length));
                    q38_qsa_sparse_attention_kernel<<<q.heads, 256, 0, stream>>>(
                        st.qsa_scratch->q(), st.qsa_scratch->gate(),
                        st.qsa_state->k_cache(), st.qsa_state->v_cache(),
                        selected_ids, st.qsa_scratch->attn(), nullptr,
                        selected_count, ratio, tail_start, pos,
                        q.heads, q.kv_heads, q.key_length, attention_scale);

                    if (override_qsa_key_cache_prefix) {
                        debug_error = cudaMemcpyAsync(st.qsa_state->k_cache(),
                            original_key_prefix.data(), key_prefix_bytes,
                            cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("restore Warp QSA key cache prefix failed");
                        }
                    }
                    if (override_qsa_value_cache_prefix) {
                        debug_error = cudaMemcpyAsync(st.qsa_state->v_cache(),
                            original_value_prefix.data(), value_prefix_bytes,
                            cudaMemcpyHostToDevice, stream);
                        if (debug_error != cudaSuccess) {
                            throw std::runtime_error("restore Warp QSA value cache prefix failed");
                        }
                    }
                    if (override_qsa_key_cache_prefix) {
                        trace->add_extended(qsa_name + "key_cache_prefix_after_restore",
                            copy_half(stream, st.qsa_state->k_cache(), key_prefix_n));
                        trace->add_integer_vector(qsa_name + "key_cache_prefix_override_position",
                            {pos});
                        trace->add_integer_vector(qsa_name + "key_cache_prefix_override_length",
                            {static_cast<int64_t>(key_prefix_n)});
                        trace->add_integer_vector(qsa_name + "key_cache_prefix_override_cache_slot_range",
                            {0, static_cast<int64_t>(key_prefix_n / kv_n) - 1});
                        ++trace->debug_qsa_key_cache_prefix_override_count;
                    }
                    if (override_qsa_value_cache_prefix) {
                        trace->add_extended(qsa_name + "value_cache_prefix_after_restore",
                            copy_half(stream, st.qsa_state->v_cache(), value_prefix_n));
                        trace->add_integer_vector(qsa_name + "value_cache_prefix_override_position",
                            {pos});
                        trace->add_integer_vector(qsa_name + "value_cache_prefix_override_length",
                            {static_cast<int64_t>(value_prefix_n)});
                        trace->add_integer_vector(qsa_name + "value_cache_prefix_override_cache_slot_range",
                            {0, static_cast<int64_t>(value_prefix_n / kv_n) - 1});
                        ++trace->debug_qsa_value_cache_prefix_override_count;
                    }
                    launch_matrix_gemv_half(
                        layer.qsa_w->out, st.qsa_scratch->attn(),
                        st.qsa_scratch->final(), stream);
                    attn_out = st.qsa_scratch->final();
                }
                if (trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_context_f32_layer &&
                    !trace->debug_qsa_context_f32.empty()) {
                    if (trace->debug_qsa_context_f32.size() !=
                        static_cast<size_t>(q.heads) * q.key_length) {
                        throw std::runtime_error("QSA attention context override has the wrong size");
                    }
                    if (!st.qsa_scratch || !layer.qsa_w) {
                        throw std::runtime_error("QSA attention context override targeted an incomplete layer");
                    }
                    const std::string context_name =
                        "layer" + std::to_string(il) + ".qsa.attention_context";
                    trace->add_extended(context_name + "_before_cpu_override",
                        copy_half(stream, st.qsa_scratch->attn(),
                            static_cast<size_t>(q.heads) * q.key_length));
                    std::vector<__half> context_half(trace->debug_qsa_context_f32.size());
                    for (size_t i = 0; i < context_half.size(); ++i) {
                        context_half[i] = __float2half(trace->debug_qsa_context_f32[i]);
                    }
                    cudaError_t debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync before QSA attention context override failed");
                    }
                    debug_error = cudaMemcpyAsync(
                        st.qsa_scratch->attn(), context_half.data(),
                        context_half.size() * sizeof(__half), cudaMemcpyHostToDevice, stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("copy QSA attention context override failed");
                    }
                    debug_error = cudaStreamSynchronize(stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("sync QSA attention context override failed");
                    }
                    launch_matrix_gemv_half(
                        layer.qsa_w->out, st.qsa_scratch->attn(),
                        st.qsa_scratch->final(), stream);
                    attn_out = st.qsa_scratch->final();
                    ++trace->debug_qsa_context_override_count;
                    trace->add_extended(context_name + "_cpu_f32",
                        trace->debug_qsa_context_f32);
                    trace->add_extended(context_name + "_after_cpu_override",
                        copy_half(stream, st.qsa_scratch->attn(),
                            static_cast<size_t>(q.heads) * q.key_length));
                }
                if (trace && trace->captures_state_position() &&
                    pos == trace->state_trace_position &&
                    il == trace->debug_qsa_pregate_f32_layer &&
                    !trace->debug_qsa_pregate_f32.empty()) {
                    const int qsa_elements = q.heads * q.key_length;
                    if (trace->debug_qsa_pregate_f32.size() !=
                        static_cast<size_t>(qsa_elements)) {
                        throw std::runtime_error("QSA pregate override has the wrong size");
                    }
                    const std::string pregate_name =
                        "layer" + std::to_string(il) + ".qsa.attention_pregate";
                    trace->add_extended(pregate_name + "_before_cpu_override",
                        copy_float(stream, st.qsa_scratch->pre_gate_trace(),
                            static_cast<size_t>(qsa_elements)));
                    cudaError_t debug_error = cudaMemcpyAsync(
                        st.qsa_scratch->pre_gate_trace(), trace->debug_qsa_pregate_f32.data(),
                        static_cast<size_t>(qsa_elements) * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
                    if (debug_error != cudaSuccess) {
                        throw std::runtime_error("copy QSA pregate override failed");
                    }
                    q38_qsa_apply_gate_to_pregate_kernel<<<(qsa_elements + 255) / 256, 256, 0, stream>>>(
                        st.qsa_scratch->pre_gate_trace(), st.qsa_scratch->gate(),
                        st.qsa_scratch->attn(), qsa_elements);
                    launch_matrix_gemv_half(
                        layer.qsa_w->out, st.qsa_scratch->attn(),
                        st.qsa_scratch->final(), stream);
                    attn_out = st.qsa_scratch->final();
                    ++trace->debug_qsa_pregate_override_count;
                    trace->add_extended(pregate_name + "_cpu_f32",
                        trace->debug_qsa_pregate_f32);
                    const cudaError_t sync_error = cudaStreamSynchronize(stream);
                    if (sync_error != cudaSuccess) {
                        throw std::runtime_error("sync QSA pregate override failed");
                    }
                }
                if (q38_qsa_f32_out_enabled() && q38_qsa_f32_mid_enabled() &&
                    st.qsa_scratch->final_f32() != nullptr) {
                    // Unrounded QSA output, unless a CPU injection recomputed the
                    // final projection through the half mirror.
                    const bool qsa_override_active = trace && trace->captures_state_position() &&
                        (il == trace->debug_qsa_query_f32_layer ||
                         il == trace->debug_qsa_key_store_f32_layer ||
                         il == trace->debug_qsa_value_store_f32_layer ||
                         il == trace->debug_qsa_context_f32_layer ||
                         il == trace->debug_qsa_pregate_f32_layer ||
                         il == trace->debug_qsa_key_cache_prefix_f32_layer ||
                         il == trace->debug_qsa_value_cache_prefix_f32_layer);
                    if (!qsa_override_active) {
                        attn_out_f32 = st.qsa_scratch->final_f32();
                    }
                }
                if (trace && trace->state_trace) {
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " QSA cache/output";
                    const size_t kv_n = static_cast<size_t>(q.kv_heads) * q.key_length;
                    const size_t idx_n = static_cast<size_t>(q.indexer_key_length);
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.query",
                        copy_half(stream, st.qsa_scratch->q(), static_cast<size_t>(q.heads) * q.key_length));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.gate",
                        copy_half(stream, st.qsa_scratch->gate(), static_cast<size_t>(q.heads) * q.key_length));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.key_projection",
                        copy_half(stream, st.qsa_scratch->k(), kv_n));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.value_projection",
                        copy_half(stream, st.qsa_scratch->v(), kv_n));
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".qsa.attention_context";
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.attention_context",
                        copy_half(stream, st.qsa_scratch->attn(), static_cast<size_t>(q.heads) * q.key_length));
                    // Unrounded QSA rows: the half mirrors above are dominated by
                    // their own rounding, so the F32 twins are what an error budget
                    // against the CPU oracle has to use.
                    if (st.qsa_scratch->q_f32() != nullptr) {
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.query_f32",
                            copy_float(stream, st.qsa_scratch->q_f32(),
                                static_cast<size_t>(q.heads) * q.key_length));
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.key_f32",
                            copy_float(stream, st.qsa_scratch->k_f32(), kv_n));
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.value_f32",
                            copy_float(stream, st.qsa_scratch->v_f32(), kv_n));
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.attn_f32",
                            copy_float(stream, st.qsa_scratch->attn_f32(),
                                static_cast<size_t>(q.heads) * q.key_length));
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.gate_f32",
                            copy_float(stream, st.qsa_scratch->gate_f32(),
                                static_cast<size_t>(q.heads) * q.key_length));
                    }
                    if (il == 7) {
                        trace->add_extended("layer7.qsa.attention_pre_gate_f32",
                            copy_float(stream, st.qsa_scratch->pre_gate_trace(),
                                static_cast<size_t>(q.heads) * q.key_length));
                    }
                    const size_t cache_n = static_cast<size_t>(pos + 1) * kv_n;
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".qsa.key_cache_prefix";
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.key_cache_prefix",
                        copy_half(stream, st.qsa_state->k_cache(), cache_n));
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".qsa.value_cache_prefix";
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.value_cache_prefix",
                        copy_half(stream, st.qsa_state->v_cache(), cache_n));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.key_cache_position",
                        copy_half(stream, st.qsa_state->k_cache() + static_cast<size_t>(pos) * kv_n, kv_n));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.value_cache_position",
                        copy_half(stream, st.qsa_state->v_cache() + static_cast<size_t>(pos) * kv_n, kv_n));
                    trace->add_extended("layer" + std::to_string(il) + ".qsa.index_key_position",
                        copy_half(stream, st.qsa_state->index_raw() + static_cast<size_t>(pos) * idx_n, idx_n));
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.write_index", {pos});
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.cache_length", {pos + 1});
                    const int ratio = st.qsa_state->ratio();
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.ratio", {ratio});
                    const int full_blocks = (pos + 1) / ratio;
                    const int block_budget = q.indexer_top_k / ratio;
                    const int selected_count = std::min(full_blocks, block_budget);
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.tail_start",
                        {full_blocks * ratio});
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.used_indexer_query",
                        {full_blocks > block_budget ? 1 : 0});
                    if (full_blocks > block_budget) {
                        trace->add_extended("layer" + std::to_string(il) + ".qsa.index_query",
                            copy_half(stream, st.qsa_scratch->idx_q(),
                                static_cast<size_t>(q.indexer_heads) * q.indexer_key_length));
                    }
                    const int * selected = full_blocks > block_budget
                        ? st.qsa_state->ids_sorted() : st.qsa_state->ids();
                    const std::vector<int64_t> selected_blocks = copy_ints(
                        stream, selected, static_cast<size_t>(selected_count));
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.selected_blocks", selected_blocks);
                    std::vector<int64_t> selected_positions;
                    selected_positions.reserve(selected_blocks.size());
                    for (int64_t block : selected_blocks) selected_positions.push_back(block * ratio);
                    trace->add_integer_vector("layer" + std::to_string(il) + ".qsa.selected_positions",
                        std::move(selected_positions));
                }
            } else {
                if (!layer.gdn_w || !st.gdn_state || !st.gdn_scratch) {
                    throw std::runtime_error("Qwen3.8 GDN state is incomplete");
                }
                const size_t gdn_state_n = static_cast<size_t>(q.ssm_state_size) *
                                           q.ssm_state_size * q.ssm_dt_rank;
                const int conv_channels = q.ssm_state_size * q.ssm_group_count * 2 +
                                          q.ssm_state_size * q.ssm_dt_rank;
                const size_t conv_state_n = static_cast<size_t>(q.ssm_conv_kernel - 1) * conv_channels;
                const bool capture_full_gdn = (il <= 8 || il == 9 || il == 10 || il == 14) && !q.is_qsa(il);
                if (trace && trace->state_trace) {
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " GDN before";
                    trace->add_extended("layer" + std::to_string(il) + ".gdn.input",
                        copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.recurrent_before",
                        copy_float(stream, st.gdn_state->state(), gdn_state_n),
                        {static_cast<uint64_t>(q.ssm_dt_rank), static_cast<uint64_t>(q.ssm_state_size),
                         static_cast<uint64_t>(q.ssm_state_size)});
                    if (capture_full_gdn) {
                        trace->add_extended("layer" + std::to_string(il) + ".gdn.recurrent_before_full",
                            copy_float(stream, st.gdn_state->state(), gdn_state_n));
                    }
                    trace->add_extended("layer" + std::to_string(il) + ".gdn.conv_before",
                        copy_half(stream, st.gdn_state->conv_state(), conv_state_n));
                    trace->add_integer_vector("layer" + std::to_string(il) + ".gdn.write_position", {pos});
                }
                if (trace && trace->state_trace && il == trace->debug_layer9_gdn_state_layer &&
                    !trace->debug_layer9_gdn_state_f32.empty()) {
                    if (trace->debug_layer9_gdn_state_f32.size() != gdn_state_n) {
                        throw std::runtime_error("GDN recurrent-state override is invalid");
                    }
                    const cudaError_t e = cudaMemcpyAsync(
                        st.gdn_state->state(), trace->debug_layer9_gdn_state_f32.data(),
                        gdn_state_n * sizeof(float), cudaMemcpyHostToDevice, stream);
                    if (e != cudaSuccess) throw std::runtime_error("copy GDN recurrent-state override failed");
                    trace->add_extended("layer" + std::to_string(il) + ".gdn.recurrent_before_cpu_override",
                        copy_float(stream, st.gdn_state->state(), gdn_state_n));
                }
                if (trace && trace->state_trace && il == trace->debug_layer9_gdn_conv_state_layer &&
                    !trace->debug_layer9_gdn_conv_state_f32.empty()) {
                    if (trace->debug_layer9_gdn_conv_state_f32.size() != conv_state_n) {
                        throw std::runtime_error("GDN convolution-state override is invalid");
                    }
                    std::vector<__half> conv_state_cpu(conv_state_n);
                    std::transform(trace->debug_layer9_gdn_conv_state_f32.begin(),
                        trace->debug_layer9_gdn_conv_state_f32.end(), conv_state_cpu.begin(),
                        [] (float value) { return __float2half(value); });
                    cudaError_t e = cudaStreamSynchronize(stream);
                    if (e != cudaSuccess) throw std::runtime_error("sync before GDN convolution-state override failed");
                    e = cudaMemcpy(st.gdn_state->conv_state(), conv_state_cpu.data(),
                        conv_state_n * sizeof(__half), cudaMemcpyHostToDevice);
                    if (e != cudaSuccess) throw std::runtime_error("copy GDN convolution-state override failed");
                    const std::string gdn_state_name = "layer" + std::to_string(il) + ".gdn.";
                    trace->add_extended(gdn_state_name + "conv_state_cpu_f32",
                        trace->debug_layer9_gdn_conv_state_f32);
                    trace->add_extended(gdn_state_name + "conv_state_after_cpu_override",
                        copy_half(stream, st.gdn_state->conv_state(), conv_state_n));
                }
                if (q38_gdn_f32_out_enabled() &&
                    q38_gdn_f32_path_available(*st.gdn_state, *st.gdn_scratch, q)) {
                    // F32 GDN: the branch output stays unrounded so the HC combine
                    // does not reintroduce a half round trip. The F32 *input*
                    // variant (run_qwen38_gdn_f32_x) is available but measured
                    // worse for the same reason as the QSA one above.
                    float * gdn_f32 = (attn_f32_row_valid && q38_gdn_f32_out_enabled())
                        ? run_qwen38_gdn_f32_x(
                              hc_f32_scratch.mixed(), *layer.gdn_w, *st.gdn_state, *st.gdn_scratch,
                              q, q.rms_eps, stream)
                        : run_qwen38_gdn_f32(
                              cur, *layer.gdn_w, *st.gdn_state, *st.gdn_scratch,
                              q, q.rms_eps, stream);
                    if (gdn_f32 != nullptr) {
                        attn_out_f32 = gdn_f32;
                        attn_out = st.gdn_scratch->final_out();
                    }
                }
                if (attn_out_f32 == nullptr) {
                    attn_out = run_qwen38_gdn(
                        cur, *layer.gdn_w, *st.gdn_state, *st.gdn_scratch,
                        q, q.rms_eps, stream);
                }
                if (trace && trace->state_trace && il == 14 &&
                    !trace->debug_layer14_gdn_core_f32.empty()) {
                    const size_t value_dim = static_cast<size_t>(q.ssm_state_size) * q.ssm_dt_rank;
                    if (trace->debug_layer14_gdn_core_f32.size() != value_dim) {
                        throw std::runtime_error("layer-14 GDN core override is invalid");
                    }
                    trace->add_extended("layer14.gdn.core_output_before_cpu_override",
                        copy_half(stream, st.gdn_scratch->core_out(), value_dim));
                    trace->add_extended("layer14.gdn.norm_output_before_cpu_core_override",
                        copy_half(stream, st.gdn_scratch->norm_out(), value_dim));
                    trace->add_extended("layer14.attention_output_before_cpu_core_override",
                        copy_half(stream, attn_out, static_cast<size_t>(q.hidden)));
                    const cudaError_t e = cudaMemcpyAsync(
                        hc_f32_scratch.residual(), trace->debug_layer14_gdn_core_f32.data(),
                        value_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
                    if (e != cudaSuccess) throw std::runtime_error("copy layer-14 GDN core override failed");
                    q38_float_to_half_kernel<<<(static_cast<int>(value_dim) + 255) / 256, 256, 0, stream>>>(
                        hc_f32_scratch.residual(), st.gdn_scratch->core_out(), static_cast<int>(value_dim));
                    q38_gdn_gated_rmsnorm_kernel<<<q.ssm_dt_rank, 256, 0, stream>>>(
                        st.gdn_scratch->core_out(), layer.gdn_w->norm.data, st.gdn_scratch->z(),
                        st.gdn_scratch->norm_out(), q.ssm_state_size, q.ssm_dt_rank, q.rms_eps);
                    launch_matrix_gemv_half(
                        layer.gdn_w->out, st.gdn_scratch->norm_out(), st.gdn_scratch->final_out(), stream);
                    attn_out = st.gdn_scratch->final_out();
                    trace->add_extended("layer14.gdn.core_output_cpu_f32",
                        trace->debug_layer14_gdn_core_f32);
                    trace->add_extended("layer14.gdn.norm_output_after_cpu_core_override",
                        copy_half(stream, st.gdn_scratch->norm_out(), value_dim));
                }
                if (trace && trace->state_trace && il == 14 &&
                    !trace->debug_layer14_gdn_norm_output_f32.empty()) {
                    const size_t value_dim = static_cast<size_t>(q.ssm_state_size) * q.ssm_dt_rank;
                    if (trace->debug_layer14_gdn_norm_output_f32.size() != value_dim) {
                        throw std::runtime_error("layer-14 GDN norm output override is invalid");
                    }
                    trace->add_extended("layer14.gdn.norm_output_before_cpu_override",
                        copy_half(stream, st.gdn_scratch->norm_out(), value_dim));
                    trace->add_extended("layer14.attention_output_before_cpu_norm_override",
                        copy_half(stream, attn_out, static_cast<size_t>(q.hidden)));
                    const cudaError_t e = cudaMemcpyAsync(
                        hc_f32_scratch.residual(), trace->debug_layer14_gdn_norm_output_f32.data(),
                        value_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
                    if (e != cudaSuccess) throw std::runtime_error("copy layer-14 GDN norm output override failed");
                    q38_float_to_half_kernel<<<(static_cast<int>(value_dim) + 255) / 256, 256, 0, stream>>>(
                        hc_f32_scratch.residual(), st.gdn_scratch->norm_out(), static_cast<int>(value_dim));
                    launch_matrix_gemv_half(
                        layer.gdn_w->out, st.gdn_scratch->norm_out(), st.gdn_scratch->final_out(), stream);
                    attn_out = st.gdn_scratch->final_out();
                    trace->add_extended("layer14.gdn.norm_output_cpu_f32",
                        trace->debug_layer14_gdn_norm_output_f32);
                    trace->add_extended("layer14.gdn.norm_output_after_cpu_override",
                        copy_half(stream, st.gdn_scratch->norm_out(), value_dim));
                }
                if (trace && trace->state_trace) {
                    trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " GDN after";
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.recurrent_after",
                        copy_float(stream, st.gdn_state->state(), gdn_state_n),
                        {static_cast<uint64_t>(q.ssm_dt_rank), static_cast<uint64_t>(q.ssm_state_size),
                         static_cast<uint64_t>(q.ssm_state_size)});
                    if (capture_full_gdn) {
                        trace->add_extended("layer" + std::to_string(il) + ".gdn.recurrent_after_full",
                            copy_float(stream, st.gdn_state->state(), gdn_state_n));
                    }
                    trace->add_extended("layer" + std::to_string(il) + ".gdn.conv_after",
                        copy_half(stream, st.gdn_state->conv_state(), conv_state_n));
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.conv_output",
                        copy_half(stream, st.gdn_scratch->conv_out(), static_cast<size_t>(conv_channels)),
                        {static_cast<uint64_t>(conv_channels)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.qkv",
                        copy_half(stream, st.gdn_scratch->qkv(), static_cast<size_t>(conv_channels)),
                        {static_cast<uint64_t>(conv_channels)});
                    const size_t value_dim = static_cast<size_t>(q.ssm_state_size) * q.ssm_dt_rank;
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.z",
                        copy_half(stream, st.gdn_scratch->z(), value_dim),
                        {static_cast<uint64_t>(q.ssm_dt_rank), static_cast<uint64_t>(q.ssm_state_size)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.beta",
                        copy_half(stream, st.gdn_scratch->beta(), static_cast<size_t>(q.ssm_dt_rank)),
                        {static_cast<uint64_t>(q.ssm_dt_rank)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.alpha",
                        copy_half(stream, st.gdn_scratch->alpha(), static_cast<size_t>(q.ssm_dt_rank)),
                        {static_cast<uint64_t>(q.ssm_dt_rank)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.gate",
                        copy_float(stream, st.gdn_scratch->gate(), static_cast<size_t>(q.ssm_dt_rank)),
                        {static_cast<uint64_t>(q.ssm_dt_rank)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.core_output",
                        copy_half(stream, st.gdn_scratch->core_out(), value_dim),
                        {static_cast<uint64_t>(q.ssm_dt_rank), static_cast<uint64_t>(q.ssm_state_size)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.norm_output",
                        copy_half(stream, st.gdn_scratch->norm_out(), value_dim),
                        {static_cast<uint64_t>(q.ssm_dt_rank), static_cast<uint64_t>(q.ssm_state_size)});
                    trace->add_state_summary("layer" + std::to_string(il) + ".gdn.final_output",
                        copy_half(stream, st.gdn_scratch->final_out(), static_cast<size_t>(q.hidden)),
                        {static_cast<uint64_t>(q.hidden)});
                    if (capture_full_gdn) {
                        const std::string gdn_name = "layer" + std::to_string(il) + ".gdn.";
                        const int key_dim = q.ssm_state_size * q.ssm_group_count;
                        const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
                        const int conv_dim = key_dim * 2 + value_dim;
                        const __half * conv = st.gdn_scratch->conv_out();
                        trace->add_extended(gdn_name + "qkv_full",
                            copy_half(stream, st.gdn_scratch->qkv(), static_cast<size_t>(conv_dim)));
                        trace->add_extended(gdn_name + "conv_output_postnorm_full",
                            copy_half(stream, conv, static_cast<size_t>(conv_dim)));
                        trace->add_extended(gdn_name + "q_conv_postnorm",
                            copy_half(stream, conv, static_cast<size_t>(key_dim)));
                        trace->add_extended(gdn_name + "k_conv_postnorm",
                            copy_half(stream, conv + key_dim, static_cast<size_t>(key_dim)));
                        trace->add_extended(gdn_name + "v_conv",
                            copy_half(stream, conv + key_dim * 2, static_cast<size_t>(value_dim)));
                        trace->add_extended(gdn_name + "z_full",
                            copy_half(stream, st.gdn_scratch->z(), static_cast<size_t>(value_dim)));
                        trace->add_extended(gdn_name + "beta_full",
                            copy_half(stream, st.gdn_scratch->beta(), static_cast<size_t>(q.ssm_dt_rank)));
                        trace->add_extended(gdn_name + "alpha_full",
                            copy_half(stream, st.gdn_scratch->alpha(), static_cast<size_t>(q.ssm_dt_rank)));
                        trace->add_extended(gdn_name + "gate_full",
                            copy_float(stream, st.gdn_scratch->gate(), static_cast<size_t>(q.ssm_dt_rank)));
                        trace->add_extended(gdn_name + "core_output_full",
                            copy_half(stream, st.gdn_scratch->core_out(), static_cast<size_t>(value_dim)));
                        trace->add_extended(gdn_name + "norm_output_full",
                            copy_half(stream, st.gdn_scratch->norm_out(), static_cast<size_t>(value_dim)));
                        trace->add_extended(gdn_name + "final_output_full",
                            copy_half(stream, st.gdn_scratch->final_out(), static_cast<size_t>(q.hidden)));
                        // F32 twins of the GDN chain; the half mirrors above sit at
                        // ~5e-4, i.e. above the error an error budget must resolve.
                        if (st.gdn_scratch->qkv_f32() != nullptr) {
                            trace->add_extended(gdn_name + "qkv_f32",
                                copy_float(stream, st.gdn_scratch->qkv_f32(),
                                    static_cast<size_t>(conv_dim)));
                            trace->add_extended(gdn_name + "z_f32",
                                copy_float(stream, st.gdn_scratch->z_f32(),
                                    static_cast<size_t>(value_dim)));
                            trace->add_extended(gdn_name + "conv_out_f32",
                                copy_float(stream, st.gdn_scratch->conv_out_f32(),
                                    static_cast<size_t>(conv_dim)));
                            trace->add_extended(gdn_name + "norm_out_f32",
                                copy_float(stream, st.gdn_scratch->norm_out_f32(),
                                    static_cast<size_t>(value_dim)));
                            trace->add_extended(gdn_name + "final_out_f32",
                                copy_float(stream, st.gdn_scratch->final_out_f32(),
                                    static_cast<size_t>(q.hidden)));
                        }
                    }
                }
                if (legacy_trace && il == 0) {
                    const int key_dim = q.ssm_state_size * q.ssm_group_count;
                    const int value_dim = q.ssm_state_size * q.ssm_dt_rank;
                    const int conv_dim = key_dim * 2 + value_dim;
                    trace->gdn_qkv = copy_half(stream, st.gdn_scratch->qkv(), static_cast<size_t>(conv_dim));
                    trace->gdn_z = copy_half(stream, st.gdn_scratch->z(), static_cast<size_t>(value_dim));
                    trace->gdn_beta = copy_half(stream, st.gdn_scratch->beta(), static_cast<size_t>(q.ssm_dt_rank));
                    trace->gdn_alpha = copy_half(stream, st.gdn_scratch->alpha(), static_cast<size_t>(q.ssm_dt_rank));
                    trace->gdn_gate = copy_float(stream, st.gdn_scratch->gate(), static_cast<size_t>(q.ssm_dt_rank));
                    trace->gdn_core_out = copy_half(stream, st.gdn_scratch->core_out(), static_cast<size_t>(value_dim));
                    trace->gdn_norm_out = copy_half(stream, st.gdn_scratch->norm_out(), static_cast<size_t>(value_dim));
                }
            }
            profile_.mark_end(kQwen38StageAttn, stream);

            if (trace && trace->state_trace && pos == trace->state_trace_position &&
                il == trace->debug_attention_output_f32_layer &&
                !trace->debug_attention_output_f32.empty()) {
                if (trace->debug_attention_output_f32.size() != static_cast<size_t>(q.hidden)) {
                    throw std::runtime_error("attention output override is invalid");
                }
                const std::string layer_name = "layer" + std::to_string(il) + ".attention_output";
                trace->add_extended(layer_name + "_before_cpu_override",
                    copy_half(stream, attn_out, static_cast<size_t>(q.hidden)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.mixed(), trace->debug_attention_output_f32.data(),
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy attention output override failed");
                q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), attn_out, q.hidden);
                ++trace->debug_attention_output_override_count;
                trace->add_extended(layer_name + "_cpu_f32", trace->debug_attention_output_f32);
            }
            if (trace && trace->state_trace) {
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " attention output";
                trace->add_extended("layer" + std::to_string(il) + ".attention_output",
                    copy_half(stream, attn_out, static_cast<size_t>(q.hidden)));
            }
            if (legacy_trace && il == 0) trace->attn_out = copy_half(stream, attn_out, static_cast<size_t>(q.hidden));
            if (trace && trace->extended_parity && il == 0) {
                trace->add_extended("layer0.attn_out", trace->attn_out);
            }
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_hc_inject_f32_layer &&
                !trace->debug_hc_inject_f32.empty()) {
                if (trace->debug_hc_inject_f32.size() != static_cast<size_t>(q.hc_count)) {
                    throw std::runtime_error("HC inject override is invalid");
                }
                const std::string inject_name = "layer" + std::to_string(il) + ".hc_attn_inject";
                trace->add_extended(inject_name + "_before_cpu_override",
                    copy_half(stream, st.hc_attn.inject(), static_cast<size_t>(q.hc_count)));
                std::vector<__half> injected(static_cast<size_t>(q.hc_count));
                for (int i = 0; i < q.hc_count; ++i) {
                    injected[static_cast<size_t>(i)] =
                        __float2half(trace->debug_hc_inject_f32[static_cast<size_t>(i)]);
                }
                cudaError_t debug_error = cudaStreamSynchronize(stream);
                if (debug_error != cudaSuccess) {
                    throw std::runtime_error("sync before HC inject override failed");
                }
                debug_error = cudaMemcpy(
                    st.hc_attn.inject(), injected.data(),
                    injected.size() * sizeof(__half), cudaMemcpyHostToDevice);
                if (debug_error != cudaSuccess) throw std::runtime_error("HC inject override failed");
                ++trace->debug_hc_inject_override_count;
                trace->add_extended(inject_name + "_cpu_f32", trace->debug_hc_inject_f32);
                trace->add_extended(inject_name + "_after_cpu_override",
                    copy_half(stream, st.hc_attn.inject(), static_cast<size_t>(q.hc_count)));
            }
            profile_.mark_begin(kQwen38StageHcCombineAttn, stream);
            if (attn_out_f32 != nullptr) {
                run_qwen38_hc_combine_f32_block_f32_inject(
                    state_.residual(), attn_out_f32, hc_f32_scratch, stream,
                    !q38_residual_f32_enabled());
            } else if (attn_f32_inject) {
                run_qwen38_hc_combine_f32_inject(state_.residual(), attn_out, hc_f32_scratch, stream,
                    !q38_residual_f32_enabled());
            } else {
                run_qwen38_hc_combine(state_.residual(), attn_out, st.hc_attn, stream,
                    !q38_residual_f32_enabled());
            }
            profile_.mark_end(kQwen38StageHcCombineAttn, stream);
            if (trace && trace->captures_state_position() &&
                pos == trace->state_trace_position &&
                il == trace->debug_hc_attn_combine_layer &&
                !trace->debug_hc_attn_combine_f32.empty()) {
                if (trace->debug_hc_attn_combine_f32.size() != static_cast<size_t>(hc_dim)) {
                    throw std::runtime_error("HC attention combine override is invalid");
                }
                const std::string residual_name =
                    "layer" + std::to_string(il) + ".after_attention_residual";
                trace->add_extended(residual_name + "_before_cpu_override",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.residual(), trace->debug_hc_attn_combine_f32.data(),
                    static_cast<size_t>(hc_dim) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy HC attention combine override failed");
                q38_float_copy_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.residual(), state_.residual(), hc_dim);
                ++trace->debug_hc_attn_combine_override_count;
                trace->add_extended(residual_name + "_cpu_f32", trace->debug_hc_attn_combine_f32);
                trace->add_extended(residual_name + "_after_cpu_override",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }
            if (trace && trace->state_trace && il == 8 &&
                !trace->debug_layer8_after_attention_residual_f32.empty()) {
                if (trace->debug_layer8_after_attention_residual_f32.size() != static_cast<size_t>(hc_dim)) {
                    throw std::runtime_error("layer-8 after-attention residual override is invalid");
                }
                trace->add_extended("layer8.after_attention_residual_before_cpu_override",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.residual(), trace->debug_layer8_after_attention_residual_f32.data(),
                    static_cast<size_t>(hc_dim) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy layer-8 after-attention residual override failed");
                q38_float_copy_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.residual(), state_.residual(), hc_dim);
                trace->add_extended("layer8.after_attention_residual_cpu_f32",
                    trace->debug_layer8_after_attention_residual_f32);
            }
            if (legacy_trace && il == 0) trace->after_attn = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));
            if (trace && trace->state_trace) {
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " HC attention combine";
                trace->add_extended("layer" + std::to_string(il) + ".after_attention_residual",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }
            if (trace && trace->extended_parity && il == 0) {
                trace->add_extended("layer0.after_attn_hc_combine", trace->after_attn);
            }

            const bool debug_hc_q8_0_ffn = (debug_hc_q8_0_ffn_through >= 0 &&
                il >= debug_hc_q8_0_ffn_from && il <= debug_hc_q8_0_ffn_through) ||
                (q38_hc_q8_0_enabled() && q38_hc_q8_0_compatible(layer.hc_ffn));
            const bool debug_f32_hc_ffn = (debug_f32_hc_ffn_0_8 && il <= 8) ||
                                          (debug_f32_hc_ffn_layer8 && il == 8) ||
                                          (debug_hc_q8_1_ffn_through >= 0 &&
                                           il >= debug_hc_q8_1_ffn_from &&
                                           il <= debug_hc_q8_1_ffn_through);
            bool ffn_used_f32 = false;
            bool ffn_f32_inject = false;
            profile_.mark_begin(kQwen38StageHcMixFfn, stream);
            if (il == 0) q38_hc_report_switches_once();
            Q38HcKernelScope hc_ffn_region_scope(
                kQ38HcSiteFfnF16Norm, kQ38HcMixRegion, stream);
            if (q38_hc_f32_enabled() && !debug_hc_q8_0_ffn && !debug_f32_hc_ffn) {
                __half * f32_cur = run_qwen38_hc_mix_f32(
                    state_.residual(), layer.hc_ffn, hc_f32_scratch, st.hc_ffn,
                    q.rms_eps, stream, kQ38HcSiteFfnF16Norm);
                if (f32_cur != nullptr) {
                    cur = f32_cur;
                    ffn_used_f32 = true;
                }
            }
            // Mirrors attn_f32_mix_active: when the exact Q8_0 replay is the one
            // that will produce the mix row, the legacy F16 pass below is skipped
            // entirely (it would be recomputed into the same scratch an instant
            // later). With WARODECODE_HC_SKIP_LEGACY=0 the old order returns.
            // The exact path must also be the one that produces the [hc] inject
            // row, otherwise the fallback combine (which reads the half-typed
            // st.hc_ffn.inject()) would consume a stale buffer: the legacy pass
            // is the only other writer. Quantised inject tensors (the blk.48 MTP
            // block) and WARODECODE_HC_INJECT_F32=0 therefore keep the old order.
            const bool ffn_exact_mix_active = q38_hc_q8_0_enabled() && !debug_hc_q8_0_ffn &&
                q38_hc_q8_0_compatible(layer.hc_ffn) && q38_hc_skip_legacy_enabled() &&
                qwen38_hc_f32_inject_enabled(layer.hc_ffn);
            if (!ffn_used_f32 && !ffn_exact_mix_active) {
                cur = run_qwen38_hc_mix(
                    state_.residual(), layer.hc_ffn, st.hc_ffn, q.rms_eps, stream,
                    nullptr, nullptr, nullptr, nullptr, kQ38HcSiteFfnF16Norm);
            }
            if (debug_hc_q8_0_ffn) {
                __half * inject_cur = run_qwen38_hc_mix_f32_with_inject(
                    state_.residual(), layer.hc_ffn, hc_f32_scratch, st.hc_ffn, q.rms_eps, stream,
                    kQ38HcSiteFfnF16Norm);
                if (inject_cur != nullptr) {
                    cur = inject_cur;
                    ffn_f32_inject = true;
                } else {
                    run_qwen38_hc_q8_0_f32_mix_from_residual(
                        state_.residual(), layer.hc_ffn, hc_f32_scratch, q.rms_eps, stream,
                        kQ38HcSiteFfnF16Norm);
                    const int blocks = (q.hidden + 255) / 256;
                    q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                        hc_f32_scratch.mixed(), st.hc_ffn.mixed(), q.hidden);
                    cur = st.hc_ffn.mixed();
                }
            } else if (debug_f32_hc_ffn) {
                if (layer.hc_ffn.debug_norm_f32.empty()) {
                    throw std::runtime_error("F32 HC FFN replay weights unavailable for requested layer");
                }
                run_qwen38_hc_f32_mix_from_residual(
                    state_.residual(), layer.hc_ffn, hc_f32_scratch, q.rms_eps, stream,
                    kQ38HcSiteFfnF16Norm);
                const int blocks = (q.hidden + 255) / 256;
                q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_ffn.mixed(), q.hidden);
                cur = st.hc_ffn.mixed();
            }
            if (trace && trace->state_trace && il == 8 &&
                !trace->debug_layer8_hc_ffn_mix_f32.empty()) {
                if (trace->debug_layer8_hc_ffn_mix_f32.size() != static_cast<size_t>(q.hidden)) {
                    throw std::runtime_error("layer-8 HC FFN mix override is invalid");
                }
                trace->add_extended("layer8.hc_ffn_mix_before_cpu_override",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                const cudaError_t e = cudaMemcpyAsync(
                    hc_f32_scratch.mixed(), trace->debug_layer8_hc_ffn_mix_f32.data(),
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) throw std::runtime_error("copy layer-8 HC FFN mix override failed");
                q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_ffn.mixed(), q.hidden);
                cur = st.hc_ffn.mixed();
                trace->add_extended("layer8.hc_ffn_mix_cpu_f32", trace->debug_layer8_hc_ffn_mix_f32);
            }
            if (trace && trace->state_trace) {
                const bool trace_hc_ffn_f32 = debug_f32_hc_ffn || debug_hc_q8_0_ffn;
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_ffn_norm";
                trace->add_extended("layer" + std::to_string(il) + ".hc_ffn_norm",
                    trace_hc_ffn_f32
                        ? copy_float(stream, hc_f32_scratch.norm(), static_cast<size_t>(hc_dim))
                        : copy_half(stream, st.hc_ffn.xn(), static_cast<size_t>(hc_dim)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_ffn_gate";
                trace->add_extended("layer" + std::to_string(il) + ".hc_ffn_gate",
                    trace_hc_ffn_f32
                        ? copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim))
                        : copy_half(stream, st.hc_ffn.gate(), static_cast<size_t>(hc_dim)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_ffn_mix";
                if (ffn_f32_inject && hc_f32_scratch.hidden() == q.hidden) {
                    trace->add_extended("layer" + std::to_string(il) + ".hc_ffn_mix_f32",
                        copy_float(stream, hc_f32_scratch.mixed(), static_cast<size_t>(q.hidden)));
                }
                trace->add_extended("layer" + std::to_string(il) + ".hc_ffn_mix",
                    copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + ".hc_ffn_inject";
                trace->add_extended("layer" + std::to_string(il) + ".hc_ffn_inject",
                    copy_half(stream, st.hc_ffn.inject(), static_cast<size_t>(q.hc_count)));
            }
            if (legacy_trace && il == 0) trace->hc_ffn_mix = copy_half(stream, cur, static_cast<size_t>(q.hidden));
            profile_.mark_end(kQwen38StageHcMixFfn, stream);
            if (trace && trace->extended_parity && il == 0) {
                trace->add_extended("layer0.hc_mixed", trace->hc_ffn_mix);
            }

            // Router + expert cache staging + routed GEMV + shared expert. The
            // period covers the wait for a cache miss as well as the compute.
            profile_.mark_begin(kQwen38StageMoe, stream);
            bool experts_preselected = f32_layer0_router_preselected && il == 0;
            Qwen38Trace * debug_trace = active_trace_;
            const Qwen38MoeRouteOverride * route_override = nullptr;
            if (debug_trace && il == 8) {
                for (const auto & candidate : debug_trace->debug_moe_routes) {
                    if (candidate.position == pos) {
                        route_override = &candidate;
                        break;
                    }
                }
            }
            const bool single_position_route_override = debug_trace && debug_trace->state_trace &&
                il == 8 && pos == debug_trace->state_trace_position && !debug_trace->debug_moe_ids.empty();
            if (route_override || single_position_route_override) {
                const std::vector<int> & override_ids = route_override
                    ? route_override->expert_ids : debug_trace->debug_moe_ids;
                const std::vector<float> * override_weights = route_override
                    ? (route_override->weights.empty() ? nullptr : &route_override->weights)
                    : (debug_trace->debug_moe_weights.empty() ? nullptr : &debug_trace->debug_moe_weights);
                if (override_ids.size() != static_cast<size_t>(q.experts_used) ||
                    (override_weights && override_weights->size() != static_cast<size_t>(q.experts_used))) {
                    throw std::runtime_error("debug MoE override count does not match selected expert count");
                }
                q38_select_experts(layer.router, cur, st.moe, q.expert_weights_scale, stream);
                for (int j = 0; j < q.experts_used; ++j) {
                    const int expert = override_ids[static_cast<size_t>(j)];
                    if (expert < 0 || expert >= q.experts) {
                        throw std::runtime_error("invalid debug MoE override");
                    }
                    for (int prior = 0; prior < j; ++prior) {
                        if (override_ids[static_cast<size_t>(prior)] == expert) {
                            throw std::runtime_error("duplicate expert in debug MoE override");
                        }
                    }
                    st.moe.h_ids()[j] = expert;
                    if (override_weights) {
                        const float weight = (*override_weights)[static_cast<size_t>(j)];
                        if (!std::isfinite(weight)) throw std::runtime_error("invalid debug MoE weight");
                        st.moe.h_weights()[j] = weight;
                    }
                }
                if (!override_weights) {
                    const std::vector<float> logits = copy_float(
                        stream, st.moe.d_logits(), static_cast<size_t>(q.experts));
                    float max_logit = -std::numeric_limits<float>::infinity();
                    for (int expert : override_ids) {
                        max_logit = std::max(max_logit, logits[static_cast<size_t>(expert)]);
                    }
                    float sum = 0.0f;
                    for (int j = 0; j < q.experts_used; ++j) {
                        const int expert = override_ids[static_cast<size_t>(j)];
                        const float weight = std::exp(logits[static_cast<size_t>(expert)] - max_logit);
                        st.moe.h_weights()[j] = weight;
                        sum += weight;
                    }
                    const float scale = sum > 0.0f ? q.expert_weights_scale / sum : 0.0f;
                    for (int j = 0; j < q.experts_used; ++j) st.moe.h_weights()[j] *= scale;
                }
                experts_preselected = true;
                ++debug_trace->debug_moe_route_override_count;
            }

            __half * selected_output_trace = nullptr;
            if (trace && (trace->state_trace || (trace->extended_parity && il == 0))) {
                selected_output_trace = st.moe.selected_output_trace();
            }
            // F32 MoE: the routed sum, the shared expert and the MoE branch stay
            // unrounded; the half buffer is refreshed only so the traces and the
            // legacy consumers keep their meaning.
            // Input-only F32 switch: the expert/router/shared-expert activations
            // are read from the unrounded HC mix row (the same rule the GDN and
            // QSA paths follow), while the output contract stays the legacy one so
            // the combine keeps consuming d_accum / d_down.
            const bool ffn_f32_row_valid = ffn_used_f32 || ffn_f32_inject ||
                debug_hc_q8_0_ffn || debug_f32_hc_ffn;
            const bool moe_f32_in = q38_moe_f32_mid_enabled() && ffn_f32_row_valid;
            const bool moe_f32_mid = q38_moe_f32_out_opt_in() && moe_f32_in &&
                st.moe.moe_out_f32() != nullptr;
            float * moe_out_f32 = nullptr;
            if (rt_.fused_experts) {
                if (!layer.routed.gate_up || !layer.routed.down) {
                    throw std::runtime_error("Qwen3.8 fused expert tensor set is incomplete");
                }
                if (moe_f32_in && selected_output_trace) {
                    run_qwen38_routed_moe_fused_f32_x<true>(
                        il, hc_f32_scratch.mixed(), layer.router,
                        *layer.routed.gate_up, *layer.routed.down,
                        rt_.fused_expert_cache, st.moe, st.moe.d_accum(),
                        q.expert_weights_scale, stream, experts_preselected, selected_output_trace);
                } else if (moe_f32_in) {
                    run_qwen38_routed_moe_fused_f32_x<false>(
                        il, hc_f32_scratch.mixed(), layer.router,
                        *layer.routed.gate_up, *layer.routed.down,
                        rt_.fused_expert_cache, st.moe, st.moe.d_accum(),
                        q.expert_weights_scale, stream, experts_preselected);
                } else if (selected_output_trace) {
                    run_qwen38_routed_moe_fused<true>(
                        il, cur, layer.router,
                        *layer.routed.gate_up, *layer.routed.down,
                        rt_.fused_expert_cache, st.moe, st.moe_out,
                        q.expert_weights_scale, stream, experts_preselected, selected_output_trace);
                } else {
                    run_qwen38_routed_moe_fused<false>(
                        il, cur, layer.router,
                        *layer.routed.gate_up, *layer.routed.down,
                        rt_.fused_expert_cache, st.moe, st.moe_out,
                        q.expert_weights_scale, stream, experts_preselected);
                }
            } else {
                if (!layer.routed.gate || !layer.routed.up || !layer.routed.down) {
                    throw std::runtime_error("Qwen3.8 expert tensor set is incomplete");
                }
                if (moe_f32_in && selected_output_trace) {
                    run_qwen38_routed_moe_separate_f32_x<true>(
                        il, hc_f32_scratch.mixed(), layer.router,
                        *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                        rt_.expert_cache, st.moe, st.moe.d_accum(),
                        q.expert_weights_scale, stream, experts_preselected, selected_output_trace);
                } else if (moe_f32_in) {
                    run_qwen38_routed_moe_separate_f32_x<false>(
                        il, hc_f32_scratch.mixed(), layer.router,
                        *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                        rt_.expert_cache, st.moe, st.moe.d_accum(),
                        q.expert_weights_scale, stream, experts_preselected);
                } else if (selected_output_trace) {
                    run_qwen38_routed_moe_separate<true>(
                        il, cur, layer.router,
                        *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                        rt_.expert_cache, st.moe, st.moe_out,
                        q.expert_weights_scale, stream, experts_preselected, selected_output_trace);
                } else {
                    run_qwen38_routed_moe_separate<false>(
                        il, cur, layer.router,
                        *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                        rt_.expert_cache, st.moe, st.moe_out,
                        q.expert_weights_scale, stream, experts_preselected);
                }
            }
            if (trace && trace->state_trace) {
                trace->add_extended("layer" + std::to_string(il) + ".ffn.selected_expert_outputs",
                    copy_half(stream, selected_output_trace,
                        static_cast<size_t>(q.experts_used) * q.hidden));
            }
            if (trace && trace->state_trace) {
                const std::string layer_name = "layer" + std::to_string(il) + ".ffn";
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " MoE router";
                if (il == 8) {
                    trace->add_extended(layer_name + ".router_input",
                        copy_half(stream, cur, static_cast<size_t>(q.hidden)));
                }
                trace->add_extended(layer_name + ".router_logits",
                    copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts)));
                std::vector<int64_t> expert_ids;
                std::vector<float> expert_weights;
                expert_ids.reserve(static_cast<size_t>(q.experts_used));
                expert_weights.reserve(static_cast<size_t>(q.experts_used));
                for (int j = 0; j < q.experts_used; ++j) {
                    expert_ids.push_back(st.moe.h_ids()[j]);
                    expert_weights.push_back(st.moe.h_weights()[j]);
                }
                trace->add_integer_vector(layer_name + ".expert_ids", std::move(expert_ids));
                trace->add_extended(layer_name + ".expert_weights", std::move(expert_weights));
                trace->add_extended(layer_name + ".ffn_moe_out",
                    copy_half(stream, st.moe_out, static_cast<size_t>(q.hidden)));
            }
            if (trace && trace->captures_state_position() && il == 8 &&
                !trace->debug_router_cpu_activation_f32.empty()) {
                if (trace->debug_router_cpu_activation_f32.size() != static_cast<size_t>(q.hidden)) {
                    throw std::runtime_error("CPU router activation size does not match hidden size");
                }
                const std::vector<float> warp_activation = copy_half(
                    stream, cur, static_cast<size_t>(q.hidden));
                const std::vector<float> router_weights = decode_router_weights(*layer.routed.router);
                const std::vector<float> cpu_reference_logits = deterministic_router_logits(
                    router_weights, trace->debug_router_cpu_activation_f32, q.experts, q.hidden);
                const std::vector<float> warp_reference_logits = deterministic_router_logits(
                    router_weights, warp_activation, q.experts, q.hidden);
                trace->add_extended("layer8.router_cross.cpu_activation_reference_logits",
                    cpu_reference_logits);
                trace->add_extended("layer8.router_cross.warp_activation_reference_logits",
                    warp_reference_logits);
                trace->add_extended("layer8.router_cross.warp_activation_warp_cuda_logits",
                    copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts)));

                const cudaError_t input_copy = cudaMemcpyAsync(
                    hc_f32_scratch.embedding(), trace->debug_router_cpu_activation_f32.data(),
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (input_copy != cudaSuccess) {
                    throw std::runtime_error("copy CPU router activation for CUDA cross failed");
                }
                launch_matrix_gemv_f32_input(
                    layer.router, hc_f32_scratch.embedding(), st.moe.d_logits(), stream);
                trace->add_extended("layer8.router_cross.cpu_activation_warp_cuda_logits",
                    copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts)));

                const cudaError_t warp_input_copy = cudaMemcpyAsync(
                    hc_f32_scratch.embedding(), warp_activation.data(),
                    static_cast<size_t>(q.hidden) * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (warp_input_copy != cudaSuccess) {
                    throw std::runtime_error("copy Warp router activation for CUDA cross failed");
                }
                launch_matrix_gemv_f32_input(
                    layer.router, hc_f32_scratch.embedding(), st.moe.d_logits(), stream);
                trace->add_extended("layer8.router_cross.warp_activation_warp_cuda_f32_input_logits",
                    copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts)));
            }
            Qwen38Trace * moe_output_override = active_trace_;
            const size_t routed_output_count = moe_output_override &&
                moe_output_override->debug_routed_moe_output_all_positions
                ? moe_output_override->debug_routed_moe_output.size() / static_cast<size_t>(q.hidden) : 0;
            const bool all_position_routed_output_override = moe_output_override &&
                moe_output_override->debug_routed_moe_output_all_positions &&
                il == moe_output_override->debug_routed_moe_output_layer &&
                pos < static_cast<int>(routed_output_count);
            const bool selected_routed_output_override = trace && trace->captures_state_position() &&
                il == trace->debug_routed_moe_output_layer &&
                !trace->debug_routed_moe_output.empty() &&
                !trace->debug_routed_moe_output_all_positions;
            if (all_position_routed_output_override || selected_routed_output_override) {
                if (moe_output_override->debug_routed_moe_output.size() % static_cast<size_t>(q.hidden) != 0 ||
                    (!moe_output_override->debug_routed_moe_output_all_positions &&
                     moe_output_override->debug_routed_moe_output.size() != static_cast<size_t>(q.hidden))) {
                    throw std::runtime_error("debug routed MoE output has an invalid size");
                }
                const size_t offset = moe_output_override->debug_routed_moe_output_all_positions
                    ? static_cast<size_t>(pos) * q.hidden : 0;
                std::vector<__half> injected(static_cast<size_t>(q.hidden));
                for (int i = 0; i < q.hidden; ++i) {
                    injected[static_cast<size_t>(i)] =
                        __float2half(moe_output_override->debug_routed_moe_output[offset +
                            static_cast<size_t>(i)]);
                }
                cudaError_t debug_error = cudaStreamSynchronize(stream);
                if (debug_error != cudaSuccess) throw std::runtime_error("sync before routed MoE output injection failed");
                debug_error = cudaMemcpy(st.moe_out, injected.data(), injected.size() * sizeof(__half),
                    cudaMemcpyHostToDevice);
                if (debug_error != cudaSuccess) throw std::runtime_error("routed MoE output injection failed");
                ++moe_output_override->debug_routed_moe_output_override_count;
                if (trace && trace->captures_state_position()) {
                    std::vector<float> injected_values(
                        moe_output_override->debug_routed_moe_output.begin() +
                            static_cast<std::ptrdiff_t>(offset),
                        moe_output_override->debug_routed_moe_output.begin() +
                            static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(q.hidden)));
                    trace->add_extended("layer" + std::to_string(il) + ".ffn_moe_out_cpu_override",
                        injected_values);
                }
            }
            if (legacy_trace && il == 0) {
                trace->router_logits = copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts));
                trace->router_weights.assign(st.moe.h_weights(), st.moe.h_weights() + q.experts_used);
                trace->expert_ids.assign(st.moe.h_ids(), st.moe.h_ids() + q.experts_used);
                if (trace->extended_parity) {
                    trace->add_extended("layer0.router_logits", trace->router_logits);
                    trace->add_extended_ints("layer0.selected_expert_ids", trace->expert_ids);
                    trace->add_extended("layer0.selected_expert_weights", trace->router_weights);
                    trace->add_extended("layer0.routed_weighted_sum_f32", copy_float(
                        stream, st.moe.d_accum(), static_cast<size_t>(q.hidden)));
                    trace->add_extended("layer0.routed_moe_output", copy_half(
                        stream, st.moe_out, static_cast<size_t>(q.hidden)));
                    const std::vector<float> selected = copy_half(
                        stream, selected_output_trace,
                        static_cast<size_t>(q.experts_used) * q.hidden);
                    for (int rank = 0; rank < q.experts_used; ++rank) {
                        const size_t offset = qwen38_expert_trace_offset(
                            static_cast<size_t>(rank), static_cast<size_t>(q.hidden));
                        const size_t end = offset + static_cast<size_t>(q.hidden);
                        if (end > selected.size()) {
                            throw std::runtime_error("selected expert trace slice is out of range");
                        }
                        std::vector<float> expert_output(
                            selected.begin() + static_cast<std::ptrdiff_t>(offset),
                            selected.begin() + static_cast<std::ptrdiff_t>(end));
                        trace->add_extended(
                            "layer0.expert_rank_" + std::to_string(rank) + "_id_" +
                                std::to_string(trace->expert_ids[static_cast<size_t>(rank)]),
                            expert_output);
                    }
                }
            }

            if (moe_f32_in) {
                run_qwen38_shared_expert_f32_x(hc_f32_scratch.mixed(), layer.shared, st.moe, stream);
                // Keep the legacy output contract: mirror the routed sum into the
                // half buffer and add the shared expert there, exactly like the
                // F16 path does, so the traces and the combine are unchanged.
                q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    st.moe.d_accum(), st.moe_out, q.hidden);
                q38_float_to_half_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    st.moe.down_f32(), st.moe.d_down(), q.hidden);
                q38_add_shared_kernel<<<(q.hidden + 255) / 256, 256, 0, stream>>>(
                    st.moe_out, st.moe.d_down(), st.moe.d_shared_gate(), q.hidden);
                if (moe_f32_mid) {
                    q38_finish_moe_f32(st.moe, st.moe.moe_out_f32(), stream);
                    moe_out_f32 = st.moe.moe_out_f32();
                }
            } else {
                run_qwen38_shared_expert(cur, layer.shared, st.moe, st.moe_out, stream);
            }
            if (trace && trace->state_trace && il == trace->debug_ffn_output_layer &&
                pos == trace->state_trace_position &&
                !trace->debug_ffn_output.empty()) {
                if (trace->debug_ffn_output.size() != static_cast<size_t>(q.hidden)) {
                    throw std::runtime_error("debug FFN output size does not match hidden size");
                }
                std::vector<__half> injected(q.hidden);
                for (int i = 0; i < q.hidden; ++i) {
                    injected[static_cast<size_t>(i)] = __float2half(trace->debug_ffn_output[static_cast<size_t>(i)]);
                }
                cudaError_t debug_error = cudaStreamSynchronize(stream);
                if (debug_error != cudaSuccess) throw std::runtime_error("sync before debug FFN output injection failed");
                debug_error = cudaMemcpy(st.moe_out, injected.data(), injected.size() * sizeof(__half), cudaMemcpyHostToDevice);
                if (debug_error != cudaSuccess) throw std::runtime_error("debug FFN output injection failed");
            }
            if (trace && trace->state_trace) {
                const std::string layer_name = "layer" + std::to_string(il);
                const std::vector<float> shared_output = copy_half(
                    stream, st.moe.d_down(), static_cast<size_t>(q.hidden));
                const std::vector<float> shared_gate = copy_float(stream, st.moe.d_shared_gate(), 1);
                std::vector<float> shared_gated(shared_output.size());
                std::transform(shared_output.begin(), shared_output.end(), shared_gated.begin(),
                    [&](float value) { return value * shared_gate[0]; });
                trace->add_extended(layer_name + ".ffn_shexp", shared_output);
                trace->add_extended(layer_name + ".shared_expert_gate_sigmoid", shared_gate);
                trace->add_extended(layer_name + ".ffn_shexp_gated", std::move(shared_gated));
                // F32 twins of the MoE branch. The half mirrors above sit at ~5e-4,
                // which is far above the 1e-4-level errors that actually drive the
                // layer-to-layer drift, so an error budget needs these.
                trace->add_extended(layer_name + ".ffn_routed_sum_f32",
                    copy_float(stream, st.moe.d_accum(), static_cast<size_t>(q.hidden)));
                if (st.moe.down_f32() != nullptr) {
                    trace->add_extended(layer_name + ".ffn_shexp_f32",
                        copy_float(stream, st.moe.down_f32(), static_cast<size_t>(q.hidden)));
                }
                trace->add_extended(layer_name + ".hc_ffn_inject_f32",
                    copy_float(stream, hc_f32_scratch.inject(), static_cast<size_t>(q.hc_count)));
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " FFN output";
                trace->add_extended("layer" + std::to_string(il) + ".ffn_output",
                    copy_half(stream, st.moe_out, static_cast<size_t>(q.hidden)));
                if (moe_out_f32 != nullptr) {
                    trace->add_extended("layer" + std::to_string(il) + ".ffn_output_f32",
                        copy_float(stream, moe_out_f32, static_cast<size_t>(q.hidden)));
                }
                if (attn_out_f32 != nullptr) {
                    trace->add_extended("layer" + std::to_string(il) + ".attention_out_f32",
                        copy_float(stream, attn_out_f32, static_cast<size_t>(q.hidden)));
                }
            }
            if (legacy_trace && il == 0) trace->ffn_out = copy_half(stream, st.moe_out, static_cast<size_t>(q.hidden));
            if (trace && trace->extended_parity && il == 0) {
                const std::vector<float> shared_down = copy_half(
                    stream, st.moe.d_down(), static_cast<size_t>(q.hidden));
                const std::vector<float> shared_gate = copy_float(stream, st.moe.d_shared_gate(), 1);
                std::vector<float> shared_weighted(shared_down.size());
                std::transform(shared_down.begin(), shared_down.end(), shared_weighted.begin(),
                    [&](float value) { return value * shared_gate[0]; });
                trace->add_extended("layer0.shared_expert_output", shared_down);
                trace->add_extended("layer0.shared_expert_gate", shared_gate);
                trace->add_extended("layer0.shared_expert_weighted_output_f32", shared_weighted);
                trace->add_extended("layer0.final_moe_output", trace->ffn_out);
            }
            profile_.mark_end(kQwen38StageMoe, stream);

            profile_.mark_begin(kQwen38StageHcCombineFfn, stream);
            const bool moe_f32_boundary = q38_moe_f32_enabled() &&
                !all_position_routed_output_override && !selected_routed_output_override &&
                !(trace && trace->debug_ffn_output_layer == il &&
                  !trace->debug_ffn_output.empty());
            if (moe_f32_boundary) {
                run_qwen38_hc_combine_moe_f32(
                    state_.residual(), st.moe.d_accum(), st.moe.d_down(),
                    st.moe.d_shared_gate(), st.hc_ffn, stream,
                    !q38_residual_f32_enabled());
            } else if (moe_out_f32 != nullptr && ffn_f32_inject) {
                run_qwen38_hc_combine_f32_block_f32_inject(
                    state_.residual(), moe_out_f32, hc_f32_scratch, stream,
                    !q38_residual_f32_enabled());
            } else if (ffn_f32_inject) {
                run_qwen38_hc_combine_moe_f32_inject(
                    state_.residual(), st.moe.d_accum(), st.moe.d_down(),
                    st.moe.d_shared_gate(), hc_f32_scratch, stream,
                    !q38_residual_f32_enabled());
            } else {
                run_qwen38_hc_combine(state_.residual(), st.moe_out, st.hc_ffn, stream,
                    !q38_residual_f32_enabled());
            }
            profile_.mark_end(kQwen38StageHcCombineFfn, stream);
            Qwen38Trace * output_override = active_trace_;
            const size_t all_position_count = output_override && output_override->debug_hc_residual_all_positions
                ? output_override->debug_hc_residual.size() / static_cast<size_t>(hc_dim) : 0;
            const bool all_position_output_override = output_override &&
                output_override->debug_hc_residual_all_positions &&
                il == output_override->debug_hc_residual_layer &&
                pos < static_cast<int>(all_position_count);
            const bool selected_output_override = trace && trace->captures_state_position() &&
                il == trace->debug_hc_residual_layer && !trace->debug_hc_residual.empty() &&
                !trace->debug_hc_residual_all_positions;
            if (all_position_output_override || selected_output_override) {
                const size_t expected_size = output_override->debug_hc_residual_all_positions
                    ? all_position_count * static_cast<size_t>(hc_dim) : static_cast<size_t>(hc_dim);
                if (output_override->debug_hc_residual.size() != expected_size) {
                    throw std::runtime_error("debug HC residual output has an invalid size");
                }
                const std::string residual_name = "layer" + std::to_string(il) + ".output";
                if (trace && trace->captures_state_position()) {
                    trace->add_extended(residual_name + "_before_cpu_override",
                        copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                }
                const size_t offset = output_override->debug_hc_residual_all_positions
                    ? static_cast<size_t>(pos) * hc_dim : 0;
                std::vector<float> injected(static_cast<size_t>(hc_dim));
                for (int i = 0; i < hc_dim; ++i) {
                    injected[static_cast<size_t>(i)] =
                        output_override->debug_hc_residual[offset + static_cast<size_t>(i)];
                }
                cudaError_t debug_error = cudaStreamSynchronize(stream);
                if (debug_error != cudaSuccess) throw std::runtime_error("sync before debug HC residual injection failed");
                debug_error = cudaMemcpy(state_.residual(), injected.data(), injected.size() * sizeof(float), cudaMemcpyHostToDevice);
                if (debug_error != cudaSuccess) throw std::runtime_error("debug HC residual injection failed");
                ++output_override->debug_hc_output_override_count;
                if (trace && trace->captures_state_position()) {
                    std::vector<float> injected_values(
                        output_override->debug_hc_residual.begin() + static_cast<std::ptrdiff_t>(offset),
                        output_override->debug_hc_residual.begin() +
                            static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(hc_dim)));
                    trace->add_extended(residual_name + "_cpu_f32", injected_values);
                    trace->add_extended(residual_name + "_after_cpu_override",
                        copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                }
            }
            if (legacy_trace && il == 0) trace->after_ffn = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));
            if (trace && trace->state_trace) {
                trace_context_ = "position " + std::to_string(pos) + " layer " + std::to_string(il) + " layer output";
                trace->add_extended("layer" + std::to_string(il) + ".output",
                    copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
                // Unrounded residual: the half mirror above hides every error below
                // its own rounding (~5e-4), which is exactly the range that decides
                // whether the next layer's Q8_0 activation codes match the CPU.
                trace->add_extended("layer" + std::to_string(il) + ".output_f32",
                    copy_float(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            }
            if (trace && trace->extended_parity && il == 0) {
                trace->add_extended("layer0.hc_ffn_combine_residual", trace->after_ffn);
                trace->add_extended("layer0.output", trace->after_ffn);
            }
            profile_.layer_end(il, stream);
        }

        if (trace && trace->state_trace && !trace->debug_final_residual_f32.empty()) {
            if (trace->debug_final_residual_f32.size() != static_cast<size_t>(hc_dim)) {
                throw std::runtime_error("final residual override is invalid");
            }
            trace->add_extended("final.residual_before_cpu_override",
                copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
            const cudaError_t e = cudaMemcpyAsync(
                hc_f32_scratch_.residual(), trace->debug_final_residual_f32.data(),
                static_cast<size_t>(hc_dim) * sizeof(float), cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) throw std::runtime_error("copy final residual override failed");
            q38_float_copy_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
                hc_f32_scratch_.residual(), state_.residual(), hc_dim);
            trace->add_extended("final.residual_cpu_f32", trace->debug_final_residual_f32);
            trace->add_extended("final.residual_after_cpu_override",
                copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim)));
        }
        profile_.mark_begin(kQwen38StageHead, stream);
        __half * head = nullptr;
        float * head_f32 = nullptr;
        const bool debug_f32_hc_head = trace && trace->state_trace && trace->debug_f32_hc_head;
        if (debug_f32_hc_head) {
            if (rt_.hc_head.debug_norm_f32.empty()) {
                throw std::runtime_error("F32 HC head replay weights unavailable");
            }
            run_qwen38_hc_f32_mix_from_residual(
                state_.residual(), rt_.hc_head, hc_f32_scratch_, q.rms_eps, stream);
            const int blocks = (q.hidden + 255) / 256;
            q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                hc_f32_scratch_.mixed(), state_.head_hc().mixed(), q.hidden);
            head = state_.head_hc().mixed();
        } else {
            head = run_qwen38_hc_mix_f32(
                state_.residual(), rt_.hc_head, hc_f32_scratch_, state_.head_hc(),
                q.rms_eps, stream);
            if (head != nullptr) head_f32 = hc_f32_scratch_.mixed();
            if (head == nullptr) {
                head = run_qwen38_hc_mix(
                    state_.residual(), rt_.hc_head, state_.head_hc(), q.rms_eps, stream);
            }
        }
        if (trace && trace->state_trace) {
            trace->add_extended("final.hc_gamma_half",
                copy_half(stream, rt_.hc_head.norm.data, static_cast<size_t>(hc_dim)));
            if (!rt_.hc_head.debug_norm_f32.empty()) {
                trace->add_extended("final.hc_gamma_f32", rt_.hc_head.debug_norm_f32);
            }
            trace->add_extended("final.hc_norm",
                debug_f32_hc_head
                    ? copy_float(stream, hc_f32_scratch_.norm(), static_cast<size_t>(hc_dim))
                    : copy_half(stream, state_.head_hc().xn(), static_cast<size_t>(hc_dim)));
            trace->add_extended("final.hc_gate",
                debug_f32_hc_head
                    ? copy_float(stream, hc_f32_scratch_.gate(), static_cast<size_t>(hc_dim))
                    : copy_half(stream, state_.head_hc().gate(), static_cast<size_t>(hc_dim)));
            trace->add_extended("final.hc_mixed",
                copy_half(stream, head, static_cast<size_t>(q.hidden)));
            trace_context_ = "position " + std::to_string(pos) + " final head input";
            trace->add_extended("final.head_input", copy_half(stream, head, static_cast<size_t>(q.hidden)));
        }
        // Feed the head GEMV the unrounded F32 HC mix when the F32 path produced
        // it: the CPU quantises its F32 head input, and the half round trip would
        // shift every Q8_K block scale.
        const bool head_f32_input = [] {
            const char * v = std::getenv("WARODECODE_HEAD_F32_IN");
            return v == nullptr || v[0] == '\0' || v[0] != '0';
        }();
        if (head_f32_input && head_f32 != nullptr &&
            lowbit_q6_k_head_q8_k_eligible(rt_.output)) {
            launch_q6_k_head_gemv_q8_k_from_f32(rt_.output, head_f32, logits_, stream, 256);
        } else {
            launch_lowbit_gemv_float(rt_.output, head, logits_, stream);
        }
        q38_argmax_kernel<<<1, 256, 0, stream>>>(
            logits_, rt_.vocab, next_token_, has_nonfinite_);
        profile_.mark_end(kQwen38StageHead, stream);

        profile_.mark_begin(kQwen38StageSync, stream);
        if (trace && trace->state_trace) {
            trace_context_ = "position " + std::to_string(pos) + " final logits";
            trace->add_extended("final.logits",
                copy_float(stream, logits_, static_cast<size_t>(rt_.vocab)));
            trace->add_state_summary("final.logits", copy_float(
                stream, logits_, static_cast<size_t>(rt_.vocab)), {static_cast<uint64_t>(rt_.vocab)});
        }

        cudaError_t e = cudaMemcpyAsync(
            host_next_token_, next_token_, sizeof(int),
            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error("copy Qwen3.8 next token failed");
        e = cudaMemcpyAsync(
            host_has_nonfinite_, has_nonfinite_, sizeof(int),
            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error("copy Qwen3.8 validation flag failed");
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("Qwen3.8 step sync failed: ") + cudaGetErrorString(e));
        }
        profile_.mark_end(kQwen38StageSync, stream);
        profile_.mark_end(kQwen38StageToken, stream);
        // Every recorded event is complete now, so the elapsed reads cannot fail
        // with cudaErrorNotReady.
        profile_.flush_token();
        Q38HcKernelProfiler::instance().flush_token();
        if (profile_on) profile_.note_token_host(Qwen38StageProfiler::host_now() - profile_token_start);
        if (*host_has_nonfinite_) throw std::runtime_error("Qwen3.8 output logits contain NaN or infinity");

        if (trace && trace->state_trace) {
            trace->add_integer_vector("output_token", {*host_next_token_});
        }
        history_.push_back(token);
        return *host_next_token_;
    }

    std::vector<std::pair<int, float>> top_logits(int count) const {
        if (count <= 0 || count > rt_.vocab) throw std::runtime_error("invalid Qwen3.8 top-k size");
        std::vector<float> host_logits(static_cast<size_t>(rt_.vocab));
        const cudaError_t e = cudaMemcpy(
            host_logits.data(), logits_, host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) throw std::runtime_error("copy Qwen3.8 logits failed");

        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(host_logits.size());
        for (int i = 0; i < rt_.vocab; ++i) ranked.emplace_back(host_logits[static_cast<size_t>(i)], i);
        const auto by_logit = [](const auto & a, const auto & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        };
        std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(), by_logit);

        std::vector<std::pair<int, float>> out;
        out.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            out.emplace_back(ranked[static_cast<size_t>(i)].second, ranked[static_cast<size_t>(i)].first);
        }
        return out;
    }

    double run(
            int start_token,
            int steps,
            std::vector<int> * generated = nullptr,
            std::vector<std::pair<int, float>> * first_top_logits = nullptr,
            int top_k = 0,
            Qwen38Trace * trace = nullptr,
            const std::vector<int> * forced_input_tokens = nullptr,
            const std::vector<int> * prefill_tokens = nullptr,
            Qwen38RunTiming * timing = nullptr) {
        if (steps <= 0) throw std::runtime_error("Qwen3.8 steps must be positive");
        if (forced_input_tokens &&
            (static_cast<int>(forced_input_tokens->size()) != steps || forced_input_tokens->front() != start_token)) {
            throw std::runtime_error("forced Qwen3.8 input tokens must match --generate and --start-token");
        }
        const int prefill_steps = prefill_tokens ? static_cast<int>(prefill_tokens->size()) : 0;
        if (prefill_steps > 0 && forced_input_tokens) {
            throw std::runtime_error("Qwen3.8 prefill tokens cannot be combined with forced input tokens");
        }
        if (prefill_steps + steps > state_.max_ctx()) {
            throw std::runtime_error("Qwen3.8 prefill + decode exceeds the context length");
        }
        if (prefill_tokens) {
            for (int id : *prefill_tokens) {
                if (id < 0 || id >= rt_.vocab) throw std::runtime_error("Qwen3.8 prefill token out of range");
            }
        }
        active_trace_ = trace;
        clear();


        cudaEvent_t begin{}, end{}, prefill_mark{};
        cudaError_t e = cudaEventCreate(&begin);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventCreate begin failed");
        e = cudaEventCreate(&end);
        if (e != cudaSuccess) {
            cudaEventDestroy(begin);
            throw std::runtime_error("cudaEventCreate end failed");
        }
        if (prefill_steps > 0) {
            e = cudaEventCreate(&prefill_mark);
            if (e != cudaSuccess) {
                cudaEventDestroy(begin);
                cudaEventDestroy(end);
                throw std::runtime_error("cudaEventCreate prefill mark failed");
            }
        }

        int token = start_token;
        const bool debug_f32_hc_attn_0_8 = trace && trace->debug_f32_hc_attn_0_8;
        const bool debug_f32_hc_ffn_0_8 = trace && trace->debug_f32_hc_ffn_0_8;
        const bool debug_f32_hc_ffn_layer8 = trace && trace->debug_f32_hc_ffn_layer8;
        const int debug_hc_q8_1_attn_from = trace
            ? trace->debug_hc_q8_1_attn_from : 0;
        const int debug_hc_q8_1_attn_through = trace
            ? trace->debug_hc_q8_1_attn_through : -1;
        const int debug_hc_q8_1_ffn_from = trace
            ? trace->debug_hc_q8_1_ffn_from : 0;
        const int debug_hc_q8_1_ffn_through = trace
            ? trace->debug_hc_q8_1_ffn_through : -1;
        const int debug_hc_q8_0_ffn_from = trace
            ? trace->debug_hc_q8_0_ffn_from : 0;
        const int debug_hc_q8_0_ffn_through = trace
            ? trace->debug_hc_q8_0_ffn_through : -1;
        e = cudaEventRecord(begin, state_.stream());
        if (e != cudaSuccess) throw std::runtime_error("cudaEventRecord begin failed");

        if (prefill_steps > 0) {
            // Prompt phase: forward every prompt token at its real position and
            // keep the sample of the last one, exactly like a decode step would.
            // That sample is the first generated token's *input*, so the decode
            // phase below reports a clean one-forward-per-token ratio. The plain
            // decode path is untouched and still lives in the else branch.
            profile_.set_decode_phase(false);
            for (int pos = 0; pos < prefill_steps; ++pos) {
                const bool capture = trace && (trace->state_trace
                    ? trace->state_trace_position < 0 || pos == trace->state_trace_position
                    : pos == 0);
                token = step((*prefill_tokens)[static_cast<size_t>(pos)], pos,
                    capture ? trace : nullptr,
                    debug_f32_hc_attn_0_8, debug_f32_hc_ffn_0_8, debug_f32_hc_ffn_layer8,
                    debug_hc_q8_1_attn_from, debug_hc_q8_1_attn_through,
                    debug_hc_q8_1_ffn_from, debug_hc_q8_1_ffn_through,
                    debug_hc_q8_0_ffn_from, debug_hc_q8_0_ffn_through);
            }
            e = cudaEventRecord(prefill_mark, state_.stream());
            if (e != cudaSuccess) throw std::runtime_error("cudaEventRecord prefill mark failed");
            profile_.set_decode_phase(true);
            for (int pos = prefill_steps; pos < prefill_steps + steps; ++pos) {
                const bool capture = trace && (trace->state_trace
                    ? trace->state_trace_position < 0 || pos == trace->state_trace_position
                    : pos == 0);
                token = step(token, pos, capture ? trace : nullptr,
                    debug_f32_hc_attn_0_8, debug_f32_hc_ffn_0_8, debug_f32_hc_ffn_layer8,
                    debug_hc_q8_1_attn_from, debug_hc_q8_1_attn_through,
                    debug_hc_q8_1_ffn_from, debug_hc_q8_1_ffn_through,
                    debug_hc_q8_0_ffn_from, debug_hc_q8_0_ffn_through);
                if (pos == prefill_steps && first_top_logits && top_k > 0) {
                    *first_top_logits = top_logits(top_k);
                }
                if (generated) generated->push_back(token);
            }
        } else {
            profile_.set_decode_phase(true);
            for (int pos = 0; pos < steps; ++pos) {
                if (forced_input_tokens) token = (*forced_input_tokens)[static_cast<size_t>(pos)];
                const bool capture = trace && (trace->state_trace
                    ? trace->state_trace_position < 0 || pos == trace->state_trace_position
                    : pos == 0);
                token = step(token, pos, capture ? trace : nullptr,
                    debug_f32_hc_attn_0_8, debug_f32_hc_ffn_0_8, debug_f32_hc_ffn_layer8,
                    debug_hc_q8_1_attn_from, debug_hc_q8_1_attn_through,
                    debug_hc_q8_1_ffn_from, debug_hc_q8_1_ffn_through,
                    debug_hc_q8_0_ffn_from, debug_hc_q8_0_ffn_through);
                if (pos == 0 && first_top_logits && top_k > 0) {
                    *first_top_logits = top_logits(top_k);
                }
                if (generated) generated->push_back(token);
            }
        }

        e = cudaEventRecord(end, state_.stream());
        if (e != cudaSuccess) throw std::runtime_error("cudaEventRecord end failed");
        e = cudaEventSynchronize(end);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventSynchronize end failed");

        float ms = 0.0f;
        e = cudaEventElapsedTime(&ms, begin, end);
        if (timing) {
            timing->prefill_tokens = prefill_steps;
            timing->prefill_seconds = 0.0;
            timing->decode_tokens = steps;
            timing->decode_seconds = static_cast<double>(ms) / 1000.0;
        }
        if (prefill_steps > 0) {
            float prefill_ms = 0.0f;
            float decode_ms = 0.0f;
            const cudaError_t prefill_error = cudaEventElapsedTime(&prefill_ms, begin, prefill_mark);
            const cudaError_t phase_error = prefill_error == cudaSuccess
                ? cudaEventElapsedTime(&decode_ms, prefill_mark, end) : prefill_error;
            if (e == cudaSuccess) e = phase_error;
            if (timing) {
                timing->prefill_seconds = static_cast<double>(prefill_ms) / 1000.0;
                timing->decode_seconds = static_cast<double>(decode_ms) / 1000.0;
            }
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        if (prefill_mark) cudaEventDestroy(prefill_mark);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventElapsedTime failed");
        active_trace_ = nullptr;
        return static_cast<double>(ms) / 1000.0;
    }

private:
    static std::vector<float> decode_router_weights(const TensorInfo & tensor) {
        const size_t count = static_cast<size_t>(tensor.elements());
        std::vector<float> values(count);
        if (tensor.type == GgmlType::F32) {
            std::memcpy(values.data(), tensor.data, count * sizeof(float));
            return values;
        }
        if (tensor.type == GgmlType::F16) {
            for (size_t i = 0; i < count; ++i) {
                __half value;
                std::memcpy(&value, tensor.data + i * sizeof(uint16_t), sizeof(uint16_t));
                values[i] = __half2float(value);
            }
            return values;
        }
        if (tensor.type == GgmlType::BF16) {
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = 0;
                std::memcpy(&bits, tensor.data + i * sizeof(uint16_t), sizeof(bits));
                const uint32_t expanded = static_cast<uint32_t>(bits) << 16;
                std::memcpy(&values[i], &expanded, sizeof(float));
            }
            return values;
        }
        if (tensor.type == GgmlType::Q8_0) {
            const size_t blocks = count / 32;
            if (count % 32 != 0) throw std::runtime_error("Q8_0 router width is not block aligned");
            for (size_t block = 0; block < blocks; ++block) {
                const uint8_t * raw = tensor.data + block * 34;
                __half scale_half;
                std::memcpy(&scale_half, raw, sizeof(uint16_t));
                const float scale = __half2float(scale_half);
                for (size_t i = 0; i < 32; ++i) {
                    int8_t quantized = 0;
                    std::memcpy(&quantized, raw + 2 + i, sizeof(quantized));
                    values[block * 32 + i] = scale * static_cast<float>(quantized);
                }
            }
            return values;
        }
        throw std::runtime_error("deterministic router reference does not support tensor type " +
            std::to_string(static_cast<uint32_t>(tensor.type)));
    }

    static std::vector<float> deterministic_router_logits(
            const std::vector<float> & weights,
            const std::vector<float> & activation,
            int experts,
            int hidden) {
        if (weights.size() != static_cast<size_t>(experts) * hidden ||
            activation.size() != static_cast<size_t>(hidden)) {
            throw std::runtime_error("deterministic router input geometry mismatch");
        }
        std::vector<float> logits(static_cast<size_t>(experts));
        for (int expert = 0; expert < experts; ++expert) {
            float sum = 0.0f;
            const size_t row = static_cast<size_t>(expert) * hidden;
            for (int i = 0; i < hidden; ++i) {
                sum = std::fma(weights[row + static_cast<size_t>(i)], activation[static_cast<size_t>(i)], sum);
            }
            logits[static_cast<size_t>(expert)] = sum;
        }
        return logits;
    }

    std::vector<float> copy_half(cudaStream_t stream, const __half * data, size_t n) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error("sync Qwen3.8 trace stream failed");
        std::vector<__half> half(n);
        e = cudaMemcpy(
            half.data(), data, n * sizeof(__half), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            throw std::runtime_error("copy Qwen3.8 trace tensor failed at " + trace_context_ +
                " (count=" + std::to_string(n) + "): " + cudaGetErrorString(e));
        }
        std::vector<float> out(n);
        std::transform(half.begin(), half.end(), out.begin(), [](__half x) {
            return __half2float(x);
        });
        return out;
    }

    std::vector<float> copy_float(cudaStream_t stream, const float * data, size_t n) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error("sync Qwen3.8 trace stream failed");
        std::vector<float> out(n);
        e = cudaMemcpy(
            out.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            throw std::runtime_error("copy Qwen3.8 trace tensor failed at " + trace_context_ +
                " (count=" + std::to_string(n) + "): " + cudaGetErrorString(e));
        }
        return out;
    }

    // The persistent HC residual stream is F32; trace call sites that were
    // written for a half residual still resolve through this overload.
    std::vector<float> copy_half(cudaStream_t stream, const float * data, size_t n) {
        return copy_float(stream, data, n);
    }

    std::vector<int64_t> copy_ints(cudaStream_t stream, const int * data, size_t n) {
        if (n == 0) return {};
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error("sync Qwen3.8 trace stream failed");
        std::vector<int> host(n);
        e = cudaMemcpy(host.data(), data, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            throw std::runtime_error("copy Qwen3.8 trace indices failed at " + trace_context_ +
                " (count=" + std::to_string(n) + "): " + cudaGetErrorString(e));
        }
        return std::vector<int64_t>(host.begin(), host.end());
    }

    Qwen38Runtime & rt_;
    Qwen38DecodeState & state_;
    Qwen38HcF32DebugScratch hc_f32_scratch_;
    Qwen38StageProfiler profile_;
    Qwen38Trace * active_trace_ = nullptr;
    float * logits_ = nullptr;
    int * next_token_ = nullptr;
    int * has_nonfinite_ = nullptr;
    int * host_next_token_ = nullptr;
    int * host_has_nonfinite_ = nullptr;
    std::vector<int> history_;
    std::string trace_context_;
};

inline uint64_t qwen38_token_hash(const std::vector<int> & ids) {
    uint64_t h = 1469598103934665603ull;
    for (int id : ids) {
        const uint32_t x = static_cast<uint32_t>(id);
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<uint8_t>(x >> (8 * b));
            h *= 1099511628211ull;
        }
    }
    return h;
}

} // namespace wd
