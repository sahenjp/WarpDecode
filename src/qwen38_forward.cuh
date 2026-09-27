#pragma once

#include "qwen38_runtime.cuh"
#include "qwen38_state.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wd {

__global__ void q38_repeat_embedding_kernel(
        const __half * __restrict__ emb,
        __half * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i < total) residual[i] = emb[i % hidden];
}

struct Q38MaxPair {
    float v;
    int i;
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
    std::vector<float> ffn_out;
    std::vector<float> after_ffn;
    std::vector<float> router_logits;
    std::vector<float> router_weights;
    std::vector<int> expert_ids;
    bool use_f32_hc_layer0 = false;
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

class Qwen38Decoder {
public:
    Qwen38Decoder(Qwen38Runtime & rt, Qwen38DecodeState & state)
        : rt_(rt), state_(state) {
        if (rt_.vocab <= 0) throw std::runtime_error("Qwen3.8 vocab is not initialized");
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

    int step(int token, int pos, Qwen38Trace * trace = nullptr) {
        const Qwen38Layout & q = rt_.layout;
        if (token < 0 || token >= rt_.vocab) throw std::runtime_error("Qwen3.8 token out of range");
        if (pos < 0 || pos >= state_.max_ctx()) throw std::runtime_error("Qwen3.8 position out of range");
        if (pos != static_cast<int>(history_.size())) {
            throw std::runtime_error("Qwen3.8 decoder currently requires contiguous single-sequence positions");
        }

        cudaStream_t stream = state_.stream();
        Qwen38HcF32DebugScratch hc_f32_scratch;

        // The wide HC residual starts as four identical copies of the token embedding.
        launch_q8_0_gather_row(rt_.token_embd, token, state_.head_hidden(), stream);
        const int hc_dim = q.hidden * q.hc_count;
        q38_repeat_embedding_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
            state_.head_hidden(), state_.residual(), q.hidden, q.hc_count);
        if (trace) trace->hc_init = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));
        if (trace) {
            const Qwen38HcWeights & hc_attn = rt_.layers.front().hc_attn;
            hc_f32_scratch.initialize(q.hidden, q.hc_count, q.hc_low_rank);
            run_qwen38_hc_f32_debug(rt_.token_embd, token, hc_attn, hc_f32_scratch, q.rms_eps, stream);
            trace->hc_f32.init = copy_float(stream, hc_f32_scratch.residual(), static_cast<size_t>(hc_dim));
            trace->hc_f32.norm = copy_float(stream, hc_f32_scratch.norm(), static_cast<size_t>(hc_dim));
            trace->hc_f32.low_raw = copy_float(stream, hc_f32_scratch.low_raw(), static_cast<size_t>(q.hc_low_rank));
            trace->hc_f32.low = copy_float(stream, hc_f32_scratch.low(), static_cast<size_t>(q.hc_low_rank));
            trace->hc_f32.gate = copy_float(stream, hc_f32_scratch.gate(), static_cast<size_t>(hc_dim));
            trace->hc_f32.mixed = copy_float(stream, hc_f32_scratch.mixed(), static_cast<size_t>(q.hidden));
        }

        for (int il = 0; il < q.trunk_layers; ++il) {
            Qwen38RuntimeLayer & layer = rt_.layers[static_cast<size_t>(il)];
            Qwen38LayerState & st = state_.layer(il);

            if (q.is_ple(il)) {
                if (!rt_.ple_table || !layer.ple_w || !st.ple_state || !st.ple_scratch) {
                    throw std::runtime_error("Qwen3.8 PLE runtime state is incomplete");
                }
                run_qwen38_ple(
                    state_.residual(), token, history_, pos,
                    *layer.ple_w, *rt_.ple_table,
                    *st.ple_state, *st.ple_scratch,
                    q, stream);
            }

            __half * cur = run_qwen38_hc_mix(
                state_.residual(), layer.hc_attn, st.hc_attn, q.rms_eps, stream);
            if (trace && il == 0) {
                trace->hc_attn_norm = copy_half(stream, st.hc_attn.xn(), static_cast<size_t>(hc_dim));
                trace->hc_attn_low = copy_half(
                    stream, st.hc_attn.lo(), static_cast<size_t>(st.hc_attn.low_rank()));
                trace->hc_attn_gate = copy_half(stream, st.hc_attn.gate(), static_cast<size_t>(hc_dim));
                trace->hc_attn_mix = copy_half(stream, cur, static_cast<size_t>(q.hidden));
            }
            if (trace && trace->use_f32_hc_layer0 && il == 0) {
                const int blocks = (q.hidden + 255) / 256;
                q38_float_to_half_kernel<<<blocks, 256, 0, stream>>>(
                    hc_f32_scratch.mixed(), st.hc_attn.mixed(), q.hidden);
                cur = st.hc_attn.mixed();
                trace->hc_attn_mix_used = copy_half(stream, cur, static_cast<size_t>(q.hidden));
            }

            __half * attn_out = nullptr;
            if (layer.qsa) {
                if (!layer.qsa_w || !st.qsa_state || !st.qsa_scratch) {
                    throw std::runtime_error("Qwen3.8 QSA state is incomplete");
                }
                attn_out = run_qwen38_qsa(
                    cur, *layer.qsa_w, *st.qsa_state, *st.qsa_scratch,
                    q, il, pos, Qwen38Position4::text(pos), stream);
            } else {
                if (!layer.gdn_w || !st.gdn_state || !st.gdn_scratch) {
                    throw std::runtime_error("Qwen3.8 GDN state is incomplete");
                }
                attn_out = run_qwen38_gdn(
                    cur, *layer.gdn_w, *st.gdn_state, *st.gdn_scratch,
                    q, q.rms_eps, stream);
                if (trace && il == 0) {
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
            if (trace && il == 0) trace->attn_out = copy_half(stream, attn_out, static_cast<size_t>(q.hidden));
            run_qwen38_hc_combine(state_.residual(), attn_out, st.hc_attn, stream);
            if (trace && il == 0) trace->after_attn = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));

            cur = run_qwen38_hc_mix(
                state_.residual(), layer.hc_ffn, st.hc_ffn, q.rms_eps, stream);
            if (trace && il == 0) trace->hc_ffn_mix = copy_half(stream, cur, static_cast<size_t>(q.hidden));

            if (rt_.fused_experts) {
                if (!layer.routed.gate_up || !layer.routed.down) {
                    throw std::runtime_error("Qwen3.8 fused expert tensor set is incomplete");
                }
                run_qwen38_routed_moe_fused(
                    il, cur, layer.router,
                    *layer.routed.gate_up, *layer.routed.down,
                    rt_.fused_expert_cache, st.moe, st.moe_out,
                    q.expert_weights_scale, stream);
            } else {
                if (!layer.routed.gate || !layer.routed.up || !layer.routed.down) {
                    throw std::runtime_error("Qwen3.8 expert tensor set is incomplete");
                }
                run_qwen38_routed_moe_separate(
                    il, cur, layer.router,
                    *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                    rt_.expert_cache, st.moe, st.moe_out,
                    q.expert_weights_scale, stream);
            }
            if (trace && il == 0) {
                trace->router_logits = copy_float(stream, st.moe.d_logits(), static_cast<size_t>(q.experts));
                trace->router_weights.assign(st.moe.h_weights(), st.moe.h_weights() + q.experts_used);
                trace->expert_ids.assign(st.moe.h_ids(), st.moe.h_ids() + q.experts_used);
            }

            run_qwen38_shared_expert(cur, layer.shared, st.moe, st.moe_out, stream);
            if (trace && il == 0) trace->ffn_out = copy_half(stream, st.moe_out, static_cast<size_t>(q.hidden));
            run_qwen38_hc_combine(state_.residual(), st.moe_out, st.hc_ffn, stream);
            if (trace && il == 0) trace->after_ffn = copy_half(stream, state_.residual(), static_cast<size_t>(hc_dim));
        }

        __half * head = run_qwen38_hc_mix(
            state_.residual(), rt_.hc_head, state_.head_hc(), q.rms_eps, stream);
        launch_lowbit_gemv_float(rt_.output, head, logits_, stream);
        q38_argmax_kernel<<<1, 256, 0, stream>>>(
            logits_, rt_.vocab, next_token_, has_nonfinite_);

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
        if (*host_has_nonfinite_) throw std::runtime_error("Qwen3.8 output logits contain NaN or infinity");

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
            Qwen38Trace * trace = nullptr) {
        if (steps <= 0) throw std::runtime_error("Qwen3.8 steps must be positive");
        clear();

        cudaEvent_t begin{}, end{};
        cudaError_t e = cudaEventCreate(&begin);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventCreate begin failed");
        e = cudaEventCreate(&end);
        if (e != cudaSuccess) {
            cudaEventDestroy(begin);
            throw std::runtime_error("cudaEventCreate end failed");
        }

        int token = start_token;
        e = cudaEventRecord(begin, state_.stream());
        if (e != cudaSuccess) throw std::runtime_error("cudaEventRecord begin failed");

        for (int pos = 0; pos < steps; ++pos) {
            token = step(token, pos, pos == 0 ? trace : nullptr);
            if (pos == 0 && first_top_logits && top_k > 0) {
                *first_top_logits = top_logits(top_k);
            }
            if (generated) generated->push_back(token);
        }

        e = cudaEventRecord(end, state_.stream());
        if (e != cudaSuccess) throw std::runtime_error("cudaEventRecord end failed");
        e = cudaEventSynchronize(end);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventSynchronize end failed");

        float ms = 0.0f;
        e = cudaEventElapsedTime(&ms, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventElapsedTime failed");
        return static_cast<double>(ms) / 1000.0;
    }

private:
    static std::vector<float> copy_half(cudaStream_t stream, const __half * data, size_t n) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error("sync Qwen3.8 trace stream failed");
        std::vector<__half> half(n);
        e = cudaMemcpy(
            half.data(), data, n * sizeof(__half), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) throw std::runtime_error("copy Qwen3.8 trace tensor failed");
        std::vector<float> out(n);
        std::transform(half.begin(), half.end(), out.begin(), [](__half x) {
            return __half2float(x);
        });
        return out;
    }

    static std::vector<float> copy_float(cudaStream_t stream, const float * data, size_t n) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error("sync Qwen3.8 trace stream failed");
        std::vector<float> out(n);
        e = cudaMemcpy(
            out.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) throw std::runtime_error("copy Qwen3.8 trace tensor failed");
        return out;
    }

    Qwen38Runtime & rt_;
    Qwen38DecodeState & state_;
    float * logits_ = nullptr;
    int * next_token_ = nullptr;
    int * has_nonfinite_ = nullptr;
    int * host_next_token_ = nullptr;
    int * host_has_nonfinite_ = nullptr;
    std::vector<int> history_;
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
