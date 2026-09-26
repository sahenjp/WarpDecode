#pragma once

#include "lowbit.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_moe.cuh"
#include "qwen38_ple.cuh"
#include "qwen38_qsa.cuh"
#include "qwen38_runtime.cuh"
#include "qwen38_state.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

__global__ void q38_repeat_embedding_kernel(
        const __half * __restrict__ emb,
        __half * __restrict__ residual,
        int hidden,
        int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden * hc;
    if (i >= total) return;
    residual[i] = emb[i % hidden];
}

struct Q38MaxPair {
    float value;
    int index;
};

__global__ void q38_argmax_kernel(
        const float * __restrict__ logits,
        int n,
        int * __restrict__ out) {
    __shared__ Q38MaxPair smem[256];
    Q38MaxPair best{-3.402823466e+38F, 0};

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > best.value || (v == best.value && i < best.index)) {
            best = {v, i};
        }
    }
    smem[threadIdx.x] = best;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const Q38MaxPair other = smem[threadIdx.x + stride];
            if (other.value > smem[threadIdx.x].value ||
                (other.value == smem[threadIdx.x].value &&
                 other.index < smem[threadIdx.x].index)) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) *out = smem[0].index;
}

class Qwen38Decoder {
public:
    Qwen38Decoder(Qwen38Runtime & rt, Qwen38DecodeState & state)
        : rt_(rt), state_(state) {
        if (rt_.vocab <= 0) throw std::runtime_error("invalid Qwen3.8 vocabulary");
        cudaError_t e = cudaMalloc(
            reinterpret_cast<void **>(&logits_),
            static_cast<size_t>(rt_.vocab) * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error(
            std::string("cudaMalloc Qwen3.8 logits failed: ") + cudaGetErrorString(e));

        e = cudaMalloc(reinterpret_cast<void **>(&next_token_), sizeof(int));
        if (e != cudaSuccess) {
            cudaFree(logits_);
            logits_ = nullptr;
            throw std::runtime_error(
                std::string("cudaMalloc Qwen3.8 next token failed: ") + cudaGetErrorString(e));
        }
    }

    Qwen38Decoder(const Qwen38Decoder &) = delete;
    Qwen38Decoder & operator=(const Qwen38Decoder &) = delete;

    ~Qwen38Decoder() {
        if (logits_) cudaFree(logits_);
        if (next_token_) cudaFree(next_token_);
    }

    int step(int token, int pos) {
        if (token < 0 || token >= rt_.vocab) throw std::runtime_error("Qwen3.8 token out of range");
        if (pos < 0 || pos >= state_.max_ctx()) throw std::runtime_error("Qwen3.8 decode position out of range");

        const auto & q = rt_.layout;
        cudaStream_t stream = state_.stream();

        // qwen4exp begins with hc identical copies of the token embedding.
        launch_q8_0_gather_row(rt_.token_embd, token, state_.head_hidden(), stream);
        const int hc_dim = q.hidden * q.hc_count;
        q38_repeat_embedding_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
            state_.head_hidden(), state_.residual(), q.hidden, q.hc_count);

        for (int il = 0; il < q.trunk_layers; ++il) {
            auto & layer = rt_.layers[static_cast<size_t>(il)];
            auto & st = state_.layer(il);

            if (q.is_ple(il)) {
                if (!rt_.ple_table || !layer.ple_w || !st.ple_state || !st.ple_scratch) {
                    throw std::runtime_error("PLE layer runtime is incomplete");
                }
                run_qwen38_ple(
                    state_.residual(), token, token_history_, pos,
                    *layer.ple_w, *rt_.ple_table,
                    *st.ple_state, *st.ple_scratch,
                    q, stream);
            }

            __half * cur = run_qwen38_hc_mix(
                state_.residual(), layer.hc_attn, st.hc_attn, q.rms_eps, stream);

            __half * mixer_out = nullptr;
            if (layer.qsa) {
                if (!layer.qsa_w || !st.qsa_state || !st.qsa_scratch) {
                    throw std::runtime_error("QSA layer runtime is incomplete");
                }
                mixer_out = run_qwen38_qsa(
                    cur, *layer.qsa_w,
                    *st.qsa_state, *st.qsa_scratch,
                    q, il, pos, Qwen38Position4::text(pos), stream);
            } else {
                if (!layer.gdn_w || !st.gdn_state || !st.gdn_scratch) {
                    throw std::runtime_error("GDN layer runtime is incomplete");
                }
                mixer_out = run_qwen38_gdn(
                    cur, *layer.gdn_w,
                    *st.gdn_state, *st.gdn_scratch,
                    q, q.rms_eps, stream);
            }
            run_qwen38_hc_combine(
                state_.residual(), mixer_out, st.hc_attn, stream);

            cur = run_qwen38_hc_mix(
                state_.residual(), layer.hc_ffn, st.hc_ffn, q.rms_eps, stream);

            if (rt_.fused_experts) {
                if (!layer.routed.gate_up || !layer.routed.down) {
                    throw std::runtime_error("fused MoE tensor set is incomplete");
                }
                run_qwen38_routed_moe_fused(
                    il, cur, layer.router,
                    *layer.routed.gate_up, *layer.routed.down,
                    rt_.fused_expert_cache, st.moe, st.moe_out,
                    q.expert_weights_scale, stream);
            } else {
                if (!layer.routed.gate || !layer.routed.up || !layer.routed.down) {
                    throw std::runtime_error("separate MoE tensor set is incomplete");
                }
                run_qwen38_routed_moe_separate(
                    il, cur, layer.router,
                    *layer.routed.gate, *layer.routed.up, *layer.routed.down,
                    rt_.expert_cache, st.moe, st.moe_out,
                    q.expert_weights_scale, stream);
            }
            run_qwen38_shared_expert(
                cur, layer.shared, st.moe, st.moe_out, stream);
            run_qwen38_hc_combine(
                state_.residual(), st.moe_out, st.hc_ffn, stream);
        }

        __half * final_hidden = run_qwen38_hc_mix(
            state_.residual(), rt_.hc_head, state_.head_hc(), q.rms_eps, stream);

        launch_lowbit_gemv_float(rt_.output, final_hidden, logits_, stream);
        q38_argmax_kernel<<<1, 256, 0, stream>>>(logits_, rt_.vocab, next_token_);

        int next = -1;
        cudaError_t e = cudaMemcpyAsync(
            &next, next_token_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(
            std::string("copy Qwen3.8 next token failed: ") + cudaGetErrorString(e));
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) throw std::runtime_error(
            std::string("Qwen3.8 decode sync failed: ") + cudaGetErrorString(e));

        token_history_.push_back(token);
        const size_t keep = static_cast<size_t>(std::max(0, q.ple_ngram_size - 1));
        if (keep > 0 && token_history_.size() > keep) {
            token_history_.erase(
                token_history_.begin(),
                token_history_.begin() + static_cast<ptrdiff_t>(token_history_.size() - keep));
        }
        return next;
    }

    double run(int start_token, int steps, std::vector<int> * generated = nullptr) {
        if (steps <= 0) throw std::runtime_error("Qwen3.8 decode step count must be positive");
        if (steps >= state_.max_ctx()) throw std::runtime_error("Qwen3.8 decode exceeds context");

        state_.clear();
        token_history_.clear();

        cudaEvent_t begin{}, end{};
        cudaError_t e = cudaEventCreate(&begin);
        if (e != cudaSuccess) throw std::runtime_error("cudaEventCreate Qwen3.8 begin failed");
        e = cudaEventCreate(&end);
        if (e != cudaSuccess) {
            cudaEventDestroy(begin);
            throw std::runtime_error("cudaEventCreate Qwen3.8 end failed");
        }

        cudaEventRecord(begin, state_.stream());
        int token = start_token;
        if (generated) generated->clear();

        for (int pos = 0; pos < steps; ++pos) {
            token = step(token, pos);
            if (generated) generated->push_back(token);
        }

        cudaEventRecord(end, state_.stream());
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        return static_cast<double>(ms) / 1000.0;
    }

private:
    Qwen38Runtime & rt_;
    Qwen38DecodeState & state_;
    float * logits_ = nullptr;
    int * next_token_ = nullptr;
    std::vector<int> token_history_;
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
