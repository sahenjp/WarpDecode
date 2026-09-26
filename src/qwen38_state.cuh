#pragma once

#include "qwen38_gdn.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"
#include "qwen38_moe.cuh"
#include "qwen38_ple.cuh"
#include "qwen38_qsa.cuh"
#include "qwen38_runtime.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace wd {

struct Qwen38LayerState {
    Qwen38HcScratch hc_attn;
    Qwen38HcScratch hc_ffn;
    Qwen38MoeScratch moe;

    std::unique_ptr<Qwen38QsaState> qsa_state;
    std::unique_ptr<Qwen38QsaScratch> qsa_scratch;

    std::unique_ptr<Qwen38GdnState> gdn_state;
    std::unique_ptr<Qwen38GdnScratch> gdn_scratch;

    std::unique_ptr<Qwen38PleState> ple_state;
    std::unique_ptr<Qwen38PleScratch> ple_scratch;

    __half * moe_out = nullptr;

    Qwen38LayerState() = default;
    Qwen38LayerState(const Qwen38LayerState &) = delete;
    Qwen38LayerState & operator=(const Qwen38LayerState &) = delete;

    ~Qwen38LayerState() {
        if (moe_out) cudaFree(moe_out);
    }
};

class Qwen38DecodeState {
public:
    Qwen38DecodeState() = default;
    Qwen38DecodeState(const Qwen38DecodeState &) = delete;
    Qwen38DecodeState & operator=(const Qwen38DecodeState &) = delete;

    ~Qwen38DecodeState() {
        if (residual_) cudaFree(residual_);
        if (head_hidden_) cudaFree(head_hidden_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void initialize(const Qwen38Runtime & rt, int max_ctx) {
        if (residual_) throw std::runtime_error("Qwen3.8 decode state already initialized");
        if (max_ctx <= 0) throw std::runtime_error("invalid Qwen3.8 max context");

        const auto & q = rt.layout;
        max_ctx_ = max_ctx;

        cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (e != cudaSuccess) throw std::runtime_error("cudaStreamCreate Qwen3.8 failed");

        const size_t residual_n = static_cast<size_t>(q.hidden) * q.hc_count;
        e = cudaMalloc(reinterpret_cast<void **>(&residual_), residual_n * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 residual failed");

        e = cudaMalloc(reinterpret_cast<void **>(&head_hidden_), static_cast<size_t>(q.hidden) * sizeof(__half));
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 head hidden failed");

        layers_.reserve(static_cast<size_t>(q.trunk_layers));
        for (int il = 0; il < q.trunk_layers; ++il) {
            auto st = std::make_unique<Qwen38LayerState>();
            st->hc_attn.initialize(q.hidden, q.hc_count, q.hc_low_rank);
            st->hc_ffn.initialize(q.hidden, q.hc_count, q.hc_low_rank);
            st->moe.initialize(q.experts, q.experts_used, q.hidden, q.expert_ffn);

            e = cudaMalloc(reinterpret_cast<void **>(&st->moe_out), static_cast<size_t>(q.hidden) * sizeof(__half));
            if (e != cudaSuccess) throw std::runtime_error("cudaMalloc Qwen3.8 MoE output failed");

            if (q.is_qsa(il)) {
                const int ratio = q.compress_ratios.empty()
                    ? q.full_attention_interval
                    : static_cast<int>(q.compress_ratios[static_cast<size_t>(il)]);
                if (ratio <= 0) throw std::runtime_error("invalid QSA compression ratio");

                st->qsa_state = std::make_unique<Qwen38QsaState>();
                st->qsa_scratch = std::make_unique<Qwen38QsaScratch>();
                initialize_qwen38_qsa_state(*st->qsa_state, max_ctx, ratio, q);
                st->qsa_scratch->initialize(q);
            } else {
                st->gdn_state = std::make_unique<Qwen38GdnState>();
                st->gdn_scratch = std::make_unique<Qwen38GdnScratch>();
                st->gdn_state->initialize(q);
                st->gdn_scratch->initialize(q);
            }

            if (q.is_ple(il)) {
                st->ple_state = std::make_unique<Qwen38PleState>();
                st->ple_scratch = std::make_unique<Qwen38PleScratch>();
                st->ple_state->initialize(q);
                st->ple_scratch->initialize(q);
            }

            layers_.push_back(std::move(st));
        }

        head_hc_.initialize(q.hidden, q.hc_count, q.hc_low_rank);
        clear();
    }

    void clear() {
        if (!stream_) return;
        for (auto & st : layers_) {
            if (st->qsa_state) st->qsa_state->clear(stream_);
            if (st->gdn_state) st->gdn_state->clear(stream_);
            if (st->ple_state) st->ple_state->clear(stream_);
        }
        cudaMemsetAsync(residual_, 0,
            residual_elements() * sizeof(__half), stream_);
        cudaStreamSynchronize(stream_);
    }

    cudaStream_t stream() const { return stream_; }
    int max_ctx() const { return max_ctx_; }
    __half * residual() const { return residual_; }
    __half * head_hidden() const { return head_hidden_; }
    Qwen38HcScratch & head_hc() { return head_hc_; }

    Qwen38LayerState & layer(int il) {
        if (il < 0 || il >= static_cast<int>(layers_.size())) {
            throw std::runtime_error("Qwen3.8 layer state index out of range");
        }
        return *layers_[static_cast<size_t>(il)];
    }

private:
    size_t residual_elements() const {
        if (layers_.empty()) return 0;
        return static_cast<size_t>(layers_.front()->hc_attn.hidden())
             * layers_.front()->hc_attn.hc();
    }

    int max_ctx_ = 0;
    cudaStream_t stream_ = nullptr;
    __half * residual_ = nullptr;
    __half * head_hidden_ = nullptr;
    Qwen38HcScratch head_hc_;
    std::vector<std::unique_ptr<Qwen38LayerState>> layers_;
};

} // namespace wd
