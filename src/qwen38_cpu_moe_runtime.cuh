#pragma once

// ---------------------------------------------------------------------------
// Task N: CUDA-side driver for the CPU MoE path (src/qwen38_cpu_moe.cuh).
//
// This is the only file that knows both worlds. It keeps the per-layer weight
// view over the GGUF mapping, hosts the CPU thread pool, and moves the three
// small buffers that have to cross the bus per layer:
//
//   hidden row (F32)  -> host    10,240 B   (so the CPU can quantise it)
//   expert ids + w    -> host       56 B
//   routed sum (F32)  <- device   10,240 B
//
// The expert *weights* never move: the CPU reads them out of the mapping. That
// is the whole point of the task - it is the difference between the engine's
// 6.15 tok/s (expert staging per token) and llama.cpp's 12.5-23.1 tok/s.
//
// Bit-identity with the GPU path is structural, not best-effort:
//
//   * the activation is quantised from the same F32 row the CUDA path uses
//     (the HC mix output), with the same QT8_K algorithm;
//   * every expert GEMV runs the transcription of the same AVX2 kernel the CUDA
//     DP4A kernels are bit-identical to (see the header comment there);
//   * the weighted sum is accumulated with std::fma in ascending expert order
//     into d_accum(), exactly like q38_weighted_accum_f32_kernel;
//   * the F32-mid contract is preserved: only d_accum() is written, so
//     q38_finish_moe() / the caller's shared expert keep working untouched.
// ---------------------------------------------------------------------------

#include "qwen38_cpu_moe.cuh"
#include "qwen38_moe.cuh"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

inline size_t q38_cpu_moe_expert_stride(const TensorInfo & t) {
    if (t.dims.size() != 3) {
        throw std::runtime_error("CPU MoE: expert tensor is not 3D: " + t.name);
    }
    const int cols = static_cast<int>(t.dims[0]);
    const int rows = static_cast<int>(t.dims[1]);
    const int type = cpu_moe::block_type_from_ggml(static_cast<int>(t.type));
    if (type == cpu_moe::kUnsupported) {
        throw std::runtime_error("CPU MoE: unsupported expert tensor type: " + t.name);
    }
    return cpu_moe::row_bytes(type, cols) * static_cast<size_t>(rows);
}

class Qwen38CpuMoe {
public:
    void configure(const Qwen38Layout & q, const std::vector<Qwen38MoeLayerTensors> & layers,
                   int cpu_layers, bool fused_experts) {
        if (cpu_layers <= 0) return;
        if (fused_experts) {
            throw std::runtime_error(
                "CPU MoE: the fused gate_up expert layout is not wired yet; "
                "WARODECODE_MOE_FUSED_GATE_UP=0 selects the separate tensors");
        }
        cpu_layers_ = std::min(cpu_layers, static_cast<int>(layers.size()));
        weights_.resize(static_cast<size_t>(cpu_layers_));
        for (int il = 0; il < cpu_layers_; ++il) {
            const Qwen38MoeLayerTensors & src = layers[static_cast<size_t>(il)];
            if (src.gate == nullptr || src.up == nullptr || src.down == nullptr) {
                throw std::runtime_error("CPU MoE: expert tensor set is incomplete");
            }
            cpu_moe::LayerWeights & w = weights_[static_cast<size_t>(il)];
            w.hidden = q.hidden;
            w.expert_ffn = q.expert_ffn;
            w.gate_type = cpu_moe::block_type_from_ggml(static_cast<int>(src.gate->type));
            w.up_type = cpu_moe::block_type_from_ggml(static_cast<int>(src.up->type));
            w.down_type = cpu_moe::block_type_from_ggml(static_cast<int>(src.down->type));
            if (w.gate_type == cpu_moe::kUnsupported || w.up_type == cpu_moe::kUnsupported ||
                w.down_type == cpu_moe::kUnsupported) {
                throw std::runtime_error("CPU MoE: expert tensor type is not supported on the CPU");
            }
            if (cpu_moe::block_elems(w.gate_type) != cpu_moe::kQK_K ||
                cpu_moe::block_elems(w.up_type) != cpu_moe::kQK_K) {
                throw std::runtime_error("CPU MoE: expert gate/up weight is not IQ2_S/IQ3_S");
            }
            if (cpu_moe::block_elems(w.down_type) == cpu_moe::kQK_K) {
                throw std::runtime_error("CPU MoE: expert down weight is not a 32-value block type");
            }
            w.gate = src.gate->data;
            w.up = src.up->data;
            w.down = src.down->data;
            w.gate_expert_stride = q38_cpu_moe_expert_stride(*src.gate);
            w.up_expert_stride = q38_cpu_moe_expert_stride(*src.up);
            w.down_expert_stride = q38_cpu_moe_expert_stride(*src.down);
        }
        scratch_.resize(static_cast<size_t>(cpu_layers_));
        for (cpu_moe::LayerScratch & s : scratch_) {
            s.reset(q.experts_used, q.hidden, q.expert_ffn);
        }
        hidden_host_.assign(static_cast<size_t>(q.hidden), 0.0f);
        ids_host_.assign(static_cast<size_t>(q.experts_used), 0);
        weights_host_.assign(static_cast<size_t>(q.experts_used), 0.0f);
        pool_ = std::make_unique<cpu_moe::ThreadPool>(cpu_moe::thread_count());
        configured_ = true;
    }

    bool active_for(int layer) const {
        return configured_ && cpu_moe::layer_is_cpu(layer, cpu_layers_);
    }

    int layers() const { return cpu_layers_; }

    // Runs the routed experts of one layer on the CPU and leaves the weighted
    // sum in scratch.d_accum(), i.e. the pre-finish state the CUDA runners hand
    // to q38_finish_moe() / q38_finish_moe_f32().
    void run_layer(int layer, const float * hidden_f32, Qwen38MoeScratch & moe,
                   cudaStream_t stream) {
        if (!active_for(layer)) {
            throw std::runtime_error("CPU MoE: layer is not configured for the CPU path");
        }
        cpu_moe::LayerWeights & w = weights_[static_cast<size_t>(layer)];
        cpu_moe::LayerScratch & scratch = scratch_[static_cast<size_t>(layer)];
        const int hidden = w.hidden;
        const int k = moe.k();

        cudaError_t e = cudaMemcpyAsync(hidden_host_.data(), hidden_f32,
                                        static_cast<size_t>(hidden) * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CPU MoE: hidden row download failed: ") +
                                     cudaGetErrorString(e));
        }
        e = cudaMemcpyAsync(ids_host_.data(), moe.d_ids(),
                            static_cast<size_t>(k) * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CPU MoE: expert id download failed: ") +
                                     cudaGetErrorString(e));
        }
        e = cudaMemcpyAsync(weights_host_.data(), moe.d_weights(),
                            static_cast<size_t>(k) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CPU MoE: expert weight download failed: ") +
                                     cudaGetErrorString(e));
        }
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CPU MoE: stream sync failed: ") +
                                     cudaGetErrorString(e));
        }

        std::vector<float> & out = scratch.shared.expert_out;  // reused as the MoE sum
        std::fill(out.begin(), out.begin() + hidden, 0.0f);
        cpu_moe::run_layer(w, *pool_, hidden_host_.data(), ids_host_.data(),
                           weights_host_.data(), k, scratch, out.data());

        e = cudaMemcpyAsync(moe.d_accum(), out.data(),
                            static_cast<size_t>(hidden) * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CPU MoE: routed sum upload failed: ") +
                                     cudaGetErrorString(e));
        }
    }

private:
    bool configured_ = false;
    int cpu_layers_ = 0;
    std::vector<cpu_moe::LayerWeights> weights_;
    std::vector<cpu_moe::LayerScratch> scratch_;
    std::unique_ptr<cpu_moe::ThreadPool> pool_;
    std::vector<float> hidden_host_;
    std::vector<int> ids_host_;
    std::vector<float> weights_host_;
};

}  // namespace wd
