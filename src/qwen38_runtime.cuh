#pragma once

#include "expert_cache.cuh"
#include "gguf.hpp"
#include "matrix.cuh"
#include "qwen38_gdn.cuh"
#include "qwen38_hc.cuh"
#include "qwen38_layout.hpp"
#include "qwen38_moe.cuh"
#include "qwen38_qsa.cuh"
#include "qwen38_tuning.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wd {

struct Qwen38RuntimeLayer {
    Qwen38HcWeights hc_attn;
    Qwen38HcWeights hc_ffn;

    bool qsa = false;
    std::unique_ptr<Qwen38QsaWeights> qsa_w;
    std::unique_ptr<Qwen38GdnWeights> gdn_w;

    LowBitMatrix router;
    Qwen38MoeLayerTensors routed;
    Qwen38SharedExpertWeights shared;
};

struct Qwen38Runtime {
    Qwen38Layout layout;
    Qwen38Tuning tuning;

    std::vector<Qwen38RuntimeLayer> layers;
    Qwen38HcWeights hc_head;

    GpuExpertCache expert_cache;
    GpuFusedExpertCache fused_expert_cache;
    bool fused_experts = false;

    size_t expert_cache_slots = 0;
    size_t resident_weight_bytes = 0;
};

inline void qwen38_check_router(
        const TensorInfo & t,
        const Qwen38Layout & q) {
    if (t.dims.size() != 2 ||
        static_cast<int>(t.dims[0]) != q.hidden ||
        static_cast<int>(t.dims[1]) != q.experts) {
        throw std::runtime_error("Qwen3.8 router shape mismatch: " + t.name);
    }
    if (!lowbit_gemv_supported(t.type)) {
        throw std::runtime_error("Qwen3.8 router type unsupported: " + t.name);
    }
}

inline void qwen38_check_expert_tensors(
        const Qwen38MoeLayerTensors & m,
        const Qwen38Layout & q,
        int il) {
    if (!m.router || !m.down) {
        throw std::runtime_error("Qwen3.8 MoE tensor set incomplete at layer " + std::to_string(il));
    }
    qwen38_check_router(*m.router, q);

    auto check3 = [&](const TensorInfo & t, int cols, int rows, const char * what) {
        if (t.dims.size() != 3 ||
            static_cast<int>(t.dims[0]) != cols ||
            static_cast<int>(t.dims[1]) != rows ||
            static_cast<int>(t.dims[2]) != q.experts) {
            throw std::runtime_error(
                std::string("Qwen3.8 ") + what + " shape mismatch at layer " +
                std::to_string(il) + ": " + t.name);
        }
        if (!lowbit_gemv_supported(t.type)) {
            throw std::runtime_error(
                std::string("Qwen3.8 ") + what + " type unsupported: " + t.name);
        }
    };

    check3(*m.down, q.expert_ffn, q.hidden, "expert down");
    if (m.fused_gate_up()) {
        check3(*m.gate_up, q.hidden, q.expert_ffn * 2, "expert gate-up");
    } else {
        if (!m.gate || !m.up) throw std::runtime_error("Qwen3.8 separate expert gate/up missing");
        check3(*m.gate, q.hidden, q.expert_ffn, "expert gate");
        check3(*m.up, q.hidden, q.expert_ffn, "expert up");
    }
}

inline Qwen38Runtime load_qwen38_runtime(
        const GgufFile & g,
        int sm,
        int memory_mib) {
    Qwen38Runtime rt;
    rt.layout = qwen38_layout_from_gguf(g);
    rt.tuning = qwen38_tuning_for_gpu(sm, memory_mib);

    if (rt.layout.trunk_layers != 48) {
        throw std::runtime_error(
            "Qwen3.8 fast path currently expects 48 trunk layers, got " +
            std::to_string(rt.layout.trunk_layers));
    }
    if (rt.layout.experts != 512 || rt.layout.experts_used != 10) {
        throw std::runtime_error("Qwen3.8 fast path currently expects 512 experts / top-10 routing");
    }

    rt.layers.reserve(static_cast<size_t>(rt.layout.trunk_layers));
    bool expert_mode_set = false;

    for (int il = 0; il < rt.layout.trunk_layers; ++il) {
        Qwen38RuntimeLayer layer;
        layer.hc_attn = load_qwen38_hc_weights(
            g, il, "attn", rt.layout.hidden, rt.layout.hc_count, rt.layout.hc_low_rank);
        layer.hc_ffn = load_qwen38_hc_weights(
            g, il, "ffn", rt.layout.hidden, rt.layout.hc_count, rt.layout.hc_low_rank);

        layer.qsa = rt.layout.is_qsa(il);
        if (layer.qsa) {
            layer.qsa_w = std::make_unique<Qwen38QsaWeights>(
                load_qwen38_qsa_weights(g, il, rt.layout));
        } else {
            layer.gdn_w = std::make_unique<Qwen38GdnWeights>(
                load_qwen38_gdn_weights(g, il, rt.layout));
        }

        layer.routed = qwen38_moe_tensors(g, il);
        qwen38_check_expert_tensors(layer.routed, rt.layout, il);
        const bool fused = layer.routed.fused_gate_up();
        if (!expert_mode_set) {
            rt.fused_experts = fused;
            expert_mode_set = true;
        } else if (rt.fused_experts != fused) {
            throw std::runtime_error("mixed fused/separate expert layout across Qwen3.8 layers");
        }

        layer.router = upload_lowbit_matrix(*layer.routed.router);
        layer.shared = load_qwen38_shared_expert(
            g, il, rt.layout.hidden, rt.layout.shared_ffn);

        rt.layers.push_back(std::move(layer));
    }

    rt.hc_head = load_qwen38_hc_head(
        g, rt.layout.hidden, rt.layout.hc_count, rt.layout.hc_low_rank);

    if (rt.layers.empty()) throw std::runtime_error("Qwen3.8 has no trunk layers");
    const auto & sample = rt.layers.front().routed;
    if (rt.fused_experts) {
        rt.fused_expert_cache.initialize(
            static_cast<size_t>(rt.tuning.expert_cache_mib),
            *sample.gate_up, *sample.down);
        rt.expert_cache_slots = rt.fused_expert_cache.slot_count();
    } else {
        rt.expert_cache.initialize(
            static_cast<size_t>(rt.tuning.expert_cache_mib),
            *sample.gate, *sample.up, *sample.down);
        rt.expert_cache_slots = rt.expert_cache.slot_count();
    }

    return rt;
}

} // namespace wd
