#pragma once

#include "gguf.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38Layout {
    int total_blocks = 0;
    int mtp_layers = 0;
    int trunk_layers = 0;

    int hidden = 0;
    int heads = 0;
    int kv_heads = 0;
    int key_length = 0;
    int value_length = 0;

    int experts = 0;
    int experts_used = 0;
    int expert_ffn = 0;
    int shared_ffn = 0;

    int ssm_conv_kernel = 0;
    int ssm_state_size = 0;
    int ssm_group_count = 0;
    int ssm_dt_rank = 0;
    int ssm_inner_size = 0;

    int full_attention_interval = 0;
    int hc_count = 0;
    int hc_low_rank = 0;

    int indexer_heads = 0;
    int indexer_key_length = 0;
    int indexer_top_k = 0;

    int ple_ngram_size = 0;
    int ple_heads_per_ngram = 0;
    int ple_conv_kernel = 0;
    int ple_embedding_length = 0;

    std::vector<int64_t> compress_ratios;
    std::vector<uint64_t> ple_layers;
    std::vector<uint64_t> ple_layer_multipliers;
    std::vector<uint64_t> ple_head_offsets;
    std::vector<uint64_t> ple_head_vocab_sizes;

    int qsa_layer_count() const {
        int n = 0;
        for (int il = 0; il < trunk_layers; ++il) if (is_qsa(il)) ++n;
        return n;
    }

    int gdn_layer_count() const {
        return trunk_layers - qsa_layer_count();
    }

    bool is_mtp(int il) const {
        return il >= trunk_layers && il < total_blocks;
    }

    bool is_qsa(int il) const {
        if (il < 0 || il >= trunk_layers) return false;
        if (compress_ratios.size() == static_cast<size_t>(total_blocks)) {
            return compress_ratios[static_cast<size_t>(il)] > 0;
        }
        return full_attention_interval > 0 && ((il + 1) % full_attention_interval == 0);
    }

    bool is_gdn(int il) const {
        return il >= 0 && il < trunk_layers && !is_qsa(il);
    }

    bool is_ple(int il) const {
        return std::find(ple_layers.begin(), ple_layers.end(), static_cast<uint64_t>(il)) != ple_layers.end();
    }

    int ple_head_count() const {
        return ple_ngram_size > 0 ? (ple_ngram_size - 1) * ple_heads_per_ngram : 0;
    }
};

inline int q38_u32(const GgufFile & g, const char * key) {
    return static_cast<int>(g.uint_meta(key));
}

inline Qwen38Layout qwen38_layout_from_gguf(const GgufFile & g) {
    if (g.string_meta("general.architecture") != "qwen4exp") {
        throw std::runtime_error("qwen38 layout requires general.architecture=qwen4exp");
    }

    Qwen38Layout q;
    q.total_blocks = q38_u32(g, "qwen4exp.block_count");
    q.mtp_layers = static_cast<int>(g.number_meta("qwen4exp.nextn_predict_layers", 0));
    q.trunk_layers = q.total_blocks - q.mtp_layers;

    q.hidden = q38_u32(g, "qwen4exp.embedding_length");
    q.heads = q38_u32(g, "qwen4exp.attention.head_count");
    q.kv_heads = q38_u32(g, "qwen4exp.attention.head_count_kv");
    q.key_length = q38_u32(g, "qwen4exp.attention.key_length");
    q.value_length = q38_u32(g, "qwen4exp.attention.value_length");

    q.experts = q38_u32(g, "qwen4exp.expert_count");
    q.experts_used = q38_u32(g, "qwen4exp.expert_used_count");
    q.expert_ffn = q38_u32(g, "qwen4exp.expert_feed_forward_length");
    q.shared_ffn = q38_u32(g, "qwen4exp.expert_shared_feed_forward_length");

    q.ssm_conv_kernel = q38_u32(g, "qwen4exp.ssm.conv_kernel");
    q.ssm_state_size = q38_u32(g, "qwen4exp.ssm.state_size");
    q.ssm_group_count = q38_u32(g, "qwen4exp.ssm.group_count");
    q.ssm_dt_rank = q38_u32(g, "qwen4exp.ssm.time_step_rank");
    q.ssm_inner_size = q38_u32(g, "qwen4exp.ssm.inner_size");

    q.full_attention_interval = static_cast<int>(g.number_meta("qwen4exp.full_attention_interval", 4));
    q.hc_count = q38_u32(g, "qwen4exp.hyper_connection.count");
    q.hc_low_rank = q38_u32(g, "qwen4exp.hyper_connection.low_rank");

    q.indexer_heads = q38_u32(g, "qwen4exp.attention.indexer.head_count");
    q.indexer_key_length = q38_u32(g, "qwen4exp.attention.indexer.key_length");
    q.indexer_top_k = q38_u32(g, "qwen4exp.attention.indexer.top_k");

    if (g.has_meta("qwen4exp.attention.compress_ratios")) {
        q.compress_ratios = g.int_array_meta("qwen4exp.attention.compress_ratios");
    }

    if (g.has_meta("qwen4exp.ple.layers")) {
        q.ple_layers = g.uint_array_meta("qwen4exp.ple.layers");
        q.ple_ngram_size = q38_u32(g, "qwen4exp.ple.ngram_size");
        q.ple_heads_per_ngram = q38_u32(g, "qwen4exp.ple.heads_per_ngram");
        q.ple_conv_kernel = q38_u32(g, "qwen4exp.ple.conv_kernel");
        q.ple_embedding_length = q38_u32(g, "qwen4exp.embedding_length_per_layer_input");
        q.ple_layer_multipliers = g.uint_array_meta("qwen4exp.ple.layer_multipliers");
        q.ple_head_offsets = g.uint_array_meta("qwen4exp.ple.head_offsets");
        q.ple_head_vocab_sizes = g.uint_array_meta("qwen4exp.ple.head_vocab_sizes");
    }

    if (q.total_blocks <= 0 || q.trunk_layers <= 0 || q.mtp_layers < 0 ||
        q.hidden <= 0 || q.heads <= 0 || q.kv_heads <= 0 ||
        q.key_length <= 0 || q.value_length <= 0 ||
        q.experts <= 0 || q.experts_used <= 0 || q.experts_used > q.experts ||
        q.expert_ffn <= 0 || q.shared_ffn <= 0 ||
        q.ssm_conv_kernel <= 0 || q.ssm_state_size <= 0 ||
        q.ssm_group_count <= 0 || q.ssm_dt_rank <= 0 || q.ssm_inner_size <= 0 ||
        q.hc_count <= 1 || q.hc_low_rank <= 0 ||
        q.indexer_heads <= 0 || q.indexer_key_length <= 0 || q.indexer_top_k <= 0) {
        throw std::runtime_error("invalid qwen3.8 layout metadata");
    }

    if (!q.compress_ratios.empty() && q.compress_ratios.size() != static_cast<size_t>(q.total_blocks)) {
        throw std::runtime_error("qwen4exp.attention.compress_ratios length mismatch");
    }

    if (!q.ple_layers.empty()) {
        if (q.ple_ngram_size < 2 || q.ple_heads_per_ngram <= 0 ||
            q.ple_conv_kernel <= 0 || q.ple_embedding_length <= 0) {
            throw std::runtime_error("invalid qwen3.8 PLE metadata");
        }
        if (q.ple_layer_multipliers.size() < static_cast<size_t>(q.ple_ngram_size)) {
            throw std::runtime_error("PLE layer multiplier count is too small");
        }
        const size_t heads = static_cast<size_t>(q.ple_head_count());
        if (q.ple_head_offsets.size() < heads || q.ple_head_vocab_sizes.size() < heads) {
            throw std::runtime_error("PLE head range metadata is too small");
        }
        for (uint64_t il : q.ple_layers) {
            if (il >= static_cast<uint64_t>(q.trunk_layers)) {
                throw std::runtime_error("PLE layer is outside trunk");
            }
            if (!q.is_gdn(static_cast<int>(il))) {
                throw std::runtime_error("PLE must be attached to a recurrent/GDN layer");
            }
        }
    }

    return q;
}

} // namespace wd
