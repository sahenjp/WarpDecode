#pragma once

#include "gguf.hpp"
#include "gpu_arch.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>

namespace wd {

struct ModelProfile {
    std::string family = "unknown";
    std::string display_name;
    std::string arch;
    std::string quant_hint = "?";
    bool qwen38_flash_next = false;
    bool runnable = false;

    int layers = 0;
    int hidden = 0;
    int experts = 0;
    int routed_experts = 0;
    int shared_experts = 0;
    int gdn_layers = 0;
    int qsa_layers = 0;
    int residual_streams = 0;
    int mtp_layers = 0;

    std::string decode_path;
    std::string qsa_path;
    std::string ple_path;
    std::string moe_path;
    std::string mtp_path;
};

inline std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline std::string quant_from_filename(const std::filesystem::path& path) {
    const std::string s = lower_ascii(path.filename().string());
    const char* known[] = {
        "iq3e", "iq4_nl", "ud-iq4_xs", "ud-q3_k_xl", "ud-q2_k_xl",
        "q8d", "q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "q3_k", "q2_k",
        "nvfp4", "fp8", "f16"
    };
    for (const char* q : known) {
        if (s.find(q) != std::string::npos) return q;
    }
    return "?";
}

inline uint64_t uint_meta_or(const GgufFile& g, const std::string& key, uint64_t fallback) {
    if (!g.has_meta(key)) return fallback;
    try { return g.uint_meta(key); } catch (...) { return fallback; }
}

inline ModelProfile profile_model(const GgufFile& g, const std::filesystem::path& path, int sm = 0) {
    ModelProfile p;
    p.arch = g.string_meta_or("general.architecture", "?");
    p.display_name = g.string_meta_or("general.name", path.stem().string());
    p.quant_hint = quant_from_filename(path);

    const std::string a = lower_ascii(p.arch);
    const std::string n = lower_ascii(p.display_name + " " + path.filename().string());
    p.qwen38_flash_next = (a == "qwen4exp") || (n.find("qwen3.8-flash-next") != std::string::npos);

    if (p.qwen38_flash_next) {
        p.family = "qwen3.8-flash-next";
        p.layers = static_cast<int>(uint_meta_or(g, "qwen4exp.block_count", 48));
        p.hidden = static_cast<int>(uint_meta_or(g, "qwen4exp.embedding_length", 2560));
        p.experts = static_cast<int>(uint_meta_or(g, "qwen4exp.expert_count", 512));
        p.routed_experts = static_cast<int>(uint_meta_or(g, "qwen4exp.expert_used_count", 10));
        p.shared_experts = 1;
        p.gdn_layers = (p.layers / 4) * 3;
        p.qsa_layers = p.layers - p.gdn_layers;
        p.residual_streams = 4;
        p.mtp_layers = static_cast<int>(uint_meta_or(g, "qwen4exp.nextn_predict_layers", 1));

        const auto gpu = kernel_plan_for_sm(sm);
        p.decode_path = sm > 0 ? std::string(gpu.decode) + "+grouped-moe" : "grouped-lowbit-moe";
        p.qsa_path = "cached-block-index+compact-gather";
        p.ple_path = "direct-read+dedup+coalesce+hot-row-cache";
        p.moe_path = "hot-expert-slot-cache+10routed+1shared";
        p.mtp_path = "adaptive-mtp2-4+M2/M4-verify";
        p.runnable = false; // planner support only until qwen4exp kernels are wired end-to-end.
        return p;
    }

    if (a == "llama" || a == "qwen2" || a == "qwen3") {
        p.family = "dense";
        p.runnable = true;
        p.decode_path = sm > 0 ? kernel_plan_for_sm(sm).decode : "q4-simt";
        p.qsa_path = "none";
        p.ple_path = "none";
        p.moe_path = "none";
        p.mtp_path = "none";
        return p;
    }

    p.family = "unknown";
    return p;
}

inline std::string profile_summary(const ModelProfile& p) {
    std::ostringstream o;
    o << p.family;
    if (p.qwen38_flash_next) {
        o << "  " << p.layers << "L " << p.gdn_layers << "GDN+" << p.qsa_layers << "QSA"
          << "  MoE " << p.routed_experts << "+" << p.shared_experts << "/" << p.experts
          << "  MTP" << p.mtp_layers;
    }
    return o.str();
}

} // namespace wd
