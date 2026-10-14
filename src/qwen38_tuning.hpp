#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace wd {

inline bool qwen38_tuning_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline std::vector<size_t> qwen38_tuning_env_slots(const char * name) {
    std::vector<size_t> out;
    const char * v = std::getenv(name);
    if (v == nullptr) return out;
    const char * p = v;
    while (*p != '\0') {
        char * end = nullptr;
        const long long n = std::strtoll(p, &end, 10);
        if (end == p) break;
        if (n > 0) out.push_back(static_cast<size_t>(n));
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

struct Qwen38Tuning {
    int qsa_block_budget = 512;
    int qsa_token_budget = 2048;
    bool qsa_persistent_index_cache = true;
    bool qsa_compact_kv_gather = true;

    int ple_prefetch_rows = 32;
    int ple_hot_cache_mib = 128;
    int ple_max_coalesce_gap = 1;

    int expert_cache_mib = 0;
    int expert_prefetch_depth = 1;
    bool expert_frequency_prefetch = true;

    // Task K: expert read-ahead. The memory tier below picks the look-ahead depth
    // (expert_prefetch_depth) and expert_prefetch is the master switch that the
    // runtime hands to the caches; the pipeline stays off until the environment
    // turns it on, because how much it helps depends on the cache size.
    bool expert_prefetch = false;      // WARODECODE_EXPERT_PREFETCH=1
    int expert_prefetch_threads = 2;   // WARODECODE_EXPERT_PREFETCH_THREADS
    int expert_prefetch_slots = 0;     // WARODECODE_EXPERT_PREFETCH_SLOTS (0 = auto)

    // Task C: expert cache / host->device transfer switches. These mirror the
    // ExpertCacheOptions fields of src/expert_cache.cuh (which is the consumer);
    // qwen38_tuning_for_gpu() reads the environment and the runtime hands the
    // result to both expert caches.
    bool expert_stats = false;          // WARODECODE_EXPERT_STATS=1
    bool expert_copy_stream = true;     // WARODECODE_EXPERT_COPY_STREAM=0 disables
    bool expert_layer_evict = false;    // WARODECODE_EXPERT_EVICT=layer
    std::vector<size_t> expert_shadow_slots;  // WARODECODE_EXPERT_SHADOW=a,b,..

    std::array<int, 3> mtp_draft_candidates{2, 3, 4};
    double mtp_reward_ewma = 0.25;
};

inline Qwen38Tuning qwen38_tuning_for_gpu(int sm, int memory_mib) {
    Qwen38Tuning t;

    // Keep the model semantics fixed. QSA's official sparse-attention budget is
    // 512 micro-blocks / 2048 tokens; only the caching and staging strategy is tuned.
    t.qsa_block_budget = 512;
    t.qsa_token_budget = 2048;

    if (memory_mib >= 120 * 1024) {
        t.ple_hot_cache_mib = 512;
        t.expert_cache_mib = 16 * 1024;
        t.expert_prefetch_depth = 3;
    } else if (memory_mib >= 72 * 1024) {
        t.ple_hot_cache_mib = 256;
        t.expert_cache_mib = 8 * 1024;
        t.expert_prefetch_depth = 2;
    } else if (memory_mib >= 40 * 1024) {
        t.ple_hot_cache_mib = 192;
        t.expert_cache_mib = 4 * 1024;
        t.expert_prefetch_depth = 2;
    } else if (memory_mib >= 20 * 1024) {
        t.ple_hot_cache_mib = 128;
        t.expert_cache_mib = 1536;
        t.expert_prefetch_depth = 1;
    } else {
        t.ple_hot_cache_mib = 64;
        t.expert_cache_mib = 512;
        t.expert_prefetch_depth = 1;
    }

    // Diagnostic override: the cache holds roughly 1.7 layers at the 1536 MiB
    // default, so the first eviction happens inside layer 2. Bumping it moves
    // that boundary and makes cache-related divergence easy to spot.
    if (const char * v = std::getenv("WARODECODE_EXPERT_CACHE_MIB")) {
        const int mib = std::atoi(v);
        if (mib > 0) t.expert_cache_mib = mib;
    }

    // Task C switches (see src/expert_cache.cuh for the transfer path).
    t.expert_stats = qwen38_tuning_env_flag("WARODECODE_EXPERT_STATS", false);
    t.expert_copy_stream = qwen38_tuning_env_flag("WARODECODE_EXPERT_COPY_STREAM", true);
    if (const char * v = std::getenv("WARODECODE_EXPERT_EVICT")) {
        t.expert_layer_evict = std::string(v) == "layer";
    }
    t.expert_shadow_slots = qwen38_tuning_env_slots("WARODECODE_EXPERT_SHADOW");

    // Task K. WARODECODE_EXPERT_PREFETCH_DEPTH overrides the tier depth (and, by
    // being a positive depth, switches the pipeline on for one A/B run); a value
    // of 0 keeps it off. The runtime only forwards these to the caches, they are
    // the consumers that size and run the read-ahead ring.
    t.expert_prefetch = qwen38_tuning_env_flag("WARODECODE_EXPERT_PREFETCH", false);
    if (const char * v = std::getenv("WARODECODE_EXPERT_PREFETCH_DEPTH")) {
        const int depth = std::atoi(v);
        t.expert_prefetch_depth = depth > 0 ? depth : 0;
        if (depth > 0) t.expert_prefetch = true;
    }

    if (sm >= 90) t.ple_prefetch_rows = 64;
    else if (sm >= 80) t.ple_prefetch_rows = 48;
    else t.ple_prefetch_rows = 32;

    return t;
}

class AdaptiveMtpController {
public:
    explicit AdaptiveMtpController(double alpha = 0.25)
        : alpha_(std::clamp(alpha, 0.01, 1.0)) {}

    int choose() const {
        // Measure every candidate once before exploiting the best reward.
        for (size_t i = 0; i < arms_.size(); ++i) {
            if (arms_[i].samples == 0) return kDraft[i];
        }
        size_t best = 0;
        for (size_t i = 1; i < arms_.size(); ++i) {
            if (arms_[i].reward > arms_[best].reward) best = i;
        }
        return kDraft[best];
    }

    void observe(int draft_n, int accepted_target_tokens, double gpu_us) {
        if (gpu_us <= 0.0 || accepted_target_tokens < 0) return;
        const int idx = index_of(draft_n);
        if (idx < 0) return;
        const double sample = static_cast<double>(accepted_target_tokens) / gpu_us;
        Arm& a = arms_[static_cast<size_t>(idx)];
        a.reward = a.samples == 0 ? sample : alpha_ * sample + (1.0 - alpha_) * a.reward;
        ++a.samples;
    }

    double reward(int draft_n) const {
        const int idx = index_of(draft_n);
        return idx < 0 ? 0.0 : arms_[static_cast<size_t>(idx)].reward;
    }

private:
    struct Arm {
        double reward = 0.0;
        uint64_t samples = 0;
    };

    static constexpr std::array<int, 3> kDraft{2, 3, 4};

    static int index_of(int n) {
        for (size_t i = 0; i < kDraft.size(); ++i) {
            if (kDraft[i] == n) return static_cast<int>(i);
        }
        return -1;
    }

    double alpha_;
    std::array<Arm, 3> arms_{};
};

} // namespace wd
