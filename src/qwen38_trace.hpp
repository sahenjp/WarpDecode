#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct Qwen38TraceVector {
    std::string name;
    std::vector<float> values;
};

struct Qwen38TraceStats {
    size_t count = 0;
    float min = 0.0f;
    float max = 0.0f;
    float max_abs = 0.0f;
    double mean = 0.0;
    double l2 = 0.0;
    double rms = 0.0;
    uint64_t checksum = 14695981039346656037ull;
};

struct Qwen38TraceSummaryVector {
    std::string name;
    std::vector<uint64_t> dims;
    Qwen38TraceStats stats;
};

struct Qwen38TraceIntegerVector {
    std::string name;
    std::vector<int64_t> values;
};

inline size_t qwen38_expert_trace_offset(size_t rank, size_t hidden) {
    if (hidden == 0 || rank > std::numeric_limits<size_t>::max() / hidden) {
        throw std::runtime_error("invalid Qwen3.8 expert trace offset");
    }
    return rank * hidden;
}

inline Qwen38TraceStats qwen38_trace_stats(const std::vector<float> & values) {
    Qwen38TraceStats out;
    out.count = values.size();
    if (values.empty()) return out;

    out.min = *std::min_element(values.begin(), values.end());
    out.max = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    double square_sum = 0.0;
    for (float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error("non-finite value in Qwen3.8 trace");
        sum += value;
        square_sum += static_cast<double>(value) * value;
        out.max_abs = std::max(out.max_abs, std::abs(value));
        const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) {
            out.checksum ^= bytes[i];
            out.checksum *= 1099511628211ull;
        }
    }
    out.mean = sum / static_cast<double>(values.size());
    out.l2 = std::sqrt(square_sum);
    out.rms = std::sqrt(square_sum / static_cast<double>(values.size()));
    return out;
}

inline void write_qwen38_trace_binary(
        const std::string & path,
        const std::vector<Qwen38TraceVector> & vectors) {
    static_assert(sizeof(float) == 4, "Qwen3.8 trace format requires float32");
    const uint32_t endian_check = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_check) != 1) {
        throw std::runtime_error("Qwen3.8 trace writer requires a little-endian host");
    }
    if (vectors.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("too many Qwen3.8 trace vectors");
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot open Qwen3.8 trace file: " + path);
    // WDPTRC1: magic, version, vector count, then (name length, value count, name, float32 data).
    constexpr char magic[8] = {'W', 'D', 'P', 'T', 'R', 'C', '1', '\0'};
    const uint32_t version = 1;
    const uint32_t count = static_cast<uint32_t>(vectors.size());
    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));

    for (const Qwen38TraceVector & vector : vectors) {
        if (vector.name.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Qwen3.8 trace name is too long");
        }
        const uint32_t name_size = static_cast<uint32_t>(vector.name.size());
        const uint64_t value_count = static_cast<uint64_t>(vector.values.size());
        file.write(reinterpret_cast<const char *>(&name_size), sizeof(name_size));
        file.write(reinterpret_cast<const char *>(&value_count), sizeof(value_count));
        file.write(vector.name.data(), static_cast<std::streamsize>(vector.name.size()));
        file.write(
            reinterpret_cast<const char *>(vector.values.data()),
            static_cast<std::streamsize>(vector.values.size() * sizeof(float)));
        if (!file) throw std::runtime_error("failed writing Qwen3.8 trace file: " + path);
    }
}

// ---------------------------------------------------------------------------
// Task F: per-token stage profiling.
//
// This header stays CUDA-free on purpose: tests/qwen38_trace_test.cpp compiles
// it with plain g++ and without a CUDA toolchain. Only the *shape* of the
// measurement lives here (stage ids, accumulators, JSON); the cudaEvent_t
// plumbing lives in qwen38_forward.cuh, which already includes cuda_runtime.h.
//
// Every stage below is measured as a pair of CUDA events recorded on the decode
// stream, so the device number is the GPU-timeline distance between the two
// records. When the GPU goes idle waiting for work (a synchronous host read, a
// launch gap, a cache miss, ...) that idle time stays inside the stage that
// follows the gap; the matching host number (wall clock around the same region)
// shows which side is waiting. Comparing the two is the point of the exercise.
// ---------------------------------------------------------------------------

// Stage ids of one decode token. kQwen38StageToken spans the whole step() body
// except the final device->host token copy and the stream synchronise, which
// kQwen38StageSync covers.
enum Qwen38Stage : int {
    kQwen38StageEmbed = 0,
    kQwen38StagePle = 1,
    kQwen38StageHcMixAttn = 2,
    kQwen38StageAttn = 3,
    kQwen38StageHcCombineAttn = 4,
    kQwen38StageHcMixFfn = 5,
    kQwen38StageMoe = 6,
    kQwen38StageHcCombineFfn = 7,
    kQwen38StageHead = 8,
    kQwen38StageSync = 9,
    kQwen38StageToken = 10,
    kQwen38StageCount = 11
};

inline const char * qwen38_stage_name(int stage) {
    switch (stage) {
        case kQwen38StageEmbed: return "embed";
        case kQwen38StagePle: return "ple";
        case kQwen38StageHcMixAttn: return "hc_mix_attn";
        case kQwen38StageAttn: return "attn";
        case kQwen38StageHcCombineAttn: return "hc_combine_attn";
        case kQwen38StageHcMixFfn: return "hc_mix_ffn";
        case kQwen38StageMoe: return "moe";
        case kQwen38StageHcCombineFfn: return "hc_combine_ffn";
        case kQwen38StageHead: return "head";
        case kQwen38StageSync: return "sync";
        case kQwen38StageToken: return "total";
        default: return "unknown";
    }
}

struct Qwen38StageSample {
    double device_ms = 0.0;   // sum of the flushed event intervals
    double host_ms = 0.0;     // sum of the host wall time of the same regions
    uint64_t calls = 0;       // flushed event intervals
    uint64_t unmatched = 0;   // recorded starts without a matching stop
    uint64_t overflow = 0;    // intervals dropped because the pool was full
};

struct Qwen38ProfileTotals {
    uint64_t tokens = 0;
    uint64_t phases = 0;      // tokens tagged as decode (not prefill)
    std::array<Qwen38StageSample, kQwen38StageCount> stage{};
    double token_host_ms = 0.0;   // host wall inside step(), D2H and sync included
    double flush_host_ms = 0.0;   // host cost of reading the events back
    std::vector<double> layer_device_ms;   // indexed by layer, shared by all tokens
    std::vector<double> layer_host_ms;
    uint64_t layer_calls = 0;
};

// Serialises the totals as one JSON object of *per-token averages* for the
// device and host numbers, plus the per-layer sums. "unit" states that so a
// reader never has to guess whether the number is a total or an average.
inline std::string qwen38_profile_json(const Qwen38ProfileTotals & t) {
    const double div = t.tokens > 0 ? static_cast<double>(t.tokens) : 1.0;
    auto fixed = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.4f", value);
        return std::string(buffer);
    };
    auto per_token = [&](double value) { return fixed(value / div); };

    std::string out;
    out.reserve(4096);
    // Every value here is numeric, so a reader may format every key without
    // inspecting it first. Units and interval counts live in the detail object.
    out += "{\"enabled\":1,\"tokens\":" + std::to_string(t.tokens);

    double stage_sum_ms = 0.0;
    for (int s = 0; s < kQwen38StageCount; ++s) {
        if (s == kQwen38StageToken || s == kQwen38StageSync) continue;
        stage_sum_ms += t.stage[static_cast<size_t>(s)].device_ms;
    }
    for (int s = 0; s < kQwen38StageCount; ++s) {
        if (s == kQwen38StageToken || s == kQwen38StageSync) continue;
        out += ",\"";
        out += qwen38_stage_name(s);
        out += "_ms\":" + per_token(t.stage[static_cast<size_t>(s)].device_ms);
    }
    out += ",\"total_ms\":" + per_token(t.stage[kQwen38StageToken].device_ms);
    out += ",\"sync_ms\":" + per_token(t.stage[kQwen38StageSync].device_ms);
    out += ",\"stage_sum_ms\":" + per_token(stage_sum_ms);
    // total = stage_sum + sync + other, so the breakdown always adds up.
    const double claimed_ms = stage_sum_ms + t.stage[kQwen38StageSync].device_ms;
    const double other_ms = t.stage[kQwen38StageToken].device_ms > claimed_ms
        ? t.stage[kQwen38StageToken].device_ms - claimed_ms : 0.0;
    out += ",\"other_ms\":" + per_token(other_ms);

    out += "}";
    return out;
}

// Everything that is not a per-token scalar: the host-side wall clock, the
// interval counts and the per-layer arrays. It is a separate JSON object so a
// reader that walks the keys of "profile" expecting plain numbers cannot trip
// over a list or a nested object.
inline std::string qwen38_profile_detail_json(const Qwen38ProfileTotals & t) {
    const double div = t.tokens > 0 ? static_cast<double>(t.tokens) : 1.0;
    auto fixed = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.4f", value);
        return std::string(buffer);
    };
    auto per_token = [&](double value) { return fixed(value / div); };

    std::string out;
    out.reserve(4096);
    out += "{\"unit\":\"ms_per_token\",\"tokens\":" + std::to_string(t.tokens);

    // The token stage spans the whole step, so it must stay out of the sum:
    // host_stage_sum_ms is the host time inside the individual regions.
    double host_stage_sum = 0.0;
    for (int s = 0; s < kQwen38StageCount; ++s) {
        if (s == kQwen38StageToken) continue;
        host_stage_sum += t.stage[static_cast<size_t>(s)].host_ms;
    }
    out += ",\"host_total_ms\":" + per_token(t.token_host_ms);
    out += ",\"host_flush_ms\":" + per_token(t.flush_host_ms);
    out += ",\"host_stage_sum_ms\":" + per_token(host_stage_sum);
    const double host_other = t.token_host_ms - host_stage_sum - t.flush_host_ms;
    out += ",\"host_other_ms\":" + per_token(host_other > 0.0 ? host_other : 0.0);
    for (int s = 0; s < kQwen38StageCount; ++s) {
        if (s == kQwen38StageToken) continue;
        out += ",\"host_";
        out += qwen38_stage_name(s);
        out += "_ms\":" + per_token(t.stage[static_cast<size_t>(s)].host_ms);
    }

    for (int s = 0; s < kQwen38StageCount; ++s) {
        if (s == kQwen38StageToken) continue;
        out += ",\"calls_";
        out += qwen38_stage_name(s);
        out += "\":" + std::to_string(t.stage[static_cast<size_t>(s)].calls);
    }
    out += ",\"calls_layers\":" + std::to_string(t.layer_calls);

    out += ",\"layer_device_ms\":[";
    for (size_t i = 0; i < t.layer_device_ms.size(); ++i) {
        if (i != 0) out += ",";
        out += per_token(t.layer_device_ms[i]);
    }
    out += "]";
    out += ",\"layer_host_ms\":[";
    for (size_t i = 0; i < t.layer_host_ms.size(); ++i) {
        if (i != 0) out += ",";
        out += per_token(t.layer_host_ms[i]);
    }
    out += "]";

    uint64_t unmatched = 0;
    uint64_t overflow = 0;
    for (int s = 0; s < kQwen38StageCount; ++s) {
        unmatched += t.stage[static_cast<size_t>(s)].unmatched;
        overflow += t.stage[static_cast<size_t>(s)].overflow;
    }
    out += ",\"unmatched\":" + std::to_string(unmatched);
    out += ",\"overflow\":" + std::to_string(overflow);
    out += ",\"decode_tokens\":" + std::to_string(t.phases);
    out += "}";
    return out;
}

} // namespace wd
