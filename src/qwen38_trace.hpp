#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    double mean = 0.0;
    double l2 = 0.0;
    uint64_t checksum = 14695981039346656037ull;
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
        const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) {
            out.checksum ^= bytes[i];
            out.checksum *= 1099511628211ull;
        }
    }
    out.mean = sum / static_cast<double>(values.size());
    out.l2 = std::sqrt(square_sum);
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

} // namespace wd
