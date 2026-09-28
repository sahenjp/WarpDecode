#include "../src/qwen38_trace.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

static void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}

template <typename T>
static T read_value(std::ifstream & file) {
    T value{};
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) throw std::runtime_error("truncated trace file");
    return value;
}

int main() {
    const std::vector<float> values{1.0f, -2.0f, 3.0f};
    const auto stats = wd::qwen38_trace_stats(values);
    require(stats.count == 3, "trace count");
    require(stats.min == -2.0f && stats.max == 3.0f, "trace range");
    require(stats.max_abs == 3.0f, "trace max absolute value");
    require(std::abs(stats.mean - (2.0 / 3.0)) < 1e-12, "trace mean");
    require(std::abs(stats.l2 - std::sqrt(14.0)) < 1e-12, "trace L2");
    require(std::abs(stats.rms - std::sqrt(14.0 / 3.0)) < 1e-12, "trace RMS");
    require(stats.checksum != 0, "trace checksum");
    require(wd::qwen38_expert_trace_offset(0, 4) == 0, "expert zero offset");
    require(wd::qwen38_expert_trace_offset(2, 4) == 8, "expert rank offset");

    const std::string path = "/tmp/warodecode-qwen38-trace-test.bin";
    const std::vector<wd::Qwen38TraceVector> vectors{
        {"layer0.expert_rank_0_id_9", values},
        {"layer0.expert_rank_1_id_4", {4.0f, 5.0f}},
    };
    wd::write_qwen38_trace_binary(path, vectors);

    std::ifstream file(path, std::ios::binary);
    char magic[8]{};
    file.read(magic, sizeof(magic));
    require(std::string(magic, sizeof(magic)) == std::string("WDPTRC1\0", 8), "trace magic");
    require(read_value<uint32_t>(file) == 1, "trace version");
    require(read_value<uint32_t>(file) == 2, "trace vector count");
    for (const auto & expected : vectors) {
        const uint32_t name_size = read_value<uint32_t>(file);
        const uint64_t count = read_value<uint64_t>(file);
        std::string name(name_size, '\0');
        file.read(name.data(), static_cast<std::streamsize>(name.size()));
        require(static_cast<bool>(file), "trace name");
        require(name == expected.name, "trace vector name");
        require(count == expected.values.size(), "trace vector size");
        std::vector<float> actual(static_cast<size_t>(count));
        file.read(reinterpret_cast<char *>(actual.data()), static_cast<std::streamsize>(actual.size() * sizeof(float)));
        require(static_cast<bool>(file), "trace vector data");
        require(actual == expected.values, "trace vector values");
    }
    require(file.peek() == std::ifstream::traits_type::eof(), "trace trailing bytes");
    file.close();
    std::remove(path.c_str());
    return 0;
}
