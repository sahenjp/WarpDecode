#include "../src/ple_reader.hpp"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    const char* path = "/tmp/warodecode-ple-reader.bin";
    constexpr uint32_t row_bytes = 7;
    constexpr uint64_t rows = 32;
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        for (uint64_t r = 0; r < rows; ++r) {
            for (uint32_t b = 0; b < row_bytes; ++b) {
                const uint8_t v = static_cast<uint8_t>((r * 17 + b) & 0xff);
                f.write(reinterpret_cast<const char*>(&v), 1);
            }
        }
    }

    wd::PleDirectReader rd(path, 0, row_bytes, rows);
    const std::vector<uint64_t> idx = {9, 3, 4, 4, 5, 31, 9, 0};
    std::vector<uint8_t> out(idx.size() * row_bytes);
    rd.read_rows(idx, out.data());
    for (size_t i = 0; i < idx.size(); ++i) {
        for (uint32_t b = 0; b < row_bytes; ++b) {
            const uint8_t want = static_cast<uint8_t>((idx[i] * 17 + b) & 0xff);
            if (out[i * row_bytes + b] != want) {
                std::cerr << "mismatch i=" << i << " b=" << b << "\n";
                return 1;
            }
        }
    }
    std::remove(path);
    std::cout << "PASS\n";
    return 0;
}
