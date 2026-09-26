#include "../src/gguf.hpp"

#include <cassert>
#include <iostream>

int main() {
    auto check = [](wd::GgmlType type, uint64_t n, size_t bytes) {
        wd::TensorInfo t;
        t.name = "test";
        t.dims = {n};
        t.type = type;
        assert(t.byte_size() == bytes);
    };

    check(wd::GgmlType::F32, 32, 128);
    check(wd::GgmlType::F16, 32, 64);
    check(wd::GgmlType::BF16, 32, 64);
    check(wd::GgmlType::Q4_0, 32, 18);
    check(wd::GgmlType::Q8_0, 32, 34);
    check(wd::GgmlType::Q6_K, 256, 210);
    check(wd::GgmlType::IQ4_NL, 32, 18);
    check(wd::GgmlType::IQ3_S, 256, 110);
    check(wd::GgmlType::IQ2_S, 256, 82);

    std::cout << "PASS\n";
    return 0;
}
