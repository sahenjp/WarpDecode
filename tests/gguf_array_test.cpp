#include "../src/gguf.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

int main(int argc, char ** argv) {
    assert(argc == 2);
    wd::GgufFile g(argv[1], false);

    const auto ratios = g.int_array_meta("qwen4exp.attention.compress_ratios");
    assert(ratios.size() == 49);
    assert(ratios[3] == 8);
    assert(ratios[48] == 0);

    const auto ple = g.uint_array_meta("qwen4exp.ple.layers");
    assert(ple.size() == 1 && ple[0] == 0);

    const auto mult = g.uint_array_meta("qwen4exp.ple.layer_multipliers");
    assert((mult == std::vector<uint64_t>{3, 5, 7}));

    const auto offsets = g.uint_array_meta("qwen4exp.ple.head_offsets");
    const auto sizes = g.uint_array_meta("qwen4exp.ple.head_vocab_sizes");
    assert(offsets.size() == 16);
    assert(sizes.size() == 16);
    assert(offsets.back() == 150);
    assert(sizes.back() == 10);

    std::cout << "PASS\n";
    return 0;
}
