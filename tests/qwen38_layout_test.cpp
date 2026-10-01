#include "../src/qwen38_layout.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main(int argc, char ** argv) {
    assert(argc == 2);
    wd::GgufFile g(argv[1], false);
    const auto q = wd::qwen38_layout_from_gguf(g);

    assert(q.total_blocks == 49);
    assert(q.mtp_layers == 1);
    assert(q.trunk_layers == 48);
    assert(q.hidden == 2560);
    assert(q.heads == 24);
    assert(q.kv_heads == 2);
    assert(q.experts == 512);
    assert(q.experts_used == 10);
    assert(q.rope_dimension_count == 64);
    assert(q.qsa_layer_count() == 12);
    assert(q.gdn_layer_count() == 36);
    assert(q.is_qsa(3));
    assert(q.is_gdn(0));
    assert(q.is_mtp(48));
    assert(q.is_ple(0));
    assert(q.ple_head_count() == 16);
    assert(std::fabs(q.rms_eps - 1e-6f) < 1e-9f);
    assert(std::fabs(q.rope_freq_base - 1000000.0f) < 1.0f);
    assert(std::fabs(q.expert_weights_scale - 1.25f) < 1e-6f);
    assert((q.rope_sections == std::vector<int64_t>{11, 11, 10, 0}));

    std::cout << "PASS\n";
    return 0;
}
