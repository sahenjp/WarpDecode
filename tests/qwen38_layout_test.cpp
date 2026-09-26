#include "../src/qwen38_layout.hpp"

#include <cassert>
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
    assert(q.qsa_layer_count() == 12);
    assert(q.gdn_layer_count() == 36);
    assert(q.is_qsa(3));
    assert(q.is_gdn(0));
    assert(q.is_mtp(48));
    assert(q.is_ple(0));
    assert(q.ple_head_count() == 16);

    std::cout << "PASS\n";
    return 0;
}
