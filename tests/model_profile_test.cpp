#include "../src/model_profile.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::filesystem::path path = argv[1];

    wd::GgufFile g(path.string(), false);
    const auto p = wd::profile_model(g, path, 80);

    assert(p.qwen38_flash_next);
    assert(p.arch == "qwen4exp");
    assert(p.total_blocks == 49);
    assert(p.mtp_layers == 1);
    assert(p.layers == 48);
    assert(p.gdn_layers == 36);
    assert(p.qsa_layers == 12);
    assert(p.experts == 512);
    assert(p.routed_experts == 10);

    std::cout << "PASS\n";
    return 0;
}
