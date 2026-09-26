#include "../src/qwen38_tuning.hpp"

#include <cassert>
#include <iostream>

int main() {
    const auto a100 = wd::qwen38_tuning_for_gpu(80, 80 * 1024);
    assert(a100.qsa_block_budget == 512);
    assert(a100.qsa_token_budget == 2048);
    assert(a100.ple_prefetch_rows == 48);
    assert(a100.expert_cache_mib == 8 * 1024);

    const auto h200 = wd::qwen38_tuning_for_gpu(90, 141 * 1024);
    assert(h200.ple_prefetch_rows == 64);
    assert(h200.expert_cache_mib == 16 * 1024);

    wd::AdaptiveMtpController c;
    assert(c.choose() == 2);
    c.observe(2, 2, 100.0);
    assert(c.choose() == 3);
    c.observe(3, 3, 120.0);
    assert(c.choose() == 4);
    c.observe(4, 3, 160.0);
    assert(c.choose() == 3);

    std::cout << "PASS\n";
    return 0;
}
