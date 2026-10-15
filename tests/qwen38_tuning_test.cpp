#include "../src/qwen38_tuning.hpp"

#include <cassert>
#include <iostream>

int main() {
    const auto a100 = wd::qwen38_tuning_for_gpu(80, 80 * 1024);
    assert(a100.qsa_block_budget == 512);
    assert(a100.qsa_token_budget == 2048);
    assert(a100.ple_prefetch_rows == 48);
    // The expert cache scales with the card: 75% of VRAM minus 8 GiB of headroom.
    // The old tier table gave 8 GiB here, which measured 15.06 tok/s against 26.64
    // at 20 GiB on A100-80GB.
    assert(a100.expert_cache_mib == 80 * 1024 * 3 / 4 - 8 * 1024);

    const auto h200 = wd::qwen38_tuning_for_gpu(90, 141 * 1024);
    assert(h200.ple_prefetch_rows == 64);
    assert(h200.expert_cache_mib == 64 * 1024);   // clamped

    // A bigger card must never get a smaller cache, and small cards keep a floor.
    int previous = 0;
    for (int mib : {8 * 1024, 16 * 1024, 24 * 1024, 40 * 1024, 80 * 1024, 141 * 1024}) {
        const auto t = wd::qwen38_tuning_for_gpu(80, mib);
        assert(t.expert_cache_mib >= previous);
        assert(t.expert_cache_mib >= 512);
        previous = t.expert_cache_mib;
    }
    assert(wd::qwen38_tuning_for_gpu(80, 16 * 1024).expert_cache_mib >= 512);

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
