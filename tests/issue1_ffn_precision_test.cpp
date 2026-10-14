// Host regression for the CPU-order F32 MoE/shared-expert/residual boundary.
#include "qwen38_precision.hpp"

#include <cmath>
#include <cfenv>
#include <cstdio>

int main() {
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::puts("FAIL: could not set IEEE round-to-nearest-even mode");
        return 1;
    }
    // Every expert output remains half, and BOTH paths use this same F32
    // fmaf weighted accumulator. Only the completed-output store differs.
    float routed = 0.0f;
    for (int expert = 0; expert < 10; ++expert) {
        const float expert_half = wd::q38_round_half(
            expert % 2 == 0 ? 1.0f : 1.0009765625f);
        routed = std::fma(0.1f, expert_half, routed);
    }
    constexpr float shared = -0.00048828125f;
    constexpr float shared_gate = 0.5f;
    constexpr float residual = 0.333251953125f;
    constexpr float hc_weight = 1.25f;

    const float moe = wd::q38_moe_branch_f32(routed, shared, shared_gate);
    const float output = wd::q38_residual_add_f32(residual, moe, hc_weight);
    volatile float weighted_shared = shared_gate * shared;
    const float expected_moe = routed + weighted_shared;
    volatile float weighted_branch = hc_weight * expected_moe;
    const float expected_output = residual + weighted_branch;
    if (moe != expected_moe || output != expected_output) {
        std::printf("FAIL: F32 branch boundary mismatch (moe=%a expected=%a output=%a expected=%a)\n",
            moe, expected_moe, output, expected_output);
        return 1;
    }

    const float routed_half = wd::q38_round_half(routed);
    const float rounded_branch = wd::q38_round_half(
        std::fma(shared_gate, shared, routed_half));
    const float rounded_residual = wd::q38_round_half(residual + hc_weight * rounded_branch);
    if (rounded_branch == expected_moe && rounded_residual == expected_output) {
        std::puts("FAIL: legacy F16 boundary did not expose the expected precision loss");
        return 1;
    }

    std::printf("PASS: legacy=%a candidate=%a expected=%a\n",
        rounded_residual, output, expected_output);
    return 0;
}
