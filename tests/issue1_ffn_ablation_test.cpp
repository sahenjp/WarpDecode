// Synthetic Issue #1 regression for the four independent FFN precision switches.
// Host only: ten half expert outputs feed an F32 weighted sum, then the routed
// and shared branches cross the legacy/candidate store boundaries for 48 layers.
#include "qwen38_precision.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kLayers = 48;
constexpr int kHidden = 128;
constexpr int kExperts = 10;
enum Variant { kBaseline, kResidualOnly, kMoeOutputOnly, kCombined, kVariantCount };

const char * variant_name(Variant variant) {
    static constexpr const char * names[] = {
        "baseline", "residual_only", "moe_output_only", "combined"};
    return names[variant];
}

double nrmse(const std::vector<float> & got, const std::vector<double> & ref) {
    double err2 = 0.0;
    double ref2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - ref[i];
        err2 += d * d;
        ref2 += ref[i] * ref[i];
    }
    return std::sqrt(err2 / ref2);
}

}  // namespace

int main() {
    constexpr float weights[kExperts] = {
        1.0f / 55.0f, 2.0f / 55.0f, 3.0f / 55.0f, 4.0f / 55.0f, 5.0f / 55.0f,
        6.0f / 55.0f, 7.0f / 55.0f, 8.0f / 55.0f, 9.0f / 55.0f, 10.0f / 55.0f};
    constexpr float hc_weight = 1.25f;

    std::vector<double> reference(kHidden);
    std::array<std::vector<float>, kVariantCount> outputs;
    for (auto & output : outputs) output.resize(kHidden);
    for (int i = 0; i < kHidden; ++i) {
        const float initial = 0.05f + static_cast<float>(i - kHidden / 2) * 0.0007f;
        reference[i] = initial;
        for (auto & output : outputs) output[i] = initial;
    }

    for (int layer = 0; layer < kLayers; ++layer) {
        const float shared_gate = 0.35f + static_cast<float>((layer * 7) % 13) * 0.025f;
        for (int i = 0; i < kHidden; ++i) {
            float routed_f32 = 0.0f;
            double routed_reference = 0.0;
            for (int expert = 0; expert < kExperts; ++expert) {
                const int code = (expert * 19 + layer * 11 + i * 5) % 37 - 18;
                const float expert_half = wd::q38_round_half(static_cast<float>(code) * 0.00019f);
                routed_f32 = std::fma(weights[expert], expert_half, routed_f32);
                routed_reference += static_cast<double>(weights[expert]) * expert_half;
            }
            const int shared_code = (layer * 7 + i * 13) % 29 - 14;
            const float shared_half = wd::q38_round_half(static_cast<float>(shared_code) * 0.00031f);

            const double reference_branch = routed_reference +
                static_cast<double>(shared_gate) * shared_half;
            reference[i] += static_cast<double>(hc_weight) * reference_branch;

            const float routed_half = wd::q38_round_half(routed_f32);
            const float legacy_branch = wd::q38_round_half(
                routed_half + shared_gate * shared_half);
            const float f32_branch = wd::q38_moe_branch_f32(
                routed_f32, shared_half, shared_gate);

            outputs[kBaseline][i] = wd::q38_round_half(
                std::fma(legacy_branch, hc_weight, outputs[kBaseline][i]));
            outputs[kResidualOnly][i] = std::fma(
                legacy_branch, hc_weight, outputs[kResidualOnly][i]);
            outputs[kMoeOutputOnly][i] = wd::q38_round_half(
                std::fma(f32_branch, hc_weight, outputs[kMoeOutputOnly][i]));
            outputs[kCombined][i] = wd::q38_residual_add_f32(
                outputs[kCombined][i], f32_branch, hc_weight);
        }
        if ((layer + 1) % 8 == 0 || layer == 0) {
            std::printf("layer=%d", layer + 1);
            for (int variant = 0; variant < kVariantCount; ++variant) {
                std::printf(" %s=%.9e", variant_name(static_cast<Variant>(variant)),
                    nrmse(outputs[variant], reference));
            }
            std::puts("");
        }
    }

    std::array<double, kVariantCount> errors{};
    for (int variant = 0; variant < kVariantCount; ++variant) {
        errors[variant] = nrmse(outputs[variant], reference);
        std::printf("%s_nrmse=%.9e\n", variant_name(static_cast<Variant>(variant)), errors[variant]);
    }

    if (!(errors[kCombined] < errors[kBaseline] &&
          errors[kCombined] < errors[kResidualOnly] &&
          errors[kCombined] < errors[kMoeOutputOnly])) {
        std::puts("FAIL: combined F32 boundaries did not improve on each independently rounded variant");
        return 1;
    }
    if (outputs[kBaseline] == outputs[kResidualOnly] ||
        outputs[kBaseline] == outputs[kMoeOutputOnly] ||
        outputs[kResidualOnly] == outputs[kCombined] ||
        outputs[kMoeOutputOnly] == outputs[kCombined]) {
        std::puts("FAIL: the two precision boundaries did not vary independently");
        return 1;
    }
    std::puts("PASS: 10 experts, routed/shared boundary and residual boundary across 48 layers");
    return 0;
}
