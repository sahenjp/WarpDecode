// Deterministic residual-accumulation regression test (host only, no GPU).
//
// Isolates a precision-boundary hypothesis for WarpDecode Issue #1: an F16
// residual stream rounds to half at every layer boundary while the pinned CPU
// reference keeps F32. Over many layers the F16 lifetime accumulates a
// systematic error that the F32 lifetime avoids.
//
// The branch term is a fixed, small pseudo-random vector repeatedly added to
// the residual. No model trace is used. The reference is computed in
// double precision.
//
// Build:  g++ -O2 -std=c++17 issue1_residual_accum_test.cpp -o accum_test
// Run:    ./accum_test
// Exit code is non-zero if the F32 path does not improve on the F16 path.

#include "qwen38_precision.hpp"

#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

uint32_t next_u32(uint32_t & s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float next_unit(uint32_t & s) {
    return (static_cast<float>(next_u32(s) >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
}

}  // namespace

int main() {
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::puts("FAIL: could not set IEEE round-to-nearest-even mode");
        return 1;
    }
    // Guard the emulator before using it as evidence about CUDA rounding.
    // Ties must round to even, mantissa carry must reach the next exponent,
    // and half subnormals must decode to their actual floating-point value.
    if (wd::q38_round_half(1.00048828125f) != 1.0f ||
        wd::q38_round_half(1.00146484375f) != 1.001953125f ||
        wd::q38_round_half(1.99951171875f) != 2.0f ||
        wd::q38_round_half(std::ldexp(1.0f, -24)) != std::ldexp(1.0f, -24) ||
        wd::q38_round_half(std::ldexp(1.0f, -25)) != 0.0f) {
        std::puts("FAIL: binary16 emulator violates round-to-nearest-even");
        return 1;
    }
    constexpr int kLayers = 48;
    constexpr int kDim = 10240;
    constexpr float kBranchRms = 0.001f;

    uint32_t seed = 0x1234567u;
    std::vector<float> branch(kDim);
    for (int i = 0; i < kDim; ++i) branch[i] = next_unit(seed) * kBranchRms;

    std::vector<float> residual(kDim);
    for (int i = 0; i < kDim; ++i) residual[i] = next_unit(seed) * 0.05f;

    std::vector<double> ref(residual.begin(), residual.end());
    std::vector<float> f16 = residual;
    std::vector<float> f32 = residual;

    auto nrmse = [&](const std::vector<float> & got) {
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < kDim; ++i) {
            const double d = static_cast<double>(got[i]) - ref[i];
            err2 += d * d;
            ref2 += ref[i] * ref[i];
        }
        return std::sqrt(err2 / ref2);
    };

    std::printf("layer  f16_nrmse   f32_nrmse\n");
    for (int layer = 0; layer < kLayers; ++layer) {
        for (int i = 0; i < kDim; ++i) {
            ref[i] += branch[i];
            f16[i] = wd::q38_round_half(f16[i] + branch[i]);
            f32[i] = f32[i] + branch[i];
        }
        if (layer % 8 == 0 || layer == kLayers - 1) {
            std::printf("%5d  %9.6f  %9.6f\n", layer, nrmse(f16), nrmse(f32));
        }
    }

    const double f16_err = nrmse(f16);
    const double f32_err = nrmse(f32);
    std::printf("\nfinal: f16=%.6e f32=%.6e improvement=%.1fx\n", f16_err, f32_err, f16_err / f32_err);
    if (!(f32_err < f16_err)) {
        std::printf("FAIL: F32 residual did not improve on F16\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
