// CUDA regression for the routed-output/shared-expert/residual boundaries.
// Run this only on the authorized Modal A100 validation path.
#include "qwen38_hc.cuh"
#include "qwen38_moe.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kHidden = 32;
constexpr int kHc = 2;
constexpr int kExperts = 10;
constexpr int kTotal = kHidden * kHc;
constexpr float kWeights[kExperts] = {
    1.0f / 55.0f, 2.0f / 55.0f, 3.0f / 55.0f, 4.0f / 55.0f, 5.0f / 55.0f,
    6.0f / 55.0f, 7.0f / 55.0f, 8.0f / 55.0f, 9.0f / 55.0f, 10.0f / 55.0f};

enum Variant { kBaseline, kResidualOnly, kMoeOutputOnly, kCombined, kVariantCount };

const char * variant_name(Variant variant) {
    static constexpr const char * names[] = {
        "baseline", "residual_only", "moe_output_only", "combined"};
    return names[variant];
}

bool check(cudaError_t status, const char * what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

bool finish_launch(const char * what) {
    return check(cudaGetLastError(), what) && check(cudaDeviceSynchronize(), what);
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
    int device_count = 0;
    if (!check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    cudaDeviceProp properties{};
    if (!check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) return 1;
    if (properties.major < 8) {
        std::fprintf(stderr, "FAIL: test is intended for SM80+; found SM%d%d\n",
            properties.major, properties.minor);
        return 1;
    }

    std::vector<__half> expert_outputs(kExperts * kHidden);
    std::vector<__half> shared(kHidden);
    std::vector<float> initial(kTotal);
    std::vector<double> reference(kTotal);
    std::vector<float> routed(kHidden, 0.0f);
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int i = 0; i < kHidden; ++i) {
            const int code = (expert * 19 + i * 5) % 37 - 18;
            expert_outputs[expert * kHidden + i] = __float2half(static_cast<float>(code) * 0.00019f);
        }
    }
    for (int i = 0; i < kHidden; ++i) {
        const int code = (i * 13) % 29 - 14;
        shared[i] = __float2half(static_cast<float>(code) * 0.00031f);
        for (int c = 0; c < kHc; ++c) {
            const int index = c * kHidden + i;
            initial[index] = 0.05f + static_cast<float>(i - kHidden / 2) * 0.0015f + c * 0.003f;
            reference[index] = initial[index];
        }
        for (int expert = 0; expert < kExperts; ++expert) {
            routed[i] = std::fma(kWeights[expert], __half2float(expert_outputs[expert * kHidden + i]), routed[i]);
        }
    }

    __half * d_expert_outputs = nullptr;
    __half * d_shared = nullptr;
    __half * d_moe_half = nullptr;
    float * d_accum = nullptr;
    float * d_gate = nullptr;
    float * d_residual = nullptr;
    auto alloc = [&](void ** ptr, size_t bytes, const char * what) {
        return check(cudaMalloc(ptr, bytes), what);
    };
    if (!alloc(reinterpret_cast<void **>(&d_expert_outputs), expert_outputs.size() * sizeof(__half), "alloc experts") ||
        !alloc(reinterpret_cast<void **>(&d_shared), shared.size() * sizeof(__half), "alloc shared") ||
        !alloc(reinterpret_cast<void **>(&d_moe_half), kHidden * sizeof(__half), "alloc routed half") ||
        !alloc(reinterpret_cast<void **>(&d_accum), kHidden * sizeof(float), "alloc routed F32") ||
        !alloc(reinterpret_cast<void **>(&d_gate), sizeof(float), "alloc shared gate") ||
        !alloc(reinterpret_cast<void **>(&d_residual), kTotal * sizeof(float), "alloc residual")) {
        return 1;
    }
    if (!check(cudaMemcpy(d_expert_outputs, expert_outputs.data(), expert_outputs.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy experts") ||
        !check(cudaMemcpy(d_shared, shared.data(), shared.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy shared")) {
        return 1;
    }
    if (!check(cudaMemset(d_accum, 0, kHidden * sizeof(float)), "zero routed F32 accumulator")) return 1;
    const float shared_gate = 0.625f;
    if (!check(cudaMemcpy(d_gate, &shared_gate, sizeof(shared_gate), cudaMemcpyHostToDevice), "copy shared gate")) return 1;

    wd::Qwen38HcScratch scratch;
    scratch.initialize(kHidden, kHc, 1);
    if (!check(cudaMemset(scratch.inject(), 0, kHc * sizeof(__half)), "zero HC inject")) return 1;

    for (int expert = 0; expert < kExperts; ++expert) {
        wd::q38_weighted_accum_kernel<<<1, 64>>>(
            d_expert_outputs + expert * kHidden, kWeights[expert], d_accum, kHidden);
        if (!finish_launch("routed F32 accumulation")) return 1;
    }

    std::array<std::vector<float>, kVariantCount> outputs;
    for (auto & output : outputs) output.resize(kTotal);
    for (int variant = 0; variant < kVariantCount; ++variant) {
        const bool moe_f32 = variant == kMoeOutputOnly || variant == kCombined;
        const bool residual_f32 = variant == kResidualOnly || variant == kCombined;
        if (!check(cudaMemcpy(d_residual, initial.data(), kTotal * sizeof(float), cudaMemcpyHostToDevice), "reset residual")) return 1;

        if (moe_f32) {
            wd::q38_hc_combine_moe_f32_residual_kernel<<<(kTotal + 255) / 256, 256>>>(
                d_residual, d_accum, d_shared, d_gate, scratch.inject(),
                kHidden, kHc, !residual_f32);
        } else {
            wd::q38_float_to_half_kernel<<<1, 64>>>(d_accum, d_moe_half, kHidden);
            if (!finish_launch("legacy routed F16 store")) return 1;
            wd::q38_add_shared_kernel<<<1, 64>>>(d_moe_half, d_shared, d_gate, kHidden);
            if (!finish_launch("legacy shared-expert add")) return 1;
            wd::q38_hc_combine_f32_residual_kernel<<<(kTotal + 255) / 256, 256>>>(
                d_residual, d_moe_half, scratch.inject(), kHidden, kHc, !residual_f32);
        }
        if (!finish_launch("HC residual combine") ||
            !check(cudaMemcpy(outputs[variant].data(), d_residual, kTotal * sizeof(float), cudaMemcpyDeviceToHost), "copy result")) {
            return 1;
        }
    }

    // Check each actual GPU path against independent host arithmetic, not
    // just the aggregate quality ranking (which could hide a broken variant).
    for (int variant = 0; variant < kVariantCount; ++variant) {
        const bool moe_f32 = variant == kMoeOutputOnly || variant == kCombined;
        const bool residual_f32 = variant == kResidualOnly || variant == kCombined;
        for (int i = 0; i < kHidden; ++i) {
            const float branch = moe_f32
                ? wd::q38_moe_branch_f32(routed[i], __half2float(shared[i]), shared_gate)
                : wd::q38_round_half(std::fma(shared_gate, __half2float(shared[i]),
                    wd::q38_round_half(routed[i])));
            for (int c = 0; c < kHc; ++c) {
                const int index = c * kHidden + i;
                const float added = moe_f32
                    ? wd::q38_residual_add_f32(initial[index], branch, 1.0f)
                    : std::fma(branch, 1.0f, initial[index]);
                const float expected = residual_f32 ? added : wd::q38_round_half(added);
                if (std::fabs(outputs[variant][index] - expected) > 5.0e-8f) {
                    std::fprintf(stderr, "FAIL: %s lane %d got=%a expected=%a\n",
                        variant_name(static_cast<Variant>(variant)), index,
                        outputs[variant][index], expected);
                    return 1;
                }
            }
        }
    }

    for (int i = 0; i < kHidden; ++i) {
        const float branch = wd::q38_moe_branch_f32(
            routed[i], __half2float(shared[i]), shared_gate);
        for (int c = 0; c < kHc; ++c) {
            const int index = c * kHidden + i;
            reference[index] = wd::q38_residual_add_f32(initial[index], branch, 1.0f);
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
        std::puts("FAIL: combined CUDA path did not improve on each single-boundary variant");
        return 1;
    }

    cudaFree(d_expert_outputs);
    cudaFree(d_shared);
    cudaFree(d_moe_half);
    cudaFree(d_accum);
    cudaFree(d_gate);
    cudaFree(d_residual);
    std::puts("PASS: CUDA routed F32 sum, shared add, and four independent FFN precision variants");
    return 0;
}
