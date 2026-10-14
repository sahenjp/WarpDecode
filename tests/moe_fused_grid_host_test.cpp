// Host bit-comparison for the Task G expert-batched MoE stages
// (WARODECODE_MOE_FUSED_GRID).
//
// The batched kernels need nvcc, and tests/moe_fused_grid_bench.cu compares them
// against the per-expert kernels on the device. This test covers the two things
// that are pure C++ and would fail silently: the *order* in which the k expert
// outputs are folded into the routed sum, and the expert/projection index
// mapping that lets one kernel serve both the separate and the fused gate/up
// layouts. Both are exact integer/float-order properties, so they are checked
// bit for bit.
//
//   accumulation: the serial runner launches
//                 accum[i] = add_rn(accum[i], mul_rn(x_j[i], w_j)) once per
//                 expert j in top-k order, while the batched kernel keeps
//                 accum[i] in a register and walks j = 0..k-1. Per element the
//                 two sequences are identical, and this test asserts that -
//                 including a negative control showing that an FMA contraction
//                 (which the device code must not introduce) would be caught.
//
//   layout:       the separate layout stores expert j's gate at 2j and its up at
//                 2j + 1 in the [2k][ffn] gate/up buffer; the fused layout
//                 stores the 2 * ffn rows of expert j at row j. This test checks
//                 that both formulas address the same (expert, projection) pair.
//
// What this does NOT check: the device code, the quantiser/SiLU arithmetic (the
// device harness and the 64/64 token parity run cover those).
//
// Run: bash tests/moe_fused_grid_host_check.sh

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(const char * what, bool ok) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// The device kernels call __fmul_rn / __fadd_rn, i.e. one rounded operation per
// step with no contraction. -ffp-contract=off makes these plain C++ equivalents
// exact mirrors.
float mul_rn(float a, float b) { return a * b; }
float add_rn(float a, float b) { return a + b; }

// accum[i] = add(accum[i], mul(x_j[i], w_j)) applied for j = 0..k-1, as one
// kernel per expert would.
std::vector<float> accumulate_sequential(
        const std::vector<float> & experts, const std::vector<float> & weights,
        int n, int k) {
    std::vector<float> accum(static_cast<size_t>(n), 0.0f);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            accum[static_cast<size_t>(i)] = add_rn(
                accum[static_cast<size_t>(i)],
                mul_rn(experts[static_cast<size_t>(j) * n + i], weights[static_cast<size_t>(j)]));
        }
    }
    return accum;
}

// The same work as one ordered loop per element, which is what
// q38_moe_weighted_accum_experts_f32_kernel does.
std::vector<float> accumulate_batched(
        const std::vector<float> & experts, const std::vector<float> & weights,
        int n, int k) {
    std::vector<float> accum(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        float a = accum[static_cast<size_t>(i)];
        for (int j = 0; j < k; ++j) {
            a = add_rn(a, mul_rn(experts[static_cast<size_t>(j) * n + i],
                                 weights[static_cast<size_t>(j)]));
        }
        accum[static_cast<size_t>(i)] = a;
    }
    return accum;
}

// The contract the batched kernels rely on: expert e's projection p lives at the
// same flattened offset whether the layer stores gate and up separately or as
// one fused [2 * ffn] matrix.
size_t separate_offset(int expert, int projection, int ffn) {
    return (static_cast<size_t>(2) * expert + projection) * static_cast<size_t>(ffn);
}
size_t fused_offset(int expert, int projection, int ffn) {
    return static_cast<size_t>(expert) * static_cast<size_t>(2 * ffn) +
           static_cast<size_t>(projection) * static_cast<size_t>(ffn);
}

} // namespace

int main() {
    const int k = 10;
    const int n = 2560;
    std::mt19937 rng(20261014);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> experts(static_cast<size_t>(k) * n);
    for (auto & v : experts) v = dist(rng);
    std::vector<float> weights(static_cast<size_t>(k));
    for (auto & v : weights) v = static_cast<float>(rng() % 1000) * 0.0001f;

    std::printf("Task G host check (k=%d, hidden=%d)\n", k, n);

    const std::vector<float> serial = accumulate_sequential(experts, weights, n, k);
    const std::vector<float> batched = accumulate_batched(experts, weights, n, k);
    check("ordered accumulation is bit-identical to k launches",
          std::memcmp(serial.data(), batched.data(), serial.size() * sizeof(float)) == 0);

    // Negative control: a fused multiply-add would change the rounding and must
    // be visible to this comparison.
    std::vector<float> contracted(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        float a = 0.0f;
        for (int j = 0; j < k; ++j) {
            a = std::fma(experts[static_cast<size_t>(j) * n + i], weights[static_cast<size_t>(j)], a);
        }
        contracted[static_cast<size_t>(i)] = a;
    }
    check("negative control: FMA contraction is detected",
          std::memcmp(serial.data(), contracted.data(), serial.size() * sizeof(float)) != 0);

    // A reversed expert order must also be caught (the order is top-k order, not
    // "any order").
    std::vector<float> reversed_weights(weights.rbegin(), weights.rend());
    const std::vector<float> reversed = accumulate_batched(experts, reversed_weights, n, k);
    check("negative control: reversed expert order is detected",
          std::memcmp(serial.data(), reversed.data(), serial.size() * sizeof(float)) != 0);

    bool layout_ok = true;
    for (int expert = 0; expert < 32 && layout_ok; ++expert) {
        for (int projection = 0; projection < 2; ++projection) {
            if (separate_offset(expert, projection, 640) !=
                fused_offset(expert, projection, 640)) {
                layout_ok = false;
            }
        }
    }
    check("separate and fused gate/up layouts share one index formula", layout_ok);

    // Every expert and projection must land in a distinct slot of the [2k][ffn]
    // buffer, otherwise a batched launch would silently overwrite an expert.
    bool unique_ok = true;
    {
        std::vector<int> seen(static_cast<size_t>(2) * k * 640, 0);
        for (int expert = 0; expert < k; ++expert) {
            for (int projection = 0; projection < 2; ++projection) {
                const size_t base = separate_offset(expert, projection, 640);
                for (int i = 0; i < 640; ++i) {
                    if (++seen[base + static_cast<size_t>(i)] != 1) unique_ok = false;
                }
            }
        }
    }
    check("batched gate/up slices are disjoint", unique_ok);

    std::printf("%s (%d failure%s)\n", g_failures ? "FAIL" : "PASS",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
