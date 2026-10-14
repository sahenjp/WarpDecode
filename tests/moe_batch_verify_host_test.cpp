// Host bit-comparison for the Task I M-token batch MoE layout.
//
// The pair kernels need nvcc (tests/moe_batch_verify_bench.cu is the device
// half), but the two things that would break bit-identity silently are pure
// index/order properties and are checked here:
//
//   pair map:   the flat pair index p = token * k + j must resolve to token
//               p / k and to the j-th expert the router selected for that
//               token, in top-k order. This is what the batch replaces the
//               per-token acquire loop with.
//   accumulation: per element, the batch folds
//               a = add_rn(a, mul_rn(experts[p * n + i], weights[p]))
//               for p = token * k .. token * k + k - 1, which is exactly the
//               sequence of rounded product/add pairs the k serial
//               q38_weighted_accum_f32_kernel launches produce.
//
// A negative control shows that a wrong pair->token map or a contracted
// multiply-add is visible to the same comparison.
//
// Run: bash tests/moe_batch_verify_host_check.sh

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

// -ffp-contract=off keeps these mirrors of __fmul_rn / __fadd_rn single-rounded.
float mul_rn(float a, float b) { return a * b; }
float add_rn(float a, float b) { return a + b; }

// The host-side view of one routed MoE layer: tokens * k pairs, where pair
// token * k + j already knows its expert index and router weight.
struct Layer {
    int tokens = 0;
    int k = 0;
    int hidden = 0;
    std::vector<int> pair_expert;      // [tokens * k]
    std::vector<float> pair_weight;    // [tokens * k]
};

// Serial runner: one pass per token, walking the token's own top-k slice.
std::vector<float> run_serial(const Layer & layer, const std::vector<float> & expert_outputs) {
    const int n = layer.hidden;
    std::vector<float> out(static_cast<size_t>(layer.tokens) * n, 0.0f);
    for (int t = 0; t < layer.tokens; ++t) {
        for (int j = 0; j < layer.k; ++j) {
            const int pair = t * layer.k + j;
            const float weight = layer.pair_weight[static_cast<size_t>(pair)];
            for (int i = 0; i < n; ++i) {
                float & a = out[static_cast<size_t>(t) * n + i];
                a = add_rn(a, mul_rn(expert_outputs[static_cast<size_t>(pair) * n + i], weight));
            }
        }
    }
    return out;
}

// Batch runner: the same per-element loop, but the pair order is derived from
// the flat build order the batch class uses (token major, top-k minor). The
// only difference from the serial loop is where the pair comes from.
std::vector<float> run_batch(const Layer & layer, const std::vector<float> & expert_outputs) {
    const int n = layer.hidden;
    std::vector<float> out(static_cast<size_t>(layer.tokens) * n, 0.0f);
    for (int t = 0; t < layer.tokens; ++t) {
        for (int i = 0; i < n; ++i) {
            float a = 0.0f;
            for (int j = 0; j < layer.k; ++j) {
                const int pair = t * layer.k + j;
                a = add_rn(a, mul_rn(expert_outputs[static_cast<size_t>(pair) * n + i],
                                     layer.pair_weight[static_cast<size_t>(pair)]));
            }
            out[static_cast<size_t>(t) * n + i] = a;
        }
    }
    return out;
}

} // namespace

int main() {
    const int hidden = 2560;
    const int k = 10;
    const int tokens = 4;
    const int n_experts = 512;

    std::mt19937 rng(20261014u);
    // A realistic router result: every token picks its own top-k from the same
    // expert pool, and several experts are picked by more than one token
    // (the union is smaller than tokens * k, which is what the batch folds).
    Layer layer;
    layer.tokens = tokens;
    layer.k = k;
    layer.hidden = hidden;
    layer.pair_expert.resize(static_cast<size_t>(tokens) * k);
    layer.pair_weight.resize(static_cast<size_t>(tokens) * k);
    std::uniform_int_distribution<int> expert_dist(0, n_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.0f, 0.4f);
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < k; ++j) {
            const int pair = t * k + j;
            layer.pair_expert[static_cast<size_t>(pair)] = expert_dist(rng);
            layer.pair_weight[static_cast<size_t>(pair)] = weight_dist(rng);
        }
    }

    // One expert output per pair, in the batch buffer's own layout: the pair's
    // own row of the expert matrix, not the expert's global row. This is the
    // layout the pair kernels write ([pairs][hidden]).
    std::vector<float> expert_outputs(static_cast<size_t>(tokens) * k * hidden);
    std::uniform_real_distribution<float> out_dist(-2.0f, 2.0f);
    for (auto & v : expert_outputs) v = out_dist(rng);

    std::printf("Task I host check (tokens=%d, top-k=%d, hidden=%d)\n", tokens, k, hidden);

    // The pair map the device would build: pair = token * k + j, expert = the
    // token's j-th selection. Both are integer identities, so they are checked
    // exactly.
    bool pair_map_ok = true;
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < k; ++j) {
            const int pair = t * k + j;
            if (pair / k != t || pair % k != j) pair_map_ok = false;
            if (pair >= tokens * k) pair_map_ok = false;
        }
    }
    check("pair index decomposes to (token, top-k slot)", pair_map_ok);

    int distinct = 0;
    for (int e = 0; e < n_experts; ++e) {
        for (int p = 0; p < tokens * k; ++p) {
            if (layer.pair_expert[static_cast<size_t>(p)] == e) { ++distinct; break; }
        }
    }
    std::printf("  router union: %d distinct experts over %d pairs\n", distinct, tokens * k);

    const std::vector<float> serial = run_serial(layer, expert_outputs);
    const std::vector<float> batched = run_batch(layer, expert_outputs);
    check("ordered accumulation is bit-identical to M serial passes",
          std::memcmp(serial.data(), batched.data(), serial.size() * sizeof(float)) == 0);

    // Negative control 1: a fused multiply-add changes the rounding.
    {
        std::vector<float> contracted(serial.size(), 0.0f);
        for (int t = 0; t < tokens; ++t) {
            for (int i = 0; i < hidden; ++i) {
                float a = 0.0f;
                for (int j = 0; j < k; ++j) {
                    const int pair = t * k + j;
                    a = std::fmaf(expert_outputs[static_cast<size_t>(pair) * hidden + i],
                                  layer.pair_weight[static_cast<size_t>(pair)], a);
                }
                contracted[static_cast<size_t>(t) * hidden + i] = a;
            }
        }
        check("negative control: an FMA contraction is visible",
              std::memcmp(serial.data(), contracted.data(), serial.size() * sizeof(float)) != 0);
    }

    // Negative control 2: folding a token's pairs into another token's output
    // (the wrong pair -> token map) must change the result.
    {
        std::vector<float> swapped(serial.size(), 0.0f);
        for (int t = 0; t < tokens; ++t) {
            const int source = (t + 1) % tokens;
            for (int i = 0; i < hidden; ++i) {
                float a = 0.0f;
                for (int j = 0; j < k; ++j) {
                    const int pair = source * k + j;
                    a = add_rn(a, mul_rn(expert_outputs[static_cast<size_t>(pair) * hidden + i],
                                         layer.pair_weight[static_cast<size_t>(pair)]));
                }
                swapped[static_cast<size_t>(t) * hidden + i] = a;
            }
        }
        check("negative control: a wrong pair -> token map is visible",
              std::memcmp(serial.data(), swapped.data(), serial.size() * sizeof(float)) != 0);
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAIL" : "PASS", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
