// Task T: bit-exactness test for the parallel exact router.
//
// The reference is the shipped one-thread kernel q38_topk_softmax_kernel(); the
// candidate is q38_router_exact_parallel_kernel<256>(). Both are compiled from
// the same headers the engine uses, and both are given the same logits rows on
// the real GPU, including the degenerate ones (all -INFINITY, +INFINITY that
// underflows every exponent, heavy ties, ±0, denormals, huge and tiny
// magnitudes). The comparison is bit-wise on both the ids and the weights, so a
// single changed rounding or a different tie-break fails the test.
//
// Build (VM): nvcc -O2 -std=c++17 -arch=sm_80 tests/router_parallel_exact_test.cu -o /content/router_test
#include <immintrin.h>  // the CPU MoE header uses SSE/AVX intrinsics unguarded

#include "qwen38_forward.cuh"

using namespace wd;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kMaxExperts = 512;
constexpr int kMaxK = 16;

struct BitResult {
    int mismatches = 0;
    int first_case = -1;
    int first_index = -1;
};

void fill_case(std::vector<float> & row, int pattern, int n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    row.assign(static_cast<size_t>(n), 0.0f);
    switch (pattern) {
        case 0:
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = normal(rng);
            break;
        case 1:
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = 0.25f;
            break;
        case 2:  // heavy ties from a small alphabet
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = static_cast<float>(rng() % 4) - 1.5f;
            break;
        case 3:  // large magnitudes (near the exp overflow/underflow edges)
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = (u01(rng) < 0.5f ? -1.0f : 1.0f) * (80.0f + u01(rng) * 60.0f);
            break;
        case 4:  // tiny magnitudes (denormal exponents, equal max)
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = (u01(rng) < 0.5f ? -1.0f : 1.0f) * 1e-30f;
            break;
        case 5:  // signed zeros and denormals mixed in
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = (i % 3 == 0) ? 0.0f : ((i % 3 == 1) ? -0.0f : 1e-38f);
            break;
        case 6:  // -INFINITY entries
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = (i % 7 == 0) ? -INFINITY : normal(rng);
            break;
        case 7:  // +INFINITY: every exponent underflows to zero
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = INFINITY;
            break;
        case 8:  // all -INFINITY
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = -INFINITY;
            break;
        case 9:  // one clear winner, everything else at the floor
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = (i == n / 3) ? 0.5f : -120.0f;
            break;
        default:
            for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = normal(rng) * 0.1f;
            break;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const int rows_per_pattern = argc > 1 ? std::atoi(argv[1]) : 64;
    const float scales[] = {1.0f, 2.5f, 0.5f};
    std::mt19937 rng(1234u);

    int * ref_ids = nullptr;
    float * ref_w = nullptr;
    int * par_ids = nullptr;
    float * par_w = nullptr;
    float * dev_logits = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&ref_ids), kMaxK * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ref_w), kMaxK * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&par_ids), kMaxK * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&par_w), kMaxK * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&dev_logits), kMaxExperts * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }

    BitResult result;
    long long comparisons = 0;
    // The device flag drives the exact/legacy dispatch inside both kernels; set
    // it explicitly instead of relying on the environment of the test process.
    for (int exact = 1; exact >= 0; --exact) {
        const int flag = exact;
        cudaMemcpyToSymbol(q38_moe_exact_router_flag, &flag, sizeof(flag));
        for (int pattern = 0; pattern < 12; ++pattern) {
            for (int rep = 0; rep < rows_per_pattern; ++rep) {
                std::vector<float> row;
                const int n = static_cast<int>(kMaxExperts);
                fill_case(row, pattern, n, rng);
                const int k = (rep % 2 == 0) ? 10 : (1 + (rep % kMaxK));
                const float scale = scales[rep % 3];
                cudaMemcpy(dev_logits, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
                q38_topk_softmax_kernel<<<1, 1, 0, nullptr>>>(
                    dev_logits, n, k, scale, ref_ids, ref_w);
                q38_router_exact_parallel_kernel<256><<<1, 256, 0, nullptr>>>(
                    dev_logits, n, k, scale, par_ids, par_w);
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    std::fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                    return 3;
                }
                int ri[kMaxK] = {};
                int pi[kMaxK] = {};
                float rw[kMaxK] = {};
                float pw[kMaxK] = {};
                cudaMemcpy(ri, ref_ids, k * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(pi, par_ids, k * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(rw, ref_w, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(pw, par_w, k * sizeof(float), cudaMemcpyDeviceToHost);
                for (int j = 0; j < k; ++j) {
                    ++comparisons;
                    const bool same = ri[j] == pi[j] &&
                        std::memcmp(&rw[j], &pw[j], sizeof(float)) == 0;
                    if (!same) {
                        ++result.mismatches;
                        if (result.first_case < 0) {
                            result.first_case = pattern * 1000 + rep * 10 + exact;
                            result.first_index = j;
                            std::fprintf(stderr,
                                "MISMATCH exact=%d pattern=%d rep=%d j=%d id %d/%d weight %g/%g bits %08x/%08x\n",
                                exact, pattern, rep, j, ri[j], pi[j], rw[j], pw[j],
                                *reinterpret_cast<unsigned *>(&rw[j]),
                                *reinterpret_cast<unsigned *>(&pw[j]));
                        }
                    }
                }
            }
        }
    }
    std::printf("{\"device\":\"");
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) std::printf("%s", prop.name);
    std::printf("\",\"comparisons\":%lld,\"mismatches\":%d,\"first_case\":%d,\"first_index\":%d}\n",
                comparisons, result.mismatches, result.first_case, result.first_index);
    cudaFree(ref_ids);
    cudaFree(ref_w);
    cudaFree(par_ids);
    cudaFree(par_w);
    cudaFree(dev_logits);
    return result.mismatches == 0 ? 0 : 1;
}
