// Bit-equality check of the Task N CPU MoE row kernels against the CUDA expert
// GEMV kernels the GPU path runs, on identical bytes.
//
// The host test (tests/qwen38_cpu_moe_host_test.cpp / _oracle_test.cpp) proves
// the CPU kernels equal the pinned llama.cpp AVX2 code. This one proves the
// *other* half of the parity contract: the CUDA kernels the engine used for the
// 64/64 runs produce the very same float for the same row, so swapping the MoE
// onto the CPU cannot move a token.
//
//   nvcc -O3 --use_fast_math --extra-device-vectorization -std=c++17 -arch=sm_80 \
//        -Xcompiler=-mavx2,-mfma,-ffp-contract=off -Isrc \
//        tests/cpu_moe_gpu_ref_test.cu -o /content/cpu_moe_gpu_ref
//   WARODECODE_DP4A=0 /content/cpu_moe_gpu_ref
//   WARODECODE_DP4A=1 /content/cpu_moe_gpu_ref

#include "lowbit.cuh"
#include "qwen38_cpu_moe.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace wd;

namespace {

int g_failures = 0;
int g_checks = 0;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check(const char * what, bool ok) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %s\n", what);
    }
}

bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

int block_bytes_of(GgmlType type) {
    switch (type) {
        case GgmlType::IQ2_S: return 82;
        case GgmlType::IQ3_S: return 110;
        case GgmlType::IQ4_NL: return 18;
        default: std::fprintf(stderr, "unsupported type\n"); std::exit(1);
    }
}

int quantum_of(GgmlType type) {
    return (type == GgmlType::IQ2_S || type == GgmlType::IQ3_S) ? 256 : 32;
}

// One row of ggml blocks with a finite normal F16 scale and random payload.
void fill_row(std::vector<uint8_t> & row, GgmlType type, int cols, std::mt19937 & rng) {
    const int q = quantum_of(type);
    const int bb = block_bytes_of(type);
    row.assign(static_cast<size_t>(cols / q) * bb, 0);
    for (size_t b = 0; b + 1 < row.size(); b += static_cast<size_t>(bb)) {
        row[b] = 0;                                   // low byte of d
        row[b + 1] = static_cast<uint8_t>(0x2e | (rng() & 0x01u));  // normal f16 scale
        for (int i = 2; i < bb; ++i) row[b + static_cast<size_t>(i)] = static_cast<uint8_t>(rng() & 0xffu);
    }
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("cpu-moe vs CUDA GEMV check (%s, WARODECODE_DP4A=%s)\n", prop.name,
                lowbit_dp4a_enabled() ? "1" : "0");

    const int hidden = 2560;
    const int ffn = 640;
    const int experts = 3;

    std::mt19937 rng(20261014u);
    cudaStream_t stream = nullptr;

    // ---------------------------------------------------------------
    // IQ2_S / IQ3_S x Q8_K (the routed gate/up projections).
    // ---------------------------------------------------------------
    for (GgmlType type : {GgmlType::IQ2_S, GgmlType::IQ3_S}) {
        const int cols = hidden;
        const int rows = ffn;
        const int bb = block_bytes_of(type);
        const int nb = cols / 256;

        std::vector<float> act(static_cast<size_t>(cols));
        for (int i = 0; i < cols; ++i) {
            act[static_cast<size_t>(i)] = (static_cast<float>(rng() & 0xffffu) / 65536.0f - 0.5f) * 5.0f;
        }
        std::vector<cpu_moe::q8_K> q8k(static_cast<size_t>(nb));
        cpu_moe::quantize_q8_K(act.data(), q8k.data(), nb);
        std::vector<int8_t> act_qs(static_cast<size_t>(cols));
        std::vector<float> act_d(static_cast<size_t>(nb));
        for (int b = 0; b < nb; ++b) {
            std::memcpy(act_qs.data() + static_cast<size_t>(b) * 256, q8k[b].qs, 256);
            act_d[static_cast<size_t>(b)] = q8k[b].d;
        }
        int8_t * dev_qs = nullptr;
        float * dev_d = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&dev_qs), act_qs.size()), "cudaMalloc act_qs");
        ck(cudaMalloc(reinterpret_cast<void **>(&dev_d), act_d.size() * sizeof(float)), "cudaMalloc act_d");
        ck(cudaMemcpy(dev_qs, act_qs.data(), act_qs.size(), cudaMemcpyHostToDevice), "copy act_qs");
        ck(cudaMemcpy(dev_d, act_d.data(), act_d.size() * sizeof(float), cudaMemcpyHostToDevice), "copy act_d");

        for (int e = 0; e < experts; ++e) {
            std::vector<uint8_t> host(static_cast<size_t>(rows) * static_cast<size_t>(cols / quantum_of(type)) * bb);
            for (int r = 0; r < rows; ++r) {
                std::vector<uint8_t> row;
                fill_row(row, type, cols, rng);
                std::memcpy(host.data() + static_cast<size_t>(r) * row.size(), row.data(), row.size());
            }
            LowBitMatrix m;
            m.rows = rows;
            m.cols = cols;
            m.type = type;
            m.owns = false;
            m.bytes = host.size();
            ck(cudaMalloc(reinterpret_cast<void **>(&m.data), m.bytes), "cudaMalloc weights");
            ck(cudaMemcpy(m.data, host.data(), m.bytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");

            float * out = nullptr;
            ck(cudaMalloc(reinterpret_cast<void **>(&out), static_cast<size_t>(rows) * sizeof(float)), "cudaMalloc out");
            if (type == GgmlType::IQ2_S) {
                launch_iq2_s_gemv_q8_k_f32(m, dev_qs, dev_d, out, stream);
            } else {
                launch_iq3_s_gemv_q8_k_f32(m, dev_qs, dev_d, out, stream);
            }
            ck(cudaStreamSynchronize(stream), "sync");
            std::vector<float> got(static_cast<size_t>(rows));
            ck(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");

            const size_t row_bytes = static_cast<size_t>(cols / 256) * static_cast<size_t>(bb);
            int bad = 0;
            float worst = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float want = (type == GgmlType::IQ2_S)
                    ? cpu_moe::row_iq2_s_q8_K(host.data() + static_cast<size_t>(r) * row_bytes, q8k.data(), cols)
                    : cpu_moe::row_iq3_s_q8_K(host.data() + static_cast<size_t>(r) * row_bytes, q8k.data(), cols);
                if (!same_bits(want, got[static_cast<size_t>(r)])) {
                    if (bad == 0) {
                        std::printf("    first mismatch row=%d cpu=%.9g cuda=%.9g\n", r, want,
                                    got[static_cast<size_t>(r)]);
                    }
                    ++bad;
                    worst = std::max(worst, std::fabs(want - got[static_cast<size_t>(r)]));
                }
            }
            char label[96];
            std::snprintf(label, sizeof(label), "%s x Q8_K row-for-row (expert %d)",
                          type == GgmlType::IQ2_S ? "IQ2_S" : "IQ3_S", e);
            check(label, bad == 0);
            if (bad) std::printf("    %d/%d rows differ, max |diff| = %.3g\n", bad, rows, worst);

            cudaFree(out);
            cudaFree(m.data);
        }
    }

    // ---------------------------------------------------------------
    // IQ4_NL x Q8_0 (the routed down projection).
    // ---------------------------------------------------------------
    {
        const int cols = ffn;
        const int rows = hidden;
        const int nb = cols / 32;
        std::vector<float> act(static_cast<size_t>(cols));
        for (int i = 0; i < cols; ++i) {
            act[static_cast<size_t>(i)] = (static_cast<float>(rng() & 0xffffu) / 65536.0f - 0.5f) * 6.0f;
        }
        std::vector<cpu_moe::q8_0> q0(static_cast<size_t>(nb));
        cpu_moe::quantize_q8_0(act.data(), q0.data(), nb);
        std::vector<int8_t> act_qs(static_cast<size_t>(cols));
        std::vector<uint16_t> act_d(static_cast<size_t>(nb));
        for (int b = 0; b < nb; ++b) {
            std::memcpy(act_qs.data() + static_cast<size_t>(b) * 32, q0[b].qs, 32);
            act_d[static_cast<size_t>(b)] = q0[b].d;
        }
        int8_t * dev_qs = nullptr;
        uint16_t * dev_d = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&dev_qs), act_qs.size()), "cudaMalloc act_qs");
        ck(cudaMalloc(reinterpret_cast<void **>(&dev_d), act_d.size() * sizeof(uint16_t)), "cudaMalloc act_d");
        ck(cudaMemcpy(dev_qs, act_qs.data(), act_qs.size(), cudaMemcpyHostToDevice), "copy act_qs");
        ck(cudaMemcpy(dev_d, act_d.data(), act_d.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy act_d");

        for (int e = 0; e < experts; ++e) {
            std::vector<uint8_t> host(static_cast<size_t>(rows) * static_cast<size_t>(nb) * 18u);
            for (int r = 0; r < rows; ++r) {
                std::vector<uint8_t> row;
                fill_row(row, GgmlType::IQ4_NL, cols, rng);
                std::memcpy(host.data() + static_cast<size_t>(r) * row.size(), row.data(), row.size());
            }
            LowBitMatrix m;
            m.rows = rows;
            m.cols = cols;
            m.type = GgmlType::IQ4_NL;
            m.owns = false;
            m.bytes = host.size();
            ck(cudaMalloc(reinterpret_cast<void **>(&m.data), m.bytes), "cudaMalloc weights");
            ck(cudaMemcpy(m.data, host.data(), m.bytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");

            float * out = nullptr;
            ck(cudaMalloc(reinterpret_cast<void **>(&out), static_cast<size_t>(rows) * sizeof(float)), "cudaMalloc out");
            launch_iq4_nl_gemv_q8_0_f32(m, dev_qs, reinterpret_cast<const __half *>(dev_d), out, stream);
            ck(cudaStreamSynchronize(stream), "sync");
            std::vector<float> got(static_cast<size_t>(rows));
            ck(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");

            const size_t row_bytes = static_cast<size_t>(nb) * 18u;
            int bad = 0;
            float worst = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float want = cpu_moe::row_iq4_nl_q8_0(
                    host.data() + static_cast<size_t>(r) * row_bytes, q0.data(), cols);
                if (!same_bits(want, got[static_cast<size_t>(r)])) {
                    if (bad == 0) {
                        std::printf("    first mismatch row=%d cpu=%.9g cuda=%.9g\n", r, want,
                                    got[static_cast<size_t>(r)]);
                    }
                    ++bad;
                    worst = std::max(worst, std::fabs(want - got[static_cast<size_t>(r)]));
                }
            }
            char label[96];
            std::snprintf(label, sizeof(label), "IQ4_NL x Q8_0 row-for-row (expert %d)", e);
            check(label, bad == 0);
            if (bad) std::printf("    %d/%d rows differ, max |diff| = %.3g\n", bad, rows, worst);

            cudaFree(out);
            cudaFree(m.data);
        }
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
