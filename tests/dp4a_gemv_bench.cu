// Isolated timing of the quantised GEMV kernels used by the decode loop.
//
// The end-to-end A/B only shows the net effect; this benchmark times one launch
// of every projection the routed experts run, for both the reference and the
// WARODECODE_DP4A=1 kernels, so the kernel-level speedup and the share of the
// token time that the GEMVs can account for are measurable separately.
//
//   nvcc -O3 --use_fast_math --extra-device-vectorization -std=c++17 \
//        -arch=sm_89 -Isrc tests/dp4a_gemv_bench.cu -o /content/dp4a_bench
//   WARODECODE_DP4A=0 /content/dp4a_bench
//   WARODECODE_DP4A=1 /content/dp4a_bench
//
// Shapes are the real Qwen3.8-Flash-Next ones: the routed gate/up projections
// are IQ2_S (blk.2 IQ3_S) with 640 rows and 2560 columns, the down projection is
// IQ4_NL with 2560 rows and 640 columns, and one token runs 10 experts through
// each of the 48 layers.

#include "lowbit.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace wd;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// (block elements, block bytes) for the four types the benchmark needs.
void type_geometry(GgmlType type, int & quantum, int & block_bytes) {
    switch (type) {
        case GgmlType::IQ2_S: quantum = 256; block_bytes = 82; break;
        case GgmlType::IQ3_S: quantum = 256; block_bytes = 110; break;
        case GgmlType::IQ4_NL: quantum = 32; block_bytes = 18; break;
        case GgmlType::Q6_K: quantum = 256; block_bytes = 210; break;
        default:
            std::fprintf(stderr, "unsupported benchmark type\n");
            std::exit(1);
    }
}

LowBitMatrix make_matrix(int rows, int cols, GgmlType type, unsigned seed) {
    int quantum = 0;
    int block_bytes = 0;
    type_geometry(type, quantum, block_bytes);
    if (cols % quantum != 0) {
        std::fprintf(stderr, "cols must be a multiple of %d\n", quantum);
        std::exit(1);
    }
    LowBitMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.type = type;
    m.owns = false;
    m.bytes = static_cast<size_t>(rows) * (cols / quantum) * block_bytes;

    std::vector<uint8_t> host(m.bytes);
    std::mt19937 rng(seed);
    for (auto & b : host) b = static_cast<uint8_t>(rng() & 0xffu);
    // Keep every block scale a finite normal F16 so nothing goes near NaN.
    for (size_t off = 0; off + 1 < host.size(); off += static_cast<size_t>(block_bytes)) {
        host[off + 1] = static_cast<uint8_t>(0x30 | (seed & 0x03u));
    }
    ck(cudaMalloc(reinterpret_cast<void **>(&m.data), m.bytes), "cudaMalloc matrix");
    ck(cudaMemcpy(m.data, host.data(), m.bytes, cudaMemcpyHostToDevice), "cudaMemcpy matrix");
    return m;
}

struct Timer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    Timer() {
        ck(cudaEventCreate(&start), "cudaEventCreate");
        ck(cudaEventCreate(&stop), "cudaEventCreate");
    }
    float measure(int iters, const std::function<void()> & launch) {
        for (int i = 0; i < 5; ++i) launch();
        ck(cudaEventRecord(start), "cudaEventRecord");
        for (int i = 0; i < iters; ++i) launch();
        ck(cudaEventRecord(stop), "cudaEventRecord");
        ck(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return ms * 1000.0f / static_cast<float>(iters); // microseconds
    }
};

void report(const char * what, int rows, int cols, double us, size_t bytes, int calls_per_token) {
    const double gib = static_cast<double>(bytes) / static_cast<double>(1u << 30);
    std::printf("  %-22s %5dx%-5d %9.2f us %8.1f GiB/s  x%4d = %8.2f ms/token\n",
                what, rows, cols, us, gib / (us * 1e-6), calls_per_token,
                us * calls_per_token / 1000.0);
}

} // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&device), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("GPU: %s sm_%d%d  WARODECODE_DP4A=%s\n", prop.name, prop.major, prop.minor,
                lowbit_dp4a_enabled() ? "1" : "0");

    cudaStream_t stream = nullptr;
    const int hidden = 2560;
    const int ffn = 640;
    const int iters = 200;

    LowBitMatrix gate_iq2 = make_matrix(ffn, hidden, GgmlType::IQ2_S, 1);
    LowBitMatrix gate_iq3 = make_matrix(ffn, hidden, GgmlType::IQ3_S, 2);
    LowBitMatrix down_iq4 = make_matrix(hidden, ffn, GgmlType::IQ4_NL, 3);

    int8_t * act_qs = nullptr;
    float * act_d = nullptr;
    __half * act_scales = nullptr;
    float * out = nullptr;
    const int k_blocks = hidden / 256;
    ck(cudaMalloc(reinterpret_cast<void **>(&act_qs), hidden), "cudaMalloc act_qs");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_d), static_cast<size_t>(k_blocks) * sizeof(float)), "cudaMalloc act_d");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_scales), static_cast<size_t>(hidden / 32) * sizeof(__half)), "cudaMalloc act_scales");
    ck(cudaMalloc(reinterpret_cast<void **>(&out), static_cast<size_t>(hidden) * sizeof(float)), "cudaMalloc out");
    ck(cudaMemset(act_qs, 0x31, hidden), "cudaMemset act_qs");
    ck(cudaMemset(act_d, 0, static_cast<size_t>(k_blocks) * sizeof(float)), "cudaMemset act_d");
    ck(cudaMemset(act_scales, 0, static_cast<size_t>(hidden / 32) * sizeof(__half)), "cudaMemset act_scales");

    Timer timer;
    std::printf("per-launch timings (%d iterations, %s dispatch):\n", iters,
                lowbit_dp4a_enabled() ? "DP4A" : "reference");

    const double us_iq2 = timer.measure(iters, [&] {
        launch_iq2_s_gemv_q8_k_f32(gate_iq2, act_qs, act_d, out, stream);
    });
    report("IQ2_S x Q8_K", ffn, hidden, us_iq2, gate_iq2.bytes, 47 * 10 * 2);

    const double us_iq3 = timer.measure(iters, [&] {
        launch_iq3_s_gemv_q8_k_f32(gate_iq3, act_qs, act_d, out, stream);
    });
    report("IQ3_S x Q8_K", ffn, hidden, us_iq3, gate_iq3.bytes, 1 * 10 * 2);

    const double us_iq4 = timer.measure(iters, [&] {
        launch_iq4_nl_gemv_q8_0_f32(down_iq4, act_qs, act_scales, out, stream);
    });
    report("IQ4_NL x Q8_0", hidden, ffn, us_iq4, down_iq4.bytes, 48 * 10);

    const double total_ms = (us_iq2 * 47 * 10 * 2 + us_iq3 * 1 * 10 * 2 + us_iq4 * 48 * 10) / 1000.0;
    std::printf("  routed-expert GEMV total: %.2f ms/token (one 64-token step is ~%.0f ms "
                "at 3.5 tok/s)\n", total_ms, 1000.0 / 3.5);

    // The Q6_K output head is one GEMV of vocab x hidden - by far the largest
    // single weight read of a token. It uses no codebook table (ql/qh/scales sit
    // in the block itself), so it is a separate optimisation target; this measures
    // what it costs today. Skipped when the 500 MiB weight buffer does not fit.
    const int vocab = 248320;
    int quantum = 0;
    int block_bytes = 0;
    type_geometry(GgmlType::Q6_K, quantum, block_bytes);
    const size_t head_bytes = static_cast<size_t>(vocab) * (hidden / quantum) * block_bytes;
    uint8_t * head_raw = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&head_raw), head_bytes) == cudaSuccess) {
        cudaMemset(head_raw, 0x24, head_bytes);
        LowBitMatrix head = borrow_lowbit_matrix(head_raw, vocab, hidden, GgmlType::Q6_K, head_bytes);
        int16_t * act_bsums = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&act_bsums),
                      static_cast<size_t>(hidden / 256) * 16u * sizeof(int16_t)), "cudaMalloc bsums");
        ck(cudaMemset(act_bsums, 0, static_cast<size_t>(hidden / 256) * 16u * sizeof(int16_t)),
           "cudaMemset bsums");
        float * head_out = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&head_out), static_cast<size_t>(vocab) * sizeof(float)),
           "cudaMalloc head_out");
        const double us_head = timer.measure(50, [&] {
            launch_q6_k_gemv_q8_k_f32(head, act_qs, act_d, act_bsums, head_out, stream);
        });
        report("Q6_K x Q8_K (head)", vocab, hidden, us_head, head_bytes, 1);
        std::printf("  head + routed experts: %.2f ms/token\n", total_ms + us_head / 1000.0);
    } else {
        std::printf("  Q6_K head skipped: %.0f MiB weight buffer does not fit\n",
                    static_cast<double>(head_bytes) / (1u << 20));
    }
    return 0;
}
