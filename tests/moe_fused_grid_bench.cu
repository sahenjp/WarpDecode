// Task G harness: expert-batched MoE (WARODECODE_MOE_FUSED_GRID) against the
// per-expert launches it replaces.
//
// Two questions are answered here, both on the device:
//   1. Are the batched kernels bit-identical to the single-expert kernels they
//      twin? Every comparison is a memcmp over the raw output bytes.
//   2. What does one MoE layer cost with 60 launches (10 experts x gate GEMV,
//      up GEMV, SiLU, Q8_0 activation quantisation, down GEMV, accumulation)
//      against the 5 launches of the batched path?
//
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 -Isrc \
//        tests/moe_fused_grid_bench.cu -o /content/moe_fused_bench
//   /content/moe_fused_bench
//
// Shapes are the real Qwen3.8-Flash-Next ones: hidden 2560, expert FFN 640,
// top-10 routing, 48 trunk layers. The 640/2560 rows here are the rows of one
// expert, so a batched launch covers all ten experts of a layer.

#include "lowbit.cuh"
#include "qwen38_moe.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void type_geometry(GgmlType type, int & quantum, int & block_bytes) {
    switch (type) {
        case GgmlType::IQ2_S: quantum = 256; block_bytes = 82; break;
        case GgmlType::IQ3_S: quantum = 256; block_bytes = 110; break;
        case GgmlType::IQ4_NL: quantum = 32; block_bytes = 18; break;
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
        for (int i = 0; i < 3; ++i) launch();
        ck(cudaEventRecord(start), "cudaEventRecord");
        for (int i = 0; i < iters; ++i) launch();
        ck(cudaEventRecord(stop), "cudaEventRecord");
        ck(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return ms * 1000.0f / static_cast<float>(iters);
    }
};

int g_failures = 0;

void check_bytes(const char * what, const void * a, const void * b, size_t bytes) {
    const bool same = std::memcmp(a, b, bytes) == 0;
    std::printf("  %-46s %s (%zu B)\n", what, same ? "bit-identical" : "MISMATCH", bytes);
    if (!same) ++g_failures;
}

std::vector<uint8_t> read_bytes(const void * device, size_t bytes) {
    std::vector<uint8_t> host(bytes);
    ck(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "copy to host");
    return host;
}

} // namespace

int main() {
    cudaFree(nullptr);
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("GPU: %s sm_%d%d  DP4A=%d smem=%d\n", prop.name, prop.major, prop.minor,
                lowbit_dp4a_enabled() ? 1 : 0, lowbit_dp4a_smem_enabled() ? 1 : 0);

    const int hidden = 2560;
    const int ffn = 640;
    const int k = 10;
    const int iters = 100;
    cudaStream_t stream = nullptr;

    std::vector<LowBitMatrix> gate(k), up(k), down(k);
    for (int e = 0; e < k; ++e) {
        gate[e] = make_matrix(ffn, hidden, GgmlType::IQ2_S, 100u + static_cast<unsigned>(e));
        up[e] = make_matrix(ffn, hidden, GgmlType::IQ2_S, 200u + static_cast<unsigned>(e));
        down[e] = make_matrix(hidden, ffn, GgmlType::IQ4_NL, 300u + static_cast<unsigned>(e));
    }

    const uint8_t ** d_gate_ptrs = nullptr;
    const uint8_t ** d_up_ptrs = nullptr;
    const uint8_t ** d_down_ptrs = nullptr;
    std::vector<const uint8_t *> h_gate(k), h_up(k), h_down(k);
    for (int e = 0; e < k; ++e) {
        h_gate[e] = gate[e].data;
        h_up[e] = up[e].data;
        h_down[e] = down[e].data;
    }
    ck(cudaMalloc(reinterpret_cast<void **>(&d_gate_ptrs), k * sizeof(uint8_t *)), "cudaMalloc ptrs");
    ck(cudaMalloc(reinterpret_cast<void **>(&d_up_ptrs), k * sizeof(uint8_t *)), "cudaMalloc ptrs");
    ck(cudaMalloc(reinterpret_cast<void **>(&d_down_ptrs), k * sizeof(uint8_t *)), "cudaMalloc ptrs");
    ck(cudaMemcpy(d_gate_ptrs, h_gate.data(), k * sizeof(uint8_t *), cudaMemcpyHostToDevice), "copy ptrs");
    ck(cudaMemcpy(d_up_ptrs, h_up.data(), k * sizeof(uint8_t *), cudaMemcpyHostToDevice), "copy ptrs");
    ck(cudaMemcpy(d_down_ptrs, h_down.data(), k * sizeof(uint8_t *), cudaMemcpyHostToDevice), "copy ptrs");

    int8_t * act_qs = nullptr;
    float * act_d = nullptr;
    __half * act_scales = nullptr;
    const int q8k_blocks = hidden / 256;
    ck(cudaMalloc(reinterpret_cast<void **>(&act_qs), hidden), "cudaMalloc act_qs");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_d), static_cast<size_t>(q8k_blocks) * sizeof(float)), "cudaMalloc act_d");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_scales), static_cast<size_t>(hidden / 32) * sizeof(__half)), "cudaMalloc act_scales");
    ck(cudaMemset(act_qs, 0x31, hidden), "cudaMemset act_qs");
    ck(cudaMemset(act_d, 0, static_cast<size_t>(q8k_blocks) * sizeof(float)), "cudaMemset act_d");
    ck(cudaMemset(act_scales, 0, static_cast<size_t>(hidden / 32) * sizeof(__half)), "cudaMemset act_scales");

    float * y_serial = nullptr;
    float * y_batched = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&y_serial), static_cast<size_t>(k) * ffn * sizeof(float)), "cudaMalloc y_serial");
    ck(cudaMalloc(reinterpret_cast<void **>(&y_batched), static_cast<size_t>(k) * ffn * sizeof(float)), "cudaMalloc y_batched");

    Timer timer;

    // -----------------------------------------------------------------------
    // 1. Gate/up GEMV: 10 separate launches against one batched launch.
    // -----------------------------------------------------------------------
    std::printf("\n[1] IQ2_S x Q8_K gate/up, %d experts of %dx%d\n", k, ffn, hidden);
    ck(cudaMemset(y_serial, 0, static_cast<size_t>(k) * ffn * sizeof(float)), "memset");
    ck(cudaMemset(y_batched, 0, static_cast<size_t>(k) * ffn * sizeof(float)), "memset");
    for (int e = 0; e < k; ++e) {
        launch_iq2_s_gemv_q8_k_f32(gate[e], act_qs, act_d, y_serial + static_cast<size_t>(e) * ffn, stream);
    }
    const bool iq2_ok = launch_iq2_s_gemv_q8_k_experts_f32(
        d_gate_ptrs, k, ffn, hidden, act_qs, act_d, y_batched, stream);
    ck(cudaStreamSynchronize(stream), "sync");
    check_bytes("IQ2_S batched vs serial", read_bytes(y_serial, static_cast<size_t>(k) * ffn * 4).data(),
                read_bytes(y_batched, static_cast<size_t>(k) * ffn * 4).data(),
                static_cast<size_t>(k) * ffn * 4);
    if (!iq2_ok) {
        std::printf("  (batched IQ2_S launcher reported no DP4A support)\n");
    }
    const double us_iq2_serial = timer.measure(iters, [&] {
        for (int e = 0; e < k; ++e) {
            launch_iq2_s_gemv_q8_k_f32(gate[e], act_qs, act_d, y_serial + static_cast<size_t>(e) * ffn, stream);
        }
    });
    const double us_iq2_batched = timer.measure(iters, [&] {
        launch_iq2_s_gemv_q8_k_experts_f32(d_gate_ptrs, k, ffn, hidden, act_qs, act_d, y_batched, stream);
    });
    std::printf("  10 launches: %8.2f us total (%6.2f us each)\n", us_iq2_serial, us_iq2_serial / k);
    std::printf("   1 launch : %8.2f us total\n", us_iq2_batched);
    std::printf("   ratio    : %8.2fx\n", us_iq2_serial / us_iq2_batched);

    // -----------------------------------------------------------------------
    // 2. Down projection (IQ4_NL).
    // -----------------------------------------------------------------------
    const int down_rows = hidden;
    const int down_cols = ffn;
    float * d_serial = nullptr;
    float * d_batched = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&d_serial), static_cast<size_t>(k) * down_rows * sizeof(float)), "cudaMalloc d_serial");
    ck(cudaMalloc(reinterpret_cast<void **>(&d_batched), static_cast<size_t>(k) * down_rows * sizeof(float)), "cudaMalloc d_batched");
    // One activation per expert: the down projection consumes each expert's own
    // SiLU(gate) * up product, so the batched buffer is k slices of ffn values
    // and the serial comparison has to walk the same slices.
    int8_t * down_act_qs = nullptr;
    __half * down_act_scales = nullptr;
    const int down_act_blocks = ffn / 32;
    ck(cudaMalloc(reinterpret_cast<void **>(&down_act_qs), static_cast<size_t>(k) * ffn), "cudaMalloc down act");
    ck(cudaMalloc(reinterpret_cast<void **>(&down_act_scales),
                  static_cast<size_t>(k) * down_act_blocks * sizeof(__half)), "cudaMalloc down scales");
    {
        std::mt19937 rng(31);
        std::vector<int8_t> qs(static_cast<size_t>(k) * ffn);
        for (auto & v : qs) v = static_cast<int8_t>(static_cast<int>(rng() % 255u) - 127);
        std::vector<__half> sc(static_cast<size_t>(k) * down_act_blocks);
        for (auto & v : sc) v = __float2half(0.01f + static_cast<float>(rng() % 100u) * 0.001f);
        ck(cudaMemcpy(down_act_qs, qs.data(), qs.size(), cudaMemcpyHostToDevice), "copy down act");
        ck(cudaMemcpy(down_act_scales, sc.data(), sc.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy down scales");
    }

    std::printf("\n[2] IQ4_NL x Q8_0 down, %d experts of %dx%d (per-expert activation)\n", k, down_rows, down_cols);
    for (int e = 0; e < k; ++e) {
        launch_iq4_nl_gemv_q8_0_f32(
            down[e], down_act_qs + static_cast<size_t>(e) * ffn,
            down_act_scales + static_cast<size_t>(e) * down_act_blocks,
            d_serial + static_cast<size_t>(e) * down_rows, stream);
    }
    const bool iq4_ok = launch_iq4_nl_gemv_q8_0_experts_f32(
        d_down_ptrs, k, down_rows, down_cols, down_act_qs, down_act_scales, d_batched, stream);
    ck(cudaStreamSynchronize(stream), "sync");
    check_bytes("IQ4_NL batched vs serial", read_bytes(d_serial, static_cast<size_t>(k) * down_rows * 4).data(),
                read_bytes(d_batched, static_cast<size_t>(k) * down_rows * 4).data(),
                static_cast<size_t>(k) * down_rows * 4);
    if (!iq4_ok) {
        std::printf("  (batched IQ4_NL launcher reported no DP4A support)\n");
    }
    const double us_iq4_serial = timer.measure(iters, [&] {
        for (int e = 0; e < k; ++e) {
            launch_iq4_nl_gemv_q8_0_f32(
                down[e], down_act_qs + static_cast<size_t>(e) * ffn,
                down_act_scales + static_cast<size_t>(e) * down_act_blocks,
                d_serial + static_cast<size_t>(e) * down_rows, stream);
        }
    });
    const double us_iq4_batched = timer.measure(iters, [&] {
        launch_iq4_nl_gemv_q8_0_experts_f32(d_down_ptrs, k, down_rows, down_cols,
                                            down_act_qs, down_act_scales, d_batched, stream);
    });
    std::printf("  10 launches: %8.2f us total (%6.2f us each)\n", us_iq4_serial, us_iq4_serial / k);
    std::printf("   1 launch : %8.2f us total\n", us_iq4_batched);
    std::printf("   ratio    : %8.2fx\n", us_iq4_serial / us_iq4_batched);

    // -----------------------------------------------------------------------
    // 3. SiLU(gate) * up and its Q8_0 quantisation.
    // -----------------------------------------------------------------------
    float * gate_up_serial = nullptr;
    float * gate_up_batched = nullptr;
    float * act_serial = nullptr;
    float * act_batched = nullptr;
    int8_t * qv_serial = nullptr;
    int8_t * qv_batched = nullptr;
    __half * qs_serial = nullptr;
    __half * qs_batched = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&gate_up_serial), static_cast<size_t>(2) * k * ffn * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&gate_up_batched), static_cast<size_t>(2) * k * ffn * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_serial), static_cast<size_t>(k) * ffn * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&act_batched), static_cast<size_t>(k) * ffn * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&qv_serial), static_cast<size_t>(k) * ffn), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&qv_batched), static_cast<size_t>(k) * ffn), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&qs_serial), static_cast<size_t>(k) * (ffn / 32) * sizeof(__half)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&qs_batched), static_cast<size_t>(k) * (ffn / 32) * sizeof(__half)), "alloc");

    // Same gate/up input in both buffers; the serial path applies SiLU in place.
    {
        std::mt19937 rng(7);
        std::vector<float> host(static_cast<size_t>(2) * k * ffn);
        for (auto & v : host) {
            v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) * 0.01f;
        }
        ck(cudaMemcpy(gate_up_serial, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(gate_up_batched, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    }
    constexpr int threads = 256;
    const int ffn_blocks = (ffn + threads - 1) / threads;
    const int q8_blocks = ((ffn / 32) + threads / 32 - 1) / (threads / 32);

    std::printf("\n[3] SiLU(gate)*up + Q8_0 activation quantisation\n");
    for (int e = 0; e < k; ++e) {
        float * g = gate_up_serial + static_cast<size_t>(2) * e * ffn;
        q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(g, ffn, 1.0f);
        q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
            g, g + ffn, act_serial + static_cast<size_t>(e) * ffn, ffn);
        q38_hc_q8_0_quantize_f32_exact_kernel<<<q8_blocks, threads, 0, stream>>>(
            act_serial + static_cast<size_t>(e) * ffn, ffn,
            qv_serial + static_cast<size_t>(e) * ffn,
            qs_serial + static_cast<size_t>(e) * (ffn / 32));
    }
    q38_moe_silu_scale_mul_experts_kernel<<<dim3(ffn_blocks, k), threads, 0, stream>>>(
        gate_up_batched, act_batched, ffn, 1.0f);
    q38_moe_q8_0_quantize_experts_kernel<<<dim3(q8_blocks, k), threads, 0, stream>>>(
        act_batched, qv_batched, qs_batched, ffn);
    ck(cudaStreamSynchronize(stream), "sync");
    check_bytes("SiLU*up batched vs serial", read_bytes(act_serial, static_cast<size_t>(k) * ffn * 4).data(),
                read_bytes(act_batched, static_cast<size_t>(k) * ffn * 4).data(),
                static_cast<size_t>(k) * ffn * 4);
    check_bytes("Q8_0 values batched vs serial", read_bytes(qv_serial, static_cast<size_t>(k) * ffn).data(),
                read_bytes(qv_batched, static_cast<size_t>(k) * ffn).data(),
                static_cast<size_t>(k) * ffn);
    check_bytes("Q8_0 scales batched vs serial",
                read_bytes(qs_serial, static_cast<size_t>(k) * (ffn / 32) * sizeof(__half)).data(),
                read_bytes(qs_batched, static_cast<size_t>(k) * (ffn / 32) * sizeof(__half)).data(),
                static_cast<size_t>(k) * (ffn / 32) * sizeof(__half));

    // -----------------------------------------------------------------------
    // 4. Weighted accumulation, whose order matters for bit-exactness.
    // -----------------------------------------------------------------------
    float * accum_serial = nullptr;
    float * accum_batched = nullptr;
    float * d_weights = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&accum_serial), static_cast<size_t>(hidden) * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&accum_batched), static_cast<size_t>(hidden) * sizeof(float)), "alloc");
    ck(cudaMalloc(reinterpret_cast<void **>(&d_weights), static_cast<size_t>(k) * sizeof(float)), "alloc");
    {
        std::mt19937 rng(11);
        std::vector<float> w(k);
        for (auto & v : w) v = static_cast<float>(rng() % 1000) * 0.0001f;
        ck(cudaMemcpy(d_weights, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
    }
    ck(cudaMemset(accum_serial, 0, static_cast<size_t>(hidden) * sizeof(float)), "memset");
    ck(cudaMemset(accum_batched, 0, static_cast<size_t>(hidden) * sizeof(float)), "memset");
    const int hidden_blocks = (hidden + threads - 1) / threads;
    std::printf("\n[4] weighted accumulation of the %d expert outputs\n", k);
    for (int e = 0; e < k; ++e) {
        std::vector<float> w(k);
        ck(cudaMemcpy(w.data(), d_weights, w.size() * sizeof(float), cudaMemcpyDeviceToHost), "read weights");
        q38_weighted_accum_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
            d_batched + static_cast<size_t>(e) * down_rows, w[static_cast<size_t>(e)], accum_serial, hidden);
    }
    q38_moe_weighted_accum_experts_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
        d_batched, d_weights, accum_batched, hidden, k);
    ck(cudaStreamSynchronize(stream), "sync");
    check_bytes("accumulation batched vs serial",
                read_bytes(accum_serial, static_cast<size_t>(hidden) * 4).data(),
                read_bytes(accum_batched, static_cast<size_t>(hidden) * 4).data(),
                static_cast<size_t>(hidden) * 4);

    // -----------------------------------------------------------------------
    // 6. The production helpers, end to end.
    //
    // Section 5 launches the raw kernels; this one goes through
    // q38_moe_upload_expert_ptr_table() + q38_moe_run_batched_experts() - the
    // exact pair the engine's batched runner calls - and compares the routed
    // accumulation against the launch sequence of
    // q38_moe_run_routed_experts_q8_act_separate(). A host/device pointer mix-up
    // in the table plumbing shows up here and nowhere else.
    // -----------------------------------------------------------------------
    std::printf("\n[6] production helpers vs the per-expert runner sequence\n");
    Qwen38MoeScratch scratch;
    scratch.initialize(512, k, hidden, ffn);
    {
        std::mt19937 rng(23);
        std::vector<int8_t> qs(static_cast<size_t>(hidden));
        for (auto & v : qs) v = static_cast<int8_t>(static_cast<int>(rng() % 255u) - 127);
        std::vector<float> ds(static_cast<size_t>(q8k_blocks));
        for (auto & v : ds) v = static_cast<float>(rng() % 100u) * 0.001f + 0.01f;
        ck(cudaMemcpy(scratch.rx_q8k_values(), qs.data(), qs.size(), cudaMemcpyHostToDevice), "copy act");
        ck(cudaMemcpy(scratch.rx_q8k_scales(), ds.data(), ds.size() * sizeof(float), cudaMemcpyHostToDevice), "copy act");
    }
    std::vector<float> h_weights(static_cast<size_t>(k));
    for (int j = 0; j < k; ++j) h_weights[static_cast<size_t>(j)] = 0.05f + 0.01f * static_cast<float>(j);
    ck(cudaMemcpy(scratch.d_weights(), h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights");

    Q38MoeExpertPtrTable table{};
    for (int j = 0; j < k; ++j) {
        table.gate_up[2 * j] = gate[static_cast<size_t>(j)].data;
        table.gate_up[2 * j + 1] = up[static_cast<size_t>(j)].data;
        table.down[j] = down[static_cast<size_t>(j)].data;
    }
    ck(cudaMemset(scratch.d_accum(), 0, static_cast<size_t>(hidden) * sizeof(float)), "memset");
    q38_moe_upload_expert_ptr_table(table, scratch, stream);
    q38_moe_run_batched_experts<false>(
        scratch.expert_ptr_table(), GgmlType::IQ2_S, 2 * k, ffn, k, nullptr, scratch, stream);
    ck(cudaStreamSynchronize(stream), "sync");
    const std::vector<uint8_t> accum_batched_e2e =
        read_bytes(scratch.d_accum(), static_cast<size_t>(hidden) * sizeof(float));

    ck(cudaMemset(scratch.d_accum(), 0, static_cast<size_t>(hidden) * sizeof(float)), "memset");
    for (int j = 0; j < k; ++j) {
        q38_moe_gemv_gate_up_q8_k_f32(
            gate[static_cast<size_t>(j)], scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
            scratch.rx_gate_f32(), stream);
        q38_moe_gemv_gate_up_q8_k_f32(
            up[static_cast<size_t>(j)], scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
            scratch.rx_up_f32(), stream);
        q38_moe_finish_routed_expert_q8_act(
            scratch.rx_gate_f32(), scratch.rx_up_f32(), down[static_cast<size_t>(j)], scratch,
            false, stream);
        q38_moe_accumulate_expert(
            scratch.rx_down_f32(), scratch.d_down(), h_weights[static_cast<size_t>(j)], scratch,
            true, stream);
    }
    ck(cudaStreamSynchronize(stream), "sync");
    check_bytes("production batched helper vs per-expert runner",
                accum_batched_e2e.data(),
                read_bytes(scratch.d_accum(), static_cast<size_t>(hidden) * sizeof(float)).data(),
                static_cast<size_t>(hidden) * sizeof(float));

    // -----------------------------------------------------------------------
    // 5. A whole layer: 60 launches against 5.
    // -----------------------------------------------------------------------
    std::printf("\n[5] one MoE layer of %d experts\n", k);
    const double us_layer_serial = timer.measure(iters, [&] {
        for (int e = 0; e < k; ++e) {
            launch_iq2_s_gemv_q8_k_f32(gate[e], act_qs, act_d, y_serial + static_cast<size_t>(e) * ffn, stream);
            launch_iq2_s_gemv_q8_k_f32(up[e], act_qs, act_d, y_serial + static_cast<size_t>(e) * ffn, stream);
            float * g = gate_up_serial + static_cast<size_t>(2) * e * ffn;
            q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(g, ffn, 1.0f);
            q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(g, g + ffn, act_serial, ffn);
            q38_hc_q8_0_quantize_f32_exact_kernel<<<q8_blocks, threads, 0, stream>>>(
                act_serial, ffn, qv_serial, qs_serial);
            launch_iq4_nl_gemv_q8_0_f32(down[e], qv_serial, qs_serial, d_serial, stream);
            float weight = 0.1f;
            q38_weighted_accum_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
                d_serial, weight, accum_serial, hidden);
        }
    });
    const double us_layer_batched = timer.measure(iters, [&] {
        launch_iq2_s_gemv_q8_k_experts_f32(d_gate_ptrs, k, ffn, hidden, act_qs, act_d, y_batched, stream);
        launch_iq2_s_gemv_q8_k_experts_f32(d_up_ptrs, k, ffn, hidden, act_qs, act_d, y_batched, stream);
        q38_moe_silu_scale_mul_experts_kernel<<<dim3(ffn_blocks, k), threads, 0, stream>>>(
            gate_up_batched, act_batched, ffn, 1.0f);
        q38_moe_q8_0_quantize_experts_kernel<<<dim3(q8_blocks, k), threads, 0, stream>>>(
            act_batched, qv_batched, qs_batched, ffn);
        launch_iq4_nl_gemv_q8_0_experts_f32(d_down_ptrs, k, down_rows, down_cols,
                                            qv_batched, qs_batched, d_batched, stream);
        q38_moe_weighted_accum_experts_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
            d_batched, d_weights, accum_batched, hidden, k);
    });
    // Host-side enqueue cost of the same two sequences (sync every 4 layers so
    // the launch queue never becomes the limiter).
    auto host_us_per_layer = [&](const std::function<void()> & f) {
        const int reps = 4;
        const int rounds = 25;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < rounds; ++r) {
            for (int i = 0; i < reps; ++i) f();
            ck(cudaStreamSynchronize(stream), "sync");
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() /
               static_cast<double>(reps * rounds);
    };
    const double host_serial = host_us_per_layer([&] {
        for (int e = 0; e < k; ++e) {
            launch_iq2_s_gemv_q8_k_f32(gate[e], act_qs, act_d, y_serial + static_cast<size_t>(e) * ffn, stream);
        }
    });
    const double host_batched = host_us_per_layer([&] {
        launch_iq2_s_gemv_q8_k_experts_f32(d_gate_ptrs, k, ffn, hidden, act_qs, act_d, y_batched, stream);
    });

    std::printf("  60 launches serial : %8.2f us/layer  -> %6.2f ms/token (48 layers)\n",
                us_layer_serial, us_layer_serial * 48.0 / 1000.0);
    std::printf("   5 launches batched: %8.2f us/layer  -> %6.2f ms/token (48 layers)\n",
                us_layer_batched, us_layer_batched * 48.0 / 1000.0);
    std::printf("   ratio             : %8.2fx\n", us_layer_serial / us_layer_batched);
    std::printf("  host enqueue of 10 gate GEMVs : %8.2f us/layer (%5.2f us/launch)\n",
                host_serial, host_serial / k);
    std::printf("  host enqueue of  1 batched GEMV: %8.2f us/layer\n", host_batched);

    std::printf("\n%s (%d mismatch%s)\n", g_failures ? "FAIL" : "PASS",
                g_failures, g_failures == 1 ? "" : "es");
    return g_failures ? 1 : 0;
}
