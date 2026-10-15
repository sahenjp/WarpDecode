// Task W harness: the ILP (software-pipelined) routed-expert GEMV twins against
// the sequential kernels they replace, on the real Qwen3.8-Flash-Next geometry.
//
// Two questions, both on the device:
//   1. Are the ILP kernels bit-identical to the sequential ones? (memcmp over
//      the raw output bytes, for every expert and every row.)
//   2. What does one 10-expert gate/up or down projection cost with ILP = 1, 2,
//      4, 8? Reported as us and as effective GB/s of weight bytes read, which is
//      what the kernel is bound by.
//
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_80 -Isrc \
//        tests/qwen38_expert_ilp_bench.cu -o /content/expert_ilp_bench
//   /content/expert_ilp_bench
//
// Shapes are the real ones: hidden 2560, expert FFN 640, top-10 routing, 48
// trunk layers; gate/up are IQ2_S (82 bytes per 256 values) and down is IQ4_NL
// (18 bytes per 32 values).
#include "lowbit.cuh"
#include "qwen38_moe.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace wd;

// ---------------------------------------------------------------------------
// Diagnostics: what is the ceiling of the IQ4_NL and IQ2_S expert GEMVs?
//
// Both kernels move 0.9-1.0 MB per expert and reach only 100-180 GB/s, far below
// the Q8_0 dense GEMV (543 GB/s on the same GPU). These two variants keep the
// identical load pattern and loop structure but replace the decode with the
// cheapest possible consumer of the same bytes:
//
//   *_raw  : the nibble bytes are used as weights directly (no codebook lookup)
//   *_flat : the IQ2_S grid lookup is replaced by the grid index itself
//
// The numbers they produce are WRONG on purpose; only the timing is meaningful.
// If they run at the same speed as the real kernels, the decode is free and the
// access pattern (18-byte blocks / 82-byte superblocks, few bytes per warp per
// iteration) is the limit. If they are much faster, the decode is the limit.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float bench_iq4_row_raw(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        int row,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int lane_in_row = lane & 15;
    const int chunk = lane_in_row & 7;
    const int parity = lane_in_row >> 3;

    const int blocks_per_row = cols >> 5;
    const size_t row_stride = static_cast<size_t>(blocks_per_row) * 18u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;
    const int byte_base = (chunk & 3) * 4;
    const bool high_nibble = chunk >= 4;

    float accumulator = 0.0f;
    for (int b = parity; b < blocks_per_row; b += 2) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 18u;
        const float d = lowbit_half(blk) * __half2float(x_scales[b]);
        const uint8_t * qs = blk + 2 + byte_base;
        const unsigned packed =
            static_cast<unsigned>(*reinterpret_cast<const uint16_t *>(qs)) |
            (static_cast<unsigned>(*reinterpret_cast<const uint16_t *>(qs + 2)) << 16);
        const unsigned weights4 = high_nibble
            ? ((packed >> 4) & 0x0f0f0f0fu)
            : (packed & 0x0f0f0f0fu);
        const unsigned activation4 = *reinterpret_cast<const unsigned *>(
            x_qs + static_cast<size_t>(b) * 32u + static_cast<size_t>(chunk) * 4u);
        accumulator = fmaf(
            d,
            static_cast<float>(lowbit_dp4a(
                static_cast<int>(activation4), static_cast<int>(weights4), 0)),
            accumulator);
    }
    const float combined =
        __fadd_rn(accumulator, __shfl_xor_sync(0xffffffffu, accumulator, 8));
    const float r = __fadd_rn(combined, __shfl_down_sync(0xffffffffu, combined, 4));
    const float s = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2));
    return __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 1));
}

__global__ void bench_iq4_raw_kernel(
        const uint8_t * const * __restrict__ w_experts,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int row_in_warp = lane >> 4;
    const int output_row =
        (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2 + row_in_warp;
    const int row = min(output_row, rows - 1);
    const int activation = blockIdx.y;
    const float value = bench_iq4_row_raw(
        w_experts[activation],
        x_qs + static_cast<size_t>(activation) * static_cast<size_t>(cols),
        x_scales + static_cast<size_t>(activation) * static_cast<size_t>(cols / 32),
        row, cols);
    if ((lane & 15) != 0 || output_row >= rows) return;
    y[static_cast<size_t>(activation) * static_cast<size_t>(rows) + output_row] = value;
}

__device__ __forceinline__ float bench_iq2_row_flat(
        const uint8_t * __restrict__ w,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        int row,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int chunk = lane & 7;
    const int phase = lane >> 3;
    const int group = chunk >> 1;
    const int grid_half = chunk & 1;
    const int superblocks_per_row = cols >> 8;
    const size_t row_stride = static_cast<size_t>(superblocks_per_row) * 82u;
    const uint8_t * row_ptr = w + static_cast<size_t>(row) * row_stride;
    const bool scale_low_nibble = chunk < 4;

    float accumulator = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; ++sb) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        const float d_w = lowbit_half(blk);
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        const int8_t * a = x_qs + static_cast<size_t>(sb) * 256u;

        int partial = 0;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ib32 = phase + 4 * p;
            const uint8_t scale_byte = scales[ib32];
            const int ls = scale_low_nibble
                ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                : 1 + 2 * static_cast<int>(scale_byte >> 4);
            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u));
            // The grid lookup becomes a function of the index itself.
            const unsigned quad = static_cast<unsigned>(grid_index) * 0x01010101u;
            const unsigned sign_bits = static_cast<unsigned>(
                signs[ib32 * 4 + group] >> (4 * grid_half));
            const unsigned weights4 = lowbit_sign_i8x4(quad, sign_bits);
            const unsigned activation4 = *reinterpret_cast<const unsigned *>(
                a + static_cast<size_t>(ib32) * 32u + static_cast<size_t>(chunk) * 4u);
            partial += lowbit_dp4a(
                static_cast<int>(activation4), static_cast<int>(weights4), 0) * ls;
        }
        partial += __shfl_down_sync(0xffffffffu, partial, 8);
        partial += __shfl_down_sync(0xffffffffu, partial, 16);
        if (lane < 8) {
            accumulator = fmaf(d_w * x_d[sb], static_cast<float>(partial), accumulator);
        }
    }
    const float r = __fadd_rn(accumulator, __shfl_down_sync(0xffffffffu, accumulator, 4));
    const float s = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2));
    return __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 1));
}

__global__ void bench_iq2_flat_kernel(
        const uint8_t * const * __restrict__ w_experts,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float value = bench_iq2_row_flat(w_experts[blockIdx.y], x_qs, x_d, row, cols);
    if ((threadIdx.x & 31) != 0) return;
    y[static_cast<size_t>(blockIdx.y) * static_cast<size_t>(rows) + row] = value;
}


namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Timer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    Timer() { ck(cudaEventCreate(&start), "cudaEventCreate"); ck(cudaEventCreate(&stop), "cudaEventCreate"); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    template <typename F>
    double measure(int iters, F && body) {
        body();
        ck(cudaStreamSynchronize(0), "warmup sync");
        ck(cudaEventRecord(start, 0), "record start");
        for (int i = 0; i < iters; ++i) body();
        ck(cudaEventRecord(stop, 0), "record stop");
        ck(cudaStreamSynchronize(0), "sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        return ms * 1000.0f / static_cast<float>(iters);
    }
};

} // namespace

int main() {
    // The production launchers pick the ILP kernel from the environment; this
    // harness always wants the sequential one as its reference.
    unsetenv("WARODECODE_EXPERT_GEMV_ILP");

    const int experts = 10;
    const int hidden = 2560;
    const int ffn = 640;
    const int iters = 200;

    std::mt19937 rng(20261015u);
    auto fill = [&](std::vector<uint8_t> & v) {
        for (uint8_t & b : v) b = static_cast<uint8_t>(rng() & 0xffu);
    };

    std::printf("== IQ2_S gate/up: %d experts x [%d rows, %d cols] (82 B / 256 values)\n",
                experts, ffn, hidden);
    const size_t iq2_bytes = static_cast<size_t>(ffn) * (hidden / 256u) * 82u;
    std::vector<uint8_t> iq2_host(iq2_bytes);
    fill(iq2_host);
    // Rows start on an even byte (the kernels use 16-bit loads for the packed
    // nibbles), so the stride of a row is 820 bytes here and the base is even.
    uint8_t * d_iq2 = nullptr;
    ck(cudaMalloc(&d_iq2, iq2_bytes * experts), "cudaMalloc iq2");
    std::vector<uint8_t *> iq2_ptrs(experts);
    for (int e = 0; e < experts; ++e) {
        ck(cudaMemcpy(d_iq2 + iq2_bytes * e, iq2_host.data(), iq2_bytes,
                      cudaMemcpyHostToDevice), "upload iq2");
        iq2_ptrs[e] = d_iq2 + iq2_bytes * e;
    }
    uint8_t ** d_iq2_ptrs = nullptr;
    ck(cudaMalloc(&d_iq2_ptrs, sizeof(uint8_t *) * experts), "cudaMalloc iq2 ptrs");
    ck(cudaMemcpy(d_iq2_ptrs, iq2_ptrs.data(), sizeof(uint8_t *) * experts,
                  cudaMemcpyHostToDevice), "upload iq2 ptrs");

    std::vector<int8_t> act_qs(hidden);
    std::vector<float> act_d(hidden / 256);
    for (int8_t & v : act_qs) v = static_cast<int8_t>(rng() & 0xffu);
    for (float & v : act_d) v = static_cast<float>(static_cast<int>(rng() % 1000) - 500) * 0.001f;
    int8_t * d_act_qs = nullptr;
    float * d_act_d = nullptr;
    ck(cudaMalloc(&d_act_qs, act_qs.size()), "cudaMalloc act qs");
    ck(cudaMalloc(&d_act_d, act_d.size() * sizeof(float)), "cudaMalloc act d");
    ck(cudaMemcpy(d_act_qs, act_qs.data(), act_qs.size(), cudaMemcpyHostToDevice), "upload act qs");
    ck(cudaMemcpy(d_act_d, act_d.data(), act_d.size() * sizeof(float), cudaMemcpyHostToDevice), "upload act d");

    const size_t iq2_out_bytes = static_cast<size_t>(experts) * ffn * sizeof(float);
    float * d_ref = nullptr;
    float * d_got = nullptr;
    ck(cudaMalloc(&d_ref, iq2_out_bytes), "cudaMalloc ref");
    ck(cudaMalloc(&d_got, iq2_out_bytes), "cudaMalloc got");
    ck(cudaMemset(d_ref, 0, iq2_out_bytes), "memset ref");

    const bool ref_ok = launch_iq2_s_gemv_q8_k_experts_f32(
        d_iq2_ptrs, experts, ffn, hidden, d_act_qs, d_act_d, d_ref, 0, 256);
    ck(cudaStreamSynchronize(0), "sync ref");
    if (!ref_ok) {
        std::fprintf(stderr, "IQ2_S DP4A kernels unavailable\n");
        return 1;
    }
    std::vector<float> ref_iq2(static_cast<size_t>(experts) * ffn);
    ck(cudaMemcpy(ref_iq2.data(), d_ref, iq2_out_bytes, cudaMemcpyDeviceToHost), "read ref iq2");

    Timer timer;
    const double iq2_base = timer.measure(iters, [&] {
        launch_iq2_s_gemv_q8_k_experts_f32(
            d_iq2_ptrs, experts, ffn, hidden, d_act_qs, d_act_d, d_ref, 0, 256);
    });
    std::printf("  ILP=1 sequential : %8.2f us  %7.1f GB/s\n",
                iq2_base, static_cast<double>(iq2_bytes) * experts / iq2_base / 1000.0);
    for (int ilp : {2, 4, 8}) {
        ck(cudaMemset(d_got, 0, iq2_out_bytes), "memset got");
        if (!launch_iq2_s_gemv_q8_k_experts_f32_ilp(
                d_iq2_ptrs, experts, ffn, hidden, d_act_qs, d_act_d, d_got, 0, 256, ilp)) {
            std::fprintf(stderr, "ILP=%d launcher rejected the configuration\n", ilp);
            return 1;
        }
        ck(cudaStreamSynchronize(0), "sync ilp");
        std::vector<float> got(static_cast<size_t>(experts) * ffn);
        ck(cudaMemcpy(got.data(), d_got, iq2_out_bytes, cudaMemcpyDeviceToHost), "read got");
        const bool same = std::memcmp(got.data(), ref_iq2.data(), iq2_out_bytes) == 0;
        const double us = timer.measure(iters, [&] {
            launch_iq2_s_gemv_q8_k_experts_f32_ilp(
                d_iq2_ptrs, experts, ffn, hidden, d_act_qs, d_act_d, d_got, 0, 256, ilp);
        });
        std::printf("  ILP=%d %-11s : %8.2f us  %7.1f GB/s  speedup %5.2fx  bit %s\n",
                    ilp, "pipelined", us,
                    static_cast<double>(iq2_bytes) * experts / us / 1000.0,
                    iq2_base / us, same ? "ok" : "MISMATCH");
    }

    std::printf("== IQ4_NL down: %d experts x [%d rows, %d cols] (18 B / 32 values)\n",
                experts, hidden, ffn);
    const size_t iq4_bytes = static_cast<size_t>(hidden) * (ffn / 32u) * 18u;
    std::vector<uint8_t> iq4_host(iq4_bytes);
    fill(iq4_host);
    uint8_t * d_iq4 = nullptr;
    ck(cudaMalloc(&d_iq4, iq4_bytes * experts), "cudaMalloc iq4");
    std::vector<uint8_t *> iq4_ptrs(experts);
    for (int e = 0; e < experts; ++e) {
        ck(cudaMemcpy(d_iq4 + iq4_bytes * e, iq4_host.data(), iq4_bytes,
                      cudaMemcpyHostToDevice), "upload iq4");
        iq4_ptrs[e] = d_iq4 + iq4_bytes * e;
    }
    uint8_t ** d_iq4_ptrs = nullptr;
    ck(cudaMalloc(&d_iq4_ptrs, sizeof(uint8_t *) * experts), "cudaMalloc iq4 ptrs");
    ck(cudaMemcpy(d_iq4_ptrs, iq4_ptrs.data(), sizeof(uint8_t *) * experts,
                  cudaMemcpyHostToDevice), "upload iq4 ptrs");

    const size_t down_qs = static_cast<size_t>(experts) * ffn;
    const size_t down_scales = static_cast<size_t>(experts) * (ffn / 32);
    std::vector<int8_t> dq(down_qs);
    std::vector<__half> dsc(down_scales);
    for (int8_t & v : dq) v = static_cast<int8_t>(rng() & 0xffu);
    for (__half & v : dsc) {
        const uint16_t h = static_cast<uint16_t>(0x2000u | (rng() & 0x3ffu));
        std::memcpy(&v, &h, sizeof(h));
    }
    int8_t * d_dq = nullptr;
    __half * d_dsc = nullptr;
    ck(cudaMalloc(&d_dq, down_qs), "cudaMalloc dq");
    ck(cudaMalloc(&d_dsc, down_scales * sizeof(__half)), "cudaMalloc dsc");
    ck(cudaMemcpy(d_dq, dq.data(), down_qs, cudaMemcpyHostToDevice), "upload dq");
    ck(cudaMemcpy(d_dsc, dsc.data(), down_scales * sizeof(__half), cudaMemcpyHostToDevice), "upload dsc");

    const size_t iq4_out_bytes = static_cast<size_t>(experts) * hidden * sizeof(float);
    float * d_down_ref = nullptr;
    float * d_down_got = nullptr;
    ck(cudaMalloc(&d_down_ref, iq4_out_bytes), "cudaMalloc down ref");
    ck(cudaMalloc(&d_down_got, iq4_out_bytes), "cudaMalloc down got");
    ck(cudaMemset(d_down_ref, 0, iq4_out_bytes), "memset down ref");

    if (!launch_iq4_nl_gemv_q8_0_experts_f32(
            d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_ref, 0, 256)) {
        std::fprintf(stderr, "IQ4_NL DP4A kernels unavailable\n");
        return 1;
    }
    ck(cudaStreamSynchronize(0), "sync down ref");
    std::vector<float> ref_down(static_cast<size_t>(experts) * hidden);
    ck(cudaMemcpy(ref_down.data(), d_down_ref, iq4_out_bytes, cudaMemcpyDeviceToHost),
       "read down ref");

    const double iq4_base = timer.measure(iters, [&] {
        launch_iq4_nl_gemv_q8_0_experts_f32(
            d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_ref, 0, 256);
    });
    std::printf("  ILP=1 sequential : %8.2f us  %7.1f GB/s\n",
                iq4_base, static_cast<double>(iq4_bytes) * experts / iq4_base / 1000.0);
    for (int ilp : {2, 4, 8}) {
        ck(cudaMemset(d_down_got, 0, iq4_out_bytes), "memset down got");
        if (!launch_iq4_nl_gemv_q8_0_experts_f32_ilp(
                d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_got, 0, 256, ilp)) {
            std::fprintf(stderr, "ILP=%d down launcher rejected the configuration\n", ilp);
            return 1;
        }
        ck(cudaStreamSynchronize(0), "sync down ilp");
        std::vector<float> got(static_cast<size_t>(experts) * hidden);
        ck(cudaMemcpy(got.data(), d_down_got, iq4_out_bytes, cudaMemcpyDeviceToHost),
           "read down got");
        const bool same = std::memcmp(got.data(), ref_down.data(), iq4_out_bytes) == 0;
        const double us = timer.measure(iters, [&] {
            launch_iq4_nl_gemv_q8_0_experts_f32_ilp(
                d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_got, 0, 256, ilp);
        });
        std::printf("  ILP=%d %-11s : %8.2f us  %7.1f GB/s  speedup %5.2fx  bit %s\n",
                    ilp, "pipelined", us,
                    static_cast<double>(iq4_bytes) * experts / us / 1000.0,
                    iq4_base / us, same ? "ok" : "MISMATCH");
    }

    // Shared-memory codebook: the same 16 bytes the constexpr table holds, but
    // one LDS per value instead of a divergent constant/select lookup.
    for (int ilp : {4, 8}) {
        ck(cudaMemset(d_down_got, 0, iq4_out_bytes), "memset smem got");
        if (!launch_iq4_nl_gemv_q8_0_experts_f32_smem(
                d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_got, 0, 256, ilp)) {
            std::fprintf(stderr, "smem launcher rejected ilp=%d\n", ilp);
            return 1;
        }
        ck(cudaStreamSynchronize(0), "sync smem");
        std::vector<float> got(static_cast<size_t>(experts) * hidden);
        ck(cudaMemcpy(got.data(), d_down_got, iq4_out_bytes, cudaMemcpyDeviceToHost),
           "read smem got");
        const bool same = std::memcmp(got.data(), ref_down.data(), iq4_out_bytes) == 0;
        const double us = timer.measure(iters, [&] {
            launch_iq4_nl_gemv_q8_0_experts_f32_smem(
                d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_got, 0, 256, ilp);
        });
        std::printf("  ILP=%d %-11s : %8.2f us  %7.1f GB/s  speedup %5.2fx  bit %s\n",
                    ilp, "smem-LUT", us,
                    static_cast<double>(iq4_bytes) * experts / us / 1000.0,
                    iq4_base / us, same ? "ok" : "MISMATCH");
    }

    // Diagnostics: identical loads, decode replaced by the cheapest possible
    // consumer, so the two numbers bracket "access pattern" against "decode".
    {
        const int warps = 256 / 32;
        const dim3 iq4_grid((hidden + warps * 2 - 1) / (warps * 2), experts);
        const double raw_us = timer.measure(iters, [&] {
            bench_iq4_raw_kernel<<<iq4_grid, 256, 0, 0>>>(
                d_iq4_ptrs, d_dq, d_dsc, d_down_got, hidden, ffn);
        });
        ck(cudaGetLastError(), "bench_iq4_raw_kernel");
        std::printf("  IQ4_NL no-codebook (WRONG values, timing only): %8.2f us  %7.1f GB/s\n",
                    raw_us, static_cast<double>(iq4_bytes) * experts / raw_us / 1000.0);

        const dim3 iq2_grid((ffn + warps - 1) / warps, experts);
        const double flat_us = timer.measure(iters, [&] {
            bench_iq2_flat_kernel<<<iq2_grid, 256, 0, 0>>>(
                d_iq2_ptrs, d_act_qs, d_act_d, d_got, ffn, hidden);
        });
        ck(cudaGetLastError(), "bench_iq2_flat_kernel");
        std::printf("  IQ2_S no-grid (WRONG values, timing only):      %8.2f us  %7.1f GB/s\n",
                    flat_us, static_cast<double>(iq2_bytes) * experts / flat_us / 1000.0);
    }

    // Per-token cost of one layer's routed MoE with the best measured ILP, so
    // the harness prints the end-to-end number the tuning is for.
    std::printf("\n== per-token routed MoE (48 layers, 10 experts each)\n");
    const double base_layer = (iq2_base + iq4_base) / 1000.0;
    std::printf("  ILP=1 : %6.2f ms/token\n", base_layer * 48.0);
    for (int ilp : {2, 4, 8}) {
        const double us_iq2 = timer.measure(iters, [&] {
            launch_iq2_s_gemv_q8_k_experts_f32_ilp(
                d_iq2_ptrs, experts, ffn, hidden, d_act_qs, d_act_d, d_got, 0, 256, ilp);
        });
        const double us_iq4 = timer.measure(iters, [&] {
            launch_iq4_nl_gemv_q8_0_experts_f32_ilp(
                d_iq4_ptrs, experts, hidden, ffn, d_dq, d_dsc, d_down_got, 0, 256, ilp);
        });
        std::printf("  ILP=%d : %6.2f ms/token (gate/up %.2f us + down %.2f us per layer)\n",
                    ilp, (us_iq2 + us_iq4) / 1000.0 * 48.0, us_iq2, us_iq4);
    }

    cudaFree(d_iq2);
    cudaFree(d_iq2_ptrs);
    cudaFree(d_iq4);
    cudaFree(d_iq4_ptrs);
    cudaFree(d_act_qs);
    cudaFree(d_act_d);
    cudaFree(d_ref);
    cudaFree(d_got);
    cudaFree(d_down_ref);
    cudaFree(d_down_got);
    cudaFree(d_dq);
    cudaFree(d_dsc);
    return 0;
}
