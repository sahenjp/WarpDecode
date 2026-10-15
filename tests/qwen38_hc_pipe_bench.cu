// Production-kernel bit check + timing for the Task V GEMV pipeline.
//
// Unlike tests/qwen38_hc_gemv_bench.cu (which mirrors the kernels by hand),
// this file includes src/qwen38_hc.cuh and drives the real launchers, so the
// reference is the shipped scalar kernel and the candidate is whatever
// q38_hc_launch_exact_q8_0_gemv_f32() selects.
//
//   nvcc -O3 -std=c++17 -arch=sm_80 -Isrc tests/qwen38_hc_pipe_bench.cu -o /tmp/b && /tmp/b
#include "qwen38_hc.cuh"

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

struct Timer {
    cudaEvent_t a{}, b{};
    Timer() { ck(cudaEventCreate(&a), "ev"); ck(cudaEventCreate(&b), "ev"); }
    float us(int n, const std::function<void()> & f) {
        for (int i = 0; i < 5; ++i) f();
        ck(cudaEventRecord(a), "r");
        for (int i = 0; i < n; ++i) f();
        ck(cudaEventRecord(b), "r");
        ck(cudaEventSynchronize(b), "s");
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, a, b), "e");
        return ms * 1000.0f / n;
    }
};

bool same(const float * a, const float * b, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    }
    return true;
}

} // namespace

int main() {
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "prop");
    std::printf("GPU: %s sm_%d%d  dp4a=%d q8_pipe=%d\n", prop.name, prop.major, prop.minor,
                lowbit_dp4a_enabled() ? 1 : 0, q38_hc_q8_pipe_enabled() ? 1 : 0);
    std::mt19937 rng(7);
    Timer timer;

    struct Shape { int rows, cols; const char * tag; };
    const Shape shapes[] = {{320, 10240, "hc down"}, {10240, 320, "hc up"},
                            {640, 2560, "shexp gate/up"}, {2560, 640, "shexp down"},
                            {8192, 2560, "gdn qkv"}, {2560, 6144, "gdn out"},
                            {6144, 2560, "gdn z"}, {16, 2560, "gdn beta"},
                            {48, 2560, "gdn alpha"}, {4096, 2560, "qsa qkv"},
                            {2560, 4096, "qsa out"}};
    for (const Shape & s : shapes) {
        const size_t wbytes = (size_t)s.rows * (s.cols / 32) * 34u;
        std::vector<uint8_t> hw(wbytes);
        for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t)(rng() & 0xffu);
        std::vector<int8_t> ha(s.cols);
        for (auto & v : ha) v = (int8_t)((int)(rng() % 255) - 127);
        std::vector<__half> hs(s.cols / 32);
        for (auto & v : hs) v = __float2half(0.01f + 0.001f * (float)(rng() % 100));

        uint8_t * dw = nullptr; int8_t * da = nullptr; __half * ds = nullptr;
        float * o0 = nullptr; float * o1 = nullptr;
        ck(cudaMalloc(&dw, wbytes), "m"); ck(cudaMalloc(&da, s.cols), "m");
        ck(cudaMalloc(&ds, (s.cols / 32) * sizeof(__half)), "m");
        ck(cudaMalloc(&o0, s.rows * sizeof(float)), "m");
        ck(cudaMalloc(&o1, s.rows * sizeof(float)), "m");
        ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(da, ha.data(), s.cols, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(ds, hs.data(), (s.cols / 32) * sizeof(__half), cudaMemcpyHostToDevice), "c");

        const int threads = 256, warps = threads / 32;
        const int blocks = (s.rows + warps - 1) / warps;
        q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<blocks, threads>>>(dw, da, ds, o0, s.rows, s.cols);
        ck(cudaDeviceSynchronize(), "s");
        std::vector<float> ref(s.rows);
        ck(cudaMemcpy(ref.data(), o0, s.rows * sizeof(float), cudaMemcpyDeviceToHost), "c");

        const int iters = 300;
        const float us_ref = timer.us(iters, [&] {
            q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<blocks, threads>>>(dw, da, ds, o1, s.rows, s.cols);
        });
        ck(cudaMemset(o1, 0, s.rows * sizeof(float)), "m");
        q38_hc_launch_exact_q8_0_gemv_f32(dw, da, ds, o1, s.rows, s.cols, nullptr);
        ck(cudaDeviceSynchronize(), "s");
        std::vector<float> got(s.rows);
        ck(cudaMemcpy(got.data(), o1, s.rows * sizeof(float), cudaMemcpyDeviceToHost), "c");
        const bool ok = same(ref.data(), got.data(), s.rows);
        const float us_new = timer.us(iters, [&] {
            q38_hc_launch_exact_q8_0_gemv_f32(dw, da, ds, o1, s.rows, s.cols, nullptr);
        });
        std::printf("%-16s %5dx%-6d  scalar %8.2f us (%6.1f GB/s)  launcher %8.2f us (%6.1f GB/s)"
                    "  x%5.2f  %s\n",
                    s.tag, s.rows, s.cols, us_ref, wbytes / (us_ref * 1e-6) / 1e9, us_new,
                    wbytes / (us_new * 1e-6) / 1e9, us_ref / us_new,
                    ok ? "bit-ok" : "*** MISMATCH ***");
        cudaFree(dw); cudaFree(da); cudaFree(ds); cudaFree(o0); cudaFree(o1);
    }

    // F32 inject: [hc, hc * n_embd] = [4, 10240]. The reference is the shipped
    // one-warp-per-row kernel with the shipped one-block-per-row launch config.
    {
        const int rows = 4, cols = 10240;
        const size_t wb = (size_t)rows * cols * sizeof(float);
        std::vector<float> hw(rows * cols), hx(cols);
        for (auto & v : hw) v = (float)((int)(rng() % 2000) - 1000) * 0.0005f;
        for (auto & v : hx) v = (float)((int)(rng() % 2000) - 1000) * 0.0005f;
        float * dw = nullptr; float * dx = nullptr; float * o0 = nullptr; float * o1 = nullptr;
        ck(cudaMalloc(&dw, wb), "m"); ck(cudaMalloc(&dx, cols * sizeof(float)), "m");
        ck(cudaMalloc(&o0, rows * sizeof(float)), "m"); ck(cudaMalloc(&o1, rows * sizeof(float)), "m");
        ck(cudaMemcpy(dw, hw.data(), wb, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dx, hx.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "c");
        q38_hc_dense_gemv_f32_avx2_warp_kernel<<<rows, 32>>>(dw, dx, o0, rows, cols);
        ck(cudaDeviceSynchronize(), "s");
        std::vector<float> ref(rows);
        ck(cudaMemcpy(ref.data(), o0, rows * sizeof(float), cudaMemcpyDeviceToHost), "c");
        const int iters = 2000;
        const float us_ref = timer.us(iters, [&] {
            q38_hc_dense_gemv_f32_avx2_warp_kernel<<<rows, 32>>>(dw, dx, o1, rows, cols);
        });
        ck(cudaMemset(o1, 0, rows * sizeof(float)), "m");
        q38_hc_launch_dense_gemv_f32(dw, dx, o1, rows, cols, nullptr);
        ck(cudaDeviceSynchronize(), "s");
        std::vector<float> got(rows);
        ck(cudaMemcpy(got.data(), o1, rows * sizeof(float), cudaMemcpyDeviceToHost), "c");
        const bool ok = same(ref.data(), got.data(), rows);
        const float us_new = timer.us(iters, [&] {
            q38_hc_launch_dense_gemv_f32(dw, dx, o1, rows, cols, nullptr);
        });
        std::printf("%-16s %5dx%-6d  warp-u4 %8.2f us (%6.1f GB/s)  launcher %8.2f us (%6.1f GB/s)"
                    "  x%5.2f  %s\n",
                    "hc inject f32", rows, cols, us_ref, wb / (us_ref * 1e-6) / 1e9, us_new,
                    wb / (us_new * 1e-6) / 1e9, us_ref / us_new,
                    ok ? "bit-ok" : "*** MISMATCH ***");
        cudaFree(dw); cudaFree(dx); cudaFree(o0); cudaFree(o1);
    }
    return 0;
}
