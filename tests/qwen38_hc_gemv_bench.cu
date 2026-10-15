// Isolated timing + bit-equality check of the exact Q8_0 x Q8_0 GEMV and the
// F32 inject GEMV used by the HC mix / shared expert / GDN / QSA paths.
//
// The per-token profile (docs/CEILING.md) puts ~30 ms/token in these two
// kernels. Every variant below computes the reference's float order exactly and
// is checked bit for bit against the shipped kernel before it is timed.
//
//   nvcc -O3 -std=c++17 -arch=sm_80 -Isrc tests/qwen38_hc_gemv_bench.cu -o /tmp/b && /tmp/b
#include "lowbit.cuh"

#include <cuda_runtime.h>

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
    float us(int iters, const std::function<void()> & f) {
        for (int i = 0; i < 5; ++i) f();
        ck(cudaEventRecord(a), "rec");
        for (int i = 0; i < iters; ++i) f();
        ck(cudaEventRecord(b), "rec");
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        return ms * 1000.0f / iters;
    }
};

// ---------------------------------------------------------------------------
// Q8_0 x Q8_0 exact GEMV. Reference order (ggml SIMD):
//   lane c in 0..7 owns 4-value chunk c of every 34-byte block and runs
//   acc = fma(w_scale * a_scale, dot4_c, acc) over b = 0..B-1 in order,
//   then the tree ((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3)).
// ---------------------------------------------------------------------------
__global__ void k_scalar(const uint8_t * __restrict__ w, const int8_t * __restrict__ act,
                         const __half * __restrict__ asc, float * __restrict__ out,
                         int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int B = cols >> 5;
    const uint8_t * rp = w + static_cast<size_t>(row) * B * 34u;
    float acc = 0.0f;
    if (lane < 8) {
        for (int b = 0; b < B; ++b) {
            const uint8_t * blk = rp + static_cast<size_t>(b) * 34u;
            const int base = lane * 4;
            const int8_t qw0 = *(const int8_t *)(blk + 2 + base + 0);
            const int8_t qw1 = *(const int8_t *)(blk + 2 + base + 1);
            const int8_t qw2 = *(const int8_t *)(blk + 2 + base + 2);
            const int8_t qw3 = *(const int8_t *)(blk + 2 + base + 3);
            const int8_t qx0 = act[(b << 5) + base + 0];
            const int8_t qx1 = act[(b << 5) + base + 1];
            const int8_t qx2 = act[(b << 5) + base + 2];
            const int8_t qx3 = act[(b << 5) + base + 3];
            const int p0 = max(-32768, min(32767,
                (qw0 == -128 ? 128 : (qw0 < 0 ? -qw0 : qw0)) * (qw0 == 0 ? 0 : (qw0 < 0 ? (qx0 == -128 ? 127 : -qx0) : qx0)) +
                (qw1 == -128 ? 128 : (qw1 < 0 ? -qw1 : qw1)) * (qw1 == 0 ? 0 : (qw1 < 0 ? (qx1 == -128 ? 127 : -qx1) : qx1))));
            const int p1 = max(-32768, min(32767,
                (qw2 == -128 ? 128 : (qw2 < 0 ? -qw2 : qw2)) * (qw2 == 0 ? 0 : (qw2 < 0 ? (qx2 == -128 ? 127 : -qx2) : qx2)) +
                (qw3 == -128 ? 128 : (qw3 < 0 ? -qw3 : qw3)) * (qw3 == 0 ? 0 : (qw3 < 0 ? (qx3 == -128 ? 127 : -qx3) : qx3))));
            const float ws = lowbit_half(blk);
            const float as = __half2float(asc[b]);
            acc = __fmaf_rn(__fmul_rn(ws, as), static_cast<float>(p0 + p1), acc);
        }
    }
    const float a0 = __shfl_sync(~0u, acc, 0), a1 = __shfl_sync(~0u, acc, 1);
    const float a2 = __shfl_sync(~0u, acc, 2), a3 = __shfl_sync(~0u, acc, 3);
    const float a4 = __shfl_sync(~0u, acc, 4), a5 = __shfl_sync(~0u, acc, 5);
    const float a6 = __shfl_sync(~0u, acc, 6), a7 = __shfl_sync(~0u, acc, 7);
    if (lane == 0) {
        const float s0 = __fadd_rn(__fadd_rn(a4, a0), __fadd_rn(a6, a2));
        const float s1 = __fadd_rn(__fadd_rn(a5, a1), __fadd_rn(a7, a3));
        out[row] = __fadd_rn(s0, s1);
    }
}

// Shipped production kernel (dp4a, unroll 4).
__global__ void k_dp4a(const uint8_t * __restrict__ w, const int8_t * __restrict__ act,
                       const __half * __restrict__ asc, float * __restrict__ out,
                       int rows, int cols, int unroll_hint) {
    (void)unroll_hint;
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int B = cols >> 5;
    const uint8_t * rp = w + static_cast<size_t>(row) * B * 34u;
    float acc = 0.0f;
    if (lane < 8) {
        const int base = lane << 2;
#pragma unroll 4
        for (int b = 0; b < B; ++b) {
            const uint8_t * blk = rp + static_cast<size_t>(b) * 34u;
            const uint16_t * w16 = reinterpret_cast<const uint16_t *>(blk + 2 + base);
            const unsigned packed = (unsigned)w16[0] | ((unsigned)w16[1] << 16);
            const int a = *reinterpret_cast<const int *>(act + static_cast<size_t>(b) * 32u + base);
            const int dot = lowbit_dp4a(a, (int)packed, 0);
            const float ws = lowbit_half(blk);
            const float as = __half2float(asc[b]);
            acc = __fmaf_rn(__fmul_rn(ws, as), (float)dot, acc);
        }
    }
    const float a0 = __shfl_sync(~0u, acc, 0), a1 = __shfl_sync(~0u, acc, 1);
    const float a2 = __shfl_sync(~0u, acc, 2), a3 = __shfl_sync(~0u, acc, 3);
    const float a4 = __shfl_sync(~0u, acc, 4), a5 = __shfl_sync(~0u, acc, 5);
    const float a6 = __shfl_sync(~0u, acc, 6), a7 = __shfl_sync(~0u, acc, 7);
    if (lane == 0) {
        const float s0 = __fadd_rn(__fadd_rn(a4, a0), __fadd_rn(a6, a2));
        const float s1 = __fadd_rn(__fadd_rn(a5, a1), __fadd_rn(a7, a3));
        out[row] = __fadd_rn(s0, s1);
    }
}

// Manual 8-deep software pipeline: issue the loads for step k+PIPE before the
// FMAs of step k. Same arithmetic, same order, more loads in flight.
template <int PIPE>
__global__ void k_pipe(const uint8_t * __restrict__ w, const int8_t * __restrict__ act,
                       const __half * __restrict__ asc, float * __restrict__ out,
                       int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int B = cols >> 5;
    const uint8_t * rp = w + static_cast<size_t>(row) * B * 34u;
    float acc = 0.0f;
    if (lane < 8) {
        const int base = lane << 2;
        unsigned pk[PIPE];
        int    aq[PIPE];
        float  sc[PIPE];
#pragma unroll
        for (int k = 0; k < PIPE; ++k) {
            const uint8_t * blk = rp + static_cast<size_t>(k) * 34u;
            const uint16_t * w16 = reinterpret_cast<const uint16_t *>(blk + 2 + base);
            pk[k] = (unsigned)w16[0] | ((unsigned)w16[1] << 16);
            aq[k] = *reinterpret_cast<const int *>(act + static_cast<size_t>(k) * 32u + base);
            sc[k] = __fmul_rn(lowbit_half(blk), __half2float(asc[k]));
        }
        int b = 0;
        for (; b + PIPE <= B; b += PIPE) {
            unsigned npk[PIPE];
            int naq[PIPE];
            float nsc[PIPE];
            if (b + 2 * PIPE <= B) {
#pragma unroll
                for (int k = 0; k < PIPE; ++k) {
                    const uint8_t * blk = rp + static_cast<size_t>(b + PIPE + k) * 34u;
                    const uint16_t * w16 = reinterpret_cast<const uint16_t *>(blk + 2 + base);
                    npk[k] = (unsigned)w16[0] | ((unsigned)w16[1] << 16);
                    naq[k] = *reinterpret_cast<const int *>(act + static_cast<size_t>(b + PIPE + k) * 32u + base);
                    nsc[k] = __fmul_rn(lowbit_half(blk), __half2float(asc[b + PIPE + k]));
                }
            }
#pragma unroll
            for (int k = 0; k < PIPE; ++k) {
                acc = __fmaf_rn(sc[k], (float)lowbit_dp4a(aq[k], (int)pk[k], 0), acc);
            }
            if (b + 2 * PIPE <= B) {
#pragma unroll
                for (int k = 0; k < PIPE; ++k) { pk[k] = npk[k]; aq[k] = naq[k]; sc[k] = nsc[k]; }
            }
        }
        for (; b < B; ++b) {
            const uint8_t * blk = rp + static_cast<size_t>(b) * 34u;
            const uint16_t * w16 = reinterpret_cast<const uint16_t *>(blk + 2 + base);
            const unsigned packed = (unsigned)w16[0] | ((unsigned)w16[1] << 16);
            const int a = *reinterpret_cast<const int *>(act + static_cast<size_t>(b) * 32u + base);
            acc = __fmaf_rn(__fmul_rn(lowbit_half(blk), __half2float(asc[b])),
                            (float)lowbit_dp4a(a, (int)packed, 0), acc);
        }
    }
    const float a0 = __shfl_sync(~0u, acc, 0), a1 = __shfl_sync(~0u, acc, 1);
    const float a2 = __shfl_sync(~0u, acc, 2), a3 = __shfl_sync(~0u, acc, 3);
    const float a4 = __shfl_sync(~0u, acc, 4), a5 = __shfl_sync(~0u, acc, 5);
    const float a6 = __shfl_sync(~0u, acc, 6), a7 = __shfl_sync(~0u, acc, 7);
    if (lane == 0) {
        const float s0 = __fadd_rn(__fadd_rn(a4, a0), __fadd_rn(a6, a2));
        const float s1 = __fadd_rn(__fadd_rn(a5, a1), __fadd_rn(a7, a3));
        out[row] = __fadd_rn(s0, s1);
    }
}

// ---------------------------------------------------------------------------
// F32 dense GEMV (ggml_vec_dot_f32 AVX2 order), rows = hyper-connection count.
// ---------------------------------------------------------------------------
__global__ void f_warp(const float * __restrict__ w, const float * __restrict__ x,
                       float * __restrict__ y, int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int j = lane >> 3, l = lane & 7;
    const float * rp = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
#pragma unroll 4
    for (int i = 0; i < cols; i += 32) {
        const int c = i + j * 8 + l;
        acc = __fmaf_rn(rp[c], x[c], acc);
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(~0u, acc, 16));
    const float s1 = __shfl_sync(~0u, acc, (lane & 7) + 8);
    if (lane < 8) acc = __fadd_rn(acc, s1);
    const float s4 = __shfl_sync(~0u, acc, (lane & 3) + 4);
    if (lane < 4) acc = __fadd_rn(acc, s4);
    const float t1 = __shfl_sync(~0u, acc, 1);
    const float t2 = __shfl_sync(~0u, acc, 2);
    const float t3 = __shfl_sync(~0u, acc, 3);
    if (lane == 0) {
        const float h0 = __fadd_rn(acc, t1), h1 = __fadd_rn(t2, t3);
        y[row] = __fadd_rn(h0, h1);
    }
}

template <int PIPE>
__global__ void f_pipe(const float * __restrict__ w, const float * __restrict__ x,
                       float * __restrict__ y, int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int j = lane >> 3, l = lane & 7;
    const float * rp = w + static_cast<size_t>(row) * cols;
    const int step = 32;
    const int n = cols / step;
    float acc = 0.0f;
    float v[PIPE], wv[PIPE];
#pragma unroll
    for (int k = 0; k < PIPE; ++k) {
        const int c = k * step + j * 8 + l;
        v[k] = x[c];
        wv[k] = rp[c];
    }
    int i = 0;
    for (; i + PIPE <= n; i += PIPE) {
        float nv[PIPE], nw[PIPE];
        if (i + 2 * PIPE <= n) {
#pragma unroll
            for (int k = 0; k < PIPE; ++k) {
                const int c = (i + PIPE + k) * step + j * 8 + l;
                nv[k] = x[c];
                nw[k] = rp[c];
            }
        }
#pragma unroll
        for (int k = 0; k < PIPE; ++k) acc = __fmaf_rn(wv[k], v[k], acc);
        if (i + 2 * PIPE <= n) {
#pragma unroll
            for (int k = 0; k < PIPE; ++k) { v[k] = nv[k]; wv[k] = nw[k]; }
        }
    }
    for (; i < n; ++i) {
        const int c = i * step + j * 8 + l;
        acc = __fmaf_rn(rp[c], x[c], acc);
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(~0u, acc, 16));
    const float s1 = __shfl_sync(~0u, acc, (lane & 7) + 8);
    if (lane < 8) acc = __fadd_rn(acc, s1);
    const float s4 = __shfl_sync(~0u, acc, (lane & 3) + 4);
    if (lane < 4) acc = __fadd_rn(acc, s4);
    const float t1 = __shfl_sync(~0u, acc, 1);
    const float t2 = __shfl_sync(~0u, acc, 2);
    const float t3 = __shfl_sync(~0u, acc, 3);
    if (lane == 0) {
        const float h0 = __fadd_rn(acc, t1), h1 = __fadd_rn(t2, t3);
        y[row] = __fadd_rn(h0, h1);
    }
}

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
    std::printf("GPU: %s sm_%d%d\n", prop.name, prop.major, prop.minor);

    std::mt19937 rng(1234);
    Timer timer;

    struct Shape { int rows, cols; const char * tag; };
    const Shape shapes[] = {{320, 10240, "hc down"}, {640, 2560, "shexp gate/up"},
                            {16, 2560, "gdn beta"}, {48, 2560, "gdn alpha"}};

    for (const Shape & s : shapes) {
        const size_t wbytes = (size_t)s.rows * (s.cols / 32) * 34u;
        std::vector<uint8_t> hw(wbytes);
        for (auto & b : hw) b = (uint8_t)(rng() & 0xffu);
        for (size_t off = 0; off + 1 < wbytes; off += 34u) {
            hw[off + 1] = (uint8_t)(0x30 | (rng() & 3u));
            hw[off + 0] = (uint8_t)(rng() & 0xffu);
        }
        std::vector<int8_t> ha(s.cols);
        for (auto & v : ha) v = (int8_t)((int)(rng() % 255) - 127);
        std::vector<__half> hs(s.cols / 32);
        for (auto & v : hs) v = __float2half(0.01f + 0.001f * (float)(rng() % 100));

        uint8_t * dw = nullptr; int8_t * da = nullptr; __half * ds = nullptr;
        float * out0 = nullptr; float * out1 = nullptr;
        ck(cudaMalloc(&dw, wbytes), "malloc");
        ck(cudaMalloc(&da, s.cols), "malloc");
        ck(cudaMalloc(&ds, (s.cols / 32) * sizeof(__half)), "malloc");
        ck(cudaMalloc(&out0, s.rows * sizeof(float)), "malloc");
        ck(cudaMalloc(&out1, s.rows * sizeof(float)), "malloc");
        ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "cpy");
        ck(cudaMemcpy(da, ha.data(), s.cols, cudaMemcpyHostToDevice), "cpy");
        ck(cudaMemcpy(ds, hs.data(), (s.cols / 32) * sizeof(__half), cudaMemcpyHostToDevice), "cpy");

        const int threads = 256, warps = threads / 32;
        const int blocks = (s.rows + warps - 1) / warps;
        const size_t bytes = wbytes;

        auto run = [&](void (*k)(const uint8_t *, const int8_t *, const __half *, float *, int, int),
                       float * o) { k<<<blocks, threads>>>(dw, da, ds, o, s.rows, s.cols); };

        run(k_scalar, out0);
        ck(cudaDeviceSynchronize(), "sync");
        std::vector<float> ref(s.rows);
        ck(cudaMemcpy(ref.data(), out0, s.rows * sizeof(float), cudaMemcpyDeviceToHost), "cpy");

        const int iters = 300;
        const float us0 = timer.us(iters, [&] { run(k_scalar, out1); });

        struct Variant { const char * name; float us; bool ok; };
        std::vector<Variant> vs;
        auto bench = [&](const char * name, const std::function<void()> & f) {
            ck(cudaMemset(out1, 0, s.rows * sizeof(float)), "memset");
            f();
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> got(s.rows);
            ck(cudaMemcpy(got.data(), out1, s.rows * sizeof(float), cudaMemcpyDeviceToHost), "cpy");
            const bool ok = same(ref.data(), got.data(), s.rows);
            vs.push_back({name, timer.us(iters, f), ok});
        };
        bench("dp4a u4 (shipped)", [&] { k_dp4a<<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols, 0); });
        bench("pipe 4", [&] { k_pipe<4><<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols); });
        bench("pipe 8", [&] { k_pipe<8><<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols); });
        bench("pipe 16", [&] { k_pipe<16><<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols); });
        bench("pipe 24", [&] { k_pipe<24><<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols); });
        bench("pipe 32", [&] { k_pipe<32><<<blocks, threads>>>(dw, da, ds, out1, s.rows, s.cols); });

        std::printf("\n%-16s %5dx%-6d  %.1f KB  ref(scalar) %8.1f us  %7.1f GB/s\n",
                    s.tag, s.rows, s.cols, bytes / 1024.0, us0,
                    bytes / (us0 * 1e-6) / 1e9);
        for (const Variant & v : vs) {
            std::printf("    %-20s %8.1f us  %7.1f GB/s  x%6.2f  %s\n", v.name, v.us,
                        bytes / (v.us * 1e-6) / 1e9, us0 / v.us, v.ok ? "bit-ok" : "*** MISMATCH ***");
        }
        cudaFree(dw); cudaFree(da); cudaFree(ds); cudaFree(out0); cudaFree(out1);
    }

    // F32 inject GEMV: rows = hc = 4.
    {
        const int rows = 4, cols = 10240;
        const size_t wbytes = (size_t)rows * cols * sizeof(float);
        std::vector<float> hw(rows * cols);
        for (auto & v : hw) v = (float)((int)(rng() % 2000) - 1000) * 0.0005f;
        std::vector<float> hx(cols);
        for (auto & v : hx) v = (float)((int)(rng() % 2000) - 1000) * 0.0005f;
        float * dw = nullptr; float * dx = nullptr; float * o0 = nullptr; float * o1 = nullptr;
        ck(cudaMalloc(&dw, wbytes), "m"); ck(cudaMalloc(&dx, cols * sizeof(float)), "m");
        ck(cudaMalloc(&o0, rows * sizeof(float)), "m"); ck(cudaMalloc(&o1, rows * sizeof(float)), "m");
        ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dx, hx.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "c");
        const int threads = 256, warps = threads / 32, blocks = (rows + warps - 1) / warps;
        f_warp<<<blocks, threads>>>(dw, dx, o0, rows, cols);
        ck(cudaDeviceSynchronize(), "s");
        std::vector<float> ref(rows);
        ck(cudaMemcpy(ref.data(), o0, rows * sizeof(float), cudaMemcpyDeviceToHost), "c");
        const int iters = 2000;
        const float us0 = timer.us(iters, [&] { f_warp<<<blocks, threads>>>(dw, dx, o1, rows, cols); });
        std::printf("\n%-16s %5dx%-6d  F32  ref(warp u4) %8.1f us  %7.1f GB/s\n", "hc inject", rows, cols,
                    us0, wbytes / (us0 * 1e-6) / 1e9);
        auto b2 = [&](const char * n, const std::function<void()> & f) {
            ck(cudaMemset(o1, 0, rows * sizeof(float)), "m");
            f(); ck(cudaDeviceSynchronize(), "s");
            std::vector<float> got(rows);
            ck(cudaMemcpy(got.data(), o1, rows * sizeof(float), cudaMemcpyDeviceToHost), "c");
            std::printf("    %-20s %8.1f us  %7.1f GB/s  x%6.2f  %s\n", n, timer.us(iters, f),
                        wbytes / (timer.us(iters, f) * 1e-6) / 1e9, us0 / timer.us(iters, f),
                        same(ref.data(), got.data(), rows) ? "bit-ok" : "*** MISMATCH ***");
        };
        b2("pipe 8", [&] { f_pipe<8><<<blocks, threads>>>(dw, dx, o1, rows, cols); });
        b2("pipe 16", [&] { f_pipe<16><<<blocks, threads>>>(dw, dx, o1, rows, cols); });
        b2("pipe 32", [&] { f_pipe<32><<<blocks, threads>>>(dw, dx, o1, rows, cols); });
    }
    return 0;
}
