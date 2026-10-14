// Task J: standalone HC mix micro-benchmark.
//
// Task F's stage profile put hc_mix_attn + hc_mix_ffn at ~40 ms each on A100
// (80 ms = 30% of a 272 ms decode token) while the same work reads only a few
// MB per site. This benchmark answers two questions that a whole-step profile
// cannot:
//
//  1. how long does a single HC mix take when it is the only work on the
//     device ("sync" mode: one launch, cudaStreamSynchronize, next),
//  2. how long does it take when thousands of kernels are queued back to back
//     ("batch" mode, which is what the decode loop actually does, and which is
//     what makes a per-stage CUDA event interval grow),
//
// for the three mix shapes the engine can take: the exact Q8_0 replay
// (default), the legacy F16 mirror, and the F32 debug replay.
//
// Usage:
//   nvcc -arch=sm_89 -O3 -I src -o hc_bench tests/qwen38_hc_bench.cu
//   ./hc_bench --hidden 10240 --hc 4 --low-rank 320 --iterations 300 --json out.json
#include "qwen38_hc.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace wd;

namespace {

void check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

double host_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Plain row-major Q8_0 packer (host): 34 bytes per 32 values, half scale, int8
// quants. The benchmark only needs the same memory layout, not the same bits.
std::vector<uint8_t> pack_q8_0(const std::vector<float> & w, int rows, int cols) {
    const int blocks = cols / 32;
    std::vector<uint8_t> out(static_cast<size_t>(rows) * blocks * 34u, 0);
    uint8_t * p = out.data();
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const float * src = w.data() + static_cast<size_t>(r) * cols + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(src[i]));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            __half hd = __float2half(d);
            std::memcpy(p, &hd, 2);
            for (int i = 0; i < 32; ++i) {
                float q = src[i] * id;
                q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
                p[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::lround(q)));
            }
            p += 34;
        }
    }
    return out;
}

float * upload_float_values(const std::vector<float> & values) {
    float * d = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&d), values.size() * sizeof(float)), "cudaMalloc f32");
    check(cudaMemcpy(d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy f32");
    return d;
}

// Upload a freshly packed Q8_0 blob (or 0 when the host vector is empty).
LowBitMatrix upload_lowbit_matrix_bytes(const std::vector<uint8_t> & bytes, int rows, int cols) {
    if (bytes.empty()) return LowBitMatrix();
    uint8_t * d = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&d), bytes.size()), "cudaMalloc upload");
    check(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy upload");
    return borrow_lowbit_matrix(d, rows, cols, GgmlType::Q8_0, bytes.size());
}

uint8_t * upload(const std::vector<uint8_t> & bytes) {
    uint8_t * d = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&d), bytes.size()), "cudaMalloc weight");
    check(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy weight");
    return d;
}

__half * upload_half(const std::vector<float> & values) {
    std::vector<__half> tmp(values.size());
    for (size_t i = 0; i < values.size(); ++i) tmp[i] = __float2half(values[i]);
    __half * d = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&d), tmp.size() * sizeof(__half)), "cudaMalloc half");
    check(cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "cudaMemcpy half");
    return d;
}

GpuMatrix make_dense_half(int rows, int cols, float value) {
    std::vector<float> values(static_cast<size_t>(rows) * cols, value);
    GpuMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.type = GgmlType::F16;
    m.lowbit = false;
    m.dense.data = upload_half(values);
    m.dense.size = values.size();
    return m;
}

GpuMatrix make_q8_0(int rows, int cols, const std::vector<float> & values, uint8_t * data) {
    GpuMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.type = GgmlType::Q8_0;
    m.lowbit = true;
    m.low = borrow_lowbit_matrix(data, rows, cols, GgmlType::Q8_0,
                                 static_cast<size_t>(rows) * (cols / 32) * 34u);
    return m;
}

struct CaseResult {
    std::string name;
    int iterations = 0;
    double total_ms = 0.0;
    double us_per_call = 0.0;
};

double run_sync(
        const std::function<void()> & launch,
        int iterations,
        cudaStream_t stream,
        cudaEvent_t start,
        cudaEvent_t stop) {
    const double t0 = host_now_ms();
    check(cudaEventRecord(start, stream), "event record start");
    for (int i = 0; i < iterations; ++i) {
        launch();
        check(cudaStreamSynchronize(stream), "sync");
    }
    check(cudaEventRecord(stop, stream), "event record stop");
    check(cudaStreamSynchronize(stream), "sync after stop");
    (void)t0;
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    return ms;
}

double run_batch(
        const std::function<void()> & launch,
        int iterations,
        cudaStream_t stream,
        cudaEvent_t start,
        cudaEvent_t stop) {
    check(cudaEventRecord(start, stream), "event record start");
    for (int i = 0; i < iterations; ++i) launch();
    check(cudaEventRecord(stop, stream), "event record stop");
    check(cudaStreamSynchronize(stream), "sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    return ms;
}

} // namespace

int main(int argc, char ** argv) {
    int hidden = 10240;
    int hc = 4;
    int low_rank = 320;
    int iterations = 300;
    int warmup = 20;
    int device = 0;
    bool skip_legacy = false;
    bool stages = false;
    double rms_eps = 1e-6;
    std::string json_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if (a == "--hidden") hidden = next();
        else if (a == "--hc") hc = next();
        else if (a == "--low-rank") low_rank = next();
        else if (a == "--iterations") iterations = next();
        else if (a == "--warmup") warmup = next();
        else if (a == "--device") device = next();
        else if (a == "--skip-legacy") skip_legacy = true;
        else if (a == "--stages") stages = true;
        else if (a == "--json") json_path = argv[++i];
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    check(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "device props");

    const int hc_dim = hidden * hc;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> down_values(static_cast<size_t>(low_rank) * hc_dim);
    std::vector<float> up_values(static_cast<size_t>(hc_dim) * low_rank);
    for (float & v : down_values) v = nd(rng) * 0.02f;
    for (float & v : up_values) v = nd(rng) * 0.02f;
    std::vector<float> norm_values(static_cast<size_t>(hc_dim));
    for (float & v : norm_values) v = 1.0f + ud(rng) * 0.1f;
    std::vector<float> inject_values(static_cast<size_t>(hc) * hc_dim);
    for (float & v : inject_values) v = nd(rng) * 0.05f;

    uint8_t * down_data = upload(pack_q8_0(down_values, low_rank, hc_dim));
    uint8_t * up_data = upload(pack_q8_0(up_values, hc_dim, low_rank));

    Qwen38HcWeights w;
    w.down = make_q8_0(low_rank, hc_dim, down_values, down_data);
    w.up = make_q8_0(hc_dim, low_rank, up_values, up_data);
    w.norm = DeviceHalfArray();  // unused: the exact path reads debug_norm_f32
    {
        // The F16 mirror needs a half norm; seed it with the same values.
        __half * n = upload_half(norm_values);
        w.norm.data = n;
        w.norm.size = static_cast<size_t>(hc_dim);
    }
    w.debug_norm_f32 = norm_values;
    w.debug_inject_f32 = inject_values;
    w.has_inject = true;
    w.has_inject_f32 = true;
    w.inject = make_dense_half(hc, hc_dim, 0.1f);
    // qwen38_hc_f32_inject_ready() gates the with_inject entry point on the
    // unrounded F32 copy, exactly as load_qwen38_hc_weights() leaves it.
    w.inject_f32.data = upload_float_values(inject_values);
    w.inject_f32.size = inject_values.size();

    Qwen38HcScratch scratch;
    scratch.initialize(hidden, hc, low_rank);
    Qwen38HcF32DebugScratch f32;
    f32.initialize(hidden, hc, low_rank);

    float * residual = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&residual), static_cast<size_t>(hc_dim) * sizeof(float)),
          "cudaMalloc residual");
    {
        std::vector<float> host_residual(static_cast<size_t>(hc_dim));
        for (float & v : host_residual) v = nd(rng);
        check(cudaMemcpy(residual, host_residual.data(), host_residual.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "cudaMemcpy residual");
    }

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "stream create");
    (void)skip_legacy;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");

    std::printf("# geometry: hidden=%d hc=%d hc_dim=%d low_rank=%d down=%dx%d up=%dx%d %s%s\n",
                hidden, hc, hc_dim, low_rank, w.down.rows, w.down.cols, w.up.rows, w.up.cols,
                w.down.lowbit ? "down=lowbit" : "down=dense",
                w.up.lowbit ? " up=lowbit" : " up=dense");
    std::fflush(stdout);
    // ---- stage isolation: one launch + sync each, so an illegal access names
    // the kernel that produced it instead of poisoning the next case.
    if (stages) {
        struct Stage { const char * name; std::function<void()> run; };
        const int down_blocks = hc_dim / 32;
        const int low_blocks = low_rank / 32;
        std::vector<Stage> list = {
            {"gamma_h2d", [&] {
                check(cudaMemcpyAsync(f32.gamma(), w.debug_norm_f32.data(),
                                      static_cast<size_t>(hc_dim) * sizeof(float),
                                      cudaMemcpyHostToDevice, stream), "stage gamma");
            }},
            {"norm", [&] {
                q38_hc_group_rmsnorm_f32_kernel<<<hc, 256, 0, stream>>>(
                    residual, f32.gamma(), f32.norm(), hidden, hc, static_cast<float>(rms_eps));
            }},
            {"quant_norm", [&] {
                q38_hc_q8_0_quantize_f32_exact_kernel<<<(down_blocks + warps - 1) / warps, 256, 0, stream>>>(
                    f32.norm(), hc_dim, f32.q8_0_values(), f32.q8_0_scales());
            }},
            {"down_gemv", [&] {
                q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(low_rank + warps - 1) / warps, 256, 0, stream>>>(
                    w.down.low.data, f32.q8_0_values(), f32.q8_0_scales(), f32.low_raw(), low_rank, hc_dim);
            }},
            {"down_copy", [&] {
                check(cudaMemcpyAsync(f32.low(), f32.low_raw(), static_cast<size_t>(low_rank) * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream), "stage down copy");
            }},
            {"silu", [&] {
                q38_hc_silu_scale_ggml_f32_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
                    f32.low(), low_rank, 1.0f / static_cast<float>(hc));
            }},
            {"quant_low", [&] {
                q38_hc_q8_0_quantize_f32_exact_kernel<<<(low_blocks + warps - 1) / warps, 256, 0, stream>>>(
                    f32.low(), low_rank, f32.q8_0_values(), f32.q8_0_scales());
            }},
            {"up_gemv", [&] {
                q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(hc_dim + warps - 1) / warps, 256, 0, stream>>>(
                    w.up.low.data, f32.q8_0_values(), f32.q8_0_scales(), f32.gate(), hc_dim, low_rank);
            }},
            {"mix", [&] {
                q38_hc_mix_f32_cpu_order_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
                    f32.norm(), f32.gate(), f32.mixed(), hidden, hc);
            }},
            {"inject_gemv", [&] {
                q38_hc_launch_dense_gemv_f32(w.debug_inject_f32.data(), f32.norm(), f32.inject(), hc, hc_dim, stream);
            }},
            {"out_copy", [&] {
                q38_hc_copy_f32_to_half_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
                    f32.mixed(), scratch.mixed(), hidden);
            }},
            {"inject_copy", [&] {
                q38_hc_copy_f32_to_half_kernel<<<(hc + 255) / 256, 256, 0, stream>>>(
                    f32.inject(), scratch.inject(), hc);
            }},
            {"f16_norm", [&] {
                q38_hc_group_rmsnorm_f32in_kernel<<<hc, 256, 0, stream>>>(
                    residual, w.norm.data, scratch.xn(), hidden, hc, static_cast<float>(rms_eps));
            }},
            {"f16_down", [&] {
                launch_matrix_gemv_half(w.down, scratch.xn(), scratch.lo(), stream);
            }},
            {"f16_silu", [&] {
                q38_hc_silu_scale_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
                    scratch.lo(), low_rank, 1.0f / static_cast<float>(hc));
            }},
            {"f16_up", [&] {
                launch_matrix_gemv_half(w.up, scratch.lo(), scratch.gate(), stream);
            }},
            {"f16_mix", [&] {
                q38_hc_mix_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
                    scratch.xn(), scratch.gate(), scratch.mixed(), hidden, hc);
            }},
            {"f16_inject", [&] {
                launch_matrix_gemv_half(w.inject, scratch.xn(), scratch.inject(), stream);
            }},
        };
        for (const Stage & s : list) {
            (void)cudaGetLastError();
            s.run();
            const cudaError_t err = cudaStreamSynchronize(stream);
            std::printf("%-16s %s\n", s.name,
                        err == cudaSuccess ? "OK" : cudaGetErrorString(err));
            std::fflush(stdout);
            if (err != cudaSuccess) return 1;
        }
        return 0;
    }

    std::vector<CaseResult> results;
    auto measure = [&](const std::string & name, const std::function<void()> & launch) {
        try {
            for (int i = 0; i < warmup; ++i) launch();
            check(cudaDeviceSynchronize(), "warmup sync");
        } catch (const std::exception & ex) {
            std::printf("%-34s SKIPPED: %s\n", name.c_str(), ex.what());
            std::fflush(stdout);
            (void)cudaGetLastError();
            return;
        }
        double sync_ms = 0.0;
        double batch_ms = 0.0;
        try {
            sync_ms = run_sync(launch, iterations, stream, start, stop);
            batch_ms = run_batch(launch, iterations, stream, start, stop);
        } catch (const std::exception & ex) {
            std::printf("%-34s FAILED: %s\n", name.c_str(), ex.what());
            std::fflush(stdout);
            (void)cudaGetLastError();
            return;
        }
        CaseResult r;
        r.name = name + " [sync]";
        r.iterations = iterations;
        r.total_ms = sync_ms;
        r.us_per_call = sync_ms * 1000.0 / iterations;
        results.push_back(r);
        r.name = name + " [batch]";
        r.total_ms = batch_ms;
        r.us_per_call = batch_ms * 1000.0 / iterations;
        results.push_back(r);
    };

    measure("q8_0_mix_with_inject", [&] {
        run_qwen38_hc_q8_0_f32_mix_with_inject_from_residual(
            residual, w, f32, static_cast<float>(rms_eps), stream);
    });
    // Same exact replay with the F32 inject projection disabled: the inject
    // weight is 160 KB, so this isolates its share.
    {
        std::vector<float> saved;
        saved.swap(w.debug_inject_f32);
        measure("q8_0_mix_no_inject", [&] {
            run_qwen38_hc_q8_0_f32_mix_from_residual(
                residual, w, f32, static_cast<float>(rms_eps), stream);
        });
        saved.swap(w.debug_inject_f32);
    }
    if (!skip_legacy) {
        measure("legacy_f16_mix", [&] {
            run_qwen38_hc_mix(residual, w, scratch, static_cast<float>(rms_eps), stream);
        });
        measure("q8_0_mix_debug_f32_variant", [&] {
            run_qwen38_hc_f32_mix_from_residual(residual, w, f32, static_cast<float>(rms_eps), stream);
        });
        // Same exact replay with the pageable 40 KB gamma H2D removed from the
        // composition (the weight is resident on the device, so the copy is the
        // only thing this isolates).
        {
            std::vector<float> saved;
            saved.swap(w.debug_norm_f32);
            measure("q8_0_mix_no_gamma_copy", [&] {
                run_qwen38_hc_q8_0_f32_mix_from_residual(
                    residual, w, f32, static_cast<float>(rms_eps), stream);
            });
            saved.swap(w.debug_norm_f32);
        }
        measure("weights.upload_q8_0_down", [&] {
            LowBitMatrix m = upload_lowbit_matrix_bytes(pack_q8_0(down_values, low_rank, hc_dim),
                                                        low_rank, hc_dim);
            (void)m;
        });
        measure("weights.upload_q8_0_up", [&] {
            LowBitMatrix m = upload_lowbit_matrix_bytes(pack_q8_0(up_values, hc_dim, low_rank),
                                                        hc_dim, low_rank);
            (void)m;
        });
    }
    if (!skip_legacy)
    measure("weights.upload_gamma_pageable", [&] {
        void * dst = nullptr;
        check(cudaMalloc(&dst, static_cast<size_t>(hc_dim) * sizeof(float)), "cudaMalloc gamma");
        check(cudaMemcpyAsync(dst, norm_values.data(), static_cast<size_t>(hc_dim) * sizeof(float),
                              cudaMemcpyHostToDevice, stream), "gamma memcpy");
        check(cudaFree(dst), "cudaFree gamma");
    });

    // Individual kernels of the exact path, so the sum can be compared with the
    // sequence total above.
    {
        q38_hc_group_rmsnorm_f32_kernel<<<hc, 256, 0, stream>>>(
            residual, f32.gamma(), f32.norm(), hidden, hc, static_cast<float>(rms_eps));
        const int down_blocks = hc_dim / 32;
        q38_hc_q8_0_quantize_f32_exact_kernel<<<(down_blocks + 7) / 8, 256, 0, stream>>>(
            f32.norm(), hc_dim, f32.q8_0_values(), f32.q8_0_scales());
        check(cudaStreamSynchronize(stream), "kernel warm");
    }
    auto kernel_case = [&](const std::string & name, const std::function<void()> & launch) {
        measure(name, launch);
    };
    const int warps = 8;
    kernel_case("kernel.q8_norm", [&] {
        q38_hc_group_rmsnorm_f32_kernel<<<hc, 256, 0, stream>>>(
            residual, f32.gamma(), f32.norm(), hidden, hc, static_cast<float>(rms_eps));
    });
    kernel_case("kernel.q8_quant_norm", [&] {
        q38_hc_q8_0_quantize_f32_exact_kernel<<<((hc_dim / 32) + warps - 1) / warps, 256, 0, stream>>>(
            f32.norm(), hc_dim, f32.q8_0_values(), f32.q8_0_scales());
    });
    kernel_case("kernel.q8_down", [&] {
        q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<(low_rank + warps - 1) / warps, 256, 0, stream>>>(
            w.down.low.data, f32.q8_0_values(), f32.q8_0_scales(), f32.low_raw(), low_rank, hc_dim);
    });
    kernel_case("kernel.q8_quant_low", [&] {
        q38_hc_q8_0_quantize_f32_exact_kernel<<<((low_rank / 32) + warps - 1) / warps, 256, 0, stream>>>(
            f32.low(), low_rank, f32.q8_0_values(), f32.q8_0_scales());
    });
    kernel_case("kernel.q8_up", [&] {
        q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<((hc_dim + warps - 1) / warps), 256, 0, stream>>>(
            w.up.low.data, f32.q8_0_values(), f32.q8_0_scales(), f32.gate(), hc_dim, low_rank);
    });
    kernel_case("kernel.q8_mix", [&] {
        q38_hc_mix_f32_cpu_order_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            f32.norm(), f32.gate(), f32.mixed(), hidden, hc);
    });
    kernel_case("kernel.q8_inject", [&] {
        q38_hc_launch_dense_gemv_f32(w.debug_inject_f32.data(), f32.norm(), f32.inject(), hc, hc_dim, stream);
    });
    kernel_case("kernel.silu", [&] {
        q38_hc_silu_scale_ggml_f32_kernel<<<(low_rank + 255) / 256, 256, 0, stream>>>(
            f32.low(), low_rank, 1.0f / static_cast<float>(hc));
    });
    kernel_case("kernel.down_copy_d2d", [&] {
        check(cudaMemcpyAsync(f32.low(), f32.low_raw(), static_cast<size_t>(low_rank) * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream), "d2d");
    });
    kernel_case("kernel.gamma_h2d_pageable", [&] {
        check(cudaMemcpyAsync(f32.gamma(), w.debug_norm_f32.data(),
                              static_cast<size_t>(hc_dim) * sizeof(float),
                              cudaMemcpyHostToDevice, stream), "h2d");
    });

    std::printf("# device=%s sm=%d.%d hidden=%d hc=%d low_rank=%d iterations=%d\n",
                prop.name, prop.major, prop.minor, hidden, hc, low_rank, iterations);
    for (const CaseResult & r : results) {
        std::printf("%-34s total=%9.3f ms  per_call=%9.3f us\n",
                    r.name.c_str(), r.total_ms, r.us_per_call);
    }

    if (!json_path.empty()) {
        FILE * fh = std::fopen(json_path.c_str(), "w");
        if (fh != nullptr) {
            std::fprintf(fh, "{\"device\":\"%s\",\"sm\":%d,\"hidden\":%d,\"hc\":%d,"
                             "\"low_rank\":%d,\"iterations\":%d,\"cases\":[",
                         prop.name, prop.major, hidden, hc, low_rank, iterations);
            for (size_t i = 0; i < results.size(); ++i) {
                std::fprintf(fh, "%s{\"name\":\"%s\",\"us_per_call\":%.3f,\"total_ms\":%.3f}",
                             i ? "," : "", results[i].name.c_str(),
                             results[i].us_per_call, results[i].total_ms);
            }
            std::fprintf(fh, "]}\n");
            std::fclose(fh);
        }
    }

    cudaFree(down_data);
    cudaFree(up_data);
    cudaFree(residual);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
}
