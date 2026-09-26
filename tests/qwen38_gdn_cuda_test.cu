#include "../src/qwen38_gdn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int main() {
    constexpr int D = 4;

    std::vector<__half> hq = {
        __float2half(1.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)
    };
    std::vector<__half> hk = hq;
    std::vector<__half> hv = {
        __float2half(2.0f), __float2half(3.0f), __float2half(4.0f), __float2half(5.0f)
    };
    std::vector<__half> hb = { __float2half(0.5f) };
    std::vector<float> hg = { 0.0f };
    std::vector<float> hs(D * D, 0.0f);

    __half * dq = nullptr;
    __half * dk = nullptr;
    __half * dv = nullptr;
    __half * db = nullptr;
    __half * dout = nullptr;
    float * dg = nullptr;
    float * ds = nullptr;

    ck(cudaMalloc(reinterpret_cast<void **>(&dq), D * sizeof(__half)), "alloc q");
    ck(cudaMalloc(reinterpret_cast<void **>(&dk), D * sizeof(__half)), "alloc k");
    ck(cudaMalloc(reinterpret_cast<void **>(&dv), D * sizeof(__half)), "alloc v");
    ck(cudaMalloc(reinterpret_cast<void **>(&db), sizeof(__half)), "alloc beta");
    ck(cudaMalloc(reinterpret_cast<void **>(&dout), D * sizeof(__half)), "alloc out");
    ck(cudaMalloc(reinterpret_cast<void **>(&dg), sizeof(float)), "alloc gate");
    ck(cudaMalloc(reinterpret_cast<void **>(&ds), D * D * sizeof(float)), "alloc state");

    ck(cudaMemcpy(dq, hq.data(), D * sizeof(__half), cudaMemcpyHostToDevice), "copy q");
    ck(cudaMemcpy(dk, hk.data(), D * sizeof(__half), cudaMemcpyHostToDevice), "copy k");
    ck(cudaMemcpy(dv, hv.data(), D * sizeof(__half), cudaMemcpyHostToDevice), "copy v");
    ck(cudaMemcpy(db, hb.data(), sizeof(__half), cudaMemcpyHostToDevice), "copy beta");
    ck(cudaMemcpy(dg, hg.data(), sizeof(float), cudaMemcpyHostToDevice), "copy gate");
    ck(cudaMemcpy(ds, hs.data(), D * D * sizeof(float), cudaMemcpyHostToDevice), "copy state");

    wd::q38_gdn_ar_kernel<<<1, D>>>(
        dq, dk, dv, dg, db, ds, dout,
        D, 1, 1);
    ck(cudaGetLastError(), "GDN AR launch");
    ck(cudaDeviceSynchronize(), "GDN AR sync");

    std::vector<__half> hout(D);
    ck(cudaMemcpy(hout.data(), dout, D * sizeof(__half), cudaMemcpyDeviceToHost), "copy out");
    ck(cudaMemcpy(hs.data(), ds, D * D * sizeof(float), cudaMemcpyDeviceToHost), "copy state back");

    const float expected[D] = {0.5f, 0.75f, 1.0f, 1.25f};
    for (int j = 0; j < D; ++j) {
        const float got = __half2float(hout[j]);
        if (std::fabs(got - expected[j]) > 0.01f) {
            throw std::runtime_error("GDN output mismatch at " + std::to_string(j) + ": " + std::to_string(got));
        }
    }

    // S = k * (beta*v)^T with k=e0.
    const float expected_state[D] = {1.0f, 1.5f, 2.0f, 2.5f};
    for (int j = 0; j < D; ++j) {
        if (std::fabs(hs[j] - expected_state[j]) > 1e-4f) {
            throw std::runtime_error("GDN state orientation mismatch");
        }
    }
    for (int i = 1; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            if (std::fabs(hs[i * D + j]) > 1e-5f) {
                throw std::runtime_error("GDN unexpected state row");
            }
        }
    }

    std::cout << "Qwen3.8 GDN AR PASS out="
              << __half2float(hout[0]) << ","
              << __half2float(hout[1]) << ","
              << __half2float(hout[2]) << ","
              << __half2float(hout[3]) << "\n";

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(dg);
    cudaFree(ds);
    return 0;
}
