#include "../src/qwen38_hc.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

static wd::TensorInfo f32_tensor(
        const char * name,
        std::vector<uint64_t> dims,
        std::vector<float> & data) {
    wd::TensorInfo t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = wd::GgmlType::F32;
    t.data = reinterpret_cast<const uint8_t *>(data.data());
    return t;
}

int main() {
    constexpr int hidden = 32;
    constexpr int hc = 4;
    constexpr int hc_dim = hidden * hc;
    constexpr int low_rank = 1;

    std::vector<float> norm_raw(hc_dim, 1.0f);
    std::vector<float> down_raw(hc_dim * low_rank, 0.0f);
    std::vector<float> up_raw(low_rank * hc_dim, 0.0f);
    std::vector<float> inject_raw(hc_dim * hc, 0.0f);

    auto norm_t = f32_tensor("norm", {hidden, hc}, norm_raw);
    auto down_t = f32_tensor("down", {hc_dim, low_rank}, down_raw);
    auto up_t = f32_tensor("up", {low_rank, hc_dim}, up_raw);
    auto inject_t = f32_tensor("inject", {hc_dim, hc}, inject_raw);

    wd::Qwen38HcWeights w;
    w.norm = wd::upload_half_array(norm_t);
    w.down = wd::upload_matrix(down_t);
    w.up = wd::upload_matrix(up_t);
    w.inject = wd::upload_matrix(inject_t);

    wd::Qwen38HcScratch scratch;
    scratch.initialize(hidden, hc, low_rank);

    std::vector<__half> hres(hc_dim, __float2half(1.0f));
    std::vector<__half> hblock(hidden, __float2half(2.0f));
    __half * dres = nullptr;
    __half * dblock = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&dres), hres.size() * sizeof(__half)), "alloc residual");
    ck(cudaMalloc(reinterpret_cast<void **>(&dblock), hblock.size() * sizeof(__half)), "alloc block");
    ck(cudaMemcpy(dres, hres.data(), hres.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy residual");
    ck(cudaMemcpy(dblock, hblock.data(), hblock.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy block");

    __half * mixed = wd::run_qwen38_hc_mix(dres, w, scratch, 1e-6f, nullptr);

    std::vector<__half> hmixed(hidden);
    ck(cudaMemcpy(hmixed.data(), mixed, hmixed.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy mixed");
    const float m0 = __half2float(hmixed[0]);
    if (std::fabs(m0 - 0.5f) > 0.01f) {
        throw std::runtime_error("HC mix mismatch: " + std::to_string(m0));
    }

    wd::run_qwen38_hc_combine(dres, dblock, scratch, nullptr);
    ck(cudaGetLastError(), "HC launch");
    ck(cudaDeviceSynchronize(), "HC sync");

    ck(cudaMemcpy(hres.data(), dres, hres.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy residual back");
    const float r0 = __half2float(hres[0]);
    if (std::fabs(r0 - 3.0f) > 0.01f) {
        throw std::runtime_error("HC combine mismatch: " + std::to_string(r0));
    }

    std::cout << "Qwen3.8 HC PASS mixed=" << m0 << " residual=" << r0 << "\n";

    cudaFree(dres);
    cudaFree(dblock);
    return 0;
}
