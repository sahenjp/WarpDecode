#include "../src/lowbit.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

static void put_half(std::vector<uint8_t> & raw, size_t off, float x) {
    const __half h = __float2half(x);
    std::memcpy(raw.data() + off, &h, sizeof(h));
}

static float run_case(wd::GgmlType type, int cols, std::vector<uint8_t> raw) {
    wd::TensorInfo t;
    t.name = "synthetic";
    t.dims = {static_cast<uint64_t>(cols), 1};
    t.type = type;
    t.data = raw.data();

    auto w = wd::upload_lowbit_matrix(t);

    std::vector<__half> hx(static_cast<size_t>(cols), __float2half(1.0f));
    __half * dx = nullptr;
    float * dy = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&dx), hx.size() * sizeof(__half)), "cudaMalloc x");
    ck(cudaMalloc(reinterpret_cast<void **>(&dy), sizeof(float)), "cudaMalloc y");
    ck(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy x");

    wd::launch_lowbit_gemv_float(w, dx, dy, nullptr, 128);
    ck(cudaGetLastError(), "low-bit launch");
    ck(cudaDeviceSynchronize(), "low-bit sync");

    float y = 0.0f;
    ck(cudaMemcpy(&y, dy, sizeof(float), cudaMemcpyDeviceToHost), "copy y");
    cudaFree(dx);
    cudaFree(dy);
    return y;
}

int main() {
    {
        std::vector<uint8_t> raw(34, 0);
        put_half(raw, 0, 1.0f);
        for (int i = 0; i < 32; ++i) raw[2 + i] = 1;
        const float y = run_case(wd::GgmlType::Q8_0, 32, raw);
        if (std::fabs(y - 32.0f) > 0.01f) throw std::runtime_error("Q8_0 mismatch");
        std::cout << "Q8_0 PASS " << y << "\n";
    }

    {
        // Q6_K: q=1 everywhere. Stored code is q+32=33.
        std::vector<uint8_t> raw(210, 0);
        for (int i = 0; i < 128; ++i) raw[i] = 0x11;
        for (int i = 0; i < 64; ++i) raw[128 + i] = 0xaa;
        for (int i = 0; i < 16; ++i) raw[192 + i] = 1;
        put_half(raw, 208, 1.0f);
        const float y = run_case(wd::GgmlType::Q6_K, 256, raw);
        if (std::fabs(y - 256.0f) > 0.02f) throw std::runtime_error("Q6_K mismatch");
        std::cout << "Q6_K PASS " << y << "\n";
    }

    {
        std::vector<uint8_t> raw(18, 0);
        put_half(raw, 0, 1.0f);
        for (int i = 0; i < 16; ++i) raw[2 + i] = 0x88;
        const float y = run_case(wd::GgmlType::IQ4_NL, 32, raw);
        if (std::fabs(y - 32.0f) > 0.01f) throw std::runtime_error("IQ4_NL mismatch");
        std::cout << "IQ4_NL PASS " << y << "\n";
    }

    {
        // IQ2_S: d=1, scale nibble=0 => 0.125. Grid entry 0 stores
        // eight 8s, so every dequantized value is 0.125*8 = 1.
        std::vector<uint8_t> raw(82, 0);
        put_half(raw, 0, 1.0f);
        const float y = run_case(wd::GgmlType::IQ2_S, 256, raw);
        if (std::fabs(y - 256.0f) > 0.02f) throw std::runtime_error("IQ2_S mismatch");
        std::cout << "IQ2_S PASS " << y << "\n";
    }

    {
        // IQ3_S: d=1, scale nibble=0 => 1. Grid entry 0 is
        // {1,1,1,1}, high bits and signs are zero, so all 256 values are 1.
        std::vector<uint8_t> raw(110, 0);
        put_half(raw, 0, 1.0f);
        const float y = run_case(wd::GgmlType::IQ3_S, 256, raw);
        if (std::fabs(y - 256.0f) > 0.02f) throw std::runtime_error("IQ3_S mismatch");
        std::cout << "IQ3_S PASS " << y << "\n";
    }

    return 0;
}
