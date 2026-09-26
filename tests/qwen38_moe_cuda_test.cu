#include "../src/qwen38_moe.cuh"

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

static std::vector<uint8_t> make_q8_0_2d(int rows, int cols, int q) {
    if (cols % 32 != 0) throw std::runtime_error("cols");
    const int blocks = cols / 32;
    std::vector<uint8_t> raw(static_cast<size_t>(rows) * blocks * 34u, 0);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const size_t off = (static_cast<size_t>(r) * blocks + b) * 34u;
            put_half(raw, off, 1.0f);
            for (int i = 0; i < 32; ++i) {
                raw[off + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
            }
        }
    }
    return raw;
}

static std::vector<uint8_t> make_router() {
    std::vector<uint8_t> raw(4u * 34u, 0);
    for (int r = 0; r < 4; ++r) {
        const size_t off = static_cast<size_t>(r) * 34u;
        put_half(raw, off, 1.0f);
        for (int i = 0; i < 32; ++i) {
            raw[off + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(r + 1));
        }
    }
    return raw;
}

int main() {
    auto router_raw = make_router();

    // Fused gate-up: 64 output rows x 32 input columns, four experts.
    auto one_gate_up = make_q8_0_2d(64, 32, 1);
    auto one_down = make_q8_0_2d(32, 32, 1);
    std::vector<uint8_t> gate_up_raw;
    std::vector<uint8_t> down_raw;
    for (int e = 0; e < 4; ++e) {
        gate_up_raw.insert(gate_up_raw.end(), one_gate_up.begin(), one_gate_up.end());
        down_raw.insert(down_raw.end(), one_down.begin(), one_down.end());
    }

    wd::TensorInfo router_t;
    router_t.name = "router";
    router_t.dims = {32, 4};
    router_t.type = wd::GgmlType::Q8_0;
    router_t.data = router_raw.data();

    wd::TensorInfo gate_up_t;
    gate_up_t.name = "gate_up";
    gate_up_t.dims = {32, 64, 4};
    gate_up_t.type = wd::GgmlType::Q8_0;
    gate_up_t.data = gate_up_raw.data();

    wd::TensorInfo down_t;
    down_t.name = "down";
    down_t.dims = {32, 32, 4};
    down_t.type = wd::GgmlType::Q8_0;
    down_t.data = down_raw.data();

    auto router = wd::upload_lowbit_matrix(router_t);

    wd::GpuFusedExpertCache cache;
    cache.initialize(1, gate_up_t, down_t);

    wd::Qwen38MoeScratch scratch;
    scratch.initialize(4, 2, 32, 32);

    std::vector<__half> hx(32, __float2half(1.0f));
    __half * dx = nullptr;
    __half * dy = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&dx), 32 * sizeof(__half)), "alloc x");
    ck(cudaMalloc(reinterpret_cast<void **>(&dy), 32 * sizeof(__half)), "alloc y");
    ck(cudaMemcpy(dx, hx.data(), 32 * sizeof(__half), cudaMemcpyHostToDevice), "copy x");

    wd::run_qwen38_routed_moe_fused(
        0, dx, router, gate_up_t, down_t, cache, scratch, dy, 1.0f, nullptr);

    // Shared expert: identical all-one Q8 matrices. A zero gate-input vector
    // gives sigmoid(0)=0.5, so it contributes half of the routed expert output.
    auto shared_gate_raw = make_q8_0_2d(32, 32, 1);
    auto shared_up_raw = make_q8_0_2d(32, 32, 1);
    auto shared_down_raw = make_q8_0_2d(32, 32, 1);
    std::vector<float> shared_gate_input_raw(32, 0.0f);

    wd::TensorInfo shared_gate_input_t;
    shared_gate_input_t.name = "shared_gate_input";
    shared_gate_input_t.dims = {32};
    shared_gate_input_t.type = wd::GgmlType::F32;
    shared_gate_input_t.data = reinterpret_cast<const uint8_t *>(shared_gate_input_raw.data());

    wd::TensorInfo shared_gate_t;
    shared_gate_t.name = "shared_gate";
    shared_gate_t.dims = {32, 32};
    shared_gate_t.type = wd::GgmlType::Q8_0;
    shared_gate_t.data = shared_gate_raw.data();

    wd::TensorInfo shared_up_t = shared_gate_t;
    shared_up_t.name = "shared_up";
    shared_up_t.data = shared_up_raw.data();

    wd::TensorInfo shared_down_t = shared_gate_t;
    shared_down_t.name = "shared_down";
    shared_down_t.data = shared_down_raw.data();

    wd::Qwen38SharedExpertWeights shared;
    shared.gate_input = wd::upload_half_vector(shared_gate_input_t);
    shared.gate = wd::upload_lowbit_matrix(shared_gate_t);
    shared.up = wd::upload_lowbit_matrix(shared_up_t);
    shared.down = wd::upload_lowbit_matrix(shared_down_t);

    wd::run_qwen38_shared_expert(dx, shared, scratch, dy, nullptr);

    ck(cudaGetLastError(), "MoE launch");
    ck(cudaDeviceSynchronize(), "MoE sync");

    if (scratch.h_ids()[0] != 3 || scratch.h_ids()[1] != 2) {
        throw std::runtime_error("router top-k mismatch");
    }
    if (!(scratch.h_weights()[0] > 0.999f && scratch.h_weights()[1] < 0.001f)) {
        throw std::runtime_error("router softmax mismatch");
    }

    std::vector<__half> hy(32);
    ck(cudaMemcpy(hy.data(), dy, 32 * sizeof(__half), cudaMemcpyDeviceToHost), "copy y");
    const float y0 = __half2float(hy[0]);
    // Routed ~=32768. Shared is identical but gated by sigmoid(0)=0.5,
    // so the combined result is ~=49152.
    if (std::fabs(y0 - 49152.0f) > 96.0f) {
        throw std::runtime_error("Qwen3.8 MoE numerical mismatch: " + std::to_string(y0));
    }

    std::cout << "Qwen3.8 routed+shared MoE PASS top=" << scratch.h_ids()[0]
              << "," << scratch.h_ids()[1] << " y=" << y0
              << " slots=" << cache.slot_count() << "\n";

    cudaFree(dx);
    cudaFree(dy);
    return 0;
}
