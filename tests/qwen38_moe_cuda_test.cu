#include "../src/matrix.cuh"
#include "../src/qwen38_moe.cuh"
#include "../src/qwen38_trace.hpp"

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

static std::vector<uint8_t> make_q8_0_2d(int rows, int cols, int q, float scale = 1.0f) {
    if (cols % 32 != 0) throw std::runtime_error("cols");
    const int blocks = cols / 32;
    std::vector<uint8_t> raw(static_cast<size_t>(rows) * blocks * 34u, 0);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const size_t off = (static_cast<size_t>(r) * blocks + b) * 34u;
            put_half(raw, off, scale);
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

static void test_separate_expert_trace(const wd::GpuMatrix & router, const __half * x) {
    auto one = make_q8_0_2d(32, 32, 1);
    std::vector<uint8_t> gate_raw;
    std::vector<uint8_t> up_raw;
    std::vector<uint8_t> down_raw;
    for (int e = 0; e < 4; ++e) {
        gate_raw.insert(gate_raw.end(), one.begin(), one.end());
        up_raw.insert(up_raw.end(), one.begin(), one.end());
        auto down = make_q8_0_2d(32, 32, 1, static_cast<float>(e + 1) * 0.25f);
        down_raw.insert(down_raw.end(), down.begin(), down.end());
    }
    auto make_tensor = [](const char * name, std::vector<uint8_t> & raw) {
        wd::TensorInfo tensor;
        tensor.name = name;
        tensor.dims = {32, 32, 4};
        tensor.type = wd::GgmlType::Q8_0;
        tensor.data = raw.data();
        return tensor;
    };
    auto gate = make_tensor("gate", gate_raw);
    auto up = make_tensor("up", up_raw);
    auto down = make_tensor("down", down_raw);

    wd::GpuExpertCache cache;
    cache.initialize(1, gate, up, down);
    wd::Qwen38MoeScratch scratch;
    scratch.initialize(4, 2, 32, 32);
    __half * output = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&output), 32 * sizeof(__half)), "alloc separate output");
    __half * selected_trace = scratch.selected_output_trace();
    wd::run_qwen38_routed_moe_separate<true>(
        0, x, router, gate, up, down, cache, scratch, output, 1.0f, nullptr,
        false, selected_trace);
    ck(cudaGetLastError(), "separate trace launch");
    ck(cudaDeviceSynchronize(), "separate trace sync");

    std::vector<__half> selected(2u * 32u);
    std::vector<float> routed(32);
    ck(cudaMemcpy(selected.data(), selected_trace, selected.size() * sizeof(__half), cudaMemcpyDeviceToHost),
       "copy separate selected outputs");
    ck(cudaMemcpy(routed.data(), scratch.d_accum(), routed.size() * sizeof(float), cudaMemcpyDeviceToHost),
       "copy separate routed sum");
    if (scratch.h_ids()[0] != 3 || scratch.h_ids()[1] != 2 ||
        !(__half2float(selected[wd::qwen38_expert_trace_offset(0, 32)]) >
          __half2float(selected[wd::qwen38_expert_trace_offset(1, 32)]))) {
        throw std::runtime_error("separate expert trace rank indexing mismatch");
    }
    for (int i = 0; i < 32; ++i) {
        float expected = 0.0f;
        for (int rank = 0; rank < 2; ++rank) {
            expected = std::fma(scratch.h_weights()[rank],
                __half2float(selected[static_cast<size_t>(rank) * 32 + i]), expected);
        }
        if (std::fabs(expected - routed[static_cast<size_t>(i)]) > 0.1f) {
            throw std::runtime_error("separate expert outputs do not reconstruct routed sum");
        }
    }
    cudaFree(output);
}

int main() {
    auto router_raw = make_router();

    // Fused gate-up: 64 output rows x 32 input columns, four experts.
    auto one_gate_up = make_q8_0_2d(64, 32, 1);
    std::vector<uint8_t> gate_up_raw;
    std::vector<uint8_t> down_raw;
    for (int e = 0; e < 4; ++e) {
        gate_up_raw.insert(gate_up_raw.end(), one_gate_up.begin(), one_gate_up.end());
        auto down = make_q8_0_2d(32, 32, 1, static_cast<float>(e + 1) * 0.25f);
        down_raw.insert(down_raw.end(), down.begin(), down.end());
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

    auto router = wd::upload_matrix(router_t);

    std::vector<__half> hx(32, __float2half(1.0f));
    __half * dx = nullptr;
    __half * dy = nullptr;
    ck(cudaMalloc(reinterpret_cast<void **>(&dx), 32 * sizeof(__half)), "alloc x");
    ck(cudaMalloc(reinterpret_cast<void **>(&dy), 32 * sizeof(__half)), "alloc y");
    ck(cudaMemcpy(dx, hx.data(), 32 * sizeof(__half), cudaMemcpyHostToDevice), "copy x");
    test_separate_expert_trace(router, dx);

    wd::GpuFusedExpertCache cache;
    cache.initialize(1, gate_up_t, down_t);

    wd::Qwen38MoeScratch scratch;
    scratch.initialize(4, 2, 32, 32);

    __half * selected_trace = scratch.selected_output_trace();
    wd::run_qwen38_routed_moe_fused<true>(
        0, dx, router, gate_up_t, down_t, cache, scratch, dy, 1.0f, nullptr,
        false, selected_trace);

    std::vector<__half> selected_outputs(2u * 32u);
    ck(cudaMemcpy(selected_outputs.data(), selected_trace, selected_outputs.size() * sizeof(__half),
                  cudaMemcpyDeviceToHost), "copy selected expert outputs");
    std::vector<float> routed_sum(32);
    ck(cudaMemcpy(routed_sum.data(), scratch.d_accum(), routed_sum.size() * sizeof(float),
                  cudaMemcpyDeviceToHost), "copy routed weighted sum");
    std::vector<__half> routed_output(32);
    ck(cudaMemcpy(routed_output.data(), dy, routed_output.size() * sizeof(__half),
                  cudaMemcpyDeviceToHost), "copy routed output");

    const size_t expert0 = wd::qwen38_expert_trace_offset(0, 32);
    const size_t expert1 = wd::qwen38_expert_trace_offset(1, 32);
    if (!(__half2float(selected_outputs[expert0]) > __half2float(selected_outputs[expert1]))) {
        throw std::runtime_error(
            "selected expert outputs were not captured in rank order: " +
            std::to_string(__half2float(selected_outputs[expert0])) + "," +
            std::to_string(__half2float(selected_outputs[expert1])) + " ids=" +
            std::to_string(scratch.h_ids()[0]) + "," + std::to_string(scratch.h_ids()[1]));
    }
    for (int i = 0; i < 32; ++i) {
        float expected = 0.0f;
        for (int rank = 0; rank < 2; ++rank) {
            expected = std::fma(
                scratch.h_weights()[rank],
                __half2float(selected_outputs[static_cast<size_t>(rank) * 32 + i]),
                expected);
        }
        if (std::fabs(expected - routed_sum[static_cast<size_t>(i)]) > 0.1f) {
            throw std::runtime_error("selected outputs do not reconstruct the weighted routed sum");
        }
        if (std::fabs(__half2float(routed_output[static_cast<size_t>(i)]) - routed_sum[static_cast<size_t>(i)]) > 1.0f) {
            throw std::runtime_error("routed output differs from the captured weighted sum");
        }
    }

    // The shared branch remains separately inspectable after its gated output
    // is added to the routed branch in place.
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
    std::vector<__half> shared_output(32);
    float shared_gate = 0.0f;
    ck(cudaMemcpy(shared_output.data(), scratch.d_down(), shared_output.size() * sizeof(__half),
                  cudaMemcpyDeviceToHost), "copy shared expert output");
    ck(cudaMemcpy(&shared_gate, scratch.d_shared_gate(), sizeof(float), cudaMemcpyDeviceToHost),
       "copy shared expert gate");
    if (std::fabs(shared_gate - 0.5f) > 1e-6f) {
        throw std::runtime_error("shared expert gate mismatch");
    }
    for (int i = 0; i < 32; ++i) {
        const size_t index = static_cast<size_t>(i);
        const float expected = __half2float(routed_output[index]) +
                              shared_gate * __half2float(shared_output[index]);
        if (std::fabs(__half2float(hy[index]) - expected) > 2.0f) {
            throw std::runtime_error("routed/shared output separation mismatch");
        }
    }

    std::cout << "Qwen3.8 routed+shared MoE PASS top=" << scratch.h_ids()[0]
              << "," << scratch.h_ids()[1] << " expert0=" << __half2float(selected_outputs[expert0])
              << " expert1=" << __half2float(selected_outputs[expert1])
              << " shared=" << __half2float(shared_output[0])
              << " gate=" << shared_gate << " final=" << __half2float(hy[0])
              << " slots=" << cache.slot_count() << "\n";

    cudaFree(dx);
    cudaFree(dy);
    return 0;
}
