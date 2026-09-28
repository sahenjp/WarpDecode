#include "gguf.hpp"
#include "matrix.cuh"
#include "qwen38_hc.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct RouterMax {
    int id;
    float logit;
    float weight;
};

__global__ void qwen38_probe_router_gemv_f32(
        const float * __restrict__ weights,
        const float * __restrict__ input,
        float * __restrict__ logits,
        int experts,
        int hidden) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int expert = blockIdx.x * warps_per_block + warp_in_block;
    if (expert >= experts) return;

    const float * row = weights + static_cast<size_t>(expert) * hidden;
    float sum = 0.0f;
    for (int i = lane; i < hidden; i += 32) sum = fmaf(row[i], input[i], sum);
    sum = wd::warp_sum(sum);
    if (lane == 0) logits[expert] = sum;
}

std::vector<float> read_f32(const char * path, size_t expected) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot open router input: ") + path);
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size != static_cast<std::streamoff>(expected * sizeof(float))) {
        throw std::runtime_error("router input must contain exactly hidden F32 values");
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(expected);
    file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("failed to read router input");
    return values;
}

wd::LowBitMatrix upload_embedding_row(const wd::TensorInfo & tensor, int hidden) {
    constexpr int token = 1;
    if (tensor.type != wd::GgmlType::Q8_0 || tensor.dims.size() != 2 ||
        static_cast<int>(tensor.dims[0]) != hidden || token < 0 || token >= static_cast<int>(tensor.dims[1]) ||
        hidden % 32 != 0) {
        throw std::runtime_error("token_embd.weight must be Q8_0 [hidden,vocab]");
    }

    const size_t row_bytes = static_cast<size_t>(hidden / 32) * 34;
    std::vector<uint8_t> two_rows(row_bytes * 2, 0);
    std::memcpy(two_rows.data() + row_bytes,
                tensor.data + static_cast<size_t>(token) * row_bytes, row_bytes);

    wd::LowBitMatrix out;
    out.rows = 2;
    out.cols = hidden;
    out.type = wd::GgmlType::Q8_0;
    out.bytes = two_rows.size();
    const cudaError_t alloc = cudaMalloc(reinterpret_cast<void **>(&out.data), out.bytes);
    if (alloc != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc token row failed: ") + cudaGetErrorString(alloc));
    const cudaError_t copy = cudaMemcpy(out.data, two_rows.data(), out.bytes, cudaMemcpyHostToDevice);
    if (copy != cudaSuccess) throw std::runtime_error(std::string("copy token row failed: ") + cudaGetErrorString(copy));
    return out;
}

std::vector<RouterMax> rank_router(const std::vector<float> & logits, float weight_scale) {
    std::vector<RouterMax> ranked;
    ranked.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        ranked.push_back({static_cast<int>(i), logits[i], 0.0f});
    }
    const auto before = [](const RouterMax & a, const RouterMax & b) {
        return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
    };
    std::partial_sort(ranked.begin(), ranked.begin() + 11, ranked.end(), before);

    float sum = 0.0f;
    for (int i = 0; i < 10; ++i) {
        ranked[i].weight = std::exp(ranked[i].logit - ranked[0].logit);
        sum += ranked[i].weight;
    }
    const float scale = weight_scale / sum;
    for (int i = 0; i < 10; ++i) ranked[i].weight *= scale;
    return ranked;
}

void dump_float_array(const char * name, const std::vector<float> & values) {
    std::cout << ",\"" << name << "\":[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << std::setprecision(9) << std::defaultfloat << values[i];
    }
    std::cout << ']';
}

std::vector<float> copy_f32(const float * data, size_t n) {
    std::vector<float> out(n);
    const cudaError_t copy = cudaMemcpy(out.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy != cudaSuccess) throw std::runtime_error(std::string("copy F32 probe output failed: ") + cudaGetErrorString(copy));
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: warodecode-qwen38-layer0-probe MODEL.gguf ROUTER_INPUT.f32");
        }
        constexpr int token = 1;
        wd::GgufFile gguf(argv[1], false);
        const wd::TensorInfo & token_embd = gguf.tensor("token_embd.weight");
        const wd::TensorInfo & norm = gguf.tensor("blk.0.hc_attn_norm.weight");
        const wd::TensorInfo & down = gguf.tensor("blk.0.hc_attn_down.weight");
        const wd::TensorInfo & up = gguf.tensor("blk.0.hc_attn_up.weight");
        const wd::TensorInfo & router = gguf.tensor("blk.0.ffn_gate_inp.weight");

        if (token_embd.dims.size() != 2 || norm.type != wd::GgmlType::F32 ||
            down.type != wd::GgmlType::Q8_0 || up.type != wd::GgmlType::Q8_0 ||
            router.type != wd::GgmlType::F32) {
            throw std::runtime_error("unexpected layer-0 probe tensor types");
        }
        const int hidden = static_cast<int>(token_embd.dims[0]);
        const int hc_dim = static_cast<int>(norm.elements());
        if (hidden <= 0 || hc_dim % hidden != 0) throw std::runtime_error("invalid HC norm geometry");
        const int hc = hc_dim / hidden;
        const int low_rank = static_cast<int>(down.dims[1]);
        const int experts = static_cast<int>(router.dims[1]);
        if (down.dims.size() != 2 || up.dims.size() != 2 || router.dims.size() != 2 ||
            experts < 11 ||
            static_cast<int>(down.dims[0]) != hc_dim ||
            static_cast<int>(up.dims[0]) != low_rank || static_cast<int>(up.dims[1]) != hc_dim ||
            static_cast<int>(router.dims[0]) != hidden) {
            throw std::runtime_error("invalid layer-0 probe tensor shapes");
        }

        wd::LowBitMatrix embedding = upload_embedding_row(token_embd, hidden);
        wd::Qwen38HcWeights hc_weights;
        hc_weights.debug_norm_f32.assign(
            reinterpret_cast<const float *>(norm.data),
            reinterpret_cast<const float *>(norm.data) + norm.elements());
        hc_weights.down = wd::upload_matrix(down);
        hc_weights.up = wd::upload_matrix(up);
        wd::Qwen38HcF32DebugScratch hc_scratch;
        hc_scratch.initialize(hidden, hc, low_rank);
        wd::run_qwen38_hc_f32_debug(embedding, token, hc_weights, hc_scratch,
                                    static_cast<float>(gguf.number_meta("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6)), nullptr);

        std::vector<float> router_input = read_f32(argv[2], static_cast<size_t>(hidden));
        float * d_router_weights = nullptr;
        float * d_router_input = nullptr;
        float * d_logits = nullptr;
        auto alloc = [](void ** p, size_t n, const char * what) {
            const cudaError_t e = cudaMalloc(p, n);
            if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc ") + what + ": " + cudaGetErrorString(e));
        };
        alloc(reinterpret_cast<void **>(&d_router_weights),
              static_cast<size_t>(experts) * hidden * sizeof(float), "router weights");
        alloc(reinterpret_cast<void **>(&d_router_input), static_cast<size_t>(hidden) * sizeof(float), "router input");
        alloc(reinterpret_cast<void **>(&d_logits), static_cast<size_t>(experts) * sizeof(float), "router logits");
        cudaError_t e = cudaMemcpy(d_router_weights, router.data,
                                   static_cast<size_t>(experts) * hidden * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("copy router weights failed: ") + cudaGetErrorString(e));
        e = cudaMemcpy(d_router_input, router_input.data(), static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("copy router input failed: ") + cudaGetErrorString(e));
        const int warps = 8;
        qwen38_probe_router_gemv_f32<<<(experts + warps - 1) / warps, warps * 32>>>(
            d_router_weights, d_router_input, d_logits, experts, hidden);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) throw std::runtime_error(std::string("router GEMV failed: ") + cudaGetErrorString(e));

        const size_t hc_size = static_cast<size_t>(hc_dim);
        const std::vector<float> init = copy_f32(hc_scratch.residual(), hc_size);
        const std::vector<float> hc_norm = copy_f32(hc_scratch.norm(), hc_size);
        const std::vector<float> hc_low_raw = copy_f32(hc_scratch.low_raw(), static_cast<size_t>(low_rank));
        const std::vector<float> hc_low = copy_f32(hc_scratch.low(), static_cast<size_t>(low_rank));
        const std::vector<float> hc_gate = copy_f32(hc_scratch.gate(), hc_size);
        const std::vector<float> hc_mixed = copy_f32(hc_scratch.mixed(), static_cast<size_t>(hidden));
        const std::vector<float> logits = copy_f32(d_logits, static_cast<size_t>(experts));
        const float weight_scale = static_cast<float>(gguf.number_meta("qwen4exp.expert_weights_scale", 1.0));
        const std::vector<RouterMax> top = rank_router(logits, weight_scale);

        std::cout << "PROBE_RESULT={\"token\":" << token << ",\"hidden\":" << hidden
                  << ",\"hc\":" << hc << ",\"low_rank\":" << low_rank << ",\"router_input_source\":\"external_f32\"";
        dump_float_array("hc_init", init);
        dump_float_array("hc_norm", hc_norm);
        dump_float_array("hc_low_raw", hc_low_raw);
        dump_float_array("hc_low", hc_low);
        dump_float_array("hc_gate", hc_gate);
        dump_float_array("hc_mixed", hc_mixed);
        dump_float_array("router_logits", logits);
        std::cout << ",\"expert_top10\":[";
        for (int i = 0; i < 10; ++i) {
            if (i != 0) std::cout << ',';
            std::cout << '[' << top[i].id << ',' << std::setprecision(9) << std::defaultfloat << top[i].weight << ']';
        }
        std::cout << "],\"rank11\":[" << top[10].id << ',' << top[10].logit
                  << "],\"margin_10_11\":" << top[9].logit - top[10].logit << "}\n";

        cudaFree(d_router_weights);
        cudaFree(d_router_input);
        cudaFree(d_logits);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "layer-0 probe error: " << e.what() << '\n';
        return 1;
    }
}
