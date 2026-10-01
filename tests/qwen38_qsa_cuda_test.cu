#include "qwen38_qsa.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main() {
    try {
        constexpr int heads = 2;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 32;
        constexpr int values_n = kv_heads * head_dim;
        constexpr int output_n = heads * head_dim;

        std::vector<__half> q(output_n, __float2half(0.0f));
        std::vector<__half> gate(output_n);
        std::vector<__half> k_cache(values_n);
        std::vector<__half> v_cache(values_n);
        std::vector<__half> output(output_n);
        for (int i = 0; i < output_n; ++i) {
            gate[static_cast<size_t>(i)] = __float2half(-0.8f + 0.05f * i);
        }
        for (int i = 0; i < values_n; ++i) {
            k_cache[static_cast<size_t>(i)] = __float2half(0.0f);
            v_cache[static_cast<size_t>(i)] = __float2half(0.01f * (i + 1));
        }

        __half * d_q = nullptr;
        __half * d_gate = nullptr;
        __half * d_k = nullptr;
        __half * d_v = nullptr;
        __half * d_out = nullptr;
        check(cudaMalloc(reinterpret_cast<void **>(&d_q), q.size() * sizeof(__half)));
        check(cudaMalloc(reinterpret_cast<void **>(&d_gate), gate.size() * sizeof(__half)));
        check(cudaMalloc(reinterpret_cast<void **>(&d_k), k_cache.size() * sizeof(__half)));
        check(cudaMalloc(reinterpret_cast<void **>(&d_v), v_cache.size() * sizeof(__half)));
        check(cudaMalloc(reinterpret_cast<void **>(&d_out), output.size() * sizeof(__half)));
        check(cudaMemcpy(d_q, q.data(), q.size() * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_k, k_cache.data(), k_cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_v, v_cache.data(), v_cache.size() * sizeof(__half), cudaMemcpyHostToDevice));

        wd::q38_qsa_sparse_attention_kernel<<<heads, 256>>>(
            d_q, d_gate, d_k, d_v, nullptr, d_out,
            0, 4, 0, 0, heads, kv_heads, head_dim, 1.0f);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(output.data(), d_out, output.size() * sizeof(__half), cudaMemcpyDeviceToHost));

        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int i = h * head_dim + d;
                const float value = __half2float(v_cache[static_cast<size_t>(d)]);
                const float g = __half2float(gate[static_cast<size_t>(i)]);
                const float expected = value / (1.0f + std::exp(-g));
                const float actual = __half2float(output[static_cast<size_t>(i)]);
                if (std::abs(actual - expected) > 2e-3f) {
                    throw std::runtime_error("single-token QSA output mismatch at head " +
                        std::to_string(h) + " feature " + std::to_string(d));
                }
            }
        }

        cudaFree(d_q);
        cudaFree(d_gate);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_out);
        std::cout << "QSA single-token softmax PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "QSA single-token softmax FAIL: " << error.what() << '\n';
        return 1;
    }
}
