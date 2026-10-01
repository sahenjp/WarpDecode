#include "qwen38_qsa.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void test_imrope_rotary_dimensions() {
    wd::Qwen38Layout layout;
    layout.rope_dimension_count = 4;
    layout.rope_freq_base = 10000.0f;
    layout.rope_sections = {1, 1, 0, 0};
    const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<__half> actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) actual[i] = __float2half(input[i]);

    __half * device = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device), actual.size() * sizeof(__half)));
    check(cudaMemcpy(device, actual.data(), actual.size() * sizeof(__half), cudaMemcpyHostToDevice));
    wd::launch_qwen38_imrope(device, 8, 1, wd::Qwen38Position4::text(1), layout, nullptr);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    cudaFree(device);

    const float theta_scale = std::pow(layout.rope_freq_base, -2.0f / layout.rope_dimension_count);
    for (int pair = 0; pair < 2; ++pair) {
        const float theta = std::pow(theta_scale, static_cast<float>(pair));
        const float cs = std::cos(theta);
        const float sn = std::sin(theta);
        const int other = pair + 2;
        const float expected_a = input[static_cast<size_t>(pair)] * cs - input[static_cast<size_t>(other)] * sn;
        const float expected_b = input[static_cast<size_t>(pair)] * sn + input[static_cast<size_t>(other)] * cs;
        if (std::abs(__half2float(actual[static_cast<size_t>(pair)]) - expected_a) > 2e-3f ||
            std::abs(__half2float(actual[static_cast<size_t>(other)]) - expected_b) > 2e-3f) {
            throw std::runtime_error("IMRoPE used the wrong rotary dimension or pair offset");
        }
    }
    for (size_t i = 4; i < input.size(); ++i) {
        if (std::abs(__half2float(actual[i]) - input[i]) > 1e-3f) {
            throw std::runtime_error("IMRoPE modified a non-rotary head dimension");
        }
    }
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
        test_imrope_rotary_dimensions();
        std::cout << "QSA single-token softmax and IMRoPE dimensions PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "QSA single-token softmax FAIL: " << error.what() << '\n';
        return 1;
    }
}
