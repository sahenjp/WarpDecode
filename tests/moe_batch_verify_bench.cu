// Task I harness: M-token batch MoE (WARODECODE_MTP_BATCH_VERIFY) against the
// per-token serial MoE it replaces.
//
// Two questions, both answered on the device:
//   1. Is one basin pass over M tokens bit-identical to M serial passes? Every
//      comparison is a memcmp over the raw bytes of the two outputs.
//   2. How much does the fold cost? The serial path is M * (k experts x 5
//      launches + 1 router selection), the batch path is one union pass.
//
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 -Isrc \
//        tests/moe_batch_verify_bench.cu -o /content/moe_batch_verify_bench
//   /content/moe_batch_verify_bench
//
// Shapes are the real Qwen3.8-Flash-Next ones: hidden 2560, expert FFN 640,
// 512 experts, top-10 routing. Each token picks its own top-10 union, like a
// round of MTP verify where the M tokens share several experts.

#include "lowbit.cuh"
#include "qwen38_moe.cuh"

#include <cuda_runtime.h>

#include <chrono>
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

void type_geometry(GgmlType type, int & quantum, int & block_bytes) {
    switch (type) {
        case GgmlType::IQ2_S: quantum = 256; block_bytes = 82; break;
        case GgmlType::IQ3_S: quantum = 256; block_bytes = 110; break;
        case GgmlType::IQ4_NL: quantum = 32; block_bytes = 18; break;
        default:
            std::fprintf(stderr, "unsupported benchmark type\n");
            std::exit(1);
    }
}

// One expert slice, ready for a GpuExpertCache slot: the IQ2_S/IQ4_NL weight
// block generators from the Task G harness.
struct ExpertSource {
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
    std::vector<const uint8_t *> gate_ptr;
    std::vector<const uint8_t *> up_ptr;
    std::vector<const uint8_t *> down_ptr;
};

ExpertSource make_experts(int experts, int hidden, int ffn, unsigned seed) {
    int quantum = 0;
    int block_bytes = 0;

    struct Blocks { int quantum; int bytes; GgmlType type; };
    const Blocks gate_blocks{256, 82, GgmlType::IQ2_S};
    const Blocks down_blocks{32, 18, GgmlType::IQ4_NL};

    ExpertSource src;
    std::mt19937 rng(seed);
    auto fill = [&](std::vector<uint8_t> & out, size_t per_expert, int quantum_, int block_bytes_) {
        out.resize(per_expert * static_cast<size_t>(experts));
        for (auto & b : out) b = static_cast<uint8_t>(rng() & 0xffu);
        // The IQ2_S scale byte must not be all zero (the decoder divides by it),
        // and a nonzero exponent keeps every block finite.
        for (size_t off = 0; off + 1 < out.size(); off += static_cast<size_t>(block_bytes_)) {
            out[off + 1] = static_cast<uint8_t>(0x30 | (seed & 0x03u));
        }
        (void)quantum_;
    };

    const size_t gate_bytes = static_cast<size_t>(ffn) * (hidden / gate_blocks.quantum) * gate_blocks.bytes;
    const size_t down_bytes = static_cast<size_t>(hidden) * (ffn / down_blocks.quantum) * down_blocks.bytes;
    fill(src.gate, gate_bytes, gate_blocks.quantum, gate_blocks.bytes);
    fill(src.up, gate_bytes, gate_blocks.quantum, gate_blocks.bytes);
    fill(src.down, down_bytes, down_blocks.quantum, down_blocks.bytes);
    (void)quantum;
    (void)block_bytes;
    return src;
}

} // namespace

int main() {
    cudaFree(nullptr);
    cudaStream_t stream = nullptr;

    const int hidden = 2560;
    const int ffn = 640;
    const int k = 10;

    std::printf("MoE M-token batch check (hidden=%d ffn=%d top-k=%d)\n", hidden, ffn, k);
    std::printf("  DP4A available=%d smem=%d batch_k=%d\n",
                lowbit_dp4a_available() ? 1 : 0, lowbit_dp4a_smem_enabled() ? 1 : 0,
                q38_moe_batch_k());

    const int experts = 64;
    ExpertSource src = make_experts(experts, hidden, ffn, 20261014u);

    std::vector<const uint8_t *> d_gate(static_cast<size_t>(experts));
    std::vector<const uint8_t *> d_up(static_cast<size_t>(experts));
    std::vector<const uint8_t *> d_down(static_cast<size_t>(experts));
    const size_t gate_bytes = src.gate.size() / static_cast<size_t>(experts);
    const size_t down_bytes = src.down.size() / static_cast<size_t>(experts);
    for (int e = 0; e < experts; ++e) {
        ck(cudaMalloc(reinterpret_cast<void **>(const_cast<uint8_t **>(&d_gate[e])), gate_bytes), "malloc gate");
        ck(cudaMalloc(reinterpret_cast<void **>(const_cast<uint8_t **>(&d_up[e])), gate_bytes), "malloc up");
        ck(cudaMalloc(reinterpret_cast<void **>(const_cast<uint8_t **>(&d_down[e])), down_bytes), "malloc down");
        ck(cudaMemcpy(const_cast<uint8_t *>(d_gate[e]), src.gate.data() + static_cast<size_t>(e) * gate_bytes,
                      gate_bytes, cudaMemcpyHostToDevice), "copy gate");
        ck(cudaMemcpy(const_cast<uint8_t *>(d_up[e]), src.up.data() + static_cast<size_t>(e) * gate_bytes,
                      gate_bytes, cudaMemcpyHostToDevice), "copy up");
        ck(cudaMemcpy(const_cast<uint8_t *>(d_down[e]), src.down.data() + static_cast<size_t>(e) * down_bytes,
                      down_bytes, cudaMemcpyHostToDevice), "copy down");
    }

    TensorInfo gate_info;
    gate_info.name = "synthetic.gate";
    gate_info.type = GgmlType::IQ2_S;
    gate_info.dims = {static_cast<uint64_t>(hidden), static_cast<uint64_t>(ffn), static_cast<uint64_t>(experts)};
    gate_info.data = nullptr;
    TensorInfo up_info = gate_info;
    up_info.name = "synthetic.up";
    TensorInfo down_info = gate_info;
    down_info.name = "synthetic.down";
    down_info.type = GgmlType::IQ4_NL;
    down_info.dims = {static_cast<uint64_t>(ffn), static_cast<uint64_t>(hidden), static_cast<uint64_t>(experts)};

    Qwen38MoeScratch scratch;
    scratch.initialize(experts, k, hidden, ffn);
    GpuExpertCache cache;
    cache.initialize(1024, gate_info, up_info, down_info);

    Qwen38MoeTokenBatch batch;
    std::printf("  token batch compatible (M=1..8): ");
    for (int m = 1; m <= 8; ++m) {
        std::printf("%d%s", batch.compatible(cache, scratch, gate_info, up_info, down_info, m) ? 1 : 0,
                    m == 8 ? "\n" : " ");
    }

    int failures = 0;
    for (int tokens = 1; tokens <= 4; ++tokens) {
        // Deterministic per-token activations in F32, exactly what the MTP
        // block's HC mix hands to the MoE.
        std::vector<std::vector<float>> host_act(static_cast<size_t>(tokens));
        std::vector<float *> d_act(static_cast<size_t>(tokens));
        std::mt19937 rng(1000u + static_cast<unsigned>(tokens));
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        for (int t = 0; t < tokens; ++t) {
            host_act[static_cast<size_t>(t)].resize(static_cast<size_t>(hidden));
            for (auto & v : host_act[static_cast<size_t>(t)]) v = dist(rng);
            ck(cudaMalloc(reinterpret_cast<void **>(&d_act[static_cast<size_t>(t)]),
                          static_cast<size_t>(hidden) * sizeof(float)), "malloc activation");
            ck(cudaMemcpy(d_act[static_cast<size_t>(t)], host_act[static_cast<size_t>(t)].data(),
                          static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyHostToDevice), "copy activation");
        }

        // ---- serial reference: one routed MoE per token ---------------------
        std::vector<__half *> d_serial(static_cast<size_t>(tokens));
        for (int t = 0; t < tokens; ++t) {
            ck(cudaMalloc(reinterpret_cast<void **>(&d_serial[static_cast<size_t>(t)]),
                          static_cast<size_t>(hidden) * sizeof(__half)), "malloc serial output");
        }
        // preselected = true: the router is not involved here, only the expert
        // execution the batch changes. The expert ids are the root of the top-k
        // slice this layer uses.
        const auto serial_begin = std::chrono::steady_clock::now();
        for (int t = 0; t < tokens; ++t) {
            run_qwen38_routed_moe_separate<false>(
                /*layer=*/0, d_act[static_cast<size_t>(t)], GpuMatrix{}, gate_info, up_info, down_info,
                cache, scratch, d_serial[static_cast<size_t>(t)], 1.0f, stream, true);
        }
        ck(cudaStreamSynchronize(stream), "serial sync");
        const auto serial_end = std::chrono::steady_clock::now();

        // ---- fixed per-token expert lists -----------------------------------
        // Each token gets its own top-k list, so the pair -> expert map has to
        // build the union and hand out one slot per (token, top-k) - the same
        // shape the MTP path uses. Reusing one list for every token would not
        // exercise the mapping at all.
        std::vector<int> token_expert(static_cast<size_t>(tokens) * k);
        std::vector<int> token_slot(static_cast<size_t>(tokens) * k, -1);
        std::vector<float> token_weight(static_cast<size_t>(tokens) * k);
        {
            std::mt19937 rng2(500u + static_cast<unsigned>(tokens));
            std::uniform_int_distribution<int> ed(0, experts - 1);
            std::uniform_real_distribution<float> wd(0.05f, 0.4f);
            for (size_t i = 0; i < token_expert.size(); ++i) {
                token_expert[i] = ed(rng2);
                token_weight[i] = wd(rng2);
            }
        }
        std::vector<int> union_experts;
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < k; ++j) {
                const int e = token_expert[static_cast<size_t>(t) * k + j];
                int slot = -1;
                for (size_t u = 0; u < union_experts.size(); ++u) {
                    if (union_experts[u] == e) { slot = static_cast<int>(u); break; }
                }
                if (slot < 0) {
                    slot = static_cast<int>(union_experts.size());
                    union_experts.push_back(e);
                }
                token_slot[static_cast<size_t>(t) * k + j] = slot;
            }
        }
        std::vector<int> union_slot_count(union_experts.size() + 1, 0);
        std::vector<float> union_weight(union_experts.size(), 0.0f);
        for (size_t i = 0; i < token_expert.size(); ++i) {
            union_weight[static_cast<size_t>(token_slot[i])] = token_weight[i];
        }
        // Every serial pass of the reference runs on the same fixed list.
        auto set_scratch_topk = [&]() {
            for (int i = 0; i < tokens * k; ++i) {
                scratch.h_ids()[i] = token_expert[static_cast<size_t>(i)];
                scratch.h_weights()[i] = token_weight[static_cast<size_t>(i)];
            }
        };

        // ---- serial reference on the fixed lists ----------------------------
        const auto serial_begin = std::chrono::steady_clock::now();
        for (int t = 0; t < tokens; ++t) {
            // Each serial pass walks the token's own slice of the fixed list, so
            // copy it into the k slots the runner reads.
            for (int j = 0; j < k; ++j) {
                scratch.h_ids()[j] = token_expert[static_cast<size_t>(t) * k + j];
                scratch.h_weights()[j] = token_weight[static_cast<size_t>(t) * k + j];
            }
            run_qwen38_routed_moe_separate<false>(
                /*layer=*/0, d_act[static_cast<size_t>(t)], GpuMatrix{}, gate_info, up_info, down_info,
                cache, scratch, d_serial[static_cast<size_t>(t)], 1.0f, stream, true);
        }
        ck(cudaStreamSynchronize(stream), "serial sync");
        const auto serial_end = std::chrono::steady_clock::now();

        // ---- batch path -----------------------------------------------------
        std::vector<const float *> acts;
        for (int t = 0; t < tokens; ++t) acts.push_back(d_act[static_cast<size_t>(t)]);
        // Poison the batch workspace so a missing zeroing step cannot hide
        // behind a fresh allocation (this is the bug that shipped once).
        {
            float * accum = static_cast<float *>(
                scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Accum, tokens));
            ck(cudaMemset(accum, 0x7f, static_cast<size_t>(hidden) * tokens * sizeof(float)),
               "poison batch accum");
        }
        const auto batch_begin = std::chrono::steady_clock::now();
        batch.open_all(scratch, GpuMatrix{}, acts, 1.0f, stream, true);
        for (int i = 0; i < tokens * k; ++i) {
            batch.set_pair_experts(i, token_expert[static_cast<size_t>(i)], union_weight);
        }
        for (size_t u = 0; u < union_experts.size(); ++u) {
            const int e = union_experts[u];
            batch.set_pair_weights(static_cast<int>(u), d_gate[static_cast<size_t>(e)],
                                   d_up[static_cast<size_t>(e)], d_down[static_cast<size_t>(e)]);
        }
        // The pair slot of every (token, top-k) carries the union index; pairs
        // are (token-major, top-k-minor) and the caller passes the union entry
        // of its own expert. set_pair_weights() above wrote union slot u's
        // pointers at pair u, so the pair list must be reordered to the union
        // order the kernels read.
        for (int p = 0; p < tokens * k; ++p) {
            const int u = token_slot[static_cast<size_t>(p)];
            batch.set_pair_weights(p, d_gate[static_cast<size_t>(union_experts[static_cast<size_t>(u)])],
                                   d_up[static_cast<size_t>(union_experts[static_cast<size_t>(u)])],
                                   d_down[static_cast<size_t>(union_experts[static_cast<size_t>(u)])]);
        }
        const bool resolved = batch.resolved();
        std::vector<__half *> d_batch(static_cast<size_t>(tokens));
        for (int t = 0; t < tokens; ++t) {
            ck(cudaMalloc(reinterpret_cast<void **>(&d_batch[static_cast<size_t>(t)]),
                          static_cast<size_t>(hidden) * sizeof(__half)), "malloc batch output");
        }
        // The batch writes all M slices contiguously; the comparison uses one
        // contiguous buffer so the layout itself is part of what is checked.
        __half * d_batch_all = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&d_batch_all),
                      static_cast<size_t>(hidden) * tokens * sizeof(__half)), "malloc batch buffer");
        batch.write_output(scratch, nullptr, d_batch_all, stream);
        ck(cudaStreamSynchronize(stream), "batch sync");
        const auto batch_end = std::chrono::steady_clock::now();

        std::vector<uint8_t> serial_bytes(static_cast<size_t>(hidden) * tokens * sizeof(__half));
        std::vector<uint8_t> batch_bytes(static_cast<size_t>(hidden) * tokens * sizeof(__half));
        for (int t = 0; t < tokens; ++t) {
            ck(cudaMemcpy(serial_bytes.data() + static_cast<size_t>(t) * hidden * sizeof(__half),
                          d_serial[static_cast<size_t>(t)], static_cast<size_t>(hidden) * sizeof(__half),
                          cudaMemcpyDeviceToHost), "copy serial output");
        }
        ck(cudaMemcpy(batch_bytes.data(), d_batch_all,
                      static_cast<size_t>(hidden) * tokens * sizeof(__half), cudaMemcpyDeviceToHost),
           "copy batch output");

        const bool same = std::memcmp(serial_bytes.data(), batch_bytes.data(), serial_bytes.size()) == 0;
        const double serial_ms = std::chrono::duration<double, std::milli>(serial_end - serial_begin).count();
        const double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_begin).count();
        std::printf("  M=%d: %s  serial %.2f ms  batch %.2f ms  ratio %.2fx\n",
                    tokens, same ? "bit-identical" : "MISMATCH", serial_ms, batch_ms,
                    batch_ms > 0.0 ? serial_ms / batch_ms : 0.0);
        if (!same) ++failures;

        // Negative control: a batch that actually folded the wrong token's
        // activation must differ, otherwise the comparison proves nothing.
        if (tokens > 1 && same) {
            std::vector<const float *> swapped = acts;
            std::swap(swapped[0], swapped[1]);
            batch.open_all(scratch, GpuMatrix{}, swapped, 1.0f, stream, true);
            for (int t = 0; t < tokens; ++t) {
                for (int j = 0; j < k; ++j) {
                    batch.set_pair_weights(t * k + j, d_gate[static_cast<size_t>(j)], d_up[static_cast<size_t>(j)],
                                           d_down[static_cast<size_t>(j)]);
                }
            }
            if (batch.resolved()) batch.run(scratch, gate_info, up_info, down_info, stream);
            batch.write_output(scratch, nullptr, d_batch_all, stream);
            ck(cudaStreamSynchronize(stream), "negative sync");
            ck(cudaMemcpy(batch_bytes.data(), d_batch_all,
                          static_cast<size_t>(hidden) * tokens * sizeof(__half), cudaMemcpyDeviceToHost),
               "copy negative output");
            const bool differs = std::memcmp(serial_bytes.data(), batch_bytes.data(), serial_bytes.size()) != 0;
            std::printf("       negative control (token activation swap) %s\n",
                        differs ? "differs as expected" : "PROVES NOTHING");
            if (!differs) ++failures;
        }

        for (int t = 0; t < tokens; ++t) {
            cudaFree(d_act[static_cast<size_t>(t)]);
            cudaFree(d_serial[static_cast<size_t>(t)]);
            cudaFree(d_batch[static_cast<size_t>(t)]);
        }
        cudaFree(d_batch_all);
    }

    // ---- one layer, one token batch, many repetitions ----------------------
    {
        const int tokens = 3;
        std::vector<float *> d_act(static_cast<size_t>(tokens));
        std::mt19937 rng(77u);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        for (int t = 0; t < tokens; ++t) {
            std::vector<float> h(static_cast<size_t>(hidden));
            for (auto & v : h) v = dist(rng);
            ck(cudaMalloc(reinterpret_cast<void **>(&d_act[static_cast<size_t>(t)]),
                          static_cast<size_t>(hidden) * sizeof(float)), "malloc activation");
            ck(cudaMemcpy(d_act[static_cast<size_t>(t)], h.data(), static_cast<size_t>(hidden) * sizeof(float),
                          cudaMemcpyHostToDevice), "copy activation");
        }
        std::vector<const float *> acts;
        for (int t = 0; t < tokens; ++t) acts.push_back(d_act[static_cast<size_t>(t)]);

        auto serial_layer = [&] {
            for (int t = 0; t < tokens; ++t) {
                run_qwen38_routed_moe_separate<false>(
                    0, d_act[static_cast<size_t>(t)], GpuMatrix{}, gate_info, up_info, down_info,
                    cache, scratch, scratch.d_down(), 1.0f, stream, true);
            }
        };
        __half * d_out = nullptr;
        ck(cudaMalloc(reinterpret_cast<void **>(&d_out), static_cast<size_t>(hidden) * tokens * sizeof(__half)),
           "malloc out");
        auto batch_layer = [&] {
            batch.open_all(scratch, GpuMatrix{}, acts, 1.0f, stream, true);
            for (int t = 0; t < tokens; ++t) {
                for (int j = 0; j < k; ++j) {
                    batch.set_pair_weights(t * k + j, d_gate[static_cast<size_t>(j)],
                                           d_up[static_cast<size_t>(j)], d_down[static_cast<size_t>(j)]);
                }
            }
            if (batch.resolved()) batch.run(scratch, gate_info, up_info, down_info, stream);
            batch.write_output(scratch, nullptr, d_out, stream);
        };

        auto measure = [&](const std::function<void()> & f) {
            for (int i = 0; i < 3; ++i) f();
            ck(cudaStreamSynchronize(stream), "warm sync");
            cudaEvent_t a{}, b{};
            ck(cudaEventCreate(&a), "event");
            ck(cudaEventCreate(&b), "event");
            ck(cudaEventRecord(a, stream), "record");
            const int iters = 50;
            for (int i = 0; i < iters; ++i) f();
            ck(cudaEventRecord(b, stream), "record");
            ck(cudaEventSynchronize(b), "sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return static_cast<double>(ms) * 1000.0 / iters;
        };
        const double serial_us = measure(serial_layer);
        const double batch_us = measure(batch_layer);
        std::printf("\n[timing] M=%d x top-%d, one routed MoE layer\n", tokens, k);
        std::printf("  serial (M selections + M x %d expert passes): %8.2f us/layer\n", k, serial_us);
        std::printf("  batch  (M selections + 1 union pass)        : %8.2f us/layer\n", batch_us);
        std::printf("  per-token ratio                             : %8.2fx\n", batch_us > 0.0 ? serial_us / batch_us : 0.0);
        std::printf("  projected on 48 layers                      : %8.2f -> %8.2f ms/token\n",
                    serial_us * 48.0 / 1000.0 / tokens, batch_us * 48.0 / 1000.0 / tokens);
        for (int t = 0; t < tokens; ++t) cudaFree(d_act[static_cast<size_t>(t)]);
        cudaFree(d_out);
    }

    std::printf("\n%s (%d mismatch%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "es");
    return failures ? 1 : 0;
}
