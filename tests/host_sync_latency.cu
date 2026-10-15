// Task R: what does one host<->device round trip cost on this VM?
//
// The MoE region of every layer starts with q38_select_experts*(): a router
// GEMV, a single-thread top-k kernel, a 40-byte D2H of the selected ids and
// weights, and a cudaStreamSynchronize() on the decode stream. That is 48
// synchronisations per token, and the region does not shrink when the *kernels*
// or the *transfer* are removed, so the fixed cost of the pattern has to be
// measured on its own:
//
//   empty_sync   : cudaStreamSynchronize() on an idle non-blocking stream
//   d2h_sync     : cudaMemcpyAsync(D2H, 40 B) + sync   (the router pattern)
//   kernel_sync  : a 1-block kernel + sync
//   events       : 22 cudaEventRecord pairs on the stream + sync (what
//                  WARODECODE_PROFILE=1 adds per layer)
//
// Usage: host_sync_latency [iterations]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void tiny_kernel(int * p) { if (threadIdx.x == 0 && blockIdx.x == 0) *p = 1; }

__global__ void topk_like_kernel(const float * logits, int n, int * out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float best = -1e30f;
    int best_i = 0;
    for (int j = 0; j < n; ++j) {
        const float v = logits[j];
        if (v > best) { best = v; best_i = j; }
    }
    *out = best_i;
}

}  // namespace

int main(int argc, char ** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
    if (cudaSetDevice(0) != cudaSuccess) { std::fprintf(stderr, "no device\n"); return 1; }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;

    int * h_ids = nullptr;
    int * d_ids = nullptr;
    int * d_scratch = nullptr;
    float * d_logits = nullptr;
    cudaMallocHost(&h_ids, 64);
    cudaMalloc(&d_ids, 64);
    cudaMalloc(&d_scratch, 64);
    cudaMalloc(&d_logits, 512 * sizeof(float));
    cudaMemset(d_logits, 0, 512 * sizeof(float));

    std::vector<cudaEvent_t> events(44);
    for (cudaEvent_t & e : events) cudaEventCreate(&e);

    // Warm up: first syncs pay one-off driver work.
    for (int i = 0; i < 50; ++i) {
        cudaMemcpyAsync(h_ids, d_ids, 40, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    double t0 = now_ms();
    for (int i = 0; i < iterations; ++i) cudaStreamSynchronize(stream);
    double t1 = now_ms();
    std::printf("empty_sync_us=%.2f\n", (t1 - t0) * 1000.0 / iterations);

    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpyAsync(h_ids, d_ids, 40, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    t1 = now_ms();
    const double d2h_sync_us = (t1 - t0) * 1000.0 / iterations;
    std::printf("d2h_sync_us=%.2f\n", d2h_sync_us);

    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        tiny_kernel<<<1, 1, 0, stream>>>(d_scratch);
        cudaStreamSynchronize(stream);
    }
    t1 = now_ms();
    std::printf("kernel_sync_us=%.2f\n", (t1 - t0) * 1000.0 / iterations);

    // The router's top-k is a single-thread scan over the 512 logits.
    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        topk_like_kernel<<<1, 1, 0, stream>>>(d_logits, 512, d_scratch);
        cudaStreamSynchronize(stream);
    }
    t1 = now_ms();
    std::printf("topk512_sync_us=%.2f\n", (t1 - t0) * 1000.0 / iterations);

    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        for (cudaEvent_t e : events) cudaEventRecord(e, stream);
        cudaStreamSynchronize(stream);
    }
    t1 = now_ms();
    std::printf("events44_sync_us=%.2f\n", (t1 - t0) * 1000.0 / iterations);

    // No sync at all: the same 44 event records, measured by wall time only.
    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        for (cudaEvent_t e : events) cudaEventRecord(e, stream);
    }
    cudaStreamSynchronize(stream);
    t1 = now_ms();
    std::printf("events44_no_sync_us=%.2f\n", (t1 - t0) * 1000.0 / iterations);

    // The engine runs this pattern once per MoE layer, i.e. 48 times per token.
    std::printf("per_token_48x_d2h_sync_ms=%.2f\n", 48.0 * d2h_sync_us / 1000.0);
    return 0;
}
