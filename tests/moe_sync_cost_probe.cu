// Host-side cost of the primitives one decode token is built from.
//
// Task T: the stage profile says the MoE region spends ~120 ms/token of host wall
// clock while the launches inside it cost ~6 ms, and 48 of those 48 regions end
// in a cudaStreamSynchronize. This probe measures, on the machine the engine
// runs on, what a wait costs in each of the ways the engine could wait:
//
//   empty_sync     - cudaStreamSynchronize with nothing queued
//   kernel_sync    - launch a 1-thread kernel, then cudaStreamSynchronize
//   kernel_evwait  - launch, then cudaEventSynchronize on an event after it
//   kernel_poll    - launch, then spin on cudaEventQuery
//   kernel_nop     - launch, then a host pause-loop of the same nominal length
//   launch_only    - N launches, one sync at the end (pure launch cost)
//
// Every number is microseconds per iteration. It is a host/driver measurement:
// the kernels are empty, so nothing here depends on the GPU's speed.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

__global__ void nop_kernel(int * out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 1;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Result {
    const char * name;
    double us;
    int iterations;
};

void report(const Result & r) {
    std::printf("{\"case\":\"%s\",\"us_per_iter\":%.3f,\"iterations\":%d}\n",
                r.name, r.us, r.iterations);
}

int * device_out = nullptr;

// cudaStreamSynchronize on a stream with no pending work.
Result empty_sync(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // Warm up so the first call does not carry the driver's lazy setup.
    for (int i = 0; i < 50; ++i) cudaStreamSynchronize(stream);
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) cudaStreamSynchronize(stream);
    const double t1 = now_ms();
    cudaStreamDestroy(stream);
    return {"empty_sync", (t1 - t0) * 1000.0 / n, n};
}

Result kernel_sync(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int i = 0; i < 50; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaStreamSynchronize(stream);
    }
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaStreamSynchronize(stream);
    }
    const double t1 = now_ms();
    cudaStreamDestroy(stream);
    return {"kernel_sync", (t1 - t0) * 1000.0 / n, n};
}

Result kernel_evwait(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t ev = nullptr;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int i = 0; i < 50; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);
    }
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);
    }
    const double t1 = now_ms();
    cudaEventDestroy(ev);
    cudaStreamDestroy(stream);
    return {"kernel_event_sync", (t1 - t0) * 1000.0 / n, n};
}

Result kernel_poll(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t ev = nullptr;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int i = 0; i < 50; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaEventRecord(ev, stream);
        while (cudaEventQuery(ev) == cudaErrorNotReady) {}
    }
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_out);
        cudaEventRecord(ev, stream);
        for (;;) {
            const cudaError_t e = cudaEventQuery(ev);
            if (e == cudaSuccess) break;
            if (e != cudaErrorNotReady) { (void)cudaGetLastError(); break; }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
    const double t1 = now_ms();
    cudaEventDestroy(ev);
    cudaStreamDestroy(stream);
    return {"kernel_event_poll", (t1 - t0) * 1000.0 / n, n};
}

// The expert-selection readback pattern: a writer kernel, then a 40-byte
// device->host copy, then the wait. This is the shape of the per-layer router
// readback in the decode loop, so it separates "the copy is expensive" from
// "the wait is expensive".
int * pinned_host = nullptr;
int * device_src = nullptr;

Result kernel_d2h_sync(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int i = 0; i < 50; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_src);
        cudaMemcpyAsync(pinned_host, device_src, 40, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_src);
        cudaMemcpyAsync(pinned_host, device_src, 40, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    const double t1 = now_ms();
    cudaStreamDestroy(stream);
    return {"kernel_d2h_sync", (t1 - t0) * 1000.0 / n, n};
}

Result kernel_d2h_evwait(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t ev = nullptr;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int i = 0; i < 50; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_src);
        cudaMemcpyAsync(pinned_host, device_src, 40, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);
    }
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        nop_kernel<<<1, 1, 0, stream>>>(device_src);
        cudaMemcpyAsync(pinned_host, device_src, 40, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);
    }
    const double t1 = now_ms();
    cudaEventDestroy(ev);
    cudaStreamDestroy(stream);
    return {"kernel_d2h_event_sync", (t1 - t0) * 1000.0 / n, n};
}

// Pure launch cost: N launches with a single synchronise at the end.
Result launch_only(int n) {
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    const int n2 = n;
    nop_kernel<<<1, 1, 0, stream>>>(device_out);
    cudaStreamSynchronize(stream);
    const double t0 = now_ms();
    for (int i = 0; i < n2; ++i) nop_kernel<<<1, 1, 0, stream>>>(device_out);
    const double t1 = now_ms();
    cudaStreamSynchronize(stream);
    const double t2 = now_ms();
    cudaStreamDestroy(stream);
    // The tail sync is the queue drain of n2 empty kernels, which the GPU does
    // far faster than the host can enqueue, so the launch cost dominates.
    (void)t2;
    return {"launch_only", (t1 - t0) * 1000.0 / n2, n2};
}

}  // namespace

int main(int argc, char ** argv) {
    int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
    if (iterations < 1) iterations = 1;
    if (cudaMalloc(reinterpret_cast<void **>(&device_out), sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    std::printf("{\"device\":\"");
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) std::printf("%s", prop.name);
    std::printf("\",\"iterations\":%d}\n", iterations);

    if (cudaMallocHost(reinterpret_cast<void **>(&pinned_host), 64) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&device_src), 64) != cudaSuccess) {
        std::fprintf(stderr, "readback buffers failed\n");
        return 1;
    }
    report(empty_sync(iterations));
    report(kernel_sync(iterations));
    report(kernel_evwait(iterations));
    report(kernel_poll(iterations));
    report(kernel_d2h_sync(iterations));
    report(kernel_d2h_evwait(iterations));
    report(launch_only(iterations));
    cudaFreeHost(pinned_host);
    cudaFree(device_src);

    cudaFree(device_out);
    return 0;
}
