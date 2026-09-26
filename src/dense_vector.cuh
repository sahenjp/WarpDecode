#pragma once

#include "gguf.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

struct DeviceHalfVector {
    __half * data = nullptr;
    int size = 0;

    DeviceHalfVector() = default;
    DeviceHalfVector(const DeviceHalfVector &) = delete;
    DeviceHalfVector & operator=(const DeviceHalfVector &) = delete;

    DeviceHalfVector(DeviceHalfVector && o) noexcept : data(o.data), size(o.size) {
        o.data = nullptr;
        o.size = 0;
    }

    DeviceHalfVector & operator=(DeviceHalfVector && o) noexcept {
        if (this != &o) {
            reset();
            data = o.data;
            size = o.size;
            o.data = nullptr;
            o.size = 0;
        }
        return *this;
    }

    ~DeviceHalfVector() { reset(); }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        size = 0;
    }
};

inline float bf16_bits_to_float(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline DeviceHalfVector upload_half_vector(const TensorInfo & t) {
    if (t.dims.size() != 1) {
        throw std::runtime_error("vector tensor must be 1D: " + t.name);
    }

    const size_t n = static_cast<size_t>(t.dims[0]);
    std::vector<__half> host(n);

    if (t.type == GgmlType::F16) {
        std::memcpy(host.data(), t.data, n * sizeof(uint16_t));
    } else if (t.type == GgmlType::F32) {
        const float * src = reinterpret_cast<const float *>(t.data);
        for (size_t i = 0; i < n; ++i) host[i] = __float2half(src[i]);
    } else if (t.type == GgmlType::BF16) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < n; ++i) host[i] = __float2half(bf16_bits_to_float(src[i]));
    } else {
        throw std::runtime_error("unsupported vector tensor type: " + t.name);
    }

    DeviceHalfVector out;
    out.size = static_cast<int>(n);
    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), n * sizeof(__half));
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc vector failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, host.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy vector failed: ") + cudaGetErrorString(c));
    }
    return out;
}

__global__ void half_dot_kernel(
        const __half * __restrict__ a,
        const __half * __restrict__ b,
        float * __restrict__ out,
        int n) {
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum = fmaf(__half2float(a[i]), __half2float(b[i]), sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = smem[0];
}

__global__ void sigmoid_scalar_kernel(float * x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float v = *x;
        *x = 1.0f / (1.0f + expf(-v));
    }
}

} // namespace wd
