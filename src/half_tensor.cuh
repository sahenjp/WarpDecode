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

struct DeviceHalfArray {
    __half * data = nullptr;
    size_t size = 0;

    DeviceHalfArray() = default;
    DeviceHalfArray(const DeviceHalfArray &) = delete;
    DeviceHalfArray & operator=(const DeviceHalfArray &) = delete;

    DeviceHalfArray(DeviceHalfArray && o) noexcept : data(o.data), size(o.size) {
        o.data = nullptr;
        o.size = 0;
    }

    DeviceHalfArray & operator=(DeviceHalfArray && o) noexcept {
        if (this != &o) {
            reset();
            data = o.data;
            size = o.size;
            o.data = nullptr;
            o.size = 0;
        }
        return *this;
    }

    ~DeviceHalfArray() { reset(); }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        size = 0;
    }
};

inline float bf16_to_float(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline DeviceHalfArray upload_half_array(const TensorInfo & t) {
    const size_t n = static_cast<size_t>(t.elements());
    std::vector<__half> host(n);

    if (t.type == GgmlType::F16) {
        std::memcpy(host.data(), t.data, n * sizeof(uint16_t));
    } else if (t.type == GgmlType::F32) {
        const float * src = reinterpret_cast<const float *>(t.data);
        for (size_t i = 0; i < n; ++i) host[i] = __float2half(src[i]);
    } else if (t.type == GgmlType::BF16) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < n; ++i) host[i] = __float2half(bf16_to_float(src[i]));
    } else {
        throw std::runtime_error("tensor cannot be uploaded as half array: " + t.name);
    }

    DeviceHalfArray out;
    out.size = n;
    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), n * sizeof(__half));
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc half array failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, host.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy half array failed: ") + cudaGetErrorString(c));
    }
    return out;
}

} // namespace wd
