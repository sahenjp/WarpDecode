#pragma once

#include "gguf.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

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

// F32 weights are kept in F32 on the device: ggml runs non-quantised tensors
// with vec_dot_type F32, so rounding them to F16 (as upload_half_array does)
// diverges from the CPU reference.
struct DeviceFloatArray {
    float * data = nullptr;
    size_t size = 0;

    DeviceFloatArray() = default;
    DeviceFloatArray(const DeviceFloatArray &) = delete;
    DeviceFloatArray & operator=(const DeviceFloatArray &) = delete;

    DeviceFloatArray(DeviceFloatArray && o) noexcept : data(o.data), size(o.size) {
        o.data = nullptr;
        o.size = 0;
    }

    DeviceFloatArray & operator=(DeviceFloatArray && o) noexcept {
        if (this != &o) {
            reset();
            data = o.data;
            size = o.size;
            o.data = nullptr;
            o.size = 0;
        }
        return *this;
    }

    ~DeviceFloatArray() { reset(); }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        size = 0;
    }
};

inline DeviceFloatArray upload_float_array(const TensorInfo & t) {
    if (t.type != GgmlType::F32) {
        throw std::runtime_error("tensor cannot be uploaded as F32 array: " + t.name);
    }
    const size_t n = static_cast<size_t>(t.elements());
    DeviceFloatArray out;
    out.size = n;
    const cudaError_t a = cudaMalloc(reinterpret_cast<void **>(&out.data), n * sizeof(float));
    if (a != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc float array failed: ") + cudaGetErrorString(a));
    }
    const cudaError_t c = cudaMemcpy(out.data, t.data, n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        out.reset();
        throw std::runtime_error(std::string("cudaMemcpy float array failed: ") + cudaGetErrorString(c));
    }
    return out;
}

// ggml's CPU rms_norm / attention scaling compute 1.0f/sqrtf(v) with libm,
// i.e. an IEEE-correctly-rounded square root followed by a correctly-rounded
// division. CUDA's rsqrtf is only accurate to 2 ulp (and --use_fast_math turns
// the division into an approximation), so the same expression differs by ~1e-7
// there - enough to flip Q8_0 activation codes downstream. Every such site uses
// this helper instead.
__device__ __forceinline__ float q38_exact_rsqrt(float v) {
    return __fdiv_rn(1.0f, __fsqrt_rn(v));
}

// Host-side twin for the cases whose scale is computed on the CPU before the
// kernel launches (attention scaling). Same two correctly-rounded float steps
// the CPU reference performs: sqrtf() then a float division.
inline float q38_exact_rsqrt_host(float v) {
    const float s = std::sqrt(v);
    return 1.0f / s;
}

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
