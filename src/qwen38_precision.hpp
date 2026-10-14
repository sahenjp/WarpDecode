#pragma once

#include <cmath>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define Q38_HOST_DEVICE __host__ __device__
#else
#define Q38_HOST_DEVICE
#endif

namespace wd {

Q38_HOST_DEVICE inline float q38_mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float result = a * b;
    return result;
#endif
}

Q38_HOST_DEVICE inline float q38_add_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    volatile float result = a + b;
    return result;
#endif
}

Q38_HOST_DEVICE inline float q38_moe_branch_f32(float routed, float shared, float gate) {
    return q38_add_rn(routed, q38_mul_rn(shared, gate));
}

Q38_HOST_DEVICE inline float q38_residual_add_f32(float residual, float branch, float weight) {
    return q38_add_rn(residual, q38_mul_rn(branch, weight));
}

Q38_HOST_DEVICE inline float q38_round_half(float value) {
#if defined(__CUDA_ARCH__)
    return __half2float(__float2half(value));
#else
    if (!std::isfinite(value) || value == 0.0f) return value;
    const float magnitude = std::fabs(value);
    if (magnitude >= 65520.0f) return std::copysign(INFINITY, value);
    if (magnitude < std::ldexp(1.0f, -14)) {
        return std::copysign(
            std::ldexp(std::nearbyint(std::ldexp(magnitude, 24)), -24), value);
    }
    const int exponent = std::ilogb(magnitude);
    const float quantum = std::ldexp(1.0f, exponent - 10);
    return std::copysign(std::nearbyint(magnitude / quantum) * quantum, value);
#endif
}

} // namespace wd

#undef Q38_HOST_DEVICE
