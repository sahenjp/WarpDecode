#pragma once

// Device-and-host compilable transcription of the glibc float sine/cosine
// routines (sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c, s_sincosf.h,
// sincosf_poly.h, s_sincosf_data.c, glibc 2.43).
//
// The CPU oracle calls cosf()/sinf() through ggml's rope_yarn(), so the kernel
// needs those two functions bit for bit. They are not correctly rounded, but
// they are deterministic, and the measured difference against
// (float)sin((double)x) is ~1% of the arguments in this model's theta range,
// which is exactly the drift this file removes.
//
// Included by src/qwen38_rope.cuh and by tests/qwen38_rope_host_test.cpp; the
// host build compiles it with -ffp-contract=off so every expression below is a
// single IEEE operation, matching nvcc's default contraction behaviour.

#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

namespace wd {
namespace q38glibc {

struct sincos_t {
    double sign[4];          // sign of sine in quadrants 0..3
    double hpi_inv;          // 2/pi ( * 2^24 for the non-intrinsic reduction)
    double hpi;              // pi/2
    double c0, c1, c2, c3, c4;
    double s1, s2, s3;
};

// Device copies of the two tables used below. They are filled from the static
// host tables (q38_glibc_fill_device_tables) instead of being written out a
// second time, so the GPU and the host cannot drift apart. They have to be
// defined before sincos_table()/inv_pio4(), which read them in device code.
#if defined(__CUDACC__)
__constant__ sincos_t q38_glibc_tbl_dev[2];
__constant__ uint32_t q38_glibc_inv_pio4_dev[24];
#endif

// The constants and polynomials for sine and cosine. The 2nd entry computes
// -cos(x) rather than cos(x) to get the negation for free.
__host__ __device__ __forceinline__ const sincos_t * sincos_table(int second) {
#if defined(__CUDA_ARCH__)
    return &q38_glibc_tbl_dev[second];
#else
    static const sincos_t tbl[2] = {
        {
            { 1.0, -1.0, -1.0, 1.0 },
            0x1.45F306DC9C883p+23,
            0x1.921FB54442D18p0,
            0x1p0,
            -0x1.ffffffd0c621cp-2,
            0x1.55553e1068f19p-5,
            -0x1.6c087e89a359dp-10,
            0x1.99343027bf8c3p-16,
            -0x1.555545995a603p-3,
            0x1.1107605230bc4p-7,
            -0x1.994eb3774cf24p-13
        },
        {
            { 1.0, -1.0, -1.0, 1.0 },
            0x1.45F306DC9C883p+23,
            0x1.921FB54442D18p0,
            -0x1p0,
            0x1.ffffffd0c621cp-2,
            -0x1.55553e1068f19p-5,
            0x1.6c087e89a359dp-10,
            -0x1.99343027bf8c3p-16,
            -0x1.555545995a603p-3,
            0x1.1107605230bc4p-7,
            -0x1.994eb3774cf24p-13
        }
    };
    return &tbl[second];
#endif
}

// Table with 4/pi to 192 bit precision. To avoid unaligned accesses only 8 new
// bits are added per entry, making the table 4 times larger.
__host__ __device__ __forceinline__ const uint32_t * inv_pio4(void) {
#if defined(__CUDA_ARCH__)
    return q38_glibc_inv_pio4_dev;
#else
    static const uint32_t tbl[24] = {
        0xa2u,       0xa2f9u,     0xa2f983u,   0xa2f9836eu,
        0xf9836e4eu, 0x836e4e44u, 0x6e4e4415u, 0x4e441529u,
        0x441529fcu, 0x1529fc27u, 0x29fc2757u, 0xfc2757d1u,
        0x2757d1f5u, 0x57d1f534u, 0xd1f534ddu, 0xf534ddc0u,
        0x34ddc0dbu, 0xddc0db62u, 0xc0db6295u, 0xdb629599u,
        0x6295993cu, 0x95993c43u, 0x993c4390u, 0x3c439041u
    };
    return tbl;
#endif
}

// 2*pi * 2^-64 and pi/4.
const double q38_pi63 = 0x1.921FB54442D18p-62;
const float  q38_pio4 = 0x1.921FB6p-1f;

// ---------------------------------------------------------------------------
// Fills the device copies declared above from the static host tables, so the
// two cannot drift apart; the host build simply does not declare them.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
__host__ inline void q38_glibc_fill_device_tables() {
    if (cudaMemcpyToSymbol(q38_glibc_tbl_dev, sincos_table(0), 2 * sizeof(sincos_t)) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    (void)cudaMemcpyToSymbol(q38_glibc_inv_pio4_dev, inv_pio4(), 24 * sizeof(uint32_t));
}
#endif

__host__ __device__ __forceinline__ uint32_t asuint32(float f) {
    uint32_t u;
#if defined(__CUDA_ARCH__)
    u = __float_as_uint(f);
#else
    std::memcpy(&u, &f, sizeof(u));
#endif
    return u;
}

__host__ __device__ __forceinline__ float asfloat32(uint32_t u) {
    float f;
#if defined(__CUDA_ARCH__)
    f = __uint_as_float(u);
#else
    std::memcpy(&f, &u, sizeof(f));
#endif
    return f;
}

// Top 12 bits of the float representation with the sign bit cleared.
__host__ __device__ __forceinline__ uint32_t abstop12(float x) {
    return (asuint32(x) >> 20) & 0x7ffu;
}

// Fast range reduction using single multiply-subtract. Returns the modulo of X
// as a value between -pi/4 and pi/4 and stores the quadrant in NP. PI/2 as a
// double is accurate to 55 bits and the worst-case cancellation happens at
// 6*pi/4, so the result is accurate for |X| <= 120.
__host__ __device__ __forceinline__ double reduce_fast(double x, const sincos_t * p, int * np) {
    const double r = x * p->hpi_inv;
    const int n = (static_cast<int>(static_cast<int32_t>(r)) + 0x800000) >> 24;
    *np = n;
    return x - n * p->hpi;
}

// Reduce the range of XI to a multiple of pi/2 using fast integer arithmetic.
// XI is a reinterpreted float and must be >= 2.0f (the sign bit is ignored).
// A 32x96 -> 128 bit multiply computes the exact 2.62-bit fixed-point modulo.
// The 64 bit products below use at most 55 significant bits, so they are bit
// exact on the CPU as well as on the GPU (no 64x64 -> 128 multiply needed).
__host__ __device__ __forceinline__ double reduce_large(uint32_t xi, int * np) {
    const uint32_t * arr = &inv_pio4()[(xi >> 26) & 15];
    const int shift = static_cast<int>((xi >> 23) & 7);
    uint64_t n, res0, res1, res2;

    xi = (xi & 0xffffffu) | 0x800000u;
    xi <<= shift;

    res0 = static_cast<uint64_t>(xi) * arr[0];
    res1 = static_cast<uint64_t>(xi) * arr[4];
    res2 = static_cast<uint64_t>(xi) * arr[8];
    res0 = (res2 >> 32) | (res0 << 32);
    res0 += res1;

    n = (res0 + (1ull << 61)) >> 62;
    res0 -= n << 62;
    const double x = static_cast<double>(static_cast<int64_t>(res0));
    *np = static_cast<int>(n);
    return x * q38_pi63;
}

// Return the sine of inputs X and X2 (X squared) using the polynomial P. N is
// the quadrant, and if odd the cosine polynomial is used. This is the routine
// the single-output sinf()/cosf() entries evaluate.
__host__ __device__ __forceinline__ float sinf_poly(
        double x, double x2, const sincos_t * p, int n) {
    double x3, x4, x6, x7, s, c, c1, c2, s1;

    if ((n & 1) == 0) {
        x3 = x * x2;
        s1 = p->s2 + x2 * p->s3;
        x7 = x3 * x2;
        s = x + x3 * p->s1;
        return static_cast<float>(s + x7 * s1);
    }
    x4 = x2 * x2;
    c2 = p->c3 + x2 * p->c4;
    c1 = p->c0 + x2 * p->c1;
    x6 = x4 * x2;
    c = c1 + x4 * p->c2;
    return static_cast<float>(c + x6 * c2);
}

// Compute sine and cosine of inputs X and X2 (X squared) using the polynomial P
// and store the results in SINP and COSP. This is the sincosf() entry point; the
// kernel calls the two single-output routines below instead, because the CPU
// oracle calls cosf() and sinf() separately and their polynomials differ.
__host__ __device__ __forceinline__ void sincosf_poly(
        double x, double x2, const sincos_t * p, int n, float * sinp, float * cosp) {
    double x3, x4, x5, x6, s, c, c1, c2, s1;

    x4 = x2 * x2;
    x3 = x2 * x;
    c2 = p->c3 + x2 * p->c4;
    s1 = p->s2 + x2 * p->s3;

    float * tmp = (n & 1 ? cosp : sinp);
    cosp = (n & 1 ? sinp : cosp);
    sinp = tmp;

    c1 = p->c0 + x2 * p->c1;
    x5 = x3 * x2;
    x6 = x4 * x2;

    s = x + x3 * p->s1;
    c = c1 + x4 * p->c2;

    *sinp = static_cast<float>(s + x5 * s1);
    *cosp = static_cast<float>(c + x6 * c2);
}

// glibc sinf(). This is the function ggml's rope_yarn() calls on the CPU.
__host__ __device__ __forceinline__ float sinf(float y) {
    double x = static_cast<double>(y);
    double s;
    int n;
    const sincos_t * p = sincos_table(0);

    if (abstop12(y) < abstop12(q38_pio4)) {
        const double x2 = x * x;
        if (abstop12(y) < abstop12(asfloat32(0x39800000u /* 0x1p-12f */))) {
            return y;
        }
        return sinf_poly(x, x2, p, 0);
    } else if (abstop12(y) < abstop12(120.0f)) {
        x = reduce_fast(x, p, &n);
        s = p->sign[n & 3];
        if (n & 2) p = sincos_table(1);
        return sinf_poly(x * s, x * x, p, n);
    } else if (abstop12(y) < abstop12(INFINITY)) {
        const uint32_t xi = asuint32(y);
        const int sign = static_cast<int>(xi >> 31);
        x = reduce_large(xi, &n);
        s = p->sign[(n + sign) & 3];
        if ((n + sign) & 2) p = sincos_table(1);
        return sinf_poly(x * s, x * x, p, n);
    }
    return y - y;
}

// glibc cosf(). This is the other half of ggml's rope_yarn() on the CPU.
__host__ __device__ __forceinline__ float cosf(float y) {
    double x = static_cast<double>(y);
    double s;
    int n;
    const sincos_t * p = sincos_table(0);

    if (abstop12(y) < abstop12(q38_pio4)) {
        const double x2 = x * x;
        if (abstop12(y) < abstop12(asfloat32(0x39800000u /* 0x1p-12f */))) {
            return 1.0f;
        }
        return sinf_poly(x, x2, p, 1);
    } else if (abstop12(y) < abstop12(120.0f)) {
        x = reduce_fast(x, p, &n);
        s = p->sign[n & 3];
        if (n & 2) p = sincos_table(1);
        return sinf_poly(x * s, x * x, p, n ^ 1);
    } else if (abstop12(y) < abstop12(INFINITY)) {
        const uint32_t xi = asuint32(y);
        const int sign = static_cast<int>(xi >> 31);
        x = reduce_large(xi, &n);
        s = p->sign[(n + sign) & 3];
        if ((n + sign) & 2) p = sincos_table(1);
        return sinf_poly(x * s, x * x, p, n ^ 1);
    }
    return y - y;
}

} // namespace q38glibc
} // namespace wd
