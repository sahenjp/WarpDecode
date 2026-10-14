#pragma once

// ---------------------------------------------------------------------------
// Task N: CPU-side MoE for the warp engine (batch = 1 decode).
//
// llama.cpp wins the VRAM-matched A/B because it keeps the MoE expert weights in
// the mmap and multiplies them with the CPU instead of copying them to the GPU
// on every token (docs/LLAMACPP_AB.md). This header is the CPU half of that
// strategy: AVX2 quantised GEMV over the GGUF mapping, the same activation
// quantisers the CUDA path uses, and a thread pool that runs the ten selected
// experts of one layer (and optionally the shared expert).
//
// The arithmetic is not approximate. Every kernel below is a direct translation
// of the AVX2 branch of the pinned oracle (ggml-org/llama.cpp
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11, ggml/src/ggml-cpu/arch/x86/quants.c)
// and of the CUDA DP4A kernels this project already proved bit-identical to it
// (src/lowbit.cuh, src/qwen38_hc.cuh). Two properties are load-bearing:
//
//   * the integer side is exact. The IQ2_S/IQ3_S codebook magnitudes are <= 43,
//     the IQ4_NL codebook is <= 127 and the activations are 7-bit, so no partial
//     product pair can saturate int16 and the whole per-superblock integer sum
//     fits int32 with room to spare. _mm256_maddubs_epi16 + _mm256_madd_epi16
//     therefore equal a plain per-element integer dot product.
//   * the float side keeps ggml's *order*: one FMA per 256-value superblock for
//     IQ2_S/IQ3_S, the even/odd block split plus the closing hsum_float_8 tree
//     for IQ4_NL, and one FMA per 32-value block for Q8_0. Nothing is
//     reassociated and nothing is contracted outside those, so
//     tests/qwen38_cpu_moe_host_test.cpp can assert bit equality against an
//     independent scalar reconstruction of the same lane choreography.
//
// The router (softmax/top-k), the activation packing they consume and every
// non-expert block stay on the GPU: only the expert projections move to the CPU,
// which is exactly where this differs from llama.cpp. Attention, HC mix,
// GDN/QSA and the head keep their optimised CUDA path.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace wd {
namespace cpu_moe {

// ---------------------------------------------------------------------------
// ggml block geometries (ggml-common.h at the pinned commit).
// ---------------------------------------------------------------------------
constexpr int kQK_K = 256;   // IQ2_S / IQ3_S / Q8_K superblock
constexpr int kQK4_NL = 32;  // IQ4_NL block
constexpr int kQK8_0 = 32;   // Q8_0 block

constexpr size_t kIq2SBlockBytes = 82;   //   2 + 64 qs + 8 qh + 8 scales
constexpr size_t kIq3SBlockBytes = 110;  //   2 + 64 qs + 8 qh + 32 signs + 4 scales
constexpr size_t kIq4NlBlockBytes = 18;  //   2 + 16 nibbles
constexpr size_t kQ8_0BlockBytes = 34;   //   2 + 32 quants

// ---------------------------------------------------------------------------
// F16 <-> F32 without a libm dependency. GGUF stores every block scale as a
// half; the conversion below is what __half2float() does on sm_80 and what
// _Float16 -> float does in the host test. F32 -> F16 is round-to-nearest-even,
// the rounding ggml applies when it writes a block scale.
// ---------------------------------------------------------------------------
inline float f16_to_f32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int shift = 0;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++shift;
            }
            m &= 0x3ffu;
            // The leading 1 sits at bit 10 after \p shift + 1 steps, so the
            // normalised value is 1.m * 2^(-14 - shift) and the F32 exponent
            // field is 127 - 14 - shift (the previous 127 - 15 - shift halved
            // every subnormal before it was rounded into the payload).
            out = sign | (static_cast<uint32_t>(127 - 14 - shift) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exp = static_cast<int>((bits >> 23) & 0xffu) - 127;
    const uint32_t mant = bits & 0x7fffffu;
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp < -24) return static_cast<uint16_t>(sign);
    if (exp < -14) {
        // The implicit 1 of the normalised mantissa sits at bit 23 + (-exp - 14)
        // and the 10-bit subnormal result starts at bit 13 + (-exp - 14).
        const uint32_t shift = static_cast<uint32_t>(13 + (-exp - 14));
        const uint32_t full = mant | 0x800000u;
        uint32_t rounded = full >> shift;
        const uint32_t rest = (shift >= 32) ? full : (full & ((1u << shift) - 1u));
        const uint32_t halfway = (shift >= 32) ? 0x80000000u : (1u << (shift - 1));
        if (rest > halfway || (rest == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mant >> 13;
    const uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x400u) {
            return static_cast<uint16_t>(sign | static_cast<uint32_t>((exp + 1 + 15) << 10));
        }
    }
    return static_cast<uint16_t>(sign | static_cast<uint32_t>((exp + 15) << 10) | rounded);
}

// Single-rounded multiply / add the compiler must not contract into an FMA.
// ggml's MoE accumulation is mul-then-add (the routing weight is applied with
// ggml_mul and the running sum with ggml_add), so it has two roundings; the CUDA
// twin q38_weighted_accum_f32_kernel spells that out with __fmul_rn + __fadd_rn
// for the same reason. An FMA here would move the result by one ulp per expert.
inline float mul_rn(float a, float b) {
#if defined(__AVX2__) || defined(__SSE__)
    return _mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
    volatile float r = a * b;
    return r;
#endif
}

inline float add_rn(float a, float b) {
#if defined(__AVX2__) || defined(__SSE__)
    return _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
    volatile float r = a + b;
    return r;
#endif
}

inline float block_half(const uint8_t * p) {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return f16_to_f32(bits);
}

inline int32_t load_i32(const uint8_t * p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// ---------------------------------------------------------------------------
// Codebooks. Byte-for-byte copies of ggml's iq2s_grid / iq3s_grid, generated
// from the tables the CUDA kernels already use (src/iq2s_grid.cuh,
// src/iq3s_grid.cuh) by the same script that writes the .inc; the host check
// compares the values against the originals again.
// ---------------------------------------------------------------------------
extern const uint64_t kIq2sGrid[1024];
extern const uint32_t kIq3sGrid[512];

#include "qwen38_cpu_moe_grid.inc"

// ggml kvalues_iq4nl. -127 stays -127 after the int8 round trip, exactly like
// the AVX2 kernel's _mm256_cvtepi8_epi32 of the same table.
inline const int8_t * iq4_nl_lut() {
    static const int8_t lut[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return lut;
}

// ---------------------------------------------------------------------------
// Activation quantisers.
//
// quantize_q8_0 mirrors q38_hc_q8_0_quantize_f32_exact_kernel (and the AVX2
// branch of ggml's quantize_row_q8_0): amax over the block, d = fp16(amax/127),
// q = round_to_nearest_even(value * (127 / amax)) saturated to [-128, 127].
// The output is the ggml AoS block layout: the CUDA kernel keeps quants and
// scales in separate arrays, this one writes the block the row dot reads.
//
// quantize_q8_K mirrors q8_k_quantize_f32_kernel, which mirrors
// quantize_row_q8_K: the signed element of largest magnitude picks the scale
// (the kernels reduce |x| instead, and the sign flip cancels out),
// iscale = -127/max, d = 1/iscale, bsums[j] = sum of qs[16j .. 16j+15].
// ---------------------------------------------------------------------------
struct alignas(32) q8_K {
    float d;
    int8_t qs[kQK_K];
    int16_t bsums[16];
};

struct alignas(2) q8_0 {
    uint16_t d;
    int8_t qs[kQK8_0];
};

inline void quantize_q8_0(const float * x, q8_0 * out, int nblocks) {
    for (int b = 0; b < nblocks; ++b) {
        const float * src = x + static_cast<size_t>(b) * kQK8_0;
        q8_0 & blk = out[b];
        float amax = 0.0f;
        for (int i = 0; i < kQK8_0; ++i) amax = std::fmax(amax, std::fabs(src[i]));
        blk.d = f32_to_f16_bits(amax / 127.0f);
        if (amax == 0.0f) {
            for (int i = 0; i < kQK8_0; ++i) blk.qs[i] = 0;
            continue;
        }
        const float inverse = 127.0f / amax;
        for (int i = 0; i < kQK8_0; ++i) {
            const int q = static_cast<int>(std::nearbyint(src[i] * inverse));
            blk.qs[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    }
}

inline void quantize_q8_K(const float * x, q8_K * out, int nblocks) {
    for (int b = 0; b < nblocks; ++b) {
        const float * src = x + static_cast<size_t>(b) * kQK_K;
        q8_K & blk = out[b];
        float amax = 0.0f;
        for (int i = 0; i < kQK_K; ++i) amax = std::fmax(amax, std::fabs(src[i]));
        if (amax == 0.0f) {
            blk.d = 0.0f;
            for (int i = 0; i < kQK_K; ++i) blk.qs[i] = 0;
            for (int i = 0; i < 16; ++i) blk.bsums[i] = 0;
            continue;
        }
        const float iscale = -127.0f / amax;
        for (int i = 0; i < kQK_K; ++i) {
            const int q = static_cast<int>(std::nearbyint(src[i] * iscale));
            blk.qs[i] = static_cast<int8_t>(std::min(127, q));
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += static_cast<int>(blk.qs[j * 16 + i]);
            blk.bsums[j] = static_cast<int16_t>(sum);
        }
        blk.d = 1.0f / iscale;
    }
}

// ---------------------------------------------------------------------------
// Scalar row dot products: the independent bit-reference the AVX2 kernels are
// checked against. Each one reconstructs ggml's lane choreography exactly - the
// eight 4-value chunk lanes of a 32-value block, the block parity of IQ4_NL and
// the closing hsum_float_8 tree - but sums the integer products one at a time.
// The AVX2 twin and this one must agree bit for bit; the only difference is the
// order of exact integer additions.
//
// Scale conventions at the pinned commit (ggml-common.h block layouts and the
// AVX2 branch of ggml-cpu/arch/x86/quants.c):
//
//   IQ2_S  block b has its own scale byte. Values 0..15 of that block (chunks 0
//          to 3) use 1 + 2*(scales[b] & 0x0f), values 16..31 (chunks 4..7) use
//          1 + 2*(scales[b] >> 4). The sign byte of grid group g is
//          signs[b*4 + g], bit 4*(chunk & 1) + j.
//   IQ3_S  one scale byte per two 32-value blocks: block 2t takes the low
//          nibble of scales[t], block 2t+1 the high nibble, and one ls scales
//          the whole 32-value block.
//   IQ4_NL chunk l < 4 holds values 4l..4l+3 decoded from the low nibbles of
//          qs[4l..4l+3]; chunk l >= 4 holds values 4l..4l+3 decoded from the
//          high nibbles of qs[4l-16..4l-13].
//
// hsum_float_8 closes the tree as (v0+v4, v1+v5, v2+v6, v3+v7) and then
// (s0+s2) + (s1+s3); the CUDA twins in src/lowbit.cuh reproduce the same tree.
// ---------------------------------------------------------------------------
inline float hsum_float_8(const float lane[8]) {
    const float r0 = lane[0] + lane[4];
    const float r1 = lane[1] + lane[5];
    const float r2 = lane[2] + lane[6];
    const float r3 = lane[3] + lane[7];
    return (r0 + r2) + (r1 + r3);
}

inline float scalar_iq2_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    const int nb = cols / kQK_K;
    float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * blk = w_row + static_cast<size_t>(sb) * kIq2SBlockBytes;
        const float d = block_half(blk) * y[sb].d;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qs + 32;
        const uint8_t * scales = qh + 8;
        const int8_t * a = y[sb].qs;
        for (int c = 0; c < 8; ++c) {
            const int group = c >> 1;
            const int grid_half = c & 1;
            int32_t partial = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int ls = 1 + 2 * static_cast<int>(
                    (c < 4) ? (scales[ib32] & 0x0f) : (scales[ib32] >> 4));
                const uint16_t gi = static_cast<uint16_t>(
                    static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                    static_cast<uint16_t>(
                        (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300));
                const uint64_t quad = kIq2sGrid[gi] >> (32 * grid_half);
                const unsigned sign_bits =
                    static_cast<unsigned>(signs[ib32 * 4 + group] >> (4 * grid_half));
                int32_t dot = 0;
                for (int j = 0; j < 4; ++j) {
                    const int magnitude = static_cast<int>((quad >> (8 * j)) & 0xffu);
                    const int sign = ((sign_bits >> j) & 1u) ? -1 : 1;
                    dot += static_cast<int>(a[ib32 * 32 + c * 4 + j]) * (sign * magnitude);
                }
                partial += dot * ls;
            }
            lane[c] = std::fma(d, static_cast<float>(partial), lane[c]);
        }
    }
    return 0.125f * hsum_float_8(lane);
}

inline float scalar_iq3_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    const int nb = cols / kQK_K;
    float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * blk = w_row + static_cast<size_t>(sb) * kIq3SBlockBytes;
        const float d = block_half(blk) * y[sb].d;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        const int8_t * a = y[sb].qs;
        for (int c = 0; c < 8; ++c) {
            const int group = c >> 1;
            const int grid_half = c & 1;
            int32_t partial = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint8_t scale_byte = scales[ib32 >> 1];
                const int ls = 1 + 2 * static_cast<int>(
                    (ib32 & 1) ? (scale_byte >> 4) : (scale_byte & 0x0f));
                const uint16_t gi = static_cast<uint16_t>(
                    static_cast<uint16_t>(qs[ib32 * 8 + 2 * group + grid_half]) |
                    static_cast<uint16_t>(
                        (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group - grid_half)) & 0x100u));
                const uint32_t quad = kIq3sGrid[gi];
                const unsigned sign_bits =
                    static_cast<unsigned>(signs[ib32 * 4 + group] >> (4 * grid_half));
                int32_t dot = 0;
                for (int j = 0; j < 4; ++j) {
                    const int magnitude = static_cast<int>((quad >> (8 * j)) & 0xffu);
                    const int sign = ((sign_bits >> j) & 1u) ? -1 : 1;
                    dot += static_cast<int>(a[ib32 * 32 + c * 4 + j]) * (sign * magnitude);
                }
                partial += dot * ls;
            }
            lane[c] = std::fma(d, static_cast<float>(partial), lane[c]);
        }
    }
    return hsum_float_8(lane);
}

inline float scalar_iq4_nl_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    const int nb = cols / kQK4_NL;
    const int8_t * lut = iq4_nl_lut();
    // Even 32-value blocks accumulate in acc_even, odd ones in acc_odd, exactly
    // like the reference's accum1/accum2 pair split.
    float acc_even[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc_odd[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int b = 0;
    for (; b + 1 < nb; b += 2) {
        for (int parity = 0; parity < 2; ++parity) {
            const int bi = b + parity;
            const uint8_t * blk = w_row + static_cast<size_t>(bi) * kIq4NlBlockBytes;
            const float d = block_half(blk) * f16_to_f32(y[bi].d);
            float * acc = parity ? acc_odd : acc_even;
            for (int l = 0; l < 8; ++l) {
                const int high = l >> 2;
                int32_t partial = 0;
                for (int j = 0; j < 4; ++j) {
                    const int idx = 4 * (l & 3) + j;
                    const uint8_t byte = blk[2 + idx];
                    const uint8_t nibble = high ? static_cast<uint8_t>(byte >> 4)
                                                : static_cast<uint8_t>(byte & 0x0fu);
                    partial += static_cast<int>(y[bi].qs[16 * high + idx]) *
                               static_cast<int>(lut[nibble]);
                }
                acc[l] = std::fma(d, static_cast<float>(partial), acc[l]);
            }
        }
    }
    float lane[8];
    for (int l = 0; l < 8; ++l) lane[l] = acc_even[l] + acc_odd[l];
    float sumf = hsum_float_8(lane);
    if (b < nb) {
        // The reference's scalar leftover for a trailing odd block.
        const uint8_t * blk = w_row + static_cast<size_t>(b) * kIq4NlBlockBytes;
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < kQK4_NL / 2; ++j) {
            sumi1 += static_cast<int>(y[b].qs[j]) * static_cast<int>(lut[blk[2 + j] & 0x0f]);
            sumi2 += static_cast<int>(y[b].qs[j + kQK4_NL / 2]) *
                     static_cast<int>(lut[blk[2 + j] >> 4]);
        }
        sumf += (block_half(blk) * f16_to_f32(y[b].d)) * static_cast<float>(sumi1 + sumi2);
    }
    return sumf;
}

inline float scalar_q8_0_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    const int nb = cols / kQK8_0;
    float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = w_row + static_cast<size_t>(b) * kQ8_0BlockBytes;
        const int8_t * w = reinterpret_cast<const int8_t *>(blk + 2);
        const float d = block_half(blk) * f16_to_f32(y[b].d);
        for (int l = 0; l < 8; ++l) {
            int32_t partial = 0;
            for (int j = 0; j < 4; ++j) {
                partial += static_cast<int>(w[4 * l + j]) * static_cast<int>(y[b].qs[4 * l + j]);
            }
            lane[l] = std::fma(d, static_cast<float>(partial), lane[l]);
        }
    }
    return hsum_float_8(lane);
}

// ---------------------------------------------------------------------------
// Block types, the layer weight view and the per-layer workspace.
//
// The weight pointers are into the GGUF mapping: the expert bytes are paged in
// on demand and never copied to VRAM, which is the entire point of this path.
// ---------------------------------------------------------------------------
enum BlockType {
    kIq2S = 0,
    kIq3S = 1,
    kIq4Nl = 2,
    kQ8_0 = 3,
    kUnsupported = -1,
};

inline int block_type_from_ggml(int ggml_type) {
    switch (ggml_type) {
        case 22: return kIq2S;
        case 21: return kIq3S;
        case 20: return kIq4Nl;
        case 8:  return kQ8_0;
        default: return kUnsupported;
    }
}

inline size_t block_bytes(int type) {
    switch (type) {
        case kIq2S: return kIq2SBlockBytes;
        case kIq3S: return kIq3SBlockBytes;
        case kIq4Nl: return kIq4NlBlockBytes;
        case kQ8_0: return kQ8_0BlockBytes;
        default: throw std::runtime_error("CPU MoE: unsupported expert block type");
    }
}

inline int block_elems(int type) {
    switch (type) {
        case kIq2S: return kQK_K;
        case kIq3S: return kQK_K;
        case kIq4Nl: return kQK4_NL;
        case kQ8_0: return kQK8_0;
        default: throw std::runtime_error("CPU MoE: unsupported expert block type");
    }
}

inline size_t row_bytes(int type, int cols) {
    if (cols <= 0 || cols % block_elems(type) != 0) {
        throw std::runtime_error("CPU MoE: expert row is not a whole number of blocks");
    }
    return static_cast<size_t>(cols / block_elems(type)) * block_bytes(type);
}

struct LayerWeights {
    const uint8_t * gate = nullptr;   // [experts][ffn][hidden]
    const uint8_t * up = nullptr;
    const uint8_t * down = nullptr;   // [experts][hidden][ffn]
    size_t gate_expert_stride = 0;
    size_t up_expert_stride = 0;
    size_t down_expert_stride = 0;
    int hidden = 0;
    int expert_ffn = 0;
    int gate_type = kUnsupported;
    int up_type = kUnsupported;
    int down_type = kUnsupported;
};

// The shared expert stays on the GPU: llama.cpp's --n-cpu-moe tensor override
// only moves blk.<n>.ffn_.*_exps to the CPU, so its shexp gate/up/down keep
// running there and so do this engine's.

inline const uint8_t * expert_rows(const uint8_t * base, size_t stride, int expert) {
    return base + static_cast<size_t>(expert) * stride;
}

// Per-layer workspace, reused across tokens so a decode step allocates nothing
// after the first one.
struct Scratch {
    std::vector<q8_K> x_q8k;        // hidden / 256
    std::vector<float> gate;        // expert_ffn
    std::vector<float> up;          // expert_ffn
    std::vector<float> gated;       // expert_ffn
    std::vector<q8_0> down_act;     // expert_ffn / 32
    std::vector<float> expert_out;  // hidden

    void reset(int hidden, int expert_ffn) {
        if (hidden % kQK_K != 0 || expert_ffn % kQK8_0 != 0) {
            throw std::runtime_error("CPU MoE: hidden/expert_ffn geometry is not block aligned");
        }
        x_q8k.assign(static_cast<size_t>(hidden / kQK_K), q8_K{});
        gate.assign(static_cast<size_t>(expert_ffn), 0.0f);
        up.assign(static_cast<size_t>(expert_ffn), 0.0f);
        gated.assign(static_cast<size_t>(expert_ffn), 0.0f);
        down_act.assign(static_cast<size_t>(expert_ffn / kQK8_0), q8_0{});
        expert_out.assign(static_cast<size_t>(hidden), 0.0f);
    }
};

// ---------------------------------------------------------------------------
// AVX2 row kernels. These are what the engine runs; the scalar twins above are
// their independent bit-reference. Both follow the AVX2 branch of
// ggml-cpu/arch/x86/quants.c at the pinned commit and the CUDA DP4A kernels
// this project already checked against it.
// ---------------------------------------------------------------------------
#if defined(__AVX2__)

// ggml's k_mask1 / k_mask2 for the IQ2_S/IQ3_S sign expansion: mask1 spreads
// the four sign bytes over the four 8-value groups, mask2 keeps one bit per
// group position. The comparison then yields 0xff exactly where the activation
// must flip sign, which is the (q8 ^ neg) - neg identity.
inline __m256i sign_expand(__m256i aux, __m256i mask1, __m256i mask2) {
    const __m256i bits = _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2);
    return _mm256_cmpeq_epi8(bits, mask2);
}

// IQ2_S: one scale byte per 32-value block, the low nibble for the first 16
// values (the four 4-value chunk lanes of the low 128-bit half) and the high
// nibble for the second 16. The madd_epi16 below therefore needs ls1 in the
// low four int16 lanes and ls2 in the high four.
inline __m256i iq2_s_scale_vector(uint8_t scale_byte) {
    const int16_t ls1 = static_cast<int16_t>(1 + 2 * (scale_byte & 0x0f));
    const int16_t ls2 = static_cast<int16_t>(1 + 2 * (scale_byte >> 4));
    return _mm256_set_m128i(_mm_set1_epi16(ls2), _mm_set1_epi16(ls1));
}

inline float row_iq2_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    const int nb = cols / kQK_K;
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                           1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * blk = w_row + static_cast<size_t>(sb) * kIq2SBlockBytes;
        const float d = block_half(blk) * y[sb].d;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qs + 32;
        const uint8_t * scales = qh + 8;
        const int8_t * a = y[sb].qs;

        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + ib32 * 32));
            const __m256i q8_2 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + (ib32 + 1) * 32));
            const uint8_t * q = qs + (ib32 >> 1) * 8;
            const __m256i q2_1 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[q[3] | ((qh[ib32] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[2] | ((qh[ib32] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[1] | ((qh[ib32] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[0] | ((qh[ib32] << 8) & 0x300)]));
            const __m256i q2_2 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[q[7] | ((qh[ib32 + 1] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[6] | ((qh[ib32 + 1] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[5] | ((qh[ib32 + 1] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[q[4] | ((qh[ib32 + 1] << 8) & 0x300)]));

            // The reference builds the 32-bit sign word for a block from its
            // four sign bytes; the shuffle spreads byte g over the eight
            // positions of group g in the low lane and g+2's byte in the high.
            const __m256i neg1 = sign_expand(
                _mm256_set1_epi32(load_i32(signs + ib32 * 4)), mask1, mask2);
            const __m256i neg2 = sign_expand(
                _mm256_set1_epi32(load_i32(signs + (ib32 + 1) * 4)), mask1, mask2);
            const __m256i q8s1 = _mm256_sub_epi8(_mm256_xor_si256(neg1, q8_1), neg1);
            const __m256i q8s2 = _mm256_sub_epi8(_mm256_xor_si256(neg2, q8_2), neg2);

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s2);
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(dot1, iq2_s_scale_vector(scales[ib32])));
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(dot2, iq2_s_scale_vector(scales[ib32 + 1])));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    alignas(32) float v[8];
    _mm256_store_ps(v, acc);
    return 0.125f * hsum_float_8(v);
}

inline float row_iq3_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    const int nb = cols / kQK_K;
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                           1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * blk = w_row + static_cast<size_t>(sb) * kIq3SBlockBytes;
        const float d = block_half(blk) * y[sb].d;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        const int8_t * a = y[sb].qs;

        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + ib32 * 32));
            // Grid index = qs byte | (qh bit << 8), bit 8-2*g-g%2 for grid slot
            // 2*g + (chunk & 1), i.e. shift 8 down to 1 across the eight slots.
            const __m256i idx = _mm256_or_si256(
                _mm256_and_si256(
                    _mm256_sllv_epi32(_mm256_set1_epi32(static_cast<int>(qh[ib32])), idx_shift),
                    idx_mask),
                _mm256_cvtepi16_epi32(_mm_cvtepu8_epi16(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs + ib32 * 8)))));
            alignas(32) int32_t index[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(index), idx);
            const __m256i q2 = _mm256_set_epi32(
                static_cast<int>(kIq3sGrid[index[7]]), static_cast<int>(kIq3sGrid[index[6]]),
                static_cast<int>(kIq3sGrid[index[5]]), static_cast<int>(kIq3sGrid[index[4]]),
                static_cast<int>(kIq3sGrid[index[3]]), static_cast<int>(kIq3sGrid[index[2]]),
                static_cast<int>(kIq3sGrid[index[1]]), static_cast<int>(kIq3sGrid[index[0]]));

            const __m256i neg = sign_expand(
                _mm256_set1_epi32(load_i32(signs + ib32 * 4)), mask1, mask2);
            const __m256i q8s = _mm256_sub_epi8(_mm256_xor_si256(neg, q8), neg);

            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = 1 + 2 * static_cast<int>(
                (ib32 & 1) ? (scale_byte >> 4) : (scale_byte & 0x0f));
            sumi = _mm256_add_epi32(
                sumi,
                _mm256_madd_epi16(_mm256_maddubs_epi16(q2, q8s),
                                  _mm256_set1_epi16(static_cast<int16_t>(ls))));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    alignas(32) float v[8];
    _mm256_store_ps(v, acc);
    return hsum_float_8(v);
}

// IQ4_NL x Q8_0. One FMA per 32-value block with the reference's block-parity
// split: even blocks accumulate in accum1, odd ones in accum2, and only the
// closing hsum sees their sum.
inline __m256i iq4_nl_mul_add_epi8(__m256i x, __m256i y) {
    // _mm256_maddubs_epi16 treats its first operand as unsigned, so the signed
    // LUT values are passed through the psignb absolute-value/sign split.
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

inline float row_iq4_nl_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    const int nb = cols / kQK4_NL;
    const __m128i values128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(iq4_nl_lut()));
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const uint8_t * blk1 = w_row + static_cast<size_t>(ib + 0) * kIq4NlBlockBytes;
        const uint8_t * blk2 = w_row + static_cast<size_t>(ib + 1) * kIq4NlBlockBytes;
        const __m128i q4bits_1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk1 + 2));
        const __m128i q4bits_2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk2 + 2));
        const __m256i q8b_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[ib + 0].qs));
        const __m256i q8b_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[ib + 1].qs));
        // Low 128 bits: the sixteen low-nibble values (columns 0..15). High
        // 128 bits: the sixteen high-nibble values (columns 16..31).
        const __m256i q4b_1 = _mm256_set_m128i(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = _mm256_set_m128i(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p_1 = _mm256_madd_epi16(iq4_nl_mul_add_epi8(q4b_1, q8b_1), mone);
        const __m256i p_2 = _mm256_madd_epi16(iq4_nl_mul_add_epi8(q4b_2, q8b_2), mone);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(block_half(blk1) * f16_to_f32(y[ib + 0].d)),
                                 _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(block_half(blk2) * f16_to_f32(y[ib + 1].d)),
                                 _mm256_cvtepi32_ps(p_2), accum2);
    }
    alignas(32) float v[8];
    _mm256_store_ps(v, _mm256_add_ps(accum1, accum2));
    float sumf = hsum_float_8(v);
    if (ib < nb) {
        // The reference's scalar leftover for a trailing odd block.
        const uint8_t * blk = w_row + static_cast<size_t>(ib) * kIq4NlBlockBytes;
        const int8_t * lut = iq4_nl_lut();
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < kQK4_NL / 2; ++j) {
            sumi1 += static_cast<int>(y[ib].qs[j]) * static_cast<int>(lut[blk[2 + j] & 0x0f]);
            sumi2 += static_cast<int>(y[ib].qs[j + kQK4_NL / 2]) *
                     static_cast<int>(lut[blk[2 + j] >> 4]);
        }
        sumf += (block_half(blk) * f16_to_f32(y[ib].d)) * static_cast<float>(sumi1 + sumi2);
    }
    return sumf;
}

// Q8_0 x Q8_0 with one FMA per 32-value block. _mm256_sign_epi8(a, b) yields
// |a| with b's sign, so the signed product comes out of an unsigned maddubs +
// madd pair.
inline float row_q8_0_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    const int nb = cols / kQK8_0;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = w_row + static_cast<size_t>(b) * kQ8_0BlockBytes;
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[b].qs));
        const __m256i prod16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w),
                                                    _mm256_sign_epi8(q, w));
        const __m256i prod32 = _mm256_madd_epi16(prod16, ones);
        const float scale = block_half(blk) * f16_to_f32(y[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(prod32), acc);
    }
    alignas(32) float v[8];
    _mm256_store_ps(v, acc);
    return hsum_float_8(v);
}


// Weight-type dispatch. The activation pointer type is fixed by the weight
// type: IQ2_S/IQ3_S consume Q8_K, IQ4_NL/Q8_0 consume Q8_0.
inline float dot_row(int type, const uint8_t * w_row, const void * activation, int cols) {
    switch (type) {
        case kIq2S:  return row_iq2_s_q8_K(w_row, static_cast<const q8_K *>(activation), cols);
        case kIq3S:  return row_iq3_s_q8_K(w_row, static_cast<const q8_K *>(activation), cols);
        case kIq4Nl: return row_iq4_nl_q8_0(w_row, static_cast<const q8_0 *>(activation), cols);
        case kQ8_0:  return row_q8_0_q8_0(w_row, static_cast<const q8_0 *>(activation), cols);
        default: throw std::runtime_error("CPU MoE: unsupported expert block type");
    }
}

#else  // !__AVX2__

inline float row_iq2_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    return scalar_iq2_s_q8_K(w_row, y, cols);
}
inline float row_iq3_s_q8_K(const uint8_t * w_row, const q8_K * y, int cols) {
    return scalar_iq3_s_q8_K(w_row, y, cols);
}
inline float row_iq4_nl_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    return scalar_iq4_nl_q8_0(w_row, y, cols);
}
inline float row_q8_0_q8_0(const uint8_t * w_row, const q8_0 * y, int cols) {
    return scalar_q8_0_q8_0(w_row, y, cols);
}
inline float dot_row(int type, const uint8_t * w_row, const void * activation, int cols) {
    switch (type) {
        case kIq2S:  return scalar_iq2_s_q8_K(w_row, static_cast<const q8_K *>(activation), cols);
        case kIq3S:  return scalar_iq3_s_q8_K(w_row, static_cast<const q8_K *>(activation), cols);
        case kIq4Nl: return scalar_iq4_nl_q8_0(w_row, static_cast<const q8_0 *>(activation), cols);
        case kQ8_0:  return scalar_q8_0_q8_0(w_row, static_cast<const q8_0 *>(activation), cols);
        default: throw std::runtime_error("CPU MoE: unsupported expert block type");
    }
}

#endif  // __AVX2__

// ---------------------------------------------------------------------------
// Bit casts and the ggml AVX2 transcendentals.
// ---------------------------------------------------------------------------
inline uint32_t float_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

inline float uint_as_float(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// ggml-cpu/vec.h:1215 (AVX2 branch), one lane at a time. ggml_silu is
// x / (1 + expf(-x)) with this expf, and op_sigmoid is 1 / (1 + expf(-x)).
inline float ggml_avx2_exp_f32(float x) {
    constexpr float r = 0x1.8p23f;
    const float z = std::fma(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float inner = std::fma(-n, 0x1.62e4p-1f, x);
    const float b = std::fma(-n, 0x1.7f7d1cp-20f, inner);
    const uint32_t e = float_bits(z) << 23;
    const float k = uint_as_float(e + float_bits(1.0f));
    const float u = b * b;
    const float t0 = std::fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f);
    const float t1 = std::fma(t0, u, std::fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f));
    const float j = std::fma(t1, u, 0x1.ffffecp-1f * b);
    if (std::fabs(n) <= 126.0f) return std::fma(j, k, k);
    const uint32_t g = n <= 0.0f ? 0x82000000u : 0u;
    const float s1 = uint_as_float(g + 0x7f000000u);
    const float s2 = uint_as_float(e - g);
    if (std::fabs(n) > 192.0f) return s1 * s1;
    return std::fma(s2, j, s2) * s1;
}

inline float silu_ggml(float value) {
    return value / (1.0f + ggml_avx2_exp_f32(-value));
}

inline float sigmoid_ggml(float value) {
    return 1.0f / (1.0f + ggml_avx2_exp_f32(-value));
}

// ggml_vec_dot_f32 on AVX2: four 8-lane FMA accumulators, the GGML_F32_ARR
// folding, then the 8-lane reduce. Same visit order as the CUDA twin
// q38_moe_gate_dot_f32_avx2_kernel().
inline float dot_f32_avx2(const float * a, const float * b, int n) {
    float sum[4][8];
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = n & ~31;
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 8; ++l) {
                const int c = i + j * 8 + l;
                sum[j][l] = std::fma(a[c], b[c], sum[j][l]);
            }
        }
    }
    for (int l = 0; l < 8; ++l) {
        sum[0][l] = sum[0][l] + sum[2][l];
        sum[1][l] = sum[1][l] + sum[3][l];
    }
    for (int l = 0; l < 8; ++l) sum[0][l] = sum[0][l] + sum[1][l];
    float t[4];
    for (int k = 0; k < 4; ++k) t[k] = sum[0][k] + sum[0][k + 4];
    float acc = (t[0] + t[1]) + (t[2] + t[3]);
    for (int c = np; c < n; ++c) acc = acc + a[c] * b[c];
    return acc;
}

// ---------------------------------------------------------------------------
// Thread pool. std::thread is a thin wrapper over pthreads on Linux, the only
// platform the engine targets.
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int threads) {
        if (threads <= 0) threads = 1;
        threads_ = threads;
        workers_.reserve(static_cast<size_t>(threads - 1));
        for (int i = 1; i < threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool & operator=(const ThreadPool &) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_cv_.notify_all();
        done_cv_.notify_all();
        for (std::thread & t : workers_) t.join();
    }

    int threads() const { return threads_; }

    // Runs fn(i) for i in [0, count) and blocks until all calls returned. The
    // pool reduces nothing, so a task value never depends on thread scheduling;
    // keep the *order* inside fn deterministic when it accumulates floats.
    // Nested calls are not supported: a task must not call parallel_for again.
    void parallel_for(int count, const std::function<void(int)> & fn) {
        if (count <= 0) return;
        if (count == 1 || threads_ <= 1) {
            for (int i = 0; i < count; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (busy_) throw std::runtime_error("CPU MoE thread pool: nested parallel_for");
            busy_ = true;
            fn_ = &fn;
            count_ = count;
            next_.store(1, std::memory_order_relaxed);
            remaining_.store(count - 1, std::memory_order_relaxed);
            generation_ += 1;
        }
        work_cv_.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
        fn_ = nullptr;
        busy_ = false;
    }

private:
    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> * fn = nullptr;
            int count = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait(lock, [this, &seen] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                fn = fn_;
                count = count_;
            }
            if (fn == nullptr) continue;
            for (;;) {
                const int index = next_.fetch_add(1, std::memory_order_relaxed);
                if (index >= count) break;
                (*fn)(index);
                if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    done_cv_.notify_all();
                }
            }
        }
    }

    int threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    const std::function<void(int)> * fn_ = nullptr;
    int count_ = 0;
    std::atomic<int> next_{0};
    std::atomic<int> remaining_{0};
    uint64_t generation_ = 0;
    bool stop_ = false;
    bool busy_ = false;
};

// ---------------------------------------------------------------------------
// Environment switches. Read once per process.
//
//   WARODECODE_CPU_MOE=1         every trunk layer's routed experts on the CPU
//   WARODECODE_CPU_MOE_LAYERS=N  only the first N trunk layers
//   WARODECODE_CPU_MOE_THREADS=n pool size (default: min(hardware, 16))
//
// The shared expert, the router, attention and every other non-expert block
// stay on the GPU; only the routed experts move, which is what llama.cpp's
// per-tensor CPU override does.
// ---------------------------------------------------------------------------
inline bool env_present(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
}

inline bool env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < 0 || parsed > 4096) return fallback;
    return static_cast<int>(parsed);
}

// True when at least one layer may use the CPU path.
inline bool enabled() {
    static const bool on = [] {
        if (env_present("WARODECODE_CPU_MOE")) return env_flag("WARODECODE_CPU_MOE", true);
        return env_int("WARODECODE_CPU_MOE_LAYERS", 0) > 0;
    }();
    return on;
}

// Number of *leading* trunk layers whose routed experts run on the CPU.
inline int layer_count(int trunk_layers) {
    static const int configured = [] {
        if (env_present("WARODECODE_CPU_MOE_LAYERS")) return env_int("WARODECODE_CPU_MOE_LAYERS", 0);
        if (env_flag("WARODECODE_CPU_MOE", false)) return 1 << 20;
        return 0;
    }();
    if (configured <= 0) return 0;
    return std::min(configured, trunk_layers);
}

inline bool layer_is_cpu(int layer, int cpu_layers) {
    return cpu_layers > 0 && layer < cpu_layers;
}

inline int thread_count() {
    static const int configured = [] {
        if (env_present("WARODECODE_CPU_MOE_THREADS")) {
            return std::max(1, env_int("WARODECODE_CPU_MOE_THREADS", 1));
        }
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 8;
        return static_cast<int>(std::min(hw, 16u));
    }();
    return configured;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fused gate+up row: one pass over the expert's weight blocks and one pass over
// the activation decode BOTH projections. The arithmetic per output element is
// exactly row_iq2_s_q8_K / row_iq3_s_q8_K, so the values are unchanged; only the
// activation loads and the grid lookups are shared.
// ---------------------------------------------------------------------------
#if defined(__AVX2__)

inline void dot_gate_up_iq2_s_row(const uint8_t * gate_row, const uint8_t * up_row,
                                  const q8_K * y, int cols, float * gate_out, float * up_out) {
    const int nb = cols / kQK_K;
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                           1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    __m256 acc_g = _mm256_setzero_ps();
    __m256 acc_u = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * blk_g = gate_row + static_cast<size_t>(sb) * kIq2SBlockBytes;
        const uint8_t * blk_u = up_row + static_cast<size_t>(sb) * kIq2SBlockBytes;
        const float dg = block_half(blk_g) * y[sb].d;
        const float du = block_half(blk_u) * y[sb].d;
        const uint8_t * qsg = blk_g + 2;
        const uint8_t * qsu = blk_u + 2;
        const uint8_t * qhg = qsg + 64;
        const uint8_t * qhu = qsu + 64;
        const uint8_t * signsg = qsg + 32;
        const uint8_t * signsu = qsu + 32;
        const uint8_t * scalesg = qhg + 8;
        const uint8_t * scalesu = qhu + 8;
        const int8_t * a = y[sb].qs;

        __m256i sumg = _mm256_setzero_si256();
        __m256i sumu = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + ib32 * 32));
            const __m256i q8_2 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + (ib32 + 1) * 32));
            const uint8_t * qg = qsg + (ib32 >> 1) * 8;
            const uint8_t * qu = qsu + (ib32 >> 1) * 8;
            const __m256i q2g_1 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[qg[3] | ((qhg[ib32] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[2] | ((qhg[ib32] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[1] | ((qhg[ib32] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[0] | ((qhg[ib32] << 8) & 0x300)]));
            const __m256i q2g_2 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[qg[7] | ((qhg[ib32 + 1] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[6] | ((qhg[ib32 + 1] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[5] | ((qhg[ib32 + 1] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qg[4] | ((qhg[ib32 + 1] << 8) & 0x300)]));
            const __m256i q2u_1 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[qu[3] | ((qhu[ib32] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[2] | ((qhu[ib32] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[1] | ((qhu[ib32] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[0] | ((qhu[ib32] << 8) & 0x300)]));
            const __m256i q2u_2 = _mm256_set_epi64x(
                static_cast<long long>(kIq2sGrid[qu[7] | ((qhu[ib32 + 1] << 2) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[6] | ((qhu[ib32 + 1] << 4) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[5] | ((qhu[ib32 + 1] << 6) & 0x300)]),
                static_cast<long long>(kIq2sGrid[qu[4] | ((qhu[ib32 + 1] << 8) & 0x300)]));

            // One 32-bit sign word per block: the four sign bytes of the block
            // spread over its own 32 values, exactly like row_iq2_s_q8_K().
            const __m256i neg_g1 = sign_expand(
                _mm256_set1_epi32(load_i32(signsg + ib32 * 4)), mask1, mask2);
            const __m256i neg_g2 = sign_expand(
                _mm256_set1_epi32(load_i32(signsg + (ib32 + 1) * 4)), mask1, mask2);
            const __m256i neg_u1 = sign_expand(
                _mm256_set1_epi32(load_i32(signsu + ib32 * 4)), mask1, mask2);
            const __m256i neg_u2 = sign_expand(
                _mm256_set1_epi32(load_i32(signsu + (ib32 + 1) * 4)), mask1, mask2);
            const __m256i q8sg1 = _mm256_sub_epi8(_mm256_xor_si256(neg_g1, q8_1), neg_g1);
            const __m256i q8sg2 = _mm256_sub_epi8(_mm256_xor_si256(neg_g2, q8_2), neg_g2);
            const __m256i q8su1 = _mm256_sub_epi8(_mm256_xor_si256(neg_u1, q8_1), neg_u1);
            const __m256i q8su2 = _mm256_sub_epi8(_mm256_xor_si256(neg_u2, q8_2), neg_u2);

            // Chunks 0..3 carry the low nibble of the block's scale byte, chunks
            // 4..7 the high nibble: the same vector row_iq2_s_q8_K() builds.
            const __m256i scg1 = iq2_s_scale_vector(scalesg[ib32]);
            const __m256i scg2 = iq2_s_scale_vector(scalesg[ib32 + 1]);
            const __m256i scu1 = iq2_s_scale_vector(scalesu[ib32]);
            const __m256i scu2 = iq2_s_scale_vector(scalesu[ib32 + 1]);
            sumg = _mm256_add_epi32(sumg, _mm256_madd_epi16(_mm256_maddubs_epi16(q2g_1, q8sg1), scg1));
            sumg = _mm256_add_epi32(sumg, _mm256_madd_epi16(_mm256_maddubs_epi16(q2g_2, q8sg2), scg2));
            sumu = _mm256_add_epi32(sumu, _mm256_madd_epi16(_mm256_maddubs_epi16(q2u_1, q8su1), scu1));
            sumu = _mm256_add_epi32(sumu, _mm256_madd_epi16(_mm256_maddubs_epi16(q2u_2, q8su2), scu2));
        }
        acc_g = _mm256_fmadd_ps(_mm256_set1_ps(dg), _mm256_cvtepi32_ps(sumg), acc_g);
        acc_u = _mm256_fmadd_ps(_mm256_set1_ps(du), _mm256_cvtepi32_ps(sumu), acc_u);
    }
    alignas(32) float vg[8];
    alignas(32) float vu[8];
    _mm256_store_ps(vg, acc_g);
    _mm256_store_ps(vu, acc_u);
    *gate_out = 0.125f * (((vg[0] + vg[4]) + (vg[2] + vg[6])) + ((vg[1] + vg[5]) + (vg[3] + vg[7])));
    *up_out = 0.125f * (((vu[0] + vu[4]) + (vu[2] + vu[6])) + ((vu[1] + vu[5]) + (vu[3] + vu[7])));
}

#endif  // __AVX2__

// Best available gate+up pair for one row. Falls back to two independent rows
// whenever the fused AVX2 kernel does not cover the weight type.
inline void dot_gate_up_row(int type, const uint8_t * gate_row, const uint8_t * up_row,
                            const void * activation, int cols, float * gate_out, float * up_out) {
#if defined(__AVX2__)
    if (type == kIq2S) {
        dot_gate_up_iq2_s_row(gate_row, up_row, static_cast<const q8_K *>(activation),
                              cols, gate_out, up_out);
        return;
    }
#endif
    *gate_out = dot_row(type, gate_row, activation, cols);
    *up_out = dot_row(type, up_row, activation, cols);
}

// ---------------------------------------------------------------------------
// One non-shared expert, entirely on the calling thread. The three row loops
// are serial: the parallelism of a layer comes from running the ten experts
// concurrently, one pool task each.
// ---------------------------------------------------------------------------
// One expert's gate/up/SiLU/Q8_0 chain plus the down rows in
// [row_begin, row_end). The gate/up stages always cover the whole expert (they
// feed the quantised activation), so a partial range only skips trailing FMA
// terms of the down projection's own per-row accumulation - the row sums are
// independent, so the concatenation of the ranges reproduces the serial result
// exactly.
// Stage 1 of one expert: gate matmul, up matmul, SiLU product and the Q8_0
// requantisation the down projection consumes.
inline void run_expert_gate_up(const LayerWeights & w, int expert, const q8_K * x_q8k,
                               Scratch & scratch) {
    const uint8_t * gate_row0 = expert_rows(w.gate, w.gate_expert_stride, expert);
    const uint8_t * up_row0 = expert_rows(w.up, w.up_expert_stride, expert);
    const size_t gate_stride = row_bytes(w.gate_type, w.hidden);
    const size_t up_stride = row_bytes(w.up_type, w.hidden);
    const int ffn = w.expert_ffn;
    const int hidden = w.hidden;

    float * gate = scratch.gate.data();
    float * up = scratch.up.data();
    float * gated = scratch.gated.data();

    // One pass over the block decode and one over the activation for both
    // projections (dot_gate_up_row falls back to two independent rows for a
    // weight type the fused kernel does not cover).
    if (w.gate_type == w.up_type) {
        for (int r = 0; r < ffn; ++r) {
            dot_gate_up_row(w.gate_type,
                            gate_row0 + static_cast<size_t>(r) * gate_stride,
                            up_row0 + static_cast<size_t>(r) * up_stride,
                            x_q8k, hidden, &gate[r], &up[r]);
        }
    } else {
        for (int r = 0; r < ffn; ++r) {
            gate[r] = dot_row(w.gate_type, gate_row0 + static_cast<size_t>(r) * gate_stride, x_q8k, hidden);
        }
        for (int r = 0; r < ffn; ++r) {
            up[r] = dot_row(w.up_type, up_row0 + static_cast<size_t>(r) * up_stride, x_q8k, hidden);
        }
    }
    for (int r = 0; r < ffn; ++r) gated[r] = silu_ggml(gate[r]) * up[r];
    quantize_q8_0(gated, scratch.down_act.data(), ffn / kQK8_0);
}

// Stage 2 of one expert: the down rows in [row_begin, row_end). Rows are
// independent, so any partition of [0, hidden) concatenates to the serial
// result bit for bit.
inline void run_expert_down(const LayerWeights & w, int expert, int row_begin, int row_end,
                            Scratch & scratch) {
    const uint8_t * down_row0 = expert_rows(w.down, w.down_expert_stride, expert);
    const size_t down_stride = row_bytes(w.down_type, w.expert_ffn);
    float * out = scratch.expert_out.data();
    const q8_0 * down_act = scratch.down_act.data();
    for (int r = row_begin; r < row_end; ++r) {
        out[r] = dot_row(w.down_type, down_row0 + static_cast<size_t>(r) * down_stride, down_act,
                         w.expert_ffn);
    }
}

// ---------------------------------------------------------------------------
// One layer's routed MoE.
//
// hidden_f32 is the unrounded MoE input (the HC mix row the GPU F32 path also
// feeds to the router and the experts). expert_ids / expert_weights are the
// router's host-side results. out receives the weighted sum accumulated in
// ascending expert order with std::fma, which is exactly what the CUDA
// q38_weighted_accum_f32_kernel produces when it is launched once per expert.
//
// The expert outputs go to per-expert buffers instead of one shared buffer, so
// no thread ever writes a byte another thread reads and the accumulation order
// is independent of the schedule.
// ---------------------------------------------------------------------------
struct LayerScratch {
    std::vector<Scratch> experts;   // one per selected expert
    Scratch shared;                 // the shared expert's own buffers
    std::vector<q8_K> x_q8k;        // the shared quantised layer input

    void reset(int k, int hidden, int expert_ffn) {
        x_q8k.assign(static_cast<size_t>(hidden / kQK_K), q8_K{});
        experts.resize(static_cast<size_t>(k));
        for (Scratch & s : experts) s.reset(hidden, expert_ffn);
        shared.reset(hidden, expert_ffn);
    }
};

inline void run_layer(const LayerWeights & w, ThreadPool & pool, const float * hidden_f32,
                      const int * expert_ids, const float * expert_weights, int k,
                      LayerScratch & scratch, float * out) {
    if (w.hidden <= 0 || w.hidden % kQK_K != 0) {
        throw std::runtime_error("CPU MoE: hidden size is not a multiple of 256");
    }
    if (k <= 0 || static_cast<size_t>(k) > scratch.experts.size()) {
        throw std::runtime_error("CPU MoE: layer scratch does not cover the expert count");
    }
    quantize_q8_K(hidden_f32, scratch.x_q8k.data(), static_cast<int>(scratch.x_q8k.size()));
    const q8_K * x = scratch.x_q8k.data();

    // Stage 1: gate/up/SiLU/Q8_0 for every selected expert, one pool task each
    // (10 experts vs 12 threads).
    pool.parallel_for(k, [&](int j) {
        run_expert_gate_up(w, expert_ids[j], x, scratch.experts[static_cast<size_t>(j)]);
    });
    // Stage 2: the down projection has four times the rows of gate or up, so it
    // is split by rows on top of the experts; k * slices tasks fill the pool.
    const int slices = std::max(1, std::min(pool.threads(), w.hidden / 64));
    if (slices > 1) {
        const int rows_per_slice = (w.hidden + slices - 1) / slices;
        pool.parallel_for(k * slices, [&](int t) {
            const int j = t / slices;
            const int slice = t % slices;
            const int from = slice * rows_per_slice;
            const int to = std::min(w.hidden, from + rows_per_slice);
            if (from >= to) return;
            run_expert_down(w, expert_ids[j], from, to, scratch.experts[static_cast<size_t>(j)]);
        });
    } else {
        pool.parallel_for(k, [&](int j) {
            run_expert_down(w, expert_ids[j], 0, w.hidden, scratch.experts[static_cast<size_t>(j)]);
        });
    }

    const int hidden = w.hidden;
    for (int j = 0; j < k; ++j) {
        const float weight = expert_weights[j];
        const float * src = scratch.experts[static_cast<size_t>(j)].expert_out.data();
        for (int i = 0; i < hidden; ++i) {
            // Two roundings, exactly like ggml_mul followed by ggml_add (and the
            // CUDA q38_weighted_accum_f32_kernel): not an FMA.
            out[i] = add_rn(out[i], mul_rn(src[i], weight));
        }
    }
}

}  // namespace cpu_moe
}  // namespace wd
