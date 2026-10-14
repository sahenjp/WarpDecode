// Oracle check for the Task N CPU MoE kernels: this file compiles the *actual*
// AVX2 kernels of the pinned llama.cpp and compares them bit for bit against
// src/qwen38_cpu_moe.cuh.
//
// The reference code below is copied verbatim from
//   ggml-org/llama.cpp @ 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
//   ggml/src/ggml-cpu/arch/x86/quants.c            (MIT licence)
//   ggml/src/ggml-common.h                         (block layouts, tables)
// with only these mechanical edits:
//   * GGML_RESTRICT / GGML_UNUSED / assert are dropped or replaced,
//   * the generic tails are kept, the AVX/AVX512 branches are dropped,
//   * GGML_CPU_FP16_TO_FP32 becomes the compiler's _Float16 conversion (both
//     are exact IEEE half -> float conversions; the header's own f16_to_f32 is
//     already checked against _Float16 in qwen38_cpu_moe_host_test.cpp).
//
// The host test proves the AVX2 kernels equal their scalar twins and the exact
// int64 lane sums. This file proves the twins are the *same function* llama.cpp
// runs, so the CPU MoE path cannot drift from the oracle by a rounding step.
//
// Run: bash tests/qwen38_cpu_moe_host_check.sh

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "qwen38_cpu_moe.cuh"

namespace {

using namespace wd::cpu_moe;

int g_failures = 0;
int g_checks = 0;

void check(const char * what, bool ok) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %s\n", what);
    }
}

bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// ---------------------------------------------------------------------------
// ggml block layouts (ggml-common.h) and the pinned reference kernels.
// ---------------------------------------------------------------------------
#define REF_QK_K 256
#define REF_QK8_0 32
#define REF_QK4_NL 32

typedef uint16_t ref_half;   // ggml_half is a plain uint16_t

typedef struct {
    ref_half d;
    uint8_t qs[REF_QK_K / 4];
    uint8_t qh[REF_QK_K / 32];
    uint8_t scales[REF_QK_K / 32];
} ref_block_iq2_s;

typedef struct {
    ref_half d;
    uint8_t qs[REF_QK_K / 4];
    uint8_t qh[REF_QK_K / 32];
    uint8_t signs[REF_QK_K / 8];
    uint8_t scales[REF_QK_K / 64];
} ref_block_iq3_s;

typedef struct {
    ref_half d;
    uint8_t qs[REF_QK4_NL / 2];
} ref_block_iq4_nl;

typedef struct {
    ref_half d;
    int8_t qs[REF_QK8_0];
} ref_block_q8_0;

typedef struct {
    float d;
    int8_t qs[REF_QK_K];
    int16_t bsums[REF_QK_K / 16];
} ref_block_q8_K;

static const int8_t ref_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static inline float ref_fp16_to_fp32(uint16_t h) {
    _Float16 v;
    std::memcpy(&v, &h, sizeof(h));
    return static_cast<float>(v);
}

// ggml-cpu/arch/x86/quants.c:43
static inline float hsum_float_8_ref(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ggml-cpu/arch/x86/quants.c:69
static inline __m256i mul_add_epi8_ref(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// ggml-cpu/arch/x86/quants.c:99
static inline __m256 sum_i16_pairs_float_ref(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

// ggml-cpu/arch/x86/quants.c:105
static inline __m256 mul_sum_us8_pairs_float_ref(const __m256i ax, const __m256i sy) {
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float_ref(dot);
}

// ggml-cpu/arch/x86/quants.c:122
static inline __m256 mul_sum_i8_pairs_float_ref(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float_ref(ax, sy);
}

// ggml-cpu/arch/x86/quants.c:527, the per-block scale shuffle. The table is
// copied verbatim; it selects int16 lanes 2i and 2i+1 in the low lane and
// 2i+8 / 2i+9 in the high lane of scales16.
static inline __m256i get_scale_shuffle_k4_ref(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_shuffle) + i);
}

// The grid tables are the ones the CUDA path already proved identical to
// ggml-common.h (qwen38_cpu_moe_host_test.cpp re-checks that equality).
static inline long long iq2s_grid_ref(uint16_t index) {
    return static_cast<long long>(kIq2sGrid[index]);
}
static inline uint32_t iq3s_grid_ref(uint16_t index) {
    return kIq3sGrid[index];
}

// ggml-cpu/arch/x86/quants.c:502, IQ2_S AVX2 branch, verbatim but for the edits
// listed at the top of this file.
float ref_vec_dot_iq2_s_q8_K(int n, const void * vx, const void * vy) {
    const ref_block_iq2_s * x = static_cast<const ref_block_iq2_s *>(vx);
    const ref_block_q8_K * y = static_cast<const ref_block_q8_K *>(vy);
    const int nb = n / REF_QK_K;

    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_mask1));
    const __m256i mask2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_mask2));

    uint64_t aux64;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = ref_fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = reinterpret_cast<const uint16_t *>(x[i].qs + REF_QK_K / 8);
        const int8_t * q8 = y[i].qs;

        std::memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(
            _mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < REF_QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
            q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(
                iq2s_grid_ref(qs[3] | ((qh[ib32 + 0] << 2) & 0x300)),
                iq2s_grid_ref(qs[2] | ((qh[ib32 + 0] << 4) & 0x300)),
                iq2s_grid_ref(qs[1] | ((qh[ib32 + 0] << 6) & 0x300)),
                iq2s_grid_ref(qs[0] | ((qh[ib32 + 0] << 8) & 0x300)));
            const __m256i q2_2 = _mm256_set_epi64x(
                iq2s_grid_ref(qs[7] | ((qh[ib32 + 1] << 2) & 0x300)),
                iq2s_grid_ref(qs[6] | ((qh[ib32 + 1] << 4) & 0x300)),
                iq2s_grid_ref(qs[5] | ((qh[ib32 + 1] << 6) & 0x300)),
                iq2s_grid_ref(qs[4] | ((qh[ib32 + 1] << 8) & 0x300)));
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (static_cast<uint32_t>(signs[1]) << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (static_cast<uint32_t>(signs[3]) << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4_ref(ib32 + 0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4_ref(ib32 + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return 0.125f * hsum_float_8_ref(accumf);
}

// ggml-cpu/arch/x86/quants.c:3384, IQ3_S AVX2 branch.
float ref_vec_dot_iq3_s_q8_K(int n, const void * vx, const void * vy) {
    const ref_block_iq3_s * x = static_cast<const ref_block_iq3_s *>(vx);
    const ref_block_q8_K * y = static_cast<const ref_block_q8_K *>(vy);
    const int nb = n / REF_QK_K;

    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m256i mask1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_mask1));
    const __m256i mask2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_mask2));

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);

    typedef union {
        __m256i vec[2];
        uint32_t index[16];
    } index_t;
    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = ref_fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = reinterpret_cast<const uint16_t *>(x[i].signs);
        const int8_t * q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < REF_QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
            q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(qs)));
            qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            const __m256i q2_1 = _mm256_set_epi32(
                iq3s_grid_ref(idx.index[7]), iq3s_grid_ref(idx.index[6]),
                iq3s_grid_ref(idx.index[5]), iq3s_grid_ref(idx.index[4]),
                iq3s_grid_ref(idx.index[3]), iq3s_grid_ref(idx.index[2]),
                iq3s_grid_ref(idx.index[1]), iq3s_grid_ref(idx.index[0]));
            const __m256i q2_2 = _mm256_set_epi32(
                iq3s_grid_ref(idx.index[15]), iq3s_grid_ref(idx.index[14]),
                iq3s_grid_ref(idx.index[13]), iq3s_grid_ref(idx.index[12]),
                iq3s_grid_ref(idx.index[11]), iq3s_grid_ref(idx.index[10]),
                iq3s_grid_ref(idx.index[9]), iq3s_grid_ref(idx.index[8]));

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return hsum_float_8_ref(accumf);
}

// ggml-cpu/arch/x86/quants.c:3920, IQ4_NL AVX2 branch plus its scalar tail.
float ref_vec_dot_iq4_nl_q8_0(int n, const void * vx, const void * vy) {
    const ref_block_iq4_nl * x = static_cast<const ref_block_iq4_nl *>(vx);
    const ref_block_q8_0 * y = static_cast<const ref_block_q8_0 *>(vy);
    const int nb = n / REF_QK4_NL;

    int ib = 0;
    float sumf = 0;
    const __m128i values128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ref_kvalues_iq4nl));
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[ib + 0].qs));
        const __m128i q4bits_2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[ib + 1].qs));
        const __m256i q8b_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[ib + 0].qs));
        const __m256i q8b_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[ib + 1].qs));
        const __m256i q4b_1 = _mm256_set_m128i(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = _mm256_set_m128i(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8_ref(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8_ref(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(
            _mm256_set1_ps(ref_fp16_to_fp32(y[ib + 0].d) * ref_fp16_to_fp32(x[ib + 0].d)),
            _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(
            _mm256_set1_ps(ref_fp16_to_fp32(y[ib + 1].d) * ref_fp16_to_fp32(x[ib + 1].d)),
            _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8_ref(_mm256_add_ps(accum1, accum2));

    for (; ib < nb; ++ib) {
        const float d = ref_fp16_to_fp32(y[ib].d) * ref_fp16_to_fp32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < REF_QK4_NL / 2; ++j) {
            sumi1 += y[ib].qs[j + 0] * ref_kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + REF_QK4_NL / 2] * ref_kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    return sumf;
}

// ggml-cpu/arch/x86/quants.c:1308, Q8_0 AVX2 branch.
float ref_vec_dot_q8_0_q8_0(int n, const void * vx, const void * vy) {
    const ref_block_q8_0 * x = static_cast<const ref_block_q8_0 *>(vx);
    const ref_block_q8_0 * y = static_cast<const ref_block_q8_0 *>(vy);
    const int nb = n / REF_QK8_0;

    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(ref_fp16_to_fp32(x[ib].d) * ref_fp16_to_fp32(y[ib].d));
        const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x[ib].qs));
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[ib].qs));
        const __m256 q = mul_sum_i8_pairs_float_ref(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    return hsum_float_8_ref(acc);
}

}  // namespace

int main() {
    std::printf("cpu-moe llama.cpp oracle check\n");
    std::mt19937 rng(20261014u);
    auto next = [&rng]() { return rng(); };
    auto uniform = [&rng](float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng);
    };

    const int cols_list[] = {256, 512, 768, 2560, 5120, 640, 224};
    for (int cols : cols_list) {
        for (int rep = 0; rep < 4; ++rep) {
            const size_t nb_k = static_cast<size_t>(cols / kQK_K);
            const size_t nb_0 = static_cast<size_t>(cols / kQK8_0);

            // IQ2_S / IQ3_S: the ggml blocks are byte-identical to ours, so the
            // Same rows are handed to both implementations.
            std::vector<uint8_t> row2(nb_k * kIq2SBlockBytes);
            std::vector<uint8_t> row3(nb_k * kIq3SBlockBytes);
            std::vector<uint8_t> row4(nb_0 * kIq4NlBlockBytes);
            std::vector<uint8_t> row5(nb_0 * kQ8_0BlockBytes);
            for (size_t b = 0; b < nb_k; ++b) {
                uint8_t * b2 = row2.data() + b * kIq2SBlockBytes;
                uint8_t * b3 = row3.data() + b * kIq3SBlockBytes;
                const uint16_t d2 = f32_to_f16_bits(uniform(0.0005f, 0.05f));
                const uint16_t d3 = f32_to_f16_bits(uniform(0.0005f, 0.05f));
                std::memcpy(b2, &d2, 2);
                std::memcpy(b3, &d3, 2);
                for (int i = 0; i < 64; ++i) b2[2 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 8; ++i) b2[66 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 8; ++i) b2[74 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 64; ++i) b3[2 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 8; ++i) b3[66 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 32; ++i) b3[74 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 4; ++i) b3[106 + i] = static_cast<uint8_t>(next() & 0xff);
            }
            for (size_t b = 0; b < nb_0; ++b) {
                uint8_t * b4 = row4.data() + b * kIq4NlBlockBytes;
                uint8_t * b5 = row5.data() + b * kQ8_0BlockBytes;
                const uint16_t d4 = f32_to_f16_bits(uniform(0.0005f, 0.05f));
                const uint16_t d5 = f32_to_f16_bits(uniform(0.0005f, 0.05f));
                std::memcpy(b4, &d4, 2);
                std::memcpy(b5, &d5, 2);
                for (int i = 0; i < 16; ++i) b4[2 + i] = static_cast<uint8_t>(next() & 0xff);
                for (int i = 0; i < 32; ++i) b5[2 + i] = static_cast<uint8_t>(next() & 0xff);
            }

            std::vector<float> act(static_cast<size_t>(cols));
            for (int i = 0; i < cols; ++i) act[static_cast<size_t>(i)] = uniform(-2.5f, 2.5f);
            std::vector<q8_K> ya(nb_k);
            quantize_q8_K(act.data(), ya.data(), static_cast<int>(nb_k));
            std::vector<q8_0> y0(nb_0);
            quantize_q8_0(act.data(), y0.data(), static_cast<int>(nb_0));

            std::vector<ref_block_q8_K> ry(nb_k);
            for (size_t b = 0; b < nb_k; ++b) {
                ry[b].d = ya[b].d;
                std::memcpy(ry[b].qs, ya[b].qs, sizeof(ry[b].qs));
                std::memcpy(ry[b].bsums, ya[b].bsums, sizeof(ry[b].bsums));
            }
            if (cols % kQK_K == 0) {
                const float r2 = ref_vec_dot_iq2_s_q8_K(cols, row2.data(), ry.data());
                const float o2 = row_iq2_s_q8_K(row2.data(), ya.data(), cols);
                check("oracle iq2_s", same_bits(r2, o2));
                const float r3 = ref_vec_dot_iq3_s_q8_K(cols, row3.data(), ry.data());
                const float o3 = row_iq3_s_q8_K(row3.data(), ya.data(), cols);
                check("oracle iq3_s", same_bits(r3, o3));
            }
            {
                const float r4 = ref_vec_dot_iq4_nl_q8_0(cols, row4.data(), y0.data());
                const float o4 = row_iq4_nl_q8_0(row4.data(), y0.data(), cols);
                check("oracle iq4_nl", same_bits(r4, o4));
                const float r5 = ref_vec_dot_q8_0_q8_0(cols, row5.data(), y0.data());
                const float o5 = row_q8_0_q8_0(row5.data(), y0.data(), cols);
                check("oracle q8_0", same_bits(r5, o5));
            }
        }
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
