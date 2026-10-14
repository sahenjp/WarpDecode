// Host bit-comparison for the DP4A GEMV variants in src/lowbit.cuh (Task D).
//
// The kernels themselves need nvcc, so this test mirrors both orderings for one
// (row, chunk) pair as plain C++ and compares the results bit for bit:
//
//   reference: the per-element products of the existing
//              iq2_s_q8_k_gemv_kernel / iq3_s_q8_k_gemv_kernel /
//              iq4_nl_q8_0_act_gemv_kernel, summed in the same order the four
//              lanes of a chunk are combined by the two shfl_down steps,
//   dp4a:      the byte-packed __dp4a of the new kernels, where the four grid
//              magnitudes (or IQ4_NL codewords) are negated as bytes and dotted
//              with the four quantised activations in a single instruction.
//
// Both must agree exactly - not approximately - because the only difference is
// the order of exact integer additions and one integer multiply. The scale
// application, IQ4_NL's even/odd block split and the closing hsum_float_8 tree
// are shared by both implementations in this file, so a mismatch can only come
// from the integer path. The DP4A integer results are additionally compared
// against an exact 64-bit per-element dot product, which catches a broken sign
// expansion or nibble order even if both implementations were changed together.
//
// What this does NOT check: the compiled device code, the sm < 80 fallback of
// __dp4a (the scalar expansion below mirrors it) or the launch geometry. The
// 64/64 token parity run on the GPU covers those.
//
// Run: bash tests/lowbit_dp4a_host_check.sh

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "iq2s_grid.cuh"
#include "iq3s_grid.cuh"

namespace {

using std::int32_t;
using std::int64_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

int32_t g_failures = 0;
int32_t g_checks = 0;

void check(bool ok, const char * what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 20) std::printf("FAIL: %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// Copies of the src/lowbit.cuh helpers under test (device functions cannot be
// linked here; tests/qwen38_cuda_syntax_check.sh type-checks the originals).
// ---------------------------------------------------------------------------

uint32_t sign_mask4(uint32_t bits) {
    const uint32_t spaced = (bits & 1u) | ((bits & 2u) << 7) |
                            ((bits & 4u) << 14) | ((bits & 8u) << 21);
    return spaced * 0xffu;
}

uint32_t sign_i8x4(uint32_t quad, uint32_t bits) {
    const uint32_t mask = sign_mask4(bits);
    return (quad ^ mask) + (mask & 0x01010101u);
}

// Mirrors the sm < 80 fallback of lowbit_dp4a().
int32_t dp4a_scalar(int32_t a, int32_t b, int32_t c) {
    int32_t sum = c;
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int32_t>(
                   static_cast<int8_t>(static_cast<uint32_t>(a) >> (8 * i))) *
               static_cast<int32_t>(
                   static_cast<int8_t>(static_cast<uint32_t>(b) >> (8 * i)));
    }
    return sum;
}

const int8_t kIq4NlLut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

uint32_t iq4_nl_pack4(uint32_t nibbles) {
    return static_cast<uint32_t>(static_cast<uint8_t>(kIq4NlLut[nibbles & 0x0fu])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(kIq4NlLut[(nibbles >> 8) & 0x0fu])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(kIq4NlLut[(nibbles >> 16) & 0x0fu])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(kIq4NlLut[(nibbles >> 24) & 0x0fu])) << 24);
}

// Exact F16 -> F32 widening, so the block scale reads identically here and in
// the kernels (lowbit_half()).
float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int shift = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0) { m <<= 1; ++shift; }
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) |
                   ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float load_half(const uint8_t * p) {
    uint16_t h = 0;
    std::memcpy(&h, p, sizeof(h));
    return half_bits_to_float(h);
}

// Same float helpers the kernels use.
float fadd_rn(float a, float b) { return a + b; }
float fmul_rn(float a, float b) { return a * b; }
float fma_rn(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

uint32_t float_bits(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// The closing tree of the reference kernels, expressed on the eight chunk
// accumulators. Both kernels now hold them in lanes row*8 + 0..7, so the same
// pairing is (v0+v4, v1+v5, v2+v6, v3+v7) then (s0+s2) + (s1+s3).
float hsum_float_8(const float v[8]) {
    const float r0 = fadd_rn(v[0], v[4]);
    const float r1 = fadd_rn(v[1], v[5]);
    const float r2 = fadd_rn(v[2], v[6]);
    const float r3 = fadd_rn(v[3], v[7]);
    return fadd_rn(fadd_rn(r0, r2), fadd_rn(r1, r3));
}

uint8_t iq2s_grid_byte(uint16_t index, int lane8) {
    return static_cast<uint8_t>(wd::k_iq2s_grid[index] >> (lane8 * 8));
}

uint8_t iq3s_grid_byte(uint16_t index, int lane4) {
    return static_cast<uint8_t>(wd::k_iq3s_grid[index] >> (lane4 * 8));
}

// ---------------------------------------------------------------------------
// IQ2_S x Q8_K: one chunk of four columns, accumulated over a whole row.
// ---------------------------------------------------------------------------

// Reference: lane 4c+j keeps its own int32 over the eight 32-value blocks and the
// two shfl_down steps then add {0,2} and {1,3} before the per-superblock scale.
float iq2s_reference_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_d, int cols, int chunk) {
    const int superblocks = cols >> 8;
    const int group = chunk >> 1;
    float acc = 0.0f;
    for (int sb = 0; sb < superblocks; ++sb) {
        const uint8_t * blk = row + static_cast<size_t>(sb) * 82u;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        int32_t lane_sum[4] = {0, 0, 0, 0};
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            for (int j = 0; j < 4; ++j) {
                const int lane = chunk * 4 + j;   // the reference warp lane
                const int lane8 = lane & 7;
                const uint8_t scale_byte = scales[ib32];
                const int ls = (lane >> 3) < 2
                    ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                    : 1 + 2 * static_cast<int>(scale_byte >> 4);
                const uint16_t grid_index = static_cast<uint16_t>(
                    static_cast<uint16_t>(qs[ib32 * 4 + (lane >> 3)]) |
                    static_cast<uint16_t>(
                        (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * (lane >> 3))) & 0x300u));
                int qv = static_cast<int>(iq2s_grid_byte(grid_index, lane8));
                if (signs[ib32 * 4 + (lane >> 3)] & (1u << lane8)) qv = -qv;
                const int col = (sb << 8) + (ib32 << 5) + lane;
                lane_sum[j] += static_cast<int32_t>(act[col]) * qv * ls;
            }
        }
        const int32_t a2 = lane_sum[0] + lane_sum[2];
        const int32_t b2 = lane_sum[1] + lane_sum[3];
        acc = fma_rn(fmul_rn(load_half(blk), act_d[sb]),
                     static_cast<float>(a2 + b2), acc);
    }
    return acc;
}

// DP4A: one __dp4a per 32-value block covers the four columns, the group scale
// multiplies the chunk sum once.
float iq2s_dp4a_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_d, int cols, int chunk,
        int64_t * exact_out) {
    const int superblocks = cols >> 8;
    const int group = chunk >> 1;
    const int grid_half = chunk & 1;
    const bool scale_low_nibble = chunk < 4;
    float acc = 0.0f;
    int64_t exact = 0;
    for (int sb = 0; sb < superblocks; ++sb) {
        const uint8_t * blk = row + static_cast<size_t>(sb) * 82u;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t scale_byte = scales[ib32];
            const int ls = scale_low_nibble
                ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                : 1 + 2 * static_cast<int>(scale_byte >> 4);
            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u));
            const uint32_t quad = static_cast<uint32_t>(
                wd::k_iq2s_grid[grid_index] >> (32 * grid_half));
            const uint32_t sign_bits =
                static_cast<uint32_t>(signs[ib32 * 4 + group] >> (4 * grid_half));
            const uint32_t weights4 = sign_i8x4(quad, sign_bits);
            const int col = (sb << 8) + (ib32 << 5) + chunk * 4;
            uint32_t activation4 = 0;
            std::memcpy(&activation4, act + col, sizeof(activation4));
            const int32_t dot = dp4a_scalar(static_cast<int32_t>(activation4),
                                            static_cast<int32_t>(weights4), 0);
            bsum += dot * ls;
            for (int j = 0; j < 4; ++j) {
                const int8_t w = static_cast<int8_t>(weights4 >> (8 * j));
                exact += static_cast<int64_t>(act[col + j]) * static_cast<int64_t>(w) * ls;
            }
        }
        acc = fma_rn(fmul_rn(load_half(blk), act_d[sb]), static_cast<float>(bsum), acc);
    }
    if (exact_out) *exact_out = exact;
    return acc;
}

// ---------------------------------------------------------------------------
// IQ3_S x Q8_K.
// ---------------------------------------------------------------------------

float iq3s_reference_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_d, int cols, int chunk) {
    const int superblocks = cols >> 8;
    float acc = 0.0f;
    for (int sb = 0; sb < superblocks; ++sb) {
        const uint8_t * blk = row + static_cast<size_t>(sb) * 110u;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        int32_t lane_sum[4] = {0, 0, 0, 0};
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * qs32 = qs + ib32 * 8;
            const uint8_t qh32 = qh[ib32];
            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = (ib32 & 1) ? 1 + 2 * static_cast<int>(scale_byte >> 4)
                                      : 1 + 2 * static_cast<int>(scale_byte & 0x0f);
            for (int j = 0; j < 4; ++j) {
                const int lane = chunk * 4 + j;
                const int group = lane >> 3;
                const int lane8 = lane & 7;
                uint16_t grid_index = 0;
                int grid_lane = 0;
                if (lane8 < 4) {
                    grid_index = static_cast<uint16_t>(
                        static_cast<uint16_t>(qs32[2 * group]) |
                        static_cast<uint16_t>(
                            (static_cast<uint16_t>(qh32) << (8 - 2 * group)) & 0x100u));
                    grid_lane = lane8;
                } else {
                    grid_index = static_cast<uint16_t>(
                        static_cast<uint16_t>(qs32[2 * group + 1]) |
                        static_cast<uint16_t>(
                            (static_cast<uint16_t>(qh32) << (7 - 2 * group)) & 0x100u));
                    grid_lane = lane8 - 4;
                }
                int qv = static_cast<int>(iq3s_grid_byte(grid_index, grid_lane));
                if (signs[ib32 * 4 + group] & (1u << lane8)) qv = -qv;
                const int col = (sb << 8) + (ib32 << 5) + lane;
                lane_sum[j] += static_cast<int32_t>(act[col]) * qv * ls;
            }
        }
        acc = fma_rn(fmul_rn(load_half(blk), act_d[sb]),
                     static_cast<float>((lane_sum[0] + lane_sum[2]) + (lane_sum[1] + lane_sum[3])), acc);
    }
    return acc;
}

float iq3s_dp4a_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_d, int cols, int chunk,
        int64_t * exact_out) {
    const int superblocks = cols >> 8;
    const int group = chunk >> 1;
    const int grid_half = chunk & 1;
    float acc = 0.0f;
    int64_t exact = 0;
    for (int sb = 0; sb < superblocks; ++sb) {
        const uint8_t * blk = row + static_cast<size_t>(sb) * 110u;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * signs = qh + 8;
        const uint8_t * scales = signs + 32;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * qs32 = qs + ib32 * 8;
            const uint8_t qh32 = qh[ib32];
            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = (ib32 & 1) ? 1 + 2 * static_cast<int>(scale_byte >> 4)
                                      : 1 + 2 * static_cast<int>(scale_byte & 0x0f);
            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs32[2 * group + grid_half]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh32) << (8 - 2 * group - grid_half)) & 0x100u));
            const uint32_t weights4 = wd::k_iq3s_grid[grid_index];
            const uint32_t sign_bits =
                static_cast<uint32_t>(signs[ib32 * 4 + group] >> (4 * grid_half));
            const uint32_t signed4 = sign_i8x4(weights4, sign_bits);
            const int col = (sb << 8) + (ib32 << 5) + chunk * 4;
            uint32_t activation4 = 0;
            std::memcpy(&activation4, act + col, sizeof(activation4));
            const int32_t dot = dp4a_scalar(static_cast<int32_t>(activation4),
                                            static_cast<int32_t>(signed4), 0);
            bsum += dot * ls;
            for (int j = 0; j < 4; ++j) {
                const int8_t w = static_cast<int8_t>(signed4 >> (8 * j));
                exact += static_cast<int64_t>(act[col + j]) * static_cast<int64_t>(w) * ls;
            }
        }
        acc = fma_rn(fmul_rn(load_half(blk), act_d[sb]), static_cast<float>(bsum), acc);
    }
    if (exact_out) *exact_out = exact;
    return acc;
}

// ---------------------------------------------------------------------------
// IQ4_NL x Q8_0.
// ---------------------------------------------------------------------------

float iq4nl_reference_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_scales, int cols, int chunk) {
    const int blocks = cols >> 5;
    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t * blk = row + static_cast<size_t>(b) * 18u;
        const float d = fmul_rn(load_half(blk), act_scales[b]);
        int32_t lane_sum[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4; ++j) {
            const int lane = chunk * 4 + j;
            const uint8_t packed = blk[2 + (lane & 15)];
            const uint8_t q = lane < 16 ? (packed & 0x0f) : (packed >> 4);
            lane_sum[j] = static_cast<int32_t>(act[(b << 5) + lane]) *
                          static_cast<int32_t>(kIq4NlLut[q]);
        }
        const int32_t prod = (lane_sum[0] + lane_sum[2]) + (lane_sum[1] + lane_sum[3]);
        if (b & 1) {
            acc_odd = fma_rn(d, static_cast<float>(prod), acc_odd);
        } else {
            acc_even = fma_rn(d, static_cast<float>(prod), acc_even);
        }
    }
    return fadd_rn(acc_even, acc_odd);
}

float iq4nl_dp4a_row_chunk(
        const uint8_t * row, const int8_t * act, const float * act_scales, int cols, int chunk,
        int64_t * exact_out) {
    const int blocks = cols >> 5;
    const int byte_base = (chunk & 3) * 4;
    const bool high_nibble = chunk >= 4;
    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    int64_t exact_even = 0;
    int64_t exact_odd = 0;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t * blk = row + static_cast<size_t>(b) * 18u;
        const float d = fmul_rn(load_half(blk), act_scales[b]);
        const uint8_t * qs = blk + 2 + byte_base;
        uint32_t packed = 0;
        std::memcpy(&packed, qs, sizeof(packed));   // the two 16-bit loads, host-side
        const uint32_t nibbles = high_nibble ? ((packed >> 4) & 0x0f0f0f0fu)
                                             : (packed & 0x0f0f0f0fu);
        const uint32_t weights4 = iq4_nl_pack4(nibbles);
        const int col = (b << 5) + chunk * 4;
        uint32_t activation4 = 0;
        std::memcpy(&activation4, act + col, sizeof(activation4));
        const int32_t prod = dp4a_scalar(static_cast<int32_t>(activation4),
                                         static_cast<int32_t>(weights4), 0);
        int64_t exact = 0;
        for (int j = 0; j < 4; ++j) {
            exact += static_cast<int64_t>(act[col + j]) *
                     static_cast<int64_t>(static_cast<int8_t>(weights4 >> (8 * j)));
        }
        if (b & 1) {
            acc_odd = fma_rn(d, static_cast<float>(prod), acc_odd);
            exact_odd += exact;
        } else {
            acc_even = fma_rn(d, static_cast<float>(prod), acc_even);
            exact_even += exact;
        }
    }
    if (exact_out) *exact_out = exact_even + exact_odd;
    return fadd_rn(acc_even, acc_odd);
}

// ---------------------------------------------------------------------------
// Exhaustive packing checks.
// ---------------------------------------------------------------------------

void check_sign_expansion() {
    // Every IQ2_S magnitude is one of the three bytes of the 1024 grid entries.
    for (int i = 0; i < 1024; ++i) {
        const uint64_t word = wd::k_iq2s_grid[i];
        for (int bits = 0; bits < 16; ++bits) {
            const uint32_t signed4 = sign_i8x4(static_cast<uint32_t>(word & 0xffffffffu),
                                               static_cast<uint32_t>(bits));
            for (int j = 0; j < 4; ++j) {
                const int v = static_cast<int>(static_cast<uint8_t>(word >> (8 * j)));
                const int want = (bits & (1 << j)) ? -v : v;
                check(static_cast<int>(static_cast<int8_t>(signed4 >> (8 * j))) == want,
                      "sign_i8x4 low half");
            }
            const uint32_t signed4_hi = sign_i8x4(static_cast<uint32_t>(word >> 32),
                                                  static_cast<uint32_t>(bits));
            for (int j = 0; j < 4; ++j) {
                const int v = static_cast<int>(static_cast<uint8_t>(word >> (32 + 8 * j)));
                const int want = (bits & (1 << j)) ? -v : v;
                check(static_cast<int>(static_cast<int8_t>(signed4_hi >> (8 * j))) == want,
                      "sign_i8x4 high half");
            }
        }
    }
    for (int i = 0; i < 512; ++i) {
        const uint32_t word = wd::k_iq3s_grid[i];
        for (int bits = 0; bits < 16; ++bits) {
            const uint32_t signed4 = sign_i8x4(word, static_cast<uint32_t>(bits));
            for (int j = 0; j < 4; ++j) {
                const int v = static_cast<int>(static_cast<uint8_t>(word >> (8 * j)));
                const int want = (bits & (1 << j)) ? -v : v;
                check(static_cast<int>(static_cast<int8_t>(signed4 >> (8 * j))) == want,
                      "sign_i8x4 iq3s");
            }
        }
    }
}

void check_iq4_nl_pack() {
    // All 16^4 nibble combinations must reproduce the codebook values byte by
    // byte, including the two's complement bytes dp4a reads as int8.
    for (int a = 0; a < 16; ++a) {
        for (int b = 0; b < 16; ++b) {
            for (int c = 0; c < 16; ++c) {
                for (int d = 0; d < 16; ++d) {
                    const uint32_t nibbles = static_cast<uint32_t>(a) |
                                             (static_cast<uint32_t>(b) << 8) |
                                             (static_cast<uint32_t>(c) << 16) |
                                             (static_cast<uint32_t>(d) << 24);
                    const uint32_t packed = iq4_nl_pack4(nibbles);
                    const uint8_t want[4] = {
                        static_cast<uint8_t>(kIq4NlLut[a]), static_cast<uint8_t>(kIq4NlLut[b]),
                        static_cast<uint8_t>(kIq4NlLut[c]), static_cast<uint8_t>(kIq4NlLut[d])};
                    bool ok = true;
                    for (int j = 0; j < 4; ++j) {
                        ok = ok && (static_cast<uint8_t>(packed >> (8 * j)) == want[j]);
                    }
                    check(ok, "iq4_nl_pack4");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Randomized end-to-end comparison per kernel type.
// ---------------------------------------------------------------------------

std::mt19937 g_rng(0x5eed1234u);

uint8_t rand_byte() { return static_cast<uint8_t>(g_rng() & 0xffu); }
int8_t rand_act() { return static_cast<int8_t>(static_cast<int>(g_rng() % 255u) - 127); }

// A finite normal F16 bit pattern: NaN and infinity would make the bit
// comparison meaningless (and the kernels never see them for a real block
// scale), so the exponent field stays in 1..30.
float rand_half_scale() {
    const uint16_t bits = static_cast<uint16_t>((g_rng() & 0x8000u) |
                                                ((1u + g_rng() % 30u) << 10) |
                                                (g_rng() & 0x3ffu));
    return half_bits_to_float(bits);
}

void check_iq2s(int trials) {
    for (int t = 0; t < trials; ++t) {
        const int cols = 256 * (1 + static_cast<int>(g_rng() % 3u));
        const int superblocks = cols >> 8;
        std::vector<uint8_t> row(static_cast<size_t>(superblocks) * 82u);
        for (auto & b : row) b = rand_byte();
        std::vector<int8_t> act(static_cast<size_t>(cols));
        for (auto & a : act) a = rand_act();
        std::vector<float> act_d(static_cast<size_t>(superblocks));
        for (auto & d : act_d) d = rand_half_scale();

        float ref[8];
        float got[8];
        for (int chunk = 0; chunk < 8; ++chunk) {
            ref[chunk] = iq2s_reference_row_chunk(row.data(), act.data(), act_d.data(), cols, chunk);
            got[chunk] = iq2s_dp4a_row_chunk(row.data(), act.data(), act_d.data(), cols, chunk, nullptr);
        }
        const float ref_value = 0.125f * hsum_float_8(ref);
        const float got_value = 0.125f * hsum_float_8(got);
        check(float_bits(ref_value) == float_bits(got_value), "iq2_s row value");
        for (int chunk = 0; chunk < 8; ++chunk) {
            check(float_bits(ref[chunk]) == float_bits(got[chunk]), "iq2_s chunk accumulator");
        }
    }
}

void check_iq3s(int trials) {
    for (int t = 0; t < trials; ++t) {
        const int cols = 256 * (1 + static_cast<int>(g_rng() % 3u));
        const int superblocks = cols >> 8;
        std::vector<uint8_t> row(static_cast<size_t>(superblocks) * 110u);
        for (auto & b : row) b = rand_byte();
        std::vector<int8_t> act(static_cast<size_t>(cols));
        for (auto & a : act) a = rand_act();
        std::vector<float> act_d(static_cast<size_t>(superblocks));
        for (auto & d : act_d) d = rand_half_scale();

        float ref[8];
        float got[8];
        for (int chunk = 0; chunk < 8; ++chunk) {
            ref[chunk] = iq3s_reference_row_chunk(row.data(), act.data(), act_d.data(), cols, chunk);
            got[chunk] = iq3s_dp4a_row_chunk(row.data(), act.data(), act_d.data(), cols, chunk, nullptr);
        }
        check(float_bits(hsum_float_8(ref)) == float_bits(hsum_float_8(got)), "iq3_s row value");
        for (int chunk = 0; chunk < 8; ++chunk) {
            check(float_bits(ref[chunk]) == float_bits(got[chunk]), "iq3_s chunk accumulator");
        }
    }
}

void check_iq4nl(int trials) {
    for (int t = 0; t < trials; ++t) {
        const int cols = 64 * (1 + static_cast<int>(g_rng() % 4u));
        const int blocks = cols >> 5;
        std::vector<uint8_t> row(static_cast<size_t>(blocks) * 18u);
        for (auto & b : row) b = rand_byte();
        std::vector<int8_t> act(static_cast<size_t>(cols));
        for (auto & a : act) a = rand_act();
        std::vector<float> act_scales(static_cast<size_t>(blocks));
        for (auto & d : act_scales) d = rand_half_scale();

        float ref[8];
        float got[8];
        for (int chunk = 0; chunk < 8; ++chunk) {
            ref[chunk] = iq4nl_reference_row_chunk(row.data(), act.data(), act_scales.data(), cols, chunk);
            got[chunk] = iq4nl_dp4a_row_chunk(row.data(), act.data(), act_scales.data(), cols, chunk, nullptr);
        }
        check(float_bits(hsum_float_8(ref)) == float_bits(hsum_float_8(got)), "iq4_nl row value");
        for (int chunk = 0; chunk < 8; ++chunk) {
            check(float_bits(ref[chunk]) == float_bits(got[chunk]), "iq4_nl chunk accumulator");
        }
    }
}

} // namespace

int main() {
    check_sign_expansion();
    check_iq4_nl_pack();
    check_iq2s(400);
    check_iq3s(400);
    check_iq4nl(400);
    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("FAIL (DP4A integer path differs from the reference order)\n");
        return 1;
    }
    std::printf("PASS (IQ2_S/IQ3_S/IQ4_NL DP4A bit-identical to the reference order)\n");
    return 0;
}
