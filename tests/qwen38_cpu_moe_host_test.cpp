// Host bit-equality check for the Task N CPU MoE kernels (src/qwen38_cpu_moe.cuh).
//
// nvcc is not required: the CPU kernels are plain C++/AVX2, so this test links
// the real header and compares it against two independent references:
//
//   1. every AVX2 row kernel against its scalar twin in the same header, which
//      reconstructs ggml's lane choreography (eight 4-value chunk lanes, the
//      even/odd block parity of IQ4_NL, the hsum_float_8 closing tree) but sums
//      the integer products one at a time. The two must agree bit for bit - the
//      only difference is the order of exact integer additions.
//   2. the integer side against an exact int64 per-element dot product, so a
//      broken sign expansion or nibble order cannot be hidden by both
//      implementations sharing the same mistake.
//
// It also checks the two things the GPU parity run cannot show: that the code
// tables in the header are identical to the ones the CUDA kernels use, and that
// the hand-written F16 conversions agree with the compiler's _Float16.
//
// Run: bash tests/qwen38_cpu_moe_host_check.sh

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "qwen38_cpu_moe.cuh"

// The CUDA headers carry the codebooks as __device__ __constant__ arrays; the
// defines let the host read the very same initialisers.
#define __device__ 
#define __constant__ 
#define __forceinline__ inline
#include "iq2s_grid.cuh"
#include "iq3s_grid.cuh"
#undef __device__
#undef __constant__
#undef __forceinline__

namespace {

using namespace wd::cpu_moe;

int g_failures = 0;
int g_checks = 0;

void check(const char * what, bool ok) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 30) std::printf("  FAIL %s\n", what);
    }
}

bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

std::mt19937 g_rng(20261014u);

float rand_uniform(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g_rng);
}

// ---------------------------------------------------------------------------
// Synthesised weight rows. The layouts are the ggml block layouts; the values
// are uniform so the codebooks, the sign bytes, the high bits and the scales
// are all exercised.
// ---------------------------------------------------------------------------
void fill_iq2_s_row(std::vector<uint8_t> & row, int cols) {
    const int nb = cols / kQK_K;
    row.assign(static_cast<size_t>(nb) * kIq2SBlockBytes, 0);
    for (int b = 0; b < nb; ++b) {
        uint8_t * blk = row.data() + static_cast<size_t>(b) * kIq2SBlockBytes;
        const uint16_t d = f32_to_f16_bits(rand_uniform(0.0005f, 0.05f));
        std::memcpy(blk, &d, sizeof(d));
        for (int i = 0; i < 64; ++i) blk[2 + i] = static_cast<uint8_t>(g_rng() & 0xff);
        for (int i = 0; i < 8; ++i) blk[66 + i] = static_cast<uint8_t>(g_rng() & 0xff);
        for (int i = 0; i < 8; ++i) blk[74 + i] = static_cast<uint8_t>(g_rng() & 0xff);
    }
}

void fill_iq3_s_row(std::vector<uint8_t> & row, int cols) {
    const int nb = cols / kQK_K;
    row.assign(static_cast<size_t>(nb) * kIq3SBlockBytes, 0);
    for (int b = 0; b < nb; ++b) {
        uint8_t * blk = row.data() + static_cast<size_t>(b) * kIq3SBlockBytes;
        const uint16_t d = f32_to_f16_bits(rand_uniform(0.0005f, 0.05f));
        std::memcpy(blk, &d, sizeof(d));
        for (int i = 0; i < 64; ++i) blk[2 + i] = static_cast<uint8_t>(g_rng() & 0xff);
        for (int i = 0; i < 8; ++i) blk[66 + i] = static_cast<uint8_t>(g_rng() & 0xff);
        for (int i = 0; i < 32; ++i) blk[74 + i] = static_cast<uint8_t>(g_rng() & 0xff);
        for (int i = 0; i < 4; ++i) blk[106 + i] = static_cast<uint8_t>(g_rng() & 0xff);
    }
}

void fill_iq4_nl_row(std::vector<uint8_t> & row, int cols) {
    const int nb = cols / kQK4_NL;
    row.assign(static_cast<size_t>(nb) * kIq4NlBlockBytes, 0);
    for (int b = 0; b < nb; ++b) {
        uint8_t * blk = row.data() + static_cast<size_t>(b) * kIq4NlBlockBytes;
        const uint16_t d = f32_to_f16_bits(rand_uniform(0.0005f, 0.05f));
        std::memcpy(blk, &d, sizeof(d));
        for (int i = 0; i < 16; ++i) blk[2 + i] = static_cast<uint8_t>(g_rng() & 0xff);
    }
}

void fill_q8_0_row(std::vector<uint8_t> & row, int cols) {
    const int nb = cols / kQK8_0;
    row.assign(static_cast<size_t>(nb) * kQ8_0BlockBytes, 0);
    for (int b = 0; b < nb; ++b) {
        uint8_t * blk = row.data() + static_cast<size_t>(b) * kQ8_0BlockBytes;
        const uint16_t d = f32_to_f16_bits(rand_uniform(0.0005f, 0.05f));
        std::memcpy(blk, &d, sizeof(d));
        for (int i = 0; i < 32; ++i) blk[2 + i] = static_cast<uint8_t>(g_rng() & 0xff);
    }
}

// Exact integer reconstructions of the row dots, decoded by hand so a broken
// sign expansion, nibble order or group scale cannot hide behind the AVX2 and
// scalar implementations sharing the same mistake.
//
// The kernels apply one integer sum per 32-value block to a per-lane float
// accumulator (ggml's eight 4-value chunk lanes and its hsum_float_8 closing
// tree), so the decoders below return the per-lane partials of one or more
// blocks and the checks rebuild the float with the very same lane tree and the
// same one-FMA-per-superblock step.
float hsum_lanes(const float lane[8]) {
    const float r0 = lane[0] + lane[4];
    const float r1 = lane[1] + lane[5];
    const float r2 = lane[2] + lane[6];
    const float r3 = lane[3] + lane[7];
    return (r0 + r2) + (r1 + r3);
}

// Lane c is the 4-value chunk c of every 32-value block. Values 0..15 (chunks
// 0..3) of a block use 1 + 2*(scales[b] & 0x0f), values 16..31 the high nibble.
void exact_iq2_s_lane_partials(const uint8_t * blk, const q8_K & y, int64_t lane[8]) {
    const uint8_t * qs = blk + 2;
    const uint8_t * qh = qs + 64;
    const uint8_t * signs = qs + 32;
    const uint8_t * scales = qh + 8;
    for (int c = 0; c < 8; ++c) {
        const int group = c >> 1;
        const int grid_half = c & 1;
        int64_t total = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const int ls = 1 + 2 * ((c < 4) ? (scales[ib32] & 0x0f) : (scales[ib32] >> 4));
            const uint16_t gi = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300));
            for (int j = 0; j < 4; ++j) {
                const int value = ib32 * 32 + c * 4 + j;
                const int magnitude =
                    static_cast<int>((kIq2sGrid[gi] >> (32 * grid_half + 8 * j)) & 0xffu);
                const int sign = ((signs[ib32 * 4 + group] >> (4 * grid_half + j)) & 1u) ? -1 : 1;
                total += static_cast<int64_t>(y.qs[value]) * sign * magnitude * ls;
            }
        }
        lane[c] = total;
    }
}

// One scale byte per two 32-value blocks: block 2t takes the low nibble of
// scales[t], block 2t+1 the high nibble, applied to the whole 32-value block.
void exact_iq3_s_lane_partials(const uint8_t * blk, const q8_K & y, int64_t lane[8]) {
    const uint8_t * qs = blk + 2;
    const uint8_t * qh = qs + 64;
    const uint8_t * signs = qh + 8;
    const uint8_t * scales = signs + 32;
    for (int c = 0; c < 8; ++c) {
        const int group = c >> 1;
        const int grid_half = c & 1;
        int64_t total = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t scale_byte = scales[ib32 >> 1];
            const int ls = 1 + 2 * ((ib32 & 1) ? (scale_byte >> 4) : (scale_byte & 0x0f));
            const uint16_t gi = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 8 + 2 * group + grid_half]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group - grid_half)) & 0x100u));
            for (int j = 0; j < 4; ++j) {
                const int value = ib32 * 32 + c * 4 + j;
                const int magnitude = static_cast<int>((kIq3sGrid[gi] >> (8 * j)) & 0xffu);
                const int sign = ((signs[ib32 * 4 + group] >> (4 * grid_half + j)) & 1u) ? -1 : 1;
                total += static_cast<int64_t>(y.qs[value]) * sign * magnitude * ls;
            }
        }
        lane[c] = total;
    }
}

// Lane l < 4 holds values 4l..4l+3 from the low nibbles of qs[4l..4l+3]; lane
// l >= 4 holds values 4l..4l+3 from the high nibbles of qs[4l-16..4l-13].
void exact_iq4_nl_lane_partials(const uint8_t * blk, const q8_0 & y, int64_t lane[8]) {
    const int8_t * lut = iq4_nl_lut();
    for (int l = 0; l < 8; ++l) {
        const int high = l >> 2;
        int64_t total = 0;
        for (int j = 0; j < 4; ++j) {
            const int idx = 4 * (l & 3) + j;
            const uint8_t byte = blk[2 + idx];
            const uint8_t nibble = high ? static_cast<uint8_t>(byte >> 4)
                                        : static_cast<uint8_t>(byte & 0x0f);
            total += static_cast<int64_t>(y.qs[16 * high + idx]) * lut[nibble];
        }
        lane[l] = total;
    }
}
}  // namespace

int main() {
    std::printf("cpu-moe host check\n");

    // ---------------------------------------------------------------------
    // Codebooks identical to the CUDA tables.
    // ---------------------------------------------------------------------
    {
        bool ok2 = true;
        for (int i = 0; i < 1024; ++i) {
            if (kIq2sGrid[i] != wd::k_iq2s_grid[i]) ok2 = false;
        }
        bool ok3 = true;
        for (int i = 0; i < 512; ++i) {
            if (kIq3sGrid[i] != wd::k_iq3s_grid[i]) ok3 = false;
        }
        check("casino.iq2s_grid matches src/iq2s_grid.cuh", ok2);
        check("iq3s_grid matches src/iq3s_grid.cuh", ok3);
    }

    // ---------------------------------------------------------------------
    // F16 conversions against the compiler's _Float16.
    // ---------------------------------------------------------------------
    {
        bool fwd = true;
        bool back = true;
        for (int i = 0; i < 200000; ++i) {
            const uint16_t bits = static_cast<uint16_t>(g_rng() & 0xffff);
            _Float16 h;
            std::memcpy(&h, &bits, sizeof(h));
            const float expect = static_cast<float>(h);
            const float got = f16_to_f32(bits);
            if (!(same_bits(expect, got) || (std::isnan(expect) && std::isnan(got)))) fwd = false;
        }
        for (int i = 0; i < 200000; ++i) {
            const float v = rand_uniform(-70000.0f, 70000.0f);
            const _Float16 h = static_cast<_Float16>(v);
            uint16_t expect;
            std::memcpy(&expect, &h, sizeof(expect));
            if (f32_to_f16_bits(v) != expect) back = false;
        }
        check("f16_to_f32 == (float)(_Float16)", fwd);
        check("f32_to_f16_bits == (_Float16) value", back);
    }

    // ---------------------------------------------------------------------
    // Activation quantisers: layout and the zero/edge blocks.
    // ---------------------------------------------------------------------
    {
        std::vector<float> x(512, 0.0f);
        for (int i = 0; i < 512; ++i) x[static_cast<size_t>(i)] = rand_uniform(-1.5f, 1.5f);
        std::vector<q8_K> y(2);
        quantize_q8_K(x.data(), y.data(), 2);
        bool bsums_ok = true;
        bool range_ok = true;
        bool scale_ok = true;
        for (int b = 0; b < 2; ++b) {
            for (int j = 0; j < 16; ++j) {
                int sum = 0;
                for (int i = 0; i < 16; ++i) sum += y[b].qs[j * 16 + i];
                if (sum != y[b].bsums[j]) bsums_ok = false;
            }
            float amax = 0.0f;
            for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(x[static_cast<size_t>(b) * 256 + i]));
            for (int i = 0; i < 256; ++i) {
                if (static_cast<int>(y[b].qs[i]) < -127) range_ok = false;
            }
            const float expect_d = 1.0f / (-127.0f / amax);
            if (y[b].d != expect_d) scale_ok = false;
        }
        check("q8_K bsums are the 16-value integer sums", bsums_ok);
        check("q8_K quants stay in [-127, 127]", range_ok);
        check("q8_K d = 1 / iscale", scale_ok);

        std::vector<float> zero(256, 0.0f);
        std::vector<q8_K> zy(1);
        quantize_q8_K(zero.data(), zy.data(), 1);
        bool zero_ok = zy[0].d == 0.0f;
        for (int i = 0; i < 256; ++i) zero_ok = zero_ok && zy[0].qs[i] == 0;
        for (int i = 0; i < 16; ++i) zero_ok = zero_ok && zy[0].bsums[i] == 0;
        check("q8_K zero block -> zero scale, quants and bsums", zero_ok);

        std::vector<q8_0> qb(1);
        std::vector<float> tiny(32, 0.0f);
        tiny[7] = 3.0f;
        quantize_q8_0(tiny.data(), qb.data(), 1);
        // ggml stores the block scale as the fp16 of amax/127, so the stored
        // bits are the rounded half and not the exact float quotient.
        bool q8_0_ok = qb[0].d == f32_to_f16_bits(3.0f / 127.0f) && qb[0].qs[7] == 127;
        for (int i = 0; i < 32; ++i) {
            if (i != 7 && qb[0].qs[i] != 0) q8_0_ok = false;
        }
        check("q8_0 block scale (fp16 bits) and peak quant", q8_0_ok);
    }

    // ---------------------------------------------------------------------
    // The routed-sum accumulation is ggml's mul-then-add (two roundings), not a
    // fused multiply-add: the CUDA twin accumulates with
    // q38_add_rn(accum, q38_mul_rn(x, weight)), and an FMA here would move the
    // sum by one ulp per selected expert.
    // ---------------------------------------------------------------------
    {
        const float a = 1.1109310388565063f;
        const float b = 1.1190425157546997f;
        const float c = -1.3467210531234741f;
        const float product = a * b;   // -ffp-contract=off keeps the two steps apart
        const float expect = product + c;
        check("routed sum accumulates mul-then-add, not FMA",
              same_bits(add_rn(mul_rn(a, b), c), expect) &&
                  !same_bits(std::fma(a, b, c), expect));
    }

    // ---------------------------------------------------------------------
    // Row kernels: AVX2 (or the scalar fallback) vs the scalar twin vs the
    // exact int64 dot.
    // ---------------------------------------------------------------------
    const int cols_list[] = {256, 512, 2560, 5120};
    for (int cols : cols_list) {
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<float> act(static_cast<size_t>(cols));
            for (int i = 0; i < cols; ++i) act[static_cast<size_t>(i)] = rand_uniform(-2.5f, 2.5f);

            std::vector<q8_K> ya(static_cast<size_t>(cols / kQK_K));
            quantize_q8_K(act.data(), ya.data(), cols / kQK_K);

            std::vector<uint8_t> row2;
            fill_iq2_s_row(row2, cols);
            const float s2 = scalar_iq2_s_q8_K(row2.data(), ya.data(), cols);
            const float a2 = row_iq2_s_q8_K(row2.data(), ya.data(), cols);
            check("iq2_s AVX2 == scalar", same_bits(s2, a2));
            if (cols == 256) {
                int64_t partial[8];
                exact_iq2_s_lane_partials(row2.data(), ya[0], partial);
                const float d =
                    f16_to_f32(*reinterpret_cast<const uint16_t *>(row2.data())) * ya[0].d;
                float lane[8];
                for (int c = 0; c < 8; ++c) {
                    lane[c] = std::fma(d, static_cast<float>(partial[c]), 0.0f);
                }
                const float expect = 0.125f * hsum_lanes(lane);
                check("iq2_s integer side == exact int64 lanes", same_bits(expect, a2));
            }

            // The fused gate+up kernel shares the activation decode between the
            // two projections; it must return exactly what the two independent
            // row kernels return (a shared scale/sign slip is invisible to a
            // check that only looks at one projection at a time).
            {
                std::vector<uint8_t> row_up;
                fill_iq2_s_row(row_up, cols);
                float gv = 0.0f;
                float uv = 0.0f;
                dot_gate_up_row(kIq2S, row2.data(), row_up.data(), ya.data(), cols, &gv, &uv);
                check("iq2_s fused gate+up == two independent rows",
                      same_bits(gv, row_iq2_s_q8_K(row2.data(), ya.data(), cols)) &&
                          same_bits(uv, row_iq2_s_q8_K(row_up.data(), ya.data(), cols)));
            }

            std::vector<uint8_t> row3;
            fill_iq3_s_row(row3, cols);
            const float s3 = scalar_iq3_s_q8_K(row3.data(), ya.data(), cols);
            const float a3 = row_iq3_s_q8_K(row3.data(), ya.data(), cols);
            check("iq3_s AVX2 == scalar", same_bits(s3, a3));
            if (cols == 256) {
                int64_t partial[8];
                exact_iq3_s_lane_partials(row3.data(), ya[0], partial);
                const float d =
                    f16_to_f32(*reinterpret_cast<const uint16_t *>(row3.data())) * ya[0].d;
                float lane[8];
                for (int c = 0; c < 8; ++c) {
                    lane[c] = std::fma(d, static_cast<float>(partial[c]), 0.0f);
                }
                const float expect = hsum_lanes(lane);
                check("iq3_s integer side == exact int64 lanes", same_bits(expect, a3));
            }

            std::vector<q8_0> y0(static_cast<size_t>(cols / kQK8_0));
            quantize_q8_0(act.data(), y0.data(), static_cast<int>(y0.size()));

            std::vector<uint8_t> row4;
            fill_iq4_nl_row(row4, cols);
            const float s4 = scalar_iq4_nl_q8_0(row4.data(), y0.data(), cols);
            const float a4 = row_iq4_nl_q8_0(row4.data(), y0.data(), cols);
            check("iq4_nl AVX2 == scalar", same_bits(s4, a4));
            if (cols == 256) {
                // Eight 32-value blocks: even blocks accumulate in the first
                // lane vector, odd blocks in the second, exactly like the
                // reference's accum1/accum2 pair split.
                int64_t partial[8][8];
                for (int b = 0; b < 8; ++b) {
                    exact_iq4_nl_lane_partials(
                        row4.data() + static_cast<size_t>(b) * kIq4NlBlockBytes, y0[b], partial[b]);
                }
                float acc_even[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float acc_odd[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (int b = 0; b < 8; ++b) {
                    const uint8_t * blk = row4.data() + static_cast<size_t>(b) * kIq4NlBlockBytes;
                    const float d = f16_to_f32(*reinterpret_cast<const uint16_t *>(blk)) *
                                    f16_to_f32(y0[b].d);
                    float * acc = (b & 1) ? acc_odd : acc_even;
                    for (int l = 0; l < 8; ++l) {
                        acc[l] = std::fma(d, static_cast<float>(partial[b][l]), acc[l]);
                    }
                }
                float lane[8];
                for (int l = 0; l < 8; ++l) lane[l] = acc_even[l] + acc_odd[l];
                check("iq4_nl integer side == exact int64 lanes", same_bits(hsum_lanes(lane), a4));
            }

            std::vector<uint8_t> row5;
            fill_q8_0_row(row5, cols);
            const float s5 = scalar_q8_0_q8_0(row5.data(), y0.data(), cols);
            const float a5 = row_q8_0_q8_0(row5.data(), y0.data(), cols);
            check("q8_0 AVX2 == scalar", same_bits(s5, a5));
        }
    }

    // ---------------------------------------------------------------------
    // The layer runner: ten experts, one thread, and the same ten with a pool.
    // The results must be bit-identical, which is the property the engine's
    // parity depends on.
    // ---------------------------------------------------------------------
    {
        const int hidden = 2560;
        const int ffn = 640;
        const int k = 10;
        const int experts = 32;

        LayerWeights w;
        w.hidden = hidden;
        w.expert_ffn = ffn;
        w.gate_type = kIq2S;
        w.up_type = kIq2S;
        w.down_type = kIq4Nl;
        const size_t gate_row = row_bytes(kIq2S, hidden);
        const size_t down_row = row_bytes(kIq4Nl, ffn);
        const size_t gate_expert = gate_row * static_cast<size_t>(ffn);
        const size_t down_expert = down_row * static_cast<size_t>(hidden);
        std::vector<uint8_t> gate(static_cast<size_t>(experts) * gate_expert);
        std::vector<uint8_t> up(static_cast<size_t>(experts) * gate_expert);
        std::vector<uint8_t> down(static_cast<size_t>(experts) * down_expert);
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < ffn; ++r) {
                std::vector<uint8_t> tmp;
                fill_iq2_s_row(tmp, hidden);
                std::memcpy(gate.data() + static_cast<size_t>(e) * gate_expert + static_cast<size_t>(r) * gate_row,
                            tmp.data(), tmp.size());
                fill_iq2_s_row(tmp, hidden);
                std::memcpy(up.data() + static_cast<size_t>(e) * gate_expert + static_cast<size_t>(r) * gate_row,
                            tmp.data(), tmp.size());
            }
            for (int r = 0; r < hidden; ++r) {
                std::vector<uint8_t> tmp;
                fill_iq4_nl_row(tmp, ffn);
                std::memcpy(down.data() + static_cast<size_t>(e) * down_expert + static_cast<size_t>(r) * down_row,
                            tmp.data(), tmp.size());
            }
        }
        w.gate = gate.data();
        w.up = up.data();
        w.down = down.data();
        w.gate_expert_stride = gate_expert;
        w.up_expert_stride = gate_expert;
        w.down_expert_stride = down_expert;

        std::vector<float> x(static_cast<size_t>(hidden));
        for (int i = 0; i < hidden; ++i) x[static_cast<size_t>(i)] = rand_uniform(-2.0f, 2.0f);
        std::vector<int> ids(static_cast<size_t>(k));
        for (int j = 0; j < k; ++j) ids[static_cast<size_t>(j)] = j * 3;
        std::vector<float> wts(static_cast<size_t>(k));
        float wsum = 0.0f;
        for (int j = 0; j < k; ++j) {
            wts[static_cast<size_t>(j)] = rand_uniform(0.01f, 1.0f);
            wsum += wts[static_cast<size_t>(j)];
        }
        for (int j = 0; j < k; ++j) wts[static_cast<size_t>(j)] /= wsum;

        std::vector<float> seq(static_cast<size_t>(hidden), 0.0f);
        {
            LayerScratch scratch;
            scratch.reset(k, hidden, ffn);
            ThreadPool pool(1);
            run_layer(w, pool, x.data(), ids.data(), wts.data(), k, scratch, seq.data());
        }
        std::vector<float> par(static_cast<size_t>(hidden), 0.0f);
        {
            LayerScratch scratch;
            scratch.reset(k, hidden, ffn);
            ThreadPool pool(8);
            run_layer(w, pool, x.data(), ids.data(), wts.data(), k, scratch, par.data());
        }
        bool identical = true;
        for (int i = 0; i < hidden; ++i) {
            if (!same_bits(seq[static_cast<size_t>(i)], par[static_cast<size_t>(i)])) identical = false;
        }
        check("run_layer: 1 thread == 8 threads (bit-identical)", identical);

        float max_abs = 0.0f;
        for (int i = 0; i < hidden; ++i) max_abs = std::max(max_abs, std::fabs(seq[static_cast<size_t>(i)]));
        check("run_layer produces a non-trivial routed sum", max_abs > 1e-6f);

        // Negative control: swapping two expert weights must change the result,
        // so the equality above is not vacuous.
        std::vector<float> swapped = wts;
        std::swap(swapped[0], swapped[1]);
        std::vector<float> alt(static_cast<size_t>(hidden), 0.0f);
        {
            LayerScratch scratch;
            scratch.reset(k, hidden, ffn);
            ThreadPool pool(1);
            run_layer(w, pool, x.data(), ids.data(), swapped.data(), k, scratch, alt.data());
        }
        bool differs = false;
        for (int i = 0; i < hidden; ++i) {
            if (!same_bits(seq[static_cast<size_t>(i)], alt[static_cast<size_t>(i)])) differs = true;
        }
        check("negative control: swapped expert weights change the sum", differs);
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
