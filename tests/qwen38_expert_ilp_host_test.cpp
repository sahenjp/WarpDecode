// Host mirror of the Task W ILP (software-pipelined) routed-expert GEMV row
// routines (src/lowbit.cuh: iq2_s_q8_k_row_dp4a_ilp<USE_SMEM, ILP>,
// iq4_nl_q8_0_act_row_dp4a_ilp<ILP>).
//
// The kernels only hoist the loads and the integer partial of ILP superblocks
// (or ILP same-parity blocks) ahead of the accumulator chain. Every rounded
// operation the accumulator sees - fmaf(d_w * x_d[sb], partial[sb], acc) for
// increasing sb, and fmaf(d, dot, acc) for increasing same-parity blocks - is
// unchanged, so the result has to be bit-identical to the sequential reference,
// including the group/tail split and rows shorter than ILP.
//
// This test mirrors both loop structures at the real byte offsets, records the
// ordered (index, scale, integer partial) sequence each variant feeds to the
// accumulator, and compares the sequences and the resulting float chain bit for
// bit. A one-byte mutation of the row is the positive control.
//
// Build/run: bash tests/qwen38_expert_ilp_host_check.sh
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char * what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

uint32_t bits(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

float half_from_bits(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
    const uint32_t exp = static_cast<uint32_t>(h >> 10) & 0x1fu;
    const uint32_t man = static_cast<uint32_t>(h) & 0x3ffu;
    float v = 0.0f;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
    } else if (exp != 31) {
        v = std::ldexp(static_cast<float>(man + 1024u), static_cast<int>(exp) - 25);
    }
    return sign ? -v : v;
}

// lowbit_half(): the block scale is the first two bytes of a block.
float lowbit_half(const uint8_t * block) {
    uint16_t h = 0;
    std::memcpy(&h, block, sizeof(h));
    return half_from_bits(h);
}

// __dp4a(a, b, 0) over four int8 lanes.
int dp4a4(const uint8_t * a, const uint8_t * b) {
    int sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int>(static_cast<int8_t>(a[i])) *
               static_cast<int>(static_cast<int8_t>(b[i]));
    }
    return sum;
}

// ggml kvalues_iq4nl, as src/lowbit.cuh uses it.
int8_t iq4_lut(uint8_t nibble) {
    static const int8_t lut[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return lut[nibble & 0x0f];
}

// The shared-memory kernel (Task W2) fills
//   lut[i] = (int8_t)iq4_nl_lut_value((uint8_t)i)      for i in 0..15
// and then reads lut[nibble] instead of going through the codebook switch. i is
// already in 0..15, so iq4_nl_lut_value(i) is lut[i] and pack4_smem has to equal
// pack4 for every 4-nibble group; the loop in main() checks all 65536 of them.
uint32_t pack4_codebook(uint32_t nibbles) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        out |= static_cast<uint32_t>(static_cast<uint8_t>(
                   iq4_lut(static_cast<uint8_t>((nibbles >> (8 * i)) & 0x0fu)))) << (8 * i);
    }
    return out;
}

uint32_t pack4_shared_lut(uint32_t nibbles, const int8_t * lut) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const uint8_t nibble = static_cast<uint8_t>((nibbles >> (8 * i)) & 0x0fu);
        out |= static_cast<uint32_t>(static_cast<uint8_t>(lut[nibble])) << (8 * i);
    }
    return out;
}

// The reference reads the activation scale as a half.
float iq4_scale_i8(const int8_t * act_scales) {
    uint16_t h = 0;
    std::memcpy(&h, act_scales, sizeof(h));
    return half_from_bits(h);
}

struct Iq4Step {
    int block;
    float scale;
    int dot;
};

// Mirrors iq4_nl_q8_0_act_row_dp4a for one (chunk, parity) lane; the returned
// chain is the order the accumulator consumes, and the final entry is the
// accumulated value of that lane.
struct Iq4Chain {
    std::vector<Iq4Step> steps;
    float value;
    int max_block_read;
};

Iq4Chain iq4_row(const uint8_t * row_ptr, const int8_t * act, const int8_t * act_scales,
                 int blocks_per_row, int chunk, int parity, int ilp) {
    const int byte_base = (chunk & 3) * 4;
    const bool high_nibble = chunk >= 4;

    auto one = [&](int block) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(block) * 18u;
        const float d = lowbit_half(blk) *
                        iq4_scale_i8(act_scales + static_cast<size_t>(block) * 2u);
        const uint8_t * qs = blk + 2 + byte_base;
        const uint8_t * nibble_bytes = qs;
        uint8_t weights4[4];
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = nibble_bytes[i];
            const uint8_t nibble = high_nibble ? static_cast<uint8_t>(byte >> 4)
                                               : static_cast<uint8_t>(byte & 0x0f);
            weights4[i] = static_cast<uint8_t>(iq4_lut(nibble));
        }
        const uint8_t * a4 = reinterpret_cast<const uint8_t *>(act) +
                             static_cast<size_t>(block) * 32u +
                             static_cast<size_t>(chunk) * 4u;
        return Iq4Step{block, d, dp4a4(a4, weights4)};
    };

    Iq4Chain chain;
    chain.value = 0.0f;
    chain.max_block_read = -1;
    if (ilp >= 2) {
        int b = parity;
        for (; b + 2 * (ilp - 1) < blocks_per_row; b += 2 * ilp) {
            std::vector<Iq4Step> group;
            for (int u = 0; u < ilp; ++u) {
                group.push_back(one(b + 2 * u));
                chain.max_block_read = std::max(chain.max_block_read, b + 2 * u);
            }
            for (const Iq4Step & step : group) {
                chain.value = std::fma(step.scale, static_cast<float>(step.dot), chain.value);
                chain.steps.push_back(step);
            }
        }
        for (; b < blocks_per_row; b += 2) {
            const Iq4Step step = one(b);
            chain.max_block_read = std::max(chain.max_block_read, b);
            chain.value = std::fma(step.scale, static_cast<float>(step.dot), chain.value);
            chain.steps.push_back(step);
        }
    } else {
        for (int b = parity; b < blocks_per_row; b += 2) {
            const Iq4Step step = one(b);
            chain.max_block_read = std::max(chain.max_block_read, b);
            chain.value = std::fma(step.scale, static_cast<float>(step.dot), chain.value);
            chain.steps.push_back(step);
        }
    }
    return chain;
}

struct Iq2Step {
    int superblock;
    float scale;
    int partial;
};

struct Iq2Chain {
    std::vector<Iq2Step> steps;
    float value;
    int max_superblock_read;
};

// Mirrors iq2_s_q8_k_row_dp4a for one chunk lane (phase 0..3 are the four lanes
// that later combine through the two shuffles).
Iq2Chain iq2_row(const uint8_t * row_ptr, const int8_t * act, const float * act_d,
                 int superblocks_per_row, int chunk, int ilp,
                 const uint64_t * grid_table) {
    const int group = chunk >> 1;
    const int grid_half = chunk & 1;
    const bool scale_low_nibble = chunk < 4;

    auto phase_partial = [&](int sb, int phase) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        const uint8_t * qs = blk + 2;
        const uint8_t * qh = qs + 64;
        const uint8_t * scales = qh + 8;
        const uint8_t * signs = qs + 32;
        const int8_t * a = act + static_cast<size_t>(sb) * 256u;
        int sum = 0;
        for (int p = 0; p < 2; ++p) {
            const int ib32 = phase + 4 * p;
            const uint8_t scale_byte = scales[ib32];
            const int ls = scale_low_nibble
                ? 1 + 2 * static_cast<int>(scale_byte & 0x0f)
                : 1 + 2 * static_cast<int>(scale_byte >> 4);
            const uint16_t grid_index = static_cast<uint16_t>(
                static_cast<uint16_t>(qs[ib32 * 4 + group]) |
                static_cast<uint16_t>(
                    (static_cast<uint16_t>(qh[ib32]) << (8 - 2 * group)) & 0x300u));
            const unsigned quad = static_cast<unsigned>(
                grid_table[grid_index] >> (32 * grid_half));
            const uint8_t sign_bits = static_cast<uint8_t>(
                signs[ib32 * 4 + group] >> (4 * grid_half));
            uint8_t weights4[4];
            for (int i = 0; i < 4; ++i) {
                const uint8_t value = static_cast<uint8_t>((quad >> (8 * i)) & 0xffu);
                const int8_t sign = static_cast<int8_t>(
                    (sign_bits & (1u << i)) ? -1 : 1);
                weights4[i] = static_cast<uint8_t>(
                    static_cast<int8_t>(static_cast<int8_t>(value) * sign));
            }
            const uint8_t * a4 = reinterpret_cast<const uint8_t *>(a) +
                                 static_cast<size_t>(ib32) * 32u +
                                 static_cast<size_t>(chunk) * 4u;
            sum += dp4a4(a4, weights4) * ls;
        }
        return sum;
    };

    auto superblock = [&](int sb) {
        // The two shuffles of the kernel: phases 0+1 and 2+3, then those halves.
        const int p0 = phase_partial(sb, 0);
        const int p1 = phase_partial(sb, 1);
        const int p2 = phase_partial(sb, 2);
        const int p3 = phase_partial(sb, 3);
        const int partial = (p0 + p1) + (p2 + p3);
        const uint8_t * blk = row_ptr + static_cast<size_t>(sb) * 82u;
        return Iq2Step{sb, lowbit_half(blk) * act_d[sb], partial};
    };

    Iq2Chain chain;
    chain.value = 0.0f;
    chain.max_superblock_read = -1;
    if (ilp >= 2) {
        int sb = 0;
        for (; sb + ilp <= superblocks_per_row; sb += ilp) {
            std::vector<Iq2Step> group;
            for (int u = 0; u < ilp; ++u) {
                group.push_back(superblock(sb + u));
                chain.max_superblock_read = std::max(chain.max_superblock_read, sb + u);
            }
            for (const Iq2Step & step : group) {
                chain.value =
                    std::fma(step.scale, static_cast<float>(step.partial), chain.value);
                chain.steps.push_back(step);
            }
        }
        for (; sb < superblocks_per_row; ++sb) {
            const Iq2Step step = superblock(sb);
            chain.max_superblock_read = std::max(chain.max_superblock_read, sb);
            chain.value = std::fma(step.scale, static_cast<float>(step.partial), chain.value);
            chain.steps.push_back(step);
        }
    } else {
        for (int sb = 0; sb < superblocks_per_row; ++sb) {
            const Iq2Step step = superblock(sb);
            chain.max_superblock_read = std::max(chain.max_superblock_read, sb);
            chain.value = std::fma(step.scale, static_cast<float>(step.partial), chain.value);
            chain.steps.push_back(step);
        }
    }
    return chain;
}

bool same_iq4(const Iq4Chain & a, const Iq4Chain & b) {
    if (a.steps.size() != b.steps.size()) return false;
    if (bits(a.value) != bits(b.value)) return false;
    for (size_t i = 0; i < a.steps.size(); ++i) {
        if (a.steps[i].block != b.steps[i].block) return false;
        if (bits(a.steps[i].scale) != bits(b.steps[i].scale)) return false;
        if (a.steps[i].dot != b.steps[i].dot) return false;
    }
    return true;
}

bool same_iq2(const Iq2Chain & a, const Iq2Chain & b) {
    if (a.steps.size() != b.steps.size()) return false;
    if (bits(a.value) != bits(b.value)) return false;
    for (size_t i = 0; i < a.steps.size(); ++i) {
        if (a.steps[i].superblock != b.steps[i].superblock) return false;
        if (bits(a.steps[i].scale) != bits(b.steps[i].scale)) return false;
        if (a.steps[i].partial != b.steps[i].partial) return false;
    }
    return true;
}

} // namespace

int main() {
    std::mt19937 rng(20261015u);
    const int kIlpValues[3] = {2, 4, 8};

    // IQ4_NL down projection: 20 blocks per row of 640 columns (blocks_per_row
    // is what a row of the real 2560x640 expert has), 16 lanes per row.
    {
        const int max_blocks = 24;
        std::vector<uint8_t> row(static_cast<size_t>(max_blocks) * 18u);
        std::vector<int8_t> act(static_cast<size_t>(max_blocks) * 32u);
        std::vector<int8_t> act_scales(static_cast<size_t>(max_blocks) * 2u);
        for (uint8_t & v : row) v = static_cast<uint8_t>(rng() & 0xffu);
        for (int8_t & v : act) v = static_cast<int8_t>(rng() & 0xffu);
        for (size_t i = 0; i + 1 < act_scales.size(); i += 2) {
            const uint16_t h = static_cast<uint16_t>(0x2000u | (rng() & 0x3ffu));
            std::memcpy(&act_scales[i], &h, sizeof(h));
        }

        int compared = 0;
        for (int blocks = 1; blocks <= max_blocks; ++blocks) {
            for (int lane = 0; lane < 16; ++lane) {
                const int chunk = lane & 7;
                const int parity = lane >> 3;
                const Iq4Chain ref = iq4_row(row.data(), act.data(), act_scales.data(),
                                             blocks, chunk, parity, 0);
                for (int ilp : kIlpValues) {
                    const Iq4Chain got = iq4_row(row.data(), act.data(), act_scales.data(),
                                                 blocks, chunk, parity, ilp);
                    check(same_iq4(ref, got), "IQ4_NL ILP chain differs from the reference");
                    check(got.max_block_read < blocks, "IQ4_NL ILP read past the row");
                    check(got.steps.size() == ref.steps.size(),
                          "IQ4_NL ILP step count differs");
                    ++compared;
                }
            }
        }
        std::printf("IQ4_NL rows compared: %d chains\n", compared);

        // Positive control: flipping one weight byte has to move the chain.
        {
            std::vector<uint8_t> mutated = row;
            mutated[2] ^= 0x11u;   // qs byte 0, the first nibble chunk 0 reads
            const Iq4Chain ref = iq4_row(row.data(), act.data(), act_scales.data(),
                                         20, 0, 0, 0);
            const Iq4Chain got = iq4_row(mutated.data(), act.data(), act_scales.data(),
                                         20, 0, 0, 4);
            check(!same_iq4(ref, got), "IQ4_NL positive control did not change the chain");
        }
    }

    // Task W2: the shared-memory codebook is a copy of the constexpr table, so
    // the expansion it feeds to __dp4a must be identical for every input.
    {
        int8_t lut[16];
        for (int i = 0; i < 16; ++i) {
            lut[i] = static_cast<int8_t>(iq4_lut(static_cast<uint8_t>(i)));
        }
        int mismatched = 0;
        for (uint32_t n = 0; n < 65536u; ++n) {
            if (pack4_codebook(n) != pack4_shared_lut(n, lut)) ++mismatched;
        }
        check(mismatched == 0, "shared-memory codebook differs from the constexpr table");
        check(pack4_shared_lut(0x00000000u, lut) == pack4_codebook(0x00000000u),
              "shared-memory codebook nibble 0");
        check(pack4_shared_lut(0x0f0f0f0fu, lut) == pack4_codebook(0x0f0f0f0fu),
              "shared-memory codebook nibble 15");
        std::printf("shared-LUT 4-nibble groups compared: 65536\n");
    }

    // IQ2_S gate/up: 10 superblocks per row of 2560 columns.
    {
        const int max_superblocks = 12;
        std::vector<uint8_t> row(static_cast<size_t>(max_superblocks) * 82u);
        std::vector<int8_t> act(static_cast<size_t>(max_superblocks) * 256u);
        std::vector<float> act_d(static_cast<size_t>(max_superblocks));
        std::vector<uint64_t> grid(1024);
        for (uint8_t & v : row) v = static_cast<uint8_t>(rng() & 0xffu);
        for (int8_t & v : act) v = static_cast<int8_t>(rng() & 0xffu);
        for (size_t i = 0; i < grid.size(); ++i) grid[i] = (static_cast<uint64_t>(rng()) << 32) | rng();
        for (float & v : act_d) v = static_cast<float>(static_cast<int>(rng() % 1000) - 500);

        int compared = 0;
        for (int superblocks = 1; superblocks <= max_superblocks; ++superblocks) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                const Iq2Chain ref = iq2_row(row.data(), act.data(), act_d.data(),
                                             superblocks, chunk, 0, grid.data());
                for (int ilp : kIlpValues) {
                    const Iq2Chain got = iq2_row(row.data(), act.data(), act_d.data(),
                                                 superblocks, chunk, ilp, grid.data());
                    check(same_iq2(ref, got), "IQ2_S ILP chain differs from the reference");
                    check(got.max_superblock_read < superblocks, "IQ2_S ILP read past the row");
                    check(got.steps.size() == ref.steps.size(),
                          "IQ2_S ILP step count differs");
                    ++compared;
                }
            }
        }
        std::printf("IQ2_S rows compared: %d chains\n", compared);

        // Positive control: one byte of the qs plane moves the integer partial.
        {
            std::vector<uint8_t> mutated = row;
            mutated[2 + 4] ^= 0x01u;
            const Iq2Chain ref = iq2_row(row.data(), act.data(), act_d.data(),
                                         10, 0, 0, grid.data());
            const Iq2Chain got = iq2_row(mutated.data(), act.data(), act_d.data(),
                                         10, 0, 4, grid.data());
            check(!same_iq2(ref, got), "IQ2_S positive control did not change the chain");
        }
    }

    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
