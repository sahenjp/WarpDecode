// Host mirror of the Task V pipelined Q8_0 x Q8_0 exact GEMV
// (src/qwen38_hc.cuh, q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_pipe_kernel<PIPE>).
//
// The kernel only adds memory-level parallelism: PIPE blocks' operands are
// fetched before the corresponding FMAs. The float chain and the closing tree
// are untouched, so the result has to be bit-identical to the scalar reference
// kernel -- including the short-row path (blocks_per_row < PIPE), the clamped
// prefetch window at the end of a row and the scalar tail.
//
// Build/run: bash tests/qwen38_hc_q8_pipe_host_check.sh
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
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t man = (uint32_t)h & 0x3ffu;
    float v = 0.0f;
    if (exp == 0) {
        v = std::ldexp((float)man, -24);
    } else if (exp != 31) {
        v = std::ldexp((float)(man + 1024u), (int)exp - 25);
    }
    return sign ? -v : v;
}

// 2-byte little-endian load of the block scale, as lowbit_half() does.
float weight_scale(const uint8_t * block) {
    uint16_t h = 0;
    std::memcpy(&h, block, sizeof(h));
    return half_from_bits(h);
}

int abs_i8(int8_t v) { return v == -128 ? 128 : (v < 0 ? -v : v); }
int sign_i8(int8_t v, int8_t sign) {
    if (sign == 0) return 0;
    if (sign < 0) return v == -128 ? 127 : -v;
    return v;
}
int sat_i16(int v) { return std::max(-32768, std::min(32767, v)); }

// The shipped scalar kernel's per-row order, one accumulator per 4-value chunk.
void scalar_chunks(const uint8_t * w, const int8_t * act, const float * ascales, int blocks,
                   float out[8]) {
    for (int c = 0; c < 8; ++c) {
        float a = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            const uint8_t * blk = w + (size_t)b * 34u;
            const int base = c * 4;
            const int8_t qw0 = (int8_t)blk[2 + base + 0];
            const int8_t qw1 = (int8_t)blk[2 + base + 1];
            const int8_t qw2 = (int8_t)blk[2 + base + 2];
            const int8_t qw3 = (int8_t)blk[2 + base + 3];
            const int8_t qx0 = act[b * 32 + base + 0];
            const int8_t qx1 = act[b * 32 + base + 1];
            const int8_t qx2 = act[b * 32 + base + 2];
            const int8_t qx3 = act[b * 32 + base + 3];
            const int p0 = sat_i16(abs_i8(qw0) * sign_i8(qx0, qw0) +
                                   abs_i8(qw1) * sign_i8(qx1, qw1));
            const int p1 = sat_i16(abs_i8(qw2) * sign_i8(qx2, qw2) +
                                   abs_i8(qw3) * sign_i8(qx3, qw3));
            a = std::fma(weight_scale(blk) * ascales[b], (float)(p0 + p1), a);
        }
        out[c] = a;
    }
}

float tree(const float a[8]) {
    const float r0 = a[4] + a[0];
    const float r1 = a[5] + a[1];
    const float r2 = a[6] + a[2];
    const float r3 = a[7] + a[3];
    return (r0 + r2) + (r1 + r3);
}

// The __dp4a integer dot, mirrored exactly (four int8 products summed as int32).
int dot4(const int8_t * qw, const int8_t * qx) {
    int s = 0;
    for (int i = 0; i < 4; ++i) s += (int)qw[i] * (int)qx[i];
    return s;
}

// The pipelined kernel's loop structure, including the clamped prefetch window.
template <int PIPE>
void pipe_chunks(const uint8_t * w, const int8_t * act, const float * ascales, int blocks,
                 float out[8]) {
    for (int c = 0; c < 8; ++c) {
        const int base = c * 4;
        unsigned packed[PIPE];
        int aq[PIPE];
        float sc[PIPE];
        const int prime = blocks < PIPE ? blocks : PIPE;
        for (int k = 0; k < PIPE; ++k) {
            const int b = k < prime ? k : prime - 1;
            const uint8_t * blk = w + (size_t)b * 34u;
            std::memcpy(&packed[k], blk + 2 + base, sizeof(unsigned));
            std::memcpy(&aq[k], act + (size_t)b * 32u + base, sizeof(int));
            sc[k] = weight_scale(blk) * ascales[b];
        }
        float acc = 0.0f;
        int b = 0;
        while (b + PIPE <= blocks) {
            unsigned np[PIPE];
            int na[PIPE];
            float ns[PIPE];
            for (int k = 0; k < PIPE; ++k) {
                const int nb = b + PIPE + k;
                const int cb = nb < blocks ? nb : blocks - 1;
                const uint8_t * blk = w + (size_t)cb * 34u;
                std::memcpy(&np[k], blk + 2 + base, sizeof(unsigned));
                std::memcpy(&na[k], act + (size_t)cb * 32u + base, sizeof(int));
                ns[k] = weight_scale(blk) * ascales[cb];
            }
            for (int k = 0; k < PIPE; ++k) {
                const int8_t * qw = (const int8_t *)&packed[k];
                const int8_t * qx = (const int8_t *)&aq[k];
                acc = std::fma(sc[k], (float)dot4(qw, qx), acc);
            }
            for (int k = 0; k < PIPE; ++k) {
                packed[k] = np[k];
                aq[k] = na[k];
                sc[k] = ns[k];
            }
            b += PIPE;
        }
        for (; b < blocks; ++b) {
            const uint8_t * blk = w + (size_t)b * 34u;
            unsigned p = 0;
            int a = 0;
            std::memcpy(&p, blk + 2 + base, sizeof(unsigned));
            std::memcpy(&a, act + (size_t)b * 32u + base, sizeof(int));
            acc = std::fma(weight_scale(blk) * ascales[b],
                           (float)dot4((const int8_t *)&p, (const int8_t *)&a), acc);
        }
        out[c] = acc;
    }
}

} // namespace

int main() {
    std::mt19937 rng(20261015u);
    // 320 is the model's widest row (10 240 / 32); 10 and 20 are the shortest;
    // 32 is the dispatch threshold kQ38HcQ8PipeMinBlocks.
    const int widths[] = {1, 2, 3, 8, 10, 16, 17, 20, 31, 32, 33, 80, 320};
    const int pipes[] = {1, 2, 4, 16};
    long long compared = 0;
    long long mismatches = 0;
    for (int blocks : widths) {
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<uint8_t> w((size_t)blocks * 34u);
            for (auto & b : w) b = (uint8_t)(rng() & 0xffu);
            // The activation stays in [-127, 127]: that is the domain the
            // production quantiser produces and the one the dp4a equality needs.
            std::vector<int8_t> act((size_t)blocks * 32u);
            for (auto & v : act) v = (int8_t)((int)(rng() % 255) - 127);
            std::vector<float> scales(blocks);
            for (auto & v : scales) v = 0.001f + 0.0001f * (float)(rng() % 500);

            float ref[8];
            scalar_chunks(w.data(), act.data(), scales.data(), blocks, ref);
            const float r = tree(ref);

            for (int pipe : pipes) {
                float got[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                switch (pipe) {
                    case 1: pipe_chunks<1>(w.data(), act.data(), scales.data(), blocks, got); break;
                    case 2: pipe_chunks<2>(w.data(), act.data(), scales.data(), blocks, got); break;
                    case 4: pipe_chunks<4>(w.data(), act.data(), scales.data(), blocks, got); break;
                    default: pipe_chunks<16>(w.data(), act.data(), scales.data(), blocks, got); break;
                }
                ++compared;
                if (bits(tree(got)) != bits(r)) ++mismatches;
            }
        }
    }
    std::printf("rows compared=%lld  mismatches=%lld\n", compared, mismatches);
    check(mismatches == 0, "pipelined kernel is bit-identical to the scalar order");
    // Positive control: dropping every other block must actually move bits, so
    // the comparison above is not vacuous.
    {
        std::vector<uint8_t> w(320u * 34u);
        for (auto & b : w) b = (uint8_t)(rng() & 0xffu);
        std::vector<int8_t> act(320u * 32u);
        for (auto & v : act) v = (int8_t)((int)(rng() % 255) - 127);
        std::vector<float> scales(320, 0.5f);
        float ref[8];
        scalar_chunks(w.data(), act.data(), scales.data(), 320, ref);
        float wrong[8];
        for (int c = 0; c < 8; ++c) {
            const int base = c * 4;
            float a = 0.0f;
            for (int b = 0; b < 320; b += 2) {
                const uint8_t * blk = w.data() + (size_t)b * 34u;
                const int p0 = dot4((const int8_t *)(blk + 2 + base), act.data() + b * 32 + base);
                a = std::fma(weight_scale(blk) * scales[b], (float)p0, a);
            }
            wrong[c] = a;
        }
        check(bits(tree(wrong)) != bits(tree(ref)),
              "positive control: a different chain order does move bits");
    }
    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
