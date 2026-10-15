// Host check for the k-quant dequantisers in src/ggml_dequant.hpp
// (ggml_q2_k_value / q3_k_value / q4_k_value / q5_k_value), which the dense path
// uses to requantise Q2_K..Q6_K tensors to Q8_0.
//
// The error-prone part is the byte/nibble/bit index mapping and the packed 6-bit
// scale extraction, so the test builds synthetic blocks whose expected value is
// known in closed form for every element and compares:
//
//   Q4_K: d=1, dmin=0, all 8 (scale,min) = (1,0), qs = 0xAB  -> value = nibble
//   Q5_K: same with the 5th bit forced off (qh=0) and on (qh=0xFF)
//   Q2_K: d=1, dmin=0, all scales = 0x01, qs = 0xE4         -> value = 2-bit field
//   Q3_K: d=1, hm=0xFF (and 0x00), scales unpacked to 15    -> value = -17*v
//
// Build/run: bash tests/qwen38_kquant_host_check.sh
// 最小の __half shim (検査で使う値は 1.0/0.0 だけなので厳密な丸めは要らない)。
#if !defined(__CUDACC__)
struct __half { unsigned short bits; };
static inline __half __float2half(float v) {
    __half h{};
    if (v == 0.0f) { h.bits = 0; return h; }
    const float a = v < 0 ? -v : v;
    int e = 0;
    float m = a;
    while (m >= 2.0f) { m *= 0.5f; e += 1; }
    while (m < 1.0f) { m *= 2.0f; e -= 1; }
    unsigned bits = (static_cast<unsigned>(e + 15) << 10) |
                    (static_cast<unsigned>((m - 1.0f) * 1024.0f + 0.5f) & 0x3FFu);
    if (v < 0) bits |= 0x8000u;
    h.bits = static_cast<unsigned short>(bits);
    return h;
}
#endif

#include "ggml_dequant.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

void put_half(uint8_t * p, float v) {
    const __half h = __float2half(v);
    std::memcpy(p, &h, sizeof(uint16_t));
}

bool close(float a, float b) { return std::fabs(a - b) < 1e-4f; }

} // namespace

int main() {
    // ---- Q4_K: 144 bytes, d + dmin + scales[12] + qs[128] ----
    {
        std::vector<uint8_t> blk(144, 0);
        put_half(blk.data(), 1.0f);
        put_half(blk.data() + 2, 0.0f);
        for (int i = 0; i < 12; ++i) blk[4 + i] = 1;      // all (sc,m) = (1,0)
        for (int i = 0; i < 128; ++i) blk[16 + i] = 0xAB; // low 0xB, high 0xA
        int mismatches = 0;
        for (int i = 0; i < 256; ++i) {
            const int j = i >> 6, r = i & 63, half = r >> 5, l = r & 31;
            const int expect = half ? 0xA : 0xB;
            if (!close(wd::ggml_q4_k_value(blk.data(), i), static_cast<float>(expect))) ++mismatches;
            (void)j; (void)l;
        }
        check(mismatches == 0, "Q4_K index/nibble mapping");
        std::printf("Q4_K elements checked: 256 (mismatches %d)\n", mismatches);
    }

    // ---- Q5_K: 176 bytes, d + dmin + scales[12] + qh[32] + qs[128] ----
    {
        int mismatches[2] = {0, 0};
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<uint8_t> blk(176, 0);
            put_half(blk.data(), 1.0f);
            put_half(blk.data() + 2, 0.0f);
            for (int i = 0; i < 12; ++i) blk[4 + i] = 1;
            for (int i = 0; i < 32; ++i) blk[16 + i] = variant ? 0xFF : 0x00;
            for (int i = 0; i < 128; ++i) blk[48 + i] = 0x21;   // low 0x1, high 0x2
            for (int i = 0; i < 256; ++i) {
                const int r = i & 63, half = r >> 5;
                const int base = half ? 0x2 : 0x1;
                const int expect = base + (variant ? 16 : 0);
                if (!close(wd::ggml_q5_k_value(blk.data(), i), static_cast<float>(expect))) {
                    ++mismatches[variant];
                }
            }
        }
        check(mismatches[0] == 0, "Q5_K with the 5th bit clear");
        check(mismatches[1] == 0, "Q5_K with the 5th bit set");
        std::printf("Q5_K elements checked: 512 (mismatches %d/%d)\n", mismatches[0], mismatches[1]);
    }

    // ---- Q2_K: 84 bytes, scales[16] + qs[64] + d + dmin ----
    {
        std::vector<uint8_t> blk(84, 0);
        for (int i = 0; i < 16; ++i) blk[i] = 0x01;        // sc = 1, m = 0
        for (int i = 0; i < 64; ++i) blk[16 + i] = 0xE4;   // 2-bit fields 0,1,2,3
        put_half(blk.data() + 80, 1.0f);
        put_half(blk.data() + 82, 0.0f);
        int mismatches = 0;
        for (int i = 0; i < 256; ++i) {
            const int n = i >> 7, r = i & 127, j = r >> 5, half = (r >> 4) & 1, l = r & 15;
            const uint8_t q = blk[16 + n * 32 + l + (half ? 16 : 0)];
            const int v = (q >> (j * 2)) & 3;
            if (!close(wd::ggml_q2_k_value(blk.data(), i), static_cast<float>(v))) ++mismatches;
        }
        check(mismatches == 0, "Q2_K index/2-bit mapping");
        std::printf("Q2_K elements checked: 256 (mismatches %d)\n", mismatches);
    }

    // ---- Q3_K: 110 bytes, hmask[32] + qs[64] + scales[12] + d ----
    {
        int mismatches[2] = {0, 0};
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<uint8_t> blk(110, 0);
            for (int i = 0; i < 32; ++i) blk[i] = variant ? 0xFF : 0x00;   // hmask
            for (int i = 0; i < 64; ++i) blk[32 + i] = 0xE4;               // 2-bit fields
            // raw scales: low nibble 0xF, high bits 0 -> every unpacked scale = 15
            // raw[0..7] = 0xFF -> low nibble 15 for scales 0..7 and high nibble 15 for
            // scales 8..15 (they are rebuilt from aux[0]/aux[1]'s high nibbles).
            for (int i = 0; i < 8; ++i) blk[96 + i] = 0xFF;
            for (int i = 8; i < 12; ++i) blk[96 + i] = 0x00;
            put_half(blk.data() + 108, 1.0f);
            for (int i = 0; i < 256; ++i) {
                const int n = i >> 7, r = i & 127, j = r >> 5, half = (r >> 4) & 1, l = r & 15;
                const uint8_t q = blk[32 + n * 32 + l + (half ? 16 : 0)];
                const int v = (q >> (j * 2)) & 3;
                const int hv = variant ? v : (v - 4);
                const float expect = -17.0f * static_cast<float>(hv);   // (15 - 32) * v
                if (!close(wd::ggml_q3_k_value(blk.data(), i), expect)) ++mismatches[variant];
            }
        }
        check(mismatches[0] == 0, "Q3_K with the high mask clear");
        check(mismatches[1] == 0, "Q3_K with the high mask set");
        std::printf("Q3_K elements checked: 512 (mismatches %d/%d)\n", mismatches[0], mismatches[1]);
    }

    // ---- Q5_0 (22 bytes: d + qh[4] + qs[16]) ----
    {
        int mismatches[2] = {0, 0};
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<uint8_t> blk(22, 0);
            put_half(blk.data(), 1.0f);
            const uint32_t qh = variant ? 0xFFFFFFFFu : 0u;
            std::memcpy(blk.data() + 2, &qh, 4);
            for (int i = 0; i < 16; ++i) blk[6 + i] = 0x21;   // low 1, high 2
            for (int i = 0; i < 32; ++i) {
                const int base = ((i >> 4) & 1) ? 2 : 1;
                const int expect = base + (variant ? 16 : 0) - 16;
                if (!close(wd::ggml_q5_0_value(blk.data(), i), static_cast<float>(expect))) {
                    ++mismatches[variant];
                }
            }
        }
        check(mismatches[0] == 0, "Q5_0 with the 5th bit clear");
        check(mismatches[1] == 0, "Q5_0 with the 5th bit set");
        std::printf("Q5_0 elements checked: 64 (mismatches %d/%d)\n", mismatches[0], mismatches[1]);
    }

    // ---- Q5_1 (24 bytes: d + m + qh[4] + qs[16]) ----
    {
        std::vector<uint8_t> blk(24, 0);
        put_half(blk.data(), 1.0f);
        put_half(blk.data() + 2, 1.0f);          // m
        const uint32_t qh = 0u;
        std::memcpy(blk.data() + 4, &qh, 4);
        for (int i = 0; i < 16; ++i) blk[8 + i] = 0x21;
        int mismatches = 0;
        for (int i = 0; i < 32; ++i) {
            const int base = ((i >> 4) & 1) ? 2 : 1;
            const int expect = base + 1;         // x*d + m
            if (!close(wd::ggml_q5_1_value(blk.data(), i), static_cast<float>(expect))) ++mismatches;
        }
        check(mismatches == 0, "Q5_1 index/nibble mapping");
        std::printf("Q5_1 elements checked: 32 (mismatches %d)\n", mismatches);
    }

    // The 6-bit (scale, min) extractor itself, for all 8 groups.
    {
        uint8_t scales[12];
        for (int i = 0; i < 12; ++i) scales[i] = 0;
        // j < 4: d = q[j] & 63, m = q[j+4] & 63. 6bit 目は q[j] の bit6-7 に入る。
        for (int i = 0; i < 4; ++i) { scales[i] = 5 | 0x80; scales[i + 4] = 7 | 0x80; }
        for (int i = 8; i < 12; ++i) scales[i] = 0x31;
        int bad = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t d = 0, m = 0;
            wd::ggml_get_scale_min_k4(j, scales, &d, &m);
            if (j < 4) {
                if (d != 5 || m != 7) ++bad;
            } else {
                // d = (0x31 & 0xF) | ((0x87 >> 6) << 4) = 1 + 32 = 33
                // m = (0x31 >> 4)  | ((0x87 >> 6) << 4) = 3 + 32 = 35
                if (d != 33 || m != 35) ++bad;
            }
        }
        check(bad == 0, "get_scale_min_k4 packing");
    }

    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
