// Host mirror of the multi-block argmax key packing (src/qwen38_forward.cuh,
// q38_argmax_key / q38_argmax_multi_kernel).
//
// The reference scan keeps {best value, lowest index on ties} starting from
// {-FLT_MAX, 0} and ignores non-finite logits. The kernel instead maximises a
// packed 64-bit key: the IEEE-monotone order of the float in the high half and
// the inverted index in the low half, with the identity key(-FLT_MAX, 0). This
// test compares the two over random vectors that include ties, -FLT_MAX, +/-inf
// and NaN, and checks the non-finite flag.
//
// Build/run: bash tests/qwen38_argmax_host_check.sh
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

uint64_t key(float v, int i) {
    const uint32_t u = bits(v);
    const uint32_t monotone = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<uint64_t>(monotone) << 32) |
           static_cast<uint32_t>(~static_cast<uint32_t>(i));
}

uint64_t identity() { return key(-3.402823466e+38F, 0); }

int decode(uint64_t k) {
    return (k == 0) ? 0 : static_cast<int>(~static_cast<uint32_t>(k & 0xffffffffu));
}

// The reference scan (q38_argmax_kernel's inner loop).
void reference(const std::vector<float> & logits, int & token, int & nonfinite) {
    float best_v = -3.402823466e+38F;
    int best_i = 0;
    nonfinite = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            nonfinite = 1;
            continue;
        }
        if (v > best_v || (v == best_v && static_cast<int>(i) < best_i)) {
            best_v = v;
            best_i = static_cast<int>(i);
        }
    }
    token = best_i;
}

void packed(const std::vector<float> & logits, int & token, int & nonfinite) {
    uint64_t best = identity();
    nonfinite = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            nonfinite = 1;
            continue;
        }
        const uint64_t k = key(v, static_cast<int>(i));
        if (k > best) best = k;
    }
    token = decode(best);
}

void compare(const std::vector<float> & logits, const char * what) {
    int rt = 0;
    int rn = 0;
    int pt = 0;
    int pn = 0;
    reference(logits, rt, rn);
    packed(logits, pt, pn);
    check(rt == pt, what);
    check(rn == pn, "non-finite flag differs");
}

} // namespace

int main() {
    std::mt19937 rng(20261015u);

    // Hand-built edge cases.
    {
        std::vector<float> v(8, -3.402823466e+38F);
        compare(v, "all logits at -FLT_MAX keep token 0");
    }
    {
        std::vector<float> v(8, 0.0f);
        compare(v, "all-equal logits keep token 0");
    }
    {
        std::vector<float> v(8, -5.0f);
        v[0] = -3.402823466e+38F;   // exactly the identity value at index 0
        v[3] = -3.402823466e+38F;   // and again at index 3
        compare(v, "identity value at a later index does not win");
    }
    {
        std::vector<float> v{1.0f, 2.0f, 2.0f, 2.0f, -1.0f};
        compare(v, "ties resolve to the lowest index");
    }
    {
        std::vector<float> v{std::numeric_limits<float>::infinity(), 1.0f, 2.0f};
        compare(v, "+inf is non-finite");
    }
    {
        std::vector<float> v{-std::numeric_limits<float>::infinity(), -1.0f, -2.0f};
        compare(v, "-inf is non-finite");
    }
    {
        std::vector<float> v{std::numeric_limits<float>::quiet_NaN(), -1.0f, -2.0f};
        compare(v, "NaN is non-finite");
    }
    {
        std::vector<float> v(4, std::numeric_limits<float>::quiet_NaN());
        compare(v, "all non-finite keeps token 0");
    }
    {
        std::vector<float> v(6, 0.0f);
        v[2] = -0.0f;
        v[5] = -0.0f;
        compare(v, "-0.0 ties with +0.0 by index");
    }
    {
        std::vector<float> v(5, 0.0f);
        v[1] = std::numeric_limits<float>::denorm_min();
        compare(v, "denormals order above zero");
    }

    // Random vectors, including heavy tie rates and the full float range.
    int cases = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 4096);
        std::vector<float> v(static_cast<size_t>(n));
        const int mode = trial % 4;
        for (int i = 0; i < n; ++i) {
            const uint32_t r = rng();
            if (mode == 0) {
                v[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(r % 7)) - 3.0f;
            } else if (mode == 1) {
                v[static_cast<size_t>(i)] = static_cast<float>(r) / 4.0e9f - 0.5f;
            } else if (mode == 2) {
                v[static_cast<size_t>(i)] = std::ldexp(static_cast<float>(static_cast<int>(r % 2001) - 1000), static_cast<int>(r % 60) - 30);
            } else {
                v[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(r % 5));
                if (r % 97 == 0) v[static_cast<size_t>(i)] = std::numeric_limits<float>::quiet_NaN();
                if (r % 89 == 0) v[static_cast<size_t>(i)] = -3.402823466e+38F;
            }
        }
        compare(v, "random vector differs from the reference scan");
        ++cases;
    }

    std::printf("argmax vectors compared: %d (+10 edge cases)\n", cases);
    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
