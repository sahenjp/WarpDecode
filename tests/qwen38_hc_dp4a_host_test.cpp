// Host bit-comparison for the __dp4a form of the exact Q8_0 x Q8_0 row dot
// (Task P, src/qwen38_hc.cuh).
//
// Background: Task J's dp4a kernel was 6.2x faster on L4 but broke the 64/64
// token parity (41/64). This test pins the cause and proves the replacement is
// bit-exact - all without nvcc, by mirroring the three orderings as plain C++:
//
//   reference (q38_hc_q8_0_q8_0_exact_gemv_f32_kernel)
//       lane c in 0..7 owns 4-value chunk c of every 32-value block and carries
//       its own accumulator = __fmaf_rn(scale_b, (float)dot_c_b, acc) chain over
//       b = 0..B-1; the eight lanes are then combined by the closing tree
//       ((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3)).
//
//   dp4a (q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_kernel, this task)
//       the same eight chains, the same tree, the same scale computation; only
//       the four integer products of a chunk become one byte-packed __dp4a.
//
//   old_dp4a (Task J, removed)
//       four blocks in flight, the eight chunk products summed in *integer*
//       space first, and one float chain over the block totals on lane 0.
//
// Integer addition may be reassociated, float addition may not, so the test
// asserts:
//   1. the reference's per-chunk integer dot equals an exact 64-bit dot product
//      (the abs/sign/sat_i16 path is a faithful integer dot for every
//      activation in [-127, 127] and every weight byte, -128 included),
//   2. the two int16 pair sums of the reference never saturate,
//   3. reference and dp4a agree bit for bit on every row, and
//   4. reference and old_dp4a do NOT agree on at least one row - the positive
//      control that gives (3) its meaning.
//
// The -128 activation corner is exercised separately and reported: both the
// reference and dp4a deviate from the exact integer dot there (the reference
// maps qx == -128 to 127 when qw < 0), which is why the production quantisers
// keep the activation in [-127, 127].
//
// What this does NOT check: the compiled device code, the sm < 80 fallback of
// __dp4a (mirrored by the scalar expansion below) or the launch geometry. The
// 64/64 token parity run on the GPU covers those.
//
// Run: bash tests/qwen38_hc_dp4a_host_check.sh

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using std::int32_t;
using std::int64_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

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
// Copies of the src/qwen38_hc.cuh helpers (device functions cannot be linked
// here; tests/qwen38_cuda_syntax_check.sh type-checks the originals).
// ---------------------------------------------------------------------------
int abs_i8(int8_t value) {
    if (value == static_cast<int8_t>(-128)) return 128;
    return value < 0 ? -static_cast<int>(value) : static_cast<int>(value);
}

int sign_i8(int8_t value, int8_t sign) {
    if (sign == 0) return 0;
    if (sign < 0) return value == static_cast<int8_t>(-128) ? 127 : -static_cast<int>(value);
    return static_cast<int>(value);
}

int sat_i16(int value) { return std::max(-32768, std::min(32767, value)); }

// Scalar expansion of lowbit_dp4a() (src/lowbit.cuh), byte-for-byte the same
// sum the sm_80+ __dp4a instruction computes.
int dp4a_scalar(uint32_t a, uint32_t b, int c) {
    int sum = c;
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int>(static_cast<int8_t>(a >> (8 * i))) *
               static_cast<int>(static_cast<int8_t>(b >> (8 * i)));
    }
    return sum;
}

// 34-byte Q8_0 block: [0,2) = __half scale, [2,34) = 32 int8 quants.
constexpr size_t kBlockBytes = 34;

uint16_t load_u16(const uint8_t * p) {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint32_t load_u32(const uint8_t * p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float half_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
    const int32_t exp = (bits >> 10) & 0x1f;
    const uint32_t mant = bits & 0x3ff;
    uint32_t out = 0;
    if (exp == 0) {
        const float value = std::ldexp(static_cast<float>(mant), -24);
        const float signed_value = sign ? -value : value;
        std::memcpy(&out, &signed_value, sizeof(out));
    } else {
        out = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float fmul_rn(float a, float b) { return a * b; }
float fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
float fadd_rn(float a, float b) { return a + b; }

float reference_tree(const float lane[8]) {
    const float r0 = fadd_rn(lane[4], lane[0]);
    const float r1 = fadd_rn(lane[5], lane[1]);
    const float r2 = fadd_rn(lane[6], lane[2]);
    const float r3 = fadd_rn(lane[7], lane[3]);
    const float s0 = fadd_rn(r0, r2);
    const float s1 = fadd_rn(r1, r3);
    return fadd_rn(s0, s1);
}

const uint8_t * block_ptr(const uint8_t * row, int b) {
    return row + static_cast<size_t>(b) * kBlockBytes;
}

// q38_hc_q8_0_q8_0_exact_gemv_f32_kernel (the parity-verified reference).
float reference_row(const uint8_t * row, const int8_t * act, const uint16_t * act_scales,
                    int blocks_per_row, bool * saturated) {
    float lanes[8];
    for (int c = 0; c < 8; ++c) {
        const int base = c * 4;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t * block = block_ptr(row, b);
            const int8_t qw[4] = {
                static_cast<int8_t>(block[2 + base + 0]), static_cast<int8_t>(block[2 + base + 1]),
                static_cast<int8_t>(block[2 + base + 2]), static_cast<int8_t>(block[2 + base + 3])};
            const int8_t qx[4] = {act[(b << 5) + base + 0], act[(b << 5) + base + 1],
                                  act[(b << 5) + base + 2], act[(b << 5) + base + 3]};
            const int raw0 = abs_i8(qw[0]) * sign_i8(qx[0], qw[0]) +
                             abs_i8(qw[1]) * sign_i8(qx[1], qw[1]);
            const int raw1 = abs_i8(qw[2]) * sign_i8(qx[2], qw[2]) +
                             abs_i8(qw[3]) * sign_i8(qx[3], qw[3]);
            if (raw0 != sat_i16(raw0) || raw1 != sat_i16(raw1)) *saturated = true;
            const int dot = sat_i16(raw0) + sat_i16(raw1);
            const float scale = fmul_rn(half_to_float(load_u16(block)), half_to_float(act_scales[b]));
            acc = fmaf_rn(scale, static_cast<float>(dot), acc);
        }
        lanes[c] = acc;
    }
    return reference_tree(lanes);
}

// q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_kernel (this task).
float dp4a_row(const uint8_t * row, const int8_t * act, const uint16_t * act_scales,
               int blocks_per_row) {
    float lanes[8];
    for (int c = 0; c < 8; ++c) {
        const int base = c * 4;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t * block = block_ptr(row, b);
            const uint32_t packed = load_u32(block + 2 + base);
            uint32_t act_bytes = 0;
            std::memcpy(&act_bytes, &act[(b << 5) + base], sizeof(act_bytes));
            const int dot = dp4a_scalar(act_bytes, packed, 0);
            const float scale = fmul_rn(half_to_float(load_u16(block)), half_to_float(act_scales[b]));
            acc = fmaf_rn(scale, static_cast<float>(dot), acc);
        }
        lanes[c] = acc;
    }
    return reference_tree(lanes);
}

// Task J's removed kernel: integer block totals, one lane-0 float chain.
float old_dp4a_row(const uint8_t * row, const int8_t * act, const uint16_t * act_scales,
                   int blocks_per_row) {
    float sum = 0.0f;
    int b = 0;
    for (; b + 4 <= blocks_per_row; b += 4) {
        int dots[4];
        for (int g = 0; g < 4; ++g) {
            int64_t total = 0;
            for (int i = 0; i < 32; ++i) {
                total += static_cast<int64_t>(static_cast<int8_t>(block_ptr(row, b + g)[2 + i])) *
                         static_cast<int64_t>(act[((b + g) << 5) + i]);
            }
            dots[g] = static_cast<int>(total);
        }
        for (int g = 0; g < 4; ++g) {
            const float scale = fmul_rn(half_to_float(load_u16(block_ptr(row, b + g))),
                                        half_to_float(act_scales[b + g]));
            sum += scale * static_cast<float>(dots[g]);
        }
    }
    for (; b < blocks_per_row; ++b) {
        int64_t total = 0;
        for (int i = 0; i < 32; ++i) {
            total += static_cast<int64_t>(static_cast<int8_t>(block_ptr(row, b)[2 + i])) *
                     static_cast<int64_t>(act[(b << 5) + i]);
        }
        const float scale = fmul_rn(half_to_float(load_u16(block_ptr(row, b))),
                                    half_to_float(act_scales[b]));
        sum += scale * static_cast<float>(static_cast<int>(total));
    }
    return sum;
}

struct Row {
    std::vector<uint8_t> weights;
    std::vector<int8_t> activation;
    std::vector<uint16_t> scales;
    int blocks_per_row = 0;
};

int8_t random_weight(std::mt19937 & rng, int mode) {
    std::uniform_int_distribution<int> full(-128, 127);
    if (mode == 0) return static_cast<int8_t>(full(rng));
    static const int values[6] = {-128, -127, -1, 0, 1, 127};
    std::uniform_int_distribution<int> pick(0, 5);
    return static_cast<int8_t>(values[pick(rng)]);
}

// Production quantisers (q38_hc_q8_0_quantize_f32_exact_kernel and the Q8_1
// kernel) can only emit [-127, 127]; -128 is exercised separately below.
int8_t random_activation(std::mt19937 & rng, int mode) {
    std::uniform_int_distribution<int> full(-127, 127);
    if (mode == 0) return static_cast<int8_t>(full(rng));
    static const int values[5] = {-127, -1, 0, 1, 127};
    std::uniform_int_distribution<int> pick(0, 4);
    return static_cast<int8_t>(values[pick(rng)]);
}

uint16_t random_scale_bits(std::mt19937 & rng, int mode) {
    if (mode == 0) {
        std::uniform_int_distribution<int> exponent(1, 24);
        std::uniform_int_distribution<int> mantissa(0, 1023);
        std::uniform_int_distribution<int> sign(0, 1);
        std::uniform_int_distribution<int> zero(0, 20);
        if (zero(rng) == 0) return static_cast<uint16_t>(0x0000);
        return static_cast<uint16_t>((sign(rng) << 15) | (exponent(rng) << 10) | mantissa(rng));
    }
    static const uint16_t values[6] = {0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x7bff};
    std::uniform_int_distribution<int> pick(0, 5);
    return values[pick(rng)];
}

Row make_row(std::mt19937 & rng, int blocks_per_row, int mode) {
    Row row;
    row.blocks_per_row = blocks_per_row;
    row.weights.assign(static_cast<size_t>(blocks_per_row) * kBlockBytes, 0);
    row.activation.assign(static_cast<size_t>(blocks_per_row) * 32, 0);
    row.scales.assign(static_cast<size_t>(blocks_per_row), 0);
    for (int b = 0; b < blocks_per_row; ++b) {
        uint8_t * block = row.weights.data() + static_cast<size_t>(b) * kBlockBytes;
        const uint16_t scale = random_scale_bits(rng, mode);
        std::memcpy(block, &scale, sizeof(scale));
        for (int i = 0; i < 32; ++i) {
            block[2 + i] = static_cast<uint8_t>(random_weight(rng, mode));
        }
        for (int i = 0; i < 32; ++i) {
            row.activation[static_cast<size_t>(b) * 32 + i] = random_activation(rng, mode);
        }
        row.scales[static_cast<size_t>(b)] = random_scale_bits(rng, mode);
    }
    return row;
}

}  // namespace

int main() {
    std::mt19937 rng(20261015u);

    // 1. The reference integer dot must equal an exact 64-bit dot, and its
    //    int16 pair sums must never saturate.
    {
        bool saturated = false;
        bool integer_exact = true;
        for (int trial = 0; trial < 200; ++trial) {
            const int blocks = 1 + (trial % 37);
            const Row row = make_row(rng, blocks, trial % 2);
            for (int b = 0; b < blocks; ++b) {
                const uint8_t * block = block_ptr(row.weights.data(), b);
                for (int c = 0; c < 8; ++c) {
                    const int base = c * 4;
                    int64_t exact = 0;
                    for (int i = 0; i < 4; ++i) {
                        exact += static_cast<int64_t>(static_cast<int8_t>(block[2 + base + i])) *
                                 static_cast<int64_t>(row.activation[(b << 5) + base + i]);
                    }
                    const int raw0 = abs_i8(static_cast<int8_t>(block[2 + base + 0])) *
                                         sign_i8(row.activation[(b << 5) + base + 0],
                                                 static_cast<int8_t>(block[2 + base + 0])) +
                                     abs_i8(static_cast<int8_t>(block[2 + base + 1])) *
                                         sign_i8(row.activation[(b << 5) + base + 1],
                                                 static_cast<int8_t>(block[2 + base + 1]));
                    const int raw1 = abs_i8(static_cast<int8_t>(block[2 + base + 2])) *
                                         sign_i8(row.activation[(b << 5) + base + 2],
                                                 static_cast<int8_t>(block[2 + base + 2])) +
                                     abs_i8(static_cast<int8_t>(block[2 + base + 3])) *
                                         sign_i8(row.activation[(b << 5) + base + 3],
                                                 static_cast<int8_t>(block[2 + base + 3]));
                    if (sat_i16(raw0) != raw0 || sat_i16(raw1) != raw1) saturated = true;
                    if (sat_i16(raw0) + sat_i16(raw1) != exact) integer_exact = false;
                }
            }
        }
        check(!saturated, "reference int16 pair sums never saturate");
        check(integer_exact, "reference integer dot equals the exact 64-bit dot");
    }

    // 2/3. Bit-exact equality of the dp4a kernel, plus the positive control
    //      that the removed Task J ordering is measurably different.
    {
        long long compared = 0;
        long long old_differing = 0;
        long long mismatches = 0;
        const int widths[] = {1, 3, 8, 10, 11, 320, 1280};
        for (int width : widths) {
            for (int mode = 0; mode < 2; ++mode) {
                for (int trial = 0; trial < 40; ++trial) {
                    const Row row = make_row(rng, width, mode);
                    bool saturated = false;
                    const float reference = reference_row(
                        row.weights.data(), row.activation.data(), row.scales.data(), width,
                        &saturated);
                    check(!saturated, "no int16 saturation in the compared rows");
                    const float dp4a = dp4a_row(
                        row.weights.data(), row.activation.data(), row.scales.data(), width);
                    const float old_dp4a = old_dp4a_row(
                        row.weights.data(), row.activation.data(), row.scales.data(), width);
                    ++compared;
                    if (float_bits(reference) != float_bits(dp4a)) ++mismatches;
                    if (float_bits(reference) != float_bits(old_dp4a)) ++old_differing;
                }
            }
        }
        std::printf("rows compared=%lld  dp4a mismatches=%lld  old_kernel differences=%lld\n",
                    compared, mismatches, old_differing);
        check(mismatches == 0, "dp4a kernel is bit-identical to the reference");
        check(old_differing > 0, "positive control: the removed Task J order does differ");
    }

    // 4. Activation == -128: the one corner where the reference itself leaves
    //    the exact integer dot (sign_i8 maps it to +127 when the weight is
    //    negative). Documented because it is why the production quantisers stay
    //    in [-127, 127].
    {
        const Row row = make_row(rng, 4, 1);
        const uint8_t * block = row.weights.data();
        const int8_t qw0 = static_cast<int8_t>(block[2]);
        const int reference_term = abs_i8(qw0) * sign_i8(static_cast<int8_t>(-128), qw0);
        const int exact_term = static_cast<int>(qw0) * (-128);
        if (qw0 < 0) {
            check(reference_term != exact_term,
                  "-128 activation: sign_i8 special case is not the exact product");
            std::printf("-128 activation corner: qw=%d reference=%d exact=%d (unreachable while the "
                        "quantiser stays in [-127,127])\n",
                        static_cast<int>(qw0), reference_term, exact_term);
        }
    }

    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
