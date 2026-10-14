// Host bit-comparison of the warp-parallel GDN F32 GEMV (Task P).
//
// q38_gdn_launch_dense_f32_gemv_avx2() picks between three kernels:
//
//   reference (q38_gdn_dense_f32_gemv_avx2_kernel, WARODECODE_EXACT_SCALAR_FMA=1)
//       one thread per row, 4x8 ggml_vec_dot_f32 accumulators filled in AVX2
//       visit order, the GGML_F32x8_REDUCE tree, then the non-fused tail.
//
//   warp (q38_gdn_dense_f32_gemv_avx2_warp_kernel, this task, default ON)
//       lane 8*j + l owns accumulator [j][l] and walks the same i = 0, 32, ...
//       chain; the tree is rebuilt with shuffles in the same association.
//
//   legacy (q38_gdn_dense_f32_gemv_kernel, WARODECODE_EXACT_SCALAR_FMA=0)
//       lane-strided accumulation plus a warp reduction: a different addition
//       order, which is why it is not the default. Positive control here.
//
// The test asserts that reference and warp agree bit for bit over random rows
// and widths, and that the legacy order does differ - the control that gives
// the equality its meaning.
//
// What this does NOT check: the compiled device code, the shuffle masks or the
// launch geometry. The 64/64 token parity run on the GPU covers those.
//
// Run: bash tests/qwen38_gdn_f32_gemv_host_check.sh

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char * what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 20) std::printf("FAIL: %s\n", what);
    }
}

float fmul_rn(float a, float b) { return a * b; }
float fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
float fadd_rn(float a, float b) { return a + b; }

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// q38_gdn_reduce_avx2() / q38_hc_reduce_avx2().
float reduce_avx2(float sum[4][8]) {
    for (int l = 0; l < 8; ++l) {
        sum[0][l] = fadd_rn(sum[0][l], sum[2][l]);
        sum[1][l] = fadd_rn(sum[1][l], sum[3][l]);
    }
    for (int l = 0; l < 8; ++l) sum[0][l] = fadd_rn(sum[0][l], sum[1][l]);
    float t0[4];
    for (int k = 0; k < 4; ++k) t0[k] = fadd_rn(sum[0][k], sum[0][k + 4]);
    const float h0 = fadd_rn(t0[0], t0[1]);
    const float h1 = fadd_rn(t0[2], t0[3]);
    return fadd_rn(h0, h1);
}

float reference_row(const float * w, const float * x, int cols) {
    float sum[4][8];
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = cols & ~31;
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 8; ++l) {
                const int c = i + j * 8 + l;
                sum[j][l] = fmaf_rn(w[c], x[c], sum[j][l]);
            }
        }
    }
    float acc = reduce_avx2(sum);
    for (int c = np; c < cols; ++c) acc = fadd_rn(acc, fmul_rn(w[c], x[c]));
    return acc;
}

// The warp kernel on a 32-lane machine, shuffle tree included.
float warp_row(const float * w, const float * x, int cols) {
    float lane_acc[32];
    for (int lane = 0; lane < 32; ++lane) {
        const int j = lane >> 3;
        const int l = lane & 7;
        float acc = 0.0f;
        for (int i = 0; i < (cols & ~31); i += 32) {
            const int c = i + j * 8 + l;
            acc = fmaf_rn(w[c], x[c], acc);
        }
        lane_acc[lane] = acc;
    }
    float acc[32];
    for (int lane = 0; lane < 32; ++lane) {
        acc[lane] = fadd_rn(lane_acc[lane], lane_acc[lane ^ 16]);
    }
    for (int lane = 0; lane < 8; ++lane) {
        acc[lane] = fadd_rn(acc[lane], acc[(lane & 7) + 8]);
    }
    for (int lane = 0; lane < 4; ++lane) {
        acc[lane] = fadd_rn(acc[lane], acc[(lane & 3) + 4]);
    }
    float out = fadd_rn(fadd_rn(acc[0], acc[1]), fadd_rn(acc[2], acc[3]));
    for (int c = cols & ~31; c < cols; ++c) out = fadd_rn(out, fmul_rn(w[c], x[c]));
    return out;
}

// Positive control: the legacy lane-strided order plus a warp tree.
float legacy_row(const float * w, const float * x, int cols) {
    float lane_acc[32];
    for (int lane = 0; lane < 32; ++lane) {
        float acc = 0.0f;
        for (int i = lane; i < cols; i += 32) acc = fmaf_rn(w[i], x[i], acc);
        lane_acc[lane] = acc;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int lane = 0; lane < 32; ++lane) {
            if (lane + offset < 32) lane_acc[lane] = fadd_rn(lane_acc[lane], lane_acc[lane + offset]);
        }
    }
    return lane_acc[0];
}

}  // namespace

int main() {
    std::mt19937 rng(20261015u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    long long compared = 0;
    long long mismatches = 0;
    long long legacy_differing = 0;
    // Qwen3.8 ssm_beta / ssm_alpha widths are multiples of 32 (hidden 10240 and
    // 2048 / 2560); the odd widths are there to prove the tail path matches too.
    const int widths[] = {32, 64, 2560, 2048, 10240, 96, 48};
    for (int cols : widths) {
        for (int trial = 0; trial < 8; ++trial) {
            std::vector<float> w(static_cast<size_t>(cols));
            std::vector<float> x(static_cast<size_t>(cols));
            for (int c = 0; c < cols; ++c) {
                w[static_cast<size_t>(c)] = dist(rng);
                x[static_cast<size_t>(c)] = dist(rng);
            }
            const float reference = reference_row(w.data(), x.data(), cols);
            const float warp = warp_row(w.data(), x.data(), cols);
            const float legacy = legacy_row(w.data(), x.data(), cols);
            ++compared;
            if (float_bits(reference) != float_bits(warp)) ++mismatches;
            if (float_bits(reference) != float_bits(legacy)) ++legacy_differing;
        }
    }
    std::printf("rows compared=%lld  warp mismatches=%lld  legacy differences=%lld\n",
                compared, mismatches, legacy_differing);
    check(mismatches == 0, "warp GEMV is bit-identical to the one-thread reference");
    check(legacy_differing > 0, "positive control: the legacy warp order does differ");

    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
