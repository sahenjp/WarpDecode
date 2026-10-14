// Host bit-comparison of the F32 weight x F32 activation dots in
// src/qwen38_moe.cuh, src/matrix.cuh and src/qwen38_hc.cuh against the real ggml
// CPU ggml_vec_dot_f32() the pinned oracle executes.
//
// Reference: llama.cpp 6fcaa16f4b360649933a54d1f91ad40ed35c0e11,
// ggml/src/ggml-cpu/vec.cpp:11, linked from build-host/bin/libggml-cpu.so (the
// same CMake flags the canonical oracle image uses: GGML_NATIVE=OFF still turns
// on -mavx -mavx2 -mfma -mf16c, so GGML_F32_STEP / GGML_F32_EPR == 32 / 8).
//
// Candidates: the extracted source text of the three AVX2-order kernels,
// compiled on the host by qwen38_shexp_gate_host_check.sh, so the device kernels
// and this test cannot drift apart:
//
//   * q38_moe_gate_dot_f32_avx2_kernel  - shared expert gate
//     (blk.N.ffn_gate_inp_shexp.weight, F32 [n_embd], one build_lora_mm row)
//   * dense_f32_gemv_avx2_kernel        - F32 router
//     (blk.N.ffn_gate_inp.weight, F32 [n_embd, n_expert], one row per expert)
//   * q38_hc_dense_gemv_f32_avx2_kernel - F32 HC inject
//     (blk.N.hc_*_inject.weight, F32 [hc_dim, hc], one row per stream)
//
// All three are "F32 weight x F32 activation": type_traits_cpu[GGML_TYPE_F32]
// has vec_dot_type == GGML_TYPE_F32 and nrows == 1, and llamafile_sgemm()
// declines the single decode token (sgemm.cpp:3819 "if (n < 2) return false"),
// so each row is one vec_dot(ne00, ..., nrc == 1) == ggml_vec_dot_f32.
//
// Acceptance rule: every candidate must be bit-identical to the linked
// ggml_vec_dot_f32 for every width that is a multiple of 32 - that is the case
// the model runs (2560 for the gate and the router, 10240 for the inject) - and
// the tail widths (n % 32 != 0) may differ only where the compiler contracted
// the last leftover elements: there the linked ggml must equal the
// compiled-tail model, the candidate the source-text model, and no other
// divergence is accepted. The tail cases are reported, never silently skipped.
//
// Usage: qwen38_shexp_gate_host_test [n_random_per_group]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Host stand-ins for the CUDA launch-index variables the extracted row kernels
// read. The checks call every candidate with rows == 1, i.e. the single thread
// that owns the row.
struct Q38HostIndex {
    int x;
    int y;
    int z;
};
static const Q38HostIndex blockIdx = {0, 0, 0};
static const Q38HostIndex blockDim = {1, 1, 1};
static const Q38HostIndex threadIdx = {0, 0, 0};

#include "qwen38_shexp_gate_extract.h"

// C linkage: vec.h declares the whole vector API inside extern "C".
extern "C" void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx,
                                 const float * y, size_t by, int nrc);

namespace {

int g_failures = 0;
long g_cases = 0;
long g_impl_cases[3] = {0, 0, 0};
long g_impl_fail[3] = {0, 0, 0};
const char * g_impl_name[3] = {"q38_moe_gate_dot_f32_avx2_kernel",
                               "dense_f32_gemv_avx2_kernel",
                               "q38_hc_dense_gemv_f32_avx2_kernel"};
long g_warp_diff = 0;
long g_warp_cases = 0;
long g_naive_diff = 0;
long g_tail_cases = 0;
long g_tail_diff = 0;
long g_tail_max_ulp = 0;
long g_tail_model_diff = 0;
long g_src_model_diff = 0;
double g_half_rel_max = 0.0;
double g_half_rel_sum = 0.0;
long g_half_rel_n = 0;

uint32_t bits(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// ---- candidates: the extracted device kernels, called with one row ---------

float candidate_gate(const std::vector<float> & w, const std::vector<float> & x) {
    float out = 0.0f;
    q38_moe_gate_dot_f32_avx2_kernel(w.data(), x.data(), &out, static_cast<int>(w.size()));
    return out;
}

float candidate_router(const std::vector<float> & w, const std::vector<float> & x) {
    float out = 0.0f;
    dense_f32_gemv_avx2_kernel(w.data(), x.data(), &out, 1, static_cast<int>(w.size()));
    return out;
}

float candidate_inject(const std::vector<float> & w, const std::vector<float> & x) {
    float out = 0.0f;
    q38_hc_dense_gemv_f32_avx2_kernel(w.data(), x.data(), &out, 1, static_cast<int>(w.size()));
    return out;
}

float reference_dot(const std::vector<float> & w, const std::vector<float> & x) {
    float out = 0.0f;
    ggml_vec_dot_f32(static_cast<int>(w.size()), &out, 0, w.data(), 0, x.data(), 0, 1);
    return out;
}

// ---- models of the ggml source text, written independently of the extract ---

// The n & ~31 blocks plus the GGML_F32x8_REDUCE tree: the part the device
// kernels and the oracle share.
float avx2_block_sum(const std::vector<float> & w, const std::vector<float> & x, int np) {
    float sum[4][8];
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 8; ++l) sum[j][l] = std::fma(w[i + j * 8 + l], x[i + j * 8 + l], sum[j][l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sum[0][l] = sum[0][l] + sum[2][l];
        sum[1][l] = sum[1][l] + sum[3][l];
    }
    for (int l = 0; l < 8; ++l) sum[0][l] = sum[0][l] + sum[1][l];
    float t[4];
    for (int k = 0; k < 4; ++k) t[k] = sum[0][k] + sum[0][k + 4];
    return (t[0] + t[1]) + (t[2] + t[3]);
}

// ggml_vec_dot_f32's "sumf += x[i]*y[i]" leftovers as the C source spells them:
// one rounded multiply and one rounded add per element.
float source_model_dot(const std::vector<float> & w, const std::vector<float> & x) {
    const int n = static_cast<int>(w.size());
    float acc = avx2_block_sum(w, x, n & ~31);
    for (int c = n & ~31; c < n; ++c) acc = acc + w[c] * x[c];
    return acc;
}

// How the compiled oracle actually handles the leftovers, read off the
// disassembly of the pinned build: GCC keeps the C source order but vectorises
// the leftover loop by 8 and then by 4 with separate rounded multiplies and one
// rounded add per element, contracting only the final 1..3 elements into
// vfmadd231ss. The n & ~31 blocks and the reduce tree are unchanged, so this
// model and source_model_dot() differ only inside the leftover elements.
// Diagnostic only: the model never reaches this tail (2560 and 10240 are
// multiples of 32).
float compiled_tail_model_dot(const std::vector<float> & w, const std::vector<float> & x) {
    const int n = static_cast<int>(w.size());
    float acc = avx2_block_sum(w, x, n & ~31);
    int c = n & ~31;
    for (; c + 8 <= n; c += 8) {
        float p[8];
        for (int j = 0; j < 8; ++j) p[j] = w[c + j] * x[c + j];
        for (int j = 0; j < 8; ++j) acc = acc + p[j];
    }
    if (n - c >= 4) {
        float p[4];
        for (int j = 0; j < 4; ++j) p[j] = w[c + j] * x[c + j];
        for (int j = 0; j < 4; ++j) acc = acc + p[j];
        c += 4;
    }
    for (; c < n; ++c) acc = std::fma(w[c], x[c], acc);
    return acc;
}

// Monotonic float ordering, for a ulp distance.
int64_t ordered_bits(float v) {
    const uint32_t u = bits(v);
    return (u & 0x80000000u) ? -static_cast<int64_t>(~u) : static_cast<int64_t>(u | 0x80000000u);
}

// ---- legacy orders, for the drift measurements -----------------------------

// q38_hc_dense_gemv_f32_kernel()/dense_f32_gemv_kernel(): lane-strided FMA
// chains reduced by warp_sum()'s shfl_down tree (offsets 16, 8, 4, 2, 1).
float legacy_warp_order_dot(const std::vector<float> & w, const std::vector<float> & x) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
        float acc = 0.0f;
        for (size_t i = static_cast<size_t>(l); i < w.size(); i += 32) {
            acc = std::fma(w[i], x[i], acc);
        }
        lane[l] = acc;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int l = 0; l < offset; ++l) lane[l] = lane[l] + lane[l + offset];
    }
    return lane[0];
}

float legacy_scalar_dot(const std::vector<float> & w, const std::vector<float> & x) {
    float acc = 0.0f;
    for (size_t i = 0; i < w.size(); ++i) acc = acc + w[i] * x[i];
    return acc;
}

// The previous gate path rounded the F32 weight to F16 before an F16 dot.
float legacy_half_weight_dot(const std::vector<float> & w, const std::vector<float> & x) {
    std::vector<float> wh(w.size());
    for (size_t i = 0; i < w.size(); ++i) wh[i] = static_cast<float>(static_cast<_Float16>(w[i]));
    return reference_dot(wh, x);
}

void compare(const std::string & what, const std::vector<float> & w, const std::vector<float> & x,
             bool measure_half) {
    const float ref = reference_dot(w, x);
    const float src = source_model_dot(w, x);
    const float tail_model = compiled_tail_model_dot(w, x);
    const bool has_tail = (w.size() & 31u) != 0;
    ++g_cases;
    if (has_tail) {
        ++g_tail_cases;
        if (bits(tail_model) != bits(ref)) ++g_tail_model_diff;
    } else if (bits(src) != bits(ref)) {
        // A multiple of 32 has no leftovers, so the two models and the oracle
        // must agree exactly.
        ++g_src_model_diff;
        std::printf("MODEL FAIL %s n=%zu ref=%.9g (%08x) src=%.9g (%08x) tail=%.9g (%08x)\n",
                    what.c_str(), w.size(), ref, bits(ref), src, bits(src), tail_model,
                    bits(tail_model));
    }

    const float cands[3] = {candidate_gate(w, x), candidate_router(w, x), candidate_inject(w, x)};
    for (int k = 0; k < 3; ++k) {
        ++g_impl_cases[k];
        bool bad = false;
        if (bits(ref) != bits(cands[k])) {
            // The only accepted difference is the compiler's contraction of the
            // leftover tail: the oracle must equal the compiled-tail model, the
            // candidate the source-text model, and the length must have a tail
            // (the model widths 2560 / 10240 never do).
            const bool explained =
                has_tail && bits(ref) == bits(tail_model) && bits(cands[k]) == bits(src);
            if (!explained) {
                bad = true;
            } else {
                ++g_tail_diff;
                const int64_t ulp = std::llabs(ordered_bits(ref) - ordered_bits(cands[k]));
                if (ulp > g_tail_max_ulp) g_tail_max_ulp = ulp;
            }
        }
        if (bits(src) != bits(cands[k])) bad = true;  // must implement the source text
        if (bad) {
            ++g_impl_fail[k];
            ++g_failures;
            if (g_impl_fail[k] <= 3) {
                std::printf("FAIL %s [%s] n=%zu ref=%.9g (%08x) src=%.9g (%08x) cand=%.9g (%08x)\n",
                            what.c_str(), g_impl_name[k], w.size(), ref, bits(ref), src, bits(src),
                            cands[k], bits(cands[k]));
            }
        }
    }

    // The two legacy orders are different arithmetics; a correct comparison must
    // reject them on a large fraction of the random rows.
    if (g_warp_cases < 5000 && w.size() >= 2048) {
        ++g_warp_cases;
        if (bits(legacy_warp_order_dot(w, x)) != bits(ref)) ++g_warp_diff;
    }
    if (bits(legacy_scalar_dot(w, x)) != bits(ref)) ++g_naive_diff;

    if (measure_half && std::isfinite(ref) && ref != 0.0f) {
        const float h = legacy_half_weight_dot(w, x);
        if (std::isfinite(h)) {
            const double rel = std::fabs(static_cast<double>(h) - ref) / std::fabs(ref);
            g_half_rel_sum += rel;
            ++g_half_rel_n;
            if (rel > g_half_rel_max) g_half_rel_max = rel;
        }
    }
}

std::vector<float> make_normal(std::mt19937 & rng, size_t n, double sigma) {
    std::normal_distribution<double> d(0.0, sigma);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(d(rng));
    return v;
}

void random_groups(int n_random) {
    std::mt19937 rng(0x5eed1234u);

    // The model geometries: 2560 both for the shared gate and the router rows,
    // 10240 for the HC inject (hc * n_embd == 4 * 2560), 2048/4096/512/640 as
    // extra widths that still cover the block boundary.
    const size_t dims[] = {2560, 2560, 10240, 4096, 2048, 512, 640};
    for (size_t d : dims) {
        for (int i = 0; i < n_random; ++i) {
            compare("random n=" + std::to_string(d), make_normal(rng, d, 0.08),
                    make_normal(rng, d, 1.0), d == 2560);
        }
    }

    // Every n in [1, 200] walks the 32-value block boundary and the non-fused
    // tail; the model never hits the tail (2560 and 10240 are multiples of 32),
    // but a different tensor could.
    for (int n = 1; n <= 200; ++n) {
        compare("len sweep n=" + std::to_string(n), make_normal(rng, static_cast<size_t>(n), 0.3),
                make_normal(rng, static_cast<size_t>(n), 2.0), false);
    }

    // Dynamic-range sweep on the model shape.
    for (int e = -12; e <= 12; e += 2) {
        const double s = std::pow(10.0, e);
        compare("weight scale 1e" + std::to_string(e),
                make_normal(rng, 2560, 0.08 * s), make_normal(rng, 2560, 1.0), false);
        compare("act scale 1e" + std::to_string(e),
                make_normal(rng, 2560, 0.08), make_normal(rng, 2560, s), false);
    }

    // Alternating signs: maximum cancellation, where the accumulator order is
    // most visible.
    {
        std::vector<float> w(2560), x(2560);
        for (size_t i = 0; i < w.size(); ++i) {
            const float s = (i % 2 == 0) ? 1.0f : -1.0f;
            w[i] = s * static_cast<float>(1.0 + (i % 97) * 0.01);
            x[i] = s * static_cast<float>(1.0 - (i % 89) * 0.01);
        }
        compare("alternating signs", w, x, false);
    }

    // Both operands exactly representable in F16: the legacy F16-weight path and
    // the F32 path must agree bit for bit here.
    {
        std::vector<float> w = make_normal(rng, 2560, 0.08);
        std::vector<float> x = make_normal(rng, 2560, 1.0);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = static_cast<float>(static_cast<_Float16>(w[i]));
            x[i] = static_cast<float>(static_cast<_Float16>(x[i]));
        }
        compare("both operands F16-exact", w, x, false);
    }

    // Special values: zeros, signed zeros, denormals, huge magnitudes, NaN and
    // infinities must round identically in every implementation.
    {
        const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, 1e-38f, -1e-38f, 1e-45f, 5.0e-40f,
                                  1e38f, -1e38f, 3.0e38f, 65504.0f, 65536.0f};
        for (float sw : specials) {
            for (float sx : specials) {
                compare("special", std::vector<float>(2560, sw), std::vector<float>(2560, sx),
                        false);
            }
        }
        std::vector<float> w(2560, 1.0f), x(2560, 1.0f);
        w[7] = std::nanf("");
        compare("nan weight", w, x, false);
        x[33] = INFINITY;
        compare("inf activation", w, x, false);
        x[33] = -INFINITY;
        compare("neg inf activation", w, x, false);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const int n_random = argc > 1 ? std::atoi(argv[1]) : 200;
    random_groups(n_random);

    const double half_rel_mean = g_half_rel_n > 0 ? g_half_rel_sum / g_half_rel_n : 0.0;
    std::printf("cases=%ld failures=%d\n", g_cases, g_failures);
    for (int k = 0; k < 3; ++k) {
        std::printf("  %-34s mismatches %ld / %ld\n", g_impl_name[k], g_impl_fail[k],
                    g_impl_cases[k]);
    }
    std::printf("tail lengths (n %% 32 != 0): %ld cases, %ld candidate/ggml divergences, max %ld ulp;"
                " compiled-tail model reproduces ggml on %ld of them (%ld misses)\n",
                g_tail_cases, g_tail_diff, g_tail_max_ulp, g_tail_cases - g_tail_model_diff,
                g_tail_model_diff);
    std::printf("legacy warp order (dense_f32_gemv_kernel) differs from ggml AVX2 on %ld / %ld"
                " model-shaped rows\n", g_warp_diff, g_warp_cases);
    std::printf("legacy ascending scalar differs from ggml on %ld / %ld cases\n", g_naive_diff,
                g_cases);
    std::printf("F16-rounded gate weight vs F32 dot: mean rel %.3e, max rel %.3e over %ld rows\n",
                half_rel_mean, g_half_rel_max, g_half_rel_n);
    if (g_failures != 0) {
        std::printf("F32 DOT HOST CHECK FAILED\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
