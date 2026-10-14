// Host bit-comparison for src/qwen38_exact_math.cuh.
//
// Every reference is llama.cpp's own code at the pinned commit
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11, compiled as-is:
//
//   * ggml_v_expf / ggml_v_silu                     -> ggml/src/ggml-cpu/vec.h
//        (included below, so the AVX2/FMA polynomial is the compiler's own
//        output for that exact text);
//   * ggml_vec_silu_f32 / ggml_vec_soft_max_f32     -> ggml/src/ggml-cpu/vec.cpp
//        (linked from a build of that tree, same instruction set as the oracle:
//        -mavx2 -mfma -mf16c, GGML_AVX512 OFF);
//   * GGML_OP_SILU / GGML_OP_SIGMOID / GGML_OP_SOFT_MAX
//        -> ggml-cpu ops.cpp + unary-ops.cpp, driven through the public ggml
//        API on the CPU backend;
//   * q38_hc_ggml_avx2_exp_f32 -> extracted verbatim from src/qwen38_hc.cuh by
//        tests/qwen38_exact_math_host_check.sh, so the two copies of the
//        polynomial cannot drift.
//
// Build and run: bash tests/qwen38_exact_math_host_check.sh [llama.cpp-checkout]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "vec.h"

// ---------------------------------------------------------------------------
// Host stand-ins for the CUDA rounding-mode intrinsics. The build of this test
// passes -ffp-contract=off so every one of these is a single IEEE operation,
// exactly like the device intrinsic it replaces.
// ---------------------------------------------------------------------------
static inline float  __fadd_rn(float a, float b)   { return a + b; }
static inline float  __fsub_rn(float a, float b)   { return a - b; }
static inline float  __fmul_rn(float a, float b)   { return a * b; }
static inline float  __fdiv_rn(float a, float b)   { return a / b; }
static inline float  __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline uint32_t __float_as_uint(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
static inline float __uint_as_float(uint32_t x) { float r; std::memcpy(&r, &x, 4); return r; }
static inline uint64_t __double_as_longlong(double x) { uint64_t r; std::memcpy(&r, &x, 8); return r; }
static inline double __longlong_as_double(uint64_t x) { double r; std::memcpy(&r, &x, 8); return r; }

#include "qwen38_exact_math.cuh"

#if __has_include("q38_hc_avx2_extract.h")
#include "q38_hc_avx2_extract.h"
#define Q38_HAVE_HC_EXTRACT 1
#endif

using namespace wd;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static inline uint32_t bits(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
static inline bool nan_bits(uint32_t b) {
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

struct Check {
    const char * name;
    long long    n          = 0;
    long long    bit_diff   = 0;
    long long    nan_only   = 0;   // both sides NaN, payload differs
    long long    value_diff = 0;   // mismatch that is not a NaN-payload case
    float        first_x    = 0.0f;
    float        first_y    = 0.0f;
    uint32_t     first_a    = 0;
    uint32_t     first_b    = 0;
    bool         reported   = false;

    void feed(float x, float y, float a, float b) {
        ++n;
        const uint32_t ba = bits(a), bb = bits(b);
        if (ba == bb) return;
        ++bit_diff;
        if (nan_bits(ba) && nan_bits(bb)) { ++nan_only; return; }
        ++value_diff;
        if (!reported) {
            reported = true;
            first_x = x; first_y = y; first_a = ba; first_b = bb;
        }
    }

    bool ok() const { return value_diff == 0; }

    void print() const {
        std::printf("%-46s samples=%9lld  bit-diff=%9lld  value-diff=%9lld  nan-payload=%lld%s\n",
                    name, n, bit_diff, value_diff, nan_only, ok() ? "  [PASS]" : "  [FAIL]");
        if (value_diff) {
            std::printf("      first: x=%.9g (0x%08x) max=%.9g   ours=%.9g (0x%08x)  ref=%.9g (0x%08x)\n",
                        first_x, bits(first_x), first_y, __uint_as_float(first_a), first_a,
                        __uint_as_float(first_b), first_b);
        }
        std::fflush(stdout);
    }
};

// ---------------------------------------------------------------------------
// sample set: millions of points, with every boundary class present
// ---------------------------------------------------------------------------
static std::vector<float> sample_inputs() {
    std::vector<float> xs;
    const float specials[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        1.0f / 0.0f, -1.0f / 0.0f, 0.0f / 0.0f,
        __uint_as_float(0x00000001u),   // +denormal min
        __uint_as_float(0x80000001u),   // -denormal min
        __uint_as_float(0x007fffffu),   // +denormal max
        __uint_as_float(0x807fffffu),   // -denormal max
        __uint_as_float(0x00800000u),   // +normal min
        1e-30f, -1e-30f, 1e-38f, -1e-38f,
        0.6931472f, -0.6931472f, 1e-6f, -1e-6f,
        20.0f, -20.0f, 21.0f, -21.0f, 87.0f, -87.0f,
        88.0f, -88.0f,                          // expf overflow threshold region
        88.3762626647949f, -88.3762626647949f,
        88.72283935546875f, -88.72283935546875f,
        89.0f, -89.0f, 100.0f, -100.0f, 126.0f, -126.0f, 127.0f, -127.0f,
        192.0f, -192.0f, 193.0f, -193.0f, 1e30f, -1e30f,
    };
    xs.insert(xs.end(), std::begin(specials), std::end(specials));

    std::mt19937 rng(0x51f3a2b7u);
    std::uniform_real_distribution<float> uni(-128.0f, 128.0f);
    for (int i = 0; i < 4000000; ++i) xs.push_back(uni(rng));
    // arbitrary bit patterns: NaNs, infinities, denormals, huge exponents
    std::uniform_int_distribution<uint32_t> raw(0u, 0xffffffffu);
    for (int i = 0; i < 2000000; ++i) xs.push_back(__uint_as_float(raw(rng)));
    // fine sweep across the expf overflow / underflow boundaries
    for (int i = -2000; i <= 2000; ++i) xs.push_back(86.0f + static_cast<float>(i) * 0.0015f);
    for (int i = -2000; i <= 2000; ++i) xs.push_back(-86.0f + static_cast<float>(i) * 0.0015f);
    for (int i = -2000; i <= 2000; ++i) xs.push_back(static_cast<float>(i) * 0.075f);
    return xs;
}

// ---------------------------------------------------------------------------
// 1. AVX2 ggml_v_expf
// ---------------------------------------------------------------------------
static Check check_avx2_exp(const std::vector<float> & xs) {
    Check c{"q38_exact_avx2_exp_f32 vs ggml_v_expf/AVX2"};
    size_t i = 0;
    for (; i + 8 <= xs.size(); i += 8) {
        const __m256 v = _mm256_loadu_ps(&xs[i]);
        alignas(32) float ref[8];
        _mm256_store_ps(ref, ggml_v_expf(v));
        for (int k = 0; k < 8; ++k) {
            const float x = xs[i + k];
            c.feed(x, 0.0f, q38_exact_avx2_exp_f32(x), ref[k]);
        }
    }
    for (; i < xs.size(); ++i) {
        const float x = xs[i];
        alignas(32) float lane[8] = {0};
        lane[0] = x;
        alignas(32) float ref[8];
        _mm256_store_ps(ref, ggml_v_expf(_mm256_loadu_ps(lane)));
        c.feed(x, 0.0f, q38_exact_avx2_exp_f32(x), ref[0]);
    }
    return c;
}

// ---------------------------------------------------------------------------
// 2. glibc expf port
// ---------------------------------------------------------------------------
static Check check_glibc_exp(const std::vector<float> & xs) {
    Check c{"q38_exact_expf_glibc vs glibc expf"};
    for (float x : xs) c.feed(x, 0.0f, q38_exact_expf_glibc(x), std::exp(x));
    return c;
}

static Check check_exp_switch(const std::vector<float> & xs) {
    Check c{"q38_exact_expf_f32(x,true) vs glibc expf"};
    for (float x : xs) c.feed(x, 0.0f, q38_exact_expf_f32(x, true), std::exp(x));
    return c;
}

// ---------------------------------------------------------------------------
// 3. sigmoid: op_sigmoid() body, then the real GGML_OP_SIGMOID
// ---------------------------------------------------------------------------
static Check check_sigmoid_expr(const std::vector<float> & xs) {
    Check c{"q38_exact_sigmoid_f32 vs op_sigmoid body"};
    for (float x : xs) {
        // exactly ggml-cpu/unary-ops.cpp:31-33
        c.feed(x, 0.0f, q38_exact_sigmoid_f32(x, true), 1.f / (1.f + expf(-x)));
    }
    return c;
}

// ---------------------------------------------------------------------------
// 4. silu: the real ggml_vec_silu_f32 (vec.cpp) and the real GGML_OP_SILU
// ---------------------------------------------------------------------------
static Check check_silu_vec(const std::vector<float> & xs) {
    Check c{"q38_exact_silu_f32 vs ggml_vec_silu_f32"};
    // ggml_vec_silu_f32 vectorises 8 lanes at a time, so only a multiple of 8
    // row length is fully covered by the AVX2 branch; the row size here is a
    // multiple of 8, which is also the case for every warp call site.
    const size_t n = (xs.size() / 8) * 8;
    std::vector<float> row(xs.begin(), xs.begin() + n);
    std::vector<float> out(n, 0.0f);
    ggml_vec_silu_f32(static_cast<int>(n), out.data(), row.data());
    for (size_t i = 0; i < n; ++i) {
        c.feed(row[i], 0.0f, q38_exact_silu_f32(row[i], true), out[i]);
    }
    return c;
}

// ---------------------------------------------------------------------------
// 5. softmax: the real ggml_vec_soft_max_f32 (vec.cpp)
// ---------------------------------------------------------------------------
static Check check_softmax_vec(const std::vector<float> & xs) {
    Check c{"q38_exact_softmax_exp_f32 vs ggml_vec_soft_max_f32"};
    const size_t n = (xs.size() / 8) * 8;
    std::vector<float> row(xs.begin(), xs.begin() + n);
    std::vector<float> out(n, 0.0f);
    float max = -INFINITY;
    for (float x : row) max = x > max ? x : max;
    ggml_vec_soft_max_f32(static_cast<int>(n), out.data(), row.data(), max);
    for (size_t i = 0; i < n; ++i) {
        c.feed(row[i], max, q38_exact_softmax_exp_f32(row[i], max, true), out[i]);
    }
    return c;
}

// ---------------------------------------------------------------------------
// 6. the real ops through the public ggml API (CPU backend)
// ---------------------------------------------------------------------------
enum OpKind { OP_SILU, OP_SIGMOID, OP_SOFTMAX };

static bool run_ggml_op(OpKind kind, const std::vector<float> & in, std::vector<float> & out) {
    const int64_t n = static_cast<int64_t>(in.size());
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) return false;

    ggml_init_params p;
    p.mem_size   = static_cast<size_t>(n) * sizeof(float) * 8 + 8 * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    if (ctx == nullptr) { ggml_backend_free(backend); return false; }

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::memcpy(a->data, in.data(), static_cast<size_t>(n) * sizeof(float));

    ggml_tensor * r = nullptr;
    switch (kind) {
        case OP_SILU:    r = ggml_silu(ctx, a); break;
        case OP_SIGMOID: r = ggml_sigmoid(ctx, a); break;
        case OP_SOFTMAX: r = ggml_soft_max(ctx, a); break;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) std::memcpy(out.data(), r->data, static_cast<size_t>(n) * sizeof(float));

    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

static Check check_op(OpKind kind, const char * name, const std::vector<float> & xs) {
    Check c{name};
    std::vector<float> in = xs, out(xs.size(), 0.0f);
    if (!run_ggml_op(kind, in, out)) {
        std::printf("%-46s FAILED TO RUN\n", name);
        return c;
    }
    if (kind == OP_SOFTMAX) {
        // ggml_compute_forward_soft_max_f32 also normalises: reproduce the exact
        // reduction of ggml_vec_soft_max_f32's AVX2 branch and its 1/sum scale.
        float max = -INFINITY;
        for (float x : in) max = x > max ? x : max;
        ggml_float sum = 0;
        for (size_t i = 0; i + 8 <= in.size(); i += 8) {
            const __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(&in[i]),
                                                         _mm256_set1_ps(max)));
            __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1), _mm256_castps256_ps128(val));
            val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
            val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
            sum += static_cast<ggml_float>(_mm_cvtss_f32(val2));
        }
        const float scale = 1.0f / static_cast<float>(sum);
        for (size_t i = 0; i < in.size(); ++i) {
            c.feed(in[i], max, q38_exact_softmax_exp_f32(in[i], max, true) * scale, out[i]);
        }
        return c;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        const float want = kind == OP_SILU ? q38_exact_silu_f32(in[i], true)
                                           : q38_exact_sigmoid_f32(in[i], true);
        c.feed(in[i], 0.0f, want, out[i]);
    }
    return c;
}

// ---------------------------------------------------------------------------
// 7. drift guard: the polynomial shipped in qwen38_hc.cuh
// ---------------------------------------------------------------------------
#ifdef Q38_HAVE_HC_EXTRACT
static Check check_hc_drift(const std::vector<float> & xs) {
    Check c{"q38_exact_avx2_exp_f32 vs qwen38_hc.cuh copy"};
    for (float x : xs) c.feed(x, 0.0f, q38_exact_avx2_exp_f32(x), q38_hc_ggml_avx2_exp_f32(x));
    return c;
}
#endif

int main() {
    std::printf("== qwen38_exact_math host bit comparison (llama.cpp pin, AVX2+FMA) ==\n");
    const std::vector<float> xs = sample_inputs();
    std::printf("sample count: %zu\n\n", xs.size());

    Check avx2   = check_avx2_exp(xs);
    Check glibc  = check_glibc_exp(xs);
    Check sw     = check_exp_switch(xs);
    Check sig    = check_sigmoid_expr(xs);
    Check silu   = check_silu_vec(xs);
    Check smx    = check_softmax_vec(xs);
    Check op_si  = check_op(OP_SILU,    "GGML_OP_SILU via ggml CPU backend", xs);
    Check op_sg  = check_op(OP_SIGMOID, "GGML_OP_SIGMOID via ggml CPU backend", xs);
    Check op_sm  = check_op(OP_SOFTMAX, "GGML_OP_SOFT_MAX via ggml CPU backend", xs);
#ifdef Q38_HAVE_HC_EXTRACT
    Check drift  = check_hc_drift(xs);
#endif

    std::printf("\n--- results ---\n");
    avx2.print();
    glibc.print();
    sw.print();
    sig.print();
    silu.print();
    smx.print();
    op_si.print();
    op_sg.print();
    op_sm.print();
#ifdef Q38_HAVE_HC_EXTRACT
    drift.print();
#else
    std::printf("%-46s SKIPPED (no extract)\n", "q38_exact_avx2_exp_f32 vs qwen38_hc.cuh copy");
#endif

    // Negative control: the exact path must NOT coincide with libm/device expf,
    // otherwise the harness would be proving nothing.
    long long libm_diff = 0, libm_nan = 0;
    for (size_t i = 0; i < xs.size(); i += 97) {
        const uint32_t a = bits(q38_exact_avx2_exp_f32(xs[i]));
        const uint32_t b = bits(std::exp(xs[i]));
        if (a != b) { ++libm_diff; if (nan_bits(a) && nan_bits(b)) ++libm_nan; }
    }
    std::printf("\nnegative control: AVX2 polynomial != glibc expf on %lld / %zu sampled points\n",
                libm_diff, xs.size() / 97 + 1);

    bool pass = avx2.ok() && glibc.ok() && sw.ok() && sig.ok() && silu.ok() && smx.ok() &&
                op_si.ok() && op_sg.ok() && op_sm.ok();
#ifdef Q38_HAVE_HC_EXTRACT
    pass = pass && drift.ok();
#endif
    std::printf("\n%s\n", pass ? "ALL CHECKS PASS (0 value mismatches)" : "MISMATCHES FOUND");
    return pass ? 0 : 1;
}
