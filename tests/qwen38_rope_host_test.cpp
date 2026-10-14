// Host bit-comparison for src/qwen38_rope.cuh's exact IMRoPE.
//
// The reference is llama.cpp's own CPU rope at the pinned commit
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11: the test drives ggml_rope_multi()
// through the public ggml API on the CPU backend, with exactly the op params
// src/models/qwen4exp.cpp passes (mode GGML_ROPE_TYPE_IMROPE, n_dims =
// qwen4exp.rope.dimension_count, sections = qwen4exp.rope.dimension_sections,
// freq_base = qwen4exp.rope.freq_base = 1e7, freq_scale 1, ext_factor 0,
// attn_factor 1, the YaRN defaults, n_ctx_orig = 262144), and compares it bit
// for bit against
//
//   * wd::q38glibc::sinf / wd::q38glibc::cosf   - the glibc transcription used
//     by the kernel, against the host libm's own sinf/cosf;
//   * wd::q38imrope::rotate_pair_exact          - the pair rotation the kernel
//     performs, including the fused multiply-adds;
//   * wd::q38imrope::rotate_pair_unfused        - the unfused form, which is
//     what the CPU source literally says, so the report shows why the fused
//     form is the one that matches.
//
// The model geometry is exercised first (head_dim 256, n_dims 64, base 1e7,
// sections [11,11,10,0]), then the boundaries of the three internal glibc range
// reduction paths (|theta| < pi/4, < 120, and the large-argument table path)
// plus other bases/dimensions.
//
// Build and run: bash tests/qwen38_rope_host_check.sh [llama.cpp-checkout]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "qwen38_imrope_math.cuh"

using namespace wd;

static const int HEAD_DIM = 256;

static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// Deterministic splitmix64 so every case is reproducible.
static uint64_t g_seed = 0x20261014ull;
static uint64_t next_u64() {
    g_seed += 0x9E3779B97F4A7C15ull;
    uint64_t z = g_seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float next_float() {
    // [-1, 1) from the top 24 bits; no infinities, no denormals.
    return (float)((int64_t)(next_u64() >> 40) - (1 << 23)) / (float)(1 << 23);
}

struct CaseResult {
    long exact_bad = 0;
    long unfused_bad = 0;
    long trig_bad = 0;
    long total = 0;
};

static CaseResult run_case(const char * label,
                           int n_dims, float base, const int sections[4],
                           int n_heads, const int pos[4], int n_ctx_orig,
                           bool report_trig) {
    CaseResult out;

    const size_t n = (size_t)HEAD_DIM * n_heads;
    std::vector<float> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = next_float();

    static uint8_t work[64 * 1024 * 1024];
    struct ggml_init_params ip = { sizeof(work), work, true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::printf("ggml_init failed\n"); std::exit(2); }

    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, n_heads, 1);
    struct ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    int sect[4] = { sections[0], sections[1], sections[2], sections[3] };
    int p4[4] = { pos[0], pos[1], pos[2], pos[3] };

    struct ggml_tensor * r = ggml_rope_multi(ctx, a, p, nullptr, n_dims, sect,
        GGML_ROPE_TYPE_IMROPE, n_ctx_orig, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { std::printf("ggml_backend_alloc_ctx_tensors failed\n"); std::exit(2); }

    ggml_backend_tensor_set(a, in.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(p, p4, 0, 4 * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> ref(n);
    ggml_backend_tensor_get(r, ref.data(), 0, n * sizeof(float));

    const float ts = q38imrope::theta_scale(base, n_dims);
    const int pairs = n_dims / 2;
    for (int h = 0; h < n_heads; ++h) {
        const int hb = h * HEAD_DIM;
        for (int j = 0; j < pairs; ++j) {
            const int axis = q38imrope::axis_for_pair(j, sect[0], sect[1], sect[2], sect[3]);
            const float theta_base = (float)p4[axis];
            const float x0 = in[hb + j];
            const float x1 = in[hb + j + pairs];

            float e0, e1;
            q38imrope::rotate_pair_exact(theta_base, ts, j, x0, x1, &e0, &e1);
            if (bits(ref[hb + j]) != bits(e0)) out.exact_bad++;
            if (bits(ref[hb + j + pairs]) != bits(e1)) out.exact_bad++;

            const float theta = q38imrope::theta_for_pair(theta_base, ts, j);
            const float sn = q38glibc::sinf(theta);
            const float cs = q38glibc::cosf(theta);
            float u0, u1;
            q38imrope::rotate_pair_unfused(x0, x1, sn, cs, &u0, &u1);
            if (bits(ref[hb + j]) != bits(u0)) out.unfused_bad++;
            if (bits(ref[hb + j + pairs]) != bits(u1)) out.unfused_bad++;

            if (report_trig) {
                if (bits(sn) != bits(sinf(theta))) out.trig_bad++;
                if (bits(cs) != bits(cosf(theta))) out.trig_bad++;
            }
            out.total += 2;
        }
    }

    std::printf("%-38s n_dims=%-3d base=%-9g sections=[%d,%d,%d,%d] heads=%-3d pos=[%d,%d,%d,%d]  "
                "exact_bad=%-4ld unfused_bad=%-4ld trig_bad=%-3ld / %ld\n",
                label, n_dims, base, sections[0], sections[1], sections[2], sections[3],
                n_heads, pos[0], pos[1], pos[2], pos[3],
                out.exact_bad, out.unfused_bad, out.trig_bad, out.total);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return out;
}

int main() {
    const int text_sections[4] = { 11, 11, 10, 0 };   // qwen4exp.rope.dimension_sections
    const int alt_sections[4]  = { 16, 16, 16, 0 };
    const int s4_sections[4]   = { 4, 4, 4, 0 };

    CaseResult total;
    const int model_heads = 24;   // qwen4exp.attention.head_count / compress ratio

    auto add = [&](const CaseResult & c) {
        total.exact_bad += c.exact_bad;
        total.unfused_bad += c.unfused_bad;
        total.trig_bad += c.trig_bad;
        total.total += c.total;
    };

    // ---- the model geometry, positions 1..3 ---------------------------------
    {
        const int p1[4]  = { 1, 1, 1, 1 };
        const int p2[4]  = { 2, 2, 2, 2 };
        const int p3[4]  = { 3, 3, 3, 3 };
        add(run_case("model pos 1", 64, 1e7f, text_sections, model_heads, p1, 262144, true));
        add(run_case("model pos 2", 64, 1e7f, text_sections, model_heads, p2, 262144, true));
        add(run_case("model pos 3", 64, 1e7f, text_sections, model_heads, p3, 262144, true));
    }

    // ---- the three glibc reduction paths, plus mixed axes -------------------
    {
        const int a[4] = {   1,   1,   1,   1 };   // |theta| < pi/4
        const int b[4] = {  59,  59,  59,  59 };   // < 120 fast reduction
        const int c[4] = { 120, 120, 120, 120 };   // exactly at the switch
        const int d[4] = { 121, 121, 121, 121 };   // large-argument table
        const int e[4] = { 5, 9, 13, 2 };          // different value per axis
        const int f[4] = { 262143, 262143, 262143, 262143 };
        add(run_case("glibc path small", 64, 1e7f, text_sections, 2, a, 262144, false));
        add(run_case("glibc path <120", 64, 1e7f, text_sections, 2, b, 262144, false));
        add(run_case("glibc path ==120", 64, 1e7f, text_sections, 2, c, 262144, false));
        add(run_case("glibc path >120", 64, 1e7f, text_sections, 2, d, 262144, false));
        add(run_case("mixed axes", 64, 1e7f, text_sections, 2, e, 262144, false));
        add(run_case("n_ctx limit", 64, 1e7f, text_sections, 2, f, 262144, false));
    }

    // ---- large positions, other n_dims / bases / section lists --------------
    {
        const int p[4] = { 200000, 200000, 200000, 200000 };
        add(run_case("large position", 64, 1e7f, text_sections, 2, p, 262144, false));
    }
    {
        const int p[4] = { 1, 1, 1, 1 };
        add(run_case("n_dims 128", 128, 1e7f, text_sections, 2, p, 262144, false));
        add(run_case("n_dims 32", 32, 1e7f, text_sections, 2, p, 262144, false));
        add(run_case("base 1e6", 64, 1e6f, text_sections, 2, p, 262144, false));
        add(run_case("base 1e4", 64, 1e4f, text_sections, 2, p, 262144, false));
        add(run_case("base 1", 64, 1.0f, text_sections, 2, p, 262144, false));
        add(run_case("sections [16,16,16,0]", 64, 1e6f, alt_sections, 2, p, 262144, false));
        add(run_case("sections [4,4,4,0]", 64, 1e4f, s4_sections, 2, p, 262144, false));
    }
    {
        const int p[4] = { 3, 3, 3, 3 };
        add(run_case("n_dims 16 base 1e4", 16, 1e4f, text_sections, 2, p, 262144, false));
    }

    std::printf("\nsummary: exact_bad=%ld  unfused_bad=%ld  trig_bad=%ld  elements=%ld\n",
                total.exact_bad, total.unfused_bad, total.trig_bad, total.total);
    if (total.exact_bad != 0) {
        std::printf("FAIL: the kernel's exact rotation does not reproduce ggml_rope_multi\n");
        return 1;
    }
    std::printf("PASS: every rotated element is bit-identical to ggml_rope_multi\n");
    return 0;
}
