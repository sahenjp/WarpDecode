// Host bit-comparison for the MoE router weight chain in src/qwen38_moe.cuh.
//
// The reference is llama.cpp's own code at the pinned commit
// 6fcaa16f4b360649933a54d1f91ad40ed35c0e11: the graph below is the router part
// of llm_graph_context::build_moe_ffn (src/llama-graph.cpp:2022-2152) for this
// model's shape (gating_op = SOFTMAX, norm_w = true, no exp_probs_b, w_scale
// passed in), evaluated by the real ggml CPU backend with -ggml-org's graph
// with per-op outputs. Every op runs the pinned implementation, so the
// comparison is against the same arithmetic the CPU oracle executed.
//
// The candidate is the host transcription of the device functions that live in
// src/qwen38_moe.cuh (q38_moe_router_exact / q38_moe_softmax_lane_sum), copied
// verbatim out of that file by tests/qwen38_router_host_check.sh so the two
// copies cannot drift.
//
// Usage: qwen38_router_host_test [corpus.bin] [n_synthetic]

#include <algorithm>
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
#include "ggml-impl.h"

// ---------------------------------------------------------------------------
// Host stand-ins for the CUDA rounding-mode intrinsics. The build passes
// -ffp-contract=off so every one of these is a single IEEE operation, exactly
// like the device intrinsic it replaces.
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

// The extracted router source is included for real: it carries
// q38_moe_router_exact / q38_moe_softmax_lane_sum verbatim from
// src/qwen38_moe.cuh, plus the qwen38_exact_math.cuh helpers they call. The
// __device__ / __forceinline__ qualifiers are defined away by the build script
// and -ffp-contract=off keeps every rounding-mode intrinsic a single IEEE op.
#include "qwen38_moe_router_extract.h"

using wd::q38_test_router_probs_row;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static inline uint32_t bits(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
static inline bool is_nan_bits(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0; }

struct Case {
    std::string name;
    std::vector<float> logits;
    std::vector<int> ids;
    std::vector<float> weights;
    float w_scale = 1.0f;
    bool synthetic = false;
};

// ---------------------------------------------------------------------------
// Oracle reference: build_moe_ffn's router part as a ggml graph.
// ---------------------------------------------------------------------------
struct RefResult {
    std::vector<float> probs;
    std::vector<int> sel;
    std::vector<float> weights;
};

struct RefRunner {
    ggml_backend_t cpu = nullptr;
    ggml_context * ctx = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * soft = nullptr;
    ggml_tensor * sel = nullptr;
    ggml_tensor * wts = nullptr;
    int n_expert = 0;
    int k = 0;

    ggml_cgraph * graph = nullptr;

    void init(int ne, int kk) {
        n_expert = ne;
        k = kk;
        cpu = ggml_backend_cpu_init();
        if (!cpu) { std::fprintf(stderr, "ggml_backend_cpu_init failed\n"); std::exit(2); }
        // A plain (non no_alloc) context is enough: the reference ops all have a
        // CPU implementation and no device buffer is involved.
        ggml_init_params p = { /*mem_size*/ 64 * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
        ctx = ggml_init(p);
        if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); std::exit(2); }

        logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, 1);
        soft   = ggml_soft_max(ctx, logits);                 // llama-graph.cpp:2043
        sel    = ggml_argsort_top_k(ctx, soft, kk);          // llama-graph.cpp:2109
        ggml_tensor * probs3 = ggml_reshape_3d(ctx, soft, 1, ne, 1);
        ggml_tensor * rows = ggml_get_rows(ctx, probs3, sel);          // llama-graph.cpp:2123
        ggml_tensor * w2 = ggml_reshape_2d(ctx, rows, kk, 1);
        ggml_tensor * wsum = ggml_sum_rows(ctx, w2);                   // llama-graph.cpp:2137
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);       // llama-graph.cpp:2141
        w2 = ggml_div(ctx, w2, wsum);                                  // llama-graph.cpp:2144
        w2 = ggml_reshape_3d(ctx, w2, 1, kk, 1);
        wts = w2;

        // One graph, recomputed per row. The context only grows when w_scale is
        // exercised (below), which is rare and bounded by the number of distinct
        // scales the caller asks for.
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, wts);
    }

    RefResult run(const float * x, float w_scale) {
        // The scale is baked in per call: the graph mirrors build_moe_ffn's
        // optional step, and this model asks for 1.0f where ggml skips it.
        ggml_tensor * out = wts;
        if (w_scale != 0.0f && w_scale != 1.0f) {
            out = ggml_scale(ctx, wts, w_scale);
            graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
        }

        std::memcpy(logits->data, x, sizeof(float) * (size_t) n_expert);
        ggml_backend_graph_compute(cpu, graph);

        RefResult r;
        r.probs.assign((float *) soft->data, (float *) soft->data + n_expert);
        const int * si = (const int *) sel->data;
        r.sel.assign(si, si + k);
        const float * wd = (const float *) out->data;
        r.weights.assign(wd, wd + k);
        return r;
    }
};

// ---------------------------------------------------------------------------
// Corpus / synthetic cases
// ---------------------------------------------------------------------------
static std::vector<Case> read_corpus(const char * path) {
    std::vector<Case> out;
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    uint32_t count = 0;
    if (std::fread(&count, sizeof(count), 1, f) != 1) { std::fprintf(stderr, "short corpus\n"); std::exit(2); }
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nlen = 0;
        std::fread(&nlen, sizeof(nlen), 1, f);
        std::string name(nlen, 0);
        std::fread(name.data(), 1, nlen, f);
        Case c;
        c.name = name;
        c.logits.resize(512);
        std::fread(c.logits.data(), sizeof(float), 512, f);
        c.ids.resize(10);
        std::fread(c.ids.data(), sizeof(int), 10, f);
        c.weights.resize(10);
        std::fread(c.weights.data(), sizeof(float), 10, f);
        out.push_back(c);
    }
    std::fclose(f);
    return out;
}

static float next_float(std::mt19937_64 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}

static std::vector<Case> make_synthetic(int per_kind) {
    std::vector<Case> out;
    const int N = 512;
    std::mt19937_64 rng(0x9e3779b97f4a7c15ull);
    auto push = [&](const std::string & name, const std::vector<float> & x) {
        Case c;
        c.name = name;
        c.logits = x;
        c.synthetic = true;
        c.ids.assign(10, 0);
        c.weights.assign(10, 0.0f);
        out.push_back(c);
    };

    for (int s = 0; s < per_kind; ++s) {
        // 1. the observed model range
        {
            std::vector<float> x(N);
            for (auto & v : x) v = next_float(rng, -14.0f, -1.0f);
            push("syn:model-range", x);
        }
        // 2. tight cluster in the top region (near-ties inside one ulp)
        {
            std::vector<float> x(N);
            const float base = next_float(rng, -6.0f, -2.0f);
            for (int i = 0; i < N; ++i) {
                x[i] = next_float(rng, -30.0f, -15.0f);
            }
            for (int i = 0; i < 24; ++i) {
                x[i] = base + next_float(rng, -1e-6f, 1e-6f) * (1.0f + (float) (i % 5));
            }
            for (int i = 0; i < 24; ++i) {
                const int a = (int) (rng() % 24), b = (int) (rng() % 24);
                x[a] = x[b];  // exact ties among the top candidates
            }
            push("syn:near-tie", x);
        }
        // 3. all values identical (full tie)
        {
            const float v = next_float(rng, -8.0f, 0.0f);
            std::vector<float> x(N, v);
            push("syn:all-equal", x);
        }
        // 4. extremes: -inf, 0, very negative, very positive
        {
            std::vector<float> x(N);
            for (int i = 0; i < N; ++i) {
                switch (i % 8) {
                    case 0: x[i] = -INFINITY; break;
                    case 1: x[i] = next_float(rng, -100.0f, -88.0f); break;
                    case 2: x[i] = next_float(rng, -1e30f, -1e20f); break;
                    case 3: x[i] = 0.0f; break;
                    case 4: x[i] = next_float(rng, 0.0f, 80.0f); break;
                    case 5: x[i] = -0.0f; break;
                    case 6: x[i] = std::ldexp(1.0f, -149); break;   // denormal
                    case 7: x[i] = next_float(rng, -20.0f, -5.0f); break;
                }
            }
            push("syn:extremes", x);
        }
        // 5. large positive everywhere (the v_expf flush region, |n| > 192)
        {
            std::vector<float> x(N);
            for (auto & v : x) v = next_float(rng, -0.5f, 60.0f);
            push("syn:hot", x);
        }
        // 6. max exactly at index 0 and exactly at index N-1
        {
            std::vector<float> x(N);
            for (auto & v : x) v = next_float(rng, -12.0f, -4.0f);
            x[0] = 0.0f;
            push("syn:max-first", x);
            x[0] = -12.0f;
            x[N - 1] = 0.0f;
            push("syn:max-last", x);
        }
        // 7a. every logit -inf: every exponent underflows, so the softmax row
        //     sum is zero and ggml_vec_scale_f32 leaves the row at -INFINITY.
        {
            std::vector<float> x(N, -INFINITY);
            push("syn:all-neg-inf", x);
        }
        // 7b. all -inf except one, so the maximum is finite but every other
        //     exponent is zero.
        {
            std::vector<float> x(N, -INFINITY);
            x[N / 2] = next_float(rng, -5.0f, -1.0f);
            push("syn:one-finite", x);
        }
        // 7c. every logit far below the maximum, so every exponent underflows
        //     even though no logit is infinite.
        {
            std::vector<float> x(N, -1.0e30f);
            push("syn:all-underflow", x);
        }
        // 7. many rounded-equal probabilities: logits that differ below the
        //    ulp of the probability
        {
            std::vector<float> x(N);
            for (int i = 0; i < N; ++i) {
                x[i] = next_float(rng, -25.0f, -18.0f);
            }
            const float top = -3.0f;
            for (int i = 0; i < 40; ++i) {
                // 2^-24 relative spacing around exp(top)
                x[i] = top + (float) (i % 3) * 1.1920929e-7f * 0.5f;
            }
            push("syn:prob-tie", x);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------
struct Stats {
    long long cases = 0;
    long long id_mismatch_cases = 0;
    long long tie_order_cases = 0;         // id order differs only inside an exact value class
    long long id_mismatch_other_cases = 0; // id order differs between different values
    long long weight_bit_diff_cases = 0;
    long long weight_value_diff_cases = 0;
    long long weight_nan_diff_cases = 0;
    long long prob_bit_diff_cases = 0;
    long long boundary_tie_cases = 0;      // 10th and 11th probability are bit equal
    long long golden_rows = 0;             // rows that carry the oracle's own recorded output
    long long golden_id_diff_cases = 0;    // candidate top-k != the oracle's recorded top-k
    long long golden_weight_diff_cases = 0;
    std::vector<int> last_ids;             // candidate result of the most recent case
    std::vector<float> last_weights;
    bool reported = false;
    bool ok = true;
    std::string first_name;
};

// True when the two selections are interchangeable: same experts (possibly in a
// different order) and the same weights, compared bit for bit. With a bit-equal
// probability row that is exactly the situation ggml's std::sort leaves open.
static bool same_selection_set(const std::vector<int> & a, const std::vector<int> & b) {
    if (a.size() != b.size()) return false;
    std::vector<int> sa = a, sb = b;
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    return sa == sb;
}

static bool same_weight_bits(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i]) != bits(b[i])) return false;
    }
    return true;
}

static bool compare_one(RefRunner & ref, const Case & c, Stats & st, bool verbose) {
    const int N = (int) c.logits.size();
    RefResult r = ref.run(c.logits.data(), c.w_scale);

    // The candidate's probs pointer is the scratch the selection consumes: the
    // device kernel passes its shared row buffer, so it comes back with -INFINITY
    // in the selected slots. Comparing the whole row means giving the candidate a
    // separate scratch and reading the probabilities out of the half of the call
    // that is deliberately the same in both implementations.
    std::vector<float> probs(N, 0.0f);
    std::vector<int> ids(10);
    std::vector<float> wts(10);
    q38_test_router_probs_row(c.logits.data(), N, 10, c.w_scale, probs.data(), ids.data(), wts.data());

    ++st.cases;
    bool ok = true;
    st.last_ids = ids;
    st.last_weights = wts;

    for (int i = 0; i < N; ++i) {
        if (bits(r.probs[i]) != bits(probs[i]) && !(is_nan_bits(bits(r.probs[i])) && is_nan_bits(bits(probs[i])))) {
            ++st.prob_bit_diff_cases;
            ok = false;
            break;
        }
    }
    bool id_differs = false;
    for (int j = 0; j < 10; ++j) {
        if (r.sel[j] != ids[j]) { id_differs = true; break; }
    }
    if (id_differs) {
        ++st.id_mismatch_cases;
        // An exact tie among probabilities leaves ggml's std::sort free to return
        // either order, and the set it returns can therefore differ by one or
        // more members. The exchange is observable only if it changes a weight,
        // so the case is accepted when the two selections are the same set of
        // experts - in which case the sorted weight vectors below must be bit
        // equal as well.
        // Sorting the bit patterns keeps the comparison exact and free of NaN
        // and signed-zero ordering questions.
        std::vector<uint32_t> ra, ca;
        for (float w : r.weights) ra.push_back(bits(w));
        for (float w : wts) ca.push_back(bits(w));
        std::sort(ra.begin(), ra.end());
        std::sort(ca.begin(), ca.end());
        if (ra == ca) {
            ++st.tie_order_cases;
        } else {
            ++st.id_mismatch_other_cases;
            ok = false;
        }
    }

    // Boundary tie: the last selected and the first rejected probability are bit
    // equal, so the top-k set itself is not unique.
    {
        float lowest_selected = r.probs[r.sel[0]];
        for (int j = 1; j < 10; ++j) lowest_selected = std::min(lowest_selected, r.probs[r.sel[j]]);
        float highest_rejected = 0.0f;
        for (int e = 0; e < N; ++e) {
            bool selected = false;
            for (int j = 0; j < 10; ++j) {
                if (r.sel[j] == e) { selected = true; break; }
            }
            if (!selected) highest_rejected = std::max(highest_rejected, r.probs[e]);
        }
        if (bits(lowest_selected) == bits(highest_rejected)) ++st.boundary_tie_cases;
    }

    for (int j = 0; j < 10; ++j) {
        const uint32_t a = bits(r.weights[j]), b = bits(wts[j]);
        if (a == b) continue;
        ++st.weight_bit_diff_cases;
        ok = false;
        if (is_nan_bits(a) && is_nan_bits(b)) ++st.weight_nan_diff_cases;
        else ++st.weight_value_diff_cases;
        break;
    }

    if (!ok && verbose && !st.reported) {
        st.reported = true;
        st.first_name = c.name;
        std::printf("  first mismatch: %s\n", c.name.c_str());
        std::printf("    ref ids    :");
        for (int j = 0; j < 10; ++j) std::printf(" %d", r.sel[j]);
        std::printf("\n    cand ids   :");
        for (int j = 0; j < 10; ++j) std::printf(" %d", ids[j]);
        std::printf("\n    ref weights:");
        for (int j = 0; j < 10; ++j) std::printf(" %08x", bits(r.weights[j]));
        std::printf("\n    cand weights:");
        for (int j = 0; j < 10; ++j) std::printf(" %08x", bits(wts[j]));
        std::printf("\n    id/data source: %s\n", c.synthetic ? "synthetic" : "oracle trace");

        // Print the first differing logit pair to characterise the case.
        for (int i = 0; i < N; ++i) {
            if (bits(r.probs[i]) != bits(probs[i])) {
                std::printf("    first prob diff at expert %d: logit %08x ref %08x cand %08x\n",
                            i, bits(c.logits[i]), bits(r.probs[i]), bits(probs[i]));
                break;
            }
        }
    }
    return ok;
}

static Stats run_set(RefRunner & ref, const std::vector<Case> & cases, bool check_golden) {
    Stats st;
    st.golden_rows = 0;
    st.golden_id_diff_cases = 0;
    st.golden_weight_diff_cases = 0;
    for (const Case & c : cases) {
        compare_one(ref, c, st, true);
        if (!check_golden) continue;
        ++st.golden_rows;
        if (c.ids != st.last_ids) {
            ++st.golden_id_diff_cases;
            st.ok = false;
        }
        for (size_t j = 0; j < c.weights.size(); ++j) {
            if (bits(c.weights[j]) != bits(st.last_weights[j])) {
                ++st.golden_weight_diff_cases;
                st.ok = false;
                break;
            }
        }
    }
    return st;
}

int main(int argc, char ** argv) {
    const char * corpus_path = argc > 1 ? argv[1] : "router_corpus.bin";
    int per_kind = argc > 2 ? std::atoi(argv[2]) : 200;

    RefRunner ref;
    ref.init(512, 10);

    Stats golden;
    Stats synth;

    std::printf("ggml reference: soft_max -> argsort_top_k -> get_rows -> sum_rows -> clamp -> div\n");
    std::printf("[build %s %s]\n", __DATE__, __TIME__);

    std::vector<Case> corpus;
    FILE * probe = std::fopen(corpus_path, "rb");
    if (probe) {
        std::fclose(probe);
        corpus = read_corpus(corpus_path);
        std::printf("golden corpus: %zu rows from %s\n", corpus.size(), corpus_path);
    } else {
        std::printf("golden corpus: %s not found (skipped)\n", corpus_path);
    }

    std::vector<Case> syn = make_synthetic(per_kind);

    if (!corpus.empty()) {
        golden = run_set(ref, corpus, true);
        std::printf("  golden rows %lld: vs oracle-recorded output: id-diff %lld, weight-diff %lld\n",
                    golden.golden_rows, golden.golden_id_diff_cases, golden.golden_weight_diff_cases);
        std::printf("  golden rows %lld: vs ggml replay: prob-bit-diff %lld, id-order-diff %lld (tie-only %lld), "
                    "weight-bit-diff %lld, boundary-ties %lld\n",
                    golden.cases, golden.prob_bit_diff_cases, golden.id_mismatch_cases, golden.tie_order_cases,
                    golden.weight_bit_diff_cases, golden.boundary_tie_cases);
    }

    synth = run_set(ref, syn, false);
    std::printf("  synthetic rows %lld: prob-bit-diff %lld, id-order-diff %lld (tie-only %lld, value %lld), "
                "weight-bit-diff %lld, boundary-ties %lld\n",
                synth.cases, synth.prob_bit_diff_cases, synth.id_mismatch_cases, synth.tie_order_cases,
                synth.id_mismatch_other_cases, synth.weight_bit_diff_cases, synth.boundary_tie_cases);

    const long long fails = golden.prob_bit_diff_cases + golden.weight_bit_diff_cases +
                            golden.id_mismatch_other_cases + golden.golden_id_diff_cases + golden.golden_weight_diff_cases +
                            synth.prob_bit_diff_cases + synth.weight_bit_diff_cases + synth.id_mismatch_other_cases;
    if (fails == 0) {
        std::printf("router host check: PASS (%lld golden + %lld synthetic rows, "
                    "0 probability / weight bit differences)\n", golden.cases, synth.cases);
        return 0;
    }
    std::printf("router host check: FAIL (%lld differences)\n", fails);
    return 1;
}
