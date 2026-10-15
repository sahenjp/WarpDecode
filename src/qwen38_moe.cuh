#pragma once

#include "expert_cache.cuh"
#include "dense_vector.cuh"
#include "gguf.hpp"
#include "lowbit.cuh"
// ggml-exact Q8_0 activation quantisation + integer-dot GEMV. qwen38_hc.cuh
// includes neither gdn/qsa/moe, so pulling it in here stays acyclic.
#include "qwen38_exact_math.cuh"
#include "qwen38_hc.cuh"
// ggml-ordered F32 arithmetic helpers (q38_mul_rn / q38_add_rn /
// q38_moe_branch_f32). Header-only, no project includes.
#include "qwen38_precision.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace wd {

// ggml-compatible activation quantisation for the MoE router and the shared
// expert. llama.cpp's CPU reference quantises the activation to Q8_0 (32-value
// blocks, integer dot, per-block scale) before every matmul against a
// quantised weight; the legacy F16-activation GEMV rounds the activation
// differently and drifts ~1% per projection. Both switches default to ON and
// only apply while the weight is Q8_0, so non-quantised routers keep the
// existing dense path. WARODECODE_ROUTER_Q8_0=0 / WARODECODE_SHEXP_Q8_0=0
// restore the legacy path.
inline bool q38_moe_env_flag(const char * name, bool default_on) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_on;
    return v[0] != '0';
}

inline bool q38_moe_router_f32_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_F32", true);
    return enabled;
}

inline bool q38_moe_router_q8_0_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_Q8_0", true);
    return enabled;
}

inline bool q38_moe_shexp_q8_0_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_Q8_0", true);
    return enabled;
}

// WARODECODE_SHEXP_GATE_F32 (default ON).
//
// blk.N.ffn_gate_inp_shexp.weight is F32 [n_embd] in the pinned GGUF, so
// llama.cpp builds the shared expert's sigmoid gate with
// build_lora_mm(ffn_gate_inp_shexp, cur) on F32 tensors: type_traits_cpu[F32]
// has vec_dot_type == F32 and llamafile_sgemm() declines the single decode
// token (sgemm.cpp:3819, "if (n < 2) return false"), so the gate is one
// ggml_vec_dot_f32 per token. upload_half_vector() rounded that weight to F16
// (~5e-4 systematic) before the sigmoid, which the expert activation
// quantisation then amplifies. With this switch the weight stays F32 and the
// dot runs in the oracle's AVX2+FMA visit order
// (q38_moe_gate_dot_f32_avx2_kernel). WARODECODE_SHEXP_GATE_F32=0 restores the
// legacy F16-weight gate, and a model whose tensor is not F32 falls back to it
// automatically.
inline bool q38_moe_shexp_gate_f32_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_GATE_F32", true);
    return enabled;
}

// Routed experts (512 experts, top-10) dominate the MoE work, so their
// activation also has to match ggml: llama.cpp's CPU reference quantises the
// gate/up input to the Q8_K vec_dot_type shared by IQ2_S and IQ3_S once and the
// SiLU output to the IQ4_NL vec_dot_type (Q8_0) for the down projection. Both
// projections stay in F32 and the quantised gate/up activation is shared by all
// ten experts of a layer. Defaults to ON and only applies to the observed expert
// tensor types, so anything else keeps the legacy F16-activation path.
inline bool q38_moe_expert_q8_act_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_EXPERT_Q8_ACT", true);
    return enabled;
}

// F32 MoE intermediates. ggml never rounds the FFN intermediates: the expert
// gate/up/down projections, the SiLU*up product, the per-expert weighted sum
// and the shared-expert addition all stay F32, and only the activation of a
// *quantised* weight is rounded (to Q8_0 / Q8_K) before the integer dot. The
// legacy MoE stored each expert output and the shared-expert output in __half,
// so every projection added one F16 rounding on top of the quantisation.
// With WARODECODE_MOE_F32_MID=1 (default) the routed experts accumulate their
// unrounded down projection into d_accum, the shared expert keeps its unrounded
// down projection in down_f32(), and q38_add_shared_f32_kernel() /
// q38_finish_moe_f32() build the MoE branch without an F16 round. The F16
// buffers and kernels stay in place for WARODECODE_MOE_F32_MID=0.
inline bool q38_moe_f32_mid_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_MOE_F32_MID", true);
    return enabled;
}

// WARODECODE_MOE_FUSED_GRID (default OFF).
//
// The routed experts of one layer run as one expert-batched grid instead of one
// launch per expert and projection (the per-expert loop issues seven launches
// per expert: gate GEMV, up GEMV, SiLU, Q8_0 activation quantisation, down GEMV,
// optional trace copy and the weighted accumulation). With this switch a layer
// costs five launches:
//
//   gate+up GEMV   grid.y = 2k (separate gate/up) or k (fused gate_up tensor)
//   SiLU(gate)*up  grid.y = k
//   Q8_0 quantise  grid.y = k
//   down GEMV      grid.y = k
//   weighted sum   ordered loop over the k experts, one launch
//
// Every kernel still decodes one row or one 32-value block with the same
// arithmetic the single-expert kernels use (the DP4A row routines are shared by
// construction, the SiLU/quantiser/accumulation bodies are literal copies of
// the originals), so the values are bit-identical - only the launch geometry and
// the launch count change. The weight pointers of the layer travel in one small
// pinned H2D copy (Q38MoeExpertPtrTable).
// Default ON since the transfer path stopped dominating: on A100 with the
// 20000 MiB cache and the pread+DMA staging this is +2.2% (7.624 -> 7.793 tok/s)
// at parity 64/64, and it interacts strongly with the staging (skipping the
// transfer entirely takes the same binary from 7.79 to 9.68 tok/s), so the
// launch-count reduction is now visible. Set WARODECODE_MOE_FUSED_GRID=0 to go
// back to the per-expert launches.
inline bool q38_moe_fused_grid_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_MOE_FUSED_GRID", true);
    return enabled;
}

// Task R: how many prefix groups one expert-batched layer pass runs in.
//
// The batched pass acquires all k expert slots and only then launches its five
// kernels, so every pread() and every DMA of the layer completes before the
// first expert GEMV starts: the transfer is fully exposed, and the consumer
// stream waits for the last miss of the layer (measured on the A100: 25 ms of
// the MoE region against 16 ms for the serial loop, which overlaps expert j's
// DMA with expert j-1's kernels).
//
// Splitting the pass into prefix groups [0,g1), [g1,g2), ... keeps the
// accumulation order the kernels already use (j ascending inside
// q38_moe_weighted_accum_experts_f32_kernel), so the result stays bit-identical
// to a single group: group g+1's acquire (pread + DMA) is issued after group
// g's kernels were launched and therefore overlaps them, while group g+1's
// kernels still wait for group g's, exactly like the serial loop.
//
// WARODECODE_MOE_BATCH_GROUPS=n (default 1 = the unsplit pass).
inline int q38_moe_batch_groups(int k) {
    static const int configured = [] {
        const char * v = std::getenv("WARODECODE_MOE_BATCH_GROUPS");
        if (v == nullptr || v[0] == '\0') return 1;
        const long n = std::strtol(v, nullptr, 10);
        return n > 1 ? static_cast<int>(std::min<long>(n, 8)) : 1;
    }();
    if (k <= 1 || configured <= 1) return 1;
    return configured > k ? k : configured;
}

// Experts in group g of a k-expert prefix split: every group gets the floor and
// the first k % groups get one extra, so the counts sum to k.
inline int q38_moe_group_count(int k, int groups, int g) {
    return k / groups + (g < k % groups ? 1 : 0);
}

// The router's F32-logits GEMV is the one projection of the MoE region that ggml
// computes with a plain F32 dot (vec_dot_type == F32 for a non-quantised
// ffn_gate_inp), so its accumulation order is fixed by ggml's AVX2 visit order
// and cannot be changed without changing which experts get selected.
// src/matrix.cuh::dense_f32_gemv_avx2_kernel reproduces that order with one
// thread per row, which for the router's [512, 2560] F32 matrix means 512
// threads in 2 blocks, each walking its own row with uncoalesced 4-byte loads
// (neighbouring lanes read 10 KB apart). Measured on the A100 it costs ~100+ us
// per layer, i.e. several ms per token, for 5.2 MB of weights.
//
// The twin below keeps the accumulation *bit* for bit: the AVX2 order has 32
// independent accumulators sum[j][l] over elements i + j*8 + l (i stepping by
// 32), so one warp owns a row and lane t owns the single accumulator j*8+l = t,
// running exactly the same FMA chain over the same elements in the same order.
// Only the *placement* changes, so the result is identical; the loads become
// coalesced because the lanes walk the row together, and there are 8x more
// blocks. q38_matrix_reduce_avx2() is reproduced with the shuffles that
// implement exactly its pairing, so the reduction is identical as well.
//
// WARODECODE_ROUTER_F32_WARP=0 goes back to the one-thread-per-row kernel.
inline bool q38_moe_router_f32_warp_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_F32_WARP", true);
    return enabled;
}

__global__ void q38_router_gemv_f32_warp_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ y,
        int rows,
        int cols) {
    const int lane = threadIdx.x & 31;
    const int row_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + row_in_block;
    if (row >= rows) return;
    const float * row_ptr = w + static_cast<size_t>(row) * cols;

    float s = 0.0f;
    const int np = cols & ~31;
    for (int i = 0; i < np; i += 32) {
        s = __fmaf_rn(row_ptr[i + lane], x[i + lane], s);
    }

    // q38_matrix_reduce_avx2(), which pairs
    //   sum[0][l] += sum[2][l]; sum[1][l] += sum[3][l];
    //   sum[0][l] += sum[1][l];
    //   t0[k] = sum[0][k] + sum[0][k+4];
    //   h0 = t0[0] + t0[1]; h1 = t0[2] + t0[3]; return h0 + h1;
    // Lane t holds sum[t >> 3][t & 7], so lane l needs sum[2][l] from lane l+16,
    // then sum[1][l] from lane l+8, and so on. Every shuffle is executed by all
    // 32 lanes (the warp never diverges before this point).
    float other = __shfl_down_sync(0xffffffffu, s, 16);
    if (lane < 16) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 8);
    if (lane < 8) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 4);
    if (lane < 4) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 1);
    if (lane == 0 || lane == 2) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 2);
    float result = s + other;

    if (lane == 0) {
        // ggml's leftover tail, the non-fused "sumf += x[i]*y[i]" chain. The
        // router is 2560 wide (a multiple of 32), so this never runs for the
        // model; it is kept for the same generality the reference kernel has.
        for (int c = np; c < cols; ++c) {
            result = __fadd_rn(result, __fmul_rn(row_ptr[c], x[c]));
        }
        y[row] = result;
    }
}

inline void launch_q38_router_gemv_f32(
        const GpuMatrix & w,
        const float * x,
        float * y,
        cudaStream_t stream,
        int threads = 256) {
    if (!matrix_has_f32_weights(w)) {
        throw std::runtime_error("router F32 warp GEMV requires an F32 dense matrix");
    }
    if (threads < 32 || threads % 32 != 0) {
        throw std::runtime_error("router F32 warp GEMV threads must be a multiple of 32");
    }
    const int warps = threads / 32;
    const int blocks = (w.rows + warps - 1) / warps;
    q38_router_gemv_f32_warp_kernel<<<blocks, threads, 0, stream>>>(
        w.dense_f32.data, x, y, w.rows, w.cols);
}

// ---------------------------------------------------------------------------
// Task S: MoE-region kernel-level attribution (WARODECODE_MOE_PROFILE).
//
// Task Q decomposed the MoE region by differencing whole runs and was left with
// "expert kernel execution, no I/O, five launches per layer" = 64.5 ms/token on
// the A100 at 20000 MiB. The Task G microbenchmark measures those same five
// launches at 179 us/layer = 8.6 ms/token on the L4 (tests/moe_fused_grid_bench.cu),
// so 64.5 ms cannot be the sum of the five kernels: the region either spends its
// time in the launches around them (router selection, its device-to-host sync,
// the shared expert, the F16 mirrors, the five-way finish) or in idle gaps
// between launches.
//
// This profiler brackets every launch of the region with its own event pair,
// exactly like the Task J HC profiler and the Task P attention profiler. Two
// numbers are reported per site and token:
//
//   dev_ms  - the GPU-timeline width of the bracket. A host stall inside the
//             bracket is included, because the GPU is idle while the host
//             prepares the next launch; that is the point.
//   host_ms - the host wall time of the same bracket. dev_ms - host_ms is the
//             time the GPU was doing work the host did not wait for, host_ms -
//             dev_ms is the time the host spent inside the bracket while the GPU
//             had nothing queued (the thing only removing host work can fix).
//
// The difference between a parent site and the sum of its children is the launch
// gap plus any uninstrumented launch.
//
// Enabled by WARODECODE_MOE_PROFILE=1. With the switch off the class is empty
// and every hook is one predictable branch, so the default decode path is
// unchanged bit for bit.
// ---------------------------------------------------------------------------
enum Q38MoeSite : int {
    kQ38MoeSiteStageTotal = 0,   // the whole kQwen38StageMoe bracket
    kQ38MoeSiteRoutedTotal,      // one layer's routed-expert runner
    kQ38MoeSiteRouter,           // router GEMV + top-k + D2H + stream sync
    kQ38MoeSiteRouterQuant,      //   the router activation Q8_0 quantisation
    kQ38MoeSiteRouterGemv,       //   the router logits GEMV
    kQ38MoeSiteRouterTopk,       //   the single-threaded softmax + top-k kernel
    kQ38MoeSiteRouterD2h,        //   the two device-to-host copies of ids/weights
    kQ38MoeSiteRouterSync,       //   the cudaStreamSynchronize that closes it
    kQ38MoeSiteAccumZero,        // cudaMemsetAsync of the routed accumulator
    kQ38MoeSiteRxQuant,          // Q8_K quantisation of the routed activation
    kQ38MoeSiteAcquire,          // the k cache acquires of one layer
    kQ38MoeSitePtrH2d,           // the expert pointer-table H2D copy
    kQ38MoeSiteBatchTotal,       // the five expert-batched launches together
    kQ38MoeSiteGateUpGemv,       // 1/5 gate+up GEMV (grid.y = k)
    kQ38MoeSiteSilu,             // 2/5 SiLU(gate)*up
    kQ38MoeSiteQ8Quant,          // 3/5 Q8_0 activation quantisation
    kQ38MoeSiteDownGemv,         // 4/5 down GEMV
    kQ38MoeSiteAccum,            // 5/5 weighted accumulation
    kQ38MoeSiteCapture,          // optional selected-expert trace copy
    kQ38MoeSiteLoopTotal,        // the per-expert loop (WARODECODE_MOE_FUSED_GRID=0)
    kQ38MoeSiteFinish,           // q38_finish_moe()
    kQ38MoeSiteShared,           // the shared expert
    kQ38MoeSiteSharedGate,       //   its sigmoid gate dot product
    kQ38MoeSiteMirror,           // the F16 mirrors + add_shared + finish_moe_f32
    kQ38MoeSiteCount
};

inline const char * q38_moe_site_name(int site) {
    switch (site) {
        case kQ38MoeSiteStageTotal: return "moe.stage_total";
        case kQ38MoeSiteRoutedTotal: return "moe.routed_total";
        case kQ38MoeSiteRouter: return "moe.router_select";
        case kQ38MoeSiteRouterQuant: return "moe.router.act_quant";
        case kQ38MoeSiteRouterGemv: return "moe.router.gemv";
        case kQ38MoeSiteRouterTopk: return "moe.router.topk";
        case kQ38MoeSiteRouterD2h: return "moe.router.d2h";
        case kQ38MoeSiteRouterSync: return "moe.router.sync";
        case kQ38MoeSiteAccumZero: return "moe.accum_zero";
        case kQ38MoeSiteRxQuant: return "moe.rx_quant";
        case kQ38MoeSiteAcquire: return "moe.acquire";
        case kQ38MoeSitePtrH2d: return "moe.ptr_h2d";
        case kQ38MoeSiteBatchTotal: return "moe.batch_total";
        case kQ38MoeSiteGateUpGemv: return "moe.batch.gate_up";
        case kQ38MoeSiteSilu: return "moe.batch.silu";
        case kQ38MoeSiteQ8Quant: return "moe.batch.q8_quant";
        case kQ38MoeSiteDownGemv: return "moe.batch.down";
        case kQ38MoeSiteAccum: return "moe.batch.accum";
        case kQ38MoeSiteCapture: return "moe.batch.capture";
        case kQ38MoeSiteLoopTotal: return "moe.loop_total";
        case kQ38MoeSiteFinish: return "moe.finish";
        case kQ38MoeSiteShared: return "moe.shared";
        case kQ38MoeSiteSharedGate: return "moe.shared.gate";
        case kQ38MoeSiteMirror: return "moe.mirror";
        default: return "unknown";
    }
}

struct Q38MoeSiteStat {
    unsigned long long calls = 0;
    double total_ms = 0.0;
    double total_host_ms = 0.0;
    unsigned long long overflow = 0;
    unsigned long long elapsed_fail = 0;
    unsigned long long recorded = 0;
};

class Q38MoeKernelProfiler {
public:
    // One token has 48 layers and every instrumented site fires at most once per
    // layer (the per-expert loop path is bracketed as a whole), so 128 slots per
    // site cover a token with room for a prefix-split pass. A site that runs out
    // reports it in the "dropped" column instead of silently losing samples.
    static constexpr int kCap = 128;

    static Q38MoeKernelProfiler & instance() {
        static Q38MoeKernelProfiler profiler;
        return profiler;
    }

    bool enabled() const { return enabled_; }

    void mark_begin(int site, cudaStream_t stream) {
        if (!enabled_) return;
        if (site < 0 || site >= kQ38MoeSiteCount) return;
        // A nested bracket stays one interval: only the outermost pair records.
        if (depth_[site]++ != 0) return;
        host_begin_[site] = std::chrono::steady_clock::now();
        if (trace_enabled_ && trace_lines_ < 120) {
            ++trace_lines_;
            std::fprintf(stderr, "[moe-profile-trace] begin %s depth=%d slot=%d\n",
                         q38_moe_site_name(site), depth_[site] - 1, written_[site]);
        }
        const int slot = written_[site];
        if (slot >= kCap) {
            ++overflow_[site];
            return;
        }
        if (cudaEventRecord(start_[index(site, slot)], stream) != cudaSuccess) {
            ++overflow_[site];
            return;
        }
        pending_start_[site] = start_[index(site, slot)];
    }

    void mark_end(int site, cudaStream_t stream) {
        if (!enabled_) return;
        if (site < 0 || site >= kQ38MoeSiteCount) return;
        if (depth_[site] <= 0) return;
        if (--depth_[site] != 0) return;
        const auto stop_host = std::chrono::steady_clock::now();
        host_ms_[site] += std::chrono::duration<double, std::milli>(
                              stop_host - host_begin_[site]).count();
        const int slot = written_[site];
        if (slot >= kCap || pending_start_[site] == nullptr) {
            ++overflow_[site];
            pending_start_[site] = nullptr;
            return;
        }
        if (cudaEventRecord(stop_[index(site, slot)], stream) != cudaSuccess) {
            ++overflow_[site];
            pending_start_[site] = nullptr;
            return;
        }
        start_[index(site, slot)] = pending_start_[site];
        pending_start_[site] = nullptr;
        ++written_[site];
    }

    // Called once per decode token, after the stream synchronise: every recorded
    // event is complete there, so cudaEventElapsedTime() cannot return
    // cudaErrorNotReady.
    void flush_token() {
        if (!enabled_) return;
        for (int s = 0; s < kQ38MoeSiteCount; ++s) {
            Q38MoeSiteStat & stat = stats_[s];
            const unsigned long long pending = written_[s];
            for (int i = 0; i < written_[s]; ++i) {
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, start_[index(s, i)], stop_[index(s, i)]) == cudaSuccess) {
                    stat.total_ms += static_cast<double>(ms);
                    ++stat.calls;
                } else {
                    ++stat.elapsed_fail;
                }
            }
            stat.recorded += pending;
            stat.total_host_ms += host_ms_[s];
            stat.overflow += overflow_[s];
            host_ms_[s] = 0.0;
            written_[s] = 0;
            depth_[s] = 0;
            pending_start_[s] = nullptr;
        }
        ++tokens_;
    }

    void print_summary() const {
        if (!enabled_ || tokens_ == 0) return;
        const double div = static_cast<double>(tokens_);
        std::fprintf(stderr, "[moe-profile] tokens=%llu unit=ms_per_token\n",
                     static_cast<unsigned long long>(tokens_));
        for (int s = 0; s < kQ38MoeSiteCount; ++s) {
            const Q38MoeSiteStat & stat = stats_[s];
            // Nothing at all happened for this site: skip it. Everything else is
            // printed, including the failure modes, so "the hook never ran",
            // "the bracket never closed" and "the events were not readable" stay
            // distinguishable in the output.
            if (stat.recorded == 0 && stat.overflow == 0 && stat.elapsed_fail == 0 &&
                stat.total_host_ms == 0.0) {
                continue;
            }
            std::fprintf(stderr,
                         "[moe-profile] %-20s dev=%9.4f host=%9.4f gap=%9.4f calls/token=%.2f"
                         " recorded=%llu dropped=%llu evfail=%llu\n",
                         q38_moe_site_name(s), stat.total_ms / div, stat.total_host_ms / div,
                         (stat.total_host_ms - stat.total_ms) / div,
                         static_cast<double>(stat.calls) / div,
                         static_cast<unsigned long long>(stat.recorded),
                         static_cast<unsigned long long>(stat.overflow),
                         static_cast<unsigned long long>(stat.elapsed_fail));
        }
    }

private:
    Q38MoeKernelProfiler() {
        const char * v = std::getenv("WARODECODE_MOE_PROFILE");
        enabled_ = v != nullptr && v[0] != '\0' && v[0] != '0';
        const char * t = std::getenv("WARODECODE_MOE_PROFILE_TRACE");
        trace_enabled_ = t != nullptr && t[0] != '\0' && t[0] != '0';
        if (!enabled_) return;
        for (size_t i = 0; i < start_.size(); ++i) {
            if (cudaEventCreate(&start_[i]) != cudaSuccess ||
                cudaEventCreate(&stop_[i]) != cudaSuccess) {
                enabled_ = false;
                return;
            }
        }
    }

    ~Q38MoeKernelProfiler() { print_summary(); }

    Q38MoeKernelProfiler(const Q38MoeKernelProfiler &) = delete;
    Q38MoeKernelProfiler & operator=(const Q38MoeKernelProfiler &) = delete;

    static size_t index(int site, int slot) {
        return static_cast<size_t>(site) * static_cast<size_t>(kCap) + static_cast<size_t>(slot);
    }

    bool enabled_ = false;
    bool trace_enabled_ = false;
    int trace_lines_ = 0;
    unsigned long long tokens_ = 0;
    std::array<cudaEvent_t, static_cast<size_t>(kQ38MoeSiteCount) * kCap> start_{};
    std::array<cudaEvent_t, static_cast<size_t>(kQ38MoeSiteCount) * kCap> stop_{};
    int written_[kQ38MoeSiteCount] = {};
    int overflow_[kQ38MoeSiteCount] = {};
    int depth_[kQ38MoeSiteCount] = {};
    cudaEvent_t pending_start_[kQ38MoeSiteCount] = {};
    double host_ms_[kQ38MoeSiteCount] = {};
    std::chrono::steady_clock::time_point host_begin_[kQ38MoeSiteCount] = {};
    Q38MoeSiteStat stats_[kQ38MoeSiteCount] = {};
};

// RAII bracket for one region of a MoE layer.
class Q38MoeKernelScope {
public:
    Q38MoeKernelScope(int site, cudaStream_t stream) : site_(site), stream_(stream) {
        Q38MoeKernelProfiler::instance().mark_begin(site_, stream_);
    }
    ~Q38MoeKernelScope() {
        Q38MoeKernelProfiler::instance().mark_end(site_, stream_);
    }
    Q38MoeKernelScope(const Q38MoeKernelScope &) = delete;
    Q38MoeKernelScope & operator=(const Q38MoeKernelScope &) = delete;

private:
    int site_;
    cudaStream_t stream_;
};

struct Qwen38MoeLayerTensors {
    const TensorInfo * router = nullptr;
    const TensorInfo * gate_up = nullptr;
    const TensorInfo * gate = nullptr;
    const TensorInfo * up = nullptr;
    const TensorInfo * down = nullptr;

    bool fused_gate_up() const { return gate_up != nullptr; }
};

struct Qwen38SharedExpertWeights {
    // Legacy F16 mirror of the shared-expert gate weight
    // (blk.N.ffn_gate_inp_shexp.weight). Only the non-F32 models and the
    // WARODECODE_SHEXP_GATE_F32=0 A/B path consume it.
    DeviceHalfVector gate_input;
    // Unrounded F32 gate weight. The pinned GGUF stores the tensor as F32
    // [n_embd], and ggml's F32 trait has vec_dot_type == F32, so the CPU builds
    // the shared gate from an F32 weight and an F32 activation. Keeping only the
    // half mirror rounds the weight by ~5e-4 before the sigmoid, which the
    // downstream activation quantisation amplifies. Empty for a quantised
    // tensor, in which case the legacy F16 path stays in charge.
    DeviceFloatArray gate_input_f32;
    LowBitMatrix gate;
    LowBitMatrix up;
    LowBitMatrix down;
};

inline Qwen38SharedExpertWeights load_qwen38_shared_expert(
        const GgufFile & g,
        int il,
        int hidden,
        int shared_ffn) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const TensorInfo & gate_input = g.tensor(p + "ffn_gate_inp_shexp.weight");
    const TensorInfo & gate = g.tensor(p + "ffn_gate_shexp.weight");
    const TensorInfo & up = g.tensor(p + "ffn_up_shexp.weight");
    const TensorInfo & down = g.tensor(p + "ffn_down_shexp.weight");

    if (gate_input.dims.size() != 1 || static_cast<int>(gate_input.dims[0]) != hidden) {
        throw std::runtime_error("shared expert gate-input shape mismatch");
    }
    auto check_matrix = [&](const TensorInfo & t, int rows, int cols, const char * what) {
        if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != cols || static_cast<int>(t.dims[1]) != rows) {
            throw std::runtime_error(std::string("shared expert ") + what + " shape mismatch: " + t.name);
        }
        if (!lowbit_gemv_supported(t.type)) {
            throw std::runtime_error(std::string("shared expert ") + what + " low-bit type unsupported: " + t.name);
        }
    };
    check_matrix(gate, shared_ffn, hidden, "gate");
    check_matrix(up, shared_ffn, hidden, "up");
    check_matrix(down, hidden, shared_ffn, "down");

    Qwen38SharedExpertWeights out;
    out.gate_input = upload_half_vector(gate_input);
    if (gate_input.type == GgmlType::F32) {
        out.gate_input_f32 = upload_float_array(gate_input);
    }
    out.gate = upload_lowbit_matrix(gate);
    out.up = upload_lowbit_matrix(up);
    out.down = upload_lowbit_matrix(down);
    return out;
}

inline Qwen38MoeLayerTensors qwen38_moe_tensors(const GgufFile & g, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    Qwen38MoeLayerTensors out;
    out.router = &g.tensor(p + "ffn_gate_inp.weight");
    out.down = &g.tensor(p + "ffn_down_exps.weight");
    out.gate_up = g.optional_tensor(p + "ffn_gate_up_exps.weight");
    if (!out.gate_up) {
        out.gate = &g.tensor(p + "ffn_gate_exps.weight");
        out.up = &g.tensor(p + "ffn_up_exps.weight");
    }
    return out;
}

// ---------------------------------------------------------------------------
// MoE router weight chain, matching build_moe_ffn (llama-graph.cpp at the
// pinned oracle commit 6fcaa16f4b360649933a54d1f91ad40ed35c0e11).
//
// The oracle path for this model - gating_op = SOFTMAX, norm_w = true,
// w_scale = qwen4exp.expert_weights_scale, exp_probs_b = null - is:
//
//   logits  = build_lora_mm(ffn_gate_inp, cur)                 [512]
//   probs   = ggml_soft_max(logits)                            [512]  softmax over ALL experts
//   sel     = ggml_argsort_top_k(probs, n_expert_used)         [10]   select by PROB, descending
//   weights = ggml_get_rows(probs, sel)                        [10]
//   wsum    = ggml_sum_rows(weights)                           [ 1]
//   wsum    = ggml_clamp(wsum, 6.103515625e-5, +inf)           [ 1]
//   weights = ggml_div(weights, wsum)                          [10]
//   weights = ggml_scale(weights, w_scale)     skipped when w_scale == 1.0f
//
// Three properties of that chain are load-bearing and easy to get wrong:
//
//   1. softmax runs over all n_expert cells, so every probability carries the
//      full 512-term denominator and the selected cells keep it. The legacy
//      warp kernel instead exponentiated only the top-k logits against their own
//      maximum and folded the expert scale into one reciprocal multiply - a
//      different sequence of roundings with a different number of divisions.
//   2. selection is by probability, not by logit. Softmax is order preserving on
//      the logits but not injective after rounding, so a near-tie can round to
//      two equal probabilities and then be ordered by ggml's std::sort instead
//      of by the raw logit. The selection below therefore compares the
//      probabilities, and where two of them are bit equal it takes the lower
//      expert index: ggml's descending comparator is a strict >, so std::sort
//      leaves equal elements unordered and its choice there is not reproducible
//      at all. That is harmless because equal probabilities carry equal weights,
//      so only the identity of the expert can differ, never the k values fed to
//      the experts.
//   3. ggml normalises inside the softmax op with a reciprocal multiply
//      (dp[i] * (1/sum), ggml-cpu/ops.cpp:5682) and renormalises the top-k with
//      a true division (ggml-cpu/binary-ops.cpp: op_div). The warp kernel must
//      not fuse or reorder those two steps.
//
// ggml_vec_soft_max_f32 (ggml-cpu/vec.cpp:531) never calls libm: its AVX2+FMA
// branch - the one the oracle dispatch picks for this model - evaluates
// dp[i] = ggml_v_expf(x[i] - max) with the polynomial at ggml-cpu/vec.h:1215 and
// accumulates the row sum with the 8-lane horizontal adds
// (_mm256_extractf128_ps / _mm_movehl_ps / _mm_movehdup_ps) into a double. The
// kernel below reproduces that per-8-element tree. q38_exact_avx2_exp_f32
// (qwen38_exact_math.cuh) is the lane-wise copy of the polynomial; n_expert =
// 512 is a multiple of 8, so the scalar tail of ggml_vec_soft_max_f32 - the only
// part the copy does not model - never runs for this model. A row length that is
// not a multiple of 8 falls back to the legacy router below instead, because
// there the oracle's tail uses libm expf, which has no device counterpart.
//
// WARODECODE_EXACT_ROUTER (default ON) selects the chain above. Setting it to 0
// restores the legacy two-rounding warp formula for A/B runs.
// ---------------------------------------------------------------------------
// The kernel below reads the switch through a device symbol (the same pattern
// qwen38_gdn.cuh uses) because q38_topk_softmax_kernel keeps its signature.
__device__ int q38_moe_exact_router_flag = 1;

inline bool q38_moe_exact_router_enabled() {
    static const bool enabled = [] {
        const int value = q38_moe_env_flag("WARODECODE_EXACT_ROUTER", true) ? 1 : 0;
        const cudaError_t e = cudaMemcpyToSymbol(q38_moe_exact_router_flag, &value, sizeof(value));
        return e == cudaSuccess && value != 0;
    }();
    return enabled;
}

__device__ __forceinline__ bool q38_moe_exact_router_on() {
    return q38_moe_exact_router_flag != 0;
}

inline const bool q38_moe_exact_router_published = q38_moe_exact_router_enabled();

constexpr int Q38_MOE_ROUTER_KMAX = 16;
// Scratch for one full softmax row. n_expert = 512 for this model.
constexpr int Q38_MOE_ROUTER_MAX_EXPERTS = 1024;

// One step of ggml_vec_soft_max_f32's AVX2 row sum. For the 8 lanes v[0..7] the
// _mm_movehl_ps / _mm_movehdup_ps sequence computes
// ((v0+v4) + (v2+v6)) + ((v1+v5) + (v3+v7)).
__device__ __forceinline__ float q38_moe_softmax_lane_sum(const float * v) {
    const float lo = __fadd_rn(v[0], v[4]);
    const float lh = __fadd_rn(v[2], v[6]);
    const float hi = __fadd_rn(v[1], v[5]);
    const float hh = __fadd_rn(v[3], v[7]);
    return __fadd_rn(__fadd_rn(lo, lh), __fadd_rn(hi, hh));
}

// ggml_soft_max over the whole row (ggml_compute_forward_soft_max_f32,
// ggml-cpu/ops.cpp:5584). probs receives n_experts probabilities.
//
//   max:  ggml_vec_max_f32 (ggml-cpu/vec.h:1541) is an order-independent scan,
//         so the plain ascending loop reproduces it exactly.
//   exp:  dp[i] = ggml_v_expf(x[i] - max), the subtraction its own IEEE step.
//   sum:  accumulated per 8 lanes into a double (ggml_float), exactly like the
//         AVX2 branch of ggml_vec_soft_max_f32.
//   norm: sum = 1.0/sum, then one multiply per element (ggml_vec_scale_f32). That
//         vector scale starts from -INFINITY so a zero row stays zero.
__device__ __forceinline__ void q38_moe_softmax_row(
        const float * __restrict__ logits,
        int n_experts,
        float * __restrict__ probs) {
    float max_v = -INFINITY;
    for (int e = 0; e < n_experts; ++e) {
        max_v = fmaxf(max_v, logits[e]);
    }

    double sum = 0.0;
    int i = 0;
    for (; i + 8 <= n_experts; i += 8) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = q38_exact_avx2_exp_f32(__fsub_rn(logits[i + j], max_v));
            probs[i + j] = v[j];
        }
        sum += static_cast<double>(q38_moe_softmax_lane_sum(v));
    }
    for (; i < n_experts; ++i) {
        const float val = q38_exact_avx2_exp_f32(__fsub_rn(logits[i], max_v));
        probs[i] = val;
        sum += static_cast<double>(val);
    }

    // ggml_vec_scale_f32 starts from -INFINITY, so a row whose exponents all
    // underflowed stays -INFINITY rather than turning into NaN.
    if (sum == 0.0) {
        for (int e = 0; e < n_experts; ++e) {
            probs[e] = -INFINITY;
        }
        return;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int e = 0; e < n_experts; ++e) {
        probs[e] = __fmul_rn(probs[e], inv);
    }
}

// ggml_argsort_top_k(probs, k) + norm_w + w_scale. probs is read only; the
// candidate ids and the normalised weights are the outputs.
//
//   selection: a descending argsort, i.e. the k largest probabilities with a tie
//              falling to the lower expert index. Taking the rungs in order
//              reproduces that order. The kernel keeps the ids and the weights in
//              ascending expert order for equal probabilities; the oracle's
//              std::sort is free to return any order there (see the block comment
//              above), and the weights are equal either way.
//   norm_w:    ggml_sum_rows, whose ggml_vec_sum_f32 accumulates into ggml_float
//              (double) and rounds once at the end (ggml-cpu/vec.h:1495), then
//              the ggml_clamp against the F16 minimum (a no-op for any row with a
//              positive sum) and the ggml_div renormalisation.
//   w_scale:   skipped by ggml when it is exactly 1.0f.
__device__ __forceinline__ void q38_moe_router_select(
        const float * __restrict__ probs,
        int n_experts,
        int k,
        float w_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    float top[Q38_MOE_ROUTER_KMAX];
    // The k rungs of the descending argsort. Each rung scans the whole row and
    // skips the experts already taken, which is the same sequence ggml's argsort
    // produces: the rungs are non-increasing, so every taken expert is >= the
    // next rung and skipping it cannot change the choice.
    for (int j = 0; j < k; ++j) {
        // Start below every probability so a row that is all -INFINITY (every
        // exponent underflowed) still resolves, and read the value back through
        // the chosen id: ggml_get_rows selects the cell, it does not invent one,
        // so that row yields -INFINITY weights, not zero.
        float best = -INFINITY;
        int best_id = 0;
        for (int e = 0; e < n_experts; ++e) {
            const float p = probs[e];
            if (p > best) {
                bool taken = false;
                for (int m = 0; m < j; ++m) {
                    if (ids[m] == e) {
                        taken = true;
                        break;
                    }
                }
                if (!taken) {
                    best = p;
                    best_id = e;
                }
            }
        }
        ids[j] = best_id;
        top[j] = probs[best_id];
    }

    double wsum = 0.0;
    for (int j = 0; j < k; ++j) {
        wsum += static_cast<double>(top[j]);
    }
    const float wsum_f32 = fmaxf(static_cast<float>(wsum), 6.103515625e-5f);
    for (int j = 0; j < k; ++j) {
        weights[j] = __fdiv_rn(top[j], wsum_f32);
    }

    if (w_scale != 0.0f && w_scale != 1.0f) {
        for (int j = 0; j < k; ++j) {
            weights[j] = __fmul_rn(weights[j], w_scale);
        }
    }
}

// The two halves together, the shape the kernel calls.
__device__ __forceinline__ void q38_moe_router_exact(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float w_scale,
        float * __restrict__ probs,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    q38_moe_softmax_row(logits, n_experts, probs);
    q38_moe_router_select(probs, n_experts, k, w_scale, ids, weights);
}

#if defined(__CUDACC__)
// Legacy two-rounding router, kept for WARODECODE_EXACT_ROUTER=0.
__device__ __forceinline__ void q38_moe_router_legacy(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    float best[Q38_MOE_ROUTER_KMAX];
    int best_id[Q38_MOE_ROUTER_KMAX];
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        best[j] = -3.402823466e+38F;
        best_id[j] = -1;
    }

    for (int e = 0; e < n_experts; ++e) {
        const float v = logits[e];
        if (v <= best[k - 1]) continue;

        int pos = k - 1;
        while (pos > 0 && v > best[pos - 1]) {
            best[pos] = best[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best[pos] = v;
        best_id[pos] = e;
    }

    const float max_v = best[0];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        if (j >= k) break;
        const float w = q38_exact_softmax_exp_f32(best[j], max_v, q38_exact_exp_on());
        weights[j] = w;
        ids[j] = best_id[j];
        sum += w;
    }

    const float inv = sum > 0.0f ? expert_scale / sum : 0.0f;
#pragma unroll
    for (int j = 0; j < Q38_MOE_ROUTER_KMAX; ++j) {
        if (j >= k) break;
        weights[j] *= inv;
    }
}

__global__ void q38_topk_softmax_kernel(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    __shared__ float probs[Q38_MOE_ROUTER_MAX_EXPERTS];
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (k <= 0 || k > Q38_MOE_ROUTER_KMAX) return;

    if (q38_moe_exact_router_on() &&
        n_experts > 0 && n_experts <= Q38_MOE_ROUTER_MAX_EXPERTS && n_experts % 8 == 0) {
        q38_moe_router_exact(logits, n_experts, k, expert_scale, probs, ids, weights);
        return;
    }
    q38_moe_router_legacy(logits, n_experts, k, expert_scale, ids, weights);
}

// ---------------------------------------------------------------------------
// Task S: block-parallel twin of q38_topk_softmax_kernel.
//
// The reference runs the whole router on one thread: 512 logits scanned for the
// max, 512 AVX2-order exponentials, 64 eight-lane sums accumulated into a double
// in order, then k rungs of a 512-wide descending argsort. That one thread is
// the entire GPU for the launch, which is why the router path costs more than
// the five expert-batched launches of the layer together.
//
// Every step is reproduced with the same arithmetic:
//   max    - fmaxf is associative and commutative, so a reduction tree returns
//            the same value the ascending scan does;
//   exp    - elementwise, neither order nor associativity involved;
//   sum    - q38_moe_softmax_lane_sum() over each group of eight is independent,
//            so only the ordered double accumulation over the group sums has to
//            stay on one thread (64 adds, the same ones the reference makes);
//   select - the ascending scan with a strict ">" keeps the lowest index of the
//            maximum among the not-yet-taken experts. A combine rule of "higher
//            value wins, equal value falls to the lower index" is the same total
//            order over (value, index) and never promotes a NaN, so the
//            reduction tree selects the same pair the scan does;
//   norm   - the same double sum, the same fmaxf clamp against the F16 minimum,
//            the same __fdiv_rn per element and the same __fmul_rn w_scale.
// WARODECODE_ROUTER_TOPK_PARALLEL=0 restores the one-thread kernel.
// ---------------------------------------------------------------------------
constexpr int kQ38RouterTopkThreads = 512;

__device__ __forceinline__ void q38_router_keep_better(
        float & value,
        int & index,
        float other_value,
        int other_index) {
    if (other_value > value || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
    }
}

__global__ void q38_topk_softmax_parallel_kernel(
        const float * __restrict__ logits,
        int n_experts,
        int k,
        float expert_scale,
        int * __restrict__ ids,
        float * __restrict__ weights) {
    __shared__ float probs[Q38_MOE_ROUTER_MAX_EXPERTS];
    __shared__ double group_sums[Q38_MOE_ROUTER_MAX_EXPERTS / 8];
    __shared__ float red_value[kQ38RouterTopkThreads];
    __shared__ int red_index[kQ38RouterTopkThreads];
    __shared__ int taken_ids[Q38_MOE_ROUTER_KMAX];
    __shared__ float top_values[Q38_MOE_ROUTER_KMAX];

    const int tid = threadIdx.x;
    if (blockIdx.x != 0) return;
    if (k <= 0 || k > Q38_MOE_ROUTER_KMAX) return;
    if (n_experts <= 0 || n_experts > Q38_MOE_ROUTER_MAX_EXPERTS) return;
    // n_experts is a multiple of 8 here (the launcher only takes this path under
    // the same condition the reference's exact branch uses), so the reference's
    // scalar tail over n_experts & 7 can never run.
    if (n_experts % 8 != 0) return;

    float local_max = -INFINITY;
    for (int e = tid; e < n_experts; e += blockDim.x) {
        local_max = fmaxf(local_max, logits[e]);
    }
    red_value[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red_value[tid] = fmaxf(red_value[tid], red_value[tid + stride]);
        __syncthreads();
    }
    const float max_v = red_value[0];
    __syncthreads();

    // dp[i] = ggml_v_expf(x[i] - max), the subtraction its own IEEE step.
    for (int e = tid; e < n_experts; e += blockDim.x) {
        probs[e] = q38_exact_avx2_exp_f32(__fsub_rn(logits[e], max_v));
    }
    __syncthreads();

    const int groups = n_experts / 8;
    for (int g = tid; g < groups; g += blockDim.x) {
        group_sums[g] = static_cast<double>(q38_moe_softmax_lane_sum(probs + g * 8));
    }
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int g = 0; g < groups; ++g) sum += group_sums[g];
        group_sums[0] = sum;
    }
    __syncthreads();
    const double sum_total = group_sums[0];

    // ggml_vec_scale_f32 starts from -INFINITY, so a row whose exponents all
    // underflowed stays -INFINITY rather than turning into NaN.
    if (sum_total == 0.0) {
        for (int e = tid; e < n_experts; e += blockDim.x) probs[e] = -INFINITY;
    } else {
        const float inv = static_cast<float>(1.0 / sum_total);
        for (int e = tid; e < n_experts; e += blockDim.x) probs[e] = __fmul_rn(probs[e], inv);
    }
    __syncthreads();

    for (int j = 0; j < k; ++j) {
        // Start below every probability so a row that is all -INFINITY still
        // resolves, and read the value back through the chosen id exactly like
        // the reference: best_id stays 0 when nothing beats -INFINITY.
        float best = -INFINITY;
        int best_id = 0;
        for (int e = tid; e < n_experts; e += blockDim.x) {
            bool taken = false;
            for (int m = 0; m < j; ++m) {
                if (taken_ids[m] == e) {
                    taken = true;
                    break;
                }
            }
            if (taken) continue;
            if (probs[e] > best) {
                best = probs[e];
                best_id = e;
            }
        }
        red_value[tid] = best;
        red_index[tid] = best_id;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                q38_router_keep_better(red_value[tid], red_index[tid],
                                       red_value[tid + stride], red_index[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            ids[j] = red_index[0];
            top_values[j] = probs[red_index[0]];
            taken_ids[j] = red_index[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double wsum = 0.0;
        for (int j = 0; j < k; ++j) wsum += static_cast<double>(top_values[j]);
        const float wsum_f32 = fmaxf(static_cast<float>(wsum), 6.103515625e-5f);
        for (int j = 0; j < k; ++j) {
            weights[j] = __fdiv_rn(top_values[j], wsum_f32);
        }
        if (expert_scale != 0.0f && expert_scale != 1.0f) {
            for (int j = 0; j < k; ++j) weights[j] = __fmul_rn(weights[j], expert_scale);
        }
    }
}
#endif  // __CUDACC__

// Host-side pick between the one-thread router and its block-parallel twin. The
// parallel kernel reimplements the exact branch only, so anything the reference
// would send to q38_moe_router_legacy() keeps the one-thread kernel.
inline bool q38_moe_router_topk_parallel_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_ROUTER_TOPK_PARALLEL", true);
    return enabled;
}

inline void q38_launch_topk_softmax(
        const float * logits,
        int n_experts,
        int k,
        float expert_scale,
        int * ids,
        float * weights,
        cudaStream_t stream) {
    if (q38_moe_router_topk_parallel_enabled() &&
        q38_moe_exact_router_enabled() &&
        n_experts > 0 && n_experts <= Q38_MOE_ROUTER_MAX_EXPERTS && n_experts % 8 == 0 &&
        k > 0 && k <= Q38_MOE_ROUTER_KMAX) {
        q38_topk_softmax_parallel_kernel<<<1, kQ38RouterTopkThreads, 0, stream>>>(
            logits, n_experts, k, expert_scale, ids, weights);
        return;
    }
    q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        logits, n_experts, k, expert_scale, ids, weights);
}
__global__ void q38_silu_mul_kernel(
        const __half * __restrict__ gate,
        const __half * __restrict__ up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    // ggml_silu -> ggml_vec_silu_f32 (ggml-cpu/vec.cpp:380) -> ggml_v_silu.
    const float s = q38_exact_silu_f32(g, q38_exact_exp_on());
    out[i] = __float2half(s * u);
}

__global__ void q38_silu_mul_fused_kernel(
        const __half * __restrict__ gate_up,
        __half * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate_up[i]);
    const float u = __half2float(gate_up[n + i]);
    const float s = q38_exact_silu_f32(g, q38_exact_exp_on());
    out[i] = __float2half(s * u);
}

__global__ void q38_weighted_accum_kernel(
        const __half * __restrict__ x,
        float weight,
        float * __restrict__ accum,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = fmaf(weight, __half2float(x[i]), accum[i]);
}

// F32 twin of q38_weighted_accum_kernel for the WARODECODE_MOE_F32_MID path.
// The expert output is already F32, so the accumulation is exactly the two
// rounding steps ggml performs per selected expert (ggml_mul by the expert
// weight, then ggml_add into the running sum) instead of a fused multiply-add
// on an F16-rounded value.
__global__ void q38_weighted_accum_f32_kernel(
        const float * __restrict__ x,
        float weight,
        float * __restrict__ accum,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = q38_add_rn(accum[i], q38_mul_rn(x[i], weight));
}

__global__ void q38_float_to_half_kernel(
        const float * __restrict__ x,
        __half * __restrict__ y,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// ggml SiLU(gate) * up product in F32. The shared expert keeps the gated
// activation unrounded so the following Q8_0 quantisation sees the same F32
// values the CPU reference sees.
__global__ void q38_silu_mul_f32_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ out,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __fmul_rn(gate[i], up[i]);
}

// ---------------------------------------------------------------------------
// Task G: expert-batched MoE stages used by WARODECODE_MOE_FUSED_GRID.
//
// Every kernel below runs the same per-item arithmetic as the single-expert
// kernel it twins; grid.y only selects which expert's slice of the batched
// buffer is read and written. The buffers are laid out as
//   gate_up[2k][ffn]   gate of expert j at 2j, up at 2j + 1
//   act[k][ffn], q8_values[k][ffn], q8_scales[k][ffn / 32], down[k][hidden]
// which is the same flattened order the separate and the fused expert layouts
// produce.
// ---------------------------------------------------------------------------

// Fixed-size device table with the weight pointers of one batched MoE layer.
// The gate/up half holds 2k entries for the separate layout (gate of expert j at
// 2j, up at 2j + 1) and k entries for the fused layout, whose two ffn halves
// already sit in gate_up order; both end up as [2k][ffn] in the same output
// buffer. Kept in one struct so a layer needs a single small H2D copy.
struct Q38MoeExpertPtrTable {
    const uint8_t * gate_up[2 * kLowBitExpertBatchMax];
    const uint8_t * down[kLowBitExpertBatchMax];
};

// ---------------------------------------------------------------------------
// Task I: M-token batch MoE (WARODECODE_MTP_BATCH_VERIFY).
//
// A layer that runs M tokens through the routed MoE has one expert list per
// token, and the same expert may be selected by several of them. The batch
// workspace below flattens the M * k selections into "pairs", so a layer still
// needs a handful of launches (five for the expert GEMMs plus M router
// selections) no matter how many tokens are folded in: the per-launch cost that
// dominates a batch=1 MoE is paid once instead of M times.
//
// Bit-identity to the serial per-token path is structural, not best-effort:
// every pair goes through the same per-row weight decode routine the
// single-expert kernel uses (the Task G contract), the down projection consumes
// the same SiLU(gate) * up product quantised the same way, and each token's
// weighted sum walks its own top-k list in top-k order, exactly the order the
// serial loop of q38_weighted_accum_f32_kernel launches produces.
// ---------------------------------------------------------------------------

// One expert's weight pointers inside the batch: the gate/up pair follows the
// separate layout (gate of the layer's own expert list, up next to it) and the
// down pointer is taken from the token the pair belongs to. "base" is the budget
// 2-byte alignment: the IQ4_NL decoder reads packed nibbles with 16-bit loads.
struct Q38MoeBatchPtrEntry {
    const uint8_t * gate_up[2 * kLowBitExpertBatchMax];
    const uint8_t * down[kLowBitExpertBatchMax];
    uintptr_t base[kLowBitExpertBatchMax];
};

// Host-side description of one (token, expert) pair, copied to the device with
// the pointer table in the same small H2D transfer.
struct Q38MoeBatchPairDesc {
    int activation_pair;   // pair owning the Q8_K/Q8_0 activation, -1 to disable
    int expert;            // expert index inside the pointer-table entry
};

// Per-token view of one batch pass: where the router selects into, where the
// per-token hidden output lands and which pair range belongs to the token.
struct Q38MoeBatchTokenDesc {
    int * ids;
    float * weights;
    float * logits;
    float * out;
    int first_pair;
    int pair_count;
};

// One H2D copy per batch pass carries the whole description; the arrays are
// capped by the largest batch the workspace ever saw (M <= kLowBitExpertBatchMax
// tokens, k <= kLowBitExpertBatchMax experts).
struct Q38MoeBatchPassDesc {
    int tokens;
    int k;
    int hidden;
    int expert_ffn;
    int n_experts;
    float expert_scale;
    int _pad[3];
    Q38MoeBatchTokenDesc token[kLowBitExpertBatchMax];
    Q38MoeBatchPairDesc pair[static_cast<int>(kLowBitExpertBatchMax) * static_cast<int>(kLowBitExpertBatchMax)];
};

// SiLU(gate) * up per expert: q38_hc_silu_scale_ggml_f32_kernel(x, n, inv_hc)
// followed by q38_silu_mul_f32_kernel(). The SiLU output keeps the rounded F32
// value the in-place kernel writes, so the product is bit-identical.
__global__ void q38_moe_silu_scale_mul_experts_kernel(
        const float * __restrict__ gate_up,
        float * __restrict__ act,
        int ffn,
        float inv_hc) {
    const float * gate =
        gate_up + static_cast<size_t>(2) * blockIdx.y * static_cast<size_t>(ffn);
    const float * up = gate + ffn;
    float * out = act + static_cast<size_t>(blockIdx.y) * static_cast<size_t>(ffn);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ffn; i += gridDim.x * blockDim.x) {
        const float v = __fmul_rn(gate[i], inv_hc);
        const float exp_neg = q38_hc_ggml_avx2_exp_f32(-v);
        out[i] = __fmul_rn(__fdiv_rn(v, __fadd_rn(1.0f, exp_neg)), up[i]);
    }
}

// Q8_0 activation quantisation per expert, copied from
// q38_hc_q8_0_quantize_f32_exact_kernel with grid.y selecting the expert.
__global__ void q38_moe_q8_0_quantize_experts_kernel(
        const float * __restrict__ input,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scales,
        int ffn) {
    const int lane = threadIdx.x & 31;
    const int block = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int blocks_per_expert = ffn / 32;
    if (block >= blocks_per_expert) return;

    const int i = block * 32 + lane;
    const float value = input[static_cast<size_t>(blockIdx.y) * ffn + i];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float inverse = amax == 0.0f ? 0.0f : __fdiv_rn(127.0f, amax);
    const int q = amax == 0.0f ? 0 : __float2int_rn(__fmul_rn(value, inverse));
    quantized[static_cast<size_t>(blockIdx.y) * ffn + i] =
        static_cast<int8_t>(max(-128, min(127, q)));
    if (lane == 0) {
        scales[static_cast<size_t>(blockIdx.y) * blocks_per_expert + block] =
            __float2half_rn(d);
    }
}

// Weighted sum of every batched expert output. The serial path launches
// q38_weighted_accum_f32_kernel once per expert in top-k order, so the loop
// below reproduces exactly that sequence of rounded product/add pairs per
// element.
__global__ void q38_moe_weighted_accum_experts_f32_kernel(
        const float * __restrict__ experts,
        const float * __restrict__ weights,
        float * __restrict__ accum,
        int n,
        int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = accum[i];
    for (int j = 0; j < k; ++j) {
        a = q38_add_rn(a, q38_mul_rn(experts[static_cast<size_t>(j) * n + i], weights[j]));
    }
    accum[i] = a;
}

// ---------------------------------------------------------------------------
// Task I: M-token batch kernels. Each one walks "pairs" where the Task G kernels
// walked "experts"; the pair -> (token, expert, activation) mapping comes from
// the pass description, and the arithmetic per pair is the same rounded sequence
// the single-expert kernel runs.
// ---------------------------------------------------------------------------

// gate/up for every pair. blocks/grid.y are indexed by pair exactly like the
// Task G expert-batched kernels index by expert; a pair selects the token's Q8_K
// activation slice instead of a per-expert one.
template <bool USE_SMEM>
__global__ void iq2_s_q8_k_gemv_pairs_dp4a_kernel(
        const uint8_t * const * __restrict__ w_pairs,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        const int * __restrict__ activation_pair,
        float * __restrict__ y,
        int pairs,
        int rows,
        int cols) {
    __shared__ uint64_t grid_table[1024];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
            grid_table[i] = k_iq2s_grid[i];
        }
        __syncthreads();
    }

    const int pair = blockIdx.y;
    if (pair >= pairs) return;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const int activation = activation_pair[pair];
    const int8_t * pair_qs = x_qs + static_cast<size_t>(activation) * static_cast<size_t>(cols);
    const float * pair_d = x_d + static_cast<size_t>(activation) * static_cast<size_t>(cols >> 8);
    const float value = iq2_s_q8_k_row_dp4a<USE_SMEM>(
        w_pairs[pair], pair_qs, pair_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    y[static_cast<size_t>(pair) * static_cast<size_t>(rows) + row] = 0.125f * value;
}

template <bool USE_SMEM>
__global__ void iq3_s_q8_k_gemv_pairs_dp4a_kernel(
        const uint8_t * const * __restrict__ w_pairs,
        const int8_t * __restrict__ x_qs,
        const float * __restrict__ x_d,
        const int * __restrict__ activation_pair,
        float * __restrict__ y,
        int pairs,
        int rows,
        int cols) {
    __shared__ uint32_t grid_table[512];
    if constexpr (USE_SMEM) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) {
            grid_table[i] = k_iq3s_grid[i];
        }
        __syncthreads();
    }

    const int pair = blockIdx.y;
    if (pair >= pairs) return;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const int activation = activation_pair[pair];
    const int8_t * pair_qs = x_qs + static_cast<size_t>(activation) * static_cast<size_t>(cols);
    const float * pair_d = x_d + static_cast<size_t>(activation) * static_cast<size_t>(cols >> 8);
    const float value = iq3_s_q8_k_row_dp4a<USE_SMEM>(
        w_pairs[pair], pair_qs, pair_d, row, cols, grid_table);
    if ((threadIdx.x & 31) != 0) return;
    y[static_cast<size_t>(pair) * static_cast<size_t>(rows) + row] = value;
}

// IQ4_NL down projection for every pair, one token activation per pair. Same
// row routine and row-to-warp mapping as the single-expert kernel, so the rows
// are bit-identical; only grid.y (pair) replaces the expert selection.
__global__ void iq4_nl_q8_0_act_gemv_pairs_dp4a_kernel(
        const uint8_t * const * __restrict__ w_pairs,
        const int8_t * __restrict__ x_qs,
        const __half * __restrict__ x_scales,
        const int * __restrict__ activation_pair,
        float * __restrict__ y,
        int pairs,
        int rows,
        int cols) {
    const int pair = blockIdx.y;
    if (pair >= pairs) return;

    const int lane = threadIdx.x & 31;
    const int row_in_warp = lane >> 4;

    const int output_row =
        (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2 + row_in_warp;
    const int row = min(output_row, rows - 1);

    const int activation = activation_pair[pair];
    const int8_t * pair_x_qs =
        x_qs + static_cast<size_t>(activation) * static_cast<size_t>(cols);
    const __half * pair_x_scales =
        x_scales + static_cast<size_t>(activation) * static_cast<size_t>(cols / 32);

    const float value = iq4_nl_q8_0_act_row_dp4a(
        w_pairs[pair], pair_x_qs, pair_x_scales, row, cols);
    if ((lane & 15) != 0 || output_row >= rows) return;
    y[static_cast<size_t>(pair) * static_cast<size_t>(rows) + output_row] = value;
}

// Routed weighted sum per token. Element i of token t walks the token's own
// top-k list in order and folds in the pair output of each selected expert -
// exactly the sequence of rounded multiply/adds M single-expert accumulation
// launches would produce.
__global__ void q38_moe_batch_accum_kernel(
        const float * __restrict__ experts,
        const float * __restrict__ pair_weights,
        const float * __restrict__ accum_in,
        float * __restrict__ out,
        const Q38MoeBatchPassDesc * __restrict__ pass) {
    const int t = blockIdx.y;
    if (t >= pass->tokens) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pass->hidden) return;

    const Q38MoeBatchTokenDesc & token = pass->token[t];
    float a = accum_in[static_cast<size_t>(t) * pass->hidden + i];
    for (int j = 0; j < token.pair_count; ++j) {
        const int pair = token.first_pair + j;
        a = q38_add_rn(a, q38_mul_rn(
            experts[static_cast<size_t>(pair) * pass->hidden + i], pair_weights[pair]));
    }
    out[static_cast<size_t>(t) * pass->hidden + i] = a;
}

// F16 copy of the M-token routed sum, the batch twin of q38_finish_moe()'s
// q38_float_to_half_kernel: the same __float2half of the same unrounded sum,
// once per token instead of once per token step.
__global__ void q38_moe_batch_to_half_kernel(
        const float * __restrict__ input,
        __half * __restrict__ output,
        int tokens,
        int hidden) {
    const int t = blockIdx.y;
    if (t >= tokens) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    output[static_cast<size_t>(t) * hidden + i] =
        __float2half(input[static_cast<size_t>(t) * hidden + i]);
}

// F16 copy of every batched expert output for the debug traces, matching
// q38_moe_capture_expert_output() on the F32-mid path.
__global__ void q38_moe_capture_experts_kernel(
        const float * __restrict__ experts,
        __half * __restrict__ destination,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t index = static_cast<size_t>(blockIdx.y) * n + i;
    destination[index] = __float2half(experts[index]);
}

// Q8_0 activation quantisation in the exact ggml order (amax/127 with
// round-to-nearest-even division, q = round(value * 127/amax)). Byte-identical
// to ggml's quantize_row_q8_0 for elements that are a multiple of 32.
inline void q38_moe_quantize_activation_q8_0(
        const float * input,
        int elements,
        int8_t * quantized,
        __half * scales,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = elements / 32;
    q38_hc_q8_0_quantize_f32_exact_kernel<<<(blocks + warps - 1) / warps, threads, 0, stream>>>(
        input, elements, quantized, scales);
}

// Task V (WARODECODE_SHEXP_GEMV_FAST, default ON).
//
// The shared expert's gate/up/down projections are Q8_0 x Q8_0 with exactly the
// layout the HC mix / GDN / QSA projections use, but this call site kept the
// scalar kernel while every other caller already went through
// q38_hc_launch_exact_q8_0_gemv_f32(), which selects the dp4a and pipelined
// twins. The three kernels accumulate in the same order, so the switch is a
// pure speed A/B: measured 32.4 -> 8.5 us for [640, 2560] and 14.4 -> 6.2 us for
// [2560, 640] on A100 (tests/qwen38_hc_gemv_bench.cu).
inline bool q38_moe_shexp_gemv_fast_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_GEMV_FAST", true);
    return enabled;
}

// Q8_0 weight x Q8_0 activation integer dot with an F32 output, in the same
// accumulation order the ggml SIMD kernels use.
inline void q38_moe_gemv_q8_0_exact(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const __half * activation_scales,
        float * output,
        cudaStream_t stream) {
    if (q38_moe_shexp_gemv_fast_enabled()) {
        q38_hc_launch_exact_q8_0_gemv_f32(
            weights.data, activation, activation_scales, output,
            weights.rows, weights.cols, stream);
        return;
    }
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int blocks = (weights.rows + warps - 1) / warps;
    q38_hc_q8_0_q8_0_exact_gemv_f32_kernel<<<blocks, threads, 0, stream>>>(
        weights.data, activation, activation_scales, output, weights.rows, weights.cols);
}

// F32-activation twin of half_dot_kernel() (dense_vector.cuh) for the shared
// expert's gate_input projection. The F16 weights widen exactly and the F32
// activation is consumed as-is instead of being rounded to F16 first. The thread
// mapping, the fmaf order and the shared-memory reduction are the same, so an F32
// activation that is exactly F16-representable reproduces half_dot_kernel() bit
// for bit.
__global__ void q38_moe_gate_dot_f32_x_kernel(
        const __half * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ out,
        int n) {
    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum = fmaf(__half2float(a[i]), b[i], sum);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = smem[0];
}

inline void q38_moe_gate_dot_f32_x(
        const __half * weights,
        const float * x,
        float * out,
        int n,
        cudaStream_t stream) {
    q38_moe_gate_dot_f32_x_kernel<<<1, 256, 0, stream>>>(weights, x, out, n);
}

// Tree reduction half of ggml_vec_dot_f32 on AVX2+FMA, the instruction set the
// pinned CPU oracle compiles (GGML_NATIVE=OFF still enables AVX2/FMA, see
// qwen38_gdn.cuh): GGML_F32_ARR == 4 accumulators of GGML_F32_EPR == 8 lanes,
// folded (0,2)/(1,3) then (0,1), then the 8-lane GGML_F32x8_REDUCE
// (t[k] = x[k] + x[k+4], h0 = t0 + t1, h1 = t2 + t3, res = h0 + h1). Callers
// must fill sum[4][8] in AVX2 visit order: base in {0, 32, 64, 96}, stream j in
// 0..3, lane l in 0..7, i = base + j*8 + l with FMA accumulation. Identical to
// q38_gdn_reduce_avx2(); kept as a local copy so this file stays independent of
// qwen38_gdn.cuh.
__device__ __forceinline__ float q38_moe_reduce_avx2(float sum[4][8]) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        sum[0][l] = sum[0][l] + sum[2][l];
        sum[1][l] = sum[1][l] + sum[3][l];
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) sum[0][l] = sum[0][l] + sum[1][l];
    float t0[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) t0[k] = sum[0][k] + sum[0][k + 4];
    const float h0 = t0[0] + t0[1];
    const float h1 = t0[2] + t0[3];
    return h0 + h1;
}

// F32 weight vector x F32 activation -> one ggml_vec_dot_f32 (ggml-cpu/vec.cpp:11)
// in its own visit order, for the shared expert's gate projection
// (build_lora_mm(ffn_gate_inp_shexp, cur) with the F32 weight and F32 activation
// the pinned GGUF selects). ggml_compute_forward_mul_mat() sees
// src1->type == vec_dot_type == F32, skips the activation quantisation and - for
// a single decode token - hands the whole row to vec_dot(ne00, ...) with
// nrc == 1 (llamafile_sgemm() returns false for n < 2, sgemm.cpp:3819).
//
// One thread walks the row, so the visit order is exactly the oracle's single
// vec_dot call: 32-value blocks of four 8-lane FMA accumulators, the
// GGML_F32x8_REDUCE tree, then the "sumf += x[i]*y[i]" non-fused tail over
// n & ~31. n == 2560 for the pinned model, so the tail never runs there.
// Launch with a single thread: thread 0 owns the whole row.
__global__ void q38_moe_gate_dot_f32_avx2_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ out,
        int n) {
    float sum[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) sum[j][l] = 0.0f;
    }
    const int np = n & ~31;
    for (int i = 0; i < np; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int c = i + j * 8 + l;
                sum[j][l] = __fmaf_rn(w[c], x[c], sum[j][l]);
            }
        }
    }
    float acc = q38_moe_reduce_avx2(sum);
    for (int c = np; c < n; ++c) acc = __fadd_rn(acc, __fmul_rn(w[c], x[c]));
    *out = acc;
}

// Warp-parallel twin of the one-thread gate dot above (WARODECODE_SHEXP_GATE_WARP).
//
// q38_moe_reduce_avx2() is the same reduction tree the router GEMV uses, over
// the same 32 accumulators sum[j][l] for elements i + j*8 + l. Assigning
// accumulator j*8+l to lane t and having the lane run that accumulator's own FMA
// chain therefore reproduces the dot bit for bit, while the row is walked
// coalesced instead of one thread reading 10 KB with strided 4-byte loads.
// The one-thread launch measures ~108 us per layer for n = 2560 on the A100
// (~5 ms/token) because a single thread is the whole GPU for that launch.
inline bool q38_moe_gate_dot_f32_warp_enabled() {
    static const bool enabled = q38_moe_env_flag("WARODECODE_SHEXP_GATE_WARP", true);
    return enabled;
}

__global__ void q38_moe_gate_dot_f32_avx2_warp_kernel(
        const float * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ out,
        int n) {
    const int lane = threadIdx.x & 31;
    if (blockIdx.x != 0 || (threadIdx.x >> 5) != 0) return;

    float s = 0.0f;
    const int np = n & ~31;
    for (int i = 0; i < np; i += 32) {
        s = __fmaf_rn(w[i + lane], x[i + lane], s);
    }
    // q38_moe_reduce_avx2() in the lane layout above (see the router GEMV twin
    // for the pairing: lane t holds sum[t >> 3][t & 7]).
    float other = __shfl_down_sync(0xffffffffu, s, 16);
    if (lane < 16) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 8);
    if (lane < 8) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 4);
    if (lane < 4) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 1);
    if (lane == 0 || lane == 2) s = s + other;
    other = __shfl_down_sync(0xffffffffu, s, 2);
    float acc = s + other;
    if (lane == 0) {
        for (int c = np; c < n; ++c) acc = __fadd_rn(acc, __fmul_rn(w[c], x[c]));
        *out = acc;
    }
}

inline void q38_moe_gate_dot_f32_avx2(
        const float * weights,
        const float * x,
        float * out,
        int n,
        cudaStream_t stream) {
    if (q38_moe_gate_dot_f32_warp_enabled()) {
        q38_moe_gate_dot_f32_avx2_warp_kernel<<<1, 32, 0, stream>>>(weights, x, out, n);
        return;
    }
    q38_moe_gate_dot_f32_avx2_kernel<<<1, 1, 0, stream>>>(weights, x, out, n);
}

__global__ void q38_add_shared_kernel(
        __half * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ gate,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float y = __half2float(routed[i]) + (*gate) * __half2float(shared[i]);
    routed[i] = __float2half(y);
}

// F32 twin of q38_add_shared_kernel: routed[i] += shared[i] * gate[0] with no
// F16 intermediate. The multiply/add order matches q38_moe_branch_f32(), which
// the HC combine kernel applies to the same two operands, so the F32 MoE branch
// stays bit-identical whichever side fuses the addition.
__global__ void q38_add_shared_f32_kernel(
        float * __restrict__ routed,
        const float * __restrict__ shared,
        const float * __restrict__ gate,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    routed[i] = q38_moe_branch_f32(routed[i], shared[i], gate[0]);
}

class Qwen38MoeScratch {
public:
    Qwen38MoeScratch() = default;
    Qwen38MoeScratch(const Qwen38MoeScratch &) = delete;
    Qwen38MoeScratch & operator=(const Qwen38MoeScratch &) = delete;

    ~Qwen38MoeScratch() { reset(); }

    void initialize(int n_experts, int k, int hidden, int expert_ffn) {
        if (d_logits_) throw std::runtime_error("MoE scratch already initialized");
        if (n_experts <= 0 || n_experts > 4096 || k <= 0 || k > 16 || k > n_experts ||
            hidden <= 0 || expert_ffn <= 0) {
            throw std::runtime_error("invalid MoE scratch geometry");
        }

        n_experts_ = n_experts;
        k_ = k;
        hidden_ = hidden;
        expert_ffn_ = expert_ffn;

        alloc(reinterpret_cast<void **>(&d_logits_), static_cast<size_t>(n_experts_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_ids_), static_cast<size_t>(k_) * sizeof(int));
        alloc(reinterpret_cast<void **>(&d_weights_), static_cast<size_t>(k_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_gate_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_up_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_gate_up_), static_cast<size_t>(2 * expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_act_), static_cast<size_t>(expert_ffn_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_down_), static_cast<size_t>(hidden_) * sizeof(__half));
        alloc(reinterpret_cast<void **>(&d_accum_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_shared_gate_), sizeof(float));

        // ggml-exact Q8_0 activation scratch (router + shared expert). The F32
        // buffers keep the projections unrounded like the CPU reference; the
        // int8/scale pair holds the quantised activation.
        const int activ = std::max(hidden_, expert_ffn_);
        alloc(reinterpret_cast<void **>(&d_act_f32_), static_cast<size_t>(activ) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_gate_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_up_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_down_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_q8_0_values_), static_cast<size_t>(activ) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_q8_0_scales_),
              static_cast<size_t>((activ + 31) / 32) * sizeof(__half));

        // ggml-exact routed expert scratch: Q8_K for the IQ2_S/IQ3_S gate/up pair
        // (one quantisation per layer, shared by every selected expert), Q8_0 for the
        // IQ4_NL down projection and F32 for the projections in between. The
        // Q8_K buffers are rounded up to whole 256-value blocks so the geometry
        // check, not the allocation, is what decides whether the path applies.
        const int q8k_blocks = (hidden_ + 255) / 256;
        alloc(reinterpret_cast<void **>(&d_rx_x_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_values_),
              static_cast<size_t>(q8k_blocks) * 256u * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_scales_), static_cast<size_t>(q8k_blocks) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8k_bsums_),
              static_cast<size_t>(q8k_blocks) * 16u * sizeof(int16_t));
        alloc(reinterpret_cast<void **>(&d_rx_gate_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_up_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_gate_up_f32_), static_cast<size_t>(2 * expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_act_f32_), static_cast<size_t>(expert_ffn_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_down_f32_), static_cast<size_t>(hidden_) * sizeof(float));
        alloc(reinterpret_cast<void **>(&d_rx_q8_0_values_), static_cast<size_t>(expert_ffn_) * sizeof(int8_t));
        alloc(reinterpret_cast<void **>(&d_rx_q8_0_scales_),
              static_cast<size_t>((expert_ffn_ + 31) / 32) * sizeof(__half));

        host_alloc(reinterpret_cast<void **>(&h_ids_), static_cast<size_t>(k_) * sizeof(int));
        host_alloc(reinterpret_cast<void **>(&h_weights_), static_cast<size_t>(k_) * sizeof(float));
    }

    __half * selected_output_trace() {
        if (!d_selected_output_trace_) {
            alloc(reinterpret_cast<void **>(&d_selected_output_trace_),
                  static_cast<size_t>(k_) * hidden_ * sizeof(__half));
        }
        return d_selected_output_trace_;
    }

    // F32 MoE branch (routed weighted sum + shared expert output * gate). Only
    // the WARODECODE_MOE_F32_MID path asks for it, so it is allocated on first
    // use instead of growing every legacy session.
    float * moe_out_f32() {
        if (!d_moe_out_f32_) {
            alloc(reinterpret_cast<void **>(&d_moe_out_f32_),
                  static_cast<size_t>(hidden_) * sizeof(float));
        }
        return d_moe_out_f32_;
    }

    // F16 mirror of the MoE activation for the legacy F16-activation fallbacks
    // of the F32-input runners (the ggml-exact Q8/Q8_K path never needs it).
    // Lazily allocated like moe_out_f32() so legacy sessions do not grow.
    __half * half_activation() {
        if (!d_half_activation_) {
            alloc(reinterpret_cast<void **>(&d_half_activation_),
                  static_cast<size_t>(hidden_) * sizeof(__half));
        }
        return d_half_activation_;
    }

    int n_experts() const { return n_experts_; }
    int k() const { return k_; }
    int hidden() const { return hidden_; }
    int expert_ffn() const { return expert_ffn_; }

    float * d_logits() const { return d_logits_; }
    int * d_ids() const { return d_ids_; }
    float * d_weights() const { return d_weights_; }
    __half * d_gate() const { return d_gate_; }
    __half * d_up() const { return d_up_; }
    __half * d_gate_up() const { return d_gate_up_; }
    __half * d_act() const { return d_act_; }
    __half * d_down() const { return d_down_; }
    float * d_accum() const { return d_accum_; }
    float * d_shared_gate() const { return d_shared_gate_; }
    float * act_f32() const { return d_act_f32_; }
    float * gate_f32() const { return d_gate_f32_; }
    float * up_f32() const { return d_up_f32_; }
    float * down_f32() const { return d_down_f32_; }
    int8_t * q8_0_values() const { return d_q8_0_values_; }
    __half * q8_0_scales() const { return d_q8_0_scales_; }
    float * rx_x_f32() const { return d_rx_x_f32_; }
    int8_t * rx_q8k_values() const { return d_rx_q8k_values_; }
    float * rx_q8k_scales() const { return d_rx_q8k_scales_; }
    int16_t * rx_q8k_bsums() const { return d_rx_q8k_bsums_; }
    float * rx_gate_f32() const { return d_rx_gate_f32_; }
    float * rx_up_f32() const { return d_rx_up_f32_; }
    float * rx_gate_up_f32() const { return d_rx_gate_up_f32_; }
    float * rx_act_f32() const { return d_rx_act_f32_; }
    float * rx_down_f32() const { return d_rx_down_f32_; }
    int8_t * rx_q8_0_values() const { return d_rx_q8_0_values_; }
    __half * rx_q8_0_scales() const { return d_rx_q8_0_scales_; }

    // Task G: expert-batched MoE workspaces (WARODECODE_MOE_FUSED_GRID). Lazily
    // allocated like moe_out_f32() so sessions that never take the batched path
    // do not grow.
    float * rx_batch_gate_up() {
        if (!d_rx_batch_gate_up_) {
            alloc(reinterpret_cast<void **>(&d_rx_batch_gate_up_),
                  static_cast<size_t>(2) * static_cast<size_t>(k_) *
                      static_cast<size_t>(expert_ffn_) * sizeof(float));
        }
        return d_rx_batch_gate_up_;
    }
    float * rx_batch_act() {
        if (!d_rx_batch_act_) {
            alloc(reinterpret_cast<void **>(&d_rx_batch_act_),
                  static_cast<size_t>(k_) * static_cast<size_t>(expert_ffn_) * sizeof(float));
        }
        return d_rx_batch_act_;
    }
    int8_t * rx_batch_q8_0_values() {
        if (!d_rx_batch_q8_0_values_) {
            alloc(reinterpret_cast<void **>(&d_rx_batch_q8_0_values_),
                  static_cast<size_t>(k_) * static_cast<size_t>(expert_ffn_) * sizeof(int8_t));
        }
        return d_rx_batch_q8_0_values_;
    }
    __half * rx_batch_q8_0_scales() {
        if (!d_rx_batch_q8_0_scales_) {
            alloc(reinterpret_cast<void **>(&d_rx_batch_q8_0_scales_),
                  static_cast<size_t>(k_) * static_cast<size_t>(expert_ffn_ / 32) * sizeof(__half));
        }
        return d_rx_batch_q8_0_scales_;
    }
    float * rx_batch_down() {
        if (!d_rx_batch_down_) {
            alloc(reinterpret_cast<void **>(&d_rx_batch_down_),
                  static_cast<size_t>(k_) * static_cast<size_t>(hidden_) * sizeof(float));
        }
        return d_rx_batch_down_;
    }
    // -----------------------------------------------------------------------
    // Task I: M-token batch workspaces (WARODECODE_MTP_BATCH_VERIFY).
    //
    // One "pair" is a (token, expert) selection, so a layer that M tokens route
    // through holds tokens * k pairs at most. The buffers are indexed by pair and
    // hold tokens * k slots; a smaller batch simply leaves the tail unused. They
    // are sized on first use and grow when a later layer needs more, exactly like
    // the Task G workspaces above.
    // -----------------------------------------------------------------------
    enum class BatchBuffer {
        GateUp,       // [2 * pairs][ffn]    gate at pair p * 2, up at pair p * 2 + 1
        Act,          // [pairs][ffn]        SiLU(gate) * up
        Q8Values,     // [pairs][ffn]        Q8_0 activation of Act
        Q8Scales,     // [pairs][ffn / 32]
        Down,         // [pairs][hidden]     expert output before the weighted sum
        Accum,        // [tokens][hidden]    routed weighted sum
        PtrTable,     // [pairs] weight pointers, one device table entry each
        Chart,        // [pairs] pass description, activation map and pair weights
        Logits,       // [tokens][n_experts]
        Ids,          // [tokens][k]      router top-k expert ids
        Weights,      // [tokens][k]      router top-k weights
    };

    static size_t batch_buffer_bytes(BatchBuffer which, int tokens, int hidden, int expert_ffn, int k, int n_experts) {
        const size_t pairs = static_cast<size_t>(tokens) * static_cast<size_t>(k);
        const size_t n = static_cast<size_t>(hidden);
        const size_t ffn = static_cast<size_t>(expert_ffn);
        const size_t t = static_cast<size_t>(tokens);
        switch (which) {
            case BatchBuffer::GateUp: return 2u * pairs * ffn * sizeof(float);
            case BatchBuffer::Act: return pairs * ffn * sizeof(float);
            case BatchBuffer::Q8Values: return pairs * ffn * sizeof(int8_t);
            case BatchBuffer::Q8Scales: return pairs * (ffn / 32u) * sizeof(__half);
            case BatchBuffer::Down: return pairs * n * sizeof(float);
            case BatchBuffer::Accum: return t * n * sizeof(float);
            case BatchBuffer::PtrTable: return pairs * sizeof(Q38MoeBatchPtrEntry);
            case BatchBuffer::Chart: return sizeof(Q38MoeBatchPassDesc) + pairs * (sizeof(Q38MoeBatchPairDesc) + sizeof(int) + sizeof(float));
            case BatchBuffer::Logits: return t * static_cast<size_t>(n_experts) * sizeof(float);
            case BatchBuffer::Ids: return pairs * sizeof(int);
            case BatchBuffer::Weights: return pairs * sizeof(float);
        }
        return 0;
    }

    static int batch_capacity_slots(BatchBuffer which, size_t bytes, int hidden, int expert_ffn, int k, int n_experts) {
        const size_t ffn = static_cast<size_t>(expert_ffn);
        const size_t n = static_cast<size_t>(hidden);
        const size_t kk = static_cast<size_t>(k);
        switch (which) {
            case BatchBuffer::GateUp: return static_cast<int>(bytes / (2u * kk * ffn * sizeof(float)));
            case BatchBuffer::Act: return static_cast<int>(bytes / (kk * ffn * sizeof(float)));
            case BatchBuffer::Q8Values: return static_cast<int>(bytes / (kk * ffn * sizeof(int8_t)));
            case BatchBuffer::Q8Scales: return static_cast<int>(bytes / (kk * (ffn / 32u) * sizeof(__half)));
            case BatchBuffer::Down: return static_cast<int>(bytes / (kk * n * sizeof(float)));
            case BatchBuffer::Accum: return static_cast<int>(bytes / (kk * n * sizeof(float)));
            case BatchBuffer::PtrTable: return static_cast<int>(bytes / (kk * sizeof(Q38MoeBatchPtrEntry)));
            case BatchBuffer::Chart:
                return static_cast<int>(bytes / (kk * (sizeof(Q38MoeBatchPairDesc) + sizeof(int) + sizeof(float))));
            case BatchBuffer::Logits:
                return n_experts > 0 ? static_cast<int>(bytes / (kk * static_cast<size_t>(n_experts) * sizeof(float))) : 0;
            case BatchBuffer::Ids: return static_cast<int>(bytes / (kk * sizeof(int)));
            case BatchBuffer::Weights: return static_cast<int>(bytes / (kk * sizeof(float)));
        }
        return 0;
    }

    void * batch_buffer(BatchBuffer which, int tokens) {
        if (tokens <= 0) throw std::runtime_error("MoE batch buffer needs at least one token");
        void ** slot = batch_slot(which);
        const size_t needed = batch_buffer_bytes(which, tokens, hidden_, expert_ffn_, k_, n_experts_);
        if (*slot == nullptr) {
            alloc(slot, needed);
            batch_capacity_[batch_slot_index(which)] = tokens;
            return *slot;
        }
        if (batch_capacity_[batch_slot_index(which)] < tokens) {
            cudaFree(*slot);
            *slot = nullptr;
            alloc(slot, needed);
            batch_capacity_[batch_slot_index(which)] = tokens;
        }
        return *slot;
    }

    Q38MoeBatchPtrEntry * host_batch_ptrs(int entries) {
        if (entries <= 0) throw std::runtime_error("MoE batch pointer table needs at least one entry");
        if (h_batch_ptrs_ == nullptr) {
            host_alloc(reinterpret_cast<void **>(&h_batch_ptrs_), static_cast<size_t>(entries) * sizeof(Q38MoeBatchPtrEntry));
            h_batch_ptr_entries_ = entries;
        } else if (h_batch_ptr_entries_ < entries) {
            cudaFreeHost(h_batch_ptrs_);
            h_batch_ptrs_ = nullptr;
            host_alloc(reinterpret_cast<void **>(&h_batch_ptrs_), static_cast<size_t>(entries) * sizeof(Q38MoeBatchPtrEntry));
            h_batch_ptr_entries_ = entries;
        }
        return h_batch_ptrs_;
    }

    int batch_capacity(BatchBuffer which) const { return batch_capacity_[batch_slot_index(which)]; }

    // Weight-pointer table of one batched layer: filled in pinned host memory
    // and copied to the device once per layer.
    // Task R: the pointer table of a layer is uploaded once per expert group,
    // and cudaMemcpyAsync() from pinned host memory reads the source while the
    // stream is still draining the previous group. One shared staging table
    // would therefore be rewritten by the next group before its predecessor's
    // H2D had read it (observed as wrong experts and, when a torn 8-byte
    // pointer reached the device, an illegal memory access). Each group gets
    // its own host and device table instead.
    static constexpr int kTableSlots = 8;
    Q38MoeExpertPtrTable * host_expert_ptr_table(int slot = 0) {
        if (!h_expert_ptr_table_) {
            host_alloc(reinterpret_cast<void **>(&h_expert_ptr_table_),
                       sizeof(Q38MoeExpertPtrTable) * kTableSlots);
        }
        return h_expert_ptr_table_ + (slot % kTableSlots);
    }
    Q38MoeExpertPtrTable * expert_ptr_table(int slot = 0) {
        if (!d_expert_ptr_table_) {
            alloc(reinterpret_cast<void **>(&d_expert_ptr_table_),
                  sizeof(Q38MoeExpertPtrTable) * kTableSlots);
        }
        return d_expert_ptr_table_ + (slot % kTableSlots);
    }
    int * h_ids() const { return h_ids_; }
    float * h_weights() const { return h_weights_; }

private:
    static void alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMalloc(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc MoE scratch failed: ") + cudaGetErrorString(e));
    }

    static void host_alloc(void ** p, size_t n) {
        const cudaError_t e = cudaMallocHost(p, n);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMallocHost MoE scratch failed: ") + cudaGetErrorString(e));
    }

    // Index of a Task I batch workspace in d_batch_[] / batch_capacity_[].
    static int batch_slot_index(BatchBuffer which) {
        switch (which) {
            case BatchBuffer::GateUp: return 0;
            case BatchBuffer::Act: return 1;
            case BatchBuffer::Q8Values: return 2;
            case BatchBuffer::Q8Scales: return 3;
            case BatchBuffer::Down: return 4;
            case BatchBuffer::Accum: return 5;
            case BatchBuffer::PtrTable: return 6;
            case BatchBuffer::Chart: return 7;
            case BatchBuffer::Logits: return 8;
            case BatchBuffer::Ids: return 9;
            case BatchBuffer::Weights: return 10;
        }
        return 0;
    }

    static constexpr int kBatchBufferCount = 11;

    void ** batch_slot(BatchBuffer which) { return &d_batch_[batch_slot_index(which)]; }

    void reset() {
        if (d_logits_) cudaFree(d_logits_);
        if (d_ids_) cudaFree(d_ids_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_gate_) cudaFree(d_gate_);
        if (d_up_) cudaFree(d_up_);
        if (d_gate_up_) cudaFree(d_gate_up_);
        if (d_act_) cudaFree(d_act_);
        if (d_down_) cudaFree(d_down_);
        if (d_accum_) cudaFree(d_accum_);
        if (d_shared_gate_) cudaFree(d_shared_gate_);
        if (d_act_f32_) cudaFree(d_act_f32_);
        if (d_gate_f32_) cudaFree(d_gate_f32_);
        if (d_up_f32_) cudaFree(d_up_f32_);
        if (d_down_f32_) cudaFree(d_down_f32_);
        if (d_q8_0_values_) cudaFree(d_q8_0_values_);
        if (d_q8_0_scales_) cudaFree(d_q8_0_scales_);
        if (d_rx_x_f32_) cudaFree(d_rx_x_f32_);
        if (d_rx_q8k_values_) cudaFree(d_rx_q8k_values_);
        if (d_rx_q8k_scales_) cudaFree(d_rx_q8k_scales_);
        if (d_rx_q8k_bsums_) cudaFree(d_rx_q8k_bsums_);
        if (d_rx_gate_f32_) cudaFree(d_rx_gate_f32_);
        if (d_rx_up_f32_) cudaFree(d_rx_up_f32_);
        if (d_rx_gate_up_f32_) cudaFree(d_rx_gate_up_f32_);
        if (d_rx_act_f32_) cudaFree(d_rx_act_f32_);
        if (d_rx_down_f32_) cudaFree(d_rx_down_f32_);
        if (d_rx_q8_0_values_) cudaFree(d_rx_q8_0_values_);
        if (d_rx_q8_0_scales_) cudaFree(d_rx_q8_0_scales_);
        if (d_rx_batch_gate_up_) cudaFree(d_rx_batch_gate_up_);
        if (d_rx_batch_act_) cudaFree(d_rx_batch_act_);
        if (d_rx_batch_q8_0_values_) cudaFree(d_rx_batch_q8_0_values_);
        if (d_rx_batch_q8_0_scales_) cudaFree(d_rx_batch_q8_0_scales_);
        if (d_rx_batch_down_) cudaFree(d_rx_batch_down_);
        if (d_expert_ptr_table_) cudaFree(d_expert_ptr_table_);
        if (h_expert_ptr_table_) cudaFreeHost(h_expert_ptr_table_);
        for (int i = 0; i < kBatchBufferCount; ++i) {
            if (d_batch_[i]) cudaFree(d_batch_[i]);
            d_batch_[i] = nullptr;
            batch_capacity_[i] = 0;
        }
        if (h_batch_ptrs_) cudaFreeHost(h_batch_ptrs_);
        h_batch_ptrs_ = nullptr;
        h_batch_ptr_entries_ = 0;
        if (d_selected_output_trace_) cudaFree(d_selected_output_trace_);
        if (d_moe_out_f32_) cudaFree(d_moe_out_f32_);
        if (d_half_activation_) cudaFree(d_half_activation_);
        if (h_ids_) cudaFreeHost(h_ids_);
        if (h_weights_) cudaFreeHost(h_weights_);
        d_logits_ = nullptr;
        d_ids_ = nullptr;
        d_weights_ = nullptr;
        d_gate_ = d_up_ = d_gate_up_ = d_act_ = d_down_ = nullptr;
        d_accum_ = nullptr;
        d_shared_gate_ = nullptr;
        d_act_f32_ = d_gate_f32_ = d_up_f32_ = d_down_f32_ = nullptr;
        d_q8_0_values_ = nullptr;
        d_q8_0_scales_ = nullptr;
        d_rx_batch_gate_up_ = d_rx_batch_act_ = nullptr;
        d_rx_batch_q8_0_values_ = nullptr;
        d_rx_batch_q8_0_scales_ = nullptr;
        d_rx_batch_down_ = nullptr;
        d_expert_ptr_table_ = nullptr;
        h_expert_ptr_table_ = nullptr;
        d_selected_output_trace_ = nullptr;
        d_moe_out_f32_ = nullptr;
        d_half_activation_ = nullptr;
        h_ids_ = nullptr;
        h_weights_ = nullptr;
    }

    int n_experts_ = 0;
    int k_ = 0;
    int hidden_ = 0;
    int expert_ffn_ = 0;

    float * d_logits_ = nullptr;
    int * d_ids_ = nullptr;
    float * d_weights_ = nullptr;
    __half * d_gate_ = nullptr;
    __half * d_up_ = nullptr;
    __half * d_gate_up_ = nullptr;
    __half * d_act_ = nullptr;
    __half * d_down_ = nullptr;
    float * d_accum_ = nullptr;
    float * d_shared_gate_ = nullptr;
    float * d_act_f32_ = nullptr;
    float * d_gate_f32_ = nullptr;
    float * d_up_f32_ = nullptr;
    float * d_down_f32_ = nullptr;
    int8_t * d_q8_0_values_ = nullptr;
    __half * d_q8_0_scales_ = nullptr;
    float * d_rx_x_f32_ = nullptr;
    int8_t * d_rx_q8k_values_ = nullptr;
    float * d_rx_q8k_scales_ = nullptr;
    int16_t * d_rx_q8k_bsums_ = nullptr;
    float * d_rx_gate_f32_ = nullptr;
    float * d_rx_up_f32_ = nullptr;
    float * d_rx_gate_up_f32_ = nullptr;
    float * d_rx_act_f32_ = nullptr;
    float * d_rx_down_f32_ = nullptr;
    int8_t * d_rx_q8_0_values_ = nullptr;
    __half * d_rx_q8_0_scales_ = nullptr;
    float * d_rx_batch_gate_up_ = nullptr;
    float * d_rx_batch_act_ = nullptr;
    int8_t * d_rx_batch_q8_0_values_ = nullptr;
    __half * d_rx_batch_q8_0_scales_ = nullptr;
    float * d_rx_batch_down_ = nullptr;
    Q38MoeExpertPtrTable * d_expert_ptr_table_ = nullptr;
    Q38MoeExpertPtrTable * h_expert_ptr_table_ = nullptr;
    // Task I: M-token batch workspaces, indexed by batch_slot_index().
    void * d_batch_[kBatchBufferCount] = {};
    int batch_capacity_[kBatchBufferCount] = {};
    Q38MoeBatchPtrEntry * h_batch_ptrs_ = nullptr;
    int h_batch_ptr_entries_ = 0;
    __half * d_selected_output_trace_ = nullptr;
    float * d_moe_out_f32_ = nullptr;
    __half * d_half_activation_ = nullptr;
    int * h_ids_ = nullptr;
    float * h_weights_ = nullptr;
};

// The router only needs the Q8_0 activation path when the weight itself is
// Q8_0 (ggml does not quantise the activation for F16/F32 weights) and the
// hidden width is a whole number of 32-value Q8_0 blocks.
inline bool q38_moe_router_q8_0_compatible(
        const GpuMatrix & router,
        const Qwen38MoeScratch & scratch) {
    return router.lowbit && router.type == GgmlType::Q8_0 &&
        router.rows == scratch.n_experts() && router.cols == scratch.hidden() &&
        scratch.hidden() > 0 && scratch.hidden() % 32 == 0;
}

// Same rule for the shared expert: every GEMV input width (hidden for
// gate/up, expert_ffn for down) must be a whole number of Q8_0 blocks.
inline bool q38_moe_shexp_q8_0_compatible(
        const Qwen38SharedExpertWeights & w,
        const Qwen38MoeScratch & scratch) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (hidden <= 0 || expert_ffn <= 0 || hidden % 32 != 0 || expert_ffn % 32 != 0) return false;
    return w.gate.type == GgmlType::Q8_0 && w.gate.rows == expert_ffn && w.gate.cols == hidden &&
        w.up.type == GgmlType::Q8_0 && w.up.rows == expert_ffn && w.up.cols == hidden &&
        w.down.type == GgmlType::Q8_0 && w.down.rows == hidden && w.down.cols == expert_ffn;
}

// ggml-exact routed expert activation path. It needs the gate/up activation to
// be a whole number of Q8_K blocks (256 values) and the down activation to be an
// even number of Q8_0 blocks (64 values), because ggml's AVX2 IQ4_NL kernel
// consumes two 32-value blocks per step and falls back to a scalar tail
// otherwise.
inline bool q38_moe_expert_q8_act_geometry(int hidden, int expert_ffn) {
    return hidden > 0 && hidden % 256 == 0 && expert_ffn > 0 && expert_ffn % 64 == 0;
}

// Both IQ2_S and IQ3_S declare Q8_K as their ggml vec_dot_type, so the packed
// gate/up activation feeds either kernel; only the weight decode differs. The
// model is IQ2_S everywhere except blk.2's routed experts, which are IQ3_S.
inline bool q38_moe_gate_up_q8_k_type(GgmlType type) {
    return type == GgmlType::IQ2_S || type == GgmlType::IQ3_S;
}

inline bool q38_moe_separate_experts_q8_act_compatible(
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        const Qwen38MoeScratch & scratch) {
    if (!q38_moe_expert_q8_act_enabled()) return false;
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (!q38_moe_expert_q8_act_geometry(hidden, expert_ffn)) return false;
    if (gate_exps.dims.size() != 3 || up_exps.dims.size() != 3 || down_exps.dims.size() != 3) return false;
    return q38_moe_gate_up_q8_k_type(gate_exps.type) &&
            static_cast<int>(gate_exps.dims[0]) == hidden &&
            static_cast<int>(gate_exps.dims[1]) == expert_ffn &&
        q38_moe_gate_up_q8_k_type(up_exps.type) &&
            static_cast<int>(up_exps.dims[0]) == hidden &&
            static_cast<int>(up_exps.dims[1]) == expert_ffn &&
        down_exps.type == GgmlType::IQ4_NL &&
            static_cast<int>(down_exps.dims[0]) == expert_ffn &&
            static_cast<int>(down_exps.dims[1]) == hidden;
}

inline bool q38_moe_fused_experts_q8_act_compatible(
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        const Qwen38MoeScratch & scratch) {
    if (!q38_moe_expert_q8_act_enabled()) return false;
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    if (!q38_moe_expert_q8_act_geometry(hidden, expert_ffn)) return false;
    if (gate_up_exps.dims.size() != 3 || down_exps.dims.size() != 3) return false;
    return q38_moe_gate_up_q8_k_type(gate_up_exps.type) &&
            static_cast<int>(gate_up_exps.dims[0]) == hidden &&
            static_cast<int>(gate_up_exps.dims[1]) == 2 * expert_ffn &&
        down_exps.type == GgmlType::IQ4_NL &&
            static_cast<int>(down_exps.dims[0]) == expert_ffn &&
            static_cast<int>(down_exps.dims[1]) == hidden;
}

inline void q38_select_experts(
        const GpuMatrix & router,
        const __half * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }
    Q38MoeKernelScope profile_scope(kQ38MoeSiteRouter, stream);

    constexpr int threads = 256;
    const int hidden = scratch.hidden();
    if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
        // ggml quantises the activation to Q8_0 before the matmul; reproduce
        // that rounding instead of the legacy F16 activation GEMV.
        q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        q38_moe_quantize_activation_q8_0(
            scratch.act_f32(), hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
        q38_moe_gemv_q8_0_exact(
            router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
    } else if (q38_moe_router_f32_enabled() && matrix_has_f32_weights(router)) {
        // This model's ffn_gate_inp is F32, and ggml keeps non-quantised tensors
        // in F32 on both sides (vec_dot_type = F32). The router decides which
        // experts run, so its rounding is amplified downstream.
        q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        launch_matrix_gemv_f32_weights(router, scratch.act_f32(), scratch.d_logits(), stream);
    } else {
        launch_matrix_gemv_float(router, x, scratch.d_logits(), stream);
    }

    q38_launch_topk_softmax(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights(), stream);

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert ids failed: ") + cudaGetErrorString(e));

    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy expert weights failed: ") + cudaGetErrorString(e));

    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("expert selection sync failed: ") + cudaGetErrorString(e));
}

inline void q38_select_experts_f32_input(
        const GpuMatrix & router,
        const float * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }
    Q38MoeKernelScope profile_scope(kQ38MoeSiteRouter, stream);

    if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
        // The F32 router input can be quantised in place, exactly like ggml.
        q38_moe_quantize_activation_q8_0(
            x, scratch.hidden(), scratch.q8_0_values(), scratch.q8_0_scales(), stream);
        q38_moe_gemv_q8_0_exact(
            router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
    } else {
        launch_matrix_gemv_f32_input(router, x, scratch.d_logits(), stream);
    }
    q38_launch_topk_softmax(
        scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
        scratch.d_ids(), scratch.d_weights(), stream);

    cudaError_t e = cudaMemcpyAsync(
        scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-router expert ids failed: ") + cudaGetErrorString(e));
    e = cudaMemcpyAsync(
        scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-router expert weights failed: ") + cudaGetErrorString(e));
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw std::runtime_error(std::string("F32-router selection sync failed: ") + cudaGetErrorString(e));
}

// F32-activation router selection for the *_f32_x runners. ggml quantises the
// activation only when the router weight is quantised; an F32 router keeps both
// sides F32 and never sees the F16 weight copy (launch_matrix_gemv_f32_weights).
// The dense fallback widens the F16 weight copy exactly, so the only difference
// from q38_select_experts_f32_input() is which of the two non-quantised weight
// copies is used. The legacy half-input q38_select_experts() is untouched.
inline void q38_moe_select_experts_f32_x(
        const GpuMatrix & router,
        const float * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if (router.rows != scratch.n_experts() || router.cols != scratch.hidden()) {
        throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
    }
    Q38MoeKernelScope profile_scope(kQ38MoeSiteRouter, stream);

    {
        Q38MoeKernelScope quant_scope(kQ38MoeSiteRouterQuant, stream);
        if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
            // The F32 router input is quantised in place, exactly like ggml.
            q38_moe_quantize_activation_q8_0(
                x, scratch.hidden(), scratch.q8_0_values(), scratch.q8_0_scales(), stream);
            q38_moe_gemv_q8_0_exact(
                router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
        } else {
            Q38MoeKernelScope gemv_scope(kQ38MoeSiteRouterGemv, stream);
            if (q38_moe_router_f32_enabled() && matrix_has_f32_weights(router)) {
                if (q38_moe_router_f32_warp_enabled()) {
                    launch_q38_router_gemv_f32(router, x, scratch.d_logits(), stream);
                } else {
                    launch_matrix_gemv_f32_weights(router, x, scratch.d_logits(), stream);
                }
            } else {
                launch_matrix_gemv_f32_input(router, x, scratch.d_logits(), stream);
            }
        }
    }

    {
        Q38MoeKernelScope topk_scope(kQ38MoeSiteRouterTopk, stream);
        q38_launch_topk_softmax(
            scratch.d_logits(), scratch.n_experts(), scratch.k(), expert_scale,
            scratch.d_ids(), scratch.d_weights(), stream);
    }

    cudaError_t e = cudaSuccess;
    {
        Q38MoeKernelScope d2h_scope(kQ38MoeSiteRouterD2h, stream);
        e = cudaMemcpyAsync(
            scratch.h_ids(), scratch.d_ids(), static_cast<size_t>(scratch.k()) * sizeof(int),
            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-x router expert ids failed: ") + cudaGetErrorString(e));
        e = cudaMemcpyAsync(
            scratch.h_weights(), scratch.d_weights(), static_cast<size_t>(scratch.k()) * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("copy F32-x router expert weights failed: ") + cudaGetErrorString(e));
    }
    {
        Q38MoeKernelScope sync_scope(kQ38MoeSiteRouterSync, stream);
        e = cudaStreamSynchronize(stream);
    }
    if (e != cudaSuccess) throw std::runtime_error(std::string("F32-x router selection sync failed: ") + cudaGetErrorString(e));
}

// Activation-type dispatch shared by the routed/expert runners: F16 activations
// take the legacy widening (exact) and F32 activations are consumed directly.
template <typename XT>
inline void q38_moe_select_experts_input(
        const GpuMatrix & router,
        const XT * x,
        Qwen38MoeScratch & scratch,
        float expert_scale,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_select_experts_f32_x(router, x, scratch, expert_scale, stream);
    } else {
        q38_select_experts(router, x, scratch, expert_scale, stream);
    }
}

// Write the routed weighted sum into the caller's MoE output. The F16 output
// (legacy contract) rounds once, like before; the F32 output keeps the sum
// unrounded for the F32 HC combine (out == d_accum() is allowed and is a no-op).
template <typename OutT>
inline void q38_finish_moe(
        Qwen38MoeScratch & scratch,
        OutT * out,
        cudaStream_t stream) {
    Q38MoeKernelScope profile_scope(kQ38MoeSiteFinish, stream);
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    const int blocks = (hidden + threads - 1) / threads;
    if constexpr (std::is_same_v<OutT, float>) {
        if (out == scratch.d_accum()) return;
        const cudaError_t e = cudaMemcpyAsync(
            out, scratch.d_accum(), static_cast<size_t>(hidden) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("copy F32 MoE output failed: ") + cudaGetErrorString(e));
        }
    } else {
        q38_float_to_half_kernel<<<blocks, threads, 0, stream>>>(
            scratch.d_accum(), out, hidden);
    }
}

// F32 twin of q38_add_shared_kernel(): routed[i] += shared[i] * gate[0].
inline void q38_moe_add_shared_f32(
        float * routed_in_out,
        const float * shared,
        const float * gate,
        int n,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    q38_add_shared_f32_kernel<<<blocks, threads, 0, stream>>>(
        routed_in_out, shared, gate, n);
}

// Complete F32 MoE branch: out = routed weighted sum + shared expert * gate,
// the F32 twin of q38_finish_moe() followed by q38_add_shared_kernel(). "out"
// must be a scratch buffer distinct from d_accum(): the routed sum stays
// available for tracing (layer*.ffn.*), and the caller typically passes
// Qwen38MoeScratch::moe_out_f32().
inline void q38_finish_moe_f32(
        Qwen38MoeScratch & scratch,
        float * out,
        cudaStream_t stream) {
    if (out == scratch.d_accum()) {
        throw std::runtime_error(
            "F32 MoE output must not alias the routed accumulation buffer");
    }
    q38_finish_moe(scratch, out, stream);
    q38_moe_add_shared_f32(
        out, scratch.down_f32(), scratch.d_shared_gate(), scratch.hidden(), stream);
}

// F16 activation for the legacy F16-activation fallbacks of the F32-input
// runners. Those GEMV kernels are F16-activation by construction, so the F32
// activation is rounded once here; the ggml-exact Q8/Q8_K paths never call this.
// An F16 input is returned unchanged (no conversion, no rounding).
template <typename XT>
inline const __half * q38_moe_legacy_half_activation(
        const XT * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        const int hidden = scratch.hidden();
        constexpr int threads = 256;
        q38_hc_copy_f32_to_half_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
            x, scratch.half_activation(), hidden);
        return scratch.half_activation();
    } else {
        return x;
    }
}

// Accumulate one selected expert into the routed F32 sum. The F32-mid path
// reads the unrounded projection (ggml keeps the per-expert product in F32);
// the legacy path reads the F16 buffer the expert wrote. Exactly one of the two
// source pointers is used.
inline void q38_moe_accumulate_expert(
        const float * expert_f32,
        const __half * expert_half,
        float weight,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    const int blocks = (hidden + threads - 1) / threads;
    if (f32_mid) {
        q38_weighted_accum_f32_kernel<<<blocks, threads, 0, stream>>>(
            expert_f32, weight, scratch.d_accum(), hidden);
    } else {
        q38_weighted_accum_kernel<<<blocks, threads, 0, stream>>>(
            expert_half, weight, scratch.d_accum(), hidden);
    }
}

// Debug trace of one selected expert's output. The trace buffer stays F16
// (forward.cuh reads it with copy_half and exposes it as
// layer*.ffn.selected_expert_outputs), so the F32-mid path rounds only this
// copy and never the value that feeds the accumulation.
inline void q38_moe_capture_expert_output(
        const float * expert_f32,
        const __half * expert_half,
        __half * destination,
        int n,
        bool f32_mid,
        cudaStream_t stream) {
    if (f32_mid) {
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        q38_float_to_half_kernel<<<blocks, threads, 0, stream>>>(expert_f32, destination, n);
        return;
    }
    const cudaError_t e = cudaMemcpyAsync(
        destination, expert_half, static_cast<size_t>(n) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("copy selected expert output failed: ") + cudaGetErrorString(e));
    }
}

// Quantise the MoE input to Q8_K once per layer. gate and up consume the same
// activation, so every selected expert reuses this single quantisation instead
// of rounding the activation per projection.
inline void q38_moe_quantize_routed_x_q8_k(
        const __half * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    constexpr int threads = 256;
    q38_hc_copy_half_to_f32_kernel<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
        x, scratch.rx_x_f32(), hidden);
    launch_q8_k_quantize_f32(
        scratch.rx_x_f32(), hidden, scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
        scratch.rx_q8k_bsums(), stream);
}

// F32-activation twin of q38_moe_quantize_routed_x_q8_k(): the caller's F32
// activation is quantised in place instead of being rounded to F16 first.
// Bit-identical to the F16 version whenever x is exactly F16-representable
// (F16 -> F32 widening is exact, so the quantiser sees the same values).
inline void q38_moe_quantize_routed_x_q8_k_f32(
        const float * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    launch_q8_k_quantize_f32(
        x, scratch.hidden(), scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
        scratch.rx_q8k_bsums(), stream);
}

// Activation-type dispatch for the routed expert gate/up quantisation.
template <typename XT>
inline void q38_moe_quantize_routed_x(
        const XT * x,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_quantize_routed_x_q8_k_f32(x, scratch, stream);
    } else {
        q38_moe_quantize_routed_x_q8_k(x, scratch, stream);
    }
}

// IQ2_S and IQ3_S both consume the Q8_K activation, so the routed gate/up
// projection only has to pick the matching weight decoder. Anything else is a
// programming error: q38_moe_gate_up_q8_k_type() already gated the dispatch.
inline void q38_moe_gemv_gate_up_q8_k_f32(
        const LowBitMatrix & weights,
        const int8_t * activation,
        const float * activation_scales,
        float * output,
        cudaStream_t stream) {
    if (weights.type == GgmlType::IQ2_S) {
        launch_iq2_s_gemv_q8_k_f32(weights, activation, activation_scales, output, stream);
    } else if (weights.type == GgmlType::IQ3_S) {
        launch_iq3_s_gemv_q8_k_f32(weights, activation, activation_scales, output, stream);
    } else {
        throw std::runtime_error("routed expert gate/up weight is neither IQ2_S nor IQ3_S");
    }
}

// Shared second half of the ggml-order routed expert FFN: ggml's vectorised SiLU
// on the F32 gate, the gated product in F32, its Q8_0 re-quantisation and the
// IQ4_NL down projection. The down projection lands in rx_down_f32(); when
// round_half is set it is additionally copied to the F16 d_down() buffer so the
// legacy q38_weighted_accum_kernel() and the legacy outputs stay unchanged.
// The F32-mid path passes round_half = false and accumulates rx_down_f32().
inline void q38_moe_finish_routed_expert_q8_act(
        float * gate_f32,
        const float * up_f32,
        const LowBitMatrix & down,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    constexpr int threads = 256;
    const int ffn_blocks = (expert_ffn + threads - 1) / threads;
    const int hidden_blocks = (hidden + threads - 1) / threads;

    q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        gate_f32, expert_ffn, 1.0f);
    q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        gate_f32, up_f32, scratch.rx_act_f32(), expert_ffn);
    q38_moe_quantize_activation_q8_0(
        scratch.rx_act_f32(), expert_ffn, scratch.rx_q8_0_values(), scratch.rx_q8_0_scales(), stream);
    launch_iq4_nl_gemv_q8_0_f32(
        down, scratch.rx_q8_0_values(), scratch.rx_q8_0_scales(), scratch.rx_down_f32(), stream);
    if (round_half) {
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.rx_down_f32(), scratch.d_down(), hidden);
    }
}

// XT is the activation type: __half keeps the legacy widen-then-quantise step,
// float quantises the caller's F32 activation directly.
template <bool CaptureSelectedOutput, typename XT = __half>
inline void q38_moe_run_routed_experts_q8_act_separate(
        int layer,
        const XT * x,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();

    q38_moe_quantize_routed_x(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);
        if (!q38_moe_gate_up_q8_k_type(w.gate.type) || !q38_moe_gate_up_q8_k_type(w.up.type) ||
            w.down.type != GgmlType::IQ4_NL) {
            throw std::runtime_error("Qwen3.8 routed expert low-bit type changed inside a layer");
        }

        q38_moe_gemv_gate_up_q8_k_f32(
            w.gate, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_gate_f32(), stream);
        q38_moe_gemv_gate_up_q8_k_f32(
            w.up, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_up_f32(), stream);
        // round_half is the F16 fallback: the F32 path keeps rx_down_f32().
        q38_moe_finish_routed_expert_q8_act(
            scratch.rx_gate_f32(), scratch.rx_up_f32(), w.down, scratch, !f32_mid, stream);

        if constexpr (CaptureSelectedOutput) {
            q38_moe_capture_expert_output(
                scratch.rx_down_f32(), scratch.d_down(),
                selected_output_trace + static_cast<size_t>(j) * hidden,
                hidden, f32_mid, stream);
        }
        q38_moe_accumulate_expert(
            scratch.rx_down_f32(), scratch.d_down(), weight, scratch, f32_mid, stream);
    }
}

template <bool CaptureSelectedOutput, typename XT = __half>
inline void q38_moe_run_routed_experts_q8_act_fused(
        int layer,
        const XT * x,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        bool f32_mid,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();

    q38_moe_quantize_routed_x(x, scratch, stream);

    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);
        if (!q38_moe_gate_up_q8_k_type(w.gate_up.type) || w.down.type != GgmlType::IQ4_NL) {
            throw std::runtime_error("Qwen3.8 fused expert low-bit type changed inside a layer");
        }

        q38_moe_gemv_gate_up_q8_k_f32(
            w.gate_up, scratch.rx_q8k_values(), scratch.rx_q8k_scales(), scratch.rx_gate_up_f32(), stream);
        // The fused matrix stores gate first and up second, matching
        // q38_silu_mul_fused_kernel on the half path.
        q38_moe_finish_routed_expert_q8_act(
            scratch.rx_gate_up_f32(), scratch.rx_gate_up_f32() + expert_ffn, w.down, scratch,
            !f32_mid, stream);

        if constexpr (CaptureSelectedOutput) {
            q38_moe_capture_expert_output(
                scratch.rx_down_f32(), scratch.d_down(),
                selected_output_trace + static_cast<size_t>(j) * hidden,
                hidden, f32_mid, stream);
        }
        q38_moe_accumulate_expert(
            scratch.rx_down_f32(), scratch.d_down(), weight, scratch, f32_mid, stream);
    }
}

// ---------------------------------------------------------------------------
// Task I: M-token batch MoE (WARODECODE_MTP_BATCH_VERIFY).
// ---------------------------------------------------------------------------

// Longest token batch one pass folds together. The description copy is fixed
// size, and the practical call site (MTP verify) never exceeds k + 1 tokens.
constexpr int kMoeBatchTokenMax = kLowBitExpertBatchMax;

// Expert batch cap for the M-token path, a separate knob from the Task G value
// so either path can be measured with the other one off.
inline int q38_moe_batch_k() {
    const char * v = std::getenv("WARODECODE_MOE_BATCH_K");
    int k = 4;
    if (v != nullptr && v[0] != '\0') {
        const int parsed = std::atoi(v);
        if (parsed > 0) k = parsed;
    }
    if (k < 1) k = 1;
    if (k > kLowBitExpertBatchMax) k = kLowBitExpertBatchMax;
    return k;
}

// M-token batch MoE (WARODECODE_MTP_BATCH_VERIFY).
//
// The batch is a flat list of "pairs": one (token, expert, activation) triple
// per selected expert. Every pair runs the same per-row weight decode the
// single-expert kernels run, and each token's weighted sum walks its own top-k
// list in order, so a batched pass over M tokens is bit-identical to M serial
// passes - the same property Task G's expert-batched grid established for the
// k experts of one token, extended over tokens.
//
// Layout of one batch pass, all staged in one device allocation:
//   gate_up_ptrs[pairs]   per-pair gate and up pointers, interleaved [g,u,g,u..]
//   down_ptrs[pairs]      per-pair down projection pointers
//   activation[pairs]     pair -> activation slice (token) index
//   weights[pairs]        pair -> router weight of the selection
//   ptr_entries[pairs]    Q38MoeBatchPtrEntry, the layout set_pair_weights fills
//   desc                  Q38MoeBatchPassDesc, per-token views of the pass
class Qwen38MoeTokenBatch {
public:
    Qwen38MoeTokenBatch() = default;
    Qwen38MoeTokenBatch(const Qwen38MoeTokenBatch &) = delete;
    Qwen38MoeTokenBatch & operator=(const Qwen38MoeTokenBatch &) = delete;

    ~Qwen38MoeTokenBatch() {
        if (host_) cudaFreeHost(host_);
        if (device_) cudaFree(device_);
    }

    int batch_k() const { return batch_k_; }

    // A batch pass is possible only for the ggml-exact F32-mid routed MoE the
    // model actually runs: IQ2_S/IQ3_S gate/up feeding an IQ4_NL down
    // projection, with the DP4A pair kernels (they have no reference fallback).
    //
    // The slot bound is 8 slots per top-k selection: one batch holds at most
    // tokens * k <= k * k pairs, and the acquire loop never holds more than
    // 2 * k of them at once (each pair's down view is dropped before the next
    // pair is acquired), so 8 * k leaves a factor of four of headroom. The
    // caller still verifies after the acquire loop that every pair's slot is
    // still the one it acquired - the bound is what makes the check pass, the
    // check is what makes it safe (see Qwen38MoeTokenBatch::note_pair_slot).
    bool compatible(
            const GpuExpertCache & cache,
            const Qwen38MoeScratch & scratch,
            const TensorInfo & gate_exps,
            const TensorInfo & up_exps,
            const TensorInfo & down_exps,
            int tokens) const {
        if (tokens < 1 || tokens > kMoeBatchTokenMax) return false;
        if (scratch.k() < 1 || scratch.k() > batch_k_) return false;
        if (!q38_moe_gate_up_q8_k_type(gate_exps.type) || !q38_moe_gate_up_q8_k_type(up_exps.type)) return false;
        if (down_exps.type != GgmlType::IQ4_NL) return false;
        if (!q38_moe_expert_q8_act_geometry(scratch.hidden(), scratch.expert_ffn())) return false;
        if (gate_exps.dims.size() != 3 || up_exps.dims.size() != 3 || down_exps.dims.size() != 3) return false;
        if (static_cast<int>(gate_exps.dims[0]) != scratch.hidden() ||
            static_cast<int>(gate_exps.dims[1]) != scratch.expert_ffn() ||
            static_cast<int>(up_exps.dims[0]) != scratch.hidden() ||
            static_cast<int>(up_exps.dims[1]) != scratch.expert_ffn() ||
            static_cast<int>(down_exps.dims[0]) != scratch.expert_ffn() ||
            static_cast<int>(down_exps.dims[1]) != scratch.hidden()) {
            return false;
        }
        if (!lowbit_dp4a_available()) return false;
        return cache.slot_count() >= 8u * static_cast<size_t>(scratch.k());
    }

    // Phase 1: select the experts of every token. The router GEMVs and the
    // top-k kernels are enqueued token by token (each one writes the shared MoE
    // scratch, so they cannot overlap), but the ids and weights are read back
    // with a single synchronise instead of the per-token cudaStreamSynchronize()
    // the serial selection performs. preselected mirrors the serial call site:
    // the caller already ran the router for the activations it hands in.
    void open_all(
            Qwen38MoeScratch & scratch,
            const GpuMatrix & router,
            const std::vector<const float *> & activations,
            float expert_scale,
            cudaStream_t stream,
            bool preselected) {
        tokens_ = static_cast<int>(activations.size());
        if (tokens_ < 1 || tokens_ > kMoeBatchTokenMax) {
            throw std::runtime_error("MoE token batch holds one to the batch bound of tokens");
        }
        if (!(expert_scale > 0.0f)) {
            throw std::runtime_error("MoE token batch needs a positive expert scale");
        }
        for (int t = 0; t < tokens_; ++t) {
            if (activations[static_cast<size_t>(t)] == nullptr) {
                throw std::runtime_error("MoE token batch got a null activation");
            }
        }

        const int k = scratch.k();
        const int hidden = scratch.hidden();
        const size_t pairs = static_cast<size_t>(tokens_) * static_cast<size_t>(k);
        k_ = k;

        activations_ = activations;
        ids_ = static_cast<int *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Ids, tokens_));
        weights_ = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Weights, tokens_));
        float * logits = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Logits, tokens_));

        if (!preselected) {
            if (router.rows != scratch.n_experts() || router.cols != hidden) {
                throw std::runtime_error("Qwen3.8 router matrix geometry mismatch");
            }
            cudaError_t cuda_error = cudaSuccess;
            for (int t = 0; t < tokens_ && cuda_error == cudaSuccess; ++t) {
                const float * x = activations[static_cast<size_t>(t)];
                if (q38_moe_router_q8_0_enabled() && q38_moe_router_q8_0_compatible(router, scratch)) {
                    q38_moe_quantize_activation_q8_0(
                        x, hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
                    q38_moe_gemv_q8_0_exact(
                        router.low, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.d_logits(), stream);
                } else if (q38_moe_router_f32_enabled() && matrix_has_f32_weights(router)) {
                    launch_matrix_gemv_f32_weights(router, x, scratch.d_logits(), stream);
                } else {
                    launch_matrix_gemv_f32_input(router, x, scratch.d_logits(), stream);
                }
                q38_topk_softmax_kernel<<<1, 1, 0, stream>>>(
                    scratch.d_logits(), scratch.n_experts(), k, expert_scale,
                    scratch.d_ids(), scratch.d_weights());
                cuda_error = cudaMemcpyAsync(
                    logits + static_cast<size_t>(t) * static_cast<size_t>(scratch.n_experts()),
                    scratch.d_logits(), static_cast<size_t>(scratch.n_experts()) * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
                if (cuda_error == cudaSuccess) {
                    cuda_error = cudaMemcpyAsync(
                        ids_ + static_cast<size_t>(t) * static_cast<size_t>(k),
                        scratch.d_ids(), static_cast<size_t>(k) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
                }
                if (cuda_error == cudaSuccess) {
                    cuda_error = cudaMemcpyAsync(
                        weights_ + static_cast<size_t>(t) * static_cast<size_t>(k),
                        scratch.d_weights(), static_cast<size_t>(k) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
                }
            }
            if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
            if (cuda_error != cudaSuccess) {
                throw std::runtime_error(std::string("MoE token batch selection failed: ") +
                                         cudaGetErrorString(cuda_error));
            }
        }

        Q38MoeBatchPassDesc * host = static_cast<Q38MoeBatchPassDesc *>(host_staging());
        host->tokens = tokens_;
        host->k = k;
        host->hidden = hidden;
        host->expert_ffn = scratch.expert_ffn();
        host->n_experts = scratch.n_experts();
        host->expert_scale = expert_scale;
        for (int t = 0; t < tokens_; ++t) {
            host->token[t].ids = ids_ + static_cast<size_t>(t) * static_cast<size_t>(k);
            host->token[t].weights = weights_ + static_cast<size_t>(t) * static_cast<size_t>(k);
            host->token[t].logits = logits + static_cast<size_t>(t) * static_cast<size_t>(scratch.n_experts());
            host->token[t].out = nullptr;
            host->token[t].first_pair = t * k;
            host->token[t].pair_count = k;
        }
        (void)pairs;
    }

    // Phase 2: resolve one pair's weights. gate/up follow the separate layout;
    // every base is checked for the 2-byte alignment the IQ4_NL decoder needs
    // and for the type the dispatcher was built for, so run() can re-check the
    // whole batch and fall back to the serial loop on any mismatch.
    // Pins the expert a pair belongs to when the caller already ran the router
    // (the batch's own preselected path). Used by the device bench to compare a
    // fixed per-token expert list against the serial runner.
    void set_pair_experts(int pair, int expert, const std::vector<float> & weights_by_pair) {
        if (pair < 0 || pair >= kMoeBatchTokenMax * kLowBitExpertBatchMax) {
            throw std::runtime_error("MoE token batch pair index out of range");
        }
        set_pair_experts_[pair] = expert;
        if (weights_ != nullptr) weights_[pair] = weights_by_pair[static_cast<size_t>(pair)];
    }

    // One pair's weight pointers. Whether the batch is still runnable after the
    // acquire loop is decided by resolved(true), which uses these pointers.
    void set_pair_weights(
            int pair,
            const LowBitMatrix & gate,
            const LowBitMatrix & up,
            const LowBitMatrix & down) {
        if (pair < 0 || pair >= kMoeBatchTokenMax * kLowBitExpertBatchMax) {
            throw std::runtime_error("MoE token batch pair index out of range");
        }
        Q38MoeBatchPtrEntry & entry = ptr_entries()[pair];
        entry.gate_up[0] = gate.data;
        entry.gate_up[1] = up.data;
        entry.down[0] = down.data;
        entry.base[0] = reinterpret_cast<uintptr_t>(gate.data) |
            reinterpret_cast<uintptr_t>(up.data) | reinterpret_cast<uintptr_t>(down.data);
        if (pair == 0) gate_type_ = gate.type;
    }

    // True when every pair staged by set_pair_weights() is runnable: the right
    // weight types, the 2-byte bases the IQ4_NL decoder needs, and - when the
    // caller reports the slots it acquired - a slot that is still live.
    //
    // The slot check is the eviction proof. A cache slot is one (layer, expert)
    // residue, so two pairs of the same batch that resolved to *different*
    // experts must not share a slot base: if they do, a later acquire recycled
    // an earlier pair and the batch must fall back to the serial loop. Pairs
    // that share an expert are expected to share the base.
    bool resolved(bool check_slots = false) {
        if (gate_type_ != GgmlType::IQ2_S && gate_type_ != GgmlType::IQ3_S) return false;
        const int pairs = tokens_ * k_;
        for (int p = 0; p < pairs; ++p) {
            const Q38MoeBatchPtrEntry & entry = ptr_entries()[p];
            if ((entry.base[0] & 1u) != 0) return false;
            if (entry.gate_up[0] == nullptr || entry.gate_up[1] == nullptr || entry.down[0] == nullptr) return false;
        }
        if (!check_slots || ids_ == nullptr) return true;
        for (int a = 0; a < pairs; ++a) {
            for (int b = a + 1; b < pairs; ++b) {
                if (ptr_entries()[b].gate_up[0] != ptr_entries()[a].gate_up[0]) continue;
                if (ptr_entries()[b].down[0] != ptr_entries()[a].down[0]) continue;
                if (ids_[a] != ids_[b]) return false;
            }
        }
        return true;
    }

    // Phase 3: one staged batch pass. The description and the pointer arrays go
    // to the device in four small H2D copies; the five imports after that are
    // the whole routed expert computation of the layer.
    void run(
            Qwen38MoeScratch & scratch,
            const TensorInfo & gate_exps,
            const TensorInfo & up_exps,
            const TensorInfo & down_exps,
            cudaStream_t stream) {
        (void)gate_exps;
        (void)up_exps;
        (void)down_exps;

        const int k = scratch.k();
        const int pairs = tokens_ * k;
        const int hidden = scratch.hidden();
        const int ffn = scratch.expert_ffn();
        if (tokens_ < 1 || k < 1 || pairs <= 0 || pairs > kMoeBatchTokenMax * kLowBitExpertBatchMax) {
            throw std::runtime_error("MoE token batch pass is not staged");
        }

        float * gate_up = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::GateUp, tokens_));
        float * act = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Act, tokens_));
        int8_t * q8_values = static_cast<int8_t *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Q8Values, tokens_));
        __half * q8_scales = static_cast<__half *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Q8Scales, tokens_));
        float * down = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Down, tokens_));
        float * accum = static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Accum, tokens_));

        uint8_t * device = device_staging();
        const size_t ptr_entries_bytes = static_cast<size_t>(pairs) * sizeof(Q38MoeBatchPtrEntry);
        const size_t gate_ptr_bytes = static_cast<size_t>(pairs) * sizeof(uint8_t *);
        uint8_t * offsets = device;
        Q38MoeBatchPtrEntry * device_entries = reinterpret_cast<Q38MoeBatchPtrEntry *>(offsets);
        offsets += ptr_entries_bytes;
        uint8_t ** device_gate_up = reinterpret_cast<uint8_t **>(offsets);
        offsets += gate_ptr_bytes;
        uint8_t ** device_down = reinterpret_cast<uint8_t **>(offsets);
        offsets += gate_ptr_bytes;
        Q38MoeBatchPassDesc * device_pass = reinterpret_cast<Q38MoeBatchPassDesc *>(offsets);
        offsets += sizeof(Q38MoeBatchPassDesc);
        int * device_activation = reinterpret_cast<int *>(offsets);
        offsets += static_cast<size_t>(pairs) * sizeof(int);
        float * device_weights = reinterpret_cast<float *>(offsets);

        // Host staging: flat arrays the kernels index by pair. The pointer table
        // stays the Q38MoeBatchPtrEntry shape set_pair_weights() fills, the GEMV
        // kernels get the two flat arrays built from it.
        uint8_t ** gate_up_host = flat_gate_up();
        uint8_t ** down_host = flat_down();
        const Q38MoeBatchPtrEntry * entries = ptr_entries();
        Q38MoeBatchPassDesc * pass = static_cast<Q38MoeBatchPassDesc *>(host_staging());
        for (int p = 0; p < pairs; ++p) {
            gate_up_host[2 * p] = const_cast<uint8_t *>(entries[p].gate_up[0]);
            gate_up_host[2 * p + 1] = const_cast<uint8_t *>(entries[p].gate_up[1]);
            down_host[p] = const_cast<uint8_t *>(entries[p].down[0]);
        }

        Q38MoeBatchPairDesc * pair_desc = pair_descriptors();
        int * activation_host = activation_map();
        float * weights_host = pair_weight_map();
        for (int t = 0; t < tokens_; ++t) {
            for (int j = 0; j < k; ++j) {
                const int pair = t * k + j;
                pair_desc[pair].activation_pair = t;
                pair_desc[pair].expert = j;
                activation_host[pair] = t;
                weights_host[pair] = weights_[pair];
            }
        }

        cudaError_t cuda_error = cudaMemcpyAsync(
            device_entries, entries, ptr_entries_bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_error == cudaSuccess) {
            cuda_error = cudaMemcpyAsync(
                device_gate_up, gate_up_host, gate_ptr_bytes, cudaMemcpyHostToDevice, stream);
        }
        if (cuda_error == cudaSuccess) {
            cuda_error = cudaMemcpyAsync(
                device_down, down_host, gate_ptr_bytes, cudaMemcpyHostToDevice, stream);
        }
        if (cuda_error == cudaSuccess) {
            cuda_error = cudaMemcpyAsync(
                device_pass, pass, sizeof(Q38MoeBatchPassDesc), cudaMemcpyHostToDevice, stream);
        }
        if (cuda_error == cudaSuccess) {
            cuda_error = cudaMemcpyAsync(
                device_activation, activation_host, static_cast<size_t>(pairs) * sizeof(int),
                cudaMemcpyHostToDevice, stream);
        }
        if (cuda_error == cudaSuccess) {
            cuda_error = cudaMemcpyAsync(
                device_weights, weights_host, static_cast<size_t>(pairs) * sizeof(float),
                cudaMemcpyHostToDevice, stream);
        }
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(std::string("MoE token batch upload failed: ") +
                                     cudaGetErrorString(cuda_error));
        }

        // The weighted-sum kernel folds the k expert products INTO the
        // accumulation buffer, so it has to start from zero - the serial runner
        // does the same with a cudaMemsetAsync on d_accum() before its loop.
        // Skipping this reads whatever the previous layer left in the workspace.
        {
            const cudaError_t zero = cudaMemsetAsync(
                accum, 0, static_cast<size_t>(tokens_) * static_cast<size_t>(hidden) * sizeof(float), stream);
            if (zero != cudaSuccess) {
                throw std::runtime_error(std::string("MoE token batch zero accumulation failed: ") +
                                         cudaGetErrorString(zero));
            }
        }

        constexpr int threads = 256;
        const int warps = threads / 32;
        const int gate_up_blocks = (ffn + warps - 1) / warps;
        const bool smem = lowbit_dp4a_smem_enabled();
        if (gate_type_ == GgmlType::IQ2_S) {
            if (smem) {
                iq2_s_q8_k_gemv_pairs_dp4a_kernel<true><<<dim3(gate_up_blocks, pairs), threads, 0, stream>>>(
                    const_cast<const uint8_t * const *>(device_gate_up), q8_values, nullptr,
                    device_activation, gate_up, pairs, ffn, hidden);
            } else {
                iq2_s_q8_k_gemv_pairs_dp4a_kernel<false><<<dim3(gate_up_blocks, pairs), threads, 0, stream>>>(
                    const_cast<const uint8_t * const *>(device_gate_up), q8_values, nullptr,
                    device_activation, gate_up, pairs, ffn, hidden);
            }
        } else {
            if (smem) {
                iq3_s_q8_k_gemv_pairs_dp4a_kernel<true><<<dim3(gate_up_blocks, pairs), threads, 0, stream>>>(
                    const_cast<const uint8_t * const *>(device_gate_up), q8_values, nullptr,
                    device_activation, gate_up, pairs, ffn, hidden);
            } else {
                iq3_s_q8_k_gemv_pairs_dp4a_kernel<false><<<dim3(gate_up_blocks, pairs), threads, 0, stream>>>(
                    const_cast<const uint8_t * const *>(device_gate_up), q8_values, nullptr,
                    device_activation, gate_up, pairs, ffn, hidden);
            }
        }

        const int ffn_blocks = (ffn + threads - 1) / threads;
        q38_moe_silu_scale_mul_experts_kernel<<<dim3(ffn_blocks, pairs), threads, 0, stream>>>(
            gate_up, act, ffn, 1.0f);

        const int q8_blocks = (ffn / 32 + (threads / 32) - 1) / (threads / 32);
        q38_moe_q8_0_quantize_experts_kernel<<<dim3(q8_blocks, pairs), threads, 0, stream>>>(
            act, q8_values, q8_scales, ffn);

        constexpr int rows_per_warp = 2;
        const int down_blocks = (hidden + warps * rows_per_warp - 1) / (warps * rows_per_warp);
        iq4_nl_q8_0_act_gemv_pairs_dp4a_kernel<<<dim3(down_blocks, pairs), threads, 0, stream>>>(
            const_cast<const uint8_t * const *>(device_down), q8_values, q8_scales,
            device_activation, down, pairs, hidden, ffn);

        const int hidden_blocks = (hidden + threads - 1) / threads;
        q38_moe_batch_accum_kernel<<<dim3(hidden_blocks, tokens_), threads, 0, stream>>>(
            down, device_weights, nullptr, accum, device_pass);
    }

    // The batch's routed weighted sum is READY in scratch.d_batch Accum; the
    // caller turns it into the per-token MoE output with a finish kernel.
    float * accumulated(Qwen38MoeScratch & scratch) {
        return static_cast<float *>(scratch.batch_buffer(Qwen38MoeScratch::BatchBuffer::Accum, tokens_));
    }

    int tokens() const { return tokens_; }

    // Writes the routed sums of every token into one contiguous output and
    // returns its dimensions. out_f32 keeps the unrounded F32 values.
    void write_output(
            Qwen38MoeScratch & scratch,
            float * out_f32_or_null,
            __half * out_half_or_null,
            cudaStream_t stream) {
        const int hidden = scratch.hidden();
        const float * accum = accumulated(scratch);
        const int hidden_blocks = (hidden + 255) / 256;
        cudaError_t cuda_error = cudaSuccess;
        if (out_half_or_null != nullptr) {
            // The unrounded sum is rounded exactly once, in the same
            // __float2half the serial q38_finish_moe() would have applied.
            q38_moe_batch_to_half_kernel<<<dim3(hidden_blocks, tokens_), 256, 0, stream>>>(
                accum, out_half_or_null, tokens_, hidden);
        } else {
            const cudaError_t copy_error = cudaMemcpyAsync(
                out_f32_or_null, accum, static_cast<size_t>(hidden) * tokens_ * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            cuda_error = copy_error;
        }
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(std::string("MoE token batch output failed: ") +
                                     cudaGetErrorString(cuda_error));
        }
    }

private:
    // Pointer arrays (host) ...
    uint8_t ** flat_gate_up() {
        ensure_host();
        return reinterpret_cast<uint8_t **>(static_cast<uint8_t *>(host_) + ptr_entries_bytes());
    }
    uint8_t ** flat_down() {
        ensure_host();
        return reinterpret_cast<uint8_t **>(static_cast<uint8_t *>(host_) + ptr_entries_bytes() + gate_ptr_bytes());
    }
    Q38MoeBatchPtrEntry * ptr_entries() {
        ensure_host();
        return reinterpret_cast<Q38MoeBatchPtrEntry *>(host_);
    }
    Q38MoeBatchPassDesc * host_staging() {
        ensure_host();
        return reinterpret_cast<Q38MoeBatchPassDesc *>(
            static_cast<uint8_t *>(host_) + ptr_entries_bytes() + 2 * gate_ptr_bytes());
    }
    Q38MoeBatchPairDesc * pair_descriptors() {
        ensure_host();
        return reinterpret_cast<Q38MoeBatchPairDesc *>(
            reinterpret_cast<uint8_t *>(host_staging()) + sizeof(Q38MoeBatchPassDesc));
    }
    int * activation_map() {
        ensure_host();
        return reinterpret_cast<int *>(
            reinterpret_cast<uint8_t *>(pair_descriptors()) + pair_desc_bytes());
    }
    float * pair_weight_map() {
        ensure_host();
        return reinterpret_cast<float *>(
            reinterpret_cast<uint8_t *>(activation_map()) + activation_bytes());
    }

    size_t ptr_entries_bytes() const { return pair_capacity() * sizeof(Q38MoeBatchPtrEntry); }
    size_t gate_ptr_bytes() const { return pair_capacity() * sizeof(uint8_t *); }
    size_t pair_desc_bytes() const { return pair_capacity() * sizeof(Q38MoeBatchPairDesc); }
    size_t activation_bytes() const { return pair_capacity() * sizeof(int); }
    size_t pair_weight_bytes() const { return pair_capacity() * sizeof(float); }
    size_t pass_offset_bytes() const { return ptr_entries_bytes() + 2 * gate_ptr_bytes(); }

    // Pairs a pass can ever hold: the largest token batch times this session's
    // expert batch cap.
    size_t pair_capacity() const {
        return static_cast<size_t>(kMoeBatchTokenMax) * static_cast<size_t>(batch_k_);
    }

    size_t host_bytes() const {
        return pass_offset_bytes() + sizeof(Q38MoeBatchPassDesc) + pair_desc_bytes() +
            activation_bytes() + pair_weight_bytes();
    }

    // The device staging mirrors the host one, minus the elements no kernel
    // reads: pair descriptors stay host-only, so only the two pointer arrays,
    // the pass description, the activation map and the pair weights are copied.
    size_t device_bytes() const {
        return ptr_entries_bytes() + 2 * gate_ptr_bytes() + sizeof(Q38MoeBatchPassDesc) +
            activation_bytes() + pair_weight_bytes();
    }

    void ensure_host() {
        if (host_ == nullptr) {
            const cudaError_t e = cudaMallocHost(&host_, host_bytes());
            if (e != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMallocHost MoE token batch failed: ") +
                                         cudaGetErrorString(e));
            }
        }
    }

    uint8_t * device_staging() {
        if (device_ == nullptr) {
            const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&device_), device_bytes());
            if (e != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc MoE token batch staging failed: ") +
                                         cudaGetErrorString(e));
            }
        }
        return device_;
    }

    int batch_k_ = q38_moe_batch_k();
    int tokens_ = 0;
    GgmlType gate_type_ = GgmlType::F32;
    int k_ = 0;
    void * host_ = nullptr;
    uint8_t * device_ = nullptr;
    std::vector<const float *> activations_;
    int * ids_ = nullptr;
    float * weights_ = nullptr;
    int set_pair_experts_[kMoeBatchTokenMax * kLowBitExpertBatchMax] = {};
};

// ---------------------------------------------------------------------------
// Task G: expert-batched layer execution (WARODECODE_MOE_FUSED_GRID).
// ---------------------------------------------------------------------------

// True when the batched path may be attempted. The path acquires all k expert
// slots of a layer before it launches anything, so a later acquire of the same
// layer must not be able to evict one of its predecessors. choose_slot() evicts
// the least recently used slot, which keeps the k slots this layer just took
// untouched as long as the cache holds at least k slots *per layer*; 64 per
// layer is a conservative bound for the 49-layer model. The batched GEMV has no
// reference fallback, so the DP4A kernels have to be available, and the legacy
// F16-mid path (which accumulates through d_down()) stays on its own loop.
inline bool q38_moe_fused_grid_candidate(
        const Qwen38MoeScratch & scratch,
        size_t cache_slots,
        bool f32_mid) {
    if (!q38_moe_fused_grid_enabled()) return false;
    const int k = scratch.k();
    const char * blocker = nullptr;
    if (!lowbit_dp4a_available()) blocker = "the DP4A GEMV kernels are unavailable";
    else if (!f32_mid) blocker = "the legacy F16-mid path is active";
    else if (k <= 0 || k > kLowBitExpertBatchMax) blocker = "top-k is outside the batch bound";
    else if (cache_slots < 64u * static_cast<size_t>(k)) blocker = "the expert cache holds too few slots";
    if (blocker != nullptr) {
        // One line per process: which of the switches decided against the
        // batched path, so an A/B run can tell "no effect" from "never taken".
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "warodecode: MoE expert-batched grid not taken (%s; WARODECODE_MOE_FUSED_GRID=1)\n",
                         blocker);
        }
        return false;
    }
    static bool announced = false;
    if (!announced) {
        announced = true;
        std::fprintf(stderr,
                     "warodecode: MoE expert-batched grid taken (top-k=%d, cache slots=%zu, 5 launches/layer)\n",
                     k, cache_slots);
    }
    return true;
}

// One small H2D copy carries the weight pointers of the whole layer. It is
// enqueued after every acquire() and before the batched kernels that read it,
// so the stream order is the acquire -> copy -> kernels order the serial loop
// would have had per expert.
inline void q38_moe_upload_expert_ptr_table(
        const Q38MoeExpertPtrTable & table,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream,
        int slot = 0) {
    Q38MoeKernelScope profile_scope(kQ38MoeSitePtrH2d, stream);
    Q38MoeExpertPtrTable * host = scratch.host_expert_ptr_table(slot);
    *host = table;
    const cudaError_t e = cudaMemcpyAsync(
        scratch.expert_ptr_table(slot), host, sizeof(Q38MoeExpertPtrTable),
        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy MoE expert pointer table failed: ") + cudaGetErrorString(e));
    }
}

// The five launches that replace the per-expert loop. gate_up_experts /
// gate_up_rows describe the pointer table: 2k rows of ffn for the separate
// gate/up layout (gate of expert j at 2j, up at 2j + 1), or k rows of 2 * ffn
// for a fused gate_up tensor. Both fill the same [2k][ffn] gate/up buffer.
// device_table must be scratch.expert_ptr_table(): the kernels read
// w_experts[blockIdx.y] from device memory, so handing them the host struct the
// upload was filled from would dereference host addresses on the device.
template <bool CaptureSelectedOutput>
inline void q38_moe_run_batched_experts(
        const Q38MoeExpertPtrTable * device_table,
        GgmlType gate_up_type,
        int gate_up_experts,
        int gate_up_rows,
        int k,
        __half * selected_output_trace,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream,
        const float * weights) {
    Q38MoeKernelScope profile_scope(kQ38MoeSiteBatchTotal, stream);
    const int hidden = scratch.hidden();
    const int ffn = scratch.expert_ffn();
    constexpr int threads = 256;

    bool launched = false;
    {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteGateUpGemv, stream);
        if (gate_up_type == GgmlType::IQ2_S) {
            launched = launch_iq2_s_gemv_q8_k_experts_f32(
                device_table->gate_up, gate_up_experts, gate_up_rows, hidden,
                scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
                scratch.rx_batch_gate_up(), stream, threads);
        } else {
            launched = launch_iq3_s_gemv_q8_k_experts_f32(
                device_table->gate_up, gate_up_experts, gate_up_rows, hidden,
                scratch.rx_q8k_values(), scratch.rx_q8k_scales(),
                scratch.rx_batch_gate_up(), stream, threads);
        }
    }
    if (!launched) {
        throw std::runtime_error("expert-batched gate/up GEMV needs the DP4A kernels");
    }

    // SiLU(gate) * up for every expert. The routed call site applies SiLU with
    // inv_hc = 1.0f, so that is the factor the batched kernel gets.
    const int ffn_blocks = (ffn + threads - 1) / threads;
    {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteSilu, stream);
        q38_moe_silu_scale_mul_experts_kernel<<<dim3(ffn_blocks, k), threads, 0, stream>>>(
            scratch.rx_batch_gate_up(), scratch.rx_batch_act(), ffn, 1.0f);
    }

    const int q8_blocks_per_expert = ffn / 32;
    const int q8_blocks = (q8_blocks_per_expert + (threads / 32) - 1) / (threads / 32);
    {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteQ8Quant, stream);
        q38_moe_q8_0_quantize_experts_kernel<<<dim3(q8_blocks, k), threads, 0, stream>>>(
            scratch.rx_batch_act(), scratch.rx_batch_q8_0_values(),
            scratch.rx_batch_q8_0_scales(), ffn);
    }

    {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteDownGemv, stream);
        launched = launch_iq4_nl_gemv_q8_0_experts_f32(
            device_table->down, k, hidden, ffn, scratch.rx_batch_q8_0_values(),
            scratch.rx_batch_q8_0_scales(), scratch.rx_batch_down(), stream, threads);
    }
    if (!launched) {
        throw std::runtime_error("expert-batched down GEMV needs the DP4A kernels");
    }

    const int hidden_blocks = (hidden + threads - 1) / threads;
    if constexpr (CaptureSelectedOutput) {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteCapture, stream);
        q38_moe_capture_experts_kernel<<<dim3(hidden_blocks, k), threads, 0, stream>>>(
            scratch.rx_batch_down(), selected_output_trace, hidden);
    }
    // "weights" is the group's slice of the layer's top-k weights, so a prefix
    // split accumulates the same per-element rounded sequence in the same order.
    {
        Q38MoeKernelScope launch_scope(kQ38MoeSiteAccum, stream);
        q38_moe_weighted_accum_experts_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.rx_batch_down(), weights, scratch.d_accum(), hidden, k);
    }
}

// Batched twin of q38_moe_run_routed_experts_q8_act_separate(). Returns false
// without launching anything when an expert base is odd (the IQ4_NL DP4A kernel
// needs 16-bit aligned nibble reads); the caller's per-expert loop then
// re-acquires the same slots as cache hits.
template <bool CaptureSelectedOutput, typename XT = __half>
inline bool q38_moe_try_run_routed_experts_q8_act_separate_batched(
        int layer,
        const XT * x,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int k = scratch.k();
    const int groups = q38_moe_batch_groups(k);

    // The routed activation is shared by every expert of the layer, so it is
    // quantised once however many groups the layer runs in.
    {
        Q38MoeKernelScope quant_scope(kQ38MoeSiteRxQuant, stream);
        q38_moe_quantize_routed_x(x, scratch, stream);
    }

    GgmlType gate_up_type = GgmlType::F32;
    int start = 0;
    for (int g = 0; g < groups; ++g) {
        const int count = q38_moe_group_count(k, groups, g);
        Q38MoeExpertPtrTable table{};
        Q38MoeKernelScope acquire_scope(kQ38MoeSiteAcquire, stream);
        for (int j = 0; j < count; ++j) {
            const int expert = scratch.h_ids()[start + j];
            auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);
            if (!q38_moe_gate_up_q8_k_type(w.gate.type) ||
                !q38_moe_gate_up_q8_k_type(w.up.type) ||
                w.down.type != GgmlType::IQ4_NL) {
                throw std::runtime_error("Qwen3.8 routed expert low-bit type changed inside a layer");
            }
            // The batched kernels are dispatched once per group, so the
            // per-expert type checks above become one type check per group.
            if (start + j == 0) {
                gate_up_type = w.gate.type;
            } else if (w.gate.type != gate_up_type || w.up.type != gate_up_type) {
                throw std::runtime_error("Qwen3.8 routed expert types differ inside a layer");
            }
            if ((reinterpret_cast<uintptr_t>(w.down.data) & 1u) != 0) {
                // Abort the layer before any later group can add to a partially
                // written accumulation: the caller falls back to the serial
                // loop, which accumulates onto d_accum.
                (void)cudaMemsetAsync(scratch.d_accum(), 0,
                    static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
                return false;
            }
            table.gate_up[2 * j] = w.gate.data;
            table.gate_up[2 * j + 1] = w.up.data;
            table.down[j] = w.down.data;
        }
        q38_moe_upload_expert_ptr_table(table, scratch, stream, g);
        q38_moe_run_batched_experts<CaptureSelectedOutput>(
            scratch.expert_ptr_table(g), gate_up_type, 2 * count, scratch.expert_ffn(), count,
            selected_output_trace == nullptr
                ? nullptr
                : selected_output_trace + static_cast<size_t>(start) * scratch.hidden(),
            scratch, stream, scratch.d_weights() + start);
        start += count;
    }
    return true;
}

// Batched twin of q38_moe_run_routed_experts_q8_act_fused().
template <bool CaptureSelectedOutput, typename XT = __half>
inline bool q38_moe_try_run_routed_experts_q8_act_fused_batched(
        int layer,
        const XT * x,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        __half * selected_output_trace,
        cudaStream_t stream) {
    const int k = scratch.k();
    const int groups = q38_moe_batch_groups(k);

    {
        Q38MoeKernelScope quant_scope(kQ38MoeSiteRxQuant, stream);
        q38_moe_quantize_routed_x(x, scratch, stream);
    }

    GgmlType gate_up_type = GgmlType::F32;
    int start = 0;
    for (int g = 0; g < groups; ++g) {
        const int count = q38_moe_group_count(k, groups, g);
        Q38MoeExpertPtrTable table{};
        Q38MoeKernelScope acquire_scope(kQ38MoeSiteAcquire, stream);
        for (int j = 0; j < count; ++j) {
            const int expert = scratch.h_ids()[start + j];
            auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);
            if (!q38_moe_gate_up_q8_k_type(w.gate_up.type) || w.down.type != GgmlType::IQ4_NL) {
                throw std::runtime_error("Qwen3.8 fused expert low-bit type changed inside a layer");
            }
            if (start + j == 0) {
                gate_up_type = w.gate_up.type;
            } else if (w.gate_up.type != gate_up_type) {
                throw std::runtime_error("Qwen3.8 fused expert types differ inside a layer");
            }
            if ((reinterpret_cast<uintptr_t>(w.down.data) & 1u) != 0) {
                (void)cudaMemsetAsync(scratch.d_accum(), 0,
                    static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
                return false;
            }
            table.gate_up[j] = w.gate_up.data;
            table.down[j] = w.down.data;
        }
        q38_moe_upload_expert_ptr_table(table, scratch, stream, g);
        q38_moe_run_batched_experts<CaptureSelectedOutput>(
            scratch.expert_ptr_table(g), gate_up_type, count, 2 * scratch.expert_ffn(), count,
            selected_output_trace == nullptr
                ? nullptr
                : selected_output_trace + static_cast<size_t>(start) * scratch.hidden(),
            scratch, stream, scratch.d_weights() + start);
        start += count;
    }
    return true;
}

// MoE runner for one layer. OutT is the routed-output element type: the legacy
// call sites pass a __half * (WARODECODE_MOE_F32_MID then only removes the F16
// roundings *inside* the layer and keeps the single final round), while the F32
// call sites pass a float * (see run_qwen38_routed_moe_fused_f32) and consume
// the unrounded routed sum with the F32 HC combine.
template <bool CaptureSelectedOutput = false, typename OutT = __half, typename XT = __half>
inline void run_qwen38_routed_moe_fused(
        int layer,
        const XT * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        OutT * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    if constexpr (CaptureSelectedOutput) {
        if (!selected_output_trace) throw std::runtime_error("selected expert trace buffer is null");
    }
    if (gate_up_exps.dims.size() != 3 || down_exps.dims.size() != 3) {
        throw std::runtime_error("Qwen3.8 fused expert tensors must be 3D");
    }
    if (static_cast<int>(gate_up_exps.dims[1]) != 2 * scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 fused gate-up FFN width mismatch");
    }

    Q38MoeKernelScope profile_scope(kQ38MoeSiteRoutedTotal, stream);
    if (!preselected) q38_moe_select_experts_input<XT>(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaSuccess;
    {
        Q38MoeKernelScope zero_scope(kQ38MoeSiteAccumZero, stream);
        e = cudaMemsetAsync(
            scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    }
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const bool f32_mid = q38_moe_f32_mid_enabled();
    if (q38_moe_fused_experts_q8_act_compatible(gate_up_exps, down_exps, scratch)) {
        // WARODECODE_MOE_FUSED_GRID: one expert-batched grid per layer instead
        // of seven launches per expert.
        if (q38_moe_fused_grid_candidate(scratch, cache.slot_count(), f32_mid) &&
            q38_moe_try_run_routed_experts_q8_act_fused_batched<CaptureSelectedOutput, XT>(
                layer, x, gate_up_exps, down_exps, cache, scratch,
                selected_output_trace, stream)) {
            q38_finish_moe(scratch, out, stream);
            return;
        }
        q38_moe_run_routed_experts_q8_act_fused<CaptureSelectedOutput, XT>(
            layer, x, gate_up_exps, down_exps, cache, scratch, f32_mid, selected_output_trace, stream);
        q38_finish_moe(scratch, out, stream);
        return;
    }

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;
    // Legacy F16-activation fallback: round the F32 input once, exactly what the
    // half-input runner receives.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);

    Q38MoeKernelScope loop_scope(kQ38MoeSiteLoopTotal, stream);
    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate_up, legacy_x, scratch.d_gate_up(), stream);
        q38_silu_mul_fused_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        if constexpr (CaptureSelectedOutput) {
            e = cudaMemcpyAsync(
                selected_output_trace + static_cast<size_t>(j) * scratch.hidden(),
                scratch.d_down(), static_cast<size_t>(scratch.hidden()) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) throw std::runtime_error(std::string("copy selected fused expert output failed: ") + cudaGetErrorString(e));
        }
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}

// F32-output twin of run_qwen38_routed_moe_fused(): the routed weighted sum is
// written unrounded so it can be combined with the F32 shared expert through
// q38_finish_moe_f32() or the F32 HC combine.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_fused_f32(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_fused<CaptureSelectedOutput, float>(
        layer, x, router, gate_up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

// F32-activation twin of run_qwen38_routed_moe_fused_f32(): the MoE input is
// consumed as F32, so the router activation, the expert Q8_K activation and the
// F32 intermediates never pass through F16. Whenever x is exactly
// F16-representable the result is bit-identical to the half-input version.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_fused_f32_x(
        int layer,
        const float * x,
        const GpuMatrix & router,
        const TensorInfo & gate_up_exps,
        const TensorInfo & down_exps,
        GpuFusedExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_fused<CaptureSelectedOutput, float, float>(
        layer, x, router, gate_up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

template <bool CaptureSelectedOutput = false, typename OutT = __half, typename XT = __half>
inline void run_qwen38_routed_moe_separate(
        int layer,
        const XT * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        OutT * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    if constexpr (CaptureSelectedOutput) {
        if (!selected_output_trace) throw std::runtime_error("selected expert trace buffer is null");
    }
    Q38MoeKernelScope profile_scope(kQ38MoeSiteRoutedTotal, stream);
    if (!preselected) q38_moe_select_experts_input<XT>(router, x, scratch, expert_scale, stream);

    cudaError_t e = cudaSuccess;
    {
        Q38MoeKernelScope zero_scope(kQ38MoeSiteAccumZero, stream);
        e = cudaMemsetAsync(
            scratch.d_accum(), 0, static_cast<size_t>(scratch.hidden()) * sizeof(float), stream);
    }
    if (e != cudaSuccess) throw std::runtime_error(std::string("zero MoE accumulation failed: ") + cudaGetErrorString(e));

    const bool f32_mid = q38_moe_f32_mid_enabled();
    if (q38_moe_separate_experts_q8_act_compatible(gate_exps, up_exps, down_exps, scratch)) {
        // WARODECODE_MOE_FUSED_GRID: one expert-batched grid per layer instead
        // of seven launches per expert.
        if (q38_moe_fused_grid_candidate(scratch, cache.slot_count(), f32_mid) &&
            q38_moe_try_run_routed_experts_q8_act_separate_batched<CaptureSelectedOutput, XT>(
                layer, x, gate_exps, up_exps, down_exps, cache, scratch,
                selected_output_trace, stream)) {
            q38_finish_moe(scratch, out, stream);
            return;
        }
        q38_moe_run_routed_experts_q8_act_separate<CaptureSelectedOutput, XT>(
            layer, x, gate_exps, up_exps, down_exps, cache, scratch, f32_mid,
            selected_output_trace, stream);
        q38_finish_moe(scratch, out, stream);
        return;
    }

    // Legacy F16-activation fallback for weight types the ggml-exact path does
    // not cover. WARODECODE_MOE_F32_MID deliberately leaves it alone: its
    // GEMV chain is F16-activation based, the model's experts are IQ2_S or IQ3_S
    // (or Q8_0 in the unit test), and the F32 runners below are the supported
    // way to get an unrounded MoE branch.
    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;
    // The F32-input runner reaches this fallback too: round its activation once
    // (a no-op for the half-input runner), which is exactly what these
    // F16-activation GEMV kernels would have received.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);

    Q38MoeKernelScope loop_scope(kQ38MoeSiteLoopTotal, stream);
    for (int j = 0; j < scratch.k(); ++j) {
        const int expert = scratch.h_ids()[j];
        const float weight = scratch.h_weights()[j];
        auto w = cache.acquire(layer, expert, gate_exps, up_exps, down_exps, stream);

        launch_lowbit_gemv_half(w.gate, legacy_x, scratch.d_gate(), stream);
        launch_lowbit_gemv_half(w.up, legacy_x, scratch.d_up(), stream);
        q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(w.down, scratch.d_act(), scratch.d_down(), stream);
        if constexpr (CaptureSelectedOutput) {
            const cudaError_t copy_error = cudaMemcpyAsync(
                selected_output_trace + static_cast<size_t>(j) * scratch.hidden(),
                scratch.d_down(), static_cast<size_t>(scratch.hidden()) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream);
            if (copy_error != cudaSuccess) throw std::runtime_error(std::string("copy selected expert output failed: ") + cudaGetErrorString(copy_error));
        }
        q38_weighted_accum_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.d_down(), weight, scratch.d_accum(), scratch.hidden());
    }

    q38_finish_moe(scratch, out, stream);
}

// F32-output twin of run_qwen38_routed_moe_separate(): the routed weighted sum
// is written unrounded so it can be combined with the F32 shared expert through
// q38_finish_moe_f32() or the F32 HC combine.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_separate_f32(
        int layer,
        const __half * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_separate<CaptureSelectedOutput, float>(
        layer, x, router, gate_exps, up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}

// F32-activation twin of run_qwen38_routed_moe_separate_f32(): the routed MoE
// input is consumed as F32 (router activation and expert Q8_K activation stay
// unrounded). Bit-identical to the half-input version whenever x is exactly
// F16-representable.
template <bool CaptureSelectedOutput = false>
inline void run_qwen38_routed_moe_separate_f32_x(
        int layer,
        const float * x,
        const GpuMatrix & router,
        const TensorInfo & gate_exps,
        const TensorInfo & up_exps,
        const TensorInfo & down_exps,
        GpuExpertCache & cache,
        Qwen38MoeScratch & scratch,
        float * out,
        float expert_scale,
        cudaStream_t stream,
        bool preselected = false,
        __half * selected_output_trace = nullptr) {
    run_qwen38_routed_moe_separate<CaptureSelectedOutput, float, float>(
        layer, x, router, gate_exps, up_exps, down_exps, cache, scratch, out,
        expert_scale, stream, preselected, selected_output_trace);
}


// True when the shared-expert gate weight is resident in F32 (the pinned GGUF's
// F32 tensor) and the F32 path is enabled, so the gate can be built the way the
// CPU oracle builds it.
inline bool q38_moe_shexp_gate_f32_ready(
        const Qwen38SharedExpertWeights & shared,
        int hidden) {
    return q38_moe_shexp_gate_f32_enabled() &&
        shared.gate_input_f32.data != nullptr &&
        shared.gate_input_f32.size == static_cast<size_t>(hidden);
}

// Shared-expert gate: sigmoid(build_lora_mm(ffn_gate_inp_shexp, cur)).
//
// With the F32 weight resident the dot is one ggml_vec_dot_f32 in AVX2 order; an
// F16 activation widens exactly into act_f32() first (the oracle's activation is
// F32, so widening only removes the path's own rounding). Otherwise the legacy
// F16-weight dot runs: q38_moe_gate_dot_f32_x() for an F32 activation and
// half_dot_kernel() for an F16 one.
template <typename XT>
inline void q38_moe_shared_gate_dot(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    if (q38_moe_shexp_gate_f32_ready(shared, hidden)) {
        if constexpr (std::is_same_v<XT, float>) {
            q38_moe_gate_dot_f32_avx2(
                shared.gate_input_f32.data, x, scratch.d_shared_gate(), hidden, stream);
        } else {
            constexpr int threads = 256;
            const int blocks = (hidden + threads - 1) / threads;
            q38_hc_copy_half_to_f32_kernel<<<blocks, threads, 0, stream>>>(
                x, scratch.act_f32(), hidden);
            q38_moe_gate_dot_f32_avx2(
                shared.gate_input_f32.data, scratch.act_f32(),
                scratch.d_shared_gate(), hidden, stream);
        }
        return;
    }
    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_gate_dot_f32_x(
            shared.gate_input.data, x, scratch.d_shared_gate(), hidden, stream);
    } else {
        half_dot_kernel<<<1, 256, 0, stream>>>(
            shared.gate_input.data, x, scratch.d_shared_gate(), hidden);
    }
}

inline void q38_moe_check_shared_expert_geometry(
        const Qwen38SharedExpertWeights & shared,
        const Qwen38MoeScratch & scratch) {
    if (shared.gate_input.size != scratch.hidden() ||
        shared.gate.rows != scratch.expert_ffn() || shared.gate.cols != scratch.hidden() ||
        shared.up.rows != scratch.expert_ffn() || shared.up.cols != scratch.hidden() ||
        shared.down.rows != scratch.hidden() || shared.down.cols != scratch.expert_ffn()) {
        throw std::runtime_error("Qwen3.8 shared expert geometry mismatch");
    }
}

// ggml-order shared expert: F32 gate/up projections with a Q8_0-quantised
// activation, ggml SiLU, then the down projection with the activation
// re-quantised from the unrounded F32 gated value. The unrounded down
// projection stays in down_f32(); round_half additionally copies it into the
// F16 d_down() buffer the legacy accumulation kernel consumes.
//
// Activation type: an F16 input is widened into act_f32() first (exact) and
// quantised from there, an F32 input is quantised in place. Everything after the
// first quantisation is identical.
template <typename XT>
inline void q38_moe_run_shared_expert_q8_0_input(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    const int hidden = scratch.hidden();
    const int expert_ffn = scratch.expert_ffn();
    constexpr int threads = 256;
    const int hidden_blocks = (hidden + threads - 1) / threads;
    const int ffn_blocks = (expert_ffn + threads - 1) / threads;

    if constexpr (std::is_same_v<XT, float>) {
        q38_moe_quantize_activation_q8_0(
            x, hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    } else {
        q38_hc_copy_half_to_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
            x, scratch.act_f32(), hidden);
        q38_moe_quantize_activation_q8_0(
            scratch.act_f32(), hidden, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    }
    q38_moe_gemv_q8_0_exact(
        shared.gate, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.gate_f32(), stream);
    q38_moe_gemv_q8_0_exact(
        shared.up, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.up_f32(), stream);

    q38_hc_silu_scale_ggml_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.gate_f32(), expert_ffn, 1.0f);
    q38_silu_mul_f32_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.gate_f32(), scratch.up_f32(), scratch.act_f32(), expert_ffn);

    q38_moe_quantize_activation_q8_0(
        scratch.act_f32(), expert_ffn, scratch.q8_0_values(), scratch.q8_0_scales(), stream);
    q38_moe_gemv_q8_0_exact(
        shared.down, scratch.q8_0_values(), scratch.q8_0_scales(), scratch.down_f32(), stream);
    if (round_half) {
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.down_f32(), scratch.d_down(), hidden);
    }
}

// F16-activation entry point of the chain above (unchanged signature/contract).
inline void run_qwen38_shared_expert_q8_0(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        bool round_half,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_q8_0_input<__half>(x, shared, scratch, round_half, stream);
}

// F32 shared expert (WARODECODE_MOE_F32_MID). The unrounded result stays in
// down_f32() and the sigmoid gate in d_shared_gate(), so the caller can build
// the MoE branch with q38_moe_add_shared_f32() / q38_finish_moe_f32() without
// rounding the shared-expert output to F16.
//
// Activation type: only three steps depend on it -- the gate dot product
// (q38_moe_shared_gate_dot() keeps the oracle's F32 weight and widens an F16
// activation into act_f32() before the AVX2 dot), the first Q8_0 quantisation
// (an F16 input is widened exactly, an F32 input is quantised in place) and the
// legacy F16 fallback below. Everything else is shared so the two entry points
// cannot drift apart.
template <typename XT>
inline void q38_moe_run_shared_expert_f32_input(
        const XT * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    Q38MoeKernelScope shared_scope(kQ38MoeSiteShared, stream);
    q38_moe_check_shared_expert_geometry(shared, scratch);

    {
        Q38MoeKernelScope gate_scope(kQ38MoeSiteSharedGate, stream);
        q38_moe_shared_gate_dot(x, shared, scratch, stream);
        sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(scratch.d_shared_gate());
    }

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    if (q38_moe_shexp_q8_0_enabled() && q38_moe_shexp_q8_0_compatible(shared, scratch)) {
        q38_moe_run_shared_expert_q8_0_input<XT>(x, shared, scratch, false, stream);
        return;
    }

    // Legacy F16-activation chain for weight types the Q8_0 path does not
    // cover; only its final output is lifted into F32 so the F32 contract holds.
    const __half * legacy_x = q38_moe_legacy_half_activation(x, scratch, stream);
    launch_lowbit_gemv_half(shared.gate, legacy_x, scratch.d_gate(), stream);
    launch_lowbit_gemv_half(shared.up, legacy_x, scratch.d_up(), stream);
    q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
        scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
    launch_lowbit_gemv_half(shared.down, scratch.d_act(), scratch.d_down(), stream);
    q38_hc_copy_half_to_f32_kernel<<<hidden_blocks, threads, 0, stream>>>(
        scratch.d_down(), scratch.down_f32(), scratch.hidden());
}

inline void run_qwen38_shared_expert_f32(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_f32_input<__half>(x, shared, scratch, stream);
}

// F32-activation twin of run_qwen38_shared_expert_f32(): the sigmoid gate dot
// product and the Q8_0 activation quantisation read the caller's F32 activation
// directly, so neither the gate weight widening nor the expert activation loses
// precision to F16. Bit-identical to the half-input version whenever x is
// exactly F16-representable.
inline void run_qwen38_shared_expert_f32_x(
        const float * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        cudaStream_t stream) {
    q38_moe_run_shared_expert_f32_input<float>(x, shared, scratch, stream);
}

// Legacy contract: routes the shared expert into the F16 buffer in place. With
// WARODECODE_MOE_F32_MID the expert is computed unrounded and only this final
// F16 store (and the existing half accumulation) rounds; the F32 path is
// run_qwen38_shared_expert_f32() plus q38_moe_add_shared_f32().
inline void run_qwen38_shared_expert(
        const __half * x,
        const Qwen38SharedExpertWeights & shared,
        Qwen38MoeScratch & scratch,
        __half * routed_in_out,
        cudaStream_t stream) {
    q38_moe_check_shared_expert_geometry(shared, scratch);

    const int threads = 256;
    const int ffn_blocks = (scratch.expert_ffn() + threads - 1) / threads;
    const int hidden_blocks = (scratch.hidden() + threads - 1) / threads;

    if (q38_moe_f32_mid_enabled()) {
        run_qwen38_shared_expert_f32(x, shared, scratch, stream);
        q38_float_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
            scratch.down_f32(), scratch.d_down(), scratch.hidden());
        q38_add_shared_kernel<<<hidden_blocks, threads, 0, stream>>>(
            routed_in_out, scratch.d_down(), scratch.d_shared_gate(), scratch.hidden());
        return;
    }

    q38_moe_shared_gate_dot(x, shared, scratch, stream);
    sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(scratch.d_shared_gate());

    if (q38_moe_shexp_q8_0_enabled() && q38_moe_shexp_q8_0_compatible(shared, scratch)) {
        run_qwen38_shared_expert_q8_0(x, shared, scratch, true, stream);
    } else {
        launch_lowbit_gemv_half(shared.gate, x, scratch.d_gate(), stream);
        launch_lowbit_gemv_half(shared.up, x, scratch.d_up(), stream);
        q38_silu_mul_kernel<<<ffn_blocks, threads, 0, stream>>>(
            scratch.d_gate(), scratch.d_up(), scratch.d_act(), scratch.expert_ffn());
        launch_lowbit_gemv_half(shared.down, scratch.d_act(), scratch.d_down(), stream);
    }
    q38_add_shared_kernel<<<hidden_blocks, threads, 0, stream>>>(
        routed_in_out, scratch.d_down(), scratch.d_shared_gate(), scratch.hidden());
}

} // namespace wd
