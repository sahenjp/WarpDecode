#include "gguf.hpp"
#include "gpu_arch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define WD_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
} while (0)

namespace wd {

struct Args {
    std::string model;
    int ctx = 4096;
    int generate = 128;
    int device = 0;
    int start_token = 1;
    bool graph = true;
    bool autotune = true;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i, const char* flag) -> const char* {
        if (++i >= argc) throw std::runtime_error(std::string("値がありません: ") + flag);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--model") a.model = need(i, "--model");
        else if (s == "--ctx") a.ctx = std::stoi(need(i, "--ctx"));
        else if (s == "--generate") a.generate = std::stoi(need(i, "--generate"));
        else if (s == "--device") a.device = std::stoi(need(i, "--device"));
        else if (s == "--start-token") a.start_token = std::stoi(need(i, "--start-token"));
        else if (s == "--no-graph") a.graph = false;
        else if (s == "--no-autotune") a.autotune = false;
        else if (s == "-h" || s == "--help") {
            std::cout << "warpdecode --model model.gguf [--ctx 4096] [--generate 128] [--device 0] [--start-token 1] [--no-graph] [--no-autotune]\n";
            std::exit(0);
        } else throw std::runtime_error("不明な引数: " + s);
    }
    if (a.model.empty()) throw std::runtime_error("--model が必要です");
    if (a.ctx < 16 || a.generate < 1 || a.start_token < 0) throw std::runtime_error("引数範囲が不正です");
    return a;
}

struct DeviceHalfBuffer {
    __half* p = nullptr;
    size_t n = 0;
    DeviceHalfBuffer() = default;
    explicit DeviceHalfBuffer(size_t count) : n(count) { WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(__half))); }
    DeviceHalfBuffer(const DeviceHalfBuffer&) = delete;
    DeviceHalfBuffer& operator=(const DeviceHalfBuffer&) = delete;
    DeviceHalfBuffer(DeviceHalfBuffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DeviceHalfBuffer& operator=(DeviceHalfBuffer&& o) noexcept {
        if (this != &o) { if (p) cudaFree(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    ~DeviceHalfBuffer() { if (p) cudaFree(p); }
};

struct DeviceFloatBuffer {
    float* p = nullptr;
    size_t n = 0;
    DeviceFloatBuffer() = default;
    explicit DeviceFloatBuffer(size_t count) : n(count) { WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float))); }
    DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer(DeviceFloatBuffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    ~DeviceFloatBuffer() { if (p) cudaFree(p); }
};

struct Q4Matrix {
    int rows = 0;
    int cols = 0;
    uint8_t* q = nullptr;
    __half* d = nullptr;
    bool owns = true;

    Q4Matrix() = default;
    Q4Matrix(const Q4Matrix&) = delete;
    Q4Matrix& operator=(const Q4Matrix&) = delete;
    Q4Matrix(Q4Matrix&& o) noexcept : rows(o.rows), cols(o.cols), q(o.q), d(o.d), owns(o.owns) {
        o.q = nullptr; o.d = nullptr; o.rows = o.cols = 0; o.owns = true;
    }
    Q4Matrix& operator=(Q4Matrix&& o) noexcept {
        if (this != &o) {
            reset(); rows = o.rows; cols = o.cols; q = o.q; d = o.d; owns = o.owns;
            o.q = nullptr; o.d = nullptr; o.rows = o.cols = 0; o.owns = true;
        }
        return *this;
    }
    ~Q4Matrix() { reset(); }
    void reset() { if (owns) { if (q) cudaFree(q); if (d) cudaFree(d); } q = nullptr; d = nullptr; owns = true; }
};

static uint16_t load_u16(const uint8_t* p) {
    uint16_t x; std::memcpy(&x, p, sizeof(x)); return x;
}

struct HostPlanarQ4 {
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> d_bits;
};

static HostPlanarQ4 planarize_q4(const TensorInfo& t) {
    if (t.type != GgmlType::Q4_0 || t.dims.size() != 2) throw std::runtime_error("Q4_0 2Dのみ対応: " + t.name);
    const int cols = static_cast<int>(t.dims[0]);
    const int rows = static_cast<int>(t.dims[1]);
    if (cols <= 0 || rows <= 0 || cols % 32 != 0) throw std::runtime_error("Q4_0行形状が不正: " + t.name);
    const int groups = cols / 32;
    HostPlanarQ4 h;
    h.rows = rows; h.cols = cols;
    h.q.resize(static_cast<size_t>(rows) * groups * 16);
    h.d_bits.resize(static_cast<size_t>(rows) * groups);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) {
            const size_t bi = static_cast<size_t>(r) * groups + g;
            const uint8_t* b = t.data + bi * 18;
            h.d_bits[bi] = load_u16(b);
            std::memcpy(h.q.data() + bi * 16, b + 2, 16);
        }
    }
    return h;
}

static HostPlanarQ4 concat_q4(const std::vector<HostPlanarQ4>& xs) {
    if (xs.empty()) throw std::runtime_error("空のQ4連結");
    HostPlanarQ4 out;
    out.cols = xs.front().cols;
    for (const auto& x : xs) {
        if (x.cols != out.cols) throw std::runtime_error("Q4連結の列数が不一致");
        out.rows += x.rows;
        out.q.insert(out.q.end(), x.q.begin(), x.q.end());
        out.d_bits.insert(out.d_bits.end(), x.d_bits.begin(), x.d_bits.end());
    }
    return out;
}

static Q4Matrix upload_q4(const HostPlanarQ4& h) {
    Q4Matrix out;
    out.rows = h.rows; out.cols = h.cols;
    WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&out.q), h.q.size()));
    WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&out.d), h.d_bits.size() * sizeof(uint16_t)));
    WD_CUDA(cudaMemcpy(out.q, h.q.data(), h.q.size(), cudaMemcpyHostToDevice));
    WD_CUDA(cudaMemcpy(out.d, h.d_bits.data(), h.d_bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return out;
}

static std::vector<uint16_t> tensor_to_f16_bits(const TensorInfo& t) {
    std::vector<uint16_t> out(static_cast<size_t>(t.elements()));
    if (t.type == GgmlType::F16) {
        std::memcpy(out.data(), t.data, out.size() * sizeof(uint16_t));
    } else if (t.type == GgmlType::F32) {
        const float* f = reinterpret_cast<const float*>(t.data);
        for (size_t i = 0; i < out.size(); ++i) {
            __half h = __float2half(f[i]);
            std::memcpy(&out[i], &h, sizeof(uint16_t));
        }
    } else throw std::runtime_error("ベクトルはF16/F32のみ対応: " + t.name);
    return out;
}

static DeviceHalfBuffer upload_f16_bits(const std::vector<uint16_t>& bits) {
    DeviceHalfBuffer d(bits.size());
    WD_CUDA(cudaMemcpy(d.p, bits.data(), bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return d;
}

static DeviceHalfBuffer upload_optional_concat_bias(const GgufFile& g, const std::vector<std::string>& names, const std::vector<int>& sizes) {
    size_t total = 0; for (int s : sizes) total += static_cast<size_t>(s);
    std::vector<uint16_t> bits(total);
    bool any = false;
    size_t off = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (const TensorInfo* t = g.optional_tensor(names[i])) {
            auto b = tensor_to_f16_bits(*t);
            if (static_cast<int>(b.size()) != sizes[i]) throw std::runtime_error("biasサイズ不一致: " + names[i]);
            std::copy(b.begin(), b.end(), bits.begin() + static_cast<ptrdiff_t>(off));
            any = true;
        }
        off += static_cast<size_t>(sizes[i]);
    }
    if (!any) return {};
    return upload_f16_bits(bits);
}

struct Layer {
    DeviceHalfBuffer attn_norm;
    Q4Matrix qkv;
    DeviceHalfBuffer qkv_bias;
    DeviceHalfBuffer q_norm;
    DeviceHalfBuffer k_norm;
    Q4Matrix o;
    DeviceHalfBuffer ffn_norm;
    Q4Matrix gate_up;
    Q4Matrix down;
};

struct Model {
    std::string arch;
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int kv_dim = 0;
    int ffn = 0;
    int vocab = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    Q4Matrix token_embd;
    std::vector<Layer> block;
    DeviceHalfBuffer output_norm;
    Q4Matrix output;
};

static std::string key(const std::string& arch, const char* suffix) { return arch + "." + suffix; }

static void expect_vector(const TensorInfo& t, int n, const std::string& what) {
    if (t.dims.size() != 1 || static_cast<int>(t.elements()) != n) {
        throw std::runtime_error(what + " の形状が不一致: " + t.name);
    }
}

static void expect_q4_shape(const HostPlanarQ4& w, int rows, int cols, const std::string& what) {
    if (w.rows != rows || w.cols != cols) {
        throw std::runtime_error(what + " の形状が不一致");
    }
}

static Model load_model(const GgufFile& g) {
    Model m;
    m.arch = g.string_meta("general.architecture");
    if (m.arch != "llama" && m.arch != "qwen2" && m.arch != "qwen3") {
        throw std::runtime_error("現段階では llama/qwen2/qwen3 のdenseモデルのみ: " + m.arch);
    }
    m.hidden = static_cast<int>(g.uint_meta(key(m.arch, "embedding_length")));
    m.layers = static_cast<int>(g.uint_meta(key(m.arch, "block_count")));
    m.heads = static_cast<int>(g.uint_meta(key(m.arch, "attention.head_count")));
    m.kv_heads = static_cast<int>(g.uint_meta(key(m.arch, "attention.head_count_kv")));
    m.ffn = static_cast<int>(g.uint_meta(key(m.arch, "feed_forward_length")));
    m.rms_eps = static_cast<float>(g.number_meta(key(m.arch, "attention.layer_norm_rms_epsilon"), 1e-6));
    m.rope_base = static_cast<float>(g.number_meta(key(m.arch, "rope.freq_base"), 10000.0));
    if (m.hidden <= 0 || m.layers <= 0 || m.heads <= 0 || m.kv_heads <= 0 || m.ffn <= 0 || m.hidden % m.heads != 0) {
        throw std::runtime_error("モデル構成が不正です");
    }
    if (m.heads % m.kv_heads != 0) throw std::runtime_error("GQAのhead比が整数ではありません");
    m.head_dim = m.hidden / m.heads;
    m.kv_dim = m.kv_heads * m.head_dim;
    if (m.head_dim > 256 || (m.head_dim & 1)) throw std::runtime_error("head_dimは偶数かつ256以下のみ対応");
    const int rope_dim = static_cast<int>(g.number_meta(key(m.arch, "rope.dimension_count"), m.head_dim));
    if (rope_dim != m.head_dim) throw std::runtime_error("現段階ではpartial RoPEを未対応として拒否します");
    const std::string rope_scaling = g.string_meta_or(key(m.arch, "rope.scaling.type"), "none");
    if (rope_scaling != "none") {
        throw std::runtime_error("現段階ではRoPE scalingを未対応として拒否します: " + rope_scaling);
    }

    auto emb = planarize_q4(g.tensor("token_embd.weight"));
    if (emb.cols != m.hidden) throw std::runtime_error("token_embdのhiddenが不一致");
    m.vocab = emb.rows;
    m.token_embd = upload_q4(emb);

    m.block.reserve(m.layers);
    for (int i = 0; i < m.layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        Layer l;
        const TensorInfo& attn_norm_t = g.tensor(p + "attn_norm.weight");
        expect_vector(attn_norm_t, m.hidden, "attn_norm");
        l.attn_norm = upload_f16_bits(tensor_to_f16_bits(attn_norm_t));
        auto q = planarize_q4(g.tensor(p + "attn_q.weight"));
        auto k = planarize_q4(g.tensor(p + "attn_k.weight"));
        auto v = planarize_q4(g.tensor(p + "attn_v.weight"));
        expect_q4_shape(q, m.hidden, m.hidden, "Q");
        expect_q4_shape(k, m.kv_dim, m.hidden, "K");
        expect_q4_shape(v, m.kv_dim, m.hidden, "V");
        l.qkv = upload_q4(concat_q4({std::move(q), std::move(k), std::move(v)}));
        l.qkv_bias = upload_optional_concat_bias(g,
            {p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias"},
            {m.hidden, m.kv_dim, m.kv_dim});
        if (const TensorInfo* t = g.optional_tensor(p + "attn_q_norm.weight")) { expect_vector(*t, m.head_dim, "q_norm"); l.q_norm = upload_f16_bits(tensor_to_f16_bits(*t)); }
        if (const TensorInfo* t = g.optional_tensor(p + "attn_k_norm.weight")) { expect_vector(*t, m.head_dim, "k_norm"); l.k_norm = upload_f16_bits(tensor_to_f16_bits(*t)); }
        auto o = planarize_q4(g.tensor(p + "attn_output.weight"));
        expect_q4_shape(o, m.hidden, m.hidden, "attention output");
        l.o = upload_q4(o);
        const TensorInfo& ffn_norm_t = g.tensor(p + "ffn_norm.weight");
        expect_vector(ffn_norm_t, m.hidden, "ffn_norm");
        l.ffn_norm = upload_f16_bits(tensor_to_f16_bits(ffn_norm_t));
        auto gate = planarize_q4(g.tensor(p + "ffn_gate.weight"));
        auto up = planarize_q4(g.tensor(p + "ffn_up.weight"));
        expect_q4_shape(gate, m.ffn, m.hidden, "FFN gate");
        expect_q4_shape(up, m.ffn, m.hidden, "FFN up");
        l.gate_up = upload_q4(concat_q4({std::move(gate), std::move(up)}));
        auto down = planarize_q4(g.tensor(p + "ffn_down.weight"));
        expect_q4_shape(down, m.hidden, m.ffn, "FFN down");
        l.down = upload_q4(down);
        m.block.push_back(std::move(l));
    }

    const TensorInfo& output_norm_t = g.tensor("output_norm.weight");
    expect_vector(output_norm_t, m.hidden, "output_norm");
    m.output_norm = upload_f16_bits(tensor_to_f16_bits(output_norm_t));
    if (const TensorInfo* out = g.optional_tensor("output.weight")) {
        m.output = upload_q4(planarize_q4(*out));
    } else {
        // tied embeddingはGPU上でも共有し、同じ巨大行列を二重確保しない。
        m.output.rows = m.token_embd.rows;
        m.output.cols = m.token_embd.cols;
        m.output.q = m.token_embd.q;
        m.output.d = m.token_embd.d;
        m.output.owns = false;
    }
    if (m.output.rows != m.vocab || m.output.cols != m.hidden) throw std::runtime_error("output形状が不一致");
    return m;
}

__device__ __forceinline__ float half_to_float(__half x) { return __half2float(x); }

__global__ void embedding_q4_kernel(const uint8_t* q, const __half* d, int cols, const int* token, __half* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const int t = *token;
    const int groups = cols / 32;
    const int g = c >> 5;
    const int lane = c & 31;
    const size_t bi = static_cast<size_t>(t) * groups + g;
    const uint8_t packed = q[bi * 16 + (lane & 15)];
    const int qi = ((lane < 16) ? (packed & 0x0f) : (packed >> 4)) - 8;
    out[c] = __float2half(static_cast<float>(qi) * half_to_float(d[bi]));
}

__global__ void rmsnorm_kernel(const __half* x, const __half* w, __half* y, int n, float eps) {
    __shared__ float smem[256];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = half_to_float(x[i]);
        s = fmaf(v, v, s);
    }
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = __float2half(half_to_float(x[i]) * inv * half_to_float(w[i]));
    }
}

__device__ __forceinline__ float halfwarp_sum(float v, unsigned mask) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off, 16);
    return v;
}

// 1 warp = 2 output rows. Each half-warp owns one row and consumes both nibbles
// from a Q4 byte, so no byte broadcast is needed. Weights remain compressed until registers.
template <bool OUT_F32>
__global__ void q4_gemv_2row_kernel(const uint8_t* __restrict__ q,
                                    const __half* __restrict__ d,
                                    const __half* __restrict__ x,
                                    const __half* __restrict__ bias,
                                    void* __restrict__ y,
                                    int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;
    const int slane = lane & 15;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int row = warp_global * 2 + half;
    if (row >= rows) return;
    const unsigned mask = half ? 0xffff0000u : 0x0000ffffu;
    const int groups = cols / 32;
    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const size_t bi = static_cast<size_t>(row) * groups + g;
        const uint8_t packed = q[bi * 16 + slane];
        const int q0 = static_cast<int>(packed & 0x0f) - 8;
        const int q1 = static_cast<int>(packed >> 4) - 8;
        const float scale = half_to_float(d[bi]);
        const int c = g * 32 + slane;
        acc = fmaf(scale * static_cast<float>(q0), half_to_float(x[c]), acc);
        acc = fmaf(scale * static_cast<float>(q1), half_to_float(x[c + 16]), acc);
    }
    acc = halfwarp_sum(acc, mask);
    if (slane == 0) {
        if (bias) acc += half_to_float(bias[row]);
        if constexpr (OUT_F32) static_cast<float*>(y)[row] = acc;
        else static_cast<__half*>(y)[row] = __float2half(acc);
    }
}

static int g_q4_threads_half = 256;
static int g_q4_threads_float = 256;

static inline void launch_q4_half(const Q4Matrix& w, const __half* x, __half* y, const __half* bias, cudaStream_t s) {
    const int threads = g_q4_threads_half;
    const int rows_per_block = (threads / 32) * 2;
    const int blocks = (w.rows + rows_per_block - 1) / rows_per_block;
    q4_gemv_2row_kernel<false><<<blocks, threads, 0, s>>>(w.q, w.d, x, bias, y, w.rows, w.cols);
}

static inline void launch_q4_float(const Q4Matrix& w, const __half* x, float* y, cudaStream_t s) {
    const int threads = g_q4_threads_float;
    const int rows_per_block = (threads / 32) * 2;
    const int blocks = (w.rows + rows_per_block - 1) / rows_per_block;
    q4_gemv_2row_kernel<true><<<blocks, threads, 0, s>>>(w.q, w.d, x, nullptr, y, w.rows, w.cols);
}

static float time_q4_half(const Q4Matrix& w, const __half* x, __half* y, int threads, cudaStream_t s) {
    const int old = g_q4_threads_half;
    g_q4_threads_half = threads;
    for (int i = 0; i < 3; ++i) launch_q4_half(w, x, y, nullptr, s);
    WD_CUDA(cudaStreamSynchronize(s));
    cudaEvent_t begin{}, end{};
    WD_CUDA(cudaEventCreate(&begin)); WD_CUDA(cudaEventCreate(&end));
    WD_CUDA(cudaEventRecord(begin, s));
    for (int i = 0; i < 12; ++i) launch_q4_half(w, x, y, nullptr, s);
    WD_CUDA(cudaEventRecord(end, s)); WD_CUDA(cudaEventSynchronize(end));
    float ms = 0.0f; WD_CUDA(cudaEventElapsedTime(&ms, begin, end));
    cudaEventDestroy(begin); cudaEventDestroy(end);
    g_q4_threads_half = old;
    return ms / 12.0f;
}

static float time_q4_float(const Q4Matrix& w, const __half* x, float* y, int threads, cudaStream_t s) {
    const int old = g_q4_threads_float;
    g_q4_threads_float = threads;
    for (int i = 0; i < 3; ++i) launch_q4_float(w, x, y, s);
    WD_CUDA(cudaStreamSynchronize(s));
    cudaEvent_t begin{}, end{};
    WD_CUDA(cudaEventCreate(&begin)); WD_CUDA(cudaEventCreate(&end));
    WD_CUDA(cudaEventRecord(begin, s));
    for (int i = 0; i < 8; ++i) launch_q4_float(w, x, y, s);
    WD_CUDA(cudaEventRecord(end, s)); WD_CUDA(cudaEventSynchronize(end));
    float ms = 0.0f; WD_CUDA(cudaEventElapsedTime(&ms, begin, end));
    cudaEventDestroy(begin); cudaEventDestroy(end);
    g_q4_threads_float = old;
    return ms / 8.0f;
}

static void autotune_q4(const Model& m) {
    if (m.block.empty()) return;
    cudaStream_t s{}; WD_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    DeviceHalfBuffer x(static_cast<size_t>(m.hidden));
    DeviceHalfBuffer y_half(static_cast<size_t>(std::max(2 * m.ffn, m.hidden)));
    DeviceFloatBuffer y_float(static_cast<size_t>(m.vocab));
    WD_CUDA(cudaMemsetAsync(x.p, 0, x.n * sizeof(__half), s));
    const int candidates[] = {64, 128, 256, 512};
    float best_half = std::numeric_limits<float>::infinity();
    float best_float = std::numeric_limits<float>::infinity();
    int best_half_threads = 256, best_float_threads = 256;
    for (int t : candidates) {
        const float mh = time_q4_half(m.block.front().gate_up, x.p, y_half.p, t, s);
        if (mh < best_half) { best_half = mh; best_half_threads = t; }
        const float mf = time_q4_float(m.output, x.p, y_float.p, t, s);
        if (mf < best_float) { best_float = mf; best_float_threads = t; }
    }
    g_q4_threads_half = best_half_threads;
    g_q4_threads_float = best_float_threads;
    WD_CUDA(cudaStreamDestroy(s));
    std::cerr << "autotune: q4-half=" << best_half_threads << " threads (" << best_half
              << " ms), logits=" << best_float_threads << " threads (" << best_float << " ms)\n";
}

__global__ void qk_norm_kernel(__half* qkv, const __half* q_w, const __half* k_w,
                               int heads, int kv_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    const bool is_q = h < heads;
    if ((!is_q && h >= heads + kv_heads) || threadIdx.x >= 256) return;
    __half* base = is_q ? qkv + h * head_dim : qkv + heads * head_dim + (h - heads) * head_dim;
    const __half* w = is_q ? q_w : k_w;
    if (!w) return;
    __shared__ float smem[256];
    float s = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = half_to_float(base[d]); s = fmaf(v, v, s);
    }
    smem[threadIdx.x] = s; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) { if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride]; __syncthreads(); }
    const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) base[d] = __float2half(half_to_float(base[d]) * inv * half_to_float(w[d]));
}

__global__ void rope_store_kernel(__half* qkv, __half* kcache, __half* vcache,
                                  const int* pos_ptr, int heads, int kv_heads,
                                  int head_dim, float base) {
    // Llama/Qwen系はGPT-NeoX型のRoPE配置: [cos側..., sin側...]。
    // pair p は p と p + head_dim/2 を回転させる。
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_pairs = heads * half_dim;
    const int kv_pairs = kv_heads * half_dim;
    const int total = q_pairs + kv_pairs;
    if (pair >= total) return;
    const int pos = *pos_ptr;
    if (pair < q_pairs) {
        const int h = pair / half_dim;
        const int p = pair % half_dim;
        const int i0 = h * head_dim + p;
        const int i1 = i0 + half_dim;
        const float ang = static_cast<float>(pos) * powf(base, -2.0f * p / static_cast<float>(head_dim));
        float sn, cs; sincosf(ang, &sn, &cs);
        const float a = half_to_float(qkv[i0]);
        const float b = half_to_float(qkv[i1]);
        qkv[i0] = __float2half(a * cs - b * sn);
        qkv[i1] = __float2half(a * sn + b * cs);
    } else {
        const int kp = pair - q_pairs;
        const int h = kp / half_dim;
        const int p = kp % half_dim;
        const int kbase = heads * head_dim;
        const int vbase = kbase + kv_heads * head_dim;
        const int i0 = kbase + h * head_dim + p;
        const int i1 = i0 + half_dim;
        const float ang = static_cast<float>(pos) * powf(base, -2.0f * p / static_cast<float>(head_dim));
        float sn, cs; sincosf(ang, &sn, &cs);
        const float a = half_to_float(qkv[i0]);
        const float b = half_to_float(qkv[i1]);
        const __half k0 = __float2half(a * cs - b * sn);
        const __half k1 = __float2half(a * sn + b * cs);
        qkv[i0] = k0; qkv[i1] = k1;
        const size_t koff = (static_cast<size_t>(pos) * kv_heads + h) * head_dim;
        kcache[koff + p] = k0;
        kcache[koff + p + half_dim] = k1;
        vcache[koff + p] = qkv[vbase + h * head_dim + p];
        vcache[koff + p + half_dim] = qkv[vbase + h * head_dim + p + half_dim];
    }
}

// Decode-only online softmax. No score buffer is materialized.
__global__ void attention_decode_kernel(const __half* q,
                                        const __half* kcache,
                                        const __half* vcache,
                                        __half* out,
                                        const int* pos_ptr,
                                        int heads, int kv_heads, int head_dim) {
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int d = threadIdx.x;
    __shared__ float red[256];
    __shared__ float sh_alpha, sh_beta, sh_l;
    const int pos = *pos_ptr;
    const int kvh = (h * kv_heads) / heads;
    const float qd = d < head_dim ? half_to_float(q[h * head_dim + d]) : 0.0f;
    float acc = 0.0f;
    float m = -CUDART_INF_F;
    float l = 0.0f;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    for (int t = 0; t <= pos; ++t) {
        const size_t base = (static_cast<size_t>(t) * kv_heads + kvh) * head_dim;
        float dot = d < head_dim ? qd * half_to_float(kcache[base + d]) : 0.0f;
        red[d] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = red[0] * inv_sqrt;
            const float nm = fmaxf(m, score);
            const float alpha = isfinite(m) ? expf(m - nm) : 0.0f;
            const float beta = expf(score - nm);
            l = l * alpha + beta;
            m = nm;
            sh_alpha = alpha;
            sh_beta = beta;
            sh_l = l;
        }
        __syncthreads();
        if (d < head_dim) acc = acc * sh_alpha + sh_beta * half_to_float(vcache[base + d]);
        __syncthreads();
    }
    if (d < head_dim) out[h * head_dim + d] = __float2half(acc / sh_l);
}

__global__ void residual_add_kernel(__half* x, const __half* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = __float2half(half_to_float(x[i]) + half_to_float(y[i]));
    }
}

__global__ void swiglu_kernel(const __half* gate_up, __half* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = half_to_float(gate_up[i]);
        const float u = half_to_float(gate_up[n + i]);
        const float silu = g / (1.0f + expf(-g));
        out[i] = __float2half(silu * u);
    }
}

struct MaxPair { float v; int i; };

__global__ void argmax_kernel(const float* logits, int n, int* token, int* out_tokens, const int* pos_ptr) {
    __shared__ MaxPair sm[256];
    MaxPair best{-CUDART_INF_F, 0};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > best.v || (v == best.v && i < best.i)) best = {v, i};
    }
    sm[threadIdx.x] = best; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            const MaxPair o = sm[threadIdx.x + stride];
            if (o.v > sm[threadIdx.x].v || (o.v == sm[threadIdx.x].v && o.i < sm[threadIdx.x].i)) sm[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *token = sm[0].i;
        out_tokens[*pos_ptr] = sm[0].i;
    }
}

__global__ void increment_pos_kernel(int* pos) { if (threadIdx.x == 0) ++(*pos); }

struct Runtime {
    Model& m;
    int max_ctx;
    cudaStream_t stream{};
    DeviceHalfBuffer hidden, norm, qkv, attn, proj, gate_up, mlp;
    DeviceFloatBuffer logits;
    DeviceHalfBuffer kcache, vcache;
    int* d_token = nullptr;
    int* d_pos = nullptr;
    int* d_out_tokens = nullptr;
    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};

    Runtime(Model& model, int ctx) : m(model), max_ctx(ctx),
        hidden(model.hidden), norm(model.hidden),
        qkv(model.hidden + 2 * model.kv_dim), attn(model.hidden), proj(model.hidden),
        gate_up(static_cast<size_t>(2) * model.ffn), mlp(model.ffn), logits(model.vocab),
        kcache(static_cast<size_t>(model.layers) * ctx * model.kv_dim),
        vcache(static_cast<size_t>(model.layers) * ctx * model.kv_dim) {
        WD_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_token), sizeof(int)));
        WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pos), sizeof(int)));
        WD_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out_tokens), static_cast<size_t>(ctx) * sizeof(int)));
        WD_CUDA(cudaMemsetAsync(kcache.p, 0, kcache.n * sizeof(__half), stream));
        WD_CUDA(cudaMemsetAsync(vcache.p, 0, vcache.n * sizeof(__half), stream));
        WD_CUDA(cudaMemsetAsync(d_out_tokens, 0xff, static_cast<size_t>(ctx) * sizeof(int), stream));
        WD_CUDA(cudaStreamSynchronize(stream));
    }

    ~Runtime() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (d_token) cudaFree(d_token);
        if (d_pos) cudaFree(d_pos);
        if (d_out_tokens) cudaFree(d_out_tokens);
        if (stream) cudaStreamDestroy(stream);
    }

    void emit_step() {
        const int emb_blocks = (m.hidden + 255) / 256;
        embedding_q4_kernel<<<emb_blocks, 256, 0, stream>>>(m.token_embd.q, m.token_embd.d, m.hidden, d_token, hidden.p);

        for (int li = 0; li < m.layers; ++li) {
            Layer& l = m.block[li];
            rmsnorm_kernel<<<1, 256, 0, stream>>>(hidden.p, l.attn_norm.p, norm.p, m.hidden, m.rms_eps);
            launch_q4_half(l.qkv, norm.p, qkv.p, l.qkv_bias.p, stream);
            if (l.q_norm.p || l.k_norm.p) {
                qk_norm_kernel<<<m.heads + m.kv_heads, 256, 0, stream>>>(qkv.p, l.q_norm.p, l.k_norm.p, m.heads, m.kv_heads, m.head_dim, m.rms_eps);
            }
            const int total_pairs = (m.heads + m.kv_heads) * (m.head_dim / 2);
            __half* kc = kcache.p + static_cast<size_t>(li) * max_ctx * m.kv_dim;
            __half* vc = vcache.p + static_cast<size_t>(li) * max_ctx * m.kv_dim;
            rope_store_kernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(qkv.p, kc, vc, d_pos, m.heads, m.kv_heads, m.head_dim, m.rope_base);
            int attn_threads = 1;
            while (attn_threads < m.head_dim) attn_threads <<= 1;
            attn_threads = std::max(32, std::min(256, attn_threads));
            attention_decode_kernel<<<m.heads, attn_threads, 0, stream>>>(qkv.p, kc, vc, attn.p, d_pos, m.heads, m.kv_heads, m.head_dim);
            launch_q4_half(l.o, attn.p, proj.p, nullptr, stream);
            residual_add_kernel<<<std::min(64, (m.hidden + 255) / 256), 256, 0, stream>>>(hidden.p, proj.p, m.hidden);

            rmsnorm_kernel<<<1, 256, 0, stream>>>(hidden.p, l.ffn_norm.p, norm.p, m.hidden, m.rms_eps);
            launch_q4_half(l.gate_up, norm.p, gate_up.p, nullptr, stream);
            swiglu_kernel<<<std::min(128, (m.ffn + 255) / 256), 256, 0, stream>>>(gate_up.p, mlp.p, m.ffn);
            launch_q4_half(l.down, mlp.p, proj.p, nullptr, stream);
            residual_add_kernel<<<std::min(64, (m.hidden + 255) / 256), 256, 0, stream>>>(hidden.p, proj.p, m.hidden);
        }
        rmsnorm_kernel<<<1, 256, 0, stream>>>(hidden.p, m.output_norm.p, norm.p, m.hidden, m.rms_eps);
        launch_q4_float(m.output, norm.p, logits.p, stream);
        argmax_kernel<<<1, 256, 0, stream>>>(logits.p, m.vocab, d_token, d_out_tokens, d_pos);
        increment_pos_kernel<<<1, 1, 0, stream>>>(d_pos);
    }

    void set_state(int token, int pos) {
        WD_CUDA(cudaMemcpyAsync(d_token, &token, sizeof(int), cudaMemcpyHostToDevice, stream));
        WD_CUDA(cudaMemcpyAsync(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, stream));
        WD_CUDA(cudaStreamSynchronize(stream));
    }

    void capture() {
        if (graph_exec) return;
        WD_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        emit_step();
        WD_CUDA(cudaStreamEndCapture(stream, &graph));
        WD_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    }

    double run(int steps, bool use_graph, int start_token) {
        cudaEvent_t a{}, b{};
        WD_CUDA(cudaEventCreate(&a)); WD_CUDA(cudaEventCreate(&b));
        if (use_graph) capture();
        // warmup: state is reset afterwards, so cache writes from warmup are ignored.
        set_state(start_token, 0);
        if (use_graph) WD_CUDA(cudaGraphLaunch(graph_exec, stream)); else emit_step();
        WD_CUDA(cudaStreamSynchronize(stream));
        set_state(start_token, 0);
        WD_CUDA(cudaEventRecord(a, stream));
        for (int i = 0; i < steps; ++i) {
            if (use_graph) WD_CUDA(cudaGraphLaunch(graph_exec, stream)); else emit_step();
        }
        WD_CUDA(cudaEventRecord(b, stream));
        WD_CUDA(cudaEventSynchronize(b));
        float ms = 0.0f; WD_CUDA(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a); cudaEventDestroy(b);
        return static_cast<double>(ms) / 1000.0;
    }

    uint64_t output_hash(int steps) {
        std::vector<int> ids(static_cast<size_t>(steps));
        WD_CUDA(cudaMemcpy(ids.data(), d_out_tokens, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
        uint64_t h = 1469598103934665603ull;
        for (int id : ids) {
            const uint32_t x = static_cast<uint32_t>(id);
            for (int b = 0; b < 4; ++b) { h ^= static_cast<uint8_t>(x >> (8 * b)); h *= 1099511628211ull; }
        }
        return h;
    }
};

} // namespace wd

int main(int argc, char** argv) {
    try {
        wd::Args args = wd::parse_args(argc, argv);
        int count = 0; WD_CUDA(cudaGetDeviceCount(&count));
        if (args.device < 0 || args.device >= count) throw std::runtime_error("指定GPUがありません");
        WD_CUDA(cudaSetDevice(args.device));
        cudaDeviceProp prop{}; WD_CUDA(cudaGetDeviceProperties(&prop, args.device));
        const int cc = prop.major * 10 + prop.minor;
        if (cc < 50) throw std::runtime_error("warpdecodeはSM50以上のNVIDIA GPUを対象にします");
        if (args.ctx <= args.generate + 2) throw std::runtime_error("--ctx は --generate より十分大きくしてください");

        const auto kernel_plan = wd::kernel_plan_for_sm(cc);
        std::cerr << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << " family=" << kernel_plan.family << "\n";
        std::cerr << "target kernel plan: decode=" << kernel_plan.decode << " verify=" << kernel_plan.verify << "\n";
        std::cerr << "GGUF読込: " << args.model << "\n";
        wd::GgufFile gguf(args.model);
        wd::Model model = wd::load_model(gguf);
        if (args.autotune) wd::autotune_q4(model);
        std::cerr << "arch=" << model.arch << " layers=" << model.layers << " hidden=" << model.hidden
                  << " heads=" << model.heads << "/" << model.kv_heads << " ffn=" << model.ffn
                  << " vocab=" << model.vocab << "\n";
        std::cerr << "制約: Q4_0 dense / batch=1 / greedy / FP16 KV / CUDA only\n";

        if (args.start_token >= model.vocab) throw std::runtime_error("--start-token が語彙数以上です");
        wd::Runtime rt(model, args.ctx);
        const double sec = rt.run(args.generate, args.graph, args.start_token);
        const double tps = args.generate / sec;
        const uint64_t token_hash = rt.output_hash(args.generate);
        std::cout << "{\"engine\":\"warpdecode-v2\",\"gpu\":\"" << prop.name
                  << "\",\"arch\":\"" << model.arch << "\",\"tokens\":" << args.generate
                  << ",\"seconds\":" << std::fixed << std::setprecision(6) << sec
                  << ",\"tok_s\":" << std::setprecision(3) << tps
                  << ",\"token_hash\":\"0x" << std::hex << token_hash << std::dec << "\""
                  << ",\"cuda_graph\":" << (args.graph ? "true" : "false")
                  << ",\"autotune\":" << (args.autotune ? "true" : "false")
                  << ",\"q4_threads\":" << wd::g_q4_threads_half
                  << ",\"kernel_family\":\"" << kernel_plan.family << "\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
