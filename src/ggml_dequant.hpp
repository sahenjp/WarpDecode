#pragma once
// ホスト側の GGUF テンソル → F16 展開。
//
// dense パスは元々 Q4_0 しか読めなかったが、llama.cpp が配る Q4_0 の GGUF でも
// output.weight は Q6_K、Llama-3.2-1B の ffn_down は Q8_0 ということが普通にあり、
// そのままだと「Q4_0 以外は未対応」でどのモデルも起動できなかった。
// ここで Q8_0 / Q6_K / F16 / F32 を F16 に落として F16 GEMV に渡す。
// 量子化のままより重い (Q6_K の 2.4 倍) が、再量子化せず値も変えない。
#include "gguf.hpp"

// ホスト側の検査では CUDA が無いので、__half / __float2half は呼び出し側が用意する
// (tests/gemv_kquant_dequant_host_test.cpp が最小の shim を置く)。
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace wd {

inline float ggml_half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
    const uint32_t exp = static_cast<uint32_t>(h >> 10) & 0x1fu;
    const uint32_t man = static_cast<uint32_t>(h) & 0x3ffu;
    float v = 0.0f;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
    } else if (exp != 31) {
        v = std::ldexp(static_cast<float>(man + 1024u), static_cast<int>(exp) - 25);
    }
    return sign ? -v : v;
}

inline uint16_t ggml_load_u16(const uint8_t * p) {
    uint16_t x = 0;
    std::memcpy(&x, p, sizeof(x));
    return x;
}

// dense パスが F16 経由で扱える型か。ブロック量子化の型は列数がブロック長の
// 倍数である必要がある (満たさない場合は呼び出し側が明示的に失敗させる)。
inline bool ggml_type_requantizable(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::Q8_0:
        case GgmlType::Q6_K:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
            return true;
        default:
            return false;
    }
}

inline int ggml_type_block_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 1;
        case GgmlType::F16: return 1;
        case GgmlType::Q8_0: return 32;
        case GgmlType::Q6_K: return 256;
        case GgmlType::Q4_1: return 32;
        case GgmlType::Q5_0: return 32;
        case GgmlType::Q5_1: return 32;
        case GgmlType::Q2_K: return 256;
        case GgmlType::Q3_K: return 256;
        case GgmlType::Q4_K: return 256;
        case GgmlType::Q5_K: return 256;
        default: return 0;
    }
}

inline int ggml_type_block_bytes(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q6_K: return 210;
        case GgmlType::Q4_1: return 20;
        case GgmlType::Q5_0: return 22;
        case GgmlType::Q5_1: return 24;
        case GgmlType::Q2_K: return 84;
        case GgmlType::Q3_K: return 110;
        case GgmlType::Q4_K: return 144;
        case GgmlType::Q5_K: return 176;
        default: return 0;
    }
}

// Q6_K の 1 値。lowbit.cuh の q6_k_value() と同じ式 (ggml 検証済み) をホストへ写した。
inline float ggml_q6_k_value(const uint8_t * blk, int i) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = ggml_half_to_float(ggml_load_u16(blk + 208));

    const int half = i >> 7;
    const int r = i & 127;
    const int l = r & 31;
    const int q = r >> 5;

    const uint8_t * ql64 = ql + half * 64;
    const uint8_t * qh32 = qh + half * 32;
    const int8_t * sc8 = sc + half * 8;

    int v = 0;
    int scale_index = 0;
    if (q == 0) {
        v = (ql64[l] & 0x0f) | (((qh32[l] >> 0) & 3) << 4);
        scale_index = (l >> 4) + 0;
    } else if (q == 1) {
        v = (ql64[l + 32] & 0x0f) | (((qh32[l] >> 2) & 3) << 4);
        scale_index = (l >> 4) + 2;
    } else if (q == 2) {
        v = (ql64[l] >> 4) | (((qh32[l] >> 4) & 3) << 4);
        scale_index = (l >> 4) + 4;
    } else {
        v = (ql64[l + 32] >> 4) | (((qh32[l] >> 6) & 3) << 4);
        scale_index = (l >> 4) + 6;
    }
    return d * static_cast<float>(sc8[scale_index]) * static_cast<float>(v - 32);
}

// 1 次元分 (n 要素) を F16 のビット列へ。
inline void ggml_dequant_to_f16(GgmlType type, const uint8_t * src, uint16_t * dst, int64_t n) {
    switch (type) {
        case GgmlType::F16:
            std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(uint16_t));
            return;
        case GgmlType::F32: {
            const float * f = reinterpret_cast<const float *>(src);
            for (int64_t i = 0; i < n; ++i) {
                const __half h = __float2half(f[i]);
                std::memcpy(&dst[i], &h, sizeof(uint16_t));
            }
            return;
        }
        case GgmlType::Q8_0: {
            for (int64_t i = 0; i < n; ++i) {
                const int64_t block = i >> 5;
                const uint8_t * b = src + block * 34;
                const float d = ggml_half_to_float(ggml_load_u16(b));
                const int8_t q = static_cast<int8_t>(b[2 + (i & 31)]);
                const __half h = __float2half(d * static_cast<float>(q));
                std::memcpy(&dst[i], &h, sizeof(uint16_t));
            }
            return;
        }
        case GgmlType::Q6_K: {
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t * blk = src + (i >> 8) * 210;
                const __half h = __float2half(ggml_q6_k_value(blk, static_cast<int>(i & 255)));
                std::memcpy(&dst[i], &h, sizeof(uint16_t));
            }
            return;
        }
        default:
            throw std::runtime_error(std::string("F16 へ展開できない型です: ") + ggml_type_name(type));
    }
}

// ---------------------------------------------------------------------------
// Q8_0 への再量子化
//
// dense パスは Q4_0 の planar カーネルしか持たないが、llama.cpp が配る GGUF は
// output.weight が Q6_K / Q8_0、Llama-3.2-1B は token_embd が Q6_K で
// ffn_down が Q4_1、というように混ざっている。Q4_0 以外の 2D テンソルは
// 一度 F32 に戻してから **Q8_0** に詰め直し、lowbit.cuh の実績ある
// q8_0_gemv_kernel / q8_0_gather_row_kernel で走らせる。
// Q8_0 は 8bit なので元が Q6_K でも精度は落ちない (バイト数は 1.3 倍)。
// ---------------------------------------------------------------------------

inline float ggml_q8_0_value(const uint8_t * blk, int i) {
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    return d * static_cast<float>(static_cast<int8_t>(blk[2 + (i & 31)]));
}

inline float ggml_q4_0_value(const uint8_t * blk, int i) {
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    const uint8_t byte = blk[2 + (i & 15)];
    const int q = (i < 16) ? (byte & 0x0f) : (byte >> 4);
    return d * static_cast<float>(q - 8);
}

inline float ggml_q4_1_value(const uint8_t * blk, int i) {
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    const float m = ggml_half_to_float(ggml_load_u16(blk + 2));
    const uint8_t byte = blk[4 + (i & 15)];
    const int q = (i < 16) ? (byte & 0x0f) : (byte >> 4);
    return d * static_cast<float>(q) + m;
}

// block_q5_0: d + qh[4] + qs[16] (22 バイト)。5bit 目は qh のビット。
inline float ggml_q5_0_value(const uint8_t * blk, int i) {
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    uint32_t qh = 0;
    std::memcpy(&qh, blk + 2, 4);
    const uint8_t * qs = blk + 6;
    const int j = i & 15;
    const bool high = ((i >> 4) & 1) != 0;
    const uint8_t xh = high
        ? static_cast<uint8_t>((qh >> (j + 12)) & 0x10u)
        : static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
    const int x = ((high ? (qs[j] >> 4) : (qs[j] & 0x0F)) | xh) - 16;
    return static_cast<float>(x) * d;
}

// block_q5_1: d + m + qh[4] + qs[16] (24 バイト)。
inline float ggml_q5_1_value(const uint8_t * blk, int i) {
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    const float m = ggml_half_to_float(ggml_load_u16(blk + 2));
    uint32_t qh = 0;
    std::memcpy(&qh, blk + 4, 4);
    const uint8_t * qs = blk + 8;
    const int j = i & 15;
    const bool high = ((i >> 4) & 1) != 0;
    const uint8_t xh = high
        ? static_cast<uint8_t>((qh >> (j + 12)) & 0x10u)
        : static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
    const int x = (high ? (qs[j] >> 4) : (qs[j] & 0x0F)) | xh;
    return static_cast<float>(x) * d + m;
}

// ---- k-quant (llama.cpp の ggml-quants.c の dequantize_row_* をホストへ写した) ----
// すべて 256 値スーパーブロック。1 値を得るため、ブロック内位置から必要な
// スケール/最小値/ビットを取り出す。

// Q4_K/Q5_K 共通の 6bit スケール・最小値の展開 (get_scale_min_k4)。
inline void ggml_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// block_q2_K: scales[16] + qs[64] + d + dmin (84 バイト)。
inline float ggml_q2_k_value(const uint8_t * blk, int i) {
    const uint8_t * scales = blk;
    const uint8_t * qs = blk + 16;
    const float d = ggml_half_to_float(ggml_load_u16(blk + 80));
    const float dmin = ggml_half_to_float(ggml_load_u16(blk + 82));
    const int n = i >> 7;              // 0 or 1 (128 値ずつ)
    const int r = i & 127;
    const int j = r >> 5;              // 0..3
    const int half = (r >> 4) & 1;
    const int l = r & 15;
    const int is = n * 8 + j * 2 + half;
    const int shift = j * 2;
    const uint8_t sc = scales[is];
    const uint8_t q = qs[n * 32 + l + (half ? 16 : 0)];
    const int v = (q >> shift) & 3;
    return d * static_cast<float>(sc & 0x0F) * static_cast<float>(v) -
           dmin * static_cast<float>(sc >> 4);
}

// block_q3_K: hmask[32] + qs[64] + scales[12] + d (110 バイト)。
inline float ggml_q3_k_value(const uint8_t * blk, int i) {
    const uint8_t * hm = blk;
    const uint8_t * qs = blk + 32;
    const uint8_t * raw = blk + 96;
    const float d_all = ggml_half_to_float(ggml_load_u16(blk + 108));

    // 12 バイト → 16 個の int8 スケール (dequantize_row_q3_K の aux 展開)。
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, raw, 12);
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * scales = reinterpret_cast<const int8_t *>(aux);

    const int n = i >> 7;
    const int r = i & 127;
    const int j = r >> 5;
    const int half = (r >> 4) & 1;
    const int l = r & 15;
    const int is = n * 8 + j * 2 + half;
    const int shift = j * 2;
    const uint8_t m = static_cast<uint8_t>(1u << (n * 4 + j));
    // qs は n ごとに 32 バイト進むが、hmask は 32 バイト固定で n に依存しない
    // (dequantize_row_q3_K も q だけを進めている)。
    const int hidx = l + (half ? 16 : 0);
    const uint8_t q = qs[n * 32 + hidx];
    const int v = static_cast<int>((q >> shift) & 3) - ((hm[hidx] & m) ? 0 : 4);
    return d_all * static_cast<float>(scales[is] - 32) * static_cast<float>(v);
}

// block_q4_K: d + dmin + scales[12] + qs[128] (144 バイト)。
inline float ggml_q4_k_value(const uint8_t * blk, int i) {
    const uint8_t * scales = blk + 4;
    const uint8_t * qs = blk + 16;
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    const float dmin = ggml_half_to_float(ggml_load_u16(blk + 2));
    const int j = i >> 6;              // 0..3 (64 値ずつ)
    const int r = i & 63;
    const int half = r >> 5;           // 下ニブル / 上ニブル
    const int l = r & 31;
    uint8_t sc = 0, m = 0;
    ggml_get_scale_min_k4(j * 2 + half, scales, &sc, &m);
    const uint8_t q = qs[j * 32 + l];
    const int v = half ? (q >> 4) : (q & 0x0F);
    return d * static_cast<float>(sc) * static_cast<float>(v) - dmin * static_cast<float>(m);
}

// block_q5_K: d + dmin + scales[12] + qh[32] + qs[128] (176 バイト)。
inline float ggml_q5_k_value(const uint8_t * blk, int i) {
    const uint8_t * scales = blk + 4;
    const uint8_t * qh = blk + 16;
    const uint8_t * ql = blk + 48;
    const float d = ggml_half_to_float(ggml_load_u16(blk));
    const float dmin = ggml_half_to_float(ggml_load_u16(blk + 2));
    const int j = i >> 6;
    const int r = i & 63;
    const int half = r >> 5;
    const int l = r & 31;
    uint8_t sc = 0, m = 0;
    ggml_get_scale_min_k4(j * 2 + half, scales, &sc, &m);
    const uint8_t q = ql[j * 32 + l];
    const uint8_t bit = static_cast<uint8_t>(1u << (j * 2 + half));
    const int v = (half ? (q >> 4) : (q & 0x0F)) + ((qh[l] & bit) ? 16 : 0);
    return d * static_cast<float>(sc) * static_cast<float>(v) - dmin * static_cast<float>(m);
}

// ブロック先頭ポインタとブロック内位置から 1 値を返す。
inline float ggml_dequant_value_f32(GgmlType type, const uint8_t * block, int index_in_block) {
    switch (type) {
        case GgmlType::F32:
            return reinterpret_cast<const float *>(block)[index_in_block];
        case GgmlType::F16:
            return ggml_half_to_float(reinterpret_cast<const uint16_t *>(block)[index_in_block]);
        case GgmlType::Q4_0: return ggml_q4_0_value(block, index_in_block);
        case GgmlType::Q5_0: return ggml_q5_0_value(block, index_in_block);
        case GgmlType::Q5_1: return ggml_q5_1_value(block, index_in_block);
        case GgmlType::Q4_1: return ggml_q4_1_value(block, index_in_block);
        case GgmlType::Q8_0: return ggml_q8_0_value(block, index_in_block);
        case GgmlType::Q6_K: return ggml_q6_k_value(block, index_in_block);
        case GgmlType::Q2_K: return ggml_q2_k_value(block, index_in_block);
        case GgmlType::Q3_K: return ggml_q3_k_value(block, index_in_block);
        case GgmlType::Q4_K: return ggml_q4_k_value(block, index_in_block);
        case GgmlType::Q5_K: return ggml_q5_k_value(block, index_in_block);
        default:
            throw std::runtime_error(std::string("再量子化できない型です: ") + ggml_type_name(type));
    }
}

// 1 行 (cols 値) を Q8_0 ブロック列 (cols/32 * 34 バイト) へ。
inline void ggml_requantize_row_to_q8_0(GgmlType type, const uint8_t * src,
                                        uint8_t * dst, int64_t cols) {
    const int block = ggml_type_block_size(type);
    const int block_bytes = ggml_type_block_bytes(type);
    if (block <= 1) throw std::runtime_error("1 要素ブロックの型は再量子化対象外です");
    if (cols % 32 != 0) {
        throw std::runtime_error("Q8_0 へ詰め直すには列数が 32 の倍数である必要があります: " +
                                 std::to_string(cols));
    }
    for (int64_t b = 0; b < cols / 32; ++b) {
        float v[32];
        for (int i = 0; i < 32; ++i) {
            const int64_t index = b * 32 + i;
            const uint8_t * bp = src + (index / block) * block_bytes;
            v[i] = ggml_dequant_value_f32(type, bp, static_cast<int>(index % block));
        }
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float d = amax / 127.0f;
        uint8_t * out = dst + b * 34;
        const __half dh = __float2half(d);
        std::memcpy(out, &dh, sizeof(uint16_t));
        for (int i = 0; i < 32; ++i) {
            int q = (d == 0.0f) ? 0 : static_cast<int>(std::lround(v[i] / d));
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        }
    }
}

} // namespace wd
