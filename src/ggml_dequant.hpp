#pragma once
// ホスト側の GGUF テンソル → F16 展開。
//
// dense パスは元々 Q4_0 しか読めなかったが、llama.cpp が配る Q4_0 の GGUF でも
// output.weight は Q6_K、Llama-3.2-1B の ffn_down は Q8_0 ということが普通にあり、
// そのままだと「Q4_0 以外は未対応」でどのモデルも起動できなかった。
// ここで Q8_0 / Q6_K / F16 / F32 を F16 に落として F16 GEMV に渡す。
// 量子化のままより重い (Q6_K の 2.4 倍) が、再量子化せず値も変えない。
#include "gguf.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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
inline bool ggml_type_as_f16_supported(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::Q8_0:
        case GgmlType::Q6_K:
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
        default: return 0;
    }
}

inline int ggml_type_block_bytes(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q6_K: return 210;
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

} // namespace wd
