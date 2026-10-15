#pragma once
// Qwen3.8-Flash-Next (qwen4exp) tokenizer: byte-level BPE + Qwen pre-tokenization.
//
// This header is the host-side port of the pinned llama.cpp tokenizer so that
// "string -> token IDs -> string" in warodecode produces exactly the token
// sequence llama.cpp produces for the same model:
//
//   llama.cpp 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
//     src/llama-vocab.cpp   llm_tokenizer_bpe / llm_tokenizer_bpe_session
//                           llama_vocab::impl::tokenize / tokenizer_st_partition
//                           token_to_piece (BPE branch) / llama_decode_text
//     src/unicode.cpp       unicode_regex_split_custom_qwen2 / _qwen35,
//                           unicode_byte_to_utf8 / unicode_utf8_to_byte
//     src/unicode-data.cpp  codepoint flags + lowercase map
//                           -> generated into qwen38_tokenizer_unicode.inc
//
// What is intentionally NOT ported (and why it cannot matter for this model):
//   * the generic std::regex pre-tokenizer fallback: the qwen2 / qwen35
//     patterns always take llama.cpp's hand-written custom splitter, which is
//     the one ported here
//   * non-BPE vocabularies ("spm", "ugm", ...) and the SPM-style
//     escape_whitespaces path (qwen4exp is GPT-2 byte-level BPE)
//   * the hybriddna / whitespace / gemma4 special BPE sessions
//
// Vocabulary data comes straight from the GGUF metadata (tokenizer.ggml.*), so
// no sidecar tokenizer.json is needed. Only the metadata section is read: the
// 85 GB of tensor data is never touched, which is why the tokenizer also runs
// on a GPU-less host.
//
// No CUDA types are used, so this file compiles with plain g++.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wd {

struct Qwen38UnicodeRange {
    uint32_t first;
    uint32_t last;   // exclusive
    uint16_t flags;
};

struct Qwen38CasePair {
    uint32_t from;
    uint32_t to;
};

#include "qwen38_tokenizer_unicode.inc"

// Minimal "left/right -> merge rank" key. Byte-level BPE symbols never contain
// the ASCII space (0x20 is encoded as U+0120 'G-dot'), which is the same
// assumption llama.cpp's merges parser makes with word.find(' ', 1).
inline std::string qwen38_merge_key(const std::string & left, const std::string & right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key += left;
    key.push_back(' ');
    key += right;
    return key;
}

// ---------------------------------------------------------------------------
// GGUF tokenizer metadata (metadata section only, tensors are never read)
// ---------------------------------------------------------------------------

class Qwen38GgufMeta {
public:
    explicit Qwen38GgufMeta(const std::string & path) : path_(path) { parse(); }

    bool has(const std::string & key) const { return values_.count(key) != 0; }

    const std::string * string(const std::string & key) const {
        auto it = values_.find(key);
        if (it == values_.end() || it->second.kind != Value::Kind::String) return nullptr;
        return &it->second.str;
    }

    const std::vector<std::string> * strings(const std::string & key) const {
        auto it = values_.find(key);
        if (it == values_.end() || it->second.kind != Value::Kind::StringArray) return nullptr;
        return &it->second.strs;
    }

    const std::vector<double> * floats(const std::string & key) const {
        auto it = values_.find(key);
        if (it == values_.end() || it->second.kind != Value::Kind::FloatArray) return nullptr;
        return &it->second.floats;
    }

    const std::vector<int64_t> * ints(const std::string & key) const {
        auto it = values_.find(key);
        if (it == values_.end() || it->second.kind != Value::Kind::IntArray) return nullptr;
        return &it->second.ints;
    }

    // GGUF stores counts as U32 and flags as Bool/U8/U32; everything integral is
    // reported as int64 here.
    bool integer(const std::string & key, int64_t & out) const {
        auto it = values_.find(key);
        if (it == values_.end()) return false;
        if (it->second.kind != Value::Kind::Int) return false;
        out = it->second.i;
        return true;
    }

    int64_t integer_or(const std::string & key, int64_t fallback) const {
        int64_t value = 0;
        return integer(key, value) ? value : fallback;
    }

    bool boolean_or(const std::string & key, bool fallback) const {
        int64_t value = 0;
        if (!integer(key, value)) return fallback;
        return value != 0;
    }

    const std::string & path() const { return path_; }

private:
    struct Value {
        enum class Kind { String, Int, Float, StringArray, IntArray, FloatArray, Other };
        Kind kind = Kind::Other;
        std::string str;
        std::vector<std::string> strs;
        std::vector<int64_t> ints;
        std::vector<double> floats;
        int64_t i = 0;
        double f = 0.0;
    };

    // gguf_type (gguf.h).
    enum : uint32_t {
        kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5,
        kF32 = 6, kBool = 7, kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
    };

    void fail(const std::string & message) const {
        throw std::runtime_error("GGUFトークナイザメタデータ: " + message + " (" + path_ + ")");
    }

    uint8_t read_u8() {
        const int byte = file_.get();
        if (byte < 0) fail("ファイル末尾に達しました");
        return static_cast<uint8_t>(byte);
    }

    uint64_t read_le(uint32_t bytes) {
        uint64_t value = 0;
        for (uint32_t i = 0; i < bytes; ++i) value |= static_cast<uint64_t>(read_u8()) << (8 * i);
        return value;
    }

    std::string read_string() {
        const uint64_t length = read_le(8);
        if (length > (1ull << 32)) fail("文字列長が異常です");
        std::string out(static_cast<size_t>(length), '\0');
        if (length > 0) {
            file_.read(&out[0], static_cast<std::streamsize>(length));
            if (static_cast<uint64_t>(file_.gcount()) != length) fail("文字列を読み切れませんでした");
        }
        return out;
    }
void read_value(uint32_t type, Value & out) {
        switch (type) {
            case kU8:   out.kind = Value::Kind::Int; out.i = read_u8(); break;
            case kI8:   out.kind = Value::Kind::Int; out.i = static_cast<int8_t>(read_u8()); break;
            case kU16:  out.kind = Value::Kind::Int; out.i = static_cast<int64_t>(read_le(2)); break;
            case kI16:  out.kind = Value::Kind::Int; out.i = static_cast<int16_t>(read_le(2)); break;
            case kU32:  out.kind = Value::Kind::Int; out.i = static_cast<int64_t>(read_le(4)); break;
            case kI32:  out.kind = Value::Kind::Int; out.i = static_cast<int32_t>(read_le(4)); break;
            case kU64:  out.kind = Value::Kind::Int; out.i = static_cast<int64_t>(read_le(8)); break;
            case kI64:  out.kind = Value::Kind::Int; out.i = static_cast<int64_t>(read_le(8)); break;
            case kF32: {
                const uint32_t bits = static_cast<uint32_t>(read_le(4));
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                out.kind = Value::Kind::Float;
                out.f = value;
                break;
            }
            case kF64: {
                const uint64_t bits = read_le(8);
                double value = 0.0;
                std::memcpy(&value, &bits, sizeof(value));
                out.kind = Value::Kind::Float;
                out.f = value;
                break;
            }
            case kBool:   out.kind = Value::Kind::Int; out.i = read_u8() != 0 ? 1 : 0; break;
            case kString: out.kind = Value::Kind::String; out.str = read_string(); break;
            case kArray: {
                const uint32_t element = static_cast<uint32_t>(read_le(4));
                const uint64_t count = read_le(8);
                if (count > (1ull << 32)) fail("配列長が異常です");
                read_array(element, count, out);
                break;
            }
            default: fail("未対応のメタデータ型です: " + std::to_string(type));
        }
    }

    void read_array(uint32_t element, uint64_t count, Value & out) {
        switch (element) {
            case kString:
                out.kind = Value::Kind::StringArray;
                out.strs.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) out.strs.push_back(read_string());
                return;
            case kU8: case kI8: case kU16: case kI16:
            case kU32: case kI32: case kU64: case kI64: case kBool:
                out.kind = Value::Kind::IntArray;
                out.ints.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) {
                    Value item;
                    read_value(element, item);
                    out.ints.push_back(item.i);
                }
                return;
            case kF32: case kF64:
                out.kind = Value::Kind::FloatArray;
                for (uint64_t i = 0; i < count; ++i) {
                    Value item;
                    read_value(element, item);
                    out.floats.push_back(item.f);
                }
                return;
            default:
                // Arrays of arrays are not needed by the tokenizer; read them so
                // the stream stays aligned and keep them out of the value map.
                for (uint64_t i = 0; i < count; ++i) {
                    Value item;
                    read_value(element, item);
                }
                out.kind = Value::Kind::Other;
                return;
        }
    }

    void parse() {
        file_.open(path_, std::ios::binary);
        if (!file_) fail("開けません");
        if (read_le(4) != 0x46554747u) fail("GGUF magicが一致しません");
        const uint64_t version = read_le(4);
        if (version < 2 || version > 3) fail("対応外のGGUF versionです");
        const uint64_t tensor_count = read_le(8);
        const uint64_t kv_count = read_le(8);
        if (tensor_count > (1ull << 24) || kv_count > (1ull << 24)) fail("ヘッダ件数が異常です");
        // Only the key/value section is read: the tensor index and the tensor
        // data behind it stay untouched.
        for (uint64_t i = 0; i < kv_count; ++i) {
            const std::string key = read_string();
            const uint32_t type = static_cast<uint32_t>(read_le(4));
            Value value;
            read_value(type, value);
            values_.emplace(key, std::move(value));
        }
    }

    std::string path_;
    std::ifstream file_;
    std::unordered_map<std::string, Value> values_;
};

// ---------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------

class Qwen38Tokenizer {
public:
    // llama_token_type (include/llama.h).
    enum TokenType {
        kUndefined = 0, kNormal = 1, kUnknown = 2, kControl = 3,
        kUserDefined = 4, kUnused = 5, kByte = 6,
    };
    // Pre types that this port implements; anything else is refused instead of
    // silently falling back to a different split rule.
    enum class PreType { Qwen2, Qwen35, Gpt2, LlamaBpe };

    explicit Qwen38Tokenizer(const std::string & gguf_path, bool verbose = false) {
        Qwen38GgufMeta meta(gguf_path);
        arch_ = meta.string("general.architecture") ? *meta.string("general.architecture") : std::string();
        const std::string * model = meta.string("tokenizer.ggml.model");
        if (model == nullptr) throw std::runtime_error("GGUFにtokenizer.ggml.modelがありません: " + gguf_path);
        model_ = *model;
        if (model_ != "gpt2" && model_ != "llama") {
            throw std::runtime_error(
                "未対応のトークナイザ種別です: " + model_ +
                "\n  対応: gpt2 (BPE) / llama (SentencePiece)");
        }
        spm_ = (model_ == "llama");
        pre_ = meta.string("tokenizer.ggml.pre") ? *meta.string("tokenizer.ggml.pre") : std::string();
        if (pre_.empty() || pre_ == "qwen2" || pre_ == "megrez") {
            pre_type_ = PreType::Qwen2;
        } else if (pre_ == "qwen35") {
            pre_type_ = PreType::Qwen35;
        } else if (pre_ == "gpt2") {
            pre_type_ = PreType::Gpt2;
        } else if (pre_ == "llama-bpe") {
            // Llama-3 系。qwen2 との違いは数字のまとめ方だけ (\p{N}{1,3} vs \p{N})。
            pre_type_ = PreType::LlamaBpe;
        } else {
            throw std::runtime_error("未対応のtokenizer.ggml.preです: " + pre_);
        }

        const std::vector<std::string> * tokens = meta.strings("tokenizer.ggml.tokens");
        if (tokens == nullptr || tokens->empty()) {
            throw std::runtime_error("GGUFにtokenizer.ggml.tokensがありません: " + gguf_path);
        }
        tokens_ = *tokens;
        const std::vector<int64_t> * types = meta.ints("tokenizer.ggml.token_type");
        token_types_.assign(tokens_.size(), kNormal);
        if (types != nullptr) {
            if (types->size() != tokens_.size()) {
                throw std::runtime_error("tokenizer.ggml.token_typeの要素数がtokensと一致しません");
            }
            for (size_t i = 0; i < types->size(); ++i) {
                const int64_t value = (*types)[i];
                token_types_[i] = (value >= 0 && value <= 6) ? static_cast<int>(value) : kUndefined;
            }
        }

        token_to_id_.reserve(tokens_.size() * 2);
        for (size_t i = 0; i < tokens_.size(); ++i) {
            // llama.cpp keeps the last id for a duplicated string.
            token_to_id_[tokens_[i]] = static_cast<int32_t>(i);
        }

        const std::vector<std::string> * merges = meta.strings("tokenizer.ggml.merges");
        if (merges == nullptr) throw std::runtime_error("GGUFにtokenizer.ggml.mergesがありません: " + gguf_path);
        bpe_ranks_.reserve(merges->size() * 2);
        for (size_t i = 0; i < merges->size(); ++i) {
            const std::string & entry = (*merges)[i];
            if (entry.find(' ', 1) == std::string::npos) continue;
            // First rank wins, exactly like llama.cpp's bpe_ranks.emplace().
            bpe_ranks_.emplace(entry, static_cast<int32_t>(i));
        }
        merge_count_ = merges->size();

        if (spm_) {
            const std::vector<double> * sc = meta.floats("tokenizer.ggml.scores");
            if (sc == nullptr) {
                throw std::runtime_error("SentencePiece の tokenizer.ggml.scores がありません: " +
                                         gguf_path);
            }
            scores_ = *sc;
        }
        bos_id_ = static_cast<int32_t>(meta.integer_or("tokenizer.ggml.bos_token_id", -1));
        eos_id_ = static_cast<int32_t>(meta.integer_or("tokenizer.ggml.eos_token_id", -1));
        unk_id_ = static_cast<int32_t>(meta.integer_or("tokenizer.ggml.unknown_token_id", -1));
        pad_id_ = static_cast<int32_t>(meta.integer_or("tokenizer.ggml.padding_token_id", -1));
        add_bos_ = meta.boolean_or("tokenizer.ggml.add_bos_token", false);
        add_eos_ = meta.boolean_or("tokenizer.ggml.add_eos_token", false);
        chat_template_ = meta.string("tokenizer.chat_template") ? *meta.string("tokenizer.chat_template") : std::string();

        // Special token cache: control / user-defined / unknown, longest text
        // first (llama.cpp tokenizer_st_partition order).
        for (size_t i = 0; i < tokens_.size(); ++i) {
            const int type = token_types_[i];
            if (type == kControl || type == kUserDefined || type == kUnknown) {
                special_ids_.push_back(static_cast<int32_t>(i));
            }
        }
        std::stable_sort(special_ids_.begin(), special_ids_.end(),
            [this](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });

        // Stop tokens: EOS plus the well-known end-of-turn texts llama.cpp
        // auto-detects.
        if (eos_id_ >= 0) stop_ids_.push_back(eos_id_);
        for (size_t i = 0; i < tokens_.size(); ++i) {
            const std::string & text = tokens_[i];
            if (text == "<|im_end|>" || text == "<|endoftext|>" || text == "<|eot_id|>" ||
                text == "<|end_of_text|>" || text == "<end_of_turn>") {
                const int32_t id = static_cast<int32_t>(i);
                if (std::find(stop_ids_.begin(), stop_ids_.end(), id) == stop_ids_.end()) stop_ids_.push_back(id);
            }
        }
        if (verbose) {
            std::fprintf(stderr,
                "[tokenizer] model=%s pre=%s vocab=%zu merges=%zu specials=%zu "
                "bos=%d eos=%d add_bos=%d add_eos=%d chat_template=%s\n",
                model_.c_str(), pre_.c_str(), tokens_.size(), merge_count_, special_ids_.size(),
                bos_id_, eos_id_, add_bos_ ? 1 : 0, add_eos_ ? 1 : 0,
                chat_template_.empty() ? "none" : "yes");
        }
    }
const std::string & architecture() const { return arch_; }
    const std::string & pre_type_name() const { return pre_; }
    const std::string & chat_template() const { return chat_template_; }
    size_t vocab_size() const { return tokens_.size(); }
    size_t merge_count() const { return merge_count_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t unknown_id() const { return unk_id_; }
    int32_t padding_id() const { return pad_id_; }
    bool add_bos() const { return add_bos_; }
    bool add_eos() const { return add_eos_; }
    const std::vector<int32_t> & stop_ids() const { return stop_ids_; }
    const std::vector<std::string> & tokens() const { return tokens_; }

    const std::string & token_text(int32_t token) const {
        check_token(token);
        return tokens_[static_cast<size_t>(token)];
    }
    int token_type(int32_t token) const {
        check_token(token);
        return token_types_[static_cast<size_t>(token)];
    }
    bool is_special(int32_t token) const {
        check_token(token);
        const int type = token_types_[static_cast<size_t>(token)];
        return type == kControl || type == kUserDefined || type == kUnknown;
    }
    bool is_stop(int32_t token) const {
        return std::find(stop_ids_.begin(), stop_ids_.end(), token) != stop_ids_.end();
    }

    // llama_tokenize(text, add_special, parse_special).
    // add_special  prepends BOS / appends EOS when the GGUF asks for it.
    // parse_special splits control / user-defined / unknown vocab entries out of
    // the raw text before BPE, so "<|im_start|>" becomes a single token.
    // ---- SentencePiece (tokenizer.ggml.model == "llama") ----
    //
    // llama.cpp の llm_tokenizer_spm と同じ手順:
    //   1. 空白を ▁ (U+2581) に置き換えて UTF-8 コードポイント単位の断片にする
    //   2. 隣接する 2 断片の連結が語彙にある間、スコア最大の組から順に併合する
    //   3. 残った断片を語彙で引き、引けなければ 1 バイトずつ <0xXX> に落とす
    std::vector<int32_t> encode_spm(const std::string & text, bool add_special) const {
        const std::string marked = spm_mark_spaces(" " + text);
        std::vector<std::string> frags = spm_split_codepoints(marked);

        std::vector<int> prev(frags.size()), next(frags.size());
        std::vector<unsigned char> alive(frags.size(), 1);
        for (size_t i = 0; i < frags.size(); ++i) {
            prev[i] = static_cast<int>(i) - 1;
            next[i] = static_cast<int>(i) + 1 < static_cast<int>(frags.size())
                          ? static_cast<int>(i) + 1 : -1;
        }

        // スコア付きの最大ヒープ (連結した断片が語彙にあるものだけ積む)
        struct Item { double score; int left; int right; };
        struct Cmp { bool operator()(const Item & a, const Item & b) const { return a.score < b.score; } };
        std::priority_queue<Item, std::vector<Item>, Cmp> queue;
        auto try_push = [&](int left, int right) {
            if (left < 0 || right < 0) return;
            const std::string joined = frags[left] + frags[right];
            auto it = token_to_id_.find(joined);
            if (it == token_to_id_.end()) return;
            const double score = it->second < static_cast<int>(scores_.size())
                                     ? scores_[it->second] : 0.0;
            queue.push({score, left, right});
        };
        for (size_t i = 0; i + 1 < frags.size(); ++i) try_push(static_cast<int>(i), static_cast<int>(i) + 1);
        while (!queue.empty()) {
            const Item item = queue.top();
            queue.pop();
            if (!alive[item.left] || !alive[item.right]) continue;
            if (next[item.left] != item.right) continue;
            frags[item.left] += frags[item.right];
            alive[item.right] = 0;
            next[item.left] = next[item.right];
            if (next[item.left] >= 0) prev[next[item.left]] = item.left;
            try_push(prev[item.left], item.left);
            try_push(item.left, next[item.left]);
        }

        std::vector<int32_t> out;
        if (add_special && bos_id_ >= 0 && add_bos_) out.push_back(bos_id_);
        for (size_t i = 0; i < frags.size(); ++i) {
            if (!alive[i]) continue;
            auto it = token_to_id_.find(frags[i]);
            if (it != token_to_id_.end()) {
                out.push_back(it->second);
                continue;
            }
            for (unsigned char byte : frags[i]) {
                char name[8];
                std::snprintf(name, sizeof(name), "<0x%02X>", byte);
                auto bit = token_to_id_.find(name);
                if (bit != token_to_id_.end()) {
                    out.push_back(bit->second);
                } else if (unk_id_ >= 0) {
                    out.push_back(unk_id_);
                }
            }
        }
        return out;
    }

    static std::string spm_mark_spaces(const std::string & text) {
        std::string out;
        out.reserve(text.size() + 4);
        for (size_t i = 0; i < text.size();) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == ' ') {
                out += "\xe2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK
                ++i;
                continue;
            }
            size_t len = 1;
            if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
            out.append(text, i, len);
            i += len;
        }
        return out;
    }

    static std::vector<std::string> spm_split_codepoints(const std::string & text) {
        std::vector<std::string> out;
        for (size_t i = 0; i < text.size();) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            size_t len = 1;
            if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
            out.push_back(text.substr(i, len));
            i += len;
        }
        return out;
    }

    // SPM のデコード: ▁ を空白に戻し、<0xXX> はバイトとして復元する。
    std::string decode_spm(const std::vector<int32_t> & tokens, bool skip_special) const {
        std::string out;
        for (int32_t id : tokens) {
            if (id < 0 || id >= static_cast<int>(tokens_.size())) continue;
            if (skip_special && is_special(id)) continue;
            const std::string & piece = tokens_[id];
            if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
                const int byte = std::strtol(piece.substr(3, 2).c_str(), nullptr, 16);
                out.push_back(static_cast<char>(byte));
                continue;
            }
            for (size_t i = 0; i < piece.size();) {
                if (i + 2 < piece.size() &&
                    static_cast<unsigned char>(piece[i]) == 0xE2 &&
                    static_cast<unsigned char>(piece[i + 1]) == 0x96 &&
                    static_cast<unsigned char>(piece[i + 2]) == 0x81) {
                    out.push_back(' ');
                    i += 3;
                    continue;
                }
                out.push_back(piece[i]);
                ++i;
            }
        }
        return out;
    }

    std::vector<int32_t> encode(const std::string & text, bool add_special, bool parse_special) const {
        std::vector<int32_t> output;
        std::vector<Fragment> fragments;
        if (!text.empty()) {
            fragments.push_back(Fragment::raw(0, text.size()));
            partition(text, parse_special, fragments);
        }
        if (add_special && add_bos_ && bos_id_ >= 0) output.push_back(bos_id_);
        for (const Fragment & fragment : fragments) {
            if (!fragment.is_text) {
                output.push_back(fragment.token);
                continue;
            }
            const std::string piece = text.substr(fragment.offset, fragment.length);
            if (spm_) {
                const std::vector<int32_t> ids = encode_spm(piece, /*add_special=*/false);
                output.insert(output.end(), ids.begin(), ids.end());
            } else {
                encode_text(piece, output);
            }
        }
        if (add_special && add_eos_ && eos_id_ >= 0) output.push_back(eos_id_);
        return output;
    }

    std::vector<int32_t> encode(const std::string & text) const { return encode(text, true, true); }

    // llama_token_get_piece / token_to_piece (BPE branch): the raw bytes a token
    // contributes. Byte-level BPE pieces can end in the middle of a UTF-8
    // sequence; callers that print text must use Qwen38StreamDecoder below.
    std::string piece(int32_t token) const {
        check_token(token);
        const std::string & text = tokens_[static_cast<size_t>(token)];
        const int type = token_types_[static_cast<size_t>(token)];
        if (type == kControl || type == kUserDefined || type == kUnknown) return text;
        if (type == kByte) {
            // BPE byte tokens carry their value in "<0xXX>" (llama.cpp
            // token_to_byte, BPE branch).
            if (text.size() == 6 && text.compare(0, 3, "<0x") == 0 && text.back() == '>') {
                const char byte = static_cast<char>(std::strtol(text.substr(3, 2).c_str(), nullptr, 16));
                return std::string(1, byte);
            }
            return text;
        }
        return decode_text_bytes(text);
    }

    // llama_detokenize(tokens, remove_special, special).
    std::string decode(const std::vector<int32_t> & tokens, bool skip_special) const {
        if (spm_) return decode_spm(tokens, skip_special);
        std::string out;
        for (int32_t token : tokens) {
            if (skip_special && is_special(token)) continue;
            out += piece(token);
        }
        return out;
    }

    std::string decode(const std::vector<int32_t> & tokens) const { return decode(tokens, true); }

    // Chat template support: the GGUF stores a Jinja template, which needs a
    // full Jinja engine to evaluate in general. Qwen's template is a fixed
    // shape, so it is reproduced here and anything else is refused loudly (the
    // caller then falls back to the raw prompt).
    bool supports_qwen_chat_template() const {
        return chat_template_.find("<|im_start|>") != std::string::npos &&
               chat_template_.find("<|im_end|>") != std::string::npos &&
               token_to_id_.count("<|im_start|>") != 0 &&
               token_to_id_.count("<|im_end|>") != 0;
    }

    std::string apply_qwen_chat_template(
            const std::vector<std::pair<std::string, std::string>> & messages,
            bool add_generation_prompt) const {
        std::string out;
        for (const auto & message : messages) {
            out += "<|im_start|>";
            out += message.first;
            out += "\n";
            out += message.second;
            out += "<|im_end|>\n";
        }
        if (add_generation_prompt) out += "<|im_start|>assistant\n";
        return out;
    }

private:
    struct Fragment {
        bool is_text = true;
        int32_t token = -1;
        size_t offset = 0;
        size_t length = 0;
        static Fragment raw(size_t offset, size_t length) {
            Fragment fragment;
            fragment.is_text = true;
            fragment.offset = offset;
            fragment.length = length;
            return fragment;
        }
        static Fragment special(int32_t token) {
            Fragment fragment;
            fragment.is_text = false;
            fragment.token = token;
            return fragment;
        }
    };

    void check_token(int32_t token) const {
        if (token < 0 || static_cast<size_t>(token) >= tokens_.size()) {
            throw std::runtime_error("Qwen3.8トークンIDが語彙範囲外です: " + std::to_string(token));
        }
    }

    // ---- unicode helpers (port of src/unicode.cpp) ----

    static uint16_t cpt_flags(uint32_t cpt) {
        constexpr uint16_t kUndefinedFlag = 0x0001;
        if (cpt >= 0x110000u) return kUndefinedFlag;
        // Sorted ranges: the first starts at 0 and the last ends at 0x110000, so
        // every codepoint is covered.
        size_t lo = 0;
        size_t hi = sizeof(kQwen38UnicodeFlags) / sizeof(kQwen38UnicodeFlags[0]);
        while (hi - lo > 1) {
            const size_t mid = lo + (hi - lo) / 2;
            if (kQwen38UnicodeFlags[mid].first <= cpt) lo = mid; else hi = mid;
        }
        return kQwen38UnicodeFlags[lo].flags;
    }

    static size_t lowercase_count() {
        return sizeof(kQwen38Lowercase) / sizeof(kQwen38Lowercase[0]);
    }

    static uint32_t to_lower(uint32_t cpt) {
        size_t lo = 0;
        size_t hi = lowercase_count();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (kQwen38Lowercase[mid].from < cpt) lo = mid + 1; else hi = mid;
        }
        if (lo < lowercase_count() && kQwen38Lowercase[lo].from == cpt) return kQwen38Lowercase[lo].to;
        return cpt;
    }
static size_t utf8_len(char lead) {
        static const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        return lookup[static_cast<uint8_t>(lead) >> 4];
    }

    static std::string cpt_to_utf8(uint32_t cpt) {
        std::string out;
        if (cpt <= 0x7f) {
            out.push_back(static_cast<char>(cpt));
        } else if (cpt <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | ((cpt >> 6) & 0x1f)));
            out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        } else if (cpt <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | ((cpt >> 12) & 0x0f)));
            out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        } else if (cpt <= 0x10ffff) {
            out.push_back(static_cast<char>(0xf0 | ((cpt >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        }
        return out;
    }

    // llama.cpp unicode_cpts_from_utf8(): invalid bytes become U+FFFD and are
    // consumed one byte at a time, so encode() never throws on bad input.
    static std::vector<uint32_t> cpts_from_utf8(const std::string & text) {
        std::vector<uint32_t> cpts;
        cpts.reserve(text.size());
        size_t offset = 0;
        while (offset < text.size()) {
            const uint8_t b0 = static_cast<uint8_t>(text[offset]);
            size_t length = 1;
            uint32_t value = b0;
            if (b0 < 0x80) {
                length = 1;
            } else if ((b0 & 0xe0) == 0xc0) {
                length = 2;
                value = b0 & 0x1fu;
            } else if ((b0 & 0xf0) == 0xe0) {
                length = 3;
                value = b0 & 0x0fu;
            } else if ((b0 & 0xf8) == 0xf0) {
                length = 4;
                value = b0 & 0x07u;
            } else {
                cpts.push_back(0xFFFD);
                ++offset;
                continue;
            }
            bool ok = offset + length <= text.size();
            for (size_t i = 1; ok && i < length; ++i) {
                const uint8_t cont = static_cast<uint8_t>(text[offset + i]);
                if ((cont & 0xc0) != 0x80) { ok = false; break; }
                value = (value << 6) | (cont & 0x3fu);
            }
            if (!ok) {
                cpts.push_back(0xFFFD);
                ++offset;
                continue;
            }
            cpts.push_back(value);
            offset += length;
        }
        return cpts;
    }

    // unicode_byte_to_utf8 / unicode_utf8_to_byte (GPT-2 byte encoder).
    static const std::vector<std::string> & byte_to_unicode() {
        static const std::vector<std::string> table = [] {
            std::vector<std::string> map(256);
            std::vector<bool> used(256, false);
            for (int ch = 0x21; ch <= 0x7E; ++ch) { map[ch] = cpt_to_utf8(static_cast<uint32_t>(ch)); used[ch] = true; }
            for (int ch = 0xA1; ch <= 0xAC; ++ch) { map[ch] = cpt_to_utf8(static_cast<uint32_t>(ch)); used[ch] = true; }
            for (int ch = 0xAE; ch <= 0xFF; ++ch) { map[ch] = cpt_to_utf8(static_cast<uint32_t>(ch)); used[ch] = true; }
            int extra = 0;
            for (int ch = 0; ch < 256; ++ch) {
                if (!used[ch]) map[ch] = cpt_to_utf8(static_cast<uint32_t>(256 + extra++));
            }
            return map;
        }();
        return table;
    }

    static const std::unordered_map<std::string, uint8_t> & unicode_to_byte() {
        static const std::unordered_map<std::string, uint8_t> map = [] {
            std::unordered_map<std::string, uint8_t> out;
            const std::vector<std::string> & table = byte_to_unicode();
            for (int ch = 0; ch < 256; ++ch) out.emplace(table[ch], static_cast<uint8_t>(ch));
            return out;
        }();
        return map;
    }

    // llama_decode_text(): GPT-2 byte-encoded piece -> raw bytes.
    static std::string decode_text_bytes(const std::string & text) {
        const auto & map = unicode_to_byte();
        std::string out;
        for (uint32_t cpt : cpts_from_utf8(text)) {
            const std::string utf8 = cpt_to_utf8(cpt);
            const auto it = map.find(utf8);
            if (it == map.end()) {
                out += "[UNK_BYTE_0x" + std::to_string(cpt) + "]";
                continue;
            }
            out.push_back(static_cast<char>(it->second));
        }
        return out;
    }

    // unicode_byte_encoding_process(): every UTF-8 byte of the word is replaced
    // by its GPT-2 byte-encoder codepoint.
    static std::string byte_encode_word(const std::string & word) {
        std::string normalized;
        for (uint32_t cpt : cpts_from_utf8(word)) normalized += cpt_to_utf8(cpt);
        const std::vector<std::string> & table = byte_to_unicode();
        std::string out;
        out.reserve(normalized.size() * 2);
        for (char c : normalized) out += table[static_cast<uint8_t>(c)];
        return out;
    }
// ---- pre-tokenization: unicode_regex_split_custom_qwen2 / _qwen35 ----

    // llama.cpp unicode_cpt_flags bits used by the splitters.
    enum : uint16_t {
        kFlagNumber = 0x0002,
        kFlagLetter = 0x0004,
        kFlagAccent = 0x0010,
        kFlagWhitespace = 0x0100,
    };

    static bool is_letter(uint16_t flags) { return (flags & kFlagLetter) != 0; }
    static bool is_number(uint16_t flags) { return (flags & kFlagNumber) != 0; }
    static bool is_accent(uint16_t flags) { return (flags & kFlagAccent) != 0; }
    static bool is_ws(uint16_t flags) { return (flags & kFlagWhitespace) != 0; }

    void pre_tokenize(const std::string & text, std::vector<std::string> & words) const {
        const std::vector<uint32_t> cpts = cpts_from_utf8(text);
        if (pre_type_ == PreType::Gpt2) {
            pre_tokenize_gpt2(cpts, words);
            return;
        }
        const bool extended = pre_type_ == PreType::Qwen35;
        const int digits_max = (pre_type_ == PreType::LlamaBpe) ? 3 : 1;
        const size_t offset_end = cpts.size();
        constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

        auto get_cpt = [&](size_t pos) -> uint32_t {
            return pos < offset_end ? cpts[pos] : kOutOfRange;
        };
        auto get_flags = [&](size_t pos) -> uint16_t {
            return pos < offset_end ? cpt_flags(cpts[pos]) : 0;
        };


        size_t prev_end = 0;
        std::vector<std::pair<size_t, size_t>> spans;   // [begin, end) codepoint spans
        auto add_token = [&](size_t end) -> size_t {
            const size_t length = end - prev_end;
            if (length > 0) spans.emplace_back(prev_end, end);
            prev_end = end;
            return length;
        };

        for (size_t pos = 0; pos < offset_end; ) {
            const uint32_t cpt = get_cpt(pos);
            const uint16_t flags = get_flags(pos);

            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (cpt == '\'' && pos + 1 < offset_end) {
                const uint32_t next = to_lower(get_cpt(pos + 1));
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    pos += add_token(pos + 2);
                    continue;
                }
                if (pos + 2 < offset_end) {
                    const uint32_t next_next = to_lower(get_cpt(pos + 2));
                    if ((next == 'r' && next_next == 'e') ||
                        (next == 'v' && next_next == 'e') ||
                        (next == 'l' && next_next == 'l')) {
                        pos += add_token(pos + 3);
                        continue;
                    }
                }
            }

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (qwen2: [^\r\n\p{L}\p{N}]?\p{L}+)
            if (!(cpt == '\r' || cpt == '\n' || is_number(flags))) {
                const uint16_t next_flags = get_flags(pos + 1);
                const bool starts = extended
                    ? (is_letter(flags) || is_accent(flags) || is_accent(next_flags) || is_letter(next_flags))
                    : (is_letter(flags) || is_letter(next_flags));
                if (starts) {
                    ++pos;
                    while (extended
                           ? (is_letter(get_flags(pos)) || is_accent(get_flags(pos)))
                           : is_letter(get_flags(pos))) {
                        ++pos;
                    }
                    add_token(pos);
                    continue;
                }
            }

            // \p{N} (llama-bpe は \p{N}{1,3})
            if (is_number(flags)) {
                int taken = 0;
                while (taken < digits_max && is_number(get_flags(pos))) {
                    ++pos;
                    ++taken;
                }
                add_token(pos);
                continue;
            }

            // <space>?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
            const uint16_t flags2 = (cpt == ' ') ? get_flags(pos + 1) : flags;
            const bool symbol2 = extended
                ? !(is_ws(flags2) || is_letter(flags2) || is_accent(flags2) || is_number(flags2))
                : !(is_ws(flags2) || is_letter(flags2) || is_number(flags2));
            if (symbol2 && flags != 0) {
                pos += (cpt == ' ') ? 1 : 0;
                for (;;) {
                    const uint16_t current = get_flags(pos);
                    const bool stop = extended
                        ? (is_ws(current) || is_letter(current) || is_accent(current) || is_number(current) || current == 0)
                        : (is_ws(current) || is_letter(current) || is_number(current) || current == 0);
                    if (stop) break;
                    ++pos;
                }
                uint32_t tail = get_cpt(pos);
                while (tail == '\r' || tail == '\n') tail = get_cpt(++pos);
                add_token(pos);
                continue;
            }

            size_t whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (is_ws(get_flags(pos + whitespaces))) {
                const uint32_t c = get_cpt(pos + whitespaces);
                if (c == '\r' || c == '\n') last_end_r_or_n = pos + whitespaces + 1;
                ++whitespaces;
            }

            // \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                add_token(pos);
                continue;
            }
            // \s+(?!\S)
            if (whitespaces > 1 && get_cpt(pos + whitespaces) != kOutOfRange) {
                pos += whitespaces - 1;
                add_token(pos);
                continue;
            }
            // \s+
            if (whitespaces > 0) {
                pos += whitespaces;
                add_token(pos);
                continue;
            }
            // no match: one codepoint
            add_token(++pos);
        }

        words.reserve(words.size() + spans.size());
        for (const auto & span : spans) {
            std::string word;
            for (size_t i = span.first; i < span.second; ++i) word += cpt_to_utf8(cpts[i]);
            words.push_back(byte_encode_word(word));
        }
    }

    // GPT-2 pre-tokenizer, used when tokenizer.ggml.pre is "gpt2" (same 256-byte
    // vocabulary family, different split rule). qwen4exp uses qwen2 / qwen35.
    void pre_tokenize_gpt2(const std::vector<uint32_t> & cpts, std::vector<std::string> & words) const {
        const size_t n = cpts.size();
        auto flags_at = [&](size_t i) -> uint16_t { return i < n ? cpt_flags(cpts[i]) : 0; };
        size_t pos = 0;
        while (pos < n) {
            const size_t start = pos;
            const uint16_t flags = flags_at(pos);
            const uint32_t cpt = cpts[pos];
            // 's|'t|'re|'ve|'m|'ll|'d
            bool matched = false;
            if (cpt == '\'' && pos + 1 < n) {
                const uint32_t next = to_lower(cpts[pos + 1]);
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    pos += 2;
                    matched = true;
                } else if (pos + 2 < n && ((next == 'r' && to_lower(cpts[pos + 2]) == 'e') ||
                                           (next == 'v' && to_lower(cpts[pos + 2]) == 'e') ||
                                           (next == 'l' && to_lower(cpts[pos + 2]) == 'l'))) {
                    pos += 3;
                    matched = true;
                }
            }
            const bool space = (cpt == ' ');
            const uint16_t body = space ? flags_at(pos + 1) : flags;
            // ?\p{L}+ / ?\p{N}+ (the leading space belongs to the run)
            if (!matched && (is_letter(body) || is_number(body))) {
                if (space) ++pos;
                if (is_letter(body)) {
                    while (pos < n && is_letter(flags_at(pos))) ++pos;
                } else {
                    while (pos < n && is_number(flags_at(pos))) ++pos;
                }
                matched = true;
            }
            // ?[^\s\p{L}\p{N}]+
            if (!matched) {
                const uint16_t body2 = space ? flags_at(pos + 1) : flags;
                if (!is_ws(body2) && !is_letter(body2) && !is_number(body2) && body2 != 0) {
                    if (space) ++pos;
                    while (pos < n) {
                        const uint16_t f = flags_at(pos);
                        if (is_ws(f) || is_letter(f) || is_number(f) || f == 0) break;
                        ++pos;
                    }
                    matched = true;
                }
            }
            // \s+(?!\S)  /  \s+
            if (!matched && is_ws(flags)) {
                size_t run = 1;
                while (pos + run < n && is_ws(flags_at(pos + run))) ++run;
                if (run > 1 && pos + run < n) --run;   // \s+(?!\S) keeps one back
                pos += run;
                matched = true;
            }
            if (!matched) ++pos;
            std::string word;
            for (size_t i = start; i < pos; ++i) word += cpt_to_utf8(cpts[i]);
            words.push_back(byte_encode_word(word));
        }
    }
// ---- BPE (llm_tokenizer_bpe_session) ----

    int32_t find_rank(const std::string & left, const std::string & right) const {
        const auto it = bpe_ranks_.find(qwen38_merge_key(left, right));
        return it == bpe_ranks_.end() ? -1 : it->second;
    }

    void encode_word(const std::string & word, std::vector<int32_t> & output) const {
        struct Symbol {
            size_t offset = 0;
            size_t length = 0;
            int prev = -1;
            int next = -1;
        };
        struct Bigram {
            int left = -1;
            int right = -1;
            int rank = 0;
            std::string text;   // concatenation when the bigram was queued
        };
        // min-heap on (rank, left): llama.cpp's llm_bigram_bpe::comparator.
        struct Compare {
            bool operator()(const Bigram & a, const Bigram & b) const {
                return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
            }
        };

        if (word.empty()) return;
        std::vector<Symbol> symbols;
        symbols.reserve(word.size());
        for (size_t offset = 0; offset < word.size();) {
            const size_t length = std::min(word.size() - offset, utf8_len(word[offset]));
            Symbol symbol;
            symbol.offset = offset;
            symbol.length = length;
            symbol.prev = static_cast<int>(symbols.size()) - 1;
            symbol.next = offset + length >= word.size() ? -1 : static_cast<int>(symbols.size()) + 1;
            symbols.push_back(symbol);
            offset += length;
        }
        std::priority_queue<Bigram, std::vector<Bigram>, Compare> queue;
        auto symbol_text = [&](int index) {
            return std::string(word, symbols[static_cast<size_t>(index)].offset,
                               symbols[static_cast<size_t>(index)].length);
        };
        auto add_bigram = [&](int left, int right) {
            if (left < 0 || right < 0) return;
            const std::string left_text = symbol_text(left);
            const std::string right_text = symbol_text(right);
            const int32_t rank = find_rank(left_text, right_text);
            if (rank < 0) return;
            Bigram bigram;
            bigram.left = left;
            bigram.right = right;
            bigram.rank = rank;
            bigram.text = left_text + right_text;
            queue.push(std::move(bigram));
        };
        for (size_t i = 1; i < symbols.size(); ++i) add_bigram(static_cast<int>(i) - 1, static_cast<int>(i));
        while (!queue.empty()) {
            const Bigram bigram = queue.top();
            queue.pop();
            Symbol & left = symbols[static_cast<size_t>(bigram.left)];
            Symbol & right = symbols[static_cast<size_t>(bigram.right)];
            if (left.length == 0 || right.length == 0) continue;
            // Stale bigram: the two symbols no longer concatenate to bigram.text.
            if (symbol_text(bigram.left) + symbol_text(bigram.right) != bigram.text) continue;
            left.length += right.length;
            right.length = 0;
            left.next = right.next;
            if (right.next >= 0) symbols[static_cast<size_t>(right.next)].prev = bigram.left;
            add_bigram(left.prev, bigram.left);
            add_bigram(bigram.left, left.next);
        }

        for (const Symbol & symbol : symbols) {
            if (symbol.length == 0) continue;
            const std::string text = word.substr(symbol.offset, symbol.length);
            const auto it = token_to_id_.find(text);
            if (it != token_to_id_.end()) {
                output.push_back(it->second);
                continue;
            }
            // Byte fallback: every byte of the symbol is looked up on its own
            // (llama.cpp does the same, and drops bytes with no token).
            for (size_t i = 0; i < text.size(); ++i) {
                const auto byte_it = token_to_id_.find(std::string(1, text[i]));
                if (byte_it != token_to_id_.end()) output.push_back(byte_it->second);
            }
        }
    }

    void encode_text(const std::string & text, std::vector<int32_t> & output) const {
        std::vector<std::string> words;
        pre_tokenize(text, words);
        for (const std::string & word : words) encode_word(word, output);
    }

    // tokenizer_st_partition(): split control / user-defined / unknown tokens
    // out of the raw text, longest special text first.
    void partition(const std::string & text, bool parse_special, std::vector<Fragment> & fragments) const {
        for (int32_t special : special_ids_) {
            const int type = token_types_[static_cast<size_t>(special)];
            // Control / unknown tokens are only parsed when asked for;
            // user-defined tokens are always split out (llama.cpp comment).
            if (!parse_special && (type == kControl || type == kUnknown)) continue;
            const std::string & needle = tokens_[static_cast<size_t>(special)];
            if (needle.empty()) continue;
            for (size_t index = 0; index < fragments.size(); ++index) {
                if (!fragments[index].is_text) continue;
                size_t base_offset = fragments[index].offset;
                size_t base_length = fragments[index].length;
                std::vector<Fragment> inserted;
                bool hit = false;
                for (;;) {
                    const size_t match = text.find(needle, base_offset);
                    if (match == std::string::npos || match + needle.size() > base_offset + base_length) break;
                    if (match > base_offset) inserted.push_back(Fragment::raw(base_offset, match - base_offset));
                    inserted.push_back(Fragment::special(special));
                    hit = true;
                    const size_t right_offset = match + needle.size();
                    const size_t right_length = base_length - ((match - base_offset) + needle.size());
                    base_offset = right_offset;
                    base_length = right_length;
                    if (right_length == 0) break;
                }
                if (!hit) continue;
                // The text after the last match is the tail of the fragment; it is
                // emitted here because the loop above only emits what precedes a
                // match.
                if (base_length > 0) inserted.push_back(Fragment::raw(base_offset, base_length));
                fragments.erase(fragments.begin() + static_cast<std::ptrdiff_t>(index));
                fragments.insert(fragments.begin() + static_cast<std::ptrdiff_t>(index),
                                 inserted.begin(), inserted.end());
                index += inserted.size() - 1;
            }
        }
    }

    std::string arch_;
    std::string model_;
    std::string pre_;
    PreType pre_type_ = PreType::Qwen2;
    std::vector<std::string> tokens_;
    std::vector<int> token_types_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    bool spm_ = false;                     // tokenizer.ggml.model == "llama"
    std::vector<double> scores_;           // tokenizer.ggml.scores (SPM only)
    std::unordered_map<std::string, int32_t> bpe_ranks_;
    size_t merge_count_ = 0;
    std::vector<int32_t> special_ids_;
    std::vector<int32_t> stop_ids_;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
    int32_t pad_id_ = -1;
    bool add_bos_ = false;
    bool add_eos_ = false;
    std::string chat_template_;
};

// Incremental printer: byte-level BPE pieces can split a UTF-8 sequence across
// tokens, so incomplete trailing codepoints are held back until they complete.
// This is what lets the CLI stream Japanese / emoji text one token at a time
// without printing broken bytes.
class Qwen38StreamDecoder {
public:
    // Returns the bytes that are safe to print now (possibly empty).
    std::string push(const Qwen38Tokenizer & tokenizer, int32_t token, bool skip_special) {
        if (skip_special && tokenizer.is_special(token)) return std::string();
        pending_ += tokenizer.piece(token);
        return take_complete();
    }

    std::string flush() {
        const std::string out = pending_;
        pending_.clear();
        return out;
    }

    const std::string & pending() const { return pending_; }

private:
    static size_t utf8_len(char lead) {
        static const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        return lookup[static_cast<uint8_t>(lead) >> 4];
    }

    std::string take_complete() {
        size_t complete = 0;
        size_t offset = 0;
        while (offset < pending_.size()) {
            const size_t length = utf8_len(pending_[offset]);
            if (offset + length > pending_.size()) break;
            bool valid = true;
            for (size_t i = 1; valid && i < length; ++i) {
                if ((static_cast<uint8_t>(pending_[offset + i]) & 0xc0) != 0x80) valid = false;
            }
            if (!valid) {
                // Stray continuation / bad lead byte: emit it instead of
                // stalling the stream forever.
                complete = ++offset;
                continue;
            }
            offset += length;
            complete = offset;
        }
        const std::string out = pending_.substr(0, complete);
        pending_.erase(0, complete);
        return out;
    }

    std::string pending_;
};

} // namespace wd
