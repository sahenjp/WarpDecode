// Host driver for src/qwen38_tokenizer.cuh: encode/decode + JSONL dump.
//
// Prints one JSON object per (corpus line, add_special, parse_special) case so
// that tests/qwen38_tokenizer_oracle.cpp - the same dump produced through the
// real llama.cpp vocabulary - can be diffed line by line:
//
//   {"text":"...","add_special":1,"parse_special":1,"tokens":[...]}
//
// The corpus is one test string per line; \n \t \r \\ are unescaped.
// Round-trip results (decode(encode(s)) == s) are reported as comments and in
// the final "# summary" line, not in the JSONL stream.
//
// Usage: qwen38_tokenizer_host_test <model.gguf> [--corpus file]
#include "qwen38_tokenizer.cuh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string unescape(const std::string & line) {
    std::string out;
    out.reserve(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] != '\\' || i + 1 >= line.size()) {
            out.push_back(line[i]);
            continue;
        }
        const char next = line[++i];
        switch (next) {
            case 'n': out.push_back('\n'); break;
            case 't': out.push_back('\t'); break;
            case 'r': out.push_back('\r'); break;
            case '\\': out.push_back('\\'); break;
            default:
                out.push_back('\\');
                out.push_back(next);
                break;
        }
    }
    return out;
}

std::string json_escape(const std::string & text) {
    std::string out;
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::vector<std::string> read_corpus(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open corpus: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(unescape(line));
    }
    return lines;
}

void print_tokens(const std::vector<int32_t> & tokens) {
    std::cout << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens[i];
    }
    std::cout << ']';
}

// True when the byte string is a well formed UTF-8 sequence: the CLI prints one
// chunk per token, so every chunk has to be complete on its own.
bool is_valid_utf8(const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        const uint8_t lead = static_cast<uint8_t>(text[offset]);
        size_t length = 1;
        if (lead < 0x80) length = 1;
        else if ((lead & 0xe0) == 0xc0) length = 2;
        else if ((lead & 0xf0) == 0xe0) length = 3;
        else if ((lead & 0xf8) == 0xf0) length = 4;
        else return false;
        if (offset + length > text.size()) return false;
        for (size_t i = 1; i < length; ++i) {
            if ((static_cast<uint8_t>(text[offset + i]) & 0xc0) != 0x80) return false;
        }
        offset += length;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string corpus = "tests/qwen38_tokenizer_corpus.txt";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) corpus = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: qwen38_tokenizer_host_test <model.gguf> [--corpus file]\n";
            return 0;
        } else if (model.empty()) model = arg;
        else { std::cerr << "unexpected argument: " << arg << "\n"; return 2; }
    }
    if (model.empty()) {
        std::cerr << "usage: qwen38_tokenizer_host_test <model.gguf> [--corpus file]\n";
        return 2;
    }

    wd::Qwen38Tokenizer tokenizer(model, /*verbose=*/true);
    std::vector<std::string> lines;
    try {
        lines = read_corpus(corpus);
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 2;
    }

    size_t cases = 0;
    size_t round_trips = 0;
    size_t round_trip_failures = 0;
    size_t stream_round_trips = 0;
    size_t stream_failures = 0;
    size_t stream_chunk_failures = 0;
    for (size_t index = 0; index < lines.size(); ++index) {
        const std::string & text = lines[index];
        for (int add_special = 0; add_special <= 1; ++add_special) {
            for (int parse_special = 0; parse_special <= 1; ++parse_special) {
                const std::vector<int32_t> tokens =
                    tokenizer.encode(text, add_special != 0, parse_special != 0);
                std::cout << "{\"text\":\"" << json_escape(text)
                          << "\",\"add_special\":" << add_special
                          << ",\"parse_special\":" << parse_special
                          << ",\"tokens\":";
                print_tokens(tokens);
                std::cout << "}\n";
                ++cases;
            }
        }
        // Round trip on the raw text path (no BOS/EOS, specials kept verbatim).
        const std::vector<int32_t> raw = tokenizer.encode(text, false, true);
        const std::string decoded = tokenizer.decode(raw, /*skip_special=*/false);
        ++round_trips;
        if (decoded != text) {
            ++round_trip_failures;
            std::cout << "# roundtrip mismatch line " << index << ": in=\""
                      << json_escape(text) << "\" out=\"" << json_escape(decoded) << "\"\n";
        }
        // Streaming decode: exactly the path the CLI prints through. Every chunk
        // must be valid UTF-8 by itself and the concatenation must be the text.
        std::string streamed;
        {
            wd::Qwen38StreamDecoder printer;
            for (int32_t token : raw) {
                const std::string chunk = printer.push(tokenizer, token, /*skip_special=*/false);
                if (!is_valid_utf8(chunk)) {
                    ++stream_chunk_failures;
                    std::cout << "# streaming chunk is not valid UTF-8 (line " << index << ")\n";
                }
                streamed += chunk;
            }
            streamed += printer.flush();
        }
        ++stream_round_trips;
        if (streamed != text) {
            ++stream_failures;
            std::cout << "# streaming mismatch line " << index << ": in=\""
                      << json_escape(text) << "\" out=\"" << json_escape(streamed) << "\"\n";
        }
    }
    // Chat template: the CLI applies the Qwen shape when the GGUF carries one.
    if (tokenizer.supports_qwen_chat_template()) {
        const std::string rendered = tokenizer.apply_qwen_chat_template(
            {{"user", "こんにちは。自己紹介してください。"}}, /*add_generation_prompt=*/true);
        const std::vector<int32_t> chat_tokens = tokenizer.encode(rendered, false, true);
        std::cout << "# chat_template supported=yes rendered=" << json_escape(rendered) << "\n";
        std::cout << "# chat_template tokens=";
        print_tokens(chat_tokens);
        std::cout << "\n";
    } else {
        std::cout << "# chat_template supported=no\n";
    }

    std::cout << "# summary cases=" << cases
              << " round_trips=" << round_trips
              << " round_trip_failures=" << round_trip_failures
              << " stream_round_trips=" << stream_round_trips
              << " stream_failures=" << stream_failures
              << " stream_chunk_failures=" << stream_chunk_failures
              << " vocab=" << tokenizer.vocab_size()
              << " merges=" << tokenizer.merge_count()
              << " pre=" << tokenizer.pre_type_name() << "\n";
    return (round_trip_failures == 0 && stream_failures == 0 && stream_chunk_failures == 0) ? 0 : 1;
}
