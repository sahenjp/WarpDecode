// llama.cpp driver for the tokenizer comparison: exactly the API surface the
// pinned llama.cpp exposes, dumped in the same JSONL format as
// tests/qwen38_tokenizer_host_test.cpp so the two can be diffed.
//
// Usage: qwen38_tokenizer_oracle <llama.cpp checkout unused> <model.gguf> [--corpus file]
//
// Build and run through tests/qwen38_tokenizer_host_check.sh; the llama.cpp
// build directory is passed with -DLLAMA_ORACLE_BUILD=<checkout>/build-host.
#include <llama.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string corpus = "tests/qwen38_tokenizer_corpus.txt";
    // Single-prompt mode: the llama-tokenize CLI subset (-m/-f/--ids/--no-bos),
    // so tests/qwen38_tokenizer_compare (or tokenizer_compare.py) can drive this
    // harness instead of llama-tokenize. llama-tokenize itself cannot be used for
    // qwen4exp: it always creates a llama_context, and the qwen4exp graph needs
    // ctx_other, so "could not create context" aborts it before tokenizing.
    std::string prompt_file;
    bool single = false;
    bool ids = false;
    bool no_bos = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) { model = argv[++i]; single = true; }
        else if ((arg == "-f" || arg == "--file") && i + 1 < argc) { prompt_file = argv[++i]; single = true; }
        else if (arg == "--ids") ids = true;
        else if (arg == "--no-bos") no_bos = true;
        else if (arg == "--no-escape") { /* prompts are read verbatim */ }
        else if (arg == "--corpus" && i + 1 < argc) corpus = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: qwen38_tokenizer_oracle <model.gguf> [--corpus file]\n"
                         "       qwen38_tokenizer_oracle -m <model.gguf> -f <prompt-file> --ids [--no-bos]\n";
            return 0;
        } else if (model.empty()) model = arg;
        else { std::cerr << "unexpected argument: " << arg << "\n"; return 2; }
    }
    if (model.empty()) {
        std::cerr << "usage: qwen38_tokenizer_oracle <model.gguf> [--corpus file]\n";
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * llama = llama_model_load_from_file(model.c_str(), model_params);
    if (llama == nullptr) {
        std::cerr << "cannot load vocabulary from " << model << "\n";
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(llama);

    if (single) {
        std::ifstream file(prompt_file, std::ios::binary);
        if (!file) {
            std::cerr << "cannot open prompt file " << prompt_file << "\n";
            return 2;
        }
        std::string prompt((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const bool add_bos = llama_vocab_get_add_bos(vocab) && !no_bos;
        std::vector<llama_token> tokens(prompt.size() + 64);
        int32_t count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                       tokens.data(), static_cast<int32_t>(tokens.size()),
                                       add_bos, /*parse_special=*/true);
        if (count < 0) {
            tokens.resize(static_cast<size_t>(-count));
            count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                   tokens.data(), static_cast<int32_t>(tokens.size()),
                                   add_bos, /*parse_special=*/true);
        }
        if (count < 0) throw std::runtime_error("llama_tokenize failed");
        if (!ids) {
            for (int32_t i = 0; i < count; ++i) {
                std::cout << (i ? " " : "") << tokens[static_cast<size_t>(i)];
            }
            std::cout << "\n";
            return 0;
        }
        std::cout << "[";
        for (int32_t i = 0; i < count; ++i) {
            if (i) std::cout << ", ";
            std::cout << tokens[static_cast<size_t>(i)];
        }
        std::cout << "]\n";
        llama_model_free(llama);
        llama_backend_free();
        return 0;
    }

    const std::vector<std::string> lines = read_corpus(corpus);

    for (size_t index = 0; index < lines.size(); ++index) {
        const std::string & text = lines[index];
        for (int add_special = 0; add_special <= 1; ++add_special) {
            for (int parse_special = 0; parse_special <= 1; ++parse_special) {
                std::vector<llama_token> tokens(text.size() + 64);
                int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                               tokens.data(), static_cast<int32_t>(tokens.size()),
                                               add_special != 0, parse_special != 0);
                if (count < 0) {
                    tokens.resize(static_cast<size_t>(-count));
                    count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                           tokens.data(), static_cast<int32_t>(tokens.size()),
                                           add_special != 0, parse_special != 0);
                }
                if (count < 0) throw std::runtime_error("llama_tokenize failed");
                tokens.resize(static_cast<size_t>(count));
                std::cout << "{\"text\":\"" << json_escape(text)
                          << "\",\"add_special\":" << add_special
                          << ",\"parse_special\":" << parse_special
                          << ",\"tokens\":[";
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << tokens[i];
                }
                std::cout << "]}\n";
            }
        }
    }
    std::cout << "# oracle vocab=" << llama_vocab_n_tokens(vocab)
              << " add_bos=" << (llama_vocab_get_add_bos(vocab) ? 1 : 0)
              << " bos=" << llama_vocab_bos(vocab)
              << " eos=" << llama_vocab_eos(vocab) << "\n";
    llama_model_free(llama);
    llama_backend_free();
    return 0;
}
