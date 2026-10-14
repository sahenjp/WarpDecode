// Task L: llama.cpp decode A/B harness.
//
// Runs the same batch=1 greedy decode as the parity reference (start token 1,
// 64 generated tokens, ctx 128, n_batch=n_ubatch=1, F16 KV, no sampler) but on
// the llama.cpp CUDA build, so the warp engine and llama.cpp can be measured on
// one machine with one model file.
//
//   taskl-llama-decode MODEL CTX N_PREDICT NGL [options]
//
// Options:
//   --n-layer N          layer count used to expand --keep-moe-gpu-last (48)
//   --keep-moe-gpu-last K  keep MoE expert tensors of the last K layers on GPU,
//                          all other layers' experts on the CPU backend
//   --no-mmap            LLAMA_LOAD_MODE_NONE instead of MMAP
//   --direct-io          LLAMA_LOAD_MODE_DIRECT_IO
//   --threads T          CPU threads for the host-resident parts (nproc)
//   --reps R             number of full 64-token rounds (3)
//   --start-token T      first input token id (1)
//   --flash 0|1          flash attention disabled (0, default) or enabled (1)
//   --moe-cpu-all        1 = every layer's experts on CPU (default when
//                        --keep-moe-gpu-last is absent)
//
// One LLAMA_AB= JSON line per round plus a final LLAMA_AB_LOAD= line.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string model;
    int ctx = 128;
    int n_predict = 64;
    int n_gpu_layers = 0;
    int n_layer = 48;
    int keep_moe_gpu_last = -1;
    int load_mode = LLAMA_LOAD_MODE_MMAP;
    int threads = 12;
    int reps = 3;
    int start_token = 1;
    int flash = 0;
};

static std::string join_ids(const std::vector<llama_token> & ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) { if (i) out << ','; out << ids[i]; }
    return out.str();
}

static int matched_prefix(const std::vector<llama_token> & ids, const std::vector<llama_token> & ref) {
    const size_t n = std::min(ids.size(), ref.size());
    for (size_t i = 0; i < n; ++i) if (ids[i] != ref[i]) return static_cast<int>(i);
    return static_cast<int>(n);
}

int main(int argc, char ** argv) {
    try {
        Options o;
        if (argc < 5) {
            std::cerr << "usage: taskl-llama-decode MODEL CTX N_PREDICT NGL [--n-layer N]"
                         " [--keep-moe-gpu-last K] [--no-mmap] [--threads T] [--reps R]"
                         " [--start-token T] [--flash 0|1]\n";
            return 2;
        }
        o.model = argv[1];
        o.ctx = std::stoi(argv[2]);
        o.n_predict = std::stoi(argv[3]);
        o.n_gpu_layers = std::stoi(argv[4]);
        for (int i = 5; i < argc; ++i) {
            const std::string s = argv[i];
            auto need = [&](const char * what) {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
                return std::string(argv[++i]);
            };
            if (s == "--n-layer") o.n_layer = std::stoi(need("--n-layer"));
            else if (s == "--keep-moe-gpu-last") o.keep_moe_gpu_last = std::stoi(need("--keep-moe-gpu-last"));
            else if (s == "--no-mmap") o.load_mode = LLAMA_LOAD_MODE_NONE;
            else if (s == "--direct-io") o.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
            else if (s == "--threads") o.threads = std::stoi(need("--threads"));
            else if (s == "--reps") o.reps = std::stoi(need("--reps"));
            else if (s == "--start-token") o.start_token = std::stoi(need("--start-token"));
            else if (s == "--flash") o.flash = std::stoi(need("--flash"));
            else throw std::runtime_error("unknown argument: " + s);
        }

        std::vector<std::string> patterns;
        std::vector<llama_model_tensor_buft_override> overrides;
        const int gpu_first_moe_layer = (o.keep_moe_gpu_last < 0)
            ? o.n_layer
            : std::max(0, o.n_layer - o.keep_moe_gpu_last);
        for (int layer = 0; layer < gpu_first_moe_layer; ++layer) {
            patterns.push_back("blk\\." + std::to_string(layer) + "\\.ffn_.*_exps");
        }
        for (const std::string & p : patterns) {
            overrides.push_back({p.c_str(), ggml_backend_cpu_buffer_type()});
        }
        overrides.push_back({nullptr, nullptr});

        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = o.n_gpu_layers;
        mp.load_mode = static_cast<llama_load_mode>(o.load_mode);
        mp.load_mtp = false;
        mp.tensor_buft_overrides = overrides.data();

        const auto load_begin = std::chrono::steady_clock::now();
        llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_begin).count();
        if (!model) throw std::runtime_error("llama_model_load_from_file failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::cout << "LLAMA_AB_LOAD={\"harness\":\"llama.cpp\",\"commit\":\"__COMMIT__\""
                  << ",\"n_gpu_layers\":" << o.n_gpu_layers
                  << ",\"moe_cpu_layers\":" << gpu_first_moe_layer
                  << ",\"keep_moe_gpu_last\":" << o.keep_moe_gpu_last
                  << ",\"load_mode\":\"" << llama_load_mode_name(static_cast<llama_load_mode>(o.load_mode)) << "\""
                  << ",\"n_layer\":" << o.n_layer
                  << ",\"threads\":" << o.threads
                  << ",\"vocab\":" << vocab
                  << ",\"ctx\":" << o.ctx
                  << ",\"n_predict\":" << o.n_predict
                  << ",\"start_token\":" << o.start_token
                  << ",\"flash\":" << o.flash
                  << ",\"load_seconds\":" << std::fixed << std::setprecision(6) << load_seconds
                  << "}" << std::endl;

        for (int round = 0; round < o.reps; ++round) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = static_cast<uint32_t>(o.ctx);
            cp.n_batch = 1;
            cp.n_ubatch = 1;
            cp.n_seq_max = 1;
            cp.flash_attn_type = o.flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = GGML_TYPE_F16;
            cp.type_v = GGML_TYPE_F16;
            cp.n_threads = o.threads;
            cp.n_threads_batch = o.threads;
            const auto ctx_begin = std::chrono::steady_clock::now();
            llama_context * ctx = llama_init_from_model(model, cp);
            const double ctx_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ctx_begin).count();
            if (!ctx) throw std::runtime_error("llama_init_from_model failed");

            std::vector<llama_token> generated;
            std::vector<double> token_ms;
            generated.reserve(static_cast<size_t>(o.n_predict));
            token_ms.reserve(static_cast<size_t>(o.n_predict));
            llama_token token = static_cast<llama_token>(o.start_token);
            double total = 0.0;
            for (int step = 0; step < o.n_predict; ++step) {
                if (token < 0 || token >= vocab) throw std::runtime_error("input token outside vocabulary");
                const auto begin = std::chrono::steady_clock::now();
                llama_batch batch = llama_batch_get_one(&token, 1);
                const int rc = llama_decode(ctx, batch);
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
                token_ms.push_back(ms);
                total += ms;
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (!logits) throw std::runtime_error("llama_get_logits_ith returned null");
                llama_token best_token = 0;
                float best = -std::numeric_limits<float>::infinity();
                for (int t = 0; t < vocab; ++t) {
                    const float value = logits[t];
                    if (!std::isfinite(value)) throw std::runtime_error("non-finite logits");
                    if (value > best || (value == best && t < best_token)) { best = value; best_token = t; }
                }
                generated.push_back(best_token);
                token = best_token;
            }
            const double seconds = total / 1000.0;
            std::cout << "LLAMA_AB={\"harness\":\"llama.cpp\",\"commit\":\"__COMMIT__\""
                      << ",\"round\":" << round
                      << ",\"n_gpu_layers\":" << o.n_gpu_layers
                      << ",\"moe_cpu_layers\":" << gpu_first_moe_layer
                      << ",\"load_mode\":\"" << llama_load_mode_name(static_cast<llama_load_mode>(o.load_mode)) << "\""
                      << ",\"threads\":" << o.threads
                      << ",\"ctx\":" << o.ctx
                      << ",\"decode_seconds\":" << std::fixed << std::setprecision(6) << seconds
                      << ",\"decode_tokens\":" << generated.size()
                      << ",\"tok_s\":" << std::setprecision(3) << (generated.size() / seconds)
                      << ",\"ctx_seconds\":" << std::setprecision(6) << ctx_seconds
                      << ",\"token_ms\":[";
            for (size_t i = 0; i < token_ms.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << std::setprecision(3) << token_ms[i];
            }
            std::cout << "],\"token_ids\":[" << join_ids(generated) << "]}" << std::endl;
            llama_free(ctx);
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
