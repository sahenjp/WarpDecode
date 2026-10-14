#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TraceStats {
    size_t count = 0;
    float min = 0.0f;
    float max = 0.0f;
    float max_abs = 0.0f;
    double mean = 0.0;
    double l2 = 0.0;
    double rms = 0.0;
    uint64_t checksum = 14695981039346656037ull;
};

struct TraceVector {
    std::string name;
    int type_id = -1;
    std::array<int64_t, 4> dims{};
    std::vector<float> values;
    TraceStats stats;
};

struct TraceSummary {
    std::string name;
    std::array<int64_t, 4> dims{};
    TraceStats stats;
};

struct TraceIntegers {
    std::string name;
    std::vector<int64_t> values;
};

struct TraceCapture {
    int position = 0;
    int trace_position = -1;
    int qsa_input_override_position = -1;
    int qsa_input_override_layer = 35;
    std::vector<float> qsa_input_override;
    bool qsa_input_override_applied = false;
    size_t qsa_input_override_count = 0;
    std::vector<TraceVector> vectors;
    std::vector<TraceSummary> summaries;
    std::vector<TraceIntegers> integer_vectors;
};

static bool named_tensor(const char * name, const char * base) {
    const size_t n = std::strlen(base);
    if (std::strncmp(name, base, n) != 0) return false;
    if (name[n] == '\0') return true;
    if (name[n] != '-') return false;
    for (const char * p = name + n + 1; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    return name[n + 1] != '\0';
}

static bool named_layer_tensor(const char * name, const char * base, int layer) {
    return std::string(name) == std::string(base) + "-" + std::to_string(layer);
}

static bool trace_wanted(const char * name) {
    static const char * const bases[] = {
        "model.input_embed", "hc_init", "hc_norm", "hc_low_pre", "hc_low",
        "hc_gate", "hc_gate_sigmoid", "hc_mixed", "hc_inject",
        "l_last", "attn_output", "hc_combine", "ffn_moe_out", "ffn_moe_down", "ffn_shexp",
        "shared_expert_gate", "shared_expert_gate_sigmoid", "ffn_shexp_gated", "ffn_out",
        "ple_embd", "ple_gate",
        "ple_gated_value", "ple_conv_out", "conv_state_at", "linear_attn_qkv_mixed", "z",
        "beta", "beta_sigmoid", "alpha", "gate", "state_predelta", "new_state",
        "conv_output_silu", "gdn_core_output", "final_output", "linear_attn_out", "indexer_k_raw",
        "indexer_k_pooled", "indexer_k", "indexer_q", "indexer_score", "indexer_top_k",
        "qsa_k_write_indices", "qsa_v_write_indices", "qsa_k_cache", "qsa_v_cache",
        "qsa_k_store", "qsa_v_store", "q_conv-8", "k_conv-8", "v_conv-8",
        "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
        "Qcur", "Kcur", "Vcur", "attn_pregate", "attn_gated", "ffn_moe_logits", "ffn_moe_topk",
        "ffn_moe_weights", "ffn_moe_weights_norm", "ffn_moe_weights_scaled",
        "result_norm", "result_output",
    };
    for (const char * base : bases) if (named_tensor(name, base)) return true;
    return false;
}

static bool summarize_only(const char * name) {
    static const int full_state_layers[] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    const std::string current(name);
    for (int layer : full_state_layers) {
        const std::string suffix = "-" + std::to_string(layer);
        if (current == std::string("conv_state_at") + suffix ||
            current == std::string("state_predelta") + suffix ||
            current == std::string("new_state") + suffix) return false;
    }
    return named_tensor(name, "conv_state_at") || named_tensor(name, "state_predelta") ||
           named_tensor(name, "new_state");
}

template <typename F>
static void visit_values(const ggml_tensor * tensor, const std::vector<uint8_t> & raw, F visit) {
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t off = static_cast<size_t>(i0) * tensor->nb[0]
                                     + static_cast<size_t>(i1) * tensor->nb[1]
                                     + static_cast<size_t>(i2) * tensor->nb[2]
                                     + static_cast<size_t>(i3) * tensor->nb[3];
                    float value = 0.0f;
                    if (tensor->type == GGML_TYPE_F32) {
                        std::memcpy(&value, raw.data() + off, sizeof(value));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        ggml_fp16_t half;
                        std::memcpy(&half, raw.data() + off, sizeof(half));
                        value = ggml_fp16_to_fp32(half);
                    } else {
                        ggml_bf16_t half;
                        std::memcpy(&half, raw.data() + off, sizeof(half));
                        value = ggml_bf16_to_fp32(half);
                    }
                    visit(value);
                }
            }
        }
    }
}

static void accumulate(TraceStats & stats, float value, double & sum, double & square_sum) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite CPU trace value");
    if (stats.count == 0) stats.min = stats.max = value;
    else {
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
    }
    stats.max_abs = std::max(stats.max_abs, std::abs(value));
    sum += value;
    square_sum += static_cast<double>(value) * value;
    const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
    for (size_t i = 0; i < sizeof(value); ++i) {
        stats.checksum ^= bytes[i];
        stats.checksum *= 1099511628211ull;
    }
    ++stats.count;
}

static void finish_stats(TraceStats & stats, double sum, double square_sum) {
    if (!stats.count) return;
    stats.mean = sum / static_cast<double>(stats.count);
    stats.l2 = std::sqrt(square_sum);
    stats.rms = std::sqrt(square_sum / static_cast<double>(stats.count));
}

static std::string trace_name(int position, const char * name) {
    return "pos" + std::to_string(position) + "." + name;
}

static bool capture_trace(ggml_tensor * tensor, bool ask, void * user_data) {
    if (ask || !trace_wanted(tensor->name)) return true;
    TraceCapture & capture = *static_cast<TraceCapture *>(user_data);
    const bool layer7_qsa_history = named_tensor(tensor->name, "qsa_k_store-7") ||
                                    named_tensor(tensor->name, "qsa_v_store-7");
    const bool layer35_qsa_history = named_tensor(tensor->name, "qsa_k_store-35") ||
                                     named_tensor(tensor->name, "qsa_v_store-35");
    const bool layer8_route_series =
        named_layer_tensor(tensor->name, "ffn_moe_logits", 8) ||
        named_layer_tensor(tensor->name, "ffn_moe_topk", 8) ||
        named_layer_tensor(tensor->name, "ffn_moe_weights", 8) ||
        named_layer_tensor(tensor->name, "ffn_moe_weights_norm", 8) ||
        named_layer_tensor(tensor->name, "ffn_moe_weights_scaled", 8) ||
        named_layer_tensor(tensor->name, "ffn_moe_out", 8) ||
        named_layer_tensor(tensor->name, "l_last", 8);
    const bool layer_output_series = named_tensor(tensor->name, "l_last");
    if (capture.trace_position >= 0 && capture.position != capture.trace_position &&
        !(capture.position <= capture.trace_position && (layer8_route_series || layer_output_series)) &&
        !layer7_qsa_history && !layer35_qsa_history) return true;
    if (!capture.qsa_input_override_applied && !capture.qsa_input_override.empty() &&
        capture.position == capture.qsa_input_override_position &&
        named_layer_tensor(tensor->name, "hc_mixed", capture.qsa_input_override_layer)) {
        if (tensor->type != GGML_TYPE_F32 ||
            capture.qsa_input_override.size() != static_cast<size_t>(ggml_nelements(tensor))) {
            throw std::runtime_error("attention input override requires a matching F32 HC mix vector");
        }
        ggml_backend_tensor_set(tensor, capture.qsa_input_override.data(), 0,
                                ggml_nbytes(tensor));
        capture.qsa_input_override_applied = true;
        ++capture.qsa_input_override_count;
    }
    const std::string name = trace_name(capture.position, tensor->name);
    const int64_t n = ggml_nelements(tensor);
    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> raw(nbytes);
    if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);
    } else {
        std::memcpy(raw.data(), tensor->data, nbytes);
    }

    if (tensor->type == GGML_TYPE_I32 && named_tensor(tensor->name, "ffn_moe_topk")) {
        TraceVector vector;
        vector.name = name;
        vector.type_id = static_cast<int>(tensor->type);
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), vector.dims.begin());
        vector.values.reserve(static_cast<size_t>(n));
        double sum = 0.0, square_sum = 0.0;
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                        const size_t off = static_cast<size_t>(i0) * tensor->nb[0]
                                         + static_cast<size_t>(i1) * tensor->nb[1]
                                         + static_cast<size_t>(i2) * tensor->nb[2]
                                         + static_cast<size_t>(i3) * tensor->nb[3];
                        int32_t value = 0;
                        std::memcpy(&value, raw.data() + off, sizeof(value));
                        const float converted = static_cast<float>(value);
                        accumulate(vector.stats, converted, sum, square_sum);
                        vector.values.push_back(converted);
                    }
                }
            }
        }
        finish_stats(vector.stats, sum, square_sum);
        capture.vectors.push_back(std::move(vector));
        return true;
    }

    if (tensor->type == GGML_TYPE_I32 &&
        (named_tensor(tensor->name, "indexer_top_k") ||
         named_tensor(tensor->name, "qsa_k_write_indices") ||
         named_tensor(tensor->name, "qsa_v_write_indices") ||
         named_tensor(tensor->name, "ffn_moe_topk"))) {
        TraceIntegers values{name, {}};
        values.values.reserve(static_cast<size_t>(n));
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                        const size_t off = static_cast<size_t>(i0) * tensor->nb[0]
                                         + static_cast<size_t>(i1) * tensor->nb[1]
                                         + static_cast<size_t>(i2) * tensor->nb[2]
                                         + static_cast<size_t>(i3) * tensor->nb[3];
                        int32_t value = 0;
                        std::memcpy(&value, raw.data() + off, sizeof(value));
                        values.values.push_back(value);
                    }
                }
            }
        }
        capture.integer_vectors.push_back(std::move(values));
        return true;
    }

    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) return true;
    if (summarize_only(tensor->name)) {
        TraceSummary summary;
        summary.name = name;
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), summary.dims.begin());
        double sum = 0.0, square_sum = 0.0;
        visit_values(tensor, raw, [&](float value) { accumulate(summary.stats, value, sum, square_sum); });
        finish_stats(summary.stats, sum, square_sum);
        capture.summaries.push_back(std::move(summary));
        return true;
    }

    TraceVector vector;
    vector.name = name;
    vector.type_id = static_cast<int>(tensor->type);
    std::copy(std::begin(tensor->ne), std::end(tensor->ne), vector.dims.begin());
    vector.values.reserve(static_cast<size_t>(n));
    double sum = 0.0, square_sum = 0.0;
    visit_values(tensor, raw, [&](float value) {
        accumulate(vector.stats, value, sum, square_sum);
        vector.values.push_back(value);
    });
    finish_stats(vector.stats, sum, square_sum);
    capture.vectors.push_back(std::move(vector));
    return true;
}

static TraceVector capture_model_tensor(const ggml_tensor * tensor, const std::string & name) {
    TraceVector vector;
    vector.name = name;
    vector.type_id = static_cast<int>(tensor->type);
    std::copy(std::begin(tensor->ne), std::end(tensor->ne), vector.dims.begin());
    const size_t n = static_cast<size_t>(ggml_nelements(tensor));
    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> raw(nbytes);
    if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);
    } else {
        std::memcpy(raw.data(), tensor->data, nbytes);
    }
    vector.values.reserve(n);
    double sum = 0.0, square_sum = 0.0;
    visit_values(tensor, raw, [&](float value) {
        accumulate(vector.stats, value, sum, square_sum);
        vector.values.push_back(value);
    });
    finish_stats(vector.stats, sum, square_sum);
    return vector;
}

static std::vector<int> parse_ids(const std::string & text) {
    if (text.empty() || text.back() == ',') throw std::runtime_error("invalid input token IDs");
    std::stringstream stream(text);
    std::vector<int> ids;
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty()) throw std::runtime_error("invalid input token IDs");
        size_t consumed = 0;
        const int id = std::stoi(part, &consumed);
        if (consumed != part.size() || id < 0) throw std::runtime_error("invalid input token IDs");
        ids.push_back(id);
    }
    return ids;
}

static TraceVector capture_logits(const float * logits, int vocab, int position) {
    TraceVector vector;
    vector.name = "pos" + std::to_string(position) + ".final.logits";
    vector.type_id = GGML_TYPE_F32;
    vector.dims = {vocab, 1, 1, 1};
    vector.values.assign(logits, logits + vocab);
    double sum = 0.0, square_sum = 0.0;
    for (float value : vector.values) accumulate(vector.stats, value, sum, square_sum);
    finish_stats(vector.stats, sum, square_sum);
    return vector;
}

static std::string json_quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << ' ';
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}

static uint64_t token_hash(const std::vector<llama_token> & ids) {
    uint64_t h = 1469598103934665603ull;
    for (llama_token id : ids) {
        const uint32_t x = static_cast<uint32_t>(id);
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<uint8_t>(x >> (8 * b));
            h *= 1099511628211ull;
        }
    }
    return h;
}

static void write_trace(const std::string & path, const std::vector<TraceVector> & vectors) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot open trace file: " + path);
    constexpr char magic[8] = {'W', 'D', 'P', 'T', 'R', 'C', '1', '\0'};
    const uint32_t version = 1;
    const uint32_t count = static_cast<uint32_t>(vectors.size());
    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const TraceVector & vector : vectors) {
        const uint32_t name_size = static_cast<uint32_t>(vector.name.size());
        const uint64_t value_count = static_cast<uint64_t>(vector.values.size());
        file.write(reinterpret_cast<const char *>(&name_size), sizeof(name_size));
        file.write(reinterpret_cast<const char *>(&value_count), sizeof(value_count));
        file.write(vector.name.data(), static_cast<std::streamsize>(vector.name.size()));
        file.write(reinterpret_cast<const char *>(vector.values.data()),
            static_cast<std::streamsize>(vector.values.size() * sizeof(float)));
        if (!file) throw std::runtime_error("failed writing CPU trace vector: " + vector.name);
    }
}

static void print_stats(const TraceStats & s) {
    std::cout << "\"count\":" << s.count
              << ",\"min\":" << std::setprecision(9) << std::defaultfloat << s.min
              << ",\"max\":" << s.max
              << ",\"max_abs\":" << s.max_abs
              << ",\"mean\":" << std::setprecision(17) << s.mean
              << ",\"l2\":" << s.l2
              << ",\"rms\":" << s.rms
              << ",\"checksum_fnv1a64\":\"0x" << std::hex << s.checksum << std::dec << '\"';
}

int main(int argc, char ** argv) {
    try {
        if (argc != 5 && argc != 6 && argc != 7 && argc != 8) {
            throw std::runtime_error(
                "usage: reference MODEL CTX INPUT_TOKEN_IDS TRACE_BIN [TRACE_POSITION [INPUT_F32 [LAYER]]]");
        }
        const std::string model_path = argv[1];
        const uint32_t ctx_size = static_cast<uint32_t>(std::stoi(argv[2]));
        const std::vector<int> input_ids = parse_ids(argv[3]);
        const std::string trace_path = argv[4];
        const int trace_position = argc >= 6 ? std::stoi(argv[5]) : -1;
        if (trace_position < -1 || trace_position >= static_cast<int>(input_ids.size())) {
            throw std::runtime_error("trace position outside the forced input sequence");
        }

        llama_backend_init();
        const llama_model_tensor_buft_override overrides[] = {
            {R"(ffn_.*_exps\.weight)", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.load_mtp = false;
        mp.tensor_buft_overrides = overrides;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) throw std::runtime_error("llama_model_load_from_file failed");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx_size;
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        TraceCapture capture;
        capture.trace_position = trace_position;
        if (argc >= 7) {
            capture.qsa_input_override_position = trace_position;
            capture.qsa_input_override_layer = argc == 8 ? std::stoi(argv[7]) : 35;
            if (capture.qsa_input_override_layer < 0 || capture.qsa_input_override_layer >= 48) {
                throw std::runtime_error("attention input override layer must be in 0..47");
            }
            std::ifstream input(argv[6], std::ios::binary);
            if (!input) throw std::runtime_error("cannot open QSA input override file");
            input.seekg(0, std::ios::end);
            const std::streamoff bytes = input.tellg();
            input.seekg(0, std::ios::beg);
            if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
                throw std::runtime_error("invalid QSA input override F32 byte length");
            }
            capture.qsa_input_override.resize(static_cast<size_t>(bytes) / sizeof(float));
            input.read(reinterpret_cast<char *>(capture.qsa_input_override.data()), bytes);
            if (!input) throw std::runtime_error("failed reading QSA input override file");
        }
        if (model->hc_head_norm) {
            capture.vectors.push_back(capture_model_tensor(model->hc_head_norm, "model.output_hc_norm.weight"));
        }
        cp.cb_eval = capture_trace;
        cp.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, cp);
        if (!context) throw std::runtime_error("llama_init_from_model failed");

        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<llama_token> outputs;
        std::vector<std::vector<std::pair<llama_token, float>>> top_logits;
        const auto begin = std::chrono::steady_clock::now();
        for (size_t pos = 0; pos < input_ids.size(); ++pos) {
            if (input_ids[pos] >= vocab) throw std::runtime_error("input token outside vocabulary");
            capture.position = static_cast<int>(pos);
            llama_token token = input_ids[pos];
            llama_batch batch = llama_batch_get_one(&token, 1);
            const int rc = llama_decode(context, batch);
            if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
            const float * logits = llama_get_logits_ith(context, -1);
            if (!logits) throw std::runtime_error("llama_get_logits_ith returned null");
            if (capture.position == trace_position) {
                capture.vectors.push_back(capture_logits(logits, vocab, capture.position));
            }

            llama_token best_token = 0;
            float best = -std::numeric_limits<float>::infinity();
            std::vector<std::pair<float, llama_token>> ranked;
            ranked.reserve(static_cast<size_t>(vocab));
            for (int t = 0; t < vocab; ++t) {
                const float value = logits[t];
                if (!std::isfinite(value)) throw std::runtime_error("non-finite reference logits");
                ranked.emplace_back(value, t);
                if (value > best || (value == best && t < best_token)) {
                    best = value;
                    best_token = t;
                }
            }
            const auto order = [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            };
            const size_t top_n = std::min<size_t>(10, ranked.size());
            std::partial_sort(ranked.begin(), ranked.begin() + top_n, ranked.end(), order);
            std::vector<std::pair<llama_token, float>> row;
            for (size_t i = 0; i < top_n; ++i) row.emplace_back(ranked[i].second, ranked[i].first);
            top_logits.push_back(std::move(row));
            outputs.push_back(best_token);
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        write_trace(trace_path, capture.vectors);

        std::cout << "LLAMA_STATE_RESULT={\"llama_cpp_commit\":\"6fcaa16f4b360649933a54d1f91ad40ed35c0e11\""
                  << ",\"ctx\":" << ctx_size << ",\"trace_position\":" << trace_position
                  << ",\"qsa_input_override_applied\":"
                  << (capture.qsa_input_override_applied ? "true" : "false")
                  << ",\"qsa_input_override_count\":" << capture.qsa_input_override_count
                  << ",\"qsa_input_override_layer\":" << capture.qsa_input_override_layer
                  << ",\"seconds\":" << std::fixed << std::setprecision(6) << seconds
                  << ",\"input_token_ids\":[";
        for (size_t i = 0; i < input_ids.size(); ++i) { if (i) std::cout << ','; std::cout << input_ids[i]; }
        std::cout << "],\"token_ids\":[";
        for (size_t i = 0; i < outputs.size(); ++i) { if (i) std::cout << ','; std::cout << outputs[i]; }
        std::cout << "],\"token_hash\":\"0x" << std::hex << token_hash(outputs) << std::dec
                  << "\",\"trace_bin\":" << json_quote(trace_path) << ",\"vectors\":[";
        for (size_t i = 0; i < capture.vectors.size(); ++i) {
            if (i) std::cout << ',';
            const auto & v = capture.vectors[i];
            std::cout << "{\"name\":" << json_quote(v.name) << ",\"type_id\":" << v.type_id << ",\"dims\":["
                      << v.dims[0] << ',' << v.dims[1] << ',' << v.dims[2] << ',' << v.dims[3] << "],";
            print_stats(v.stats);
            std::cout << '}';
        }
        std::cout << "],\"summaries\":[";
        for (size_t i = 0; i < capture.summaries.size(); ++i) {
            if (i) std::cout << ',';
            const auto & v = capture.summaries[i];
            std::cout << "{\"name\":" << json_quote(v.name) << ",\"dims\":["
                      << v.dims[0] << ',' << v.dims[1] << ',' << v.dims[2] << ',' << v.dims[3] << "],";
            print_stats(v.stats);
            std::cout << '}';
        }
        std::cout << "],\"integer_vectors\":[";
        for (size_t i = 0; i < capture.integer_vectors.size(); ++i) {
            if (i) std::cout << ',';
            const auto & v = capture.integer_vectors[i];
            std::cout << "{\"name\":" << json_quote(v.name) << ",\"values\":[";
            for (size_t j = 0; j < v.values.size(); ++j) { if (j) std::cout << ','; std::cout << v.values[j]; }
            std::cout << "]}";
        }
        std::cout << "],\"top_logits\":[";
        for (size_t i = 0; i < top_logits.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '[';
            for (size_t j = 0; j < top_logits[i].size(); ++j) {
                if (j) std::cout << ',';
                std::cout << '[' << top_logits[i][j].first << ','
                          << std::setprecision(9) << std::defaultfloat << top_logits[i][j].second << ']';
            }
            std::cout << ']';
        }
        std::cout << "]}\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "reference error: " << e.what() << '\n';
        return 1;
    }
}
