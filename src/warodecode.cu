#include "gguf.hpp"
#include "gpu_arch.hpp"
#include "lowbit.cuh"
#include "qwen38_layout.hpp"
#include "qwen38_hc.cuh"
#include "qwen38_gdn.cuh"
#include "qwen38_forward.cuh"
#include "qwen38_moe.cuh"
#include "qwen38_mtp.cuh"
#include "qwen38_runtime.cuh"
#include "qwen38_state.cuh"
#include "qwen38_tokenizer.cuh"
#include "qwen38_tuning.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
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

static std::string json_quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 0x20) {
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}

struct Args {
    std::string model;
    int ctx = 4096;
    int generate = 128;
    int device = 0;
    int start_token = 1;
    int top_k = 0;
    bool graph = true;
    bool autotune = true;
    bool load_only = false;
    bool debug_f32_hc_layer0 = false;
    bool debug_f32_hc_attn_0_8 = false;
    bool debug_f32_hc_ffn_0_8 = false;
    bool debug_f32_hc_ffn_layer8 = false;
    int debug_hc_q8_1_attn_from = 0;
    int debug_hc_q8_1_attn_through = -1;
    int debug_hc_q8_1_ffn_from = 0;
    int debug_hc_q8_1_ffn_through = -1;
    int debug_hc_q8_0_ffn_from = 0;
    int debug_hc_q8_0_ffn_through = -1;
    bool debug_f32_hc_head = false;
    bool debug_layer0_hc_gate_f32_store = false;
    bool debug_layer0_hc_gate_f32_sigmoid = false;
    int debug_hc_gate_q8_0_layer = -1;
    int debug_hc_full_q8_0_layer = -1;
    int debug_qsa_q8_0_layer = -1;
    int debug_qsa_v_q8_0_layer = -1;
    int debug_hc_norm_f32_layer = -1;
    std::string debug_hc_norm_f32_file;
    std::vector<float> debug_hc_norm_f32;
    int debug_attention_output_f32_layer = -1;
    std::string debug_attention_output_f32_file;
    std::vector<float> debug_attention_output_f32;
    int debug_attention_input_f32_layer = -1;
    std::string debug_attention_input_f32_file;
    std::vector<float> debug_attention_input_f32;
    int debug_qsa_context_f32_layer = -1;
    std::string debug_qsa_context_f32_file;
    std::vector<float> debug_qsa_context_f32;
    int debug_qsa_pregate_f32_layer = -1;
    std::string debug_qsa_pregate_f32_file;
    std::vector<float> debug_qsa_pregate_f32;
    int debug_qsa_query_f32_layer = -1;
    std::string debug_qsa_query_f32_file;
    std::vector<float> debug_qsa_query_f32;
    int debug_qsa_key_store_f32_layer = -1;
    std::string debug_qsa_key_store_f32_file;
    std::vector<float> debug_qsa_key_store_f32;
    int debug_qsa_value_store_f32_layer = -1;
    std::string debug_qsa_value_store_f32_file;
    std::vector<float> debug_qsa_value_store_f32;
    int debug_qsa_key_cache_prefix_f32_layer = -1;
    std::string debug_qsa_key_cache_prefix_f32_file;
    std::vector<float> debug_qsa_key_cache_prefix_f32;
    int debug_qsa_value_cache_prefix_f32_layer = -1;
    std::string debug_qsa_value_cache_prefix_f32_file;
    std::vector<float> debug_qsa_value_cache_prefix_f32;
    int debug_hc_attn_combine_layer = -1;
    std::string debug_hc_attn_combine_f32_file;
    std::vector<float> debug_hc_attn_combine_f32;
    int debug_hc_inject_f32_layer = -1;
    std::string debug_hc_inject_f32_file;
    std::vector<float> debug_hc_inject_f32;
    int debug_layer_input_residual_layer = -1;
    std::string debug_layer_input_residual_f32_file;
    std::vector<float> debug_layer_input_residual_f32;
    std::string debug_layer8_after_attention_residual_f32_file;
    std::vector<float> debug_layer8_after_attention_residual_f32;
    std::string debug_layer8_hc_ffn_mix_f32_file;
    std::vector<float> debug_layer8_hc_ffn_mix_f32;
    std::string debug_layer8_output_f32_file;
    int debug_layer8_output_layer = 8;
    std::string debug_layer_output_all_positions_f32_file;
    std::vector<float> debug_layer_output_all_positions_f32;
    std::string debug_final_residual_f32_file;
    std::vector<float> debug_final_residual_f32;
    std::string debug_layer14_gdn_core_f32_file;
    std::vector<float> debug_layer14_gdn_core_f32;
    std::string debug_layer14_gdn_norm_output_f32_file;
    std::vector<float> debug_layer14_gdn_norm_output_f32;
    std::string debug_layer9_gdn_state_f32_file;
    std::vector<float> debug_layer9_gdn_state_f32;
    int debug_layer9_gdn_state_layer = 9;
    std::string debug_layer9_gdn_conv_state_f32_file;
    std::vector<float> debug_layer9_gdn_conv_state_f32;
    int debug_layer9_gdn_conv_state_layer = 9;
    std::string extended_parity_trace;
    std::string state_trace_path;
    std::vector<int> forced_input_tokens;
    std::vector<int> prefill_tokens;
    std::string prefill_tokens_file;
    int bench_repeat = 1;
    bool mtp = false;
    int mtp_draft = 0;
    std::vector<int> debug_moe_ids;
    std::vector<float> debug_moe_weights;
    std::string debug_moe_routes_file;
    std::vector<wd::Qwen38MoeRouteOverride> debug_moe_routes;
    std::string debug_router_cpu_activation_f32_file;
    std::vector<float> debug_router_cpu_activation_f32;
    std::string debug_routed_moe_output_f32_file;
    std::vector<float> debug_routed_moe_output;
    std::string debug_routed_moe_output_all_positions_f32_file;
    std::vector<float> debug_routed_moe_output_all_positions;
    int debug_ffn_output_layer = 8;
    std::vector<float> debug_ffn_output;
    std::vector<float> debug_hc_residual;
    int state_trace_position = -1;
    // Task M: llama-cli style text mode. All of these are additive: when
    // --prompt / --interactive are absent the legacy token-ID JSON path runs
    // unchanged.
    std::string prompt;              // -p / --prompt
    std::string prompt_file;         // -f / --prompt-file (verbatim bytes)
    bool prompt_given = false;       // -p or -f was passed (empty prompt counts)
    bool interactive = false;        // -i / --interactive
    bool no_display_prompt = false;  // --no-display-prompt
    int chat_template = -1;          // --chat-template / --no-chat-template (-1 auto)
    int add_bos = -1;                // --add-bos / --no-bos (-1 = follow the GGUF)
    bool parse_special = true;       // --no-parse-special
    bool tokenize_only = false;      // --tokenize-only (no GPU needed)
    std::string stream_json;         // --stream-json PATH
    int stop_eos = 1;                // --stop-eos / --no-stop-eos
    bool temp_set = false;           // --temp (accepted, greedy only)
    double temp = 0.0;
};

static std::vector<int> parse_token_ids(const std::string & value) {
    if (value.empty() || value.back() == ',') throw std::runtime_error("invalid forced input token list");
    std::stringstream stream(value);
    std::vector<int> ids;
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty()) throw std::runtime_error("invalid forced input token list");
        size_t consumed = 0;
        const int id = std::stoi(part, &consumed);
        if (consumed != part.size() || id < 0) throw std::runtime_error("invalid forced input token list");
        ids.push_back(id);
    }
    return ids;
}

// Prompt token IDs for the prefill/decode phase benchmark. Commas, whitespace
// and newlines are all accepted, so a tokenizer dump can be fed in as-is.
static std::vector<int> read_token_ids_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open token ID file: " + path);
    std::string text(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (char & c : text) {
        if (c == ',' || c == ';' || c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    std::istringstream stream(text);
    std::vector<int> ids;
    long long value = 0;
    while (stream >> value) {
        if (value < 0 || value > std::numeric_limits<int>::max()) {
            throw std::runtime_error("token ID out of range in " + path);
        }
        ids.push_back(static_cast<int>(value));
    }
    if (!stream.eof()) throw std::runtime_error("token ID file has a non-numeric field: " + path);
    if (ids.empty()) throw std::runtime_error("token ID file is empty: " + path);
    return ids;
}

static std::vector<float> parse_float_values(const std::string & value) {
    if (value.empty() || value.back() == ',') throw std::runtime_error("invalid float list");
    std::stringstream stream(value);
    std::vector<float> values;
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty()) throw std::runtime_error("invalid float list");
        size_t consumed = 0;
        const float parsed = std::stof(part, &consumed);
        if (consumed != part.size() || !std::isfinite(parsed)) throw std::runtime_error("invalid float list");
        values.push_back(parsed);
    }
    return values;
}

static std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open F32 vector: " + path);
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 vector file size: " + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!file || std::any_of(values.begin(), values.end(), [](float x) { return !std::isfinite(x); })) {
        throw std::runtime_error("invalid F32 vector data: " + path);
    }
    return values;
}

static std::vector<wd::Qwen38MoeRouteOverride> read_moe_routes_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open MoE route override file: " + path);
    std::vector<wd::Qwen38MoeRouteOverride> routes;
    std::string line;
    int previous_position = -1;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream row(line);
        std::string field;
        while (std::getline(row, field, ',')) fields.push_back(field);
        if (fields.size() < 2) {
            throw std::runtime_error("MoE route rows must include a position and expert IDs: " + path);
        }
        wd::Qwen38MoeRouteOverride route;
        size_t consumed = 0;
        route.position = std::stoi(fields[0], &consumed);
        if (consumed != fields[0].size() || route.position < 0 || route.position <= previous_position) {
            throw std::runtime_error("MoE route positions must be nonnegative and strictly increasing");
        }
        previous_position = route.position;
        const bool ids_only = fields.size() >= 2 && fields[1] == "ids";
        const bool weighted = fields.size() >= 2 && fields[1] == "weights";
        const size_t first_value = (ids_only || weighted) ? 2 : 1;
        if (!ids_only && (fields.size() - first_value) % 2 != 0) {
            throw std::runtime_error("weighted MoE route rows must contain ID/weight pairs: " + path);
        }
        for (size_t i = first_value; i < fields.size(); i += ids_only ? 1 : 2) {
            consumed = 0;
            const int expert = std::stoi(fields[i], &consumed);
            if (consumed != fields[i].size() || expert < 0) {
                throw std::runtime_error("invalid expert ID in MoE route file");
            }
            route.expert_ids.push_back(expert);
            if (ids_only) continue;
            consumed = 0;
            const float weight = std::stof(fields[i + 1], &consumed);
            if (consumed != fields[i + 1].size() || !std::isfinite(weight)) {
                throw std::runtime_error("invalid routing weight in MoE route file");
            }
            route.weights.push_back(weight);
        }
        if (!route.weights.empty() && route.expert_ids.size() != route.weights.size()) {
            throw std::runtime_error("MoE route ID/weight count mismatch");
        }
        routes.push_back(std::move(route));
    }
    if (routes.empty()) throw std::runtime_error("MoE route override file is empty: " + path);
    return routes;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i, const char* flag) -> const char* {
        if (++i >= argc) throw std::runtime_error(std::string("値がありません: ") + flag);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--model" || s == "-m") a.model = need(i, "--model");
        else if (s == "--ctx" || s == "-c") a.ctx = std::stoi(need(i, "--ctx"));
        else if (s == "--generate" || s == "-n") a.generate = std::stoi(need(i, "--generate"));
        else if (s == "--prompt" || s == "-p") { a.prompt = need(i, "--prompt"); a.prompt_given = true; }
        else if (s == "--prompt-file" || s == "-f") { a.prompt_file = need(i, "--prompt-file"); a.prompt_given = true; }
        else if (s == "--interactive" || s == "-i") a.interactive = true;
        else if (s == "--no-display-prompt") a.no_display_prompt = true;
        else if (s == "--chat-template") a.chat_template = 1;
        else if (s == "--no-chat-template") a.chat_template = 0;
        else if (s == "--add-bos") a.add_bos = 1;
        else if (s == "--no-bos") a.add_bos = 0;
        else if (s == "--no-parse-special") a.parse_special = false;
        else if (s == "--tokenize-only") a.tokenize_only = true;
        else if (s == "--stream-json") a.stream_json = need(i, "--stream-json");
        else if (s == "--stop-eos") a.stop_eos = 1;
        else if (s == "--no-stop-eos") a.stop_eos = 0;
        else if (s == "--temp") { a.temp = std::stod(need(i, "--temp")); a.temp_set = true; }
        else if (s == "--device") a.device = std::stoi(need(i, "--device"));
        else if (s == "--start-token") a.start_token = std::stoi(need(i, "--start-token"));
        else if (s == "--top-k") a.top_k = std::stoi(need(i, "--top-k"));
        else if (s == "--no-graph") a.graph = false;
        else if (s == "--no-autotune") a.autotune = false;
        else if (s == "--load-only") a.load_only = true;
        else if (s == "--debug-f32-hc-layer0") a.debug_f32_hc_layer0 = true;
        else if (s == "--debug-f32-hc-attn-0-8") a.debug_f32_hc_attn_0_8 = true;
        else if (s == "--debug-f32-hc-ffn-0-8") a.debug_f32_hc_ffn_0_8 = true;
        else if (s == "--debug-hc-q8_1-attn-from-layer") a.debug_hc_q8_1_attn_from = std::stoi(need(i, "--debug-hc-q8_1-attn-from-layer"));
        else if (s == "--debug-hc-q8_1-attn-through-layer") a.debug_hc_q8_1_attn_through = std::stoi(need(i, "--debug-hc-q8_1-attn-through-layer"));
        else if (s == "--debug-hc-q8_1-ffn-from-layer") a.debug_hc_q8_1_ffn_from = std::stoi(need(i, "--debug-hc-q8_1-ffn-from-layer"));
        else if (s == "--debug-hc-q8_1-ffn-through-layer") a.debug_hc_q8_1_ffn_through = std::stoi(need(i, "--debug-hc-q8_1-ffn-through-layer"));
        else if (s == "--debug-hc-q8_0-ffn-from-layer") a.debug_hc_q8_0_ffn_from = std::stoi(need(i, "--debug-hc-q8_0-ffn-from-layer"));
        else if (s == "--debug-hc-q8_0-ffn-through-layer") a.debug_hc_q8_0_ffn_through = std::stoi(need(i, "--debug-hc-q8_0-ffn-through-layer"));
        else if (s == "--debug-f32-hc-ffn-layer8") a.debug_f32_hc_ffn_layer8 = true;
        else if (s == "--debug-f32-hc-head") a.debug_f32_hc_head = true;
        else if (s == "--debug-layer0-hc-gate-f32-store") a.debug_layer0_hc_gate_f32_store = true;
        else if (s == "--debug-layer0-hc-gate-f32-sigmoid") a.debug_layer0_hc_gate_f32_sigmoid = true;
        else if (s == "--debug-layer0-hc-gate-q8_0-activation") a.debug_hc_gate_q8_0_layer = 0;
        else if (s == "--debug-hc-gate-q8_0-layer") a.debug_hc_gate_q8_0_layer = std::stoi(need(i, "--debug-hc-gate-q8_0-layer"));
        else if (s == "--debug-hc-attn-q8_0-full-layer") a.debug_hc_full_q8_0_layer = std::stoi(need(i, "--debug-hc-attn-q8_0-full-layer"));
        else if (s == "--debug-qsa-q8_0-layer") a.debug_qsa_q8_0_layer = std::stoi(need(i, "--debug-qsa-q8_0-layer"));
        else if (s == "--debug-qsa-v-q8_0-layer") a.debug_qsa_v_q8_0_layer = std::stoi(need(i, "--debug-qsa-v-q8_0-layer"));
        else if (s == "--debug-layer0-hc-norm-f32-file") {
            a.debug_hc_norm_f32_layer = 0;
            a.debug_hc_norm_f32_file = need(i, "--debug-layer0-hc-norm-f32-file");
        }
        else if (s == "--debug-hc-norm-f32-layer") a.debug_hc_norm_f32_layer = std::stoi(need(i, "--debug-hc-norm-f32-layer"));
        else if (s == "--debug-hc-norm-f32-file") a.debug_hc_norm_f32_file = need(i, "--debug-hc-norm-f32-file");
        else if (s == "--debug-attention-output-f32-layer") a.debug_attention_output_f32_layer = std::stoi(need(i, "--debug-attention-output-f32-layer"));
        else if (s == "--debug-attention-output-f32-file") a.debug_attention_output_f32_file = need(i, "--debug-attention-output-f32-file");
        else if (s == "--debug-attention-input-f32-layer") a.debug_attention_input_f32_layer = std::stoi(need(i, "--debug-attention-input-f32-layer"));
        else if (s == "--debug-attention-input-f32-file") a.debug_attention_input_f32_file = need(i, "--debug-attention-input-f32-file");
        else if (s == "--debug-qsa-context-f32-layer") a.debug_qsa_context_f32_layer = std::stoi(need(i, "--debug-qsa-context-f32-layer"));
        else if (s == "--debug-qsa-context-f32-file") a.debug_qsa_context_f32_file = need(i, "--debug-qsa-context-f32-file");
        else if (s == "--debug-qsa-pregate-f32-layer") a.debug_qsa_pregate_f32_layer = std::stoi(need(i, "--debug-qsa-pregate-f32-layer"));
        else if (s == "--debug-qsa-pregate-f32-file") a.debug_qsa_pregate_f32_file = need(i, "--debug-qsa-pregate-f32-file");
        else if (s == "--debug-qsa-query-f32-layer") a.debug_qsa_query_f32_layer = std::stoi(need(i, "--debug-qsa-query-f32-layer"));
        else if (s == "--debug-qsa-query-f32-file") a.debug_qsa_query_f32_file = need(i, "--debug-qsa-query-f32-file");
        else if (s == "--debug-qsa-key-store-f32-layer") a.debug_qsa_key_store_f32_layer = std::stoi(need(i, "--debug-qsa-key-store-f32-layer"));
        else if (s == "--debug-qsa-key-store-f32-file") a.debug_qsa_key_store_f32_file = need(i, "--debug-qsa-key-store-f32-file");
        else if (s == "--debug-qsa-value-store-f32-layer") a.debug_qsa_value_store_f32_layer = std::stoi(need(i, "--debug-qsa-value-store-f32-layer"));
        else if (s == "--debug-qsa-value-store-f32-file") a.debug_qsa_value_store_f32_file = need(i, "--debug-qsa-value-store-f32-file");
        else if (s == "--debug-qsa-key-cache-prefix-f32-layer") a.debug_qsa_key_cache_prefix_f32_layer = std::stoi(need(i, "--debug-qsa-key-cache-prefix-f32-layer"));
        else if (s == "--debug-qsa-key-cache-prefix-f32-file") a.debug_qsa_key_cache_prefix_f32_file = need(i, "--debug-qsa-key-cache-prefix-f32-file");
        else if (s == "--debug-qsa-value-cache-prefix-f32-layer") a.debug_qsa_value_cache_prefix_f32_layer = std::stoi(need(i, "--debug-qsa-value-cache-prefix-f32-layer"));
        else if (s == "--debug-qsa-value-cache-prefix-f32-file") a.debug_qsa_value_cache_prefix_f32_file = need(i, "--debug-qsa-value-cache-prefix-f32-file");
        else if (s == "--debug-hc-attn-combine-layer") a.debug_hc_attn_combine_layer = std::stoi(need(i, "--debug-hc-attn-combine-layer"));
        else if (s == "--debug-hc-attn-combine-f32-file") a.debug_hc_attn_combine_f32_file = need(i, "--debug-hc-attn-combine-f32-file");
        else if (s == "--debug-hc-inject-f32-layer") a.debug_hc_inject_f32_layer = std::stoi(need(i, "--debug-hc-inject-f32-layer"));
        else if (s == "--debug-hc-inject-f32-file") a.debug_hc_inject_f32_file = need(i, "--debug-hc-inject-f32-file");
        else if (s == "--debug-layer-input-residual-layer") a.debug_layer_input_residual_layer = std::stoi(need(i, "--debug-layer-input-residual-layer"));
        else if (s == "--debug-layer-input-residual-f32-file") a.debug_layer_input_residual_f32_file = need(i, "--debug-layer-input-residual-f32-file");
        else if (s == "--debug-layer8-after-attn-residual-f32-file") a.debug_layer8_after_attention_residual_f32_file = need(i, "--debug-layer8-after-attn-residual-f32-file");
        else if (s == "--debug-layer8-hc-ffn-mix-f32-file") a.debug_layer8_hc_ffn_mix_f32_file = need(i, "--debug-layer8-hc-ffn-mix-f32-file");
        else if (s == "--debug-layer8-output-f32-file") a.debug_layer8_output_f32_file = need(i, "--debug-layer8-output-f32-file");
        else if (s == "--debug-layer8-output-layer") a.debug_layer8_output_layer = std::stoi(need(i, "--debug-layer8-output-layer"));
        else if (s == "--debug-layer-output-all-positions-f32-file") {
            a.debug_layer_output_all_positions_f32_file = need(i, "--debug-layer-output-all-positions-f32-file");
        }
        else if (s == "--debug-final-residual-f32-file") a.debug_final_residual_f32_file = need(i, "--debug-final-residual-f32-file");
        else if (s == "--debug-layer14-gdn-core-f32-file") a.debug_layer14_gdn_core_f32_file = need(i, "--debug-layer14-gdn-core-f32-file");
        else if (s == "--debug-layer14-gdn-norm-output-f32-file") a.debug_layer14_gdn_norm_output_f32_file = need(i, "--debug-layer14-gdn-norm-output-f32-file");
        else if (s == "--debug-layer9-gdn-state-f32-file") a.debug_layer9_gdn_state_f32_file = need(i, "--debug-layer9-gdn-state-f32-file");
        else if (s == "--debug-layer9-gdn-conv-state-f32-file") a.debug_layer9_gdn_conv_state_f32_file = need(i, "--debug-layer9-gdn-conv-state-f32-file");
        else if (s == "--debug-layer9-gdn-state-layer") a.debug_layer9_gdn_state_layer = std::stoi(need(i, "--debug-layer9-gdn-state-layer"));
        else if (s == "--debug-layer9-gdn-conv-state-layer") a.debug_layer9_gdn_conv_state_layer = std::stoi(need(i, "--debug-layer9-gdn-conv-state-layer"));
        else if (s == "--debug-moe-ids") a.debug_moe_ids = parse_token_ids(need(i, "--debug-moe-ids"));
        else if (s == "--debug-moe-weights") a.debug_moe_weights = parse_float_values(need(i, "--debug-moe-weights"));
        else if (s == "--debug-moe-routes-file") a.debug_moe_routes_file = need(i, "--debug-moe-routes-file");
        else if (s == "--debug-router-cpu-activation-f32-file") {
            a.debug_router_cpu_activation_f32_file = need(i, "--debug-router-cpu-activation-f32-file");
        }
        else if (s == "--debug-routed-moe-output-f32-file") {
            a.debug_routed_moe_output_f32_file = need(i, "--debug-routed-moe-output-f32-file");
        }
        else if (s == "--debug-routed-moe-output-all-positions-f32-file") {
            a.debug_routed_moe_output_all_positions_f32_file =
                need(i, "--debug-routed-moe-output-all-positions-f32-file");
        }
        else if (s == "--debug-ffn-output-layer") a.debug_ffn_output_layer = std::stoi(need(i, "--debug-ffn-output-layer"));
        else if (s == "--debug-ffn-output") a.debug_ffn_output = parse_float_values(need(i, "--debug-ffn-output"));
        else if (s == "--debug-hc-residual") {
            const auto values = parse_float_values(need(i, "--debug-hc-residual"));
            a.debug_hc_residual.insert(a.debug_hc_residual.end(), values.begin(), values.end());
        }
        else if (s == "--extended-parity-trace") a.extended_parity_trace = need(i, "--extended-parity-trace");
        else if (s == "--state-trace") a.state_trace_path = need(i, "--state-trace");
        else if (s == "--forced-input-tokens") a.forced_input_tokens = parse_token_ids(need(i, "--forced-input-tokens"));
        else if (s == "--prefill-tokens") a.prefill_tokens = parse_token_ids(need(i, "--prefill-tokens"));
        else if (s == "--prefill-tokens-file") a.prefill_tokens_file = need(i, "--prefill-tokens-file");
        else if (s == "--bench-repeat") a.bench_repeat = std::stoi(need(i, "--bench-repeat"));
        else if (s == "--mtp") a.mtp = true;
        else if (s == "--mtp-draft") a.mtp_draft = std::stoi(need(i, "--mtp-draft"));
        else if (s == "--state-trace-position") a.state_trace_position = std::stoi(need(i, "--state-trace-position"));
        else if (s == "-h" || s == "--help") {
            std::cout << "text mode: --prompt/-p \"文字列\" | --prompt-file/-f path | --interactive/-i | -n N | -m model.gguf"
                         " [--no-display-prompt] [--temp 0] [--chat-template|--no-chat-template]"
                         " [--add-bos|--no-bos] [--stop-eos|--no-stop-eos] [--stream-json path]"
                         " [--tokenize-only]\n";
            std::cout << "   -m/-p/-n/-c は --model/--prompt/--generate/--ctx の別名です\n";
            std::cout << "warodecode --model model.gguf [--ctx 4096] [--generate 128] [--device 0] [--start-token 1] [--top-k 10] [--debug-f32-hc-layer0] [--debug-f32-hc-attn-0-8] [--debug-f32-hc-ffn-0-8] [--debug-hc-q8_1-attn-from-layer N --debug-hc-q8_1-attn-through-layer N] [--debug-hc-q8_1-ffn-from-layer N --debug-hc-q8_1-ffn-through-layer N] [--debug-f32-hc-ffn-layer8] [--debug-hc-norm-f32-layer N --debug-hc-norm-f32-file path | --debug-layer0-hc-gate-f32-store | --debug-layer0-hc-gate-f32-sigmoid | --debug-hc-gate-q8_0-layer N | --debug-hc-attn-q8_0-full-layer N | --debug-qsa-q8_0-layer N | --debug-qsa-v-q8_0-layer N] [--debug-attention-input-f32-layer N --debug-attention-input-f32-file path] [--debug-qsa-context-f32-layer N --debug-qsa-context-f32-file path] [--debug-qsa-pregate-f32-layer N --debug-qsa-pregate-f32-file path] [--debug-qsa-query-f32-layer N --debug-qsa-query-f32-file path] [--debug-attention-output-f32-layer N --debug-attention-output-f32-file path] [--debug-hc-attn-combine-layer N --debug-hc-attn-combine-f32-file path] [--debug-hc-inject-f32-layer N --debug-hc-inject-f32-file path] [--debug-layer-input-residual-layer N --debug-layer-input-residual-f32-file path] [--debug-router-cpu-activation-f32-file path] [--debug-moe-ids ids [--debug-moe-weights weights] | --debug-moe-routes-file path] [--debug-routed-moe-output-f32-file path | --debug-routed-moe-output-all-positions-f32-file path] [--debug-layer8-after-attn-residual-f32-file path] [--debug-layer8-hc-ffn-mix-f32-file path | --debug-layer8-output-layer N --debug-layer8-output-f32-file path | --debug-layer-output-all-positions-f32-file path] [--debug-ffn-output-layer N --debug-ffn-output values] [--debug-final-residual-f32-file path | --debug-layer9-gdn-state-layer N --debug-layer9-gdn-state-f32-file path | --debug-layer9-gdn-conv-state-layer N --debug-layer9-gdn-conv-state-f32-file path | --debug-layer14-gdn-core-f32-file path | --debug-layer14-gdn-norm-output-f32-file] [--extended-parity-trace path] [--state-trace path --forced-input-tokens 1,271,2 [--state-trace-position N]] [--prefill-tokens ids | --prefill-tokens-file path] [--bench-repeat N] [--no-graph] [--no-autotune] [--load-only]\n";
            std::cout << "[--debug-qsa-key-store-f32-layer N --debug-qsa-key-store-f32-file path] [--debug-qsa-value-store-f32-layer N --debug-qsa-value-store-f32-file path] [--debug-qsa-key-cache-prefix-f32-layer N --debug-qsa-key-cache-prefix-f32-file path] [--debug-qsa-value-cache-prefix-f32-layer N --debug-qsa-value-cache-prefix-f32-file path]\n";
            std::exit(0);
        } else throw std::runtime_error("不明な引数: " + s);
    }
    // MTP (speculative decode) defaults come from the environment so the
    // Colab runner can switch it without new CLI plumbing. --mtp / --mtp-draft
    // still win when given explicitly.
    if (!a.mtp && wd::q38_mtp_enabled()) a.mtp = true;
    if (a.mtp_draft <= 0) a.mtp_draft = wd::q38_mtp_draft_len();
    if (a.debug_f32_hc_layer0 && a.top_k == 0) a.top_k = 10;
    if (a.debug_f32_hc_layer0 && a.generate != 1 && a.state_trace_path.empty()) {
        throw std::runtime_error("--debug-f32-hc-layer0 requires --generate 1");
    }
    const bool debug_hc_q8_1_prefix =
        a.debug_hc_q8_1_attn_through >= 0 || a.debug_hc_q8_1_ffn_through >= 0;
    const bool debug_hc_q8_0_ffn_prefix = a.debug_hc_q8_0_ffn_through >= 0;
    if ((debug_hc_q8_1_prefix || debug_hc_q8_0_ffn_prefix) &&
        (a.debug_f32_hc_attn_0_8 || a.debug_f32_hc_ffn_0_8 || a.debug_f32_hc_ffn_layer8)) {
        throw std::runtime_error("choose either legacy HC F32 flags or a HC Q8 cutoff replay");
    }
    if (debug_hc_q8_1_prefix && debug_hc_q8_0_ffn_prefix) {
        throw std::runtime_error("choose either HC Q8_1 or exact HC Q8_0 FFN replay");
    }
    if ((a.debug_f32_hc_attn_0_8 || a.debug_f32_hc_ffn_0_8 || a.debug_f32_hc_ffn_layer8) &&
        a.state_trace_path.empty() && a.top_k == 0) {
        throw std::runtime_error("layer 0-8 HC F32 experiments require tracing");
    }
    if (!a.extended_parity_trace.empty() && !a.debug_f32_hc_layer0) {
        throw std::runtime_error("--extended-parity-trace requires --debug-f32-hc-layer0");
    }
    const int hc_gate_replays = static_cast<int>(a.debug_layer0_hc_gate_f32_store) +
        static_cast<int>(a.debug_layer0_hc_gate_f32_sigmoid) +
        static_cast<int>(a.debug_hc_gate_q8_0_layer >= 0) +
        static_cast<int>(a.debug_hc_full_q8_0_layer >= 0) +
        static_cast<int>(a.debug_qsa_q8_0_layer >= 0) +
        static_cast<int>(a.debug_qsa_v_q8_0_layer >= 0) +
        static_cast<int>(debug_hc_q8_0_ffn_prefix);
    if (hc_gate_replays > 1) throw std::runtime_error("choose one HC diagnostic projection replay at a time");
    if (hc_gate_replays > 0 &&
        (a.debug_hc_q8_1_attn_through >= 0 || a.debug_hc_q8_1_ffn_through >= 0)) {
        throw std::runtime_error("HC Q8_1 prefix replay conflicts with another HC/QSA projection replay");
    }
    const bool has_hc_norm_override = !a.debug_hc_norm_f32_file.empty();
    if (hc_gate_replays > 0 &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("Q8_0 projection replay requires a selected state trace position");
    }
    if (has_hc_norm_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("HC norm override requires a selected state trace position");
    }
    const bool has_attention_output_override = !a.debug_attention_output_f32_file.empty();
    const bool has_attention_input_override = !a.debug_attention_input_f32_file.empty();
    const bool has_qsa_context_override = !a.debug_qsa_context_f32_file.empty();
    const bool has_qsa_pregate_override = !a.debug_qsa_pregate_f32_file.empty();
    const bool has_qsa_query_override = !a.debug_qsa_query_f32_file.empty();
    const bool has_qsa_key_store_override = !a.debug_qsa_key_store_f32_file.empty();
    const bool has_qsa_value_store_override = !a.debug_qsa_value_store_f32_file.empty();
    const bool has_qsa_store_override =
        has_qsa_key_store_override || has_qsa_value_store_override;
    const bool has_qsa_key_cache_prefix_override =
        !a.debug_qsa_key_cache_prefix_f32_file.empty();
    const bool has_qsa_value_cache_prefix_override =
        !a.debug_qsa_value_cache_prefix_f32_file.empty();
    const bool has_qsa_cache_prefix_override =
        has_qsa_key_cache_prefix_override || has_qsa_value_cache_prefix_override;
    const bool qsa_store_q8_conflict =
        (has_qsa_key_store_override &&
         (a.debug_qsa_q8_0_layer == a.debug_qsa_key_store_f32_layer ||
          a.debug_qsa_v_q8_0_layer == a.debug_qsa_key_store_f32_layer)) ||
        (has_qsa_value_store_override &&
         (a.debug_qsa_q8_0_layer == a.debug_qsa_value_store_f32_layer ||
          a.debug_qsa_v_q8_0_layer == a.debug_qsa_value_store_f32_layer));
    const bool qsa_cache_prefix_q8_conflict =
        (has_qsa_key_cache_prefix_override &&
         (a.debug_qsa_q8_0_layer == a.debug_qsa_key_cache_prefix_f32_layer ||
          a.debug_qsa_v_q8_0_layer == a.debug_qsa_key_cache_prefix_f32_layer)) ||
        (has_qsa_value_cache_prefix_override &&
         (a.debug_qsa_q8_0_layer == a.debug_qsa_value_cache_prefix_f32_layer ||
          a.debug_qsa_v_q8_0_layer == a.debug_qsa_value_cache_prefix_f32_layer));
    const bool has_hc_attn_combine_override = !a.debug_hc_attn_combine_f32_file.empty();
    const bool has_hc_inject_override = !a.debug_hc_inject_f32_file.empty();
    const bool has_layer_input_residual_override = !a.debug_layer_input_residual_f32_file.empty();
    const bool has_layer8_residual_override = !a.debug_layer8_after_attention_residual_f32_file.empty();
    const bool has_layer8_hc_ffn_mix_override = !a.debug_layer8_hc_ffn_mix_f32_file.empty();
    const bool has_layer8_output_override = !a.debug_layer8_output_f32_file.empty();
    const bool has_layer_output_all_positions_override =
        !a.debug_layer_output_all_positions_f32_file.empty();
    const bool has_routed_moe_output_override = !a.debug_routed_moe_output_f32_file.empty();
    const bool has_routed_moe_output_all_positions_override =
        !a.debug_routed_moe_output_all_positions_f32_file.empty();
    const bool has_final_residual_override = !a.debug_final_residual_f32_file.empty();
    const bool has_layer14_gdn_core_override = !a.debug_layer14_gdn_core_f32_file.empty();
    const bool has_layer14_gdn_norm_override = !a.debug_layer14_gdn_norm_output_f32_file.empty();
    const bool has_layer9_gdn_state_override = !a.debug_layer9_gdn_state_f32_file.empty();
    const bool has_layer9_gdn_conv_state_override = !a.debug_layer9_gdn_conv_state_f32_file.empty();
    if (a.debug_layer8_output_layer < 0 || a.debug_layer8_output_layer >= 48 ||
        (a.debug_layer8_output_layer != 8 && !has_layer8_output_override &&
         !has_layer_output_all_positions_override && a.debug_hc_residual.empty())) {
        throw std::runtime_error("HC residual output target requires an override and layer in 0..47");
    }
    if (a.debug_layer9_gdn_state_layer < 0 || a.debug_layer9_gdn_state_layer >= 48 ||
        (a.debug_layer9_gdn_state_layer != 9 && !has_layer9_gdn_state_override)) {
        throw std::runtime_error("layer-9 GDN state target requires a state file and layer in 0..47");
    }
    if (a.debug_layer9_gdn_conv_state_layer < 0 || a.debug_layer9_gdn_conv_state_layer >= 48 ||
        (a.debug_layer9_gdn_conv_state_layer != 9 && !has_layer9_gdn_conv_state_override)) {
        throw std::runtime_error("layer-9 GDN convolution-state target requires a state file and layer in 0..47");
    }
    if (a.debug_attention_output_f32_layer < -1 || a.debug_attention_output_f32_layer >= 48 ||
        has_attention_output_override != (a.debug_attention_output_f32_layer >= 0)) {
        throw std::runtime_error("attention output override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_attention_input_f32_layer < -1 || a.debug_attention_input_f32_layer >= 48 ||
        has_attention_input_override != (a.debug_attention_input_f32_layer >= 0)) {
        throw std::runtime_error("attention input override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_context_f32_layer < -1 || a.debug_qsa_context_f32_layer >= 48 ||
        has_qsa_context_override != (a.debug_qsa_context_f32_layer >= 0)) {
        throw std::runtime_error("QSA attention context override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_pregate_f32_layer < -1 || a.debug_qsa_pregate_f32_layer >= 48 ||
        has_qsa_pregate_override != (a.debug_qsa_pregate_f32_layer >= 0)) {
        throw std::runtime_error("QSA pregate override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_query_f32_layer < -1 || a.debug_qsa_query_f32_layer >= 48 ||
        has_qsa_query_override != (a.debug_qsa_query_f32_layer >= 0)) {
        throw std::runtime_error("QSA query override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_key_store_f32_layer < -1 || a.debug_qsa_key_store_f32_layer >= 48 ||
        has_qsa_key_store_override != (a.debug_qsa_key_store_f32_layer >= 0)) {
        throw std::runtime_error("QSA key store override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_value_store_f32_layer < -1 || a.debug_qsa_value_store_f32_layer >= 48 ||
        has_qsa_value_store_override != (a.debug_qsa_value_store_f32_layer >= 0)) {
        throw std::runtime_error("QSA value store override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_key_cache_prefix_f32_layer < -1 ||
        a.debug_qsa_key_cache_prefix_f32_layer >= 48 ||
        has_qsa_key_cache_prefix_override != (a.debug_qsa_key_cache_prefix_f32_layer >= 0)) {
        throw std::runtime_error("QSA key cache prefix override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_qsa_value_cache_prefix_f32_layer < -1 ||
        a.debug_qsa_value_cache_prefix_f32_layer >= 48 ||
        has_qsa_value_cache_prefix_override != (a.debug_qsa_value_cache_prefix_f32_layer >= 0)) {
        throw std::runtime_error("QSA value cache prefix override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_hc_attn_combine_layer < -1 || a.debug_hc_attn_combine_layer >= 48 ||
        has_hc_attn_combine_override != (a.debug_hc_attn_combine_layer >= 0)) {
        throw std::runtime_error("HC attention combine override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_hc_inject_f32_layer < -1 || a.debug_hc_inject_f32_layer >= 48 ||
        has_hc_inject_override != (a.debug_hc_inject_f32_layer >= 0)) {
        throw std::runtime_error("HC inject override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_layer_input_residual_layer < -1 || a.debug_layer_input_residual_layer >= 48 ||
        has_layer_input_residual_override != (a.debug_layer_input_residual_layer >= 0)) {
        throw std::runtime_error("layer input residual override requires a layer in 0..47 and an F32 file");
    }
    if ((has_attention_input_override || has_layer_input_residual_override) &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("component override requires a selected state trace position");
    }
    if (has_qsa_context_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("QSA attention context override requires a selected state trace position");
    }
    if (has_qsa_context_override &&
        (has_attention_output_override || has_hc_attn_combine_override)) {
        throw std::runtime_error("choose one QSA context/output override at a time");
    }
    if (has_qsa_pregate_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("QSA pregate override requires a selected state trace position");
    }
    if (has_qsa_pregate_override &&
        (has_qsa_context_override || has_attention_output_override || has_hc_attn_combine_override)) {
        throw std::runtime_error("choose one QSA pregate/context/output override at a time");
    }
    if (has_qsa_query_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("QSA query override requires a selected state trace position");
    }
    if (has_qsa_query_override &&
        (has_qsa_context_override || has_qsa_pregate_override ||
         has_attention_output_override || has_hc_attn_combine_override)) {
        throw std::runtime_error("choose one QSA query/context/output override at a time");
    }
    if (has_qsa_store_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("QSA key/value store override requires a selected state trace position");
    }
    if (has_qsa_store_override &&
        (has_qsa_context_override || has_qsa_pregate_override || has_qsa_query_override ||
         has_attention_output_override || has_hc_attn_combine_override)) {
        throw std::runtime_error("choose one QSA key/value store or context/query/output override at a time");
    }
    if (qsa_store_q8_conflict) {
        throw std::runtime_error("QSA key/value store override cannot be combined with Q8_0 replay");
    }
    if (has_qsa_cache_prefix_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("QSA cache prefix override requires a selected state trace position");
    }
    if (has_qsa_cache_prefix_override &&
        (has_qsa_context_override || has_qsa_pregate_override || has_qsa_query_override ||
         has_qsa_store_override || has_attention_output_override ||
         has_hc_attn_combine_override)) {
        throw std::runtime_error("choose one QSA cache prefix or context/query/store/output override at a time");
    }
    if (qsa_cache_prefix_q8_conflict) {
        throw std::runtime_error("QSA cache prefix override cannot be combined with Q8_0 replay");
    }
    if (has_hc_attn_combine_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("HC attention combine override requires a selected state trace position");
    }
    if (has_hc_attn_combine_override &&
        (has_attention_input_override || has_attention_output_override ||
         has_hc_inject_override || has_layer8_residual_override)) {
        throw std::runtime_error("choose one attention output/residual override at a time");
    }
    if (has_hc_inject_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("HC inject override requires a selected state trace position");
    }
    if (has_attention_output_override && has_layer8_residual_override) {
        throw std::runtime_error("choose one attention/residual override at a time");
    }
    if ((has_layer8_residual_override ||
         has_final_residual_override || has_layer14_gdn_core_override ||
         has_layer14_gdn_norm_override) &&
        (a.state_trace_path.empty() || a.state_trace_position != 36)) {
        throw std::runtime_error("diagnostic override requires state trace position 36");
    }
    // TEMPORARY DIAGNOSTIC RELAXATION (Issue #1 MoE same-input test): allow the
    // layer-8 HC FFN mix override at the traced position so the MoE execution
    // can be measured with exact CPU inputs. REVERT BEFORE COMMIT.
    // TEMPORARY DIAGNOSTIC RELAXATION (Issue #1 GDN same-input test): allow the
    // layer-9 GDN recurrent/convolution state overrides at the traced position
    // so the AR-core execution can be separated from state inheritance.
    // REVERT BEFORE COMMIT.
    if (has_layer8_hc_ffn_mix_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("layer-8 HC FFN mix override requires a selected state trace position");
    }
    if ((has_layer9_gdn_state_override || has_layer9_gdn_conv_state_override) &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("GDN state override requires a selected state trace position");
    }
    if (has_attention_output_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("attention output override requires a selected state trace position");
    }
    if (has_layer8_output_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("layer output override requires a selected state trace position");
    }
    if ((has_layer8_output_override || has_layer_output_all_positions_override) &&
        !a.debug_hc_residual.empty()) {
        throw std::runtime_error("choose one layer-8 output override at a time");
    }
    if (has_layer_output_all_positions_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("all-position layer output replay requires a selected state trace position");
    }
    if (has_routed_moe_output_override && has_routed_moe_output_all_positions_override) {
        throw std::runtime_error("choose a single-position or all-position routed MoE output override");
    }
    if (has_routed_moe_output_all_positions_override &&
        (a.state_trace_path.empty() || a.state_trace_position < 0)) {
        throw std::runtime_error("all-position routed MoE output replay requires a selected state trace position");
    }
    if (a.debug_hc_q8_0_ffn_from < 0 || a.debug_hc_q8_0_ffn_from >= 48 ||
        a.debug_hc_q8_0_ffn_through < -1 || a.debug_hc_q8_0_ffn_through >= 48 ||
        (a.debug_hc_q8_0_ffn_through >= 0 &&
         a.debug_hc_q8_0_ffn_from > a.debug_hc_q8_0_ffn_through) ||
        (a.debug_hc_q8_0_ffn_through < 0 && a.debug_hc_q8_0_ffn_from != 0)) {
        throw std::runtime_error("exact HC Q8_0 FFN layer range must satisfy 0 <= start <= end < 48");
    }
    if (a.debug_hc_q8_1_attn_from < 0 || a.debug_hc_q8_1_attn_from >= 48 ||
        a.debug_hc_q8_1_ffn_from < 0 || a.debug_hc_q8_1_ffn_from >= 48 ||
        a.debug_hc_q8_1_attn_through < -1 || a.debug_hc_q8_1_attn_through >= 48 ||
        a.debug_hc_q8_1_ffn_through < -1 || a.debug_hc_q8_1_ffn_through >= 48 ||
        (a.debug_hc_q8_1_attn_through >= 0 &&
         a.debug_hc_q8_1_attn_from > a.debug_hc_q8_1_attn_through) ||
        (a.debug_hc_q8_1_ffn_through >= 0 &&
         a.debug_hc_q8_1_ffn_from > a.debug_hc_q8_1_ffn_through)) {
        throw std::runtime_error("HC Q8_1 layer range must satisfy 0 <= start <= end < 48");
    }
    if ((a.debug_hc_q8_1_attn_through < 0 && a.debug_hc_q8_1_attn_from != 0) ||
        (a.debug_hc_q8_1_ffn_through < 0 && a.debug_hc_q8_1_ffn_from != 0)) {
        throw std::runtime_error("HC Q8_1 from-layer requires a matching through-layer");
    }
    if ((debug_hc_q8_1_prefix || debug_hc_q8_0_ffn_prefix) &&
        a.state_trace_path.empty() && a.top_k == 0) {
        throw std::runtime_error("HC Q8 prefix replay requires tracing");
    }
    if (a.debug_hc_gate_q8_0_layer < -1 || a.debug_hc_gate_q8_0_layer > 8) {
        throw std::runtime_error("Q8_0 HC gate replay layer must be in 0..8");
    }
    if (a.debug_hc_full_q8_0_layer < -1 || a.debug_hc_full_q8_0_layer >= 48) {
        throw std::runtime_error("full Q8_0 HC replay layer must be in 0..47");
    }
    if (a.debug_qsa_q8_0_layer < -1 || a.debug_qsa_q8_0_layer >= 48) {
        throw std::runtime_error("QSA Q8_0 replay layer must be in 0..47");
    }
    if (a.debug_qsa_v_q8_0_layer < -1 || a.debug_qsa_v_q8_0_layer >= 48) {
        throw std::runtime_error("QSA V Q8_0 replay layer must be in 0..47");
    }
    if (a.debug_ffn_output_layer < 0 || a.debug_ffn_output_layer >= 48) {
        throw std::runtime_error("FFN output override layer must be in 0..47");
    }
    if (a.debug_hc_norm_f32_file.empty() != (a.debug_hc_norm_f32_layer < 0) ||
        a.debug_hc_norm_f32_layer < -1 || a.debug_hc_norm_f32_layer >= 48) {
        throw std::runtime_error("HC norm override requires a layer in 0..47 and an F32 file");
    }
    if (a.debug_hc_full_q8_0_layer >= 0 && has_hc_norm_override &&
        a.debug_hc_full_q8_0_layer != a.debug_hc_norm_f32_layer) {
        throw std::runtime_error("full Q8_0 HC replay and norm override layers must match");
    }
    if (a.debug_hc_norm_f32_layer >= 0) {
        a.debug_hc_norm_f32 = read_f32_file(a.debug_hc_norm_f32_file);
        if (a.debug_hc_norm_f32.size() != 10240) {
            throw std::runtime_error("HC norm override must contain 10240 F32 values");
        }
    }
    if (has_attention_output_override) {
        a.debug_attention_output_f32 = read_f32_file(a.debug_attention_output_f32_file);
    }
    if (has_attention_input_override) {
        a.debug_attention_input_f32 = read_f32_file(a.debug_attention_input_f32_file);
        if (a.debug_attention_input_f32.size() != 2560) {
            throw std::runtime_error("attention input override must contain 2560 F32 values");
        }
    }
    if (has_qsa_context_override) {
        a.debug_qsa_context_f32 = read_f32_file(a.debug_qsa_context_f32_file);
    }
    if (has_qsa_pregate_override) {
        a.debug_qsa_pregate_f32 = read_f32_file(a.debug_qsa_pregate_f32_file);
    }
    if (has_qsa_query_override) {
        a.debug_qsa_query_f32 = read_f32_file(a.debug_qsa_query_f32_file);
    }
    if (has_qsa_key_store_override) {
        a.debug_qsa_key_store_f32 = read_f32_file(a.debug_qsa_key_store_f32_file);
    }
    if (has_qsa_value_store_override) {
        a.debug_qsa_value_store_f32 = read_f32_file(a.debug_qsa_value_store_f32_file);
    }
    if (has_qsa_key_cache_prefix_override) {
        a.debug_qsa_key_cache_prefix_f32 =
            read_f32_file(a.debug_qsa_key_cache_prefix_f32_file);
    }
    if (has_qsa_value_cache_prefix_override) {
        a.debug_qsa_value_cache_prefix_f32 =
            read_f32_file(a.debug_qsa_value_cache_prefix_f32_file);
    }
    if (has_hc_attn_combine_override) {
        a.debug_hc_attn_combine_f32 = read_f32_file(a.debug_hc_attn_combine_f32_file);
        if (a.debug_hc_attn_combine_f32.size() != 10240) {
            throw std::runtime_error("HC attention combine override must contain 10240 F32 values");
        }
    }
    if (has_hc_inject_override) {
        a.debug_hc_inject_f32 = read_f32_file(a.debug_hc_inject_f32_file);
        if (a.debug_hc_inject_f32.size() != 4) {
            throw std::runtime_error("HC inject override must contain 4 F32 values");
        }
    }
    if (has_layer_input_residual_override) {
        a.debug_layer_input_residual_f32 = read_f32_file(a.debug_layer_input_residual_f32_file);
        if (a.debug_layer_input_residual_f32.size() != 10240) {
            throw std::runtime_error("layer input residual override must contain 10240 F32 values");
        }
    }
    if (!a.debug_layer8_after_attention_residual_f32_file.empty()) {
        a.debug_layer8_after_attention_residual_f32 = read_f32_file(a.debug_layer8_after_attention_residual_f32_file);
    }
    if (!a.debug_layer8_hc_ffn_mix_f32_file.empty()) {
        a.debug_layer8_hc_ffn_mix_f32 = read_f32_file(a.debug_layer8_hc_ffn_mix_f32_file);
    }
    if (!a.debug_router_cpu_activation_f32_file.empty()) {
        a.debug_router_cpu_activation_f32 = read_f32_file(a.debug_router_cpu_activation_f32_file);
    }
    if (!a.debug_routed_moe_output_f32_file.empty()) {
        a.debug_routed_moe_output = read_f32_file(a.debug_routed_moe_output_f32_file);
    }
    if (has_routed_moe_output_all_positions_override) {
        a.debug_routed_moe_output_all_positions =
            read_f32_file(a.debug_routed_moe_output_all_positions_f32_file);
    }
    if (!a.debug_moe_routes_file.empty()) {
        a.debug_moe_routes = read_moe_routes_file(a.debug_moe_routes_file);
    }
    if (has_layer8_output_override) {
        a.debug_hc_residual = read_f32_file(a.debug_layer8_output_f32_file);
    }
    if (has_layer_output_all_positions_override) {
        a.debug_layer_output_all_positions_f32 = read_f32_file(
            a.debug_layer_output_all_positions_f32_file);
        a.debug_hc_residual = a.debug_layer_output_all_positions_f32;
    }
    if (has_final_residual_override) {
        a.debug_final_residual_f32 = read_f32_file(a.debug_final_residual_f32_file);
    }
    if (has_layer14_gdn_core_override) {
        a.debug_layer14_gdn_core_f32 = read_f32_file(a.debug_layer14_gdn_core_f32_file);
    }
    if (has_layer14_gdn_norm_override) {
        a.debug_layer14_gdn_norm_output_f32 = read_f32_file(a.debug_layer14_gdn_norm_output_f32_file);
    }
    if (has_layer9_gdn_state_override) {
        a.debug_layer9_gdn_state_f32 = read_f32_file(a.debug_layer9_gdn_state_f32_file);
    }
    if (has_layer9_gdn_conv_state_override) {
        a.debug_layer9_gdn_conv_state_f32 = read_f32_file(a.debug_layer9_gdn_conv_state_f32_file);
    }
    if (a.state_trace_path.empty() != a.forced_input_tokens.empty()) {
        throw std::runtime_error("--state-trace and --forced-input-tokens must be used together");
    }
    if (a.state_trace_path.empty() && a.state_trace_position != -1) {
        throw std::runtime_error("--state-trace-position requires --state-trace");
    }
    if ((a.debug_moe_ids.empty() && !a.debug_moe_weights.empty()) ||
        ((!a.debug_moe_ids.empty() || !a.debug_moe_routes_file.empty() ||
          !a.debug_router_cpu_activation_f32_file.empty() ||
          !a.debug_routed_moe_output_f32_file.empty() ||
          !a.debug_routed_moe_output_all_positions_f32_file.empty() ||
          !a.debug_ffn_output.empty() ||
          !a.debug_hc_residual.empty()) &&
         a.state_trace_path.empty())) {
        throw std::runtime_error("debug MoE/FFN overrides require --state-trace");
    }
    if (!a.debug_moe_routes_file.empty() &&
        (!a.debug_moe_ids.empty() || !a.debug_moe_weights.empty())) {
        throw std::runtime_error("choose a single-position or all-position MoE route override");
    }
    if (has_layer8_output_override && has_layer_output_all_positions_override) {
        throw std::runtime_error("choose a single-position or all-position layer output override");
    }
    if (!a.state_trace_path.empty() &&
        (!a.extended_parity_trace.empty() ||
         static_cast<int>(a.forced_input_tokens.size()) != a.generate ||
         a.forced_input_tokens.front() != a.start_token || a.state_trace_position < -1 ||
         a.state_trace_position >= a.generate)) {
        throw std::runtime_error("state trace requires a forced input for every generated position and cannot be combined with extended parity tracing");
    }
    if (a.model.empty()) throw std::runtime_error("--model が必要です");
    if (!a.prompt.empty() && !a.prompt_file.empty()) {
        throw std::runtime_error("--prompt と --prompt-file は同時に指定できません");
    }
    if ((a.prompt_given || a.interactive || a.tokenize_only) &&
        (!a.forced_input_tokens.empty() || !a.prefill_tokens.empty() || !a.state_trace_path.empty() ||
         !a.extended_parity_trace.empty() || a.mtp || a.bench_repeat > 1 || a.debug_moe_routes_file.size() > 0)) {
        throw std::runtime_error("--prompt/--interactive/--tokenize-only は "
                                 "--forced-input-tokens/--prefill-tokens/トレース/--mtp/--bench-repeat と併用できません");
    }
    if (a.ctx < 16 || a.generate < 1 || a.start_token < 0 || a.top_k < 0 || a.top_k > 256) {
        throw std::runtime_error("引数範囲が不正です");
    }
    if (a.bench_repeat < 1) throw std::runtime_error("--bench-repeat は1以上にしてください");
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
    float m = -INFINITY;
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
    MaxPair best{-INFINITY, 0};
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

// ---------------------------------------------------------------------------
// Task M: llama-cli style prompt streaming
// ---------------------------------------------------------------------------
//
// The helpers below live in the namespace that is already open (namespace wd at
// the top of this file) and are only reached from the qwen4exp branch of main().

// Set by SIGINT so Ctrl-C ends generation after the current token instead of
// killing the process in the middle of a CUDA step.
static volatile std::sig_atomic_t g_stream_interrupted = 0;

static void qwen38_stream_sigint(int) { g_stream_interrupted = 1; }

struct Qwen38StreamRequest {
    std::string prompt;
    bool interactive = false;
    bool no_display_prompt = false;
    bool chat_template = false;
    bool stop_eos = true;
    bool add_bos = false;
    bool show_progress = true;
    int generate = 128;
    int start_token = 1;
    int ctx = 0;
    std::string stream_json;
};

struct Qwen38StreamResult {
    std::vector<int> prompt_tokens;
    std::vector<int> generated;
    std::string text;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    bool stopped_on_eos = false;
    bool interrupted = false;
};

// One generation round: prime the KV cache with the prompt tokens and stream
// the continuation one token at a time.
//
// Step order is the one the CPU oracle uses: the sample taken after the last
// prompt token is the first generated token, and every further token is fed
// back at the next position. With an empty prompt the caller passes the start
// token, which reproduces the historical --start-token run exactly
// (one BOS at position 0 -> 64/64 against reference_tokens_64.json).
static Qwen38StreamResult qwen38_stream_generate(
        Qwen38Decoder & decoder,
        const Qwen38Tokenizer & tokenizer,
        const Qwen38StreamRequest & request,
        const std::vector<int> & prompt_tokens) {
    Qwen38StreamResult result;
    result.prompt_tokens = prompt_tokens;
    Qwen38StreamDecoder printer;
    const int steps = request.generate;
    const int positions = static_cast<int>(prompt_tokens.size()) + steps - 1;
    if (positions > request.ctx) {
        throw std::runtime_error("--ctx がプロンプト長 + --generate に対して不足しています (必要 " +
                                 std::to_string(positions) + " 位置)");
    }

    decoder.clear();
    auto begin = std::chrono::steady_clock::now();
    int position = 0;
    int token = -1;
    for (int id : prompt_tokens) token = decoder.step(id, position++);
    auto prefill_end = std::chrono::steady_clock::now();

    result.generated.reserve(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        result.generated.push_back(token);
        const std::string piece = printer.push(tokenizer, token, /*skip_special=*/true);
        if (!piece.empty()) {
            std::cout << piece << std::flush;
            result.text += piece;
        }
        if (request.stop_eos && tokenizer.is_stop(token)) {
            result.stopped_on_eos = true;
            break;
        }
        if (g_stream_interrupted) {
            result.interrupted = true;
            break;
        }
        if (i + 1 >= steps) break;
        token = decoder.step(token, position++);
    }
    const std::string tail = printer.flush();
    if (!tail.empty()) {
        std::cout << tail << std::flush;
        result.text += tail;
    }
    auto end = std::chrono::steady_clock::now();
    result.prefill_seconds = std::chrono::duration<double>(prefill_end - begin).count();
    result.decode_seconds = std::chrono::duration<double>(end - prefill_end).count();
    return result;
}

// Prompt token IDs for the CLI: the model's own tokenizer, with BOS only when
// the GGUF asks for it (llama-cli behaviour) or when --add-bos forces it.
static std::vector<int> qwen38_prompt_tokens(
        const Qwen38Tokenizer & tokenizer,
        const std::string & text,
        bool add_bos,
        int start_token) {
    std::vector<int> ids = tokenizer.encode(text, /*add_special=*/false, /*parse_special=*/true);
    if (add_bos && tokenizer.bos_id() >= 0) ids.insert(ids.begin(), tokenizer.bos_id());
    if (ids.empty()) ids.push_back(start_token);
    return ids;
}

static void qwen38_write_stream_json(
        const std::string & path,
        const std::string & gpu,
        const Qwen38StreamRequest & request,
        const Qwen38StreamResult & result,
        bool chat_template_applied) {
    std::ostringstream out;
    const double seconds = result.prefill_seconds + result.decode_seconds;
    out << "{\"engine\":\"warodecode-qwen38-stream\",\"gpu\":\"" << gpu << "\""
        << ",\"tokens\":" << result.generated.size()
        << ",\"seconds\":" << std::fixed << std::setprecision(6) << seconds
        << ",\"prefill_seconds\":" << std::setprecision(6) << result.prefill_seconds
        << ",\"prefill_tokens\":" << result.prompt_tokens.size()
        << ",\"decode_seconds\":" << std::setprecision(6) << result.decode_seconds
        << ",\"tok_s\":" << std::setprecision(3)
        << (result.decode_seconds > 0.0
                ? static_cast<double>(result.generated.size()) / result.decode_seconds : 0.0)
        << ",\"prompt_tokens\":" << result.prompt_tokens.size()
        << ",\"first_token\":" << (result.generated.empty() ? -1 : result.generated.front())
        << ",\"last_token\":" << (result.generated.empty() ? -1 : result.generated.back())
        << ",\"stopped_on_eos\":" << (result.stopped_on_eos ? "true" : "false")
        << ",\"interrupted\":" << (result.interrupted ? "true" : "false")
        << ",\"chat_template\":" << (chat_template_applied ? "true" : "false")
        << ",\"prompt_token_ids\":[";
    for (size_t i = 0; i < result.prompt_tokens.size(); ++i) {
        if (i) out << ',';
        out << result.prompt_tokens[i];
    }
    out << "],\"token_ids\":[";
    for (size_t i = 0; i < result.generated.size(); ++i) {
        if (i) out << ',';
        out << result.generated[i];
    }
    out << "]}\n";

    if (path.empty() || path == "-") {
        // stdout carries generated text, so the record goes to stderr unless a
        // file was requested.
        std::cerr << out.str();
        return;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("--stream-json を書き込めません: " + path);
    file << out.str();
}

static int qwen38_run_stream_mode(
        const Args & args,
        Qwen38Decoder & decoder,
        const Qwen38Tokenizer & tokenizer,
        const std::string & gpu) {
    Qwen38StreamRequest request;
    request.prompt = args.prompt;
    request.interactive = args.interactive;
    request.no_display_prompt = args.no_display_prompt;
    request.chat_template = args.chat_template == 1 ||
        (args.chat_template < 0 && args.interactive);
    request.stop_eos = args.stop_eos != 0;
    request.generate = args.generate;
    request.start_token = args.start_token;
    request.ctx = args.ctx;
    request.stream_json = args.stream_json;
    request.add_bos = args.add_bos == 1 || (args.add_bos < 0 && tokenizer.add_bos());
    if (request.chat_template && !tokenizer.supports_qwen_chat_template()) {
        std::cerr << "[chat-template] GGUFのchat_templateがQwen定型 (<|im_start|>/<|im_end|>) ではないため無効化します\n";
        request.chat_template = false;
    }
    if (request.chat_template) {
        std::cerr << "[chat-template] applied: qwen <|im_start|>role\\ncontent<|im_end|>\\n (+ '<|im_start|>assistant\\n')\n";
    }
    if (args.temp_set && args.temp != 0.0) {
        std::cerr << "[warn] --temp " << args.temp << " は未対応です (greedy固定)。0として扱います\n";
    } else if (args.temp_set) {
        std::cerr << "[warn] --temp 0 = greedy固定です (このランタイムはgreedy専用)\n";
    }
    if (request.interactive) {
        std::cerr << "[interactive] generate=" << request.generate
                  << " stop_eos=" << (request.stop_eos ? 1 : 0)
                  << " chat_template=" << (request.chat_template ? 1 : 0)
                  << " (/exit で終了, Ctrl-D で終了)\n";
    } else {
        std::cerr << "[prompt] tokens="
                  << qwen38_prompt_tokens(tokenizer, request.prompt, request.add_bos,
                                          request.start_token).size()
                  << " generate=" << request.generate
                  << " stop_eos=" << (request.stop_eos ? 1 : 0) << "\n";
    }
    if (request.interactive) {
        std::signal(SIGINT, qwen38_stream_sigint);
        std::vector<std::pair<std::string, std::string>> history;
        std::string line;
        while (!g_stream_interrupted) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line == "/exit" || line == "/quit") break;
            if (line.empty()) continue;
            history.emplace_back("user", line);
            const std::string rendered = request.chat_template
                ? tokenizer.apply_qwen_chat_template(history, /*add_generation_prompt=*/true)
                : line;
            if (request.no_display_prompt) {
                std::cout << "\n";
            } else {
                std::cout << rendered << std::flush;
            }
            const std::vector<int> ids = qwen38_prompt_tokens(
                tokenizer, rendered, request.add_bos, request.start_token);
            const Qwen38StreamResult result =
                qwen38_stream_generate(decoder, tokenizer, request, ids);
            std::cout << "\n";
            history.emplace_back("assistant", result.text);
            if (!args.stream_json.empty()) {
                qwen38_write_stream_json(args.stream_json, gpu, request, result, request.chat_template);
            }
            if (result.interrupted) break;
        }
        std::cout << "\n";
        return 0;
    }

    std::signal(SIGINT, qwen38_stream_sigint);
    const std::vector<int> ids =
        qwen38_prompt_tokens(tokenizer, request.prompt, request.add_bos, request.start_token);
    if (!request.no_display_prompt) std::cout << request.prompt << std::flush;
    const Qwen38StreamResult result = qwen38_stream_generate(decoder, tokenizer, request, ids);
    std::cout << "\n";
    qwen38_write_stream_json(args.stream_json, gpu, request, result, request.chat_template);
    std::cerr << "[done] tokens=" << result.generated.size()
              << " prefill_s=" << std::fixed << std::setprecision(2) << result.prefill_seconds
              << " decode_s=" << std::setprecision(2) << result.decode_seconds
              << " tok_s=" << std::setprecision(2)
              << (result.decode_seconds > 0.0
                      ? static_cast<double>(result.generated.size()) / result.decode_seconds : 0.0)
              << (result.interrupted ? " interrupted" : "")
              << (result.stopped_on_eos ? " eos" : "") << "\n";
    return 0;
}

} // namespace wd

int main(int argc, char** argv) {
    try {
        wd::Args args = wd::parse_args(argc, argv);
        if (!args.prompt_file.empty()) {
            std::ifstream prompt_in(args.prompt_file, std::ios::binary);
            if (!prompt_in) throw std::runtime_error("--prompt-file を開けません: " + args.prompt_file);
            args.prompt.assign(std::istreambuf_iterator<char>(prompt_in),
                               std::istreambuf_iterator<char>());
        }
        if (!args.prefill_tokens_file.empty()) {
            if (!args.prefill_tokens.empty()) {
                throw std::runtime_error("--prefill-tokens と --prefill-tokens-file は同時に指定できません");
            }
            args.prefill_tokens = wd::read_token_ids_file(args.prefill_tokens_file);
        }
        if (!args.prefill_tokens.empty() && !args.forced_input_tokens.empty()) {
            throw std::runtime_error("--prefill-tokens と --forced-input-tokens は同時に指定できません");
        }
        // Task M: tokenizer-only mode never touches the GPU, so it runs before
        // any CUDA call and works on a host without an NVIDIA device.
        if (args.tokenize_only) {
            wd::Qwen38Tokenizer tokenizer(args.model, /*verbose=*/true);
            const bool add_bos = args.add_bos == 1 || (args.add_bos < 0 && tokenizer.add_bos());
            std::vector<int> ids = tokenizer.encode(args.prompt, /*add_special=*/false, args.parse_special);
            if (add_bos && tokenizer.bos_id() >= 0) ids.insert(ids.begin(), tokenizer.bos_id());
            std::cout << "{\"engine\":\"warodecode-tokenize\""
                      << ",\"arch\":\"" << tokenizer.architecture() << "\""
                      << ",\"pre\":\"" << tokenizer.pre_type_name() << "\""
                      << ",\"vocab\":" << tokenizer.vocab_size()
                      << ",\"merges\":" << tokenizer.merge_count()
                      << ",\"add_bos\":" << (add_bos ? "true" : "false")
                      << ",\"parse_special\":" << (args.parse_special ? "true" : "false")
                      << ",\"token_ids\":[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << ids[i];
            }
            std::cout << "]}\n";
            return 0;
        }
        int count = 0; WD_CUDA(cudaGetDeviceCount(&count));
        if (args.device < 0 || args.device >= count) throw std::runtime_error("指定GPUがありません");
        WD_CUDA(cudaSetDevice(args.device));
        cudaDeviceProp prop{}; WD_CUDA(cudaGetDeviceProperties(&prop, args.device));
        const int cc = prop.major * 10 + prop.minor;
        if (cc < 50) throw std::runtime_error("warodecodeはSM50以上のNVIDIA GPUを対象にします");
        if (args.ctx <= args.generate + 2) throw std::runtime_error("--ctx は --generate より十分大きくしてください");
        if (static_cast<int>(args.prefill_tokens.size()) + args.generate > args.ctx) {
            throw std::runtime_error("--ctx は prefillトークン数 + --generate 以上にしてください");
        }

        const auto kernel_plan = wd::kernel_plan_for_sm(cc);
        std::cerr << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << " family=" << kernel_plan.family << "\n";
        std::cerr << "target kernel plan: decode=" << kernel_plan.decode << " verify=" << kernel_plan.verify << "\n";
        std::cerr << "GGUF読込: " << args.model << "\n";
        wd::GgufFile gguf(args.model);
        const std::string arch = gguf.string_meta("general.architecture");

        if (arch == "qwen4exp") {
            const int memory_mib = static_cast<int>(prop.totalGlobalMem / (1024ull * 1024ull));
            std::cerr << "Qwen3.8-Flash-Next専用runtimeを構築します\n";
            // Task M: text mode needs the GGUF vocabulary; the tokenizer reads the
            // metadata section only, so this line appears with the load messages.
            const bool text_mode = args.prompt_given || args.interactive;
            std::unique_ptr<wd::Qwen38Tokenizer> tokenizer;
            if (text_mode) tokenizer.reset(new wd::Qwen38Tokenizer(args.model, /*verbose=*/true));
            auto qrt = wd::load_qwen38_runtime(gguf, args.model, cc, memory_mib);
            wd::Qwen38DecodeState qstate;
            qstate.initialize(qrt, args.ctx);
            std::cerr << "arch=qwen4exp trunk=" << qrt.layout.trunk_layers
                      << " qsa=" << qrt.layout.qsa_layer_count()
                      << " gdn=" << qrt.layout.gdn_layer_count()
                      << " experts=" << qrt.layout.experts
                      << " topk=" << qrt.layout.experts_used
                      << " mtp=" << qrt.layout.mtp_layers << "\n";
            std::cerr << "expert-cache=" << qrt.tuning.expert_cache_mib
                      << " MiB slots=" << qrt.expert_cache_slots
                      << " fused=" << (qrt.fused_experts ? "true" : "false") << "\n";
            if (qrt.cpu_moe) {
                std::cerr << "cpu-moe=on layers=" << qrt.cpu_moe_layers << "/"
                          << qrt.layout.trunk_layers
                          << " threads=" << wd::cpu_moe::thread_count()
                          << " (WARODECODE_CPU_MOE="
                          << (wd::cpu_moe::env_flag("WARODECODE_CPU_MOE", false) ? 1 : 0)
                          << ")\n";
            } else {
                std::cerr << "cpu-moe=off\n";
            }
            const bool head_q8_k = wd::lowbit_q6_k_head_q8_k_eligible(qrt.output);
            std::cerr << "output-head q8_k=" << (head_q8_k ? "on" : "off")
                      << " (WARODECODE_HEAD_Q8_K="
                      << (wd::lowbit_head_q8_k_enabled() ? "1" : "0") << ")\n";

            if (args.load_only) {
                std::cout << "{\"engine\":\"warodecode-qwen38\",\"gpu\":\"" << prop.name
                          << "\",\"arch\":\"qwen4exp\""
                          << ",\"load_only\":true"
                          << ",\"trunk_layers\":" << qrt.layout.trunk_layers
                          << ",\"qsa_layers\":" << qrt.layout.qsa_layer_count()
                          << ",\"gdn_layers\":" << qrt.layout.gdn_layer_count()
                          << ",\"experts\":" << qrt.layout.experts
                          << ",\"experts_used\":" << qrt.layout.experts_used
                          << ",\"vocab\":" << qrt.vocab
                          << ",\"expert_cache_mib\":" << qrt.tuning.expert_cache_mib
                          << ",\"expert_cache_slots\":" << qrt.expert_cache_slots
                          << ",\"ctx\":" << qstate.max_ctx()
                          << ",\"decode_state_initialized\":true"
                          << ",\"ple\":" << (qrt.ple_table ? "true" : "false")
                          << ",\"fused_experts\":" << (qrt.fused_experts ? "true" : "false")
                          << ",\"head_q8_k\":" << (head_q8_k ? "true" : "false")
                          << ",\"kernel_family\":\"" << kernel_plan.family << "\"}\n";
                return 0;
            }

            if (args.start_token >= qrt.vocab) {
                throw std::runtime_error("--start-token がQwen3.8語彙数以上です");
            }
            if (!args.debug_attention_output_f32.empty() &&
                args.debug_attention_output_f32.size() != static_cast<size_t>(qrt.layout.hidden)) {
                throw std::runtime_error("attention output override must match hidden size");
            }
            if (!args.debug_layer8_after_attention_residual_f32.empty() &&
                args.debug_layer8_after_attention_residual_f32.size() !=
                    static_cast<size_t>(qrt.layout.hidden) * qrt.layout.hc_count) {
                throw std::runtime_error("layer-8 after-attention residual override must match HC width");
            }
            if (!args.debug_layer8_hc_ffn_mix_f32.empty() &&
                args.debug_layer8_hc_ffn_mix_f32.size() != static_cast<size_t>(qrt.layout.hidden)) {
                throw std::runtime_error("layer-8 HC FFN mix override must match hidden size");
            }
            const size_t hc_width = static_cast<size_t>(qrt.layout.hidden) * qrt.layout.hc_count;
            if (!args.debug_hc_residual.empty() &&
                (args.debug_layer_output_all_positions_f32_file.empty()
                    ? args.debug_hc_residual.size() != hc_width
                    : args.debug_hc_residual.size() != static_cast<size_t>(args.generate) * hc_width)) {
                throw std::runtime_error("HC residual output override must match the selected position range");
            }
            if (!args.debug_router_cpu_activation_f32.empty() &&
                args.debug_router_cpu_activation_f32.size() != static_cast<size_t>(qrt.layout.hidden)) {
                throw std::runtime_error("CPU router activation override must match hidden size");
            }
            if (!args.debug_routed_moe_output.empty() &&
                args.debug_routed_moe_output.size() != static_cast<size_t>(qrt.layout.hidden)) {
                throw std::runtime_error("routed MoE output override must match hidden size");
            }
            if (!args.debug_routed_moe_output_all_positions.empty() &&
                args.debug_routed_moe_output_all_positions.size() !=
                    static_cast<size_t>(args.generate) * qrt.layout.hidden) {
                throw std::runtime_error("all-position routed MoE output must match generated positions");
            }
            if (!args.debug_moe_ids.empty() &&
                args.debug_moe_ids.size() != static_cast<size_t>(qrt.layout.experts_used)) {
                throw std::runtime_error("debug MoE expert IDs must match top-k width");
            }
            if (!args.debug_moe_weights.empty() &&
                args.debug_moe_weights.size() != static_cast<size_t>(qrt.layout.experts_used)) {
                throw std::runtime_error("debug MoE weights must match top-k width");
            }
            for (const auto & route : args.debug_moe_routes) {
                if (route.position >= args.generate ||
                    route.expert_ids.size() != static_cast<size_t>(qrt.layout.experts_used) ||
                    (!route.weights.empty() &&
                     route.weights.size() != static_cast<size_t>(qrt.layout.experts_used))) {
                    throw std::runtime_error("MoE route file row does not match generated positions/top-k width");
                }
            }
            if (!args.debug_final_residual_f32.empty() && args.debug_final_residual_f32.size() !=
                static_cast<size_t>(qrt.layout.hidden) * qrt.layout.hc_count) {
                throw std::runtime_error("final residual override must match HC width");
            }
            if (!args.debug_layer14_gdn_core_f32.empty() && args.debug_layer14_gdn_core_f32.size() !=
                static_cast<size_t>(qrt.layout.ssm_state_size) * qrt.layout.ssm_dt_rank) {
                throw std::runtime_error("layer-14 GDN core override must match value width");
            }
            if (!args.debug_layer14_gdn_norm_output_f32.empty() && args.debug_layer14_gdn_norm_output_f32.size() !=
                static_cast<size_t>(qrt.layout.ssm_state_size) * qrt.layout.ssm_dt_rank) {
                throw std::runtime_error("layer-14 GDN norm output override must match value width");
            }
            if (!args.debug_layer9_gdn_state_f32.empty() && args.debug_layer9_gdn_state_f32.size() !=
                static_cast<size_t>(qrt.layout.ssm_state_size) * qrt.layout.ssm_state_size * qrt.layout.ssm_dt_rank) {
                throw std::runtime_error("layer-9 GDN state override must match recurrent state width");
            }
            const size_t gdn_conv_channels = static_cast<size_t>(qrt.layout.ssm_state_size) *
                (2 * qrt.layout.ssm_group_count + qrt.layout.ssm_dt_rank);
            if (!args.debug_layer9_gdn_conv_state_f32.empty() && args.debug_layer9_gdn_conv_state_f32.size() !=
                static_cast<size_t>(qrt.layout.ssm_conv_kernel - 1) * gdn_conv_channels) {
                throw std::runtime_error("layer-9 GDN convolution state override must match state width");
            }
            for (int token : args.forced_input_tokens) {
                if (token >= qrt.vocab) throw std::runtime_error("forced input token is outside the Qwen3.8 vocabulary");
            }

            wd::Qwen38Decoder decoder(qrt, qstate);
            std::vector<int> generated;
            generated.reserve(static_cast<size_t>(args.generate));
            std::vector<std::pair<int, float>> top_logits;
            wd::Qwen38Trace trace;
            trace.use_f32_hc_layer0 = args.debug_f32_hc_layer0;
            trace.debug_f32_hc_attn_0_8 = args.debug_f32_hc_attn_0_8;
            trace.debug_f32_hc_ffn_0_8 = args.debug_f32_hc_ffn_0_8;
            trace.debug_hc_q8_1_attn_from = args.debug_hc_q8_1_attn_from;
            trace.debug_hc_q8_1_attn_through = args.debug_hc_q8_1_attn_through;
            trace.debug_hc_q8_1_ffn_from = args.debug_hc_q8_1_ffn_from;
            trace.debug_hc_q8_1_ffn_through = args.debug_hc_q8_1_ffn_through;
            trace.debug_hc_q8_0_ffn_from = args.debug_hc_q8_0_ffn_from;
            trace.debug_hc_q8_0_ffn_through = args.debug_hc_q8_0_ffn_through;
            trace.debug_f32_hc_ffn_layer8 = args.debug_f32_hc_ffn_layer8;
            trace.debug_f32_hc_head = args.debug_f32_hc_head;
            trace.debug_layer0_hc_gate_f32_store = args.debug_layer0_hc_gate_f32_store;
            trace.debug_layer0_hc_gate_f32_sigmoid = args.debug_layer0_hc_gate_f32_sigmoid;
            trace.debug_hc_gate_q8_0_layer = args.debug_hc_gate_q8_0_layer;
            trace.debug_hc_full_q8_0_layer = args.debug_hc_full_q8_0_layer;
            trace.debug_qsa_q8_0_layer = args.debug_qsa_q8_0_layer;
            trace.debug_qsa_v_q8_0_layer = args.debug_qsa_v_q8_0_layer;
            trace.debug_hc_norm_f32_layer = args.debug_hc_norm_f32_layer;
            trace.debug_hc_norm_f32 = args.debug_hc_norm_f32;
            trace.debug_attention_output_f32_layer = args.debug_attention_output_f32_layer;
            trace.debug_attention_output_f32 = args.debug_attention_output_f32;
            trace.debug_attention_input_f32_layer = args.debug_attention_input_f32_layer;
            trace.debug_attention_input_f32 = args.debug_attention_input_f32;
            trace.debug_qsa_context_f32_layer = args.debug_qsa_context_f32_layer;
            trace.debug_qsa_context_f32 = args.debug_qsa_context_f32;
            trace.debug_qsa_pregate_f32_layer = args.debug_qsa_pregate_f32_layer;
            trace.debug_qsa_pregate_f32 = args.debug_qsa_pregate_f32;
            trace.debug_qsa_query_f32_layer = args.debug_qsa_query_f32_layer;
            trace.debug_qsa_query_f32 = args.debug_qsa_query_f32;
            trace.debug_qsa_key_store_f32_layer = args.debug_qsa_key_store_f32_layer;
            trace.debug_qsa_key_store_f32 = args.debug_qsa_key_store_f32;
            trace.debug_qsa_value_store_f32_layer = args.debug_qsa_value_store_f32_layer;
            trace.debug_qsa_value_store_f32 = args.debug_qsa_value_store_f32;
            trace.debug_qsa_key_cache_prefix_f32_layer = args.debug_qsa_key_cache_prefix_f32_layer;
            trace.debug_qsa_key_cache_prefix_f32 = args.debug_qsa_key_cache_prefix_f32;
            trace.debug_qsa_value_cache_prefix_f32_layer = args.debug_qsa_value_cache_prefix_f32_layer;
            trace.debug_qsa_value_cache_prefix_f32 = args.debug_qsa_value_cache_prefix_f32;
            trace.debug_hc_attn_combine_layer = args.debug_hc_attn_combine_layer;
            trace.debug_hc_attn_combine_f32 = args.debug_hc_attn_combine_f32;
            trace.debug_hc_inject_f32_layer = args.debug_hc_inject_f32_layer;
            trace.debug_hc_inject_f32 = args.debug_hc_inject_f32;
            trace.debug_layer_input_residual_layer = args.debug_layer_input_residual_layer;
            trace.debug_layer_input_residual_f32 = args.debug_layer_input_residual_f32;
            trace.debug_layer8_after_attention_residual_f32 = args.debug_layer8_after_attention_residual_f32;
            trace.debug_layer8_hc_ffn_mix_f32 = args.debug_layer8_hc_ffn_mix_f32;
            trace.debug_hc_residual_layer = args.debug_layer8_output_layer;
            trace.debug_final_residual_f32 = args.debug_final_residual_f32;
            trace.debug_layer14_gdn_core_f32 = args.debug_layer14_gdn_core_f32;
            trace.debug_layer14_gdn_norm_output_f32 = args.debug_layer14_gdn_norm_output_f32;
            trace.debug_layer9_gdn_state_f32 = args.debug_layer9_gdn_state_f32;
            trace.debug_layer9_gdn_conv_state_f32 = args.debug_layer9_gdn_conv_state_f32;
            trace.debug_layer9_gdn_state_layer = args.debug_layer9_gdn_state_layer;
            trace.debug_layer9_gdn_conv_state_layer = args.debug_layer9_gdn_conv_state_layer;
            trace.debug_moe_ids = args.debug_moe_ids;
            trace.debug_moe_weights = args.debug_moe_weights;
            trace.debug_moe_routes = args.debug_moe_routes;
            trace.debug_router_cpu_activation_f32 = args.debug_router_cpu_activation_f32;
            trace.debug_routed_moe_output = !args.debug_routed_moe_output_all_positions_f32_file.empty()
                ? args.debug_routed_moe_output_all_positions : args.debug_routed_moe_output;
            trace.debug_routed_moe_output_layer = 8;
            trace.debug_routed_moe_output_all_positions =
                !args.debug_routed_moe_output_all_positions_f32_file.empty();
            trace.debug_ffn_output_layer = args.debug_ffn_output_layer;
            trace.debug_ffn_output = args.debug_ffn_output;
            trace.debug_hc_residual = args.debug_hc_residual;
            trace.debug_hc_residual_all_positions =
                !args.debug_layer_output_all_positions_f32_file.empty();
            trace.extended_parity = !args.extended_parity_trace.empty();
            trace.state_trace = !args.state_trace_path.empty();
            trace.state_trace_position = args.state_trace_position;
            const std::vector<int> * forced_tokens_ptr =
                args.forced_input_tokens.empty() ? nullptr : &args.forced_input_tokens;
            const std::vector<int> * prefill_tokens_ptr =
                args.prefill_tokens.empty() ? nullptr : &args.prefill_tokens;
            // Task M: -p / -i run the text path. The step() order is the one the
            // CPU oracle uses (sample after the last prompt token = first
            // generated token), and the legacy JSON path below is untouched.
            if (text_mode) {
                return wd::qwen38_run_stream_mode(args, decoder, *tokenizer, prop.name);
            }
            wd::Qwen38Trace * const trace_ptr =
                args.top_k > 0 || trace.state_trace ? &trace : nullptr;
            // --bench-repeat>=2 repeats the whole prefill+decode run against the
            // same loaded model (expert cache stays warm) and reports the fastest
            // repetition. Only the first repetition fills the caller's outputs, so
            // tokens, top logits and traces are unaffected by the extra runs.
            //
            // MTP (speculative draft/verify) path. It drives step() itself, so
            // the tracing entry points and prefill token batches are excluded.
            const bool mtp_active =
                args.mtp && trace_ptr == nullptr && prefill_tokens_ptr == nullptr;
            if (args.mtp && !mtp_active) {
                std::cerr << "[mtp] disabled: tracing / prefill batches are not supported in MTP mode\n";
            }
            wd::Qwen38Mtp mtp_engine;
            if (mtp_active) {
                mtp_engine.initialize(gguf, qrt, qstate.max_ctx());
                mtp_engine.set_draft_len(args.mtp_draft);
                std::cerr << "[mtp] draft block blk." << mtp_engine.layer()
                          << " ratio=" << mtp_engine.ratio()
                          << " k=" << mtp_engine.draft_len()
                          << " expert_slots=" << mtp_engine.expert_cache_slots() << "\n";
            }
            std::vector<int> repeat_tokens;
            wd::Qwen38MtpStats mtp_stats;
            wd::Qwen38RunTiming timing;
            double sec = 0.0;
            for (int repeat = 0; repeat < args.bench_repeat; ++repeat) {
                const bool collect = repeat == 0;
                if (mtp_active) {
                    const wd::Qwen38MtpStats round = wd::qwen38_run_speculative(
                        decoder, mtp_engine, args.start_token, args.generate,
                        collect ? &generated : &repeat_tokens);
                    if (collect || round.seconds < sec) {
                        sec = round.seconds;
                        mtp_stats = round;
                        timing = wd::Qwen38RunTiming{};
                        timing.decode_seconds = round.seconds;
                        timing.decode_tokens = args.generate;
                    }
                    continue;
                }
                wd::Qwen38RunTiming run_timing;
                const double run_seconds = decoder.run(
                    args.start_token, args.generate,
                    collect ? &generated : nullptr,
                    collect && args.top_k > 0 ? &top_logits : nullptr,
                    args.top_k,
                    collect ? trace_ptr : nullptr,
                    forced_tokens_ptr, prefill_tokens_ptr, &run_timing);
                if (collect || run_seconds < sec) {
                    sec = run_seconds;
                    timing = run_timing;
                }
            }
            std::vector<wd::Qwen38TraceStats> extended_trace_stats;
            if (trace.extended_parity || trace.state_trace) {
                wd::write_qwen38_trace_binary(
                    trace.state_trace ? args.state_trace_path : args.extended_parity_trace,
                    trace.extended_vectors);
                extended_trace_stats.reserve(trace.extended_vectors.size());
                for (const auto & vector : trace.extended_vectors) {
                    extended_trace_stats.push_back(wd::qwen38_trace_stats(vector.values));
                }
            }
            const double tps = args.generate / sec;
            const uint64_t token_hash = wd::qwen38_token_hash(generated);

            std::cout << "{\"engine\":\"warodecode-qwen38\",\"gpu\":\"" << prop.name
                      << "\",\"arch\":\"qwen4exp\""
                      << ",\"tokens\":" << args.generate
                      << ",\"seconds\":" << std::fixed << std::setprecision(6) << sec
                      << ",\"prefill_seconds\":" << std::setprecision(6) << timing.prefill_seconds
                      << ",\"prefill_tokens\":" << timing.prefill_tokens
                      << ",\"decode_seconds\":" << std::setprecision(6) << timing.decode_seconds
                      << ",\"decode_tokens\":" << timing.decode_tokens
                      << ",\"bench_repeats\":" << args.bench_repeat
                      << ",\"tok_s\":" << std::setprecision(3) << tps
                      << ",\"first_token\":" << (generated.empty() ? -1 : generated.front())
                      << ",\"last_token\":" << (generated.empty() ? -1 : generated.back())
                      << ",\"token_hash\":\"0x" << std::hex << token_hash << std::dec << "\""
                      << ",\"ctx\":" << qstate.max_ctx()
                      << ",\"ple\":" << (qrt.ple_table ? "true" : "false")
                      << ",\"mtp\":" << (mtp_active ? "true" : "false")
                      << ",\"mtp_draft\":" << mtp_stats.draft_len
                      << ",\"mtp_calls\":" << mtp_stats.draft_steps
                      << ",\"mtp_draft_tokens\":" << mtp_stats.draft_tokens
                      << ",\"mtp_accepts\":" << mtp_stats.accepts
                      << ",\"mtp_full_rounds\":" << mtp_stats.full_rounds
                      << ",\"mtp_rounds\":" << mtp_stats.rounds
                      << ",\"mtp_accept_rate\":" << std::setprecision(4) << mtp_stats.accept_rate()
                      << ",\"mtp_draft_seconds\":" << std::setprecision(6) << mtp_stats.draft_seconds
                      << ",\"mtp_draft_ms\":" << std::setprecision(4) << mtp_stats.draft_ms_per_step()
                      << ",\"mtp_stage_norm_s\":" << std::setprecision(6) << mtp_stats.stage_norm_seconds
                      << ",\"mtp_stage_attn_s\":" << std::setprecision(6) << mtp_stats.stage_attn_seconds
                      << ",\"mtp_stage_moe_s\":" << std::setprecision(6) << mtp_stats.stage_moe_seconds
                      << ",\"mtp_stage_head_s\":" << std::setprecision(6) << mtp_stats.stage_head_seconds
                      << ",\"debug_f32_hc_layer0\":" << (args.debug_f32_hc_layer0 ? "true" : "false")
                      << ",\"cuda_graph\":false"
                      << ",\"head_q8_k\":" << (head_q8_k ? "true" : "false")
                      << ",\"head_q8_k_calls\":" << wd::lowbit_head_q8_k_launch_count()
                      << ",\"kernel_family\":\"" << kernel_plan.family << "\""
                      // Task F: per-stage CUDA-event breakdown of one decode token.
                      // {"enabled":false} unless WARODECODE_PROFILE is set.
                      << ",\"profile\":" << decoder.profile_json()
                      << ",\"profile_detail\":" << decoder.profile_detail_json()
                      << ",\"token_ids\":[";
            for (size_t i = 0; i < generated.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << generated[i];
            }
            std::cout << ']';
            if (!top_logits.empty()) {
                std::cout << ",\"top_logits\":[";
                for (size_t i = 0; i < top_logits.size(); ++i) {
                    if (i != 0) std::cout << ',';
                    std::cout << '[' << top_logits[i].first << ','
                              << std::setprecision(9) << std::defaultfloat << top_logits[i].second << ']';
                }
                std::cout << ']';
                auto dump_trace = [](const char * name, const std::vector<float> & values) {
                    std::cout << ",\"" << name << "\":[";
                    for (size_t i = 0; i < values.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << std::setprecision(9) << std::defaultfloat << values[i];
                    }
                    std::cout << ']';
                };
                if (!trace.extended_parity) {
                    dump_trace("hc_init", trace.hc_init);
                    dump_trace("hc_attn_norm", trace.hc_attn_norm);
                    dump_trace("hc_attn_low", trace.hc_attn_low);
                    dump_trace("hc_attn_gate", trace.hc_attn_gate);
                    dump_trace("hc_attn_mix", trace.hc_attn_mix);
                    dump_trace("hc_attn_mix_used", trace.hc_attn_mix_used);
                    dump_trace("hc_f32_init", trace.hc_f32.init);
                    dump_trace("hc_f32_norm", trace.hc_f32.norm);
                    dump_trace("hc_f32_low_raw", trace.hc_f32.low_raw);
                    dump_trace("hc_f32_low", trace.hc_f32.low);
                    dump_trace("hc_f32_gate", trace.hc_f32.gate);
                    dump_trace("hc_f32_mix", trace.hc_f32.mixed);
                    dump_trace("gdn_f32_out", trace.gdn_f32_out);
                    dump_trace("after_attn_f32", trace.after_attn_f32);
                    dump_trace("gdn_qkv", trace.gdn_qkv);
                    dump_trace("gdn_z", trace.gdn_z);
                    dump_trace("gdn_beta", trace.gdn_beta);
                    dump_trace("gdn_alpha", trace.gdn_alpha);
                    dump_trace("gdn_gate", trace.gdn_gate);
                    dump_trace("gdn_core_out", trace.gdn_core_out);
                    dump_trace("gdn_norm_out", trace.gdn_norm_out);
                    dump_trace("attn_out", trace.attn_out);
                    dump_trace("after_attn", trace.after_attn);
                    dump_trace("hc_ffn_mix", trace.hc_ffn_mix);
                    dump_trace("hc_ffn_mix_f32", trace.hc_ffn_mix_f32);
                    dump_trace("ffn_out", trace.ffn_out);
                    dump_trace("after_ffn", trace.after_ffn);
                    dump_trace("router_logits", trace.router_logits);
                    dump_trace("router_weights", trace.router_weights);
                }
                std::cout << ",\"expert_ids\":[";
                for (size_t i = 0; i < trace.expert_ids.size(); ++i) {
                    if (i != 0) std::cout << ',';
                    std::cout << trace.expert_ids[i];
                }
                std::cout << ']';
            }
            if (trace.extended_parity || trace.state_trace) {
                const std::string & trace_path = trace.state_trace
                    ? args.state_trace_path : args.extended_parity_trace;
                std::cout << (trace.state_trace ? ",\"state_trace\":{\"format\":\"WDPTRC1-f32le\",\"path\":"
                                                : ",\"extended_parity_trace\":{\"format\":\"WDPTRC1-f32le\",\"path\":")
                          << wd::json_quote(trace_path);
                if (trace.state_trace) {
                    std::cout << ",\"forced_input_token_ids\":[";
                    for (size_t i = 0; i < args.forced_input_tokens.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << args.forced_input_tokens[i];
                    }
                    std::cout << ']';
                    if (trace.state_trace_position >= 0) {
                        std::cout << ",\"trace_position\":" << trace.state_trace_position;
                    }
                    std::cout << ",\"debug_injection_counts\":{\"moe_routes\":"
                              << trace.debug_moe_route_override_count
                              << ",\"routed_moe_outputs\":"
                              << trace.debug_routed_moe_output_override_count
                              << ",\"attention_outputs\":"
                              << trace.debug_attention_output_override_count
                              << ",\"attention_inputs\":"
                              << trace.debug_attention_input_override_count
                              << ",\"qsa_contexts\":"
                              << trace.debug_qsa_context_override_count
                              << ",\"qsa_pregates\":"
                              << trace.debug_qsa_pregate_override_count
                              << ",\"qsa_queries\":"
                              << trace.debug_qsa_query_override_count
                              << ",\"qsa_key_stores\":"
                              << trace.debug_qsa_key_store_override_count
                              << ",\"qsa_value_stores\":"
                              << trace.debug_qsa_value_store_override_count
                              << ",\"qsa_key_cache_prefixes\":"
                              << trace.debug_qsa_key_cache_prefix_override_count
                              << ",\"qsa_value_cache_prefixes\":"
                              << trace.debug_qsa_value_cache_prefix_override_count
                              << ",\"hc_attention_residuals\":"
                              << trace.debug_hc_attn_combine_override_count
                              << ",\"hc_injects\":"
                              << trace.debug_hc_inject_override_count
                              << ",\"hc_norms\":"
                              << trace.debug_hc_norm_override_count
                              << ",\"layer_inputs\":"
                              << trace.debug_layer_input_residual_override_count
                              << ",\"layer_outputs\":"
                              << trace.debug_hc_output_override_count << '}';
                }
                std::cout << ",\"vectors\":[";
                for (size_t i = 0; i < trace.extended_vectors.size(); ++i) {
                    if (i != 0) std::cout << ',';
                    const auto & vector = trace.extended_vectors[i];
                    const auto & stats = extended_trace_stats[i];
                    std::cout << "{\"name\":" << wd::json_quote(vector.name)
                              << ",\"count\":" << stats.count
                              << ",\"min\":" << std::setprecision(9) << std::defaultfloat << stats.min
                              << ",\"max\":" << stats.max
                              << ",\"max_abs\":" << stats.max_abs
                              << ",\"mean\":" << std::setprecision(17) << stats.mean
                              << ",\"l2\":" << stats.l2
                              << ",\"rms\":" << stats.rms
                              << ",\"checksum_fnv1a64\":\"0x" << std::hex << stats.checksum << std::dec << "\"}";
                }
                std::cout << ']';
                if (trace.state_trace) {
                    std::cout << ",\"summaries\":[";
                    for (size_t i = 0; i < trace.state_summaries.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        const auto & item = trace.state_summaries[i];
                        const auto & stats = item.stats;
                        std::cout << "{\"name\":" << wd::json_quote(item.name) << ",\"dims\":[";
                        for (size_t d = 0; d < item.dims.size(); ++d) {
                            if (d != 0) std::cout << ',';
                            std::cout << item.dims[d];
                        }
                        std::cout << "],\"count\":" << stats.count
                                  << ",\"min\":" << std::setprecision(9) << std::defaultfloat << stats.min
                                  << ",\"max\":" << stats.max
                                  << ",\"max_abs\":" << stats.max_abs
                                  << ",\"mean\":" << std::setprecision(17) << stats.mean
                                  << ",\"l2\":" << stats.l2
                                  << ",\"rms\":" << stats.rms
                                  << ",\"checksum_fnv1a64\":\"0x" << std::hex << stats.checksum << std::dec << "\"}";
                    }
                    std::cout << "],\"integer_vectors\":[";
                    for (size_t i = 0; i < trace.integer_vectors.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        const auto & item = trace.integer_vectors[i];
                        std::cout << "{\"name\":" << wd::json_quote(item.name) << ",\"values\":[";
                        for (size_t j = 0; j < item.values.size(); ++j) {
                            if (j != 0) std::cout << ',';
                            std::cout << item.values[j];
                        }
                        std::cout << "]}";
                    }
                    std::cout << ']';
                }
                std::cout << '}';
            }
            std::cout << "}\n";
            return 0;
        }

        if (!args.prefill_tokens.empty()) {
            throw std::runtime_error("--prefill-tokens は qwen4exp (Qwen3.8-Flash-Next) 専用です");
        }
        if (args.prompt_given || args.interactive) {
            throw std::runtime_error("--prompt/--interactive は qwen4exp (Qwen3.8-Flash-Next) 専用です "
                                     "(このモデルのarch: " + arch + ")");
        }
        wd::Model model = wd::load_model(gguf);
        if (args.autotune) wd::autotune_q4(model);
        std::cerr << "arch=" << model.arch << " layers=" << model.layers << " hidden=" << model.hidden
                  << " heads=" << model.heads << "/" << model.kv_heads << " ffn=" << model.ffn
                  << " vocab=" << model.vocab << "\n";
        std::cerr << "制約: Q4_0 dense / batch=1 / greedy / FP16 KV / CUDA only\n";

        if (args.load_only) {
            std::cout << "{\"engine\":\"warodecode-v2\",\"gpu\":\"" << prop.name
                      << "\",\"arch\":\"" << model.arch << "\",\"load_only\":true}\n";
            return 0;
        }

        if (args.start_token >= model.vocab) throw std::runtime_error("--start-token が語彙数以上です");
        wd::Runtime rt(model, args.ctx);
        const double sec = rt.run(args.generate, args.graph, args.start_token);
        const double tps = args.generate / sec;
        const uint64_t token_hash = rt.output_hash(args.generate);
        std::cout << "{\"engine\":\"warodecode-v2\",\"gpu\":\"" << prop.name
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
