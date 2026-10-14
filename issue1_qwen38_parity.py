from __future__ import annotations

import json
import pathlib
import subprocess

import modal

APP_NAME = "warpdecode-qwen38-issue1-parity"
MODEL_VOL = "warpdecode-qwen38-models"
REPO_ID = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=False)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ninja-build", "build-essential", "ca-certificates", "pkg-config")
    .pip_install("cmake>=3.31,<4")
    .add_local_dir(
        ".",
        "/workspace/warodecode",
        copy=True,
        ignore=[
            "build*",
            ".git",
            ".serena",
            "__pycache__",
            "*.pyc",
            "*.zip",
            "*.webp",
            "bench_flash_next*.py",
            "probe_flash_next.py",
            "warodecode_qwen38_smoke*.py",
            "issue1_qwen38_parity.py",
        ],
    )
    .run_commands(
        "cmake -S /workspace/warodecode -B /workspace/warodecode/build-sm80 "
        "-G Ninja -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /workspace/warodecode/build-sm80 -j8",
        "git init /opt/llama.cpp",
        "git -C /opt/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git",
        "git -C /opt/llama.cpp fetch --depth=1 origin refs/pull/28243/head",
        "git -C /opt/llama.cpp checkout --detach FETCH_HEAD",
        f'test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "{LLAMA_COMMIT}"',
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja "
        "-DGGML_CUDA=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
        "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /opt/llama.cpp/build -j8 --target llama-bench llama-cli",
    )
)

REFERENCE_CPP = r'''#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct TraceTensor {
    std::string name;
    std::array<int64_t, 4> dims{};
    std::vector<float> values;
};

struct TraceCapture {
    std::vector<TraceTensor> tensors;
};

static bool trace_wanted(const char * name) {
    static const char * const names[] = {
        "hc_init", "hc_mixed-0", "linear_attn_out-0", "hc_combine-0",
        "ffn_out-0", "l_last-0", "result_norm",
    };
    for (const char * candidate : names) if (std::strcmp(name, candidate) == 0) return true;
    return false;
}

static bool capture_trace(ggml_tensor * t, bool ask, void * user_data) {
    if (!trace_wanted(t->name)) return true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) return true;

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    } else {
        std::memcpy(raw.data(), t->data, nbytes);
    }

    TraceTensor item;
    item.name = t->name;
    std::copy(std::begin(t->ne), std::end(t->ne), item.dims.begin());
    const int64_t n = ggml_nelements(t);
    item.values.reserve(static_cast<size_t>(n));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off = static_cast<size_t>(i0) * t->nb[0]
                                     + static_cast<size_t>(i1) * t->nb[1]
                                     + static_cast<size_t>(i2) * t->nb[2]
                                     + static_cast<size_t>(i3) * t->nb[3];
                    float v = 0.0f;
                    if (t->type == GGML_TYPE_F32) {
                        std::memcpy(&v, raw.data() + off, sizeof(v));
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        std::memcpy(&h, raw.data() + off, sizeof(h));
                        v = ggml_fp16_to_fp32(h);
                    } else {
                        ggml_bf16_t h;
                        std::memcpy(&h, raw.data() + off, sizeof(h));
                        v = ggml_bf16_to_fp32(h);
                    }
                    item.values.push_back(v);
                }
            }
        }
    }
    static_cast<TraceCapture *>(user_data)->tensors.push_back(std::move(item));
    return true;
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

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: reference MODEL CTX STEPS");
        const std::string path = argv[1];
        const uint32_t ctx_size = static_cast<uint32_t>(std::stoi(argv[2]));
        const int steps = std::stoi(argv[3]);

        llama_backend_init();
        const llama_model_tensor_buft_override buft_overrides[] = {
            {R"(ffn_.*_exps\.weight)", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.load_mtp = false;
        mp.tensor_buft_overrides = buft_overrides;
        llama_model * model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) throw std::runtime_error("llama_model_load_from_file failed");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx_size;
        cp.n_batch = 1024;
        cp.n_ubatch = 256;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        TraceCapture trace;
        cp.cb_eval = capture_trace;
        cp.cb_eval_user_data = &trace;
        llama_context * context = llama_init_from_model(model, cp);
        if (!context) throw std::runtime_error("llama_init_from_model failed");

        std::vector<llama_token> ids;
        ids.reserve(static_cast<size_t>(steps));
        std::vector<std::pair<llama_token, float>> first_top;
        llama_token token = 1;
        llama_token first = -1;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) {
            llama_batch batch = llama_batch_get_one(&token, 1);
            const int rc = llama_decode(context, batch);
            if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));

            const float * logits = llama_get_logits_ith(context, -1);
            if (!logits) throw std::runtime_error("llama_get_logits_ith returned null");
            float best = -std::numeric_limits<float>::infinity();
            llama_token next = 0;
            std::vector<std::pair<float, llama_token>> ranked;
            if (i == 0) ranked.reserve(static_cast<size_t>(n_vocab));
            for (int t = 0; t < n_vocab; ++t) {
                const float value = logits[t];
                if (!std::isfinite(value)) throw std::runtime_error("llama logits contain NaN or infinity");
                if (i == 0) ranked.emplace_back(value, t);
                if (value > best || (value == best && t < next)) {
                    best = value;
                    next = t;
                }
            }
            if (i == 0) {
                const auto by_logit = [](const auto & a, const auto & b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                };
                std::partial_sort(ranked.begin(), ranked.begin() + 10, ranked.end(), by_logit);
                for (int j = 0; j < 10; ++j) {
                    first_top.emplace_back(ranked[j].second, ranked[j].first);
                }
            }
            if (i == 0) first = next;
            ids.push_back(next);
            token = next;
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();

        std::cout << "LLAMA_RESULT={\"ctx\":" << ctx_size
                  << ",\"tokens\":" << ids.size()
                  << ",\"seconds\":" << std::fixed << std::setprecision(6) << seconds
                  << ",\"tok_s\":" << std::setprecision(3) << ids.size() / seconds
                  << ",\"first_token\":" << first
                  << ",\"token_hash\":\"0x" << std::hex << token_hash(ids) << std::dec
                  << "\",\"token_ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << "],\"top_logits\":[";
        for (size_t i = 0; i < first_top.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '[' << first_top[i].first << ','
                      << std::setprecision(9) << std::defaultfloat << first_top[i].second << ']';
        }
        std::cout << "],\"trace\":[";
        for (size_t i = 0; i < trace.tensors.size(); ++i) {
            const auto & t = trace.tensors[i];
            if (i) std::cout << ',';
            std::cout << "{\"name\":\"" << t.name << "\",\"dims\":["
                      << t.dims[0] << ',' << t.dims[1] << ',' << t.dims[2] << ',' << t.dims[3]
                      << "],\"values\":[";
            for (size_t j = 0; j < t.values.size(); ++j) {
                if (j) std::cout << ',';
                std::cout << std::setprecision(9) << std::defaultfloat << t.values[j];
            }
            std::cout << "]}";
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
'''


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def result_line(output: str, prefix: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError(f"missing {prefix} output")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": models.with_mount_options(read_only=True)},
    cpu=8,
    memory=32768,
    timeout=7200,
)
def parity(repo_id: str = REPO_ID, ctx: int = 128, steps: int = 64) -> dict:
    model_root = pathlib.Path("/models") / repo_id.replace("/", "__")
    model = model_root / "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")

    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    lowbit = run(["/workspace/warodecode/build-sm80/warodecode-lowbit-test"])

    source = pathlib.Path("/tmp/warodecode-qwen38-reference.cpp")
    source.write_text(REFERENCE_CPP)
    cmake_file = pathlib.Path("/opt/llama.cpp/CMakeLists.txt")
    cmake_file.write_text(
        cmake_file.read_text()
        + "\nadd_executable(warodecode-qwen38-reference /tmp/warodecode-qwen38-reference.cpp)\n"
        + "target_link_libraries(warodecode-qwen38-reference PRIVATE llama)\n"
    )
    run([
        "cmake", "-S", "/opt/llama.cpp", "-B", "/opt/llama.cpp/build", "-G", "Ninja",
        "-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=80",
    ])
    run(["cmake", "--build", "/opt/llama.cpp/build", "-j8", "--target", "warodecode-qwen38-reference"])

    wd_out = run([
        "/workspace/warodecode/build-sm80/warodecode",
        "--model", str(model), "--ctx", str(ctx), "--device", "0",
        "--start-token", "1", "--generate", str(steps), "--no-graph",
        "--top-k", "10",
    ])
    ref_out = run([
        "/opt/llama.cpp/build/bin/warodecode-qwen38-reference",
        str(model), str(ctx), str(steps),
    ])
    ref = result_line(ref_out, "LLAMA_RESULT=")
    wd = result_line(wd_out, "")
    def compare_vectors(reference: list[float], candidate: list[float]) -> dict:
        dot = sum(a * b for a, b in zip(reference, candidate))
        ref_norm = sum(a * a for a in reference) ** 0.5
        candidate_norm = sum(b * b for b in candidate) ** 0.5
        return {
            "max_abs": max(abs(a - b) for a, b in zip(reference, candidate)),
            "rmse": (sum((a - b) ** 2 for a, b in zip(reference, candidate)) / len(reference)) ** 0.5,
            "cosine": dot / (ref_norm * candidate_norm) if ref_norm and candidate_norm else None,
            "llama_prefix": reference[:8],
            "warodecode_prefix": candidate[:8],
        }

    layer0_input = compare_vectors(ref.pop("layer0_input"), wd.pop("hc_init"))
    layer1_input = compare_vectors(ref.pop("layer1_input"), wd.pop("after_ffn"))
    same = ref["token_ids"] == wd["token_ids"]
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(ref["token_ids"], wd["token_ids"])) if a != b),
        None,
    )
    result = {
        "repo": repo_id,
        "model": str(model),
        "gpu": run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip(),
        "llama_commit": LLAMA_COMMIT,
        "ctx": ctx,
        "steps": steps,
        "mtp": False,
        "lowbit": [line for line in lowbit.splitlines() if " PASS " in line],
        "llama_cpp": ref,
        "warodecode": wd,
        "layer0_input_diff": layer0_input,
        "layer1_input_diff": layer1_input,
        "first_token_match": ref["first_token"] == wd["token_ids"][0],
        "sequence_match": same,
        "first_mismatch_position": mismatch,
    }
    print("PARITY_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main(context: int = 128, steps: int = 64):
    print(json.dumps(parity.remote(REPO_ID, context, steps), ensure_ascii=False, indent=2))
