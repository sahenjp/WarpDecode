from __future__ import annotations

import json
import os
import pathlib
import struct
import subprocess
import tempfile

import modal

ROOT = pathlib.Path(__file__).resolve().parent
ARTIFACTS = ROOT / "issue1_artifacts"
CPU_VOL = "warpdecode-qwen38-cpu-oracle-42506dd9"
MODEL_VOL = "warpdecode-qwen38-models"
MODEL_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"
CPU_ROOT = "/oracle/qwen38-cpu-oracle/42506dd9-6fcaa16f"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"

app = modal.App("warpdecode-qwen38-layer9-cpu-state")
cpu_oracle = modal.Volume.from_name(CPU_VOL, create_if_missing=False)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=False)
image = (
    modal.Image.from_registry("ubuntu:26.04", add_python="3.12")
    .apt_install("build-essential", "ca-certificates", "git")
    .add_local_file(
        ROOT / "qwen38_forced_cpu_trace.cpp",
        "/workspace/qwen38_forced_cpu_trace.cpp",
        copy=True,
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
        f"git -C /opt/llama.cpp checkout --detach {LLAMA_COMMIT}",
        f"test \"$(git -C /opt/llama.cpp rev-parse HEAD)\" = {LLAMA_COMMIT}",
    )
)


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def select_trace_vectors(path: pathlib.Path, wanted: set[str]) -> tuple[bytes, dict[str, int]]:
    raw = path.read_bytes()
    if raw[:8] != b"WDPTRC1\0":
        raise RuntimeError("invalid CPU trace header")
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 1:
        raise RuntimeError(f"unsupported CPU trace version: {version}")
    offset = 16
    entries = []
    vectors = {}
    for _ in range(count):
        entry_start = offset
        name_size, value_count = struct.unpack_from("<IQ", raw, offset)
        offset += 12
        name = raw[offset : offset + name_size].decode()
        offset += name_size + value_count * 4
        vectors[name] = value_count
        if name in wanted:
            entries.append(raw[entry_start:offset])
    if offset != len(raw):
        raise RuntimeError("unexpected trailing CPU trace data")
    if not wanted <= vectors.keys():
        raise RuntimeError(f"CPU trace is missing requested vectors: {wanted - vectors.keys()}")
    header = b"WDPTRC1\0" + struct.pack("<II", 1, len(entries))
    return header + b"".join(entries), {name: vectors[name] for name in wanted}


@app.function(
    image=image,
    volumes={"/oracle": cpu_oracle.with_mount_options(read_only=True),
             "/models": models.with_mount_options(read_only=True)},
    cpu=4,
    memory=16384,
    timeout=3600,
)
def capture(input_ids: list[int], expected_ids: list[int]) -> dict:
    model = pathlib.Path("/models") / MODEL_REPO.replace("/", "__") / "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
    if model.stat().st_size != 85923705344:
        raise RuntimeError("mounted model size does not match the canonical GGUF")
    cpu_bin = pathlib.Path(CPU_ROOT) / "bin"
    helper = pathlib.Path("/tmp/qwen38-forced-cpu-trace")
    compile_cmd = [
        "c++", "-std=c++17", "-O3", "-DNDEBUG",
        "-I/opt/llama.cpp/include", "-I/opt/llama.cpp/src",
        "-I/opt/llama.cpp/ggml/include", "-I/opt/llama.cpp/ggml/src",
        "/workspace/qwen38_forced_cpu_trace.cpp",
        str(cpu_bin / "libllama.so.0"), str(cpu_bin / "libggml-cpu.so.0"),
        str(cpu_bin / "libggml-base.so.0"), str(cpu_bin / "libggml.so.0"),
        "-fopenmp", "-pthread", f"-Wl,-rpath,{cpu_bin}", "-o", str(helper),
    ]
    print(run(["c++", "--version"]).splitlines()[0], flush=True)
    print(run(compile_cmd), flush=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(cpu_bin) + ":" + env.get("LD_LIBRARY_PATH", "")

    with tempfile.TemporaryDirectory(prefix="qwen38-layer9-state-") as tmp:
        trace_path = pathlib.Path(tmp) / "cpu.bin"
        output = run([
            str(helper), str(model), "128", ",".join(map(str, input_ids)),
            str(trace_path), "36",
        ], env=env)
        result_line = next(line for line in reversed(output.splitlines())
                           if line.startswith("LLAMA_STATE_RESULT="))
        payload = result_line.removeprefix("LLAMA_STATE_RESULT=")
        token_start = payload.index('"token_ids":[') + len('"token_ids":[')
        token_end = payload.index("]", token_start)
        cpu_tokens = [int(token) for token in payload[token_start:token_end].split(",")]
        if cpu_tokens != expected_ids or cpu_tokens[-1] != 56212:
            raise RuntimeError("instrumented CPU trace did not reproduce the saved canonical output")
        required = {
            f"pos36.{name}-{layer}"
            for layer in (9, 10)
            for name in ("state_predelta", "new_state", "conv_state_at")
        }
        trace, vectors = select_trace_vectors(trace_path, required)
        return {
            "cpu_tokens_match_canonical": True,
            "token36": cpu_tokens[-1],
            "llama_cpp_commit": LLAMA_COMMIT,
            "trace_position": 36,
            "state_vector_counts": {name: vectors[name] for name in sorted(required)},
            "trace_bytes": trace,
        }


@app.local_entrypoint()
def main() -> None:
    warp = json.loads((ARTIFACTS / "warodecode-state-position-36-gdn.json").read_text())
    cpu = json.loads((ARTIFACTS / "llama-state-position-36-gdn.json").read_text())
    result = capture.remote(
        warp["state_trace"]["forced_input_token_ids"], cpu["token_ids"])
    trace = result.pop("trace_bytes")
    with tempfile.TemporaryDirectory(prefix="position36-cpu-gdn-state-") as tmp:
        combined_path = pathlib.Path(tmp) / "cpu-state.bin"
        combined_path.write_bytes(trace)
        for layer in (9, 10):
            required = {
                f"pos36.{name}-{layer}"
                for name in ("state_predelta", "new_state", "conv_state_at")
            }
            layer_trace, counts = select_trace_vectors(combined_path, required)
            trace_path = ARTIFACTS / f"position36_layer{layer}_cpu_state_vectors.bin"
            trace_path.write_bytes(layer_trace)
            layer_result = {
                **result,
                "layer": layer,
                "state_vector_counts": {name: counts[name] for name in sorted(required)},
                "trace_file": str(trace_path),
                "trace_size_bytes": len(layer_trace),
            }
            result_path = ARTIFACTS / f"position36_layer{layer}_cpu_state_vectors.json"
            result_path.write_text(json.dumps(layer_result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
