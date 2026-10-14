from __future__ import annotations

import ast
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = "warpdecode-qwen38-t4"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"

REMOTE = r'''import pathlib
import subprocess
import sys
import time
import json
import os

LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
model = "/content/qwen38-http/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
model_url = ("https://huggingface.co/pentacoxian-dev/"
             "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/resolve/"
             "0cd618ade637402b7870e993135233f7ae78387c/"
             "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
model_bytes = 85923705344
repo = pathlib.Path("/content/llama.cpp")
FUSE_WORKER_SOURCE = __FUSE_WORKER__

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open("/content/llama-reference.log", "a") as f:
        f.write("+ " + " ".join(cmd) + "\n" + p.stdout + "\n")
    if p.returncode:
        print("\\n".join(p.stdout.splitlines()[-80:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    if p.stdout:
        print("\\n".join(p.stdout.splitlines()[-20:]), flush=True)
    return p.stdout

def capture(cmd):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        print("\\n".join(p.stdout.splitlines()[-80:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    for line in p.stdout.splitlines()[-8:]:
        if line.startswith("LLAMA_RESULT=") or line.startswith("{"):
            print(line[:2000], flush=True)
    return p.stdout

def result_line(output, prefix):
    for line in reversed(output.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError(f"missing {prefix} result")

if not repo.exists():
    repo.mkdir()
    run(["git", "-C", str(repo), "init"])
    run(["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"])
    run(["git", "-C", str(repo), "fetch", "--depth=1", "origin", "refs/pull/28243/head"])
    run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"])
head = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
if head != LLAMA_COMMIT:
    raise RuntimeError(f"unexpected llama.cpp revision: {head}")

source = pathlib.Path("/content/warodecode-qwen38-reference.cpp")
source.write_text(__REFERENCE_CPP__)
cmake_file = repo / "CMakeLists.txt"
text = cmake_file.read_text()
target = "add_executable(warodecode-qwen38-reference"
if target not in text:
    cmake_file.write_text(
        text + "\nadd_executable(warodecode-qwen38-reference /content/warodecode-qwen38-reference.cpp)\n"
        + "target_link_libraries(warodecode-qwen38-reference PRIVATE llama)\n"
    )

build = repo / "build"
run([
    "cmake", "-S", str(repo), "-B", str(build),
    "-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
])
run(["cmake", "--build", str(build), "-j2", "--target", "warodecode-qwen38-reference"])

mount = pathlib.Path("/content/qwen38-http")
mount.mkdir(exist_ok=True)
fuse_log = None
if not os.path.ismount(mount):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "fusepy"], check=True)
    worker = pathlib.Path("/content/qwen38_fuse_worker.py")
    worker.write_text(FUSE_WORKER_SOURCE)
    fuse_log = open("/content/qwen38-fuse.log", "w")
    fuse = subprocess.Popen([sys.executable, str(worker), model_url, str(mount)],
                            stdout=fuse_log, stderr=fuse_log, start_new_session=True)
    pathlib.Path("/content/qwen38-fuse.pid").write_text(str(fuse.pid))
try:
    for _ in range(120):
        if os.path.ismount(mount) and (mount / pathlib.Path(model).name).exists():
            break
        if fuse.poll() is not None:
            fuse_log.flush()
            tail = open("/content/qwen38-fuse.log").read()[-2000:]
            raise RuntimeError(f"FUSE model mount process exited; log tail: {tail}")
        time.sleep(1)
    target = mount / pathlib.Path(model).name
    if not os.path.ismount(mount) or not target.exists():
        raise RuntimeError("FUSE model path was not mounted")
    if target.stat().st_size != model_bytes:
        raise RuntimeError(f"mounted model size mismatch: {target.stat().st_size}")

    warp_out = capture([
        "/content/warodecode/build-sm75/warodecode",
        "--model", str(target), "--ctx", "128", "--device", "0",
        "--start-token", "1", "--generate", "1", "--no-graph", "--top-k", "10",
    ])
    warp = result_line(warp_out, "")
    llama_out = capture([str(build / "bin/warodecode-qwen38-reference"), str(target), "128", "1"])
    llama = result_line(llama_out, "LLAMA_RESULT=")

    token_match = llama["token_ids"] == warp["token_ids"]
    warp_top = [x[0] for x in warp["top_logits"]]
    llama_top = [x[0] for x in llama["top_logits"]]
    trace = llama.get("trace", [])

    def diff(reference, candidate):
        if len(reference) != len(candidate):
            return {"llama_len": len(reference), "warp_len": len(candidate)}
        dot = sum(x * y for x, y in zip(reference, candidate))
        nr = sum(x * x for x in reference) ** 0.5
        nc = sum(y * y for y in candidate) ** 0.5
        return {
            "max_abs": max(abs(x - y) for x, y in zip(reference, candidate)),
            "rmse": (sum((x - y) ** 2 for x, y in zip(reference, candidate)) / len(reference)) ** 0.5,
            "cosine": dot / (nr * nc) if nr and nc else None,
            "llama_prefix": reference[:8],
            "warp_prefix": candidate[:8],
        }

    def trace_values(name, index=0):
        matches = [entry["values"] for entry in trace if entry["name"] == name]
        return matches[index] if len(matches) > index else None

    comparisons = {}
    for label, name, index, candidate in (
        ("hc_init", "hc_init", 0, "hc_init"),
        ("hc_attn_mix", "hc_mixed-0", 0, "hc_attn_mix"),
        ("attn_out", "linear_attn_out-0", 0, "attn_out"),
        ("after_attn", "hc_combine-0", 0, "after_attn"),
        ("hc_ffn_mix", "hc_mixed-0", 1, "hc_ffn_mix"),
        ("ffn_out", "ffn_out-0", 0, "ffn_out"),
        ("after_ffn", "hc_combine-0", 1, "after_ffn"),
    ):
        reference = trace_values(name, index)
        if reference is not None:
            comparisons[label] = diff(reference, warp[candidate])

    result = {
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              text=True, capture_output=True).stdout.strip(),
        "model": model,
        "llama_commit": LLAMA_COMMIT,
        "context": 128,
        "input_token_ids": [1],
        "mtp": False,
        "warodecode_token": warp["token_ids"][0],
        "llama_token": llama["token_ids"][0],
        "token_match": token_match,
        "top10_overlap": len(set(warp_top) & set(llama_top)),
        "warodecode_top_logits": warp["top_logits"],
        "llama_top_logits": llama["top_logits"],
        "llama_trace_names": [entry["name"] for entry in trace],
        "layer_diffs": comparisons,
        "warodecode_seconds": warp["seconds"],
        "llama_seconds": llama["seconds"],
    }
    print("T4_PARITY_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
finally:
    if fuse_log:
        fuse_log.close()
'''


def reference_cpp() -> str:
    tree = ast.parse((ROOT / "issue1_qwen38_parity.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REFERENCE_CPP"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("REFERENCE_CPP not found")


def fuse_worker_source() -> str:
    tree = ast.parse((ROOT / "issue1_colab_probe.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FUSE_WORKER"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("FUSE_WORKER not found")


def main() -> None:
    worker_code = REMOTE.replace("__REFERENCE_CPP__", repr(reference_cpp()))
    worker_code = worker_code.replace("__FUSE_WORKER__", repr(fuse_worker_source()))
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-ref-") as temp:
        worker = pathlib.Path(temp) / "run_reference.py"
        worker.write_text(worker_code)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(worker)]
        print("+", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.stdout:
            print(p.stdout, flush=True)
        if p.returncode:
            raise RuntimeError(f"Colab reference run failed ({p.returncode})")


if __name__ == "__main__":
    main()
