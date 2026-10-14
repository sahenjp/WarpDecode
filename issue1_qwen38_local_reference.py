from __future__ import annotations

import ast
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_URL = (
    "https://huggingface.co/pentacoxian-dev/"
    "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/resolve/main/"
    "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
)
MODEL_NAME = "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
T4_TRACE_LOG = pathlib.Path(
    "/home/sahen/.local/share/opencode/tool-output/tool_0df0dcc6f001cTViTzGMWU45fa"
)


def reference_cpp() -> str:
    tree = ast.parse((ROOT / "issue1_qwen38_parity.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REFERENCE_CPP"
            for target in node.targets
        ):
            cpp = ast.literal_eval(node.value)
            cpp = cpp.replace(
                "mp.n_gpu_layers = -1;",
                "mp.n_gpu_layers = 0;\n        mp.load_mode = LLAMA_LOAD_MODE_MMAP;",
            )
            cpp = cpp.replace(
                '"hc_init", "hc_mixed-0",',
                '"hc_init", "hc_norm-0", "hc_gate-0", "hc_mixed-0",',
            )
            cpp = cpp.replace(
                '"hc_gate-0", "hc_mixed-0",',
                '"hc_gate-0", "hc_low-0", "hc_mixed-0",',
            )
            cpp = cpp.replace(
                '"hc_low-0", "hc_mixed-0",',
                '"hc_low-0", "linear_attn_qkv_mixed-0", "z-0", "beta_sigmoid-0", '
                '"alpha-0", "gate-0", "attn_output-0", "final_output-0", "hc_mixed-0",',
            )
            cpp = cpp.replace(
                "cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;",
                "cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;",
            )
            cpp = cpp.replace("cp.n_batch = 1024;", "cp.n_batch = 1;")
            cpp = cpp.replace("cp.n_ubatch = 256;", "cp.n_ubatch = 1;")
            cpp = cpp.replace(
                "cp.type_v = GGML_TYPE_F16;",
                "cp.type_v = GGML_TYPE_F16;\n        cp.n_threads = 2;\n        cp.n_threads_batch = 2;",
            )
            return cpp
    raise RuntimeError("REFERENCE_CPP not found")


def fuse_worker_source(cache_path: pathlib.Path, progress_path: pathlib.Path) -> str:
    tree = ast.parse((ROOT / "issue1_colab_probe.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FUSE_WORKER"
            for target in node.targets
        ):
            worker = ast.literal_eval(node.value)
            return (
                worker.replace("/content/qwen38-range-cache.bin", str(cache_path))
                .replace("/content/qwen38-fuse-progress.json", str(progress_path))
            )
    raise RuntimeError("FUSE_WORKER not found")


def warp_t4_trace() -> dict:
    outer = json.loads(T4_TRACE_LOG.read_text())
    for line in outer.get("log_tail", []):
        if line.startswith('{"engine":"warodecode-qwen38"'):
            return json.loads(line)
    raise RuntimeError("saved T4 warodecode diagnostic trace not found")


def run(cmd: list[str], *, cwd: pathlib.Path | None = None, timeout: int = 3600) -> str:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(
        cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    if p.returncode:
        print("\n".join((p.stdout + "\n" + p.stderr).splitlines()[-80:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def compare(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) != len(candidate):
        return {"llama_len": len(reference), "warp_len": len(candidate)}
    dot = sum(a * b for a, b in zip(reference, candidate))
    nr = sum(a * a for a in reference) ** 0.5
    nc = sum(b * b for b in candidate) ** 0.5
    return {
        "max_abs": max(abs(a - b) for a, b in zip(reference, candidate)),
        "rmse": (sum((a - b) ** 2 for a, b in zip(reference, candidate)) / len(reference)) ** 0.5,
        "cosine": dot / (nr * nc) if nr and nc else None,
        "llama_prefix": reference[:8],
        "warp_prefix": candidate[:8],
    }


def main() -> None:
    t4 = warp_t4_trace()
    if t4.get("token_ids") != [238920]:
        raise RuntimeError(f"unexpected saved T4 input/output: {t4.get('token_ids')}")
    if not pathlib.Path("/dev/fuse").exists():
        raise RuntimeError("local /dev/fuse is unavailable")

    with tempfile.TemporaryDirectory(prefix="qwen38-cpu-ref-", dir="/home/sahen/.cache") as tmp:
        work = pathlib.Path(tmp)
        mount = work / "model"
        mount.mkdir()
        cache = work / "sparse-cache.gguf"
        progress = work / "range-progress.json"
        worker_file = work / "fuse_worker.py"
        worker_file.write_text(fuse_worker_source(cache, progress))
        fuse_log = open(work / "fuse.log", "w")
        fuse = subprocess.Popen(
            [sys.executable, str(worker_file), MODEL_URL, str(mount)],
            stdout=fuse_log, stderr=fuse_log, start_new_session=True,
        )

        try:
            model = mount / MODEL_NAME
            for _ in range(90):
                if pathlib.Path(f"/proc/{fuse.pid}").exists() and model.exists():
                    break
                if fuse.poll() is not None:
                    fuse_log.flush()
                    print(pathlib.Path(fuse_log.name).read_text(errors="replace")[-4000:])
                    raise RuntimeError("HF FUSE worker exited")
                time.sleep(1)
            if not model.exists():
                raise RuntimeError("HF GGUF FUSE path was not created")

            with model.open("rb") as f:
                if f.read(4) != b"GGUF": raise RuntimeError("bad streamed GGUF magic")

            llama = work / "llama.cpp"
            run(["rtk", "git", "init", str(llama)])
            run(["rtk", "git", "-C", str(llama), "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"])
            run(["rtk", "git", "-C", str(llama), "fetch", "--depth=1", "origin", "refs/pull/28243/head"], timeout=900)
            run(["rtk", "git", "-C", str(llama), "checkout", "--detach", "FETCH_HEAD"])
            rev = run(["rtk", "git", "-C", str(llama), "rev-parse", "HEAD"]).strip()
            if rev != LLAMA_COMMIT: raise RuntimeError(f"unexpected llama commit: {rev}")

            cpp = work / "qwen38_reference.cpp"
            cpp.write_text(reference_cpp())
            cmake = llama / "CMakeLists.txt"
            cmake.write_text(
                cmake.read_text()
                + f"\nadd_executable(qwen38-reference {cpp})\n"
                + "target_link_libraries(qwen38-reference PRIVATE llama)\n"
            )
            build = work / "build"
            run([
                "rtk", "cmake", "-S", str(llama), "-B", str(build),
                "-DGGML_CUDA=OFF", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF",
                "-DGGML_CPU_REPACK=OFF", "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF",
            ])
            cache = (build / "CMakeCache.txt").read_text()
            if "GGML_CPU_REPACK:BOOL=OFF" not in cache:
                raise RuntimeError("CMake did not apply GGML_CPU_REPACK=OFF")
            run(["rtk", "cmake", "--build", str(build), "-j1", "--target", "qwen38-reference"], timeout=3600)
            output = run([str(build / "bin/qwen38-reference"), str(model), "128", "1"], timeout=21600)
            result = next(
                json.loads(line.removeprefix("LLAMA_RESULT="))
                for line in reversed(output.splitlines())
                if line.startswith("LLAMA_RESULT=")
            )

            trace = result.get("trace", [])
            by_name: dict[str, list[list[float]]] = {}
            for entry in trace:
                by_name.setdefault(entry["name"], []).append(entry["values"])

            pairs = [
                ("hc_init", "hc_init", 0),
                ("hc_attn_mix", "hc_mixed-0", 0),
                ("attn_out", "linear_attn_out-0", 0),
                ("after_attn", "hc_combine-0", 0),
                ("hc_ffn_mix", "hc_mixed-0", 1),
                ("ffn_out", "ffn_out-0", 0),
                ("after_ffn", "hc_combine-0", 1),
            ]
            diffs = {}
            for warp_name, llama_name, index in pairs:
                vals = by_name.get(llama_name, [])
                if len(vals) > index and t4.get(warp_name):
                    diffs[warp_name] = compare(vals[index], t4[warp_name])

            print("LOCAL_LLAMA_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
            print("T4_VS_LLAMA_CPU_TRACE=" + json.dumps({
                "t4_token": t4["token_ids"][0],
                "llama_cpu_token": result["token_ids"][0],
                "llama_commit": LLAMA_COMMIT,
                "trace_names": [x["name"] for x in trace],
                "layer_diffs": diffs,
            }, ensure_ascii=False), flush=True)
        finally:
            subprocess.run(["fusermount", "-u", str(mount)], check=False, capture_output=True)
            if fuse.poll() is None:
                fuse.terminate()
                fuse.wait(timeout=10)
            fuse_log.close()


if __name__ == "__main__":
    main()
