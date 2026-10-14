from __future__ import annotations

"""
warodecode / Qwen3.8-Flash-Next Modal benchmark runner.

Run this file from the warodecode repository root.

Setup:
    python -m pip install -U modal
    modal setup

1) Download/cache the GGUF into a persistent Modal Volume without a GPU:
    modal run bench_flash_next.py --action prefetch

2) Benchmark on A100 80GB:
    modal run bench_flash_next.py --action bench --gpu A100-80GB

Optional:
    modal run bench_flash_next.py --action bench --gpu A100-80GB --depths 0,8192,32768
"""

import json
import pathlib
import subprocess
import time

import modal

APP_NAME = "warpdecode-qwen38-bench"
MODEL_VOL = "warpdecode-qwen38-models"
DEFAULT_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"

# Pinned so a later llama.cpp change does not silently move the baseline.
LLAMA_COMMIT = "4e7481175cbd4759df8bee2f1c1a0073effbebd7"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "ninja-build",
        "build-essential",
        "curl",
        "ca-certificates",
        "pkg-config",
    )
    .pip_install("cmake>=3.31,<4", "huggingface_hub[hf_xet]")
    # Run `modal run` from the warodecode repository root.
    .add_local_dir(
        ".",
        "/workspace/warodecode",
        copy=True,
        ignore=["build*", ".git", "__pycache__", "*.pyc"],
    )
    # These builds happen while constructing the image, before paid GPU benchmark time.
    .run_commands(
        "echo BENCH_FLASH_NEXT_V2 && which cmake && cmake --version",
        "cmake -S /workspace/warodecode -B /workspace/warodecode/build-sm80 "
        "-G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DWARODECODE_BUILD_ENGINE=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /workspace/warodecode/build-sm80 -j8",
        "git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
        f"git -C /opt/llama.cpp checkout {LLAMA_COMMIT}",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja "
        "-DGGML_CUDA=ON "
        "-DLLAMA_CURL=OFF "
        "-DGGML_NATIVE=OFF "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /opt/llama.cpp/build -j8 "
        "--target llama-bench llama-cli",
    )
)


def run(
    cmd: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        text=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def find_target(root: pathlib.Path) -> pathlib.Path:
    # A detached draft sidecar normally starts with "mtp-".
    files = [
        p
        for p in root.rglob("*.gguf")
        if not p.name.lower().startswith("mtp-")
    ]

    # llama.cpp accepts the first shard path and discovers the other shards.
    first_shards = [p for p in files if "-00001-of-" in p.name]
    candidates = first_shards or files

    if not candidates:
        raise RuntimeError(f"target GGUF not found under {root}")

    # Usually there is only one quant family in this repository.
    # Prefer the largest first shard if there are multiple candidates.
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


@app.function(
    image=image,
    volumes={"/models": models},
    cpu=4,
    memory=16384,
    timeout=7200,
)
def prefetch_model(
    repo_id: str = DEFAULT_REPO,
    revision: str = "main",
) -> dict:
    from huggingface_hub import snapshot_download

    dest = pathlib.Path("/models") / repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["*.gguf", "*.json", "README*", "*.md"],
        max_workers=16,
    )
    models.commit()

    ggufs = sorted(
        str(p.relative_to(dest))
        for p in dest.rglob("*.gguf")
    )

    result = {
        "repo": repo_id,
        "path": path,
        "seconds": round(time.perf_counter() - t0, 2),
        "gguf_count": len(ggufs),
        "ggufs": ggufs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@app.function(
    image=image,
    volumes={"/models": models.with_mount_options(read_only=True)},
    cpu=8,
    memory=32768,
    timeout=1500,
)
def benchmark(
    repo_id: str = DEFAULT_REPO,
    depths: str = "0,8192,32768",
) -> dict:
    model_root = pathlib.Path("/models") / repo_id.replace("/", "__")
    target = find_target(model_root)

    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,memory.total,driver_version,pstate,"
            "clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ]
    ).stdout.strip()
    print("GPU:", gpu, flush=True)

    # CUDA compilation is done during the image build.
    wd_bin = pathlib.Path(
        "/workspace/warodecode/build-sm80/warodecode"
    )
    if not wd_bin.exists():
        raise RuntimeError(
            "warodecode CUDA binary missing from image build"
        )

    # Confirm that the qwen4exp/split-GGUF planner sees this model.
    tui = run(
        [
            "/workspace/warodecode/build-sm80/warodecode-tui",
            "--dump-state",
            "--model-dir",
            str(model_root),
        ]
    ).stdout
    print("warodecode model scan:\n", tui, flush=True)

    # Current warodecode qwen4exp forward is not wired yet.
    # This run establishes an upstream llama.cpp baseline on exactly
    # the same Modal GPU while also proving warodecode SM80 compiles.
    bench_cmd = [
        "/opt/llama.cpp/build/bin/llama-bench",
        "-m",
        str(target),
        "-ngl",
        "all",
        "-ncmoe",
        "0",
        "-fa",
        "1",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-p",
        "0",
        "-n",
        "64",
        "-d",
        depths,
        "-b",
        "1024",
        "-ub",
        "256",
        "-r",
        "3",
        "-o",
        "jsonl",
    ]

    t0 = time.perf_counter()
    proc = run(bench_cmd)
    elapsed = time.perf_counter() - t0
    out = proc.stdout
    print(out, flush=True)

    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    compact = [
        {
            "depth": r.get("n_depth"),
            "avg_ts": r.get("avg_ts"),
            "stddev_ts": r.get("stddev_ts"),
            "model_size": r.get("model_size"),
            "gpu_info": r.get("gpu_info"),
            "build_commit": r.get("build_commit"),
        }
        for r in rows
        if r.get("n_gen", 0) > 0
    ]

    result = {
        "repo": repo_id,
        "target": str(target),
        "gpu": gpu,
        "elapsed_s": round(elapsed, 2),
        "depths": depths,
        "llama_cpp": compact,
        "warodecode_qwen4exp_forward": "not-wired-yet",
    }

    print(
        "RESULT=" + json.dumps(result, ensure_ascii=False),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main(
    action: str = "bench",
    repo: str = DEFAULT_REPO,
    gpu: str = "A100-80GB",
    depths: str = "0,8192,32768",
):
    if action == "prefetch":
        result = prefetch_model.remote(repo)
    elif action == "bench":
        result = benchmark.with_options(gpu=gpu).remote(
            repo,
            depths,
        )
    else:
        raise ValueError("action must be prefetch or bench")

    print(json.dumps(result, ensure_ascii=False, indent=2))
