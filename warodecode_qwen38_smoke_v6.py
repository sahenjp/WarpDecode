from __future__ import annotations

import json
import pathlib
import subprocess

import modal

APP_NAME = "warpdecode-qwen38-smoke"
MODEL_VOL = "warpdecode-qwen38-models"
DEFAULT_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=False)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "ninja-build",
        "build-essential",
        "ca-certificates",
        "pkg-config",
    )
    .pip_install("cmake>=3.31,<4")
    .add_local_dir(
        ".",
        "/workspace/warodecode",
        copy=True,
        ignore=[
            "build*",
            ".git",
            "__pycache__",
            "*.pyc",
            "bench_flash_next*.py",
            "probe_flash_next.py",
            "warodecode_qwen38_smoke*.py",
        ],
    )
    .run_commands(
        "echo WARODECODE_QWEN38_SMOKE_V6 && which cmake && cmake --version",
        "test -f /workspace/warodecode/src/qwen38_runtime.cuh || "
        "(echo 'ERROR: local warodecode source is stale: qwen38_runtime.cuh missing' >&2; exit 20)",
        "grep -q -- '--load-only' /workspace/warodecode/src/warodecode.cu || "
        "(echo 'ERROR: local warodecode source is stale: --load-only path missing' >&2; exit 21)",
        "test -f /workspace/warodecode/src/iq3s_grid.cuh || "
        "(echo 'ERROR: local warodecode source is stale: IQ3_S support missing' >&2; exit 22)",
        "test -f /workspace/warodecode/src/qwen38_forward.cuh || "
        "(echo 'ERROR: local warodecode source is stale: forward path missing' >&2; exit 23)",
        "grep -q 'Q6_K PASS' /workspace/warodecode/tests/lowbit_cuda_test.cu || "
        "(echo 'ERROR: local warodecode source is stale: Q6_K self-test missing' >&2; exit 24)",
        "sed -i 's/-CUDART_INF_F/-INFINITY/g' /workspace/warodecode/src/warodecode.cu",
        "cmake -S /workspace/warodecode -B /workspace/warodecode/build-sm80 "
        "-G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DWARODECODE_BUILD_ENGINE=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /workspace/warodecode/build-sm80 -j8",
    )
)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    return proc


def find_target(root: pathlib.Path) -> pathlib.Path:
    files = sorted(root.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    files = [p for p in files if not p.name.lower().startswith("mtp-")]
    if not files:
        raise RuntimeError(f"no GGUF found under {root}")
    return files[0]


@app.function(
    image=image,
    volumes={"/models": models.with_mount_options(read_only=True)},
    cpu=4,
    memory=8192,
    timeout=1200,
)
def smoke(repo_id: str = DEFAULT_REPO) -> dict:
    root = pathlib.Path("/models") / repo_id.replace("/", "__")
    target = find_target(root)

    lowbit_test = run(
        [
            "/workspace/warodecode/build-sm80/warodecode-lowbit-test",
        ]
    ).stdout.strip()

    gpu_info = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ]
    ).stdout.strip()

    tui = run(
        [
            "/workspace/warodecode/build-sm80/warodecode-tui",
            "--dump-state",
            "--model-dir",
            str(root),
        ]
    ).stdout

    load = run(
        [
            "/workspace/warodecode/build-sm80/warodecode",
            "--model",
            str(target),
            "--ctx",
            "32768",
            "--device",
            "0",
            "--load-only",
        ]
    ).stdout

    forward1 = run(
        [
            "/workspace/warodecode/build-sm80/warodecode",
            "--model",
            str(target),
            "--ctx",
            "32768",
            "--device",
            "0",
            "--start-token",
            "1",
            "--generate",
            "1",
            "--no-graph",
        ]
    ).stdout

    result = {
        "repo": repo_id,
        "target": str(target),
        "gpu": gpu_info,
        "lowbit_test": lowbit_test,
        "tui": tui,
        "load_only": load,
        "forward_1_token": forward1,
    }
    print("RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main(
    repo: str = DEFAULT_REPO,
    gpu: str = "A100-80GB",
):
    result = smoke.with_options(gpu=gpu).remote(repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
