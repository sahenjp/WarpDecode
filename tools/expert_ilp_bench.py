#!/usr/bin/env python3
"""Task W: run tests/qwen38_expert_ilp_bench.cu on the GPU session.

The A/B harness already leaves the engine's source tree built in /content/wd_taskl
(phase build_engine), so this only uploads the new benchmark file, compiles it
with the same nvcc flags the engine uses, and prints its output.

Usage:
  export PATH=/home/sahen/.local/bin:$PATH
  WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
    python3 tools/expert_ilp_bench.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-a100")
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "80")
WORK = "/content/wd_taskl"
SRC = WORK + "/warodecode"
BENCH = ROOT / "tests/qwen38_expert_ilp_bench.cu"
RESULT = ROOT / "issue1_artifacts/expert_ilp_bench.txt"

# google-colab-cli "exec -f" runs the uploaded file as Python (the A/B harness
# does the same), so the runner is a Python snippet that shells out.
RUNNER = """import pathlib, subprocess, sys
root = pathlib.Path(%(src)r)
bench = root / "tests/qwen38_expert_ilp_bench.cu"
cmd = ["nvcc", "-O3", "--use_fast_math", "--extra-device-vectorization", "-std=c++17",
       "-arch=sm_%(arch)s", "-Isrc", str(bench), "-o", "/content/expert_ilp_bench"]
print("+ " + " ".join(cmd), flush=True)
build = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
print(build.stdout[-4000:], flush=True)
print(build.stderr[-4000:], flush=True)
if build.returncode != 0:
    print("BENCH_EXIT=" + str(build.returncode), flush=True)
    sys.exit(0)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm",
                      "--format=csv,noheader"], text=True, capture_output=True).stdout.strip(), flush=True)
run = subprocess.run(["/content/expert_ilp_bench"], text=True, capture_output=True)
print(run.stdout, flush=True)
print(run.stderr[-2000:], flush=True)
print("BENCH_EXIT=" + str(run.returncode), flush=True)
""" % {"src": SRC, "arch": ARCH}


def run(cmd, quiet=False):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet:
        print("\n".join(proc.stdout.splitlines()[-3:]), flush=True)
    if proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))
    return proc.stdout


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="expert-ilp-") as tmp:
        tmp_path = pathlib.Path(tmp)
        runner = tmp_path / "expert_ilp_run.py"
        runner.write_text(RUNNER)
        run(CLI + ["upload", "-s", SESSION, str(ROOT / "src/lowbit.cuh"), SRC + "/src/lowbit.cuh"], quiet=True)
        run(CLI + ["upload", "-s", SESSION, str(BENCH), SRC + "/tests/qwen38_expert_ilp_bench.cu"], quiet=True)
        output = subprocess.run(CLI + ["exec", "-s", SESSION, "--timeout", "900", "-f", str(runner)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(output)
        print(output)


if __name__ == "__main__":
    main()
