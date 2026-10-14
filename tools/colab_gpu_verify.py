#!/usr/bin/env python3
"""Build warodecode on a Colab GPU session and run the synthetic CUDA tests.

Every test executed here uses synthetic weights, so no model download is
required. The point is to verify that the uncommitted tree compiles with a
real nvcc and that the kernels produce the expected numbers on real silicon.

Usage:
    python3 tools/colab_gpu_verify.py --session warpdecode-t4
    python3 tools/colab_gpu_verify.py --new --gpu T4      # provision + verify
    WARODECODE_COLAB_ARCH=86 python3 tools/colab_gpu_verify.py --session warpdecode-sm86

The source archive is uploaded to /content/warodecode-source.zip and the
worker prints one COLAB_ROW= JSON line per test so partial progress survives.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SESSION = "warpdecode-t4"
DEFAULT_ARCH = "75"

# CMake targets that build a self-contained synthetic CUDA test binary.
CMAKE_CUDA_TESTS = [
    "warodecode-lowbit-test",
    "warodecode-qwen38-moe-cuda-test",
    "warodecode-qwen38-qsa-cuda-test",
    "warodecode-qwen38-gdn-cuda-test",
    "warodecode-qwen38-hc-cuda-test",
    "warodecode-issue1-moe-precision-cuda-test",
]
# Tests that exist in tests/ but are not registered in CMakeLists.txt would be
# compiled here directly; the list is empty now that both are CMake targets.
ORPHAN_CUDA_TESTS: list[str] = []

REMOTE = r'''import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

ARCH = __ARCH__
EXPECT_SM = __EXPECT_SM__
ROOT = pathlib.Path("/content/warodecode")
CMAKE_TESTS = __CMAKE_TESTS__
ORPHAN_TESTS = __ORPHAN_TESTS__

with zipfile.ZipFile("/content/warodecode-source.zip") as archive:
    archive.extractall(ROOT)


def run(cmd, cwd=None, check=False):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))
    return proc


def emit(record):
    print("COLAB_ROW=" + json.dumps(record, allow_nan=False), flush=True)


gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], text=True,
                     stdout=subprocess.PIPE, check=True).stdout.strip()
nvcc = subprocess.run(["nvcc", "--version"], text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
print("GPU=" + gpu, flush=True)
print("CUDA=" + nvcc.splitlines()[-1].strip(), flush=True)

build = ROOT / ("build-sm" + ARCH)
configure = run(["cmake", "-S", str(ROOT), "-B", str(build),
                 "-DCMAKE_BUILD_TYPE=Release", "-DWARODECODE_BUILD_ENGINE=ON",
                 "-DCMAKE_CUDA_ARCHITECTURES=" + ARCH])
if configure.returncode:
    emit({"stage": "configure", "ok": False, "log_tail": configure.stdout.splitlines()[-40:]})
    sys.exit(1)

compile_result = run(["cmake", "--build", str(build), "-j4"])
emit({"stage": "build", "ok": compile_result.returncode == 0,
      "log_tail": compile_result.stdout.splitlines()[-30:]})

results = []
for target in CMAKE_TESTS:
    exe = build / target
    if not exe.is_file():
        results.append({"test": target, "status": "MISSING"})
        continue
    proc = run([exe])
    lines = proc.stdout.splitlines()
    results.append({
        "test": target,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "exit": proc.returncode,
        "pass_lines": [line for line in lines if " PASS " in line],
        "fail_lines": [line for line in lines if "FAIL" in line or "error" in line.lower()][:5],
    })

objects = build / "CMakeFiles"
for name in ORPHAN_TESTS:
    source = ROOT / "tests" / name
    if not source.is_file():
        results.append({"test": name, "status": "MISSING"})
        continue
    exe = pathlib.Path("/tmp") / (name.rsplit(".", 1)[0] + ".bin")
    compile_proc = run(["nvcc", "-O3", "-std=c++17", "-I", str(ROOT / "src"),
                        "-arch=sm_" + ARCH, str(source), "-o", str(exe)])
    if compile_proc.returncode:
        results.append({"test": name, "status": "COMPILE_FAIL"})
        continue
    proc = run([exe])
    results.append({"test": name, "status": "PASS" if proc.returncode == 0 else "FAIL",
                    "exit": proc.returncode, "log_tail": proc.stdout.splitlines()[-10:]})

engine = build / "warodecode"
emit({
    "stage": "tests",
    "gpu": gpu,
    "arch": ARCH,
    "expect_sm": EXPECT_SM,
    "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest() if engine.is_file() else None,
    "engine_built": engine.is_file(),
    "results": results,
    "free_disk": shutil.disk_usage("/").free,
})
'''


def cli() -> list[str]:
    return ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(cmd)))
    return proc.stdout


def make_source_archive(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("src", "tests", "tools", "bench"):
            for file in (ROOT / name).rglob("*"):
                if file.is_file() and "qwen38-fixture" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
        for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--gpu", default=None, help="provision a new session with this GPU")
    parser.add_argument("--new", action="store_true", help="create the session before verifying")
    args = parser.parse_args()

    base = cli()
    if args.new:
        cmd = base + ["new", "-s", args.session]
        if args.gpu:
            cmd += ["--gpu", args.gpu]
        run(cmd)

    worker_text = (REMOTE
                   .replace("__ARCH__", json.dumps(str(args.arch)))
                   .replace("__EXPECT_SM__", json.dumps("sm_" + str(args.arch)))
                   .replace("__CMAKE_TESTS__", json.dumps(CMAKE_CUDA_TESTS))
                   .replace("__ORPHAN_TESTS__", json.dumps(ORPHAN_CUDA_TESTS)))

    with tempfile.TemporaryDirectory(prefix="warodecode-colab-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "warodecode-source.zip"
        worker = tmp_path / "colab_gpu_verify_remote.py"
        make_source_archive(archive)
        worker.write_text(worker_text)
        run(base + ["upload", "-s", args.session, str(archive), "/content/warodecode-source.zip"])
        run(base + ["exec", "-s", args.session, "--timeout", "3600", "-f", str(worker)])


if __name__ == "__main__":
    main()
