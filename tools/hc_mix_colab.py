#!/usr/bin/env python3
"""Task J: run the HC mix micro-benchmark and the engine A/B plan on Colab.

Unlike colab_run_bg.py this runner does not kill anyone: it writes its own log
and pid (/content/taskj_<label>.{log,pid}), refuses to start when another job's
pid file is alive or a compute process already sits on the GPU, and never
reuses /content/run_bg.log.

Usage:
  export PATH=/home/sahen/.local/bin:$PATH
  WARODECODE_COLAB_SESSION=warpdecode-l4 python3 tools/hc_mix_colab.py \
      --label taskJ_micro --mode bench --iterations 300
  WARODECODE_COLAB_SESSION=warpdecode-l4 python3 tools/hc_mix_colab.py \
      --label taskJ_ab --mode ab --plan l4 --repeat 2
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import colab_run_bg as runner  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

BENCH_SCRIPT = """#!/bin/bash
cd /content/warodecode
mkdir -p /content/taskj_build
if [ ! -x /content/taskj_build/hc_bench ] || [ "__REBUILD__" = "1" ]; then
nvcc -arch=sm___ARCH__ -O3 -I src -o /content/taskj_build/hc_bench tests/qwen38_hc_bench.cu > /content/taskj_build/bench_build.log 2>&1
fi
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/taskj_build/bench_build.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BENCH_BUILD_FAILED: see /content/taskj_build/bench_build.log" > /content/taskj___LABEL__.txt
  echo BENCH_EXIT=99 >> /content/taskj___LABEL__.txt
  exit 1
fi
/content/taskj_build/hc_bench __BENCH_ARGS__ > /content/taskj___LABEL__.json 2> /content/taskj___LABEL__.txt
echo BENCH_EXIT=$? >> /content/taskj___LABEL__.txt
"""

AB_SCRIPT = """#!/bin/bash
cd /content/warodecode
if [ ! -d build-sm__ARCH__ ]; then
  cmake -S . -B build-sm__ARCH__ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=__ARCH__ > /content/taskj_build___LABEL__.log 2>&1
fi
cmake --build build-sm__ARCH__ -j4 --target warodecode >> /content/taskj_build___LABEL__.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/taskj_build___LABEL__.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED (stale binary would be used): see /content/taskj_build___LABEL__.log" > /content/taskj___LABEL__.log
  echo RUN_EXIT=99 >> /content/taskj___LABEL__.log
  exit 1
fi
cp /content/reference_tokens_64.json /content/reference_tokens_64.json.bak 2>/dev/null
WD_ENGINE=/content/warodecode/build-sm__ARCH__/warodecode \
  python3 tests/hc_mix_ab.py --plan __PLAN__ --repeat __REPEAT__ --generate __GENERATE__ \
    --out /content/taskj_ab___LABEL__.json > /content/taskj___LABEL__.log 2>&1
echo RUN_EXIT=$? >> /content/taskj___LABEL__.log
echo AB_LOG_READY >> /content/taskj___LABEL__.log
"""

STARTER = '''import pathlib, signal, subprocess, sys, zipfile, os

zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")

def job_state(pid):
    try:
        os.kill(pid, 0)
    except Exception:
        return None
    # A zombie still answers signal 0 but owns no GPU work: run_bg.pid is left
    # behind by tasks whose parent shell already reaped the process.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    if not state or state.startswith("Z"):
        return None
    return state

jobs = []
for p in sorted(pathlib.Path("/content").glob("*.pid")):
    try:
        pid = int(p.read_text().strip())
    except Exception:
        continue
    state = job_state(pid)
    if state:
        jobs.append("%s:%d:%s" % (p.name, pid, state))

gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader"], text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
if jobs or gpu:
    print("BUSY jobs=%s gpu=%r" % (jobs, gpu))
    sys.exit(3)

label = "__LABEL__"
script = pathlib.Path("/content/taskj_%s.sh" % label)
script.write_text("""__BODY__""")
script.chmod(0o755)
for stale in ("/content/taskj___LABEL__.log", "/content/taskj___LABEL__.txt",
              "/content/taskj___LABEL__.json"):
    pathlib.Path(stale).unlink(missing_ok=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/taskj___LABEL__.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''


def run(cmd, check=True, quiet=False):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet:
        print(proc.stdout[-2000:], flush=True)
    if check and proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(cmd)))
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("bench", "ab"), default="bench")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=10240)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--low-rank", type=int, default=320)
    parser.add_argument("--plan", default="l4")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--wait-for-free", type=int, default=0)
    parser.add_argument("--retry", type=int, default=45)
    parser.add_argument("--bench-args", default="",
                        help="ベンチ本体へ追加で渡す引数 (例: --skip-legacy)")
    parser.add_argument("--reuse-bench", action="store_true",
                        help="走 VM 上の既存ベンチバイナリをそのまま使う (再ビルドしない)")
    args = parser.parse_args()

    if args.mode == "bench":
        bench_args = "--hidden %d --hc %d --low-rank %d --iterations %d --json /content/taskj_%s.json %s" % (
            args.hidden, args.hc, args.low_rank, args.iterations, args.label, args.bench_args)
        body = (BENCH_SCRIPT.replace("__BENCH_ARGS__", bench_args)
                         .replace("__LABEL__", args.label)
                         .replace("__ARCH__", runner.ARCH)
                         .replace("__REBUILD__", "0" if args.reuse_bench else "1"))
    else:
        body = (AB_SCRIPT.replace("__PLAN__", args.plan)
                        .replace("__REPEAT__", str(args.repeat))
                        .replace("__GENERATE__", str(args.generate))
                        .replace("__LABEL__", args.label))
    starter = STARTER.replace("__BODY__", body).replace("__LABEL__", args.label)

    with tempfile.TemporaryDirectory(prefix="warodecode-taskj-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "src.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (runner.SOURCE / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts:
                        z.write(f, f.relative_to(runner.SOURCE))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = runner.SOURCE / name
                if f.is_file():
                    z.write(f, name)
        print("+ upload source (%d bytes)" % archive.stat().st_size, flush=True)
        run(runner.CLI + ["upload", "-s", runner.SESSION, str(archive),
                          "/content/warodecode-source.zip"])

        starter_path = tmp_path / "start.py"
        starter_path.write_text(starter)
        wait_until = time.monotonic() + args.wait_for_free
        while True:
            out = run(runner.CLI + ["exec", "-s", runner.SESSION, "--timeout", "600",
                                    "-f", str(starter_path)], check=False)
            if "BUSY" in out:
                if time.monotonic() >= wait_until:
                    raise SystemExit("セッションが使用中です: " + out.strip()[-300:])
                print("[wait] session busy, retry in %ds" % args.retry, flush=True)
                time.sleep(args.retry)
                continue
            if "STARTED" not in out:
                raise SystemExit("VM 上で起動できませんでした: " + out.strip()[-600:])
            break

        remote_txt = "/content/taskj_%s.txt" % args.label
        remote_json = "/content/taskj_%s.json" % args.label
        remote_log = "/content/taskj_%s.log" % args.label
        local_txt = tmp_path / "bench.txt"
        local_json = tmp_path / "bench.json"
        local_log = tmp_path / "ab.log"
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            for remote, local in ((remote_txt, local_txt), (remote_json, local_json),
                                  (remote_log, local_log)):
                local.unlink(missing_ok=True)
                subprocess.run(runner.CLI + ["download", "-s", runner.SESSION, remote, str(local)],
                               text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            done = local_txt.exists() and "BENCH_EXIT=" in local_txt.read_text(errors="replace")
            done = done or (local_log.exists() and "AB_LOG_READY" in local_log.read_text(errors="replace"))
            print("[poll] %s" % ("DONE" if done else "running"), flush=True)
            if done:
                break

        out_dir = ROOT / "issue1_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "bench":
            if local_txt.exists():
                (out_dir / ("taskj_bench_%s.txt" % args.label)).write_text(
                    local_txt.read_text(errors="replace"))
                print(local_txt.read_text(errors="replace"))
            if local_json.exists():
                (out_dir / ("taskj_bench_%s.json" % args.label)).write_text(
                    local_json.read_text(errors="replace"))
            else:
                print("no bench json yet")
        else:
            if local_log.exists():
                (out_dir / ("taskj_ab_%s.log" % args.label)).write_text(
                    local_log.read_text(errors="replace"))
                print(local_log.read_text(errors="replace")[-4000:])
            if local_json.exists():
                (out_dir / ("taskj_ab_%s.json" % args.label)).write_text(
                    local_json.read_text(errors="replace"))
                print(local_json.read_text(errors="replace")[:4000])


if __name__ == "__main__":
    main()
