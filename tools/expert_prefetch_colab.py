#!/usr/bin/env python3
"""Task K: run the expert read-ahead A/B on one Colab session and collect it.

Like tools/token_profile_colab.py this never kills another task's job: it writes
its own log/pid, refuses to start while any other /content/*.pid is alive or a
compute process already sits on the GPU, and samples the GPU during the run so a
shared session is visible in the result.

Usage:
    export PATH=/home/sahen/.local/bin:$PATH
    WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
        python3 tools/expert_prefetch_colab.py --label a100 --plan a100 --wait-for-free 2700
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
import time
import zipfile

import colab_run_bg as runner

STARTER = '''import json, os, pathlib, shutil, subprocess, sys, zipfile

# The archive is a frozen revision, so a stale file from an earlier upload must
# not survive the extraction and shadow it.
for sub in ("src", "tests", "tools", "bench"):
    shutil.rmtree("/content/warodecode/" + sub, ignore_errors=True)
zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")
pathlib.Path("/content/reference_tokens_64.json").write_text(
    json.dumps({"token_ids": __TOKENS__}))

def alive(pid):
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    try:
        stat = pathlib.Path("/proc/%d/stat" % pid).read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        return True

def busy_jobs():
    out = []
    for p in sorted(pathlib.Path("/content").glob("*.pid")):
        try:
            pid = int(p.read_text().strip())
        except Exception:
            continue
        if alive(pid):
            out.append("%s:%d" % (p, pid))
    return out

jobs = busy_jobs()
gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader"], text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
if jobs or gpu:
    print("BUSY jobs=%s gpu=%r" % (jobs, gpu))
    sys.exit(3)

label = "__LABEL__"
script = pathlib.Path("/content/prefetch_%s.sh" % label)
script.write_text("""#!/bin/bash
cd /content/warodecode
if [ ! -d build-sm__ARCH__ ]; then
  cmake -S . -B build-sm__ARCH__ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=__ARCH__ > /content/build___LABEL__.log 2>&1
fi
cmake --build build-sm__ARCH__ -j4 --target warodecode >> /content/build___LABEL__.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/build___LABEL__.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED (stale binary would be used): see /content/build___LABEL__.log" > /content/prefetch___LABEL__.log
  echo "RUN_EXIT=99" >> /content/prefetch___LABEL__.log
  exit 1
fi
WD_ENGINE=/content/warodecode/build-sm__ARCH__/warodecode python3 tests/expert_prefetch_ab.py --plan __PLAN__ --engine /content/warodecode/build-sm__ARCH__/warodecode > /content/prefetch___LABEL__.log 2>&1
echo "RUN_EXIT=$?" >> /content/prefetch___LABEL__.log
pkill -f sampler___LABEL__.sh 2>/dev/null
rm -f /content/prefetch___LABEL__.pid
""")
script.chmod(0o755)
monitor = pathlib.Path("/content/monitor___LABEL__.log")
monitor.write_text("")
sampler = pathlib.Path("/content/sampler___LABEL__.sh")
sampler.write_text("""#!/bin/bash
for i in $(seq 1 400); do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . >> /content/monitor___LABEL__.log
  sleep 5
done
""")
sampler.chmod(0o755)
for stale in ("/content/prefetch___LABEL__.log", "/content/build___LABEL__.log"):
    pathlib.Path(stale).unlink(missing_ok=True)
subprocess.Popen(["bash", str(sampler)], start_new_session=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/prefetch___LABEL__.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''

REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--plan", default="a100", choices=("a100", "l4"))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--wait-for-free", type=int, default=0)
    parser.add_argument("--retry", type=int, default=45)
    args = parser.parse_args()

    raw = (runner.ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise SystemExit("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]

    starter = (STARTER.replace("__TOKENS__", json.dumps(tokens))
                      .replace("__ARCH__", runner.ARCH)
                      .replace("__PLAN__", args.plan)
                      .replace("__LABEL__", args.label))

    with tempfile.TemporaryDirectory(prefix="warodecode-prefetch-") as tmp:
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
        runner.run(runner.CLI + ["upload", "-s", runner.SESSION, str(archive),
                                 "/content/warodecode-source.zip"])

        starter_path = tmp_path / "start.py"
        starter_path.write_text(starter)
        print("+ start VM background run", flush=True)
        wait_until = time.monotonic() + args.wait_for_free
        while True:
            try:
                out = runner.run(runner.CLI + ["exec", "-s", runner.SESSION, "--timeout", "600",
                                               "-f", str(starter_path)], check=False)
            except Exception as exc:  # noqa: BLE001
                out = "CONNECTION_ERROR " + str(exc)[-200:]
                print("[wait] exec failed: %s" % str(exc)[-160:], flush=True)
            if "CONNECTION_ERROR" in out or "Connection was lost" in out:
                if time.monotonic() >= wait_until:
                    raise SystemExit("セッション接続が復旧しません: " + out[-300:])
                time.sleep(args.retry)
                continue
            if "BUSY" in out:
                if time.monotonic() >= wait_until:
                    raise SystemExit("セッションが使用中です (他のタスクが実行中): " + out.strip()[-300:])
                print("[wait] session busy, retry in %ds" % args.retry, flush=True)
                time.sleep(args.retry)
                continue
            if "STARTED" not in out:
                raise SystemExit("VM 上で起動できませんでした: " + out.strip()[-600:])
            break

        remote_log = "/content/prefetch_%s.log" % args.label
        local_log = tmp_path / ("prefetch_%s.log" % args.label)
        deadline = time.monotonic() + args.timeout
        text = ""
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            local_log.unlink(missing_ok=True)
            subprocess.run(runner.CLI + ["download", "-s", runner.SESSION, remote_log, str(local_log)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            text = local_log.read_text(errors="replace") if local_log.exists() else ""
            cases = sum(1 for l in text.splitlines() if l.startswith("WD_CASE "))
            print("[poll] bytes=%d cases=%d %s" % (
                len(text), cases, "DONE" if "WD_AB_DONE" in text else "running"), flush=True)
            if "WD_AB_DONE" in text or "RUN_EXIT=99" in text:
                break

        log_path = runner.ROOT / ("issue1_artifacts/colab_log_prefetch_%s.txt" % args.label)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
        print("+ saved log -> %s" % log_path.relative_to(runner.ROOT), flush=True)

        build_text = ""
        local_build = tmp_path / ("build_%s.log" % args.label)
        subprocess.run(runner.CLI + ["download", "-s", runner.SESSION,
                                     "/content/build_%s.log" % args.label, str(local_build)],
                       text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if local_build.exists():
            build_text = local_build.read_text(errors="replace")
        if "BUILD_EXIT=0" not in build_text:
            print(build_text[-4000:], flush=True)
            raise SystemExit("ビルドが成功していません (BUILD_EXIT != 0)")

        cases = [json.loads(l[len("WD_CASE "):]) for l in text.splitlines()
                 if l.startswith("WD_CASE ")]
        sampler_stats = None
        local_mon = tmp_path / ("monitor_%s.log" % args.label)
        subprocess.run(runner.CLI + ["download", "-s", runner.SESSION,
                                     "/content/monitor_%s.log" % args.label, str(local_mon)],
                       text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if local_mon.exists():
            values = [int(v) for v in local_mon.read_text(errors="replace").split() if v.strip().isdigit()]
            if values:
                sampler_stats = {"samples": len(values), "max_gpu_apps": max(values),
                                 "samples_over_one": sum(1 for v in values if v > 1)}

        out_path = runner.ROOT / ("issue1_artifacts/expert_prefetch_%s.json" % args.label)
        out_path.write_text(json.dumps({"label": args.label, "session": runner.SESSION,
                                        "arch": runner.ARCH, "plan": args.plan,
                                        "gpu_sampler": sampler_stats,
                                        "cases": cases}, indent=1) + "\n")
        print("+ saved results -> %s" % out_path.relative_to(runner.ROOT), flush=True)
        for rec in cases:
            print(json.dumps({k: rec.get(k) for k in
                              ("case", "tok_s", "matched_prefix", "exit", "error")}))
        print("gpu_sampler:", sampler_stats)


if __name__ == "__main__":
    main()
