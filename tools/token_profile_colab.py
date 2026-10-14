#!/usr/bin/env python3
"""Task F: run the per-stage token profiler on one Colab session and collect it.

colab_run_cmd.py owns /content/run_bg.pid and kills whatever it finds there, so
it cannot be used while another task owns the VM. This runner never kills
anything: it writes its own log and pid (/content/profile_<label>.{log,pid}),
refuses to start when any other job on the session is still alive, and refuses
to start when a compute process already sits on the GPU.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-l4b python3 tools/token_profile_colab.py \
      --label l4_cache8000 --switches WARODECODE_EXPERT_CACHE_MIB=8000
  WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
      python3 tools/token_profile_colab.py --label a100_cache20000 \
      --switches WARODECODE_EXPERT_CACHE_MIB=20000

The engine always runs with WARODECODE_PROFILE=1; --no-profile flips it back to
the off case so the switch overhead itself can be measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import zipfile

import colab_run_bg as runner

STARTER = '''import pathlib, signal, subprocess, sys, zipfile, os

zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")

def alive(pid):
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    # A finished wrapper can linger as a zombie because its starter already
    # exited; /proc/<pid>/stat state 'Z' means it holds no GPU and no work.
    try:
        stat = pathlib.Path("/proc/%d/stat" % pid).read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        return state != "Z"
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
script = pathlib.Path("/content/profile_%s.sh" % label)
script.write_text("""#!/bin/bash
cd /content/warodecode
if [ ! -d build-sm__ARCH__ ]; then
  cmake -S . -B build-sm__ARCH__ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=__ARCH__ > /content/build___LABEL__.log 2>&1
fi
cmake --build build-sm__ARCH__ -j4 --target warodecode >> /content/build___LABEL__.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/build___LABEL__.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED (stale binary would be used): see /content/build___LABEL__.log" > /content/profile___LABEL__.log
  echo RUN_EXIT=99 >> /content/profile___LABEL__.log
  exit 1
fi
__ENGINE_CMD__ > /content/profile___LABEL__.log 2>&1
echo RUN_EXIT=$? >> /content/profile___LABEL__.log
pkill -f sampler___LABEL__.sh 2>/dev/null
# Drop the pid file here: the next case of the same series must not read this
# run's still-dying wrapper as "session busy".
rm -f /content/profile___LABEL__.pid
""")
script.chmod(0o755)
# Sample the GPU during the run: the token time is meaningless if another task
# started sharing the device halfway through, and the "busy" gate only checks
# the state at start time.
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
for stale in ("/content/profile___LABEL__.log", "/content/build___LABEL__.log"):
    pathlib.Path(stale).unlink(missing_ok=True)
subprocess.Popen(["bash", str(sampler)], start_new_session=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/profile___LABEL__.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--switches", default="",
                        help="engine の前に置く KEY=VALUE (カンマ区切り)")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--forced-prefix", type=int, default=0)
    parser.add_argument("--no-profile", action="store_true",
                        help="WARODECODE_PROFILE を付けない (OFF の比較用)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--wait-for-free", type=int, default=0,
                        help="セッションが使用中なら空くまで待つ秒数 (0 で即時失敗)")
    parser.add_argument("--retry", type=int, default=45,
                        help="--wait-for-free の再試行間隔 (秒)")
    args = parser.parse_args()

    tokens = json.loads((runner.ROOT / "issue1_artifacts/reference_tokens_64.json").read_text())["token_ids"]
    engine_cmd = [runner.ENGINE, "--model", runner.MODEL, "--ctx", str(args.ctx),
                  "--device", "0", "--start-token", "1", "--no-graph", "--no-autotune",
                  "--generate", str(args.generate)]
    if args.forced_prefix:
        ids = [1] + tokens[:args.generate - 1]
        engine_cmd += ["--forced-input-tokens", ",".join(str(t) for t in ids)]
    engine_cmd += [a for a in args.extra_args.split(" ") if a]

    switches = [kv for kv in args.switches.split(",") if "=" in kv]
    if not args.no_profile:
        switches.insert(0, "WARODECODE_PROFILE=1")
    cmd_line = (" ".join(switches) + " " if switches else "") + " ".join(engine_cmd)

    starter = (STARTER.replace("__ENGINE_CMD__", cmd_line)
                      .replace("__ARCH__", runner.ARCH)
                      .replace("__LABEL__", args.label))

    with tempfile.TemporaryDirectory(prefix="warodecode-profile-") as tmp:
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
            # The Colab exec websocket drops intermittently ("Connection was
            # lost."): treat that like a busy session and retry inside the wait
            # window instead of failing the whole measurement.
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

        remote_log = "/content/profile_%s.log" % args.label
        local_log = tmp_path / ("profile_%s.log" % args.label)
        deadline = time.monotonic() + args.timeout
        text = ""
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            local_log.unlink(missing_ok=True)
            subprocess.run(runner.CLI + ["download", "-s", runner.SESSION, remote_log, str(local_log)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            text = local_log.read_text(errors="replace") if local_log.exists() else ""
            print("[poll] bytes=%d %s" % (len(text),
                  "DONE" if "RUN_EXIT=" in text else "running"), flush=True)
            if "RUN_EXIT=" in text:
                break

        log_path = runner.ROOT / ("issue1_artifacts/colab_log_profile_%s.txt" % args.label)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
        print("+ saved log -> %s" % log_path.relative_to(runner.ROOT), flush=True)

        records = [l for l in text.splitlines() if l.startswith('{"engine":"warodecode-qwen38"')]
        if not records:
            print(text[-3000:])
            raise SystemExit("engine JSON がログにありません (RUN_EXIT=%s)" %
                             ("0" if "RUN_EXIT=0" in text else "非0"))
        record = json.loads(records[-1])
        # A run started by another task while this one was executing would
        # inflate every interval, so record what the GPU looked like right after
        # the run instead of assuming the session stayed ours.
        # GPU app count sampled every 5 s during the run: my own engine counts as
        # 1, so anything above 1 means the session was shared.
        sampler_stats = None
        try:
            local_mon = tmp_path / ("monitor_%s.log" % args.label)
            local_mon.unlink(missing_ok=True)
            subprocess.run(runner.CLI + ["download", "-s", runner.SESSION,
                                         "/content/monitor_%s.log" % args.label, str(local_mon)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if local_mon.exists():
                values = [int(v) for v in local_mon.read_text(errors="replace").split() if v.strip().isdigit()]
                if values:
                    sampler_stats = {
                        "samples": len(values),
                        "max_gpu_apps": max(values),
                        "samples_over_one": sum(1 for v in values if v > 1),
                        "mean_gpu_apps": round(sum(values) / len(values), 3),
                    }
        except Exception as exc:  # noqa: BLE001
            sampler_stats = {"error": str(exc)[:200]}
        contamination = None
        try:
            check = runner.run(runner.CLI + ["exec", "-s", runner.SESSION, "--timeout", "120",
                                             "-f", "/tmp/wd_contam.py"], check=False, quiet=True)
            line = next((l.strip() for l in check.splitlines() if l.startswith("CONTAM ")), "")
            if line:
                fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
                contamination = {
                    "gpu_apps": int(fields.get("apps", -1)),
                    "alive_pid_files": int(fields.get("pids", -1)),
                    "gpu_mem_used": fields.get("mem", "?"),
                }
                contamination["clean"] = contamination["gpu_apps"] == 0
        except Exception as exc:  # noqa: BLE001 - the check must never fail the run
            contamination = {"error": str(exc)[:200]}
        got = record.get("token_ids", [])
        mismatch = next((i for i, (a, b) in enumerate(zip(tokens, got)) if a != b), None)
        result = {
            "label": args.label,
            "session": runner.SESSION,
            "arch": runner.ARCH,
            "switches": args.switches,
            "profile_enabled": not args.no_profile,
            "generate": args.generate,
            "matched_prefix": mismatch if mismatch is not None else len(got),
            "first_mismatch": mismatch,
            "matched_prefix_of": min(len(tokens), len(got)),
            "tok_s": record.get("tok_s"),
            "seconds": record.get("seconds"),
            "profile": record.get("profile"),
            "profile_detail": record.get("profile_detail"),
            "contamination_check": contamination,
            "gpu_sampler": sampler_stats,
            "token_ids": got,
        }
        out_path = runner.ROOT / ("issue1_artifacts/token_profile_%s.json" % args.label)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print("+ saved profile -> %s" % out_path.relative_to(runner.ROOT), flush=True)
        print(json.dumps({k: result[k] for k in
                          ("label", "tok_s", "matched_prefix", "matched_prefix_of")}, indent=2))
        if result["profile"]:
            print(json.dumps(result["profile"], indent=2)[:4000])


if __name__ == "__main__":
    main()
