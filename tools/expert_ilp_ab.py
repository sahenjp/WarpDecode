#!/usr/bin/env python3
"""Task W: run tests/expert_ilp_ab.sh detached on the GPU session and collect it.

The engine binary already in /content/wd_taskl/warodecode/build-sm80 contains the
ILP kernels (default OFF), so the A/B only flips WARODECODE_EXPERT_GEMV_ILP.

Usage:
  export PATH=/home/sahen/.local/bin:$PATH
  python3 tools/expert_ilp_ab.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-a100")
WORK = "/content/wd_taskl"
SCRIPT = ROOT / "tests/expert_ilp_ab.sh"
CASE = ROOT / "tests/ceiling_case.py"
SRC = WORK + "/warodecode"
RESULTS = ROOT / "issue1_artifacts/expert_ilp_ab.jsonl"

BUILDER = """import subprocess, sys
cmd = ["cmake", "--build", "%(src)s/build-sm80", "-j", "8", "--target", "warodecode"]
print("+ " + " ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd="%(src)s", text=True, capture_output=True)
print(r.stdout[-3000:])
print(r.stderr[-3000:])
print("BUILD_OK" if r.returncode == 0 else "BUILD_FAIL")
""" % {"src": SRC}

STARTER = """import pathlib, subprocess
dst = pathlib.Path("/content/wd_taskl/expert_ilp_ab.sh")
dst.write_text(pathlib.Path("/content/wd_taskl_expert_ilp_ab.sh").read_text())
dst.chmod(0o755)
pathlib.Path("/content/expert_ilp_ab_status.txt").unlink(missing_ok=True)
pathlib.Path("/content/expert_ilp_ab.jsonl").unlink(missing_ok=True)
p = subprocess.Popen(["bash", str(dst)], start_new_session=True)
pathlib.Path("/content/expert_ilp_ab.pid").write_text(str(p.pid))
print("STARTED", p.pid)
"""


def run(cmd, quiet=False):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet:
        print("\n".join(proc.stdout.splitlines()[-3:]), flush=True)
    if proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))
    return proc.stdout


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="expert-ilp-ab-") as tmp:
        tmp_path = pathlib.Path(tmp)
        starter = tmp_path / "start.py"
        starter.write_text(STARTER)
        builder = tmp_path / "build.py"
        builder.write_text(BUILDER)
        run(CLI + ["upload", "-s", SESSION, str(SCRIPT), WORK + "_expert_ilp_ab.sh"], quiet=True)
        run(CLI + ["upload", "-s", SESSION, str(CASE), WORK + "/ceiling_case.py"], quiet=True)
        run(CLI + ["upload", "-s", SESSION, str(ROOT / "src/lowbit.cuh"), SRC + "/src/lowbit.cuh"],
            quiet=True)
        build = run(CLI + ["exec", "-s", SESSION, "--timeout", "900", "-f", str(builder)])
        if "BUILD_OK" not in build:
            print(build[-3000:])
            raise RuntimeError("the engine rebuild failed")
        if "STARTED" not in run(CLI + ["exec", "-s", SESSION, "--timeout", "300", "-f", str(starter)]):
            raise RuntimeError("the A/B did not start")
        status = tmp_path / "status.txt"
        for _ in range(80):
            subprocess.run(CLI + ["download", "-s", SESSION, "/content/expert_ilp_ab_status.txt", str(status)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            text = status.read_text(errors="replace") if status.exists() else ""
            print("[ilp-ab] %s" % (text.splitlines()[-1:] or [""])[0][:120], flush=True)
            if "DONE" in text:
                break
            time.sleep(30)
        results = tmp_path / "results.jsonl"
        subprocess.run(CLI + ["download", "-s", SESSION, "/content/expert_ilp_ab.jsonl", str(results)],
                       text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not results.exists():
            print("no results yet")
            return
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(results.read_text())
        rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
        print("")
        print("%-10s %8s %9s %6s %s" % ("case", "tok/s", "decode_s", "match", "ilp"))
        for row in rows:
            print("%-10s %8.3f %9.3f %4d/64 %s" % (
                row["case"], row["tok_s"] or 0.0, row["decode_seconds"] or 0.0,
                row["matched_prefix"], row["extra"]))


if __name__ == "__main__":
    main()
