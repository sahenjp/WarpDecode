#!/usr/bin/env python3
"""Task R: run the whole expert-transfer matrix as ONE Colab background job.

The Colab runner (tools/colab_run_bg.py) starts one job, and only one job may
exist per session (a new one kills the old), so the matrix runs *inside* the VM
(tests/moe_transfer_sweep.sh) and this script only uploads, builds, starts and
polls. Every case is one engine process with the canonical greedy command, so
each case checks its own 64/64 against the reference token list.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-a100 python3 tests/moe_transfer_sweep.py \
      --label r1 --arch 80
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]

STARTER = '''import pathlib, subprocess, zipfile
zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")
script = pathlib.Path("/content/start_sweep.sh")
script.write_text("""#!/bin/bash
cd /content/warodecode
if [ ! -d build-sm__ARCH__ ]; then
  cmake -S . -B build-sm__ARCH__ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=__ARCH__ > /content/build_bg.log 2>&1
fi
cmake --build build-sm__ARCH__ -j8 --target warodecode >> /content/build_bg.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/build_bg.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED" > /content/sweep_status.txt
  echo "BUILD_FAILED: see /content/build_bg.log" >> /content/sweep.log
  exit 1
fi
bash /content/warodecode/__MATRIX__
""")
script.chmod(0o755)
for name in ("/content/sweep.log", "/content/sweep_status.txt", "/content/sweep_probe.txt",
             "/content/sweep_err.txt", "/content/build_bg.log"):
    pathlib.Path(name).unlink(missing_ok=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/run_bg.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''


def run(cmd, check=True):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout[-1500:], flush=True)
    if check and proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(cmd)))
    return proc.stdout


def download(cli, session, remote, local):
    pathlib.Path(local).unlink(missing_ok=True)
    subprocess.run(cli + ["download", "-s", session, remote, str(local)],
                   text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arch", default="80")
    ap.add_argument("--script", default="tests/moe_transfer_sweep.sh",
                    help="VM-side matrix script inside the uploaded tree")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--poll", type=int, default=45)
    args = ap.parse_args()
    session = "warpdecode-a100"
    cli = CLI

    ref = ROOT / "issue1_artifacts/reference_tokens_64.json"
    with tempfile.TemporaryDirectory(prefix="warodecode-sweep-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "src.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (ROOT / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts and "__pycache__" not in f.parts:
                        z.write(f, f.relative_to(ROOT))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = ROOT / name
                if f.is_file():
                    z.write(f, name)
        print("+ upload source (%d bytes)" % archive.stat().st_size, flush=True)
        run(cli + ["upload", "-s", session, str(archive), "/content/warodecode-source.zip"])
        run(cli + ["upload", "-s", session, str(ref), "/content/reference_tokens_64.json"])

        starter = tmp_path / "start.py"
        # __MATRIX__ is substituted here, not by the VM shell: the starter is
        # Python, so a $MATRIX left for bash would expand to nothing and
        # "bash /content/warodecode/" would fail without starting the matrix.
        starter.write_text(STARTER.replace("__ARCH__", args.arch)
                                  .replace("__MATRIX__", args.script))
        print("+ start VM background sweep", flush=True)
        try:
            run(cli + ["exec", "-s", session, "--timeout", "600", "-f", str(starter)], check=False)
        except Exception as exc:  # the exec websocket wedges; the job may still be up
            print("! exec returned an error (%s); polling anyway" % exc, flush=True)

        log_local = tmp_path / "sweep.log"
        status_local = tmp_path / "status.txt"
        build_local = tmp_path / "build.log"
        probe_local = tmp_path / "probe.txt"
        seen = 0
        idle = 0
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            download(cli, session, "/content/sweep.log", log_local)
            download(cli, session, "/content/sweep_status.txt", status_local)
            download(cli, session, "/content/build_bg.log", build_local)
            download(cli, session, "/content/sweep_probe.txt", probe_local)
            text = log_local.read_text(errors="replace") if log_local.exists() else ""
            lines = [l for l in text.splitlines() if l.startswith("{")]
            build_tail = build_local.read_text(errors="replace").splitlines()[-1:] if build_local.exists() else []
            stxt = status_local.read_text(errors="replace") if status_local.exists() else ""
            if not stxt and not build_tail:
                idle += 1
            print("[poll] build=%s cases=%d %s" % (build_tail, len(lines),
                  "DONE" if "SWEEP_DONE" in stxt else ("FAILED" if "BUILD_FAILED" in stxt else "running")),
                  flush=True)
            for line in lines[seen:]:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                p = r.get("profile") or {}
                s = r.get("stats") or {}
                print("  %-16s tok_s=%-7s parity=%s/64 moe=%s stage=%s pread=%s hits=%s/%s dma=%s stall=%s"
                      % (r["case"], r["tok_s"], r["matched_prefix"], p.get("moe_ms"),
                         s.get("stage_ms"), s.get("io_pread_ms"), s.get("hits"), s.get("accesses"),
                         s.get("io_dma_ms"), s.get("io_dma_stall_ms")), flush=True)
                if r.get("stderr_tail"):
                    print("    stderr: %s" % r["stderr_tail"][-400:].replace("\n", " | "), flush=True)
            seen = len(lines)
            if "SWEEP_DONE" in stxt or "BUILD_FAILED" in stxt:
                out = ROOT / ("issue1_artifacts/moe_sweep_%s.jsonl" % args.label)
                out.write_text(text)
                probe = ROOT / ("issue1_artifacts/moe_sweep_%s_probe.txt" % args.label)
                if probe_local.exists():
                    probe.write_text(probe_local.read_text(errors="replace"))
                print("+ wrote %s" % out, flush=True)
                if "BUILD_FAILED" in stxt:
                    raise RuntimeError("VM build failed; see /content/build_bg.log")
                return
        raise RuntimeError("sweep timed out")


if __name__ == "__main__":
    main()
