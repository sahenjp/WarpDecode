#!/usr/bin/env python3
"""Upload the tree, build the engine and run it on the Colab VM, then poll.

All heavy work happens on the VM under a detached process (the Colab CLI drops
long-lived exec connections). File names are fixed so the flow is easy to debug:
  /content/run_bg.sh, /content/build_bg.log, /content/run_bg.log, /content/run_bg.pid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-l4")
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "89")
ENGINE = "/content/warodecode/build-sm%s/warodecode" % ARCH
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"

STARTER = '''import subprocess, pathlib, zipfile
zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")
script = pathlib.Path("/content/run_bg.sh")
script.write_text("""#!/bin/bash
cd /content/warodecode
if [ ! -d build-sm__ARCH__ ]; then
  cmake -S . -B build-sm__ARCH__ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=__ARCH__ > /content/build_bg.log 2>&1
fi
cmake --build build-sm__ARCH__ -j4 --target warodecode >> /content/build_bg.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> /content/build_bg.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED (stale binary would be used): see /content/build_bg.log" > /content/run_bg.log
  echo RUN_EXIT=99 >> /content/run_bg.log
  exit 1
fi
__ENGINE_CMD__ > /content/run_bg.log 2>&1
echo RUN_EXIT=$? >> /content/run_bg.log
""")
script.chmod(0o755)
for name in ("/content/run_bg.log", "/content/build_bg.log"):
    pathlib.Path(name).unlink(missing_ok=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/run_bg.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''

STATUS = '''import pathlib, os, json
pidf = pathlib.Path("/content/run_bg.pid")
pid = int(pidf.read_text()) if pidf.exists() else -1
alive = False
try:
    os.kill(pid, 0); alive = True
except Exception:
    pass
b = pathlib.Path("/content/build_bg.log")
if b.exists():
    print("BUILD:", " || ".join(b.read_text(errors="replace").splitlines()[-2:])[:200])
t = pathlib.Path("/content/run_bg.log")
tt = t.read_text(errors="replace") if t.exists() else ""
print("alive=", alive)
rec = [l for l in tt.splitlines() if l.startswith('{"engine":"warodecode-qwen38"')]
if rec:
    j = json.loads(rec[-1])
    print("TOKENS=" + ",".join(str(x) for x in j["token_ids"]))
elif "RUN_EXIT" in tt:
    print("FAILED:", " || ".join(tt.splitlines()[-4:])[:400])
else:
    print("RUNNING")
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
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--switches", default="")
    parser.add_argument("--forced-prefix", type=int, default=0)
    parser.add_argument("--trace-position", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]

    engine_cmd = [ENGINE, "--model", MODEL, "--ctx", "128", "--device", "0",
                  "--start-token", "1", "--no-graph", "--no-autotune",
                  "--generate", str(args.generate)]
    if args.trace_position >= 0:
        engine_cmd += ["--state-trace", "/content/tr_%s.bin" % args.label,
                       "--state-trace-position", str(args.trace_position)]
    if args.forced_prefix:
        # The engine requires one forced input token per generated position, and
        # the first one must equal --start-token.
        ids = [1] + tokens[:args.generate - 1]
        assert len(ids) == args.generate
        engine_cmd += ["--forced-input-tokens", ",".join(str(t) for t in ids)]
    engine_cmd += [a for a in args.extra_args.split(" ") if a]
    env_prefix = " ".join(kv for kv in args.switches.split(",") if "=" in kv)
    cmd_line = (env_prefix + " " if env_prefix else "") + " ".join(engine_cmd)

    with tempfile.TemporaryDirectory(prefix="warodecode-bg-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "src.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (ROOT / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts:
                        z.write(f, f.relative_to(ROOT))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = ROOT / name
                if f.is_file():
                    z.write(f, name)
        print("+ upload source (%d bytes)" % archive.stat().st_size, flush=True)
        run(CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"])

        starter = tmp_path / "start.py"
        starter.write_text(
            STARTER.replace("__ENGINE_CMD__", cmd_line).replace("__ARCH__", ARCH))
        print("+ start VM background run", flush=True)
        run(CLI + ["exec", "-s", SESSION, "--timeout", "600", "-f", str(starter)])


        # Poll over HTTP (colab download) instead of the Jupyter kernel: the CLI's
        # exec websocket wedges intermittently and would stall the loop forever.
        result = None
        deadline = time.monotonic() + args.timeout
        local_run = tmp_path / "run_bg.log"
        local_build = tmp_path / "build_bg.log"
        while time.monotonic() < deadline:
            time.sleep(30)
            for remote, local in (("/content/run_bg.log", local_run),
                                  ("/content/build_bg.log", local_build)):
                local.unlink(missing_ok=True)
                subprocess.run(CLI + ["download", "-s", SESSION, remote, str(local)],
                               text=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            build_tail = local_build.read_text(errors="replace").splitlines()[-1:] if local_build.exists() else []
            text = local_run.read_text(errors="replace") if local_run.exists() else ""
            rec = [l for l in text.splitlines() if l.startswith('{"engine":"warodecode-qwen38"')]
            print("[poll] build=%s run_bytes=%d %s" % (build_tail, len(text),
                  "DONE" if rec else ("FAILED" if "RUN_EXIT" in text else "running")), flush=True)
            if rec:
                got = json.loads(rec[-1])["token_ids"]
                mismatch = next((i for i, (a, b) in enumerate(zip(tokens, got)) if a != b), None)
                result = {"matched_prefix": mismatch if mismatch is not None else len(got),
                          "first_mismatch": mismatch, "tokens": got}
                break
            if "RUN_EXIT" in text and not rec:
                result = {"error": " || ".join(text.splitlines()[-4:])[:800]}
                break

        out_path = ROOT / ("issue1_artifacts/colab_run_%s.json" % args.label)
        out_path.write_text(json.dumps({
            "label": args.label, "generate": args.generate, "extra_args": args.extra_args,
            "switches": args.switches, "forced_prefix": args.forced_prefix,
            "result": result}, indent=2) + "\n")
        print(json.dumps(result, indent=2) if result else "NO RESULT", flush=True)


if __name__ == "__main__":
    main()
