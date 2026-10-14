#!/usr/bin/env python3
"""Run an arbitrary shell command on the Colab VM after building the engine.

Task H variant of colab_run_cmd.py: everything lives under /content/wd_h
(source tree, build dir, per-label logs, pid file) so a sibling task on the same
Colab session cannot overwrite the binary or the log, and the engine only starts
once nvidia-smi reports no foreign compute process (WARODECODE_GPU_WAIT_SECONDS,
default 900).

colab_run_bg.py drives the fixed parity command; this one drives anything else
(the phase benchmark, ad-hoc probes, ...) with the same proven mechanics:
upload the tree, extract it, build incrementally, run the command detached and
poll /content/run_bg.log over HTTP until RUN_EXIT appears.

Usage:
  python3 tools/colab_run_cmd.py --label phases --cmd "bash bench/phases.sh ..."
  python3 tools/colab_run_cmd.py --label phases --cmd "..." \
      --upload issue1_artifacts/reference_tokens_64.json:/content/reference_tokens_64.json \
      --download /content/bench-out/phases-p64d64.json:issue1_artifacts/phases-p64d64.json
"""
from __future__ import annotations

import argparse
import base64
import os
import pathlib
import subprocess
import tempfile
import time
import zipfile

import colab_run_bg as runner

STARTER = '''import base64, os, pathlib, signal, shutil, subprocess, zipfile
ROOT = "__ROOT__"
LABEL = "__LABEL__"
os.makedirs(ROOT, exist_ok=True)
tree = os.path.join(ROOT, "warodecode")
if os.path.isdir(tree): shutil.rmtree(tree)
zipfile.ZipFile("/content/warodecode-source.zip").extractall(tree)
pathlib.Path(os.path.join(ROOT, "run_cmd.sh")).write_bytes(base64.b64decode("__CMD_B64__"))
script = pathlib.Path(os.path.join(ROOT, "run_%s.sh" % LABEL))
script.write_text("""#!/bin/bash
ROOT=__ROOT__
LABEL=__LABEL__
ARCH=__ARCH__
cd $ROOT/warodecode
BUILD=$ROOT/build-sm$ARCH
mkdir -p $BUILD
if [ ! -f $BUILD/CMakeCache.txt ]; then
  cmake -S . -B $BUILD -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH > $ROOT/build_$LABEL.log 2>&1
fi
cmake --build $BUILD -j4 --target warodecode >> $ROOT/build_$LABEL.log 2>&1
BUILD_EXIT=$?
echo BUILD_EXIT=$BUILD_EXIT >> $ROOT/build_$LABEL.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_FAILED: see $ROOT/build_$LABEL.log" > $ROOT/run_$LABEL.log
  echo RUN_EXIT=99 >> $ROOT/run_$LABEL.log
  exit 1
fi
DEADLINE=$(( $(date +%s) + GATE ))
while true; do
  FOREIGN=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c [0-9] || true)
  if [ "$FOREIGN" -eq 0 ]; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "GPU_BUSY_AT_START:" >> $ROOT/run_$LABEL.log
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $ROOT/run_$LABEL.log
    break
  fi
  sleep 15
done
export WD_ENGINE=$ROOT/build-sm$ARCH/warodecode
bash $ROOT/run_cmd.sh > $ROOT/run_$LABEL.log 2>&1
echo RUN_EXIT=$? >> $ROOT/run_$LABEL.log
""")
script.chmod(0o755)
for name in (os.path.join(ROOT, "run_%s.log" % LABEL), os.path.join(ROOT, "build_%s.log" % LABEL)):
    if os.path.exists(name): os.unlink(name)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path(os.path.join(ROOT, "pid_%s" % LABEL)).write_text(str(proc.pid))
print("STARTED", proc.pid)
'''


def split_pair(value: str, option: str) -> tuple[str, str]:
    if ":" not in value:
        raise SystemExit("%s は local:remote 形式で指定してください: %s" % (option, value))
    local, remote = value.split(":", 1)
    return local, remote


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--upload", action="append", default=[],
                        help="起動前に送る local:remote ファイル (複数可)")
    parser.add_argument("--download", action="append", default=[],
                        help="完了後に取り寄せる remote:local ファイル (複数可)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--source-dir", default="",
                        help="アップロードするソースツリー (既定: WARODECODE_SOURCE_DIR か repo)")
    args = parser.parse_args()
    source = pathlib.Path(args.source_dir) if args.source_dir else runner.SOURCE

    uploads = [split_pair(value, "--upload") for value in args.upload]
    downloads = [split_pair(value, "--download") for value in args.download]

    for local, remote in uploads:
        print("+ upload %s -> %s" % (local, remote), flush=True)
        runner.run(runner.CLI + ["upload", "-s", runner.SESSION, str(local), remote])

    with tempfile.TemporaryDirectory(prefix="warodecode-cmd-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "src.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (source / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts:
                        z.write(f, f.relative_to(source))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = source / name
                if f.is_file():
                    z.write(f, name)
        print("+ upload source (%d bytes)" % archive.stat().st_size, flush=True)
        runner.run(runner.CLI + ["upload", "-s", runner.SESSION, str(archive),
                                 "/content/warodecode-source.zip"])

        starter = tmp_path / "start.py"
        starter.write_text(
            STARTER.replace("__CMD_B64__", base64.b64encode(args.cmd.encode()).decode())
                   .replace("__ARCH__", runner.ARCH)
                   .replace("__ROOT__", "/content/wd_h")
                   .replace("__LABEL__", args.label)
                   .replace("GATE", os.environ.get("WARODECODE_GPU_WAIT_SECONDS", "900")))
        print("+ start VM background run", flush=True)
        runner.run(runner.CLI + ["exec", "-s", runner.SESSION, "--timeout", "600",
                                 "-f", str(starter)])

        remote_run = "/content/wd_h/run_%s.log" % args.label
        remote_build = "/content/wd_h/build_%s.log" % args.label
        local_run = tmp_path / "run.log"
        local_build = tmp_path / "build.log"
        deadline = time.monotonic() + args.timeout
        status = None
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            for remote, local in ((remote_run, local_run), (remote_build, local_build)):
                local.unlink(missing_ok=True)
                subprocess.run(runner.CLI + ["download", "-s", runner.SESSION, remote, str(local)],
                               text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            build_tail = local_build.read_text(errors="replace").splitlines()[-1:] if local_build.exists() else []
            text = local_run.read_text(errors="replace") if local_run.exists() else ""
            done = "RUN_EXIT=" in text
            print("[poll] build=%s run_bytes=%d %s" % (build_tail, len(text), "DONE" if done else "running"),
                  flush=True)
            if done:
                status = text
                break

        if status is None:
            print("TIMEOUT: %s が完了しませんでした" % remote_run, flush=True)
            status = local_run.read_text(errors="replace") if local_run.exists() else ""

        log_path = runner.ROOT / ("issue1_artifacts/colab_log_h_%s.txt" % args.label)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(status)
        print("+ saved log -> %s" % log_path.relative_to(runner.ROOT), flush=True)
        print(status[-4000:], flush=True)

        for remote, local in downloads:
            target = pathlib.Path(local)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            print("+ download %s -> %s" % (remote, local), flush=True)
            runner.run(runner.CLI + ["download", "-s", runner.SESSION, remote, str(target)],
                       check=False)

        if "BUILD_FAILED" in status:
            raise SystemExit("BUILD_FAILED: %s を確認してください" % remote_build)
        if "RUN_EXIT=0" not in status:
            raise SystemExit("コマンドが失敗しました (RUN_EXIT != 0)")


if __name__ == "__main__":
    main()
