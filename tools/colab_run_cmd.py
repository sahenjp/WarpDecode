#!/usr/bin/env python3
"""Run an arbitrary shell command on the Colab VM after building the engine.

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
import pathlib
import subprocess
import tempfile
import time
import zipfile

import colab_run_bg as runner

STARTER = '''import base64, os, pathlib, signal, subprocess, zipfile
zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")
# The Colab exec websocket can be retried, which would otherwise start a second
# run against the same GPU. Only one run may own the VM: kill any previous one.
pidf = pathlib.Path("/content/run_bg.pid")
if pidf.exists():
    try:
        old_pid = int(pidf.read_text().strip())
        cmdline = pathlib.Path("/proc/%d/cmdline" % old_pid).read_bytes().decode(errors="replace")
        if "run_bg.sh" in cmdline:
            os.killpg(os.getpgid(old_pid), signal.SIGKILL)
            print("KILLED previous run", old_pid)
        else:
            print("previous pid belongs to another process:", cmdline[:60])
    except Exception as exc:
        print("no previous run:", exc)
pathlib.Path("/content/run_cmd.sh").write_bytes(base64.b64decode("__CMD_B64__"))
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
bash /content/run_cmd.sh > /content/run_bg.log 2>&1
echo RUN_EXIT=$? >> /content/run_bg.log
""")
script.chmod(0o755)
for name in ("/content/run_bg.log", "/content/build_bg.log"):
    pathlib.Path(name).unlink(missing_ok=True)
proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
pathlib.Path("/content/run_bg.pid").write_text(str(proc.pid))
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
                   .replace("__ARCH__", runner.ARCH))
        print("+ start VM background run", flush=True)
        runner.run(runner.CLI + ["exec", "-s", runner.SESSION, "--timeout", "600",
                                 "-f", str(starter)])

        local_run = tmp_path / "run_bg.log"
        local_build = tmp_path / "build_bg.log"
        deadline = time.monotonic() + args.timeout
        status = None
        while time.monotonic() < deadline:
            time.sleep(args.poll)
            for remote, local in (("/content/run_bg.log", local_run),
                                  ("/content/build_bg.log", local_build)):
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
            print("TIMEOUT: /content/run_bg.log が完了しませんでした", flush=True)
            status = local_run.read_text(errors="replace") if local_run.exists() else ""

        log_path = runner.ROOT / ("issue1_artifacts/colab_log_%s.txt" % args.label)
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
            raise SystemExit("BUILD_FAILED: /content/build_bg.log を確認してください")
        if "RUN_EXIT=0" not in status:
            raise SystemExit("コマンドが失敗しました (RUN_EXIT != 0)")


if __name__ == "__main__":
    main()
