#!/usr/bin/env python3
"""Read-only Colab probe for the Qwen3.8 tokenizer metadata (Task M).

Nothing is built and no GPU is used: the remote worker only parses the GGUF
metadata of the model shard and reports whether a pinned llama.cpp checkout with
a built llama-tokenize already exists on the VM. Both answers decide how
tests/qwen38_tokenizer_host_check.sh is driven on the VM later.

Session exclusivity: the worker refuses to start when another run owns
/content/run_bg.pid (1 session = 1 job, see docs/COLAB_SESSIONS.md).

Usage:
  export PATH=/home/sahen/.local/bin:$PATH
  WARODECODE_COLAB_SESSION=warpdecode-a100 python3 tools/tokenizer_colab_probe.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import time

import colab_run_bg as runner

REMOTE = r'''
import json
import pathlib
import subprocess

MODEL_CANDIDATES = [
    "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf",
    "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf",
]
out = {"models": [], "llama_cpp": {}}
model = next((p for p in MODEL_CANDIDATES if pathlib.Path(p).exists()), None)
out["model"] = model
if model:
    dumped = subprocess.run(
        ["python3", "/content/warodecode/tools/gguf_tokenizer_dump.py", model],
        capture_output=True, text=True, timeout=1800)
    out["dump_ok"] = dumped.returncode == 0
    out["dump"] = dumped.stdout[-4000:]
    out["dump_err"] = dumped.stderr[-2000:]
llama = pathlib.Path("/content/llama.cpp")
out["llama_cpp"]["exists"] = llama.exists()
if llama.exists():
    binaries = sorted(str(p) for p in llama.rglob("llama-tokenize"))
    out["llama_cpp"]["tokenize_binaries"] = binaries
    out["llama_cpp"]["builds"] = sorted(p.name for p in llama.iterdir() if p.is_dir() and p.name.startswith("build"))
    head = subprocess.run(["git", "-C", str(llama), "log", "--oneline", "-1"],
                          capture_output=True, text=True)
    out["llama_cpp"]["head"] = head.stdout.strip()
print("TOKENIZER_PROBE_JSON=" + json.dumps(out))
'''


def main() -> None:
    session = runner.SESSION
    print("+ session=%s" % session, flush=True)

    # 1. upload the workspace (tools + src + tests) and the probe worker
    with tempfile.TemporaryDirectory(prefix="warodecode-tok-probe-") as tmp:
        tmp_path = pathlib.Path(tmp)
        worker = tmp_path / "worker.py"
        worker.write_text(REMOTE)
        archive = tmp_path / "src.zip"
        import zipfile
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (runner.ROOT / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts:
                        z.write(f, f.relative_to(runner.ROOT))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = runner.ROOT / name
                if f.is_file():
                    z.write(f, name)
        runner.run(runner.CLI + ["upload", "-s", session, str(archive),
                                 "/content/warodecode-source.zip"])
        runner.run(runner.CLI + ["upload", "-s", session, str(worker),
                                 "/content/tokenizer_probe_worker.py"])

        starter = tmp_path / "start.py"
        starter.write_text(
            'import pathlib, subprocess, zipfile, os\n'
            'pidf = pathlib.Path("/content/run_bg.pid")\n'
            'busy = False\n'
            'if pidf.exists():\n'
            '    try:\n'
            '        os.kill(int(pidf.read_text().strip()), 0); busy = True\n'
            '    except Exception:\n'
            '        busy = False\n'
            'if busy:\n'
            '    print("SESSION_BUSY: another run owns /content/run_bg.pid")\n'
            'else:\n'
            '    zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")\n'
            '    out = subprocess.run(["python3", "/content/tokenizer_probe_worker.py"],\n'
            '                         capture_output=True, text=True, timeout=2400)\n'
            '    pathlib.Path("/content/tokenizer_probe.log").write_text(out.stdout + out.stderr)\n'
            '    print("PROBE_DONE", out.returncode)\n')
        print("+ run read-only probe", flush=True)
        runner.run(runner.CLI + ["exec", "-s", session, "--timeout", "600", "-f", str(starter)])

        # 2. poll for the result file over HTTP
        local = tmp_path / "tokenizer_probe.log"
        deadline = time.monotonic() + 2400
        text = ""
        while time.monotonic() < deadline:
            time.sleep(20)
            local.unlink(missing_ok=True)
            subprocess.run(runner.CLI + ["download", "-s", session,
                                         "/content/tokenizer_probe.log", str(local)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            text = local.read_text(errors="replace") if local.exists() else ""
            marker = [l for l in text.splitlines() if l.startswith("TOKENIZER_PROBE_JSON=")]
            print("[poll] bytes=%d %s" % (len(text), "DONE" if marker else "running"), flush=True)
            if marker:
                break
        out_path = runner.ROOT / "issue1_artifacts/tokenizer_probe.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print("+ saved -> %s" % out_path.relative_to(runner.ROOT), flush=True)
        marker = [l for l in text.splitlines() if l.startswith("TOKENIZER_PROBE_JSON=")]
        if marker:
            print(marker[-1][len("TOKENIZER_PROBE_JSON="):][:6000], flush=True)
        else:
            print(text[-2000:], flush=True)
            raise SystemExit("probe did not produce a result")


if __name__ == "__main__":
    main()
