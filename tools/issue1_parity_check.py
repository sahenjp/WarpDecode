#!/usr/bin/env python3
"""Run a 64-token greedy parity check on the Colab GPU with configurable flags.

The canonical reference is issue1_artifacts/reference_tokens_64.json. This
script feeds only the start token (1) and lets the engine generate greedily,
then reports the first mismatching index.

WARODECODE_EXTRA_ARGS : extra engine flags (space separated)
WARODECODE_SWITCHES   : comma separated KEY=VALUE env vars for the engine
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-l4")
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "89")
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
EXTRA_ARGS = [a for a in os.environ.get("WARODECODE_EXTRA_ARGS", "").split(" ") if a]
SWITCHES = dict(kv.split("=", 1) for kv in os.environ.get("WARODECODE_SWITCHES", "").split(",") if "=" in kv)
LABEL = os.environ.get("WARODECODE_LABEL", "parity")

REMOTE = r'''import json, os, pathlib, subprocess

MODEL = __MODEL__
ARCH = __ARCH__
EXTRA_ARGS = __EXTRA_ARGS__
SWITCHES = __SWITCHES__
REF64 = __REF64__
ENGINE = pathlib.Path("/content/warodecode/build-sm__ARCH__/warodecode")

_build = subprocess.run(["cmake", "--build", "/content/warodecode/build-sm__ARCH__",
                        "-j4", "--target", "warodecode"], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
print(_build.stdout[-1500:], flush=True)
if _build.returncode:
    raise RuntimeError("engine build failed (%d)" % _build.returncode)
import hashlib as _hashlib
print("ENGINE_SHA256=" + _hashlib.sha256(ENGINE.read_bytes()).hexdigest(), flush=True)

env = {k: v for k, v in os.environ.items() if not k.startswith("WARODECODE_")}
env.update(SWITCHES)
cmd = [str(ENGINE), "--model", MODEL, "--ctx", "128", "--device", "0",
       "--start-token", "1", "--no-graph", "--no-autotune",
       "--generate", str(len(REF64))] + EXTRA_ARGS
print("+ " + " ".join(cmd) + " env=" + str(SWITCHES), flush=True)
proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, timeout=3600, env=env)
if proc.returncode:
    print(proc.stdout[-2000:], flush=True)
    print(proc.stderr[-2000:], flush=True)
    raise RuntimeError("engine failed (%d)" % proc.returncode)
record = next(json.loads(line) for line in reversed(proc.stdout.splitlines())
              if line.startswith('{"engine":"warodecode-qwen38"'))
got = record["token_ids"]
mismatch = next((i for i, (a, b) in enumerate(zip(REF64, got)) if a != b), None)
print("COLAB_ROW=" + json.dumps({
    "extra_args": EXTRA_ARGS,
    "switches": SWITCHES,
    "match64": mismatch is None and len(got) == len(REF64),
    "first_mismatch": mismatch,
    "got_prefix": got[:24],
    "ref_prefix": REF64[:24],
    "tok_s": record.get("tok_s"),
}, allow_nan=False), flush=True)
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]

    with tempfile.TemporaryDirectory(prefix="warodecode-parity-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "warodecode-source.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for file in (ROOT / name).rglob("*"):
                    if file.is_file() and "qwen38-fixture" not in file.parts:
                        z.write(file, file.relative_to(ROOT))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                file = ROOT / name
                if file.is_file():
                    z.write(file, name)
        worker_text = (REMOTE
                       .replace("__MODEL__", json.dumps(MODEL))
                       .replace("__ARCH__", str(ARCH))
                       .replace("__EXTRA_ARGS__", json.dumps(EXTRA_ARGS))
                       .replace("__SWITCHES__", json.dumps(SWITCHES))
                       .replace("__REF64__", json.dumps(tokens)))
        worker = tmp_path / "run_parity.py"
        worker.write_text(worker_text)
        compile(worker_text, "run_parity.py", "exec")

        print("+ upload source", flush=True)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"], check=True)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "7200", "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout[-4000:], flush=True)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("COLAB_ROW=")]
        result = json.loads(rows[-1][len("COLAB_ROW="):]) if rows else None
        out = {"status": "completed" if result else "INCOMPLETE",
               "reference_sha256": digest, "label": LABEL,
               "extra_args": EXTRA_ARGS, "switches": SWITCHES, "result": result}
        (ROOT / ("issue1_artifacts/parity_" + LABEL + ".json")).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2)[:1500])


if __name__ == "__main__":
    main()
