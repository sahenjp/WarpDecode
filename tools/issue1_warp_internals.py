#!/usr/bin/env python3
"""Capture the warp (CUDA engine) internal vectors at positions 0 and 10.

The CPU oracle writes WDPTRC1 vectors through qwen38_forced_cpu_trace.cpp; the
engine writes the same format through Qwen38Trace. tools/issue1_compare_internals.py
aligns and compares them block by block.

This script only produces the engine side: it runs the engine twice (position 0
and 10, forced canonical prefix) and downloads a slim trace that keeps every
per-position vector except the full vocab-sized logits.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-l4")
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "89")
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
POSITIONS = [int(p) for p in os.environ.get("WARODECODE_POSITIONS", "0,10").split(",")]
EXTRA_ARGS = [a for a in os.environ.get("WARODECODE_EXTRA_ARGS", "").split(" ") if a]
SLIM_NAME = os.environ.get("WARODECODE_TRACE_NAME", "warp_pos_internals")

REMOTE = r'''import json, pathlib, struct, subprocess

POSITIONS = __POSITIONS__
MODEL = __MODEL__
ENGINE = pathlib.Path("/content/warodecode/build-sm__ARCH__/warodecode")
SLIM = pathlib.Path("/content/__SLIM_NAME__.bin")
FORCED = __FORCED__
EXTRA_ARGS = __EXTRA_ARGS__

def run(cmd, timeout=1800):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
        raise RuntimeError("command failed (%d)" % p.returncode)
    return p.stdout

# Rebuild is a no-op when cmake sees no changes, but it picks up the freshly
# extracted sources (and their mtimes) whenever the engine is stale.
run(["cmake", "--build", "/content/warodecode/build-sm__ARCH__", "-j4", "--target", "warodecode"])
import hashlib as _hashlib
print("ENGINE_SHA256=" + _hashlib.sha256(ENGINE.read_bytes()).hexdigest(), flush=True)

def parse(path):
    raw = pathlib.Path(path).read_bytes()
    if raw[:8] != b"WDPTRC1\x00":
        raise RuntimeError("bad trace magic: " + path)
    _, count = struct.unpack_from("<II", raw, 8)
    out, off = [], 16
    for _ in range(count):
        ns, vc = struct.unpack_from("<IQ", raw, off); off += 12
        name = raw[off:off + ns].decode(); off += ns
        values = raw[off:off + vc * 4]; off += vc * 4
        out.append((name, vc, values))
    return out

records = []
summary = []
for position in POSITIONS:
    trace = "/content/warp_pos%d.bin" % position
    out = run([str(ENGINE), "--model", MODEL, "--ctx", "128", "--device", "0",
               "--start-token", "1", "--no-graph", "--no-autotune",
               "--generate", str(position + 1), "--state-trace", trace,
               "--state-trace-position", str(position),
               "--forced-input-tokens", FORCED[str(position)]] + EXTRA_ARGS)
    record = next(json.loads(line) for line in reversed(out.splitlines())
                  if line.startswith('{"engine":"warodecode-qwen38"'))
    vectors = parse(trace)
    kept = 0
    for name, count, values in vectors:
        if name.endswith("final.logits"):
            continue
        records.append((name, count, values))
        kept += 1
    summary.append({"position": position, "token": record["token_ids"][position],
                    "vectors_kept": kept, "vectors_total": len(vectors)})
    print("position %d: kept %d/%d vectors" % (position, kept, len(vectors)), flush=True)
    pathlib.Path(trace).unlink(missing_ok=True)

payload = bytearray()
for name, count, values in records:
    nb = name.encode()
    payload += struct.pack("<IQ", len(nb), count) + nb + values
SLIM.write_bytes(b"WDPTRC1\x00" + struct.pack("<II", 1, len(records)) + bytes(payload))

names = [n for n, _, _ in records]
print("COLAB_ROW=" + json.dumps({
    "stage": "trace",
    "slim_bytes": SLIM.stat().st_size,
    "vector_count": len(records),
    "positions": summary,
    "layer0_vectors": [n for n in names if ".layer0." in n or n.startswith("layer0.")],
    "sample_names": names[:20],
}, allow_nan=False), flush=True)
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]
    forced = {str(p): ",".join(str(t) for t in [1] + tokens[:p]) for p in POSITIONS}

    with tempfile.TemporaryDirectory(prefix="warodecode-warp-internals-") as tmp:
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
                       .replace("__POSITIONS__", json.dumps(POSITIONS))
                       .replace("__MODEL__", json.dumps(MODEL))
                       .replace("__ARCH__", str(ARCH))
                       .replace("__FORCED__", json.dumps(forced))
                       .replace("__SLIM_NAME__", SLIM_NAME)
                       .replace("__EXTRA_ARGS__", json.dumps(EXTRA_ARGS)))
        worker = tmp_path / "run_warp_internals.py"
        worker.write_text(worker_text)
        compile(worker_text, "run_warp_internals.py", "exec")

        print("+ upload source", flush=True)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"], check=True)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout[-4000:], flush=True)

        out_path = ROOT / ("issue1_artifacts/" + SLIM_NAME + ".bin")
        dl = CLI + ["download", "-s", SESSION, "/content/" + SLIM_NAME + ".bin", str(out_path)]
        print("+ " + " ".join(dl), flush=True)
        subprocess.run(dl, check=True)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("COLAB_ROW=")]
        meta = json.loads(rows[-1][len("COLAB_ROW="):]) if rows else {}
        (ROOT / ("issue1_artifacts/" + SLIM_NAME + ".json")).write_text(json.dumps(
            {"reference_sha256": digest, "downloaded_bytes": out_path.stat().st_size, **meta}, indent=2) + "\n")
        print(json.dumps(meta, indent=2)[:2000])


if __name__ == "__main__":
    main()
