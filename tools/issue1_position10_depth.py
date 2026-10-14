#!/usr/bin/env python3
"""Position-10 depth sweep: find the first layer that diverges on Colab.

The 64-token free-generation parity fails at index 10 on both A100 (sm_80)
and L4 (sm_89), so the divergence at position 10 is architecture independent
and is the most likely root-cause lead.

This script feeds the canonical first 11 tokens (forced prefix), captures the
state trace at position 10, and compares every layer's HC residual output
against the pinned CPU oracle vectors that issue1_artifacts already stores as
position39_cpu_layer<N>_output_all_positions.f32 (40 positions x 10240).

Only the position-10 slice of each layer (48 x 10240 floats = ~2 MiB) is
uploaded; the 86 GB GGUF stays on the Colab VM.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = __import__("os").environ.get("WARODECODE_COLAB_SESSION", "warpdecode-l4")
ARCH = __import__("os").environ.get("WARODECODE_COLAB_ARCH", "89")
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
HC_DIM = 10240          # hidden 2560 * hc_count 4
POSITION = 10
LAYERS = 48

REMOTE = r'''import json, pathlib, struct, subprocess, sys

POSITION = __POSITION__
LAYERS = __LAYERS__
HC_DIM = __HC_DIM__
MODEL = __MODEL__
REF = pathlib.Path("/content/pos10_cpu_layer_outputs.f32")
TRACE = "/content/pos10_state_trace.bin"
ENGINE = pathlib.Path("/content/warodecode/build-sm__ARCH__/warodecode")

ref = REF.read_bytes()
if len(ref) != LAYERS * HC_DIM * 4:
    raise RuntimeError(f"reference size mismatch: {len(ref)}")
reference = [struct.unpack_from(f"<{HC_DIM}f", ref, lay * HC_DIM * 4) for lay in range(LAYERS)]

def run(cmd, timeout=1800):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode})")
    return p.stdout

build_root = pathlib.Path("/content/warodecode/build-sm__ARCH__")
if not ENGINE.is_file() or FORCE_BUILD_MARKER:
    run(["cmake", "--build", str(build_root), "-j4", "--target", "warodecode"])

out = run([str(ENGINE), "--model", MODEL, "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(POSITION + 1), "--state-trace", TRACE,
           "--state-trace-position", str(POSITION),
           "--forced-input-tokens", FORCED])
record = next(json.loads(line) for line in reversed(out.splitlines())
              if line.startswith('{"engine":"warodecode-qwen38"'))

raw = pathlib.Path(TRACE).read_bytes()
if raw[:8] != b"WDPTRC1\0":
    raise RuntimeError("bad trace magic")
_, count = struct.unpack_from("<II", raw, 8)
vecs = {}
off = 16
for _ in range(count):
    ns, vc = struct.unpack_from("<IQ", raw, off)
    off += 12
    name = raw[off:off + ns].decode()
    off += ns
    if name.startswith(f"pos{POSITION}.layer") and name.endswith(".output"):
        vecs[name] = struct.unpack_from(f"<{vc}f", raw, off)
    off += vc * 4

def nrmse(a, b):
    err2 = sum((y - x) ** 2 for x, y in zip(a, b))
    ref2 = sum(x * x for x in a)
    return (err2 / len(a)) ** 0.5 / max((ref2 / len(a)) ** 0.5, 1e-30)

rows = []
first_bad = None
for lay in range(LAYERS):
    name = f"pos{POSITION}.layer{lay}.output"
    if name not in vecs:
        rows.append({"layer": lay, "status": "MISSING"})
        continue
    got = vecs[name]
    err = nrmse(reference[lay], got)
    max_abs = max(abs(y - x) for x, y in zip(reference[lay], got))
    rows.append({"layer": lay, "nrmse": err, "max_abs": max_abs})
    if first_bad is None and err > 1e-3:
        first_bad = lay

print("RESULT=" + json.dumps({
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          text=True, stdout=subprocess.PIPE).stdout.strip(),
    "position": POSITION,
    "token_ids": record["token_ids"],
    "first_layer_over_1e-3": first_bad,
    "layers": rows,
}, allow_nan=False), flush=True)
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]
    if len(tokens) != 64 or tokens[39] != 264 or tokens[36] != 56212:
        raise RuntimeError("unexpected canonical reference contents")
    forced = [1] + tokens[:POSITION]

    with tempfile.TemporaryDirectory(prefix="warodecode-pos10-") as tmp:
        tmp_path = pathlib.Path(tmp)
        slim = tmp_path / "pos10_cpu_layer_outputs.f32"
        payload = bytearray()
        for lay in range(LAYERS):
            path = ROOT / f"issue1_artifacts/position39_cpu_layer{lay}_output_all_positions.f32"
            data = path.read_bytes()
            if len(data) < (POSITION + 1) * HC_DIM * 4:
                raise RuntimeError(f"layer {lay} reference is too short: {len(data)}")
            payload += data[POSITION * HC_DIM * 4:(POSITION + 1) * HC_DIM * 4]
        slim.write_bytes(bytes(payload))
        print(f"slim position-{POSITION} reference: {slim.stat().st_size} bytes", flush=True)

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
                       .replace("__POSITION__", str(POSITION))
                       .replace("__LAYERS__", str(LAYERS))
                       .replace("__HC_DIM__", str(HC_DIM))
                       .replace("__MODEL__", json.dumps(MODEL))
                       .replace("__ARCH__", str(ARCH))
                       .replace("FORCED", json.dumps(",".join(map(str, forced))))
                       .replace("FORCE_BUILD_MARKER", "False"))
        worker = tmp_path / "run_pos10.py"
        worker.write_text(worker_text)
        compile(worker_text, "run_pos10.py", "exec")

        for src, dst in ((archive, "/content/warodecode-source.zip"),
                         (slim, "/content/pos10_cpu_layer_outputs.f32")):
            print("+ upload", src.name, flush=True)
            subprocess.run(CLI + ["upload", "-s", SESSION, str(src), dst], check=True)

        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout[-6000:], flush=True)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
        result = json.loads(rows[-1][len("RESULT="):]) if rows else None
        out = {"status": "completed" if result else "INCOMPLETE",
               "reference_sha256": digest,
               "forced_tokens": forced,
               "result": result}
        (ROOT / "issue1_artifacts/position10_depth.json").write_text(json.dumps(out, indent=2) + "\n")
        if not result:
            raise RuntimeError("position-10 depth sweep produced no result")


if __name__ == "__main__":
    main()
