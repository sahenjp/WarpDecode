#!/usr/bin/env python3
"""Position sweep: does the engine/CPU divergence start at position 0?

The injection sweep at position 10 flipped the emitted token non-monotonically
(L=5, 33, 41 -> 4613 but L=1, 3, 4, 9, 17 -> 1083), so the argmax token at that
position is a knife-edge indicator. The layer-output NRMSE is the stable
measure instead.

This script captures the state trace at several positions and reports the
per-layer NRMSE against the pinned CPU layer outputs:

  position 0 clean, later positions drifting -> the divergence is created by
      the recurrent/conv state update (GDN) as context grows.
  position 0 already ~1% -> the divergence is a per-kernel mismatch that needs
      no history.

Only the selected position slices are uploaded; the 86 GB GGUF stays on the VM.
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
HC_DIM = 10240
LAYERS = 48
POSITIONS = [0, 1, 2, 3, 5, 10]

REMOTE = r'''import json, pathlib, struct, subprocess

FORCED = __FORCED__
EXPECTED = __EXPECTED__

POSITIONS = __POSITIONS__
LAYERS = __LAYERS__
HC_DIM = __HC_DIM__
MODEL = __MODEL__
REF = pathlib.Path("/content/pos_sweep_cpu_layer_outputs.f32")
ENGINE = pathlib.Path("/content/warodecode/build-sm__ARCH__/warodecode")

ref = REF.read_bytes()
expected = len(POSITIONS) * LAYERS * HC_DIM * 4
if len(ref) != expected:
    raise RuntimeError("reference size mismatch: %d != %d" % (len(ref), expected))

def slice_of(pi, lay):
    base = ((pi * LAYERS) + lay) * HC_DIM * 4
    return struct.unpack_from("<%df" % HC_DIM, ref, base)

def run(cmd, timeout=1800):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
        raise RuntimeError("command failed (%d)" % p.returncode)
    return p.stdout

def read_layer_outputs(path, position):
    raw = pathlib.Path(path).read_bytes()
    if raw[:8] != b"WDPTRC1\x00":
        raise RuntimeError("bad trace magic")
    _, count = struct.unpack_from("<II", raw, 8)
    out, off = {}, 16
    prefix = "pos%d.layer" % position
    for _ in range(count):
        ns, vc = struct.unpack_from("<IQ", raw, off); off += 12
        name = raw[off:off + ns].decode(); off += ns
        if name.startswith(prefix) and name.endswith(".output"):
            out[int(name[len(prefix):-len(".output")])] = struct.unpack_from("<%df" % vc, raw, off)
        off += vc * 4
    return out

def nrmse(a, b):
    err2 = sum((y - x) ** 2 for x, y in zip(a, b))
    ref2 = sum(x * x for x in a)
    return (err2 / len(a)) ** 0.5 / max((ref2 / len(a)) ** 0.5, 1e-30)

rows = []
for pi, position in enumerate(POSITIONS):
    trace = "/content/sweep_pos%d.bin" % position
    out = run([str(ENGINE), "--model", MODEL, "--ctx", "128", "--device", "0",
               "--start-token", "1", "--no-graph", "--no-autotune",
               "--generate", str(position + 1), "--state-trace", trace,
               "--state-trace-position", str(position),
               "--forced-input-tokens", FORCED[str(position)]])
    record = next(json.loads(line) for line in reversed(out.splitlines())
                  if line.startswith('{"engine":"warodecode-qwen38"'))
    vecs = read_layer_outputs(trace, position)
    profile = {}
    for lay in range(LAYERS):
        if lay in vecs:
            profile[str(lay)] = nrmse(slice_of(pi, lay), vecs[lay])
    rows.append({"position": position, "token": record["token_ids"][position],
                 "expected_token": EXPECTED[str(position)], "layer_nrmse": profile})
    print("POSITION %d done: L0=%.5f L4=%.5f L47=%.5f" %
          (position, profile.get("0", float("nan")), profile.get("4", float("nan")),
           profile.get("47", float("nan"))), flush=True)
    pathlib.Path(trace).unlink(missing_ok=True)

print("RESULT=" + json.dumps({
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          text=True, stdout=subprocess.PIPE).stdout.strip(),
    "positions": POSITIONS,
    "rows": rows,
}, allow_nan=False), flush=True)
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]

    with tempfile.TemporaryDirectory(prefix="warodecode-pos-sweep-") as tmp:
        tmp_path = pathlib.Path(tmp)
        slim = tmp_path / "pos_sweep_cpu_layer_outputs.f32"
        payload = bytearray()
        for position in POSITIONS:
            for lay in range(LAYERS):
                path = ROOT / ("issue1_artifacts/position39_cpu_layer%d_output_all_positions.f32" % lay)
                data = path.read_bytes()
                if len(data) < (position + 1) * HC_DIM * 4:
                    raise RuntimeError("layer %d reference too short for position %d" % (lay, position))
                payload += data[position * HC_DIM * 4:(position + 1) * HC_DIM * 4]
        slim.write_bytes(bytes(payload))
        print("slim reference: %d bytes" % slim.stat().st_size, flush=True)

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

        forced = {str(p): ",".join(str(t) for t in [1] + tokens[:p]) for p in POSITIONS}
        expected = {str(p): tokens[p] for p in POSITIONS}
        worker_text = (REMOTE
                       .replace("__POSITIONS__", json.dumps(POSITIONS))
                       .replace("__LAYERS__", str(LAYERS))
                       .replace("__HC_DIM__", str(HC_DIM))
                       .replace("__MODEL__", json.dumps(MODEL))
                       .replace("__ARCH__", str(ARCH))
                       .replace("__FORCED__", json.dumps(forced))
                       .replace("__EXPECTED__", json.dumps(expected)))
        worker = tmp_path / "run_pos_sweep.py"
        worker.write_text(worker_text)
        compile(worker_text, "run_pos_sweep.py", "exec")

        for src, dst in ((archive, "/content/warodecode-source.zip"),
                         (slim, "/content/pos_sweep_cpu_layer_outputs.f32")):
            print("+ upload " + src.name, flush=True)
            subprocess.run(CLI + ["upload", "-s", SESSION, str(src), dst], check=True)

        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "5400", "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout[-8000:], flush=True)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
        result = json.loads(rows[-1][len("RESULT="):]) if rows else None
        out = {"status": "completed" if result else "INCOMPLETE",
               "reference_sha256": digest, "result": result}
        (ROOT / "issue1_artifacts/position_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
        if not result:
            raise RuntimeError("position sweep produced no result")


if __name__ == "__main__":
    main()
