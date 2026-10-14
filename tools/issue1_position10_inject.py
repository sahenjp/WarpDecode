#!/usr/bin/env python3
"""Position-10 CPU-injection sweep: which layer must be repaired?

At position 10 the engine emits 1083 where the canonical (llama.cpp) token is
4613, and this mismatch is architecture independent (A100 sm_80 and L4 sm_89
both fail at index 10). The layer-output depth sweep showed the largest single
step at layer 4 (+108% nrmse), immediately after layer 3, the first QSA layer.

This sweep injects the pinned CPU residual (issue1_artifacts/
position39_cpu_layer<N>_output_all_positions.f32, position 10 slice) into the
input of layer L and reports the token produced at position 10:

  token == 4613  -> repairing everything up to layer L is sufficient
  token != 4613  -> the corruption already dominates before layer L

Only the position-10 slices (48 x 10240 floats = ~2 MiB) are uploaded; the
86 GB GGUF stays on the Colab VM.
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
POSITION = 10
LAYERS = 48
INJECT_LAYERS = [1, 3, 4, 5, 9, 17, 33, 41]

REMOTE = r'''import json, pathlib, struct, subprocess

POSITION = __POSITION__
LAYERS = __LAYERS__
HC_DIM = __HC_DIM__
MODEL = __MODEL__
INJECT = __INJECT__
REF = pathlib.Path("/content/pos10_cpu_layer_outputs.f32")
TRACE = "/content/pos10_inject_trace.bin"
ENGINE = pathlib.Path("/content/warodecode/build-sm__ARCH__/warodecode")

ref = REF.read_bytes()
if len(ref) != LAYERS * HC_DIM * 4:
    raise RuntimeError("reference size mismatch")

def run(cmd, timeout=1800):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
        raise RuntimeError("command failed (%d)" % p.returncode)
    return p.stdout

BASE = [str(ENGINE), "--model", MODEL, "--ctx", "128", "--device", "0",
        "--start-token", "1", "--no-graph", "--no-autotune",
        "--generate", str(POSITION + 1), "--state-trace", TRACE,
        "--state-trace-position", str(POSITION),
        "--forced-input-tokens", FORCED]

def token_of(out):
    record = next(json.loads(line) for line in reversed(out.splitlines())
                  if line.startswith('{"engine":"warodecode-qwen38"'))
    return record["token_ids"]

rows = []
tokens = token_of(run(list(BASE)))
rows.append({"inject_layer": None, "source_layer": None, "token10": tokens[POSITION],
             "token_ids": tokens})
print("BASELINE token10=%d" % tokens[POSITION], flush=True)

for layer in INJECT:
    source = layer - 1
    path = "/content/inject_source_layer%d.f32" % source
    pathlib.Path(path).write_bytes(ref[source * HC_DIM * 4:(source + 1) * HC_DIM * 4])
    cmd = list(BASE) + ["--debug-layer-input-residual-layer", str(layer),
                        "--debug-layer-input-residual-f32-file", path]
    tokens = token_of(run(cmd))
    rows.append({"inject_layer": layer, "source_layer": source,
                 "token10": tokens[POSITION], "token_ids": tokens})
    print("INJECT layer=%d token10=%d" % (layer, tokens[POSITION]), flush=True)

print("RESULT=" + json.dumps({
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          text=True, stdout=subprocess.PIPE).stdout.strip(),
    "position": POSITION,
    "expected_token": EXPECTED,
    "rows": rows,
}, allow_nan=False), flush=True)
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]
    expected = tokens[POSITION]
    forced = [1] + tokens[:POSITION]

    with tempfile.TemporaryDirectory(prefix="warodecode-pos10-inject-") as tmp:
        tmp_path = pathlib.Path(tmp)
        slim = tmp_path / "pos10_cpu_layer_outputs.f32"
        payload = bytearray()
        for lay in range(LAYERS):
            path = ROOT / ("issue1_artifacts/position39_cpu_layer%d_output_all_positions.f32" % lay)
            data = path.read_bytes()
            if len(data) < (POSITION + 1) * HC_DIM * 4:
                raise RuntimeError("layer %d reference is too short" % lay)
            payload += data[POSITION * HC_DIM * 4:(POSITION + 1) * HC_DIM * 4]
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

        worker_text = (REMOTE
                       .replace("__POSITION__", str(POSITION))
                       .replace("__LAYERS__", str(LAYERS))
                       .replace("__HC_DIM__", str(HC_DIM))
                       .replace("__INJECT__", json.dumps(INJECT_LAYERS))
                       .replace("__MODEL__", json.dumps(MODEL))
                       .replace("__ARCH__", str(ARCH))
                       .replace("EXPECTED", str(expected))
                       .replace("FORCED", json.dumps(",".join(map(str, forced)))))
        worker = tmp_path / "run_pos10_inject.py"
        worker.write_text(worker_text)
        compile(worker_text, "run_pos10_inject.py", "exec")

        for src, dst in ((archive, "/content/warodecode-source.zip"),
                         (slim, "/content/pos10_cpu_layer_outputs.f32")):
            print("+ upload " + src.name, flush=True)
            subprocess.run(CLI + ["upload", "-s", SESSION, str(src), dst], check=True)

        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "5400", "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout[-8000:], flush=True)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
        result = json.loads(rows[-1][len("RESULT="):]) if rows else None
        out = {"status": "completed" if result else "INCOMPLETE",
               "reference_sha256": digest,
               "forced_tokens": forced,
               "expected_token10": expected,
               "result": result}
        (ROOT / "issue1_artifacts/position10_inject.json").write_text(json.dumps(out, indent=2) + "\n")
        if not result:
            raise RuntimeError("position-10 injection sweep produced no result")


if __name__ == "__main__":
    main()
