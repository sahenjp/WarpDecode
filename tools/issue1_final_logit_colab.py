#!/usr/bin/env python3
"""Final-logit error structure on Colab A100.

Compares Warp vs CPU final logits (NRMSE, top-5, margin decomposition) for
baseline / res_hc / full at position 39. Determines whether the ~4.6 margin
deficit is diffuse noise or a systematic late-layer bias.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_final_logit_colab.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-qwen38-a100")
REFERENCE_SHA256 = '2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60'
CPU_LOGITS_REMOTE = "/content/position39_cpu_final_logits.bin"

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
VARIANTS = [("baseline", 0, 0, 0), ("res_hc", 1, 0, 1), ("full", 1, 1, 1)]
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-final-logit.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    return (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30)

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
cpu_logits = struct.unpack(
    '<' + 'f' * (pathlib.Path(CPU_LOGITS).stat().st_size // 4),
    pathlib.Path(CPU_LOGITS).read_bytes())
cpu_top = sorted(range(len(cpu_logits)), key=lambda i: (-cpu_logits[i], i))[:8]

def engine(args, env_extra, timeout):
    env = {k: v for k, v in os.environ.items() if not k.startswith('WARODECODE_')}
    env.update(env_extra)
    cmd = [str(exe), "--model", str(target), "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune"] + args
    begin = time.monotonic()
    p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    wall_ms = (time.monotonic() - begin) * 1000
    log.write("+ " + " ".join(cmd) + "\n" + p.stdout[-2000:] + "\n")
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-30:]), flush=True)
        raise RuntimeError("engine failed")
    record = next(json.loads(line) for line in reversed(p.stdout.splitlines())
                  if line.startswith('{"engine":"warodecode-qwen38"'))
    return record, wall_ms

with tempfile.TemporaryDirectory(prefix="issue1-finlog-") as tmp:
    for name, r, m, h in VARIANTS:
        trace = str(pathlib.Path(tmp) / f"{name}.bin")
        record, wall_ms = engine(
            ["--generate", "40", "--state-trace", trace,
             "--state-trace-position", "39",
             "--forced-input-tokens", ",".join(map(str, FORCED39))],
            {"WARODECODE_RESIDUAL_F32": str(r), "WARODECODE_MOE_F32": str(m),
             "WARODECODE_HC_F32": str(h)}, 1500)
        raw = pathlib.Path(trace).read_bytes()
        _, count = struct.unpack_from('<II', raw, 8)
        off = 16
        logits = None
        for _ in range(count):
            ns, vc = struct.unpack_from('<IQ', raw, off)
            off += 12
            vname = raw[off:off + ns].decode()
            off += ns
            if vname == "pos39.final.logits":
                logits = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
            off += vc * 4
        wtop = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:8]
        emit({"variant": name, "token39": record["token_ids"][39],
              "margin": logits[264] - logits[449],
              "cpu_margin": cpu_logits[264] - cpu_logits[449],
              "logits_nrmse": nrmse(cpu_logits, logits),
              "logits_max_abs": max(abs(b - a) for a, b in zip(cpu_logits, logits)),
              "warp_top8": [(i, logits[i]) for i in wtop],
              "cpu_top8": [(i, cpu_logits[i]) for i in cpu_top],
              "wall_ms": wall_ms})
emit({"phase": "done"})
log.close()
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    trav = (ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin").read_bytes()
    _, count = struct.unpack_from('<II', trav, 8)
    off = 16
    payload = None
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', trav, off)
        off += 12
        vname = trav[off:off + ns].decode()
        off += ns
        if vname == "pos39.final.logits":
            payload = trav[off:off + vc * 4]
        off += vc * 4
    if payload is None:
        raise RuntimeError("CPU final logits missing")
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("CPU_LOGITS", repr(CPU_LOGITS_REMOTE))
    compile(worker, "run_finlog.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-finlog-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "position39_cpu_final_logits.bin"
        slim.write_bytes(payload)
        print(f"cpu logits bytes: {slim.stat().st_size}", flush=True)
        up = CLI + ["upload", "-s", SESSION, str(slim), CPU_LOGITS_REMOTE]
        subprocess.run(up, text=True, check=True)
        path = temp_path / "run_finlog.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r)[:1500])
        (ROOT / "issue1_artifacts/position39_colab_final_logit.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("finlog failed")


if __name__ == "__main__":
    main()
