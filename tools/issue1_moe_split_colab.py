#!/usr/bin/env python3
"""MoE routed-vs-shared split under exact inputs (Colab A100).

Exact layer-8 MoE input (CPU hc_mixed-8) + CPU norm routes, res_hc env.
Compares routed-only (ffn.ffn_moe_out vs ffn_out minus shexp_gated) and
shared (ffn_shexp_gated vs CPU ffn_shexp_gated-8) separately to isolate
which MoE half carries the 0.011 execution error.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_moe_split_colab.py
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
CPU_MIX_REMOTE = "/content/cpu_hc_mix8.f32"
CPU_SLIM_REMOTE = "/content/cpu_moe_split_slim.bin"

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
ROUTES = __ROUTES__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
CPU_SLIM = "/content/cpu_moe_split_slim.bin"
CPU_MIX = "/content/cpu_hc_mix8.f32"

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-moe-split.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    return {"nrmse": (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30),
            "rmse": (err2 / len(ref)) ** 0.5,
            "max_abs": max(abs(b - a) for a, b in zip(ref, got))}

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
_slim = pathlib.Path(CPU_SLIM).read_bytes()
_, _sc = struct.unpack_from('<II', _slim, 8)
_cpu = {}
_soff = 16
for _ in range(_sc):
    _sns, _svc = struct.unpack_from('<IQ', _slim, _soff)
    _soff += 12
    _snm = _slim[_soff:_soff + _sns].decode()
    _soff += _sns
    _cpu[_snm] = struct.unpack(f'<{_svc}f', _slim[_soff:_soff + _svc * 4])
    _soff += _svc * 4
CPU_ROUTED = [a - b for a, b in zip(_cpu['ffn_out'], _cpu['shexp_gated'])]
CPU_SHARED = _cpu['shexp_gated']
CPU_COMBINED = _cpu['ffn_out']

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

def read_vec(trace_path, name):
    raw = pathlib.Path(trace_path).read_bytes()
    _, count = struct.unpack_from('<II', raw, 8)
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        vname = raw[off:off + ns].decode()
        off += ns
        if vname == name:
            return struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
        off += vc * 4
    raise RuntimeError(name + " missing")

ENV = {"WARODECODE_RESIDUAL_F32": "1", "WARODECODE_MOE_F32": "0",
       "WARODECODE_HC_F32": "1"}
with tempfile.TemporaryDirectory(prefix="issue1-moe-split-") as tmp:
    trace = str(pathlib.Path(tmp) / "split.bin")
    args = ["--generate", "40", "--state-trace", trace,
            "--state-trace-position", "39",
            "--forced-input-tokens", ",".join(map(str, FORCED39)),
            "--debug-layer8-hc-ffn-mix-f32-file", CPU_MIX,
            "--debug-moe-ids", ",".join(map(str, ROUTES["ids"])),
            "--debug-moe-weights", ",".join(map(str, ROUTES["weights"]))]
    record, wall_ms = engine(args, ENV, 1500)
    routed = read_vec(trace, "pos39.layer8.ffn.ffn_moe_out")
    shared = read_vec(trace, "pos39.layer8.ffn_shexp_gated")
    combined = read_vec(trace, "pos39.layer8.ffn_output")
    logits = read_vec(trace, "pos39.final.logits")
    emit({"token39": record["token_ids"][39],
          "margin": logits[264] - logits[449],
          "routed_vs_cpu": nrmse(CPU_ROUTED, routed),
          "shared_vs_cpu": nrmse(CPU_SHARED, shared),
          "combined_vs_cpu": nrmse(CPU_COMBINED, combined),
          "wall_ms": wall_ms})
emit({"phase": "done"})
log.close()
'''


def read_cpu_vec(name: str, occurrence: int = 0) -> list:
    raw = (ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin").read_bytes()
    _, count = struct.unpack_from('<II', raw, 8)
    off = 16
    seen = 0
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        vname = raw[off:off + ns].decode()
        off += ns
        if vname == name:
            if seen == occurrence:
                return list(struct.unpack(f'<{vc}f', raw[off:off + vc * 4]))
            seen += 1
        off += vc * 4
    raise RuntimeError(f"{name} occ {occurrence} missing")


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    routes = {"ids": [int(v) for v in read_cpu_vec('pos39.ffn_moe_topk-8')],
              "weights": [float(v) for v in read_cpu_vec('pos39.ffn_moe_weights_norm-8')]}
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__ROUTES__", repr(routes))
    compile(worker, "run_moe_split.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-moesplit-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "cpu_moe_split_slim.bin"
        with open(slim, "wb") as _f:
            items = [("ffn_out", read_cpu_vec('pos39.ffn_out-8')),
                     ("shexp_gated", read_cpu_vec('pos39.ffn_shexp_gated-8'))]
            _f.write(b'WDPTRC1\0')
            _f.write(struct.pack('<II', 1, len(items)))
            for _nm, _vals in items:
                _nb = _nm.encode()
                _f.write(struct.pack('<IQ', len(_nb), len(_vals)))
                _f.write(_nb)
                _f.write(struct.pack(f'<{len(_vals)}f', *_vals))
        subprocess.run(CLI + ["upload", "-s", SESSION, str(slim), CPU_SLIM_REMOTE],
                       text=True, check=True)
        path = temp_path / "run_moe_split.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / "issue1_artifacts/position39_colab_moe_split.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("moe split failed")


if __name__ == "__main__":
    main()
