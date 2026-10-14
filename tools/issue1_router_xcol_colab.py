#!/usr/bin/env python3
"""Layer-8 router 5-way cross on Colab A100 (uses built-in router_cross hook).

Compares, against pinned CPU logits-8:
  A. cpu_activation_reference_logits   (exact F32 GEMV on CPU input)
  B. warp_activation_reference_logits (exact F32 GEMV on Warp input)
  C. warp_activation_warp_cuda_logits  (production GEMV on Warp input)
  D. cpu_activation_warp_cuda_logits   (production GEMV on CPU input)
  E. warp_activation_warp_cuda_f32_input_logits
Reports NRMSE + top-10 set for each: isolates router-input error (B vs A)
from GEMV-arithmetic error (D vs A, C vs B).

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_router_xcol_colab.py
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

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
CPU_VECS = "/content/cpu_vecs.bin"
CPU_ACT = "/content/cpu_router_act.f32"

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-router-xcol.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    return (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30)

def topk(vec, k=10):
    return sorted(range(len(vec)), key=lambda i: (-vec[i], i))[:k]

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
cpu_raw = pathlib.Path(CPU_VECS).read_bytes()
_, _cc = struct.unpack_from('<II', cpu_raw, 8)
_cpu = {}
_coff = 16
for _ in range(_cc):
    _ns, _vc = struct.unpack_from('<IQ', cpu_raw, _coff)
    _coff += 12
    _cn = cpu_raw[_coff:_coff + _ns].decode()
    _coff += _ns
    _cpu[_cn] = struct.unpack(f'<{_vc}f', cpu_raw[_coff:_coff + _vc * 4])
    _coff += _vc * 4
cpu_logits = _cpu['pos39.ffn_moe_logits-8']
cpu_set = set(topk(cpu_logits))

tmp = tempfile.mkdtemp(prefix="issue1-xcol-")
trace = tmp + "/xcol.bin"
env = {k: v for k, v in os.environ.items() if not k.startswith('WARODECODE_')}
env.update({"WARODECODE_RESIDUAL_F32": "1", "WARODECODE_MOE_F32": "0",
            "WARODECODE_HC_F32": "1"})
cmd = [str(exe), "--model", str(target), "--ctx", "128", "--device", "0",
       "--start-token", "1", "--no-graph", "--no-autotune",
       "--generate", "40", "--state-trace", trace,
       "--state-trace-position", "39",
       "--forced-input-tokens", ",".join(map(str, FORCED39)),
       "--debug-router-cpu-activation-f32-file", CPU_ACT]
p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, timeout=1500)
log.write(p.stdout[-2000:] + "\n")
assert p.returncode == 0, p.stdout[-2000:]
raw = pathlib.Path(trace).read_bytes()
_, count = struct.unpack_from('<II', raw, 8)
off = 16
vecs = {}
for _ in range(count):
    ns, vc = struct.unpack_from('<IQ', raw, off)
    off += 12
    vname = raw[off:off + ns].decode()
    off += ns
    if "router_cross" in vname:
        vecs[vname] = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
    off += vc * 4
for vname, vals in sorted(vecs.items()):
    tks = topk(vals)
    emit({"vector": vname, "nrmse_vs_cpu": nrmse(cpu_logits, vals),
          "top10": tks, "set_match_cpu": set(tks) == cpu_set,
          "max_abs": max(abs(b - a) for a, b in zip(cpu_logits, vals))})
emit({"phase": "done", "cpu_top10": sorted(cpu_set),
      "vectors": sorted(vecs)})
log.close()
'''


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    compile(worker, "run_xcol.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-xcol-") as temp:
        path = pathlib.Path(temp) / "run_xcol.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r)[:600])
        (ROOT / "issue1_artifacts/position39_colab_router_xcol.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("xcol failed")


if __name__ == "__main__":
    main()
