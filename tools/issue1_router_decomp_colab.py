#!/usr/bin/env python3
"""Layer-8 router decomposition: logits, weights, execution (Colab A100).

Single res_hc native run at position 39. Compares Warp vs CPU for:
  router logits (512) -> softmax weights (10) -> MoE output (2560)
against embedded CPU vectors, to locate where the 0.13 native error and the
0.81 forced error come from.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_router_decomp_colab.py
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
import math
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
CPU_VECS = "/content/cpu_vecs.bin"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-router-decomp.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    n = len(ref)
    return (err2 / n) ** 0.5 / max((ref2 / n) ** 0.5, 1e-30)

def read_vec(trace_path, names):
    raw = pathlib.Path(trace_path).read_bytes()
    _, count = struct.unpack_from('<II', raw, 8)
    off = 16
    out = {}
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        vname = raw[off:off + ns].decode()
        off += ns
        if vname in names:
            out[vname] = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
        off += vc * 4
    return out

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
_cpu_raw = pathlib.Path(CPU_VECS).read_bytes()
assert _cpu_raw[:8] == b'WDPTRC1\0'
_, _cc = struct.unpack_from('<II', _cpu_raw, 8)
_cpu_vecs = {}
_coff = 16
for _ in range(_cc):
    _ns, _vc = struct.unpack_from('<IQ', _cpu_raw, _coff)
    _coff += 12
    _cn = _cpu_raw[_coff:_coff + _ns].decode()
    _coff += _ns
    _cpu_vecs[_cn] = struct.unpack(f'<{_vc}f', _cpu_raw[_coff:_coff + _vc * 4])
    _coff += _vc * 4
CPU_LOGITS8 = _cpu_vecs['pos39.ffn_moe_logits-8']
CPU_W8 = _cpu_vecs['pos39.ffn_moe_weights-8']
CPU_FFN_OUT8 = _cpu_vecs['pos39.ffn_out-8']
tmp = tempfile.mkdtemp(prefix="issue1-decomp-")
trace = tmp + "/decomp.bin"
env = {k: v for k, v in os.environ.items() if not k.startswith('WARODECODE_')}
env.update({"WARODECODE_RESIDUAL_F32": "1", "WARODECODE_MOE_F32": "0",
            "WARODECODE_HC_F32": "1"})
cmd = [str(exe), "--model", str(target), "--ctx", "128", "--device", "0",
       "--start-token", "1", "--no-graph", "--no-autotune",
       "--generate", "40", "--state-trace", trace,
       "--state-trace-position", "39",
       "--forced-input-tokens", ",".join(map(str, FORCED39))]
p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, timeout=1500)
log.write(p.stdout[-2000:] + "\n")
assert p.returncode == 0, p.stdout[-2000:]
record = next(json.loads(line) for line in reversed(p.stdout.splitlines())
              if line.startswith('{"engine":"warodecode-qwen38"'))
vecs = read_vec(trace, {"pos39.layer8.ffn.router_logits",
                        "pos39.layer8.ffn.expert_weights",
                        "pos39.layer8.ffn_output", "pos39.final.logits"})
wlogits = vecs["pos39.layer8.ffn.router_logits"]
wweights = vecs["pos39.layer8.ffn.expert_weights"]
wout = vecs["pos39.layer8.ffn_output"]
logits = vecs["pos39.final.logits"]
top = sorted(range(len(wlogits)), key=lambda i: -wlogits[i])[:12]
emit({"warp_top12_logits": [(i, wlogits[i]) for i in top],
      "cpu_top12_logits": sorted([(i, CPU_LOGITS8[i]) for i in top],
                                 key=lambda t: -t[1])[:12],
      "warp_weights": list(wweights),
      "cpu_weights": CPU_W8,
      "logits_nrmse": nrmse(CPU_LOGITS8, wlogits),
      "weights_max_abs": max(abs(a - b) for a, b in zip(CPU_W8, wweights)),
      "moe_nrmse": nrmse(CPU_FFN_OUT8, wout),
      "token39": record["token_ids"][39],
      "margin": logits[264] - logits[449]})
emit({"phase": "done"})
log.close()
'''


def read_cpu_vec(name: str) -> list:
    raw = (ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin").read_bytes()
    _, count = struct.unpack_from('<II', raw, 8)
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        vname = raw[off:off + ns].decode()
        off += ns
        if vname == name:
            return list(struct.unpack(f'<{vc}f', raw[off:off + vc * 4]))
        off += vc * 4
    raise RuntimeError(f"{name} missing")


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    for _tok in ("__CPU_LOGITS8__", "__CPU_W8__", "__CPU_FFN_OUT8__"):
        assert _tok not in worker, _tok
    compile(worker, "run_decomp.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-decomp-") as temp:
        path = pathlib.Path(temp) / "run_decomp.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r)[:2000])
        (ROOT / "issue1_artifacts/position39_colab_router_decomp.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError(f"decomp failed (rc={proc.returncode}, done={bool(done)})")


if __name__ == "__main__":
    main()
