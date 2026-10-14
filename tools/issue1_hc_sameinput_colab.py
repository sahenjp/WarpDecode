#!/usr/bin/env python3
"""HC-mix same-input test at layer 0 on Colab A100.

Overrides the layer-0 input residual with CPU hc_init and measures the HC
attention mix (norm/gate/mix) against CPU hc_norm-0[0]/hc_gate-0[0]/
hc_mixed-0[0]. The F32 HC path (res_hc env) has never been same-input
validated; with 96 HC mixes per forward pass it is the largest unmeasured
component.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_hc_sameinput_colab.py
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
CPU_INIT_REMOTE = "/content/cpu_hc_init0.f32"

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
CPU_SLIM = "/content/cpu_hc_slim.bin"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
CPU_INIT = "/content/cpu_hc_init0.f32"

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-hc-si.log", "a")

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
assert _slim[:8] == b'WDPTRC1\0'
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
CPU_NORM = _cpu['hc_norm']
CPU_GATE = _cpu['hc_gate']
CPU_MIX = _cpu['hc_mix']

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
with tempfile.TemporaryDirectory(prefix="issue1-hc-si-") as tmp:
    trace = str(pathlib.Path(tmp) / "hc_si.bin")
    args = ["--generate", "40", "--state-trace", trace,
            "--state-trace-position", "39",
            "--forced-input-tokens", ",".join(map(str, FORCED39)),
            "--debug-layer-input-residual-layer", "0",
            "--debug-layer-input-residual-f32-file", CPU_INIT]
    record, wall_ms = engine(args, ENV, 1500)
    norm = read_vec(trace, "pos39.layer0.hc_attn_norm")
    gate = read_vec(trace, "pos39.layer0.hc_attn_gate")
    mix = read_vec(trace, "pos39.layer0.hc_attn_mix")
    logits = read_vec(trace, "pos39.final.logits")
    emit({"token39": record["token_ids"][39],
          "margin": logits[264] - logits[449],
          "norm_vs_cpu": nrmse(CPU_NORM, norm),
          "gate_vs_cpu": nrmse(CPU_GATE, gate),
          "mix_vs_cpu": nrmse(CPU_MIX, mix),
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
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    compile(worker, "run_hc_si.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-hcsi-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "cpu_hc_slim.bin"
        with open(slim, "wb") as _f:
            items = [("hc_norm", read_cpu_vec('pos39.hc_norm-0', 0)),
                     ("hc_gate", read_cpu_vec('pos39.hc_gate-0', 0)),
                     ("hc_mix", read_cpu_vec('pos39.hc_mixed-0', 0))]
            _f.write(b'WDPTRC1\0')
            _f.write(struct.pack('<II', 1, len(items)))
            for _nm, _vals in items:
                _nb = _nm.encode()
                _f.write(struct.pack('<IQ', len(_nb), len(_vals)))
                _f.write(_nb)
                _f.write(struct.pack(f'<{len(_vals)}f', *_vals))
        print(f"slim bytes: {slim.stat().st_size}", flush=True)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(slim), "/content/cpu_hc_slim.bin"],
                       text=True, check=True)
        init = temp_path / "cpu_hc_init0.f32"
        init.write_bytes(struct.pack("<10240f", *read_cpu_vec('pos39.hc_init')))
        subprocess.run(CLI + ["upload", "-s", SESSION, str(init), CPU_INIT_REMOTE],
                       text=True, check=True)
        path = temp_path / "run_hc_si.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / "issue1_artifacts/position39_colab_hc_sameinput.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("hc same-input failed")


if __name__ == "__main__":
    main()
