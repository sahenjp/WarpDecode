#!/usr/bin/env python3
"""MoE same-input execution test at layer 8 (Colab A100).

Overrides the layer-8 MoE INPUT with CPU hc_mixed-8 (half-rounded, i.e. the
production input precision), with and without CPU route override, and measures
layer-8 ffn_output vs CPU ffn_out-8 plus the final margin:
  A. exact input + CPU routes  -> pure execution error
  B. exact input + native routes -> input vs routing contribution

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_moe_sameinput_colab.py
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
CPU_VECS = "/content/cpu_vecs.bin"
CPU_MIX = "/content/cpu_hc_mix8.f32"

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-moe-sameinput.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    return (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30)

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
CPU_FFN_OUT8 = _cpu['pos39.ffn_out-8']

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
with tempfile.TemporaryDirectory(prefix="issue1-moe-si-") as tmp:
    for name, with_routes in [("exact_input_cpu_routes", True),
                              ("exact_input_native_routes", False)]:
        trace = str(pathlib.Path(tmp) / f"{name}.bin")
        args = ["--generate", "40", "--state-trace", trace,
                "--state-trace-position", "39",
                "--forced-input-tokens", ",".join(map(str, FORCED39)),
                "--debug-layer8-hc-ffn-mix-f32-file", CPU_MIX]
        if with_routes:
            args += ["--debug-moe-ids", ",".join(map(str, ROUTES["ids"])),
                     "--debug-moe-weights", ",".join(map(str, ROUTES["weights"]))]
        record, wall_ms = engine(args, ENV, 1500)
        ffn_out = read_vec(trace, "pos39.layer8.ffn_output")
        logits = read_vec(trace, "pos39.final.logits")
        emit({"name": name, "token39": record["token_ids"][39],
              "margin": logits[264] - logits[449],
              "ffn8_vs_cpu": nrmse(CPU_FFN_OUT8, ffn_out),
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
    assert "CPU_MIX" in worker
    compile(worker, "run_moe_si.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-moesi-") as temp:
        temp_path = pathlib.Path(temp)
        mix = temp_path / "cpu_hc_mix8.f32"
        mix.write_bytes(struct.pack(f"<2560f", *read_cpu_vec('pos39.hc_mixed-8', 1)))
        up = CLI + ["upload", "-s", SESSION, str(mix), CPU_MIX_REMOTE]
        subprocess.run(up, text=True, check=True)
        path = temp_path / "run_moe_si.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / "issue1_artifacts/position39_colab_moe_sameinput.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("moe same-input failed")


if __name__ == "__main__":
    main()
