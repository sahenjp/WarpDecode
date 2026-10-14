#!/usr/bin/env python3
"""MoE execution fidelity under forced CPU routes (Colab A100).

Forces the pinned-CPU layer-8 routes and measures the layer-8 FFN output
(routed + shared) NRMSE against the CPU ffn_out-8 vector, on-VM. This
separates expert-execution error from routing error completely:
  - forced MoE out ~= CPU  -> execution faithful, problem is downstream
  - forced MoE out still far -> expert execution itself diverges

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_moe_exec_colab.py
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
ROUTES39 = __ROUTES39__
CPU_FFN_OUT8 = __CPU_FFN_OUT8__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
SCENARIOS = __SCENARIOS__

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-moe-exec.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

target = pathlib.Path(MODEL)
if target.stat().st_size != MODEL_BYTES:
    raise RuntimeError("local model size mismatch")

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    n = len(ref)
    return {"nrmse": (err2 / n) ** 0.5 / max((ref2 / n) ** 0.5, 1e-30),
            "rmse": (err2 / n) ** 0.5,
            "max_abs": max(abs(b - a) for a, b in zip(ref, got))}

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
    raise RuntimeError(f"{name} not in trace")

with tempfile.TemporaryDirectory(prefix="issue1-moe-exec-") as tmp:
    for sc in SCENARIOS:
        name, r, m, h, routes = sc["name"], sc["r"], sc["m"], sc["h"], sc["routes"]
        trace = str(pathlib.Path(tmp) / f"{name}.bin")
        args = ["--generate", "40", "--state-trace", trace,
                "--state-trace-position", "39",
                "--forced-input-tokens", ",".join(map(str, FORCED39))]
        if routes:
            args += ["--debug-moe-ids", ",".join(map(str, ROUTES39["ids"])),
                     "--debug-moe-weights", ",".join(map(str, ROUTES39["weights"]))]
        record, wall_ms = engine(
            args,
            {"WARODECODE_RESIDUAL_F32": str(r), "WARODECODE_MOE_F32": str(m),
             "WARODECODE_HC_F32": str(h)}, 1500)
        ffn_out = read_vec(trace, "pos39.layer8.ffn_output")
        logits = read_vec(trace, "pos39.final.logits")
        emit({"name": name, "token39": record["token_ids"][39],
              "margin": logits[264] - logits[449],
              "ffn_out_vs_cpu": nrmse(CPU_FFN_OUT8, ffn_out),
              "wall_ms": wall_ms})
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
    routes39 = {"ids": [int(v) for v in read_cpu_vec('pos39.ffn_moe_topk-8')],
                "weights": [float(v) for v in read_cpu_vec('pos39.ffn_moe_weights_norm-8')]}
    scenarios = [
        {"name": "res_hc_forced", "r": 1, "m": 0, "h": 1, "routes": True},
        {"name": "res_hc_native", "r": 1, "m": 0, "h": 1, "routes": False},
        {"name": "baseline_forced", "r": 0, "m": 0, "h": 0, "routes": True},
        {"name": "moe_hc_forced", "r": 0, "m": 1, "h": 1, "routes": True},
    ]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__ROUTES39__", repr(routes39))
    worker = worker.replace("__CPU_FFN_OUT8__", repr([float(v) for v in read_cpu_vec('pos39.ffn_out-8')]))
    worker = worker.replace("__SCENARIOS__", repr(scenarios))
    compile(worker, "run_moe_exec.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-moeexec-") as temp:
        path = pathlib.Path(temp) / "run_moe_exec.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / "issue1_artifacts/position39_colab_moe_exec.json").write_text(
            json.dumps({"status": "completed" if proc.returncode == 0 else "INCOMPLETE",
                        "reference_sha256": REFERENCE_SHA256,
                        "rows": rows}, indent=2) + "\n")
        if proc.returncode:
            print(proc.stdout[-3000:])
            raise RuntimeError(f"moe exec audit failed ({proc.returncode})")


if __name__ == "__main__":
    main()
