#!/usr/bin/env python3
"""CPU layer-8 route forcing on Colab A100 (causal routing test).

Forces the pinned-CPU top-k IDs + weights at layer 8 (the only layer with a
route-override hook) under various precision switches, for positions 39 and
36. Decides whether the remaining margin gap is caused by wrong expert
selection (routing) or by wrong expert execution / downstream numerics.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_route_force_colab.py
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
FORCED36 = __FORCED36__
ROUTES39 = __ROUTES39__
ROUTES36 = __ROUTES36__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
SCENARIOS = __SCENARIOS__

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-route-force.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

target = pathlib.Path(MODEL)
if target.stat().st_size != MODEL_BYTES:
    raise RuntimeError("local model size mismatch")

def engine(args, env_extra, timeout):
    env = {k: v for k, v in os.environ.items() if not k.startswith('WARODECODE_')}
    env.update(env_extra)
    cmd = [str(exe), "--model", str(target), "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune"] + args
    print("+ " + " ".join(cmd[:8]) + f" ... env={env_extra}", flush=True)
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

def read_logits(trace_path):
    raw = pathlib.Path(trace_path).read_bytes()
    _, count = struct.unpack_from('<II', raw, 8)
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        name = raw[off:off + ns].decode()
        off += ns
        if name == 'pos39.final.logits' or name == 'pos36.final.logits':
            return struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
        off += vc * 4
    raise RuntimeError("final logits not in trace")

with tempfile.TemporaryDirectory(prefix="issue1-route-force-") as tmp:
    for sc in SCENARIOS:
        name, pos, r, m, h, routes = (sc["name"], sc["pos"], sc["r"], sc["m"],
                                      sc["h"], sc["routes"])
        forced = FORCED39 if pos == 39 else FORCED36
        gen = 40 if pos == 39 else 37
        trace = str(pathlib.Path(tmp) / f"{name}.bin")
        args = ["--generate", str(gen), "--state-trace", trace,
                "--state-trace-position", str(pos),
                "--forced-input-tokens", ",".join(map(str, forced))]
        if routes:
            args += ["--debug-moe-ids", ",".join(map(str, routes["ids"]))]
            if routes["weights"] is not None:
                args += ["--debug-moe-weights", ",".join(map(str, routes["weights"]))]
        record, wall_ms = engine(
            args,
            {"WARODECODE_RESIDUAL_F32": str(r), "WARODECODE_MOE_F32": str(m),
             "WARODECODE_HC_F32": str(h)}, 1500)
        idx = pos
        tok = record["token_ids"][idx]
        row = {"name": name, "pos": pos, "token": tok, "wall_ms": wall_ms,
               "routes_forced": bool(routes)}
        if pos == 39:
            logits = read_logits(trace)
            row.update({"logit264": logits[264], "logit449": logits[449],
                        "margin": logits[264] - logits[449]})
        emit(row)
emit({"phase": "done"})
log.close()
'''


def read_cpu_routes() -> dict:
    raw = (ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin").read_bytes()
    if raw[:8] != b'WDPTRC1\0':
        raise RuntimeError("bad CPU trace magic")
    _, count = struct.unpack_from('<II', raw, 8)
    out = {}
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        name = raw[off:off + ns].decode()
        off += ns
        if name in ('pos39.ffn_moe_topk-8', 'pos39.ffn_moe_weights_norm-8',
                    'pos36.ffn_moe_topk-8', 'pos36.ffn_moe_weights_norm-8'):
            out[name] = list(struct.unpack(f'<{vc}f', raw[off:off + vc * 4]))
        off += vc * 4
    return out


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    cpu = read_cpu_routes()
    routes39 = {"ids": [int(v) for v in cpu['pos39.ffn_moe_topk-8']],
                "weights": [float(v) for v in cpu['pos39.ffn_moe_weights_norm-8']]}
    routes36 = {"ids": [int(v) for v in cpu['pos36.ffn_moe_topk-8']],
                "weights": [float(v) for v in cpu['pos36.ffn_moe_weights_norm-8']]}
    scenarios = [
        {"name": "res_hc_cpuRoutes39", "pos": 39, "r": 1, "m": 0, "h": 1, "routes": routes39},
        {"name": "baseline_cpuRoutes39", "pos": 39, "r": 0, "m": 0, "h": 0, "routes": routes39},
        {"name": "moe_hc_cpuRoutes39", "pos": 39, "r": 0, "m": 1, "h": 1, "routes": routes39},
        {"name": "res_hc_cpuIdsOnly39", "pos": 39, "r": 1, "m": 0, "h": 1,
         "routes": {"ids": routes39["ids"], "weights": None}},
        {"name": "res_hc_cpuRoutes36", "pos": 36, "r": 1, "m": 0, "h": 1, "routes": routes36},
        {"name": "moe_hc_cpuRoutes36", "pos": 36, "r": 0, "m": 1, "h": 1, "routes": routes36},
        {"name": "combined_cpuRoutes36", "pos": 36, "r": 1, "m": 1, "h": 0, "routes": routes36},
    ]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__FORCED36__", repr([1] + tokens[:36]))
    worker = worker.replace("__ROUTES39__", repr(routes39))
    worker = worker.replace("__ROUTES36__", repr(routes36))
    worker = worker.replace("__SCENARIOS__", repr([
        {"name": s["name"], "pos": s["pos"], "r": s["r"], "m": s["m"], "h": s["h"],
         "routes": s["routes"]} for s in scenarios]))
    compile(worker, "run_route_force.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-routes-") as temp:
        path = pathlib.Path(temp) / "run_route_force.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / "issue1_artifacts/position39_colab_route_force.json").write_text(
            json.dumps({"status": "completed" if proc.returncode == 0 else "INCOMPLETE",
                        "reference_sha256": REFERENCE_SHA256,
                        "rows": rows}, indent=2) + "\n")
        if proc.returncode:
            print(proc.stdout[-3000:])
            raise RuntimeError(f"route force failed ({proc.returncode})")


if __name__ == "__main__":
    main()
