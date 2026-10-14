#!/usr/bin/env python3
"""Per-layer router-set audit + missing (moe+HC) cube corner on Colab A100.

Remote worker runs position-39 teacher-forced traces for baseline / res_hc /
hc_only and returns every integer vector (expert IDs per layer). Locally we
compare each layer's top-k SET against the pinned CPU trace. It also runs the
missing (moe=1, hc=1, res=0) corner for positions 39 and 36.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_router_audit_colab.py
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
import subprocess
import sys
import tempfile
import time

FORCED39 = __FORCED39__
FORCED36 = __FORCED36__
CPU_TOPK = __CPU_TOPK__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-router-audit.log", "a")

def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    log.write("+ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    log.write(p.stdout + "\n")
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-30:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode})")
    return p.stdout

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

with tempfile.TemporaryDirectory(prefix="issue1-router-audit-") as tmp:
    for name, r, m, h in [("baseline", 0, 0, 0), ("res_hc", 1, 0, 1),
                          ("hc_only", 0, 0, 1), ("moe_hc", 0, 1, 1)]:
        trace = str(pathlib.Path(tmp) / f"{name}.bin")
        env_extra = {"WARODECODE_RESIDUAL_F32": str(r),
                     "WARODECODE_MOE_F32": str(m), "WARODECODE_HC_F32": str(h)}
        record, wall_ms = engine(
            ["--generate", "40", "--state-trace", trace,
             "--state-trace-position", "39", "--forced-input-tokens",
             ",".join(map(str, FORCED39))], env_extra, 1500)
        ints = {v["name"]: v["values"] for v in record["state_trace"]["integer_vectors"]}
        ids = {k: v for k, v in ints.items() if k.endswith(".ffn.expert_ids")}
        emit({"phase": "audit39", "variant": name, "token39": record["token_ids"][39],
              "expert_ids": ids, "all_int_names": sorted(ints),
              "wall_ms": wall_ms})
        record36, wall36 = engine(
            ["--generate", "37", "--state-trace", trace + ".36",
             "--state-trace-position", "36", "--forced-input-tokens",
             ",".join(map(str, FORCED36))], env_extra, 1500)
        ints36 = {v["name"]: v["values"] for v in record36["state_trace"]["integer_vectors"]}
        ids36 = {k: v for k, v in ints36.items() if k.endswith(".ffn.expert_ids")}
        emit({"phase": "audit36", "variant": name, "token36": record36["token_ids"][36],
              "expert_ids": ids36, "wall_ms": wall36})
emit({"phase": "done"})
log.close()
'''


def read_cpu_topk() -> dict:
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
        if name.startswith('pos39.ffn_moe_topk-'):
            lay = int(name.rsplit('-', 1)[1])
            out[lay] = [int(v) for v in struct.unpack(f'<{vc}f', raw[off:off + vc * 4])]
        off += vc * 4
    if len(out) != 48:
        raise RuntimeError("CPU trace topk coverage incomplete")
    return out


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    cpu_topk = read_cpu_topk()
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__FORCED36__", repr([1] + tokens[:36]))
    worker = worker.replace("__CPU_TOPK__", json.dumps({str(k): v for k, v in cpu_topk.items()}))
    compile(worker, "run_audit.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-audit-") as temp:
        path = pathlib.Path(temp) / "run_audit.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        audits39 = [r for r in rows if r.get("phase") == "audit39"]
        audits36 = [r for r in rows if r.get("phase") == "audit36"]
        for audit in audits39:
            print(f"=== {audit['variant']} token39={audit['token39']} ===")
            mism = []
            for key in sorted(audit["expert_ids"], key=lambda k: int(k.split("layer")[1].split(".")[0])):
                lay = int(key.split("layer")[1].split(".")[0])
                wset, cset = set(audit["expert_ids"][key]), set(cpu_topk[lay])
                flag = "OK " if wset == cset else "DIFF"
                if wset != cset:
                    mism.append((lay, sorted(cset - wset), sorted(wset - cset)))
                print(f"  L{lay:2d} {flag} cpu_out={sorted(cset - wset)} warp_out={sorted(wset - cset)}")
            print(f"  mismatch layers: {[m[0] for m in mism]}")
        for audit in audits36:
            print(f"--- {audit['variant']} token36={audit['token36']} ---")
        out = {"status": "completed" if proc.returncode == 0 else "INCOMPLETE",
               "audits39": audits39, "audits36": audits36,
               "cpu_topk": cpu_topk}
        (ROOT / "issue1_artifacts/position39_colab_router_audit.json").write_text(
            json.dumps(out, indent=2) + "\n")
        if proc.returncode:
            print(proc.stdout[-3000:])
            raise RuntimeError(f"audit failed ({proc.returncode})")


if __name__ == "__main__":
    main()
