#!/usr/bin/env python3
"""Per-layer FFN/output depth curve under fixed switches (Colab A100).

For baseline and res_hc at position 39, records per layer:
  ffn_output NRMSE, layer output NRMSE, router top-k set match vs CPU.
CPU vectors (48 ffn_out + 48 l_last + 48 topk) travel as a ~3MB slim file.
Finds where error still grows after the res_hc fix.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_depth_colab.py
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
CPU_SLIM_REMOTE = "/content/position39_cpu_depth_slim.bin"

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED39 = __FORCED39__
VARIANTS = [("baseline", 0, 0, 0), ("res_hc", 1, 0, 1)]
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-depth.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    n = len(ref)
    return (err2 / n) ** 0.5 / max((ref2 / n) ** 0.5, 1e-30)

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
cpu_raw = pathlib.Path(CPU_SLIM).read_bytes()
assert cpu_raw[:8] == b'WDPTRC1\0'
_, _cc = struct.unpack_from('<II', cpu_raw, 8)
cpu_vecs = {}
_coff = 16
for _ in range(_cc):
    _ns, _vc = struct.unpack_from('<IQ', cpu_raw, _coff)
    _coff += 12
    _cn = cpu_raw[_coff:_coff + _ns].decode()
    _coff += _ns
    _cpu_vecs[_cn] = struct.unpack(f'<{_vc}f', cpu_raw[_coff:_coff + _vc * 4])
    _coff += _vc * 4
cpu_vecs = dict(_cpu_vecs)

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

with tempfile.TemporaryDirectory(prefix="issue1-depth-") as tmp:
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
        warp = {}
        off = 16
        for _ in range(count):
            ns, vc = struct.unpack_from('<IQ', raw, off)
            off += 12
            vname = raw[off:off + ns].decode()
            off += ns
            if vname.startswith("pos39.layer") and (
                    vname.endswith(".ffn_output") or vname.endswith(".output")):
                warp[vname] = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
            off += vc * 4
        ids = {v["name"]: v["values"] for v in record["state_trace"]["integer_vectors"]
               if v["name"].endswith(".ffn.expert_ids")}
        rows = []
        for lay in range(48):
            fkey, okey = f"pos39.layer{lay}.ffn_output", f"pos39.layer{lay}.output"
            ck = f"pos39.ffn_out-{lay}"
            ok = f"pos39.l_last-{lay}"
            wtop = ids.get(f"pos39.layer{lay}.ffn.expert_ids", [])
            ctop = [int(v) for v in cpu_vecs.get(f"pos39.ffn_moe_topk-{lay}", [])]
            rows.append({"layer": lay,
                         "ffn_nrmse": nrmse(cpu_vecs[ck], warp[fkey]),
                         "out_nrmse": nrmse(cpu_vecs[ok], warp[okey]),
                         "topk_match": set(wtop) == set(ctop)})
        logits = None
        emit({"variant": name, "token39": record["token_ids"][39],
              "layers": rows, "wall_ms": wall_ms})
emit({"phase": "done"})
log.close()
'''


def build_cpu_slim() -> bytes:
    raw = (ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin").read_bytes()
    if raw[:8] != b'WDPTRC1\0':
        raise RuntimeError("bad CPU trace magic")
    _, count = struct.unpack_from('<II', raw, 8)
    wanted = set()
    for lay in range(48):
        wanted.add(f'pos39.ffn_out-{lay}')
        wanted.add(f'pos39.l_last-{lay}')
        wanted.add(f'pos39.ffn_moe_topk-{lay}')
    kept = []
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from('<IQ', raw, off)
        off += 12
        name = raw[off:off + ns].decode()
        off += ns
        payload = raw[off:off + vc * 4]
        off += vc * 4
        if name in wanted and not any(k[0] == name for k in kept):
            kept.append((name, payload))
    missing = wanted - {k for k, _ in kept}
    if missing:
        raise RuntimeError(f"CPU trace missing: {sorted(missing)[:5]}")
    buf = bytearray(b'WDPTRC1\0')
    buf += struct.pack('<II', 1, len(kept))
    for name, payload in kept:
        nb = name.encode()
        buf += struct.pack('<IQ', len(nb), len(payload) // 4)
        buf += nb
        buf += payload
    return bytes(buf)


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("CPU_SLIM", repr(CPU_SLIM_REMOTE))
    compile(worker, "run_depth.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-depth-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "position39_cpu_depth_slim.bin"
        slim.write_bytes(build_cpu_slim())
        print(f"slim bytes: {slim.stat().st_size}", flush=True)
        up = CLI + ["upload", "-s", SESSION, str(slim), CPU_SLIM_REMOTE]
        subprocess.run(up, text=True, check=True)
        path = temp_path / "run_depth.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            if r.get("phase") == "done":
                continue
            print(f"=== {r['variant']} token39={r['token39']} ===")
            print(f"{'L':>3s} {'ffn_out':>9s} {'layer_out':>9s} {'topk':>5s}")
            for lay in r["layers"]:
                print(f"{lay['layer']:3d} {lay['ffn_nrmse']:9.5f} {lay['out_nrmse']:9.5f} "
                      f"{'same' if lay['topk_match'] else 'DIFF':>5s}")
        (ROOT / "issue1_artifacts/position39_colab_depth.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("depth run failed")


if __name__ == "__main__":
    main()
