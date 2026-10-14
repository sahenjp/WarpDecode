#!/usr/bin/env python3
"""Causal layer-output injection boundary for positions 39 and 36 (Colab A100).

Injects CPU l_last at one layer (single traced position) under the res_hc
switches and records token/margin. Bisects depth to find, per position, the
earliest layer from which CPU values restore the expected token. Tells how
much more accuracy is needed and where.

Caveat (known): injection replaces the residual output but not persistent
GDN/QSA states; a FAIL does not exonerate downstream layers.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_boundary_colab.py [L...] [--pos 39|36]
"""
from __future__ import annotations

import argparse
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
N_LAYERS = 48

REMOTE = r'''import json
import os
import pathlib
import struct
import subprocess
import tempfile
import time

FORCED = __FORCED__
POS = __POS__
LAYERS = __LAYERS__
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-boundary.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

target = pathlib.Path(MODEL)
assert target.stat().st_size == MODEL_BYTES
_slim = pathlib.Path("/content/cpu_llast.bin").read_bytes()
assert _slim[:8] == b'WDPTRC1\0'
_, _sc = struct.unpack_from('<II', _slim, 8)
CPU_VECS = {}
_soff = 16
for _ in range(_sc):
    _sns, _svc = struct.unpack_from('<IQ', _slim, _soff)
    _soff += 12
    _snm = _slim[_soff:_soff + _sns].decode()
    _soff += _sns
    CPU_VECS[int(_snm[1:])] = _slim[_soff:_soff + _svc * 4]
    _soff += _svc * 4

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

ENV = {"WARODECODE_RESIDUAL_F32": "1", "WARODECODE_MOE_F32": "0",
       "WARODECODE_HC_F32": "1"}
with tempfile.TemporaryDirectory(prefix="issue1-boundary-") as tmp:
    for lay in LAYERS:
        ivec = pathlib.Path(tmp) / f"cpu_l{lay}.f32"
        ivec.write_bytes(CPU_VECS[lay])
        trace = str(pathlib.Path(tmp) / f"inj_{lay}.bin")
        record, wall_ms = engine(
            ["--generate", str(POS + 1), "--state-trace", trace,
             "--state-trace-position", str(POS),
             "--forced-input-tokens", ",".join(map(str, FORCED)),
             "--debug-layer8-output-layer", str(lay),
             "--debug-layer8-output-f32-file", str(ivec)], ENV, 1500)
        tok = record["token_ids"][POS]
        row = {"pos": POS, "layer": lay, "token": tok, "wall_ms": wall_ms}
        if POS == 39:
            raw = pathlib.Path(trace).read_bytes()
            _, count = struct.unpack_from('<II', raw, 8)
            off = 16
            for _ in range(count):
                ns, vc = struct.unpack_from('<IQ', raw, off)
                off += 12
                vname = raw[off:off + ns].decode()
                off += ns
                if vname == "pos39.final.logits":
                    logits = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
                    row["margin"] = logits[264] - logits[449]
                off += vc * 4
        emit(row)
emit({"phase": "done"})
log.close()
'''


def read_cpu_l_last(pos: int) -> dict:
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
        if name.startswith(f'pos{pos}.l_last-'):
            lay = int(name.rsplit('-', 1)[1])
            if lay not in out:
                out[lay] = raw[off:off + vc * 4]
        off += vc * 4
    if len(out) != N_LAYERS:
        raise RuntimeError(f"CPU l_last coverage at pos{pos}: {len(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", nargs="*", type=int, default=[8, 16, 24, 32, 40, 47])
    parser.add_argument("--pos", type=int, default=39, choices=(36, 39))
    args = parser.parse_args()
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    n = args.pos
    forced = [1] + tokens[:n]
    cpu = read_cpu_l_last(n)
    # embed as hex to keep worker small-ish (48*10240*4 = 2MB -> hex 4MB; use slim upload instead)
    worker = REMOTE.replace("__FORCED__", repr(forced))
    worker = worker.replace("__POS__", repr(n))
    worker = worker.replace("__LAYERS__", repr(args.layers))
    compile(worker, "run_boundary.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-boundary-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "cpu_llast.bin"
        with open(slim, "wb") as _f:
            _f.write(b'WDPTRC1\0')
            _f.write(struct.pack('<II', 1, N_LAYERS))
            for lay in range(N_LAYERS):
                payload = cpu[lay]
                _nb = f"l{lay:02d}".encode()
                _f.write(struct.pack('<IQ', len(_nb), len(payload) // 4))
                _f.write(_nb)
                _f.write(payload)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(slim), "/content/cpu_llast.bin"],
                       text=True, check=True)
        path = temp_path / "run_boundary.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            print(json.dumps(r))
        (ROOT / f"issue1_artifacts/position{n}_colab_boundary.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("boundary failed")


if __name__ == "__main__":
    main()
