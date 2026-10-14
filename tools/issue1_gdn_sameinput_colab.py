#!/usr/bin/env python3
"""GDN same-input bisect at layer 0 on Colab A100.

Overrides the layer-0 GDN INPUT with CPU hc_mixed-0 (half-rounded, i.e. the
production input precision) and measures every GDN stage against the CPU
trace: qkv, q/k/v conv, gate, beta, core, norm, output projection. A second
scenario additionally overrides recurrent + convolution states, isolating the
AR-core/norm/out execution from QKV/conv/state inheritance.

Usage:
  WARODECODE_COLAB_SESSION=warpdecode-qwen38-a100 python3 \
      tools/issue1_gdn_sameinput_colab.py
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
CPU_GDN_REMOTE = "/content/position39_cpu_gdn_slim.bin"
CPU_GDN_INPUT_REMOTE = "/content/cpu_gdn_input0.f32"

# (cpu trace name, occurrence, warp trace name)
STAGES = [
    ("pos39.linear_attn_qkv_mixed-0", 0, "pos39.layer0.gdn.qkv_full"),
    ("pos39.q_conv_predelta-0", 0, "pos39.layer0.gdn.q_conv_postnorm"),
    ("pos39.k_conv_predelta-0", 0, "pos39.layer0.gdn.k_conv_postnorm"),
    ("pos39.v_conv_predelta-0", 0, "pos39.layer0.gdn.v_conv"),
    ("pos39.alpha-0", 0, "pos39.layer0.gdn.alpha_full"),
    ("pos39.beta_sigmoid-0", 0, "pos39.layer0.gdn.beta_full"),
    ("pos39.gate-0", 0, "pos39.layer0.gdn.gate_full"),
    ("pos39.z-0", 0, "pos39.layer0.gdn.z_full"),
    ("pos39.gdn_core_output-0", 0, "pos39.layer0.gdn.core_output_full"),
    ("pos39.final_output-0", 0, "pos39.layer0.gdn.norm_output_full"),
    ("pos39.linear_attn_out-0", 0, "pos39.layer0.gdn.final_output_full"),
]

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
CPU_SLIM = "/content/position39_cpu_gdn_slim.bin"
CPU_INPUT = "/content/cpu_gdn_input0.f32"
WANT = __WANT__

root = pathlib.Path("/content/warodecode")
exe = root / "build-sm80/warodecode"
log = open("/content/colab-gdn-si.log", "a")

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

def nrmse(ref, got):
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    n = len(ref)
    return {"nrmse": (err2 / n) ** 0.5 / max((ref2 / n) ** 0.5, 1e-30),
            "rmse": (err2 / n) ** 0.5,
            "max_abs": max(abs(b - a) for a, b in zip(ref, got))}

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

ENV = {"WARODECODE_RESIDUAL_F32": "1", "WARODECODE_MOE_F32": "0",
       "WARODECODE_HC_F32": "1"}
SCEN = [
    ("input_only", []),
    ("input_states", ["--debug-layer9-gdn-state-layer", "0",
                      "--debug-layer9-gdn-state-f32-file", "/content/cpu_gdn_state0.f32",
                      "--debug-layer9-gdn-conv-state-layer", "0",
                      "--debug-layer9-gdn-conv-state-f32-file", "/content/cpu_gdn_conv0.f32"]),
]
with tempfile.TemporaryDirectory(prefix="issue1-gdn-si-") as tmp:
    for scen_name, extra in SCEN:
        trace = str(pathlib.Path(tmp) / f"gdn_si_{scen_name}.bin")
        args = ["--generate", "40", "--state-trace", trace,
                "--state-trace-position", "39",
                "--forced-input-tokens", ",".join(map(str, FORCED39)),
                "--debug-attention-input-f32-layer", "0",
                "--debug-attention-input-f32-file", CPU_INPUT] + extra
        record, wall_ms = engine(args, ENV, 1500)
        raw = pathlib.Path(trace).read_bytes()
        _, count = struct.unpack_from('<II', raw, 8)
        off = 16
        warp = {}
        for _ in range(count):
            ns, vc = struct.unpack_from('<IQ', raw, off)
            off += 12
            vname = raw[off:off + ns].decode()
            off += ns
            if vname in WANT or vname == "pos39.final.logits":
                warp[vname] = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
            off += vc * 4
        stages = {}
        for warp_name, cpu_name in WANT.items():
            stages[warp_name] = dict(nrmse(cpu_vecs[cpu_name], warp[warp_name]))
        fl = warp.get("pos39.final.logits")
        emit({"scenario": scen_name, "token39": record["token_ids"][39],
              "margin": (fl[264] - fl[449]) if fl else None,
              "stages": stages, "wall_ms": wall_ms})
emit({"phase": "done"})
log.close()
'''


def read_cpu_trace() -> dict:
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
        out.setdefault(name, []).append(raw[off:off + vc * 4])
        off += vc * 4
    return out


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    tokens = json.loads(raw)["token_ids"]
    cpu = read_cpu_trace()
    want = {}
    for cpu_name, occ, warp_name in STAGES:
        want[warp_name] = cpu_name
    buf = bytearray(b'WDPTRC1\0')
    items = [(n, cpu[n][o]) for (n, o, _) in STAGES]
    buf += struct.pack('<II', 1, len(items))
    for name, payload in items:
        nb = name.encode()
        buf += struct.pack('<IQ', len(nb), len(payload) // 4)
        buf += nb
        buf += payload
    gdn_input = cpu['pos39.hc_mixed-0'][0]
    import array as _array
    _rec = _array.array('f')
    _rec.frombytes(cpu['pos39.state_predelta-0'][0])
    _w = 128
    _out = _array.array('f', [0.0]) * len(_rec)
    for _h in range(48):
        _base = _h * _w * _w
        for _r in range(_w):
            for _c in range(_w):
                _out[_base + _r * _w + _c] = _rec[_base + _c * _w + _r]
    rec_warp_order = _out.tobytes()
    conv_direct = cpu['pos39.conv_state_at-0'][0]
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__WANT__", repr(want))
    compile(worker, "run_gdn_si.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-gdnsi-") as temp:
        temp_path = pathlib.Path(temp)
        slim = temp_path / "position39_cpu_gdn_slim.bin"
        slim.write_bytes(bytes(buf))
        print(f"slim bytes: {slim.stat().st_size}", flush=True)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(slim), CPU_GDN_REMOTE],
                       text=True, check=True)
        inp = temp_path / "cpu_gdn_input0.f32"
        inp.write_bytes(gdn_input)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(inp), CPU_GDN_INPUT_REMOTE],
                       text=True, check=True)
        recf = temp_path / "cpu_gdn_state0.f32"
        recf.write_bytes(rec_warp_order)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(recf), "/content/cpu_gdn_state0.f32"],
                       text=True, check=True)
        convf = temp_path / "cpu_gdn_conv0.f32"
        convf.write_bytes(conv_direct)
        subprocess.run(CLI + ["upload", "-s", SESSION, str(convf), "/content/cpu_gdn_conv0.f32"],
                       text=True, check=True)
        path = temp_path / "run_gdn_si.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        for r in rows:
            if r.get("phase") == "done":
                continue
            print(f"[{r['scenario']}] token39={r['token39']} margin={r.get('margin')}")
            for k, v in r["stages"].items():
                print(f"  {k:38s} nrmse={v['nrmse']:.6f} rmse={v['rmse']:.6f} maxabs={v['max_abs']:.5f}")
        (ROOT / "issue1_artifacts/position39_colab_gdn_sameinput.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        done = [r for r in rows if r.get("phase") == "done"]
        if proc.returncode or not done:
            print(proc.stdout[-3000:])
            raise RuntimeError("gdn same-input failed")


if __name__ == "__main__":
    main()
