#!/usr/bin/env python3
"""Position-39 four-way precision ablation on the Colab T4 session.

Mirrors tools/issue1_position39_ablation.py (Modal) but executes on the
existing Colab session via `colab exec`. One worker runs, in order:

  phase 1: 4 position-39 teacher-forced variants
           (baseline / residual-F32 / MoE-F32 / combined)
  phase 2: position-36 gate for every variant
  phase 3: 64-token free-generation parity for the best variant

The 86GB model is streamed over the FUSE HTTP mount (no download).
Each completed row is printed immediately so partial progress survives.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-qwen38-t4")
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "75")
REFERENCE_SHA256 = '2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60'
VARIANTS = [('baseline', 0, 0, 0), ('residual_only', 1, 0, 0),
            ('moe_output_only', 0, 1, 0), ('combined', 1, 1, 0),
            ('hc_only', 0, 0, 1), ('res_hc', 1, 0, 1), ('full', 1, 1, 1)]

REMOTE = r'''import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time

FORCED39 = __FORCED39__
FORCED36 = __FORCED36__
REF64 = __REF64__
EXPECTED39 = 264
EXPECTED36 = 56212
VARIANTS = [('baseline', 0, 0, 0), ('residual_only', 1, 0, 0),
            ('moe_output_only', 0, 1, 0), ('combined', 1, 1, 0),
            ('hc_only', 0, 0, 1), ('res_hc', 1, 0, 1), ('full', 1, 1, 1)]
FUSE_WORKER_SOURCE = __FUSE_WORKER__
MODEL_URL = "https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/resolve/0cd618ade637402b7870e993135233f7ae78387c/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
CPU_TRACE = "/content/position39_cpu_reference_hc_trace.bin"
CHECKPOINTS = (0, 8, 16, 24, 35, 47)

root = pathlib.Path("/content/warodecode")
import zipfile as _zf
with _zf.ZipFile("/content/warodecode-source.zip") as _z:
    _z.extractall(root)
print("extracted fresh source archive", flush=True)
exe = root / "build-sm__ARCH__/warodecode"
log = open("/content/colab-ablation.log", "a")

def run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    log.write("+ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, **kw)
    log.write(p.stdout + "\n")
    if p.returncode:
        print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout

def emit(obj):
    print("COLAB_ROW=" + json.dumps(obj, allow_nan=False), flush=True)

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"],
                     text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
print("GPU=" + gpu, flush=True)
run(["cmake", "-S", str(root), "-B", str(root / "build-sm__ARCH__"),
     "-DCMAKE_BUILD_TYPE=Release", "-DWARODECODE_BUILD_ENGINE=ON",
     "-DCMAKE_CUDA_ARCHITECTURES=__ARCH__"])
run(["cmake", "--build", str(root / "build-sm__ARCH__"), "-j4",
     "--target", "warodecode"])
exe_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
print("BINARY_SHA256=" + exe_sha, flush=True)

# Model source: prefer the fully downloaded local file; fall back to the
# FUSE HTTP stream only if it is absent.
import importlib.util
USE_FUSE = not pathlib.Path(MODEL).is_file()
target = None
if not USE_FUSE:
    target = pathlib.Path(MODEL)
    if target.stat().st_size != MODEL_BYTES:
        raise RuntimeError(f"local model size mismatch: {target.stat().st_size}")
    print(f"using local model file {target} ({target.stat().st_size} bytes)", flush=True)
mount = pathlib.Path("/content/qwen38-http")
mount.mkdir(exist_ok=True)
fuse = None
if USE_FUSE and not os.path.ismount(mount):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "fusepy"], check=True)
    worker = pathlib.Path("/content/qwen38_fuse_worker.py")
    worker.write_text(FUSE_WORKER_SOURCE)
    fuse_log = open("/content/qwen38-fuse.log", "w")
    fuse = subprocess.Popen([sys.executable, str(worker), MODEL_URL, str(mount)],
                            stdout=fuse_log, stderr=fuse_log, start_new_session=True)
try:
    if USE_FUSE:
        for _ in range(180):
            if os.path.ismount(mount) and (mount / pathlib.Path(MODEL).name).exists():
                break
            if fuse is not None and fuse.poll() is not None:
                fuse_log.flush()
                tail = pathlib.Path("/content/qwen38-fuse.log").read_text(errors="replace")[-2000:]
                raise RuntimeError(f"FUSE model mount process exited; log tail: {tail}")
            time.sleep(1)
        target = mount / pathlib.Path(MODEL).name
        if not os.path.ismount(mount) or not target.exists():
            raise RuntimeError("FUSE model path was not mounted")
        if target.stat().st_size != MODEL_BYTES:
            raise RuntimeError(f"mounted model size mismatch: {target.stat().st_size}")

    def read_trace(path, wanted=None):
        raw = pathlib.Path(path).read_bytes()
        if raw[:8] != b'WDPTRC1\0':
            raise RuntimeError('bad trace magic')
        _, count = struct.unpack_from('<II', raw, 8)
        out = {}
        off = 16
        for _ in range(count):
            ns, vc = struct.unpack_from('<IQ', raw, off)
            off += 12
            name = raw[off:off + ns].decode()
            off += ns
            if wanted is None or name in wanted:
                out[name] = struct.unpack(f'<{vc}f', raw[off:off + vc * 4])
            off += vc * 4
        return out

    def nrmse(ref, got):
        err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
        ref2 = sum(a * a for a in ref)
        return (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30)

    cpu_raw = pathlib.Path(CPU_TRACE).read_bytes()
    if cpu_raw[:8] != b'WDPTRC1\0':
        raise RuntimeError('bad CPU trace magic')
    _, _cc = struct.unpack_from('<II', cpu_raw, 8)
    cpu_layers = {}
    _coff = 16
    for _ in range(_cc):
        _ns, _vc = struct.unpack_from('<IQ', cpu_raw, _coff)
        _coff += 12
        _cname = cpu_raw[_coff:_coff + _ns].decode()
        _coff += _ns
        if _cname.startswith('pos39.l_last-'):
            _lay = int(_cname.rsplit('-', 1)[1])
            if _lay in CHECKPOINTS and _lay not in cpu_layers:
                cpu_layers[_lay] = struct.unpack(f'<{_vc}f', cpu_raw[_coff:_coff + _vc * 4])
        _coff += _vc * 4
    if set(cpu_layers) != set(CHECKPOINTS):
        raise RuntimeError(f'CPU trace missing checkpoint layers: {set(CHECKPOINTS) - set(cpu_layers)}')

    def engine(args, env_extra, timeout):
        env = {k: v for k, v in os.environ.items() if not k.startswith('WARODECODE_')}
        env.update(env_extra)
        cmd = [str(exe), "--model", str(target), "--ctx", "128", "--device", "0",
               "--start-token", "1", "--no-graph", "--no-autotune"] + args
        print("+ " + " ".join(cmd) + f" env={env_extra}", flush=True)
        begin = time.monotonic()
        p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        wall_ms = (time.monotonic() - begin) * 1000
        log.write("+ " + " ".join(cmd) + "\n" + p.stdout[-4000:] + "\n")
        if p.returncode:
            print("\n".join(p.stdout.splitlines()[-40:]), flush=True)
            raise RuntimeError(f"engine failed ({p.returncode})")
        record = next(json.loads(line) for line in reversed(p.stdout.splitlines())
                      if line.startswith('{"engine":"warodecode-qwen38"'))
        return record, wall_ms

    with tempfile.TemporaryDirectory(prefix="issue1-ablation-") as tmp:
        phase1 = []
        for name, r, m, h in VARIANTS:
            trace = str(pathlib.Path(tmp) / f"{name}.bin")
            record, wall_ms = engine(
                ["--generate", "40", "--state-trace", trace,
                 "--state-trace-position", "39", "--forced-input-tokens",
                 ",".join(map(str, FORCED39))],
                {"WARODECODE_RESIDUAL_F32": str(r), "WARODECODE_MOE_F32": str(m),
                 "WARODECODE_HC_F32": str(h)}, 1500)
            state = record["state_trace"]
            if state["trace_position"] != 39:
                raise RuntimeError("wrong trace position")
            if any(state.get("debug_injection_counts", {}).values()):
                raise RuntimeError("unexpected diagnostic injection")
            wanted = {"pos39.final.logits", "pos39.layer8.ffn.router_logits"}
            wanted |= {f"pos39.layer{lay}.output" for lay in CHECKPOINTS}
            vecs = read_trace(trace, wanted)
            logits = vecs["pos39.final.logits"]
            router = vecs["pos39.layer8.ffn.router_logits"]
            ids = next(v["values"] for v in state["integer_vectors"]
                       if v["name"] == "pos39.layer8.ffn.expert_ids")
            layer_nrmse = {str(lay): nrmse(cpu_layers[lay], vecs[f"pos39.layer{lay}.output"])
                           for lay in CHECKPOINTS}
            row = {"phase": 1, "variant": name, "residual_f32": r, "moe_f32": m,
                   "hc_f32": h,
                   "token39": record["token_ids"][39], "expected39": EXPECTED39,
                   "logit264": logits[264], "logit449": logits[449],
                   "margin264_449": logits[264] - logits[449],
                   "layer8_topk": ids, "layer_nrmse": layer_nrmse, "wall_ms": wall_ms}
            emit(row)
            phase1.append(row)

        order = sorted(phase1, key=lambda row: (row["token39"] != EXPECTED39,
                                                -(row["margin264_449"])))
        phase2 = []
        for row in phase1:
            name = row["variant"]
            trace36 = str(pathlib.Path(tmp) / f"{name}_36.bin")
            record, wall_ms = engine(
                ["--generate", "37", "--state-trace", trace36,
                 "--state-trace-position", "36", "--forced-input-tokens",
                 ",".join(map(str, FORCED36))],
                {"WARODECODE_RESIDUAL_F32": str(row["residual_f32"]),
                 "WARODECODE_MOE_F32": str(row["moe_f32"]),
                 "WARODECODE_HC_F32": str(row["hc_f32"])}, 1500)
            gate = {"phase": 2, "variant": name,
                    "token36": record["token_ids"][36], "expected36": EXPECTED36,
                    "pass36": record["token_ids"][36] == EXPECTED36,
                    "wall_ms": wall_ms}
            emit(gate)
            phase2.append(gate)

        winner = next((row for row in order
                       if any(g["variant"] == row["variant"] and g["pass36"]
                              for g in phase2)), None)
        parity = None
        if winner is not None:
            record, wall_ms = engine(
                ["--generate", "64"],
                {"WARODECODE_RESIDUAL_F32": str(winner["residual_f32"]),
                 "WARODECODE_MOE_F32": str(winner["moe_f32"]),
                 "WARODECODE_HC_F32": str(winner["hc_f32"])}, 1700)
            got = record["token_ids"]
            mismatch = next((i for i, (a, b) in enumerate(zip(REF64, got)) if a != b), None)
            parity = {"phase": 3, "variant": winner["variant"],
                      "match64": mismatch is None and len(got) == 64,
                      "first_mismatch": mismatch, "got_len": len(got),
                      "wall_ms": wall_ms}
            emit(parity)

    emit({"phase": "done", "gpu": gpu, "phase1": phase1, "phase2": phase2,
          "parity": parity})
finally:
    log.close()
'''


def fuse_worker_source() -> str:
    tree = ast.parse((ROOT / "issue1_colab_probe.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FUSE_WORKER"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("FUSE_WORKER not found")


def make_source_archive(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("src", "tests", "tools", "bench"):
            for file in (ROOT / name).rglob("*"):
                if file.is_file() and "qwen38-fixture" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
        for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]
    if len(tokens) != 64 or tokens[39] != 264 or tokens[36] != 56212:
        raise RuntimeError("unexpected canonical reference contents")
    worker = REMOTE.replace("__FORCED39__", repr([1] + tokens[:39]))
    worker = worker.replace("__FORCED36__", repr([1] + tokens[:36]))
    worker = worker.replace("__REF64__", repr(tokens))
    worker = worker.replace("__FUSE_WORKER__", repr(fuse_worker_source()))
    worker = worker.replace("__ARCH__", ARCH)
    compile(worker, "run_ablation.py", "exec")
    for token in ("CHECKPOINTS", "CPU_TRACE", "MODEL_URL", "MODEL_BYTES",
                  "FORCED39", "FORCED36", "REF64", "FUSE_WORKER_SOURCE"):
        if token not in worker:
            raise RuntimeError(f"worker template missing {token}")
    if "__ARCH__" in worker or "__FORCED" in worker or "__REF64__" in worker or "__FUSE_WORKER__" in worker:
        raise RuntimeError("worker template has unreplaced placeholders")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-ablation-") as temp:
        temp_path = pathlib.Path(temp)
        archive = temp_path / "warodecode-source.zip"
        make_source_archive(archive)
        up = CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"]
        print("+", " ".join(up), flush=True)
        subprocess.run(up, text=True, check=True)
        cputrace = ROOT / "issue1_artifacts/position39_cpu_reference_hc_trace.bin"
        # The upload endpoint rejects the full 179MB trace; slim it down to
        # the checkpoint layer outputs the worker actually compares.
        import struct as _struct
        raw = cputrace.read_bytes()
        if raw[:8] != b'WDPTRC1\0':
            raise RuntimeError("bad local CPU trace magic")
        _, _count = _struct.unpack_from('<II', raw, 8)
        wanted = {f'pos39.l_last-{lay}' for lay in (0, 8, 16, 24, 35, 47)}
        kept = []
        _off = 16
        for _ in range(_count):
            _ns, _vc = _struct.unpack_from('<IQ', raw, _off)
            _off += 12
            _name = raw[_off:_off + _ns].decode()
            _off += _ns
            _payload = raw[_off:_off + _vc * 4]
            _off += _vc * 4
            if _name in wanted and not any(k[0] == _name for k in kept):
                kept.append((_name, _payload))
        if {k for k, _ in kept} != wanted:
            raise RuntimeError("local CPU trace missing checkpoint layers")
        slim = temp_path / "position39_cpu_reference_hc_trace.bin"
        with open(slim, "wb") as _f:
            _f.write(b'WDPTRC1\0')
            _f.write(_struct.pack('<II', 1, len(kept)))
            for _name, _payload in kept:
                _nb = _name.encode()
                _f.write(_struct.pack('<IQ', len(_nb), len(_payload) // 4))
                _f.write(_nb)
                _f.write(_payload)
        print(f"slim CPU trace: {slim.stat().st_size} bytes, {len(kept)} vectors", flush=True)
        up2 = CLI + ["upload", "-s", SESSION, str(slim),
                     "/content/position39_cpu_reference_hc_trace.bin"]
        print("+", " ".join(up2), flush=True)
        subprocess.run(up2, text=True, check=True)
        path = temp_path / "run_ablation.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "7200", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        print(proc.stdout[-6000:], flush=True)
        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        done = next((r for r in rows if r.get("phase") == "done"), None)
        out = {"status": "completed" if proc.returncode == 0 and done else "INCOMPLETE",
               "returncode": proc.returncode, "gpu": done["gpu"] if done else None,
               "reference_sha256": digest,
               "phase1": done["phase1"] if done else [],
               "phase2": done["phase2"] if done else [],
               "parity": done["parity"] if done else None}
        (ROOT / "issue1_artifacts/position39_colab_ablation.json").write_text(
            json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        if proc.returncode:
            raise RuntimeError(f"Colab ablation failed ({proc.returncode})")


if __name__ == "__main__":
    main()
