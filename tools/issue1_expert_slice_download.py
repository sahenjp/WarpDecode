#!/usr/bin/env python3
"""Download layer-8 expert weight slices for host-side exact MoE reference.

Extracts gate/up (IQ2_S) and down (IQ4_NL) slices for 10 experts at layer 8
plus shared-expert Q8_0 tensors, via gguf-py memmap slicing (no full-tensor
RAM load), and downloads them with per-tensor SHA-256 for integrity.

Usage:
  python3 tools/issue1_expert_slice_download.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = "warpdecode-qwen38-a100"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
EXPERTS = [66, 399, 127, 365, 224, 68, 500, 237, 216, 426]

REMOTE = r'''import hashlib
import json
import os
import pathlib
import subprocess

MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
EXPERTS = __EXPERTS__

subprocess.run(["python3", "-m", "pip", "install", "--quiet", "gguf"], check=True)
from gguf import GGUFReader

reader = GGUFReader(MODEL)
tensors = {}
for tensor in reader.tensors:
    tensors[tensor.name] = tensor

outdir = pathlib.Path("/tmp/expert_slices")
outdir.mkdir(exist_ok=True)
manifest = {}
# NOTE: gguf-py presents tensors in reversed (numpy) order: the expert axis
# is FIRST, and each row is stored as raw quantization blocks.
for kind, base in (("gate", "blk.8.ffn_gate_exps.weight"),
                   ("up", "blk.8.ffn_up_exps.weight"),
                   ("down", "blk.8.ffn_down_exps.weight")):
    tensor = tensors[base]
    data = tensor.data  # memmap, no full copy
    for expert in EXPERTS:
        slc = data[expert]
        path = outdir / f"{kind}_e{expert}.bin"
        with open(path, "wb") as stream:
            stream.write(slc.tobytes())
        manifest[f"{kind}_e{expert}"] = {
            "tensor": base, "expert": expert, "shape": [int(v) for v in slc.shape],
            "dtype": str(tensor.tensor_type),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        }
for base in ("blk.8.ffn_gate_shexp.weight", "blk.8.ffn_up_shexp.weight",
             "blk.8.ffn_down_shexp.weight", "blk.8.ffn_gate_inp_shexp.weight"):
    tensor = tensors[base]
    path = outdir / (base.split(".")[-2] + ".bin")
    with open(path, "wb") as stream:
        stream.write(tensor.data.tobytes())
    manifest[base] = {"shape": list(tensor.data.shape),
                      "dtype": str(tensor.tensor_type),
                      "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
print("MANIFEST=" + json.dumps(manifest))
'''


def main() -> None:
    worker = REMOTE.replace("__EXPERTS__", repr(EXPERTS))
    compile(worker, "run_slices.py", "exec")
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-slices-") as temp:
        path = pathlib.Path(temp) / "run_slices.py"
        path.write_text(worker)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", "3600", "-f", str(path)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        manifest = None
        for line in proc.stdout.splitlines():
            if line.startswith("MANIFEST="):
                manifest = json.loads(line[len("MANIFEST="):])
        if proc.returncode or manifest is None:
            print(proc.stdout[-3000:])
            raise RuntimeError("slice export failed")
        dest = ROOT / "issue1_artifacts/expert_slices_l8"
        dest.mkdir(exist_ok=True)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        by_sha = {}
        for key, info in manifest.items():
            fname = (key + ".bin") if "." not in key else (key.split(".")[-2] + ".bin")
            by_sha.setdefault(fname, []).append((key, info["sha256"]))
        for fname, entries in sorted(by_sha.items()):
            remote = f"/tmp/expert_slices/{fname}"
            local = dest / fname
            cmd = CLI + ["download", "-s", SESSION, remote, str(local)]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, text=True, check=True)
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            assert all(digest == want for _, want in entries), (fname, digest)
            print(f"OK {fname} {local.stat().st_size} sha={digest[:16]}")
        print("downloaded to", dest)


if __name__ == "__main__":
    main()
