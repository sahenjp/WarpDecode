#!/usr/bin/env python3
"""Host-side exact (float64) MoE reference for layer 8, position 39.

Dequantizes the downloaded L8 expert slices exactly per the pinned ggml spec
(IQ2_S / IQ4_NL / Q8_0, tables verified byte-identical against ggml-common.h
and Warp's iq2s_grid.cuh) and evaluates the routed + shared MoE path in
float64 on the exact CPU router input. Compares against:
  - CPU ffn_moe_out-derived routed output and ffn_shexp_gated-8 (oracle check)
  - Warp's MoE output under exact inputs (from position39_colab_moe_exec.json)

Decides whether the 0.011 MoE same-input error is input-F16-rounding
(reference ~= CPU) or GEMV arithmetic (reference ~= Warp, both off CPU).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = '2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60'
SLICE_DIR = ROOT / "issue1_artifacts/expert_slices_l8"

IQ4NL_LUT = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float64)


def load_grid() -> np.ndarray:
    text = (ROOT / "src/iq2s_grid.cuh").read_text()
    vals = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', text)]
    assert len(vals) == 1024, len(vals)
    grid = np.zeros((1024, 8), dtype=np.float64)
    for i, v in enumerate(vals):
        for lane in range(8):
            grid[i, lane] = (v >> (lane * 8)) & 0xFF
    return grid


GRID = load_grid()


def f16_to_f64(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype='<f2').astype(np.float64)


def dequant_q8_0(buf: bytes, rows: int, cols: int) -> np.ndarray:
    assert cols % 32 == 0
    out = np.zeros((rows, cols))
    off = 0
    for r in range(rows):
        for b in range(cols // 32):
            d = f16_to_f64(buf[off:off + 2])[0]
            qs = np.frombuffer(buf[off + 2:off + 34], dtype=np.int8).astype(np.float64)
            out[r, b * 32:(b + 1) * 32] = d * qs
            off += 34
    assert off == len(buf), (off, len(buf))
    return out


def dequant_iq4_nl(buf: bytes, rows: int, cols: int) -> np.ndarray:
    assert cols % 32 == 0
    out = np.zeros((rows, cols))
    off = 0
    for r in range(rows):
        for b in range(cols // 32):
            d = f16_to_f64(buf[off:off + 2])[0]
            qs = np.frombuffer(buf[off + 2:off + 18], dtype=np.uint8)
            lo = IQ4NL_LUT[qs & 0xF]
            hi = IQ4NL_LUT[qs >> 4]
            out[r, b * 32:b * 32 + 16] = d * lo
            out[r, b * 32 + 16:(b + 1) * 32] = d * hi
            off += 18
    assert off == len(buf), (off, len(buf))
    return out


def dequant_iq2_s(buf: bytes, rows: int, cols: int) -> np.ndarray:
    assert cols % 256 == 0
    out = np.zeros((rows, cols))
    off = 0
    for r in range(rows):
        for sb in range(cols // 256):
            d = f16_to_f64(buf[off:off + 2])[0]
            qs = np.frombuffer(buf[off + 2:off + 66], dtype=np.uint8)
            qh = np.frombuffer(buf[off + 66:off + 74], dtype=np.uint8)
            sc = np.frombuffer(buf[off + 74:off + 82], dtype=np.uint8)
            for ib in range(8):
                s_lo = sc[ib] & 0xF
                s_hi = sc[ib] >> 4
                for g in range(4):
                    scale4 = s_lo if g < 2 else s_hi
                    dl = d * (0.5 + scale4) * 0.25
                    idx = int(qs[ib * 4 + g]) | ((int(qh[ib]) << (8 - 2 * g)) & 0x300)
                    grid = GRID[idx]
                    signs = qs[32 + ib * 4 + g]
                    for lane in range(8):
                        v = dl * grid[lane]
                        if (signs >> lane) & 1:
                            v = -v
                        out[r, sb * 256 + ib * 32 + g * 8 + lane] = v
            off += 82
    assert off == len(buf), (off, len(buf))
    return out


def read_cpu_vec(name: str, occurrence: int = 0) -> np.ndarray:
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
                return np.frombuffer(raw[off:off + vc * 4], dtype='<f4').astype(np.float64)
            seen += 1
        off += vc * 4
    raise RuntimeError(f"{name} occ {occurrence} missing")


def nrmse(ref: np.ndarray, got: np.ndarray) -> dict:
    d = got - ref
    rmse = math.sqrt(float(np.mean(d * d)))
    ref_rms = math.sqrt(float(np.mean(ref * ref)))
    return {"nrmse": rmse / max(ref_rms, 1e-30), "rmse": rmse,
            "max_abs": float(np.max(np.abs(d)))}


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")
    manifest = json.loads((SLICE_DIR / "manifest.json").read_text())
    experts = [66, 399, 127, 365, 224, 68, 500, 237, 216, 426]
    weights = np.array([float(v) for v in read_cpu_vec('pos39.ffn_moe_weights_norm-8')],
                       dtype=np.float64)
    x = read_cpu_vec('pos39.hc_mixed-8', 1)  # exact F32 router input
    x_half = x.astype('<f2').astype(np.float64)  # production input precision

    def run_with(xin: np.ndarray) -> np.ndarray:
        acc = np.zeros(2560)
        for j, e in enumerate(experts):
            gate = dequant_iq2_s((SLICE_DIR / f"gate_e{e}.bin").read_bytes(), 640, 2560)
            up = dequant_iq2_s((SLICE_DIR / f"up_e{e}.bin").read_bytes(), 640, 2560)
            down = dequant_iq4_nl((SLICE_DIR / f"down_e{e}.bin").read_bytes(), 2560, 640)
            g = gate @ xin
            u = up @ xin
            act = (g / (1.0 + np.exp(-g))) * u
            acc += weights[j] * (down @ act)
        return acc

    routed_f32 = run_with(x)
    routed_half = run_with(x_half)
    cpu_routed = (read_cpu_vec('pos39.ffn_out-8')
                  - read_cpu_vec('pos39.ffn_shexp_gated-8'))
    warp_exec = json.load(open(ROOT / "issue1_artifacts/position39_colab_moe_exec.json"))
    warp_row = next(r for r in warp_exec["rows"] if r.get("name") == "res_hc_forced")
    report = {
        "host_f32_input_vs_cpu": nrmse(cpu_routed, routed_f32),
        "host_half_input_vs_cpu": nrmse(cpu_routed, routed_half),
        "host_f32_vs_half": nrmse(routed_half, routed_f32),
        "warp_reported_nrmse": warp_row["ffn_out_vs_cpu"]["nrmse"],
    }
    print(json.dumps(report, indent=2))
    (ROOT / "issue1_artifacts/position39_host_exact_moe.json").write_text(
        json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
