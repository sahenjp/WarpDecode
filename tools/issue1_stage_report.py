#!/usr/bin/env python3
"""Print the verified stage-by-stage error table (CPU oracle vs warp trace).

Pairs are the ones hand-checked against the real traces; duplicates such as
pos0.hc_mixed-<L> carry the HC attn mix first and the HC ffn mix second, so an
explicit occurrence index is part of the table.

Usage:
  python3 tools/issue1_stage_report.py --warp issue1_artifacts/warp_pos0_experts.bin \
      --cpu issue1_artifacts/cpu_pos0_internals.bin --position 0
"""
from __future__ import annotations

import argparse
import pathlib
import struct

# (label, cpu name template, warp name template, cpu occurrence)
PAIRS = [
    ("input residual",      "hc_init",                "layer{L}.input_residual", 0),
    ("hc_attn norm",        "hc_norm-{L}",            "layer{L}.hc_attn_norm", 0),
    ("hc_attn gate",        "hc_gate-{L}",            "layer{L}.hc_attn_gate", 0),
    ("hc_attn mix",         "hc_mixed-{L}",           "layer{L}.hc_attn_mix", 0),
    ("hc_attn inject",      "hc_inject-{L}",          "layer{L}.hc_attn_inject", 0),
    ("gdn qkv",             "linear_attn_qkv_mixed-{L}", "layer{L}.gdn.qkv_full", 0),
    ("gdn z",               "z-{L}",                  "layer{L}.gdn.z_full", 0),
    ("gdn alpha",           "alpha-{L}",              "layer{L}.gdn.alpha_full", 0),
    ("gdn beta_sigmoid",    "beta_sigmoid-{L}",       "layer{L}.gdn.beta_full", 0),
    ("gdn gate",            "gate-{L}",               "layer{L}.gdn.gate_full", 0),
    ("gdn core",            "gdn_core_output-{L}",    "layer{L}.gdn.core_output_full", 0),
    ("gdn norm_out",        "final_output-{L}",       "layer{L}.gdn.norm_output_full", 0),
    ("attention/linear",    "linear_attn_out-{L}",    "layer{L}.attention_output", 0),
    ("hc combine resid",    "hc_combine-{L}",         "layer{L}.after_attention_residual", 0),
    ("hc_ffn norm",         "hc_norm-{L}",            "layer{L}.hc_ffn_norm", 1),
    ("hc_ffn gate",         "hc_gate-{L}",            "layer{L}.hc_ffn_gate", 1),
    ("hc_ffn mix",          "hc_mixed-{L}",           "layer{L}.hc_ffn_mix", 1),
    ("hc_ffn inject",       "hc_inject-{L}",          "layer{L}.hc_ffn_inject", 1),
    ("moe router logits",   "ffn_moe_logits-{L}",     "layer{L}.ffn.router_logits", 0),
    ("moe weights_norm",    "ffn_moe_weights_norm-{L}", "layer{L}.ffn.expert_weights", 0),
    ("moe out",             "ffn_moe_out-{L}",        "layer{L}.ffn.ffn_moe_out", 0),
    ("shared expert",       "ffn_shexp-{L}",          "layer{L}.ffn_shexp", 0),
    ("ffn_out",             "ffn_out-{L}",            "layer{L}.ffn_output", 0),
    ("layer output",        "l_last-{L}",             "layer{L}.output", 0),
]


def parse(path: pathlib.Path) -> dict[str, list[list[float]]]:
    raw = path.read_bytes()
    if raw[:8] != b"WDPTRC1\x00":
        raise SystemExit("bad trace magic: %s" % path)
    _, count = struct.unpack_from("<II", raw, 8)
    out: dict[str, list[list[float]]] = {}
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from("<IQ", raw, off)
        off += 12
        name = raw[off:off + ns].decode()
        off += ns
        out.setdefault(name, []).append(struct.unpack_from("<%df" % vc, raw, off))
        off += vc * 4
    return out


def nrmse(ref: list[float], got: list[float]) -> float | None:
    if len(ref) != len(got) or not ref:
        return None
    err2 = sum((b - a) ** 2 for a, b in zip(ref, got))
    ref2 = sum(a * a for a in ref)
    return (err2 / len(ref)) ** 0.5 / max((ref2 / len(ref)) ** 0.5, 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--layers", default="0,1,2,3,4,5,7,8,14,24,32,47")
    ap.add_argument("--threshold", type=float, default=1e-2)
    args = ap.parse_args()

    cpu = parse(pathlib.Path(args.cpu))
    warp = parse(pathlib.Path(args.warp))
    layers = [int(x) for x in args.layers.split(",")]
    print("cpu vectors: %d, warp vectors: %d" % (len(cpu), len(warp)))
    worst = []
    for layer in layers:
        print("\n--- layer %d ---" % layer)
        for label, cpat, wpat, occ in PAIRS:
            cname = "pos%d.%s" % (args.position, cpat.format(L=layer))
            wname = "pos%d.%s" % (args.position, wpat.format(L=layer))
            cvals = cpu.get(cname)
            wvals = warp.get(wname)
            if cvals is None or wvals is None or len(cvals) <= occ or not wvals:
                continue
            v = nrmse(cvals[occ], wvals[0])
            if v is None:
                continue
            flag = " <== over threshold" if v > args.threshold else ""
            print("  %-20s %.5f%s" % (label, v, flag))
            worst.append((v, layer, label))
    worst.sort(reverse=True)
    print("\nworst stages:")
    for v, layer, label in worst[:8]:
        print("  L%-2d %-20s %.5f" % (layer, label, v))


if __name__ == "__main__":
    main()
