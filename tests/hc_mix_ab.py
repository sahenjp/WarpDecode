#!/usr/bin/env python3
"""Task J: engine A/B for the HC mix switches.

Runs the canonical 64-token greedy decode for a list of WARODECODE_HC_*
configurations, checks 64/64 parity against the reference token file, and
collects tok_s, the Task F stage profile and the Task J per-kernel HC profile.

Usage (on the VM):
  python3 tests/hc_mix_ab.py --plan l4 --repeat 2 --generate 64 \
      --out /content/taskj_ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

MODEL = os.environ.get("WD_MODEL", "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
REF = os.environ.get("WD_REF", "/content/reference_tokens_64.json")
ENGINE = os.environ.get("WD_ENGINE", "/content/warodecode/build-sm89/warodecode")

HC_LINE = re.compile(r"^\[hc-profile\]\s+(\S+)\s+([0-9.]+)\s+calls/token=([0-9.]+)")


def foreign_gpu_processes() -> list:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def parse_identity(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith('{"engine":"warodecode-qwen38"'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def parse_hc_profile(text: str):
    sites = {}
    tokens = None
    for line in text.splitlines():
        if not line.startswith("[hc-profile]"):
            continue
        if "tokens=" in line:
            m = re.search(r"tokens=(\d+)", line)
            if m:
                tokens = int(m.group(1))
            continue
        m = HC_LINE.match(line)
        if m:
            sites[m.group(1)] = {"ms_per_token": float(m.group(2)),
                                 "calls_per_token": float(m.group(3))}
    return {"tokens": tokens, "sites": sites} if sites else None


def run_case(name: str, env_extra: dict, reference: list, generate: int, profile: bool) -> dict:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update(env_extra)
    if profile:
        env["WARODECODE_PROFILE"] = "1"
    cmd = [ENGINE, "--model", MODEL, "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(generate)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = time.monotonic() - t0
    ident = parse_identity(proc.stdout)
    rec = {"case": name, "env": env_extra, "profile": profile, "wall_s": round(wall, 2),
           "exit": proc.returncode}
    if ident is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        rec["error"] = " || ".join(tail)[-800:]
        return rec
    tokens = ident.get("token_ids", [])
    mismatch = next((i for i, (a, b) in enumerate(zip(reference, tokens)) if a != b), None)
    rec["matched_prefix"] = mismatch if mismatch is not None else len(tokens)
    rec["parity_ok"] = rec["matched_prefix"] == min(generate, len(reference))
    for key in ("tok_s", "seconds", "decode_seconds", "decode_tokens", "prefill_tokens", "token_hash"):
        if key in ident:
            rec[key] = ident[key]
    if "profile" in ident:
        rec["profile"] = ident["profile"]
    if "profile_detail" in ident:
        detail = ident["profile_detail"]
        rec["profile_detail"] = {k: v for k, v in detail.items() if not isinstance(v, list)}
    hc = parse_hc_profile(proc.stderr)
    if hc:
        rec["hc_profile"] = hc
    return rec


def plans(plan: str, repeat: int):
    """Yield (name, env). The first entry is always a warmup (model page cache)."""
    if plan == "l4":
        base = {"WARODECODE_EXPERT_CACHE_MIB": "8000"}
    elif plan == "quick":
        base = {"WARODECODE_EXPERT_CACHE_MIB": "8000"}
    else:
        base = {"WARODECODE_EXPERT_CACHE_MIB": "20000"}

    def merged(extra):
        env = dict(base)
        env.update(extra)
        return env

    yield ("warmup", merged({}))
    if plan == "quick":
        for i in range(repeat):
            yield ("base-%d" % i, merged({}))
            yield ("no_skip_legacy-%d" % i, merged({"WARODECODE_HC_SKIP_LEGACY": "0"}))
            yield ("kprofile-%d" % i, merged({"WARODECODE_HC_PROFILE": "1"}))
        return
    for i in range(repeat):
        yield ("base-%d" % i, merged({}))
        yield ("base_offprofile-%d" % i, merged({}))
        yield ("base_kprofile-%d" % i, merged({"WARODECODE_HC_PROFILE": "1"}))
        # Task J A/B switches.
        yield ("no_skip_legacy-%d" % i, merged({"WARODECODE_HC_SKIP_LEGACY": "0"}))
        yield ("no_inject_f32-%d" % i, merged({"WARODECODE_HC_INJECT_F32": "0"}))
        yield ("no_legacy_q8attn-%d" % i, merged({"WARODECODE_HC_Q8_0_ATTN": "0"}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--out", default="/content/taskj_ab.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--plan", default="l4")
    args = ap.parse_args()

    foreign = foreign_gpu_processes()
    if foreign:
        print("WARNING: foreign GPU processes: %s" % foreign, flush=True)

    reference = json.load(open(REF))["token_ids"][:args.generate]
    results = []
    for name, env_extra in plans(args.plan, args.repeat):
        if args.only and args.only not in name:
            continue
        rec = run_case(name, env_extra, reference, args.generate, profile=True)
        rec["foreign_gpu"] = foreign
        results.append(rec)
        brief = {k: rec.get(k) for k in ("case", "tok_s", "matched_prefix", "parity_ok",
                                         "error", "wall_s") if k in rec}
        if rec.get("profile"):
            brief["hc_mix_attn_ms"] = rec["profile"].get("hc_mix_attn_ms")
            brief["hc_mix_ffn_ms"] = rec["profile"].get("hc_mix_ffn_ms")
            brief["moe_ms"] = rec["profile"].get("moe_ms")
        print(json.dumps(brief), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
    print("AB_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
