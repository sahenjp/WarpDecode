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
    if os.environ.get("WD_AB_DEBUG"):
        print("[ab-debug] keys=%s profile_type=%s" % (sorted(ident.keys()), type(ident.get("profile")).__name__),
              flush=True)
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
    default_mib = "8000" if plan in ("l4", "quick", "parity", "bisect") else "20000"
    base = {"WARODECODE_EXPERT_CACHE_MIB": os.environ.get("WD_EXPERT_CACHE_MIB", default_mib)}

    def merged(extra):
        env = dict(base)
        env.update(extra)
        return env

    yield ("warmup", merged({}))
    if plan == "bisect":
        # One Task J switch disabled at a time: the first case that returns to
        # 64/64 names the switch that changed the arithmetic.
        yield ("all_on", merged({}))
        yield ("off_skip_legacy", merged({"WARODECODE_HC_SKIP_LEGACY": "0"}))
        yield ("off_gamma_resident", merged({"WARODECODE_HC_GAMMA_RESIDENT": "0"}))
        yield ("off_inject_warp", merged({"WARODECODE_HC_INJECT_F32_WARP": "0"}))
        yield ("off_down_dp4a", merged({"WARODECODE_HC_Q8_DOWN_DP4A": "0"}))
        return
    if plan == "parity":
        # Smallest plan that still proves both directions: the Task J defaults
        # must reproduce the reference tokens, and the same run with every Task J
        # switch forced back to the pre-J order must produce the same tokens.
        for i in range(repeat):
            yield ("base-%d" % i, merged({}))
            yield ("all_off-%d" % i, merged({"WARODECODE_HC_SKIP_LEGACY": "0",
                                             "WARODECODE_HC_GAMMA_RESIDENT": "0",
                                             "WARODECODE_HC_INJECT_F32_WARP": "0",
                                             "WARODECODE_HC_Q8_DOWN_DP4A": "0"}))
        return
    if plan == "quick":
        for i in range(repeat):
            yield ("base-%d" % i, merged({}))
            yield ("kprofile-%d" % i, merged({"WARODECODE_HC_PROFILE": "1"}))
        return
    for i in range(repeat):
        # base_kprofile also proves the per-kernel event brackets did not move
        # the token timing (compare tok_s against base).
        yield ("base_kprofile-%d" % i, merged({"WARODECODE_HC_PROFILE": "1"}))
        yield ("base-%d" % i, merged({}))
        # Task J A/B switches, one at a time.
        yield ("no_skip_legacy-%d" % i, merged({"WARODECODE_HC_SKIP_LEGACY": "0"}))
        yield ("no_gamma_resident-%d" % i, merged({"WARODECODE_HC_GAMMA_RESIDENT": "0"}))
        yield ("no_inject_warp-%d" % i, merged({"WARODECODE_HC_INJECT_F32_WARP": "0"}))
        yield ("no_down_dp4a-%d" % i, merged({"WARODECODE_HC_Q8_DOWN_DP4A": "0"}))
        yield ("no_q8_0-%d" % i, merged({"WARODECODE_HC_Q8_0": "0"}))
        yield ("all_off-%d" % i, merged({"WARODECODE_HC_SKIP_LEGACY": "0",
                                         "WARODECODE_HC_GAMMA_RESIDENT": "0",
                                         "WARODECODE_HC_INJECT_F32_WARP": "0",
                                         "WARODECODE_HC_Q8_DOWN_DP4A": "0"}))


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
        prof = rec.get("profile")
        if isinstance(prof, dict) and prof.get("enabled"):
            for key in ("total_ms", "hc_mix_attn_ms", "hc_mix_ffn_ms", "attn_ms", "moe_ms"):
                if key in prof:
                    brief[key] = prof[key]
        else:
            brief["profile"] = "unavailable:%s" % type(prof).__name__
        hc = rec.get("hc_profile")
        if isinstance(hc, dict) and hc.get("sites"):
            for site in ("attn.region_total", "ffn.region_total", "attn.q8_down", "ffn.q8_down",
                         "attn.q8_inject", "ffn.q8_inject", "ffn.f16_norm"):
                if site in hc["sites"]:
                    brief["k_" + site] = hc["sites"][site]["ms_per_token"]
        print(json.dumps(brief), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
    print("AB_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
