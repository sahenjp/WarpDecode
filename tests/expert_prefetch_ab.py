#!/usr/bin/env python3
"""Task K: A/B the expert read-ahead pipeline on the Colab VM.

Runs the canonical 64-token greedy decode for a list of WARODECODE_* expert-cache
configurations with WARODECODE_PROFILE=1 and WARODECODE_EXPERT_STATS=1, checks
64/64 parity against the reference token file, and prints one compact JSON record
per case (prefixed WD_CASE) so the launcher can collect them without dragging the
whole log around.

Usage (on the VM):
    python3 tests/expert_prefetch_ab.py --plan a100 --engine /content/warodecode/build-sm80/warodecode
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MODEL = os.environ.get("WD_MODEL", "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
REF = os.environ.get("WD_REF", "/content/reference_tokens_64.json")
ENGINE = os.environ.get("WD_ENGINE", "/content/warodecode/build-sm80/warodecode")

# The stats fields the read-ahead and the wait decomposition add. Everything else
# (hits/misses/bytes) is unchanged by Task K.
STAT_KEYS = (
    "capacity_mib", "slots", "hits", "misses", "hit_rate", "bytes_copied", "gib_copied",
    "stage_ms", "acquire_ms", "stage_share", "host_ms_per_token", "io_pread_ms",
    "io_ring_wait_ms", "io_served", "io_fallbacks", "io_enabled", "io_ring", "io_threads",
    "io_prefetched", "prefetch_depth", "prefetch_slots", "prefetch_posted",
    "prefetch_dropped", "prefetch_used", "prefetch_delivered", "prefetch_failed",
    "prefetch_candidates", "prefetch_resident_skip", "prefetch_layer_skip",
    "prefetch_post_calls", "prefetch_post_refused",
    "intra_gap_ms", "layer_gap_ms", "intra_gap_per_token", "layer_gap_per_token",
    "acquire_per_token", "tokens", "distinct_pairs", "prev_token_reuse",
)
PROFILE_KEYS = ("total_ms", "sync_ms", "stage_sum_ms", "moe_ms", "attn_ms",
                "hc_mix_attn_ms", "hc_mix_ffn_ms", "head_ms")
DETAIL_KEYS = ("host_total_ms", "host_moe_ms", "host_attn_ms", "host_hc_mix_attn_ms",
               "host_hc_mix_ffn_ms", "calls_moe", "tokens")


def foreign_gpu_processes() -> list:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def last_identity(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith('{"engine":"warodecode-qwen38"'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def last_stats(text: str):
    for line in reversed(text.splitlines()):
        if '"expert_cache_stats"' in line:
            try:
                return json.loads(line)["expert_cache_stats"]
            except Exception:
                return None
    return None


def run_case(name: str, env_extra: dict, reference: list, generate: int,
             engine: str) -> dict:
    env = dict(os.environ)
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    env.update(env_extra)
    cmd = [engine, "--model", MODEL, "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(generate)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    wall = time.monotonic() - t0
    combined = proc.stdout + "\n" + proc.stderr
    ident = last_identity(proc.stdout)
    stats = last_stats(proc.stderr)
    rec = {"case": name, "env": env_extra, "wall_s": round(wall, 2),
           "exit": proc.returncode}
    if ident is None:
        rec["error"] = " || ".join(combined.strip().splitlines()[-6:])[-700:]
        return rec
    tokens = ident.get("token_ids", [])
    mismatch = next((i for i, (a, b) in enumerate(zip(reference, tokens)) if a != b), None)
    rec["matched_prefix"] = mismatch if mismatch is not None else len(tokens)
    rec["parity_ok"] = rec["matched_prefix"] == generate
    for key in ("tok_s", "seconds", "decode_seconds", "decode_tokens", "token_hash"):
        if key in ident:
            rec[key] = ident[key]
    profile = ident.get("profile") or {}
    rec["profile"] = {k: profile[k] for k in PROFILE_KEYS if k in profile}
    detail = ident.get("profile_detail") or {}
    rec["detail"] = {k: detail[k] for k in DETAIL_KEYS if k in detail}
    if stats:
        rec["stats"] = {k: stats[k] for k in STAT_KEYS if k in stats}
        if stats.get("stage_ms") and stats.get("gib_copied"):
            rec["stats"]["stage_gb_s"] = round(
                stats["gib_copied"] * 1024 ** 3 / (stats["stage_ms"] / 1000.0) / 1e9, 3)
    return rec


def plans(plan: str):
    """Yields (name, env). Every case except the warmup carries PROFILE=1 and
    EXPERT_STATS=1, so the A/B compares like with like."""
    profiled = {"WARODECODE_PROFILE": "1", "WARODECODE_EXPERT_STATS": "1"}

    def case(name, mib, extra):
        env = {"WARODECODE_EXPERT_CACHE_MIB": str(mib)}
        env.update(profiled)
        env.update(extra)
        return (name, env)

    sizes = {"a100": (20000, 1536), "l4": (8000, 1536)}[plan]
    big, small = sizes
    yield ("warmup", {"WARODECODE_EXPERT_CACHE_MIB": str(small),
                      "WARODECODE_EXPERT_IO_PREAD": "0"})
    for mib in (big, small):
        yield case("%s-c%d-off" % (plan, mib), mib, {"WARODECODE_EXPERT_PREFETCH": "0"})
        yield case("%s-c%d-pf1" % (plan, mib), mib,
                   {"WARODECODE_EXPERT_PREFETCH": "1",
                    "WARODECODE_EXPERT_PREFETCH_DEPTH": "1"})
        yield case("%s-c%d-pf2" % (plan, mib), mib,
                   {"WARODECODE_EXPERT_PREFETCH": "1",
                    "WARODECODE_EXPERT_PREFETCH_DEPTH": "2"})
        # Diagnostic: POST even when the prediction is already resident. Without
        # it a run on this model posts nothing (the previous token's list is the
        # resident set), so this is the case that proves the ring works end to end
        # on the GPU and prices the wasted reads.
        yield case("%s-c%d-pfall" % (plan, mib), mib,
                   {"WARODECODE_EXPERT_PREFETCH": "1",
                    "WARODECODE_EXPERT_PREFETCH_DEPTH": "1",
                    "WARODECODE_EXPERT_PREFETCH_ALL": "1"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="a100", choices=("a100", "l4"))
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--out", default="/content/prefetch_ab.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    foreign = foreign_gpu_processes()
    print("FOREIGN_GPU %s" % json.dumps(foreign), flush=True)

    reference = json.load(open(REF))["token_ids"][:args.generate]
    results = []
    for name, env_extra in plans(args.plan):
        if args.only and args.only not in name:
            continue
        rec = run_case(name, env_extra, reference, args.generate, args.engine)
        rec["foreign_gpu"] = foreign
        results.append(rec)
        print("WD_CASE " + json.dumps(rec), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
    print("WD_AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
