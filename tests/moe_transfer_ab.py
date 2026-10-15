#!/usr/bin/env python3
"""Task Q: attribute the MoE region of one decode token on the real engine.

Every case is a separate engine process with the same greedy parity command
(64 tokens, ctx 128, no graph, no autotune), so the cases are directly
comparable and each one reports its own 64/64 check against the reference token
list.

Two independent instrumentations are collected per case:
  * the engine's own stdout JSON: tok_s plus the Task F per-stage CUDA-event
    profile (moe_ms and friends) and the emitted token ids;
  * the expert cache's stderr JSON (WARODECODE_EXPERT_STATS=1): hit rate, bytes
    copied, pread ms, and the Task Q copy-path timing.

WARODECODE_PROFILE / _STATS / _IO_TIMING all perturb the pipeline a little, so
the "plain" case carries the unperturbed tok_s.

Usage (on the VM, from the extracted tree):
  python3 tests/moe_transfer_ab.py --engine "$WD_ENGINE" --model "$MODEL" \
      --ref /content/reference_tokens_64.json --out /content/wd_q/moe_q1.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

GENERATE = 64
CTX = 128


def cases_q3(cache_mib: int) -> list[dict]:
    """Task Q round 3: the clean A/B of the implemented switches.

    Every entry runs the same greedy 64-token command with no profiler, so tok_s
    is directly comparable; the instrumented entries at the end exist only to
    show where the time went and to re-check parity under eviction.
    """
    base = {"WARODECODE_EXPERT_CACHE_MIB": str(cache_mib)}
    return [
        {"name": "plain", "env": dict(base), "note": "baseline"},
        {"name": "st2", "env": dict(base, WARODECODE_EXPERT_IO_STREAMS="2"),
         "note": "DMA spread over 2 copy streams"},
        {"name": "st4", "env": dict(base, WARODECODE_EXPERT_IO_STREAMS="4"),
         "note": "DMA spread over 4 copy streams"},
        {"name": "st8", "env": dict(base, WARODECODE_EXPERT_IO_STREAMS="8"),
         "note": "DMA spread over 8 copy streams"},
        {"name": "th8", "env": dict(base, WARODECODE_EXPERT_IO_THREADS="8"),
         "note": "8 pread workers"},
        {"name": "st4th8", "env": dict(base, WARODECODE_EXPERT_IO_STREAMS="4",
                                       WARODECODE_EXPERT_IO_THREADS="8"),
         "note": "both"},
        {"name": "grid", "env": dict(base, WARODECODE_MOE_FUSED_GRID="1"),
         "note": "expert-batched grid (recommended: parks 60 -> 5 launches/layer)"},
        {"name": "st4grid", "env": dict(base, WARODECODE_MOE_FUSED_GRID="1",
                                        WARODECODE_EXPERT_IO_STREAMS="4"),
         "note": "grid + 4 copy streams"},
        {"name": "diag", "env": dict(base, WARODECODE_PROFILE="1",
                                     WARODECODE_EXPERT_STATS="1",
                                     WARODECODE_EXPERT_IO_TIMING="1"),
         "note": "instrumented baseline"},
        {"name": "diag-st4", "env": dict(base, WARODECODE_PROFILE="1",
                                         WARODECODE_EXPERT_STATS="1",
                                         WARODECODE_EXPERT_IO_TIMING="1",
                                         WARODECODE_EXPERT_IO_STREAMS="4"),
         "note": "instrumented, 4 copy streams"},
        {"name": "diag-c1536", "env": {"WARODECODE_EXPERT_CACHE_MIB": "1536",
                                       "WARODECODE_PROFILE": "1",
                                       "WARODECODE_EXPERT_STATS": "1"},
         "note": "eviction-heavy parity check (new retire path)"},
        {"name": "st4-c1536", "env": {"WARODECODE_EXPERT_CACHE_MIB": "1536",
                                      "WARODECODE_EXPERT_IO_STREAMS": "4"},
         "note": "eviction-heavy parity check with 4 copy streams"},
    ]


def cases(cache_mib: int) -> list[dict]:
    """Diagnostic matrix. Each entry: name, extra environment, note."""
    base = {"WARODECODE_EXPERT_CACHE_MIB": str(cache_mib)}
    diag = dict(base, WARODECODE_PROFILE="1", WARODECODE_EXPERT_STATS="1")
    out = [
        {"name": "plain", "env": dict(base),
         "note": "unperturbed tok_s / parity reference"},
        {"name": "base", "env": dict(diag, WARODECODE_EXPERT_IO_TIMING="1"),
         "note": "stage profile + cache stats + DMA/wait events"},
        {"name": "base-notiming", "env": dict(diag),
         "note": "same without the extra event pairs"},
        {"name": "skip-fill", "env": dict(diag, WARODECODE_EXPERT_IO_SKIP="1"),
         "note": "DIAGNOSTIC: no pread and no DMA at all (parity must break)"},
        {"name": "nodma", "env": dict(diag, WARODECODE_EXPERT_IO_NODMA="1"),
         "note": "DIAGNOSTIC: pread runs, DMA skipped (parity must break)"},
        {"name": "pageable", "env": dict(diag, WARODECODE_EXPERT_IO_PREAD="0"),
         "note": "pre-Task-H path: pageable cudaMemcpyAsync from the mmap"},
        {"name": "no-copy-stream", "env": dict(diag, WARODECODE_EXPERT_COPY_STREAM="0"),
         "note": "DMA ordered on the decode stream instead of a private one"},
        {"name": "dp4a", "env": dict(diag, WARODECODE_DP4A="1"),
         "note": "DP4A GEMV kernels"},
        {"name": "dp4a-grid", "env": dict(diag, WARODECODE_DP4A="1",
                                          WARODECODE_MOE_FUSED_GRID="1"),
         "note": "DP4A + one expert-batched grid per layer (5 launches/layer)"},
        {"name": "skip-grid", "env": dict(diag, WARODECODE_DP4A="1",
                                          WARODECODE_MOE_FUSED_GRID="1",
                                          WARODECODE_EXPERT_IO_SKIP="1"),
         "note": "DIAGNOSTIC: no I/O and 5 launches/layer = the compute floor"},
        {"name": "threads8", "env": dict(diag, WARODECODE_EXPERT_IO_THREADS="8"),
         "note": "more pread workers"},
        {"name": "ring16", "env": dict(diag, WARODECODE_EXPERT_IO_RING="16"),
         "note": "deeper pinned ring"},
        # Shadow comparison: which replacement policy would have hit more often on
        # the identical access trace. No profiler, so the trace is the plain one.
        {"name": "shadow", "env": dict(base, WARODECODE_EXPERT_STATS="1",
                                       WARODECODE_EXPERT_SHADOW="9002,12000,16000",
                                       WARODECODE_EXPERT_SHADOW_FREQ="lru"),
         "note": "LRU vs exact-LFU hit rate on the same trace"},
    ]
    return out


def parse_stats(stderr: str) -> dict:
    for line in reversed(stderr.splitlines()):
        if line.startswith('{"expert_cache_stats"'):
            try:
                return json.loads(line)["expert_cache_stats"]
            except Exception:
                return {"parse_error": True}
    return {}


def parse_engine(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith('{"engine":"warodecode-qwen38"'):
            try:
                return json.loads(line)
            except Exception:
                return {}
    return {}


def run_case(engine: str, model: str, ref: list[int], case: dict) -> dict:
    env = dict(os.environ)
    env.update(case["env"])
    cmd = [engine, "--model", model, "--ctx", str(CTX), "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(GENERATE)]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.monotonic() - started
    engine_json = parse_engine(proc.stdout)
    stats = parse_stats(proc.stderr)
    tokens = engine_json.get("token_ids") or []
    matched = 0
    for a, b in zip(ref, tokens):
        if a != b:
            break
        matched += 1
    profile = engine_json.get("profile") or {}
    detail = engine_json.get("profile_detail") or {}
    record = {
        "case": case["name"],
        "note": case["note"],
        "env": case["env"],
        "exit": proc.returncode,
        "wall_s": round(wall, 2),
        "tok_s": engine_json.get("tok_s"),
        "decode_seconds": engine_json.get("decode_seconds"),
        "matched_prefix": matched,
        "parity_ok": matched == len(ref) and len(tokens) == len(ref),
        "tokens": tokens,
        "profile": {k: v for k, v in profile.items() if not isinstance(v, list)},
        "stats": stats,
        "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else "",
    }
    # Per-layer host/device means are the only lists worth keeping.
    for key in ("layer_device_ms", "layer_host_ms"):
        values = detail.get(key)
        if isinstance(values, list):
            live = [v for v in values if v]
            record["detail_" + key + "_mean"] = round(sum(live) / len(live), 4) if live else None
    for key in ("host_moe_ms", "host_attn_ms", "host_hc_mix_attn_ms", "host_hc_mix_ffn_ms",
                "host_total_ms", "host_sync_ms", "host_ple_ms", "host_other_ms"):
        if key in detail:
            record["detail_" + key] = detail[key]
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-mib", type=int, default=20000)
    ap.add_argument("--extra-cache-mib", type=int, default=0,
                    help="run the diagnostic matrix for a second cache size too")
    ap.add_argument("--extra-case-names", default="plain,base",
                    help="which case names to repeat at --extra-cache-mib")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--matrix", default="default", choices=("default", "q3"))
    args = ap.parse_args()

    if not args.skip_warm:
        print("+ warming the page cache (cat model > /dev/null)", flush=True)
        with open(args.model, "rb") as fh, open(os.devnull, "wb") as devnull:
            while True:
                block = fh.read(1 << 24)
                if not block:
                    break
                devnull.write(block)
        print("+ page cache warm", flush=True)

    ref = json.loads(pathlib.Path(args.ref).read_text())["token_ids"]
    records = []
    if args.matrix == "q3":
        matrix = cases_q3(args.cache_mib)
    else:
        matrix = cases(args.cache_mib)
    if args.extra_cache_mib:
        wanted = [n for n in args.extra_case_names.split(",") if n]
        matrix += [dict(c, name=c["name"] + "-c%d" % args.extra_cache_mib,
                        env=dict(c["env"], WARODECODE_EXPERT_CACHE_MIB=str(args.extra_cache_mib)))
                   for c in cases(args.extra_cache_mib) if c["name"] in wanted]
    for case in matrix:
        print("+ case %s" % case["name"], flush=True)
        record = run_case(args.engine, args.model, ref, case)
        records.append(record)
        st = record["stats"]
        print("  tok_s=%s parity=%s/%s moe_ms=%s stage_ms=%s pread_ms=%s hits=%s/%s dma=%s stall=%s"
              % (record["tok_s"], record["matched_prefix"], len(ref),
                 (record["profile"] or {}).get("moe_ms"), st.get("stage_ms"),
                 st.get("io_pread_ms"), st.get("hits"), st.get("accesses"),
                 st.get("io_dma_ms"), st.get("io_dma_stall_ms")), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
    print("+ wrote %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
