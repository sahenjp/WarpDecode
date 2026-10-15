#!/usr/bin/env python3
"""Task S: kernel-level attribution of the MoE region on the real engine.

Task Q decomposed the MoE region by differencing whole runs and was left with
"expert kernel execution, no I/O, 5 launches/layer" = 64.5 ms/token on the A100
at 20000 MiB. The Task G microbenchmark measures the same five launches at
179 us/layer = 8.6 ms/token on the L4, so the two numbers cannot both be the
sum of the kernels. This harness runs the engine with WARODECODE_MOE_PROFILE=1,
which brackets every launch of the MoE region with its own event pair and prints
a per-site dev/host/gap line per token on stderr.

Every case is a separate engine process with the same greedy parity command
(64 tokens, ctx 128, no graph, no autotune), so the cases are comparable and
each one reports its own 64/64 check against the reference token list.

Usage (on the VM, from the extracted tree):
  python3 tests/moe_kernel_profile_ab.py --engine "$WD_ENGINE" \
      --model /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
      --ref /content/reference_tokens_64.json \
      --out /content/wd_q/moe_kernel_profile_1.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

GENERATE = 64
CTX = 128

PROFILE_LINE = re.compile(
    r"^\[moe-profile\]\s+(\S+)\s+dev=\s*([-0-9.]+)\s+host=\s*([-0-9.]+)"
    r"\s+gap=\s*([-0-9.]+)\s+calls/token=([0-9.]+)\s+recorded=(\d+)"
    r"\s+dropped=(\d+)\s+evfail=(\d+)")


def cases(cache_mib: int, matrix: str) -> list[dict]:
    """Diagnostic matrix. Each entry: name, extra environment, note."""
    base = {"WARODECODE_EXPERT_CACHE_MIB": str(cache_mib)}
    prof = dict(base, WARODECODE_MOE_PROFILE="1", WARODECODE_PROFILE="1",
                WARODECODE_EXPERT_STATS="1")
    off = dict(base, WARODECODE_ROUTER_F32_WARP="0",
               WARODECODE_ROUTER_TOPK_PARALLEL="0", WARODECODE_SHEXP_GATE_WARP="0")
    if matrix == "opt3":
        return [
            {"name": "base", "env": dict(off),
             "note": "all three Task S kernel switches off (the pre-task path)"},
            {"name": "opt", "env": dict(base),
             "note": "all three switches on (the new default tree)"},
            {"name": "prof-opt", "env": dict(prof),
             "note": "kernel profile with all three switches on"},
            {"name": "prof-opt-skip", "env": dict(prof, WARODECODE_EXPERT_IO_SKIP="1"),
             "note": "DIAGNOSTIC: no pread/DMA - the case Task Q called 64.5 ms"},
            {"name": "opt-g2", "env": dict(base, WARODECODE_MOE_BATCH_GROUPS="2"),
             "note": "expert pass split into 2 prefix groups (DMA overlap)"},
            {"name": "opt-g5", "env": dict(base, WARODECODE_MOE_BATCH_GROUPS="5"),
             "note": "expert pass split into 5 prefix groups"},
        ]
    if matrix == "opt2":
        return [
            {"name": "base", "env": dict(off),
             "note": "all three Task S kernel switches off (the pre-task path)"},
            {"name": "opt", "env": dict(base),
             "note": "all three switches on (the new default tree)"},
            {"name": "prof-opt", "env": dict(prof),
             "note": "kernel profile with all three switches on"},
            {"name": "prof-opt-skip", "env": dict(prof, WARODECODE_EXPERT_IO_SKIP="1"),
             "note": "DIAGNOSTIC: no pread/DMA - the case Task Q called 64.5 ms"},
            {"name": "prof-opt-topk0", "env": dict(prof, WARODECODE_ROUTER_TOPK_PARALLEL="0"),
             "note": "all but the parallel router top-k"},
            {"name": "prof-opt-gate0", "env": dict(prof, WARODECODE_SHEXP_GATE_WARP="0"),
             "note": "all but the warp-parallel shared-expert gate dot"},
        ]
    if matrix == "opt":
        return [
            {"name": "warp0", "env": dict(base, WARODECODE_ROUTER_F32_WARP="0"),
             "note": "one-thread-per-row router GEMV (the pre-task baseline)"},
            {"name": "warp1", "env": dict(base, WARODECODE_ROUTER_F32_WARP="1"),
             "note": "warp-parallel router GEMV"},
            {"name": "prof-warp0", "env": dict(prof, WARODECODE_ROUTER_F32_WARP="0"),
             "note": "kernel profile of the baseline router path"},
            {"name": "prof-warp1", "env": dict(prof, WARODECODE_ROUTER_F32_WARP="1"),
             "note": "kernel profile with the warp-parallel router GEMV"},
            {"name": "prof-warp1-skip", "env": dict(prof, WARODECODE_ROUTER_F32_WARP="1",
                                                    WARODECODE_EXPERT_IO_SKIP="1"),
             "note": "DIAGNOSTIC: same with no pread/DMA (parity breaks)"},
        ]
    if matrix == "profile":
        return [
            {"name": "plain", "env": dict(base),
             "note": "unperturbed tok_s / parity reference (nothing instrumented)"},
            {"name": "prof-grid", "env": dict(prof),
             "note": "MoE kernel profile, expert-batched grid (the default tree)"},
            {"name": "prof-loop", "env": dict(prof, WARODECODE_MOE_FUSED_GRID="0"),
             "note": "MoE kernel profile, per-expert launch loop"},
            {"name": "prof-skipgrid", "env": dict(prof, WARODECODE_EXPERT_IO_SKIP="1"),
             "note": "DIAGNOSTIC: grid with no pread/DMA - the 64.5 ms case (parity breaks)"},
        ]
    raise SystemExit("unknown matrix %r" % matrix)


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


def parse_moe_profile(stderr: str) -> dict:
    """Pull the [moe-profile] table out of stderr.

    The profiler prints one summary at process exit, so the last block wins. A
    line is "<site> dev=<ms/token> host=<ms/token> gap=<host-dev> calls=<n>".
    """
    header = None
    sites: dict[str, dict] = {}
    for line in stderr.splitlines():
        if line.startswith("[moe-profile] tokens="):
            header = line
            continue
        m = PROFILE_LINE.match(line)
        if not m:
            continue
        sites[m.group(1)] = {
            "dev_ms": float(m.group(2)),
            "host_ms": float(m.group(3)),
            "gap_ms": float(m.group(4)),
            "calls_per_token": float(m.group(5)),
            "recorded": int(m.group(6)),
            "dropped": int(m.group(7)),
            "evfail": int(m.group(8)),
        }
    return {"header": header, "sites": sites}


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
        "moe_sites": parse_moe_profile(proc.stderr),
        "moe_profile_raw": [ln for ln in proc.stderr.splitlines() if ln.startswith("[moe-profile")],
        "stderr_tail": proc.stderr[-4000:] if proc.returncode != 0 else "",
    }
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-mib", type=int, default=20000)
    ap.add_argument("--matrix", default="profile", choices=("profile", "opt", "opt2", "opt3"))
    ap.add_argument("--skip-warm", action="store_true")
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
    for case in cases(args.cache_mib, args.matrix):
        print("+ case %s" % case["name"], flush=True)
        record = run_case(args.engine, args.model, ref, case)
        records.append(record)
        st = record["stats"]
        print("  tok_s=%s parity=%s/%s moe_ms=%s stats_stage=%s prof_sites=%d"
              % (record["tok_s"], record["matched_prefix"], len(ref),
                 (record["profile"] or {}).get("moe_ms"), st.get("stage_ms"),
                 len(record["moe_sites"]["sites"])), flush=True)
        sites = record["moe_sites"]["sites"]
        for name in sorted(sites, key=lambda n: -sites[n]["dev_ms"])[:8]:
            s = sites[name]
            print("      %-20s dev=%8.3f host=%8.3f gap=%8.3f calls=%s rec=%d drop=%d evfail=%d"
                  % (name, s["dev_ms"], s["host_ms"], s["gap_ms"], s["calls_per_token"],
                     s["recorded"], s["dropped"], s["evfail"]), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
    print("+ wrote %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
