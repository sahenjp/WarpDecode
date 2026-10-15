#!/bin/bash
# Task R round 2: A/B the prefix-split expert-batched pass and the grid switch.
#
# Every case is the canonical 64-token greedy decode, so each one checks its own
# 64/64 against the reference token list. The instrumented repeats carry
# WARODECODE_PROFILE / _EXPERT_STATS / _IO_TIMING so the MoE ledger (device
# span, host span, pread, DMA, consumer stall) can be read for every config.
#
# Usage (on the VM): bash tests/moe_transfer_sweep_ab.sh
ENGINE=/content/warodecode/build-sm80/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
REF=/content/reference_tokens_64.json
LOG=/content/sweep.log
STATUS=/content/sweep_status.txt
PROBE=/content/sweep_probe.txt

: > "$LOG"
: > "$STATUS"

cat > /content/run_case.py <<'PY'
import json, os, pathlib, subprocess, sys, time

name, cache_mib, extra, generate, ref_path, engine, model = sys.argv[1:8]
env = dict(os.environ)
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if cache_mib != "-":
    env["WARODECODE_EXPERT_CACHE_MIB"] = cache_mib
for kv in extra.split(","):
    if kv:
        key, _, value = kv.partition("=")
        env[key] = value
cmd = [engine, "--model", model, "--ctx", "128", "--device", "0",
       "--start-token", "1", "--no-graph", "--no-autotune", "--generate", generate]
started = time.monotonic()
proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
wall = time.monotonic() - started
ident = None
for line in reversed(proc.stdout.splitlines()):
    if line.startswith('{"engine":"warodecode-qwen38"'):
        ident = json.loads(line)
        break
stats = None
for line in reversed(proc.stderr.splitlines()):
    if '"expert_cache_stats"' in line:
        stats = json.loads(line)["expert_cache_stats"]
        break
ref = json.loads(pathlib.Path(ref_path).read_text())["token_ids"]
tokens = (ident or {}).get("token_ids") or []
matched = 0
for a, b in zip(ref, tokens):
    if a != b:
        break
    matched += 1
profile = (ident or {}).get("profile") or {}
detail = (ident or {}).get("profile_detail") or {}
head = [l for l in proc.stderr.splitlines()
        if ("expert-cache " in l or "arch=qwen4exp" in l or "grid not taken" in l
            or "grid taken" in l or "DP4A" in l)]
record = {
    "case": name,
    "cache_mib": cache_mib,
    "extra": extra,
    "exit": proc.returncode,
    "wall_s": round(wall, 2),
    "tok_s": (ident or {}).get("tok_s"),
    "matched_prefix": matched,
    "parity_ok": matched == len(ref) and len(tokens) == len(ref),
    "tokens": tokens,
    "profile": {k: v for k, v in profile.items() if not isinstance(v, list)},
    "detail": {k: v for k, v in detail.items() if not isinstance(v, list)},
    "stats": stats,
    "log_lines": head[:8],
    "stderr_tail": proc.stderr[-1500:] if (proc.returncode != 0 or ident is None) else "",
}
print(json.dumps(record), flush=True)
PY

run_one() {
    echo "+ $1 cache=$2 $3" >> "$STATUS"
    python3 /content/run_case.py "$1" "$2" "$3" 64 "$REF" "$ENGINE" "$MODEL" >> "$LOG" 2>>/content/sweep_err.txt
}

{
    echo "=== round 2"
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv
} > "$PROBE" 2>&1

DIAG="WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1"
GRID="WARODECODE_MOE_FUSED_GRID=1"

# Unperturbed tok_s / parity pairs (no profiler), each followed by the
# instrumented twin that carries the MoE ledger.
run_one plain 20000 ""
run_one diag 20000 "$DIAG"
run_one grid 20000 "$GRID"
run_one diag-grid 20000 "$DIAG,$GRID"
run_one g2 20000 "$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
run_one diag-g2 20000 "$DIAG,$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
run_one g4 20000 "$GRID,WARODECODE_MOE_BATCH_GROUPS=4"
run_one diag-g4 20000 "$DIAG,$GRID,WARODECODE_MOE_BATCH_GROUPS=4"
run_one g2-only 20000 "WARODECODE_MOE_BATCH_GROUPS=2"
# Repeat the two candidates to separate the split's effect from run-to-run noise.
run_one grid-b 20000 "$GRID"
run_one g2-b 20000 "$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
run_one diag-g2-b 20000 "$DIAG,$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
# The small cache is where the transfer matters most (more misses per token).
run_one g2-c4096 4096 "$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
run_one diag-g2-c4096 4096 "$DIAG,$GRID,WARODECODE_MOE_BATCH_GROUPS=2"
run_one diag-grid-c4096 4096 "$DIAG,$GRID"

echo "SWEEP_DONE" >> "$STATUS"
