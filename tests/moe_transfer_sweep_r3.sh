#!/bin/bash
# Task R round 3: the prefix-split A/B after fixing the per-group staging table.
#
# Round 2 was polluted twice: three cases died with "cudaMalloc expert cache
# failed: out of memory" (another process held the A100), and every split case
# failed because all groups shared one pinned host pointer table. This script
# therefore waits for a free GPU before each case and records what it saw.
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


def foreign():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], text=True,
                         capture_output=True).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


# Wait for an idle GPU: a concurrent process both fails the 20 GiB cache
# allocation and slows the run, and a polluted case is worse than a missing one.
waited = 0
while waited < 300 and foreign():
    time.sleep(10)
    waited += 10
foreign_before = foreign()

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
    "case": name, "cache_mib": cache_mib, "extra": extra, "exit": proc.returncode,
    "wall_s": round(wall, 2), "tok_s": (ident or {}).get("tok_s"),
    "matched_prefix": matched,
    "parity_ok": matched == len(ref) and len(tokens) == len(ref),
    "tokens": tokens,
    "profile": {k: v for k, v in profile.items() if not isinstance(v, list)},
    "detail": {k: v for k, v in detail.items() if not isinstance(v, list)},
    "stats": stats, "log_lines": head[:8],
    "foreign_gpu": foreign_before, "waited_s": waited,
    "stderr_tail": proc.stderr[-1200:] if (proc.returncode != 0 or ident is None) else "",
}
print(json.dumps(record), flush=True)
PY

run_one() {
    echo "+ $1 cache=$2 $3" >> "$STATUS"
    python3 /content/run_case.py "$1" "$2" "$3" 64 "$REF" "$ENGINE" "$MODEL" >> "$LOG" 2>>/content/sweep_err.txt
}

{
    echo "=== round 3"
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv
} > "$PROBE" 2>&1

DIAG="WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1"

# The grid path is ON by default now, so "plain" is the batched pass with one
# group; "serial" turns it off to keep the pre-flip baseline comparable.
run_one plain 20000 ""
run_one serial 20000 "WARODECODE_MOE_FUSED_GRID=0"
run_one g2 20000 "WARODECODE_MOE_BATCH_GROUPS=2"
run_one g4 20000 "WARODECODE_MOE_BATCH_GROUPS=4"
run_one diag-serial 20000 "$DIAG,WARODECODE_MOE_FUSED_GRID=0"
run_one diag-plain 20000 "$DIAG"
run_one diag-g2 20000 "$DIAG,WARODECODE_MOE_BATCH_GROUPS=2"
run_one diag-g4 20000 "$DIAG,WARODECODE_MOE_BATCH_GROUPS=4"
run_one g2-b 20000 "WARODECODE_MOE_BATCH_GROUPS=2"
run_one g2-c1536 1536 "WARODECODE_MOE_BATCH_GROUPS=2"
run_one plain-c1536 1536 ""
run_one diag-g2-c1536 1536 "$DIAG,WARODECODE_MOE_BATCH_GROUPS=2"

echo "SWEEP_DONE" >> "$STATUS"
