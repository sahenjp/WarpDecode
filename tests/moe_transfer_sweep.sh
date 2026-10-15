#!/bin/bash
# Task R: attribute and reduce the MoE expert-transfer region on the Colab VM.
#
# Runs the canonical 64-token greedy decode once per configuration and appends
# one JSON record per case to /content/sweep.log, so the whole matrix is a single
# background job (the runner kills earlier jobs, and one job per session is the
# rule). The page cache is warmed first because every read path here reads the
# model from the page cache, so a cold cache would measure the disk instead.

ENGINE=/content/warodecode/build-sm80/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
REF=/content/reference_tokens_64.json
LOG=/content/sweep.log
STATUS=/content/sweep_status.txt
PROBE=/content/sweep_probe.txt

: > "$LOG"
: > "$STATUS"

SKIP_WARM=0
if [ -n "${WARODECODE_SWEEP_SKIP_WARM:-}" ]; then SKIP_WARM="$WARODECODE_SWEEP_SKIP_WARM"; fi
SKIP_BENCH=0
if [ -n "${WARODECODE_SWEEP_SKIP_BENCH:-}" ]; then SKIP_BENCH="$WARODECODE_SWEEP_SKIP_BENCH"; fi
echo "sweep script started pid=$$" >> "$STATUS"

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
    "detail": {k: v for k, v in ((ident or {}).get("profile_detail") or {}).items()
               if not isinstance(v, list)},
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
    echo "=== nvidia-smi"
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
    echo "=== cores"
    nproc
} > "$PROBE" 2>&1

if [ "$SKIP_WARM" != "1" ]; then
    cat "$MODEL" > /dev/null
fi
echo "warm done" >> "$PROBE"

# H2D ceiling at large chunks: does the ~2 MiB per-expert DMA sit at the link
# limit, or is there headroom that a larger transfer would take?
if [ "$SKIP_BENCH" != "1" ]; then
    bash /content/warodecode/tests/build_io_bench.sh >> "$PROBE" 2>&1 || true
    for chunk in 2 8 32; do
        echo "--- chunk ${chunk} MiB" >> "$PROBE"
        /content/warodecode/tests/expert_io_bench "$MODEL" 32 "$chunk" pin+h2d,pinpread,pinpread+h2d >> "$PROBE" 2>&1
    done
fi

run_one plain 20000 ""
run_one diag 20000 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1"
run_one diag-grid 20000 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1,WARODECODE_MOE_FUSED_GRID=1"
run_one st4 20000 "WARODECODE_EXPERT_IO_STREAMS=4"
run_one st4-grid 20000 "WARODECODE_MOE_FUSED_GRID=1,WARODECODE_EXPERT_IO_STREAMS=4"
run_one skip 20000 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_SKIP=1"
run_one skip-grid 20000 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_MOE_FUSED_GRID=1,WARODECODE_EXPERT_IO_SKIP=1"
run_one plain-4096 4096 ""
run_one diag-4096 4096 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1"
run_one diag-grid-4096 4096 "WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1,WARODECODE_MOE_FUSED_GRID=1"
run_one plain-2000 2000 ""
run_one plain-8000 8000 ""

echo "SWEEP_DONE" >> "$STATUS"
