#!/bin/bash
# Task R round 4: the fixed cost of the per-layer host<->device round trip.
#
# The router path (q38_select_experts*) does a D2H of the selected ids plus a
# cudaStreamSynchronize() once per MoE layer, 48 times per token. Round 3 showed
# the MoE region does not shrink when the consumer's DMA stall is removed, so
# the remaining suspect is that fixed cost. tests/host_sync_latency.cu measures
# it directly, and the two SKIP cases show whether the I/O-free MoE floor moves
# with the expert group count at all.
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


waited = 0
while waited < 300 and foreign():
    time.sleep(10)
    waited += 10
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
record = {
    "case": name, "cache_mib": cache_mib, "extra": extra, "exit": proc.returncode,
    "wall_s": round(wall, 2), "tok_s": (ident or {}).get("tok_s"),
    "matched_prefix": matched,
    "parity_ok": matched == len(ref) and len(tokens) == len(ref),
    "tokens": tokens,
    "profile": {k: v for k, v in profile.items() if not isinstance(v, list)},
    "detail": {k: v for k, v in detail.items() if not isinstance(v, list)},
    "stats": stats, "foreign_gpu": foreign(), "waited_s": waited,
    "stderr_tail": proc.stderr[-1200:] if (proc.returncode != 0 or ident is None) else "",
}
print(json.dumps(record), flush=True)
PY

run_one() {
    echo "+ $1 cache=$2 $3" >> "$STATUS"
    python3 /content/run_case.py "$1" "$2" "$3" 64 "$REF" "$ENGINE" "$MODEL" >> "$LOG" 2>>/content/sweep_err.txt
}

{
    echo "=== round 4: host<->device round-trip cost"
    nvcc -O2 -std=c++17 -arch=sm_80 /content/warodecode/tests/host_sync_latency.cu -o /content/host_sync_latency 2>&1
    /content/host_sync_latency 3000
    echo "=== round 4 end"
} > "$PROBE" 2>&1

DIAG="WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1,WARODECODE_EXPERT_IO_TIMING=1"
SKIP="WARODECODE_EXPERT_IO_SKIP=1"
run_one plain 20000 ""
run_one skip-grid 20000 "$DIAG,$SKIP"
run_one skip-g2 20000 "$DIAG,$SKIP,WARODECODE_MOE_BATCH_GROUPS=2"
run_one skip-g8 20000 "$DIAG,$SKIP,WARODECODE_MOE_BATCH_GROUPS=8"
run_one diag-g8 20000 "$DIAG,WARODECODE_MOE_BATCH_GROUPS=8"

echo "SWEEP_DONE" >> "$STATUS"
