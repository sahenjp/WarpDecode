#!/bin/bash
# Task T: the sync/other A/B matrix.
#
# One job, five configurations, one JSON record per case on stdout. The baseline
# and the two candidate switches must all keep parity 64/64; the io-skip case is
# the no-transfer reference and is expected to break parity by design.
set -u

ENGINE=/content/warodecode/build-sm80/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
REF=/content/reference_tokens_64.json
LOG=/content/sync_sweep.log
STATUS=/content/sync_sweep_status.txt
PROBE=/content/sync_sweep_probe.txt

: > "$LOG"
: > "$STATUS"

cat > /content/sync_case.py <<'PY'
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
moe_sites = None
for line in reversed(proc.stderr.splitlines()):
    if '"moe_sites"' in line:
        try:
            moe_sites = json.loads(line)["moe_sites"]
        except Exception:
            moe_sites = {"raw": line[-400:]}
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
            or "grid taken" in l or "DP4A" in l or "select" in l.lower())]
record = {
    "case": name, "cache_mib": cache_mib, "extra": extra, "exit": proc.returncode,
    "wall_s": round(wall, 2), "tok_s": (ident or {}).get("tok_s"),
    "matched_prefix": matched,
    "parity_ok": matched == len(ref) and len(tokens) == len(ref),
    "tokens": tokens,
    "profile": {k: v for k, v in profile.items() if not isinstance(v, list)},
    "detail": {k: v for k, v in detail.items() if not isinstance(v, list)},
    "stats": stats, "moe_sites": moe_sites, "log_lines": head[:8],
    "foreign_gpu": foreign_before, "waited_s": waited,
    "stderr_tail": proc.stderr[-1500:] if (proc.returncode != 0 or ident is None) else "",
}
print(json.dumps(record), flush=True)
PY

run_one() {
    echo "+ $1 cache=$2 $3" >> "$STATUS"
    python3 /content/sync_case.py "$1" "$2" "$3" 64 "$REF" "$ENGINE" "$MODEL" >> "$LOG" 2>>/content/sync_sweep_err.txt
}

{
    echo "=== sync/other A/B"
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv
    echo "=== clock probe"
    nvcc -O2 -std=c++17 -arch=sm_80 /content/warodecode/tests/moe_sync_cost_probe.cu -o /content/moe_sync_cost_probe 2>&1 | tail -3
    if [ -x /content/moe_sync_cost_probe ]; then /content/moe_sync_cost_probe 3000; fi
    echo "=== router bit-exactness"
    nvcc -O2 -std=c++17 -arch=sm_80 -I/content/warodecode/src \
        /content/warodecode/tests/router_parallel_exact_test.cu -o /content/router_parallel_test 2>&1 | tail -5
    if [ -x /content/router_parallel_test ]; then /content/router_parallel_test 64; echo "ROUTER_TEST_EXIT=$?"; fi
} > "$PROBE" 2>&1

DIAG="WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1"

# 1. Shipped default twice, so the run-to-run spread of the machine is visible
#    next to every switch result below.
run_one base-a 20000 "$DIAG"
run_one base-b 20000 "$DIAG"
# 2. The selection hoisted out of the runner, keeping the shipped
#    cudaStreamSynchronize wait: this separates the call-site move from the wait.
run_one split-sync 20000 "$DIAG,WARODECODE_MOE_SELECT_SPLIT=1"
run_one split-sync-b 20000 "$DIAG,WARODECODE_MOE_SELECT_SPLIT=1"
# 3. Hoisted + the one-thread router replaced by the parallel twin, still with
#    the shipped wait.
run_one router-sync 20000 "$DIAG,WARODECODE_MOE_SELECT_SPLIT=1,WARODECODE_ROUTER_PARALLEL=1"
# 4. The piece-level device spans of the selection bracket, with the shipped
#    wait (WARODECODE_SELECT_PROBE=1 also switches the call site to the probe
#    variant, but not the wait).
run_one probe-sync 20000 "$DIAG,WARODECODE_MOE_SELECT_SPLIT=1,WARODECODE_SELECT_PROBE=1"
# 5. No-transfer reference (parity breaks by design).
run_one skip 20000 "$DIAG,WARODECODE_MOE_SELECT_SPLIT=1,WARODECODE_ROUTER_PARALLEL=1,WARODECODE_EXPERT_IO_SKIP=1"

# The parallel router is the candidate that has to survive the A/B: with it the
# token must be measurably shorter than the shipped default in the same run.
echo "CANDIDATE router-sync vs base" >> "$STATUS"

echo "DONE" >> "$STATUS"
