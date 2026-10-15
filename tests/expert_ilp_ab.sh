#!/bin/bash
# Task W: in-job A/B of the ILP routed-expert GEMV against the sequential kernel.
# Both arms are the same binary; only WARODECODE_EXPERT_GEMV_ILP differs, and the
# runs alternate so a drifting VM cannot favour either arm.
set -u

ENGINE=${WD_ENGINE:-/content/wd_taskl/warodecode/build-sm80/warodecode}
MODEL=${WD_MODEL:-/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf}
REF=${WD_REF:-/content/wd_taskl/reference_tokens_64.json}
CASE=${WD_CASE:-/content/wd_taskl/ceiling_case.py}
OUT=${WD_OUT:-/content/expert_ilp_ab.jsonl}
LOG=${WD_LOG:-/content/expert_ilp_ab.log}
STATUS=${WD_STATUS:-/content/expert_ilp_ab_status.txt}

: > "$OUT"
: > "$LOG"
: > "$STATUS"

run_one() {
    echo "+ $1 ilp=$2 rep=$3" >> "$STATUS"
    python3 "$CASE" "$1" 20000 "WARODECODE_EXPERT_GEMV_ILP=$2" 64 "$3" "$REF" "$ENGINE" "$MODEL" \
        >> "$OUT" 2>>"$LOG"
}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$STATUS"
md5sum "$ENGINE" >> "$STATUS"

run_one ilp1_r1 0 3
run_one ilp8_r1 8 3
run_one ilp4_r1 4 3
run_one ilp1_r2 0 3
run_one ilp8_r2 8 3
run_one ilp4_r2 4 3
run_one ilp1_r3 0 3
run_one ilp8_r3 8 3
run_one ilp4_r3 4 3

echo "DONE" >> "$STATUS"
