#!/bin/bash
# Task V final: in-job A/B, the warm/cold interval decomposition of the new
# state, and the cache-size saturation check, all in one Colab job.
set -u

ENGINE=${WD_ENGINE:-/content/wd_q/build-sm80/warodecode}
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
REF=/content/reference_tokens_64.json
LOG=/content/ceiling_ab.log
STATUS=/content/ceiling_ab_status.txt

: > "$LOG"
: > "$STATUS"

run_one() {
    echo "+ $1 cache=$2 rep=$3 $4" >> "$STATUS"
    python3 "$(dirname "$0")/ceiling_case.py" "$1" "$2" "$4" 64 "$3" "$REF" "$ENGINE" "$MODEL" \
        >> "$LOG" 2>>/content/ceiling_ab_err.txt
}

OLD="WARODECODE_SHEXP_GEMV_FAST=0,WARODECODE_HC_Q8_PIPE=0"
NEW=""
PROF="WARODECODE_MOE_PROFILE=1,WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$STATUS"
# 1-4. interleaved baseline / new defaults, the shipped metric (best of 3)
run_one ab_base_1 20000 3 "$OLD"
run_one ab_new_1  20000 3 "$NEW"
run_one ab_base_2 20000 3 "$OLD"
run_one ab_new_2  20000 3 "$NEW"
# 5-6. cold (one repeat) and warm (three) profiles of the new state, so the
#      per-interval warm decomposition can be derived exactly
run_one pr_new_cold 20000 1 "$PROF"
run_one pr_new_warm 20000 3 "$PROF"
# 7-8. the cache-size saturation, re-measured with the new binary
run_one sat_20000 20000 3 "WARODECODE_EXPERT_STATS=1"
run_one sat_30000 30000 3 "WARODECODE_EXPERT_STATS=1"

echo "DONE" >> "$STATUS"
