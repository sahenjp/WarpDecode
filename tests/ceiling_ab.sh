#!/bin/bash
# Task V round 2: in-job A/B of the IQ4_NL unroll + PIPE=32 against the round-1
# state and the shipped baseline, all in one Colab job.
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$STATUS"
run_one r2_base_1 20000 3 "$OLD"
run_one r2_new_1  20000 3 "$NEW"
run_one r2_base_2 20000 3 "$OLD"
run_one r2_new_2  20000 3 "$NEW"
run_one r2_base_3 20000 3 "$OLD"
run_one r2_new_3  20000 3 "$NEW"
run_one r2_sat_30000 30000 3 "WARODECODE_EXPERT_STATS=1"
run_one r2_prof 20000 1 "WARODECODE_MOE_PROFILE=1,WARODECODE_PROFILE=1,WARODECODE_EXPERT_STATS=1"

echo "DONE" >> "$STATUS"
