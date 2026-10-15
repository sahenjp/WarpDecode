#!/bin/bash
# Task V (ceiling): where the 55 ms/token actually goes, measured in one job.
#
# Every case runs sequentially in a single Colab job, so the numbers are
# comparable to each other. r1 = one repeat (cold expert cache), r3 = three
# repeats (the run the shipped A/B reports, i.e. warm).
set -u

ENGINE=${WD_ENGINE:-/content/wd_q/build-sm80/warodecode}
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
REF=/content/reference_tokens_64.json
LOG=/content/ceiling_sweep.log
STATUS=/content/ceiling_sweep_status.txt
PROBE=/content/ceiling_sweep_probe.txt

: > "$LOG"
: > "$STATUS"

run_one() {
    echo "+ $1 cache=$2 rep=$3 $4" >> "$STATUS"
    python3 "$(dirname "$0")/ceiling_case.py" "$1" "$2" "$4" 64 "$3" "$REF" "$ENGINE" "$MODEL" \
        >> "$LOG" 2>>/content/ceiling_sweep_err.txt
}

{
    echo "=== ceiling sweep v2"
    nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
        --format=csv
    nproc
    free -g | head -2
} > "$PROBE" 2>&1

STAT="WARODECODE_EXPERT_STATS=1"
PROF="WARODECODE_EXPERT_STATS=1,WARODECODE_MOE_PROFILE=1,WARODECODE_PROFILE=1"

# 1-2. shipped defaults at cache 20000, cold vs warm
run_one b20r1 20000 1 "$STAT"
run_one b20r3 20000 3 "$STAT"
# 3-4. the same two with the MoE + stage profilers on
run_one p20r1 20000 1 "$PROF"
run_one p20r3 20000 3 "$PROF"
# 5. HC / attention kernel breakdown, warm
run_one h20r3 20000 3 "$STAT,WARODECODE_HC_PROFILE=1,WARODECODE_ATTN_PROFILE=1"
# 6. no-transfer reference: the token with the expert staging removed (parity
#    breaks by design), warm
run_one s20r3 20000 3 "WARODECODE_EXPERT_IO_SKIP=1,WARODECODE_MOE_PROFILE=1,WARODECODE_PROFILE=1"
# 7-8. cache 30000, to show what the extra 10 GiB buys
run_one p30r3 30000 3 "$PROF"
run_one b30r3 30000 3 "$STAT"

echo "DONE" >> "$STATUS"
