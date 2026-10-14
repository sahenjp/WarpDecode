#!/bin/bash
# Task I remote A/B round 2: warm cache, alternating off / seq / batched.
set -u
ARCH=89
ENGINE=/content/warodecode/build-sm${ARCH}/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf

run_case() {
  local label=$1; shift
  local switches=$1; shift
  echo "=== $label : $switches"
  rm -f "/content/taskI2_${label}.log"
  env $switches "$ENGINE" --model "$MODEL" --ctx 128 --device 0 \
      --start-token 1 --no-graph --no-autotune --generate 64 \
      > "/content/taskI2_${label}.log" 2>&1
  echo "RUN_EXIT=$? label=$label" >> "/content/taskI2_${label}.log"
}

BASE="WARODECODE_EXPERT_CACHE_MIB=8000"
MTP="WARODECODE_MTP=1 WARODECODE_MTP_EXPERT_CACHE_MIB=3072"
BATCH="WARODECODE_MTP_BATCH_VERIFY=1 WARODECODE_MOE_BATCH_K=4"

run_case off "$BASE"
for k in 2 3 4; do
  run_case seq_k${k} "$BASE $MTP WARODECODE_MTP_DRAFT=${k}"
  run_case bat_k${k} "$BASE $MTP WARODECODE_MTP_DRAFT=${k} $BATCH"
done
run_case off2 "$BASE"
run_case seq_k2b "$BASE $MTP WARODECODE_MTP_DRAFT=2"
run_case bat_k2b "$BASE $MTP WARODECODE_MTP_DRAFT=2 $BATCH"

echo DONE > /content/taskI2_done
