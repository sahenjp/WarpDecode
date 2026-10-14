#!/bin/bash
# Task I remote A/B: MTP off / on / on+batched block MoE, one session, one model
# download. Runs the four engine cases back to back and leaves one log per case
# in /content/taskI_<label>.log plus /content/taskI_done.
set -u
ARCH=${1:-89}
ENGINE=/content/warodecode/build-sm${ARCH}/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
CACHE=/content/taskI_ab.cache
cleanup() { rm -f "$CACHE" /content/taskI_done; }
trap cleanup EXIT

if [ ! -x "$ENGINE" ]; then
  echo "NO_ENGINE $ENGINE" > /content/taskI_done
  exit 1
fi

run_case() {
  local label=$1; shift
  local switches=$1; shift
  echo "=== $label : $switches"
  rm -f "$CACHE"
  rm -f "/content/taskI_${label}.log"
  env $switches "$ENGINE" --model "$MODEL" --ctx 128 --device 0 \
      --start-token 1 --no-graph --no-autotune --generate 64 \
      > "/content/taskI_${label}.log" 2>&1
  echo "RUN_EXIT=$? label=$label" >> "/content/taskI_${label}.log"
}

BASE="WARODECODE_EXPERT_CACHE_MIB=8000"
# The first case is normally the run the launcher itself started; only run it
# here when that log is missing (e.g. the launcher was pointed at another case).
if [ ! -s /content/taskI_off.log ]; then
  run_case off "$BASE"
else
  echo "=== off : reusing /content/taskI_off.log"
fi
run_case seq_k2 "$BASE WARODECODE_MTP=1 WARODECODE_MTP_DRAFT=2 WARODECODE_MTP_EXPERT_CACHE_MIB=3072"
run_case bat_k2 "$BASE WARODECODE_MTP=1 WARODECODE_MTP_DRAFT=2 WARODECODE_MTP_EXPERT_CACHE_MIB=3072 WARODECODE_MTP_BATCH_VERIFY=1 WARODECODE_MOE_BATCH_K=4"
run_case bat_k3 "$BASE WARODECODE_MTP=1 WARODECODE_MTP_DRAFT=3 WARODECODE_MTP_EXPERT_CACHE_MIB=3072 WARODECODE_MTP_BATCH_VERIFY=1 WARODECODE_MOE_BATCH_K=4"
run_case bat_k4 "$BASE WARODECODE_MTP=1 WARODECODE_MTP_DRAFT=4 WARODECODE_MTP_EXPERT_CACHE_MIB=3072 WARODECODE_MTP_BATCH_VERIFY=1 WARODECODE_MOE_BATCH_K=4"

echo DONE > /content/taskI_done
