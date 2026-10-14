#!/bin/bash
# Task N: compact final verification (one placement round, no --bench-repeat, no
# page-cache prewarm). Keeps the session time short and the numbers honest.
set -u
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
ENGINE=/content/warodecode/build-sm80/warodecode
OUT=/content/cpu_moe_final2
mkdir -p $OUT
echo '=== final2 ==='; date -u; grep -m1 'model name' /proc/cpuinfo
free -m | grep -E 'Mem|Cached'
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader

one() {
  local label=$1; shift
  local t0=$(date +%s.%N)
  timeout 900 env "$@" > $OUT/$label.log 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  echo "RESULT $label rc=$rc wall=$(echo "$t1 - $t0" | bc) $(grep -o '\"tok_s\":[0-9.]*' $OUT/$label.log | tail -1) $(grep -o 'cpu-moe=[a-z]* layers=[0-9/]*' $OUT/$label.log | tail -1)"
}

A="--model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64"
echo '--- gpu baseline ---'
one gpu_c30000 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE $A
echo '--- parity all-CPU (the absolute condition) ---'
one parity_all WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $A
if grep -q '\"tokens\":64' $OUT/parity_all.log; then
  python3 - "$OUT/parity_all.log" <<'PY'
import json, sys
for line in open(sys.argv[1], errors='replace'):
    line = line.strip()
    if line.startswith('{"engine":"warodecode-qwen38"'):
        tok = json.loads(line)["token_ids"]
        ref = json.load(open('/content/reference_tokens_64.json'))["token_ids"]
        n = 0
        while n < 64 and tok[n] == ref[n]:
            n += 1
        print('PARITY matched_prefix=%d/64' % n)
PY
fi
echo '--- placement sweep ---'
one gpu_c8000    WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $A
one cpu48_c8000  WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $A
one cpu24_c8000  WARODECODE_CPU_MOE_LAYERS=24 WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $A
one cpu8_c8000   WARODECODE_CPU_MOE_LAYERS=8 WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $A
echo CPU_MOE_FINAL2_DONE
