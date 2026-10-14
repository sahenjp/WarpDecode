#!/bin/bash
# Task N: parity-only re-verification against the pinned CPU oracle tokens.
set -u
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
ENGINE=/content/warodecode/build-sm80/warodecode
OUT=/content/cpu_moe_parity
mkdir -p $OUT
echo '=== parity ==='; date -u; free -m | grep -E 'Mem|Cached'
for cfg in "WARODECODE_CPU_MOE=1" "WARODECODE_CPU_MOE_LAYERS=24" "WARODECODE_CPU_MOE_LAYERS=8" "WARODECODE_EXPERT_CACHE_MIB=30000"; do
  label=$(echo $cfg | tr '=' '_')
  t0=$(date +%s)
  timeout 900 env $cfg WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 > $OUT/$label.log 2>&1
  rc=$?
  t1=$(date +%s)
  echo "CASE $cfg rc=$rc secs=$((t1-t0)) $(grep -o '\"tok_s\":[0-9.]*' $OUT/$label.log | tail -1)"
  grep -a 'cpu-moe=' $OUT/$label.log | tail -1
done
python3 - <<'PY'
import json, glob
ref = json.load(open('/content/reference_tokens_64.json'))['token_ids']
for path in sorted(glob.glob('/content/cpu_moe_parity/*.log')):
    for line in open(path, errors='replace'):
        line = line.strip()
        if line.startswith('{"engine":"warodecode-qwen38"'):
            j = json.loads(line)
            tok = j['token_ids']
            n = 0
            while n < min(len(tok), len(ref)) and tok[n] == ref[n]:
                n += 1
            print('PARITY %-58s %d/64 tok_s=%.3f' % (path.split('/')[-1], n, j['tok_s']))
PY
echo CPU_MOE_PARITY_DONE
