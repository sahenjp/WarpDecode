#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "使い方: $0 ./warodecode /path/to/llama-bench model.gguf GPU_LAYERS" >&2
  exit 2
fi
WD=$1
LLAMA=$2
MODEL=$3
NGL=$4
OUT=${OUT:-bench-out}
mkdir -p "$OUT"

# warodecodeは生成専用。llama.cpp側もbatch=1のtoken generationを主比較にする。
"$WD" --model "$MODEL" --ctx 4096 --generate 128 | tee "$OUT/warodecode.json"
"$LLAMA" -m "$MODEL" -ngl "$NGL" -fa on -ctk f16 -ctv f16 -p 0 -n 128 -b 1 -ub 1 -r 5 -o json > "$OUT/llama.json"
python3 "$(dirname "$0")/compare.py" "$OUT/warodecode.json" "$OUT/llama.json" | tee "$OUT/compare.json"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv,noheader > "$OUT/gpu-state.txt" || true
fi

echo "保存先: $OUT"
