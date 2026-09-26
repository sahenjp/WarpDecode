#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "使い方: $0 ./warpdecode model.gguf" >&2
  exit 2
fi
BIN=$1
MODEL=$2
if ! command -v ncu >/dev/null; then
  echo "Nsight Compute(ncu)がありません" >&2
  exit 3
fi
ncu --set full --kernel-name-base demangled \
  --kernel-name 'regex:q4_gemv_2row_kernel.*|attention_decode_kernel.*|rmsnorm_kernel.*' \
  "$BIN" --model "$MODEL" --ctx 1024 --generate 4 --no-graph
