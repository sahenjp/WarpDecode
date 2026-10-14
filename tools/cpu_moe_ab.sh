#!/bin/bash
# Task N: CPU MoE placement sweep vs llama.cpp on the same A100 session.
# Runs interleaved (llama.cpp / engine) so the GPU state does not favour one side.
W=/content/wd_taskl
R=$W/cpu_moe_ab
mkdir -p $R
LOG=$W/cpu_moe_ab.log
exec > "$LOG" 2>&1
set -x
date -u
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
ENGINE=/content/warodecode/build-sm80/warodecode
LLAMA=$W/llama.cpp/build-cuda-sm80/bin/taskl-llama-decode
EMIT=$W/taskl_ab_emit.py
RESULTS=$R/cpu_moe_ab.jsonl
ROUNDS="${CPU_MOE_ROUNDS:-2}"
REPS="${CPU_MOE_REPS:-2}"
: > "$RESULTS"
md5sum $LLAMA $ENGINE
nvidia-smi --query-gpu=name,memory.total,memory.used,clocks.sm,clocks.max.sm --format=csv,noheader

maxvram() { sort -n "$1" 2>/dev/null | tail -1; }

run_one() {   # run_one <label> <kind> <round> <command...>
  local label=$1; shift
  local kind=$1; shift
  local rnd=$1; shift
  local out=$R/${label}_r${rnd}.out
  local samples=$R/${label}_r${rnd}.vram
  local stop=$R/${label}_r${rnd}.stop
  rm -f "$stop"; : > "$samples"
  ( while [ ! -f "$stop" ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$samples"; sleep 1; done ) &
  local sampler=$!
  local t0=$(date +%s.%N)
  "$@" > "$out" 2> $R/${label}_r${rnd}.err
  local rc=$?
  local t1=$(date +%s.%N)
  touch "$stop"; wait $sampler 2>/dev/null
  local wall=$(echo "$t1 - $t0" | bc)
  echo "RESULT $label rc=$rc wall_s=$wall max_vram_mib=$(maxvram $samples)"
  python3 $EMIT "$label" "$kind" "$rnd" "$wall" "$(maxvram $samples)" "$out" "$RESULTS"
}

engine_run() {  # engine_run <label> <round> <cpu_layers> <cache_mib>
  local label=$1; shift
  local rnd=$1; shift
  local layers=$1; shift
  local cache=$1; shift
  run_one "$label" engine "$rnd" env WARODECODE_CPU_MOE_LAYERS=$layers \
    WARODECODE_EXPERT_CACHE_MIB=$cache \
    $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune \
    --generate 64 --bench-repeat $REPS
}

llama_run() {   # llama_run <label> <round> <keep-moe-gpu-last>
  local label=$1; shift
  local rnd=$1; shift
  local k=$1; shift
  run_one "$label" llama "$rnd" $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last $k \
    --threads 12 --reps $REPS
}

for rnd in $(seq 1 $ROUNDS); do
  llama_run llama_k0   $rnd 0
  engine_run engine_cpu48 $rnd 48 8000
  llama_run llama_k16  $rnd 16
  engine_run engine_cpu32 $rnd 32 16000
  llama_run llama_k24  $rnd 24
  engine_run engine_cpu16 $rnd 16 30000
  engine_run engine_gpu    $rnd 0 30000
done

date -u
echo CPU_MOE_AB_DONE
