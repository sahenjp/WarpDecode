#!/bin/bash
# Task N: final A100 verification. The session was reclaimed, so /content is
# empty: start the 86 GB model download first, then build the engine, then run
# the interleaved placement sweep and the parity sweep.
set -u
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
ENGINE=/content/warodecode/build-sm80/warodecode
REPO=pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF
FILE=Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
OUT=/content/cpu_moe_final
mkdir -p $OUT
echo '=== final ==='; date -u; nproc; grep -m1 'model name' /proc/cpuinfo
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
free -g | head -2

# --- model ---
if [ ! -f $MODEL ]; then
  echo 'MODEL_MISSING: start the download'
  mkdir -p /content/models
  nohup bash -lc "hf download $REPO $FILE --local-dir /content/models > /content/download.log 2>&1" >/dev/null 2>&1 &
  for i in $(seq 1 90); do
    sleep 20
    SZ=$(stat -c %s $MODEL 2>/dev/null || echo 0)
    echo "download bytes=$SZ"
    if [ "$SZ" -ge 85923705344 ]; then break; fi
  done
fi
ls -la $MODEL || { echo NO_MODEL; exit 1; }

# --- build the engine ---
cd /content/warodecode || exit 1
cmake -S . -B build-sm80 -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=80 > $OUT/cmake.log 2>&1
echo CMAKE_EXIT=$?
cmake --build build-sm80 -j8 --target warodecode > $OUT/build.log 2>&1
echo BUILD_EXIT=$?
tail -3 $OUT/build.log
[ -x $ENGINE ] || { echo NO_ENGINE; exit 1; }
md5sum $ENGINE
echo '--- host kernel checks ---'
bash tests/qwen38_cpu_moe_host_check.sh
echo HOST_CHECK_EXIT=$?

run_case() {
  local label=$1; shift
  local samples=$OUT/$label.vram; local stop=$OUT/$label.stop
  rm -f $stop; : > $samples
  ( while [ ! -f "$stop" ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $samples; sleep 1; done ) &
  local sampler=$!
  local t0=$(date +%s.%N)
  env "$@" > $OUT/$label.log 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  touch $stop; wait $sampler 2>/dev/null
  echo "RESULT $label rc=$rc wall=$(echo "$t1 - $t0" | bc) vram=$(sort -n $samples | tail -1) $(grep -o '\"tok_s\":[0-9.]*' $OUT/$label.log | tail -1) $(grep -o 'cpu-moe=[a-z]* layers=[0-9/]*' $OUT/$label.log | tail -1)"
}

ENGINE_ARGS="--model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 --bench-repeat 2"
for r in 1 2 3; do
  echo "--- round $r ---"
  run_case gpu_c30000_r$r   WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE $ENGINE_ARGS
  run_case cpu48_c30000_r$r WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE $ENGINE_ARGS
  run_case cpu24_c30000_r$r WARODECODE_CPU_MOE_LAYERS=24 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE $ENGINE_ARGS
  run_case cpu48_c8000_r$r  WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $ENGINE_ARGS
  run_case gpu_c8000_r$r    WARODECODE_EXPERT_CACHE_MIB=8000 $ENGINE $ENGINE_ARGS
done

echo '--- parity sweep ---'
for l in 48 24 8; do
  env WARODECODE_CPU_MOE_LAYERS=$l WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 > $OUT/parity_$l.log 2>&1
  echo "parity layers=$l $(grep -o '\"token_hash\":\"[^\"]*\"' $OUT/parity_$l.log | tail -1) $(grep -o '\"tok_s\":[0-9.]*' $OUT/parity_$l.log | tail -1)"
done
env WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 > $OUT/parity_all.log 2>&1
echo "parity all $(grep -o '\"token_hash\":\"[^\"]*\"' $OUT/parity_all.log | tail -1)"
echo CPU_MOE_FINAL_DONE
