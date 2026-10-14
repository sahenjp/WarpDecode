#!/bin/bash
# Task N stage 5: rebuild the pinned llama.cpp decode harness and A/B it against
# the engine's CPU MoE on the same session (same GGUF, ctx 128, 64 greedy tokens).
set -u
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
ENGINE=/content/warodecode/build-sm80/warodecode
W=/content/wd_taskl
COMMIT=6fcaa16f4b360649933a54d1f91ad40ed35c0e11
OUT=/content/cpu_moe_stage5
mkdir -p $OUT $W
cd $W || exit 1
echo '=== stage5 ==='; date -u; nproc
if [ ! -d llama.cpp/.git ]; then git clone --quiet https://github.com/ggml-org/llama.cpp llama.cpp; fi
cd llama.cpp || exit 1
git cat-file -e $COMMIT^{commit} 2>/dev/null || git fetch --quiet origin $COMMIT
git config advice.detachedHead false
git checkout --quiet $COMMIT
echo HEAD=$(git rev-parse HEAD)
mkdir -p examples/taskl-decode
cp /content/qwen38_llamacpp_gpu_bench.cpp examples/taskl-decode/main.cpp 2>/dev/null || echo 'HARNESS_MISSING'
if [ ! -f examples/taskl-decode/CMakeLists.txt ]; then
  printf 'set(TARGET taskl-llama-decode)\nadd_executable(\${TARGET} main.cpp)\ntarget_link_libraries(\${TARGET} PRIVATE common llama)\ninstall(TARGETS \${TARGET} RUNTIME)\n' > examples/taskl-decode/CMakeLists.txt
fi
grep -q 'taskl-decode' examples/CMakeLists.txt || echo 'add_subdirectory(taskl-decode)' >> examples/CMakeLists.txt
cmake -S . -B build-cuda-sm80 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=ON > $OUT/cmake.log 2>&1
echo CMAKE_EXIT=$?
cmake --build build-cuda-sm80 -j8 --target taskl-llama-decode > $OUT/build.log 2>&1
echo BUILD_EXIT=$?
tail -5 $OUT/build.log
LLAMA=$W/llama.cpp/build-cuda-sm80/bin/taskl-llama-decode
ls -la $LLAMA || { echo LLAMA_MISSING; exit 1; }

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
  echo "RESULT $label rc=$rc wall=$(echo "$t1 - $t0" | bc) peak_vram_mib=$(sort -n $samples | tail -1) $(grep -o '\"tok_s\":[0-9.]*' $OUT/$label.log | tail -1) $(grep -o '\"matched_prefix\":[0-9]*' $OUT/$label.log | tail -1)"
}

echo '--- cuda llama harness smoke ---'
run_case llama_smoke $LLAMA $MODEL 128 8 99 --keep-moe-gpu-last 0 --threads 12 --reps 1
echo '--- interleaved A/B ---'
for r in 1 2; do
  run_case llama_k0_r$r  $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last 0  --threads 12 --reps 1
  run_case eng_cpu48_r$r WARODECODE_CPU_MOE=1 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64
  run_case llama_k16_r$r $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last 16 --threads 12 --reps 1
  run_case eng_cpu24_r$r WARODECODE_CPU_MOE_LAYERS=24 WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64
  run_case llama_k24_r$r $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last 24 --threads 12 --reps 1
  run_case eng_gpu_r$r  WARODECODE_EXPERT_CACHE_MIB=30000 $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64
done
echo CPU_MOE_STAGE5_DONE
