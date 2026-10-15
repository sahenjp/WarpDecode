#!/usr/bin/env python3
"""Task L: llama.cpp <-> warp engine A/B on one GPU session (batch=1 decode).

Builds the pinned llama.cpp (CUDA, sm80), the warp engine and a small llama.cpp
decode harness on the Colab VM, then measures the two runtimes alternately on the
same GPU, the same 85.9 GB GGUF and the same decode conditions (start token 1,
64 greedy tokens, ctx 128, n_batch=n_ubatch=1, F16 KV). Every run is checked
against the canonical CPU-oracle tokens, so a fast-but-divergent run is visible
instead of silently being reported as a win.

Phases (all detached on the VM, polled by downloading their logs):

  build_llamacpp   git clone + cmake (GGML_CUDA=ON, sm80) the pinned commit
  build_engine     engine + harness build from the uploaded source tree
  probe            CPU-oracle check, VRAM / load-time sweep, llama.cpp MoE
                   placement sweep (--keep-moe-gpu-last 0..N)
  matrix           interleaved A/B: engine 8000 MiB, llama.cpp MoE-on-CPU,
                   engine 16000 MiB, llama.cpp MoE-partly-on-GPU (VRAM matched),
                   repeated; one CPU-only llama.cpp run at the end
  matrix_big       larger expert caches (engine 24000/30000 MiB) against
                   llama.cpp with the same VRAM budget (K=20/24)

Usage:

  export PATH=/home/sahen/.local/bin:$PATH
  WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
    python3 tools/llamacpp_ab.py --rounds 3 --reps 3 --llama-k 8 --llama-k2 16

  python3 tools/llamacpp_ab.py --phases matrix            # re-run only the matrix
  python3 tools/llamacpp_ab.py --phases none --collect    # download + print

Logs and results live in /content/wd_taskl on the VM; the merged results land in
issue1_artifacts/taskl_ab_results.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-a100")
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "80")
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
WORK = "/content/wd_taskl"
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"
HARNESS_SRC = ROOT / "qwen38_llamacpp_gpu_bench.cpp"
REFERENCE = ROOT / "issue1_artifacts/reference_tokens_64.json"
RESULT_JSONL = ROOT / "issue1_artifacts/taskl_ab_results.jsonl"

BUILD_LLAMACPP = """#!/bin/bash
# Build the pinned llama.cpp with CUDA for the session's GPU arch.
LOG=${WORK}/build_llamacpp.log
mkdir -p ${WORK}
exec > "$LOG" 2>&1
set -x
date -u
PREFIX=${COMMIT}
cd ${WORK}
if [ ! -d llama.cpp/.git ]; then
  rm -rf llama.cpp
  git clone --quiet https://github.com/ggml-org/llama.cpp llama.cpp
fi
cd llama.cpp
git cat-file -e $PREFIX^{commit} 2>/dev/null || git fetch --quiet origin $PREFIX
git config advice.detachedHead false
git checkout --quiet $PREFIX || { git fetch --quiet origin refs/pull/28243/head && git checkout --quiet FETCH_HEAD; }
echo "HEAD=$(git rev-parse HEAD)"
echo "SUBJECT=$(git log -1 --format=%s)"
cmake -S ${WORK}/llama.cpp -B ${WORK}/llama.cpp/build-cuda-sm${ARCH} \\
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \\
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON
echo "CMAKE_EXIT=$?"
cmake --build ${WORK}/llama.cpp/build-cuda-sm${ARCH} -j 8 --target llama-cli llama-bench
echo "BUILD_EXIT=$?"
date -u
"""

STEP_A = """#!/bin/bash
# Build the warp engine from the uploaded tree and the llama.cpp decode harness
# as an extra example target of the pinned checkout.
LOG=${WORK}/stepA.log
exec > "$LOG" 2>&1
set -x
date -u
W=${WORK}
rm -rf $W/warodecode
mkdir -p $W/warodecode
cd $W/warodecode
unzip -q -o $W/warodecode-source.zip
cmake -S . -B build-sm${ARCH} -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=${ARCH}
echo "ENGINE_CMAKE_EXIT=$?"
cmake --build build-sm${ARCH} -j 8 --target warodecode
echo "ENGINE_BUILD_EXIT=$?"
ls -la $W/warodecode/build-sm${ARCH}/warodecode
L=$W/llama.cpp
mkdir -p $L/examples/taskl-decode
sed 's/__COMMIT__/${COMMIT}/g' $W/qwen38_llamacpp_gpu_bench.cpp > $L/examples/taskl-decode/taskl-llama-decode.cpp
printf 'add_executable(taskl-llama-decode taskl-llama-decode.cpp)\\ntarget_link_libraries(taskl-llama-decode PRIVATE llama)\\n' > $L/examples/taskl-decode/CMakeLists.txt
if ! grep -q 'file(GLOB' $L/examples/CMakeLists.txt && ! grep -q 'taskl-decode' $L/examples/CMakeLists.txt; then
  echo 'add_subdirectory(taskl-decode)' >> $L/examples/CMakeLists.txt
fi
cmake -S $L -B $L/build-cuda-sm${ARCH} -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${ARCH} -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON
echo "HARNESS_CMAKE_EXIT=$?"
cmake --build $L/build-cuda-sm${ARCH} -j 8 --target taskl-llama-decode
echo "HARNESS_BUILD_EXIT=$?"
ls -la $L/build-cuda-sm${ARCH}/bin/taskl-llama-decode
date -u
echo STEP_A_DONE
"""

PROBE = """#!/bin/bash
# Correctness, VRAM and placement probes that fix the A/B matrix.
W=${WORK}
R=$W/probe
LOG=$W/stepB.log
mkdir -p $R
exec > "$LOG" 2>&1
set -x
date -u
MODEL=${MODEL}
ENGINE=$W/warodecode/build-sm${ARCH}/warodecode
LLAMA=$W/llama.cpp/build-cuda-sm${ARCH}/bin/taskl-llama-decode

maxvram() { sort -n "$1" 2>/dev/null | tail -1; }

run_timed() {
  local label=$1; shift
  local samples=$R/$label.vram
  local stop=$R/$label.stop
  rm -f "$stop"; : > "$samples"
  ( while [ ! -f "$stop" ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$samples"; sleep 1; done ) &
  local sampler=$!
  local t0=$(date +%s.%N)
  "$@" > $R/$label.out 2> $R/$label.err
  local rc=$?
  local t1=$(date +%s.%N)
  touch "$stop"; wait $sampler 2>/dev/null
  echo "RESULT $label rc=$rc wall_s=$(echo "$t1 - $t0" | bc) max_vram_mib=$(maxvram $samples)"
}

echo "=== B0: llama.cpp CPU path (oracle validation, threads 2, -ngl 0) ==="
run_timed oracle_cpu $LLAMA $MODEL 128 64 0 --threads 2 --reps 1
python3 - "$W" <<'PY'
import json, pathlib, sys
W = pathlib.Path(sys.argv[1])
ref = json.loads((W / "reference_tokens_64.json").read_text())["token_ids"]
rows = [json.loads(l.split("=", 1)[1]) for l in (W / "probe" / "oracle_cpu.out").read_text(errors="replace").splitlines() if l.startswith("LLAMA_AB=")]
if rows:
    got = rows[-1]["token_ids"]
    pref = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
    print("ORACLE_CHECK matched_prefix=%d/%d tok_s=%.3f" % (pref, len(ref), rows[-1]["tok_s"]))
else:
    print("ORACLE_CHECK FAILED no LLAMA_AB line")
PY

echo "=== B1: engine VRAM / load time probes ==="
for mib in 8000 16000; do
  run_timed engine_$mib env WARODECODE_EXPERT_CACHE_MIB=$mib $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 --bench-repeat 1
done

echo "=== B2: llama.cpp MoE placement sweep (-ngl 99, last K layers' experts on GPU) ==="
for k in 0 1 4 8 12; do
  run_timed llama_k$k $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last $k --threads 12 --reps 1
done

echo "=== llama.cpp buffer sizes ==="
grep -h "model buffer size" $R/llama_k*.err 2>/dev/null | sort -u | head -20
echo "=== engine JSON ==="
grep -h '"engine":"warodecode-qwen38"' $R/engine_*.out
echo "=== llama.cpp JSON ==="
grep -h 'LLAMA_AB=' $R/llama_k*.out $R/oracle_cpu.out
date -u
echo STEP_B_DONE
"""

EMIT = '''#!/usr/bin/env python3
"""Merge one A/B run into the Task L results jsonl."""
import json, pathlib, sys

label, kind, round_s, wall_s, max_vram, outfile, results = sys.argv[1:8]
ref = json.loads(pathlib.Path("${WORK}/reference_tokens_64.json").read_text())["token_ids"]
text = pathlib.Path(outfile).read_text(errors="replace")
rec = {
    "label": label,
    "kind": kind,
    "round": int(round_s),
    "wall_s": float(wall_s),
    "max_vram_mib": int(max_vram) if max_vram not in ("", "none") else None,
}
if kind == "engine":
    rows = [json.loads(l) for l in text.splitlines() if l.startswith('{"engine":"warodecode-qwen38"')]
    if not rows:
        print("NO_ENGINE_JSON"); sys.exit(1)
    j = rows[-1]
    for key in ("tok_s", "decode_seconds", "decode_tokens", "seconds", "first_token", "last_token",
                "token_hash", "bench_repeats", "ctx", "cuda_graph", "head_q8_k", "kernel_family"):
        rec[key] = j.get(key)
    ids = j.get("token_ids", [])
else:
    rows = [json.loads(l.split("=", 1)[1]) for l in text.splitlines() if l.startswith("LLAMA_AB=")]
    if not rows:
        print("NO_LLAMA_JSON"); sys.exit(1)
    j = rows[-1]
    for key in ("tok_s", "decode_seconds", "decode_tokens", "n_gpu_layers", "moe_cpu_layers",
                "mmap", "threads", "ctx", "ctx_seconds"):
        rec[key] = j.get(key)
    rec["rounds_tok_s"] = [r["tok_s"] for r in rows]
    rec["rounds_seconds"] = [r["decode_seconds"] for r in rows]
    ids = j.get("token_ids", [])
pref = next((i for i, (a, b) in enumerate(zip(ref, ids)) if a != b), min(len(ref), len(ids)))
rec["matched_prefix"] = pref
rec["token_match"] = (pref == 64)
with pathlib.Path(results).open("a") as handle:
    handle.write(json.dumps(rec) + "\\n")
print(json.dumps(rec))
'''

STEP_C = """#!/bin/bash
# Interleaved A/B matrix on one GPU.
#   TASKL_ROUNDS   alternating rounds (default 3)
#   TASKL_REPS     repetitions inside one run (default 2)
#   TASKL_LLAMA_K  llama.cpp --keep-moe-gpu-last for the VRAM-matched config
W=${WORK}
R=$W/ab
mkdir -p $R
LOG=$W/stepC.log
exec > "$LOG" 2>&1
set -x
date -u
MODEL=${MODEL}
ENGINE=$W/warodecode/build-sm${ARCH}/warodecode
LLAMA=$W/llama.cpp/build-cuda-sm${ARCH}/bin/taskl-llama-decode
EMIT=$W/taskl_ab_emit.py
RESULTS=$R/taskl_ab.jsonl
ROUNDS=${TASKL_ROUNDS:-3}
REPS=${TASKL_REPS:-2}
LLAMA_K=${TASKL_LLAMA_K:-8}
LLAMA_K2=${TASKL_LLAMA_K2:-16}
: > "$RESULTS"
md5sum $LLAMA $ENGINE
nvidia-smi --query-gpu=name,memory.total,memory.used,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader

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
  echo "RUN $label round=$rnd rc=$rc wall_s=$wall max_vram_mib=$(maxvram $samples)"
  if [ "$rc" -ne 0 ]; then echo "FAILED $label r$rnd"; tail -5 $R/${label}_r${rnd}.err; return; fi
  python3 "$EMIT" "$label" "$kind" "$rnd" "$wall" "$(maxvram $samples)" "$out" "$RESULTS"
}

engine_run() {  # engine_run <label> <round> <mib>
  local label=$1 rnd=$2 mib=$3
  run_one "$label" engine "$rnd" env WARODECODE_EXPERT_CACHE_MIB=$mib $ENGINE \\
    --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune \\
    --generate 64 --bench-repeat $REPS
}

llama_run() {   # llama_run <label> <round> <keep-moe-gpu-last> <ngl> <threads> [extra...]
  local label=$1 rnd=$2 k=$3 ngl=$4 thr=$5; shift 5
  run_one "$label" llama "$rnd" $LLAMA $MODEL 128 64 $ngl --keep-moe-gpu-last $k \\
    --threads $thr --reps $REPS "$@"
}

for rnd in $(seq 1 $ROUNDS); do
  echo "===== ROUND $rnd ====="
  engine_run engine_8000  $rnd 8000
  llama_run  llama_moe_cpu $rnd 0 99 12
  engine_run engine_16000 $rnd 16000
  llama_run  llama_moe_gpu${LLAMA_K} $rnd $LLAMA_K 99 12
  llama_run  llama_moe_gpu${LLAMA_K2} $rnd $LLAMA_K2 99 12
done
echo "===== extras ====="
llama_run llama_cpu_only 0 0 0 12
date -u
echo STEP_C_DONE
"""


STEP_E = """#!/bin/bash
# Larger expert cache on the engine side and the same VRAM budget on the
# llama.cpp side (one round, interleaved).
W=${WORK}
R=$W/ab
LOG=$W/stepE.log
mkdir -p $R
exec > "$LOG" 2>&1
set -x
date -u
MODEL=${MODEL}
ENGINE=$W/warodecode/build-sm${ARCH}/warodecode
LLAMA=$W/llama.cpp/build-cuda-sm${ARCH}/bin/taskl-llama-decode
EMIT=$W/taskl_ab_emit.py
RESULTS=$R/taskl_ab_E.jsonl
: > "$RESULTS"
REPS=${TASKL_REPS:-3}

maxvram() { sort -n "$1" 2>/dev/null | tail -1; }

run_one() {
  local label=$1; shift
  local kind=$1; shift
  local out=$R/${label}_rE.out
  local samples=$R/${label}_rE.vram
  local stop=$R/${label}_rE.stop
  rm -f "$stop"; : > "$samples"
  ( while [ ! -f "$stop" ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$samples"; sleep 1; done ) &
  local sampler=$!
  local t0=$(date +%s.%N)
  "$@" > "$out" 2> $R/${label}_rE.err
  local rc=$?
  local t1=$(date +%s.%N)
  touch "$stop"; wait $sampler 2>/dev/null
  local wall=$(echo "$t1 - $t0" | bc)
  echo "RUN $label rc=$rc wall_s=$wall max_vram_mib=$(maxvram $samples)"
  if [ "$rc" -ne 0 ]; then echo "FAILED $label"; tail -5 $R/${label}_rE.err; return; fi
  python3 "$EMIT" "$label" "$kind" 0 "$wall" "$(maxvram $samples)" "$out" "$RESULTS"
}

engine_run() { local label=$1 mib=$2
  run_one "$label" engine env WARODECODE_EXPERT_CACHE_MIB=$mib $ENGINE --model $MODEL --ctx 128 \
    --device 0 --start-token 1 --no-graph --no-autotune --generate 64 --bench-repeat $REPS
}
llama_run() { local label=$1 k=$2
  run_one "$label" llama $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last $k --threads 12 --reps $REPS
}

engine_run engine_24000 24000
llama_run  llama_moe_gpu20 20
engine_run engine_30000 30000
llama_run  llama_moe_gpu24 24
engine_run engine_30000b 30000
date -u
echo STEP_E_DONE
"""

EMIT3 = '''#!/usr/bin/env python3
"""Merge one interleaved A/B (ab3) run into the results jsonl."""
import json, pathlib, sys

label, kind, round_s, wall_s, max_vram, max_apps, pids, outfile, results = sys.argv[1:10]
ref = json.loads(pathlib.Path("${WORK}/reference_tokens_64.json").read_text())["token_ids"]
text = pathlib.Path(outfile).read_text(errors="replace")
rec = {
    "label": label,
    "kind": kind,
    "round": int(round_s),
    "wall_s": float(wall_s),
    "max_vram_mib": int(max_vram) if max_vram.isdigit() else None,
    "max_gpu_apps": int(max_apps) if max_apps.isdigit() else None,
    "gpu_app_pids": sorted({p for p in pids.split() if p}),
}
if kind == "engine":
    rows = [json.loads(l) for l in text.splitlines() if l.startswith('{"engine":"warodecode-qwen38"')]
    if not rows:
        print("NO_ENGINE_JSON"); sys.exit(1)
    j = rows[-1]
    for key in ("tok_s", "decode_seconds", "decode_tokens", "seconds", "bench_repeats", "ctx",
                "cuda_graph", "head_q8_k", "kernel_family"):
        rec[key] = j.get(key)
    rec["reps_tok_s"] = [j.get("tok_s")]
    ids = j.get("token_ids", [])
else:
    rows = [json.loads(l.split("=", 1)[1]) for l in text.splitlines() if l.startswith("LLAMA_AB=")]
    if not rows:
        print("NO_LLAMA_JSON"); sys.exit(1)
    rec["reps_tok_s"] = [r["tok_s"] for r in rows]
    rec["reps_seconds"] = [r["decode_seconds"] for r in rows]
    j = rows[-1]
    for key in ("n_gpu_layers", "moe_cpu_layers", "load_mode", "threads", "ctx", "ctx_seconds"):
        rec[key] = j.get(key)
    # best-of-N over the in-process repetitions, symmetric with the engine's
    # --bench-repeat (which also reports its fastest repeat)
    rec["tok_s"] = min(rec["reps_tok_s"])
    rec["tok_s_last"] = rec["reps_tok_s"][-1]
    rec["decode_seconds"] = min(rec["reps_seconds"])
    rec["decode_tokens"] = j.get("decode_tokens")
    ids = j.get("token_ids", [])
pref = next((i for i, (a, b) in enumerate(zip(ref, ids)) if a != b), min(len(ref), len(ids)))
rec["matched_prefix"] = pref
rec["token_match"] = (pref == len(ref))
with pathlib.Path(results).open("a") as handle:
    handle.write(json.dumps(rec) + chr(10))
print(json.dumps(rec))
'''

STEP_AB3 = """#!/bin/bash
# Interleaved A/B with a GPU-emptiness guard before every run.
#
#   TASKL_ROUNDS  rounds (default 3); even rounds run the pairs in reverse order
#   TASKL_REPS    repetitions inside one process (default 3)
#   TASKL_CACHES  engine expert cache MiB, paired position by position with ...
#   TASKL_KS      llama.cpp --keep-moe-gpu-last values (VRAM budget pairing)
W=${WORK}
R=$W/ab3
mkdir -p $R
LOG=$W/stepAB3.log
exec > "$LOG" 2>&1
set -x
date -u
MODEL=${MODEL}
ENGINE=$W/warodecode/build-sm${ARCH}/warodecode
LLAMA=$W/llama.cpp/build-cuda-sm${ARCH}/bin/taskl-llama-decode
EMIT=$W/taskl_ab3_emit.py
RESULTS=$R/taskl_ab3.jsonl
ROUNDS=${TASKL_ROUNDS:-3}
REPS=${TASKL_REPS:-3}
CACHES=${TASKL_CACHES:-8000,20000,30000}
KS=${TASKL_KS:-0,16,24}
: > "$RESULTS"
md5sum $LLAMA $ENGINE
nvidia-smi --query-gpu=name,memory.total,memory.used,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader
[ -f $W/engine_revision.txt ] && cat $W/engine_revision.txt

maxvram() { sort -n "$1" 2>/dev/null | tail -1; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$'; }

wait_gpu_free() {
  local waited=0
  while [ "$waited" -lt 1800 ]; do
    local a=$(gpu_apps | paste -sd, -)
    if [ -z "$a" ]; then echo "GPU_FREE waited_s=$waited"; return 0; fi
    echo "GPU_WAIT foreign=$a waited_s=$waited"; sleep 15; waited=$((waited+15))
  done
  echo "GPU_BUSY_TIMEOUT"; return 1
}

run_one() {   # run_one <label> <kind> <round> <command...>
  local label=$1; shift
  local kind=$1; shift
  local rnd=$1; shift
  local out=$R/${label}_r${rnd}.out
  local samples=$R/${label}_r${rnd}.vram
  local procs=$R/${label}_r${rnd}.procs
  local stop=$R/${label}_r${rnd}.stop
  if ! wait_gpu_free; then echo "SKIP $label r$rnd (GPU stayed busy)"; return 1; fi
  rm -f "$stop"; : > "$samples"; : > "$procs"
  ( while [ ! -f "$stop" ]; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$samples"
      nvidia-smi --query-compute-apps=pid --format=csv,noheader >> "$procs"
      sleep 1
    done ) &
  local sampler=$!
  local t0=$(date +%s.%N)
  "$@" > "$out" 2> $R/${label}_r${rnd}.err
  local rc=$?
  local t1=$(date +%s.%N)
  touch "$stop"; wait $sampler 2>/dev/null
  local wall=$(echo "$t1 - $t0" | bc)
  local maxapps=$(sort -n $procs 2>/dev/null | tail -1)
  echo "RUN $label round=$rnd rc=$rc wall_s=$wall max_vram_mib=$(maxvram $samples) max_gpu_apps=$maxapps pids=$(sort -u $procs | paste -sd, -)"
  if [ "$rc" -ne 0 ]; then echo "FAILED $label r$rnd"; tail -5 $R/${label}_r${rnd}.err; return 1; fi
  python3 "$EMIT" "$label" "$kind" "$rnd" "$wall" "$(maxvram $samples)" "$maxapps" "$(sort -u $procs | paste -sd' ' -)" "$out" "$RESULTS"
}

engine_run() { local label=$1 rnd=$2 mib=$3
  run_one "$label" engine "$rnd" env WARODECODE_EXPERT_CACHE_MIB=$mib $ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune --generate 64 --bench-repeat $REPS
}
llama_run() { local label=$1 rnd=$2 k=$3
  run_one "$label" llama "$rnd" $LLAMA $MODEL 128 64 99 --keep-moe-gpu-last $k --threads 12 --reps $REPS
}

IFS=',' read -ra CA <<< "$CACHES"
IFS=',' read -ra KA <<< "$KS"
[ "${#CA[@]}" = "${#KA[@]}" ] || { echo "CACHES/KS length mismatch"; echo STEP_AB3_DONE; exit 1; }

for rnd in $(seq 1 $ROUNDS); do
  echo "===== ROUND $rnd ====="
  if [ $((rnd % 2)) -eq 0 ]; then
    for ((i=${#CA[@]}-1; i>=0; i--)); do
      llama_run "llama_k${KA[$i]}" $rnd "${KA[$i]}"
      engine_run "engine_${CA[$i]}" $rnd "${CA[$i]}"
    done
  else
    for ((i=0; i<${#CA[@]}; i++)); do
      engine_run "engine_${CA[$i]}" $rnd "${CA[$i]}"
      llama_run "llama_k${KA[$i]}" $rnd "${KA[$i]}"
    done
  fi
done
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
date -u
echo STEP_AB3_DONE
"""


def phase_ab3(args):
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        tmp_path = pathlib.Path(tmp)
        emit_path = tmp_path / "taskl_ab3_emit.py"
        emit_path.write_text(render(EMIT3))
        upload(emit_path, WORK + "/taskl_ab3_emit.py")
        script = (render(STEP_AB3)
                  .replace("${TASKL_ROUNDS:-3}", "${TASKL_ROUNDS:-%d}" % args.rounds)
                  .replace("${TASKL_REPS:-3}", "${TASKL_REPS:-%d}" % args.reps)
                  .replace("${TASKL_CACHES:-8000,20000,30000}", "${TASKL_CACHES:-%s}" % args.engine_caches)
                  .replace("${TASKL_KS:-0,16,24}", "${TASKL_KS:-%s}" % args.llama_ks))
        local = tmp_path / "taskl_stepAB3.sh"
        local.write_text(script)
        upload(local, WORK + "_taskl_stepAB3.sh")
    launch("taskl_stepAB3.sh", "stepAB3.log")
    return wait_for_log(WORK + "/stepAB3.log", pathlib.Path("/tmp/taskl_stepAB3.log"),
                        "STEP_AB3_DONE", args.timeout, "ab3")


def collect_ab3(args):
    local = RESULT_JSONL.parent / "taskl_ab3_results.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    download(WORK + "/ab3/taskl_ab3.jsonl", local)
    if not local.exists():
        print("no ab3 results yet")
        return
    rows = [json.loads(l) for l in local.read_text().splitlines() if l.strip()]
    print("")
    print("%-16s %5s %8s %9s %8s %7s %s" % ("label", "round", "tok/s", "decode_s", "wall_s", "vram", "match"))
    for row in rows:
        print("%-16s %5s %8.3f %9.3f %8.1f %7s %d/64" % (
            row["label"], row["round"], row["tok_s"], row["decode_seconds"], row["wall_s"],
            row["max_vram_mib"] if row["max_vram_mib"] is not None else "-", row["matched_prefix"]))



def run(cmd, quiet=False):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet:
        print("\\n".join(proc.stdout.splitlines()[-4:]), flush=True)
    if proc.returncode:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))
    return proc.stdout


def upload(local, remote):
    run(CLI + ["upload", "-s", SESSION, str(local), remote], quiet=True)


def download(remote, local):
    subprocess.run(CLI + ["download", "-s", SESSION, remote, str(local)],
                   text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def exec_file(path, quiet=True):
    return run(CLI + ["exec", "-s", SESSION, "--timeout", "300", "-f", str(path)], quiet=quiet)


def exec_inline(code):
    # google-colab-cli has no "exec -c" in the pinned version, so every snippet
    # goes through a temporary file (the CLI uploads it and runs it in the VM).
    with tempfile.TemporaryDirectory(prefix="taskl-inline-") as tmp:
        path = pathlib.Path(tmp) / "inline_snippet.py"
        path.write_text(code)
        exec_file(path)


def launch(shell_name, log_name, attempts=3):
    """Start the detached shell on the VM and confirm that it really started.

    A second client on the same Colab session can cut an exec short between the
    log cleanup and the Popen, so the starter's "STARTED" line is required and
    the launch is retried when it is missing.
    """
    starter = pathlib.Path(tempfile.mkdtemp(prefix="taskl-launch-")) / "start.py"
    lines = [
        "import subprocess, pathlib",
        "dst = pathlib.Path(%r)" % (WORK + "/" + shell_name),
        "dst.write_text(pathlib.Path(%r).read_text()); dst.chmod(0o755)" % (WORK + "_" + shell_name),
        "pathlib.Path(%r).unlink(missing_ok=True)" % (WORK + "/" + log_name),
        "p = subprocess.Popen(['bash', str(dst)], start_new_session=True)",
        "pathlib.Path(%r).write_text(str(p.pid))" % (WORK + "/" + shell_name.replace(".sh", ".pid")),
        "print('STARTED', p.pid)",
    ]
    starter.write_text(chr(10).join(lines) + chr(10))
    for attempt in range(1, attempts + 1):
        output = exec_file(starter) or ""
        if "STARTED" in output:
            print("+ launched %s (%s)" % (shell_name, output.strip().splitlines()[-1]), flush=True)
            return True
        print("+ launch attempt %d for %s produced no STARTED line" % (attempt, shell_name), flush=True)
        time.sleep(15)
    raise RuntimeError("could not start %s on the VM" % shell_name)


def wait_for_log(remote, local, marker, timeout_s, label):
    # A log left over from an earlier session would let the marker match before
    # the VM has written anything, so the local copy always starts empty.
    pathlib.Path(local).unlink(missing_ok=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        download(remote, local)
        text = local.read_text(errors="replace") if local.exists() else ""
        if marker in text:
            print("[%s] done" % label, flush=True)
            return True
        last = text.splitlines()[-1:] or [""]
        print("[%s] %s" % (label, last[0][:120]), flush=True)
        time.sleep(30)
    print("[%s] TIMEOUT after %ds" % (label, timeout_s), flush=True)
    return False


def revision_text():
    """Identity of the uploaded tree: commit, dirty state and a src hash."""
    def git(*args):
        proc = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return proc.stdout.strip()

    lines = ["rev=" + git("rev-parse", "HEAD"),
             "subject=" + git("log", "-1", "--format=%s"),
             "date=" + git("log", "-1", "--format=%cI")]
    status = git("status", "--short")
    lines.append("dirty=" + ("yes" if status else "no"))
    lines.extend("status " + line for line in status.splitlines())
    digest = hashlib.sha256()
    for name in ("src", "CMakeLists.txt"):
        entry = ROOT / name
        files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
        for file in files:
            if file.is_file():
                digest.update(str(file.relative_to(ROOT)).encode())
                digest.update(file.read_bytes())
    lines.append("engine_src_sha256=" + digest.hexdigest())
    return "\n".join(lines) + "\n"


def make_source_archive(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("REVISION.txt", revision_text())
        for name in ("src", "tests", "tools", "bench"):
            for file in (ROOT / name).rglob("*"):
                if file.is_file() and "qwen38-fixture" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
        for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)


def render(text):
    return (text.replace("${WORK}", WORK).replace("${COMMIT}", LLAMA_COMMIT)
                .replace("${ARCH}", ARCH).replace("${MODEL}", MODEL))


def phase_build_llamacpp(args):
    exec_inline("import pathlib; pathlib.Path(%r).mkdir(parents=True, exist_ok=True)" % WORK)
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        local = pathlib.Path(tmp) / "taskl_build_llamacpp.sh"
        local.write_text(render(BUILD_LLAMACPP))
        upload(local, WORK + "_taskl_build_llamacpp.sh")
    launch("taskl_build_llamacpp.sh", "build_llamacpp.log")
    return wait_for_log(WORK + "/build_llamacpp.log", pathlib.Path("/tmp/taskl_build_llamacpp.log"),
                        "BUILD_EXIT", args.timeout, "build_llamacpp")


def phase_build_engine(args):
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "warodecode-source.zip"
        make_source_archive(archive)
        upload(archive, WORK + "/warodecode-source.zip")
        upload(HARNESS_SRC, WORK + "/qwen38_llamacpp_gpu_bench.cpp")
        upload(REFERENCE, WORK + "/reference_tokens_64.json")
        local = tmp_path / "taskl_stepA.sh"
        local.write_text(render(STEP_A))
        upload(local, WORK + "_taskl_stepA.sh")
    launch("taskl_stepA.sh", "stepA.log")
    return wait_for_log(WORK + "/stepA.log", pathlib.Path("/tmp/taskl_stepA.log"),
                        "STEP_A_DONE", args.timeout, "build_engine")


def phase_probe(args):
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        local = pathlib.Path(tmp) / "taskl_stepB.sh"
        local.write_text(render(PROBE))
        upload(local, WORK + "_taskl_stepB.sh")
    launch("taskl_stepB.sh", "stepB.log")
    return wait_for_log(WORK + "/stepB.log", pathlib.Path("/tmp/taskl_stepB.log"),
                        "STEP_B_DONE", args.timeout, "probe")


def phase_matrix(args):
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        tmp_path = pathlib.Path(tmp)
        emit_path = tmp_path / "taskl_ab_emit.py"
        emit_path.write_text(render(EMIT))
        upload(emit_path, WORK + "/taskl_ab_emit.py")
        script = (render(STEP_C)
                  .replace("${TASKL_ROUNDS:-3}", "${TASKL_ROUNDS:-%d}" % args.rounds)
                  .replace("${TASKL_REPS:-2}", "${TASKL_REPS:-%d}" % args.reps)
                  .replace("${TASKL_LLAMA_K:-8}", "${TASKL_LLAMA_K:-%d}" % args.llama_k)
                  .replace("${TASKL_LLAMA_K2:-16}", "${TASKL_LLAMA_K2:-%d}" % args.llama_k2))
        local = tmp_path / "taskl_stepC.sh"
        local.write_text(script)
        upload(local, WORK + "_taskl_stepC.sh")
    launch("taskl_stepC.sh", "stepC.log")
    return wait_for_log(WORK + "/stepC.log", pathlib.Path("/tmp/taskl_stepC.log"),
                        "STEP_C_DONE", args.timeout, "matrix")


def phase_matrix_big(args):
    with tempfile.TemporaryDirectory(prefix="taskl-") as tmp:
        tmp_path = pathlib.Path(tmp)
        emit_path = tmp_path / "taskl_ab_emit.py"
        emit_path.write_text(render(EMIT))
        upload(emit_path, WORK + "/taskl_ab_emit.py")
        script = render(STEP_E).replace("§{TASKL_REPS:-3}", "§{TASKL_REPS:-%d}" % args.reps)
        local = tmp_path / "taskl_stepE.sh"
        local.write_text(script)
        upload(local, WORK + "_taskl_stepE.sh")
    launch("taskl_stepE.sh", "stepE.log")
    return wait_for_log(WORK + "/stepE.log", pathlib.Path("/tmp/taskl_stepE.log"),
                        "STEP_E_DONE", args.timeout, "matrix_big")


def collect(args):
    RESULT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    download(WORK + "/ab/taskl_ab.jsonl", RESULT_JSONL)
    download(WORK + "/ab/taskl_ab_E.jsonl", RESULT_JSONL.parent / "taskl_ab_extra_results.jsonl")
    if not RESULT_JSONL.exists():
        print("no results yet")
        return
    rows = [json.loads(l) for l in RESULT_JSONL.read_text().splitlines() if l.strip()]
    print("")
    print("%-20s %5s %7s %9s %8s %9s  %s" % ("label", "round", "tok/s", "decode_s", "wall_s", "vram_mib", "match"))
    for row in rows:
        print("%-20s %5s %7.3f %9.3f %8.1f %9s  %d/64" % (
            row["label"], row["round"], row["tok_s"], row["decode_seconds"], row["wall_s"],
            row["max_vram_mib"] if row["max_vram_mib"] is not None else "-", row["matched_prefix"]))


PHASES = {
    "build_llamacpp": phase_build_llamacpp,
    "build_engine": phase_build_engine,
    "probe": phase_probe,
    "matrix": phase_matrix,
    "matrix_big": phase_matrix_big,
    "ab3": phase_ab3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phases", default="build_llamacpp,build_engine,probe,matrix",
                        help="comma separated phases, or none")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--llama-k", type=int, default=8,
                        help="llama.cpp --keep-moe-gpu-last matched to the 8000 MiB cache")
    parser.add_argument("--llama-k2", type=int, default=16,
                        help="llama.cpp --keep-moe-gpu-last matched to the 16000 MiB cache")
    parser.add_argument("--timeout", type=int, default=5400, help="per phase timeout in seconds")
    parser.add_argument("--engine-caches", default="8000,20000,30000",
                        help="engine WARODECODE_EXPERT_CACHE_MIB values, paired with --llama-ks")
    parser.add_argument("--llama-ks", default="0,16,24",
                        help="llama.cpp --keep-moe-gpu-last values, paired with --engine-caches")
    parser.add_argument("--collect", action="store_true", help="only download and print results")
    parser.add_argument("--collect-ab3", action="store_true", help="download + print the ab3 results")
    args = parser.parse_args()

    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch")

    print("+ session=%s arch=sm%s work=%s" % (SESSION, ARCH, WORK), flush=True)
    if args.phases != "none":
        for name in args.phases.split(","):
            name = name.strip()
            if not name:
                continue
            print("+ phase %s" % name, flush=True)
            PHASES[name](args)
    if args.collect:
        collect(args)
    if args.collect_ab3 or (args.phases != "none" and "ab3" in args.phases):
        collect_ab3(args)
    elif args.phases != "none":
        collect(args)


if __name__ == "__main__":
    main()
