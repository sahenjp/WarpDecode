#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-auto}

normalize_arch() {
  case "$1" in
    auto)
      if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smiがありません。sm86のようにアーキテクチャを指定してください" >&2; exit 2
      fi
      local cc
      cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d ' ')
      [[ "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]] || { echo "compute capabilityを取得できません: $cc" >&2; exit 2; }
      echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
      ;;
    maxwell|sm50|50) echo 50 ;;
    p100|sm60|60) echo 60 ;;
    p40|p4|sm61|61) echo 61 ;;
    v100|titanv|sm70|70) echo 70 ;;
    t4|rtx20|sm75|75) echo 75 ;;
    a100|sm80|80) echo 80 ;;
    rtx30|a10|sm86|86) echo 86 ;;
    rtx40|l40|sm89|89) echo 89 ;;
    h100|h200|sm90|90) echo 90 ;;
    b100|b200|sm100|100) echo 100 ;;
    b300|sm103|103) echo 103 ;;
    rtx50|sm120|120) echo 120 ;;
    sm[0-9]*) echo "${1#sm}" ;;
    [0-9]*) echo "$1" ;;
    *) echo "不明なGPU指定: $1" >&2; exit 2 ;;
  esac
}

ARCH=$(normalize_arch "$GPU")

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvccがありません" >&2
  exit 3
fi
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | tail -n1)
read -r CUDA_MAJOR CUDA_MINOR <<<"${NVCC_VER:-0 0}"

if (( ARCH < 75 && CUDA_MAJOR >= 13 )); then
  echo "CUDA 13以降はMaxwell/Pascal/Volta(sm<75)をoffline compileできません。CUDA 12.9以前を使ってください。" >&2
  exit 4
fi
if (( ARCH >= 120 )) && { (( CUDA_MAJOR < 12 )) || (( CUDA_MAJOR == 12 && CUDA_MINOR < 8 )); }; then
  echo "sm_${ARCH}にはCUDA 12.8以降が必要です" >&2
  exit 4
fi

BUILD="build-sm${ARCH}"
echo "warpdecode: target=sm_${ARCH} build=$BUILD"
cmake -S . -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWARPDECODE_BUILD_ENGINE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$BUILD" -j"$(nproc)"
