#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

bash -n setup.sh bench/build.sh bench/ab.sh bench/profile.sh
python3 -m py_compile bench/compare.py
cmake -S . -B build-ui -DCMAKE_BUILD_TYPE=Release -DWARPDECODE_BUILD_ENGINE=OFF >/dev/null
cmake --build build-ui -j"$(nproc)" >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/gguf_parser_test.cpp -o /tmp/warpdecode-gguf-parser-test
/tmp/warpdecode-gguf-parser-test tests/min.gguf >/dev/null

./build-ui/warpdecode-tui --dump-state \
  --gpu-fixture tests/nvidia-smi.fixture.csv \
  --model-dir tests > /tmp/warpdecode-state.json
python3 -m json.tool /tmp/warpdecode-state.json >/dev/null

echo PASS
