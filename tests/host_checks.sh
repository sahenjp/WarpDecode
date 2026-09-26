#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

bash -n setup.sh bench/build.sh bench/ab.sh bench/profile.sh
python3 -m py_compile bench/compare.py tests/make_qwen38_fixture.py
python3 tests/make_qwen38_fixture.py
cmake -S . -B build-ui -DCMAKE_BUILD_TYPE=Release -DWARPDECODE_BUILD_ENGINE=OFF >/dev/null
cmake --build build-ui -j"$(nproc)" >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/gguf_parser_test.cpp -o /tmp/warpdecode-gguf-parser-test
/tmp/warpdecode-gguf-parser-test tests/min.gguf >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/ple_reader_test.cpp -o /tmp/warpdecode-ple-reader-test
/tmp/warpdecode-ple-reader-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/qwen38_tuning_test.cpp -o /tmp/warpdecode-qwen38-tuning-test
/tmp/warpdecode-qwen38-tuning-test >/dev/null

./build-ui/warpdecode-tui --dump-state \
  --gpu-fixture tests/nvidia-smi.fixture.csv \
  --model-dir tests > /tmp/warpdecode-state.json
python3 -m json.tool /tmp/warpdecode-state.json >/dev/null

./build-ui/warpdecode-tui --dump-state \
  --gpu-fixture tests/nvidia-smi.fixture.csv \
  --model-dir tests/qwen38-fixture > /tmp/warpdecode-q38-state.json
python3 - <<'PY2'
import json
x=json.load(open('/tmp/warpdecode-q38-state.json'))
assert len(x['models']) == 1, x
m=x['models'][0]
assert m['qwen38_flash_next'] is True, m
assert m['arch'] == 'qwen4exp', m
assert m['shards'] == 2, m
assert m['quant'] == 'iq3e', m
PY2

rm -rf tests/qwen38-fixture

echo PASS
