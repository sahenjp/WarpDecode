#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

bash -n setup.sh bench/build.sh bench/ab.sh bench/profile.sh
python3 -m py_compile bench/compare.py tests/make_qwen38_fixture.py
python3 tests/make_qwen38_fixture.py
cmake -S . -B build-ui -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=OFF >/dev/null
cmake --build build-ui -j"$(nproc)" >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/gguf_parser_test.cpp -o /tmp/warodecode-gguf-parser-test
/tmp/warodecode-gguf-parser-test tests/min.gguf >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/gguf_type_size_test.cpp -o /tmp/warodecode-gguf-type-size-test
/tmp/warodecode-gguf-type-size-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/gguf_array_test.cpp -o /tmp/warodecode-gguf-array-test
/tmp/warodecode-gguf-array-test tests/qwen38-fixture/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/ple_reader_test.cpp -o /tmp/warodecode-ple-reader-test
/tmp/warodecode-ple-reader-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/qwen38_tuning_test.cpp -o /tmp/warodecode-qwen38-tuning-test
/tmp/warodecode-qwen38-tuning-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/qwen38_trace_test.cpp -o /tmp/warodecode-qwen38-trace-test
/tmp/warodecode-qwen38-trace-test

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc tests/issue1_residual_accum_test.cpp -o /tmp/warodecode-issue1-residual-accum-test
/tmp/warodecode-issue1-residual-accum-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc tests/issue1_ffn_precision_test.cpp -o /tmp/warodecode-issue1-ffn-precision-test
/tmp/warodecode-issue1-ffn-precision-test >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc tests/issue1_ffn_ablation_test.cpp -o /tmp/warodecode-issue1-ffn-ablation-test
/tmp/warodecode-issue1-ffn-ablation-test

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/model_profile_test.cpp -o /tmp/warodecode-model-profile-test
/tmp/warodecode-model-profile-test tests/qwen38-fixture/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf >/dev/null

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tests/qwen38_layout_test.cpp -o /tmp/warodecode-qwen38-layout-test
/tmp/warodecode-qwen38-layout-test tests/qwen38-fixture/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf >/dev/null

./build-ui/warodecode-tui --dump-state \
  --gpu-fixture tests/nvidia-smi.fixture.csv \
  --model-dir tests > /tmp/warodecode-state.json
python3 -m json.tool /tmp/warodecode-state.json >/dev/null

./build-ui/warodecode-tui --dump-state \
  --gpu-fixture tests/nvidia-smi.fixture.csv \
  --model-dir tests/qwen38-fixture > /tmp/warodecode-q38-state.json
python3 - <<'PY2'
import json
x=json.load(open('/tmp/warodecode-q38-state.json'))
assert len(x['models']) == 1, x
m=x['models'][0]
assert m['qwen38_flash_next'] is True, m
assert m['arch'] == 'qwen4exp', m
assert m['shards'] == 2, m
assert m['quant'] == 'iq3e', m
PY2

# DP4A 経路 (IQ2_S/IQ3_S/IQ4_NL) の整数部が参照実装と bit 一致することを確認する
bash tests/lowbit_dp4a_host_check.sh

# Task G: expert 統合 MoE の加算順序と index 対応が従来と bit 一致することを確認する
bash tests/moe_fused_grid_host_check.sh

# nvcc が無くても CUDA カーネルの構文・型を検査できる (CI で実行する)
bash tests/qwen38_cuda_syntax_check.sh

rm -rf tests/qwen38-fixture

echo PASS
