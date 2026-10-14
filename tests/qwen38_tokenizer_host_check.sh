#!/usr/bin/env bash
# Tokenizer parity: src/qwen38_tokenizer.cuh vs the pinned llama.cpp vocabulary.
#
# Both drivers tokenize the same corpus (tests/qwen38_tokenizer_corpus.txt) for
# all four (add_special, parse_special) combinations and print one JSON object
# per case. Any difference in a token ID is a real mismatch: the CUDA engine's
# prompt path would then feed different tokens than the CPU oracle.
#
#   tests/qwen38_tokenizer_host_test.cpp   our tokenizer (plain g++, no CUDA)
#   tests/qwen38_tokenizer_oracle.cpp      llama_tokenize() through libllama
#
# Usage:
#   bash tests/qwen38_tokenizer_host_check.sh <llama.cpp checkout> <model.gguf>
#
# The checkout needs a CPU-only build directory (only libllama / libggml are
# linked, no GPU):
#
#   cmake -S "$LLAMA" -B "$LLAMA/build-host" -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
#     -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
#   cmake --build "$LLAMA/build-host" --target llama -j"$(nproc)"
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA=${1:-}
if [[ -z "$LLAMA" ]]; then LLAMA=${LLAMA_CPP:-}; fi
MODEL=${2:-}
if [[ -z "$MODEL" ]]; then MODEL=${WARODECODE_TOKENIZER_MODEL:-}; fi
CORPUS=${WARODECODE_TOKENIZER_CORPUS:-$ROOT/tests/qwen38_tokenizer_corpus.txt}
WORK=${Q38_TOKENIZER_WORK:-/tmp/qwen38-tokenizer-host}

if [[ -z "$LLAMA" || ! -f "$LLAMA/include/llama.h" ]]; then
  echo "usage: bash tests/qwen38_tokenizer_host_check.sh <llama.cpp checkout> <model.gguf>" >&2
  echo "       (or export LLAMA_CPP=... WARODECODE_TOKENIZER_MODEL=...)" >&2
  exit 2
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "missing model.gguf (second argument or WARODECODE_TOKENIZER_MODEL)" >&2
  exit 2
fi
BUILD=$LLAMA/build-host
if [[ ! -f "$BUILD/bin/libllama.so" && ! -f "$BUILD/bin/libllama.a" ]]; then
  echo "missing $BUILD/bin/libllama - see the header of this script" >&2
  exit 2
fi

mkdir -p "$WORK"

# 0. The Unicode tables are generated from the same checkout. Regenerate into the
#    work dir and fail when the committed copy is stale.
python3 "$ROOT/tests/make_qwen38_tokenizer_tables.py" "$LLAMA" --out "$WORK/qwen38_tokenizer_unicode.inc" >/dev/null
if ! diff -q "$WORK/qwen38_tokenizer_unicode.inc" "$ROOT/src/qwen38_tokenizer_unicode.inc" >/dev/null; then
  echo "src/qwen38_tokenizer_unicode.inc が $LLAMA の生成結果と一致しません。" >&2
  echo "  python3 tests/make_qwen38_tokenizer_tables.py $LLAMA で再生成してください" >&2
  diff -u "$ROOT/src/qwen38_tokenizer_unicode.inc" "$WORK/qwen38_tokenizer_unicode.inc" | head -20 >&2
  exit 1
fi
echo "+ unicode tables match $LLAMA"

# 1. Our tokenizer, no llama.cpp dependency.
g++ -std=c++17 -O2 -Wall -Wextra -I "$ROOT/src" \
    "$ROOT/tests/qwen38_tokenizer_host_test.cpp" -o "$WORK/tokenizer_host_test"

# 2. The llama.cpp vocabulary oracle.
g++ -std=c++17 -O2 -Wall -Wextra \
    -I "$LLAMA/include" -I "$LLAMA/ggml/include" -I "$BUILD/include" \
    "$ROOT/tests/qwen38_tokenizer_oracle.cpp" \
    -L "$BUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin" -o "$WORK/tokenizer_oracle"

echo "+ corpus: $(grep -c '' "$CORPUS") lines x 4 (add_special, parse_special) cases"
"$WORK/tokenizer_host_test" "$MODEL" --corpus "$CORPUS" > "$WORK/host.jsonl"
"$WORK/tokenizer_oracle"    "$MODEL" --corpus "$CORPUS" > "$WORK/oracle.jsonl"

grep -E '^\{' "$WORK/host.jsonl"   > "$WORK/host.cases"
grep -E '^\{' "$WORK/oracle.jsonl" > "$WORK/oracle.cases"
grep -E '^#' "$WORK/host.jsonl" || true
grep -E '^#' "$WORK/oracle.jsonl" || true

if diff -q "$WORK/host.cases" "$WORK/oracle.cases" >/dev/null; then
  echo "TOKENIZER MATCH: $(grep -c '' "$WORK/host.cases") cases identical to llama.cpp"
  exit 0
fi

echo "TOKENIZER MISMATCH" >&2
diff -u "$WORK/oracle.cases" "$WORK/host.cases" | head -40 >&2
echo "--- first mismatching corpus lines ---" >&2
python3 - "$WORK/oracle.cases" "$WORK/host.cases" <<'PY' >&2
import json, sys
oracle = [json.loads(line) for line in open(sys.argv[1])]
host = [json.loads(line) for line in open(sys.argv[2])]
shown = 0
for index, (a, b) in enumerate(zip(oracle, host)):
    if a == b:
        continue
    print("case %d text=%r add_special=%d parse_special=%d" % (
        index, a["text"], a["add_special"], a["parse_special"]))
    print("  llama.cpp: %s" % a["tokens"])
    print("  warodecode: %s" % b["tokens"])
    shown += 1
    if shown >= 10:
        break
print("mismatching cases: %d / %d" % (
    sum(1 for a, b in zip(oracle, host) if a != b), len(oracle)))
PY
exit 1
