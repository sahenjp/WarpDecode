#!/usr/bin/env bash
# Task R host check: the HC legacy-skip guard (src/qwen38_hc.cuh + forward.cuh).
#
# Part 1 mirrors the guard predicate and pins its safety property: a skip must
# imply every premise that makes it sound, so "skipped but wrong" cannot be
# expressed.
# Part 2 asserts textually that both forward-pass call sites still route through
# the shared predicate, that the legacy mix calls sit behind the guard (they must
# stay reachable for the A/B configuration) and that the fail-closed invariant
# checks are still present. nvcc is not required.
#
# Usage: bash tests/qwen38_hc_legacy_skip_host_check.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${Q38_HC_LEGACY_WORK:-/tmp/qwen38-hc-legacy-skip-host}"
mkdir -p "$WORK"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    "$ROOT/tests/qwen38_hc_legacy_skip_host_test.cpp" -o "$WORK/qwen38_hc_legacy_skip_host_test"
"$WORK/qwen38_hc_legacy_skip_host_test"

python3 - "$ROOT/src/qwen38_hc.cuh" "$ROOT/src/qwen38_forward.cuh" <<'PY'
import pathlib
import sys

hc = pathlib.Path(sys.argv[1]).read_text()
fwd = pathlib.Path(sys.argv[2]).read_text()

def need(haystack, needle, what):
    if needle not in haystack:
        raise SystemExit("HC legacy skip check: missing %s\n  expected: %s" % (what, needle))

need(hc, "inline bool q38_hc_legacy_mix_skippable(", "predicate definition")
need(hc, "return skip_switch_enabled && exact_mix_active && exact_writes_inject &&",
     "predicate conjunction")
need(hc, "!half_mirror_consumed;", "half-mirror premise")

for site in ("attn", "ffn"):
    need(fwd, "const bool %s_skip_legacy = q38_hc_legacy_mix_skippable(" % site,
         "%s site predicate call" % site)
    need(fwd, "q38_hc_skip_legacy_%s_enabled()" % site, "%s site switch" % site)
    need(fwd, "if (%s_skip_legacy && !%s_f32_inject) {" % (site, site),
         "%s site fail-closed invariant" % site)

need(fwd, "if (!attn_skip_legacy && cur == nullptr) {", "attn legacy call guard")
need(fwd, "if (!ffn_used_f32 && !ffn_skip_legacy) {", "ffn legacy call guard")

# The legacy mixes stay reachable in the A/B configuration; only the guard
# around the call changed, so exactly the attn and ffn sites may keep one.
calls = fwd.count("cur = run_qwen38_hc_mix(")
if calls != 2:
    raise SystemExit("HC legacy skip check: expected 2 legacy mix calls (attn+ffn), found %d" % calls)
print("structural: predicate + 2 guarded call sites + 2 fail-closed checks present")
PY

echo "PASS (Task R HC legacy skip guard)"
