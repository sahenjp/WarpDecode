// Host pin for the Task R HC legacy-skip guard (src/qwen38_hc.cuh).
//
// The legacy F16 HC mix may only be dropped when all four facts below hold. The
// predicate itself is four booleans wide, so the value of this test is not the
// truth table but the *safety property* it pins: every fact has a concrete
// failure mode, and a "skipped but wrong" configuration must never be
// expressible. The real predicate is compiled by
// tests/qwen38_cuda_syntax_check.sh; tests/qwen38_hc_legacy_skip_host_check.sh
// additionally asserts that the real call sites still route through it.
//
//  - skip_switch_enabled: WARODECODE_HC_SKIP_LEGACY[_ATTN|_FFN].
//  - exact_mix_active:    the exact Q8_0 replay runs for this layer. Skipping
//                         without it leaves st.hc_*.mixed() at the previous
//                         layer's value (the row feeds the router and MoE).
//  - exact_writes_inject: that replay takes its non-null branch, which is the
//                         one that copies the unrounded inject back into the
//                         half scratch. Without it run_qwen38_hc_combine()
//                         reads the half inject of the previous layer.
//  - half_mirror_consumed: a trace/debug consumer reads xn / gate / lo, which
//                         only the legacy pass fills. Skipping then silently
//                         changes the traced intermediates (parity runs are
//                         compared token-wise, so a silent trace change is a
//                         regression nobody would notice).
//
// Usage: bash tests/qwen38_hc_legacy_skip_host_check.sh
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

// Mirror of src/qwen38_hc.cuh:q38_hc_legacy_mix_skippable(). Keep the body and
// the parameter order identical; the structural half of the host check asserts
// that the header still spells the conjunction this way.
bool legacy_mix_skippable(
        bool skip_switch_enabled,
        bool exact_mix_active,
        bool exact_writes_inject,
        bool half_mirror_consumed) {
    return skip_switch_enabled && exact_mix_active && exact_writes_inject &&
           !half_mirror_consumed;
}

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error("HC legacy skip guard: " + what);
}

} // namespace

int main() {
    int skipped = 0;
    // Exhaustive truth table with the invariant checked on every entry.
    for (int bits = 0; bits < 16; ++bits) {
        const bool skip_switch_enabled = (bits & 1) != 0;
        const bool exact_mix_active = (bits & 2) != 0;
        const bool exact_writes_inject = (bits & 4) != 0;
        const bool half_mirror_consumed = (bits & 8) != 0;
        const bool skippable = legacy_mix_skippable(skip_switch_enabled, exact_mix_active,
                                                    exact_writes_inject, half_mirror_consumed);
        const char * what = nullptr;
        if (skippable) {
            ++skipped;
            // The safety property: a skip implies every premise of the skip.
            if (!skip_switch_enabled) what = "skipped with the switch off";
            else if (!exact_mix_active) what = "skipped without the exact replay";
            else if (!exact_writes_inject) what = "skipped without the inject rewrite";
            else if (half_mirror_consumed) what = "skipped while the half mirrors are read";
        } else if (skip_switch_enabled && exact_mix_active && exact_writes_inject &&
                   !half_mirror_consumed) {
            what = "did not skip a fully satisfied configuration";
        }
        if (what != nullptr) {
            std::fprintf(stderr, "bits=%d: %s\n", bits, what);
            return 1;
        }
    }
    // Exactly one of the 16 rows skips: all three positive premises present and
    // the half-mirror consumer absent (bits = 0b0111). Everything else keeps the
    // legacy pass, which is the conservative direction.
    require(skipped == 1, "expected 1 skipping row, got " + std::to_string(skipped));
    // The documented A/B switch: off wins over every other fact.
    require(!legacy_mix_skippable(false, true, true, false), "switch off must not skip");
    // Default decode configuration: switch on, exact replay on with the F32
    // inject, no trace attached - the case Task R makes fast.
    require(legacy_mix_skippable(true, true, true, false), "default configuration must skip");
    // A quantised inject tensor (blk.48 MTP) or WARODECODE_HC_INJECT_F32=0.
    require(!legacy_mix_skippable(true, true, false, false), "quantised inject must not skip");
    // Any attached trace, and the debug F32 replays.
    require(!legacy_mix_skippable(true, true, true, true), "attached trace must not skip");
    std::printf("PASS (16 rows, %d skipping)\n", skipped);
    return 0;
}
