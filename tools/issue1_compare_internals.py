#!/usr/bin/env python3
"""Compare Qwen3.8 warp (CUDA engine) internals against the CPU oracle.

Both sides write the same binary vector container ("WDPTRC1", little endian):

    magic    "WDPTRC1\0"            8 bytes
    version  uint32
    count    uint32
    entry*   { name_len uint32, value_count uint64, name UTF-8, float32[value_count] }

CPU side   : qwen38_forced_cpu_trace.cpp (llama.cpp graph capture; tensor names
             such as "pos39.hc_norm-7", "pos39.ffn_moe_logits-8").
Engine side: Qwen38Trace in src/qwen38_forward.cuh.  With --state-trace every
             extended vector is prefixed with "pos<P>." (add_extended() at
             src/qwen38_forward.cuh:205-209), e.g. "pos39.layer7.hc_attn_norm".

Usage
-----
    python3 tools/issue1_compare_internals.py --cpu CPU.bin --warp WARP.bin \
        --position 39 [--json out.json] [--threshold 1e-2] [--verbose] [--audit]
    python3 tools/issue1_compare_internals.py --self-test

The CPU<->engine name mapping is the PAIRS table below.  Every rule records the
evidence (CPU tensor-name site in qwen38_forced_cpu_trace.cpp / llama.cpp graph,
engine add_extended() line in src/qwen38_forward.cuh) and a confidence:

    確実 (sure)  - name and role agree, or the engine source proves the identity.
    推測 (guess) - element count and pipeline stage agree but the names differ,
                   and the llama.cpp tensor semantics cannot be read from this
                   repo (there is no llama.cpp checkout here).

Pairs whose element counts differ are excluded as 比較不可 (size mismatch) and
reported in the "skipped" section; the same happens when one side does not emit
the vector at all.  Note that qwen38_forced_cpu_trace.cpp only records the full
internals for its TRACE_POSITION argument - other positions keep l_last, the
layer-8 routing series and the layer-7/35 QSA history (see
qwen38_forced_cpu_trace.cpp:186-189) - so a position 0 comparison against a
trace captured with TRACE_POSITION=39 can only score l_last unless the CPU
oracle is re-run with TRACE_POSITION=0.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"WDPTRC1\x00"
LAYERS = 48
DEFAULT_THRESHOLD = 1e-2
STRICT_EXIT = 3

ALL_LAYERS = tuple(range(LAYERS))
# QSA (indexed attention) sits on layers 3, 7, ... 47, every other layer runs
# GDN (linear attention).  Verified against the CPU trace itself: Qcur-<L> /
# qsa_k_store-<L> exist exactly for L % 4 == 3 (issue1_artifacts/
# position39_cpu_reference_hc_trace.bin), and the engine trace agrees
# (issue1_artifacts/warodecode-state-position-36-gdn.bin).
QSA_LAYERS = tuple(L for L in ALL_LAYERS if L % 4 == 3)
GDN_LAYERS = tuple(L for L in ALL_LAYERS if L % 4 != 3)
LAYER_SETS = {
    "none": (),
    "all": ALL_LAYERS,
    "qsa": QSA_LAYERS,
    "gdn": GDN_LAYERS,
    "layer1": (1,),   # PLE layer
    "layer7": (7,),   # only QSA layer with a pre-gate F32 trace
}

# Nominal execution order inside one step.  Used only to answer "which block
# diverges first"; PLE is listed after the attention blocks although it runs
# before them (it only exists on layer 1).
BLOCK_ORDER = (
    "embedding", "hc_attn", "gdn", "qsa", "attention_output",
    "after_attention_residual", "hc_ffn", "moe", "ple", "output", "final",
)
BLOCK_RANK = {name: rank for rank, name in enumerate(BLOCK_ORDER)}


@dataclass(frozen=True)
class Pair:
    """One CPU vector <-> one engine vector."""

    block: str
    layers: str                        # key into LAYER_SETS
    cpu: str                           # template, {pos} and {layer}
    warp: str                          # template, {pos} and {layer}
    confidence: str                    # 確実 / 推測
    cpu_ref: str
    warp_ref: str
    cpu_occurrence: int | None = None  # None: first size-compatible occurrence
    note: str = ""


# ---------------------------------------------------------------------------
# CPU <-> engine mapping.  Engine references are "file:line" of the
# trace->add_extended()/add_state_summary() call that emits the vector.
# ---------------------------------------------------------------------------
PAIRS: tuple[Pair, ...] = (
    # -- embedding ----------------------------------------------------------
    Pair("embedding", "none", "pos{pos}.hc_init", "embedding.hc_init", "確実",
         "qwen38_forced_cpu_trace.cpp:80 (trace_wanted hc_init), :202 name",
         "src/qwen38_forward.cuh:337 (copy of state_.residual() right after the "
         "token embedding was broadcast to hc_count copies)",
         note="both are the wide HC residual (hidden*hc_count = 10240) before layer 0"),

    # -- HC attention mix (1st hc_norm/hc_gate/hc_mixed/hc_inject of a layer) --
    Pair("hc_attn", "all", "pos{pos}.hc_norm-{layer}", "layer{layer}.hc_attn_norm", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_norm); 1st occurrence in the graph",
         "src/qwen38_forward.cuh:743 (st.hc_attn.xn(), the RMS-normed HC residual)",
         cpu_occurrence=0,
         note="the CPU trace stores hc_norm twice per layer; occurrence 0 is the "
              "attention side (file order per layer: hc_norm,hc_gate,hc_mixed,<attn>,"
              "hc_inject,hc_combine,hc_norm,hc_gate,hc_mixed,<moe>,hc_inject,l_last)"),
    Pair("hc_attn", "all", "pos{pos}.hc_gate-{layer}", "layer{layer}.hc_attn_gate", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_gate); 1st occurrence",
         "src/qwen38_forward.cuh:748 (st.hc_attn.gate(), pre-sigmoid gate projection)",
         cpu_occurrence=0),
    Pair("hc_attn", "all", "pos{pos}.hc_mixed-{layer}", "layer{layer}.hc_attn_mix", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_mixed); 1st occurrence",
         "src/qwen38_forward.cuh:753 (cur = HC mix output = attention block input)",
         cpu_occurrence=0,
         note="hidden-sized (2560), not the 10240 HC residual"),
    Pair("hc_attn", "all", "pos{pos}.hc_inject-{layer}", "layer{layer}.hc_attn_inject", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_inject); 1st occurrence",
         "src/qwen38_forward.cuh:756 (st.hc_attn.inject(), the hc_count values added by "
         "run_qwen38_hc_combine at src/qwen38_forward.cuh:1561)",
         cpu_occurrence=0, note="hc_count == 4"),

    # -- GDN (linear attention) ---------------------------------------------
    Pair("gdn", "gdn", "pos{pos}.linear_attn_qkv_mixed-{layer}", "layer{layer}.gdn.qkv_full", "確実",
         "qwen38_forced_cpu_trace.cpp:85 (linear_attn_qkv_mixed)",
         "src/qwen38_forward.cuh:1467 (st.gdn_scratch->qkv(), the w.qkv projection fed "
         "to q38_gdn_conv_kernel at src/qwen38_gdn.cuh:780)"),
    Pair("gdn", "gdn", "pos{pos}.conv_output_silu-{layer}", "layer{layer}.gdn.conv_output_postnorm_full", "確実",
         "qwen38_forced_cpu_trace.cpp:87 (conv_output_silu)",
         "src/qwen38_forward.cuh:1469 (conv_out; q38_gdn_conv_kernel applies SiLU at "
         "src/qwen38_gdn.cuh:361)"),
    Pair("gdn", "gdn", "pos{pos}.q_conv_predelta-{layer}", "layer{layer}.gdn.q_conv_postnorm", "推測",
         "qwen38_forced_cpu_trace.cpp:91 (q_conv_predelta = q conv output before the "
         "delta rule, i.e. after the per-head norm)",
         "src/qwen38_forward.cuh:1471 (conv_out[0:key_dim] after q38_l2_norm_heads_kernel "
         "at src/qwen38_gdn.cuh:788)",
         note="names differ; same 2048 span and same pipeline stage"),
    Pair("gdn", "gdn", "pos{pos}.k_conv_predelta-{layer}", "layer{layer}.gdn.k_conv_postnorm", "推測",
         "qwen38_forced_cpu_trace.cpp:91 (k_conv_predelta)",
         "src/qwen38_forward.cuh:1473 (conv_out[key_dim:2*key_dim] after the per-head norm)"),
    Pair("gdn", "gdn", "pos{pos}.v_conv_predelta-{layer}", "layer{layer}.gdn.v_conv", "推測",
         "qwen38_forced_cpu_trace.cpp:91 (v_conv_predelta; no norm is applied to v)",
         "src/qwen38_forward.cuh:1475 (conv_out[2*key_dim:] = v_conv)"),
    Pair("gdn", "gdn", "pos{pos}.z-{layer}", "layer{layer}.gdn.z_full", "確実",
         "qwen38_forced_cpu_trace.cpp:85 (z)",
         "src/qwen38_forward.cuh:1477 (st.gdn_scratch->z(), the w.z projection)"),
    Pair("gdn", "gdn", "pos{pos}.beta_sigmoid-{layer}", "layer{layer}.gdn.beta_full", "確実",
         "qwen38_forced_cpu_trace.cpp:86 (beta_sigmoid)",
         "src/qwen38_forward.cuh:1479 (scratch.beta() AFTER q38_gdn_gate_params_kernel "
         "overwrote it with sigmoid(beta) in place, src/qwen38_gdn.cuh:603-604)",
         note="the engine keeps no raw pre-sigmoid beta"),
    Pair("gdn", "gdn", "pos{pos}.alpha-{layer}", "layer{layer}.gdn.alpha_full", "確実",
         "qwen38_forced_cpu_trace.cpp:86 (alpha)",
         "src/qwen38_forward.cuh:1481 (st.gdn_scratch->alpha(), raw w.alpha projection)"),
    Pair("gdn", "gdn", "pos{pos}.gate-{layer}", "layer{layer}.gdn.gate_full", "確実",
         "qwen38_forced_cpu_trace.cpp:86 (gate)",
         "src/qwen38_forward.cuh:1483 (softplus(alpha+dt)*a, src/qwen38_gdn.cuh:605-606)"),
    Pair("gdn", "gdn", "pos{pos}.gdn_core_output-{layer}", "layer{layer}.gdn.core_output_full", "確実",
         "qwen38_forced_cpu_trace.cpp:87 (gdn_core_output)",
         "src/qwen38_forward.cuh:1485 (st.gdn_scratch->core_out(), the delta-rule output "
         "before the gated norm)"),
    Pair("gdn", "gdn", "pos{pos}.final_output-{layer}", "layer{layer}.gdn.norm_output_full", "推測",
         "qwen38_forced_cpu_trace.cpp:87 (final_output, value_dim sized: the normalized "
         "core output before the output projection)",
         "src/qwen38_forward.cuh:1487 (st.gdn_scratch->norm_out() from "
         "q38_gdn_gated_rmsnorm_kernel, src/qwen38_gdn.cuh:803-805)",
         note="names differ; 6144 span + stage match.  The engine's "
              "'final_output_full' (2560) is the output projection result and maps to "
              "the CPU 'linear_attn_out' instead"),
    Pair("gdn", "gdn", "pos{pos}.new_state-{layer}", "layer{layer}.gdn.recurrent_after_full", "推測",
         "qwen38_forced_cpu_trace.cpp:86 (new_state; exported as a full vector only for "
         "layers 0,1,2,4,5,6,8,9,10 via summarize_only(), qwen38_forced_cpu_trace.cpp:101-110)",
         "src/qwen38_forward.cuh:1428 (st.gdn_state->state() after run_qwen38_gdn; only for "
         "capture_full_gdn layers, src/qwen38_forward.cuh:1316)",
         note="both are the post-update recurrent state (48*128*128).  CPU "
              "'state_predelta-<L>' is NOT mapped: it is unclear from this repo whether "
              "it is pre- or post-delta"),
    Pair("gdn", "gdn", "pos{pos}.conv_state_at-{layer}", "layer{layer}.gdn.conv_before", "推測",
         "qwen38_forced_cpu_trace.cpp:85 (conv_state_at)",
         "src/qwen38_forward.cuh:1329 (st.gdn_state->conv_state() before the conv update)",
         cpu_occurrence=None,
         note="layer 1 emits conv_state_at twice (92160 then 30720): the 92160 entry is the "
              "PLE convolution state.  The rule takes the size-compatible occurrence, i.e. "
              "the 30720 GDN state"),

    # -- QSA (indexed attention) --------------------------------------------
    Pair("qsa", "qsa", "pos{pos}.Qcur-{layer}", "layer{layer}.qsa.query", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (Qcur)",
         "src/qwen38_forward.cuh:1251 (st.qsa_scratch->q(), the query projection fed to the "
         "attention kernel together with the gate)"),
    Pair("qsa", "qsa", "pos{pos}.attn_gated-{layer}", "layer{layer}.qsa.attention_context", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (attn_gated = gate applied)",
         "src/qwen38_forward.cuh:1260 (st.qsa_scratch->attn() is written by "
         "q38_qsa_gate_kernel from pre_gate_trace and gate, see :1233-1234)"),
    Pair("qsa", "layer7", "pos{pos}.attn_pregate-{layer}", "layer{layer}.qsa.attention_pre_gate_f32", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (attn_pregate)",
         "src/qwen38_forward.cuh:1263 (only il == 7 keeps the F32 pre-gate trace; the kernel "
         "writes it at src/qwen38_qsa.cuh:601)",
         note="other QSA layers do not export a pre-gate vector on the engine side"),
    Pair("qsa", "qsa", "pos{pos}.Kcur-{layer}", "layer{layer}.qsa.key_projection", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (Kcur)",
         "src/qwen38_forward.cuh:1255 (st.qsa_scratch->k(), kv_heads*key_length)"),
    Pair("qsa", "qsa", "pos{pos}.Vcur-{layer}", "layer{layer}.qsa.value_projection", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (Vcur)",
         "src/qwen38_forward.cuh:1257 (st.qsa_scratch->v())"),
    Pair("qsa", "qsa", "pos{pos}.qsa_k_store-{layer}", "layer{layer}.qsa.key_cache_position", "推測",
         "qwen38_forced_cpu_trace.cpp:89 (qsa_k_store = the K written at this position)",
         "src/qwen38_forward.cuh:1274 (k_cache() + pos*kv_n, the slot written at pos)",
         note="if the cache stores the projected K unchanged this equals "
              "'key_projection'; a difference localizes the store path"),
    Pair("qsa", "qsa", "pos{pos}.qsa_v_store-{layer}", "layer{layer}.qsa.value_cache_position", "推測",
         "qwen38_forced_cpu_trace.cpp:89 (qsa_v_store)",
         "src/qwen38_forward.cuh:1276 (v_cache() + pos*kv_n)"),
    Pair("qsa", "qsa", "pos{pos}.indexer_k_raw-{layer}", "layer{layer}.qsa.index_key_position", "推測",
         "qwen38_forced_cpu_trace.cpp:87 (indexer_k_raw)",
         "src/qwen38_forward.cuh:1278 (qsa_state->index_raw() + pos*idx_n)"),
    Pair("qsa", "qsa", "pos{pos}.indexer_q-{layer}", "layer{layer}.qsa.index_query", "推測",
         "qwen38_forced_cpu_trace.cpp:88 (indexer_q)",
         "src/qwen38_forward.cuh:1292 (only emitted when full_blocks > block_budget, "
         "src/qwen38_forward.cuh:1291)",
         note="on short contexts the engine skips the indexer query entirely, so the pair is "
              "reported as missing instead of compared"),

    # -- attention block output ---------------------------------------------
    Pair("attention_output", "qsa", "pos{pos}.attn_output-{layer}", "layer{layer}.attention_output", "確実",
         "qwen38_forced_cpu_trace.cpp:82 (attn_output)",
         "src/qwen38_forward.cuh:1526 (attn_out, hidden sized)"),
    Pair("attention_output", "gdn", "pos{pos}.linear_attn_out-{layer}", "layer{layer}.attention_output", "確実",
         "qwen38_forced_cpu_trace.cpp:87 (linear_attn_out)",
         "src/qwen38_forward.cuh:1526 (attn_out == st.gdn_scratch->final_out() for GDN layers; "
         "src/qwen38_forward.cuh:1489 copies the same buffer)"),

    # -- HC residual update after attention ---------------------------------
    Pair("after_attention_residual", "all", "pos{pos}.hc_combine-{layer}", "layer{layer}.after_attention_residual", "確実",
         "qwen38_forced_cpu_trace.cpp:82 (hc_combine; stored once per layer)",
         "src/qwen38_forward.cuh:1604 (state_.residual() right after run_qwen38_hc_combine, "
         "src/qwen38_forward.cuh:1561)"),

    # -- HC FFN mix (2nd hc_norm/hc_gate/hc_mixed/hc_inject of a layer) -------
    Pair("hc_ffn", "all", "pos{pos}.hc_norm-{layer}", "layer{layer}.hc_ffn_norm", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_norm); 2nd occurrence",
         "src/qwen38_forward.cuh:1669 (st.hc_ffn.xn())",
         cpu_occurrence=1, note="occurrence 1 of the CPU name (the FFN side)"),
    Pair("hc_ffn", "all", "pos{pos}.hc_gate-{layer}", "layer{layer}.hc_ffn_gate", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_gate); 2nd occurrence",
         "src/qwen38_forward.cuh:1674 (st.hc_ffn.gate())",
         cpu_occurrence=1),
    Pair("hc_ffn", "all", "pos{pos}.hc_mixed-{layer}", "layer{layer}.hc_ffn_mix", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_mixed); 2nd occurrence",
         "src/qwen38_forward.cuh:1679 (cur = HC mix output = MoE router input)",
         cpu_occurrence=1),
    Pair("hc_ffn", "all", "pos{pos}.hc_inject-{layer}", "layer{layer}.hc_ffn_inject", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_inject); 2nd occurrence",
         "src/qwen38_forward.cuh:1682 (st.hc_ffn.inject(), applied by the FFN HC combine)",
         cpu_occurrence=1),

    # -- MoE ----------------------------------------------------------------
    Pair("moe", "all", "pos{pos}.ffn_moe_logits-{layer}", "layer{layer}.ffn.router_logits", "確実",
         "qwen38_forced_cpu_trace.cpp:92 (ffn_moe_logits, n_experts)",
         "src/qwen38_forward.cuh:1803 (st.moe.d_logits())"),
    Pair("moe", "all", "pos{pos}.ffn_moe_weights_norm-{layer}", "layer{layer}.ffn.expert_weights", "確実",
         "qwen38_forced_cpu_trace.cpp:93 (ffn_moe_weights_norm = softmax + top-k renorm)",
         "src/qwen38_forward.cuh:1814 (st.moe.h_weights())",
         note="docs/ISSUE1_FINDINGS_20261012.md section 6: the engine matches the normalized "
              "CPU weights; the raw 'ffn_moe_weights-<L>' (sum 0.148) is NOT the right "
              "counterpart"),
    Pair("moe", "all", "pos{pos}.ffn_moe_out-{layer}", "layer{layer}.ffn.ffn_moe_out", "推測",
         "qwen38_forced_cpu_trace.cpp:82 (ffn_moe_out)",
         "src/qwen38_forward.cuh:1815 (st.moe_out right after the routed experts)",
         note="llama.cpp calls the routed result ffn_moe_out and the routed+shared sum ffn_out; "
              "the engine reuses one buffer and re-copies it as 'ffn_output' after the shared "
              "expert (src/qwen38_forward.cuh:1962)"),
    Pair("moe", "all", "pos{pos}.ffn_moe_down-{layer}", "layer{layer}.ffn.selected_expert_outputs", "推測",
         "qwen38_forced_cpu_trace.cpp:82 (ffn_moe_down, n_expert_used*hidden)",
         "src/qwen38_forward.cuh:1792 (selected_output_trace, experts_used*hidden)",
         note="per-expert down projections flattened in top-k order; a reordering of the "
              "selected experts shows up as a large NRMSE here"),
    Pair("moe", "all", "pos{pos}.ffn_shexp-{layer}", "layer{layer}.ffn_shexp", "確実",
         "qwen38_forced_cpu_trace.cpp:82 (ffn_shexp)",
         "src/qwen38_forward.cuh:1958 (st.moe.d_down(), shared expert output)"),
    Pair("moe", "all", "pos{pos}.shared_expert_gate_sigmoid-{layer}", "layer{layer}.shared_expert_gate_sigmoid", "確実",
         "qwen38_forced_cpu_trace.cpp:83 (shared_expert_gate_sigmoid)",
         "src/qwen38_forward.cuh:1959 (st.moe.d_shared_gate(), 1 value)"),
    Pair("moe", "all", "pos{pos}.ffn_shexp_gated-{layer}", "layer{layer}.ffn_shexp_gated", "確実",
         "qwen38_forced_cpu_trace.cpp:83 (ffn_shexp_gated)",
         "src/qwen38_forward.cuh:1960 (shared_output * shared_gate)"),
    Pair("moe", "all", "pos{pos}.ffn_out-{layer}", "layer{layer}.ffn_output", "確実",
         "qwen38_forced_cpu_trace.cpp:83 (ffn_out = routed + shared)",
         "src/qwen38_forward.cuh:1962 (st.moe_out after the shared expert was added)"),

    # -- layer residual ------------------------------------------------------
    Pair("output", "all", "pos{pos}.l_last-{layer}", "layer{layer}.output", "確実",
         "qwen38_forced_cpu_trace.cpp:82 (l_last = residual stream after layer L)",
         "src/qwen38_forward.cuh:2037 (state_.residual() at the end of layer L)",
         note="the CPU trace keeps l_last for every position even outside TRACE_POSITION "
              "(qwen38_forced_cpu_trace.cpp:186-189), so this is the one family that can be "
              "compared for every position"),

    # -- PLE (layer 1) -------------------------------------------------------
    Pair("ple", "layer1", "pos{pos}.ple_gate-{layer}", "layer{layer}.ple_gate", "確実",
         "qwen38_forced_cpu_trace.cpp:84 (ple_gate)",
         "src/qwen38_forward.cuh:472 (st.ple_scratch->gate(), hc_count values)"),
    Pair("ple", "layer1", "pos{pos}.ple_conv_out-{layer}", "layer{layer}.ple_conv_out", "確実",
         "qwen38_forced_cpu_trace.cpp:84 (ple_conv_out)",
         "src/qwen38_forward.cuh:470 (st.ple_scratch->conv_out(), hc_dim values)"),

    # -- final head ----------------------------------------------------------
    Pair("final", "none", "pos{pos}.hc_norm", "final.hc_norm", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_norm with no layer suffix = the final HC norm)",
         "src/qwen38_forward.cuh:2089 (state_.head_hc().xn())"),
    Pair("final", "none", "pos{pos}.hc_gate", "final.hc_gate", "確実",
         "qwen38_forced_cpu_trace.cpp:81,96 (hc_gate with no layer suffix)",
         "src/qwen38_forward.cuh:2093 (state_.head_hc().gate())"),
    Pair("final", "none", "pos{pos}.result_norm", "final.hc_mixed", "推測",
         "qwen38_forced_cpu_trace.cpp:94 (result_norm = the tensor fed to the lm_head)",
         "src/qwen38_forward.cuh:2097 (head = HC mix output, hidden sized)",
         note="the engine stores the same value again as 'final.head_input' "
              "(src/qwen38_forward.cuh:2100), so a mismatch here also shows there"),
    Pair("final", "none", "pos{pos}.final.logits", "final.logits", "確実",
         "qwen38_forced_cpu_trace.cpp:332-342 (capture_logits writes pos<P>.final.logits)",
         "src/qwen38_forward.cuh:2107 (the fp32 logits buffer before argmax)"),
)

# Vectors that cannot be compared (or would duplicate another comparison).
NON_COMPARABLE: tuple[tuple[str, str, str], ...] = (
    ("pos<P>.model.input_embed", "-",
     "the engine never exports the 2560 head_hidden embedding; it goes straight from the "
     "q8_0 gather into the hc_count broadcast (src/qwen38_forward.cuh:343-346)"),
    ("pos<P>.ple_embd", "-",
     "the engine does not trace the PLE table row gathered for the token"),
    ("pos<P>.ple_gated_value-1", "-",
     "the engine applies the PLE gate inside run_qwen38_ple and never exports the gated "
     "value; only ple_pre/post_residual and ple_conv_out/ple_gate exist"),
    ("pos<P>.beta-<L>", "-",
     "the engine overwrites beta with sigmoid(beta) in place (src/qwen38_gdn.cuh:603-604), "
     "so only the post-sigmoid value exists (gdn.beta_full); compare 'beta_sigmoid-<L>'"),
    ("pos<P>.ffn_moe_weights-<L>", "-",
     "raw (non-normalized) softmax weights; the engine keeps the normalized ones "
     "(docs/ISSUE1_FINDINGS_20261012.md section 6), so this is the known raw/normalized "
     "trap, not a defect signal"),
    ("pos<P>.ffn_moe_topk-<L>", "layer{L}.ffn.expert_ids",
     "the engine emits expert ids as an integer vector in the engine JSON, not in the "
     "WDPTRC1 file (Qwen38Trace::add_integer_vector, src/qwen38_forward.cuh:227-231)"),
    ("pos<P>.shared_expert_gate-<L>", "-",
     "the engine only exports the post-sigmoid shared gate"),
    ("pos<P>.state_predelta-<L>", "-",
     "the engine does not name the pre-delta recurrent state"),
    ("pos<P>.result_output", "final.logits",
     "identical comparison to 'pos<P>.final.logits' (llama.cpp's lm_head tensor vs the "
     "explicitly captured logits); mapped once to avoid double counting"),
    ("pos<P>.qsa_k_cache-<L>", "layer{L}.qsa.key_cache_prefix",
     "size mismatch (比較不可): the CPU exports the whole cache array (131072 = max_ctx "
     "slots), the engine only the written prefix ((pos+1)*kv_n).  Compare prefix slices "
     "instead of the raw vectors"),
    ("pos<P>.qsa_v_cache-<L>", "layer{L}.qsa.value_cache_prefix", "same as qsa_k_cache"),
    ("pos<P>.indexer_k-<L>", "-", "the engine has no corresponding trace vector"),
    ("pos<P>.indexer_k_pooled-<L>", "-", "the engine has no corresponding trace vector"),
    ("pos<P>.indexer_score-<L>", "-", "the engine has no corresponding trace vector"),
    ("pos<P>.attn_pregate-<L>", "-",
     "only layer 7 is exported by the engine (layer7.qsa.attention_pre_gate_f32, "
     "src/qwen38_forward.cuh:1263); the other 11 QSA layers cannot be checked at this "
     "granularity and must be localised through attn_gated-<L> / attention_context"),
    ("pos<P>.q_conv-8", "-",
     "layer-8-only raw conv slice; the engine exposes the same 10240 buffer as "
     "gdn.conv_output_postnorm_full, which is already compared through "
     "conv_output_silu-8.  Consider it a duplicate span, not a missing vector"),
    ("pos<P>.k_conv-8", "-", "same as q_conv-8"),
    ("pos<P>.conv_state_at-<L> (2nd occurrence)", "layer{L}.gdn.conv_before",
     "layer 1 stores the state twice (92160 PLE / 30720 GDN); the rule picks the "
     "size-compatible occurrence"),
    ("-", "final.hc_gamma_half", "engine-only: the HC head norm weights (rt_.hc_head.norm.data)"),
    ("-", "final.hc_gamma_f32", "engine-only: F32 copy of the HC head norm weights"),
    ("-", "final.head_input",
     "duplicate of final.hc_mixed (same head pointer, src/qwen38_forward.cuh:2100)"),
    ("-", "layer{L}.input_residual", "engine-only: equals the previous layer's "
     "'output' pair (l_last-<L-1> <-> layer<L-1>.output)"),
    ("-", "layer{L}.residual_after_ple", "engine-only: equals 'layer{L}.output' when the "
     "layer is not the PLE layer"),
    ("-", "layer{L}.gdn.input", "engine-only duplicate of layer{L}.hc_attn_mix (same cur pointer)"),
    ("-", "layer{L}.gdn.recurrent_before_full", "pre-update recurrent state; the CPU export "
     "of the same slot would be 'new_state-<L>' one position earlier, which the "
     "position-indexed table cannot express"),
    ("-", "layer{L}.gdn.conv_after", "post-update convolution state; compare it against "
     "'conv_state_at-<L>' of the NEXT position, not of this one"),
    ("-", "layer{L}.gdn.final_output_full",
     "duplicate of layer{L}.attention_output for GDN layers: both copy attn_out = "
     "st.gdn_scratch->final_out() (src/qwen38_forward.cuh:1489 and :1526), so the "
     "comparison runs once through linear_attn_out-<L>"),
    ("-", "layer{L}.qsa.gate", "the CPU trace does not export the QSA gate projection "
     "(only the pre-gate and gated attention values)"),
    ("-", "layer{L}.ffn.router_input", "engine-only duplicate of layer{L}.hc_ffn_mix "
     "(emitted for layer 8 only, src/qwen38_forward.cuh:1799-1802)"),
    ("-", "layer{L}.ple_history_before", "the CPU trace does not export the PLE history "
     "buffer (92160 values)"),
    ("-", "layer{L}.ple_history_after", "same as ple_history_before"),
    ("-", "layer{L}.ple_pre_residual", "equals 'l_last-0' (the residual entering layer 1); "
     "the PLE layer 1 pair already covers l_last-0/last-1"),
    ("-", "layer{L}.ple_post_residual", "inside layer 1; folded into l_last-1"),
    ("-", "layer0.hc_attn_*", "engine-only profiling vectors from the layer-0 HC Q8_0 replay "
     "(src/qwen38_forward.cuh:490-710); no CPU counterpart"),
    ("-", "layer2.hc_attn_*", "engine-only profiling vectors from the layer-2 HC projection "
     "probe (src/qwen38_forward.cuh:536-556); no CPU counterpart"),
    ("-", "layer{L}.*_cpu_override", "engine-only injection bookkeeping (present only when a "
     "debug override was configured)"),
    ("-", "final.*_cpu_override", "engine-only injection bookkeeping (final residual override)"),
)


# ---------------------------------------------------------------------------
# WDPTRC1 container
# ---------------------------------------------------------------------------
@dataclass
class Vector:
    name: str
    values: np.ndarray


def read_wdptrc1(path: str | Path) -> tuple[int, list[Vector]]:
    """Parse a WDPTRC1 file into (version, vectors).  Raises ValueError on junk."""
    raw = Path(path).read_bytes()
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a WDPTRC1 file")
    version, count = struct.unpack_from("<II", raw, 8)
    vectors: list[Vector] = []
    off = 16
    for index in range(count):
        if off + 12 > len(raw):
            raise ValueError(f"{path}: truncated header of entry {index}")
        name_len, value_count = struct.unpack_from("<IQ", raw, off)
        off += 12
        if off + name_len + value_count * 4 > len(raw):
            raise ValueError(f"{path}: truncated payload of entry {index}")
        name = raw[off:off + name_len].decode("utf-8", "replace")
        off += name_len
        values = np.frombuffer(raw, dtype="<f4", count=value_count, offset=off).copy()
        off += value_count * 4
        vectors.append(Vector(name, values))
    if off != len(raw):
        print(f"warning: {path}: {len(raw) - off} trailing bytes after {count} entries",
              file=sys.stderr)
    return version, vectors


def write_wdptrc1(path: str | Path, vectors: list[Vector], version: int = 1) -> None:
    payload = bytearray()
    for vector in vectors:
        name = vector.name.encode("utf-8")
        payload += struct.pack("<IQ", len(name), len(vector.values))
        payload += name
        payload += np.asarray(vector.values, dtype="<f4").tobytes()
    Path(path).write_bytes(MAGIC + struct.pack("<II", version, len(vectors)) + bytes(payload))


def index_by_name(vectors: list[Vector]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for vector in vectors:
        out.setdefault(vector.name, []).append(vector.values)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@dataclass
class Comparison:
    pair: Pair
    layer: int
    cpu_name: str
    warp_name: str
    n: int
    nrmse: float
    max_abs: float
    cpu_rms: float
    warp_rms: float

    def sort_key(self) -> tuple[int, int, str]:
        if self.pair.block == "embedding":
            layer = -1
        elif self.pair.block == "final":
            layer = LAYERS
        else:
            layer = self.layer
        return (layer, BLOCK_RANK[self.pair.block], self.cpu_name)


@dataclass
class Skip:
    pair: Pair
    layer: int
    cpu_name: str
    warp_name: str
    reason: str
    detail: str

    def sort_key(self) -> tuple[int, int, str]:
        return (self.layer, BLOCK_RANK[self.pair.block], self.cpu_name)


def format_name(template: str, pos: int, layer: int | None) -> str:
    return template.format(pos=pos, layer=layer)


def warp_candidates(name: str, pos: int) -> list[str]:
    """State traces prefix every name with "pos<P>."; keep a bare fallback."""
    return [f"pos{pos}.{name}", name]


def compare_pair(pair: Pair, layer: int, pos: int, cpu_index, warp_index) -> Comparison | Skip:
    cpu_name = format_name(pair.cpu, pos, layer)
    warp_name = format_name(pair.warp, pos, layer)
    cpu_list = cpu_index.get(cpu_name, [])
    warp_list: list[np.ndarray] = []
    for name in warp_candidates(warp_name, pos):
        warp_list.extend(warp_index.get(name, []))

    if not cpu_list:
        return Skip(pair, layer, cpu_name, warp_name, "cpu 側に名前なし",
                    "CPU トレースは TRACE_POSITION 以外では l_last / layer8 route / "
                    "qsa layer7,35 history のみを記録する "
                    "(qwen38_forced_cpu_trace.cpp:186-189)")
    if not warp_list:
        return Skip(pair, layer, cpu_name, warp_name, "warp 側に名前なし",
                    "このエンジン版はこのベクトルを出力していない（または --state-trace でない）")

    if pair.cpu_occurrence is not None:
        if pair.cpu_occurrence >= len(cpu_list):
            return Skip(pair, layer, cpu_name, warp_name, "cpu 側の出現が不足",
                        f"occurrence {pair.cpu_occurrence} を要求したが {len(cpu_list)} 個しかない")
        cpu_values = cpu_list[pair.cpu_occurrence]
    else:
        cpu_values = next(
            (candidate for candidate in cpu_list
             if any(candidate.shape[0] == other.shape[0] for other in warp_list)),
            cpu_list[0])

    warp_values = next(
        (candidate for candidate in warp_list if candidate.shape[0] == cpu_values.shape[0]),
        warp_list[0])

    if cpu_values.shape[0] != warp_values.shape[0]:
        return Skip(pair, layer, cpu_name, warp_name, "要素数不一致（比較不可）",
                    f"cpu={cpu_values.shape[0]} warp={warp_values.shape[0]}")
    if cpu_values.size == 0:
        return Skip(pair, layer, cpu_name, warp_name, "要素数不一致（比較不可）", "空ベクトル")

    nrmse, max_abs, cpu_rms, warp_rms = difference_stats(cpu_values, warp_values)
    return Comparison(pair, layer, cpu_name, warp_name, int(cpu_values.shape[0]),
                      nrmse, max_abs, cpu_rms, warp_rms)


def difference_stats(cpu_values: np.ndarray, warp_values: np.ndarray) -> tuple[float, float, float, float]:
    """NRMSE = ||warp-cpu||2 / ||cpu||2, plus max|d|, cpu RMS and warp RMS."""
    cpu = cpu_values.astype(np.float64, copy=False)
    warp = warp_values.astype(np.float64, copy=False)
    diff = warp - cpu
    d2 = float(np.dot(diff, diff))
    c2 = float(np.dot(cpu, cpu))
    nrmse = math.sqrt(d2 / c2) if c2 > 0.0 else (0.0 if d2 == 0.0 else math.inf)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    cpu_rms = math.sqrt(c2 / cpu.size)
    warp_rms = math.sqrt(float(np.dot(warp, warp)) / warp.size)
    return nrmse, max_abs, cpu_rms, warp_rms


def compare_files(cpu_path: str | Path, warp_path: str | Path, pos: int) -> dict:
    cpu_version, cpu_vectors = read_wdptrc1(cpu_path)
    warp_version, warp_vectors = read_wdptrc1(warp_path)
    cpu_index = index_by_name(cpu_vectors)
    warp_index = index_by_name(warp_vectors)

    comparisons: list[Comparison] = []
    skips: list[Skip] = []
    for pair in PAIRS:
        layers = LAYER_SETS[pair.layers] or (-1,)
        for layer in layers:
            outcome = compare_pair(pair, layer, pos, cpu_index, warp_index)
            (comparisons if isinstance(outcome, Comparison) else skips).append(outcome)
    comparisons.sort(key=Comparison.sort_key)
    skips.sort(key=Skip.sort_key)
    return {
        "position": pos,
        "cpu_path": str(cpu_path),
        "warp_path": str(warp_path),
        "cpu_version": cpu_version,
        "warp_version": warp_version,
        "cpu_vector_count": len(cpu_vectors),
        "warp_vector_count": len(warp_vectors),
        "comparisons": comparisons,
        "skips": skips,
        "cpu_index": cpu_index,
        "warp_index": warp_index,
    }


def block_summary(comparisons: list[Comparison]) -> list[dict]:
    buckets: dict[str, list[Comparison]] = {}
    for comparison in comparisons:
        buckets.setdefault(comparison.pair.block, []).append(comparison)
    summary = []
    for block, entries in buckets.items():
        worst = max(entries, key=lambda item: item.nrmse)
        summary.append({
            "block": block,
            "pairs": len(entries),
            "max_nrmse": worst.nrmse,
            "max_nrmse_layer": worst.layer,
            "max_nrmse_cpu": worst.cpu_name,
            "max_nrmse_warp": worst.warp_name,
            "max_abs": worst.max_abs,
            "cpu_rms": worst.cpu_rms,
            "max_nrmse_confidence": worst.pair.confidence,
        })
    summary.sort(key=lambda item: BLOCK_RANK[item["block"]])
    return summary


def first_exceeding(comparisons: list[Comparison], threshold: float) -> Comparison | None:
    for comparison in sorted(comparisons, key=Comparison.sort_key):
        if comparison.nrmse > threshold:
            return comparison
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(value: float) -> str:
    return "inf" if value == math.inf else f"{value:.6g}"


def print_report(result: dict, threshold: float, verbose: bool) -> None:
    comparisons: list[Comparison] = result["comparisons"]
    skips: list[Skip] = result["skips"]
    blocks = block_summary(comparisons)
    first = first_exceeding(comparisons, threshold)
    exceeding = [item for item in comparisons if item.nrmse > threshold]

    print(f"== WDPTRC1 internal comparison (position {result['position']}, "
          f"threshold {fmt(threshold)}) ==")
    print(f"cpu : {result['cpu_path']}  "
          f"({result['cpu_vector_count']} vectors, version {result['cpu_version']})")
    print(f"warp: {result['warp_path']}  "
          f"({result['warp_vector_count']} vectors, version {result['warp_version']})")
    print(f"compared {len(comparisons)} pairs / skipped {len(skips)}")

    print("\n-- block summary (NRMSE = ||warp-cpu||2 / ||cpu||2) --")
    header = (f"{'':>2}{'block':<26}{'pairs':>6}{'max NRMSE':>12}{'layer':>6}"
              f"{'max|d|':>12}{'cpu RMS':>12}  worst vector")
    print(header)
    print("-" * len(header))
    for item in blocks:
        flag = "!" if item["max_nrmse"] > threshold else " "
        print(f"{flag:>2}{item['block']:<26}{item['pairs']:>6}{fmt(item['max_nrmse']):>12}"
              f"{item['max_nrmse_layer']:>6}{fmt(item['max_abs']):>12}"
              f"{fmt(item['cpu_rms']):>12}  {item['max_nrmse_cpu']} "
              f"[{item['max_nrmse_confidence']}]")
    print("  '!' = this block has at least one pair above the threshold")

    if exceeding:
        print(f"\n-- {len(exceeding)} pair(s) above the threshold, in execution order --")
        for item in exceeding:
            print(f"  layer {item.layer:<3} {item.pair.block:<26} nrmse={fmt(item.nrmse):>11} "
                  f"max|d|={fmt(item.max_abs):>11} cpu_rms={fmt(item.cpu_rms):>11} "
                  f"[{item.pair.confidence}]\n"
                  f"      cpu ={item.cpu_name}\n"
                  f"      warp={item.warp_name}")

    if first is None:
        print(f"\nFIRST EXCEEDING BLOCK: none (every compared pair is <= {fmt(threshold)})")
    else:
        print(f"\nFIRST EXCEEDING BLOCK: {first.pair.block} at layer {first.layer} "
              f"(nrmse={fmt(first.nrmse)} > {fmt(threshold)}, "
              f"confidence={first.pair.confidence})")
        print(f"  cpu ={first.cpu_name}")
        print(f"  warp={first.warp_name}")

    if skips:
        counts: dict[str, int] = {}
        for item in skips:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        print("\n-- skipped --")
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")
            if reason == "cpu 側に名前なし" and count > 50:
                print("    hint: the CPU oracle only keeps the full internals at "
                      "TRACE_POSITION; re-run it with TRACE_POSITION=<position> "
                      "(qwen38_forced_cpu_trace.cpp:186-189)")
        print("  all skipped pairs (--verbose) :" if verbose else "  examples:")
        for item in (skips if verbose else skips[:8]):
            print(f"    layer {item.layer:<3} {item.cpu_name} <-> {item.warp_name}: "
                  f"{item.reason} ({item.detail})")

    if verbose:
        print("\n-- every compared pair (execution order) --")
        for item in comparisons:
            flag = "!" if item.nrmse > threshold else " "
            print(f"{flag} layer {item.layer:<3} {item.pair.block:<26} n={item.n:<6} "
                  f"nrmse={fmt(item.nrmse):>11} max|d|={fmt(item.max_abs):>11} "
                  f"cpu_rms={fmt(item.cpu_rms):>11} [{item.pair.confidence}]\n"
                  f"      cpu ={item.cpu_name}\n"
                  f"      warp={item.warp_name}")


def documented_pattern(text: str) -> "re.Pattern[str] | None":
    """Turn a NON_COMPARABLE name template into a matcher.

    "<L>" / "<P>" match any layer/position digits and "*" matches any suffix.
    Entries that mix several names into one string simply never match.
    """
    if text == "-":
        return None
    pattern = re.escape(text)
    for placeholder in ("<L>", "<P>", "{L}", "{P}"):
        pattern = pattern.replace(re.escape(placeholder), r"\d+")
    pattern = pattern.replace(r"\*", ".*")
    return re.compile("^" + pattern + "$")


def audit_names(result: dict) -> None:
    """List vector names that neither the PAIRS rules nor NON_COMPARABLE explain."""
    covered_cpu = {item.cpu_name for item in result["comparisons"]} | \
                  {item.cpu_name for item in result["skips"]}
    covered_warp = {item.warp_name for item in result["comparisons"]} | \
                   {item.warp_name for item in result["skips"]}
    cpu_patterns = [pattern for pattern in
                    (documented_pattern(cpu) for cpu, _, _ in NON_COMPARABLE) if pattern]
    warp_patterns = [pattern for pattern in
                     (documented_pattern(warp) for _, warp, _ in NON_COMPARABLE) if pattern]
    pos = result["position"]
    prefix = f"pos{pos}."
    cpu_unmapped = sorted(
        name for name in result["cpu_index"]
        if prefix in name and name not in covered_cpu
        and not any(pattern.match(name) for pattern in cpu_patterns))
    warp_unmapped = sorted(
        name for name in result["warp_index"]
        if name.startswith(prefix) and name[len(prefix):] not in covered_warp
        and not any(pattern.match(name[len(prefix):]) for pattern in warp_patterns))
    print(f"\n-- mapping audit at position {pos} "
          f"(names with no rule and no NON_COMPARABLE entry) --")
    for label, names in (("cpu", cpu_unmapped), ("warp", warp_unmapped)):
        print(f"{label} vectors with no rule ({len(names)}):")
        for name in names[:40]:
            print(f"  {name}")
        if len(names) > 40:
            print(f"  ... and {len(names) - 40} more")


def to_json(result: dict, threshold: float) -> dict:
    first = first_exceeding(result["comparisons"], threshold)
    return {
        "position": result["position"],
        "threshold": threshold,
        "cpu": {"path": result["cpu_path"], "vectors": result["cpu_vector_count"]},
        "warp": {"path": result["warp_path"], "vectors": result["warp_vector_count"]},
        "blocks": block_summary(result["comparisons"]),
        "first_exceeding": None if first is None else {
            "block": first.pair.block,
            "layer": first.layer,
            "cpu_name": first.cpu_name,
            "warp_name": first.warp_name,
            "confidence": first.pair.confidence,
            "nrmse": first.nrmse,
            "max_abs": first.max_abs,
            "cpu_rms": first.cpu_rms,
        },
        "pairs": [{
            "block": item.pair.block,
            "layer": item.layer,
            "cpu_name": item.cpu_name,
            "warp_name": item.warp_name,
            "confidence": item.pair.confidence,
            "note": item.pair.note,
            "cpu_ref": item.pair.cpu_ref,
            "warp_ref": item.pair.warp_ref,
            "n": item.n,
            "nrmse": item.nrmse,
            "max_abs": item.max_abs,
            "cpu_rms": item.cpu_rms,
            "warp_rms": item.warp_rms,
            "exceeds": item.nrmse > threshold,
        } for item in result["comparisons"]],
        "skipped": [{
            "block": item.pair.block,
            "layer": item.layer,
            "cpu_name": item.cpu_name,
            "warp_name": item.warp_name,
            "reason": item.reason,
            "detail": item.detail,
        } for item in result["skips"]],
        "non_comparable": [{"cpu": cpu, "warp": warp, "why": why}
                           for cpu, warp, why in NON_COMPARABLE],
    }


# ---------------------------------------------------------------------------
# Self-test: synthetic WDPTRC1 pairs, no external files
# ---------------------------------------------------------------------------
def rule_layers(pair: Pair) -> tuple[int, ...]:
    return LAYER_SETS[pair.layers] or (-1,)


def synth_case(pos: int, seed: int, *, warp_noise: float = 0.0,
               noisy: set[tuple[str, int]] | None = None) -> tuple[list[Vector], list[Vector]]:
    """Build a synthetic CPU/engine pair covering every rule in PAIRS.

    Every vector has 8 elements and a payload unique to its (name, occurrence),
    so a wrong name or occurrence resolution cannot pass by accident.  With
    warp_noise > 0 the engine side gets independent noise; noisy limits that to
    the listed (block, layer) keys.
    """
    rng = np.random.default_rng(seed)
    cpu_map: dict[str, list[np.ndarray | None]] = {}
    warp_map: dict[str, np.ndarray] = {}

    for pair in PAIRS:
        for layer in rule_layers(pair):
            cpu_name = format_name(pair.cpu, pos, layer)
            occurrence = pair.cpu_occurrence if pair.cpu_occurrence is not None else 0
            slots = cpu_map.setdefault(cpu_name, [])
            while len(slots) <= occurrence:
                slots.append(None)
            if slots[occurrence] is None:
                slots[occurrence] = rng.normal(size=8).astype(np.float32)
            base = slots[occurrence]
            assert base is not None
            if warp_noise and (noisy is None or (pair.block, layer) in noisy):
                value = (base + rng.normal(size=8).astype(np.float32) * warp_noise).astype(np.float32)
            else:
                value = base.copy()
            warp_name = f"pos{pos}." + format_name(pair.warp, pos, layer)
            warp_map.setdefault(warp_name, value)

    cpu_vectors = [Vector(name, values)
                   for name, slots in cpu_map.items() for values in slots if values is not None]
    warp_vectors = [Vector(name, values) for name, values in warp_map.items()]
    return cpu_vectors, warp_vectors


def run_self_test() -> int:
    pos = 7
    threshold = DEFAULT_THRESHOLD
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"  ok   {message}")
        else:
            print(f"  FAIL {message}")
            failures.append(message)

    with tempfile.TemporaryDirectory(prefix="issue1-compare-selftest-") as tmp:
        tmp_path = Path(tmp)
        expected_pairs = sum(len(rule_layers(pair)) for pair in PAIRS)
        print("self-test: building synthetic WDPTRC1 pairs")

        # 1. identical traces ------------------------------------------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=1)
        cpu_file = tmp_path / "cpu_identical.bin"
        warp_file = tmp_path / "warp_identical.bin"
        write_wdptrc1(cpu_file, cpu_vectors)
        write_wdptrc1(warp_file, warp_vectors)
        result = compare_files(cpu_file, warp_file, pos)
        comparisons = result["comparisons"]
        print(f"[1] identical: {len(comparisons)} compared (expected {expected_pairs}), "
              f"{len(result['skips'])} skipped")
        check(len(comparisons) == expected_pairs, "identical case compares every rule")
        check(all(item.nrmse == 0.0 and item.max_abs == 0.0 for item in comparisons),
              "identical vectors give NRMSE 0 (name + occurrence resolution correct)")
        check(all(item.n == 8 for item in comparisons), "every compared pair has 8 elements")
        check(first_exceeding(comparisons, threshold) is None,
              "identical case has no threshold crossing")
        check(any(item.pair.cpu_occurrence is not None for item in comparisons),
              "duplicate-name occurrence rules were exercised")

        # 2. tiny difference: below threshold --------------------------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=2, warp_noise=1e-4)
        cpu_file2 = tmp_path / "cpu_tiny.bin"
        warp_file2 = tmp_path / "warp_tiny.bin"
        write_wdptrc1(cpu_file2, cpu_vectors)
        write_wdptrc1(warp_file2, warp_vectors)
        tiny = compare_files(cpu_file2, warp_file2, pos)
        tiny_max = max(item.nrmse for item in tiny["comparisons"])
        print(f"[2] tiny diff: max NRMSE {tiny_max:.6g} (threshold {threshold})")
        check(0.0 < tiny_max < threshold, "tiny difference stays below the threshold")
        check(first_exceeding(tiny["comparisons"], threshold) is None,
              "tiny case reports no first-exceeding block")

        # 3. large difference confined to one block --------------------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=3, warp_noise=5e-2, noisy={("moe", 3)})
        cpu_file3 = tmp_path / "cpu_big.bin"
        warp_file3 = tmp_path / "warp_big.bin"
        write_wdptrc1(cpu_file3, cpu_vectors)
        write_wdptrc1(warp_file3, warp_vectors)
        big = compare_files(cpu_file3, warp_file3, pos)
        first = first_exceeding(big["comparisons"], threshold)
        exceeding = [item for item in big["comparisons"] if item.nrmse > threshold]
        print(f"[3] big diff in moe@3: {len(exceeding)} pair(s) above threshold")
        check(first is not None and first.pair.block == "moe" and first.layer == 3,
              "first-exceeding block is moe at layer 3")
        check(exceeding and all(item.layer == 3 and item.pair.block == "moe" for item in exceeding),
              "only the perturbed block exceeds the threshold")
        worst = max(big["comparisons"], key=lambda item: item.nrmse)
        check(worst.nrmse > threshold and worst.pair.block == "moe",
              "block summary picks up the perturbed block")
        json_path = tmp_path / "big.json"
        json_path.write_text(json.dumps(to_json(big, threshold), ensure_ascii=False, indent=2))
        parsed = json.loads(json_path.read_text())
        check(parsed["first_exceeding"]["block"] == "moe"
              and parsed["first_exceeding"]["layer"] == 3,
              "--json carries the first-exceeding block")
        check(len(parsed["pairs"]) == len(big["comparisons"]) and bool(parsed["blocks"]),
              "--json carries every pair and the block table")

        # 4. duplicate occurrence is really occurrence-sensitive -------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=4)
        duplicates = [vector for vector in cpu_vectors if vector.name == f"pos{pos}.hc_norm-5"]
        check(len(duplicates) == 2, "synthetic CPU trace stores hc_norm-5 twice")
        if len(duplicates) == 2:
            first_value = duplicates[0].values.copy()
            duplicates[0].values = duplicates[1].values.copy()
            duplicates[1].values = first_value
        cpu_file4 = tmp_path / "cpu_swap.bin"
        warp_file4 = tmp_path / "warp_swap.bin"
        write_wdptrc1(cpu_file4, cpu_vectors)
        write_wdptrc1(warp_file4, warp_vectors)
        swapped = compare_files(cpu_file4, warp_file4, pos)
        hc5 = [item for item in swapped["comparisons"]
               if item.layer == 5 and item.pair.block in ("hc_attn", "hc_ffn")]
        broken = [item for item in hc5 if item.nrmse > 0.5]
        check(len(hc5) == 8, "layer 5 exposes 8 hc_attn/hc_ffn pairs")
        check(len(broken) == 2
              and {item.cpu_name for item in broken} == {f"pos{pos}.hc_norm-5"}
              and {item.pair.block for item in broken} == {"hc_attn", "hc_ffn"},
              "swapping the two CPU occurrences breaks only the two hc_norm-5 pairs")

        # 5. size mismatch is excluded, not compared -------------------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=5)
        for vector in warp_vectors:
            if vector.name == f"pos{pos}.layer0.hc_attn_norm":
                vector.values = vector.values[:4]
        cpu_file5 = tmp_path / "cpu_size.bin"
        warp_file5 = tmp_path / "warp_size.bin"
        write_wdptrc1(cpu_file5, cpu_vectors)
        write_wdptrc1(warp_file5, warp_vectors)
        sized = compare_files(cpu_file5, warp_file5, pos)
        mismatch = [item for item in sized["skips"] if item.reason == "要素数不一致（比較不可）"]
        print(f"[4] size mismatch: {len(mismatch)} skipped")
        check(len(mismatch) == 1 and mismatch[0].cpu_name == f"pos{pos}.hc_norm-0",
              "a 8 vs 4 element mismatch is skipped as 比較不可")
        check(len(sized["comparisons"]) == expected_pairs - 1,
              "the mismatching pair is not compared")

        # 6. missing engine vector is reported, not compared ------------------
        cpu_vectors, warp_vectors = synth_case(pos, seed=6)
        warp_vectors = [vector for vector in warp_vectors
                        if vector.name != f"pos{pos}.layer2.output"]
        cpu_file6 = tmp_path / "cpu_missing.bin"
        warp_file6 = tmp_path / "warp_missing.bin"
        write_wdptrc1(cpu_file6, cpu_vectors)
        write_wdptrc1(warp_file6, warp_vectors)
        missing = compare_files(cpu_file6, warp_file6, pos)
        missing_skips = [item for item in missing["skips"] if item.reason == "warp 側に名前なし"]
        check(len(missing_skips) == 1 and missing_skips[0].cpu_name == f"pos{pos}.l_last-2",
              "a missing engine vector is reported as warp 側に名前なし")

        # 7. container round-trip --------------------------------------------
        version, reread = read_wdptrc1(cpu_file6)
        check(version == 1 and len(reread) == len(cpu_vectors)
              and reread[0].name == cpu_vectors[0].name
              and np.array_equal(reread[0].values, cpu_vectors[0].values),
              "WDPTRC1 writer/reader round-trip")

        # 8. malformed input is rejected -------------------------------------
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"NOTATRACE" + bytes(64))
        try:
            read_wdptrc1(bad)
            check(False, "a file without the WDPTRC1 magic is rejected")
        except ValueError:
            check(True, "a file without the WDPTRC1 magic is rejected")

    if failures:
        print(f"\nSELF-TEST FAILED: {len(failures)} check(s)")
        return 1
    print("\nSELF-TEST PASSED")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare warp (CUDA engine) and CPU oracle WDPTRC1 internal vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", help="CPU oracle WDPTRC1 trace (qwen38_forced_cpu_trace.cpp)")
    parser.add_argument("--warp", help="engine WDPTRC1 trace (--state-trace output)")
    parser.add_argument("--position", type=int, help="position whose vectors to compare")
    parser.add_argument("--json", dest="json_out", help="write the full result as JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="NRMSE threshold for the first-exceeding report "
                             f"(default {DEFAULT_THRESHOLD:g})")
    parser.add_argument("--verbose", action="store_true",
                        help="print every compared pair and all skips")
    parser.add_argument("--audit", action="store_true",
                        help="list vector names present in the files but not covered by the mapping")
    parser.add_argument("--strict", action="store_true",
                        help=f"exit {STRICT_EXIT} when any pair exceeds --threshold")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in synthetic WDPTRC1 checks (no external files)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.cpu is None or args.warp is None or args.position is None:
        parser.error("--cpu, --warp and --position are required (or use --self-test)")

    try:
        result = compare_files(args.cpu, args.warp, args.position)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print_report(result, args.threshold, args.verbose)
    if args.audit:
        audit_names(result)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(to_json(result, args.threshold), ensure_ascii=False, indent=2) + "\n")
        print(f"\njson written to {args.json_out}")
    if args.strict and first_exceeding(result["comparisons"], args.threshold) is not None:
        return STRICT_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
