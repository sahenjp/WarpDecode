#!/usr/bin/env python3
"""Task K: turn an expert_prefetch_<label>.json result file into tables.

Usage: python3 tools/expert_prefetch_report.py --label l4taskk2 [--detail]
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def num(rec, section, key):
    return (rec.get(section) or {}).get(key)


def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % digits) % v
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    path = ROOT / ("issue1_artifacts/expert_prefetch_%s.json" % args.label)
    data = json.loads(path.read_text())
    print("label=%s session=%s arch=%s plan=%s" % (
        data["label"], data["session"], data["arch"], data["plan"]))
    print("gpu_sampler=%s" % json.dumps(data.get("gpu_sampler")))
    print()
    head = ("case", "tok_s", "parity", "moe_dev", "moe_host", "total_dev",
            "stage", "pread", "prefetched", "pf_posted", "pf_cand", "pf_resid",
            "pf_used", "pf_drop", "pf_fail", "intra_gap", "layer_gap")
    rows = []
    for rec in data["cases"]:
        rows.append((
            rec.get("case", "?"),
            fmt(rec.get("tok_s"), 3),
            "%s/%s" % (rec.get("matched_prefix"), rec.get("matched_prefix_of", 64)),
            fmt(num(rec, "profile", "moe_ms")),
            fmt(num(rec, "detail", "host_moe_ms")),
            fmt(num(rec, "profile", "total_ms")),
            fmt(num(rec, "stats", "stage_ms"), 0),
            fmt(num(rec, "stats", "io_pread_ms"), 0),
            fmt(num(rec, "stats", "io_prefetched"), 0),
            fmt(num(rec, "stats", "prefetch_posted"), 0),
            fmt(num(rec, "stats", "prefetch_candidates"), 0),
            fmt(num(rec, "stats", "prefetch_resident_skip"), 0),
            fmt(num(rec, "stats", "prefetch_used"), 0),
            fmt(num(rec, "stats", "prefetch_dropped"), 0),
            fmt(num(rec, "stats", "prefetch_failed"), 0),
            fmt(num(rec, "stats", "intra_gap_ms"), 0),
            fmt(num(rec, "stats", "layer_gap_ms"), 0),
        ))
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(head))))
    print()
    print("per-token (stats totals / tokens):")
    print("  case                        acquire  stage  pread  intra_gap  layer_gap  pf_used")
    for rec in data["cases"]:
        st = rec.get("stats") or {}
        tokens = st.get("tokens") or 0
        if not tokens:
            continue
        print("  %-26s %7.2f %6.2f %6.2f %10.2f %10.2f %8s" % (
            rec.get("case", "?"), (st.get("acquire_ms") or 0) / tokens,
            (st.get("stage_ms") or 0) / tokens, (st.get("io_pread_ms") or 0) / tokens,
            (st.get("intra_gap_ms") or 0) / tokens, (st.get("layer_gap_ms") or 0) / tokens,
            st.get("prefetch_used", "-")))
    if args.detail:
        print()
        for rec in data["cases"]:
            print(json.dumps(rec, indent=1)[:2000])


if __name__ == "__main__":
    main()
