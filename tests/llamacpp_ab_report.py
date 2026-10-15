#!/usr/bin/env python3
"""Turn the interleaved A/B jsonl into the win/lose table used by docs/LLAMACPP_AB.md.

Only same-round pairs are compared: the two runtimes are measured alternately
inside one job, so the round is the only unit whose ratio is meaningful (this VM
drifts by 30 % between time slots).

Both sides report their FASTEST in-process repetition (the engine's
--bench-repeat N keeps the fastest of N, and llama.cpp's --reps N gives one
line per repetition, so max(reps_tok_s) is taken here). The first repetition is
consistently slower because it faults in pages / loads CUDA modules, and the
engine never reports that one, so using it for llama.cpp would be unfair.

  python3 tests/llamacpp_ab_report.py issue1_artifacts/taskl_ab3_results.jsonl \\
      --pairs engine_8000=llama_k0,engine_20000=llama_k16,engine_30000=llama_k24
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys


def median(values):
    return statistics.median(values) if values else float("nan")


def run_value(row, metric):
    """Engine: single best-of-N value. llama.cpp: one value per repetition."""
    reps = [v for v in row.get("reps_tok_s", []) if v]
    if not reps:
        return float("nan")
    if metric == "best":
        return max(reps)
    if metric == "first":
        return reps[0]
    if metric == "last":
        return reps[-1]
    return median(reps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=pathlib.Path)
    parser.add_argument("--pairs", default="engine_8000=llama_k0,engine_20000=llama_k16,engine_30000=llama_k24")
    parser.add_argument("--floor", type=float, default=1.05,
                        help="ratio above this is a win, below 1/floor a loss")
    parser.add_argument("--metric", default="best", choices=["best", "median", "first", "last"])
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["round"]] = row

    def stats(label):
        rounds = by_label.get(label, {})
        values = [run_value(r, args.metric) for r in rounds.values()]
        values = [v for v in values if v == v and v > 0]
        vr = [r["max_vram_mib"] for r in rounds.values() if r.get("max_vram_mib")]
        match = [r["matched_prefix"] for r in rounds.values()]
        pids = [len(r.get("gpu_app_pids", [])) for r in rounds.values()]
        return {
            "label": label,
            "n": len(values),
            "best": max(values) if values else float("nan"),
            "median": median(values),
            "worst": min(values) if values else float("nan"),
            "vram_max": max(vr) if vr else None,
            "match_min": min(match) if match else None,
            "gpu_pids_max": max(pids) if pids else None,
        }

    print("## per-run statistics (metric=%s over the rounds)" % args.metric)
    print("")
    print("| run | rounds | best | median | worst | max VRAM (MiB) | 64-token prefix match | distinct GPU pids seen |")
    print("|---|---|---|---|---|---|---|---|")
    summary = {"metric": args.metric, "runs": {}, "pairs": [], "verdict": None}
    for label in sorted(by_label):
        st = stats(label)
        summary["runs"][label] = st
        print("| %s | %d | %.3f | %.3f | %.3f | %s | %s/64 | %s |" % (
            label, st["n"], st["best"], st["median"], st["worst"],
            st["vram_max"] if st["vram_max"] is not None else "-",
            st["match_min"], st["gpu_pids_max"]))

    print("")
    print("## same-round ratios (engine / llama.cpp), metric=%s" % args.metric)
    print("")
    print("| pairing | round | engine tok/s | llama tok/s | ratio | engine VRAM | llama VRAM | engine match | llama match |")
    print("|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for pair in args.pairs.split(","):
        engine_label, llama_label = pair.split("=")
        per_round = []
        for rnd in sorted(by_label.get(engine_label, {})):
            e = by_label[engine_label].get(rnd)
            l = by_label.get(llama_label, {}).get(rnd)
            if not e or not l:
                continue
            ev = run_value(e, args.metric)
            lv = run_value(l, args.metric)
            if not ev or not lv or ev != ev or lv != lv:
                continue
            ratio = ev / lv
            per_round.append(ratio)
            ratios.append(ratio)
            print("| %s vs %s | %d | %.3f | %.3f | %.3f | %s | %s | %d/64 | %d/64 |" % (
                engine_label, llama_label, rnd, ev, lv, ratio,
                e["max_vram_mib"], l["max_vram_mib"],
                e["matched_prefix"], l["matched_prefix"]))
        summary["pairs"].append({
            "engine": engine_label,
            "llama": llama_label,
            "rounds": len(per_round),
            "ratios": per_round,
            "median_ratio": median(per_round),
            "engine_median_tok_s": stats(engine_label)["median"],
            "llama_median_tok_s": stats(llama_label)["median"],
        })

    print("")
    print("## verdict per pairing")
    print("")
    print("| pairing | rounds | median ratio (engine/llama) | engine median | llama median | verdict |")
    print("|---|---|---|---|---|---|")
    wins = losses = draws = 0
    for entry in summary["pairs"]:
        ratio = entry["median_ratio"]
        if ratio >= args.floor:
            verdict = "warp wins (>=%.2fx)" % args.floor
            wins += 1
        elif ratio <= 1.0 / args.floor:
            verdict = "llama.cpp wins (<=%.2fx)" % (1.0 / args.floor)
            losses += 1
        else:
            verdict = "draw (within +-%.0f%%)" % ((args.floor - 1.0) * 100)
            draws += 1
        print("| %s vs %s | %d | %.3f | %.3f | %.3f | %s |" % (
            entry["engine"], entry["llama"], entry["rounds"], ratio,
            entry["engine_median_tok_s"], entry["llama_median_tok_s"], verdict))

    overall = median(ratios) if ratios else float("nan")
    if not ratios:
        summary["verdict"] = "no paired data"
    elif overall >= args.floor:
        summary["verdict"] = "warp wins overall (median ratio %.3f)" % overall
    elif overall <= 1.0 / args.floor:
        summary["verdict"] = "llama.cpp wins overall (median ratio %.3f)" % overall
    else:
        summary["verdict"] = "draw overall (median ratio %.3f)" % overall
    print("")
    print("median ratio over every paired round: **%.3f** -> %s" % (overall, summary["verdict"]))
    print("pairings: %d win, %d loss, %d draw" % (wins, losses, draws))

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())