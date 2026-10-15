#!/usr/bin/env python3
"""Turn the interleaved A/B jsonl into the win/lose table used by docs/LLAMACPP_AB.md.

Only same-round pairs are compared: the two runtimes are measured alternately
inside one job, so the round is the only unit whose ratio is meaningful (the VM
drifts by 30 % between time slots).

  python3 tests/llamacpp_ab_report.py issue1_artifacts/taskl_ab3_results.jsonl \
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=pathlib.Path)
    parser.add_argument("--pairs", default="engine_8000=llama_k0,engine_20000=llama_k16,engine_30000=llama_k24")
    parser.add_argument("--floor", type=float, default=1.05,
                        help="ratio above this is a win, below 1/floor a loss")
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["round"]] = row

    def stats(label):
        values = [r["tok_s"] for r in by_label.get(label, {}).values() if r.get("tok_s")]
        vr = [r["max_vram_mib"] for r in by_label.get(label, {}).values() if r.get("max_vram_mib")]
        match = [r["matched_prefix"] for r in by_label.get(label, {}).values()]
        secs = [r["decode_seconds"] for r in by_label.get(label, {}).values() if r.get("decode_seconds")]
        return {
            "label": label,
            "n": len(values),
            "best": max(values) if values else float("nan"),
            "median": median(values),
            "worst": min(values) if values else float("nan"),
            "vram_max": max(vr) if vr else None,
            "match_min": min(match) if match else None,
            "decode_median_s": median(secs),
        }

    print("## per-run statistics (best / median / worst tok/s over the rounds)")
    print("")
    print("| run | n | best | median | worst | max VRAM (MiB) | decode median (s) | min 64-token prefix match |")
    print("|---|---|---|---|---|---|---|---|")
    summary = {"runs": {}, "pairs": [], "verdict": None}
    for label in sorted(by_label):
        st = stats(label)
        summary["runs"][label] = st
        print("| %s | %d | %.3f | %.3f | %.3f | %s | %.2f | %s/64 |" % (
            label, st["n"], st["best"], st["median"], st["worst"],
            st["vram_max"] if st["vram_max"] is not None else "-",
            st["decode_median_s"], st["match_min"]))

    print("")
    print("## same-round ratios (engine / llama.cpp)")
    print("")
    print("| pair | round | engine tok/s | llama tok/s | ratio | engine VRAM | llama VRAM | parity |")
    print("|---|---|---|---|---|---|---|---|")
    ratios = []
    for pair in args.pairs.split(","):
        engine_label, llama_label = pair.split("=")
        per_round = []
        for rnd in sorted(by_label.get(engine_label, {})):
            e = by_label[engine_label].get(rnd)
            l = by_label.get(llama_label, {}).get(rnd)
            if not e or not l or not e.get("tok_s") or not l.get("tok_s"):
                continue
            ratio = e["tok_s"] / l["tok_s"]
            per_round.append(ratio)
            ratios.append(ratio)
            print("| %s vs %s | %d | %.3f | %.3f | %.3f | %s | %s | %s vs %s |" % (
                engine_label, llama_label, rnd, e["tok_s"], l["tok_s"], ratio,
                e["max_vram_mib"], l["max_vram_mib"],
                "%d/64" % e["matched_prefix"], "%d/64" % l["matched_prefix"]))
        entry = {
            "engine": engine_label,
            "llama": llama_label,
            "rounds": len(per_round),
            "ratios": per_round,
            "median_ratio": median(per_round),
            "engine_median_tok_s": stats(engine_label)["median"],
            "llama_median_tok_s": stats(llama_label)["median"],
        }
        summary["pairs"].append(entry)

    print("")
    print("## verdict per pairing")
    print("")
    print("| pairing | rounds | median ratio (engine/llama) | engine median | llama median | verdict |")
    print("|---|---|---|---|---|---|")
    wins = losses = draws = 0
    for entry in summary["pairs"]:
        ratio = entry["median_ratio"]
        if ratio >= args.floor:
            verdict = "warp WINS (>=%.2fx)" % args.floor
            wins += 1
        elif ratio <= 1.0 / args.floor:
            verdict = "warp LOSES (<=%.2fx)" % (1.0 / args.floor)
            losses += 1
        else:
            verdict = "DRAW (within +-%.0f%%)" % ((args.floor - 1.0) * 100)
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
