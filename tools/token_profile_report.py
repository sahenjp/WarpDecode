#!/usr/bin/env python3
"""Task F: 収集した token_profile_*.json を表にまとめる。

使い方:
  python3 tools/token_profile_report.py                       # issue1_artifacts/token_profile_*.json 全部
  python3 tools/token_profile_report.py --prefix taskF_l4     # ラベルで絞る
  python3 tools/token_profile_report.py --label taskF_l4_c8000 --detail
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGES = ["embed", "ple", "hc_mix_attn", "attn", "hc_combine_attn",
          "hc_mix_ffn", "moe", "hc_combine_ffn", "head", "sync", "other"]


def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(ROOT / "issue1_artifacts" / pattern))):
        record = json.loads(pathlib.Path(path).read_text())
        record["_path"] = pathlib.Path(path).name
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="taskF_")
    parser.add_argument("--label", default="")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    pattern = "token_profile_%s.json" % (args.label if args.label else args.prefix + "*")
    records = load(pattern)
    if not records:
        raise SystemExit("該当する profile がありません: " + pattern)

    print("%-26s %8s %6s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s" %
          ("label", "tok_s", "match", "total", "embed", "ple", "hcMixA", "attn", "cCombA",
           "hcMixF", "moe", "cCombF", "head", "sync", "other"))
    for r in records:
        p = r.get("profile") or {}
        if not p or not p.get("enabled"):
            print("%-26s %8s %6s  (profile disabled)" % (r["label"], r.get("tok_s"), r.get("matched_prefix")))
            continue
        cols = [p.get(k + "_ms", 0.0) for k in STAGES]
        print("%-26s %8.3f %3d/%-2d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" %
              (r["label"], r.get("tok_s") or 0.0, r.get("matched_prefix", -1),
               r.get("matched_prefix_of", 64), p.get("total_ms", 0.0), *cols))
        if args.detail:
            d = r.get("profile_detail") or {}
            layers = d.get("layer_device_ms") or []
            if layers:
                top = sorted(range(len(layers)), key=lambda i: -layers[i])[:5]
                print("    heaviest layers (idx:ms/token): " +
                      ", ".join("%d:%.2f" % (i, layers[i]) for i in top))
            print("    host total=%.2f host_other=%.2f flush=%.2f calls_moe=%s unmatched=%s" %
                  (d.get("host_total_ms", 0.0), d.get("host_other_ms", 0.0),
                   d.get("host_flush_ms", 0.0), d.get("calls_moe"), d.get("unmatched")))
    print()
    print("列の意味: total=1トークンの device 合計[ms], 各列=device 区間の和周辺, sync=D2H+同期, "
          "other=どの区間にも入らない残り。match は 64/64 の一致数。")


if __name__ == "__main__":
    main()
