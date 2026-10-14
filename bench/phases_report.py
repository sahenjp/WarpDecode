#!/usr/bin/env python3
"""bench/phases.sh が残した生レコードを prefill / decode の表と JSON にまとめる。

入力は JSONL (1行1レコード)。header レコードの後に run レコードが続く。
計測値はエンジン JSON の prefill_seconds / decode_seconds を使うので、
プロセス起動とモデルロードは含まれない (wall_seconds は参考値として残す)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_records(path: Path):
    header = None
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("kind") == "header":
            header = record
        elif record.get("kind") == "run":
            runs.append(record)
    if header is None:
        raise SystemExit("header レコードがありません: %s" % path)
    if not runs:
        raise SystemExit("run レコードがありません: %s" % path)
    return header, runs


def phase(engine: dict, tokens_key: str, seconds_key: str):
    if tokens_key not in engine or seconds_key not in engine:
        raise SystemExit("engine JSON に %s / %s がありません (--prefill/--bench-repeat 対応ビルドが必要)"
                         % (tokens_key, seconds_key))
    tokens = int(engine[tokens_key])
    seconds = float(engine[seconds_key])
    if tokens <= 0 or seconds <= 0.0:
        return None
    return {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
        "ms_per_token": seconds * 1000.0 / tokens,
    }


def first_mismatch(expected, got):
    for index, (want, have) in enumerate(zip(expected, got)):
        if want != have:
            return index
    if len(expected) != len(got):
        return min(len(expected), len(got))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", default="")
    parser.add_argument("--json", action="store_true", help="JSONだけを出力する")
    args = parser.parse_args()

    header, runs = load_records(Path(args.records))
    prefill_len = int(header["prefill"])
    generate_len = int(header["generate"])

    expected = None
    if args.ref:
        ref = json.loads(Path(args.ref).read_text())
        ids = ref["token_ids"] if isinstance(ref, dict) else ref
        if prefill_len + generate_len > len(ids):
            raise SystemExit("--ref が短すぎます: prefill + generate = %d > %d"
                             % (prefill_len + generate_len, len(ids)))
        expected = list(ids[prefill_len:prefill_len + generate_len])

    summary = {"header": header, "runs": [], "parity": None}
    rows = []
    parity_matched = 0
    parity_total = 0
    for record in runs:
        engine = record["engine"]
        entry = {
            "config": record.get("config", ""),
            "cache_mib": record.get("cache_mib", ""),
            "wall_seconds": record.get("wall_seconds"),
            "prefill": phase(engine, "prefill_tokens", "prefill_seconds"),
            "decode": phase(engine, "decode_tokens", "decode_seconds"),
            "total_seconds": float(engine.get("seconds", 0.0)),
            "engine": engine,
        }
        if expected is not None:
            matched = first_mismatch(expected, list(engine.get("token_ids", [])))
            entry["parity"] = {
                "expected_tokens": len(expected),
                "generated_tokens": len(engine.get("token_ids", [])),
                "matched_prefix": len(expected) if matched is None else matched,
                "first_mismatch": matched,
                "ok": matched is None,
            }
            parity_total += len(expected)
            parity_matched += entry["parity"]["matched_prefix"]
        summary["runs"].append(entry)
        rows.append(entry)

    if expected is not None:
        summary["parity"] = {
            "matched": parity_matched,
            "total": parity_total,
            "ok": parity_matched == parity_total,
        }

    def num(value, digits=3):
        return "-" if value is None else ("%.*f" % (digits, value))

    lines = []
    lines.append("")
    lines.append("| cache | prefill tok | prefill tok/s | prefill ms/tok | decode tok | decode tok/s | decode ms/tok | total s | parity |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for entry in rows:
        prefill = entry["prefill"]
        decode = entry["decode"]
        parity = entry.get("parity")
        parity_text = "-"
        if parity is not None:
            parity_text = "%d/%d%s" % (parity["matched_prefix"], parity["expected_tokens"],
                                       "" if parity["ok"] else " (mismatch@%d)" % parity["first_mismatch"])
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            entry["cache_mib"],
            "-" if prefill is None else prefill["tokens"],
            num(None if prefill is None else prefill["tok_s"]),
            num(None if prefill is None else prefill["ms_per_token"]),
            "-" if decode is None else decode["tokens"],
            num(None if decode is None else decode["tok_s"]),
            num(None if decode is None else decode["ms_per_token"]),
            num(entry["total_seconds"], 2),
            parity_text,
        ))
    table = "\n".join(lines) + "\n"
    summary["table"] = table

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(table)
        if summary["parity"] is not None:
            parity = summary["parity"]
            print("parity: %d/%d %s" % (parity["matched"], parity["total"],
                                        "OK" if parity["ok"] else "MISMATCH"))
        print(json.dumps({
            "runs": len(rows),
            "decode_tok_s": [None if entry["decode"] is None else round(entry["decode"]["tok_s"], 3)
                             for entry in rows],
            "prefill_tok_s": [None if entry["prefill"] is None else round(entry["prefill"]["tok_s"], 3)
                              for entry in rows],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
