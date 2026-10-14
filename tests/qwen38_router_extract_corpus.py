#!/usr/bin/env python3
"""Golden router corpus for tests/qwen38_router_host_check.sh.

Reads every WDPTRC1 trace under issue1_artifacts/, pairs each
ffn_moe_logits-<il> row with the ffn_moe_topk-<il> expert ids and the
ffn_moe_weights_norm-<il> weights the CPU oracle recorded for it, and writes:

    uint32 count
    count * { uint32 name_len, name bytes, float32 logits[512], int32 ids[10],
              float32 weights[10] }

Duplicate rows (the same logits recorded by more than one trace) are dropped.
"""
from __future__ import annotations

import glob
import pathlib
import re
import struct
import sys


def read_trace(path: pathlib.Path):
    raw = path.read_bytes()
    if raw[:8] != b"WDPTRC1\x00":
        return None
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 1:
        return None
    out = {}
    off = 16
    for _ in range(count):
        ns, vc = struct.unpack_from("<IQ", raw, off)
        off += 12
        name = raw[off:off + ns].decode()
        off += ns
        out.setdefault(name, struct.unpack_from("<%df" % vc, raw, off))
        off += 4 * vc
    return out


def collect(root: pathlib.Path):
    cases = []
    seen = set()
    for path in sorted(glob.glob(str(root / "issue1_artifacts" / "*.bin"))):
        try:
            trace = read_trace(pathlib.Path(path))
        except Exception:
            continue
        if not trace:
            continue
        for name, logits in sorted(trace.items()):
            match = re.match(r"(.*)ffn_moe_logits-(\d+)$", name)
            if not match:
                continue
            prefix, layer = match.group(1), match.group(2)
            norm = trace.get(prefix + "ffn_moe_weights_norm-" + layer)
            topk = trace.get(prefix + "ffn_moe_topk-" + layer)
            if norm is None or topk is None:
                continue
            if len(logits) != 512 or len(topk) != 10 or len(norm) != 10:
                continue
            key = (struct.pack("<512f", *logits),)
            if key in seen:
                continue
            seen.add(key)
            cases.append((name, logits, topk, norm))
    return cases


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    dest = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "router_corpus.bin")
    cases = collect(root)
    with dest.open("wb") as fh:
        fh.write(struct.pack("<I", len(cases)))
        for name, logits, ids, weights in cases:
            raw_name = name.encode()
            fh.write(struct.pack("<I", len(raw_name)))
            fh.write(raw_name)
            fh.write(struct.pack("<512f", *logits))
            fh.write(struct.pack("<10i", *[int(v) for v in ids]))
            fh.write(struct.pack("<10f", *weights))
    print("golden corpus: %d rows -> %s" % (len(cases), dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
