#!/usr/bin/env python3
"""Dump the tokenizer metadata of a GGUF file (no tensor data is read).

Used as the ground truth for src/qwen38_tokenizer.cuh: it prints the keys the
C++ tokenizer reads (model / pre / tokens / merges / token_type / special ids /
chat template) so the two readers can be compared, and it is also the probe that
runs on the Colab VM against the real Qwen3.8-Flash-Next shard.

Usage:
  python3 tools/gguf_tokenizer_dump.py model.gguf [--json out.json] [--tokens N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
ARRAY_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
TYPE_NAMES = {8: "string", 9: "array"}


def read_string(fh) -> str:
    (n,) = struct.unpack("<Q", fh.read(8))
    raw = fh.read(n)
    return raw.decode("utf-8", errors="replace")


def read_value(fh, vtype):
    if vtype == 8:
        return read_string(fh)
    if vtype == 9:
        (etype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        if etype == 8:
            return [read_string(fh) for _ in range(count)]
        if etype == 9:
            return [read_value(fh, 9) for _ in range(count)]
        fmt = ARRAY_TYPES[etype]
        size = struct.calcsize(fmt)
        return list(struct.unpack("<%d%s" % (count, fmt), fh.read(size * count)))
    fmt = SCALAR[vtype]
    return struct.unpack("<" + fmt, fh.read(struct.calcsize(fmt)))[0]


def dump(path: pathlib.Path):
    with path.open("rb") as fh:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fh.read(24))
        if magic != 0x46554747:
            raise SystemExit("not a GGUF file: %s" % path)
        meta = {}
        for _ in range(n_kv):
            key = read_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = read_value(fh, vtype)
        header_bytes = fh.tell()
    return version, n_tensors, n_kv, meta, header_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--json")
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()

    path = pathlib.Path(args.model)
    version, n_tensors, n_kv, meta, header_bytes = dump(path)
    tokens = meta.get("tokenizer.ggml.tokens", [])
    merges = meta.get("tokenizer.ggml.merges", [])
    types = meta.get("tokenizer.ggml.token_type", [])
    histogram = {}
    for value in types:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    template = meta.get("tokenizer.chat_template", "")
    summary = {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "gguf_version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv_bytes": header_bytes,
        "arch": meta.get("general.architecture"),
        "tokenizer_model": meta.get("tokenizer.ggml.model"),
        "tokenizer_pre": meta.get("tokenizer.ggml.pre"),
        "n_vocab": len(tokens),
        "n_merges": len(merges),
        "n_token_types": len(types),
        "token_type_histogram": histogram,
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
        "unk_token_id": meta.get("tokenizer.ggml.unknown_token_id"),
        "pad_token_id": meta.get("tokenizer.ggml.padding_token_id"),
        "add_bos_token": meta.get("tokenizer.ggml.add_bos_token"),
        "add_eos_token": meta.get("tokenizer.ggml.add_eos_token"),
        "add_space_prefix": meta.get("tokenizer.ggml.add_space_prefix"),
        "chat_template": template[:600] if isinstance(template, str) else None,
        "chat_template_bytes": len(template) if isinstance(template, str) else 0,
        "first_tokens": tokens[: args.tokens],
        "last_tokens": tokens[-args.tokens:],
        "first_merges": merges[:3],
        "last_merges": merges[-3:],
        "sample_token_types": types[:16],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
