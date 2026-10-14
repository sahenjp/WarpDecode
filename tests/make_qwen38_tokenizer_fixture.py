#!/usr/bin/env python3
"""Build a metadata-only GGUF (0 tensors) from a real model's header.

The tokenizer comparison in tests/qwen38_tokenizer_host_check.sh needs two
readers of the *same* vocabulary: src/qwen38_tokenizer.cuh and llama.cpp's
llama_tokenize(). Both only need the GGUF key/value section, so a Qwen3.8 header
(11 MB) is enough and the 86 GB of tensor data is not required:

  # first 64 MiB of the model are enough for the whole key/value section
  curl -sL -r 0-67108863 -o /tmp/qwen38_head.bin <model-url>
  python3 tests/make_qwen38_tokenizer_fixture.py /tmp/qwen38_head.bin /tmp/qwen38-tokenizer-fixture.gguf
  bash tests/qwen38_tokenizer_host_check.sh /path/to/llama.cpp /tmp/qwen38-tokenizer-fixture.gguf

A full model file works too (its tensor index is simply dropped).

Usage: make_qwen38_tokenizer_fixture.py SOURCE.gguf OUT.gguf
"""
from __future__ import annotations

import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from gguf_tokenizer_dump import read_string, read_value  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    source = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])

    with source.open("rb") as fh:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fh.read(24))
        if magic != 0x46554747:
            raise SystemExit("not a GGUF file: %s" % source)
        for _ in range(n_kv):
            read_string(fh)
            (value_type,) = struct.unpack("<I", fh.read(4))
            read_value(fh, value_type)
        kv_bytes = fh.tell()

    with source.open("rb") as fh:
        payload = fh.read(kv_bytes)
    header = struct.pack("<IIQQ", magic, version, 0, n_kv)
    out.write_bytes(header + payload[24:])
    print("wrote %s (%d bytes, %d key/value pairs, %d tensors dropped)"
          % (out, out.stat().st_size, n_kv, n_tensors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
