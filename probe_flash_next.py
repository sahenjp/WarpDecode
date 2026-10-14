from __future__ import annotations

import json
import mmap
import os
import pathlib
import struct
from collections import Counter

import modal

APP_NAME = "warpdecode-qwen38-probe"
MODEL_VOL = "warpdecode-qwen38-models"
DEFAULT_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    4: "REMOVED_Q4_2", 5: "REMOVED_Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "REMOVED_31", 32: "REMOVED_32", 33: "REMOVED_33",
    34: "TQ1_0", 35: "TQ2_0", 36: "REMOVED_36", 37: "REMOVED_37",
    38: "REMOVED_38", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


class Reader:
    def __init__(self, f):
        self.f = f

    def read_exact(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"unexpected EOF: wanted {n}, got {len(b)}")
        return b

    def u8(self): return struct.unpack("<B", self.read_exact(1))[0]
    def i8(self): return struct.unpack("<b", self.read_exact(1))[0]
    def u16(self): return struct.unpack("<H", self.read_exact(2))[0]
    def i16(self): return struct.unpack("<h", self.read_exact(2))[0]
    def u32(self): return struct.unpack("<I", self.read_exact(4))[0]
    def i32(self): return struct.unpack("<i", self.read_exact(4))[0]
    def u64(self): return struct.unpack("<Q", self.read_exact(8))[0]
    def i64(self): return struct.unpack("<q", self.read_exact(8))[0]
    def f32(self): return struct.unpack("<f", self.read_exact(4))[0]
    def f64(self): return struct.unpack("<d", self.read_exact(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > (1 << 31):
            raise ValueError(f"absurd GGUF string length: {n}")
        return self.read_exact(n).decode("utf-8", errors="replace")


# GGUF value type IDs.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)


def read_scalar(r: Reader, t: int):
    if t == U8: return r.u8()
    if t == I8: return r.i8()
    if t == U16: return r.u16()
    if t == I16: return r.i16()
    if t == U32: return r.u32()
    if t == I32: return r.i32()
    if t == F32: return r.f32()
    if t == BOOL: return bool(r.u8())
    if t == STRING: return r.string()
    if t == U64: return r.u64()
    if t == I64: return r.i64()
    if t == F64: return r.f64()
    raise ValueError(f"not a scalar GGUF value type: {t}")


def skip_value(r: Reader, t: int):
    if t != ARRAY:
        return read_scalar(r, t)

    elem = r.u32()
    n = r.u64()
    if n > (1 << 34):
        raise ValueError(f"absurd GGUF array length: {n}")

    fixed = {
        U8: 1, I8: 1, BOOL: 1,
        U16: 2, I16: 2,
        U32: 4, I32: 4, F32: 4,
        U64: 8, I64: 8, F64: 8,
    }
    if elem in fixed:
        r.f.seek(n * fixed[elem], os.SEEK_CUR)
        return {"array_type": elem, "count": n}
    if elem == STRING:
        for _ in range(n):
            ln = r.u64()
            r.f.seek(ln, os.SEEK_CUR)
        return {"array_type": elem, "count": n}
    if elem == ARRAY:
        for _ in range(n):
            skip_value(r, ARRAY)
        return {"array_type": elem, "count": n}
    raise ValueError(f"unknown GGUF array element type: {elem}")


def find_gguf(root: pathlib.Path) -> pathlib.Path:
    files = sorted(root.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError(f"no GGUF found under {root}")
    return files[0]


@app.function(
    image=image,
    volumes={"/models": models.with_mount_options(read_only=True)},
    cpu=1,
    memory=2048,
    timeout=600,
)
def probe(repo_id: str = DEFAULT_REPO) -> dict:
    root = pathlib.Path("/models") / repo_id.replace("/", "__")
    path = find_gguf(root)

    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_gib": round(path.stat().st_size / (1024 ** 3), 3),
    }

    # Check whether Modal Volume supports the access patterns llama.cpp commonly uses.
    fd = os.open(path, os.O_RDONLY)
    try:
        result["pread_4k"] = len(os.pread(fd, 4096, 0))
        try:
            mm = mmap.mmap(fd, 4096, access=mmap.ACCESS_READ)
            result["mmap_4k"] = len(mm)
            mm.close()
        except Exception as e:
            result["mmap_error"] = repr(e)
    finally:
        os.close(fd)

    selected = {}
    meta_keys = []
    tensor_types = Counter()
    tensor_examples = []
    unknown_type_ids = set()

    with open(path, "rb", buffering=1024 * 1024) as f:
        r = Reader(f)
        magic = r.read_exact(4)
        if magic != b"GGUF":
            raise RuntimeError(f"bad GGUF magic: {magic!r}")

        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        result["gguf_version"] = version
        result["tensor_count"] = tensor_count
        result["kv_count"] = kv_count

        for _ in range(kv_count):
            key = r.string()
            t = r.u32()
            meta_keys.append(key)

            wanted = (
                key.startswith("general.")
                or key.startswith("qwen4")
                or "architecture" in key
                or "quant" in key
                or "split" in key
                or "expert" in key
                or "nextn" in key
            )
            if t == ARRAY:
                value = skip_value(r, t)
            else:
                value = read_scalar(r, t)

            if wanted and not isinstance(value, dict):
                # Avoid dumping giant strings if a model embeds one.
                if isinstance(value, str) and len(value) > 1000:
                    value = value[:1000] + "...<truncated>"
                selected[key] = value
            elif wanted and isinstance(value, dict):
                selected[key] = value

        for i in range(tensor_count):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            type_id = r.u32()
            offset = r.u64()

            type_name = GGML_TYPES.get(type_id, f"UNKNOWN_{type_id}")
            tensor_types[type_name] += 1
            if type_id not in GGML_TYPES:
                unknown_type_ids.add(type_id)
            if len(tensor_examples) < 30:
                tensor_examples.append({
                    "name": name,
                    "dims": dims,
                    "type_id": type_id,
                    "type": type_name,
                    "offset": offset,
                })

        result["header_end_offset"] = f.tell()

    result["selected_metadata"] = selected
    result["tensor_type_counts"] = dict(tensor_types.most_common())
    result["unknown_tensor_type_ids"] = sorted(unknown_type_ids)
    result["tensor_examples"] = tensor_examples
    result["qwen4exp_hint"] = selected.get("general.architecture") in (
        "qwen4exp", "qwen4_exp"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo: str = DEFAULT_REPO):
    result = probe.remote(repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
