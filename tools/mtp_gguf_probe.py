#!/usr/bin/env python3
"""Dump GGUF metadata + blk.47/48 tensor inventory (header only, no weight reads)."""
import struct, sys, json

DEFAULT_PATH = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
# The colab kernel forwards the CLI args, so argv may contain "-f <path>".
args = [a for a in sys.argv[1:] if not a.startswith("-") and a.endswith(".gguf")]
PATH = args[-1] if args else DEFAULT_PATH

TYPES = {0:"U8",1:"I8",2:"U16",3:"I16",4:"U32",5:"I32",6:"F32",7:"BOOL",8:"STRING",9:"ARRAY",10:"U64",11:"I64",12:"F64"}
GGML = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0"}

class R:
    def __init__(self, f): self.f = f
    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def i8(self): return struct.unpack("<b", self.f.read(1))[0]
    def i16(self): return struct.unpack("<h", self.f.read(2))[0]
    def i32(self): return struct.unpack("<i", self.f.read(4))[0]
    def i64(self): return struct.unpack("<q", self.f.read(8))[0]
    def f32(self): return struct.unpack("<f", self.f.read(4))[0]
    def f64(self): return struct.unpack("<d", self.f.read(8))[0]
    def b(self): return struct.unpack("<?", self.f.read(1))[0]
    def string(self):
        n = self.u64(); return self.f.read(n).decode("utf-8", "replace")

def read_value(r, t):
    if t == 0: return struct.unpack("<B", r.f.read(1))[0]
    if t == 1: return r.i8()
    if t == 2: return struct.unpack("<H", r.f.read(2))[0]
    if t == 3: return r.i16()
    if t == 4: return r.u32()
    if t == 5: return r.i32()
    if t == 6: return r.f32()
    if t == 7: return r.b()
    if t == 8: return r.string()
    if t == 10: return r.u64()
    if t == 11: return r.i64()
    if t == 12: return r.f64()
    if t == 9:
        et = r.u32(); n = r.u64()
        if et == 8:
            return [r.string() for _ in range(n)]
        return [read_value(r, et) for _ in range(n)]
    raise RuntimeError("bad type %d" % t)

with open(PATH, "rb") as f:
    r = R(f)
    magic = f.read(4)
    print("magic", magic)
    ver = r.u32(); nt = r.u64(); nkv = r.u64()
    print("version", ver, "tensors", nt, "kvs", nkv)
    meta = {}
    for _ in range(nkv):
        k = r.string(); t = r.u32(); v = read_value(r, t)
        meta[k] = v
    tensors = []
    for _ in range(nt):
        name = r.string(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]
        tt = r.u32(); off = r.u64()
        tensors.append((name, dims, tt, off))
    data_start = f.tell()
print("data_start", data_start)
print("=== metadata ===")
for k in sorted(meta):
    v = meta[k]
    s = repr(v)
    if len(s) > 300: s = s[:300] + "...(len %d)" % len(v)
    print("  %s = %s" % (k, s))
print("=== tensor names with blk.47 / blk.48 ===")
for (name, dims, tt, off) in tensors:
    if name.startswith("blk.47.") or name.startswith("blk.48."):
        nbytes = 1
        for d in dims: nbytes *= d
        print("  %-52s dims=%-24s type=%s off=%d" % (name, str(dims), GGML.get(tt, tt), off))
print("=== all tensor name prefixes (non blk) ===")
for (name, dims, tt, off) in tensors:
    if not name.startswith("blk."):
        print("  %-52s dims=%-24s type=%s" % (name, str(dims), GGML.get(tt, tt)))
print("=== layer tensor-name set for blk.47 vs blk.48 ===")
def suffixes(prefix):
    out = []
    for (name, dims, tt, off) in tensors:
        if name.startswith(prefix):
            out.append((name[len(prefix):], tuple(dims), GGML.get(tt, str(tt))))
    return out
a = dict((s, (d, t)) for s, d, t in suffixes("blk.47."))
b = dict((s, (d, t)) for s, d, t in suffixes("blk.48."))
for k in sorted(set(a) | set(b)):
    print("  %-40s blk47=%-40s blk48=%s" % (k, a.get(k), b.get(k)))
