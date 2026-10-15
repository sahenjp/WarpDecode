#!/usr/bin/env python3
"""One ceiling-profile case: run the engine, dump the intervals we care about.

Args: name cache_mib switches generate engine_args_bench_repeat ref engine model
Prints one JSON record.
"""
import json, os, pathlib, re, subprocess, sys, time

name, cache_mib, extra, generate, bench_repeat, ref_path, engine, model = sys.argv[1:9]
env = dict(os.environ)
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if cache_mib != "-":
    env["WARODECODE_EXPERT_CACHE_MIB"] = cache_mib
for kv in extra.split(","):
    if kv and "=" in kv:
        key, _, value = kv.partition("=")
        env[key] = value


def foreign():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], text=True,
                         capture_output=True).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


waited = 0
while waited < 300 and foreign():
    time.sleep(10)
    waited += 10

cmd = [engine, "--model", model, "--ctx", "128", "--device", "0",
       "--start-token", "1", "--no-graph", "--no-autotune",
       "--generate", generate, "--bench-repeat", bench_repeat]
foreign_seen = set()
sampler = subprocess.Popen(
    ["bash", "-c",
     "for i in $(seq 1 400); do nvidia-smi --query-compute-apps=pid --format=csv,noheader; "
     "echo ---; sleep 1; done"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
started = time.monotonic()
proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
wall = time.monotonic() - started
sampler.terminate()
try:
    gpu_log = sampler.stdout.read()
except Exception:
    gpu_log = ""
groups = [g for g in gpu_log.split("---")]
max_apps = max([sum(1 for l in g.splitlines() if l.strip()) for g in groups] or [0])

ident = None
for line in reversed(proc.stdout.splitlines()):
    if line.startswith('{"engine":"warodecode-qwen38"'):
        ident = json.loads(line)
        break

# [moe-profile] name dev= 1.234 host= 5.678 gap= -4.444 calls/token=48.00 recorded=.. dropped=.. evfail=..
moe_re = re.compile(
    r"^\[moe-profile\]\s+(\S+)\s+dev=\s*(\S+)\s+host=\s*(\S+)\s+gap=\s*(\S+)\s+"
    r"calls/token=\s*(\S+)\s+recorded=\s*(\S+)\s+dropped=\s*(\S+)\s+evfail=\s*(\S+)")
moe_sites = {}
for line in proc.stderr.splitlines():
    m = moe_re.match(line)
    if m:
        moe_sites[m.group(1)] = {
            "dev": float(m.group(2)), "host": float(m.group(3)), "gap": float(m.group(4)),
            "calls": float(m.group(5)), "recorded": int(m.group(6)),
            "dropped": int(m.group(7)), "evfail": int(m.group(8))}
hc_sites = {}
for line in proc.stderr.splitlines():
    if line.startswith("[hc-profile] ") or line.startswith("[attn-profile] "):
        parts = line.split()
        if len(parts) >= 3 and parts[2] not in ("unit=ms_per_token",):
            try:
                hc_sites[parts[1]] = float(parts[2])
            except ValueError:
                pass

stats = None
for line in reversed(proc.stderr.splitlines()):
    if '"expert_cache_stats"' in line:
        stats = json.loads(line)["expert_cache_stats"]
        break
if stats:
    stats = {k: v for k, v in stats.items() if not isinstance(v, list)}

ref = json.loads(pathlib.Path(ref_path).read_text())["token_ids"]
tokens = (ident or {}).get("token_ids") or []
matched = 0
for a, b in zip(ref, tokens):
    if a != b:
        break
    matched += 1

def ms(x):
    return round(x, 3) if isinstance(x, (int, float)) else None

record = {
    "case": name, "cache_mib": cache_mib, "extra": extra, "bench_repeat": bench_repeat,
    "exit": proc.returncode, "wall_s": round(wall, 2),
    "tok_s": (ident or {}).get("tok_s"),
    "decode_seconds": (ident or {}).get("decode_seconds"),
    "bench_repeats": (ident or {}).get("bench_repeats"),
    "ms_per_token": ms(1000.0 / ident["tok_s"]) if (ident or {}).get("tok_s") else None,
    "matched_prefix": matched,
    "parity_ok": matched == len(ref) and len(tokens) == len(ref),
    "tokens": tokens,
    "profile": {k: v for k, v in ((ident or {}).get("profile") or {}).items()
                if not isinstance(v, list)},
    "detail": {k: v for k, v in ((ident or {}).get("profile_detail") or {}).items()
               if not isinstance(v, list)},
    "moe_sites": moe_sites,
    "hc_sites": hc_sites,
    "stats": stats,
    "max_gpu_apps": max_apps,
    "waited_s": waited,
    "stderr_tail": proc.stderr[-1500:] if (proc.returncode != 0 or ident is None) else "",
}
print(json.dumps(record), flush=True)
