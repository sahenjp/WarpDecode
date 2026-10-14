#!/usr/bin/env python3
"""Task H: A/B the expert staging read path on the Colab VM.

Runs the engine over the canonical 64-token greedy decode for a list of
WARODECODE_* configurations, checks 64/64 parity against the reference token
file and collects the WARODECODE_EXPERT_STATS JSON of every run.

Usage (on the VM):  python3 tests/expert_io_ab.py --plan a100 [--repeat 2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MODEL = os.environ.get("WD_MODEL", "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
REF = os.environ.get("WD_REF", "/content/reference_tokens_64.json")
ENGINE = os.environ.get("WD_ENGINE", "/content/warodecode/build-sm80/warodecode")


def foreign_gpu_processes() -> list:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def parse_identity(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith('{"engine":"warodecode-qwen38"'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def parse_stats(text: str):
    for line in reversed(text.splitlines()):
        if '"expert_cache_stats"' in line:
            try:
                return json.loads(line)["expert_cache_stats"]
            except Exception:
                return None
    return None


def run_case(name: str, env_extra: dict, reference: list, generate: int) -> dict:
    env = dict(os.environ)
    env.update({
        "WARODECODE_EXPERT_STATS": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    env.update(env_extra)
    cmd = [ENGINE, "--model", MODEL, "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(generate)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    wall = time.monotonic() - t0
    ident = parse_identity(proc.stdout)
    stats = parse_stats(proc.stderr)
    rec = {"case": name, "env": env_extra, "wall_s": round(wall, 2),
           "exit": proc.returncode}
    if ident is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        rec["error"] = " || ".join(tail)[-600:]
        return rec
    tokens = ident.get("token_ids", [])
    mismatch = next((i for i, (a, b) in enumerate(zip(reference, tokens)) if a != b), None)
    rec["matched_prefix"] = mismatch if mismatch is not None else len(tokens)
    rec["parity_ok"] = rec["matched_prefix"] == generate
    for key in ("tok_s", "seconds", "decode_seconds", "decode_tokens", "prefill_tokens",
                "token_hash"):
        if key in ident:
            rec[key] = ident[key]
    if stats:
        for key in ("capacity_mib", "slots", "hits", "misses", "hit_rate", "bytes_copied",
                    "gib_copied", "stage_ms", "acquire_ms", "stage_share", "host_ms_per_token",
                    "io_pread_ms", "io_ring_wait_ms", "io_served", "io_fallbacks",
                    "io_enabled", "io_ring", "io_fadvise", "io_threads"):
            if key in stats:
                rec[key] = stats[key]
        if stats.get("stage_ms") and stats.get("gib_copied"):
            rec["stage_gb_s"] = round(stats["gib_copied"] * 1024 ** 3 / (stats["stage_ms"] / 1000.0) / 1e9, 3)
    return rec


def plans(plan: str, repeat: int):
    """Yield (name, env) cases. A "base" case is the pageable path, "tN" the
    pinned pread ring with N threads."""
    def case(tag, mib, threads):
        env = {"WARODECODE_EXPERT_CACHE_MIB": str(mib)}
        if threads is None:
            env["WARODECODE_EXPERT_IO_PREAD"] = "0"
        else:
            env["WARODECODE_EXPERT_IO_PREAD"] = "1"
            env["WARODECODE_EXPERT_IO_THREADS"] = str(threads)
        return ("%s-%d-%s" % (plan, mib, tag), env)

    if plan in ("default", "default_l4"):
        sizes = (1536, 8000) if plan == "default_l4" else (1536, 20000)
        yield ("warmup", {"WARODECODE_EXPERT_CACHE_MIB": str(sizes[0]),
                          "WARODECODE_EXPERT_IO_PREAD": "0"})
        for i in range(repeat):
            for mib in sizes:
                yield ("default-%d-default-%d" % (mib, i), {"WARODECODE_EXPERT_CACHE_MIB": str(mib)})
                yield ("default-%d-off-%d" % (mib, i), {"WARODECODE_EXPERT_CACHE_MIB": str(mib),
                                                        "WARODECODE_EXPERT_IO_PREAD": "0"})
        return
    if plan == "a100":
        combos = [(1536, None), (1536, 1), (1536, 2), (1536, 4),
                  (20000, None), (20000, 1), (20000, 4)]
    elif plan == "l4":
        combos = [(1536, None), (1536, 1), (1536, 3),
                  (8000, None), (8000, 1), (8000, 3)]
    else:
        combos = [(1536, None), (1536, 1), (1536, 4)]
    yield ("warmup", dict(case("base", combos[0][0], None)[1]))
    for i in range(repeat):
        for mib, threads in combos:
            tag = "base" if threads is None else "t%d" % threads
            name, env = case(tag, mib, threads)
            yield ("%s-%d" % (name, i), env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--out", default="/content/wd_h/taskh_ab.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--plan", default="a100")
    args = ap.parse_args()

    foreign = foreign_gpu_processes()
    if foreign:
        print("WARNING: foreign GPU processes: %s" % foreign, flush=True)

    reference = json.load(open(REF))["token_ids"][:args.generate]
    results = []
    for name, env_extra in plans(args.plan, args.repeat):
        if args.only and args.only not in name:
            continue
        rec = run_case(name, env_extra, reference, args.generate)
        rec["foreign_gpu"] = foreign
        results.append(rec)
        print(json.dumps(rec), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
    print("AB_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
