#!/usr/bin/env python3
"""Task P: engine A/B for the attention/HC kernel work.

Runs the canonical 64-token greedy decode for a list of configurations, checks
64/64 parity against the reference token file, and collects:
  * the Task F per-stage profile (attn / hc_mix_attn / hc_mix_ffn / ...),
  * the Task J per-kernel HC profile (WARODECODE_HC_PROFILE=1),
  * the Task P per-kernel attention profile (WARODECODE_ATTN_PROFILE=1).

Usage (on the VM):
  python3 tests/taskp_attn_ab.py --plan ab --generate 64 --out /content/taskp_ab.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

MODEL = os.environ.get("WD_MODEL", "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
REF = os.environ.get("WD_REF", "/content/reference_tokens_64.json")
ENGINE = os.environ.get("WD_ENGINE", "/content/warodecode/build-sm80/warodecode")

KERNEL_LINE = re.compile(r"^\[(hc|attn)-profile\]\s+(\S+)\s+(-?[0-9.]+)\s+calls/token=([0-9.]+)")

PLANS = {
    # Interleaved so a drift in the VM shows up as base-a != base-b.
    "ab": [
        ("base-a", {}),
        ("dp4a-a", {"WARODECODE_Q8_DP4A_GEMV": "1"}),
        ("base-b", {}),
        ("dp4a-b", {"WARODECODE_Q8_DP4A_GEMV": "1"}),
    ],
    # Task P switches: each improvement on its own against the same baseline,
    # then the combination, then the baseline again to show the drift.
    "taskp": [
        ("base-a", {"WARODECODE_Q8_DP4A_GEMV": "0", "WARODECODE_GDN_F32_GEMV_WARP": "0"}),
        ("dp4a", {"WARODECODE_Q8_DP4A_GEMV": "1", "WARODECODE_GDN_F32_GEMV_WARP": "0"}),
        ("warp", {"WARODECODE_Q8_DP4A_GEMV": "0", "WARODECODE_GDN_F32_GEMV_WARP": "1"}),
        ("both", {"WARODECODE_Q8_DP4A_GEMV": "1", "WARODECODE_GDN_F32_GEMV_WARP": "1"}),
        ("base-b", {"WARODECODE_Q8_DP4A_GEMV": "0", "WARODECODE_GDN_F32_GEMV_WARP": "0"}),
    ],
    # The shipped defaults with no switch set at all: proves that the default
    # path (both switches ON) is the one the A/B measured.
    "defaults": [
        ("default-a", {}),
        ("default-b", {}),
    ],
    # Percent-only comparison inside one measurement session: four repeats each,
    # alternated, with the profilers off so they cannot perturb the timing.
    "tok": [
        ("base-1", {}),
        ("dp4a-1", {"WARODECODE_Q8_DP4A_GEMV": "1"}),
        ("base-2", {}),
        ("dp4a-2", {"WARODECODE_Q8_DP4A_GEMV": "1"}),
        ("base-3", {}),
        ("dp4a-3", {"WARODECODE_Q8_DP4A_GEMV": "1"}),
    ],
}


def source_hashes() -> dict:
    root = pathlib.Path("/content/warodecode/src")
    out = {}
    for name in ("qwen38_hc.cuh", "qwen38_gdn.cuh", "qwen38_qsa.cuh", "lowbit.cuh",
                 "qwen38_forward.cuh"):
        path = root / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def parse_identity(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith('{"engine":"warodecode-qwen38'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def parse_kernel_profile(text: str) -> dict:
    profiles = {}
    for line in text.splitlines():
        m = KERNEL_LINE.match(line)
        if not m:
            continue
        which, site, ms, calls = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        profiles.setdefault(which, {})[site] = {"ms_per_token": ms, "calls_per_token": calls}
    return profiles


def free_vram_mib() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    return out.splitlines()[0] if out else "?"


def run_case(name, env_extra, reference, generate, out_dir):
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU tier default is 4 GiB (src/qwen38_tuning.hpp); 16 GiB keeps the
    # routed experts mostly resident like the Task J runs without leaving the
    # 40 GiB card one cudaMalloc away from an out-of-memory abort. 20000 MiB is
    # what the earlier notes quote and it does fit, but only with a fully idle
    # GPU: measured twice here, an engine started seconds after another one was
    # killed still failed to allocate it.
    # 4 GiB is also the tuning default for a 40 GiB card (src/qwen38_tuning.hpp).
    # The expert cache size only moves the MoE stage, which this task does not
    # touch, so a small cache keeps the engine inside whatever VRAM is free
    # without changing the attention / HC numbers being compared.
    env["WARODECODE_EXPERT_CACHE_MIB"] = env.get("WARODECODE_EXPERT_CACHE_MIB", "4096")
    env["WARODECODE_PROFILE"] = "1"
    env["WARODECODE_HC_PROFILE"] = "1"
    env["WARODECODE_ATTN_PROFILE"] = "1"
    env.update(env_extra)
    cmd = [ENGINE, "--model", MODEL, "--ctx", "128", "--device", "0",
           "--start-token", "1", "--no-graph", "--no-autotune",
           "--generate", str(generate)]
    # The engine aborts on a failed cudaMalloc, so a GPU that is still releasing
    # a killed process' context costs one retry: wait, then run once more.
    attempts = 0
    while True:
        attempts += 1
        print("    free_vram_mib=%s attempt=%d" % (free_vram_mib(), attempts), flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        wall = time.monotonic() - t0
        blob = proc.stdout + proc.stderr
        if "out of memory" not in blob or attempts >= 3:
            break
        print("    out of memory, retrying after 30 s", flush=True)
        time.sleep(30)
    log = out_dir / (name + ".log")
    log.write_text(proc.stdout + "\n=== STDERR ===\n" + proc.stderr)
    ident = parse_identity(proc.stdout)
    rec = {"case": name, "env": env_extra, "wall_s": round(wall, 2), "exit": proc.returncode,
           "log": str(log)}
    if ident is None:
        rec["error"] = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        return rec
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-4:])
    rec["engine_tail"] = tail
    tokens = ident.get("token_ids") or []
    matched = 0
    while matched < min(len(tokens), len(reference)) and tokens[matched] == reference[matched]:
        matched += 1
    rec.update({
        "tok_s": ident.get("tok_s"),
        "parity": "%d/%d" % (matched, len(reference)),
        "parity_n": matched,
        "profile": ident.get("profile"),
        "kernel_profile": parse_kernel_profile(proc.stderr),
    })
    return rec


def print_table(results):
    print("")
    print("%-10s %9s %8s %10s %10s %10s %10s" % (
        "case", "tok_s", "parity", "attn_ms", "hcmix_a", "hcmix_f", "total_ms"))
    for r in results:
        if "error" in r:
            print("%-10s FAILED %s" % (r["case"], r["error"][:90]))
            continue
        p = r.get("profile") or {}
        print("%-10s %9.3f %8s %10s %10s %10s %10s" % (
            r["case"], r["tok_s"], r["parity"],
            p.get("attn_ms", "-"), p.get("hc_mix_attn_ms", "-"),
            p.get("hc_mix_ffn_ms", "-"), p.get("total_ms", "-")))
        kp = r.get("kernel_profile") or {}
        for which in ("attn", "hc"):
            sites = kp.get(which)
            if not sites:
                continue
            # Biggest first: that is the order a reader acts on.
            for site, v in sorted(sites.items(), key=lambda kv: -kv[1]["ms_per_token"]):
                print("    [%s] %-22s %8.4f ms/token  calls=%.2f" % (
                    which, site, v["ms_per_token"], v["calls_per_token"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="ab", choices=sorted(PLANS))
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--out", default="/content/taskp_ab.json")
    parser.add_argument("--out-dir", default="/content/taskp_logs")
    args = parser.parse_args()

    reference = json.loads(pathlib.Path(REF).read_text())["token_ids"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    print("source sha256[:16]:", json.dumps(hashes))

    results = []
    for name, env_extra in PLANS[args.plan]:
        print("=== case %s env=%s ===" % (name, env_extra), flush=True)
        rec = run_case(name, env_extra, reference, args.generate, out_dir)
        rec["source_sha16"] = hashes
        results.append(rec)
        print("    tok_s=%s parity=%s wall=%ss" % (
            rec.get("tok_s"), rec.get("parity", rec.get("error", ""))[:60], rec.get("wall_s")),
            flush=True)
        if "error" in rec:
            # A failing case usually means the VM is in a bad state (stale GPU
            # process, missing model); repeating it three more times only burns
            # the session budget with the same error.
            print("    engine tail:\n%s" % rec.get("engine_tail", ""), flush=True)
            print("    aborting plan after the first failure", flush=True)
            break

    print_table(results)
    pathlib.Path(args.out).write_text(json.dumps(
        {"plan": args.plan, "generate": args.generate, "source_sha16": hashes,
         "results": results}, indent=2) + "\n")
    print("TASKP_AB_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
