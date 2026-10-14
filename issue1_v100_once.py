#!/usr/bin/env python3
"""Run Issue #1 parity once, only when the V100 is already idle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import textwrap

DEFAULT_REMOTE = "sahen@z170-desktop.tail5ecbed.ts.net"
DEFAULT_KEY = "/home/sahen/.ssh/warodecode_v100_ed25519"
DEFAULT_ROOT = "~/warodecode-v100-run-20260928"
DEFAULT_MODEL = "~/llama-models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
DEFAULT_LOCAL_BINARY = pathlib.Path(__file__).resolve().parent / "build-sm70-cuda128-static/warodecode"
DEFAULT_REFERENCE_JSON = pathlib.Path(__file__).resolve().parent / "issue1_artifacts/reference_tokens_64.json"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
MODEL_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"
MODEL_FILE = "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=2",
    "-o", "ServerAliveCountMax=2",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "StrictHostKeyChecking=accept-new",
]

REMOTE_RUNNER = r'''
import datetime
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

cfg = json.loads(@@CONFIG@@)
root = pathlib.Path(os.path.expanduser(cfg["root"]))
binary = pathlib.Path(os.path.expanduser(cfg["binary"]))
model = pathlib.Path(os.path.expanduser(cfg["model"]))


def stop(message, code=75):
    print("ISSUE1_V100_SKIP=" + message, flush=True)
    raise SystemExit(code)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_is_idle():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader", "-i", str(cfg["device"])],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if query.returncode:
        print("GPU query failed: " + query.stderr.strip(), flush=True)
        return False
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        print("GPU query did not return exactly one device", flush=True)
        return False
    fields = [field.strip() for field in rows[0].split(",")]
    try:
        name = fields[0]
        used, total, free, util = [int(field.split()[0]) for field in fields[1:]]
    except (IndexError, ValueError):
        print("Could not parse GPU state: " + rows[0], flush=True)
        return False
    if "V100" not in name:
        print(f"GPU {cfg['device']} is {name}, not V100", flush=True)
        return False

    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
         "-i", str(cfg["device"])],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if processes.returncode:
        print("GPU process query failed: " + processes.stderr.strip(), flush=True)
        return False
    min_free = math.ceil(total * cfg["min_free_ratio"])
    print(f"GPU {cfg['device']}: {name}, {used} MiB used, {free}/{total} MiB free, {util}% busy", flush=True)
    if processes.stdout.strip() or util > cfg["max_util_percent"] or free < min_free:
        print("GPU is not naturally idle enough; leaving all processes untouched", flush=True)
        return False
    return True


def parse_warodecode(stdout):
    for line in reversed(stdout.splitlines()):
        if not line.lstrip().startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("engine") == "warodecode-qwen38":
            return record
    raise RuntimeError("warodecode JSON result not found")


def vector_diff(reference, candidate):
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return None
    if len(reference) != len(candidate) or not reference:
        return {"lengths": [len(reference), len(candidate)]}
    errors = [abs(a - b) for a, b in zip(reference, candidate)]
    return {
        "length": len(reference),
        "max_abs": max(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def read_trace_binary(path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != b"WDPTRC1\0":
        raise RuntimeError("invalid extended parity trace header")
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 1:
        raise RuntimeError(f"unsupported extended parity trace version: {version}")
    offset = 16
    vectors = {}
    for _ in range(count):
        if offset + 12 > len(raw):
            raise RuntimeError("truncated extended parity trace entry")
        name_size, value_count = struct.unpack_from("<IQ", raw, offset)
        offset += 12
        byte_count = value_count * 4
        end_name = offset + name_size
        end_values = end_name + byte_count
        if end_values > len(raw):
            raise RuntimeError("truncated extended parity trace vector")
        name = raw[offset:end_name].decode("utf-8")
        values = list(struct.unpack_from(f"<{value_count}f", raw, end_name)) if value_count else []
        if name in vectors:
            raise RuntimeError(f"duplicate extended parity trace vector: {name}")
        vectors[name] = values
        offset = end_values
    if offset != len(raw):
        raise RuntimeError("unexpected trailing data in extended parity trace")
    return vectors


def first_trace_divergence(reference, candidate_vectors):
    if not reference:
        return {"stage": None, "reason": "saved llama.cpp record has no trace vectors"}
    ref_by_name = {}
    for item in reference.get("trace", []):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            ref_by_name.setdefault(item["name"], []).append(item.get("values", []))
    stages = [
        ("hc_mixed", "layer0.hc_mixed_f32", "hc_mixed-0", 1),
        ("router logits", "layer0.router_logits", "layer0.router_logits", 0),
        ("selected expert IDs", "layer0.selected_expert_ids", "expert_ids", 0),
        ("selected expert weights", "layer0.selected_expert_weights", "router_weights", 0),
        ("selected expert outputs", "layer0.expert_rank_", "layer0.expert_rank_", 0),
        ("routed weighted sum", "layer0.routed_weighted_sum_f32", "layer0.routed_weighted_sum_f32", 0),
        ("shared expert output", "layer0.shared_expert_output", "layer0.shared_expert_output", 0),
        ("final MoE output", "layer0.final_moe_output", "ffn_out-0", 0),
        ("layer0 HC FFN combine", "layer0.hc_ffn_combine_residual", "hc_combine-0", 1),
        ("layer0 output", "layer0.output", "layer0.output", 0),
        ("layer1 PLE before", "layer1.ple_pre_residual", "layer1.ple_pre_residual", 0),
        ("layer1 PLE after", "layer1.ple_post_residual", "layer1.ple_post_residual", 0),
        ("layer1 HC norm", "layer1.hc_attn_norm", "layer1.hc_attn_norm", 0),
        ("layer1 HC gate", "layer1.hc_attn_gate", "layer1.hc_attn_gate", 0),
        ("layer1 HC mix", "layer1.hc_attn_mix", "layer1.hc_attn_mix", 0),
        ("layer1 GDN/QSA input", "layer1.gdn_input", "layer1.gdn_input", 0),
    ]
    comparisons = []
    missing = []
    for label, candidate_name, reference_name, ref_index in stages:
        candidates = sorted(name for name in candidate_vectors if name.startswith(candidate_name)) if label == "selected expert outputs" else [candidate_name]
        for vector_name in candidates:
            ref_names = [vector_name, reference_name]
            if reference_name.startswith("layer0."):
                ref_names.append(reference_name.split(".", 1)[1])
            ref_values = next((reference[name] for name in ref_names if isinstance(reference.get(name), list)), None)
            refs = next((ref_by_name[name] for name in ref_names if ref_by_name.get(name)), [])
            if len(refs) > ref_index:
                ref_values = refs[ref_index]
            candidate = candidate_vectors.get(vector_name)
            if not isinstance(ref_values, list) or candidate is None:
                missing.append(vector_name)
                continue
            diff = vector_diff(ref_values, candidate)
            if diff is None:
                missing.append(vector_name)
                continue
            comparisons.append({"stage": label, "vector": vector_name, **diff})
            if "max_abs" in diff and (diff["max_abs"] > 1e-4 or diff["rmse"] > 1e-5):
                return {
                    "stage": label,
                    "vector": vector_name,
                    "comparison": diff,
                    "prior_stage_checks": comparisons[:-1],
                    "reference_gaps_before": missing,
                }
    return {"stage": None, "comparisons": comparisons, "no_reference_vector_for": missing}


def save_report(directory, name, report):
    path = directory / name
    path.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print("ISSUE1_V100_REPORT=" + str(path), flush=True)


if cfg.get("preflight_only"):
    if not root.is_dir() or not os.access(root, os.W_OK):
        stop(f"existing run directory is missing or not writable: {root}")
    if not model.is_file():
        stop(f"existing sparse GGUF path not found: {model}; no download or remount")
    if model.name != cfg["model_identity"]["filename"] or model.stat().st_size != cfg["model_identity"]["size"]:
        stop("remote GGUF path or byte size does not match the canonical model identity")
    reference_ids = cfg["reference_ids"]
    reference_source = cfg["reference_source"]
    if reference_ids[0] != 271:
        stop(f"saved llama.cpp reference begins with {reference_ids[0]}, expected 271")
    if len(reference_ids) < 64:
        stop(f"saved llama.cpp reference has only {len(reference_ids)} tokens; 64-token run is unsafe")
    if not gpu_is_idle():
        stop("V100 is occupied or lacks the required free VRAM; no file was copied")
    model_sha256 = file_sha256(model)
    if model_sha256 != cfg["model_identity"]["sha256"]:
        stop("remote GGUF SHA-256 does not match the canonical model identity")
    binary_ready = binary.is_file() and os.access(binary, os.X_OK)
    binary_hash = file_sha256(binary) if binary_ready else None
    help_result = subprocess.run(
        [str(binary), "--help"] if binary_ready else [],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False,
    ) if binary_ready else None
    flags_ready = bool(help_result and help_result.returncode == 0 and
                       "--debug-f32-hc-layer0" in help_result.stdout and
                       "--extended-parity-trace" in help_result.stdout and
                       "--no-graph" in help_result.stdout)
    print("ISSUE1_V100_PREFLIGHT=" + json.dumps({
        "root": str(root), "model": str(model), "reference_source": reference_source,
        "reference_tokens": len(reference_ids), "model_sha256": model_sha256,
        "binary": str(binary),
        "binary_hash": binary_hash, "local_hash": cfg["local_binary_hash"],
        "need_upload": not (binary_hash == cfg["local_binary_hash"] and flags_ready),
    }, separators=(",", ":")), flush=True)
    raise SystemExit(0)


def run_stage(directory, ref_ids, reference, steps, debug):
    if not gpu_is_idle():
        stop(f"GPU no longer idle before {steps}-token stage")
    command = [
        str(binary), "--model", str(model), "--ctx", "16" if debug else "128", "--device", str(cfg["device"]),
        "--start-token", "1", "--generate", str(steps), "--no-graph",
    ]
    trace_file = directory / "token1.extended.trace"
    if debug:
        command.extend([
            "--debug-f32-hc-layer0", "--extended-parity-trace", str(trace_file),
        ])
    print("ISSUE1_V100_STAGE=" + json.dumps({
        "tokens": steps, "debug_f32_hc_layer0": debug, "extended_trace": debug,
    }), flush=True)
    try:
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=cfg["stage_timeout_seconds"], check=False,
        )
    except subprocess.TimeoutExpired:
        stop(f"our {steps}-token command timed out; no other process was touched", 124)
    (directory / f"warodecode-{steps}.json").write_text(proc.stdout)
    (directory / f"warodecode-{steps}.stderr").write_text(proc.stderr)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).splitlines()[-20:]
        stop(f"our {steps}-token command exited {proc.returncode}: {' | '.join(detail)}", proc.returncode)
    result = parse_warodecode(proc.stdout)
    ids = result.get("token_ids")
    if not isinstance(ids, list) or len(ids) != steps:
        stop(f"{steps}-token command returned an invalid token_ids list")
    if result.get("mtp") is not False or result.get("cuda_graph") is not False:
        stop("runtime did not confirm MTP=false and CUDA graph=false")
    if debug and result.get("debug_f32_hc_layer0") is not True:
        stop("runtime did not confirm the layer-0 F32 path")
    trace_meta = result.get("extended_parity_trace") if debug else None
    trace_vectors = None
    if debug:
        if not isinstance(trace_meta, dict) or trace_meta.get("format") != "WDPTRC1-f32le":
            stop("runtime did not confirm the extended binary trace format")
        if trace_meta.get("path") != str(trace_file) or not trace_file.is_file():
            stop("extended parity trace file is missing or has an unexpected path")
        summaries = trace_meta.get("vectors", [])
        summary_by_name = {
            entry.get("name"): entry for entry in summaries
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
        names = set(summary_by_name)
        required = {
            "layer0.hc_mixed", "layer0.hc_mixed_f32", "layer0.router_logits",
            "layer0.selected_expert_ids", "layer0.selected_expert_weights",
            "layer0.routed_weighted_sum_f32", "layer0.routed_moe_output",
            "layer0.shared_expert_output", "layer0.shared_expert_gate",
            "layer0.shared_expert_weighted_output_f32", "layer0.final_moe_output",
            "layer0.hc_ffn_combine_residual", "layer0.output",
            "layer1.ple_pre_residual", "layer1.ple_post_residual",
            "layer1.hc_attn_norm", "layer1.hc_attn_gate", "layer1.hc_attn_mix",
        }
        if not required.issubset(names):
            stop("extended parity trace is missing required vectors: " + ",".join(sorted(required - names)))
        summary_fields = {"count", "min", "max", "mean", "l2", "checksum_fnv1a64"}
        if any(not summary_fields.issubset(summary_by_name[name]) for name in required):
            stop("extended parity trace JSON is missing vector statistics")
        expert_names = sorted(name for name in names if name.startswith("layer0.expert_rank_"))
        if len(expert_names) != 10:
            stop(f"extended trace expected 10 separate selected expert outputs, found {len(expert_names)}")
        if not ({"layer1.gdn_input", "layer1.qsa_input"} & names):
            stop("extended trace is missing the layer-1 attention input")
        trace_vectors = read_trace_binary(trace_file)
        if set(trace_vectors) != names:
            stop("extended binary vector names do not match the JSON summary")

    expected = ref_ids[:steps]
    mismatch = next((i for i, (want, got) in enumerate(zip(expected, ids)) if want != got), None)
    matched = len(expected) == steps and len(ids) == steps and mismatch is None
    prefix_checks = []
    for prefix in (1, 2, 8, 64):
        if prefix > steps:
            continue
        prefix_mismatch = next(
            (i for i, (want, got) in enumerate(zip(ref_ids[:prefix], ids[:prefix])) if want != got),
            None,
        )
        prefix_match = len(ref_ids) >= prefix and len(ids) >= prefix and prefix_mismatch is None
        prefix_checks.append({"tokens": prefix, "match": prefix_match})
        if not prefix_match:
            break
    mismatch_index = mismatch
    mismatch_report = {
        "first_mismatch_index": mismatch_index,
        "expected_token": expected[mismatch_index] if mismatch_index is not None and mismatch_index < len(expected) else None,
        "actual_token": ids[mismatch_index] if mismatch_index is not None and mismatch_index < len(ids) else None,
        "matching_prefix_length": mismatch_index if mismatch_index is not None else (steps if matched else len(ids)),
    }
    print("ISSUE1_V100_STAGE_RESULT=" + json.dumps({
        "tokens": steps, "reference": expected, "warodecode": ids, "match": matched,
        "prefix_checks": prefix_checks, **mismatch_report,
    }, separators=(",", ":")), flush=True)
    if not matched:
        report = {
            "stage_tokens": steps,
            "reference_source": cfg["reference_source"],
            "reference": expected,
            "warodecode": ids,
            **mismatch_report,
            "prefix_checks": prefix_checks,
            "mtp": result.get("mtp"),
            "cuda_graph": result.get("cuda_graph"),
        }
        if debug:
            report["extended_trace_path"] = str(trace_file)
            report["extended_trace_format"] = trace_meta["format"]
            report["trace_vector_summaries"] = trace_meta["vectors"]
            divergence = first_trace_divergence(reference, trace_vectors)
            report["first_trace_divergence"] = divergence
            if not divergence or divergence.get("stage") is None:
                report["layer0_boundary_trace"] = [
                    key for key in (
                        "layer0.hc_mixed_f32", "layer0.routed_weighted_sum_f32",
                        "layer0.shared_expert_output", "layer0.final_moe_output",
                        "layer0.hc_ffn_combine_residual", "layer1.ple_pre_residual",
                        "layer1.ple_post_residual", "layer1.hc_attn_norm",
                        "layer1.hc_attn_gate", "layer1.hc_attn_mix",
                    ) if key in trace_vectors
                ]
                report["top_logits"] = {
                    "llama_cpp": reference.get("top_logits") if reference else None,
                    "warodecode": result.get("top_logits"),
                }
            report["router"] = {
                "expert_ids": result.get("expert_ids"),
                "weights": trace_vectors.get("layer0.selected_expert_weights"),
                "logits_summary": next(
                    (entry for entry in trace_meta["vectors"] if entry["name"] == "layer0.router_logits"), None
                ),
            }
        save_report(directory, "first-divergence.json", report)
        stop(f"first mismatch at generated token index {mismatch}; no longer stage will run", 2)
    return result


if not root.is_dir():
    stop(f"existing run directory not found: {root}")
if not os.access(root, os.W_OK):
    stop(f"cannot save a minimal result under the existing run directory: {root}")
if not binary.is_file() or not os.access(binary, os.X_OK):
    stop(f"prebuilt executable missing or not executable: {binary}; runner will not build")
if not model.is_file():
    stop(f"existing sparse GGUF path not found: {model}; runner will not download or remount")
help_result = subprocess.run(
    [str(binary), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    timeout=10, check=False,
)
if ("--debug-f32-hc-layer0" not in help_result.stdout or
        "--extended-parity-trace" not in help_result.stdout or "--no-graph" not in help_result.stdout):
    stop("prebuilt executable lacks required parity flags; runner will not rebuild")

reference_ids = cfg["reference_ids"]
reference_record = None
reference_source = cfg["reference_source"]
if reference_ids[0] != 271:
    stop(f"saved llama.cpp reference begins with {reference_ids[0]}, expected 271")
if len(reference_ids) < 64:
    stop(f"saved llama.cpp reference has only {len(reference_ids)} tokens; refusing 64-token run")
print("ISSUE1_V100_REFERENCE=" + json.dumps({
    "source": reference_source, "tokens_available": len(reference_ids), "first_token": reference_ids[0],
}, separators=(",", ":")), flush=True)
if not gpu_is_idle():
    stop("V100 is occupied or lacks the required free VRAM; no process was touched")

run_dir = root / ("issue1-once-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
run_dir.mkdir(mode=0o700)
print("ISSUE1_V100_ARTIFACT_DIR=" + str(run_dir), flush=True)
stages = [64]
for steps in stages:
    if len(reference_ids) < steps:
        stop(f"saved llama.cpp sequence has only {len(reference_ids)} tokens; refusing unreferenced {steps}-token run")
    run_stage(run_dir, reference_ids, reference_record, steps, debug=False)

summary = {
    "reference_source": reference_source,
    "llama_cpp_rerun": False,
    "build_performed": False,
    "model_copied_or_remounted": False,
    "tokens_generated": 64,
    "prefix_checks": [1, 2, 8, 64],
    "run_directory": str(run_dir),
}
save_report(run_dir, "parity-summary.json", summary)
print("ISSUE1_V100_COMPLETE", flush=True)
'''


REMOTE_PUBLISHER = r'''
import hashlib
import json
import os
import pathlib
import subprocess
import sys

cfg = json.loads(@@CONFIG@@)
target = pathlib.Path(cfg["target"])
temporary = pathlib.Path(cfg["temporary"])
if not target.parent.is_dir() or not temporary.is_file():
    raise SystemExit("prebuilt binary upload is incomplete")
digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
if digest != cfg["sha256"]:
    raise SystemExit("prebuilt binary checksum mismatch")
help_result = subprocess.run(
    [str(temporary), "--help"], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, check=False,
)
if (help_result.returncode != 0 or "--debug-f32-hc-layer0" not in help_result.stdout or
        "--extended-parity-trace" not in help_result.stdout or "--no-graph" not in help_result.stdout):
    raise SystemExit("uploaded binary is missing required parity options")
dynamic = subprocess.run(["ldd", str(temporary)], text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
if dynamic.returncode not in (0, 1) or "not found" in dynamic.stdout:
    raise SystemExit("uploaded binary has unresolved runtime dependencies: " + dynamic.stdout.strip())
os.chmod(temporary, 0o755)
os.replace(temporary, target)
print("ISSUE1_V100_BINARY=" + json.dumps({"path": str(target), "sha256": digest}, separators=(",", ":")), flush=True)
'''


def build_remote_program(config: dict) -> str:
    program = REMOTE_RUNNER.replace("@@CONFIG@@", repr(json.dumps(config)), 1)
    compile(program, "<issue1-v100-remote-runner>", "exec")
    return textwrap.dedent(program)


def build_publisher(config: dict) -> str:
    program = REMOTE_PUBLISHER.replace("@@CONFIG@@", repr(json.dumps(config)), 1)
    compile(program, "<issue1-v100-binary-publisher>", "exec")
    return textwrap.dedent(program)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--ssh-key", default=DEFAULT_KEY)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--binary")
    parser.add_argument("--local-binary", default=str(DEFAULT_LOCAL_BINARY))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference-json", default=str(DEFAULT_REFERENCE_JSON))
    parser.add_argument("--model-identity-json", default=str(DEFAULT_REFERENCE_JSON.parent / "model_identity.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min-free-ratio", type=float, default=0.95)
    parser.add_argument("--max-util-percent", type=int, default=1)
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without connecting")
    args = parser.parse_args()

    root = args.root
    binary = args.binary or f"{root.rstrip('/')}/warodecode-issue1-extended-sm70"
    reference_file = pathlib.Path(args.reference_json).expanduser()
    if not reference_file.is_file():
        parser.error(f"saved 64-token llama.cpp reference not found: {reference_file}")
    try:
        reference_record = json.loads(reference_file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"invalid saved llama.cpp reference: {error}")
    identity_file = pathlib.Path(args.model_identity_json).expanduser()
    if not identity_file.is_file():
        parser.error(f"model identity manifest not found: {identity_file}")
    try:
        model_identity = json.loads(identity_file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"invalid model identity manifest: {error}")
    runtime_identity_file = reference_file.parent / "reference_runtime_identity.json"
    if not runtime_identity_file.is_file():
        parser.error(f"reference runtime identity record not found: {runtime_identity_file}")
    try:
        runtime_identity = json.loads(runtime_identity_file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"invalid reference runtime identity record: {error}")
    reference_model = reference_record.get("model", {})
    canonical_sha256 = model_identity.get("local_sha256")
    reference_repo = reference_model.get("repo_id", reference_model.get("repo"))
    reference_size = reference_model.get("size", reference_model.get("gguf_size_bytes"))
    if (model_identity.get("repo") != MODEL_REPO or
            model_identity.get("filename") != MODEL_FILE or
            model_identity.get("size") != reference_size or
            reference_repo != MODEL_REPO or
            reference_model.get("filename") != MODEL_FILE):
        parser.error("saved reference model metadata does not match the model identity manifest")
    reference_sha256 = reference_model.get("sha256")
    if reference_sha256 is None:
        recovered = model_identity.get("saved_reference", {})
        if (recovered.get("artifact_sha256") != sha256_file(reference_file) or
                recovered.get("model_sha256") != canonical_sha256 or
                recovered.get("identity_status") != "reconstructed_inferred_and_prefix_verified"):
            parser.error("historical reference lacks an embedded model SHA-256 and has no matching identity record")
        reference_sha256 = recovered["model_sha256"]
    if not isinstance(canonical_sha256, str) or reference_sha256 != canonical_sha256:
        parser.error("saved reference model SHA-256 does not match the canonical model identity")
    reproduced = runtime_identity.get("canonical_cpu_prefix_reproduction", {})
    if (runtime_identity.get("llama_cpp_commit") != reference_record.get("llama_cpp_commit") or
            runtime_identity.get("reference_artifact_sha256") != sha256_file(reference_file) or
            runtime_identity.get("reconstructed_model_sha256") != canonical_sha256 or
            reproduced.get("first_mismatch_index") is not None or
            reproduced.get("matched_saved_prefix") is not True):
        parser.error("reference runtime/model identity record does not validate this saved reference")
    reference_ids = reference_record.get("token_ids")
    if reference_record.get("llama_cpp_commit") != LLAMA_COMMIT:
        parser.error("saved reference was not generated by the pinned llama.cpp commit")
    decode = reference_record.get("decode", {})
    if (not isinstance(reference_ids, list) or len(reference_ids) != 64 or
            decode.get("start_token") != 1 or decode.get("mtp") is not False or
            decode.get("ggml_cpu_repack") is not False or decode.get("ggml_cuda") is not False or
            decode.get("mmap") is not True or decode.get("mlock") is not False):
        parser.error("saved reference must contain 64 tokens generated from raw start token 1")
    if reference_ids[0] != 271:
        parser.error(f"saved reference starts with {reference_ids[0]}, expected 271")
    config = {
        "root": root,
        "binary": binary,
        "model": args.model,
        "reference_source": str(reference_file),
        "reference_ids": reference_ids,
        "model_identity": {
            "repo": MODEL_REPO,
            "filename": MODEL_FILE,
            "size": model_identity["size"],
            "sha256": canonical_sha256,
        },
        "reference_artifact_sha256": runtime_identity["reference_artifact_sha256"],
        "reference_runtime_identity": str(runtime_identity_file),
        "device": args.device,
        "min_free_ratio": args.min_free_ratio,
        "max_util_percent": args.max_util_percent,
        "stage_timeout_seconds": args.stage_timeout_seconds,
    }
    if not 0.5 <= args.min_free_ratio <= 1.0:
        parser.error("--min-free-ratio must be between 0.5 and 1")
    if args.dry_run:
        build_remote_program({**config, "preflight_only": True, "local_binary_hash": "<sha256>"})
        build_publisher({"target": "/tmp/warodecode", "temporary": "/tmp/warodecode.upload",
                         "sha256": "0" * 64})
        print(json.dumps({
            "remote": args.remote,
            "ssh_key": args.ssh_key,
            "local_binary": args.local_binary,
            "config": {key: value for key, value in config.items() if key != "reference_ids"},
            "stages": [{"start_token": 1, "generate": 64, "mtp": False, "cuda_graph": False,
                        "extended_trace": False, "check_prefixes": [1, 2, 8, 64]}],
            "warodecode_commands": [
                [binary, "--model", args.model, "--ctx", "128", "--device", str(args.device),
                 "--start-token", "1", "--generate", str(n), "--no-graph"]
                for n in (64,)
            ],
            "prebuilt_binary": "upload to a separate remote path only after the read-only idle-V100 preflight",
            "build": "never",
            "llama_cpp": "read saved token_ids only; never run",
            "model_cache": "reuse existing path; never copy or remount",
            "gpu_gate": f"V100 device {args.device}, no compute process, utilization <= {args.max_util_percent}%, at least {args.min_free_ratio:.0%} VRAM free",
            "ssh": "ControlMaster=no, ControlPath=none, ConnectionAttempts=1, ServerAliveInterval=2, ServerAliveCountMax=2, ConnectTimeout=5, accept-new",
        }, indent=2))
        return 0

    key = pathlib.Path(args.ssh_key)
    if not key.is_file():
        parser.error(f"SSH key not found: {key}")
    local_binary = pathlib.Path(args.local_binary).expanduser()
    if not local_binary.is_file() or not os.access(local_binary, os.X_OK):
        parser.error(f"prebuilt SM70 binary missing or not executable: {local_binary}; no remote build will run")
    local_hash = sha256_file(local_binary)
    config["local_binary_hash"] = local_hash
    preflight_program = build_remote_program({**config, "preflight_only": True})

    def remote_python(program: str, timeout: int | None = None) -> subprocess.CompletedProcess[str] | None:
        command = ["ssh", "-T", *SSH_OPTIONS, "-i", str(key), args.remote,
                   "/bin/sh -c 'python3 - 1>&2'"]
        try:
            return subprocess.run(
                command, input=program, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as e:
            print(f"SSH command timed out after {timeout}s; no retry", file=sys.stderr, flush=True)
            stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            if stdout:
                print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
            if stderr:
                print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
            return None

    print("Using fresh SSH transport; ControlMaster disabled, no retry/poll loop.", flush=True)
    preflight = remote_python(preflight_program, timeout=21600)
    if preflight is None:
        return 124
    if preflight.stdout:
        print(preflight.stdout, end="", flush=True)
    if preflight.stderr:
        print(preflight.stderr, end="", file=sys.stderr, flush=True)
    if preflight.returncode:
        return preflight.returncode
    preflight_output = preflight.stdout + "\n" + preflight.stderr
    marker = next((line.split("=", 1)[1] for line in reversed(preflight_output.splitlines())
                   if line.startswith("ISSUE1_V100_PREFLIGHT=")), None)
    if marker is None:
        print("remote preflight result missing; stopping before any decode", file=sys.stderr, flush=True)
        return 1
    remote_state = json.loads(marker)
    if remote_state["need_upload"]:
        temporary = remote_state["binary"] + f".upload-{os.getpid()}"
        copied = subprocess.run(
            ["scp", *SSH_OPTIONS, "-i", str(key), str(local_binary), f"{args.remote}:{temporary}"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False,
        )
        if copied.returncode:
            print(copied.stderr or copied.stdout, file=sys.stderr, flush=True)
            return copied.returncode
        publisher = build_publisher({
            "target": remote_state["binary"], "temporary": temporary, "sha256": local_hash,
        })
        published = remote_python(publisher, timeout=15)
        if published is None:
            return 124
        if published.stdout:
            print(published.stdout, end="", flush=True)
        if published.stderr:
            print(published.stderr, end="", file=sys.stderr, flush=True)
        if published.returncode:
            return published.returncode
    else:
        print("Reusing the matching prebuilt SM70 binary.", flush=True)

    config["binary"] = remote_state["binary"]
    config["prebuilt_binary_sha256"] = local_hash
    result = remote_python(build_remote_program(config))
    if result is None:
        return 124
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    result_output = result.stdout + "\n" + result.stderr
    artifact_dir = next((line.split("=", 1)[1] for line in reversed(result_output.splitlines())
                        if line.startswith("ISSUE1_V100_ARTIFACT_DIR=")), None)
    if artifact_dir:
        local_parent = pathlib.Path(tempfile.mkdtemp(prefix="issue1-v100-result-", dir="/tmp/opencode"))
        copied = subprocess.run(
            ["scp", *SSH_OPTIONS, "-r", "-i", str(key),
             f"{args.remote}:{artifact_dir}", str(local_parent)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False,
        )
        if copied.stdout:
            print(copied.stdout, end="", flush=True)
        if copied.stderr:
            print(copied.stderr, end="", file=sys.stderr, flush=True)
        if copied.returncode == 0:
            print("ISSUE1_V100_LOCAL_ARTIFACT_DIR=" + str(local_parent / pathlib.Path(artifact_dir).name), flush=True)
        else:
            print("ISSUE1_V100_ARTIFACT_COPY_FAILED", file=sys.stderr, flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
