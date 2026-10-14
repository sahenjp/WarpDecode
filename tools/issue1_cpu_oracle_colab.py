#!/usr/bin/env python3
"""CPU oracle internals at positions 0 and 10 on the existing Colab session.

Issue #1: the warp (CUDA) engine does not reproduce the pinned llama.cpp
generation.  The position sweep showed the layer-0 output already differs by
NRMSE 0.0141 at position 0, i.e. before any history exists.  This script builds
the CPU oracle on the Colab VM and captures the CPU side of the vectors the
engine records in tools/issue1_warp_internals.py, so that a follow-up
tools/issue1_compare_internals.py can attribute the layer-0 error block by
block (HC mix / GDN / embedding / MoE).

Layout follows tools/issue1_position39_colab_ablation.py:

  local  main()  verify the canonical reference ABI, upload the source zip,
                 ``colab exec -f`` the worker, download the slim trace, write
                 issue1_artifacts/cpu_pos0_internals.json
  remote worker  fetch pinned llama.cpp (PR #28243 head), build CPU only,
                 link qwen38_forced_cpu_trace.cpp, run positions 0 and 10 with
                 the canonical forced prefix, and merge the WDPTRC1 vectors
                 named pos0.* / pos10.* that belong to the selected layers into
                 /content/cpu_pos0_internals.bin

The 86 GB GGUF is already on the VM (/content/models); nothing is downloaded
and no GPU is used.

Position mapping: the engine takes --state-trace-position N; the CPU harness
takes the same value as its fifth positional argument, after
MODEL CTX INPUT_TOKEN_IDS TRACE_BIN.  The forced prefix is canonical:
1,tokens[0],...,tokens[N-1] (BOS at position 0, so the decode at position N sees
exactly the N reference tokens of context).

Environment knobs (all optional):
  WARODECODE_COLAB_SESSION               session name (default warpdecode-l4)
  WARODECODE_CPU_ORACLE_POSITIONS        default "0,10"
  WARODECODE_CPU_ORACLE_LAYERS           default "0,7,8" (layer 0 plus the
                                         layer-7 QSA and layer-8 MoE neighbours)
  WARODECODE_CPU_ORACLE_TIMEOUT          colab exec timeout in seconds (7200)
  WARODECODE_CPU_ORACLE_ALLOW_TOKEN_MISMATCH
                                         "1" keeps the trace even when the
                                         position-0 token does not reproduce
                                         the canonical reference
  WARODECODE_CPU_ORACLE_STRICT=0         alias for the knob above

Dry run (template substitution + compile + archive check, no Colab call):
  python3 tools/issue1_cpu_oracle_colab.py --dry-run
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]
SESSION = os.environ.get("WARODECODE_COLAB_SESSION", "warpdecode-l4")
REFERENCE_SHA256 = "2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
MODEL = "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
MODEL_BYTES = 85923705344
HARNESS = "qwen38_forced_cpu_trace.cpp"
# Canonical greedy tokens of the pinned CPU reference (issue1_artifacts/
# reference_tokens_64.json): position 0 -> 271, position 1 -> 2, position 10 ->
# 4613.  Used both to sanity check the reference file and as the oracle gate.
CANONICAL_TOKENS = {0: 271, 1: 2, 10: 4613}
# CPU-side names the previous CPU/engine comparisons rely on, plus the bases
# qwen38_forced_cpu_trace.cpp asks for.  Purely diagnostic: names that never
# appear are reported so a missing trace point is visible instead of silent.
EXPECTED_LAYER0_NAMES = (
    "model.input_embed",
    "hc_init",
    "hc_norm-0",
    "hc_low-0",
    "hc_gate-0",
    "hc_mixed-0",
    "hc_inject-0",
    "linear_attn_qkv_mixed-0",
    "q_conv_predelta-0",
    "k_conv_predelta-0",
    "v_conv_predelta-0",
    "conv_state_at-0",
    "state_predelta-0",
    "new_state-0",
    "conv_output_silu-0",
    "linear_attn_out-0",
    "attn_output-0",
    "hc_combine-0",
    "ffn_out-0",
    "l_last-0",
)
MAX_ROW_VECTORS = 512

REMOTE = r'''import hashlib
import json
import pathlib
import struct
import subprocess
import time
import zipfile

POSITIONS = __POSITIONS__
LAYERS = __LAYERS__
MODEL = __MODEL__
MODEL_BYTES = __MODEL_BYTES__
LLAMA_COMMIT = __LLAMA_COMMIT__
HARNESS_NAME = __HARNESS_NAME__
FORCED = __FORCED__
EXPECTED = __EXPECTED__
EXPECTED_LAYER0_NAMES = __EXPECTED_LAYER0_NAMES__
ALLOW_TOKEN_MISMATCH = __ALLOW_TOKEN_MISMATCH__
MAX_ROW_VECTORS = __MAX_ROW_VECTORS__

SOURCE_ZIP = pathlib.Path("/content/warodecode-source.zip")
SOURCE_ROOT = pathlib.Path("/content/warodecode")
HARNESS = pathlib.Path("/content/" + HARNESS_NAME)
LLAMA = pathlib.Path("/content/llama.cpp")
BUILD = LLAMA / "build-cpu-oracle"
EXE = BUILD / "bin/qwen38-forced-cpu-trace"
SLIM = pathlib.Path("/content/cpu_pos0_internals.bin")
META = pathlib.Path("/content/cpu_pos0_internals.meta.json")
REMOTE_LOG = pathlib.Path("/content/cpu-oracle-remote.log")
# Same CPU-only recipe as the verified local CPU reference
# (issue1_qwen38_local_reference.py).
CMAKEFLAGS = ["-DGGML_CUDA=OFF", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF",
              "-DGGML_CPU_REPACK=OFF", "-DCMAKE_BUILD_TYPE=Release",
              "-DLLAMA_BUILD_TESTS=OFF"]


def log(message):
    print(message, flush=True)


def run_harness(cmd, timeout=5400):
    # llama.cpp writes progress/log lines to stderr using '\r'; merging the
    # streams corrupts the single-line LLAMA_STATE_RESULT JSON on stdout.
    cmd = [str(part) for part in cmd]
    log("+ " + " ".join(cmd))
    begin = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)
    log("  -> rc=%d in %.1fs (stderr %d bytes)" % (
        proc.returncode, time.monotonic() - begin, len(proc.stderr or "")))
    with REMOTE_LOG.open("a") as handle:
        handle.write("+ " + " ".join(cmd) + "\n[stderr]\n" + (proc.stderr or "") + "\n[stdout]\n" + proc.stdout + "\n")
    if proc.returncode:
        log((proc.stdout or "")[-3000:])
        log((proc.stderr or "")[-3000:])
        raise RuntimeError("harness failed (%d): %s" % (proc.returncode, " ".join(cmd)))
    proc.stdout = proc.stdout.replace("\r", "\n")
    return proc


def run(cmd, timeout=3600, check=True):
    cmd = [str(part) for part in cmd]
    log("+ " + " ".join(cmd))
    begin = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout)
    log("  -> rc=%d in %.1fs" % (proc.returncode, time.monotonic() - begin))
    with REMOTE_LOG.open("a") as handle:
        handle.write("+ " + " ".join(cmd) + "\n" + proc.stdout + "\n")
    if proc.returncode and check:
        log("\n".join(proc.stdout.splitlines()[-60:]))
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(cmd)))
    return proc


def current_head():
    proc = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def prepare_llama():
    if not (LLAMA / ".git").is_dir():
        LLAMA.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", str(LLAMA), "init"])
        run(["git", "-C", str(LLAMA), "remote", "add", "origin",
             "https://github.com/ggml-org/llama.cpp.git"], check=False)
    head = current_head()
    if head != LLAMA_COMMIT:
        run(["git", "-C", str(LLAMA), "fetch", "--depth=1", "origin",
             "refs/pull/28243/head"], timeout=1800)
        run(["git", "-C", str(LLAMA), "checkout", "--detach", "FETCH_HEAD"])
        head = current_head()
    if head != LLAMA_COMMIT:
        raise RuntimeError("unexpected llama.cpp revision: %s" % head)
    log("llama.cpp revision ok: %s" % head)


def add_harness_target():
    cmake_file = LLAMA / "CMakeLists.txt"
    text = cmake_file.read_text()
    if "qwen38-forced-cpu-trace" in text:
        log("CMakeLists.txt already declares qwen38-forced-cpu-trace")
        return
    # The harness includes llama-model.h and ggml-backend.h, so the four
    # include roots match the canonical recipe in issue1_cpu_layer9_state.py
    # (-I include -I src -I ggml/include -I ggml/src).
    block = ("\nadd_executable(qwen38-forced-cpu-trace " + str(HARNESS) + ")\n"
             + "target_link_libraries(qwen38-forced-cpu-trace PRIVATE llama)\n"
             + "target_include_directories(qwen38-forced-cpu-trace PRIVATE "
             + "${CMAKE_CURRENT_SOURCE_DIR}/include "
             + "${CMAKE_CURRENT_SOURCE_DIR}/src "
             + "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include "
             + "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src)\n")
    cmake_file.write_text(text + block)


def build_harness():
    run(["cmake", "-S", str(LLAMA), "-B", str(BUILD)] + CMAKEFLAGS, timeout=1800)
    cache = (BUILD / "CMakeCache.txt").read_text()
    if "GGML_CPU_REPACK:BOOL=OFF" in cache:
        log("GGML_CPU_REPACK=OFF confirmed by CMakeCache")
    elif "GGML_CPU_REPACK" in cache:
        raise RuntimeError("CMakeCache does not hold GGML_CPU_REPACK=OFF")
    else:
        log("WARNING: this llama.cpp revision has no GGML_CPU_REPACK option")
    run(["cmake", "--build", str(BUILD), "-j8", "--target", "qwen38-forced-cpu-trace"],
        timeout=5400)
    if not EXE.is_file():
        raise RuntimeError("harness binary was not produced: %s" % EXE)


def parse_trace(path):
    raw = pathlib.Path(path).read_bytes()
    if raw[:8] != b"WDPTRC1\0":
        raise RuntimeError("bad WDPTRC1 magic: %s" % path)
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 1:
        raise RuntimeError("unsupported WDPTRC1 version: %d" % version)
    records, offset = [], 16
    for _ in range(count):
        name_size, value_count = struct.unpack_from("<IQ", raw, offset)
        offset += 12
        name = raw[offset:offset + name_size].decode()
        offset += name_size
        values = raw[offset:offset + value_count * 4]
        offset += value_count * 4
        records.append((name, value_count, values))
    if offset != len(raw):
        raise RuntimeError("trailing bytes in %s" % path)
    return records


def classify(name, prefix):
    """Return (has_position_prefix, layer_or_None_for_layerless)."""
    if not name.startswith(prefix + "."):
        return (False, None)
    head, sep, tail = name[len(prefix) + 1:].rpartition("-")
    if sep and head and tail.isdigit():
        return (True, int(tail))
    return (True, None)


def write_slim(records, path):
    payload = bytearray()
    for name, count, values in records:
        name_bytes = name.encode()
        payload += struct.pack("<IQ", len(name_bytes), count) + name_bytes + values
    path.write_bytes(b"WDPTRC1\0" + struct.pack("<II", 1, len(records)) + bytes(payload))


begin_all = time.monotonic()
SLIM.unlink(missing_ok=True)
META.unlink(missing_ok=True)

# 1. fresh source archive (it carries qwen38_forced_cpu_trace.cpp)
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    archive.extractall(SOURCE_ROOT)
log("extracted fresh source archive -> %s" % SOURCE_ROOT)
archived_harness = SOURCE_ROOT / HARNESS_NAME
if not archived_harness.is_file():
    raise RuntimeError("source archive is missing %s" % HARNESS_NAME)
HARNESS.write_bytes(archived_harness.read_bytes())
harness_sha = hashlib.sha256(HARNESS.read_bytes()).hexdigest()
log("harness source: %s sha256=%s" % (HARNESS, harness_sha))

model = pathlib.Path(MODEL)
if not model.is_file():
    raise RuntimeError("model is not present on the VM: %s" % model)
if model.stat().st_size != MODEL_BYTES:
    raise RuntimeError("model size mismatch: %d != %d" % (model.stat().st_size, MODEL_BYTES))
log("model: %s (%d bytes)" % (model, MODEL_BYTES))

# 2. pinned llama.cpp, CPU-only build, harness linked against it
prepare_llama()
add_harness_target()
build_harness()
exe_sha = hashlib.sha256(EXE.read_bytes()).hexdigest()
log("harness binary: %s sha256=%s" % (EXE, exe_sha))

# 3. positions 0 and 10, canonical forced prefix, WDPTRC1 vectors
records = []
position_rows = []
pos0_kept_names = set()
for position in POSITIONS:
    prefix = "pos%d" % position
    trace = pathlib.Path("/content/cpu_oracle_pos%d.bin" % position)
    trace.unlink(missing_ok=True)
    log("--- position %d (prefix %s) ---" % (position, prefix))
    begin = time.monotonic()
    proc = run_harness([EXE, MODEL, "128", FORCED[str(position)], trace, str(position)], timeout=5400)
    wall_seconds = time.monotonic() - begin

    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("LLAMA_STATE_RESULT="):
            result = json.loads(line[len("LLAMA_STATE_RESULT="):])
            break
    if result is None:
        raise RuntimeError("harness printed no LLAMA_STATE_RESULT for position %d" % position)
    if result.get("trace_position") != position:
        raise RuntimeError("harness traced position %s, expected %d" % (result.get("trace_position"), position))
    if result.get("qsa_input_override_applied"):
        raise RuntimeError("unexpected QSA input override in the CPU oracle run")

    tokens = result["token_ids"]
    if len(tokens) != position + 1:
        raise RuntimeError("harness returned %d tokens for position %d" % (len(tokens), position))
    expected_sequence = ([EXPECTED[str(i)] for i in range(position + 1)]
                         if all(str(i) in EXPECTED for i in range(position + 1)) else None)
    token_mismatches = []
    if expected_sequence is not None:
        token_mismatches = [{"position": i, "cpu": got, "canonical": want}
                            for i, (got, want) in enumerate(zip(tokens, expected_sequence))
                            if got != want]
    token = tokens[position]
    expected = EXPECTED[str(position)]
    token_match = token == expected
    log("position %d: cpu token=%s canonical=%s match=%s (harness %.1fs, wall %.1fs)"
        % (position, token, expected, token_match, result.get("seconds", -1.0), wall_seconds))

    vectors = parse_trace(trace)
    kept, dropped_layer, dropped_other = 0, 0, 0
    for name, count, values in vectors:
        matched, layer = classify(name, prefix)
        if not matched:
            dropped_other += 1
            continue
        if layer is not None and layer not in LAYERS:
            dropped_layer += 1
            continue
        records.append((name, count, values))
        if position == POSITIONS[0]:
            pos0_kept_names.add(name)
        kept += 1
    log("position %d: kept %d/%d vectors (dropped %d other-layer, %d foreign-prefix)"
        % (position, kept, len(vectors), dropped_layer, dropped_other))

    position_rows.append({
        "position": position,
        "forced_tokens": len(FORCED[str(position)].split(",")),
        "cpu_token": token,
        "expected_token": expected,
        "token_match": token_match,
        "token_mismatches": token_mismatches,
        "token_ids": tokens,
        "token_hash": result.get("token_hash"),
        "top_logits_at_position": (result.get("top_logits", [[]])[position]
                                   if len(result.get("top_logits", [])) > position else []),
        "vectors_total": len(vectors),
        "vectors_kept": kept,
        "vectors_dropped_other_layer": dropped_layer,
        "vectors_dropped_foreign_prefix": dropped_other,
        "integer_vectors": [{"name": entry["name"], "count": len(entry["values"])}
                            for entry in result.get("integer_vectors", [])],
        "tensor_vectors": [{"name": entry["name"], "count": entry["count"]}
                           for entry in result.get("vectors", [])],
        "harness_seconds": result.get("seconds"),
        "wall_seconds": round(wall_seconds, 3),
    })
    trace.unlink(missing_ok=True)

    if not token_match:
        message = "position %d: cpu token %s != canonical %s" % (position, token, expected)
        if position == POSITIONS[0] and not ALLOW_TOKEN_MISMATCH:
            raise RuntimeError(message + " (primary oracle gate; refusing an unvalidated trace)")
        log("WARNING: " + message)

# 4. slim WDPTRC1 file: same order, header count rewritten
if not records:
    raise RuntimeError("no vector matched the position/layer filter")
write_slim(records, SLIM)
slim_bytes = SLIM.stat().st_size
slim_sha = hashlib.sha256(SLIM.read_bytes()).hexdigest()
log("slim trace: %s (%d bytes, %d vectors, sha256=%s)" % (SLIM, slim_bytes, len(records), slim_sha))

row_vectors = []
for name, count, values in records:
    item = {"name": name, "count": count,
            "prefix3": list(struct.unpack_from("<3f", values)) if count >= 3 else []}
    row_vectors.append(item)
missing_names = [name for name in EXPECTED_LAYER0_NAMES
                 if ("pos%d.%s" % (POSITIONS[0], name)) not in pos0_kept_names]
if missing_names:
    log("WARNING: expected layer-0 names absent from the position-0 trace: %s"
        % ", ".join(missing_names))

ok = bool(records) and all(entry["token_match"] for entry in position_rows)
row = {
    "stage": "cpu-oracle",
    "ok": ok,
    "slim_written": True,
    "slim_path": str(SLIM),
    "slim_bytes": slim_bytes,
    "slim_sha256": slim_sha,
    "vector_count": len(records),
    "layers": LAYERS,
    "positions": position_rows,
    "vectors": row_vectors[:MAX_ROW_VECTORS],
    "vectors_truncated": len(row_vectors) > MAX_ROW_VECTORS,
    "missing_expected_layer0_names": missing_names,
    "model": str(model),
    "model_bytes": model.stat().st_size,
    "llama_cpp_commit": LLAMA_COMMIT,
    "harness_source_sha256": harness_sha,
    "harness_binary_sha256": exe_sha,
    "cmake_flags": CMAKEFLAGS,
    "elapsed_seconds": round(time.monotonic() - begin_all, 3),
}
META.write_text(json.dumps({**row, "all_captured_names": [
    {"position": entry["position"], "names": entry["tensor_vectors"]}
    for entry in position_rows]}, indent=2) + "\n")
log("COLAB_ROW=" + json.dumps(row, allow_nan=False))
'''


def _int_list(text: str, name: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values or any(value < 0 for value in values):
        raise RuntimeError(f"{name} must be a non-empty list of non-negative integers")
    return values


POSITIONS = _int_list(os.environ.get("WARODECODE_CPU_ORACLE_POSITIONS", "0,10"),
                      "WARODECODE_CPU_ORACLE_POSITIONS")
LAYERS = _int_list(os.environ.get("WARODECODE_CPU_ORACLE_LAYERS", "0,7,8"),
                   "WARODECODE_CPU_ORACLE_LAYERS")
EXEC_TIMEOUT = os.environ.get("WARODECODE_CPU_ORACLE_TIMEOUT", "7200")
ALLOW_TOKEN_MISMATCH = (
    os.environ.get("WARODECODE_CPU_ORACLE_ALLOW_TOKEN_MISMATCH", "").strip() == "1"
    or os.environ.get("WARODECODE_CPU_ORACLE_STRICT", "").strip() == "0"
)


def cli_env() -> dict[str, str]:
    """PATH that finds rtk even when ~/.local/bin is not in the caller's."""
    env = dict(os.environ)
    local_bin = pathlib.Path.home() / ".local/bin"
    if local_bin.is_dir():
        env["PATH"] = str(local_bin) + os.pathsep + env.get("PATH", "")
    return env


def render_worker(tokens: list[int]) -> str:
    forced = {str(p): ",".join(str(t) for t in [1] + tokens[:p]) for p in POSITIONS}
    expected = {str(p): tokens[p] for p in POSITIONS}
    return (REMOTE
            .replace("__POSITIONS__", json.dumps(POSITIONS))
            .replace("__LAYERS__", json.dumps(LAYERS))
            .replace("__MODEL__", json.dumps(MODEL))
            .replace("__MODEL_BYTES__", str(MODEL_BYTES))
            .replace("__LLAMA_COMMIT__", json.dumps(LLAMA_COMMIT))
            .replace("__HARNESS_NAME__", json.dumps(HARNESS))
            .replace("__FORCED__", json.dumps(forced))
            .replace("__EXPECTED__", json.dumps(expected))
            .replace("__EXPECTED_LAYER0_NAMES__", json.dumps(list(EXPECTED_LAYER0_NAMES)))
            .replace("__ALLOW_TOKEN_MISMATCH__", "True" if ALLOW_TOKEN_MISMATCH else "False")
            .replace("__MAX_ROW_VECTORS__", str(MAX_ROW_VECTORS)))


def verify_worker(worker: str) -> None:
    leftovers = sorted(set(re.findall(r"__[A-Z0-9_]+__", worker)))
    if leftovers:
        raise RuntimeError(f"worker template has unreplaced placeholders: {leftovers}")
    compile(worker, "run_cpu_oracle.py", "exec")
    for token in ("POSITIONS", "LAYERS", "MODEL_BYTES", "LLAMA_COMMIT", "FORCED",
                  "EXPECTED", "EXPECTED_LAYER0_NAMES", "HARNESS_NAME",
                  "ALLOW_TOKEN_MISMATCH", "build-cpu-oracle"):
        if token not in worker:
            raise RuntimeError(f"worker template is missing {token}")
    if "GGML_CUDA=OFF" not in worker or "CMakeLists.txt" not in worker:
        raise RuntimeError("worker template lost the CPU-only cmake recipe")
    if "WDPTRC1" not in worker or "slim" not in worker:
        raise RuntimeError("worker template lost the WDPTRC1 slim step")


def make_source_archive(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("src", "tests", "tools", "bench"):
            for file in (ROOT / name).rglob("*"):
                if file.is_file() and "qwen38-fixture" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
        for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore", HARNESS):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)
    with zipfile.ZipFile(path) as archive:
        if HARNESS not in archive.namelist():
            raise RuntimeError(f"source archive does not contain {HARNESS}")


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("canonical reference SHA-256 mismatch; refusing to run")
    tokens = json.loads(raw)["token_ids"]
    if max(POSITIONS) >= len(tokens):
        raise RuntimeError(f"reference has {len(tokens)} tokens; positions {POSITIONS} exceed it")
    for position, token in CANONICAL_TOKENS.items():
        if position < len(tokens) and tokens[position] != token:
            raise RuntimeError(
                f"canonical reference token at position {position} is {tokens[position]}, expected {token}")

    worker_text = render_worker(tokens)
    verify_worker(worker_text)
    dry_run = "--dry-run" in sys.argv

    with tempfile.TemporaryDirectory(prefix="warodecode-cpu-oracle-") as temp:
        temp_path = pathlib.Path(temp)
        archive = temp_path / "warodecode-source.zip"
        make_source_archive(archive)
        print(f"+ source archive {archive.stat().st_size} bytes, {HARNESS} included", flush=True)

        if dry_run:
            print("dry run: no Colab command executed", flush=True)
            print(f"  reference sha256 : {digest}", flush=True)
            print(f"  positions        : {POSITIONS}", flush=True)
            print(f"  layers           : {LAYERS}", flush=True)
            print(f"  token mismatch ok: {ALLOW_TOKEN_MISMATCH}", flush=True)
            for position in POSITIONS:
                forced = [1] + tokens[:position]
                assert position < len(forced)
                if position:
                    assert forced[position] == tokens[position - 1]
                print(f"  pos{position}: forced={len(forced)} tokens, "
                      f"trace_position={position}, expected token={tokens[position]}, "
                      f"prefix={forced[:6]}{'...' if position > 6 else ''}", flush=True)
            print(f"  worker           : {len(worker_text)} chars, compile() OK, "
                  f"no placeholder left", flush=True)
            print("  worker head      : " + worker_text.splitlines()[0], flush=True)
            print("  worker tail      : " + worker_text.rstrip().splitlines()[-1], flush=True)
            print("DRY_RUN=OK", flush=True)
            return

        upload = CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"]
        print("+ " + " ".join(upload), flush=True)
        subprocess.run(upload, check=True, env=cli_env())

        worker = temp_path / "run_cpu_oracle.py"
        worker.write_text(worker_text)
        cmd = CLI + ["exec", "-s", SESSION, "--timeout", str(EXEC_TIMEOUT), "-f", str(worker)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=cli_env())
        print(proc.stdout[-4000:], flush=True)

        rows = [json.loads(line[len("COLAB_ROW="):]) for line in proc.stdout.splitlines()
                if line.startswith("COLAB_ROW=")]
        row = rows[-1] if rows else None
        out_path = ROOT / "issue1_artifacts/cpu_pos0_internals.bin"
        downloaded_bytes = None
        if row is not None and row.get("slim_written"):
            out_path.unlink(missing_ok=True)
            download = CLI + ["download", "-s", SESSION,
                              row.get("slim_path", "/content/cpu_pos0_internals.bin"),
                              str(out_path)]
            print("+ " + " ".join(download), flush=True)
            subprocess.run(download, check=True, env=cli_env())
            downloaded_bytes = out_path.stat().st_size

        result = {
            "status": ("completed" if row and row.get("ok")
                       else "mismatch" if row and row.get("slim_written")
                       else "INCOMPLETE"),
            "returncode": proc.returncode,
            "reference_sha256": digest,
            "session": SESSION,
            "positions": POSITIONS,
            "layers": LAYERS,
            "row": row,
            "downloaded_bytes": downloaded_bytes,
            "slim_sha256": (hashlib.sha256(out_path.read_bytes()).hexdigest()
                            if downloaded_bytes is not None else None),
        }
        (ROOT / "issue1_artifacts/cpu_pos0_internals.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "row"},
                         indent=2), flush=True)

        print("===== REMOTE STDOUT (tail) =====", flush=True)
        print(proc.stdout[-12000:] if proc.returncode or row is None
              else proc.stdout[-3000:], flush=True)
        if proc.returncode:
            raise RuntimeError(f"CPU oracle run failed ({proc.returncode})")
        if row is None or not row.get("slim_written"):
            raise RuntimeError("CPU oracle run produced no slim trace")


if __name__ == "__main__":
    main()
