#!/usr/bin/env python3
"""Task M end-to-end run on the Colab VM: tokenizer parity, streaming demo, parity.

Phases (all in one detached remote job so the GPU session is used once):

  1. build the CUDA engine (incremental)
  2. build llama-tokenize from the pinned llama.cpp commit (CPU only)
  3. tokenizer comparison on the real Qwen3.8 vocabulary:
       llama-tokenize -f <raw prompt> --ids --no-escape
       warodecode --tokenize-only --prompt-file <raw prompt>
     for every line of tests/qwen38_tokenizer_corpus.txt
  4. streaming demo: -p "..." --chat-template -n 48 (text must arrive token by token)
  5. prompt parity: -p "" -n 64 --stream-json ... vs issue1_artifacts/reference_tokens_64.json
  6. legacy parity: the historical --start-token JSON command, unchanged
  7. interactive demo: two lines through -i

Session exclusivity (docs/COLAB_SESSIONS.md): the remote worker refuses to start
while another run owns /content/run_bg.pid.

Usage:
  export PATH=/home/sahen/.local/bin:$PATH
  WARODECODE_COLAB_SESSION=warpdecode-a100 python3 tools/tokenizer_colab.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import zipfile

import colab_run_bg as runner

ROOT = runner.ROOT
SESSION = runner.SESSION
MODEL = runner.MODEL
ARCH = os.environ.get("WARODECODE_COLAB_ARCH", "89")
ENGINE = "/content/warodecode/build-sm%s/warodecode" % ARCH
LLAMA = "/content/llama.cpp"
LLAMA_COMMIT = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11"
REFERENCE_SHA256 = runner.REFERENCE_SHA256
LOG = "/content/run_tok.log"
PREFIX = "qwen38-task-m"
MODEL_URL = ("https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/"
             "resolve/main/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")
MODEL_BYTES = 85923705344

REMOTE = r'''#!/bin/bash
# Detached Task M run. Everything is appended to /content/run_tok.log.
cd /content/warodecode || exit 1
MODEL="__MODEL__"
ENGINE="__ENGINE__"
ARCH="__ARCH__"
LLAMA="__LLAMA__"
LLAMA_COMMIT="__LLAMA_COMMIT__"
REFERENCE="/content/reference_tokens_64.json"

step() { echo "TOK_PHASE $1 elapsed=$(($(date +%s) - START))s"; }
START=$(date +%s)

# 1 session = 1 job: refuse to start while another run owns the VM. A finished
# job is a zombie until it is reaped, and kill -0 succeeds for zombies, so the
# process state is read from /proc instead.
if [ -f /content/run_bg.pid ]; then
  OTHER=$(cat /content/run_bg.pid 2>/dev/null)
  if [ -n "$OTHER" ] && [ -r "/proc/$OTHER/stat" ]; then
    STATE=$(awk '{print $3}' "/proc/$OTHER/stat" 2>/dev/null)
    case "$STATE" in
      Z|X|"") ;;
      *) echo "TOK_BUSY another run owns /content/run_bg.pid (pid=$OTHER state=$STATE)"
         echo TOK_ALL_DONE
         exit 3 ;;
    esac
  fi
fi
echo $$ > /content/run_tok.pid

step download_model
if [ ! -f "$MODEL" ] || [ "$(stat -c %s "$MODEL" 2>/dev/null || echo 0)" != "__MODEL_BYTES__" ]; then
  mkdir -p "$(dirname "$MODEL")"
  curl -sL --retry 5 --retry-delay 5 -C - -o "$MODEL" "__MODEL_URL__"
  SIZE=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
  echo "TOK_MODEL_BYTES=$SIZE"
  if [ "$SIZE" != "__MODEL_BYTES__" ]; then echo TOK_MODEL_FAILED; echo TOK_ALL_DONE; exit 1; fi
fi
echo TOK_MODEL_OK

step build_engine
if [ ! -d "build-sm$ARCH" ]; then
  cmake -S . -B "build-sm$ARCH" -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >> /content/build_tok.log 2>&1
fi
cmake --build "build-sm$ARCH" -j4 --target warodecode >> /content/build_tok.log 2>&1
BUILD_EXIT=$?
echo "TOK_BUILD_EXIT=$BUILD_EXIT"
tail -6 /content/build_tok.log
if [ ! -x "$ENGINE" ]; then echo TOK_ENGINE_FAILED; echo TOK_ALL_DONE; exit 1; fi
echo TOK_ENGINE_OK

step build_llama
if [ ! -x "$LLAMA/build-tok/bin/llama-tokenize" ]; then
  if [ ! -d "$LLAMA/.git" ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA" >> /content/llama_tok.log 2>&1 \
      || { echo TOK_LLAMA_CLONE_FAILED; echo TOK_ALL_DONE; exit 1; }
  fi
  git -C "$LLAMA" fetch --depth 1 origin "$LLAMA_COMMIT" >> /content/llama_tok.log 2>&1
  git -C "$LLAMA" checkout -q FETCH_HEAD >> /content/llama_tok.log 2>&1
  cmake -S "$LLAMA" -B "$LLAMA/build-tok" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
    -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF >> /content/llama_tok.log 2>&1
  cmake --build "$LLAMA/build-tok" -j4 --target llama-tokenize >> /content/llama_tok.log 2>&1
fi
if [ ! -x "$LLAMA/build-tok/bin/llama-tokenize" ]; then echo TOK_LLAMA_FAILED; tail -5 /content/llama_tok.log; echo TOK_ALL_DONE; exit 1; fi
echo TOK_LLAMA_OK

step tokenizer_compare
python3 /content/tokenizer_compare.py "$MODEL" "$ENGINE" "$LLAMA/build-tok/bin/llama-tokenize" \
  /content/warodecode/tests/qwen38_tokenizer_corpus.txt /content/tokenizer_compare.json 2>&1 | tail -30
echo "TOK_COMPARE_EXIT=$?"

step stream_demo
timeout 2400 "$ENGINE" --model "$MODEL" --device 0 --ctx 256 --no-graph --no-autotune --generate 48 \
  -p "こんにちは。あなたの名前と得意なことを1文で教えてください。" --chat-template \
  --stream-json /content/stream_demo.json > /content/stream_demo.raw 2> /content/stream_demo.err
echo "TOK_DEMO_EXIT=$?"
python3 /content/stream_tap.py /content/stream_demo.raw /content/stream_demo.timeline | tail -25
echo "--- demo json ---"; cat /content/stream_demo.json 2>/dev/null | head -c 600; echo
echo "--- demo stderr ---"; tail -6 /content/stream_demo.err

step parity_prompt
timeout 2400 "$ENGINE" --model "$MODEL" --device 0 --ctx 128 --no-graph --no-autotune --generate 64 \
  -p "" --no-display-prompt --stream-json /content/stream_parity.json \
  > /content/stream_parity.out 2> /content/stream_parity.err
echo "TOK_PARITY_PROMPT_EXIT=$?"
python3 - "$REFERENCE" /content/stream_parity.json <<'PY' | tee /content/stream_parity.check
import json, sys
reference = json.load(open(sys.argv[1]))["token_ids"]
stream = json.load(open(sys.argv[2]))
got = stream["token_ids"]
match = 0
for a, b in zip(reference, got):
    if a != b:
        break
    match += 1
print("PROMPT_PARITY %d/%d" % (match, len(reference)))
print("first mismatch:", None if match == len(reference) else match)
print("prompt_tokens:", stream["prompt_token_ids"])
print("stopped_on_eos:", stream["stopped_on_eos"], "prefill_s:", stream["prefill_seconds"], "decode_s:", stream["decode_seconds"])
PY

step parity_legacy
timeout 2400 "$ENGINE" --model "$MODEL" --ctx 128 --device 0 --start-token 1 --no-graph --no-autotune \
  --generate 64 > /content/legacy_parity.json 2> /content/legacy_parity.err
echo "TOK_PARITY_LEGACY_EXIT=$?"
python3 - "$REFERENCE" /content/legacy_parity.json <<'PY' | tee /content/legacy_parity.check
import json, sys
reference = json.load(open(sys.argv[1]))["token_ids"]
line = [l for l in open(sys.argv[2]) if l.startswith('{"engine":"warodecode-qwen38"')]
if not line:
    print("LEGACY_PARITY run failed")
    raise SystemExit(1)
record = json.loads(line[-1])
got = record["token_ids"]
match = 0
for a, b in zip(reference, got):
    if a != b:
        break
    match += 1
print("LEGACY_PARITY %d/%d" % (match, len(reference)))
print("tok_s:", record["tok_s"], "token_hash:", record["token_hash"])
PY

step interactive
printf 'こんにちは\n1+1は？\n/exit\n' | timeout 2400 "$ENGINE" --model "$MODEL" --device 0 --ctx 256 \
  --no-graph --no-autotune --generate 24 -i --chat-template --stream-json /content/interactive.json \
  > /content/interactive_demo.txt 2> /content/interactive.err
echo "TOK_INTERACTIVE_EXIT=$?"
tail -20 /content/interactive_demo.txt

echo TOK_ALL_DONE
'''
# ---------------------------------------------------------------------------
# Remote helper: tokenizer comparison (llama-tokenize vs warodecode)
# ---------------------------------------------------------------------------

COMPARE = r'''#!/usr/bin/env python3
"""Compare warodecode --tokenize-only with llama-tokenize on one GGUF.

usage: tokenizer_compare.py MODEL ENGINE LLAMA_TOKENIZE CORPUS OUT_JSON

Both sides read the identical bytes from a scratch file (llama-tokenize -f and
warodecode --prompt-file), so escape handling can never be the reason for a
difference. Every corpus line runs in the default (model-driven BOS) mode; the
first five lines also run with --no-bos on both sides.
"""
import json
import pathlib
import subprocess
import sys


def unescape(line):
    out = []
    i = 0
    while i < len(line):
        if line[i] != "\\" or i + 1 >= len(line):
            out.append(line[i])
            i += 1
            continue
        nxt = line[i + 1]
        i += 2
        out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(nxt, "\\" + nxt))
    return "".join(out)


def main():
    model, engine, llama_tok, corpus, out_json = sys.argv[1:6]
    lines = pathlib.Path(corpus).read_text(encoding="utf-8").splitlines()
    prompt_path = pathlib.Path("/tmp/tok_prompt.bin")
    cases = []
    for index, line in enumerate(lines):
        text = unescape(line)
        prompt_path.write_bytes(text.encode("utf-8"))
        modes = [("default", [], [])]
        if index < 5:
            modes.append(("no_bos", ["--no-bos"], ["--no-bos"]))
        for mode, llama_flags, warodecode_flags in modes:
            llama = subprocess.run(
                [llama_tok, "-m", model, "-f", str(prompt_path), "--ids", "--no-escape"] + llama_flags,
                capture_output=True, text=True, timeout=3600)
            waro = subprocess.run(
                [engine, "--model", model, "--tokenize-only", "--prompt-file", str(prompt_path)] + warodecode_flags,
                capture_output=True, text=True, timeout=3600)
            llama_ids = None
            for candidate in reversed(llama.stdout.splitlines()):
                if candidate.startswith("["):
                    llama_ids = json.loads(candidate)
                    break
            waro_record = None
            for candidate in reversed(waro.stdout.splitlines()):
                if candidate.startswith("{"):
                    waro_record = json.loads(candidate)
                    break
            waro_ids = waro_record["token_ids"] if waro_record else None
            case = {
                "index": index,
                "mode": mode,
                "text": text,
                "llama": llama_ids,
                "warodecode": waro_ids,
                "match": llama_ids is not None and llama_ids == waro_ids,
                "llama_stderr": llama.stderr[-300:] if llama_ids is None else "",
                "warodecode_stderr": waro.stderr[-300:] if waro_ids is None else "",
                "bos": waro_record["add_bos"] if waro_record else None,
                "parse_special": waro_record["parse_special"] if waro_record else None,
            }
            cases.append(case)
            print("case %3d mode=%-8s match=%-3s n_llama=%s n_waro=%s text=%s" % (
                index, mode, "yes" if case["match"] else "NO",
                len(llama_ids) if llama_ids else "-", len(waro_ids) if waro_ids else "-",
                json.dumps(text, ensure_ascii=False)[:60]))

    mismatches = [case for case in cases if not case["match"]]
    summary = {
        "model": model,
        "engine": engine,
        "llama_tokenize": llama_tok,
        "cases": len(cases),
        "matches": len(cases) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:5],
    }
    pathlib.Path(out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("TOK_MATCH=%s" % ("yes" if not mismatches else "no"))
    print("TOK_SUMMARY cases=%d matches=%d" % (summary["cases"], summary["matches"]))
    if mismatches:
        first = mismatches[0]
        print("first mismatch: text=%s" % json.dumps(first["text"], ensure_ascii=False))
        print("  llama:      %s" % first["llama"])
        print("  warodecode: %s" % first["warodecode"])
        if first["llama_stderr"] or first["warodecode_stderr"]:
            print("  stderr llama=%r warodecode=%r" % (first["llama_stderr"], first["warodecode_stderr"]))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
'''

# ---------------------------------------------------------------------------
# Remote helper: byte-level arrival timeline (proof that tokens are streamed)
# ---------------------------------------------------------------------------

TAP = r'''#!/usr/bin/env python3
"""Timestamp every byte that arrives on stdin (used as: engine ... | stream_tap.py).

usage: stream_tap.py RAW_OUT TIMELINE_OUT

Byte-at-a-time reads make the arrival time of each streamed chunk visible, so
the log proves the CLI flushes per token instead of printing at the end.
"""
import sys
import time

raw = open(sys.argv[1], "wb")
timeline = open(sys.argv[2], "w")
start = time.time()
last = start
total = 0
events = 0
while True:
    chunk = sys.stdin.buffer.read(1)
    if not chunk:
        break
    raw.write(chunk)
    raw.flush()
    total += 1
    now = time.time()
    if now - last >= 0.05 or chunk in (b"\n", b"\r"):
        timeline.write("T+%7.3f bytes=%4d %s\n" % (
            now - start, total, repr(chunk.decode("utf-8", "replace"))))
        timeline.flush()
        events += 1
        last = now
timeline.write("TOTAL bytes=%d elapsed=%.3f events=%d\n" % (total, time.time() - start, events))
timeline.close()
raw.close()
'''

def _starter(script_b64: str) -> str:
    return '''import base64, os, pathlib, subprocess, zipfile
pathlib.Path("/content/tokenizer_compare.py").write_bytes(base64.b64decode("__COMPARE_B64__"))
pathlib.Path("/content/stream_tap.py").write_bytes(base64.b64decode("__TAP_B64__"))
zipfile.ZipFile("/content/warodecode-source.zip").extractall("/content/warodecode")
pathlib.Path("/content/run_tok.sh").write_bytes(base64.b64decode("__SCRIPT_B64__"))
pathlib.Path("/content/run_tok.log").unlink(missing_ok=True)
proc = subprocess.Popen(["bash", "/content/run_tok.sh"], start_new_session=True,
                        stdout=open("/content/run_tok.log", "w"), stderr=subprocess.STDOUT)
pathlib.Path("/content/run_tok.pid").write_text(str(proc.pid))
print("STARTED", proc.pid)
'''


def remote_lock_state() -> str:
    # A finished detached run stays a zombie until the kernel reaps it, and
    # os.kill(pid, 0) still succeeds for zombies. Read /proc/<pid>/stat instead
    # so a finished job is not mistaken for a running one.
    probe = '''import pathlib
pidf = pathlib.Path("/content/run_bg.pid")
busy = False
if pidf.exists():
    try:
        pid = int(pidf.read_text().strip())
        stat = pathlib.Path("/proc/%d/stat" % pid).read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        busy = state not in ("Z", "X")
    except Exception:
        busy = False
print("REMOTE_LOCK=" + ("BUSY" if busy else "FREE"))
'''
    with tempfile.TemporaryDirectory(prefix="warodecode-tok-lock-") as tmp:
        path = pathlib.Path(tmp) / "lock.py"
        path.write_text(probe)
        out = runner.run(runner.CLI + ["exec", "-s", SESSION, "--timeout", "120", "-f", str(path)],
                         check=False)
    # Only an explicit FREE starts the run: an exec that failed or was queued
    # behind another job must never be read as "the session is idle".
    return "FREE" if "REMOTE_LOCK=FREE" in out else "BUSY"


def main() -> None:
    raw = (ROOT / "issue1_artifacts/reference_tokens_64.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise SystemExit("canonical reference SHA-256 mismatch")

    state = remote_lock_state()
    print("+ remote lock: %s" % state, flush=True)
    if state == "BUSY":
        print("session %s は他担当が使用中です。空くまで待って再実行してください。" % SESSION)
        raise SystemExit(3)

    with tempfile.TemporaryDirectory(prefix="warodecode-tok-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "src.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("src", "tests", "tools", "bench"):
                for f in (ROOT / name).rglob("*"):
                    if f.is_file() and "qwen38-fixture" not in f.parts:
                        z.write(f, f.relative_to(ROOT))
            for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
                f = ROOT / name
                if f.is_file():
                    z.write(f, name)
        print("+ upload source (%d bytes)" % archive.stat().st_size, flush=True)
        runner.run(runner.CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"])
        ref_path = tmp_path / "reference_tokens_64.json"
        ref_path.write_bytes(raw)
        runner.run(runner.CLI + ["upload", "-s", SESSION, str(ref_path), "/content/reference_tokens_64.json"])

        script = (REMOTE
                  .replace("__MODEL__", MODEL)
                  .replace("__ENGINE__", ENGINE)
                  .replace("__ARCH__", ARCH)
                  .replace("__LLAMA__", LLAMA)
                  .replace("__LLAMA_COMMIT__", LLAMA_COMMIT)
                  .replace("__MODEL_URL__", MODEL_URL)
                  .replace("__MODEL_BYTES__", str(MODEL_BYTES)))
        starter = tmp_path / "start.py"
        starter.write_text(
            _starter(script)
            .replace("__SCRIPT_B64__", base64.b64encode(script.encode()).decode())
            .replace("__COMPARE_B64__", base64.b64encode(COMPARE.encode()).decode())
            .replace("__TAP_B64__", base64.b64encode(TAP.encode()).decode()))
        print("+ start detached remote run", flush=True)
        runner.run(runner.CLI + ["exec", "-s", SESSION, "--timeout", "600", "-f", str(starter)])

        deadline = time.monotonic() + float(os.environ.get("WARODECODE_TOKENIZER_TIMEOUT", "3600"))
        local_log = tmp_path / "run_tok.log"
        log_text = ""
        while time.monotonic() < deadline:
            time.sleep(30)
            local_log.unlink(missing_ok=True)
            subprocess.run(runner.CLI + ["download", "-s", SESSION, LOG, str(local_log)],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_text = local_log.read_text(errors="replace") if local_log.exists() else ""
            tail = [line for line in log_text.splitlines() if line.startswith("TOK_")][-1:]
            print("[poll] %s | bytes=%d" % (tail[0] if tail else "(no phase yet)", len(log_text)), flush=True)
            if "TOK_ALL_DONE" in log_text:
                break

        out_dir = ROOT / "issue1_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer_colab_run.log").write_text(log_text)
        print("+ saved remote log -> issue1_artifacts/tokenizer_colab_run.log", flush=True)

        for remote, local in (
                ("/content/tokenizer_compare.json", str(out_dir / "tokenizer_compare_real.json")),
                ("/content/stream_demo.json", str(out_dir / "stream_demo.json")),
                ("/content/stream_demo.timeline", str(out_dir / "stream_demo.timeline")),
                ("/content/stream_demo.raw", str(out_dir / "stream_demo.txt")),
                ("/content/stream_parity.json", str(out_dir / "stream_parity_prompt.json")),
                ("/content/interactive_demo.txt", str(out_dir / "interactive_demo.txt")),
                ("/content/legacy_parity.json", str(out_dir / "legacy_parity.json"))):
            subprocess.run(runner.CLI + ["download", "-s", SESSION, remote, local],
                           text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("+ artifacts under issue1_artifacts/", flush=True)

    if "TOK_BUSY" in log_text:
        print("session busy: 他担当のジョブが動いています")
        raise SystemExit(3)
    for marker in ("TOK_MATCH=yes", "PROMPT_PARITY 64/64", "LEGACY_PARITY 64/64"):
        print("%-24s %s" % (marker, "OK" if marker in log_text else "MISSING"))
    if "TOK_ALL_DONE" not in log_text:
        print("remote run did not finish")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

