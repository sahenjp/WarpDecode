#!/bin/bash
# Sequential MTP A/B runs on one Colab session, saving the engine JSON record
# (tok_s, mtp_* fields) next to the runner's colab_run_<label>.json artifact.
#
#   usage: tools/mtp_colab_bench.sh <session> <arch> <label>::<switches>[::<generate>] ...
set -u
export PATH=/home/sahen/.local/bin:$PATH
SESSION=$1; shift
ARCH=$1; shift
export WARODECODE_COLAB_SESSION="$SESSION"
export WARODECODE_COLAB_ARCH="$ARCH"
CLI=(rtk uvx --from google-colab-cli colab --auth=oauth2)
for spec in "$@"; do
  LABEL=${spec%%::*}
  rest=${spec#*::}
  SWITCHES=${rest%%::*}
  GEN=64
  if [ "$rest" != "$SWITCHES" ]; then GEN=${rest#*::}; fi
  echo "=== $LABEL switches=$SWITCHES generate=$GEN"
  python3 tools/colab_run_bg.py --label "$LABEL" --generate "$GEN" --switches "$SWITCHES" || echo "RUNNER_FAILED $LABEL"
  rm -f "/tmp/mtp_run_$LABEL.log"
  timeout 300 "${CLI[@]}" download -s "$SESSION" /content/run_bg.log "/tmp/mtp_run_$LABEL.log" >/dev/null 2>&1
  python3 - "$LABEL" <<'PY'
import json, pathlib, sys
label = sys.argv[1]
src = pathlib.Path("/tmp/mtp_run_%s.log" % label)
if not src.exists():
    print("NO LOG", label); sys.exit(0)
text = src.read_text(errors="replace")
rec = [l for l in text.splitlines() if l.startswith('{"engine":')]
if not rec:
    print("NO ENGINE RECORD", label); print("\n".join(text.splitlines()[-6:])); sys.exit(0)
j = json.loads(rec[-1])
for k in ("token_ids", "profile_detail"):
    j.pop(k, None)
pathlib.Path("issue1_artifacts/mtp_engine_%s.json" % label).write_text(json.dumps(j, indent=1) + "\n")
keys = ["tok_s","seconds","mtp","mtp_draft","mtp_calls","mtp_accepts","mtp_accept_rate",
        "mtp_full_rounds","mtp_draft_seconds","mtp_draft_ms","mtp_stage_norm_s",
        "mtp_stage_attn_s","mtp_stage_moe_s","mtp_stage_head_s"]
print(json.dumps({k: j.get(k) for k in keys}))
PY
done
