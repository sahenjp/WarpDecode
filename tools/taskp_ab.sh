#!/bin/bash
# Task P A/B driver.
#
# The A100 session is shared with the other in-flight task. Its engine can hold
# 36 of the 40 GiB while its Colab driver is already gone, so the only thing that
# helps is waiting for a window; nothing here kills a foreign process.
set -u
report() {
  echo "--- $(date -u +%H:%M:%S) ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
  ps -eo pid,etimes,cmd | grep '[w]arodecode --model' | head -10
}
echo "=== GPU state before ==="
report
echo "=== waiting for >= 20000 MiB free (up to 25 min) ==="
window=0
for i in $(seq 1 50); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d " ")
  if [ "${free_mib:-0}" -ge 20000 ]; then window=1; echo "window open at ${free_mib} MiB free ($(date -u +%H:%M:%S))"; break; fi
  if [ $((i % 10)) -eq 0 ]; then echo "still waiting: ${free_mib} MiB free ($(date -u +%H:%M:%S))"; fi
  sleep 30
done
report
if [ "$window" -ne 1 ]; then echo "NO_GPU_WINDOW: another engine held the card for 25 min"; fi
echo "=== engine run ==="
python3 tests/taskp_attn_ab.py --plan taskp --generate 64 --out /content/taskp_ab.json
echo TASKP_RUN_EXIT=$?