#!/usr/bin/env python3
"""Task H: read-only environment probe for the expert staging bandwidth work."""
import json, os, subprocess

MODEL = os.environ.get("WD_MODEL", "/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf")

def sh(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.stdout.strip()
    except Exception as exc:
        return "ERR: %s" % exc

out = {}
out["nproc"] = sh("nproc")
out["meminfo"] = sh("grep -E 'MemTotal|MemFree|MemAvailable|Buffers|^Cached|Dirty' /proc/meminfo")
out["df"] = sh("df -hT /content /tmp /")
out["lsblk"] = sh("lsblk -o NAME,SIZE,ROTA,TYPE,MOUNTPOINT 2>/dev/null | head -20")
out["mount"] = sh("findmnt -T /content 2>/dev/null")
out["model"] = sh("ls -la " + MODEL)
out["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv")
out["nvidia_apps"] = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
out["procs"] = sh("ps -eo pid,etimes,rss,cmd | grep -E 'warodecode' | grep -v grep | head -20")
out["readahead_kb"] = sh("cat /sys/block/*/queue/read_ahead_kb 2>/dev/null | head -5")
out["thp"] = sh("cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null")
out["toolchain"] = sh("gcc --version | head -1; nvcc --version | tail -2")
print(json.dumps(out, indent=1))
