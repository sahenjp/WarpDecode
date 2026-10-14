import json
import os
import pathlib
import shutil
import subprocess

pid_file = pathlib.Path("/content/qwen38-forward.pid")
log_file = pathlib.Path("/content/qwen38-forward.log")
if not pid_file.exists():
    pid_file = pathlib.Path("/content/qwen38-load-only.pid")
    log_file = pathlib.Path("/content/qwen38-load-only.log")
progress_file = pathlib.Path("/content/qwen38-fuse-progress.json")
pid = int(pid_file.read_text()) if pid_file.exists() else None
try:
    os.kill(pid, 0)
    running = pathlib.Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
except (TypeError, ProcessLookupError, FileNotFoundError):
    running = False

progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
log = log_file.read_text(errors="replace").splitlines() if log_file.exists() else []
fuse_log_file = pathlib.Path("/content/qwen38-fuse.log")
fuse_log = fuse_log_file.read_text(errors="replace").splitlines() if fuse_log_file.exists() else []
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader"],
    text=True, capture_output=True,
).stdout.strip()
proc = subprocess.run(
    ["ps", "-o", "pid,stat,%cpu,%mem,rss,etime,cmd", "-p", str(pid)],
    text=True, capture_output=True,
).stdout.strip()
print(json.dumps({
    "pid": pid,
    "running": running,
    "fuse_progress": progress,
    "gpu": gpu,
    "process": proc,
    "fuse_log_tail": fuse_log[-8:],
    "free_disk": shutil.disk_usage("/").free,
    "log_tail": log[-20:],
}, ensure_ascii=False))
