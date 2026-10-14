import mmap
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

import requests

MODEL_URL = (
    "https://huggingface.co/pentacoxian-dev/"
    "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/resolve/main/"
    "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
)

raw = requests.head(MODEL_URL, allow_redirects=False, timeout=60)
location = urlsplit(raw.headers.get("location", ""))
print(
    "INITIAL_HEAD",
    raw.status_code,
    "redirect_host=",
    location.hostname,
    "path_tail=",
    location.path.rsplit("/", 1)[-1][:80],
    "path_ends_slash=",
    location.path.endswith("/"),
)

head = requests.head(MODEL_URL, allow_redirects=True, timeout=60)
print(
    "HEAD",
    head.status_code,
    "bytes=",
    head.headers.get("content-length"),
    "ranges=",
    head.headers.get("accept-ranges"),
    "host=",
    urlsplit(head.url).hostname,
)

part = requests.get(
    MODEL_URL,
    headers={"Range": "bytes=0-4095"},
    allow_redirects=True,
    timeout=60,
)
print("RANGE", part.status_code, "bytes=", len(part.content), "magic=", part.content[:4])

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "fusepy"], check=True)

FUSE_WORKER = r'''import collections
import errno
import json
import os
import requests
import stat
import sys
import threading
import time
from fuse import FUSE, FuseOSError, Operations

url, mountpoint = sys.argv[1:]
name = "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
session = requests.Session()
head = session.head(url, allow_redirects=True, timeout=60)
head.raise_for_status()
total = int(head.headers["content-length"])
block_size = 4 * 1024 * 1024
cache_limit = 32
cache = collections.OrderedDict()
lock = threading.Lock()
requests_read = 0
bytes_read = 0
disk_reads = 0
progress_path = "/content/qwen38-fuse-progress.json"
cache_fd = os.open("/content/qwen38-range-cache.bin", os.O_RDWR | os.O_CREAT, 0o600)
os.ftruncate(cache_fd, total)
cached_blocks = set()

class RemoteModel(Operations):
    def getattr(self, path, fh=None):
        now = int(time.time())
        if path == "/":
            return {"st_mode": stat.S_IFDIR | 0o555, "st_nlink": 2, "st_mtime": now}
        if path == "/" + name:
            return {"st_mode": stat.S_IFREG | 0o444, "st_nlink": 1, "st_size": total, "st_mtime": now}
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        return [".", "..", name] if path == "/" else []

    def open(self, path, flags):
        if path != "/" + name:
            raise FuseOSError(errno.ENOENT)
        return 0

    def read(self, path, size, offset, fh):
        global requests_read, bytes_read, disk_reads
        if path != "/" + name:
            raise FuseOSError(errno.ENOENT)
        if offset >= total:
            return b""
        end = min(offset + size, total)
        output = bytearray()
        while offset < end:
            block = offset // block_size
            start = block * block_size
            key = block
            with lock:
                data = cache.get(key)
                if data is not None:
                    cache.move_to_end(key)
                disk_cached = key in cached_blocks
            if data is None:
                expected = min(block_size, total - start)
                if disk_cached:
                    data = os.pread(cache_fd, expected, start)
                    with lock:
                        disk_reads += 1
                else:
                    stop = start + expected - 1
                    response = session.get(
                        url,
                        headers={"Range": f"bytes={start}-{stop}"},
                        allow_redirects=True,
                        stream=True,
                        timeout=(15, 120),
                    )
                    if response.status_code != 206:
                        response.close()
                        raise FuseOSError(errno.EIO)
                    data = response.content
                    response.close()
                    if len(data) != expected or os.pwrite(cache_fd, data, start) != expected:
                        raise FuseOSError(errno.EIO)
                if len(data) != expected:
                    raise FuseOSError(errno.EIO)
                with lock:
                    cache[key] = data
                    cached_blocks.add(key)
                    while len(cache) > cache_limit:
                        cache.popitem(last=False)
                    if not disk_cached:
                        requests_read += 1
                        bytes_read += len(data)
                    if (requests_read + disk_reads) % 32 == 0:
                        tmp = progress_path + ".tmp"
                        with open(tmp, "w") as f:
                            json.dump({"ranges": requests_read, "bytes": bytes_read,
                                       "disk_reads": disk_reads}, f)
                        os.replace(tmp, progress_path)
            within = offset - start
            count = min(end - offset, len(data) - within)
            output.extend(data[within:within + count])
            offset += count
        return bytes(output)

FUSE(RemoteModel(), mountpoint, foreground=True, ro=True)
'''

mount = pathlib.Path("/content/qwen38-http")
mount.mkdir(exist_ok=True)
keep_mount = False
with tempfile.TemporaryDirectory(prefix="qwen38-fuse-") as tmp:
    worker = pathlib.Path(tmp) / "fuse_worker.py"
    worker.write_text(FUSE_WORKER)
    log = open("/content/qwen38-fuse.log", "w")
    daemon = subprocess.Popen(
        [sys.executable, str(worker), MODEL_URL, str(mount)],
        stdout=log, stderr=log, start_new_session=True,
    )
    pathlib.Path("/content/qwen38-fuse.pid").write_text(str(daemon.pid))
    try:
        target = mount / "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
        for _ in range(60):
            if os.path.ismount(mount) and target.exists():
                break
            if daemon.poll() is not None:
                break
            time.sleep(1)
        if not os.path.ismount(mount) or not target.exists():
            log.flush()
            print("FUSE_LOG", pathlib.Path(log.name).read_text(errors="replace")[-2000:])
            raise RuntimeError("remote GGUF FUSE mount failed: " + log.name)

        size = target.stat().st_size
        with target.open("rb") as f:
            fd = f.fileno()
            print("FUSE_STAT", size)
            print("FUSE_PREAD_MAGIC", os.pread(fd, 4, 0))
            with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mm:
                print("FUSE_MMAP_MAGIC", mm[:4])
                offset = (size // 2) & ~4095
                print("FUSE_MMAP_RANDOM", offset, mm[offset:offset + 4])
        print("FUSE_DISK_FREE", shutil.disk_usage("/").free, "sparse_mount_ok")
        engine = pathlib.Path("/content/warodecode/build-sm75/warodecode")
        command = [
            str(engine), "--model", str(target), "--ctx", "128", "--device", "0",
            "--start-token", "1", "--generate", "1", "--no-graph", "--top-k", "10",
        ]
        print("RUN", " ".join(command), flush=True)
        output = open("/content/qwen38-forward.log", "w")
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        pathlib.Path("/content/qwen38-forward.pid").write_text(str(process.pid))
        output.close()
        keep_mount = True
        print("WARP_FORWARD_STARTED", process.pid, flush=True)
    finally:
        if not keep_mount:
            subprocess.run(["fusermount", "-u", str(mount)], check=False, capture_output=True)
            if daemon.poll() is None:
                daemon.terminate()
                daemon.wait(timeout=10)
        log.close()
