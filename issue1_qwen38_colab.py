from __future__ import annotations

import pathlib
import subprocess
import tempfile
import zipfile

SESSION = "warpdecode-qwen38-t4"
ROOT = pathlib.Path(__file__).resolve().parent
CLI = ["rtk", "uvx", "--from", "google-colab-cli", "colab", "--auth=oauth2"]

REMOTE = r'''import json
import pathlib
import shutil
import subprocess
import zipfile

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, flush=True)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout

root = pathlib.Path("/content/warodecode")
with zipfile.ZipFile("/content/warodecode-source.zip") as z:
    z.extractall(root)

gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).strip()
nvcc = run(["nvcc", "--version"])
run(["cmake", "--version"])
run(["cmake", "-S", str(root), "-B", str(root / "build-sm75"),
     "-DCMAKE_BUILD_TYPE=Release", "-DWARODECODE_BUILD_ENGINE=ON",
     "-DCMAKE_CUDA_ARCHITECTURES=75"])
run(["cmake", "--build", str(root / "build-sm75"), "-j2"])
lowbit = run([str(root / "build-sm75/warodecode-lowbit-test")])

disk = shutil.disk_usage("/content")
print("COLAB_T4_RESULT=" + json.dumps({
    "gpu": gpu,
    "cuda_toolkit": nvcc.splitlines()[-1].strip(),
    "lowbit": [line for line in lowbit.splitlines() if " PASS " in line],
    "model_bytes": 85923705344,
    "free_disk_bytes": disk.free,
    "model_fits_local_disk": disk.free >= 85923705344,
}, ensure_ascii=False), flush=True)
'''


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def make_source_archive(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("src", "tests", "tools", "bench"):
            for file in (ROOT / name).rglob("*"):
                if file.is_file() and "qwen38-fixture" not in file.parts:
                    archive.write(file, file.relative_to(ROOT))
        for name in ("CMakeLists.txt", "README.md", "setup.sh", ".gitignore"):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="warodecode-colab-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "warodecode-source.zip"
        worker = tmp_path / "issue1_qwen38_colab_remote.py"
        make_source_archive(archive)
        worker.write_text(REMOTE)

        run(CLI + ["upload", "-s", SESSION, str(archive), "/content/warodecode-source.zip"])
        run(CLI + ["exec", "-s", SESSION, "--timeout", "1800", "-f", str(worker)])


if __name__ == "__main__":
    main()
