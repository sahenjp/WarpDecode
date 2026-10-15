#!/usr/bin/env python3
"""Upload the source tree to the GPU session and build just the engine.

Unlike tools/llamacpp_ab.py --phases build_engine this does not also build the
llama.cpp harness (which costs 15+ minutes), so it is the quick path for work
that only needs the engine (dense model tests).
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llamacpp_ab as ab

STEP = """#!/bin/bash
LOG=@WORK@/build_only.log
mkdir -p @WORK@
exec > "$LOG" 2>&1
set -x
date -u
W=@WORK@
rm -rf $W/warodecode
mkdir -p $W/warodecode
cd $W/warodecode
unzip -q -o $W/warodecode-source.zip
cmake -S . -B build-sm@ARCH@ -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON -DCMAKE_CUDA_ARCHITECTURES=@ARCH@ 2>&1 | tail -3
cmake --build build-sm@ARCH@ -j 8 --target warodecode > build_full.log 2>&1
echo "ENGINE_BUILD_EXIT=$?"
grep -n -m 6 -A 4 "error" build_full.log | head -40
ls -la $W/warodecode/build-sm@ARCH@/warodecode
date -u
echo BUILD_ONLY_DONE
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wd-build-") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "warodecode-source.zip"
        ab.make_source_archive(archive)
        ab.exec_inline("import pathlib; pathlib.Path(%r).mkdir(parents=True, exist_ok=True)" % ab.WORK)
        ab.upload(archive, ab.WORK + "/warodecode-source.zip")
        local = tmp_path / "build_only.sh"
        local.write_text(STEP.replace("@WORK@", ab.WORK).replace("@ARCH@", ab.ARCH))
        ab.upload(local, ab.WORK + "_build_only.sh")
        ab.launch("build_only.sh", "build_only.log")
        ab.wait_for_log(ab.WORK + "/build_only.log", pathlib.Path("/tmp/wd_build.log"),
                        "BUILD_ONLY_DONE", 3600, "build")
        log = pathlib.Path("/tmp/wd_build.log")
        print(log.read_text(errors="replace")[-1500:] if log.exists() else "(no log)")


if __name__ == "__main__":
    main()
