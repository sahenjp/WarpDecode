#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: warodecode_gdn_state_layer_cli_test.py PATH_TO_WARODECODE")

    executable = sys.argv[1]
    for option in (
        "--debug-layer9-gdn-state-layer",
        "--debug-layer9-gdn-conv-state-layer",
        "--debug-layer8-output-layer",
    ):
        result = subprocess.run(
            [executable, option, "10", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, f"{option} was rejected:\n{result.stdout}"
        assert option in result.stdout, f"{option} is missing from help:\n{result.stdout}"

    with tempfile.NamedTemporaryFile() as norm_file:
        norm_file.write(bytes(10240 * 4))
        norm_file.flush()
        result = subprocess.run(
            [
                executable,
                "--debug-hc-attn-q8_0-full-layer",
                "10",
                "--debug-hc-norm-f32-layer",
                "10",
                "--debug-hc-norm-f32-file",
                norm_file.name,
                "--state-trace",
                "/tmp/unused-state-trace.bin",
                "--state-trace-position",
                "36",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    assert result.returncode != 0
    assert "full Q8_0 HC replay layer must be in 0..8" not in result.stdout, result.stdout
    assert "HC norm override requires a layer in 0..8" not in result.stdout, result.stdout
    assert "--state-trace and --forced-input-tokens must be used together" in result.stdout, result.stdout


if __name__ == "__main__":
    main()
