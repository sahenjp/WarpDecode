#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
import struct
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: warodecode_position39_cli_test.py PATH_TO_WARODECODE")

    executable = sys.argv[1]
    result = subprocess.run(
        [executable, "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for option in (
        "--debug-router-cpu-activation-f32-file",
        "--debug-moe-routes-file",
        "--debug-routed-moe-output-f32-file",
        "--debug-routed-moe-output-all-positions-f32-file",
        "--debug-layer-output-all-positions-f32-file",
        "--debug-attention-output-f32-layer",
        "--debug-attention-output-f32-file",
        "--debug-hc-attn-combine-layer",
        "--debug-hc-attn-combine-f32-file",
        "--debug-layer-input-residual-layer",
        "--debug-layer-input-residual-f32-file",
        "--debug-attention-input-f32-layer",
        "--debug-attention-input-f32-file",
        "--debug-qsa-context-f32-layer",
        "--debug-qsa-context-f32-file",
        "--debug-qsa-pregate-f32-layer",
        "--debug-qsa-pregate-f32-file",
        "--debug-qsa-query-f32-layer",
        "--debug-qsa-query-f32-file",
        "--debug-qsa-key-store-f32-layer",
        "--debug-qsa-key-store-f32-file",
        "--debug-qsa-value-store-f32-layer",
        "--debug-qsa-value-store-f32-file",
        "--debug-qsa-key-cache-prefix-f32-layer",
        "--debug-qsa-key-cache-prefix-f32-file",
        "--debug-qsa-value-cache-prefix-f32-layer",
        "--debug-qsa-value-cache-prefix-f32-file",
        "--debug-hc-inject-f32-layer",
        "--debug-hc-inject-f32-file",
    ):
        assert option in result.stdout, f"{option} is missing from help:\n{result.stdout}"

    # Exercise parser validation without loading a model. Position 39 must be
    # accepted for the selected-position attention-output diagnostic.
    with tempfile.TemporaryDirectory(prefix="warodecode-position39-cli-") as temp:
        root = Path(temp)
        attention_output = root / "attention_output.f32"
        attention_output.write_bytes(struct.pack("<2560f", *([0.0] * 2560)))
        forced_tokens = ",".join(str(token) for token in [1, *range(2, 41)])
        missing_model = root / "missing.gguf"
        run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-attention-output-f32-layer", "35",
                "--debug-attention-output-f32-file", str(attention_output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "diagnostic override requires state trace position 36" not in run.stdout, run.stdout
        assert "attention output override requires a selected state trace position" not in run.stdout, run.stdout
        assert f"GGUF読込: {missing_model}" in run.stdout, run.stdout

        hc_combine = root / "hc_combine.f32"
        hc_combine.write_bytes(struct.pack("<10240f", *([0.0] * 10240)))
        combine_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "combine-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-hc-attn-combine-layer", "35",
                "--debug-hc-attn-combine-f32-file", str(hc_combine),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "HC attention combine override requires a selected state trace position" not in combine_run.stdout, combine_run.stdout
        assert f"GGUF読込: {missing_model}" in combine_run.stdout, combine_run.stdout

        norm_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "norm-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-hc-norm-f32-layer", "35",
                "--debug-hc-norm-f32-file", str(hc_combine),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "HC diagnostic replay requires state trace position 36" not in norm_run.stdout, norm_run.stdout
        assert f"GGUF読込: {missing_model}" in norm_run.stdout, norm_run.stdout

        input_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "input-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-attention-input-f32-layer", "35",
                "--debug-attention-input-f32-file", str(attention_output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "attention input override requires a selected state trace position" not in input_run.stdout, input_run.stdout
        assert f"GGUF読込: {missing_model}" in input_run.stdout, input_run.stdout

        layer_input_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "layer-input-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-layer-input-residual-layer", "35",
                "--debug-layer-input-residual-f32-file", str(hc_combine),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "layer input residual override requires a selected state trace position" not in layer_input_run.stdout, layer_input_run.stdout
        assert f"GGUF読込: {missing_model}" in layer_input_run.stdout, layer_input_run.stdout

        inject = root / "hc_inject.f32"
        inject.write_bytes(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
        inject_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "inject-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-hc-inject-f32-layer", "35",
                "--debug-hc-inject-f32-file", str(inject),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "HC inject override requires a selected state trace position" not in inject_run.stdout, inject_run.stdout
        assert f"GGUF読込: {missing_model}" in inject_run.stdout, inject_run.stdout

        combined_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "combined-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-attention-output-f32-layer", "35",
                "--debug-attention-output-f32-file", str(attention_output),
                "--debug-hc-inject-f32-layer", "35",
                "--debug-hc-inject-f32-file", str(inject),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "choose one attention output/residual override at a time" not in combined_run.stdout, combined_run.stdout
        assert f"GGUF読込: {missing_model}" in combined_run.stdout, combined_run.stdout

        qsa_context = root / "qsa_context.f32"
        qsa_context.write_bytes(struct.pack("<6144f", *([0.0] * 6144)))
        context_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "context-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-qsa-context-f32-layer", "35",
                "--debug-qsa-context-f32-file", str(qsa_context),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "QSA attention context override requires a selected state trace position" not in context_run.stdout, context_run.stdout
        assert f"GGUF読込: {missing_model}" in context_run.stdout, context_run.stdout

        qsa_pregate = root / "qsa_pregate.f32"
        qsa_pregate.write_bytes(struct.pack("<6144f", *([0.0] * 6144)))
        pregate_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "pregate-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-qsa-pregate-f32-layer", "35",
                "--debug-qsa-pregate-f32-file", str(qsa_pregate),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "QSA pregate override requires a selected state trace position" not in pregate_run.stdout, pregate_run.stdout
        assert f"GGUF読込: {missing_model}" in pregate_run.stdout, pregate_run.stdout

        qsa_query = root / "qsa_query.f32"
        qsa_query.write_bytes(struct.pack("<6144f", *([0.0] * 6144)))
        query_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "query-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-qsa-query-f32-layer", "35",
                "--debug-qsa-query-f32-file", str(qsa_query),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "QSA query override requires a selected state trace position" not in query_run.stdout, query_run.stdout
        assert f"GGUF読込: {missing_model}" in query_run.stdout, query_run.stdout

        qsa_store = root / "qsa_store.f32"
        qsa_store.write_bytes(struct.pack("<512f", *([0.0] * 512)))
        for name, options in (
            ("key", ("--debug-qsa-key-store-f32-layer", "--debug-qsa-key-store-f32-file")),
            ("value", ("--debug-qsa-value-store-f32-layer", "--debug-qsa-value-store-f32-file")),
        ):
            store_run = subprocess.run(
                [
                    executable,
                    "--model", str(missing_model),
                    "--ctx", "128",
                    "--generate", "40",
                    "--state-trace", str(root / f"{name}-store-trace.bin"),
                    "--state-trace-position", "39",
                    "--forced-input-tokens", forced_tokens,
                    options[0], "35",
                    options[1], str(qsa_store),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert "QSA .* store override requires a selected state trace position" not in store_run.stdout, store_run.stdout
            assert f"GGUF読込: {missing_model}" in store_run.stdout, store_run.stdout

        qsa_cache_prefix = root / "qsa_cache_prefix.f32"
        qsa_cache_prefix.write_bytes(struct.pack("<20480f", *([0.0] * 20480)))
        for name, options in (
            ("key", ("--debug-qsa-key-cache-prefix-f32-layer", "--debug-qsa-key-cache-prefix-f32-file")),
            ("value", ("--debug-qsa-value-cache-prefix-f32-layer", "--debug-qsa-value-cache-prefix-f32-file")),
        ):
            prefix_run = subprocess.run(
                [
                    executable,
                    "--model", str(missing_model),
                    "--ctx", "128",
                    "--generate", "40",
                    "--state-trace", str(root / f"{name}-cache-prefix-trace.bin"),
                    "--state-trace-position", "39",
                    "--forced-input-tokens", forced_tokens,
                    options[0], "35",
                    options[1], str(qsa_cache_prefix),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert "QSA cache prefix override requires a selected state trace position" not in prefix_run.stdout, prefix_run.stdout
            assert f"GGUF読込: {missing_model}" in prefix_run.stdout, prefix_run.stdout

        qsa_past_value_prefix = root / "qsa_past_value_cache_prefix.f32"
        qsa_past_value_prefix.write_bytes(struct.pack("<19968f", *([0.0] * 19968)))
        past_prefix_run = subprocess.run(
            [
                executable,
                "--model", str(missing_model),
                "--ctx", "128",
                "--generate", "40",
                "--state-trace", str(root / "past-value-cache-prefix-trace.bin"),
                "--state-trace-position", "39",
                "--forced-input-tokens", forced_tokens,
                "--debug-qsa-value-cache-prefix-f32-layer", "35",
                "--debug-qsa-value-cache-prefix-f32-file", str(qsa_past_value_prefix),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "QSA cache prefix override requires a selected state trace position" not in past_prefix_run.stdout, past_prefix_run.stdout
        assert f"GGUF読込: {missing_model}" in past_prefix_run.stdout, past_prefix_run.stdout


if __name__ == "__main__":
    main()
