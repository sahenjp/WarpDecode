from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import tempfile

import modal

ROOT = pathlib.Path(__file__).resolve().parent
APP_NAME = "warpdecode-qwen38-position36-diagnostics"
MODEL_VOL = "warpdecode-qwen38-models"
MODEL_REPO = "pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF"
ARTIFACTS = ROOT / "issue1_artifacts"
CPU_TRACE = ARTIFACTS / "position36_hc_gate_cpu.bin"
CPU_GDN_TRACE_9 = ARTIFACTS / "position36_layer9_cpu_gdn_trace.bin"
CPU_GDN_TRACE_10 = ARTIFACTS / "position36_layer10_cpu_gdn_trace.bin"
CPU_GDN_HC_TRACE_10 = ARTIFACTS / "position36_layer10_cpu_hc_trace.bin"
CPU_LAYER9_FFN_TRACE = ARTIFACTS / "position36_layer9_cpu_ffn_trace.bin"
CPU_LAYER9_OUTPUT_F32 = ARTIFACTS / "position36_layer9_cpu_layer_output.f32"
CPU_GDN_STATE_9 = ARTIFACTS / "position36_layer9_cpu_state_vectors.bin"
CPU_GDN_STATE_10 = ARTIFACTS / "position36_layer10_cpu_state_vectors.bin"
CPU_GDN_PRESTATE_10_F32 = ARTIFACTS / "position36_layer10_cpu_prestate_warp_order.f32"
CPU_GDN_CONV_STATE_10_F32 = ARTIFACTS / "position36_layer10_cpu_conv_state.f32"
CPU_GDN_TRACE_14 = ARTIFACTS / "position36_layer14_cpu_gdn_trace.bin"
CPU_LAYER_OUTPUTS = ARTIFACTS / "position36_cpu_layer_outputs.bin"
LAYER10_FULL_Q8_VARIANT = "cpu-layer8-output-gdn9-attention-ffn-layer9-output-gdn10-full-q8"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOL, create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "ca-certificates", "ninja-build", "pkg-config")
    .pip_install("cmake>=3.31,<4")
    .add_local_dir(
        ROOT,
        "/workspace/warodecode",
        copy=True,
        ignore=[
            ".git", ".serena", "build*", "__pycache__", "*.pyc", "*.zip", "*.webp", "**/*.bin",
            "issue1_cpu_layer9_state.py", "qwen38_forced_cpu_trace.cpp",
            "issue1_artifacts/position36_*.json",
        ],
    )
    .add_local_file(
        CPU_TRACE,
        "/workspace/warodecode/issue1_artifacts/position36_hc_gate_cpu.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_TRACE_14,
        "/workspace/warodecode/issue1_artifacts/position36_layer14_cpu_gdn_trace.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_TRACE_9,
        "/workspace/warodecode/issue1_artifacts/position36_layer9_cpu_gdn_trace.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_TRACE_10,
        "/workspace/warodecode/issue1_artifacts/position36_layer10_cpu_gdn_trace.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_HC_TRACE_10,
        "/workspace/warodecode/issue1_artifacts/position36_layer10_cpu_hc_trace.bin",
        copy=True,
    )
    .add_local_file(
        CPU_LAYER9_FFN_TRACE,
        "/workspace/warodecode/issue1_artifacts/position36_layer9_cpu_ffn_trace.bin",
        copy=True,
    )
    .add_local_file(
        CPU_LAYER9_OUTPUT_F32,
        "/workspace/warodecode/issue1_artifacts/position36_layer9_cpu_layer_output.f32",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_STATE_9,
        "/workspace/warodecode/issue1_artifacts/position36_layer9_cpu_state_vectors.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_STATE_10,
        "/workspace/warodecode/issue1_artifacts/position36_layer10_cpu_state_vectors.bin",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_PRESTATE_10_F32,
        "/workspace/warodecode/issue1_artifacts/position36_layer10_cpu_prestate_warp_order.f32",
        copy=True,
    )
    .add_local_file(
        CPU_GDN_CONV_STATE_10_F32,
        "/workspace/warodecode/issue1_artifacts/position36_layer10_cpu_conv_state.f32",
        copy=True,
    )
    .add_local_file(
        CPU_LAYER_OUTPUTS,
        "/workspace/warodecode/issue1_artifacts/position36_cpu_layer_outputs.bin",
        copy=True,
    )
    .run_commands(
        "cmake -S /workspace/warodecode -B /workspace/warodecode/build-sm80 "
        "-G Ninja -DCMAKE_BUILD_TYPE=Release -DWARODECODE_BUILD_ENGINE=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "cmake --build /workspace/warodecode/build-sm80 -j8 --target warodecode",
    )
)


def read_trace(path: pathlib.Path) -> dict[str, list[float]]:
    raw = path.read_bytes()
    if raw[:8] != b"WDPTRC1\0":
        raise RuntimeError(f"invalid trace header: {path}")
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 1:
        raise RuntimeError(f"unsupported trace version: {version}")
    vectors = {}
    offset = 16
    for _ in range(count):
        name_size, value_count = struct.unpack_from("<IQ", raw, offset)
        offset += 12
        name = raw[offset : offset + name_size].decode()
        offset += name_size
        byte_count = value_count * 4
        vectors[name] = list(struct.unpack_from(f"<{value_count}f", raw, offset))
        offset += byte_count
    if offset != len(raw):
        raise RuntimeError("unexpected trailing trace data")
    return vectors


def compare(reference: list[float], candidate: list[float]) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise RuntimeError(f"trace vector sizes differ: {len(reference)} != {len(candidate)}")
    errors = [b - a for a, b in zip(reference, candidate)]
    rms = (sum(a * a for a in reference) / len(reference)) ** 0.5
    return {
        "count": len(reference),
        "rmse": (sum(e * e for e in errors) / len(errors)) ** 0.5,
        "nrmse": ((sum(e * e for e in errors) / len(errors)) ** 0.5) / rms,
        "max_abs": max(abs(e) for e in errors),
    }


def transpose_gdn_state(values: list[float]) -> list[float]:
    heads, width = 48, 128
    if len(values) != heads * width * width:
        raise RuntimeError("unexpected GDN recurrent state size")
    out = [0.0] * len(values)
    for head in range(heads):
        base = head * width * width
        for row in range(width):
            for col in range(width):
                out[base + row * width + col] = values[base + col * width + row]
    return out


@app.function(
    image=image,
    volumes={"/models": models.with_mount_options(read_only=True)},
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=1800,
)
def replay(variant: str) -> dict:
    if variant not in {
        "gdn-output", "after-attention-residual", "cpu-hc-ffn-mix",
        "cpu-ffn-output", "cpu-layer8-moe-output", "cpu-layer8-output",
        "cpu-final-residual", "layer47-qsa-q8", "layer47-qsa-output",
        "gdn-layer14-output", "gdn-layer14-trace", "gdn-layer14-core",
        "gdn-layer14-norm-projection", "gdn-layer9-trace",
        "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state",
        "cpu-layer8-output-gdn9-histories", "cpu-layer8-output-gdn9-attention",
        "cpu-layer8-output-gdn9-attention-ffn",
        "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
        "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
        LAYER10_FULL_Q8_VARIANT,
    }:
        raise ValueError(f"unknown replay variant: {variant}")
    root = pathlib.Path("/workspace/warodecode")
    model_root = pathlib.Path("/models") / MODEL_REPO.replace("/", "__")
    candidates = [p for p in model_root.rglob("*.gguf") if not p.name.lower().startswith("mtp-")]
    if not candidates:
        raise RuntimeError(f"no model GGUF found under {model_root}")
    model = max(candidates, key=lambda p: p.stat().st_size)
    identity = json.loads((root / "issue1_artifacts/model_identity.json").read_text())
    if model.name != identity["filename"] or model.stat().st_size != identity["size"]:
        raise RuntimeError("mounted GGUF path or size does not match the canonical model identity")
    input_record = json.loads((root / "issue1_artifacts/warodecode-state-position-36-gdn.json").read_text())
    forced = input_record["state_trace"]["forced_input_token_ids"]
    if len(forced) != 37 or input_record["last_token"] != 8206:
        raise RuntimeError("saved position-36 Warp input does not match the baseline")
    reference = json.loads((root / "issue1_artifacts/reference_tokens_64.json").read_text())["token_ids"]

    with tempfile.TemporaryDirectory(prefix="layer8-gdn-") as tmp:
        trace_path = pathlib.Path(tmp) / "warp.bin"
        attention_layer = (
            47 if variant in {"layer47-qsa-q8", "layer47-qsa-output"}
            else 9 if variant in {
                "gdn-layer9-trace", "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state",
                "cpu-layer8-output-gdn9-histories", "cpu-layer8-output-gdn9-attention",
                "cpu-layer8-output-gdn9-attention-ffn",
                "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
                "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
                LAYER10_FULL_Q8_VARIANT,
            }
            else 14 if variant in {
                "gdn-layer14-output", "gdn-layer14-trace", "gdn-layer14-core",
                "gdn-layer14-norm-projection",
            }
            else 8
        )
        if variant == "layer47-qsa-q8":
            override_args = ["--debug-qsa-q8_0-layer", "47"]
        elif variant == "gdn-layer14-trace":
            override_args = []
        elif variant == "gdn-layer9-trace":
            override_args = []
        elif variant == "gdn-layer14-core":
            override_args = [
                "--debug-layer14-gdn-core-f32-file",
                str(root / "issue1_artifacts/position36_layer14_cpu_gdn_core_output.f32"),
            ]
        elif variant == "gdn-layer14-norm-projection":
            override_args = [
                "--debug-layer14-gdn-norm-output-f32-file",
                str(root / "issue1_artifacts/position36_layer14_cpu_gdn_norm_output.f32"),
            ]
        elif variant in {"gdn-output", "layer47-qsa-output", "gdn-layer14-output"}:
            output_file = {
                8: "position36_layer8_cpu_attention_output.f32",
                9: "position36_layer9_cpu_gdn_output.f32",
                14: "position36_layer14_cpu_gdn_output.f32",
                47: "position36_layer47_cpu_qsa_output.f32",
            }[attention_layer]
            override_args = [
                "--debug-attention-output-f32-layer", str(attention_layer),
                "--debug-attention-output-f32-file", str(root / "issue1_artifacts" / output_file),
            ]
        elif variant == "cpu-final-residual":
            override_args = ["--debug-final-residual-f32-file",
                             str(root / "issue1_artifacts/position36_cpu_final_residual.f32")]
        elif variant in {"cpu-layer8-output", "cpu-layer8-output-gdn9"}:
            override_args = ["--debug-layer8-output-f32-file",
                             str(root / "issue1_artifacts/position36_layer8_cpu_layer_output.f32")]
        elif variant == "cpu-layer8-output-gdn9-state":
            override_args = [
                "--debug-layer8-output-f32-file",
                str(root / "issue1_artifacts/position36_layer8_cpu_layer_output.f32"),
                "--debug-layer9-gdn-state-f32-file",
                str(root / "issue1_artifacts/position36_layer9_cpu_prestate_warp_order.f32"),
            ]
        elif variant == "cpu-layer8-output-gdn9-histories":
            override_args = [
                "--debug-layer8-output-f32-file",
                str(root / "issue1_artifacts/position36_layer8_cpu_layer_output.f32"),
                "--debug-layer9-gdn-state-f32-file",
                str(root / "issue1_artifacts/position36_layer9_cpu_prestate_warp_order.f32"),
                "--debug-layer9-gdn-conv-state-f32-file",
                str(root / "issue1_artifacts/position36_layer9_cpu_conv_state.f32"),
            ]
        elif variant == "cpu-layer8-output-gdn9-attention":
            override_args = [
                "--debug-layer8-output-f32-file",
                str(root / "issue1_artifacts/position36_layer8_cpu_layer_output.f32"),
                "--debug-attention-output-f32-layer", "9",
                "--debug-attention-output-f32-file",
                str(root / "issue1_artifacts/position36_layer9_cpu_gdn_output.f32"),
            ]
        elif variant in {
            "cpu-layer8-output-gdn9-attention-ffn",
            "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
            "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
            LAYER10_FULL_Q8_VARIANT,
        }:
            cpu_layer9_ffn = read_trace(
                root / "issue1_artifacts/position36_layer9_cpu_ffn_trace.bin")
            ffn_output = cpu_layer9_ffn["pos36.layer9.ffn_output"]
            override_args = [
                "--debug-layer8-output-f32-file",
                str(root / "issue1_artifacts/position36_layer8_cpu_layer_output.f32"),
                "--debug-attention-output-f32-layer", "9",
                "--debug-attention-output-f32-file",
                str(root / "issue1_artifacts/position36_layer9_cpu_gdn_output.f32"),
                "--debug-ffn-output-layer", "9",
                "--debug-ffn-output", ",".join(f"{value:.9g}" for value in ffn_output),
            ]
            if variant == "cpu-layer8-output-gdn9-attention-ffn-gdn10-state":
                override_args.extend([
                    "--debug-layer9-gdn-state-layer", "10",
                    "--debug-layer9-gdn-state-f32-file",
                    str(root / "issue1_artifacts/position36_layer10_cpu_prestate_warp_order.f32"),
                    "--debug-layer9-gdn-conv-state-layer", "10",
                    "--debug-layer9-gdn-conv-state-f32-file",
                    str(root / "issue1_artifacts/position36_layer10_cpu_conv_state.f32"),
                ])
            if variant in {
                "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
                LAYER10_FULL_Q8_VARIANT,
            }:
                override_args.extend([
                    "--debug-layer8-output-layer", "9",
                    "--debug-layer8-output-f32-file",
                    str(root / "issue1_artifacts/position36_layer9_cpu_layer_output.f32"),
                ])
            if variant == LAYER10_FULL_Q8_VARIANT:
                cpu_hc_10 = read_trace(
                    root / "issue1_artifacts/position36_layer10_cpu_hc_trace.bin")
                cpu_norm = cpu_hc_10["pos36.layer10.hc_attn_norm"]
                norm_path = pathlib.Path(tmp) / "position36_layer10_cpu_hc_attn_norm.f32"
                norm_path.write_bytes(struct.pack(f"<{len(cpu_norm)}f", *cpu_norm))
                override_args.extend([
                    "--debug-hc-attn-q8_0-full-layer", "10",
                    "--debug-hc-norm-f32-layer", "10",
                    "--debug-hc-norm-f32-file", str(norm_path),
                ])
        elif variant in {"after-attention-residual", "cpu-hc-ffn-mix", "cpu-layer8-moe-output"}:
            override_args = ["--debug-layer8-after-attn-residual-f32-file",
                             str(root / "issue1_artifacts/position36_layer8_cpu_after_attention_residual.f32")]
        else:
            ffn_output = struct.unpack(
                "<2560f", (root / "issue1_artifacts/position36_layer8_cpu_ffn_output.f32").read_bytes())
            override_args = ["--debug-ffn-output", ",".join(f"{value:.9g}" for value in ffn_output)]
        if variant in {"cpu-hc-ffn-mix", "cpu-layer8-moe-output"}:
            override_args.extend([
                "--debug-layer8-hc-ffn-mix-f32-file",
                str(root / "issue1_artifacts/position36_layer8_cpu_ffn_mix.f32"),
            ])
        if variant == "cpu-layer8-moe-output":
            ffn_output = struct.unpack(
                "<2560f", (root / "issue1_artifacts/position36_layer8_cpu_ffn_output.f32").read_bytes())
            override_args.extend(["--debug-ffn-output", ",".join(f"{value:.9g}" for value in ffn_output)])
        cmd = [
            "/workspace/warodecode/build-sm80/warodecode",
            "--model", str(model), "--ctx", "128", "--device", "0",
            "--start-token", "1", "--generate", "37", "--no-graph",
            "--state-trace", str(trace_path), "--forced-input-tokens",
            ",".join(map(str, forced)), "--state-trace-position", "36",
            *override_args,
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout, flush=True)
        if proc.returncode:
            raise RuntimeError(f"warodecode exited with status {proc.returncode}")
        record = next(json.loads(line) for line in reversed(proc.stdout.splitlines())
                      if line.startswith('{"engine":"warodecode-qwen38"'))
        candidate = read_trace(trace_path)

    cpu = read_trace(root / "issue1_artifacts/position36_hc_gate_cpu.bin")
    cpu_hc_ffn_mix = list(struct.unpack(
        "<2560f", (root / "issue1_artifacts/position36_layer8_cpu_ffn_mix.f32").read_bytes()))
    cpu_ffn_output = list(struct.unpack(
        "<2560f", (root / "issue1_artifacts/position36_layer8_cpu_ffn_output.f32").read_bytes()))
    cpu_final_residual = list(struct.unpack(
        "<10240f", (root / "issue1_artifacts/position36_cpu_final_residual.f32").read_bytes()))
    if attention_layer == 8:
        cpu_output = cpu["pos36.layer8.attention_output"]
    else:
        output_file = (
            {9: "position36_layer9_cpu_gdn_output.f32",
             14: "position36_layer14_cpu_gdn_output.f32",
             47: "position36_layer47_cpu_qsa_output.f32"}[attention_layer]
        )
        cpu_output = list(struct.unpack(
            "<2560f", (root / "issue1_artifacts" / output_file).read_bytes()))
    cpu_router = cpu["pos36.layer8.ffn.router_logits"]
    warp_router = candidate["pos36.layer8.ffn.router_logits"]
    cpu_moe = json.loads((root / "issue1_artifacts/llama-state-position-36-moe.json").read_text())
    cpu_expert_ids = next(
        item["values"] for item in cpu_moe["integer_vectors"]
        if item["name"] == "pos36.ffn_moe_topk-8")
    warp_expert_ids = next(
        item["values"] for item in record["state_trace"]["integer_vectors"]
        if item["name"] == "pos36.layer8.ffn.expert_ids")
    cpu_margin = cpu_router[335] - cpu_router[204]
    warp_margin = warp_router[335] - warp_router[204]
    gdn_layer_metrics = None
    gdn_layer = 9 if variant in {
        "gdn-layer9-trace", "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state",
        "cpu-layer8-output-gdn9-histories", "cpu-layer8-output-gdn9-attention",
        "cpu-layer8-output-gdn9-attention-ffn",
        "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
        "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
        LAYER10_FULL_Q8_VARIANT,
    } else 14
    if variant in {
        "gdn-layer9-trace", "gdn-layer14-trace", "gdn-layer14-core", "gdn-layer14-norm-projection",
        "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state", "cpu-layer8-output-gdn9-histories",
        "cpu-layer8-output-gdn9-attention", "cpu-layer8-output-gdn9-attention-ffn",
        "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
        "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
        LAYER10_FULL_Q8_VARIANT,
    }:
        cpu_gdn = read_trace(root / f"issue1_artifacts/position36_layer{gdn_layer}_cpu_gdn_trace.bin")
        warp_gdn = f"pos36.layer{gdn_layer}.gdn."
        gdn_pairs = {
            "qkv_projection": (f"pos36.linear_attn_qkv_mixed-{gdn_layer}", warp_gdn + "qkv_full"),
            "recurrent_core_output": (f"pos36.gdn_core_output-{gdn_layer}", warp_gdn + "core_output_full"),
            "gated_norm_output": (f"pos36.final_output-{gdn_layer}", warp_gdn + "norm_output_full"),
            "projected_attention_output": (
                f"pos36.linear_attn_out-{gdn_layer}", f"pos36.layer{gdn_layer}.attention_output"),
        }
        if f"pos36.layer{gdn_layer}.hc_attn_mix" in cpu_gdn:
            gdn_pairs["gdn_input"] = (
                f"pos36.layer{gdn_layer}.hc_attn_mix", f"pos36.layer{gdn_layer}.gdn.input")
        gdn_layer_metrics = {
            label: compare(cpu_gdn[cpu_name], candidate[warp_name])
            for label, (cpu_name, warp_name) in gdn_pairs.items()
        }
        if variant in {
            "cpu-layer8-output-gdn9-attention",
            "cpu-layer8-output-gdn9-attention-ffn",
            "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
            "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
            LAYER10_FULL_Q8_VARIANT,
        }:
            cpu_layer9_ffn = read_trace(
                root / "issue1_artifacts/position36_layer9_cpu_ffn_trace.bin")
            gdn_layer_metrics.update({
                "after_attention_residual": compare(
                    cpu_layer9_ffn["pos36.layer9.after_attention_residual"],
                    candidate["pos36.layer9.after_attention_residual"]),
                "hc_ffn_mix": compare(
                    cpu_layer9_ffn["pos36.layer9.hc_ffn_mix"],
                    candidate["pos36.layer9.hc_ffn_mix"]),
                "router_logits": compare(
                    cpu_layer9_ffn["pos36.layer9.ffn.router_logits"],
                    candidate["pos36.layer9.ffn.router_logits"]),
                "ffn_output": compare(
                    cpu_layer9_ffn["pos36.layer9.ffn_output"],
                    candidate["pos36.layer9.ffn_output"]),
                "layer_output": compare(
                    cpu_layer9_ffn["pos36.layer9.output"],
                    candidate["pos36.layer9.output"]),
            })
            if variant in {
                "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
                LAYER10_FULL_Q8_VARIANT,
            }:
                gdn_layer_metrics["layer_output_cpu_vs_half_override"] = compare(
                    cpu_layer9_ffn["pos36.layer9.output"],
                    candidate["pos36.layer9.output_after_cpu_override"])
        if variant in {
            "cpu-layer8-output-gdn9-attention-ffn",
            "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
            "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
            LAYER10_FULL_Q8_VARIANT,
        }:
            cpu_gdn_10 = read_trace(
                root / "issue1_artifacts/position36_layer10_cpu_gdn_trace.bin")
            gdn_layer10_pairs = {
                "gdn_input": ("pos36.layer10.gdn.input", "pos36.layer10.gdn.input"),
                "qkv_projection": (
                    "pos36.layer10.gdn.qkv_projection", "pos36.layer10.gdn.qkv_full"),
                "alpha": ("pos36.layer10.gdn.alpha", "pos36.layer10.gdn.alpha_full"),
                "beta": ("pos36.layer10.gdn.beta", "pos36.layer10.gdn.beta_full"),
                "gate": ("pos36.layer10.gdn.gate", "pos36.layer10.gdn.gate_full"),
                "z": ("pos36.layer10.gdn.z", "pos36.layer10.gdn.z_full"),
                "recurrent_core_output": (
                    "pos36.layer10.gdn.core_output", "pos36.layer10.gdn.core_output_full"),
                "gated_norm_output": (
                    "pos36.layer10.gdn.norm_output", "pos36.layer10.gdn.norm_output_full"),
                "projected_attention_output": (
                    "pos36.layer10.attention_output", "pos36.layer10.attention_output"),
                "after_attention_residual": (
                    "pos36.layer10.after_attention_residual",
                    "pos36.layer10.after_attention_residual"),
                "hc_ffn_mix": ("pos36.layer10.hc_ffn_mix", "pos36.layer10.hc_ffn_mix"),
                "router_logits": (
                    "pos36.layer10.ffn.router_logits", "pos36.layer10.ffn.router_logits"),
                "ffn_output": ("pos36.layer10.ffn_output", "pos36.layer10.ffn_output"),
                "layer_output": ("pos36.layer10.output", "pos36.layer10.output"),
            }
            gdn_layer_metrics["next_layer10_stage_errors"] = {
                label: compare(cpu_gdn_10[cpu_name], candidate[warp_name])
                for label, (cpu_name, warp_name) in gdn_layer10_pairs.items()
            }
            if variant in {
                "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
                LAYER10_FULL_Q8_VARIANT,
            }:
                cpu_hc_10 = read_trace(
                    root / "issue1_artifacts/position36_layer10_cpu_hc_trace.bin")
                layer10_hc_pairs = {
                    "input_residual": (
                        "pos36.layer10.input_residual", "pos36.layer10.input_residual"),
                    "attention_norm": (
                        "pos36.layer10.hc_attn_norm", "pos36.layer10.hc_attn_norm"),
                    "attention_gate": (
                        "pos36.layer10.hc_attn_gate", "pos36.layer10.hc_attn_gate"),
                    "attention_mix": (
                        "pos36.layer10.hc_attn_mix", "pos36.layer10.hc_attn_mix"),
                    "attention_inject": (
                        "pos36.layer10.hc_attn_inject", "pos36.layer10.hc_attn_inject"),
                    "after_attention_residual": (
                        "pos36.layer10.after_attention_residual",
                        "pos36.layer10.after_attention_residual"),
                    "ffn_norm": (
                        "pos36.layer10.hc_ffn_norm", "pos36.layer10.hc_ffn_norm"),
                    "ffn_gate": (
                        "pos36.layer10.hc_ffn_gate", "pos36.layer10.hc_ffn_gate"),
                    "ffn_mix": (
                        "pos36.layer10.hc_ffn_mix", "pos36.layer10.hc_ffn_mix"),
                }
                gdn_layer_metrics["next_layer10_hc_stage_errors"] = {
                    label: compare(cpu_hc_10[cpu_name], candidate[warp_name])
                    for label, (cpu_name, warp_name) in layer10_hc_pairs.items()
                }
            cpu_gdn_state_10 = read_trace(
                root / "issue1_artifacts/position36_layer10_cpu_state_vectors.bin")
            gdn_layer_metrics["next_layer10_state_errors"] = {
                "state_predelta_warp_order_vs_recurrent_before": compare(
                    transpose_gdn_state(cpu_gdn_state_10["pos36.state_predelta-10"]),
                    candidate["pos36.layer10.gdn.recurrent_before_full"]),
                "new_state_warp_order_vs_recurrent_after": compare(
                    transpose_gdn_state(cpu_gdn_state_10["pos36.new_state-10"]),
                    candidate["pos36.layer10.gdn.recurrent_after_full"]),
                "conv_state_vs_warp_before": compare(
                    cpu_gdn_state_10["pos36.conv_state_at-10"],
                    candidate["pos36.layer10.gdn.conv_before"]),
            }
            if variant == "cpu-layer8-output-gdn9-attention-ffn-gdn10-state":
                gdn_layer_metrics["next_layer10_state_injection_errors"] = {
                    "cpu_prestate_vs_injected": compare(
                        transpose_gdn_state(cpu_gdn_state_10["pos36.state_predelta-10"]),
                        candidate["pos36.layer10.gdn.recurrent_before_cpu_override"]),
                    "cpu_conv_state_vs_half_override": compare(
                        cpu_gdn_state_10["pos36.conv_state_at-10"],
                        candidate["pos36.layer10.gdn.conv_state_after_cpu_override"]),
                }
        if variant == "gdn-layer14-core":
            gdn_layer_metrics.update({
                "core_output_cpu_vs_warp_before_override": compare(
                    cpu_gdn["pos36.gdn_core_output-14"],
                    candidate["pos36.layer14.gdn.core_output_before_cpu_override"]),
                "gated_norm_cpu_vs_warp_before_override": compare(
                    cpu_gdn["pos36.final_output-14"],
                    candidate["pos36.layer14.gdn.norm_output_before_cpu_core_override"]),
                "projected_output_cpu_vs_warp_before_override": compare(
                    cpu_gdn["pos36.linear_attn_out-14"],
                    candidate["pos36.layer14.attention_output_before_cpu_core_override"]),
            })
        if variant == "gdn-layer14-norm-projection":
            gdn_layer_metrics.update({
                "gated_norm_cpu_vs_warp_before_override": compare(
                    cpu_gdn["pos36.final_output-14"],
                    candidate["pos36.layer14.gdn.norm_output_before_cpu_override"]),
                "gated_norm_cpu_vs_half_override": compare(
                    cpu_gdn["pos36.final_output-14"],
                    candidate["pos36.layer14.gdn.norm_output_after_cpu_override"]),
                "projected_output_cpu_vs_warp_before_override": compare(
                    cpu_gdn["pos36.linear_attn_out-14"],
                    candidate["pos36.layer14.attention_output_before_cpu_norm_override"]),
            })
    layer_output_drift_after_layer8 = None
    if variant in {
        "cpu-layer8-moe-output", "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state",
        "cpu-layer8-output-gdn9-histories", "cpu-layer8-output-gdn9-attention",
        "cpu-layer8-output-gdn9-attention-ffn",
        "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
        "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
        LAYER10_FULL_Q8_VARIANT,
    }:
        cpu_layer_outputs = read_trace(root / "issue1_artifacts/position36_cpu_layer_outputs.bin")
        layer_metrics = [
            {
                "layer": layer,
                **compare(cpu_layer_outputs[f"pos36.l_last-{layer}"],
                          candidate[f"pos36.layer{layer}.output"]),
            }
            for layer in range(8, 48)
        ]
        layer_steps = [
            {"from": left["layer"], "to": right["layer"],
             "delta_nrmse": right["nrmse"] - left["nrmse"]}
            for left, right in zip(layer_metrics, layer_metrics[1:])
        ]
        layer_output_drift_after_layer8 = {
            "nrmse_by_layer": {str(item["layer"]): item["nrmse"] for item in layer_metrics},
            "largest_positive_step": max(layer_steps, key=lambda item: item["delta_nrmse"]),
        }
    gdn_layer9_state_errors = None
    if variant in {
        "gdn-layer9-trace", "cpu-layer8-output-gdn9", "cpu-layer8-output-gdn9-state",
        "cpu-layer8-output-gdn9-histories", "cpu-layer8-output-gdn9-attention",
        "cpu-layer8-output-gdn9-attention-ffn",
        "cpu-layer8-output-gdn9-attention-ffn-gdn10-state",
        "cpu-layer8-output-gdn9-attention-ffn-layer9-output",
        LAYER10_FULL_Q8_VARIANT,
    }:
        cpu_gdn_state = read_trace(root / "issue1_artifacts/position36_layer9_cpu_state_vectors.bin")
        cpu_predelta = cpu_gdn_state["pos36.state_predelta-9"]
        cpu_new_state = cpu_gdn_state["pos36.new_state-9"]
        cpu_predelta_transposed = transpose_gdn_state(cpu_predelta)
        cpu_new_state_transposed = transpose_gdn_state(cpu_new_state)
        state_pairs = {
            "state_predelta_warp_order_vs_recurrent_before": (
                cpu_predelta_transposed, candidate["pos36.layer9.gdn.recurrent_before_full"]),
            "new_state_warp_order_vs_recurrent_after": (
                cpu_new_state_transposed, candidate["pos36.layer9.gdn.recurrent_after_full"]),
            "conv_state_vs_warp_before": (
                cpu_gdn_state["pos36.conv_state_at-9"], candidate["pos36.layer9.gdn.conv_before"]),
        }
        gdn_layer9_state_errors = {
            label: compare(cpu_values, warp_values)
            for label, (cpu_values, warp_values) in state_pairs.items()
        }
        if variant == "cpu-layer8-output-gdn9-histories":
            gdn_layer9_state_errors.update({
                "cpu_prestate_vs_override": compare(
                    cpu_predelta_transposed,
                    candidate["pos36.layer9.gdn.recurrent_before_cpu_override"]),
                "cpu_conv_state_vs_half_override": compare(
                    cpu_gdn_state["pos36.conv_state_at-9"],
                    candidate["pos36.layer9.gdn.conv_state_after_cpu_override"]),
            })
    return {
        "variant": variant,
        "attention_layer": attention_layer,
        "model": str(model),
        "token36": record["token_ids"][-1],
        "first_mismatch_index": next((i for i, (a, b) in enumerate(zip(reference, record["token_ids"])) if a != b), None),
        "attention_output_cpu_vs_warp_before_override": (
            compare(cpu_output, candidate[f"pos36.layer{attention_layer}.attention_output_before_cpu_override"])
            if f"pos36.layer{attention_layer}.attention_output_before_cpu_override" in candidate else None),
        "attention_output_cpu_vs_candidate": compare(
            cpu_output, candidate[f"pos36.layer{attention_layer}.attention_output"]),
        "after_attention_residual_warp_before_override": (
            compare(cpu["pos36.layer8.after_attention_residual"],
                    candidate["pos36.layer8.after_attention_residual_before_cpu_override"])
            if "pos36.layer8.after_attention_residual_before_cpu_override" in candidate else None),
        "after_attention_residual_cpu_vs_candidate": compare(
            cpu["pos36.layer8.after_attention_residual"],
            candidate["pos36.layer8.after_attention_residual"]),
        "hc_ffn_mix_cpu_vs_warp_before_override": (
            compare(cpu_hc_ffn_mix, candidate["pos36.layer8.hc_ffn_mix_before_cpu_override"])
            if "pos36.layer8.hc_ffn_mix_before_cpu_override" in candidate
            else compare(cpu_hc_ffn_mix, candidate["pos36.layer8.hc_ffn_mix"])),
        "hc_ffn_mix_cpu_vs_candidate": compare(
            cpu_hc_ffn_mix, candidate["pos36.layer8.hc_ffn_mix"]),
        "ffn_output_cpu_vs_candidate": compare(
            cpu_ffn_output, candidate["pos36.layer8.ffn_output"]),
        "router_logits_cpu_vs_candidate": compare(
            cpu_router, warp_router),
        "layer8_output_cpu_vs_candidate": compare(
            cpu["pos36.layer8.layer_output"], candidate["pos36.layer8.output"]),
        "gdn_layer": gdn_layer if gdn_layer_metrics is not None else None,
        "gdn_layer_stage_errors": gdn_layer_metrics,
        "gdn_layer9_state_errors": gdn_layer9_state_errors,
        "layer_output_drift_after_cpu_layer8": layer_output_drift_after_layer8,
        "final_residual_cpu_vs_warp_before_override": (
            compare(cpu_final_residual, candidate["pos36.final.residual_before_cpu_override"])
            if "pos36.final.residual_before_cpu_override" in candidate else None),
        "final_residual_cpu_vs_candidate": (
            compare(cpu_final_residual, candidate["pos36.final.residual_after_cpu_override"])
            if "pos36.final.residual_after_cpu_override" in candidate
            else compare(cpu_final_residual, candidate["pos36.layer47.output"])),
        "router_selected_experts": {
            "cpu": cpu_expert_ids,
            "candidate": warp_expert_ids,
            "exact_match": cpu_expert_ids == warp_expert_ids,
        },
        "expert335_minus_expert204": {
            "cpu": cpu_margin,
            "warp_before": -0.011368274688720703,
            "warp_after": warp_margin,
            "distance_to_cpu": abs(warp_margin - cpu_margin),
        },
        "saved_reference_modified": False,
        "production_fix_applied": False,
    }


@app.local_entrypoint()
def main(variant: str = "gdn-output") -> None:
    result = replay.remote(variant)
    report_name = (
        "position36_layer8_gdn_output_override.json"
        if variant == "gdn-output"
        else "position36_layer8_after_attention_residual_override.json"
        if variant == "after-attention-residual"
        else "position36_layer8_hc_ffn_mix_override.json"
        if variant == "cpu-hc-ffn-mix"
        else "position36_layer8_moe_output_override.json"
        if variant in {"cpu-ffn-output", "cpu-layer8-moe-output"}
        else "position36_layer8_output_override.json"
        if variant == "cpu-layer8-output"
        else "position36_final_residual_override.json"
        if variant == "cpu-final-residual"
        else "position36_layer47_qsa_q8_override.json"
        if variant == "layer47-qsa-q8"
        else "position36_layer47_qsa_output_override.json"
        if variant == "layer47-qsa-output"
        else "position36_layer14_gdn_output_override.json"
        if variant == "gdn-layer14-output"
        else "position36_layer14_gdn_trace.json"
        if variant == "gdn-layer14-trace"
        else "position36_layer14_gdn_core_override.json"
        if variant == "gdn-layer14-core"
        else "position36_layer14_gdn_norm_projection.json"
        if variant == "gdn-layer14-norm-projection"
        else "position36_layer9_gdn_trace.json"
        if variant == "gdn-layer9-trace"
        else "position36_layer8_output_gdn9.json"
        if variant == "cpu-layer8-output-gdn9"
        else "position36_layer8_output_gdn9_state.json"
        if variant == "cpu-layer8-output-gdn9-state"
        else "position36_layer8_output_gdn9_histories.json"
        if variant == "cpu-layer8-output-gdn9-histories"
        else "position36_layer8_output_gdn9_attention.json"
        if variant == "cpu-layer8-output-gdn9-attention"
        else "position36_layer8_output_gdn9_attention_ffn.json"
        if variant == "cpu-layer8-output-gdn9-attention-ffn"
        else "position36_layer8_output_gdn9_attention_ffn_gdn10_state.json"
        if variant == "cpu-layer8-output-gdn9-attention-ffn-gdn10-state"
        else "position36_layer8_output_gdn9_attention_ffn_layer9_output_gdn10_full_q8.json"
        if variant == LAYER10_FULL_Q8_VARIANT
        else "position36_layer8_output_gdn9_attention_ffn_layer9_output.json"
        if variant == "cpu-layer8-output-gdn9-attention-ffn-layer9-output"
        else "position36_layer8_ffn_output_override.json"
    )
    (ARTIFACTS / report_name).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
