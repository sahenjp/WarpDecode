#!/usr/bin/env python3
"""Offline plan by default; --run-a100 executes one allocation / four processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = '2bb72a88252610cf9ec75b14aefe6dfda1e99d9639ff625c9e967cfeab253b60'
VARIANTS = [('baseline', 0, 0), ('residual_only', 1, 0),
            ('moe_output_only', 0, 1), ('combined', 1, 1)]


def request(root: Path) -> dict:
    raw = (root / 'issue1_artifacts/reference_tokens_64.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError('canonical reference SHA-256 mismatch')
    tokens = json.loads(raw)['token_ids']
    if len(tokens) != 64 or tokens[39] != 264:
        raise RuntimeError('unexpected canonical reference')
    return {'position': 39, 'expected39': tokens[39],
            'forced_input_tokens': [1] + tokens[:39],
            'reference_sha256': digest, 'sampling': 'greedy (no stochastic seed)',
            'ctx': 128, 'cuda_graph': False,
            'variants': [{'variant': name, 'WARODECODE_RESIDUAL_F32': r,
                          'WARODECODE_MOE_F32': m} for name, r, m in VARIANTS]}


def selected_trace(path: Path) -> dict[str, list[float]]:
    wanted = {'pos39.final.logits', 'pos39.layer8.ffn.router_logits'}
    result = {}
    with path.open('rb') as stream:
        if stream.read(8) != b'WDPTRC1\0':
            raise RuntimeError('invalid state trace magic')
        version, count = struct.unpack('<II', stream.read(8))
        if version != 1:
            raise RuntimeError('unsupported state trace version')
        for _ in range(count):
            name_size, values = struct.unpack('<IQ', stream.read(12))
            name = stream.read(name_size).decode('utf-8')
            if name in wanted:
                if name in result:
                    raise RuntimeError(f'duplicate trace vector: {name}')
                result[name] = list(struct.unpack(f'<{values}f', stream.read(values * 4)))
            else:
                stream.seek(values * 4, 1)
    if not wanted.issubset(result):
        raise RuntimeError(f'missing trace vectors: {wanted - result.keys()}')
    return result


def summarize(record: dict, vectors: dict, name: str, r: int, m: int,
              spec: dict, wall_ms: float) -> dict:
    state = record['state_trace']
    if state['trace_position'] != 39 or state['forced_input_token_ids'] != spec['forced_input_tokens']:
        raise RuntimeError('position39 forced prefix was not used')
    if len(record['token_ids']) != 40:
        raise RuntimeError('unexpected generated token count')
    if any(state['debug_injection_counts'].values()):
        raise RuntimeError('unexpected diagnostic injection')
    logits = vectors['pos39.final.logits']
    if len(logits) <= 449 or not all(math.isfinite(x) for x in logits):
        raise RuntimeError('invalid position39 logits')
    router = vectors['pos39.layer8.ffn.router_logits']
    ids = next(item['values'] for item in state['integer_vectors']
               if item['name'] == 'pos39.layer8.ffn.expert_ids')
    if len(ids) != 10:
        raise RuntimeError('unexpected layer8 top-k count')
    return {'variant': name, 'residual_f32': r, 'moe_f32': m,
            'token39': record['token_ids'][39], 'expected39': spec['expected39'],
            'logit264': logits[264], 'logit449': logits[449],
            'margin264_449': logits[264] - logits[449],
            'runtime_ms': record['seconds'] * 1000, 'process_wall_ms': wall_ms,
            'reference_sha256': spec['reference_sha256'],
            'layer8_topk_order': ids, 'layer8_topk_set': sorted(ids),
            'layer8_selected_router_logits': [router[i] for i in ids]}


def run_a100() -> dict:
    import modal

    request(ROOT)  # Fail locally before requesting resources if the reference changed.
    app = modal.App('warpdecode-issue1-position39-four-way')
    models = modal.Volume.from_name('warpdecode-qwen38-models', create_if_missing=False)
    image = (modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')
             .apt_install('build-essential', 'ninja-build')
             .pip_install('cmake>=3.31,<4')
             .add_local_dir(ROOT / 'src', '/workspace/warodecode/src', copy=True)
             .add_local_dir(ROOT / 'tests', '/workspace/warodecode/tests', copy=True,
                            ignore=['__pycache__', '*.pyc', '*fixture*'])
             .add_local_file(ROOT / 'CMakeLists.txt', '/workspace/warodecode/CMakeLists.txt', copy=True)
             .add_local_file(ROOT / 'tools/warodecode_tui.cpp', '/workspace/warodecode/tools/warodecode_tui.cpp', copy=True)
             .add_local_file(ROOT / 'tools/qwen38_layer0_probe.cu', '/workspace/warodecode/tools/qwen38_layer0_probe.cu', copy=True)
             .add_local_file(Path(__file__), '/workspace/warodecode/tools/issue1_position39_ablation.py', copy=True))
    for filename in ('reference_tokens_64.json', 'model_identity.json'):
        image = image.add_local_file(ROOT / 'issue1_artifacts' / filename,
                                    '/workspace/warodecode/issue1_artifacts/' + filename, copy=True)
    image = image.run_commands(
        'cmake -S /workspace/warodecode -B /workspace/warodecode/build-sm80 -G Ninja '
        '-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80',
        'cmake --build /workspace/warodecode/build-sm80 -j8 --target '
        'warodecode warodecode-issue1-moe-precision-cuda-test')

    @app.function(image=image, gpu='A100-80GB', cpu=8, memory=32768, timeout=7200,
                  volumes={'/models': models.with_mount_options(read_only=True)})
    def ablate() -> dict:
        root = Path('/workspace/warodecode')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        if 'A100' not in gpu:
            raise RuntimeError(f'A100 required; found {gpu}')
        spec = request(root)
        identity = json.loads((root / 'issue1_artifacts/model_identity.json').read_text())
        model = Path(identity['modal_volume']['path'])
        if not model.is_file() or model.name != identity['filename'] or model.stat().st_size != identity['size']:
            raise RuntimeError('existing mounted model does not match canonical identity; no download allowed')
        executable = root / 'build-sm80/warodecode'
        build_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
        unit = subprocess.run([str(root / 'build-sm80/warodecode-issue1-moe-precision-cuda-test')],
                              text=True, capture_output=True, check=True)
        print(unit.stdout, flush=True)
        rows = []
        for name, residual, moe in VARIANTS:
            with tempfile.TemporaryDirectory(prefix='issue1-' + name + '-') as tmp:
                trace = Path(tmp) / 'position39.bin'
                command = [str(executable), '--model', str(model), '--ctx', '128',
                           '--device', '0', '--start-token', '1', '--generate', '40',
                           '--no-graph', '--no-autotune', '--state-trace', str(trace),
                           '--state-trace-position', '39', '--forced-input-tokens',
                           ','.join(map(str, spec['forced_input_tokens']))]
                env = {key: value for key, value in os.environ.items() if not key.startswith('WARODECODE_')}
                env.update(WARODECODE_RESIDUAL_F32=str(residual), WARODECODE_MOE_F32=str(moe),
                           WARODECODE_HC_F32='0')
                begin = time.monotonic()
                process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=1700)
                wall_ms = (time.monotonic() - begin) * 1000
                if process.returncode:
                    raise RuntimeError(f'{name} failed: {process.stderr[-8000:]}\n{process.stdout[-8000:]}')
                record = next(json.loads(line) for line in reversed(process.stdout.splitlines())
                              if line.startswith('{"engine":"warodecode-qwen38"'))
                row = summarize(record, selected_trace(trace), name, residual, moe, spec, wall_ms)
                print(json.dumps(row, allow_nan=False), flush=True)
                rows.append(row)
        return {'status': 'completed', 'gpu': gpu, 'build_sha256': build_sha,
                'model_identity': identity['modal_volume'],
                'model_validation': 'canonical mounted path, filename and size; recorded model checksum not rehashed in this run',
                'settings': spec,
                'cuda_unit_test_stdout': unit.stdout, 'results': rows,
                'accuracy_gates': {'position36': 'NOT RUN', 'position39': 'MEASURED', '64_token': 'NOT RUN'}}

    with app.run():
        return ablate.remote()  # Exactly one A100 allocation; four distinct engine processes.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-a100', action='store_true', help='explicitly allocate A100 after spend block clears')
    parser.add_argument('--output', type=Path, default=ROOT / 'issue1_artifacts/position39_four_way.json')
    args = parser.parse_args()
    if not args.run_a100:
        print(json.dumps({'status': 'DRY RUN; no allocation', **request(ROOT)}, indent=2))
        return
    result = run_a100()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
