#!/usr/bin/env python3
"""Offline checks: pinned prefix, position39 logits, trace skips, and validation."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('issue1_ablation', ROOT / 'tools/issue1_position39_ablation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def test_reference_and_four_independent_variants(self):
        request = runner.request(ROOT)
        self.assertEqual(len(request['forced_input_tokens']), 40)
        self.assertEqual(request['forced_input_tokens'][0], 1)
        tokens = json.loads((ROOT / 'issue1_artifacts/reference_tokens_64.json').read_text())['token_ids']
        self.assertEqual(request['forced_input_tokens'][1:], tokens[:39])
        self.assertEqual([(v['WARODECODE_RESIDUAL_F32'], v['WARODECODE_MOE_F32'])
                          for v in request['variants']], [(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_changed_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'issue1_artifacts').mkdir()
            (root / 'issue1_artifacts/reference_tokens_64.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                runner.request(root)

    def test_position39_logits_come_from_selected_binary_trace(self):
        logits = [0.0] * 512
        logits[264], logits[449] = 3.25, 3.0
        router = [float(i) / 10 for i in range(512)]
        vectors = [('pos39.unrelated', [1.0] * 2000),
                   ('pos39.final.logits', logits),
                   ('pos39.layer8.ffn.router_logits', router)]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'trace.bin'
            with trace.open('wb') as stream:
                stream.write(b'WDPTRC1\0' + struct.pack('<II', 1, len(vectors)))
                for name, values in vectors:
                    raw_name = name.encode()
                    stream.write(struct.pack('<IQ', len(raw_name), len(values)) + raw_name)
                    stream.write(struct.pack(f'<{len(values)}f', *values))
            selected = runner.selected_trace(trace)
        self.assertEqual(set(selected), {'pos39.final.logits', 'pos39.layer8.ffn.router_logits'})
        request = runner.request(ROOT)
        record = {'token_ids': [0] * 39 + [264], 'seconds': 2.5,
                  'top_logits': [[264, -100.0], [449, 100.0]],
                  'state_trace': {'trace_position': 39,
                                  'forced_input_token_ids': request['forced_input_tokens'],
                                  'debug_injection_counts': {'moe_routes': 0},
                                  'integer_vectors': [{'name': 'pos39.layer8.ffn.expert_ids',
                                                       'values': list(range(10))}]}}
        row = runner.summarize(record, selected, 'combined', 1, 1, request, 3000.0)
        self.assertEqual((row['token39'], row['logit264'], row['logit449']), (264, 3.25, 3.0))
        self.assertEqual(row['margin264_449'], 0.25)
        self.assertEqual(row['runtime_ms'], 2500.0)
        record['state_trace']['trace_position'] = 0
        with self.assertRaisesRegex(RuntimeError, 'forced prefix'):
            runner.summarize(record, selected, 'combined', 1, 1, request, 3000.0)


if __name__ == '__main__':
    unittest.main()
