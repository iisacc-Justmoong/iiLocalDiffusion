"""Offline contracts for Krea 2 AI Toolkit edit/pose LoRA names."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_ostris_lora as adapter


class Krea2OstrisTests(unittest.TestCase):
    def test_published_pose_header_maps_all_targets(self):
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-ostris-pose-header.json').read_text())
        self.assertEqual(fixture['revision'], '1da9762ca46e6c9e4014b309bf0147889f10982a')
        self.assertEqual(fixture['weight_size_bytes'], 228587504)
        self.assertEqual(fixture['weight_sha256'], '0ddc3aafce4abdf7af3309b2f00c1bacdf15df1f2b4fb7adc9ff71795da90ecf')
        shapes = {key: tuple(value['shape']) for key, value in fixture['header'].items() if key != '__metadata__'}
        modules = {}
        for name, shape in shapes.items():
            if name.endswith('.lora_A.weight'):
                target = adapter.target_name(name[:-len('.lora_A.weight')])
                up = shapes[name.replace('.lora_A.weight', '.lora_B.weight')]
                modules[target] = (up[0], shape[1])
        plan = adapter.plan_weights(shapes, modules)
        self.assertEqual(len(plan), 256)
        self.assertEqual(len(shapes), 512)
        self.assertIn('text_fusion.refiner_blocks.0.attn.to_q', [item[0] for item in plan])
        broken = dict(shapes)
        broken.pop(next(name for name in broken if name.endswith('.lora_B.weight')))
        with self.assertRaises(ValueError): adapter.plan_weights(broken, modules)
        with self.assertRaises(ValueError): adapter.target_name('diffusion_model.unrecognized.weight')


if __name__ == '__main__':
    unittest.main()
