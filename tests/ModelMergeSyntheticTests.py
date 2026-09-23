#!/usr/bin/env python3
"""Executable contracts for explicitly untrained cross-family adaptation."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))
if HAS_RUNTIME:
    import torch
    from safetensors.torch import save_file, load_file
from model_merge import merge_models, inspect_merge_request
from model_merge_options import resolve_merge_request, build_parser
from iild_package import materialize_archive


@unittest.skipUnless(HAS_RUNTIME, "Requires the SDK tensor runtime")
class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='synthetic-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.lora = self.root / 'base.safetensors', self.root / 'lora.safetensors'
        self.output = self.root / 'merged.safetensors'
        save_file({'model.diffusion_model.blocks.0.attn.q_proj.weight': torch.ones(4, 5),
                   'model.diffusion_model.blocks.0.attn.k_proj.weight': torch.ones(4, 5),
                   'steps': torch.tensor(5)}, self.base)
        self.module = 'lora_unet_down_blocks_0_attentions_0_to_q'
        self.adapter()

    def adapter(self, rows=2, cols=3, module=None, invalid=False):
        self.down = torch.arange(1., cols + 1).reshape(1, cols)
        self.up = torch.arange(1., rows + 1).reshape(rows, 1)
        if invalid:
            self.up[0, 0] = float('nan')
        name = module or self.module
        save_file({name + '.lora_down.weight': self.down,
                   name + '.lora_up.weight': self.up}, self.lora)

    def merge(self, **kwargs):
        return merge_models(self.base, self.lora, output=self.output, lora_policy='synthetic', **kwargs)

    def test_padding_has_nonzero_effect_and_preserves_base_layout(self):
        before = [p.read_bytes() for p in (self.base, self.lora)]
        report = self.merge(weights=0.25)
        expected = torch.ones(4, 5)
        expected[:2, :3] += 0.25 * (self.up @ self.down)
        state = load_file(self.output)
        torch.testing.assert_close(state['model.diffusion_model.blocks.0.attn.q_proj.weight'], expected)
        torch.testing.assert_close(state['model.diffusion_model.blocks.0.attn.k_proj.weight'], torch.ones(4, 5))
        self.assertEqual(state['steps'].item(), 5)
        self.assertEqual(before, [p.read_bytes() for p in (self.base, self.lora)])
        adaptation = report['sources'][1]['targets'][0]['adaptation']
        self.assertEqual(adaptation['zero_filled_values'], 14)
        self.assertEqual(adaptation['cropped_values'], 0)
        self.assertFalse(adaptation['semantic_equivalence'])

    def test_crop_and_subtraction(self):
        self.adapter(rows=6, cols=7)
        report = self.merge(mode='weighted-difference', weights=0.5)
        torch.testing.assert_close(load_file(self.output)['model.diffusion_model.blocks.0.attn.q_proj.weight'],
                                   torch.ones(4, 5) - 0.5 * (self.up @ self.down)[:4, :5])
        self.assertEqual(report['sources'][1]['targets'][0]['adaptation']['cropped_values'], 22)

    def test_exact_alias_with_wrong_size_is_adapted(self):
        self.adapter(module='model.diffusion_model.blocks.0.attn.q_proj')
        report = self.merge()
        self.assertEqual(report['sources'][1]['targets'][0]['adaptation']['selection'], 'exact-alias-shape-adaptation')

    def test_strict_still_rejects(self):
        with self.assertRaises(ValueError):
            merge_models(self.base, self.lora, output=self.output)
        self.assertFalse(self.output.exists())

    def test_inspection_is_repeatable_and_does_not_write(self):
        request = resolve_merge_request(self.base, self.lora, output=self.output, lora_policy='synthetic')
        first, second = inspect_merge_request(request), inspect_merge_request(request)
        self.assertEqual(first, second)
        self.assertEqual(len(first['synthetic_adaptations'][1]), 1)
        self.assertFalse(self.output.exists())

    def test_unified_fuses_into_supplied_checkpoint_without_bridge(self):
        self.output = self.root / 'merged.iildmodel'
        report = self.merge(mode='unified', compatibility_models=[])
        self.assertEqual(len(report['stages']), 1)
        self.assertEqual(report['compatibility_bridge_count'], 0)
        self.assertTrue(report['stages'][0]['loras'][0]['synthetic_adaptations'])
        package = materialize_archive(self.output, self.root / 'package-cache')
        target = package / report['stages'][0]['model']
        self.assertGreater(load_file(target)['model.diffusion_model.blocks.0.attn.q_proj.weight'].sum().item(), 20)
        self.assertEqual(json.loads((package / 'merge.json').read_text())['lora_policy'], 'synthetic')

    def test_nonfinite_and_unpaired_remain_errors(self):
        self.adapter(invalid=True)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.merge()
        self.assertFalse(self.output.exists())
        save_file({self.module + '.lora_down.weight': self.down}, self.lora)
        with self.assertRaisesRegex(ValueError, 'paired'):
            self.merge()

    def test_convolution_to_matrix_preserves_scaled_intersection(self):
        name = self.module
        down = torch.arange(1., 19.).reshape(1, 2, 3, 3)
        up = torch.tensor([2., 3.]).reshape(2, 1, 1, 1)
        save_file({name + '.lora_down.weight': down, name + '.lora_up.weight': up,
                   name + '.alpha': torch.tensor(0.5)}, self.lora)
        report = self.merge(weights=0.2)
        expected = torch.ones(4, 5)
        expected[:2] += 0.1 * (up.flatten(1) @ down.flatten(1))[:, :5]
        torch.testing.assert_close(load_file(self.output)['model.diffusion_model.blocks.0.attn.q_proj.weight'], expected)
        adaptation = report['sources'][1]['targets'][0]['adaptation']
        self.assertEqual(adaptation['zero_filled_values'], 10)
        self.assertEqual(adaptation['cropped_values'], 26)

    def test_exact_compatible_targets_are_unchanged_by_policy(self):
        self.adapter(rows=4, cols=5, module='model.diffusion_model.blocks.0.attn.q_proj')
        strict_output = self.root / 'strict.safetensors'
        merge_models(self.base, self.lora, output=strict_output)
        report = self.merge()
        self.assertIsNone(report['sources'][1]['targets'][0]['adaptation'])
        for name, value in load_file(strict_output).items():
            torch.testing.assert_close(load_file(self.output)[name], value)

    def test_unified_uses_nearest_preceding_checkpoint(self):
        second = self.root / 'second.safetensors'
        save_file({'transformer.blocks.0.attn.q_proj.weight': torch.ones(3, 4)}, second)
        self.output = self.root / 'two.iildmodel'
        report = merge_models(self.base, second, additional_models=[self.lora], mode='unified',
                              lora_policy='synthetic', output=self.output, compatibility_models=[])
        self.assertEqual(len(report['stages']), 2)
        self.assertEqual(report['stages'][0]['loras'], [])
        self.assertEqual(report['stages'][1]['loras'][0]['source_index'], 2)
        expected = torch.ones(3, 4)
        expected[:2, :3] += self.up @ self.down
        package = materialize_archive(self.output, self.root / 'package-cache')
        torch.testing.assert_close(load_file(package / report['stages'][1]['model'])['transformer.blocks.0.attn.q_proj.weight'], expected)

    def test_parser_and_unknown_policy(self):
        self.assertEqual(build_parser().parse_args(['--base-model', 'a', '--additional-model', 'b',
                                                   '--lora-policy', 'synthetic']).lora_policy, 'synthetic')
        with self.assertRaisesRegex(ValueError, 'policy'):
            resolve_merge_request(self.base, self.lora, lora_policy='random')


if __name__ == '__main__':
    unittest.main()
