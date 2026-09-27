"""Base-defined, cross-family coordinate fitting with independent value oracles."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))
if HAS_RUNTIME:
    import torch
    from safetensors.torch import save_file, load_file
from model_merge import merge_models, inspect_merge_request
from model_merge_options import resolve_merge_request


@unittest.skipUnless(HAS_RUNTIME, 'Requires Torch and safetensors')
class BaseLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='base-layout-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.material, self.output = [self.root / (name + '.safetensors') for name in ('base', 'material', 'output')]
        self.key = 'model.diffusion_model.blocks.0.attn.q_proj.weight'
        self.tensor = torch.full((2, 3), 2., dtype=torch.bfloat16)
        save_file({self.key: self.tensor, 'steps': torch.tensor(42), 'denoiser.sigmas': torch.tensor([1., 0.])},
                  self.base, metadata={'modelspec.architecture': 'anima'})

    def save(self, tensors):
        save_file(tensors, self.material, metadata={'modelspec.architecture': 'stable-diffusion-xl'})

    def merge(self, **kwargs):
        report = merge_models(self.base, self.material, output=self.output, checkpoint_policy='base-layout', **kwargs)
        state = load_file(self.output)
        self.assertEqual(set(state), set(load_file(self.base)))
        self.assertEqual(state[self.key].dtype, self.tensor.dtype)
        self.assertEqual(state[self.key].shape, self.tensor.shape)
        self.assertEqual(state['steps'].item(), 42)
        torch.testing.assert_close(state['denoiser.sigmas'], torch.tensor([1., 0.]))
        self.assertEqual(report['output_verification']['status'], 'passed')
        self.assertEqual(report['excluded_material_count'], 0)
        return report, state

    def test_cross_family_unrelated_shape_is_fitted_and_repeatable(self):
        self.save({'foreign.conv.weight': torch.tensor([[4., 8.]])})
        originals = [p.read_bytes() for p in (self.base, self.material)]
        report, state = self.merge(weights=.5)
        torch.testing.assert_close(state[self.key], torch.tensor([[3., 3., 5.], [3., 3., 5.]], dtype=torch.bfloat16))
        self.assertEqual(report['common_layers']['1']['forced_source_tensors'], 1)
        self.assertEqual(report['common_layers']['1']['shape_changed_tensors'], 1)
        self.assertEqual(originals, [p.read_bytes() for p in (self.base, self.material)])
        self.output.unlink()
        _, repeated = self.merge(weights=.5)
        for key in state:
            torch.testing.assert_close(state[key], repeated[key], rtol=0, atol=0)

    def test_no_learned_source_uses_zero_fill_in_sum_and_difference(self):
        self.save({'steps': torch.tensor(7)})
        report, state = self.merge(weights=.5)
        torch.testing.assert_close(state[self.key], torch.ones_like(self.tensor))
        self.assertEqual(report['common_layers']['1']['zero_filled_tensors'], 1)
        self.output.unlink()
        _, state = self.merge(weights=.5, mode='weighted-difference')
        torch.testing.assert_close(state[self.key], self.tensor)

    def test_foreign_lora_delta_is_cropped_and_zero_padded(self):
        module = 'lora_te1_text_model_encoder_layers_0_self_attn_q_proj'
        self.save({module + '.lora_down.weight': torch.tensor([[2., 4.]]),
                   module + '.lora_up.weight': torch.tensor([[3.]])})
        report, state = self.merge(weights=.5)
        torch.testing.assert_close(state[self.key], torch.tensor([[5., 8., 2.], [2., 2., 2.]], dtype=torch.bfloat16))
        self.assertEqual(report['lora_policy'], 'synthetic')
        self.assertEqual(report['sources'][1]['targets'][0]['adaptation']['zero_filled_values'], 4)

    def test_mixed_checkpoint_adapter_is_projected_instead_of_excluded(self):
        self.save({'foreign.weight': torch.full((2, 3), 6.),
                   'orphan.lora_down.weight': torch.ones(1, 2), 'alphas_cumprod': torch.ones(5)})
        report, state = self.merge(weights=.5)
        self.assertTrue(report['resource_compatibility'][0]['raw_tensor_projection'])
        self.assertEqual(report['sources'][1]['kind'], 'checkpoint')
        torch.testing.assert_close(state[self.key], torch.full_like(self.tensor, 4.))

    def test_all_foreign_materials_keep_their_requested_coefficients(self):
        self.save({'foreign.weight': torch.full((2, 3), 6.)})
        second = self.root / 'second.safetensors'
        save_file({'alien.weight': torch.full((2, 3), 10.)}, second, metadata={'modelspec.architecture': 'flux2'})
        report, state = self.merge(weights=[.2, .3], additional_models=[second])
        torch.testing.assert_close(state[self.key], torch.full_like(self.tensor, 5.2))
        self.assertEqual(report['included_material_count'], 2)
        self.assertEqual(report['base_weight'], .5)

    def test_inspection_includes_foreign_lora_without_creating_output(self):
        self.save({'foreign.lora_down.weight': torch.ones(1, 2), 'foreign.lora_up.weight': torch.ones(1, 1)})
        request = resolve_merge_request(self.base, self.material, output=self.output, checkpoint_policy='base-layout')
        first = inspect_merge_request(request)
        self.assertEqual(first, inspect_merge_request(request))
        self.assertEqual(first['included_material_count'], 1)
        self.assertEqual(first['excluded_material_count'], 0)
        self.assertTrue(first['synthetic_adaptations'][1])
        self.assertFalse(self.output.exists())

    def test_common_layer_keeps_its_existing_ecosystem_boundary(self):
        self.save({'foreign.weight': torch.ones(2, 3)})
        report = merge_models(self.base, self.material, output=self.output, checkpoint_policy='common-layer')
        self.assertEqual(report['excluded_material_count'], 1)
        torch.testing.assert_close(load_file(self.output)[self.key], self.tensor)

    def test_bad_material_remains_explicitly_excluded(self):
        self.material.write_bytes(b'truncated')
        report = merge_models(self.base, self.material, output=self.output, checkpoint_policy='base-layout')
        self.assertEqual(report['excluded_material_count'], 1)
        self.assertEqual(report['result_kind'], 'base-fallback')

    def test_anima_namespace_mapping_is_reported_without_shape_conversion(self):
        self.save({self.key.replace('model.diffusion_model.', 'net.'): torch.full((2, 3), 6.)})
        report, state = self.merge(weights=.5)
        torch.testing.assert_close(state[self.key], torch.full_like(self.tensor, 4.))
        mapping = report['common_layers']['1']
        self.assertEqual(mapping['normalized_name_tensors'], 1)
        self.assertEqual(mapping['shape_changed_tensors'], 0)
        self.assertEqual(mapping['forced_source_tensors'], 0)

    def test_nonfinite_foreign_coordinates_are_repaired_and_saved_finite(self):
        self.save({'foreign.weight': torch.tensor([[float('nan'), 6., float('inf')], [10., 12., 14.]])})
        report, state = self.merge(weights=.5)
        torch.testing.assert_close(state[self.key], torch.tensor([[2., 4., 2.], [6., 7., 8.]], dtype=torch.bfloat16))
        self.assertEqual(report['numeric_normalization']['nonfinite_material_values'], 2)
        self.assertTrue(torch.isfinite(state[self.key]).all())


if __name__ == '__main__':
    unittest.main()
