"""Contract checks for the FLUX.2 Fun ControlNet adapter."""
import sys
import json
import hashlib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import flux2_controlnet as control


class Flux2ControlTests(unittest.TestCase):
    def test_published_header_matches_flux2_control_contract(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'resources/test-fixtures/flux2-fun-header.json').read_text())
        header = fixture['header']
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, 'd185ed28a216b22739613b57ce2403a81868847380c7588e7d5d0b95306b808c')
        shapes = {name: value['shape'] for name, value in header.items() if name != '__metadata__'}
        base = {f'attn.{name}.weight': (6144, 6144) for name in
                ('add_k_proj', 'add_q_proj', 'add_v_proj', 'to_add_out', 'to_k', 'to_q', 'to_v')}
        base['attn.to_out.0.weight'] = (6144, 6144)
        for name in ('norm_added_k', 'norm_added_q', 'norm_k', 'norm_q'):
            base[f'attn.{name}.weight'] = (128,)
        for name in ('ff', 'ff_context'):
            base[f'{name}.linear_in.weight'] = (36864, 6144)
            base[f'{name}.linear_out.weight'] = (6144, 18432)
        control.validate_shapes(shapes, base, hidden=6144, latent_channels=128)
        self.assertEqual(len(shapes), 76)
        self.assertEqual(fixture['weight_size_bytes'], 8232506680)
        self.assertEqual(fixture['weight_sha256'], '516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794')

    def test_control_window_requires_the_whole_step_to_fit(self):
        # Match Diffusers' ControlNet start/end semantics for non-aligned bounds.
        self.assertEqual([control.control_active(i, 5, .3, .7) for i in range(5)],
                         [False, False, True, False, False])
        self.assertEqual([control.control_active(i, 5, .2, .6) for i in range(5)],
                         [False, True, True, False, False])
        self.assertEqual([control.control_active(i, 5, 0., 1.) for i in range(5)], [True] * 5)
        self.assertFalse(control.control_active(0, 0, 0., 1.))

    def test_published_flux2_dev_tensor_shapes(self):
        base = {f'attn.{name}.weight': (6144, 6144) for name in
                ('add_k_proj', 'add_q_proj', 'add_v_proj', 'to_add_out', 'to_k', 'to_q', 'to_v')}
        base['attn.to_out.0.weight'] = (6144, 6144)
        for name in ('norm_added_k', 'norm_added_q', 'norm_k', 'norm_q'):
            base[f'attn.{name}.weight'] = (128,)
        for name in ('ff', 'ff_context'):
            base[f'{name}.linear_in.weight'] = (36864, 6144)
            base[f'{name}.linear_out.weight'] = (6144, 18432)
        shapes = {'control_img_in.weight': (6144, 260), 'control_img_in.bias': (6144,)}
        for i in range(4):
            prefix = f'control_transformer_blocks.{i}.'
            shapes.update({prefix + key: shape for key, shape in base.items()})
            shapes[prefix + 'after_proj.weight'] = (6144, 6144)
            shapes[prefix + 'after_proj.bias'] = (6144,)
        shapes['control_transformer_blocks.0.before_proj.weight'] = (6144, 6144)
        shapes['control_transformer_blocks.0.before_proj.bias'] = (6144,)
        self.assertEqual(len(shapes), 76)
        control.validate_shapes(shapes, base, hidden=6144, latent_channels=128)

    def test_published_four_block_layout_and_strict_validation(self):
        base = {'attn.to_q.weight': (12, 12), 'ff.linear_in.weight': (72, 12)}
        shapes = {'control_img_in.weight': (12, 20), 'control_img_in.bias': (12,)}
        for index in range(4):
            prefix = f'control_transformer_blocks.{index}.'
            shapes.update({prefix + key: shape for key, shape in base.items()})
            shapes[prefix + 'after_proj.weight'] = (12, 12)
            shapes[prefix + 'after_proj.bias'] = (12,)
        shapes['control_transformer_blocks.0.before_proj.weight'] = (12, 12)
        shapes['control_transformer_blocks.0.before_proj.bias'] = (12,)
        control.validate_shapes(shapes, base, hidden=12, latent_channels=8)
        for mutation in ('extra', 'missing', 'shape'):
            bad = shapes.copy()
            if mutation == 'extra': bad['unexpected'] = (1,)
            if mutation == 'missing': del bad['control_transformer_blocks.2.attn.to_q.weight']
            if mutation == 'shape': bad['control_img_in.weight'] = (12, 19)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                control.validate_shapes(bad, base, hidden=12, latent_channels=8)


if __name__ == '__main__':
    unittest.main()
