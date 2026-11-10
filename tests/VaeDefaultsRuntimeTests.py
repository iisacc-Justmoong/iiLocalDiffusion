"""Strict local VAE loading with real, small Diffusers components."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
from vae_defaults import _select_directory, load_selected_vae, select_fallback_vae

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical VAE loader tests')
class VaeDefaultsRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('IILD_VERIFY_BUNDLED_VAE_PAYLOAD') == '1',
                         'Set IILD_VERIFY_BUNDLED_VAE_PAYLOAD=1 for full resource hashes')
    def test_existing_local_vae_payloads_match_the_resource_manifest(self):
        from generation_defaults import read_defaults
        from weight_files import file_sha256, file_signature

        root, manifest = read_defaults(str(ROOT / 'resources'))
        entries = [manifest['fallback_vae'], *manifest.get('fallback_vaes', [])]
        for family in ('qwen-image', 'flux1', 'flux2'):
            with self.subTest(family=family):
                item = next(entry for entry in entries if family in entry['families'])
                path = root / item['file']
                if not path.is_file():
                    self.skipTest(f'Provision the local {family} VAE resource for its payload check')
                before = file_signature(path)
                self.assertEqual(path.stat().st_size, item['size'])
                self.assertEqual(file_sha256(path), item['sha256'])
                self.assertEqual(file_signature(path), before)

    def test_existing_local_vae_headers_match_real_diffusers_architectures(self):
        import diffusers as d
        import torch
        from safetensors import safe_open

        checked = 0
        with patch.dict('os.environ', {'IILD_MODEL_VALIDATION': 'metadata'}):
            for family, cls, relative in (
                    ('qwen-image', d.AutoencoderKLQwenImage, 'vae/qwen-image/vae/diffusion_pytorch_model.safetensors'),
                    ('flux1', d.AutoencoderKL, 'vae/flux1/diffusion_pytorch_model.safetensors'),
                    ('flux2', d.AutoencoderKLFlux2, 'vae/flux2/vae/diffusion_pytorch_model.safetensors')):
                with self.subTest(family=family):
                    if not (ROOT / 'resources' / relative).is_file():
                        self.skipTest(f'Provision the local {family} VAE resource for its header check')
                    selection = select_fallback_vae(family, str(ROOT / 'resources'))
                    config = cls.load_config(selection.directory, local_files_only=True)
                    with torch.device('meta'):
                        model = cls.from_config(config)
                    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
                    with safe_open(selection.weight.path, framework='pt', device='cpu') as source:
                        actual = {key: tuple(source.get_slice(key).get_shape()) for key in source.keys()}
                    self.assertEqual(set(actual), set(expected))
                    for key, shape in expected.items():
                        self.assertEqual(actual[key], shape, key)
                    checked += 1
        if not checked:
            self.skipTest('Provision the local fallback VAE resources for header checks')

    def test_local_flux_qwen_and_flux2_vaes_require_complete_weights(self):
        import diffusers as d
        import torch
        from safetensors.torch import load_file, save_file

        torch.set_num_threads(2)
        torch.manual_seed(7)
        cases = (
            ('flux1', d.AutoencoderKL, dict(latent_channels=16, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
                scaling_factor=.3611, shift_factor=.1159)),
            ('flux2', d.AutoencoderKLFlux2, dict(latent_channels=32, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), layers_per_block=1)),
            ('qwen-image', d.AutoencoderKLQwenImage, dict(base_dim=8, z_dim=16,
                dim_mult=[1, 2, 2, 2], num_res_blocks=1)),
        )
        with resource_fixture.TemporaryDirectory(prefix='strict-vae-runtime-') as directory:
            for family, cls, kwargs in cases:
                with self.subTest(family=family):
                    folder = Path(directory) / family
                    original = cls(**kwargs).eval()
                    original.save_pretrained(folder, safe_serialization=True)
                    selection = _select_directory(folder, cls.__name__, family)
                    loaded = load_selected_vae(selection, d, torch.float32)
                    for key, expected in original.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)
                    weights = folder / 'diffusion_pytorch_model.safetensors'
                    state = load_file(str(weights))
                    missing = next(key for key in state if key.endswith('.weight'))
                    del state[missing]
                    save_file(state, str(weights), metadata={'format': 'pt'})
                    selection = _select_directory(folder, cls.__name__, family)
                    with self.assertRaisesRegex(RuntimeError, 'VAE weights'):
                        load_selected_vae(selection, d, torch.float32)


if __name__ == '__main__':
    unittest.main()
