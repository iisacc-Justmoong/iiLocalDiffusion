"""Offline preprocessing with a real, randomly initialized tiny Depth Anything model."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import resource_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import depth_preprocessing as depth
import krea2_controlnet as krea_control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'PIL', 'numpy', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install reference runtime for Depth Anything numerical tests')
class DepthPreprocessingRuntimeTests(unittest.TestCase):
    def test_depth_model_receives_selected_float_precision_without_casting_integer_inputs(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import BatchFeature
        from types import SimpleNamespace

        observed = []
        class Model:
            def to(self, device=None, dtype=None): return self
            def eval(self): return self
            def __call__(self, **kwargs):
                observed.append((kwargs['pixel_values'].dtype, kwargs['mask'].dtype))
                return SimpleNamespace(predicted_depth=torch.tensor([[[0., 1.], [2., 3.]]]))

        def features(**kwargs):
            return BatchFeature(data={'pixel_values': torch.zeros(1, 3, 2, 2, dtype=torch.float32),
                                      'mask': torch.ones(1, 2, 2, dtype=torch.int64)})

        with patch('transformers.DPTImageProcessor.from_pretrained', return_value=features), \
             patch('transformers.DepthAnythingForDepthEstimation.from_pretrained', return_value=(Model(), {})):
            for precision in ('float16', 'float32'):
                depth.apply(Image.new('RGB', (2, 2)), depth.Selection('unused', precision), torch, np, Image, 'cpu')
        self.assertEqual(observed, [(torch.float16, torch.int64), (torch.float32, torch.int64)])

    def test_provisioned_depth_model_loads_with_no_weight_mismatches(self):
        variant = os.environ.get('IILD_VERIFY_DEPTH_V2_PAYLOAD')
        if variant not in ('small', 'large'):
            self.skipTest('Set IILD_VERIFY_DEPTH_V2_PAYLOAD to small or large to audit the actual resource')
        import torch
        from transformers import DPTImageProcessor, DepthAnythingForDepthEstimation
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        import provision_depth_anything_v2
        provision_depth_anything_v2.provision(root / 'resources', variant, check=True)
        folder = root / 'resources/depth-estimators' / f'depth-anything-v2-{variant}'
        selection = depth.resolve('depth', str(folder))
        self.assertEqual(selection.directory, str(folder))
        processor = DPTImageProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        self.assertIsInstance(processor, DPTImageProcessor)
        model, info = DepthAnythingForDepthEstimation.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            output_loading_info=True, dtype=torch.float32)
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            self.assertFalse(info.get(key), (key, info.get(key)))
        self.assertTrue(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()))

    def test_extreme_finite_depth_normalizes_without_overflow(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace

        class Features(dict):
            def to(self, device=None, dtype=None): return self
        class Model:
            def to(self, device): return self
            def eval(self): return self
            def __call__(self, **kwargs):
                return SimpleNamespace(predicted_depth=torch.tensor([[[-3.e38, 3.e38]]]))

        with patch('transformers.DPTImageProcessor.from_pretrained', return_value=lambda **kwargs: Features()), \
             patch('transformers.DepthAnythingForDepthEstimation.from_pretrained', return_value=(Model(), {})):
            for preserve_float in (False, True):
                with self.subTest(preserve_float=preserve_float):
                    result, _ = depth.apply(Image.new('RGB', (2, 1)), depth.Selection('unused'),
                                            torch, np, Image, 'cpu', preserve_float=preserve_float)
                    if preserve_float:
                        self.assertTrue(torch.isfinite(result.values).all())
                        torch.testing.assert_close(result.values, torch.tensor([[0., 1.]]))
                    else:
                        self.assertEqual(np.asarray(result)[:, :, 0].tolist(), [[0, 255]])
            with patch.object(torch.nn.functional, 'interpolate',
                              return_value=torch.tensor([[[[0., float('inf')]]]])):
                with self.assertRaisesRegex(ValueError, 'non-finite resized prediction'):
                    depth.apply(Image.new('RGB', (2, 1)), depth.Selection('unused'),
                                torch, np, Image, 'cpu')

    def test_known_relative_depth_normalization(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace

        class Features(dict):
            def to(self, device=None, dtype=None):
                return self

        class Model:
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, **kwargs):
                return SimpleNamespace(predicted_depth=torch.tensor([[[0., 1.], [2., 3.]]]))

        with patch('transformers.DPTImageProcessor.from_pretrained', return_value=lambda **kwargs: Features()), \
             patch('transformers.DepthAnythingForDepthEstimation.from_pretrained', return_value=(Model(), {})):
            result, _ = depth.apply(Image.new('RGB', (2, 2)), depth.Selection('unused'), torch, np, Image, 'cpu')
        self.assertEqual(np.asarray(result)[:, :, 0].tolist(), [[0, 84], [169, 254]])

    def test_krea_depth_uses_unquantized_values_in_vae(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace

        class Features(dict):
            def to(self, device=None, dtype=None): return self
        class Model:
            def to(self, device): return self
            def eval(self): return self
            def __call__(self, **kwargs):
                return SimpleNamespace(predicted_depth=torch.tensor([[[0., 1.], [2., 3.]]]))
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(latents_mean=[0.], latents_std=[1.])
            def encode(self, pixels):
                self.pixels = pixels.clone()
                return SimpleNamespace(latent_dist=SimpleNamespace(
                    sample=lambda generator: torch.zeros((1, 1, 1, 2, 2))))

        with patch('transformers.DPTImageProcessor.from_pretrained', return_value=lambda **kwargs: Features()), \
             patch('transformers.DepthAnythingForDepthEstimation.from_pretrained', return_value=(Model(), {})):
            result, trace = depth.apply(Image.new('RGB', (2, 2)), depth.Selection('unused'), torch, np,
                                        Image, 'cpu', (2, 2), preserve_float=True)
        vae = Vae()
        pipe = SimpleNamespace(vae=vae, _pack_latents=lambda latent, *args: latent)
        identity_from_vae_input = {}
        krea_control._control_latents(pipe, result, {'width': 2, 'height': 2,
            'generator': torch.Generator('cpu').manual_seed(7)}, torch, np, output_identity=identity_from_vae_input)
        expected = torch.tensor([[0., 1.], [2., 3.]]) / (3. + 1e-6) * 2 - 1
        torch.testing.assert_close(vae.pixels[0, 0, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(vae.pixels[0, 1, 0], expected, rtol=0, atol=0)
        self.assertEqual(trace['output'], 'relative-depth-float32')
        import hashlib
        import control_hint_identity
        identity = control_hint_identity.fingerprint(result, np)
        self.assertEqual(identity_from_vae_input, identity)
        canonical = result.values.numpy().astype('<f4', copy=False).tobytes()
        self.assertEqual(identity, {'encoding': 'float32-le-row-major', 'size': [2, 2],
                                    'sha256': hashlib.sha256(canonical).hexdigest()})
        for invalid in (torch.zeros(1, 2), torch.tensor([[0., float('nan')], [0., 1.]]),
                        torch.tensor([[0., -0.1], [0., 1.]])):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'relative depth'):
                krea_control._control_latents(pipe, depth.FloatDepth(invalid), {'width': 2, 'height': 2,
                    'generator': torch.Generator('cpu').manual_seed(7)}, torch, np)

    def test_local_model_crop_rgb_and_repeatability(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import (DepthAnythingConfig, DepthAnythingForDepthEstimation,
                                  Dinov2Config, DPTImageProcessor)
        torch.set_num_threads(2)
        torch.manual_seed(9)
        backbone = Dinov2Config(image_size=56, patch_size=14, hidden_size=32,
            num_hidden_layers=4, num_attention_heads=4, intermediate_size=64,
            out_indices=[1, 2, 3, 4], reshape_hidden_states=False)
        config = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=32,
            neck_hidden_sizes=[8, 16, 32, 32], fusion_hidden_size=16, head_hidden_size=8)
        model = DepthAnythingForDepthEstimation(config)
        processor = DPTImageProcessor(size={'height': 56, 'width': 56},
            keep_aspect_ratio=False, ensure_multiple_of=14)
        with resource_fixture.TemporaryDirectory(prefix='depth-runtime-') as folder:
            model.save_pretrained(folder, safe_serialization=True)
            processor.save_pretrained(folder)
            selection = depth.resolve('depth', folder)
            image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (56, 84, 3), dtype=np.uint8))
            first, trace = depth.apply(image, selection, torch, np, Image, 'cpu', (56, 56))
            second, _ = depth.apply(image, selection, torch, np, Image, 'cpu', (56, 56))
            pixels = np.asarray(first)
            self.assertEqual(first.size, (56, 56))
            self.assertEqual(first.mode, 'RGB')
            self.assertTrue(np.array_equal(pixels, np.asarray(second)))
            self.assertTrue(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]))
            self.assertEqual(trace['near'], 'white')
            self.assertEqual(trace['crop_size'], [56, 56])
            half, half_trace = depth.apply(image, depth.resolve('depth', folder, 'float16'),
                                           torch, np, Image, 'cpu', (56, 56), preserve_float=True)
            self.assertEqual(half.values.dtype, torch.float32)
            self.assertTrue(torch.isfinite(half.values).all())
            self.assertEqual(half_trace['precision'], 'float16')


if __name__ == '__main__':
    unittest.main()
