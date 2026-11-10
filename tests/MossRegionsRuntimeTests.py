"""Small-model numerical checks for MOSS region attention."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_regions

AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime')
class MossRegionsRuntimeTests(unittest.TestCase):
    def test_training_region_geometry_follows_center_crop(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace
        import regional_prompt

        pixels = np.zeros((32, 64, 3), dtype=np.uint8)
        pixels[:, :21] = (255, 0, 0)
        pixels[:, 21:] = (0, 255, 0)
        image = Image.fromarray(pixels)
        selection = SimpleNamespace(regions=[SimpleNamespace(color=(0, 255, 0))])
        fitted = regional_prompt.image_groups(image, selection, 32, 32, 'cpu', torch, np, map_fit='center')
        stretched = regional_prompt.image_groups(image, selection, 32, 32, 'cpu', torch, np)
        self.assertEqual(fitted.tolist(), [1, 1, 1, 1])
        self.assertEqual(stretched.tolist(), [0, 1, 0, 1])
    def test_region_tokens_match_color_control_after_output_resolution_resize(self):
        import hashlib
        import numpy as np
        import torch
        from PIL import Image
        import regional_prompt
        pixels = np.zeros((17, 17, 3), dtype=np.uint8)
        pixels[:, :, 0] = 255
        pixels[:, 15] = (0, 255, 0)
        image = Image.fromarray(pixels)
        selection = SimpleNamespace(regions=(regional_prompt.Region((255, 0, 0), 'red', ''),
                                             regional_prompt.Region((0, 255, 0), 'green', '')))
        expected_pixels = np.asarray(image.resize((64, 64), Image.Resampling.NEAREST)
                                     .resize((4, 4), Image.Resampling.NEAREST))
        groups = regional_prompt.image_groups(image, selection, 64, 64, 'cpu', torch, np).reshape(4, 4)
        expected = np.where(expected_pixels[:, :, 1] == 255, 2, 1)
        np.testing.assert_array_equal(groups.numpy(), expected)
        self.assertTrue(bool((groups == 2).any()))
        import moss_controlnet
        import krea2_color_control
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.bn = SimpleNamespace(running_mean=torch.zeros(8), running_var=torch.ones(8))
                self.config = SimpleNamespace(batch_norm_eps=0., latents_mean=[0., 0.], latents_std=[1., 1.])
            def encode(self, values):
                self.pixels = values.clone()
                shape = (1, 2, 1, 8, 8) if values.ndim == 5 else (1, 2, 8, 8)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(shape)))
        vae = Vae()
        pipeline = SimpleNamespace(vae=vae, _pack_latents=lambda value, *args: value)
        canvas = np.array(image.resize((64, 64), Image.Resampling.NEAREST), copy=True)
        expected_input = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.)
        for adapter in (moss_controlnet, krea2_color_control):
            with self.subTest(adapter=adapter.__name__):
                identity = {}
                adapter._encode_hint(pipeline, image, {'width': 64, 'height': 64}, torch, np,
                                     color=True, output_identity=identity)
                actual = vae.pixels.squeeze(2) if vae.pixels.ndim == 5 else vae.pixels
                torch.testing.assert_close(actual, expected_input, rtol=0, atol=0)
                self.assertEqual(identity, {'encoding': 'rgb8-row-major', 'size': [64, 64],
                                            'sha256': hashlib.sha256(canvas.tobytes()).hexdigest()})

    def test_attention_routes_color_groups_and_early_image_isolation(self):
        import torch
        image = torch.tensor([1, 2, 0])
        text = torch.tensor([0, 1, 2])
        attn = moss_regions.Attention(image, text, .5, torch)
        tt, early = attn.masks(0, 2)
        _, late = attn.masks(1, 2)
        self.assertTrue(bool(early[0, 3]))  # region one sees the global prompt
        self.assertTrue(bool(early[0, 4]))  # region one sees its own prompt
        self.assertFalse(bool(early[0, 5]))
        self.assertFalse(bool(early[0, 1]))  # image regions initially isolated
        self.assertTrue(bool(late[0, 1]))
        self.assertFalse(bool(late[0, 5]))  # prompt routing stays regional
        self.assertFalse(bool(tt[1, 2]))
        self.assertTrue(bool(early.any(dim=1).all()))

    def test_other_region_text_does_not_directly_affect_isolated_image_tokens(self):
        import torch
        from vendor_moss.pipeline import MOSSImageTransformer
        torch.set_num_threads(2)
        torch.manual_seed(123)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).eval()
        attn = moss_regions.Attention(torch.tensor([1, 2, 1, 2]), torch.tensor([0, 1, 2]), 1., torch)
        latent = torch.randn(1, 8, 2, 2)
        text = torch.randn(1, 3, 16)
        changed = text.clone()
        changed[:, 2] += torch.randn(16) * 4
        with torch.no_grad():
            first = model(latent, 500., text, region_attention=attn.masks(0, 2)).sample
            second = model(latent, 500., changed, region_attention=attn.masks(0, 2)).sample
        torch.testing.assert_close(first[:, :, :, 0], second[:, :, :, 0], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(first[:, :, :, 1], second[:, :, :, 1]))

    def test_region_manifest_runs_normal_and_few_step_sampling_and_restores_state(self):
        import numpy as np
        import torch
        from PIL import Image
        import moss_rapid
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler, MOSSImagePipeline
        torch.set_num_threads(2)
        torch.manual_seed(22)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).eval()
        pipeline = MOSSImagePipeline(transformer=model, scheduler=MOSSImageScheduler())
        pipeline.set_progress_bar_config(disable=True)
        def encode(prompts, maximum):
            rows = [torch.full((2, 16), {'scene': .1, 'first': .2, 'second': .8}[p]) for p in prompts]
            return torch.stack(rows), torch.ones((len(rows), 2), dtype=torch.bool)
        with resource_fixture.TemporaryDirectory(prefix='moss-regions-runtime-') as temporary:
            root = Path(temporary)
            image = Image.new('RGB', (32, 32), (255, 0, 0))
            image.paste((0, 255, 0), (16, 0, 32, 32))
            image.save(root / 'map.png')
            path = root / 'regions.json'
            path.write_text(json.dumps({'map': 'map.png', 'regions': [
                {'color': '#ff0000', 'prompt': 'first'}, {'color': '#00ff00', 'prompt': 'second'}],
                'isolate_until': .5}))
            args = SimpleNamespace(moss_regions=str(path), controlnet=None, control_color_image=None,
                control_image=None, controlnet_scale=None, control_color_end=None, control_depth_end=None,
                inputs={'prompt': 'scene'}, offload='none')
            selection = moss_regions.resolve(args, 'MOSSImagePipeline')
            def run(rapid, mapped, guidance=1.):
                inputs = {'prompt': 'scene', 'width': 32, 'height': 32, 'num_inference_steps': 2,
                          'guidance_scale': guidance, 'output_type': 'latent',
                          'generator': torch.Generator('cpu').manual_seed(42), 'uncond_skip_layers': ()}
                if guidance != 1.:
                    inputs['negative_prompt'] = 'scene'
                with torch.no_grad(), patch.object(pipeline, 'encode_prompt', side_effect=encode):
                    with moss_regions.execution(pipeline, inputs, selection, mapped, torch, np) as trace:
                        if rapid:
                            result = moss_rapid.generate(pipeline, inputs, {'Q': torch.eye(8) * .1,
                                'codes': torch.tensor([1000., 968., 926., 871.])}, None, torch).images
                        else:
                            result = pipeline(**inputs).images
                self.assertEqual(trace['token_counts'], [2, 2, 2])
                self.assertEqual(inputs['prompt'], 'scene')
                self.assertNotIn('prompt_embeds', inputs)
                self.assertNotIn('_iild_regions', vars(pipeline))
                return result
            for rapid in (False, True):
                first = run(rapid, image)
                self.assertTrue(torch.equal(first, run(rapid, image)))
                self.assertFalse(torch.equal(first, run(rapid, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
            self.assertTrue(torch.equal(run(False, image, 2.), run(False, image, 2.)))


if __name__ == '__main__':
    unittest.main()
