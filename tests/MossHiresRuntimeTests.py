"""Synthetic MOSS image-to-image contracts; never loads pretrained weights."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
AVAILABLE = all(importlib.util.find_spec(name) for name in
                ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL', 'numpy'))
HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'Install torch for synthetic MOSS math tests')
class MossHiresMathTests(unittest.TestCase):
    def test_noise_schedule_and_scheduler_restoration(self):
        import types
        import torch
        import moss_hires
        from types import SimpleNamespace

        class Scheduler:
            config = {}
            def __init__(self):
                self.begin_index = None
            @classmethod
            def from_config(cls, _config):
                return cls()
            def set_timesteps(self, count, device):
                self.timesteps = torch.arange(count, 0, -1, device=device).float()
                self.sigmas = torch.linspace(1., 0., count + 1, device=device)
                self.begin_index = None
                self.step_index = None
            def set_begin_index(self, index):
                self.begin_index = index
            def scale_noise(self, source, timestep, noise):
                sigma = self.sigmas[self.begin_index]
                return sigma * noise + (1 - sigma) * source
            def step(self, prediction, timestep, sample, return_dict=False):
                if self.step_index is None:
                    self.step_index = self.begin_index
                i = self.step_index
                self.step_index += 1
                return (sample + (self.sigmas[i + 1] - self.sigmas[i]) * prediction,)

        class Vae:
            dtype = torch.float32
            bn = SimpleNamespace(running_mean=torch.zeros(8), running_var=torch.ones(8))
            config = SimpleNamespace(batch_norm_eps=0.)
            def encode(self, pixels):
                b, _, h, w = pixels.shape
                source = torch.full((b, 2, h // 8, w // 8), .25)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: source))

        class Pipeline:
            _execution_device = 'cpu'
            transformer = SimpleNamespace(config=SimpleNamespace(latent_channels=8))
            vae = Vae()
            image_processor = SimpleNamespace(preprocess=lambda images, height, width:
                                              torch.zeros(len(images), 3, height, width))
            def __init__(self):
                self.scheduler = Scheduler()
                self.corrupt = False
            def __call__(self, *, latents, num_inference_steps, progress_callback, **_):
                self.scheduler.set_timesteps(num_inference_steps, device='cpu')
                for index, timestep in enumerate(self.scheduler.timesteps):
                    value = float('nan') if self.corrupt else 1.
                    latents = self.scheduler.step(torch.full_like(latents, value), timestep,
                                                  latents, return_dict=False)[0]
                    progress_callback(index + 1, len(self.scheduler.timesteps))
                return SimpleNamespace(images=latents)

        fake_utils = types.ModuleType('diffusers.utils.torch_utils')
        fake_utils.randn_tensor = lambda shape, generator, device, dtype: torch.randn(
            shape, generator=generator, device=device, dtype=dtype)
        pipeline = Pipeline()
        original = pipeline.scheduler
        progress = []
        inputs = dict(num_inference_steps=4, generator=torch.Generator().manual_seed(11),
                      progress_callback=lambda step, total: progress.append((step, total)))
        plan = moss_hires.Plan((32, 32), (64, 64), .5, 'lanczos')
        module_name = 'diffusers.utils.torch_utils'
        had_module = module_name in sys.modules
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = fake_utils
        try:
            result = moss_hires.refine(pipeline, [object()], inputs, plan, torch)
            expected_noise = torch.randn((1, 8, 4, 4), generator=torch.Generator().manual_seed(11))
            torch.testing.assert_close(result.images, .5 * expected_noise + .5 * .25 - .5)
            self.assertEqual(progress, [(1, 2), (2, 2)])
            self.assertIs(pipeline.scheduler, original)
            pipeline.corrupt = True
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                moss_hires.refine(pipeline, [object()], inputs, plan, torch)
            self.assertIs(pipeline.scheduler, original)
        finally:
            # Reverting the entire sys.modules mapping also removes unrelated
            # modules lazily imported by Torch; their native registrations are
            # not undone and a later import attempts to register them twice.
            if had_module:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for synthetic MOSS tests')
class MossHiresRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from diffusers import AutoencoderKLFlux2
        from vendor_moss.pipeline import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
        torch.set_num_threads(2)
        torch.manual_seed(27)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).eval()
        vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                 layers_per_block=1, latent_channels=2, sample_size=64).eval()
        self.pipeline = MOSSImagePipeline(model, MOSSImageScheduler(), vae=vae)
        self.pipeline.set_progress_bar_config(disable=True)
        self.prompt = torch.randn(1, 3, 16)

    def test_refinement_encodes_source_and_denoises_selected_schedule(self):
        import torch
        from PIL import Image
        import moss_hires
        from vendor_moss.pipeline import MOSSImageScheduler
        p = self.pipeline
        calls = []
        captures = []
        image = Image.new('RGB', (64, 64), (80, 40, 20))
        original = p.scheduler
        scale_noise = MOSSImageScheduler.scale_noise
        def noise(scheduler, source, timestep, noise):
            result = scale_noise(scheduler, source, timestep, noise=noise)
            captures.append((source.clone(), timestep.clone(), noise.clone(),
                             scheduler.sigmas[scheduler.begin_index].clone(), result.clone()))
            return result
        inputs = dict(prompt_embeds=self.prompt, height=64, width=64, guidance_scale=1.,
                      num_inference_steps=4, uncond_skip_layers=(), output_type='latent',
                      generator=torch.Generator('cpu').manual_seed(9),
                      progress_callback=lambda a, b: calls.append((a, b)))
        with torch.no_grad(), patch.object(p.vae, 'encode', wraps=p.vae.encode) as encode, \
             patch.object(MOSSImageScheduler, 'scale_noise', noise):
            result = moss_hires.refine(p, [image], inputs, moss_hires.Plan((32, 32), (64, 64), .5, 'lanczos'), torch)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(len(captures), 1)
        self.assertEqual(calls, [(1, 2), (2, 2)])
        self.assertIs(p.scheduler, original)
        self.assertEqual(tuple(result.images.shape), (1, 8, 4, 4))
        self.assertTrue(torch.isfinite(result.images).all().item())
        source, timestep, random, sigma, actual = captures[0]
        torch.testing.assert_close(actual, sigma * random + (1 - sigma) * source)
        self.assertEqual(timestep.shape, (1,))

    def test_source_pixels_change_refinement_and_invalid_requests_fail(self):
        import torch
        from PIL import Image
        import moss_hires
        p = self.pipeline
        def run(color):
            inputs = dict(prompt_embeds=self.prompt, height=64, width=64, guidance_scale=1.,
                          num_inference_steps=4, uncond_skip_layers=(), output_type='latent',
                          generator=torch.Generator('cpu').manual_seed(11))
            return moss_hires.refine(p, [Image.new('RGB', (64, 64), color)], inputs,
                                     moss_hires.Plan((32, 32), (64, 64), .5, 'lanczos'), torch).images
        with torch.no_grad():
            first = run((20, 30, 40))
            self.assertTrue(torch.equal(first, run((20, 30, 40))))
            self.assertFalse(torch.equal(first, run((220, 30, 40))))
            vae = p.vae
            p.vae = None
            try:
                with self.assertRaisesRegex(ValueError, 'VAE'):
                    run((20, 30, 40))
            finally:
                p.vae = vae

    def test_complete_two_pass_path_returns_target_and_monotonic_progress(self):
        import torch
        from PIL import Image
        import moss_hires
        progress = []
        inputs = dict(prompt_embeds=self.prompt, width=64, height=64, guidance_scale=1.,
                      num_inference_steps=4, uncond_skip_layers=(),
                      generator=torch.Generator('cpu').manual_seed(3),
                      progress_callback=lambda step, total: progress.append((step, total)))
        plan = moss_hires.Plan((32, 32), (64, 64), .5, 'lanczos')
        with torch.no_grad():
            result, report = moss_hires.run(self.pipeline, inputs, plan, expected_count=1, Image=Image, torch=torch)
        self.assertEqual(result.images[0].size, (64, 64))
        self.assertEqual((report['base_steps'], report['refinement_steps']), (4, 2))
        self.assertEqual(progress, [(125, 1000), (250, 1000), (375, 1000),
                                    (500, 1000), (750, 1000), (1000, 1000)])


if __name__ == '__main__':
    unittest.main()
