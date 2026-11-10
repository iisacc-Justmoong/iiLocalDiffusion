"""Compare unmasked MOSS control execution with the pinned public source."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in
                ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for MOSS comparison')
class MossUpstreamRuntimeTests(unittest.TestCase):
    def test_unmasked_color_depth_control_matches_pinned_upstream_pipeline(self):
        import numpy as np
        import torch
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file
        import resource_fixture
        from weight_files import resolve_weight_file
        from vendor_moss import pipeline as local

        fixture = ROOT / 'resources/test-fixtures/moss-image-pipeline-upstream-85975793.py'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31')
        spec = importlib.util.spec_from_file_location('iild_test_moss_upstream', fixture)
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream
        try:
            spec.loader.exec_module(upstream)
        finally:
            sys.modules.pop(spec.name, None)

        torch.manual_seed(873)
        torch.set_num_threads(2)
        config = dict(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                      text_dim=16, freq_dim=16, latent_channels=8)
        model = local.MOSSImageTransformer(**config).eval()
        reference_model = upstream.MOSSImageTransformer(**config).eval()
        reference_model.load_state_dict(model.state_dict(), strict=True)
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        actual = local.MOSSImagePipeline(transformer=model, scheduler=local.MOSSImageScheduler(), vae=vae)
        reference = upstream.MOSSImagePipeline(transformer=reference_model,
                                                scheduler=upstream.MOSSImageScheduler(), vae=vae)
        actual.set_progress_bar_config(disable=True)
        reference.set_progress_bar_config(disable=True)
        modules = dict(model.named_modules())
        weights = {'video_embedder.weight': torch.cat((model.video_embedder.weight.detach().clone(),
                    torch.ones_like(model.video_embedder.weight) * .05,
                    torch.ones_like(model.video_embedder.weight) * .04), dim=1),
                   'video_embedder.bias': model.video_embedder.bias.detach().clone()}
        for name in [*(f'layers.0.proj.{item}' for item in control.IMAGE_TARGETS),
                     *(f'text_embedder.blocks.{i}.{item}' for i in range(2) for item in control.TEXT_TARGETS)]:
            weight = modules[name].weight
            weights[name + '.lora_A.weight'] = torch.ones(2, weight.shape[1]) * .005
            weights[name + '.lora_B.weight'] = torch.ones(weight.shape[0], 2) * .005
        metadata = {'hayate_control': '1', 'base_model': 'moss-base', 'control_channels': 'color,depth',
                    'control_preprocess': 'vae', 'rank': '2', 'alpha': '2'}
        embeds = torch.randn(1, 3, 16)
        color = Image.new('RGB', (64, 64), (160, 40, 80))
        from native_canny_fixture import hints as canny_hints
        from depth_hint_fixture import hints as depth_hints
        _, canny = canny_hints()
        depth, _ = depth_hints()
        with resource_fixture.TemporaryDirectory(prefix='moss-upstream-control-') as folder:
            adapter = Path(folder) / 'control.safetensors'
            save_file(weights, str(adapter), metadata=metadata)
            selection = control.Selection(resolve_weight_file(str(adapter), 'control'), None, None,
                                          .8, 0., .5, 1.)
            def run(pipeline, image, guidance):
                inputs = dict(prompt_embeds=embeds, height=64, width=64, num_inference_steps=3,
                              guidance_scale=guidance, generator=torch.Generator('cpu').manual_seed(9),
                              uncond_skip_layers=(), output_type='latent')
                if guidance != 1.:
                    inputs['negative_prompt_embeds'] = torch.zeros_like(embeds)
                progress = []
                if pipeline is actual:
                    inputs['progress_callback'] = lambda step, total: progress.append((step, total))
                with torch.no_grad(), control.execution(pipeline, {'depth': image, 'color': color},
                                                         inputs, selection, torch, np) as trace:
                    result = pipeline(**inputs).images
                if pipeline is actual:
                    self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
                return result, trace['control_activity']
            for name, hint in (('canny', canny), ('depth', depth)):
                for guidance in (1., 2.):
                    with self.subTest(preprocessor=name, guidance=guidance):
                        output, activity = run(actual, hint, guidance)
                        expected, expected_activity = run(reference, hint, guidance)
                        self.assertTrue(torch.equal(output, expected))
                        self.assertEqual(activity, expected_activity)
                        self.assertEqual(activity, {'color_steps': [0, 1, 2], 'depth_steps': [0, 1]})
                        self.assertNotIn('step', vars(actual.scheduler))
                        self.assertNotIn('step', vars(reference.scheduler))


if __name__ == '__main__':
    unittest.main()
