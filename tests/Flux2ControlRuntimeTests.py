"""Exercise FLUX.2 control against tiny real Diffusers transformer and VAE."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import flux2_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical FLUX.2 tests')
class Flux2ControlRuntimeTests(unittest.TestCase):
    def test_klein_comfy_reference_resize_and_restore(self):
        import math
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace
        import klein_reference_resize as resize

        pixels = np.random.default_rng(17).integers(0, 256, (59, 103, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        result, info = resize.prepare([source, source], torch, np, Image)
        factor = math.sqrt(1024 * 1024 / (103 * 59))
        size = (round(103 * factor), round(59 * factor))
        expected = torch.nn.functional.interpolate(torch.from_numpy(pixels).permute(2, 0, 1)[None],
            size=(size[1], size[0]), mode='nearest-exact')[0].permute(1, 2, 0).numpy()
        self.assertEqual(result[0].size, size)
        self.assertTrue(np.array_equal(np.asarray(result[0]), expected))
        self.assertEqual(info['images'][0]['output_size'], list(size))

        class Processor:
            def _resize_to_target_area(self, image, target_area=1024 * 1024):
                return 'original'
        processor = Processor()
        pipeline = SimpleNamespace(image_processor=processor)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with resize.execution(pipeline):
                self.assertIs(processor._resize_to_target_area(source), source)
                raise RuntimeError('fixture')
        self.assertEqual(processor._resize_to_target_area(source), 'original')
        self.assertNotIn('_resize_to_target_area', vars(processor))

    def test_klein_comfy_resize_reaches_real_diffusers_processor_without_second_resize(self):
        import numpy as np
        import torch
        from PIL import Image
        from types import SimpleNamespace
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
        import klein_reference_resize as resize

        torch.set_num_threads(2)
        pixels = np.random.default_rng(19).integers(0, 256, (115, 755, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        (scaled, _), _ = resize.prepare([image, image], torch, np, Image)
        self.assertEqual(scaled.size, (2624, 400))
        self.assertGreater(scaled.width * scaled.height, 1024 * 1024)
        processor = Flux2ImageProcessor(vae_scale_factor=16)
        with resize.execution(SimpleNamespace(image_processor=processor)):
            preserved = processor._resize_to_target_area(scaled, 1024 * 1024)
            self.assertIs(preserved, scaled)
            actual = processor.preprocess(preserved, width=2624, height=400, resize_mode='crop')
        expected = torch.from_numpy(np.array(scaled)).permute(2, 0, 1)[None].float() / 255 * 2 - 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertNotIn('_resize_to_target_area', vars(processor))

    def test_klein_reference_vae_matches_comfy_16_crop_and_latent_normalization(self):
        import numpy as np
        import torch
        import diffusers as d
        from PIL import Image

        torch.set_num_threads(2)
        torch.manual_seed(71)
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        resource_config = json.loads((ROOT / 'resources/vae/flux2/vae/config.json').read_text())
        self.assertEqual(vae.config.batch_norm_eps, resource_config['batch_norm_eps'])
        vae.bn.running_mean.copy_(torch.linspace(-.4, .4, 8))
        vae.bn.running_var.copy_(torch.linspace(.5, 1.5, 8))
        pipe = d.Flux2KleinPipeline(vae=vae, text_encoder=None, tokenizer=None,
                                    scheduler=None, transformer=None)
        pixels = np.random.default_rng(22).integers(0, 256, (73, 99, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        width, height = 99 // 16 * 16, 73 // 16 * 16
        x, y = (99 % 16) // 2, (73 % 16) // 2
        comfy_crop = pixels[y:y + height, x:x + width]
        expected_pixels = torch.from_numpy(comfy_crop.copy()).permute(2, 0, 1)[None].float() / 255 * 2 - 1
        actual_pixels = pipe.image_processor.preprocess(image, width=width, height=height, resize_mode='crop')
        torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)
        with torch.no_grad():
            raw = vae.encode(expected_pixels).latent_dist.mode()
            batch, channels, h, w = raw.shape
            packed = raw.reshape(batch, channels, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
            packed = packed.reshape(batch, channels * 4, h // 2, w // 2)
            expected = torch.nn.functional.batch_norm(packed, vae.bn.running_mean, vae.bn.running_var,
                                                       training=False, eps=vae.config.batch_norm_eps)
            actual = pipe._encode_vae_image(actual_pixels, generator=torch.Generator('cpu').manual_seed(8))
        # Separate VAE evaluations can differ slightly despite identical pixels.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
        import klein_reference_resize as resize
        from types import SimpleNamespace
        with patch.object(vae, 'encode', return_value=SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: raw))):
            with torch.no_grad(), resize.comfy_vae_execution(pipe, torch):
                identical_latent = pipe._encode_vae_image(actual_pixels, generator=torch.Generator('cpu').manual_seed(8))
        torch.testing.assert_close(identical_latent, expected, rtol=0, atol=0)
        with torch.no_grad(), resize.comfy_vae_execution(pipe, torch):
            actual = pipe._encode_vae_image(actual_pixels, generator=torch.Generator('cpu').manual_seed(8))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
        self.assertNotIn('_encode_vae_image', vars(pipe))
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with resize.comfy_vae_execution(pipe, torch):
                raise RuntimeError('fixture')
        self.assertNotIn('_encode_vae_image', vars(pipe))
        reference_latents = [actual, actual[:, :, :2, :3]]
        actual_ids = pipe._prepare_image_ids(reference_latents)[0]
        expected_ids = []
        for index, latent in enumerate(reference_latents, start=1):
            for row in range(latent.shape[2]):
                for column in range(latent.shape[3]):
                    expected_ids.append((index * 10, row, column, 0))
        self.assertTrue(torch.equal(actual_ids, torch.tensor(expected_ids)))

    def test_public_klein_reference_header_converts_to_4b_transformer_shapes(self):
        import torch
        from diffusers import Flux2Transformer2DModel
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lora_to_diffusers

        fixture = json.loads((ROOT / 'resources/test-fixtures/flux2-klein-reference-depth-header.json').read_text())
        self.assertEqual(fixture['base_model_version'], 'flux2_klein_4b')
        self.assertEqual(len(fixture['header']), 160)
        with torch.device('meta'):
            model = Flux2Transformer2DModel(in_channels=128, num_layers=5, num_single_layers=20,
                attention_head_dim=128, num_attention_heads=24, joint_attention_dim=7680,
                timestep_guidance_channels=256, axes_dims_rope=(32, 32, 32, 32))
            original = {name: torch.empty(info['shape']) for name, info in fixture['header'].items()}
            converted = _convert_non_diffusers_flux2_lora_to_diffusers(original)
        modules = dict(model.named_modules())
        self.assertTrue(converted)
        self.assertEqual(sum(name.endswith('.lora_A.weight') for name in converted) * 2, len(converted))
        for name, a in converted.items():
            if not name.endswith('.lora_A.weight'):
                continue
            target = name.removeprefix('transformer.').removesuffix('.lora_A.weight')
            b = converted[name.replace('.lora_A.weight', '.lora_B.weight')]
            self.assertIn(target, modules)
            self.assertEqual(tuple(a.shape), (a.shape[0], modules[target].weight.shape[1]), target)
            self.assertEqual(tuple(b.shape), (modules[target].weight.shape[0], a.shape[0]), target)

    def test_klein_reference_lora_and_ordered_control_hints(self):
        import torch
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file
        from types import SimpleNamespace
        import resource_fixture
        from lora import apply_lora, resolve_lora_selection, validate_lora_activation
        from native_canny_fixture import hints

        torch.set_num_threads(2)
        torch.manual_seed(63)
        transformer = d.Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
            timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2)).eval()
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        pipe = d.Flux2KleinPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(),
                                    vae=vae, text_encoder=None, tokenizer=None, is_distilled=True)
        pipe.set_progress_bar_config(disable=True)
        prompt = torch.randn(1, 4, 16)
        reference = Image.new('RGB', (64, 64), (110, 160, 190))
        blank, edges = hints()
        def run(control, ref=reference):
            with torch.no_grad():
                return pipe(prompt_embeds=prompt, image=[control, ref], width=64, height=64,
                            num_inference_steps=2, guidance_scale=1., output_type='latent',
                            generator=torch.Generator('cpu').manual_seed(11)).images
        baseline = run(edges)
        with resource_fixture.TemporaryDirectory(prefix='klein-reference-') as folder:
            path = Path(folder) / 'control.safetensors'
            # AI Toolkit's original FLUX.2 naming, converted by the installed Diffusers loader.
            key = 'diffusion_model.double_blocks.0.img_attn.proj'
            save_file({key + '.lora_A.weight': torch.full((2, 16), .2),
                       key + '.lora_B.weight': torch.full((16, 2), .2)}, str(path))
            selected = resolve_lora_selection(SimpleNamespace(lora=str(path), lora_revision=None,
                lora_weight_name=None, lora_scale=1.))
            activation = apply_lora(pipe, selected, Path(folder) / 'cache', True)
            controlled = run(edges)
            self.assertFalse(torch.equal(controlled, baseline))
            self.assertTrue(torch.equal(controlled, run(edges)))
            self.assertFalse(torch.equal(controlled, run(blank)))
            self.assertFalse(torch.equal(controlled, run(reference, ref=edges)))
            from depth_hint_fixture import hints as depth_hints
            first_depth, second_depth = depth_hints()
            depth_result = run(first_depth)
            self.assertTrue(torch.equal(depth_result, run(first_depth)))
            self.assertFalse(torch.equal(depth_result, run(second_depth)))
            import klein_reference_resize as resize
            with resize.comfy_vae_execution(pipe, torch):
                comfy_result = run(edges)
                self.assertTrue(torch.equal(comfy_result, run(edges)))
                self.assertFalse(torch.equal(comfy_result, run(blank)))
            self.assertNotIn('_encode_vae_image', vars(pipe))
            validate_lora_activation(pipe, activation)
            pipe.unload_lora_weights()
            self.assertTrue(torch.equal(baseline, run(edges)))

    def test_control_inpaint_mask_and_latents_follow_reference_contract(self):
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.bn = SimpleNamespace(running_mean=torch.zeros(8), running_var=torch.ones(8))
                self.config = SimpleNamespace(batch_norm_eps=0.)
                self.seen = []
            def encode(self, pixels):
                self.seen.append(pixels.clone())
                latent = torch.nn.functional.interpolate(pixels[:, :2], size=(4, 4), mode='nearest')
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))
        vae = Vae()
        control_image = Image.new('RGB', (64, 64), (64, 64, 64))
        inpaint = Image.new('RGB', (64, 64), (192, 192, 192))
        mask_pixels = np.zeros((64, 64), dtype=np.uint8)
        mask_pixels[:, 32:] = 255
        mask = Image.fromarray(mask_pixels)
        result = control._control_latents(SimpleNamespace(vae=vae), control_image,
                 {'width': 64, 'height': 64}, torch, np, inpaint_image=inpaint, mask_image=mask)
        self.assertEqual(tuple(result.shape), (1, 4, 20))
        torch.testing.assert_close(result[0, :, 8:12], torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.],
                                                                     [1., 1., 1., 1.], [0., 0., 0., 0.]]))
        self.assertEqual(len(vae.seen), 2)
        self.assertTrue(torch.all(vae.seen[1][:, :, :, 32:] == 0))
        self.assertTrue(torch.all(vae.seen[1][:, :, :, :32] > 0))
        self.assertTrue(torch.all(result[0, 1::2, 12:] == 0))
        default_mask = control._control_latents(SimpleNamespace(vae=vae), control_image,
                       {'width': 64, 'height': 64}, torch, np, inpaint_image=inpaint)
        self.assertTrue(torch.all(default_mask[:, :, 8:12] == 0))
        self.assertTrue(torch.all(vae.seen[-1] == 0))

    def test_published_header_matches_actual_flux2_transformer_block(self):
        import torch
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
        fixture = json.loads((ROOT / 'resources/test-fixtures/flux2-fun-header.json').read_text())
        shapes = {name: tuple(item['shape']) for name, item in fixture['header'].items() if name != '__metadata__'}
        with torch.device('meta'):
            block = Flux2TransformerBlock(dim=6144, num_attention_heads=48, attention_head_dim=128, mlp_ratio=3.0)
        base = {name: tuple(value.shape) for name, value in block.state_dict().items()}
        control.validate_shapes(shapes, base, hidden=6144, latent_channels=128)
        self.assertEqual(len(shapes), 76)

    def test_control_image_matches_diffusers_preprocessing_before_bfloat16_vae(self):
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        from diffusers.image_processor import VaeImageProcessor

        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
                self.bn = SimpleNamespace(running_mean=torch.zeros(8), running_var=torch.ones(8))
                self.config = SimpleNamespace(batch_norm_eps=1e-5)
            def encode(self, pixels):
                self.pixels = pixels.clone()
                latent = torch.zeros((1, 2, 4, 4), dtype=pixels.dtype)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

        vae = Vae()
        image = Image.new('RGB', (19, 13), (65, 130, 195))
        expected = VaeImageProcessor(vae_scale_factor=16).preprocess(image, width=64, height=64).to(torch.bfloat16)
        result = control._control_latents(SimpleNamespace(vae=vae), image, {'width': 64, 'height': 64}, torch, np)
        torch.testing.assert_close(vae.pixels, expected, rtol=0, atol=0)
        self.assertEqual(tuple(result.shape), (1, 4, 20))

    def test_control_changes_sampling_and_restores_hooks(self):
        import torch
        import numpy as np
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file
        from dataclasses import replace
        from weight_files import resolve_weight_file
        import resource_fixture

        torch.set_num_threads(2)
        torch.manual_seed(27)
        transformer = d.Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=8,
            num_single_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
            timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2)).eval()
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        pipeline = d.Flux2Pipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(),
                                   vae=vae, text_encoder=None, tokenizer=None)
        pipeline.set_progress_bar_config(disable=True)
        weights = {'control_img_in.weight': torch.randn(16, 20) * .1,
                   'control_img_in.bias': torch.randn(16) * .1}
        for i in range(4):
            prefix = f'control_transformer_blocks.{i}.'
            weights.update({prefix + name: value.detach().clone()
                            for name, value in transformer.transformer_blocks[i].state_dict().items()})
            weights[prefix + 'after_proj.weight'] = torch.randn(16, 16) * .05
            weights[prefix + 'after_proj.bias'] = torch.randn(16) * .05
        weights['control_transformer_blocks.0.before_proj.weight'] = torch.randn(16, 16) * .05
        weights['control_transformer_blocks.0.before_proj.bias'] = torch.randn(16) * .05
        with resource_fixture.TemporaryDirectory(prefix='flux2-control-') as temporary:
            path = Path(temporary) / 'control.safetensors'
            save_file(weights, str(path))
            selection = control.Selection(resolve_weight_file(str(path), 'control'), '', 1., 0., 1.)
            prompt = torch.randn(1, 4, 16)
            def run(color=None, end=1., reference=None, inpaint=None, mask=None, hint=None):
                inputs = dict(prompt_embeds=prompt, width=64, height=64, num_inference_steps=2,
                              guidance_scale=4., generator=torch.Generator('cpu').manual_seed(11), output_type='latent')
                if reference is not None:
                    inputs['image'] = Image.new('RGB', (64, 64), (reference, reference, reference))
                with torch.no_grad():
                    if color is None:
                        return pipeline(**inputs).images
                    image = hint if hint is not None else Image.new('RGB', (64, 64), (color, color, color))
                    with control.execution(pipeline, image, inputs, replace(selection, end=end), torch, np,
                                           inpaint_image=inpaint, mask_image=mask):
                        return pipeline(**inputs).images
            baseline = run()
            first, repeat, different = run(20), run(20), run(220)
            self.assertTrue(torch.equal(first, repeat))
            self.assertFalse(torch.equal(first, baseline))
            self.assertFalse(torch.equal(first, different))
            from native_canny_fixture import hints
            with self.subTest(preprocessor='native-canny'):
                blank, edges = hints()
                conditioned = run(20, hint=edges)
                self.assertTrue(torch.equal(conditioned, run(20, hint=edges)))
                self.assertFalse(torch.equal(conditioned, run(20, hint=blank)))
            with self.subTest(preprocessor='depth-anything'):
                from depth_hint_fixture import hints as depth_hints
                first_hint, second_hint = depth_hints()
                conditioned = run(20, hint=first_hint)
                self.assertTrue(torch.equal(conditioned, run(20, hint=first_hint)))
                self.assertFalse(torch.equal(conditioned, run(20, hint=second_hint)))
            # With two steps, the first spans [0, .5]; an end of .4 keeps no step.
            self.assertTrue(torch.equal(baseline, run(20, .4)))
            self.assertTrue(torch.equal(baseline, run()))
            reference_baseline = run(reference=80)
            reference_controlled = run(20, reference=80)
            self.assertTrue(torch.equal(reference_controlled, run(20, reference=80)))
            self.assertFalse(torch.equal(reference_baseline, reference_controlled))
            self.assertFalse(torch.equal(reference_controlled, run(220, reference=80)))
            self.assertTrue(torch.equal(reference_baseline, run(reference=80)))
            inpaint = Image.new('RGB', (64, 64), (190, 100, 70))
            black = Image.new('L', (64, 64), 0)
            white = Image.new('L', (64, 64), 255)
            masked = run(20, reference=80, inpaint=inpaint, mask=black)
            self.assertTrue(torch.equal(masked, run(20, reference=80, inpaint=inpaint, mask=black)))
            self.assertFalse(torch.equal(masked, run(20, reference=80, inpaint=inpaint, mask=white)))
            self.assertTrue(torch.equal(reference_baseline, run(reference=80)))
            self.assertNotIn('step', vars(pipeline.scheduler))
            # Run the actual generic CLI through native Canny and tensor I/O.
            import contextlib
            import io
            import generate_any
            import native_control_preprocessing
            from PIL import ImageDraw
            from safetensors.torch import load_file
            try:
                native_library = native_control_preprocessing.library_path()
            except ValueError as error:
                self.skipTest(f'Build the native Canny library for the full runner test: {error}')
            root = Path(temporary)
            model_dir, embed_file = root / 'model', root / 'embeds.safetensors'
            pipeline.save_pretrained(str(model_dir), safe_serialization=True)
            save_file({'prompt': prompt}, str(embed_file))
            blank = Image.new('RGB', (64, 64), 'black')
            subject = blank.copy()
            ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
            blank_path, subject_path = root / 'blank.png', root / 'subject.png'
            blank.save(blank_path)
            subject.save(subject_path)
            embed = {'tensor_path': str(embed_file), 'key': 'prompt', 'semantic': 'embeddings',
                     'layout': 'BSC', 'representation_space': 'test-flux2-text'}
            def run_main(image, name, method='canny', depth_model=None):
                output = root / name
                preprocessing = (['--control-preprocess', 'canny', '--control-preprocessor-library', str(native_library)]
                                 if method == 'canny' else
                                 ['--control-preprocess', 'depth', '--control-depth-model', str(depth_model)])
                with contextlib.redirect_stdout(io.StringIO()):
                    status = generate_any.main(['--model', str(model_dir), '--controlnet', str(path),
                        '--control-image', str(image), *preprocessing, '--device', 'cpu', '--seed', '11',
                        '--width', '64', '--height', '64', '--steps', '2', '--guidance-scale', '4',
                        '--pipeline-inputs', json.dumps({'prompt_embeds': embed, 'output_type': 'latent'}),
                        '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BSC',
                            'representation_space': 'test-flux2-packed'}}), '--no-default-modifiers',
                        '--cache-dir', str(root / 'cache'), '--output-dir', str(output)])
                self.assertEqual(status, 0)
                report = json.loads((output / 'generation.json').read_text())
                self.assertEqual(report['control_preprocessor']['name'],
                                 'native-canny' if method == 'canny' else 'depth-anything-relative')
                return load_file(str(output / 'images.safetensors'))['images'], report
            empty_latent, empty_report = run_main(blank_path, 'blank-output')
            edge_latent, edge_report = run_main(subject_path, 'edge-output')
            repeat_latent, repeat_report = run_main(subject_path, 'repeat-output')
            self.assertFalse(torch.equal(empty_latent, edge_latent))
            self.assertTrue(torch.equal(edge_latent, repeat_latent))
            self.assertNotEqual(empty_report['control_preprocessor']['output_identity'],
                                edge_report['control_preprocessor']['output_identity'])
            self.assertEqual(edge_report['control_preprocessor']['output_identity'],
                             repeat_report['control_preprocessor']['output_identity'])
            from depth_hint_fixture import save_model
            depth_model = root / 'depth-model'
            save_model(depth_model)
            depth_blank, blank_depth_report = run_main(blank_path, 'depth-blank', 'depth', depth_model)
            depth_subject, subject_depth_report = run_main(subject_path, 'depth-subject', 'depth', depth_model)
            depth_repeat, _ = run_main(subject_path, 'depth-repeat', 'depth', depth_model)
            self.assertFalse(torch.equal(depth_blank, depth_subject))
            self.assertTrue(torch.equal(depth_subject, depth_repeat))
            self.assertEqual(subject_depth_report['control_preprocessor']['output'], 'rgb8')
            self.assertNotEqual(blank_depth_report['control_preprocessor']['output_identity'],
                                subject_depth_report['control_preprocessor']['output_identity'])
            # Exercise the catalogued 128-channel FLUX.2 VAE with a tiny
            # compatible transformer when the model has no bundled VAE.
            bundled_vae = ROOT / 'resources/vae/flux2/vae/diffusion_pytorch_model.safetensors'
            if bundled_vae.is_file():
                wide = d.Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=8,
                    num_single_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
                    timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2)).eval()
                wide.save_pretrained(model_dir / 'transformer', safe_serialization=True)
                (model_dir / 'vae').rename(root / 'tiny-vae-backup')
                wide_weights = {'control_img_in.weight': torch.randn(16, 260) * .1,
                                'control_img_in.bias': torch.randn(16) * .1}
                for i in range(4):
                    prefix = f'control_transformer_blocks.{i}.'
                    wide_weights.update({prefix + name: value.detach().clone()
                                         for name, value in wide.transformer_blocks[i].state_dict().items()})
                    wide_weights[prefix + 'after_proj.weight'] = torch.randn(16, 16) * .05
                    wide_weights[prefix + 'after_proj.bias'] = torch.randn(16) * .05
                wide_weights['control_transformer_blocks.0.before_proj.weight'] = torch.randn(16, 16) * .05
                wide_weights['control_transformer_blocks.0.before_proj.bias'] = torch.randn(16) * .05
                path = root / 'wide-control.safetensors'
                save_file(wide_weights, str(path))
                fallback_blank, fallback_report = run_main(blank_path, 'fallback-blank')
                fallback_edge, _ = run_main(subject_path, 'fallback-edge')
                fallback_repeat, _ = run_main(subject_path, 'fallback-repeat')
                fallback_depth_blank, _ = run_main(blank_path, 'fallback-depth-blank', 'depth', depth_model)
                fallback_depth_edge, _ = run_main(subject_path, 'fallback-depth-edge', 'depth', depth_model)
                fallback_depth_repeat, _ = run_main(subject_path, 'fallback-depth-repeat', 'depth', depth_model)
                self.assertEqual(fallback_report['request']['vae']['status'], 'fallback')
                self.assertEqual(Path(fallback_report['request']['vae']['selection']['weight']['path']), bundled_vae)
                self.assertEqual(fallback_blank.shape[-1], 128)
                self.assertFalse(torch.equal(fallback_blank, fallback_edge))
                self.assertTrue(torch.equal(fallback_edge, fallback_repeat))
                self.assertFalse(torch.equal(fallback_depth_blank, fallback_depth_edge))
                self.assertTrue(torch.equal(fallback_depth_edge, fallback_depth_repeat))
            with self.subTest(style_lora=True):
                if importlib.util.find_spec('peft') is None:
                    self.skipTest('Install PEFT for the LoRA composition test')
                from lora import apply_lora, resolve_lora_selection, validate_lora_activation
                from types import SimpleNamespace
                style = Path(temporary) / 'style.safetensors'
                key = 'transformer.transformer_blocks.0.attn.to_q'
                save_file({key + '.lora_A.weight': torch.full((2, 16), .15),
                           key + '.lora_B.weight': torch.full((16, 2), .15)}, str(style))
                style_selection = resolve_lora_selection(SimpleNamespace(lora=str(style), lora_revision=None,
                    lora_weight_name=None, lora_scale=1.))
                activation = apply_lora(pipeline, style_selection, Path(temporary) / 'cache', True)
                validate_lora_activation(pipeline, activation)
                style_only, both = run(), run(20)
                self.assertFalse(torch.equal(style_only, baseline))
                self.assertFalse(torch.equal(both, style_only))
                self.assertTrue(torch.equal(both, run(20)))
                validate_lora_activation(pipeline, activation)
                failure_inputs = dict(width=64, height=64, num_inference_steps=2,
                    generator=torch.Generator('cpu').manual_seed(11))
                with self.assertRaisesRegex(RuntimeError, 'forced failure'), torch.no_grad(), \
                     control.execution(pipeline, Image.new('RGB', (64, 64)), failure_inputs,
                                       selection, torch, np):
                    raise RuntimeError('forced failure')
                validate_lora_activation(pipeline, activation)
                self.assertTrue(torch.equal(style_only, run()))
                self.assertNotIn('step', vars(pipeline.scheduler))
                pipeline.unload_lora_weights()
                self.assertTrue(torch.equal(baseline, run()))


if __name__ == '__main__':
    unittest.main()
