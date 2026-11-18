"""Numerical MOSS control tests using tiny local Diffusers components."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical MOSS tests')
class MossControlRuntimeTests(unittest.TestCase):
    def test_provisioned_control_adapter_identity_and_finite_values(self):
        if os.environ.get('IILD_VERIFY_MOSS_CONTROL_PAYLOAD') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_CONTROL_PAYLOAD=1 to audit the real control adapter')
        import hashlib
        import torch
        from safetensors import safe_open
        path = Path(os.environ.get('IILD_MOSS_CONTROL_AUDIT_PATH',
                                   ROOT / 'resources/models/moss_image_color_depth_control.safetensors')).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('MOSS control test input must stay inside the project workspace.')
        item = json.loads((ROOT / 'resources/generation-defaults.json').read_text())['moss_color_depth_control_resource']
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-color-depth-header.json').read_text())['header']
        self.assertTrue(path.is_file(), 'Provision the MOSS control adapter first')
        before = path.stat()
        self.assertEqual(before.st_size, item['size'])
        with path.open('rb') as stream:
            self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), item['sha256'])
        with safe_open(str(path), framework='pt', device='cpu') as source:
            expected = {name: info for name, info in fixture.items() if name != '__metadata__'}
            self.assertEqual(set(source.keys()), set(expected))
            self.assertEqual(source.metadata(), fixture['__metadata__'])
            for name, info in expected.items():
                view = source.get_slice(name)
                self.assertEqual(view.get_shape(), info['shape'], name)
                self.assertEqual(view.get_dtype(), info['dtype'], name)
                self.assertTrue(bool(torch.isfinite(source.get_tensor(name)).all()), name)
        after = path.stat()
        self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_training_preprocessing_matches_published_center_fit_for_both_maps(self):
        import hashlib
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        from types import SimpleNamespace

        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.bn = SimpleNamespace(running_mean=torch.zeros(4), running_var=torch.ones(4))
                self.config = SimpleNamespace(batch_norm_eps=1e-4)
            def encode(self, pixels):
                self.pixels = pixels.clone()
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros((1, 1, 2, 2))))

        pixels = np.random.default_rng(31).integers(0, 256, (8, 24, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        expected_image = ImageOps.fit(image, (8, 8), method=Image.Resampling.LANCZOS)
        expected = torch.from_numpy(np.array(expected_image)).permute(2, 0, 1)[None].float().div(127.5).sub(1)
        for color in (True, False):
            vae = Vae()
            identity = {}
            control._encode_hint(SimpleNamespace(vae=vae), image, {'width': 8, 'height': 8},
                                 torch, np, color=color, preprocess='training', output_identity=identity)
            torch.testing.assert_close(vae.pixels, expected, rtol=0, atol=0)
            self.assertEqual(identity, {'encoding': 'rgb8-row-major', 'size': [8, 8],
                                        'sha256': hashlib.sha256(expected_image.tobytes()).hexdigest()})

    def test_published_header_matches_real_meta_transformer(self):
        import torch
        from vendor_moss.pipeline import MOSSImageTransformer
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-color-depth-header.json').read_text())
        header = fixture['header']
        with torch.device('meta'):
            model = MOSSImageTransformer()
        modules = dict(model.named_modules())
        shapes = {name: tuple(item['shape']) for name, item in header.items() if name != '__metadata__'}
        targets = {name: tuple(module.weight.shape) for name, module in modules.items()
                   if getattr(module, 'weight', None) is not None and module.weight.ndim == 2}
        plan = control.plan_weights(shapes, targets, header['__metadata__'],
                                    projection_shape=tuple(model.video_embedder.weight.shape),
                                    image_layers=len(model.layers), text_layers=len(model.text_embedder.blocks))
        self.assertEqual(len(plan), 262)
        self.assertEqual(model.video_embedder.weight.device.type, 'meta')

    def test_control_pixels_are_normalized_before_bfloat16_cast(self):
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
                self.bn = SimpleNamespace(running_mean=torch.zeros(4), running_var=torch.ones(4))
                self.config = SimpleNamespace(batch_norm_eps=1e-5)
            def encode(self, pixels):
                self.pixels = pixels.clone()
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros((1, 1, 2, 2), dtype=torch.bfloat16)))
        vae = Vae()
        image = Image.new('RGB', (2, 2), (65, 130, 195))
        control._encode_hint(SimpleNamespace(vae=vae), image, {'width': 2, 'height': 2}, torch, np, color=True)
        expected = torch.tensor([65, 130, 195], dtype=torch.float32).div(127.5).sub(1).to(torch.bfloat16)
        torch.testing.assert_close(vae.pixels[0, :, 0, 0], expected, rtol=0, atol=0)

    def test_depth_condition_is_grayscale_while_color_keeps_rgb(self):
        import torch
        import numpy as np
        import diffusers as d
        from PIL import Image
        torch.manual_seed(18)
        torch.set_num_threads(2)
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        from types import SimpleNamespace
        pipe = SimpleNamespace(vae=vae)
        image = Image.new('RGB', (64, 64), (255, 0, 0))
        gray = image.convert('L').convert('RGB')
        inputs = {'width': 64, 'height': 64}
        with torch.no_grad():
            depth = control._encode_hint(pipe, image, inputs, torch, np, color=False)
            depth_gray = control._encode_hint(pipe, gray, inputs, torch, np, color=False)
            color = control._encode_hint(pipe, image, inputs, torch, np, color=True)
        self.assertTrue(torch.equal(depth, depth_gray))
        self.assertFalse(torch.equal(depth, color))

    def test_tiny_moss_transformer_mps_timestep_precision(self):
        import torch
        from vendor_moss.pipeline import MOSSImageTransformer
        if not torch.backends.mps.is_available():
            self.skipTest('MPS is unavailable')
        torch.manual_seed(21)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).eval()
        latent, text = torch.randn(1, 8, 4, 4), torch.randn(1, 3, 16)
        with torch.no_grad():
            cpu = model(latent, torch.tensor([400.]), text).sample
            model = model.to('mps')
            actual = model(latent.to('mps'), torch.tensor([400.], device='mps'), text.to('mps')).sample.cpu()
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, cpu, rtol=.01, atol=.002)

    def test_vetted_loader_uses_local_components_and_strict_weights(self):
        import torch
        import diffusers as d
        import resource_fixture
        import moss_runtime
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler
        with resource_fixture.TemporaryDirectory(prefix='moss-loader-') as temporary:
            folder = Path(temporary)
            model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                         text_dim=16, freq_dim=16, latent_channels=8).eval()
            vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                       layers_per_block=1, latent_channels=2, sample_size=64).eval()
            model.save_pretrained(folder / 'transformer', safe_serialization=True)
            vae.save_pretrained(folder / 'vae', safe_serialization=True)
            MOSSImageScheduler().save_pretrained(folder / 'scheduler')
            (folder / 'text_encoder').mkdir()
            (folder / 'tokenizer').mkdir()
            text_class = type('Qwen3_5Model', (torch.nn.Identity,), {})
            with patch('transformers.AutoModel.from_pretrained',
                       return_value=(text_class(), {'missing_keys': [], 'unexpected_keys': []})) as load_text, \
                    patch('transformers.AutoTokenizer.from_pretrained', return_value=None) as load_tokenizer:
                pipeline = moss_runtime.load(folder, torch.float32, None, d)
            self.assertEqual(type(pipeline.transformer).__name__, 'MOSSImageTransformer')
            self.assertTrue(torch.equal(model.video_embedder.weight, pipeline.transformer.video_embedder.weight))
            self.assertTrue(load_text.call_args.kwargs['local_files_only'])
            self.assertFalse(load_text.call_args.kwargs['trust_remote_code'])
            self.assertTrue(load_tokenizer.call_args.kwargs['local_files_only'])

    def test_vetted_loader_accepts_explicit_pytorch_vae_without_inference(self):
        import torch
        import diffusers as d
        import resource_fixture
        import moss_runtime
        from types import SimpleNamespace
        from vae_defaults import resolve_vae_selection
        from VaeDefaultsRuntimeTests import legacy_vae_name
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='moss-legacy-vae-') as temporary:
            root = Path(temporary);folder = root / 'model'
            transformer = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                              text_dim=16, freq_dim=16, latent_channels=128).eval()
            vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                       layers_per_block=1, latent_channels=32, sample_size=64).eval()
            transformer.save_pretrained(folder / 'transformer', safe_serialization=True)
            vae.save_config(folder / 'vae')
            MOSSImageScheduler().save_pretrained(folder / 'scheduler')
            (folder / 'text_encoder').mkdir();(folder / 'tokenizer').mkdir()
            index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
            text_class = type('Qwen3_5Model', (torch.nn.Identity,), {})
            for legacy in (False, True):
                with self.subTest(legacy=legacy):
                    state = {}
                    for name, tensor in vae.state_dict().items():
                        key = legacy_vae_name(name) if legacy else name
                        if legacy and '.mid.attn_1.' in key and tensor.ndim == 2 and key.endswith('.weight'):
                            tensor = tensor[:, :, None, None].contiguous()
                        state[key] = tensor
                    source = root / ('legacy-vae.pt' if legacy else 'external-vae.pt')
                    torch.save(state, source)
                    before = source.read_bytes()
                    args = SimpleNamespace(vae=str(source), pipeline_class=None, source_kind='directory',
                                           model=str(folder), generation_resources=None)
                    selection, status = resolve_vae_selection(args, index, folder)
                    self.assertEqual((status, selection.source_kind, selection.container_kind), ('explicit', 'file', 'pytorch'))
                    with patch('transformers.AutoModel.from_pretrained',
                               return_value=(text_class(), {'missing_keys': [], 'unexpected_keys': []})), \
                         patch('transformers.AutoTokenizer.from_pretrained', return_value=None):
                        pipeline = moss_runtime.load(folder, torch.float32, selection, d, vae_cache_dir=root / 'cache')
                    for name, expected in vae.state_dict().items():
                        torch.testing.assert_close(pipeline.vae.state_dict()[name], expected, rtol=0, atol=0, msg=name)
                    self.assertEqual(source.read_bytes(), before)

    def test_vetted_loader_converts_tiny_sharded_variant_without_inference(self):
        import torch
        import diffusers as d
        import resource_fixture
        import directory_checkpoint_conversion
        import moss_runtime
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='moss-variant-loader-') as temporary:
            root = Path(temporary);folder = root / 'model'
            model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                         text_dim=16, freq_dim=16, latent_channels=8).eval()
            vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                       layers_per_block=1, latent_channels=2, sample_size=64).eval()
            model.save_pretrained(folder / 'transformer', safe_serialization=False,
                                  variant='fp16', max_shard_size='32KB')
            vae.save_pretrained(folder / 'vae', safe_serialization=True)
            MOSSImageScheduler().save_pretrained(folder / 'scheduler')
            (folder / 'text_encoder').mkdir();(folder / 'tokenizer').mkdir()
            (folder / 'text_encoder/model.safetensors').write_bytes(b'mocked encoder')
            index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
            (folder / 'model_index.json').write_text(json.dumps(index))
            shard_index = folder / 'transformer/diffusion_pytorch_model.bin.index.fp16.json'
            self.assertTrue(shard_index.is_file())
            original = {path: path.read_bytes() for path in (folder / 'transformer').glob('*.bin')}
            original[shard_index] = shard_index.read_bytes()
            self.assertEqual(directory_checkpoint_conversion.automatic_variant(folder, index), 'fp16')
            text_class = type('Qwen3_5Model', (torch.nn.Identity,), {})
            with directory_checkpoint_conversion.prepare(folder, index, root / 'cache', root / 'output', 'fp16') as prepared:
                with patch('transformers.AutoModel.from_pretrained',
                           return_value=(text_class(), {'missing_keys': [], 'unexpected_keys': []})) as load_text, \
                     patch('transformers.AutoTokenizer.from_pretrained', return_value=None):
                    loaded = moss_runtime.load(prepared.path, torch.float32, None, d, variant='fp16')
                self.assertNotIn('variant', load_text.call_args.kwargs)
                actual = loaded.transformer.state_dict()
                self.assertEqual(set(actual), set(model.state_dict()))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(actual[name], value), name)
                self.assertEqual(len(prepared.conversions), 1)
            self.assertEqual({path: path.read_bytes() for path in original}, original)
            self.assertFalse((root / 'output').exists())

    def test_rapid_bfloat16_lora_is_unmerged_and_restored_on_failure(self):
        import torch
        from safetensors.torch import save_file
        from types import SimpleNamespace
        import resource_fixture
        import moss_rapid
        from weight_files import resolve_weight_file
        from vendor_moss.pipeline import MOSSImageTransformer

        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).to(torch.bfloat16).eval()
        modules = dict(model.named_modules())
        weights = {}
        for name in moss_rapid.TARGETS:
            target = f'layers.0.proj.{name}'
            shape = modules[target].weight.shape
            key = 'diffusion_model.' + target
            weights[key + '.lora_down.weight'] = torch.randn(2, shape[1]) * .1
            weights[key + '.lora_up.weight'] = torch.randn(shape[0], 2) * .1
            weights[key + '.alpha'] = torch.tensor(2.)
        times = {key: value.float().clone() for key, value in model.state_dict().items() if key in moss_rapid.TIME_KEYS}
        with resource_fixture.TemporaryDirectory(prefix='moss-rapid-bf16-') as temporary:
            root = Path(temporary)
            for name, tensors in ((moss_rapid.FILES[0], weights), (moss_rapid.FILES[1], times),
                                  (moss_rapid.FILES[2], {'latent_mix.Q': torch.eye(8),
                                                       'step_codes': torch.tensor([1000., 968., 926., 871.])})):
                save_file(tensors, str(root / name))
            selection = moss_rapid.Selection(tuple(resolve_weight_file(str(root / n), n) for n in moss_rapid.FILES))
            linear = modules['layers.0.proj.q']
            original_weight = linear.weight.detach().clone()
            original_time = model.time_embedding
            value = torch.randn(1, 3, 16).to(torch.bfloat16)
            baseline = linear(value)
            key = 'diffusion_model.layers.0.proj.q'
            delta = torch.nn.functional.linear(torch.nn.functional.linear(value.float(),
                    weights[key + '.lora_down.weight']), weights[key + '.lora_up.weight']).to(torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, 'test interruption'), torch.no_grad():
                with moss_rapid.execution(SimpleNamespace(transformer=model), selection, torch):
                    self.assertEqual(model.time_embedding[0].weight.dtype, torch.float32)
                    torch.testing.assert_close(linear(value), baseline + delta, rtol=0, atol=0)
                    self.assertTrue(torch.equal(linear.weight, original_weight))
                    raise RuntimeError('test interruption')
            self.assertIs(model.time_embedding, original_time)
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
            self.assertTrue(torch.equal(linear.weight, original_weight))

    def test_rapid_moss_combines_control_and_restores_model(self):
        import torch
        import numpy as np
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file
        import resource_fixture
        import moss_rapid
        import moss_regions
        from weight_files import resolve_weight_file
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler, MOSSImagePipeline

        torch.set_num_threads(2)
        torch.manual_seed(18)
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                     text_dim=16, freq_dim=16, latent_channels=8).eval()
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=32).eval()
        pipeline = MOSSImagePipeline(transformer=model, scheduler=MOSSImageScheduler(), vae=vae)
        modules = dict(model.named_modules())
        rapid = {}
        for name in moss_rapid.TARGETS:
            target = f'layers.0.proj.{name}'
            weight = modules[target].weight
            key = 'diffusion_model.' + target
            rapid[key + '.lora_down.weight'] = torch.randn(2, weight.shape[1]) * .1
            rapid[key + '.lora_up.weight'] = torch.randn(weight.shape[0], 2) * .1
            rapid[key + '.alpha'] = torch.tensor(2.)
        times = {key: value.clone() for key, value in model.state_dict().items() if key in moss_rapid.TIME_KEYS}
        times['time_embedding.0.bias'] += .1
        control_weights = {'video_embedder.weight': torch.cat((model.video_embedder.weight.detach(),
                            torch.full_like(model.video_embedder.weight, .05),
                            torch.full_like(model.video_embedder.weight, .05)), 1),
                           'video_embedder.bias': model.video_embedder.bias.detach().clone()}
        for target in [*(f'layers.0.proj.{n}' for n in control.IMAGE_TARGETS),
                       *(f'text_embedder.blocks.{i}.{n}' for i in range(2) for n in control.TEXT_TARGETS)]:
            weight = modules[target].weight
            control_weights[target + '.lora_A.weight'] = torch.full((2, weight.shape[1]), .005)
            control_weights[target + '.lora_B.weight'] = torch.full((weight.shape[0], 2), .005)
        metadata = {'hayate_control': '1', 'base_model': 'moss-base', 'control_channels': 'color,depth',
                    'control_preprocess': 'vae', 'rank': '2', 'alpha': '2'}
        with resource_fixture.TemporaryDirectory(prefix='moss-rapid-') as temporary:
            root = Path(temporary)
            for name, tensors in ((moss_rapid.FILES[0], rapid), (moss_rapid.FILES[1], times),
                                  (moss_rapid.FILES[2], {'latent_mix.Q': torch.eye(8) * .1,
                                                       'step_codes': torch.tensor([1000., 968., 926., 871.])})):
                save_file(tensors, str(root / name))
            save_file(control_weights, str(root / 'control.safetensors'), metadata=metadata)
            selection = moss_rapid.Selection(tuple(resolve_weight_file(str(root / n), n) for n in moss_rapid.FILES))
            ctrl = control.Selection(resolve_weight_file(str(root / 'control.safetensors'), 'control'),
                                     None, None, 1., 0., .5, 1.)
            original = {name: getattr(model, name) for name in ('time_embedding', 'shared_block_modulation', 'video_head_modulation')}
            inputs = dict(prompt_embeds=torch.randn(1, 3, 16), width=32, height=32,
                          num_inference_steps=2, guidance_scale=1., output_type='latent')
            def run(value, verify_iteration=False, regional=False):
                inputs['generator'] = torch.Generator('cpu').manual_seed(42)
                image = Image.new('RGB', (32, 32), (value, value, value))
                if regional:
                    pipeline._iild_regions = moss_regions.Attention(torch.tensor([1, 2, 1, 2]),
                                                                    torch.tensor([0, 1, 2]), .5, torch)
                try:
                    with torch.no_grad(), control.execution(pipeline, {'depth': image, 'color': image}, inputs, ctrl, torch, np) as trace:
                        with moss_rapid.execution(pipeline, selection, torch) as adapter:
                            result = moss_rapid.generate(pipeline, inputs, adapter, trace, torch).images
                            activity = {key: list(steps) for key, steps in trace['control_activity'].items()}
                            if verify_iteration:
                                noise = torch.randn((1, 8, 2, 2), generator=torch.Generator('cpu').manual_seed(42))
                                expected = torch.zeros_like(noise)
                                mask = torch.ones((1, 3), dtype=torch.bool)
                                for index in range(2):
                                    trace['completed_steps']['step'] = index
                                    current = noise + torch.einsum('oc,bchw->bohw', adapter['Q'], expected)
                                    expected = noise - model(current, adapter['codes'][index], inputs['prompt_embeds'], mask).sample.float()
                                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                finally:
                    if regional:
                        del pipeline._iild_regions
                return result, activity
            first, activity = run(20, verify_iteration=True)
            repeat, _ = run(20)
            different, _ = run(200)
            self.assertTrue(torch.equal(first, repeat))
            self.assertFalse(torch.equal(first, different))
            self.assertEqual(activity, {'color_steps': [0, 1], 'depth_steps': [0]})
            regional, _ = run(20, regional=True)
            self.assertTrue(torch.equal(regional, run(20, regional=True)[0]))
            self.assertFalse(torch.equal(regional, first))
            self.assertTrue(all(getattr(model, name) is module for name, module in original.items()))
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
            self.assertNotIn('step', vars(pipeline.scheduler))

    def test_real_transformer_vae_and_adapter_replay_and_restore(self):
        from dataclasses import replace
        import torch
        import numpy as np
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file
        import resource_fixture
        from weight_files import resolve_weight_file
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler, MOSSImagePipeline

        torch.set_num_threads(2)
        torch.manual_seed(42)
        transformer = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                           text_dim=16, freq_dim=16, latent_channels=8).eval()
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=2, sample_size=64).eval()
        pipeline = MOSSImagePipeline(transformer=transformer, scheduler=MOSSImageScheduler(), vae=vae)
        pipeline.set_progress_bar_config(disable=True)
        modules = dict(transformer.named_modules())
        weights = {'video_embedder.weight': torch.cat((transformer.video_embedder.weight.detach().clone(),
                                                        torch.ones_like(transformer.video_embedder.weight) * .05,
                                                        torch.ones_like(transformer.video_embedder.weight) * .05), dim=1),
                   'video_embedder.bias': transformer.video_embedder.bias.detach().clone()}
        for name in [*(f'layers.0.proj.{item}' for item in control.IMAGE_TARGETS),
                     *(f'text_embedder.blocks.{index}.{item}' for index in range(2) for item in control.TEXT_TARGETS)]:
            weight = modules[name].weight
            weights[name + '.lora_A.weight'] = torch.ones(2, weight.shape[1]) * .005
            weights[name + '.lora_B.weight'] = torch.ones(weight.shape[0], 2) * .005
        metadata = {'hayate_control': '1', 'base_model': 'moss-base', 'control_channels': 'color,depth',
                    'control_preprocess': 'vae', 'rank': '2', 'alpha': '2'}
        with resource_fixture.TemporaryDirectory(prefix='moss-control-') as temporary:
            path = Path(temporary) / 'control.safetensors'
            save_file(weights, str(path), metadata=metadata)
            selection = control.Selection(resolve_weight_file(str(path), 'control'), None, None, 1., 0., 1., 1.)
            image = Image.new('RGB', (64, 64), (128, 128, 128))
            embeds = torch.randn(1, 3, 16)
            # Use an actual local MOSS package and generic runner with native
            # Canny and text encoding; all model weights remain synthetic.
            import contextlib
            import io
            import generate_any
            import moss_runtime
            import native_control_preprocessing
            from PIL import ImageDraw
            from safetensors.torch import load_file
            from transformers import Qwen2Tokenizer, Qwen3_5Config, Qwen3_5Model
            try:
                native_library = native_control_preprocessing.library_path()
            except ValueError as error:
                self.skipTest(f'Build the native Canny library for the full runner test: {error}')
            root = Path(temporary)
            model_dir = root / 'model'
            transformer.save_pretrained(model_dir / 'transformer', safe_serialization=True)
            vae.save_pretrained(model_dir / 'vae', safe_serialization=True)
            MOSSImageScheduler().save_pretrained(model_dir / 'scheduler')
            text_config = {'vocab_size': 151936, 'hidden_size': 16, 'intermediate_size': 32,
                           'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2,
                           'head_dim': 8, 'layer_types': ['full_attention']}
            vision_config = {'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                             'out_hidden_size': 16, 'num_position_embeddings': 16}
            Qwen3_5Model(Qwen3_5Config(text_config=text_config, vision_config=vision_config)).save_pretrained(
                model_dir / 'text_encoder', safe_serialization=True)
            tokenizer_path = ROOT / 'resources/text_encoders/qwen3vl-4b'
            Qwen2Tokenizer.from_pretrained(str(tokenizer_path), local_files_only=True).save_pretrained(model_dir / 'tokenizer')
            (model_dir / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                    **moss_runtime.COMPONENTS}))
            blank = Image.new('RGB', (64, 64), 'black')
            subject = blank.copy()
            ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
            blank_path, subject_path = root / 'blank.png', root / 'subject.png'
            blank.save(blank_path)
            subject.save(subject_path)
            def run_main(image_path, name, method='canny', depth_model=None, prompt='a red fox', guidance='1'):
                output = root / name
                preprocessing = (['--control-preprocess', 'canny', '--control-preprocessor-library', str(native_library)]
                                 if method == 'canny' else
                                 ['--control-preprocess', 'depth', '--control-depth-model', str(depth_model)])
                with contextlib.redirect_stdout(io.StringIO()):
                    status = generate_any.main(['--model', str(model_dir), '--controlnet', str(path),
                        '--control-image', str(image_path), *preprocessing, '--device', 'cpu', '--seed', '9',
                        '--width', '64', '--height', '64', '--steps', '2', '--guidance-scale', guidance,
                        '--prompt', prompt, '--pipeline-inputs', json.dumps({'output_type': 'latent',
                                                       'max_sequence_length': 64, 'uncond_skip_layers': []}),
                        '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BCHW',
                            'representation_space': 'test-moss-packed'}}), '--no-default-modifiers',
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
            different_prompt, _ = run_main(subject_path, 'different-prompt', prompt='a blue whale')
            guided, _ = run_main(subject_path, 'guided', guidance='2')
            guided_repeat, _ = run_main(subject_path, 'guided-repeat', guidance='2')
            self.assertFalse(torch.equal(edge_latent, different_prompt))
            self.assertFalse(torch.equal(edge_latent, guided))
            self.assertTrue(torch.equal(guided, guided_repeat))
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
            # The real catalogued FLUX.2 VAE has 128 packed channels.  Verify
            # that a package omitting its VAE selects and actually runs it.
            bundled_vae = ROOT / 'resources/vae/flux2/vae/diffusion_pytorch_model.safetensors'
            if bundled_vae.is_file():
                wide = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                            text_dim=16, freq_dim=16, latent_channels=128).eval()
                wide.save_pretrained(model_dir / 'transformer', safe_serialization=True)
                (model_dir / 'vae').rename(root / 'tiny-vae-backup')
                wide_modules = dict(wide.named_modules())
                wide_weights = {'video_embedder.weight': torch.cat((wide.video_embedder.weight.detach().clone(),
                                torch.ones_like(wide.video_embedder.weight) * .05,
                                torch.ones_like(wide.video_embedder.weight) * .05), dim=1),
                                'video_embedder.bias': wide.video_embedder.bias.detach().clone()}
                for name in [*(f'layers.0.proj.{item}' for item in control.IMAGE_TARGETS),
                             *(f'text_embedder.blocks.{index}.{item}' for index in range(2)
                               for item in control.TEXT_TARGETS)]:
                    weight = wide_modules[name].weight
                    wide_weights[name + '.lora_A.weight'] = torch.ones(2, weight.shape[1]) * .005
                    wide_weights[name + '.lora_B.weight'] = torch.ones(weight.shape[0], 2) * .005
                path = root / 'wide-control.safetensors'
                save_file(wide_weights, str(path), metadata=metadata)
                fallback_blank, fallback_report = run_main(blank_path, 'fallback-blank')
                fallback_edge, _ = run_main(subject_path, 'fallback-edge')
                fallback_repeat, _ = run_main(subject_path, 'fallback-repeat')
                fallback_depth_blank, _ = run_main(blank_path, 'fallback-depth-blank', 'depth', depth_model)
                fallback_depth_edge, _ = run_main(subject_path, 'fallback-depth-edge', 'depth', depth_model)
                fallback_depth_repeat, _ = run_main(subject_path, 'fallback-depth-repeat', 'depth', depth_model)
                self.assertEqual(fallback_report['request']['vae']['status'], 'fallback')
                self.assertEqual(Path(fallback_report['request']['vae']['selection']['weight']['path']), bundled_vae)
                self.assertEqual(fallback_blank.shape[1], 128)
                self.assertFalse(torch.equal(fallback_blank, fallback_edge))
                self.assertTrue(torch.equal(fallback_edge, fallback_repeat))
                self.assertFalse(torch.equal(fallback_depth_blank, fallback_depth_edge))
                self.assertTrue(torch.equal(fallback_depth_edge, fallback_depth_repeat))
            def run(controlled, *, settings=None, guidance=1., activity=None, forwards=None, depth_hint=image,
                    input_hints=None):
                inputs = dict(prompt_embeds=embeds, height=64, width=64, num_inference_steps=2,
                              guidance_scale=guidance, generator=torch.Generator('cpu').manual_seed(9),
                              uncond_skip_layers=(), output_type='latent')
                if guidance != 1.:
                    inputs['negative_prompt_embeds'] = torch.zeros_like(embeds)
                if not controlled:
                    return pipeline(**inputs).images
                with torch.no_grad(), control.execution(pipeline, {'depth': depth_hint, 'color': image},
                                                         inputs, settings or selection, torch, np) as trace:
                    handle = None
                    if forwards is not None:
                        handle = transformer.video_embedder.register_forward_hook(
                            lambda *unused: forwards.append(trace['completed_steps']['step']))
                    try:
                        result = pipeline(**inputs).images
                    finally:
                        if handle is not None: handle.remove()
                    if activity is not None:
                        activity.update(trace['control_activity'])
                    if input_hints is not None:
                        input_hints.update(trace['vae_input_hints'])
                    return result
            baseline = run(False)
            first = run(True)
            repeated = run(True)
            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(baseline, first))
            from native_canny_fixture import hints
            with self.subTest(preprocessor='native-canny'):
                blank, edges = hints()
                conditioned = run(True, depth_hint=edges)
                self.assertTrue(torch.equal(conditioned, run(True, depth_hint=edges)))
                self.assertFalse(torch.equal(conditioned, run(True, depth_hint=blank)))
            for precision in ('float32', 'float16'):
                with self.subTest(preprocessor='depth-anything', precision=precision):
                    from depth_hint_fixture import hints as depth_hints
                    first_hint, second_hint = depth_hints(precision=precision)
                    conditioned = run(True, depth_hint=first_hint)
                    self.assertTrue(torch.equal(conditioned, run(True, depth_hint=first_hint)))
                    self.assertFalse(torch.equal(conditioned, run(True, depth_hint=second_hint)))
            with self.subTest(preprocessor='training'):
                rectangle = Image.fromarray(np.random.default_rng(51).integers(0, 256, (32, 64, 3), dtype=np.uint8))
                settings = replace(selection, map_preprocess='training')
                recorded = {}
                fitted = run(True, settings=settings, depth_hint=rectangle, input_hints=recorded)
                import hashlib
                from PIL import ImageOps
                expected_hint = ImageOps.fit(rectangle, (64, 64), method=Image.Resampling.LANCZOS)
                self.assertEqual(recorded['depth']['sha256'], hashlib.sha256(expected_hint.tobytes()).hexdigest())
                self.assertEqual(set(recorded), {'color', 'depth'})
                self.assertTrue(torch.equal(fitted, run(True, settings=settings, depth_hint=rectangle)))
                self.assertFalse(torch.equal(fitted, run(True, depth_hint=rectangle)))
            self.assertTrue(torch.equal(baseline, run(False)))
            activity, forwards = {}, []
            run(True, settings=replace(selection, color_end=.5, depth_end=.75),
                guidance=2., activity=activity, forwards=forwards)
            self.assertEqual(forwards, [0, 0, 1, 1])
            self.assertEqual(activity, {'color_steps': [0], 'depth_steps': [0, 1]})
            self.assertNotIn('step', vars(pipeline.scheduler))


if __name__ == '__main__':
    unittest.main()
