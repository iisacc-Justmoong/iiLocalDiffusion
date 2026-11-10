"""Local MOSS single-file transformer conversion and control execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_single_file as single
import moss_text_encoder
import moss_tokenizer

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL'))


@unittest.skipUnless(AVAILABLE, 'Install reference dependencies for MOSS single-file tests')
class MossSingleFileRuntimeTests(unittest.TestCase):
    def test_synthetic_lora_provenance_is_reported_without_guessing_quality(self):
        history = {'sources': [
            {'kind': 'checkpoint'},
            {'kind': 'lora', 'targets': [
                {'module': 'diffusion_model.layers.0.proj.ffn_down',
                 'tensor': 'time_embedding.2.weight',
                 'adaptation': {'synthetic': True, 'selection': 'role-depth-shape-nearest'}},
                {'module': 'diffusion_model.layers.1.proj.qkv',
                 'tensor': 'shared_block_modulation.weight',
                 'adaptation': {'synthetic': True, 'selection': 'role-depth-shape-nearest'}},
                {'module': 'exact.layer', 'tensor': 'exact.layer.weight', 'adaptation': None}]}]}
        report = single.merge_history_diagnostics({'iild_merge': json.dumps(history)})
        self.assertEqual(report['synthetic_lora_targets'], 2)
        self.assertEqual(report['examples'][0]['target'], 'time_embedding.2.weight')
        self.assertFalse(report['quality_verified'])
        self.assertIsNone(single.merge_history_diagnostics({'iild_merge': '{invalid'}))
        self.assertIsNone(single.merge_history_diagnostics({'iild_merge': json.dumps({'sources': []})}))

    def test_scheduler_rejects_hidden_flow_options_that_change_sampling(self):
        import moss_runtime
        import resource_fixture
        import torch
        from vendor_moss.pipeline import MOSSImageScheduler
        config = json.loads((ROOT / 'resources/model-configs/moss-image-sft/scheduler/scheduler_config.json').read_text())
        # Diffusers reports this subclass option as ignored, but retains it in
        # the configuration read by the parent scheduler's set_timesteps.
        unsafe = MOSSImageScheduler.from_config({**config, 'invert_sigmas': True})
        unsafe.set_timesteps(4)
        self.assertEqual(unsafe.sigmas[0].item(), 0.)
        self.assertEqual(unsafe.sigmas[-1].item(), 1.)
        with resource_fixture.TemporaryDirectory(prefix='moss-scheduler-config-') as temporary:
            folder = Path(temporary)
            scheduler_dir = folder / 'scheduler'
            scheduler_dir.mkdir()
            config_path = scheduler_dir / 'scheduler_config.json'
            config_path.write_text(json.dumps(config))
            scheduler = moss_runtime.load_scheduler(folder)
            scheduler.set_timesteps(4)
            self.assertEqual(scheduler.sigmas[0].item(), 1.)
            self.assertEqual(scheduler.sigmas[-1].item(), 0.)
            for option, value in (('use_dynamic_shifting', True), ('invert_sigmas', True),
                                  ('shift_terminal', .2), ('use_karras_sigmas', True),
                                  ('use_exponential_sigmas', True), ('use_beta_sigmas', True),
                                  ('stochastic_sampling', True)):
                with self.subTest(option=option):
                    config_path.write_text(json.dumps({**config, option: value}))
                    with self.assertRaisesRegex(ValueError, option):
                        moss_runtime.load_scheduler(folder)
                    if option == 'invert_sigmas':
                        with patch.object(single, 'load') as transformer_load:
                            with self.assertRaisesRegex(ValueError, option):
                                moss_runtime.load(folder, torch.float32, None, None,
                                                  transformer_file=folder / 'unused.safetensors')
                            transformer_load.assert_not_called()

    def test_provisioned_control_attaches_and_restores_without_inference(self):
        if os.environ.get('IILD_VERIFY_MOSS_CONTROL_HOOKS') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_CONTROL_HOOKS=1 to attach the actual MOSS adapter')
        from types import SimpleNamespace
        import moss_controlnet
        import torch
        from weight_files import resolve_weight_file
        torch.set_num_threads(2)
        model = single.load(ROOT / 'resources/models/moss_image_sft_bf16.safetensors',
                            ROOT / 'resources/model-configs/moss-image-sft/transformer',
                            torch.bfloat16, storage='cpu-stream')
        class Scheduler:
            def step(self):
                return None
        scheduler = Scheduler()
        pipeline = SimpleNamespace(transformer=model, scheduler=scheduler)
        modules = dict(model.named_modules())
        def hooks():
            return {name: len(module._forward_hooks) for name, module in modules.items()}
        original = hooks()
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            weight = resolve_weight_file(str(ROOT / 'resources/models/moss_image_color_depth_control.safetensors'),
                                         'MOSS control')
            selected = moss_controlnet.Selection(weight, 'depth', 'color', 1.0, 0.0, 1.0, 1.0)
            # A synthetic latent bypasses the VAE; adapter weights and hooks are real.
            with patch.object(moss_controlnet, '_encode_hint', return_value=torch.zeros(1, 1, 128)):
                with self.assertRaisesRegex(RuntimeError, 'injected exit'):
                    with moss_controlnet.execution(pipeline, {'color': object(), 'depth': object()},
                                                   {'width': 16, 'height': 16}, selected, torch, None):
                        active = hooks()
                        self.assertEqual(sum(active[name] - original[name] for name in active), 263)
                        self.assertIn('step', vars(scheduler))
                        raise RuntimeError('injected exit')
            self.assertEqual(hooks(), original)
            self.assertNotIn('step', vars(scheduler))

    def test_provisioned_components_load_as_one_pipeline_without_inference(self):
        if os.environ.get('IILD_VERIFY_MOSS_PIPELINE_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_PIPELINE_LOAD=1 to load the real local MOSS components')
        import torch
        import diffusers
        import generate_any
        import moss_controlnet
        import moss_runtime
        import resource_fixture
        from PIL import Image
        from safetensors import safe_open
        from streamed_linear import streamed_linear_type
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='moss-real-pipeline-load-') as temporary:
            folder = Path(temporary)
            image = folder / 'control.png'
            Image.new('RGB', (16, 16), (64, 128, 192)).save(image)
            model_path = ROOT / 'resources/models/moss_image_sft_bf16.safetensors'
            adapter_path = ROOT / 'resources/models/moss_image_color_depth_control.safetensors'
            encoder_path = ROOT / 'resources/text_encoders/moss_qwen35_int8_convrot.safetensors'
            depth_model = ROOT / 'resources/depth-estimators/depth-anything-v2-small'
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                parsed = generate_any.build_parser().parse_args([
                    '--model', str(model_path), '--model-config-preset', 'moss-image-sft',
                    '--moss-transformer-storage', 'cpu-stream', '--moss-text-encoder', str(encoder_path),
                    '--controlnet', str(adapter_path), '--control-image', str(image), '--control-color-image', str(image),
                    '--control-preprocess', 'depth', '--control-depth-model', str(depth_model),
                    '--control-map-preprocess', 'training', '--prompt', 'a sunlit studio',
                    '--width', '1024', '--height', '1024', '--steps', '50', '--guidance-scale', '3',
                    '--seed', '42', '--dtype', 'bfloat16', '--offload', 'none', '--no-default-modifiers',
                    '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(parsed)
                pipeline = moss_runtime.load(
                    Path(selected.model_config), torch.bfloat16, selected.vae_selection, diffusers,
                    transformer_file=model_path, transformer_storage='cpu-stream',
                    text_encoder_file=selected.moss_text_encoder_selection,
                    text_config_selection=selected.moss_text_config_selection,
                    tokenizer_selection=selected.moss_tokenizer_selection)
            self.assertEqual(pipeline.transformer._iild_single_file_loading['streamed_layers'], 262)
            self.assertTrue(pipeline.text_encoder._iild_cpu_int8_convrot)
            self.assertIsInstance(pipeline.transformer.layers[0].proj['q'], streamed_linear_type())
            self.assertEqual(type(pipeline.vae).__name__, 'AutoencoderKLFlux2')
            self.assertEqual(pipeline.tokenizer.eos_token_id, 248046)
            with safe_open(str(adapter_path), framework='pt', device='cpu') as source:
                shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
                modules = dict(pipeline.transformer.named_modules())
                targets = {name: tuple(module.weight.shape) for name, module in modules.items()
                           if getattr(module, 'weight', None) is not None and module.weight.ndim == 2}
                plan = moss_controlnet.plan_weights(
                    shapes, targets, source.metadata(), projection_shape=tuple(pipeline.transformer.video_embedder.weight.shape),
                    image_layers=len(pipeline.transformer.layers), text_layers=len(pipeline.transformer.text_embedder.blocks))
            self.assertEqual(len(plan), 262)
            self.assertFalse((folder / 'unused-output').exists())

    def test_provisioned_sft_stream_loads_actual_transformer_weights(self):
        if os.environ.get('IILD_VERIFY_MOSS_STREAM_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_STREAM_LOAD=1 to load the real 10 GB transformer')
        import torch
        from safetensors import safe_open
        from streamed_linear import streamed_linear_type
        torch.set_num_threads(2)
        path = ROOT / 'resources/models/moss_image_sft_bf16.safetensors'
        config = ROOT / 'resources/model-configs/moss-image-sft/transformer'
        model = single.load(path, config, torch.bfloat16, storage='cpu-stream')
        report = model._iild_single_file_loading
        self.assertEqual(report['runtime_weights'], 'cpu-stream')
        self.assertEqual(report['streamed_layers'], 262)
        self.assertGreater(report['streamed_source_bytes'], 0)
        self.assertTrue(all(not value.is_meta for value in model.parameters()))
        with safe_open(str(path), framework='pt', device='cpu') as source:
            self.assertEqual(len(model.layers), 42)
            for block, transformer_layer in enumerate(model.layers):
                fused = source.get_slice(f'layers.{block}.proj.qkv.weight')
                width = fused.get_shape()[0] // 3
                for index, kind in enumerate(('q', 'k', 'v')):
                    with self.subTest(block=block, projection=kind):
                        layer = getattr(transformer_layer.proj, kind)
                        self.assertIsInstance(layer, streamed_linear_type())
                        self.assertEqual(layer._iild_streamed_weight.device.type, 'cpu')
                        self.assertEqual(layer._iild_streamed_weight.dtype, torch.bfloat16)
                        expected = fused[index * width:(index + 1) * width]
                        self.assertTrue(torch.equal(layer._iild_streamed_weight, expected))

            # Check every remaining parameter and the persistent RoPE buffer;
            # the 42 fused QKV matrices were checked independently above.
            from vendor_moss.pipeline import MOSSImageTransformer
            with torch.device('meta'):
                template = MOSSImageTransformer.from_config(json.loads((config / 'config.json').read_text()))
            expected_shapes = {name: tuple(value.shape) for name, value in template.state_dict().items()}
            shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
            mapping = single.plan(shapes, expected_shapes)
            self.assertEqual(len(mapping), 379)
            checked = set()
            for original, targets in mapping.items():
                if len(targets) == 3:
                    continue
                target, = targets
                parent_name, _, leaf = target.rpartition('.')
                parent = model.get_submodule(parent_name)
                if leaf == 'weight' and isinstance(parent, streamed_linear_type()):
                    actual = parent._iild_streamed_weight
                elif target == 'rope.bands':
                    actual = model.get_buffer(target)
                else:
                    actual = model.get_parameter(target).detach()
                self.assertEqual(tuple(actual.shape), expected_shapes[target], target)
                stored = source.get_slice(original)
                for start in range(0, actual.shape[0], 512):
                    end = min(start + 512, actual.shape[0])
                    want = stored[start:end].to(actual.dtype)
                    self.assertTrue(torch.equal(actual[start:end].view(torch.uint8), want.view(torch.uint8)), target)
                checked.add(target)
            self.assertEqual(len(checked), len(expected_shapes) - 42 * 3)

    def test_provisioned_sft_control_and_existing_resources_resolve_together(self):
        if os.environ.get('IILD_VERIFY_MOSS_PREFLIGHT') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_PREFLIGHT=1 to resolve the real local resource combination')
        import generate_any
        import moss_controlnet
        import resource_fixture
        from PIL import Image
        with resource_fixture.TemporaryDirectory(prefix='moss-real-preflight-') as temporary:
            folder = Path(temporary)
            image = folder / 'control.png'
            Image.new('RGB', (16, 16), (64, 128, 192)).save(image)
            model = ROOT / 'resources/models/moss_image_sft_bf16.safetensors'
            adapter = ROOT / 'resources/models/moss_image_color_depth_control.safetensors'
            encoder = ROOT / 'resources/text_encoders/moss_qwen35_int8_convrot.safetensors'
            depth_model = ROOT / 'resources/depth-estimators/depth-anything-v2-small'
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                args = generate_any.build_parser().parse_args([
                    '--model', str(model), '--model-config-preset', 'moss-image-sft',
                    '--moss-transformer-storage', 'cpu-stream', '--moss-text-encoder', str(encoder),
                    '--controlnet', str(adapter), '--control-image', str(image), '--control-color-image', str(image),
                    '--control-preprocess', 'depth', '--control-depth-model', str(depth_model),
                    '--control-map-preprocess', 'training', '--prompt', 'a sunlit studio',
                    '--width', '1024', '--height', '1024', '--steps', '50', '--guidance-scale', '3',
                    '--seed', '42', '--dtype', 'bfloat16', '--offload', 'none', '--no-default-modifiers',
                    '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(args)
            self.assertEqual(Path(selected.model), model)
            self.assertEqual(selected.model_config_status, 'bundled')
            self.assertEqual(selected.moss_text_encoder_status, 'explicit')
            self.assertEqual(Path(selected.moss_text_encoder_selection.resolved_file), encoder.resolve())
            self.assertEqual(selected.moss_tokenizer_status, 'fallback')
            self.assertEqual(selected.vae_status, 'fallback')
            self.assertEqual(Path(selected.vae_selection.directory), ROOT / 'resources/vae/flux2/vae')
            self.assertIsInstance(selected.generic_controlnet, moss_controlnet.Selection)
            self.assertEqual(Path(selected.generic_controlnet.weight.resolved_file), adapter.resolve())
            self.assertEqual(selected.generic_controlnet.map_preprocess, 'training')
            self.assertEqual(Path(selected.depth_preprocessor.directory), depth_model.resolve())
            self.assertEqual(selected.moss_transformer_storage, 'cpu-stream')
            self.assertIsNone(selected.lora_selection)
            report = generate_any.configuration(selected)
            self.assertEqual(report['model_config_preset'], 'moss-image-sft')
            self.assertEqual(report['seed'], 42)
            self.assertEqual(report['moss_transformer_storage'], 'cpu-stream')
            self.assertEqual(report['moss_text_encoder_status'], 'explicit')
            self.assertEqual(report['controlnet']['kind'], 'moss-image-color-depth-control')
            self.assertEqual(report['controlnet']['map_preprocess'], 'training')
            self.assertEqual(Path(report['control_depth_model']), depth_model.resolve())
            self.assertFalse((folder / 'unused-output').exists())

    def test_provisioned_sft_transformer_header_and_sample_values(self):
        mode = os.environ.get('IILD_VERIFY_MOSS_TRANSFORMER_PAYLOAD')
        if mode not in ('1', 'metadata'):
            self.skipTest('Set IILD_VERIFY_MOSS_TRANSFORMER_PAYLOAD=metadata or 1 to audit the 10 GB file')
        import hashlib
        import torch
        from safetensors import safe_open
        path = Path(os.environ.get('IILD_MOSS_TRANSFORMER_AUDIT_PATH',
                                   ROOT / 'resources/models/moss_image_sft_bf16.safetensors')).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('MOSS transformer test input must stay inside the project workspace.')
        item = json.loads((ROOT / 'resources/generation-defaults.json').read_text())['moss_sft_transformer_resource']
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-sft-base-header.json').read_text())['header']
        self.assertTrue(path.is_file(), 'Provision the MOSS SFT transformer first')
        before = path.stat()
        self.assertEqual(before.st_size, item['size'])
        if mode == '1':
            with path.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), item['sha256'])
        with safe_open(str(path), framework='pt', device='cpu') as source:
            self.assertEqual(set(source.keys()), set(fixture))
            for name, info in fixture.items():
                tensor = source.get_slice(name)
                self.assertEqual(tensor.get_shape(), info['shape'], name)
                self.assertEqual(tensor.get_dtype(), info['dtype'], name)
            for name in ('video_embedder.weight', 'layers.0.proj.qkv.weight', 'final_linear_video.weight'):
                self.assertTrue(bool(torch.isfinite(source.get_slice(name)[:1]).all()), name)
        after = path.stat()
        self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_fallback_tokenizer_matches_official_moss_template_cases(self):
        import argparse
        import moss_runtime
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix='moss-tokenizer-') as temporary:
            selection, status = moss_tokenizer.resolve(argparse.Namespace(pipeline_class=None, generation_resources=None),
                {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}, Path(temporary))
            self.assertEqual(status, 'fallback')
            tokenizer = moss_tokenizer.load(selection)
            cases = json.loads((ROOT / 'resources/test-fixtures/moss-qwen35-tokenizer-cases.json').read_text())['cases']
            for case in cases:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                self.assertEqual(prompt, case['formatted'])
                self.assertEqual(tokenizer(prompt).input_ids, case['input_ids'])
            config = moss_text_encoder.resolve_config(Path(temporary))
            self.assertEqual(config.status, 'fallback')
            self.assertEqual(json.loads(Path(config.path).read_text())['model_type'], 'qwen3_5')
            moss_text_encoder.verify_config(config)

    def test_public_fused_header_matches_meta_model(self):
        import torch
        from vendor_moss.pipeline import MOSSImageTransformer
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-sft-base-header.json').read_text())
        with torch.device('meta'):
            model = MOSSImageTransformer()
            config = json.loads((ROOT / 'resources/model-configs/moss-image-sft/transformer/config.json').read_text())
            from_bundled_config = MOSSImageTransformer.from_config(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        self.assertEqual({name: tuple(value.shape) for name, value in from_bundled_config.state_dict().items()}, expected)
        shapes = {name: tuple(value['shape']) for name, value in fixture['header'].items()}
        plan = single.plan(shapes, expected)
        self.assertEqual(len(plan), 379)
        self.assertEqual(sum(len(targets) == 3 for targets in plan.values()), 42)
        self.assertEqual({name for targets in plan.values() for name in targets}, set(expected))
        linears = [name for name, module in model.named_modules()
                   if name.startswith(('layers.', 'text_embedder.blocks.')) and isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(linears), 262)

    def test_public_qwen35_text_header_matches_meta_model(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5TextModel
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-qwen35-encoder-header.json').read_text())
        config = Qwen3_5Config.from_dict(fixture['config'])
        with torch.device('meta'):
            model = Qwen3_5TextModel(config.text_config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        shapes = {name: tuple(value['shape']) for name, value in fixture['header'].items() if name != '__metadata__'}
        mapping = moss_text_encoder.plan(shapes, expected)
        self.assertEqual(len(mapping), 426)
        self.assertEqual(set(mapping.values()), set(expected))

    def test_int8_convrot_transformer_routes_fused_weights_without_losing_outputs(self):
        import copy
        import warnings
        import torch
        import resource_fixture
        from safetensors.torch import save_file
        from MossConvRotRuntimeTests import comfy_eager_reference
        from vendor_moss.pipeline import MOSSImageTransformer

        torch.manual_seed(623)
        torch.set_num_threads(2)
        source = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                      text_dim=16, freq_dim=16, latent_channels=128).eval()
        reference = copy.deepcopy(source)
        weights = {name: value.detach().clone() for name, value in source.state_dict().items()}
        prefix = 'layers.0.proj.'
        weights[prefix + 'qkv.weight'] = torch.cat([weights.pop(prefix + name + '.weight') for name in 'qkv'])
        hadamard = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                                  [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
        encoded = {}
        for name in ('qkv', 'o', 'ffn_up_gate', 'ffn_down'):
            key = prefix + name + '.weight'
            weight = weights[key]
            rotated = (weight.reshape(-1, 4) @ hadamard.T).reshape(weight.shape)
            scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-30)
            quantized = (rotated / scale).round().clamp(-128, 127).to(torch.int8)
            weights[key] = quantized
            weights[key[:-7] + '.weight_scale'] = scale
            meta = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4}).encode()
            weights[key[:-7] + '.comfy_quant'] = torch.tensor(list(meta), dtype=torch.uint8)
            if name == 'qkv':
                encoded.update({prefix + part: pair for part, pair in zip('qkv', zip(quantized.chunk(3), scale.chunk(3)))})
            else:
                encoded[prefix + name] = (quantized, scale)

        oracle = comfy_eager_reference()
        class ReferenceLinear(torch.nn.Linear):
            def __init__(self, original, weight, scale):
                super().__init__(original.in_features, original.out_features, bias=original.bias is not None)
                self.weight = original.weight
                self.bias = original.bias
                self.encoded, self.scale = weight, scale
            def forward(self, value):
                return oracle(value, self.encoded, self.scale, self.bias, out_dtype=value.dtype,
                              convrot=True, convrot_groupsize=4)
        for name, (weight, scale) in encoded.items():
            reference.set_submodule(name, ReferenceLinear(reference.get_submodule(name), weight, scale))
        preference = json.dumps({'attention': 'comfy_kitchen_int8'}).encode()
        weights['layers.0.comfy_attention.config'] = torch.tensor(list(preference), dtype=torch.uint8)

        with resource_fixture.TemporaryDirectory(prefix='moss-transformer-int8-') as temporary:
            root = Path(temporary)
            source.save_config(root)
            checkpoint = root / 'quantized.safetensors'
            history = {'sources': [{'kind': 'lora', 'targets': [{
                'module': 'diffusion_model.layers.0.proj.qkv', 'tensor': 'time_embedding.2.weight',
                'adaptation': {'synthetic': True, 'selection': 'role-depth-shape-nearest'}}]}]}
            save_file(weights, str(checkpoint), metadata={'iild_merge': json.dumps(history)})
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', RuntimeWarning)
                events = []
                loaded = single.load(checkpoint, root, torch.float32, storage='cpu-stream',
                                     progress=lambda *event: events.append(event))
            self.assertTrue(events)
            self.assertTrue(all(event[0] == 'transformer' for event in events))
            self.assertEqual(events[-1][1], events[-1][2])
            self.assertGreater(events[-1][2], 0)
            self.assertEqual([event[1] for event in events], sorted(event[1] for event in events))
            self.assertTrue(any('PyTorch SDPA' in str(item.message) for item in captured))
            self.assertTrue(any('synthetic LoRA' in str(item.message) for item in captured))
            self.assertEqual(loaded._iild_single_file_loading['merge_provenance']['synthetic_lora_targets'], 1)
            self.assertFalse(loaded._iild_single_file_loading['merge_provenance']['quality_verified'])
            self.assertTrue(loaded._iild_cpu_int8_convrot)
            self.assertEqual(loaded._iild_single_file_loading['attention']['preferred'], ['comfy_kitchen_int8'])
            self.assertEqual(loaded._iild_single_file_loading['attention']['selected'], 'pytorch-sdpa')
            self.assertEqual(loaded._iild_single_file_loading['attention']['layers'], 1)
            from moss_convrot import module_types
            quantized_linear, _ = module_types()
            for name, (weight, scale) in encoded.items():
                layer = loaded.get_submodule(name)
                self.assertIsInstance(layer, quantized_linear)
                self.assertTrue(torch.equal(layer._iild_int8, weight))
                self.assertTrue(torch.equal(layer._iild_scale, scale))
            latent, text = torch.randn(1, 128, 2, 2), torch.randn(1, 3, 16)
            loading_event_count = len(events)
            with torch.no_grad():
                expected = reference(latent, 0.5, text).sample
                actual = loaded(latent, 0.5, text).sample
            compute_events = [event for event in events[loading_event_count:] if event[0] == 'transformer_compute']
            self.assertTrue(compute_events)
            self.assertEqual(compute_events[-1][1], compute_events[-1][2])
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
            if torch.backends.mps.is_available():
                loaded.to('mps')
                with torch.no_grad():
                    accelerated = loaded(latent.to('mps'), 0.5, text.to('mps')).sample.cpu()
                torch.testing.assert_close(accelerated, expected, rtol=0.005, atol=0.005)
            missing_scale = dict(weights)
            del missing_scale[prefix + 'qkv.weight_scale']
            save_file(missing_scale, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'scale'):
                single.load(checkpoint, root, torch.float32)
            bad_scale = dict(weights)
            bad_scale[prefix + 'qkv.weight_scale'] = -torch.ones_like(weights[prefix + 'qkv.weight_scale'])
            save_file(bad_scale, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'scale values'):
                single.load(checkpoint, root, torch.float32)
            bad_quant = dict(weights)
            data = json.dumps({'format': 'int8_tensorwise', 'convrot': False, 'convrot_groupsize': 4}).encode()
            bad_quant[prefix + 'qkv.comfy_quant'] = torch.tensor(list(data), dtype=torch.uint8)
            save_file(bad_quant, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'quantization metadata'):
                single.load(checkpoint, root, torch.float32)
            unmatched = dict(weights)
            unmatched[prefix + 'unknown.weight_scale'] = torch.ones(1)
            save_file(unmatched, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'unpaired'):
                single.load(checkpoint, root, torch.float32)

    def test_fused_single_file_runs_control_and_matches_directory(self):
        import contextlib
        import io
        import torch
        import diffusers as d
        from PIL import Image
        from safetensors.torch import save_file, load_file
        from transformers import Qwen2Tokenizer, Qwen3_5Config, Qwen3_5Model
        import resource_fixture
        import moss_controlnet
        import moss_runtime
        import generate_any
        from vendor_moss.pipeline import MOSSImageTransformer, MOSSImageScheduler
        torch.set_num_threads(2)
        torch.manual_seed(813)
        transformer = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                           text_dim=16, freq_dim=16, latent_channels=128).eval()
        state = {name: value.detach().clone() for name, value in transformer.state_dict().items()}
        fused = dict(state)
        prefix = 'layers.0.proj.'
        fused[prefix + 'qkv.weight'] = torch.cat([fused.pop(prefix + name + '.weight') for name in 'qkv'], dim=0)
        vae = d.AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                                   layers_per_block=1, latent_channels=32, sample_size=64).eval()
        with resource_fixture.TemporaryDirectory(prefix='moss-single-') as temporary:
            root = Path(temporary)
            package = root / 'package'
            transformer.save_pretrained(package / 'transformer', safe_serialization=True)
            vae.save_pretrained(package / 'vae', safe_serialization=True)
            MOSSImageScheduler().save_pretrained(package / 'scheduler')
            config = Qwen3_5Config(text_config={'vocab_size': 248320, 'hidden_size': 16, 'intermediate_size': 32,
                'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,
                'head_dim': 8, 'layer_types': ['linear_attention', 'full_attention'],
                'linear_key_head_dim': 8, 'linear_value_head_dim': 8,
                'linear_num_key_heads': 2, 'linear_num_value_heads': 2},
                vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                               'out_hidden_size': 16, 'num_position_embeddings': 16})
            full_encoder = Qwen3_5Model(config).eval()
            full_encoder.save_pretrained(package / 'text_encoder', safe_serialization=True)
            tokenizer = Qwen2Tokenizer.from_pretrained(str(ROOT / 'resources/text_encoders/qwen35-9b'),
                                                       local_files_only=True)
            tokenizer.save_pretrained(package / 'tokenizer')
            encoder_file = root / 'qwen35.safetensors'
            encoder_weights = {'model.' + name: value.detach().clone() for name, value in full_encoder.state_dict().items()}
            encoder_weights['lm_head.weight'] = torch.ones(1)
            encoder_weights['mtp.layers.0.placeholder'] = torch.ones(1)
            save_file(encoder_weights, str(encoder_file))
            from weight_files import resolve_weight_file
            encoder_selection = resolve_weight_file(str(encoder_file), 'MOSS text encoder')
            text_encoder = moss_text_encoder.load(encoder_selection, package / 'text_encoder', torch.float32)
            text_bf16 = moss_text_encoder.load(encoder_selection, package / 'text_encoder', torch.bfloat16)
            for name, expected in full_encoder.language_model.state_dict().items():
                self.assertTrue(torch.equal(text_bf16.state_dict()[name], expected.to(torch.bfloat16)), name)
            encoded = tokenizer('a red fox', return_tensors='pt')
            with torch.no_grad():
                full_output = full_encoder(**encoded, use_cache=False).last_hidden_state
                text_output = text_encoder(**encoded, use_cache=False).last_hidden_state
            torch.testing.assert_close(text_output, full_output, rtol=0, atol=0)
            expected_text = {name: tuple(value.shape) for name, value in text_encoder.state_dict().items()}
            bad_shapes = {name: tuple(value.shape) for name, value in encoder_weights.items()}
            bad_shapes['unknown.weight'] = (1,)
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                moss_text_encoder.plan(bad_shapes, expected_text)
            bad_weights = dict(encoder_weights)
            text_key = next(name for name in bad_weights if name.startswith(moss_text_encoder.PREFIX))
            bad_weights.pop(text_key)
            bad_file = root / 'invalid-encoder.safetensors'
            save_file(bad_weights, str(bad_file))
            with self.assertRaisesRegex(ValueError, 'missing text tensors'):
                moss_text_encoder.load(resolve_weight_file(str(bad_file), 'MOSS text encoder'),
                                       package / 'text_encoder', torch.float32)
            (package / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                  **moss_runtime.COMPONENTS}))
            checkpoint = root / 'moss.safetensors'
            save_file(fused, str(checkpoint))
            loaded = single.load(checkpoint, package / 'transformer', torch.float32)
            for name, expected in state.items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], expected), name)
            bf16 = single.load(checkpoint, package / 'transformer', torch.bfloat16)
            for name, expected in state.items():
                target = expected if name == 'rope.bands' else expected.to(torch.bfloat16)
                self.assertTrue(torch.equal(bf16.state_dict()[name], target), name)
            streamed = single.load(checkpoint, package / 'transformer', torch.bfloat16, storage='cpu-stream')
            layer = streamed.layers[0].proj['q']
            self.assertIsInstance(layer, single.streamed_linear_type())
            self.assertEqual(layer._iild_streamed_weight.dtype, torch.float32)
            self.assertEqual(layer._iild_streamed_weight.device.type, 'cpu')
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
                streamed.state_dict()
            source_weight = layer._iild_streamed_weight
            streamed.to(dtype=torch.float32)
            self.assertIs(layer._iild_streamed_weight, source_weight)
            self.assertEqual(layer.weight.dtype, torch.float32)
            wrong = dict(fused)
            wrong[prefix + 'qkv.weight'] = wrong[prefix + 'qkv.weight'][:-1]
            with self.assertRaisesRegex(ValueError, 'shape'):
                single.plan({key: tuple(value.shape) for key, value in wrong.items()},
                            {key: tuple(value.shape) for key, value in state.items()})
            duplicate = {**fused, prefix + 'q.weight': state[prefix + 'q.weight']}
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                single.plan({key: tuple(value.shape) for key, value in duplicate.items()},
                            {key: tuple(value.shape) for key, value in state.items()})
            corrupt_file = root / 'bad-transformer.safetensors'
            corrupt = dict(fused)
            corrupt[prefix + 'qkv.weight'] = torch.full_like(corrupt[prefix + 'qkv.weight'], float('nan'))
            save_file(corrupt, str(corrupt_file))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                single.load(corrupt_file, package / 'transformer', torch.float32)
            corrupt[prefix + 'qkv.weight'] = torch.ones_like(fused[prefix + 'qkv.weight'], dtype=torch.int8)
            save_file(corrupt, str(corrupt_file))
            with self.assertRaisesRegex(ValueError, 'quantized scale'):
                single.load(corrupt_file, package / 'transformer', torch.float32)
            modules = dict(transformer.named_modules())
            weights = {'video_embedder.weight': torch.cat((transformer.video_embedder.weight.detach().clone(),
                        torch.ones_like(transformer.video_embedder.weight) * .05,
                        torch.ones_like(transformer.video_embedder.weight) * .04), dim=1),
                       'video_embedder.bias': transformer.video_embedder.bias.detach().clone()}
            for name in [*(f'layers.0.proj.{item}' for item in moss_controlnet.IMAGE_TARGETS),
                         *(f'text_embedder.blocks.{i}.{item}' for i in range(2) for item in moss_controlnet.TEXT_TARGETS)]:
                weight = modules[name].weight
                weights[name + '.lora_A.weight'] = torch.ones(2, weight.shape[1]) * .005
                weights[name + '.lora_B.weight'] = torch.ones(weight.shape[0], 2) * .005
            adapter = root / 'control.safetensors'
            save_file(weights, str(adapter), metadata={'hayate_control': '1', 'base_model': 'moss-base',
                'control_channels': 'color,depth', 'control_preprocess': 'vae', 'rank': '2', 'alpha': '2'})
            hint = root / 'hint.png'
            Image.new('RGB', (64, 64), (128, 128, 128)).save(hint)
            embeds = root / 'embeds.safetensors'
            save_file({'prompt': torch.randn(1, 3, 16)}, str(embeds))
            descriptor = {'tensor_path': str(embeds), 'key': 'prompt', 'semantic': 'embeddings',
                          'layout': 'BSC', 'representation_space': 'test-moss-text'}
            inputs = {'prompt_embeds': descriptor, 'output_type': 'latent', 'uncond_skip_layers': []}
            def run(name, single_file, *, single_encoder=False, text_prompt=False, device='cpu', precision='float32',
                    control_image=None, extra=(), encoder_path=None):
                output = root / name
                model_args = ['--model', str(checkpoint), '--model-config', str(package)] if single_file else ['--model', str(package)]
                prompt_inputs = {'prompt': 'a red fox', 'output_type': 'latent', 'uncond_skip_layers': [],
                                 'max_sequence_length': 64} if text_prompt else inputs
                encoder_args = ['--moss-text-encoder', str(encoder_path or encoder_file)] if single_encoder else []
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(generate_any.main([*model_args, *encoder_args, '--controlnet', str(adapter),
                        '--control-image', str(control_image or hint), *extra,
                        '--device', device, '--dtype', precision, '--seed', '9', '--width', '64', '--height', '64', '--steps', '2',
                        '--guidance-scale', '1', '--pipeline-inputs', json.dumps(prompt_inputs), '--tensor-outputs',
                        json.dumps({'images': {'semantic': 'latents', 'layout': 'BCHW', 'representation_space': 'test-moss'}}),
                        '--no-default-modifiers', '--cache-dir', str(root / 'cache'), '--output-dir', str(output)]), 0)
                return load_file(str(output / 'images.safetensors'))['images'], json.loads((output / 'generation.json').read_text())
            expected_output, _ = run('directory', False)
            bf16_expected, _ = run('bf16-directory', False, precision='bfloat16')
            if torch.backends.mps.is_available():
                mps_expected, _ = run('mps-directory', False, device='mps')
            (package / 'transformer/diffusion_pytorch_model.safetensors').rename(root / 'unused-transformer.safetensors')
            actual_output, report = run('single', True)
            self.assertEqual(report['request']['source_kind'], 'single-file')
            self.assertTrue(torch.equal(actual_output, expected_output))
            streamed_output, streamed_report = run('single-stream', True,
                extra=('--moss-transformer-storage', 'cpu-stream'))
            torch.testing.assert_close(streamed_output, actual_output, rtol=0, atol=0)
            self.assertEqual(streamed_report['request']['moss_transformer_storage'], 'cpu-stream')
            self.assertEqual(streamed_report['model']['single_file_loading']['runtime_weights'], 'cpu-stream')
            self.assertEqual(streamed_report['model']['single_file_loading']['streamed_layers'], 16)
            bf16_actual, _ = run('bf16-single', True, precision='bfloat16')
            torch.testing.assert_close(bf16_actual, bf16_expected, rtol=0, atol=0)
            bf16_stream, _ = run('bf16-single-stream', True, precision='bfloat16',
                extra=('--moss-transformer-storage', 'cpu-stream'))
            torch.testing.assert_close(bf16_stream, bf16_actual, rtol=0, atol=0)
            if torch.backends.mps.is_available():
                mps_actual, _ = run('mps-single', True, device='mps')
                torch.testing.assert_close(mps_actual, mps_expected, rtol=0, atol=0)
                mps_stream, _ = run('mps-single-stream', True, device='mps',
                    extra=('--moss-transformer-storage', 'cpu-stream'))
                torch.testing.assert_close(mps_stream, mps_actual, rtol=0, atol=0)
            baseline_prompt, _ = run('package-encoder', True, text_prompt=True)
            bf16_prompt, _ = run('bf16-package-encoder', True, text_prompt=True, precision='bfloat16')
            if torch.backends.mps.is_available():
                mps_prompt, _ = run('mps-package-encoder', True, text_prompt=True, device='mps')
            (package / 'text_encoder/model.safetensors').rename(root / 'unused-text-encoder.safetensors')
            single_prompt, single_report = run('single-encoder', True, single_encoder=True, text_prompt=True)
            self.assertTrue(torch.equal(baseline_prompt, single_prompt))
            self.assertEqual(single_report['request']['moss_text_encoder']['mode'], 'single-file-language-only')
            # Exercise the generic runner with an actual text prompt, INT8 language
            # weights, and the same synthetic color/depth control adapter.
            import moss_convrot
            quantized_weights = dict(encoder_weights)
            for module_name, module in full_encoder.language_model.named_modules():
                if type(module) not in (torch.nn.Linear, torch.nn.Embedding):
                    continue
                key = 'model.language_model.' + module_name + '.weight'
                rotated = moss_convrot.rotate(quantized_weights[key], 4)
                scale = (rotated.abs().amax(dim=-1, keepdim=True).float() / 127).clamp(min=1e-30)
                quantized_weights[key] = (rotated / scale).round().clamp(-128, 127).to(torch.int8)
                quantized_weights[key[:-7] + '.weight_scale'] = scale
                meta = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4}).encode()
                quantized_weights[key[:-7] + '.comfy_quant'] = torch.tensor(list(meta), dtype=torch.uint8)
            quantized_file = root / 'qwen35-int8.safetensors'
            save_file(quantized_weights, str(quantized_file))
            quantized_prompt, _ = run('int8-prompt', True, single_encoder=True, text_prompt=True,
                                      encoder_path=quantized_file)
            quantized_repeat, _ = run('int8-repeat', True, single_encoder=True, text_prompt=True,
                                      encoder_path=quantized_file)
            self.assertTrue(torch.isfinite(quantized_prompt).all())
            torch.testing.assert_close(quantized_prompt, quantized_repeat, rtol=0, atol=0)
            import hashlib
            import shutil
            fallback_root = root / 'int8-fallback-resources'
            fallback_root.mkdir()
            fallback_file = fallback_root / 'encoder.safetensors'
            shutil.copyfile(quantized_file, fallback_file)
            fallback_catalog = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
            fallback_catalog['fallback_moss_text_weights'].update(
                file='encoder.safetensors', size=fallback_file.stat().st_size,
                sha256=hashlib.sha256(fallback_file.read_bytes()).hexdigest())
            (fallback_root / 'generation-defaults.json').write_text(json.dumps(fallback_catalog))
            auto_output, auto_report = run('int8-fallback', True, text_prompt=True,
                                           extra=('--generation-resources', str(fallback_root)))
            torch.testing.assert_close(auto_output, quantized_prompt, rtol=0, atol=0)
            self.assertEqual(auto_report['request']['moss_text_encoder']['status'], 'fallback-int8-convrot')
            bf16_single_prompt, _ = run('bf16-single-encoder', True, single_encoder=True,
                                        text_prompt=True, precision='bfloat16')
            torch.testing.assert_close(bf16_single_prompt, bf16_prompt, rtol=0, atol=0)
            if torch.backends.mps.is_available():
                mps_single_prompt, _ = run('mps-single-encoder', True, single_encoder=True,
                                           text_prompt=True, device='mps')
                torch.testing.assert_close(mps_single_prompt, mps_prompt, rtol=0, atol=0)
            # Keep the tiny architecture but exercise the same manifest-selected fallback path.
            import hashlib
            import shutil
            resources = root / 'fallback-resources'
            shutil.copytree(ROOT / 'resources/text_encoders/qwen35-9b', resources / 'text_encoders/qwen35-9b')
            config_file = resources / 'text_encoders/qwen35-9b/config.json'
            shutil.copyfile(package / 'text_encoder/config.json', config_file)
            catalog = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
            catalog['fallback_moss_text_config']['size'] = config_file.stat().st_size
            catalog['fallback_moss_text_config']['sha256'] = hashlib.sha256(config_file.read_bytes()).hexdigest()
            (resources / 'generation-defaults.json').write_text(json.dumps(catalog))
            (package / 'tokenizer').rename(root / 'unused-tokenizer')
            (package / 'text_encoder/config.json').rename(root / 'unused-text-config.json')
            fallback, fallback_report = run('fallback-text', True, single_encoder=True, text_prompt=True,
                                            extra=('--generation-resources', str(resources)))
            self.assertTrue(torch.equal(single_prompt, fallback))
            self.assertEqual(fallback_report['request']['moss_tokenizer']['status'], 'fallback')
            self.assertEqual(fallback_report['request']['moss_text_encoder']['config']['status'], 'fallback')
            # Restore the package configuration for the independent VAE fallback checks.
            (root / 'unused-text-config.json').rename(package / 'text_encoder/config.json')
            (root / 'unused-tokenizer').rename(package / 'tokenizer')
            bundled_vae = ROOT / 'resources/vae/flux2/vae/diffusion_pytorch_model.safetensors'
            # The ordinary regression is synthetic. Pretrained VAE execution is
            # an explicit separate opt-in, never implied by a local file's presence.
            if bundled_vae.is_file() and os.environ.get('IILD_VERIFY_PRETRAINED_VAE') == '1':
                (package / 'vae').rename(root / 'unused-vae')
                from PIL import ImageDraw
                import native_control_preprocessing
                from depth_hint_fixture import save_model
                blank = Image.new('RGB', (64, 64), 'black')
                subject = blank.copy()
                ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
                blank_path, subject_path = root / 'blank.png', root / 'subject.png'
                blank.save(blank_path)
                subject.save(subject_path)
                try:
                    native = native_control_preprocessing.library_path()
                except ValueError:
                    native = None
                if native:
                    opts = ('--control-preprocess', 'canny', '--control-preprocessor-library', str(native))
                    blank_out, _ = run('fallback-canny-blank', True, single_encoder=True, text_prompt=True,
                                       control_image=blank_path, extra=opts)
                    edge_out, edge_report = run('fallback-canny-edge', True, single_encoder=True, text_prompt=True,
                                                control_image=subject_path, extra=opts)
                    repeated, _ = run('fallback-canny-repeat', True, single_encoder=True, text_prompt=True,
                                      control_image=subject_path, extra=opts)
                    self.assertEqual(edge_report['request']['vae']['status'], 'fallback')
                    self.assertEqual(Path(edge_report['request']['vae']['selection']['weight']['path']), bundled_vae)
                    self.assertFalse(torch.equal(blank_out, edge_out))
                    self.assertTrue(torch.equal(edge_out, repeated))
                    int8_blank, _ = run('int8-canny-blank', True, single_encoder=True, text_prompt=True,
                                        encoder_path=quantized_file, control_image=blank_path, extra=opts)
                    int8_edge, _ = run('int8-canny-edge', True, single_encoder=True, text_prompt=True,
                                       encoder_path=quantized_file, control_image=subject_path, extra=opts)
                    self.assertFalse(torch.equal(int8_blank, int8_edge))
                    stream_edge, _ = run('int8-canny-stream', True, single_encoder=True, text_prompt=True,
                        encoder_path=quantized_file, control_image=subject_path,
                        extra=(*opts, '--moss-transformer-storage', 'cpu-stream'))
                    torch.testing.assert_close(stream_edge, int8_edge, rtol=0, atol=0)
                depth_model = root / 'depth-model'
                save_model(depth_model)
                opts = ('--control-preprocess', 'depth', '--control-depth-model', str(depth_model))
                blank_out, _ = run('fallback-depth-blank', True, single_encoder=True, text_prompt=True,
                                   control_image=blank_path, extra=opts)
                depth_out, depth_report = run('fallback-depth-edge', True, single_encoder=True, text_prompt=True,
                                              control_image=subject_path, extra=opts)
                repeated, _ = run('fallback-depth-repeat', True, single_encoder=True, text_prompt=True,
                                  control_image=subject_path, extra=opts)
                self.assertEqual(depth_report['control_preprocessor']['output'], 'rgb8')
                self.assertFalse(torch.equal(blank_out, depth_out))
                self.assertTrue(torch.equal(depth_out, repeated))
                int8_blank, _ = run('int8-depth-blank', True, single_encoder=True, text_prompt=True,
                                    encoder_path=quantized_file, control_image=blank_path, extra=opts)
                int8_depth, _ = run('int8-depth-edge', True, single_encoder=True, text_prompt=True,
                                    encoder_path=quantized_file, control_image=subject_path, extra=opts)
                self.assertFalse(torch.equal(int8_blank, int8_depth))
                stream_depth, _ = run('int8-depth-stream', True, single_encoder=True, text_prompt=True,
                    encoder_path=quantized_file, control_image=subject_path,
                    extra=(*opts, '--moss-transformer-storage', 'cpu-stream'))
                torch.testing.assert_close(stream_depth, int8_depth, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
