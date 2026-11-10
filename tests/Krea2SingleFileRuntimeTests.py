"""Strict mapping and local loading of original Krea 2 single-file weights."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_single_file as single

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for Krea 2 single-file tests')
class Krea2SingleFileRuntimeTests(unittest.TestCase):
    def test_streamed_fp8_layer_rejects_weight_reload(self):
        import torch
        for assign in (False, True):
            for nested in (False, True):
                with self.subTest(assign=assign, nested=nested):
                    source = torch.nn.Linear(3, 2, bias=False)
                    source.weight = torch.nn.Parameter(source.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
                    layer = single.streamed_linear_type()(source, torch.tensor(0.25))
                    original = layer.weight.detach().view(torch.uint8).clone()
                    scale = layer._iild_fp8_scale.clone()
                    target = torch.nn.Sequential(layer) if nested else layer
                    key = '0.weight' if nested else 'weight'
                    with self.assertRaisesRegex(RuntimeError, 'cannot replace'):
                        target.load_state_dict({key: torch.ones(2, 3)}, strict=False, assign=assign)
                    self.assertTrue(torch.equal(layer.weight.detach().view(torch.uint8), original))
                    self.assertTrue(torch.equal(layer._iild_fp8_scale, scale))
                    target.load_state_dict({}, strict=False, assign=assign)

    def test_streamed_fp8_layer_rejects_incomplete_serialization(self):
        import torch
        layer = torch.nn.Linear(3, 2)
        layer.weight = torch.nn.Parameter(layer.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
        streamed = single.streamed_linear_type()(layer, torch.tensor(0.25))
        with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
            streamed.state_dict()
        parent = torch.nn.Sequential(streamed)
        with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
            parent.state_dict()

    def test_adapter_state_inspection_is_scoped_and_restored_after_failure(self):
        from types import SimpleNamespace
        import resource_fixture
        import torch
        from safetensors.torch import save_file
        def streamed_layer():
            layer = torch.nn.Linear(3, 2)
            layer.weight = torch.nn.Parameter(layer.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
            return single.streamed_linear_type()(layer, torch.tensor(0.25))
        active, unrelated = streamed_layer(), streamed_layer()
        model = torch.nn.Sequential(active)
        original = active.weight
        with resource_fixture.TemporaryDirectory(prefix='krea-peft-state-') as temporary:
            adapter = Path(temporary) / 'style.safetensors'
            save_file({'0.lora_A.weight': torch.ones(1, 3), '0.lora_B.weight': torch.ones(2, 1)}, str(adapter))
            selection = SimpleNamespace(local_file=SimpleNamespace(path=str(adapter)))
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                with single.adapter_loading(model, selection, torch.float32):
                    self.assertEqual(model.state_dict()['0.weight'].dtype, torch.float32)
                    with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
                        unrelated.state_dict()
                    raise RuntimeError('injected failure')
        self.assertIs(active.weight, original)
        with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
            model.state_dict()

    def test_provisioned_adapters_attach_and_restore_without_inference(self):
        if os.environ.get('IILD_VERIFY_KREA2_CONTROL_HOOKS') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_CONTROL_HOOKS=1 to attach the actual Krea adapters')
        from types import SimpleNamespace
        from unittest.mock import patch
        import krea2_controlnet
        import krea2_ostris_lora
        import torch
        from weight_files import resolve_weight_file
        torch.set_num_threads(2)
        model = single.load(ROOT / 'resources/models/krea2_turbo_fp8_scaled.safetensors',
                            ROOT / 'resources/model-configs/krea2-turbo/transformer',
                            torch.bfloat16, storage='cpu-stream')
        pipeline = SimpleNamespace(transformer=model)
        modules = dict(model.named_modules())
        def hooks():
            return {name: len(module._forward_hooks) for name, module in modules.items()}
        original = hooks()
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            depth = resolve_weight_file(str(ROOT / 'resources/models/krea2-depth-control-lora.safetensors'), 'depth')
            pose = resolve_weight_file(str(ROOT / 'resources/models/krea2-turbo-openpose-controlnet.safetensors'), 'pose')
            selection = SimpleNamespace(weight=depth, scale=1.0)
            # A synthetic latent bypasses the VAE; adapter weights and hooks are real.
            with patch.object(krea2_controlnet, '_control_latents', return_value=torch.zeros(1, 1, 64)):
                with self.assertRaisesRegex(RuntimeError, 'injected exit'):
                    with krea2_controlnet.execution(pipeline, object(), {'width': 16, 'height': 16},
                                                    selection, torch, None):
                        active = hooks()
                        self.assertEqual(sum(active[name] - original[name] for name in active), 225)
                        raise RuntimeError('injected exit')
            self.assertEqual(hooks(), original)
            with self.assertRaisesRegex(RuntimeError, 'injected exit'):
                with krea2_ostris_lora.execution(model, pose, 1.0, torch) as report:
                    self.assertEqual(report['targets'], 256)
                    active = hooks()
                    self.assertEqual(sum(active[name] - original[name] for name in active), 256)
                    raise RuntimeError('injected exit')
            self.assertEqual(hooks(), original)

    def test_provisioned_pose_pipeline_loads_without_inference(self):
        if os.environ.get('IILD_VERIFY_KREA2_POSE_PIPELINE_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_POSE_PIPELINE_LOAD=1 to load the real pose pipeline')
        from unittest.mock import patch
        import diffusers
        import generate_any
        import krea2_ostris_lora
        import resource_fixture
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='krea-real-pose-pipeline-') as temporary:
            folder = Path(temporary)
            image = folder / 'pose.png'
            image.write_bytes(b'preflight-placeholder')
            pose = ROOT / 'resources/models/krea2-turbo-openpose-controlnet.safetensors'
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                parsed = generate_any.build_parser().parse_args([
                    '--model', str(ROOT / 'resources/models/krea2_turbo_fp8_scaled.safetensors'),
                    '--model-config-preset', 'krea2-turbo', '--krea2-variant', 'turbo',
                    '--krea2-fp8-storage', 'cpu-stream', '--controlnet-format', 'krea2-ostris-edit',
                    '--controlnet', str(pose), '--control-image', str(image),
                    '--prompt', 'a dancer in a sunlit studio', '--width', '1024', '--height', '1024',
                    '--steps', '10', '--guidance-scale', '0', '--seed', '42', '--dtype', 'bfloat16',
                    '--offload', 'none', '--no-default-modifiers', '--cache-dir', str(folder / 'cache'),
                    '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(parsed)
                pipeline, report = generate_any.load_pipeline(selected, diffusers, torch.bfloat16)
            self.assertEqual(type(pipeline).__name__, 'Krea2Pipeline')
            self.assertEqual(type(pipeline.text_encoder).__name__, 'Qwen3VLModel')
            self.assertTrue(all(not value.is_meta for value in pipeline.text_encoder.parameters()))
            self.assertEqual(report['single_file_loading']['runtime_weights'], 'cpu-stream-fp8')
            modules = {name: tuple(module.weight.shape) for name, module in pipeline.transformer.named_modules()
                       if isinstance(module, torch.nn.Linear)}
            with safe_open(str(pose), framework='pt', device='cpu') as source:
                shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
                self.assertEqual(len(krea2_ostris_lora.plan_weights(shapes, modules)), 256)
            self.assertFalse((folder / 'unused-output').exists())

    def test_provisioned_depth_pipeline_and_pose_targets_load_without_inference(self):
        if os.environ.get('IILD_VERIFY_KREA2_PIPELINE_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_PIPELINE_LOAD=1 to load the real local Krea components')
        from unittest.mock import patch
        import diffusers
        import generate_any
        import krea2_controlnet
        import krea2_ostris_lora
        import resource_fixture
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='krea-real-pipeline-load-') as temporary:
            folder = Path(temporary)
            image = folder / 'control.png'
            # Selection inspects file identity only; this test does not decode it.
            image.write_bytes(b'preflight-placeholder')
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                parsed = generate_any.build_parser().parse_args([
                    '--model', str(ROOT / 'resources/models/krea2_turbo_fp8_scaled.safetensors'),
                    '--model-config-preset', 'krea2-turbo', '--krea2-variant', 'turbo',
                    '--krea2-fp8-storage', 'cpu-stream',
                    '--controlnet', str(ROOT / 'resources/models/krea2-depth-control-lora.safetensors'),
                    '--control-image', str(image), '--control-preprocess', 'depth',
                    '--control-depth-model', str(ROOT / 'resources/depth-estimators/depth-anything-v2-large'),
                    '--prompt', 'a sunlit studio', '--width', '1024', '--height', '1024',
                    '--steps', '8', '--guidance-scale', '0', '--seed', '42', '--dtype', 'bfloat16',
                    '--offload', 'none', '--no-default-modifiers', '--cache-dir', str(folder / 'cache'),
                    '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(parsed)
                pipeline, report = generate_any.load_pipeline(selected, diffusers, torch.bfloat16)
            self.assertEqual(type(pipeline).__name__, 'Krea2Pipeline')
            self.assertEqual(report['single_file_loading']['runtime_weights'], 'cpu-stream-fp8')
            self.assertEqual(type(pipeline.vae).__name__, 'AutoencoderKLQwenImage')
            modules = {name: tuple(module.weight.shape) for name, module in pipeline.transformer.named_modules()
                       if isinstance(module, torch.nn.Linear)}
            depth = ROOT / 'resources/models/krea2-depth-control-lora.safetensors'
            with safe_open(str(depth), framework='pt', device='cpu') as source:
                shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
                plan = krea2_controlnet.plan_weights(shapes, modules,
                    projection_shape=tuple(pipeline.transformer.img_in.weight.shape),
                    block_count=len(pipeline.transformer.transformer_blocks))
            self.assertEqual(len(plan), 224)
            pose = ROOT / 'resources/models/krea2-turbo-openpose-controlnet.safetensors'
            with safe_open(str(pose), framework='pt', device='cpu') as source:
                shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
                self.assertEqual(len(krea2_ostris_lora.plan_weights(shapes, modules)), 256)
            self.assertFalse((folder / 'unused-output').exists())

    def test_provisioned_turbo_stream_loads_actual_transformer_weights(self):
        if os.environ.get('IILD_VERIFY_KREA2_STREAM_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_STREAM_LOAD=1 to load the real FP8 transformer')
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        path = ROOT / 'resources/models/krea2_turbo_fp8_scaled.safetensors'
        config = ROOT / 'resources/model-configs/krea2-turbo/transformer'
        model = single.load(path, config, torch.bfloat16, storage='cpu-stream')
        report = model._iild_single_file_loading
        self.assertEqual(report['runtime_weights'], 'cpu-stream-fp8')
        self.assertEqual(report['quantized_layers'], 256)
        self.assertEqual(report['checkpoint_full_precision_matmul_required_layers'], 96)
        self.assertEqual(report['checkpoint_full_precision_matmul_not_required_layers'], 160)
        self.assertGreater(report['quantized_source_bytes'], 0)
        self.assertTrue(all(not value.is_meta for value in model.parameters()))
        with safe_open(str(path), framework='pt', device='cpu') as source:
            shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
            metadata = source.metadata()
            quantized = single.quantized_layers(shapes, metadata)
            # The source-to-target mapping is validated before any comparison.
            from diffusers import Krea2Transformer2DModel
            with torch.device('meta'):
                template = Krea2Transformer2DModel.from_config(json.loads((config / 'config.json').read_text()))
            expected = {name: tuple(value.shape) for name, value in template.state_dict().items()}
            mapping = single.plan(shapes, expected, metadata=metadata)
            for source_name, scale_name in quantized.items():
                target, _ = mapping[source_name]
                with self.subTest(source=source_name, target=target):
                    layer = model.get_submodule(target.removesuffix('.weight'))
                    self.assertIsInstance(layer, single.streamed_linear_type())
                    self.assertEqual(layer.weight.device.type, 'cpu')
                    self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                    stored = source.get_slice(source_name)
                    self.assertEqual(tuple(layer.weight.shape), tuple(stored.get_shape()))
                    # Bound peak memory while checking every original FP8 bit,
                    # including representations that a float comparison could hide.
                    for start in range(0, shapes[source_name][0], 512):
                        end = min(start + 512, shapes[source_name][0])
                        self.assertTrue(torch.equal(layer.weight[start:end].view(torch.uint8),
                                                    stored[start:end].view(torch.uint8)),
                                        f'{source_name}: rows {start}:{end}')
                    torch.testing.assert_close(layer._iild_fp8_scale, source.get_tensor(scale_name), rtol=0, atol=0)
            # Quantized layers above are only part of the complete transformer.
            # Compare all remaining source tensors, including reshaped block
            # modulation, in the precision selected by the loader.
            remaining = {name: value for name, value in mapping.items() if name not in quantized}
            self.assertEqual(len(remaining), 174)
            for source_name, (target, shape) in remaining.items():
                with self.subTest(source=source_name, target=target):
                    actual = model.get_parameter(target).detach()
                    expected_value = source.get_tensor(source_name)
                    if shape is not None:
                        expected_value = expected_value.reshape(shape)
                    expected_value = expected_value.to(actual.dtype)
                    self.assertEqual(tuple(actual.shape), tuple(expected_value.shape))
                    self.assertTrue(torch.equal(actual.view(torch.uint8), expected_value.view(torch.uint8)), target)

    def test_published_base_header_matches_real_meta_transformer(self):
        import torch
        from diffusers import Krea2Transformer2DModel
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-turbo-base-header.json').read_text())
        with torch.device('meta'):
            model = Krea2Transformer2DModel()
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        shapes = {name: tuple(value['shape']) for name, value in fixture['header'].items()}
        plan = single.plan(shapes, expected)
        self.assertEqual(len(plan), 430)
        self.assertEqual({target for target, _ in plan.values()}, set(expected))

    def test_published_scaled_fp8_header_matches_real_meta_transformer(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-turbo-fp8-header.json').read_text())
        header = fixture['header']
        with torch.device('meta'):
            model = Krea2Transformer2DModel()
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for variant in ('raw', 'turbo'):
            folder = ROOT / 'resources/model-configs' / ('krea2-' + variant)
            config = json.loads((folder / 'transformer/config.json').read_text())
            with torch.device('meta'):
                configured = Krea2Transformer2DModel.from_config(config)
            self.assertEqual({name: tuple(value.shape) for name, value in configured.state_dict().items()}, expected)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(folder / 'scheduler', local_files_only=True)
            self.assertTrue(scheduler.config.use_dynamic_shifting)
            self.assertEqual(scheduler.config.max_image_seq_len, 6400)
        shapes = {name: tuple(value['shape']) for name, value in header.items() if name != '__metadata__'}
        plan = single.plan(shapes, expected, metadata=header['__metadata__'])
        policy = {}
        quantized = single.quantized_layers(shapes, header['__metadata__'], matmul_policy=policy)
        self.assertEqual(len(plan), 430)
        self.assertEqual(len(quantized), 256)
        self.assertEqual(policy, {'required': 96, 'not_required': 160})
        for weight, scale in quantized.items():
            self.assertEqual(header[weight]['dtype'], 'F8_E4M3')
            self.assertEqual(header[scale]['dtype'], 'F32')
            self.assertTrue(plan[weight][0].startswith(('transformer_blocks.', 'text_fusion.')))
        self.assertEqual({target for target, _ in plan.values()}, set(expected))

    def test_small_original_checkpoint_loads_and_rejects_incompatible_files(self):
        import torch
        import diffusers as d
        from diffusers import Krea2Transformer2DModel
        from safetensors.torch import save_file
        import resource_fixture
        torch.set_num_threads(2)
        torch.manual_seed(43)
        model = Krea2Transformer2DModel(in_channels=64, num_layers=1, attention_head_dim=16,
            num_attention_heads=2, num_key_value_heads=2, intermediate_size=64, timestep_embed_dim=32,
            text_hidden_dim=32, num_text_layers=1, text_num_attention_heads=2, text_num_key_value_heads=2,
            text_intermediate_size=64, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
            axes_dims_rope=(4, 6, 6)).eval()
        # Reverse the public original-to-Diffusers contract independently to
        # produce a small, complete original-format checkpoint.
        prefix = {'img_in.': 'first.', 'time_embed.linear_1.': 'tmlp.0.',
            'time_embed.linear_2.': 'tmlp.2.', 'time_mod_proj.': 'tproj.1.',
            'txt_in.norm.weight': 'txtmlp.0.scale', 'txt_in.linear_1.': 'txtmlp.1.',
            'txt_in.linear_2.': 'txtmlp.3.', 'text_fusion.': 'txtfusion.',
            'transformer_blocks.': 'blocks.', 'final_layer.linear.': 'last.linear.',
            'final_layer.norm.weight': 'last.norm.scale', 'final_layer.scale_shift_table': 'last.modulation.lin'}
        block = {'.attn.to_q.': '.attn.wq.', '.attn.to_k.': '.attn.wk.',
            '.attn.to_v.': '.attn.wv.', '.attn.to_out.0.': '.attn.wo.', '.attn.to_gate.': '.attn.gate.',
            '.attn.norm_q.weight': '.attn.qknorm.qnorm.scale',
            '.attn.norm_k.weight': '.attn.qknorm.knorm.scale', '.ff.': '.mlp.',
            '.norm1.weight': '.prenorm.scale', '.norm2.weight': '.postnorm.scale',
            '.scale_shift_table': '.mod.lin'}
        weights = {}
        for name, value in model.state_dict().items():
            target = name
            for new, old in block.items():
                if name == 'final_layer.scale_shift_table' and new == '.scale_shift_table':
                    continue
                target = target.replace(new, old)
            for new, old in prefix.items():
                if target.startswith(new):
                    target = old + target[len(new):]
                    break
            value = value.detach().clone()
            if name.startswith('transformer_blocks.') and name.endswith('.scale_shift_table'):
                value = value.reshape(-1)
            weights[target] = value
        with resource_fixture.TemporaryDirectory(prefix='krea-single-') as folder:
            folder = Path(folder)
            model.save_config(folder / 'transformer')
            path = folder / 'base.safetensors'
            save_file(weights, str(path))
            loaded = single.load(path, folder / 'transformer', torch.float32)
            with self.assertRaisesRegex(ValueError, 'requires a scaled FP8 checkpoint'):
                single.load(path, folder / 'transformer', torch.float32, storage='cpu-stream')
            for name, expected in model.state_dict().items():
                self.assertTrue(torch.equal(expected, loaded.state_dict()[name]), name)
            fp8_path = folder / 'scaled-fp8.safetensors'
            fp8_key = 'blocks.0.attn.wq.weight'
            fp8_layer = fp8_key.removesuffix('.weight')
            scale_key = fp8_layer + '.weight_scale'
            second_key = 'blocks.0.mlp.up.weight'
            second_layer = second_key.removesuffix('.weight')
            second_scale = second_layer + '.weight_scale'
            third_key = 'txtfusion.layerwise_blocks.0.attn.wk.weight'
            third_layer = third_key.removesuffix('.weight')
            third_scale = third_layer + '.weight_scale'
            quantized = dict(weights)
            quantized[fp8_key] = (weights[fp8_key] / .125).clamp(-448, 448).to(torch.float8_e4m3fn)
            quantized[scale_key] = torch.tensor(.125, dtype=torch.float32)
            quantized[second_key] = (weights[second_key] / .0625).clamp(-448, 448).to(torch.float8_e4m3fn)
            quantized[second_scale] = torch.tensor(.0625, dtype=torch.float32)
            quantized[third_key] = (weights[third_key] / .03125).clamp(-448, 448).to(torch.float8_e4m3fn)
            quantized[third_scale] = torch.tensor(.03125, dtype=torch.float32)
            metadata = {'_quantization_metadata': json.dumps({'layers': {fp8_layer:
                        {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True},
                        second_layer: {'format': 'float8_e4m3fn'}, third_layer: {'format': 'float8_e4m3fn'}}})}
            save_file(quantized, str(fp8_path), metadata=metadata)
            fp8_valid_path = folder / 'scaled-fp8-valid.safetensors'
            save_file(quantized, str(fp8_valid_path), metadata=metadata)
            dense_equivalent = dict(weights)
            dense_equivalent[fp8_key] = quantized[fp8_key].float() * quantized[scale_key]
            dense_equivalent[second_key] = quantized[second_key].float() * quantized[second_scale]
            dense_equivalent[third_key] = quantized[third_key].float() * quantized[third_scale]
            dense_equivalent_path = folder / 'dense-equivalent.safetensors'
            save_file(dense_equivalent, str(dense_equivalent_path))
            for dtype in (torch.float32, torch.bfloat16):
                actual = single.load(fp8_path, folder / 'transformer', dtype)
                self.assertEqual(actual._iild_single_file_loading['checkpoint_full_precision_matmul_required_layers'], 1)
                self.assertEqual(actual._iild_single_file_loading['checkpoint_full_precision_matmul_not_required_layers'], 2)
                self.assertEqual(actual._iild_single_file_loading['quantized_matmul_execution'], 'dequantized-runtime-dtype')
                expected = quantized[fp8_key].to(dtype) * quantized[scale_key].to(dtype)
                self.assertTrue(torch.equal(actual.state_dict()['transformer_blocks.0.attn.to_q.weight'], expected))
                expected = quantized[second_key].to(dtype) * quantized[second_scale].to(dtype)
                self.assertTrue(torch.equal(actual.state_dict()['transformer_blocks.0.ff.up.weight'], expected))
                expected = quantized[third_key].to(dtype) * quantized[third_scale].to(dtype)
                self.assertTrue(torch.equal(actual.state_dict()['text_fusion.layerwise_blocks.0.attn.to_k.weight'], expected))
                streamed = single.load(fp8_path, folder / 'transformer', dtype, storage='cpu-stream')
                self.assertEqual(streamed._iild_single_file_loading['checkpoint_full_precision_matmul_required_layers'], 1)
                self.assertEqual(streamed._iild_single_file_loading['checkpoint_full_precision_matmul_not_required_layers'], 2)
                self.assertEqual(streamed._iild_single_file_loading['quantized_matmul_execution'], 'dequantized-runtime-dtype')
                layer = streamed.get_submodule('transformer_blocks.0.attn.to_q')
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(layer.weight.device.type, 'cpu')
                sample = torch.randn(1, 3, layer.in_features, dtype=dtype)
                self.assertTrue(torch.equal(layer(sample), actual.get_submodule('transformer_blocks.0.attn.to_q')(sample)))
                streamed.to(dtype=dtype)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                if torch.backends.mps.is_available():
                    layer.to('mps')
                    mps_input = sample.to('mps')
                    reference_layer = actual.get_submodule('transformer_blocks.0.attn.to_q').to('mps')
                    torch.testing.assert_close(layer(mps_input), reference_layer(mps_input), rtol=0, atol=0)
                    self.assertEqual(layer.weight.device.type, 'cpu')
            # AI Toolkit control LoRA also needs compute-device residuals when
            # the target base layer retains FP8 weights on the CPU.
            import krea2_ostris_lora
            from weight_files import resolve_weight_file
            edit_path = folder / 'edit-control.safetensors'
            edit_prefix = 'diffusion_model.blocks.0.attn.wq'
            save_file({edit_prefix + '.lora_A.weight': torch.full((2, 32), .1),
                       edit_prefix + '.lora_B.weight': torch.full((32, 2), .1)}, str(edit_path))
            edit_selection = resolve_weight_file(str(edit_path), 'edit-control')
            for device in (('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',)):
                dense_model = single.load(fp8_valid_path, folder / 'transformer', torch.float32).to(device)
                streamed_model = single.load(fp8_valid_path, folder / 'transformer', torch.float32,
                                             storage='cpu-stream').to(device)
                dense_layer = dense_model.get_submodule('transformer_blocks.0.attn.to_q')
                stream_layer = streamed_model.get_submodule('transformer_blocks.0.attn.to_q')
                values = torch.randn(1, 2, 32, device=device)
                with torch.no_grad(), krea2_ostris_lora.execution(dense_model, edit_selection, 1., torch):
                    dense_result = dense_layer(values)
                with torch.no_grad(), krea2_ostris_lora.execution(streamed_model, edit_selection, 1., torch):
                    stream_result = stream_layer(values)
                torch.testing.assert_close(stream_result, dense_result, rtol=0, atol=0)
            without_scale = dict(quantized)
            without_scale.pop(scale_key)
            save_file(without_scale, str(fp8_path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'scale'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            bad_scale = dict(quantized)
            bad_scale[scale_key] = torch.tensor(float('nan'))
            save_file(bad_scale, str(fp8_path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'scale'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            bad_scale[scale_key] = torch.tensor(-.125)
            save_file(bad_scale, str(fp8_path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'scale'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            bad_type = dict(quantized)
            bad_type[fp8_key] = weights[fp8_key]
            save_file(bad_type, str(fp8_path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            bad_metadata = {'_quantization_metadata': json.dumps({'layers': {fp8_layer: {'format': 'nvfp4'}}})}
            save_file(quantized, str(fp8_path), metadata=bad_metadata)
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            save_file(quantized, str(fp8_path))
            with self.assertRaisesRegex(ValueError, 'metadata|scale'):
                single.load(fp8_path, folder / 'transformer', torch.float32)
            unsupported = dict(weights)
            unsupported['first.weight'] = weights['first.weight'].to(torch.float8_e4m3fn)
            unsupported['first.weight_scale'] = torch.tensor(1.)
            save_file(unsupported, str(fp8_path), metadata={'_quantization_metadata': json.dumps(
                {'layers': {'first': {'format': 'float8_e4m3fn'}}})})
            with self.assertRaisesRegex(ValueError, 'does not support this layer'):
                single.load(fp8_path, folder / 'transformer', torch.float32, storage='cpu-stream')
            for prefix_text in ('', 'model.diffusion_model.'):
                self.assertEqual(len(single.plan({prefix_text + key: tuple(value.shape) for key, value in weights.items()},
                    {key: tuple(value.shape) for key, value in model.state_dict().items()})), len(weights))
            shapes = {key: tuple(value.shape) for key, value in weights.items()}
            expected_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
            duplicate = next(iter(shapes))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                single.plan({**shapes, 'model.diffusion_model.' + duplicate: shapes[duplicate]}, expected_shapes)
            with self.assertRaisesRegex(ValueError, 'unexpected tensor or shape'):
                single.plan({**shapes, 'unrecognized.scale': (1,)}, expected_shapes)
            # The generic runner must inject the original checkpoint into a
            # config-only transformer and execute its real control path.
            import contextlib
            import io
            import generate_any
            import krea2_controlnet
            from PIL import Image
            from safetensors.torch import load_file
            pipe = d.Krea2Pipeline(transformer=model,
                vae=d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1),
                scheduler=d.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=.5,
                    max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400),
                text_encoder=None, tokenizer=None, text_encoder_select_layers=[0], is_distilled=True)
            package = folder / 'package'
            pipe.save_pretrained(str(package), safe_serialization=True)
            (package / 'transformer/diffusion_pytorch_model.safetensors').rename(folder / 'saved-transformer.safetensors')
            embed_file = folder / 'embeds.safetensors'
            save_file({'prompt': torch.randn(1, 4, 1, 32), 'mask': torch.ones(1, 4, dtype=torch.bool)}, str(embed_file))
            def descriptor(key, semantic, layout):
                return {'tensor_path': str(embed_file), 'key': key, 'semantic': semantic,
                        'layout': layout, 'representation_space': 'test-krea-text'}
            inputs = {'prompt_embeds': descriptor('prompt', 'embeddings', 'BSLC'),
                      'prompt_embeds_mask': descriptor('mask', 'attention-mask', 'BS'), 'output_type': 'latent'}
            modules = dict(model.named_modules())
            control_weights = {'first.weight': torch.cat((model.img_in.weight.detach().clone(),
                               torch.ones_like(model.img_in.weight) * .05), dim=1),
                               'first.bias': model.img_in.bias.detach().clone()}
            for original, target in krea2_controlnet.BLOCK_TARGETS.items():
                weight = modules['transformer_blocks.0.' + target].weight
                control_weights['blocks.0.' + original + '.A'] = torch.ones(2, weight.shape[1]) * .005
                control_weights['blocks.0.' + original + '.B'] = torch.ones(weight.shape[0], 2) * .005
            adapter, hint = folder / 'control.safetensors', folder / 'hint.png'
            save_file(control_weights, str(adapter))
            Image.new('RGB', (64, 64), (128, 128, 128)).save(hint)
            def run(name, image, extra=(), model_file=None, device='cpu', control_file=None, preset_resources=None):
                output = folder / name
                config_args = (['--model-config', str(package)] if preset_resources is None else
                               ['--model-config-preset', 'krea2-turbo', '--generation-resources', str(preset_resources)])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(generate_any.main(['--model', str(model_file or path), *config_args,
                        '--controlnet', str(control_file or adapter), '--control-image', str(image), *extra,
                        '--seed', '7', '--device', device,
                        '--width', '64', '--height', '64', '--steps', '2', '--pipeline-inputs', json.dumps(inputs),
                        '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BSC',
                            'representation_space': 'test-krea-packed'}}), '--no-default-modifiers',
                        '--cache-dir', str(folder / 'cache'), '--output-dir', str(output)]), 0)
                report = json.loads((output / 'generation.json').read_text())
                self.assertEqual(report['request']['source_kind'], 'single-file')
                if control_file is None:
                    self.assertEqual(report['control_execution']['blocks'], 1)
                else:
                    self.assertEqual(report['control_execution']['adapter'], 'krea2-color-depth-control')
                return load_file(str(output / 'images.safetensors'))['images'], report
            first, _ = run('first', hint)
            repeat, _ = run('repeat', hint)
            # Exercise the bundled index through the actual generic runner;
            # only the tapped-layer list changes for this one-layer fixture.
            model_index = package / 'model_index.json'
            saved_index = model_index.read_text()
            bundled_index = json.loads((ROOT / 'resources/model-configs/krea2-turbo/model_index.json').read_text())
            bundled_index['text_encoder_select_layers'] = [0]
            try:
                model_index.write_text(json.dumps(bundled_index))
                bundled_output, _ = run('bundled-index', hint)
                torch.testing.assert_close(bundled_output, first, rtol=0, atol=0)
                # Adapt only the fixture dimensions and pins; preserve the real
                # preset selection, package loading, and control execution path.
                import hashlib
                import shutil
                from unittest.mock import patch
                import model_config_defaults
                preset_resources = folder / 'preset-resources'
                preset_folder = preset_resources / 'model-configs/krea2-turbo'
                shutil.copytree(package, preset_folder)
                (preset_resources / 'generation-defaults.json').write_text(json.dumps({'version': 1}))
                pinned = {}
                for relative in ('model_index.json', 'scheduler/scheduler_config.json', 'transformer/config.json'):
                    data = (preset_folder / relative).read_bytes()
                    pinned[relative] = (len(data), hashlib.sha256(data).hexdigest())
                with patch.dict(model_config_defaults.FILES, {'krea2-turbo': pinned}):
                    preset_output, preset_report = run('bundled-preset', hint, preset_resources=preset_resources)
                torch.testing.assert_close(preset_output, first, rtol=0, atol=0)
                self.assertEqual(preset_report['request']['model_config_status'], 'bundled')
            finally:
                model_index.write_text(saved_index)
            import krea2_contract
            from types import SimpleNamespace
            from weight_files import resolve_weight_file
            embed_data = load_file(str(embed_file))
            direct_inputs = {'prompt_embeds': embed_data['prompt'], 'prompt_embeds_mask': embed_data['mask'],
                             'output_type': 'latent', 'height': 64, 'width': 64, 'num_inference_steps': 2,
                             'guidance_scale': 0., 'generator': torch.Generator('cpu').manual_seed(7)}
            contract = krea2_contract.resolve(SimpleNamespace(inputs=direct_inputs, dtype='float32',
                krea2_variant='turbo', krea2_mu=None), {'_class_name': 'Krea2Pipeline', 'is_distilled': True})
            control_selection = krea2_controlnet.Selection(resolve_weight_file(str(adapter), 'control'), str(hint), 1.)
            pipe.set_progress_bar_config(disable=True)
            import numpy as np
            with torch.no_grad(), krea2_contract.execution(pipe, direct_inputs, contract, torch), \
                 krea2_controlnet.execution(pipe, Image.open(hint).convert('RGB'), direct_inputs,
                                            control_selection, torch, np):
                direct = pipe(**direct_inputs).images
            self.assertTrue(torch.equal(first, direct))
            other = folder / 'other.png'
            Image.new('RGB', (64, 64), (230, 230, 230)).save(other)
            changed, _ = run('changed', other)
            self.assertTrue(torch.equal(first, repeat))
            self.assertFalse(torch.equal(first, changed))
            scaled, scaled_report = run('scaled-control', hint, model_file=fp8_valid_path)
            expanded, dense_report = run('expanded-control', hint, model_file=dense_equivalent_path)
            self.assertTrue(torch.equal(scaled, expanded))
            self.assertEqual(scaled_report['model']['single_file_loading']['format'], 'comfy-scalar-fp8-e4m3fn')
            self.assertEqual(scaled_report['model']['single_file_loading']['quantized_layers'], 3)
            self.assertEqual(scaled_report['model']['single_file_loading']['runtime_weights'], 'dense')
            self.assertFalse(scaled_report['model']['single_file_loading']['native_fp8_matmul'])
            self.assertEqual(scaled_report['model']['single_file_loading']['checkpoint_full_precision_matmul_required_layers'], 1)
            self.assertEqual(scaled_report['model']['single_file_loading']['checkpoint_full_precision_matmul_not_required_layers'], 2)
            self.assertEqual(scaled_report['model']['single_file_loading']['quantized_matmul_execution'], 'dequantized-runtime-dtype')
            self.assertEqual(dense_report['model']['single_file_loading']['format'], 'original-dense')
            streamed, stream_report = run('stream-control', hint,
                extra=('--krea2-fp8-storage', 'cpu-stream'), model_file=fp8_valid_path)
            self.assertTrue(torch.equal(streamed, scaled))
            self.assertEqual(stream_report['model']['single_file_loading']['runtime_weights'], 'cpu-stream-fp8')
            self.assertEqual(stream_report['model']['single_file_loading']['quantized_layers'], 3)
            self.assertEqual(stream_report['model']['single_file_loading']['quantized_source_bytes'], 4096)
            self.assertEqual(stream_report['model']['single_file_loading']['largest_expanded_layer_bytes'], 8192)
            if torch.backends.mps.is_available():
                mps_dense, _ = run('mps-dense', hint, model_file=fp8_valid_path, device='mps')
                mps_stream, _ = run('mps-stream', hint,
                    extra=('--krea2-fp8-storage', 'cpu-stream'), model_file=fp8_valid_path, device='mps')
                torch.testing.assert_close(mps_stream, mps_dense, rtol=0, atol=0)
            color_weights = {'img_in.weight': torch.cat((model.img_in.weight.detach().clone(),
                             torch.ones_like(model.img_in.weight) * .05,
                             torch.ones_like(model.img_in.weight) * .04), dim=1),
                             'img_in.bias': model.img_in.bias.detach().clone()}
            for name, layer in model.named_modules():
                if name != 'img_in' and isinstance(layer, torch.nn.Linear):
                    color_weights[name + '.lora_A.weight'] = torch.ones(2, layer.weight.shape[1]) * .005
                    color_weights[name + '.lora_B.weight'] = torch.ones(layer.weight.shape[0], 2) * .005
            color_adapter = folder / 'color-control.safetensors'
            save_file(color_weights, str(color_adapter))
            color_opts = ('--controlnet-format', 'krea2-color-depth', '--control-color-image', str(hint))
            for device in (('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',)):
                dense_color, _ = run(f'color-dense-{device}', hint, extra=color_opts,
                                     model_file=fp8_valid_path, device=device, control_file=color_adapter)
                stream_color, _ = run(f'color-stream-{device}', hint,
                    extra=(*color_opts, '--krea2-fp8-storage', 'cpu-stream'), model_file=fp8_valid_path,
                    device=device, control_file=color_adapter)
                torch.testing.assert_close(stream_color, dense_color, rtol=0, atol=0)
            if importlib.util.find_spec('peft'):
                style_path = folder / 'style.safetensors'
                style_key = 'transformer.transformer_blocks.0.attn.to_q'
                save_file({style_key + '.lora_A.weight': torch.full((2, 32), .1),
                           style_key + '.lora_B.weight': torch.full((32, 2), .1)}, str(style_path))
                style = ('--lora', str(style_path))
                styled_dense, _ = run('styled-dense', hint, extra=style, model_file=fp8_valid_path)
                styled_stream, _ = run('styled-stream', hint,
                    extra=(*style, '--krea2-fp8-storage', 'cpu-stream'), model_file=fp8_valid_path)
                self.assertTrue(torch.equal(styled_stream, styled_dense))
                self.assertFalse(torch.equal(styled_stream, streamed))
                if torch.backends.mps.is_available():
                    styled_mps_dense, _ = run('styled-mps-dense', hint, extra=style,
                                              model_file=fp8_valid_path, device='mps')
                    styled_mps_stream, _ = run('styled-mps-stream', hint,
                        extra=(*style, '--krea2-fp8-storage', 'cpu-stream'), model_file=fp8_valid_path, device='mps')
                    torch.testing.assert_close(styled_mps_stream, styled_mps_dense, rtol=0, atol=0)
                from lora import resolve_lora_selection
                style_selection = resolve_lora_selection(SimpleNamespace(lora=str(style_path), lora_revision=None,
                    lora_weight_name=None, lora_scale=1.))
                stream_model = single.load(fp8_valid_path, folder / 'transformer', torch.float32, storage='cpu-stream')
                layer = stream_model.get_submodule('transformer_blocks.0.attn.to_q')
                original = layer.weight
                with self.assertRaisesRegex(RuntimeError, 'forced failure'):
                    with single.adapter_loading(stream_model, style_selection, torch.float32):
                        self.assertEqual(layer.weight.dtype, torch.float32)
                        self.assertEqual(layer.weight.untyped_storage().nbytes(), 4)
                        raise RuntimeError('forced failure')
                self.assertIs(layer.weight, original)
                bad_style = folder / 'unsupported-style.safetensors'
                save_file({style_key + '.lora_magnitude_vector': torch.ones(32)}, str(bad_style))
                bad_selection = resolve_lora_selection(SimpleNamespace(lora=str(bad_style), lora_revision=None,
                    lora_weight_name=None, lora_scale=1.))
                with self.assertRaisesRegex(ValueError, 'ordinary LoRA'):
                    with single.adapter_loading(stream_model, bad_selection, torch.float32):
                        pass
                self.assertIs(layer.weight, original)
            bundled_vae = ROOT / 'resources/vae/qwen-image/vae/diffusion_pytorch_model.safetensors'
            if bundled_vae.is_file():
                (package / 'vae').rename(folder / 'tiny-vae-backup')
                fallback, fallback_report = run('fallback', hint)
                fallback_repeat, _ = run('fallback-repeat', hint)
                self.assertTrue(torch.equal(fallback, fallback_repeat))
                self.assertEqual(fallback_report['request']['vae']['status'], 'fallback')
                self.assertEqual(Path(fallback_report['request']['vae']['selection']['weight']['path']), bundled_vae)
                from PIL import ImageDraw
                import native_control_preprocessing
                from depth_hint_fixture import save_model
                blank = Image.new('RGB', (64, 64), 'black')
                subject = blank.copy()
                ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
                blank_path, subject_path = folder / 'blank.png', folder / 'subject.png'
                blank.save(blank_path)
                subject.save(subject_path)
                try:
                    native_library = native_control_preprocessing.library_path()
                except ValueError:
                    native_library = None
                if native_library:
                    opts = ('--control-preprocess', 'canny', '--control-preprocessor-library', str(native_library))
                    blank_result, _ = run('fallback-canny-blank', blank_path, opts)
                    subject_result, trace = run('fallback-canny-subject', subject_path, opts)
                    repeated, _ = run('fallback-canny-repeat', subject_path, opts)
                    self.assertEqual(trace['control_preprocessor']['name'], 'native-canny')
                    self.assertFalse(torch.equal(blank_result, subject_result))
                    self.assertTrue(torch.equal(subject_result, repeated))
                depth_model = folder / 'depth-model'
                save_model(depth_model)
                opts = ('--control-preprocess', 'depth', '--control-depth-model', str(depth_model))
                blank_result, _ = run('fallback-depth-blank', blank_path, opts)
                subject_result, trace = run('fallback-depth-subject', subject_path, opts)
                repeated, _ = run('fallback-depth-repeat', subject_path, opts)
                self.assertEqual(trace['control_preprocessor']['output'], 'relative-depth-float32')
                self.assertFalse(torch.equal(blank_result, subject_result))
                self.assertTrue(torch.equal(subject_result, repeated))
            broken = dict(weights)
            broken.pop(next(iter(broken)))
            save_file(broken, str(path))
            with self.assertRaisesRegex(ValueError, 'missing|unexpected'):
                single.load(path, folder / 'transformer', torch.float32)
            broken = dict(weights)
            key = next(iter(broken))
            broken[key] = torch.full_like(broken[key], float('nan'))
            save_file(broken, str(path))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                single.load(path, folder / 'transformer', torch.float32)
            broken[key] = torch.ones_like(weights[key], dtype=torch.int8)
            save_file(broken, str(path))
            with self.assertRaisesRegex(ValueError, 'unquantized'):
                single.load(path, folder / 'transformer', torch.float32)


if __name__ == '__main__':
    unittest.main()
