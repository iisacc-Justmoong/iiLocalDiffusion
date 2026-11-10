"""Contracts for embedded and standalone public FLUX.1 single files."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install the Diffusers reference runtime')
class Flux1SingleFileRuntimeTests(unittest.TestCase):
    def test_streamed_layers_reject_weight_reload(self):
        import flux_t5_stream
        import streamed_linear
        import torch
        for kind in ('transformer', 't5'):
            for assign in (False, True):
                for nested in (False, True):
                    with self.subTest(kind=kind, assign=assign, nested=nested):
                        source = torch.nn.Linear(3, 2, bias=False)
                        if kind == 'transformer':
                            layer = streamed_linear.streamed_linear_type()(source, torch.float32)
                            stored = layer._iild_streamed_weight
                        else:
                            source.weight = torch.nn.Parameter(source.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
                            layer = flux_t5_stream.linear_type()(source, torch.tensor(0.25), torch.float32)
                            stored = layer._iild_fp8_weight
                        original = stored.clone()
                        placeholder = layer.weight
                        target = torch.nn.Sequential(layer) if nested else layer
                        key = '0.weight' if nested else 'weight'
                        with self.assertRaisesRegex(RuntimeError, 'cannot replace'):
                            target.load_state_dict({key: torch.ones(2, 3)}, strict=False, assign=assign)
                        self.assertIs(layer.weight, placeholder)
                        self.assertTrue(torch.equal(stored.view(torch.uint8), original.view(torch.uint8)))
                        target.load_state_dict({}, strict=False, assign=assign)

    def test_provisioned_canny_pipeline_loads_without_inference(self):
        if os.environ.get('IILD_VERIFY_FLUX1_PIPELINE_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_FLUX1_PIPELINE_LOAD=1 to load the real local FLUX components')
        from unittest.mock import patch
        import diffusers
        import generate_any
        import resource_fixture
        import torch
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='flux-real-pipeline-load-') as temporary:
            folder = Path(temporary)
            image = folder / 'control.png'
            # Argument resolution records the file; no image decoding or generation is performed.
            image.write_bytes(b'preflight-placeholder')
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                parsed = generate_any.build_parser().parse_args([
                    '--model', str(ROOT / 'resources/models/flux1-schnell-fp8.safetensors'),
                    '--model-config-preset', 'flux1-schnell', '--flux-transformer-storage', 'cpu-stream',
                    '--flux-t5-fp8-storage', 'cpu-stream',
                    '--controlnet', str(ROOT / 'resources/models/flux1-instantx-canny'),
                    '--control-image', str(image), '--control-preprocess', 'canny',
                    '--prompt', 'a city at night', '--width', '1024', '--height', '1024',
                    '--steps', '4', '--guidance-scale', '3.5', '--seed', '42',
                    '--dtype', 'bfloat16', '--offload', 'none', '--no-default-modifiers',
                    '--cache-dir', str(folder / 'cache'), '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(parsed)
                pipeline, report = generate_any.load_pipeline(selected, diffusers, torch.bfloat16)
            self.assertEqual(type(pipeline).__name__, 'FluxControlNetPipeline')
            self.assertEqual(type(pipeline.controlnet).__name__, 'FluxControlNetModel')
            self.assertFalse(pipeline.transformer.config.guidance_embeds)
            self.assertTrue(pipeline.controlnet.config.guidance_embeds)
            self.assertEqual(report['single_file_loading']['runtime_weights'], 'cpu-stream')
            self.assertEqual(selected.flux_text_encoder_status['text_encoder_2'], 'embedded')
            self.assertEqual(report['flux_text_encoder_loading']['text_encoder_2']['runtime_weights'], 'cpu-stream-fp8')
            self.assertIs(pipeline.text_encoder_2.shared.weight, pipeline.text_encoder_2.encoder.embed_tokens.weight)
            self.assertFalse((folder / 'unused-output').exists())

    def test_provisioned_schnell_stream_loads_actual_transformer_weights(self):
        if os.environ.get('IILD_VERIFY_FLUX1_STREAM_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_FLUX1_STREAM_LOAD=1 to load the real FP8 transformer')
        import torch
        from safetensors import safe_open
        import flux1_transformer_stream
        from streamed_linear import streamed_linear_type
        torch.set_num_threads(2)
        path = ROOT / 'resources/models/flux1-schnell-fp8.safetensors'
        config = ROOT / 'resources/model-configs/flux1-schnell'
        model = flux1_transformer_stream.load(path, config, torch.bfloat16)
        report = model._iild_single_file_loading
        layers = [module for module in model.modules() if isinstance(module, streamed_linear_type())]
        self.assertEqual(report['runtime_weights'], 'cpu-stream')
        self.assertEqual(report['streamed_layers'], len(layers))
        self.assertGreater(len(layers), 100)
        self.assertGreater(report['streamed_source_bytes'], 0)
        self.assertTrue(all(not value.is_meta for value in model.parameters()))
        with safe_open(str(path), framework='pt', device='cpu') as source:
            def check_fused(name, targets):
                fused = source.get_slice('model.diffusion_model.' + name)
                offset = 0
                for layer in targets:
                    self.assertIsInstance(layer, streamed_linear_type())
                    loaded = layer._iild_streamed_weight
                    self.assertEqual(loaded.device.type, 'cpu')
                    self.assertEqual(loaded.dtype, torch.float8_e4m3fn)
                    end = offset + loaded.shape[0]
                    expected = fused[offset:end]
                    # Compare the complete FP8 representation without creating
                    # full-size floating-point copies of the matrices.
                    self.assertEqual(tuple(loaded.shape), tuple(expected.shape))
                    self.assertTrue(torch.equal(loaded.view(torch.uint8), expected.view(torch.uint8)), name)
                    offset = end
                self.assertEqual(offset, fused.get_shape()[0], name)
            self.assertEqual(len(model.transformer_blocks), 19)
            self.assertEqual(len(model.single_transformer_blocks), 38)
            for index, block in enumerate(model.transformer_blocks):
                check_fused(f'double_blocks.{index}.img_attn.qkv.weight',
                            (block.attn.to_q, block.attn.to_k, block.attn.to_v))
                check_fused(f'double_blocks.{index}.txt_attn.qkv.weight',
                            (block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj))
            for index, block in enumerate(model.single_transformer_blocks):
                check_fused(f'single_blocks.{index}.linear1.weight',
                            (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.proj_mlp))

            # Check the remaining tensors as well: direct projections and
            # norms, split QKV/MLP biases, and swapped final modulation. The
            # large split matrices were checked independently above.
            from diffusers import FluxTransformer2DModel
            from flux1_single_file import PREFIX
            with torch.device('meta'):
                template = FluxTransformer2DModel.from_config(json.loads(
                    (config / 'transformer/config.json').read_text()))
            expected = {name: tuple(value.shape) for name, value in template.state_dict().items()}
            shapes = {name[len(PREFIX):]: tuple(source.get_slice(name).get_shape())
                      for name in source.keys() if name.startswith(PREFIX)}
            mapping = flux1_transformer_stream.plan(shapes, expected, 19, 38)
            self.assertEqual(len(mapping), 776)
            checked = set()
            for original, (targets, mode) in mapping.items():
                if mode == 'split' and len(shapes[original]) == 2:
                    continue
                stored = source.get_slice(PREFIX + original)
                source_offset = 0
                for target in targets:
                    module_name, _, leaf = target.rpartition('.')
                    module = model.get_submodule(module_name)
                    actual = (module._iild_streamed_weight if leaf == 'weight'
                              and isinstance(module, streamed_linear_type())
                              else model.get_parameter(target).detach())
                    self.assertEqual(tuple(actual.shape), expected[target], target)
                    rows = actual.shape[0]
                    if mode == 'swap':
                        half = rows // 2
                        segments = ((0, half, half), (half, 0, half))
                    else:
                        segments = ((0, source_offset, rows),)
                    for target_start, source_start, count in segments:
                        for offset in range(0, count, 512):
                            take = min(512, count - offset)
                            want = stored[source_start + offset:source_start + offset + take].to(actual.dtype)
                            got = actual[target_start + offset:target_start + offset + take]
                            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)), target)
                    source_offset += rows
                    checked.add(target)
            self.assertEqual(len(checked), len(expected) - 19 * 6 - 38 * 4)

    def test_streamed_transformer_rejects_ambiguous_quantization(self):
        import struct
        import torch
        import flux1_transformer_stream
        import resource_fixture
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())['header']
        config = ROOT / 'resources/model-configs/flux1-schnell'
        variants = []
        scaled = dict(header)
        scaled['__metadata__'] = {'_quantization_metadata': '{}'}
        variants.append((scaled, 'unscaled original tensors'))
        unknown_quantization = dict(header)
        unknown_quantization['__metadata__'] = {'comfy_quant_format': 'unknown'}
        variants.append((unknown_quantization, 'unscaled original tensors'))
        mixed = dict(header)
        mixed['img_in.weight'] = header['model.diffusion_model.img_in.weight']
        variants.append((mixed, 'mixes original tensor namespaces'))
        with resource_fixture.TemporaryDirectory(prefix='flux-stream-header-') as temporary:
            target = Path(temporary) / 'model.safetensors'
            for value, error in variants:
                encoded = json.dumps(value).encode()
                target.write_bytes(struct.pack('<Q', len(encoded)) + encoded)
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    flux1_transformer_stream.load(target, config, torch.float32)

    def test_public_embedded_vae_header_matches_bundled_config(self):
        import struct
        import diffusers
        import flux1_single_file
        import resource_fixture
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())['header']
        folder = ROOT / 'resources/model-configs/flux1-schnell'
        with resource_fixture.TemporaryDirectory(prefix='flux-vae-header-') as temporary:
            target = Path(temporary) / 'model.safetensors'
            def write_header(value):
                encoded = json.dumps(value).encode()
                target.write_bytes(struct.pack('<Q', len(encoded)) + encoded)
            write_header(header)
            flux1_single_file.validate_embedded_vae(target, folder, diffusers)
            broken = dict(header)
            del broken['vae.encoder.conv_in.bias']
            write_header(broken)
            with self.assertRaisesRegex(ValueError, r'missing converted weights:.*encoder\.conv_in\.bias'):
                flux1_single_file.validate_embedded_vae(target, folder, diffusers)

    def test_public_fp8_embedded_text_shapes_match_bundled_configs(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
        import flux_text_encoder
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())['header']
        folder = ROOT / 'resources/model-configs/flux1-schnell'
        configs = (('clip_l', CLIPTextModel, CLIPTextConfig, 'text_encoder'),
                   ('t5xxl', T5EncoderModel, T5Config, 'text_encoder_2'))
        for name, model_type, config_type, subdir in configs:
            with self.subTest(role=name), torch.device('meta'):
                model = model_type(config_type.from_pretrained(folder / subdir, local_files_only=True))
                expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
                if name == 't5xxl':
                    # The original Comfy file stores only the shared alias.
                    expected.pop('encoder.embed_tokens.weight')
                prefix = f'text_encoders.{name}.transformer.'
                actual = {key[len(prefix):]: tuple(value['shape']) for key, value in header.items()
                          if key.startswith(prefix)}
                if name == 'clip_l':
                    # The published full CLIP includes a projection unused by CLIPTextModel.
                    self.assertEqual(actual.pop('text_projection.weight'), (768, 768))
                self.assertEqual(set(flux_text_encoder.plan(actual, expected,
                                 'text_encoder' if name == 'clip_l' else 'text_encoder_2')), set(expected))
                if name == 't5xxl':
                    linear = [key for key in actual if key.startswith('encoder.block.') and key.endswith('.weight')
                              and isinstance(model.get_submodule(key.removesuffix('.weight')), torch.nn.Linear)]
                    self.assertEqual(len(linear), 168)
                    self.assertTrue(all(header[prefix + key]['dtype'] == 'F8_E4M3' for key in linear))

    def test_public_schnell_transformer_headers_match_bundled_config_and_converter(self):
        import torch
        from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
        from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers, infer_diffusers_model_type
        import generate_any
        import flux1_transformer_stream
        folder = ROOT / 'resources/model-configs/flux1-schnell'
        index = generate_any.validate_model_index(json.loads((folder / 'model_index.json').read_text()))
        self.assertEqual(index['_class_name'], 'FluxPipeline')
        config = json.loads((folder / 'transformer/config.json').read_text())
        with torch.device('meta'):
            model = FluxTransformer2DModel.from_config(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(folder / 'scheduler', local_files_only=True)
        self.assertEqual(scheduler.config.shift, 1.0)
        self.assertFalse(scheduler.config.use_dynamic_shifting)
        for fixture, prefix, dtype in (
            ('flux1-schnell-bf16-transformer-header.json', '', torch.bfloat16),
            ('flux1-schnell-fp8-full-header.json', 'model.diffusion_model.', torch.float8_e4m3fn),
        ):
            with self.subTest(fixture=fixture):
                header = json.loads((ROOT / 'resources/test-fixtures' / fixture).read_text())['header']
                tensors = {name: torch.empty(value['shape'], device='meta', dtype=dtype)
                           for name, value in header.items() if name != '__metadata__' and name.startswith(prefix)}
                self.assertEqual(infer_diffusers_model_type(tensors), 'flux-schnell')
                if prefix:
                    tensors = {name: tensor for name, tensor in tensors.items() if name.startswith(prefix)}
                shapes = {name.removeprefix(prefix): tuple(tensor.shape) for name, tensor in tensors.items()}
                stream_plan = flux1_transformer_stream.plan(shapes, expected, config['num_layers'], config['num_single_layers'])
                self.assertEqual({target for targets, _ in stream_plan.values() for target in targets}, set(expected))
                missing = dict(shapes)
                missing.pop('double_blocks.0.img_attn.qkv.weight')
                with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                    flux1_transformer_stream.plan(missing, expected, config['num_layers'], config['num_single_layers'])
                bad_shape = dict(shapes)
                bad_shape['single_blocks.0.linear1.weight'] = (1, 3072)
                with self.assertRaisesRegex(ValueError, 'shape does not match'):
                    flux1_transformer_stream.plan(bad_shape, expected, config['num_layers'], config['num_single_layers'])
                with self.assertRaisesRegex(ValueError, 'unexpected tensors'):
                    flux1_transformer_stream.plan({**shapes, 'double_blocks.0.unknown': (1,)}, expected,
                                                  config['num_layers'], config['num_single_layers'])
                converted = convert_flux_transformer_checkpoint_to_diffusers(tensors)
                self.assertEqual({name: tuple(value.shape) for name, value in converted.items()}, expected)
                self.assertEqual(tensors, {})

    def test_public_schnell_mapping_matches_upstream_source_and_row_order(self):
        """Check every production-shaped target against Diffusers, including fused rows."""
        from unittest.mock import patch
        import torch
        from diffusers import FluxTransformer2DModel
        from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
        import flux1_transformer_stream

        torch.set_num_threads(2)
        folder = ROOT / 'resources/model-configs/flux1-schnell'
        config = json.loads((folder / 'transformer/config.json').read_text())
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())['header']
        prefix = 'model.diffusion_model.'
        shapes = {name[len(prefix):]: tuple(value['shape']) for name, value in header.items()
                  if name.startswith(prefix)}
        with torch.device('meta'):
            model = FluxTransformer2DModel.from_config(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        plan = flux1_transformer_stream.plan(shapes, expected, config['num_layers'], config['num_single_layers'])

        # Each source row has a distinct marker; broadcast columns without
        # allocating production-sized matrices. Patch only one-item cat calls,
        # which are copies in the upstream converter and do not change values.
        sources = {}
        for index, (name, shape) in enumerate(sorted(shapes.items()), start=1):
            row = torch.arange(shape[0], dtype=torch.int32) + index * 100_000
            sources[name] = row.reshape((shape[0],) + (1,) * (len(shape) - 1)).expand(shape)
        original_cat = torch.cat
        def one_item_cat(values, *args, **kwargs):
            return values[0] if len(values) == 1 else original_cat(values, *args, **kwargs)
        checkpoint = {prefix + name: value for name, value in sources.items()}
        with patch.object(torch, 'cat', side_effect=one_item_cat):
            converted = convert_flux_transformer_checkpoint_to_diffusers(checkpoint)
        self.assertFalse(checkpoint)
        self.assertEqual(set(converted), set(expected))
        for source, (targets, mode) in plan.items():
            offset = 0
            for target in targets:
                rows = expected[target][0]
                for row in {0, rows // 2, rows - 1}:
                    original_row = (offset + row if mode == 'split' else
                                    (row + rows // 2) % rows if mode == 'swap' else row)
                    with self.subTest(source=source, target=target, row=row):
                        upstream = converted[target][(row,) + (0,) * (len(expected[target]) - 1)].item()
                        planned = sources[source][(original_row,) + (0,) * (len(shapes[source]) - 1)].item()
                        self.assertEqual(upstream, planned)
                offset += rows

    def test_embedded_t5_single_shared_weight_is_repaired(self):
        import torch
        from diffusers.loaders.single_file_utils import create_diffusers_t5_model_from_checkpoint
        from transformers import T5Config, T5EncoderModel
        import resource_fixture
        import flux_text_encoder
        torch.manual_seed(37)
        config = T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2,
                          dropout_rate=0.)
        source = T5EncoderModel(config).eval()
        state = {'text_encoders.t5xxl.transformer.' + name: tensor.detach().clone()
                 for name, tensor in source.state_dict().items() if name != 'encoder.embed_tokens.weight'}
        with resource_fixture.TemporaryDirectory(prefix='flux-embedded-t5-') as temporary:
            root = Path(temporary)
            config.save_pretrained(root / 'text_encoder_2')
            actual = create_diffusers_t5_model_from_checkpoint(T5EncoderModel, state, subfolder='text_encoder_2',
                                                               config=str(root), torch_dtype=torch.float32,
                                                               local_files_only=True).eval()
            self.assertFalse(actual.shared.weight.is_meta)
            self.assertEqual(flux_text_encoder.repair_embedded_t5(actual), 'repaired-shared-alias')
            self.assertIs(actual.shared, actual.encoder.embed_tokens)
            self.assertEqual(flux_text_encoder.repair_embedded_t5(actual), 'already-materialized')
            ids = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                expected = source(input_ids=ids).last_hidden_state
                result = actual(input_ids=ids).last_hidden_state
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_embedded_t5_rejects_other_unmaterialized_weights(self):
        import torch
        from transformers import T5Config, T5EncoderModel
        import flux_text_encoder
        with torch.device('meta'):
            model = T5EncoderModel(T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32,
                                             num_layers=1, num_heads=2))
        model.shared = torch.nn.Embedding(64, 16)
        with self.assertRaisesRegex(RuntimeError, 'meta device'):
            flux_text_encoder.repair_embedded_t5(model)


if __name__ == '__main__':
    unittest.main()
