"""Strict FLUX.1 split text encoder loading and public checkpoint contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import flux_text_encoder as loader

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install reference runtime for FLUX text encoders')
class FluxTextEncoderRuntimeTests(unittest.TestCase):
    def test_embedded_unscaled_fp8_t5_streams_and_matches_dense(self):
        import torch
        from safetensors.torch import save_file
        from transformers import T5Config, T5EncoderModel
        import resource_fixture
        torch.manual_seed(541)
        torch.set_num_threads(2)
        config = T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32,
                          num_layers=2, num_heads=2, dropout_rate=0.)
        original = T5EncoderModel(config).eval()
        prefix = 'text_encoders.t5xxl.transformer.'
        state = {prefix + name: value.detach().to(torch.float8_e4m3fn)
                 for name, value in original.state_dict().items() if name != 'encoder.embed_tokens.weight'}
        state['text_encoders.t5xxl.logit_scale'] = torch.tensor(1.)
        state['model.diffusion_model.fixture'] = torch.tensor([1.])
        with resource_fixture.TemporaryDirectory(prefix='flux-embedded-stream-') as temporary:
            folder = Path(temporary)
            config.save_pretrained(folder / 'text_encoder_2')
            path = folder / 'combined.safetensors'
            save_file(state, str(path))
            selected = loader.resolve_embedded_t5(str(path), 'FluxPipeline', folder, None)
            self.assertEqual(selected.metadata()['mode'], 'embedded-single-file')
            self.assertEqual(selected.metadata()['status'], 'embedded')
            dense = loader.load(selected, torch.float32)
            streamed = loader.load(selected, torch.float32, storage='cpu-stream')
            self.assertEqual(streamed._iild_flux_text_encoder_loading['format'], 'unscaled-fp8-e4m3fn')
            self.assertEqual(streamed._iild_flux_text_encoder_loading['runtime_weights'], 'cpu-stream-fp8')
            layer = streamed.get_submodule('encoder.block.0.layer.0.SelfAttention.q')
            self.assertEqual(layer._iild_fp8_weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(layer._iild_fp8_weight.device.type, 'cpu')
            self.assertEqual(layer._iild_fp8_scale.item(), 1.)
            self.assertIs(streamed.shared.weight, streamed.encoder.embed_tokens.weight)
            inputs = {'input_ids': torch.tensor([[4, 12, 1]])}
            for device in (('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',)):
                dense.to(device)
                streamed.to(device)
                with torch.no_grad():
                    expected = dense(**{key: value.to(device) for key, value in inputs.items()}).last_hidden_state
                    actual = streamed(**{key: value.to(device) for key, value in inputs.items()}).last_hidden_state
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(layer._iild_fp8_weight.device.type, 'cpu')
            for precision in (torch.float16, torch.bfloat16):
                dense_low = loader.load(selected, precision)
                streamed_low = loader.load(selected, precision, storage='cpu-stream')
                with torch.no_grad():
                    torch.testing.assert_close(streamed_low(**inputs).last_hidden_state,
                                               dense_low(**inputs).last_hidden_state, rtol=0, atol=0)
            invalid = dict(state)
            invalid[prefix + 'encoder.embed_tokens.weight'] = torch.zeros_like(state[prefix + 'shared.weight'])
            bad = folder / 'bad.safetensors'
            save_file(invalid, str(bad))
            with self.assertRaisesRegex(ValueError, 'embedding weights differ'):
                loader.load(loader.resolve_embedded_t5(str(bad), 'FluxPipeline', folder, None), torch.float32,
                            storage='cpu-stream')
            embedding_only = {prefix + name: value.detach().clone()
                              for name, value in original.state_dict().items() if name != 'encoder.embed_tokens.weight'}
            embedding_only[prefix + 'shared.weight'] = state[prefix + 'shared.weight']
            save_file(embedding_only, str(bad))
            with self.assertRaisesRegex(ValueError, 'at least one FP8 linear'):
                loader.load(loader.resolve_embedded_t5(str(bad), 'FluxPipeline', folder, None), torch.float32,
                            storage='cpu-stream')

    def test_scaled_header_rejects_ambiguous_or_unsupported_metadata(self):
        marker = {'scaled_fp8': (0,), 'layer.weight': (2, 2), 'layer.scale_weight': ()}
        types = {'scaled_fp8': 'F8_E4M3', 'layer.weight': 'F8_E4M3', 'layer.scale_weight': 'F32'}
        invalid = [
            ({key: shape for key, shape in marker.items() if key != 'scaled_fp8'}, types, 'text_encoder_2'),
            ({**marker, 'scaled_fp8': (2,)}, types, 'text_encoder_2'),
            (marker, {**types, 'layer.scale_weight': 'F16'}, 'text_encoder_2'),
            (marker, {**types, 'layer.weight': 'F16'}, 'text_encoder_2'),
            (marker, types, 'text_encoder'),
            ({key: shape for key, shape in marker.items() if key != 'layer.scale_weight'}, types, 'text_encoder_2'),
        ]
        for shapes, dtypes, role in invalid:
            with self.subTest(shapes=shapes, dtypes=dtypes, role=role), self.assertRaises(ValueError):
                loader.plan_scaling(shapes, dtypes, role)

    def test_public_headers_match_actual_configurations(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
        for role, cls, config_cls, folder, fixtures in (
            ('text_encoder', CLIPTextModel, CLIPTextConfig, 'flux1-clip-config', ('clip_l',)),
            ('text_encoder_2', T5EncoderModel, T5Config, 'flux1-t5-config', ('t5xxl_fp16', 't5xxl_fp8_e4m3fn', 't5xxl_fp8_e4m3fn_scaled')),
        ):
            config = json.loads((ROOT / 'resources/text_encoders' / folder / 'config.json').read_text())
            with torch.device('meta'):
                model = cls(config_cls.from_dict(config))
            expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
            for fixture in fixtures:
                header = json.loads((ROOT / f'resources/test-fixtures/flux1-{fixture}-header.json').read_text())['header']
                shapes = {name: tuple(value['shape']) for name, value in header.items() if name != '__metadata__'}
                dtypes = {name: value['dtype'] for name, value in header.items() if name != '__metadata__'}
                with self.subTest(fixture=fixture):
                    core, scales = loader.plan_scaling(shapes, dtypes, role)
                    self.assertEqual(set(loader.plan(core, expected, role)), set(expected))
                    if fixture.endswith('_scaled'):
                        self.assertEqual(len(scales), 168)
                        self.assertTrue(all(isinstance(model.get_submodule(name.removesuffix('.weight')), torch.nn.Linear)
                                            for name in scales))

    def test_single_file_matches_full_precision_models_and_rejects_bad_files(self):
        import torch
        from safetensors.torch import save_file
        from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
        import resource_fixture
        torch.manual_seed(42)
        torch.set_num_threads(2)
        specs = (
            ('text_encoder', CLIPTextModel, CLIPTextConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16, bos_token_id=0, eos_token_id=1)),
            ('text_encoder_2', T5EncoderModel, T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32,
                num_layers=1, num_heads=2, dropout_rate=0.)),
        )
        with resource_fixture.TemporaryDirectory(prefix='flux-encoder-') as temp:
            root = Path(temp)
            for role, cls, config in specs:
                with self.subTest(role=role):
                    source = cls(config).eval()
                    folder = root / role
                    folder.mkdir()
                    config.save_pretrained(folder)
                    weights = {key: value.detach().clone() for key, value in source.state_dict().items()}
                    file = root / (role + '.safetensors')
                    save_file(weights, str(file))
                    fallback = loader.resolve_file(str(file), role, 'FluxPipeline', root / 'missing-package', None)
                    self.assertEqual(fallback.config_status, 'fallback')
                    loader.verify_selection(fallback)
                    selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                    target = loader.load(selection, torch.float32)
                    with self.assertRaisesRegex(ValueError, 'cpu-stream'):
                        loader.load(selection, torch.float32, storage='cpu-stream')
                    inputs = {'input_ids': torch.tensor([[3, 5, 7, 1]])}
                    with torch.no_grad():
                        self.assertTrue(torch.equal(source(**inputs).last_hidden_state, target(**inputs).last_hidden_state))
                    if role == 'text_encoder_2':
                        fp8 = {key: value.to(torch.float8_e4m3fn) for key, value in weights.items()}
                        save_file(fp8, str(file))
                        selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                        quantized = loader.load(selection, torch.float32)
                        for key, value in quantized.state_dict().items():
                            self.assertTrue(torch.equal(value, fp8[key].float()), key)
                        scaled = dict(weights)
                        expected_scaled = dict(weights)
                        for index, key in enumerate(name for name in weights if name.endswith(('SelfAttention.q.weight', 'SelfAttention.k.weight', 'DenseReluDense.wo.weight'))):
                            scale = torch.tensor(.125 * (index + 1), dtype=torch.float32)
                            value = (weights[key] / scale).to(torch.float8_e4m3fn)
                            scaled[key] = value
                            scaled[key[:-len('.weight')] + '.scale_weight'] = scale
                            expected_scaled[key] = value.float() * scale
                        scaled['scaled_fp8'] = torch.empty(0, dtype=torch.float8_e4m3fn)
                        save_file(scaled, str(file))
                        selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                        result = loader.load(selection, torch.float32)
                        for key, value in result.state_dict().items():
                            self.assertTrue(torch.equal(value, expected_scaled[key]), key)
                        streamed = loader.load(selection, torch.float32, storage='cpu-stream')
                        self.assertEqual(streamed._iild_flux_text_encoder_loading['runtime_weights'], 'cpu-stream-fp8')
                        layer = streamed.get_submodule('encoder.block.0.layer.1.DenseReluDense.wo')
                        self.assertEqual(layer._iild_fp8_weight.dtype, torch.float8_e4m3fn)
                        self.assertEqual(layer._iild_fp8_weight.device.type, 'cpu')
                        self.assertEqual(layer.weight.dtype, torch.float32)
                        self.assertTrue(all(stride == 0 for stride in layer.weight.stride()))
                        with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
                            streamed.state_dict()
                        for device in (('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',)):
                            with self.subTest(device=device):
                                streamed.to(device)
                                result.to(device)
                                device_inputs = {key: value.to(device) for key, value in inputs.items()}
                                with torch.no_grad():
                                    actual = streamed(**device_inputs).last_hidden_state
                                    dense = result(**device_inputs).last_hidden_state
                                torch.testing.assert_close(actual, dense, rtol=0, atol=0)
                                self.assertEqual(layer._iild_fp8_weight.device.type, 'cpu')
                        for precision in (torch.float16, torch.bfloat16):
                            result_low = loader.load(selection, precision)
                            for key, value in result_low.state_dict().items():
                                scale_key = key[:-len('.weight')] + '.scale_weight' if key.endswith('.weight') else ''
                                expected_value = (scaled[key].to(precision) * scaled[scale_key].to(precision)
                                                  if scale_key in scaled else weights[key].to(precision))
                                self.assertTrue(torch.equal(value, expected_value), key)
                        for invalid_scale in (-1., float('nan'), float('inf')):
                            invalid = dict(scaled)
                            scale_key = next(name for name in scaled if name.endswith('.scale_weight'))
                            invalid[scale_key] = torch.tensor(invalid_scale)
                            save_file(invalid, str(file))
                            selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                            with self.assertRaisesRegex(ValueError, 'scale'):
                                loader.load(selection, torch.float32)
                        invalid = dict(scaled)
                        invalid.pop(next(name for name in invalid if name.endswith('.scale_weight')))
                        save_file(invalid, str(file))
                        selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                        with self.assertRaisesRegex(ValueError, 'scale'):
                            loader.load(selection, torch.float32)
                        mismatched = dict(weights)
                        mismatched['shared.weight'] = mismatched['shared.weight'] + 1
                        save_file(mismatched, str(file))
                        selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                        with self.assertRaisesRegex(ValueError, 'embedding weights differ'):
                            loader.load(selection, torch.float32)
                    bad = dict(weights)
                    bad.pop(next(iter(bad)))
                    save_file(bad, str(file))
                    selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                    with self.assertRaisesRegex(ValueError, 'missing'):
                        loader.load(selection, torch.float32)
                    bad = dict(weights)
                    key = next(name for name in bad if name not in ('shared.weight', 'encoder.embed_tokens.weight'))
                    bad[key] = torch.full_like(bad[key], float('nan'))
                    save_file(bad, str(file))
                    selection = loader.resolve_file(str(file), role, 'FluxPipeline', root, None)
                    with self.assertRaisesRegex(ValueError, 'non-finite'):
                        loader.load(selection, torch.float32)


if __name__ == '__main__':
    unittest.main()
