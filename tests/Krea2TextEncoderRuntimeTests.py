"""Compare the fallback language-only loader to real Qwen3-VL text execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_text_encoder as encoder

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical Qwen3-VL tests')
class Krea2TextEncoderRuntimeTests(unittest.TestCase):
    def test_provisioned_pose_multimodal_encoder_loads_without_inference(self):
        if os.environ.get('IILD_VERIFY_KREA2_VISION_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_VISION_LOAD=1 to load the real Qwen3-VL model')
        from unittest.mock import patch
        import generate_any
        import resource_fixture
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        with resource_fixture.TemporaryDirectory(prefix='krea-real-vision-load-') as temporary:
            folder = Path(temporary)
            image = folder / 'pose.png'
            image.write_bytes(b'preflight-placeholder')
            with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
                parsed = generate_any.build_parser().parse_args([
                    '--model', str(ROOT / 'resources/models/krea2_turbo_fp8_scaled.safetensors'),
                    '--model-config-preset', 'krea2-turbo', '--krea2-variant', 'turbo',
                    '--krea2-fp8-storage', 'cpu-stream', '--controlnet-format', 'krea2-ostris-edit',
                    '--controlnet', str(ROOT / 'resources/models/krea2-turbo-openpose-controlnet.safetensors'),
                    '--control-image', str(image), '--prompt', 'a dancer in a studio',
                    '--width', '1024', '--height', '1024', '--steps', '10', '--guidance-scale', '0',
                    '--seed', '42', '--dtype', 'bfloat16', '--offload', 'none', '--no-default-modifiers',
                    '--output-dir', str(folder / 'unused-output')])
                selected = generate_any.resolve_arguments(parsed)
                model = encoder.load_full(selected.text_encoder_selection, torch.bfloat16)
            self.assertEqual(type(model).__name__, 'Qwen3VLModel')
            self.assertTrue(all(not value.is_meta for value in model.parameters()))
            self.assertTrue(all(not value.is_meta for value in model.buffers()))
            with safe_open(selected.text_encoder_selection.weight.path, framework='pt', device='cpu') as source:
                name = next(name for name in model.state_dict() if name.startswith('visual.')
                            and model.state_dict()[name].ndim > 1)
                torch.testing.assert_close(model.state_dict()[name].flatten()[:32],
                    source.get_tensor('model.' + name).flatten()[:32], rtol=0, atol=0)
            self.assertFalse((folder / 'unused-output').exists())

    def test_full_qwen_vision_and_language_weights_reproduce_image_conditioning(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLModel
        from safetensors.torch import save_file
        from weight_files import resolve_weight_file
        import resource_fixture
        torch.set_num_threads(2)
        torch.manual_seed(34)
        config = Qwen3VLConfig(
            image_token_id=55, video_token_id=57, vision_start_token_id=56,
            text_config={'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'max_position_embeddings': 128,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                             'mrope_interleaved': True}},
            vision_config={'depth': 1, 'hidden_size': 16, 'num_heads': 2, 'intermediate_size': 32,
                           'out_hidden_size': 32, 'num_position_embeddings': 16,
                           'deepstack_visual_indexes': []})
        original = Qwen3VLModel(config).eval()
        with torch.device('meta'):
            restored = Qwen3VLModel(config)
        weights = {'model.' + name: value.detach().clone() for name, value in original.state_dict().items()}
        with resource_fixture.TemporaryDirectory(prefix='krea2-qwen-vision-') as temporary:
            path = Path(temporary) / 'encoder.safetensors'
            save_file(weights, str(path))
            encoder.load_full_weights(restored, resolve_weight_file(str(path), 'encoder'), torch.float32, torch)
            ids = torch.tensor([[1, 56, 55, 55, 55, 55, 2]])
            grid = torch.tensor([[1, 4, 4]])
            pixels = torch.randn(16, 3 * 2 * 16 * 16)
            with torch.no_grad():
                expected = original(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(ids == 55).long(),
                                    output_hidden_states=True, use_cache=False)
                actual = restored(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(ids == 55).long(),
                                  output_hidden_states=True, use_cache=False)
            for left, right in zip(expected.hidden_states, actual.hidden_states):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertFalse(any(p.is_meta for p in restored.parameters()))
            self.assertFalse(any(b.is_meta for b in restored.buffers()))
            vision_key = next(name for name in weights if name.startswith('model.visual.'))
            for problem in ('missing', 'shape', 'nonfinite'):
                broken = dict(weights)
                if problem == 'missing':
                    broken.pop(vision_key)
                elif problem == 'shape':
                    broken[vision_key] = torch.zeros(1)
                else:
                    broken[vision_key] = torch.full_like(broken[vision_key], float('nan'))
                invalid = Path(temporary) / f'{problem}.safetensors'
                save_file(broken, str(invalid))
                with torch.device('meta'):
                    empty = Qwen3VLModel(config)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    encoder.load_full_weights(empty, resolve_weight_file(str(invalid), 'encoder'), torch.float32, torch)

    def test_full_qwen_model_and_loaded_text_only_hidden_states_match(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLModel, Qwen3VLTextModel
        from safetensors.torch import save_file
        from weight_files import resolve_weight_file
        import resource_fixture
        torch.set_num_threads(2)
        torch.manual_seed(33)
        config = Qwen3VLConfig(
            text_config={'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'max_position_embeddings': 128,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                             'mrope_interleaved': True}},
            vision_config={'depth': 1, 'hidden_size': 16, 'num_heads': 2, 'intermediate_size': 32,
                           'out_hidden_size': 32, 'num_position_embeddings': 16,
                           'deepstack_visual_indexes': []})
        full = Qwen3VLModel(config).eval()
        with torch.device('meta'):
            text = Qwen3VLTextModel(config.text_config)
        weights = {'model.' + name: value.detach().clone() for name, value in full.state_dict().items()}
        with resource_fixture.TemporaryDirectory(prefix='krea2-qwen-') as temporary:
            path = Path(temporary) / 'encoder.safetensors'
            save_file(weights, str(path))
            encoder.load_language_weights(text, resolve_weight_file(str(path), 'text'), torch.float32, torch)
            ids = torch.tensor([[3, 6, 9, 0, 12]])
            mask = torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.bool)
            positions = (mask.long().cumsum(-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
            with torch.no_grad():
                expected = full(input_ids=ids, attention_mask=mask, position_ids=positions,
                                output_hidden_states=True, use_cache=False)
                actual = text(input_ids=ids, attention_mask=mask, position_ids=positions,
                              output_hidden_states=True, use_cache=False)
            self.assertEqual(len(expected.hidden_states), len(actual.hidden_states))
            for left, right in zip(expected.hidden_states, actual.hidden_states):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertFalse(any(p.device.type == 'meta' for p in text.parameters()))
            self.assertFalse(any(b.device.type == 'meta' for b in text.buffers()))

    def test_bundled_resource_language_tensor_shapes_match_official_config(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLTextModel
        from safetensors import safe_open
        config_path = ROOT / 'resources/text_encoders/qwen3vl-4b/config.json'
        weights_path = ROOT / 'resources/text_encoders/qwen3vl_4b_bf16.safetensors'
        if not weights_path.is_file():
            self.skipTest('Bundled Qwen3-VL weights are not provisioned')
        config = Qwen3VLConfig.from_dict(json.loads(config_path.read_text())).text_config
        with torch.device('meta'):
            model = Qwen3VLTextModel(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        with safe_open(str(weights_path), framework='pt', device='cpu') as source:
            actual = {name[len(encoder.PREFIX):]: name for name in source.keys() if name.startswith(encoder.PREFIX)}
            self.assertEqual(set(expected), set(actual))
            for name, shape in expected.items():
                self.assertEqual(tuple(source.get_slice(actual[name]).get_shape()), shape, name)

    def test_bundled_full_qwen_resource_shapes_match_vision_and_language_model(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLModel
        from safetensors import safe_open
        config_path = ROOT / 'resources/text_encoders/qwen3vl-4b/config.json'
        weights_path = ROOT / 'resources/text_encoders/qwen3vl_4b_bf16.safetensors'
        if not weights_path.is_file():
            self.skipTest('Bundled Qwen3-VL weights are not provisioned')
        config = Qwen3VLConfig.from_dict(json.loads(config_path.read_text()))
        with torch.device('meta'):
            model = Qwen3VLModel(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        with safe_open(str(weights_path), framework='pt', device='cpu') as source:
            actual = {name[len('model.'):]: name for name in source.keys()
                      if name.startswith(('model.language_model.', 'model.visual.'))}
            self.assertEqual(len(actual), 713)
            self.assertEqual(set(expected), set(actual))
            for name, shape in expected.items():
                self.assertEqual(tuple(source.get_slice(actual[name]).get_shape()), shape, name)


if __name__ == '__main__':
    unittest.main()
