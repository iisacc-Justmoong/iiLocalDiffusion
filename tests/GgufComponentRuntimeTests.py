"""Load a tiny quantized component with the actual Diffusers GGUF loader."""

import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'gguf', 'accelerate', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install torch, diffusers, gguf, accelerate and numpy for the synthetic GGUF loader test')
class GgufComponentRuntimeTests(unittest.TestCase):
    def test_q8_flux_component_is_loaded_and_other_tensors_preserved(self):
        import diffusers
        import gguf
        from gguf.quants import quantize
        import torch
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
        import gguf_component
        from weight_files import LocalWeightFile

        torch.set_num_threads(2)
        torch.manual_seed(19)
        model = diffusers.FluxTransformer2DModel(in_channels=4, out_channels=4, num_layers=1,
            num_single_layers=1, attention_head_dim=8, num_attention_heads=4,
            joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=(2, 2, 4)).eval()
        quantized_name = 'transformer_blocks.0.attn.to_q.weight'
        expected = model.state_dict()
        with resource_fixture.TemporaryDirectory(prefix='gguf-component-runtime-') as temporary:
            root = Path(temporary) / 'model';root.mkdir()
            cache = Path(temporary) / 'cache'
            (root / 'transformer').mkdir()
            model.save_config(root / 'transformer')
            # The contents, not the source extension, determine the loader.
            source = root / 'transformer.safetensors'
            writer = gguf.GGUFWriter(source, arch='flux')
            for name, value in expected.items():
                array = value.detach().cpu().float().contiguous().numpy()
                if name == quantized_name:
                    writer.add_tensor(name, quantize(array, gguf.GGMLQuantizationType.Q8_0),
                                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                else:
                    writer.add_tensor(name, array)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            weight = LocalWeightFile(str(source), str(source.resolve()), before, source.stat().st_size)
            index = {'transformer': ['diffusers', 'FluxTransformer2DModel']}
            selected = gguf_component.select(index, diffusers.FluxPipeline, diffusers, root)
            loaded, report = gguf_component.load(selected, weight, root, cache, torch.float32, diffusers)
            actual = loaded.state_dict()
            self.assertEqual(set(actual), set(expected))
            self.assertIsInstance(actual[quantized_name], GGUFParameter)
            self.assertEqual(actual[quantized_name].quant_type, gguf.GGMLQuantizationType.Q8_0)
            torch.testing.assert_close(dequantize_gguf_tensor(actual[quantized_name]).float(),
                                       expected[quantized_name], rtol=0.03, atol=0.005)
            for name, value in expected.items():
                if name != quantized_name:
                    torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)
            self.assertEqual(report['component'], 'transformer')
            self.assertEqual(report['format'], 'gguf')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertEqual(list((cache / 'single-file-aliases').iterdir()), [])

            # Construct the actual pipeline through the generic runner without
            # calling the model or executing pretrained inference.
            import json
            import generate_any as generic
            index.update({'_class_name': 'FluxPipeline', '_diffusers_version': diffusers.__version__,
                **{name: [None, None] for name in ('scheduler', 'vae', 'text_encoder', 'text_encoder_2',
                                                    'tokenizer', 'tokenizer_2', 'image_encoder', 'feature_extractor')}})
            (root / 'model_index.json').write_text(json.dumps(index))
            with patch.object(generic, 'resolve_vae_selection', return_value=(None, 'not-declared')):
                args = generic.resolve_arguments(generic.build_parser().parse_args([
                    '--model', str(source), '--model-config', str(root), '--cache-dir', str(cache / 'runner'),
                    '--output-dir', str(Path(temporary) / 'output'), '--no-default-modifiers']))
            pipeline, pipeline_report = generic.load_pipeline(args, diffusers, torch.float32)
            self.assertIsInstance(pipeline, diffusers.FluxPipeline)
            self.assertIsInstance(pipeline.transformer.state_dict()[quantized_name], GGUFParameter)
            self.assertEqual(pipeline_report['single_file_loading']['format'], 'gguf')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertFalse((Path(temporary) / 'output').exists())

            packaged = root / 'transformer/diffusion_pytorch_model.gguf'
            shutil.copyfile(source, packaged)
            with patch.object(generic, 'resolve_vae_selection', return_value=(None, 'not-declared')):
                directory_args = generic.resolve_arguments(generic.build_parser().parse_args([
                    '--model', str(root), '--cache-dir', str(cache / 'directory'),
                    '--output-dir', str(Path(temporary) / 'directory-output'), '--no-default-modifiers']))
                adjacent_args = generic.resolve_arguments(generic.build_parser().parse_args([
                    '--model', str(packaged), '--cache-dir', str(cache / 'adjacent'),
                    '--output-dir', str(Path(temporary) / 'adjacent-output'), '--no-default-modifiers']))
            self.assertEqual(adjacent_args.model_config_status, 'adjacent')
            for args, loading_key in ((directory_args, 'directory_gguf_loading'),
                                      (adjacent_args, 'single_file_loading')):
                pipeline, report = generic.load_pipeline(args, diffusers, torch.float32)
                self.assertIsInstance(pipeline, diffusers.FluxPipeline)
                self.assertIsInstance(pipeline.transformer.state_dict()[quantized_name], GGUFParameter)
                self.assertIn(loading_key, report)
            self.assertEqual(hashlib.sha256(packaged.read_bytes()).hexdigest(), before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
