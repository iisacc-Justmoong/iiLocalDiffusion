"""Public MOSS INT8 header and strict format validation, without model weights."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_int8_format
import moss_single_file


class MossInt8FormatTests(unittest.TestCase):
    def test_comfy_explicit_per_row_metadata_requires_row_scales(self):
        weight = 'layer.weight'
        quant = 'layer.comfy_quant'
        scale = 'layer.weight_scale'
        header = {weight: {'dtype': 'I8', 'shape': [2, 4]},
                  scale: {'dtype': 'F32', 'shape': [2, 1]},
                  quant: {'dtype': 'U8', 'shape': [80]}}
        spec = {'format': 'int8_tensorwise', 'convrot': True,
                'convrot_groupsize': 4, 'per_row': True}
        self.assertEqual(moss_int8_format.tensor_spec(header, weight, {quant: spec})[-1], 4)
        header[scale]['shape'] = []
        with self.assertRaises(ValueError):
            moss_int8_format.tensor_spec(header, weight, {quant: spec})

    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / 'resources/test-fixtures'
        cls.quant = json.loads((fixtures / 'moss-qwen35-int8-convrot-header.json').read_text())
        full = json.loads((fixtures / 'moss-qwen35-encoder-header.json').read_text())
        cls.expected = {key[len(moss_int8_format.PREFIX):]: info['shape']
                        for key, info in full['header'].items() if key.startswith(moss_int8_format.PREFIX)}

    def test_public_checkpoint_header_and_every_quantization_record(self):
        floating, quantized = moss_int8_format.plan(self.quant['header'], self.expected, self.quant['quant_metadata'])
        self.assertEqual(len(floating), 225)
        self.assertEqual(len(quantized), 201)
        self.assertEqual({item.group_size for item in quantized.values()}, {256})
        self.assertIn('model.language_model.embed_tokens.weight', quantized)

    def test_rejects_unpaired_and_inconsistent_quantization(self):
        header = self.quant['header']
        metadata = self.quant['quant_metadata']
        name = 'model.language_model.embed_tokens.comfy_quant'
        changes = [
            {'format': 'other', 'convrot': True, 'convrot_groupsize': 256},
            {'format': 'int8_tensorwise', 'convrot': False, 'convrot_groupsize': 256},
            {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 8},
            {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 65536},
        ]
        for spec in changes:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                moss_int8_format.plan(header, self.expected, {**metadata, name: spec})
        broken = copy.deepcopy(header)
        broken['model.language_model.embed_tokens.weight_scale']['shape'] = [2]
        with self.assertRaisesRegex(ValueError, 'scale'):
            moss_int8_format.plan(broken, self.expected, metadata)
        broken = dict(header)
        broken['model.language_model.unknown.comfy_quant'] = {'dtype': 'U8', 'shape': [72]}
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            moss_int8_format.plan(broken, self.expected, metadata)
        broken = dict(header)
        del broken['model.language_model.embed_tokens.weight']
        with self.assertRaises(ValueError):
            moss_int8_format.plan(broken, self.expected, metadata)

    def test_transformer_attention_preferences_are_validated_separately_from_weights(self):
        weight = 'layers.0.proj.qkv.weight'
        scale = 'layers.0.proj.qkv.weight_scale'
        quant = 'layers.0.proj.qkv.comfy_quant'
        attention = 'layers.0.comfy_attention.config'
        header = {weight: {'dtype': 'I8', 'shape': [12, 4]}, scale: {'dtype': 'F32', 'shape': [12, 1]},
                  quant: {'dtype': 'U8', 'shape': [75]}, attention: {'dtype': 'U8', 'shape': [35]}}
        metadata = {quant: {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4},
                    attention: {'attention': 'comfy_kitchen_int8'}}
        expected = {f'layers.0.proj.{part}.weight': (4, 4) for part in 'qkv'}
        mapping, quantized = moss_single_file.quantized_plan(header, expected, metadata)
        self.assertEqual(len(mapping[weight]), 3)
        self.assertEqual(len(quantized), 1)
        self.assertEqual(moss_single_file.attention_preferences(header, expected, metadata),
                         {attention: ['comfy_kitchen_int8']})
        for name, replacement in [
            (attention, {'dtype': 'F32', 'shape': [35]}),
            ('layers.1.comfy_attention.config', header[attention]),
        ]:
            broken = dict(header)
            del broken[attention]
            broken[name] = replacement
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'attention'):
                moss_single_file.quantized_plan(broken, expected, {**metadata, name: metadata[attention]})
        with self.assertRaisesRegex(ValueError, 'attention'):
            moss_single_file.quantized_plan(header, expected, {**metadata, attention: {'attention': 1}})


if __name__ == '__main__':
    unittest.main()
