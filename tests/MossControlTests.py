"""Offline MOSS color/depth adapter shape and input contracts."""
import sys
import json
import hashlib
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import moss_controlnet as control


class MossControlTests(unittest.TestCase):
    def test_published_control_header_matches_the_shipped_architecture_contract(self):
        root = Path(__file__).resolve().parents[1]
        fixture = json.loads((root / 'resources/test-fixtures/moss-color-depth-header.json').read_text())
        header = fixture['header']
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '4a2e82af34eed2d2182d3f09f6bf45b2216d7cd4beecf426f218d8a09a71f6b0')
        metadata = header['__metadata__']
        shapes = {name: value['shape'] for name, value in header.items() if name != '__metadata__'}
        image = {'q': (3072, 3072), 'k': (3072, 3072), 'v': (3072, 3072),
                 'o': (3072, 3072), 'ffn_up_gate': (16384, 3072), 'ffn_down': (3072, 8192)}
        text = {'qkv': (9216, 3072), 'proj': (3072, 3072), 'ffn_gate': (8192, 3072),
                'ffn_up': (8192, 3072), 'ffn_down': (3072, 8192)}
        modules = {**{f'layers.{i}.proj.{name}': shape for i in range(42) for name, shape in image.items()},
                   **{f'text_embedder.blocks.{i}.{name}': shape for i in range(2) for name, shape in text.items()}}
        plan = control.plan_weights(shapes, modules, metadata, projection_shape=(3072, 128),
                                    image_layers=42, text_layers=2)
        self.assertEqual(len(shapes), 526)
        self.assertEqual(len(plan), 262)
        self.assertEqual(fixture['weight_sha256'], 'e5c0c3efd05e4269207a39de6005798f14c90c6687c89c54eaffe4b76c43a763')

    def fixture(self):
        names = [*(f'layers.0.proj.{name}' for name in control.IMAGE_TARGETS),
                 *(f'text_embedder.blocks.0.{name}' for name in control.TEXT_TARGETS)]
        modules = {name: (12, 8) for name in names}
        tensors = {'video_embedder.weight': (12, 24), 'video_embedder.bias': (12,)}
        for name in names:
            tensors[name + '.lora_A.weight'] = (2, 8)
            tensors[name + '.lora_B.weight'] = (12, 2)
        metadata = {'hayate_control': '1', 'base_model': 'moss-base',
                    'control_channels': 'color,depth', 'control_preprocess': 'vae',
                    'rank': '2', 'alpha': '2'}
        return tensors, modules, metadata

    def test_control_adapter_maps_all_image_and_text_layers(self):
        tensors, modules, metadata = self.fixture()
        plan = control.plan_weights(tensors, modules, metadata,
                                    projection_shape=(12, 8), image_layers=1, text_layers=1)
        self.assertEqual(set(plan), set(modules))

    def test_base_and_sft_metadata_are_supported_but_other_families_are_rejected(self):
        tensors, modules, metadata = self.fixture()
        for name in ('moss-base', 'moss-sft'):
            metadata['base_model'] = name
            self.assertEqual(set(control.plan_weights(tensors, modules, metadata,
                projection_shape=(12, 8), image_layers=1, text_layers=1)), set(modules))
        metadata['base_model'] = 'moss-other'
        with self.assertRaises(ValueError):
            control.plan_weights(tensors, modules, metadata, projection_shape=(12, 8), image_layers=1, text_layers=1)

    def test_incompatible_or_partial_adapter_is_rejected(self):
        tensors, modules, metadata = self.fixture()
        for change in ('missing', 'wrong_projection', 'unexpected', 'rank', 'metadata', 'target'):
            shapes, targets, meta = dict(tensors), dict(modules), dict(metadata)
            if change == 'missing': shapes.pop('layers.0.proj.q.lora_B.weight')
            elif change == 'wrong_projection': shapes['video_embedder.weight'] = (12, 16)
            elif change == 'unexpected': shapes['unrelated'] = (1,)
            elif change == 'rank': shapes['layers.0.proj.q.lora_A.weight'] = (3, 8)
            elif change == 'metadata': meta['control_channels'] = 'depth'
            else: targets.pop('layers.0.proj.q')
            with self.subTest(change=change), self.assertRaises(ValueError):
                control.plan_weights(shapes, targets, meta,
                                     projection_shape=(12, 8), image_layers=1, text_layers=1)

    def test_published_color_depth_header_dimensions(self):
        image = {'q': (3072, 3072), 'k': (3072, 3072), 'v': (3072, 3072),
                 'o': (3072, 3072), 'ffn_up_gate': (16384, 3072), 'ffn_down': (3072, 8192)}
        text = {'qkv': (9216, 3072), 'proj': (3072, 3072), 'ffn_gate': (8192, 3072),
                'ffn_up': (8192, 3072), 'ffn_down': (3072, 8192)}
        modules = {**{f'layers.{i}.proj.{name}': shape for i in range(42) for name, shape in image.items()},
                   **{f'text_embedder.blocks.{i}.{name}': shape for i in range(2) for name, shape in text.items()}}
        weights = {'video_embedder.weight': (3072, 384), 'video_embedder.bias': (3072,)}
        for name, (outputs, inputs) in modules.items():
            weights[name + '.lora_A.weight'] = (64, inputs)
            weights[name + '.lora_B.weight'] = (outputs, 64)
        metadata = {'hayate_control': '1', 'base_model': 'moss-base', 'control_channels': 'color,depth',
                    'control_preprocess': 'vae', 'rank': '64', 'alpha': '64.0'}
        self.assertEqual(len(control.plan_weights(weights, modules, metadata, projection_shape=(3072, 128),
                                                  image_layers=42, text_layers=2)), 262)


if __name__ == '__main__':
    unittest.main()
