"""Offline selection and integrity for the MOSS INT8 encoder resource."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_text_encoder
import moss_tokenizer
import moss_runtime
import resource_fixture


class MossEncoderFallbackTests(unittest.TestCase):
    def test_official_moss_sft_configuration_is_pinned_and_accepted(self):
        folder = ROOT / 'resources/model-configs/moss-image-sft'
        pinned = {
            'model_index.json': (412, 'ca80b1f562a506e0eb24afb1acb464fb583163e3c240c04e4bc69fea090db173'),
            'scheduler/scheduler_config.json': (494, '707322832cc507dfb8c51c0e0cd04d44ee2d30a9b036b6f3c47d6c17aa71adf0'),
            'transformer/config.json': (240, '3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666'),
        }
        for name, (size, digest) in pinned.items():
            data = (folder / name).read_bytes()
            self.assertEqual(len(data), size, name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)
        moss_runtime.validate_index(json.loads((folder / 'model_index.json').read_text()))
        config = json.loads((folder / 'transformer/config.json').read_text())
        self.assertEqual((config['dim'], config['num_layers'], config['text_dim']), (3072, 42, 4096))

    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='moss-encoder-fallback-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.package, self.resources = root / 'package', root / 'resources'
        self.package.mkdir()
        self.resources.mkdir()
        self.data = b'synthetic-moss-encoder'
        item = {'file': 'text_encoders/moss-int8.safetensors', 'format': 'int8-tensorwise-convrot',
                'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.path = self.resources / item['file']
        self.path.parent.mkdir()
        self.path.write_bytes(self.data)
        self.catalog = self.resources / 'generation-defaults.json'
        self.catalog.write_text(json.dumps({'version': 1, 'fallback_moss_text_weights': item}))
        self.args = SimpleNamespace(pipeline_class=None, generation_resources=self.resources)
        self.index = {'_class_name': 'MOSSImagePipeline', 'text_encoder': ['transformers', 'Qwen3_5Model']}

    def test_selects_verified_fallback_and_rejects_tampering_even_in_metadata_mode(self):
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata', 'IILD_MODEL_HASH_CACHE': 'off'}):
            selected, status = moss_text_encoder.resolve_fallback(self.args, self.index, self.package)
            self.assertEqual(status, 'fallback-int8-convrot')
            self.assertEqual(selected.sha256, hashlib.sha256(self.data).hexdigest())
            moss_text_encoder.verify_weight_file(selected, 'MOSS')
            self.path.write_bytes(b'x' * len(self.data))
            with self.assertRaisesRegex(ValueError, 'differs'):
                moss_text_encoder.resolve_fallback(self.args, self.index, self.package)

    def test_package_weights_take_precedence_and_missing_is_unavailable(self):
        component = self.package / 'text_encoder'
        component.mkdir()
        (component / 'model-00001-of-00002.safetensors').write_bytes(b'partial')
        self.assertEqual(moss_text_encoder.resolve_fallback(self.args, self.index, self.package), (None, 'model'))
        (component / 'model-00001-of-00002.safetensors').unlink()
        self.path.unlink()
        self.assertEqual(moss_text_encoder.resolve_fallback(self.args, self.index, self.package), (None, 'unavailable'))
        self.index['_class_name'] = 'FluxPipeline'
        self.assertEqual(moss_text_encoder.resolve_fallback(self.args, self.index, self.package),
                         (None, 'not-configured-for-pipeline'))

    def test_rejects_path_escape_and_incompatible_component(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['fallback_moss_text_weights']['file'] = '../outside.safetensors'
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            moss_text_encoder.resolve_fallback(self.args, self.index, self.package)
        self.index['text_encoder'] = ['transformers', 'WrongModel']
        with self.assertRaisesRegex(ValueError, 'declaration'):
            moss_text_encoder.resolve_fallback(self.args, self.index, self.package)

    def test_absent_society_components_use_verified_sdk_catalog(self):
        sdk = self.resources.parent / 'sdk-resources'
        sdk.mkdir()
        encoder = sdk / 'encoder.safetensors'
        encoder.write_bytes(self.data)
        config = sdk / 'qwen/config.json'
        config.parent.mkdir()
        config.write_text(json.dumps({'model_type': 'qwen3_5'}))
        tokenizer = sdk / 'tokenizer'
        tokenizer.mkdir()
        files = []
        for name in sorted(moss_tokenizer.NAMES):
            path = tokenizer / name
            data = ('fixture ' + name).encode()
            path.write_bytes(data)
            files.append({'file': str(path.relative_to(sdk)), 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        config_data = config.read_bytes()
        manifest = {'version': 1,
                    'fallback_moss_text_weights': {'file': 'encoder.safetensors', 'format': 'int8-tensorwise-convrot',
                        'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()},
                    'fallback_moss_text_config': {'file': 'qwen/config.json', 'size': len(config_data),
                                                  'sha256': hashlib.sha256(config_data).hexdigest()},
                    'fallback_tokenizers': [{'families': ['moss-image'], 'class_name': 'Qwen2Tokenizer', 'files': files}]}
        (sdk / 'generation-defaults.json').write_text(json.dumps(manifest))
        self.catalog.write_text(json.dumps({'version': 1}))
        index = {**self.index, 'tokenizer': ['transformers', 'Qwen2Tokenizer']}
        with patch('generation_defaults.sdk_resource_directory', return_value=sdk), \
                patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata', 'IILD_MODEL_HASH_CACHE': 'off'}):
            selected, status = moss_text_encoder.resolve_fallback(self.args, index, self.package)
            self.assertEqual(Path(selected.resolved_file), encoder)
            self.assertEqual(status, 'fallback-int8-convrot')
            self.assertEqual(Path(moss_text_encoder.resolve_config(self.package, self.resources).path), config)
            selected_tokenizer, tokenizer_status = moss_tokenizer.resolve(self.args, index, self.package)
            self.assertEqual(Path(selected_tokenizer.directory), tokenizer)
            self.assertEqual(tokenizer_status, 'fallback')


if __name__ == '__main__':
    unittest.main()
