"""Pinned, explicit-family model configuration selection."""
import json
from pathlib import Path
import shutil
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import generate_any
import model_config_defaults as defaults
import resource_fixture


class ModelConfigDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='model-config-defaults-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.resources = self.root / 'resources'
        self.resources.mkdir()
        (self.resources / 'generation-defaults.json').write_text(json.dumps({'version': 1}))
        shutil.copytree(ROOT / 'resources/model-configs', self.resources / 'model-configs')
        self.checkpoint = self.root / 'model.safetensors'
        self.checkpoint.write_bytes(b'fixture')

    def test_only_explicit_moss_and_krea_variants_select_defaults(self):
        for family, variant, name in (('Krea2Pipeline', 'raw', 'krea2-raw'),
                                      ('Krea2Pipeline', 'turbo', 'krea2-turbo')):
            with self.subTest(family=family, variant=variant):
                self.assertEqual(defaults.resolve(family, variant, self.resources),
                                 self.resources / 'model-configs' / name)
        self.assertIsNone(defaults.resolve('Krea2Pipeline', 'auto', self.resources))
        self.assertIsNone(defaults.resolve('MOSSImagePipeline', 'auto', self.resources))
        self.assertIsNone(defaults.resolve('FluxPipeline', 'turbo', self.resources))
        self.assertEqual(defaults.resolve(None, 'auto', self.resources, preset='moss-image-sft'),
                         self.resources / 'model-configs/moss-image-sft')
        self.assertEqual(defaults.resolve('FluxPipeline', 'auto', self.resources, preset='flux1-schnell'),
                         self.resources / 'model-configs/flux1-schnell')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            defaults.resolve('FluxPipeline', 'auto', self.resources, preset='moss-image-sft')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            defaults.resolve(None, 'raw', self.resources, preset='krea2-turbo')

    def test_refuses_modified_missing_or_escaping_config(self):
        config = self.resources / 'model-configs/krea2-turbo/transformer/config.json'
        config.write_bytes(config.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'pinned identity'):
            defaults.resolve('Krea2Pipeline', 'turbo', self.resources)
        config.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            defaults.resolve('Krea2Pipeline', 'turbo', self.resources)
        outside = self.root / 'outside.json'
        outside.write_text('{}')
        config.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            defaults.resolve('Krea2Pipeline', 'turbo', self.resources)

    def test_missing_external_preset_uses_verified_sdk_config_but_damaged_override_fails(self):
        shutil.rmtree(self.resources / 'model-configs/moss-image-sft')
        bundled = defaults.resolve(None, 'auto', self.resources, preset='moss-image-sft')
        self.assertEqual(bundled, ROOT / 'resources/model-configs/moss-image-sft')
        (self.resources / 'model-configs/moss-image-sft').mkdir()
        with self.assertRaisesRegex(ValueError, 'missing'):
            defaults.resolve(None, 'auto', self.resources, preset='moss-image-sft')

    def test_single_file_suffix_is_case_insensitive_and_accepts_singular_form(self):
        parser = generate_any.build_parser()
        for name in ('model.safetensor', 'model.SAFETENSORS'):
            with self.subTest(name=name):
                source = self.root / name
                source.write_bytes(b'fixture')
                args = parser.parse_args(['--model', str(source), '--model-config-preset', 'moss-image-sft',
                                          '--generation-resources', str(self.resources)])
                with patch.object(generate_any, 'load_model_index', side_effect=RuntimeError('accepted')):
                    with self.assertRaisesRegex(RuntimeError, 'accepted'):
                        generate_any.resolve_arguments(args)

    def test_runner_selects_default_before_loading_and_keeps_explicit_precedence(self):
        parser = generate_any.build_parser()
        def parse(*extra):
            return parser.parse_args(['--model', str(self.checkpoint), '--generation-resources', str(self.resources),
                                      '--output-dir', str(self.root / 'out'), *extra])
        for tokens, expected, status in [
            (('--model-config-preset', 'moss-image-sft'), 'moss-image-sft', 'bundled'),
            (('--model-config-preset', 'flux1-schnell'), 'flux1-schnell', 'bundled'),
            (('--pipeline-class', 'Krea2Pipeline', '--krea2-variant', 'turbo'), 'krea2-turbo', 'bundled'),
            (('--pipeline-class', 'MOSSImagePipeline', '--model-config',
              str(self.resources / 'model-configs/krea2-raw')), 'krea2-raw', 'explicit'),
        ]:
            def inspect(args):
                self.assertEqual(Path(args.model_config).name, expected)
                self.assertEqual(args.model_config_status, status)
                raise RuntimeError('selection inspected')
            with self.subTest(tokens=tokens), patch.object(generate_any.flux1_single_file, 'validate_schnell'), \
                    patch.object(generate_any, 'load_model_index', side_effect=inspect), \
                    self.assertRaisesRegex(RuntimeError, 'selection inspected'):
                generate_any.resolve_arguments(parse(*tokens))
        with self.assertRaisesRegex(ValueError, 'requires --model-config'):
            generate_any.resolve_arguments(parse('--pipeline-class', 'Krea2Pipeline'))
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            generate_any.resolve_arguments(parse('--model-config-preset', 'moss-image-sft', '--model-config',
                                                  str(self.resources / 'model-configs/krea2-raw')))

    def test_flux_preset_checks_public_header_before_loading(self):
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-bf16-transformer-header.json').read_text())['header']
        def write_header(value):
            content = json.dumps(value).encode()
            self.checkpoint.write_bytes(struct.pack('<Q', len(content)) + content)
        parser = generate_any.build_parser()
        args = parser.parse_args(['--model', str(self.checkpoint), '--model-config-preset', 'flux1-schnell',
                                  '--generation-resources', str(self.resources), '--output-dir', str(self.root / 'out')])
        write_header(header)
        with patch.object(generate_any, 'load_model_index', side_effect=RuntimeError('header accepted')):
            with self.assertRaisesRegex(RuntimeError, 'header accepted'):
                generate_any.resolve_arguments(args)
        write_header({**header, 'guidance_in.in_layer.bias': {'dtype': 'BF16', 'shape': [3072]}})
        with patch.object(generate_any, 'load_model_index') as loader:
            with self.assertRaisesRegex(ValueError, 'tensor names'):
                generate_any.resolve_arguments(args)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
