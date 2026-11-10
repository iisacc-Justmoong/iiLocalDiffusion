"""Offline selection of the existing FLUX.1 split encoder resources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import flux_text_encoder as encoder
import resource_fixture


class FluxTextEncoderTests(unittest.TestCase):
    def test_public_schnell_single_file_component_detection(self):
        import vae_defaults
        import flux1_single_file
        with resource_fixture.TemporaryDirectory(prefix='flux-schnell-header-') as temporary:
            target = Path(temporary) / 'model.safetensors'
            for fixture, expected in (('flux1-schnell-bf16-transformer-header.json', False),
                                      ('flux1-schnell-fp8-full-header.json', True)):
                header = json.loads((ROOT / 'resources/test-fixtures' / fixture).read_text())['header']
                encoded = json.dumps(header).encode()
                target.write_bytes(struct.pack('<Q', len(encoded)) + encoded)
                with self.subTest(fixture=fixture):
                    self.assertEqual(encoder._single_file_has_encoder(target, 'text_encoder'), expected)
                    self.assertEqual(encoder._single_file_has_encoder(target, 'text_encoder_2'), expected)
                    self.assertEqual(vae_defaults.single_file_has_vae(target), expected)
                    flux1_single_file.validate_schnell(target, ROOT / 'resources/model-configs/flux1-schnell')

    def test_schnell_preset_rejects_mismatched_or_scaled_transformers(self):
        import flux1_single_file
        header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-bf16-transformer-header.json').read_text())['header']
        config = ROOT / 'resources/model-configs/flux1-schnell'
        key = 'img_in.weight'
        variants = []
        broken = dict(header)
        broken[key] = {**header[key], 'shape': [3072, 128]}
        variants.append(broken)
        broken = dict(header)
        broken[key] = {**header[key], 'dtype': 'I8'}
        variants.append(broken)
        variants.append({**header, 'guidance_in.in_layer.bias': {'dtype': 'BF16', 'shape': [3072]}})
        variants.append({**header, 'scaled_fp8': {'dtype': 'F8_E4M3', 'shape': [0]}})
        variants.append({**header, 'model.diffusion_model.' + key: header[key]})
        with resource_fixture.TemporaryDirectory(prefix='flux-schnell-invalid-') as temporary:
            target = Path(temporary) / 'model.safetensors'
            for broken in variants:
                payload = json.dumps(broken).encode()
                target.write_bytes(struct.pack('<Q', len(payload)) + payload)
                with self.subTest(index=variants.index(broken)), self.assertRaises(ValueError):
                    flux1_single_file.validate_schnell(target, config)

    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='flux-encoder-selection-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.resources = self.root / 'resources'
        self.resources.mkdir()
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
        for item in manifest['fallback_flux_text_configs']:
            dst = self.resources / item['file']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'resources' / item['file'], dst)
        for item in manifest['fallback_flux_text_weights']:
            dst = self.resources / item['file']
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = (item['component'] + '-fixture').encode()
            dst.write_bytes(data)
            item['size'] = len(data)
            item['sha256'] = hashlib.sha256(data).hexdigest()
        (self.resources / 'generation-defaults.json').write_text(json.dumps(manifest))
        self.package = self.root / 'model'
        self.package.mkdir()
        self.args = SimpleNamespace(pipeline_class=None, source_kind='directory', model=str(self.package),
                                    generation_resources=self.resources, inputs={})
        self.index = {'_class_name': 'FluxPipeline', 'text_encoder': ['transformers', 'CLIPTextModel'],
                      'text_encoder_2': ['transformers', 'T5EncoderModel']}

    def test_fallback_when_encoder_is_missing(self):
        selection, status = encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder')
        self.assertEqual(status, 'fallback')
        self.assertEqual(selection.weight_status, 'fallback')
        self.assertEqual(selection.config_status, 'fallback')
        encoder.verify_selection(selection)
        self.assertEqual(selection.metadata()['status'], 'fallback')

    def test_existing_partial_encoder_and_precomputed_embeddings_take_precedence(self):
        component = self.package / 'text_encoder'
        component.mkdir()
        (component / 'model.safetensors').write_bytes(b'partial')
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder'), (None, 'model'))
        self.args.inputs = {'prompt_embeds': {}, 'pooled_prompt_embeds': {}}
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder_2'),
                         (None, 'precomputed-embeddings'))
        self.args.inputs['negative_prompt'] = 'blur'
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder_2')[1], 'fallback')

    def test_embedded_single_file_encoder_is_never_replaced(self):
        header = json.dumps({'text_encoders.t5xxl.transformer.shared.weight': {}}).encode()
        checkpoint = self.root / 'checkpoint.safetensors'
        checkpoint.write_bytes(struct.pack('<Q', len(header)) + header)
        self.args.source_kind, self.args.model = 'single-file', str(checkpoint)
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder_2'), (None, 'model'))
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder')[1], 'fallback')

    def test_generic_runner_selects_embedded_t5_for_cpu_streaming(self):
        import generate_any
        component = self.package / 'text_encoder_2'
        component.mkdir()
        (component / 'config.json').write_text(json.dumps({'model_type': 't5'}))
        (self.package / 'model_index.json').write_text(json.dumps({
            '_class_name': 'FluxPipeline', 'text_encoder': [None, None],
            'text_encoder_2': ['transformers', 'T5EncoderModel']}))
        header = json.dumps({'text_encoders.t5xxl.transformer.shared.weight':
                             {'dtype': 'F8_E4M3', 'shape': [1, 1], 'data_offsets': [0, 1]}}).encode()
        checkpoint = self.root / 'combined.safetensors'
        checkpoint.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        parser = generate_any.build_parser()
        args = parser.parse_args(['--model', str(checkpoint), '--model-config', str(self.package),
                                  '--flux-t5-fp8-storage', 'cpu-stream', '--no-default-modifiers',
                                  '--output-dir', str(self.root / 'out')])
        with patch.object(generate_any, 'resolve_vae_selection', return_value=(None, 'model')):
            result = generate_any.resolve_arguments(args)
        self.assertEqual(result.flux_text_encoder_status['text_encoder_2'], 'embedded')
        self.assertEqual(result.flux_text_encoders['text_encoder_2'].namespace, 'text_encoders.t5xxl.transformer.')
        self.assertEqual(result.flux_text_encoders['text_encoder_2'].weight.path, str(checkpoint))
        self.assertEqual(result.flux_text_encoders['text_encoder_2'].metadata()['mode'], 'embedded-single-file')
        override = self.root / 'explicit.safetensors'
        override.write_bytes(b'fixture')
        explicit_args = parser.parse_args(['--model', str(checkpoint), '--model-config', str(self.package),
            '--flux-t5-encoder', str(override), '--flux-t5-fp8-storage', 'cpu-stream', '--no-default-modifiers',
            '--output-dir', str(self.root / 'out')])
        with patch.object(generate_any, 'resolve_vae_selection', return_value=(None, 'model')):
            explicit = generate_any.resolve_arguments(explicit_args)
        self.assertEqual(explicit.flux_text_encoder_status['text_encoder_2'], 'explicit')
        self.assertEqual(explicit.flux_text_encoders['text_encoder_2'].namespace, '')
        ambiguous = json.loads(header)
        ambiguous['text_encoders.t5xxl.scaled_fp8'] = {'dtype': 'F8_E4M3', 'shape': [0], 'data_offsets': [1, 1]}
        encoded = json.dumps(ambiguous).encode()
        checkpoint.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b'x')
        with patch.object(generate_any, 'resolve_vae_selection', return_value=(None, 'model')):
            with self.assertRaisesRegex(ValueError, 'ambiguous quantization'):
                generate_any.resolve_arguments(args)
        incompatible = json.dumps({'text_encoder_2.shared.weight': {'dtype': 'F8_E4M3',
                                    'shape': [1, 1], 'data_offsets': [0, 1]}}).encode()
        checkpoint.write_bytes(struct.pack('<Q', len(incompatible)) + incompatible + b'x')
        with patch.object(generate_any, 'resolve_vae_selection', return_value=(None, 'model')):
            with self.assertRaisesRegex(ValueError, 'Comfy T5 tensor namespace'):
                generate_any.resolve_arguments(args)

    def test_missing_or_tampered_resource_is_not_silently_accepted(self):
        catalog = json.loads((self.resources / 'generation-defaults.json').read_text())
        path = self.resources / catalog['fallback_flux_text_weights'][0]['file']
        path.unlink()
        self.assertEqual(encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder'), (None, 'unavailable'))
        path.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'differs'):
            encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder')

    def test_metadata_only_policy_still_hashes_pinned_fallback_weights(self):
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata', 'IILD_MODEL_HASH_CACHE': 'off'}):
            selected, _ = encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder')
            self.assertEqual(selected.metadata()['validation'], 'sha256')
            path = Path(selected.weight.path)
            path.write_bytes(b'x' * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'differs'):
                encoder.resolve_fallback(self.args, self.index, self.package, 'text_encoder')


if __name__ == '__main__':
    unittest.main()
