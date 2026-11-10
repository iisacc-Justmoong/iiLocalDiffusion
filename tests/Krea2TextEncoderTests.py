"""Selection and integrity contracts for the Krea 2 fallback text encoder."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_text_encoder as encoder


class Krea2TextEncoderTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('IILD_VERIFY_KREA2_ENCODER_PAYLOAD') == '1',
                         'Set IILD_VERIFY_KREA2_ENCODER_PAYLOAD=1 for the full resource hash')
    def test_provisioned_encoder_payload_matches_resource_manifest(self):
        from weight_files import file_sha256, file_signature
        resources = ROOT / 'resources'
        manifest = json.loads((resources / 'generation-defaults.json').read_text())
        item = next(entry for entry in manifest['fallback_llms'] if 'krea2' in entry['families'])
        path = resources / item['file']
        if not path.is_file():
            self.skipTest('Provision the local Krea 2 text encoder for its payload check')
        before = file_signature(path)
        self.assertEqual(path.stat().st_size, item['size'])
        self.assertEqual(file_sha256(path), item['sha256'])
        self.assertEqual(file_signature(path), before)

    def setUp(self):
        self.tmp = resource_fixture.TemporaryDirectory(prefix='krea2-text-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.resources = self.root / 'resources'
        self.resources.mkdir()
        weights = self.resources / 'encoder.safetensors'
        weights.write_bytes(b'local resource fixture')
        config = self.resources / 'config.json'
        config.write_text(json.dumps(encoder.EXPECTED_CONFIG))
        item = {'file': weights.name, 'size': weights.stat().st_size,
                'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'families': ['krea2'],
                'class_name': 'Qwen3VLForConditionalGeneration',
                'config': {'file': config.name, 'size': config.stat().st_size,
                           'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}}
        folder = self.resources / 'tokenizer'
        folder.mkdir()
        files = []
        for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
            path = folder / name
            shutil.copyfile(ROOT / 'resources/text_encoders/qwen3vl-4b' / name, path)
            files.append({'file': str(path.relative_to(self.resources)), 'size': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.resources / 'generation-defaults.json').write_text(json.dumps({'version': 1, 'fallback_llms': [item],
            'fallback_tokenizers': [{'families': ['krea2'], 'class_name': 'Qwen2Tokenizer', 'files': files}]}))
        self.args = SimpleNamespace(pipeline_class=None, source_kind='directory', model=str(self.model),
                                    generation_resources=str(self.resources))
        self.index = {'_class_name': 'Krea2Pipeline', 'text_encoder': ['transformers', 'Qwen3VLModel']}

    def test_fallback_selected_for_missing_and_configuration_only_components(self):
        for component in (None, [None, None], ['transformers', 'Qwen3VLModel']):
            self.index['text_encoder'] = component
            selected, status = encoder.resolve(self.args, self.index, self.model)
            self.assertEqual(status, 'fallback')
            self.assertEqual(selected.weight.path, str(self.resources / 'encoder.safetensors'))
            self.assertEqual(selected.config_path, str(self.resources / 'config.json'))
            encoder.verify_selection(selected)

    def test_embedded_component_wins_and_partial_component_is_not_hidden(self):
        component = self.model / 'text_encoder'
        component.mkdir()
        (component / 'model.safetensors.index.json').write_text('{}')
        self.args.generation_resources = str(self.root / 'missing')
        self.assertEqual(encoder.resolve(self.args, self.index, self.model), (None, 'model'))

    def test_embedded_single_file_text_encoder_wins(self):
        source = self.root / 'model.safetensors'
        header = json.dumps({'text_encoders.qwen3vl_4b.transformer.x':
                             {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        source.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
        self.args.source_kind, self.args.model = 'single-file', str(source)
        self.assertEqual(encoder.resolve(self.args, self.index, self.model), (None, 'model'))

    def test_wrong_family_or_incompatible_component_does_not_receive_fallback(self):
        self.index['_class_name'] = 'FluxPipeline'
        self.assertEqual(encoder.resolve(self.args, self.index, self.model), (None, 'not-configured-for-pipeline'))
        self.index['_class_name'] = 'Krea2Pipeline'
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        with self.assertRaisesRegex(ValueError, 'Qwen3VL'):
            encoder.resolve(self.args, self.index, self.model)

    def test_complete_precomputed_embeddings_do_not_require_a_text_encoder(self):
        self.args.generation_resources = str(self.root / 'missing')
        self.args.inputs = {'prompt_embeds': 'fixture', 'guidance_scale': 0}
        self.assertEqual(encoder.resolve(self.args, self.index, self.model), (None, 'precomputed-embeddings'))
        self.args.inputs['guidance_scale'] = 3.5
        self.args.inputs['negative_prompt_embeds'] = 'fixture'
        self.assertEqual(encoder.resolve(self.args, self.index, self.model), (None, 'precomputed-embeddings'))

    def test_config_or_resource_mutation_is_detected(self):
        selection, _ = encoder.resolve(self.args, self.index, self.model)
        Path(selection.config_path).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            encoder.verify_selection(selection)

    def test_metadata_mode_does_not_claim_a_hash(self):
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            selected, _ = encoder.resolve(self.args, self.index, self.model)
            self.assertIsNone(selected.weight.sha256)
            encoder.verify_selection(selected)

    def test_generic_runner_reports_selected_local_encoder(self):
        import generate_any
        index = {**self.index, 'is_distilled': True}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        args = generate_any.build_parser().parse_args(['--model', str(self.model), '--output-dir', str(self.root / 'out'),
                    '--generation-resources', str(self.resources), '--no-default-modifiers'])
        with patch('generate_any.resolve_vae_selection', return_value=(None, 'fixture')):
            result = generate_any.resolve_arguments(args)
        reported = generate_any.configuration(result)['text_encoder']
        self.assertEqual(reported['status'], 'fallback')
        self.assertEqual(reported['weight'], str(self.resources / 'encoder.safetensors'))
        self.assertEqual(reported['config_sha256'], hashlib.sha256((self.resources / 'config.json').read_bytes()).hexdigest())
        self.assertEqual(generate_any.configuration(result)['tokenizer']['status'], 'fallback')
        self.assertEqual(generate_any.configuration(result)['tokenizer']['directory'], str(self.resources / 'tokenizer'))

    def test_ostris_control_loads_full_existing_qwen_encoder(self):
        import generate_any
        index = {**self.index, 'is_distilled': True}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        adapter, image = self.root / 'pose.safetensors', self.root / 'pose.png'
        adapter.write_bytes(b'fixture')
        image.write_bytes(b'fixture')
        parsed = generate_any.build_parser().parse_args(['--model', str(self.model), '--output-dir', str(self.root / 'out'),
            '--generation-resources', str(self.resources), '--no-default-modifiers', '--seed', '1', '--prompt', 'scene',
            '--controlnet-format', 'krea2-ostris-edit', '--controlnet', str(adapter), '--control-image', str(image)])
        with patch.object(generate_any, 'resolve_vae_selection', return_value=(None, 'fixture')):
            result = generate_any.resolve_arguments(parsed)
        self.assertEqual(generate_any.configuration(result)['text_encoder']['class'], 'Qwen3VLModel')
        class Krea2Pipeline:
            def __init__(self, text_encoder=None, tokenizer=None): pass
            def __call__(self, **kwargs): pass
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.loaded_kwargs = kwargs
                return cls()
        with patch.object(generate_any, 'validate_components'), \
             patch.object(generate_any, 'builtin_pipeline', return_value=Krea2Pipeline), \
             patch.object(generate_any, 'validate_base_model', return_value={}), \
             patch.object(generate_any, 'validate_call_arguments'), \
             patch.object(generate_any, 'model_identity', return_value={'directory': str(self.model), 'files': []}), \
             patch.object(generate_any, 'verify_identity'), \
             patch.object(generate_any.krea2_contract, 'validate_pipeline'), \
             patch.object(generate_any.krea2_text_encoder, 'verify_selection'), \
             patch.object(generate_any.krea2_text_encoder, 'load_full', return_value='full-encoder') as load_full, \
             patch.object(generate_any.krea2_text_encoder, 'load') as load_text, \
             patch.object(generate_any.krea2_tokenizer, 'verify_selection'), \
             patch.object(generate_any.krea2_tokenizer, 'load', return_value='tokenizer'):
            generate_any.load_pipeline(result, SimpleNamespace(), 'float32')
        load_full.assert_called_once_with(result.text_encoder_selection, 'float32')
        load_text.assert_not_called()
        self.assertEqual(Krea2Pipeline.loaded_kwargs['text_encoder'], 'full-encoder')

    def test_resource_provisioner_checks_encoder_configuration(self):
        spec = importlib.util.spec_from_file_location('prepare_defaults', ROOT / 'scripts/prepare_generation_defaults.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest_path = self.resources / 'generation-defaults.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['negative_embeddings'] = []
        manifest_path.write_text(json.dumps(manifest))
        module.prepare(self.resources, check=True)
        (self.resources / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            module.prepare(self.resources, check=True)

    def test_provisioned_encoder_header_matches_qwen3_vl_4b(self):
        path = ROOT / 'resources/text_encoders/qwen3vl_4b_bf16.safetensors'
        if not path.is_file():
            self.skipTest('Bundled encoder is not provisioned')
        with path.open('rb') as source:
            length = struct.unpack('<Q', source.read(8))[0]
            self.assertLessEqual(length, 16 * 1024 * 1024)
            header = json.loads(source.read(length))
        base = {'input_layernorm.weight': (2560,), 'post_attention_layernorm.weight': (2560,),
                'mlp.down_proj.weight': (2560, 9728), 'mlp.gate_proj.weight': (9728, 2560),
                'mlp.up_proj.weight': (9728, 2560), 'self_attn.k_norm.weight': (128,),
                'self_attn.q_norm.weight': (128,), 'self_attn.q_proj.weight': (4096, 2560),
                'self_attn.k_proj.weight': (1024, 2560), 'self_attn.v_proj.weight': (1024, 2560),
                'self_attn.o_proj.weight': (2560, 4096)}
        expected = {'embed_tokens.weight': (151936, 2560), 'norm.weight': (2560,)}
        for i in range(36):
            expected.update({f'layers.{i}.{key}': shape for key, shape in base.items()})
        actual = {name[len(encoder.PREFIX):]: tuple(value['shape']) for name, value in header.items()
                  if name.startswith(encoder.PREFIX)}
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
