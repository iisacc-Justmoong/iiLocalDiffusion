#!/usr/bin/env python3
"""User-facing Civitai routing and derived-preset argument contracts."""
import resource_fixture

import argparse
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace
import unittest
from local_model_fixture import local_parser, MODEL
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iild_generation_router_tests", ROOT / "reference" / "generate.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)
import generate
import presets


class GenerationRouterTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = resource_fixture.TemporaryDirectory(prefix="generation-router-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def backend(self, base=None, backend="auto", *flags):
        return router.select_backend(argparse.Namespace(base_model=base, backend=backend), list(flags))

    def test_named_derivatives_choose_presets(self):
        for name in ("Illustrious", "NoobAI", "Pony", "Flux.1 Krea"):
            with self.subTest(name=name):
                self.assertEqual(self.backend(name), "preset")

    def test_other_architectures_and_explicit_workflows_route(self):
        self.assertEqual(self.backend("SD 3.5", "diffusers"), "diffusers")
        self.assertEqual(self.backend("Other", "auto", "--workflow=x.json"), "comfyui")
        self.assertEqual(self.backend("Illustrious", "auto", "--pipeline-inputs", "{}"), "diffusers")
        with self.assertRaises(ValueError):
            self.backend("Other")

    def test_model_config_preset_selects_generic_diffusers(self):
        self.assertEqual(self.backend(None, 'auto', '--model-config-preset', 'krea2-turbo'), 'diffusers')
        self.assertEqual(self.backend(None, 'auto', '--model-config-preset=moss-image-sft'), 'diffusers')

    def test_explicit_pytorch_dequantization_routes_to_diffusers(self):
        self.assertEqual(self.backend(None, 'auto', '--dequantize-pytorch'), 'diffusers')
        self.assertEqual(self.backend(None, 'auto', '--no-dequantize-pytorch'), 'diffusers')
        configuration = self.directory / 'quantized-request.json'
        configuration.write_text(json.dumps({'dequantize_pytorch': True}))
        self.assertEqual(self.backend(None, 'auto', '--config', str(configuration)), 'diffusers')
        configuration.write_text(json.dumps({'dequantize_pytorch': False}))
        self.assertEqual(self.backend(None, 'auto', '--config', str(configuration)), 'diffusers')

    def test_json_model_config_routes_gguf_to_diffusers(self):
        source = self.directory / 'model.gguf';source.write_bytes(b'GGUF' + bytes(64))
        configuration = self.directory / 'request.json'
        configuration.write_text(json.dumps({'model': str(source), 'model_config': 'local-config'}))
        self.assertEqual(self.backend(None, 'auto', '--config', str(configuration)), 'diffusers')
        configuration.write_text(json.dumps({'model': 'model.gguf', 'model_config': 'local-config'}))
        args = argparse.Namespace(base_model=None, backend='auto')
        self.assertEqual(router.select_backend(args, ['--config', str(configuration)]), 'diffusers')
        self.assertEqual(args._configuration['model'], str(source))

    def test_gguf_component_file_routes_by_adjacent_pipeline_without_overriding_native(self):
        root = self.directory / 'pipeline';root.mkdir()
        (root / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline',
                                                          'transformer': ['diffusers', 'FluxTransformer2DModel']}))
        component = root / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'model.gguf';source.write_bytes(b'GGUF' + bytes(64))
        self.assertEqual(self.backend(None, 'auto', '--model', str(source)), 'diffusers')
        self.assertEqual(self.backend(None, 'auto', '--model', str(source), '--engine', 'native'), 'local')
        (root / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline',
                                                          'transformer': ['other', 'Transformer']}))
        self.assertEqual(self.backend(None, 'auto', '--model', str(source)), 'local')
        other = self.directory / 'other.gguf';other.write_bytes(b'GGUF' + bytes(64))
        self.assertEqual(self.backend(None, 'auto', '--model', str(other)), 'local')

    def test_directory_variant_reaches_generic_runner(self):
        self.assertEqual(self.backend('Illustrious', 'auto', '--variant', 'fp16'), 'diffusers')
        model = self.directory / 'directory-model';model.mkdir()
        (model / 'model_index.json').write_text(json.dumps({'_class_name': 'FakePipeline'}))
        calls = []
        with patch.object(router.importlib, 'import_module',
                          return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)):
            self.assertEqual(router.main(['--model', str(model), '--variant', 'fp16']), 0)
        self.assertEqual(calls[0][calls[0].index('--variant') + 1], 'fp16')
        self.assertEqual(calls[0][calls[0].index('--model') + 1], str(model))

    def test_only_native_foreground_moss_preparation_is_deferred(self):
        with patch.object(router, '_is_moss_checkpoint', return_value=True) as inspect:
            self.assertTrue(router.defer_unsupported_foreground(['--model-path', 'moss.safetensors', '--backend', 'local']))
            self.assertTrue(router.defer_unsupported_foreground(['--model=moss.safetensors', '--backend=local']))
            self.assertFalse(router.defer_unsupported_foreground(['--model', 'moss.safetensors', '--backend', 'diffusers']))
            self.assertFalse(router.defer_unsupported_foreground(['--model', 'moss.safetensors']))
            inspect.assert_any_call('moss.safetensors')
        with patch.object(router, '_is_moss_checkpoint', return_value=False):
            self.assertFalse(router.defer_unsupported_foreground(['--model', 'other.safetensors', '--backend', 'local']))

    def test_foreground_does_not_convert_a_legacy_checkpoint(self):
        path = self.directory / 'legacy.safetensors'
        path.write_bytes(b'PK\x03\x04' + bytes(20))
        with patch.object(router, '_is_moss_checkpoint', side_effect=AssertionError('no model inspection')):
            for backend in ([], ['--backend', 'auto'], ['--backend', 'local'], ['--backend', 'diffusers']):
                with self.subTest(backend=backend):
                    self.assertTrue(router.defer_unsupported_foreground(['--model', str(path), *backend]))

    def test_pytorch_moss_routes_by_weights_only_tensor_identity(self):
        import downloaded_model
        path = self.directory / 'model.safetensors'
        path.write_bytes(b'PK\x03\x04' + bytes(20))
        shapes = {'video_embedder.weight': (4, 2), 'final_linear_video.weight': (2, 4),
                  'layers.0.proj.qkv.weight': (12, 4)}
        with patch.object(downloaded_model, '_torch_checkpoint', return_value=(shapes, {}, [])):
            self.assertEqual(self.backend(None, 'auto', '--model', str(path)), 'diffusers')
            self.assertEqual(self.backend(None, 'local', '--model', str(path)), 'local')
            calls = []
            with patch.object(router.importlib, 'import_module',
                              return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)):
                self.assertEqual(router.main(['--model', str(path), '--prompt', 'kept',
                                              '--num-images', '2', '--no-hires-fix']), 0)
            self.assertIn('moss-image-sft', calls[0])
            self.assertEqual(calls[0][calls[0].index('--model') + 1], str(path))
            self.assertEqual(calls[0][calls[0].index('--prompt') + 1], 'kept')

    def test_auto_moss_router_does_not_inspect_the_same_checkpoint_twice(self):
        path = self.directory / 'moss.ckpt'
        path.write_bytes(b'PK\x03\x04' + bytes(20))
        calls = []
        with patch.object(router, '_is_moss_checkpoint', return_value=True) as detect, \
             patch.object(router.importlib, 'import_module',
                          return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)):
            self.assertEqual(router.main(['--model', str(path), '--prompt', 'kept']), 0)
        detect.assert_called_once_with(str(path))
        self.assertIn('moss-image-sft', calls[0])

    def make_nested_moss_package(self):
        source = self.directory / 'moss-package-input'
        def weights(path, tensors):
            path.parent.mkdir(parents=True, exist_ok=True)
            header, payload = {}, bytearray()
            for name, shape in tensors:
                count = 4
                for dimension in shape:
                    count *= dimension
                header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [len(payload), len(payload) + count]}
                payload.extend(bytes(count))
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        weights(source / 'main.safetensors', [('unrelated.weight', [1])])
        weights(source / 'Checkpoint/part-1.safetensors', [('video_embedder.weight', [4, 2])])
        weights(source / 'Checkpoint/part-2.safetensors', [('final_linear_video.weight', [2, 4]),
                                                            ('layers.0.proj.qkv.weight', [12, 4])])
        (source / 'Checkpoint/model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'video_embedder.weight': 'part-1.safetensors', 'final_linear_video.weight': 'part-2.safetensors',
            'layers.0.proj.qkv.weight': 'part-2.safetensors'}}))
        weights(source / 'LLM/text.safetensors', [('model.language_model.embed_tokens.weight', [1]),
                                                  ('model.language_model.layers.0.weight', [1])])
        weights(source / 'VAE/vae.safetensors', [('encoder.conv_in.weight', [1]), ('decoder.conv_out.weight', [1])])
        package = self.directory / 'nested-moss.safetensors'
        command = [str(ROOT / 'build/iild-model-package'), 'create', '--input', str(source), '--output', str(package),
                   '--assign-role', 'main.safetensors=model', '--assign-position', 'main.safetensors=main',
                   '--assign-role', 'Checkpoint/part-1.safetensors=model',
                   '--assign-position', 'Checkpoint/part-1.safetensors=component',
                   '--assign-role', 'LLM/text.safetensors=text_encoder', '--assign-role', 'VAE/vae.safetensors=video_vae']
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        return package

    def make_nested_anima_package(self):
        source = self.directory / 'anima-package-input'
        def weights(path, names):
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {name: {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [4 * i, 4 * i + 4]}
                      for i, name in enumerate(names)}
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bytes(4 * len(names)))
        weights(source / 'main.safetensors', ['unrelated.weight'])
        weights(source / 'Checkpoint/part-1.safetensors', ['model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight'])
        weights(source / 'Checkpoint/part-2.safetensors', ['model.diffusion_model.blocks.0.self_attn.q_proj.weight'])
        (source / 'Checkpoint/model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight': 'part-1.safetensors',
            'model.diffusion_model.blocks.0.self_attn.q_proj.weight': 'part-2.safetensors'}}))
        weights(source / 'LLM/text.safetensors', ['model.embed_tokens.weight', 'model.norm.weight', 'model.layers.0.weight'])
        weights(source / 'VAE/vae.safetensors', ['encoder.conv1.weight', 'encoder.head.2.weight',
                                                  'decoder.conv1.weight', 'decoder.head.2.weight'])
        package = self.directory / 'nested-anima.safetensors'
        command = [str(ROOT / 'build/iild-model-package'), 'create', '--input', str(source), '--output', str(package),
                   '--assign-role', 'main.safetensors=model', '--assign-position', 'main.safetensors=main',
                   '--assign-role', 'Checkpoint/part-1.safetensors=model',
                   '--assign-position', 'Checkpoint/part-1.safetensors=component',
                   '--assign-role', 'LLM/text.safetensors=text_encoder', '--assign-role', 'VAE/vae.safetensors=video_vae']
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        return package

    def make_generic_package(self, family='sdxl', sharded=False, ambiguous=False, separate=False, selected_main=False,
                             recognizable_encoders=False):
        source = self.directory / ('generic-' + family + ('-shards' if sharded else '-single')
                                   + ('-split' if separate else '') + ('-selected' if selected_main else ''))
        def weights(path, tensors):
            path.parent.mkdir(parents=True, exist_ok=True)
            header, payload = {}, bytearray()
            for name, shape in tensors:
                size = 4
                for dimension in shape:
                    size *= dimension
                header[name] = {'dtype': 'F32', 'shape': shape,
                                'data_offsets': [len(payload), len(payload) + size]}
                payload.extend(bytes(size))
            raw = json.dumps(header, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
        if family == 'sdxl':
            first = [('model.diffusion_model.input_blocks.0.0.weight', [1, 4, 1, 1])]
            second = [('model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight', [1, 2048]),
                      ('conditioner.embedders.0.transformer.weight', [1]),
                      ('conditioner.embedders.1.model.weight', [1]),
                      ('first_stage_model.encoder.conv_in.weight', [1])]
        else:
            first = [('img_in.weight', [1, 64])]
            second = [('double_blocks.0.img_attn.norm.key_norm.scale', [1]),
                      ('text_encoders.clip_l.weight', [1]), ('text_encoders.t5xxl.weight', [1]),
                      ('first_stage_model.encoder.conv_in.weight', [1])]
        if separate:
            second = second[:1]
        weights(source / 'main.safetensors', [('unrelated.weight', [1])])
        weights(source / 'LoRA/adapter.safetensors', [('lora_A.weight', [1, 1]), ('lora_B.weight', [1, 1])])
        weights(source / 'VAE/one.safetensors', [('encoder.conv_in.weight', [1]), ('decoder.conv_out.weight', [1])])
        weights(source / 'VAE/two.safetensors', [('encoder.conv_in.weight', [1]), ('decoder.conv_out.weight', [1]),
                                                ('extra.weight', [1])])
        if separate:
            if recognizable_encoders:
                common = [('text_model.embeddings.position_embedding.weight', [1, 1]),
                          ('text_model.encoder.layers.0.layer_norm1.weight', [1])]
                weights(source / 'Encoders/l.safetensors', common + [('text_model.embeddings.token_embedding.weight', [1, 2])])
                weights(source / 'Encoders/g.safetensors', common + [('text_model.embeddings.token_embedding.weight', [1, 3])])
                if family == 'flux':
                    weights(source / 'Encoders/t5.safetensors', [('shared.weight', [1, 1]),
                        ('encoder.block.0.layer.0.SelfAttention.q.weight', [1, 1])])
            else:
                weights(source / 'Encoders/l.safetensors', [('text_model.l.weight', [1])])
                weights(source / 'Encoders/g.safetensors', [('text_model.g.weight', [1])])
        if sharded:
            weights(source / 'Checkpoint/part-1.safetensors', first)
            weights(source / 'Checkpoint/part-2.safetensors', second)
            weight_map = {name: 'part-1.safetensors' for name, _ in first}
            weight_map.update({name: 'part-2.safetensors' for name, _ in second})
            (source / 'Checkpoint/model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
            model = 'Checkpoint/part-1.safetensors'
            selected = 'Checkpoint/model.safetensors.index.json'
        else:
            model = selected = 'Checkpoint/model.safetensors'
            weights(source / model, first + second)
        if ambiguous:
            weights(source / 'Checkpoint/another.safetensors', first + second + [('extra.weight', [1])])
        package = self.directory / (source.name + '.safetensors')
        command = [str(ROOT / 'build/iild-model-package'), 'create', '--input', str(source), '--output', str(package),
                   '--assign-role', 'main.safetensors=model', '--assign-position', 'main.safetensors=' + ('component' if selected_main else 'main'),
                   '--assign-role', model + '=model', '--assign-position', model + ('=main' if selected_main else '=component')]
        if ambiguous:
            command += ['--assign-role', 'Checkpoint/another.safetensors=model',
                        '--assign-position', 'Checkpoint/another.safetensors=component']
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        return package, selected

    def make_multi_anima_package(self, multiple_encoders=False):
        source = self.directory / ('multi-anima-encoders' if multiple_encoders else 'multi-anima')
        def weights(path, names):
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {name: {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [i * 4, (i + 1) * 4]}
                      for i, name in enumerate(names)}
            raw = json.dumps(header, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(len(names) * 4))
        model = ['model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight',
                 'model.diffusion_model.blocks.0.self_attn.q_proj.weight']
        weights(source / 'Checkpoint/one.safetensors', model)
        weights(source / 'Checkpoint/two.safetensors', model + ['other.weight'])
        encoder = ['model.embed_tokens.weight', 'model.norm.weight', 'model.layers.0.weight']
        weights(source / 'Encoders/one.safetensors', encoder)
        if multiple_encoders:
            weights(source / 'Encoders/two.safetensors', encoder + ['other.weight'])
        weights(source / 'VAE/vae.safetensors', ['encoder.conv1.weight', 'encoder.head.2.weight',
                                                  'decoder.conv1.weight', 'decoder.head.2.weight'])
        package = self.directory / (source.name + '.safetensors')
        subprocess.run([str(ROOT / 'build/iild-model-package'), 'create', '--input', str(source), '--output', str(package),
            '--assign-role', 'Checkpoint/one.safetensors=model', '--assign-position', 'Checkpoint/one.safetensors=main',
            '--assign-role', 'Checkpoint/two.safetensors=model', '--assign-position', 'Checkpoint/two.safetensors=component',
            '--assign-role', 'Encoders/one.safetensors=text_encoder', '--assign-role', 'VAE/vae.safetensors=video_vae'],
            capture_output=True, text=True, check=True, timeout=30)
        return package

    def make_multi_moss_package(self, multiple_encoders=False, conflicting=False):
        source = self.directory / ('multi-moss-encoders' if multiple_encoders else 'multi-moss')
        if conflicting: source = self.directory / 'multi-moss-conflicting'
        def weights(path, tensors):
            path.parent.mkdir(parents=True, exist_ok=True)
            header, payload = {}, bytearray()
            for name, shape in tensors:
                size = 4
                for dim in shape: size *= dim
                header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [len(payload), len(payload) + size]}
                payload.extend(bytes(size))
            raw = json.dumps(header, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
        model = [('video_embedder.weight', [4, 2]), ('final_linear_video.weight', [2, 4]),
                 ('layers.0.proj.qkv.weight', [12, 4])]
        if conflicting:
            model += [('model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight', [1, 1]),
                      ('model.diffusion_model.blocks.0.self_attn.q_proj.weight', [1, 1])]
        weights(source / 'Checkpoint/one.safetensors', model)
        weights(source / 'Checkpoint/two.safetensors', model + [('other.weight', [1])])
        encoder = [('model.language_model.embed_tokens.weight', [1]), ('model.language_model.layers.0.weight', [1])]
        weights(source / 'Encoders/one.safetensors', encoder)
        if multiple_encoders:
            weights(source / 'Encoders/two.safetensors', encoder + [('other.weight', [1])])
        weights(source / 'VAE/vae.safetensors', [('encoder.conv_in.weight', [1]), ('decoder.conv_out.weight', [1])])
        package = self.directory / (source.name + '.safetensors')
        subprocess.run([str(ROOT / 'build/iild-model-package'), 'create', '--input', str(source), '--output', str(package),
            '--assign-role', 'Checkpoint/one.safetensors=model', '--assign-position', 'Checkpoint/one.safetensors=main',
            '--assign-role', 'Checkpoint/two.safetensors=model', '--assign-position', 'Checkpoint/two.safetensors=component',
            '--assign-role', 'Encoders/one.safetensors=text_encoder', '--assign-role', 'VAE/vae.safetensors=video_vae'],
            capture_output=True, text=True, check=True, timeout=30)
        return package

    def test_generic_package_inspection_is_per_source_and_understands_shards(self):
        for family, expected in (('sdxl', 'sdxl'), ('flux', 'flux1-schnell')):
            for sharded in (False, True):
                with self.subTest(family=family, sharded=sharded):
                    package, selected = self.make_generic_package(family, sharded)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(router.main(['--inspect-model', '--model', str(package)]), 0)
                    report = json.loads(output.getvalue())
                    self.assertEqual((report['architecture'], report['role']), (expected, 'checkpoint'))
                    self.assertEqual(report['embedded_package']['transformer'], selected)
                    self.assertTrue(report['backend_plan']['ready_for_loading'])
                    self.assertFalse((self.directory / 'cache').exists())

    def test_standalone_shards_route_to_native_and_revalidate_every_source(self):
        import standalone_image
        from weight_files import verify_weight_file
        package, selected = self.make_generic_package('sdxl', sharded=True)
        index = package.with_suffix('') / selected
        received = []
        def resolve(tokens):
            _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(tokens))
            received.append(args)
            return 0
        with patch.dict('os.environ', {'IILD_MODEL_VALIDATION': 'metadata', 'IILD_MODEL_HASH_CACHE': 'off'}), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model-path', str(index), '--prompt', 'kept',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        args = received[0]
        self.assertEqual(args.architecture, 'sdxl')
        self.assertEqual(Path(args.model), index.resolve())
        weight = args.model_selection.single_file
        self.assertEqual(len(weight.shards), 2)
        verify_weight_file(weight, 'model')
        shard = Path(weight.shards[0].path)
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_weight_file(weight, 'model')

    def test_standalone_shards_reject_explicit_diffusers_single_file_route(self):
        import standalone_image
        package, selected = self.make_generic_package('sdxl', sharded=True)
        index = package.with_suffix('') / selected
        args = standalone_image.build_parser().parse_args(['--model', str(index), '--prompt', 'kept', '--engine', 'diffusers'])
        with self.assertRaisesRegex(ValueError, 'shard'):
            standalone_image.resolve_arguments(args)

    def test_generic_package_prepares_selected_source_before_calling_backend(self):
        import image_package
        package, selected = self.make_generic_package('sdxl', True)
        original = package.read_bytes()
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append((tokens, dict(image_package.current()))) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'keep',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens, origin = calls[0]
        prepared = Path(tokens[tokens.index('--model') + 1])
        self.assertNotEqual(prepared, package)
        self.assertTrue(prepared.is_file())
        self.assertEqual(origin['transformer'], selected)
        self.assertEqual(origin['source'], str(package))
        self.assertEqual(package.read_bytes(), original)

    def test_generic_package_does_not_silently_choose_between_models(self):
        package, _ = self.make_generic_package('sdxl', False, True)
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit) as failure:
            router.main(['--inspect-model', '--model', str(package)])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn('Multiple image checkpoints', output.getvalue())

    def test_generic_package_allows_an_explicit_checkpoint_selection(self):
        import image_package
        from downloaded_model import inspect_downloaded_model
        from inference_session import InferenceSession
        package, _ = self.make_generic_package('sdxl', False, True)
        selected = 'Checkpoint/another.safetensors'
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(router.main(['--inspect-model', '--model', str(package), '--package-model', selected]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['embedded_package']['transformer'], selected)
        with InferenceSession():
            for path in (selected, 'Checkpoint/model.safetensors'):
                report = inspect_downloaded_model(str(package), package_model=path)
                self.assertEqual(report['embedded_package']['transformer'], path)
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append(dict(image_package.current())) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--package-model', selected, '--prompt', 'kept',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertEqual(calls[0]['transformer'], selected)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            router.main(['--inspect-model', '--model', str(package), '--package-model', 'Checkpoint/missing.safetensors'])

    def test_generic_package_respects_the_explicit_main_model_without_cli_options(self):
        import image_package
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                package, selected = self.make_generic_package('sdxl', sharded, ambiguous=True, selected_main=True)
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(router.main(['--inspect-model', '--model', str(package)]), 0)
                inspected = json.loads(output.getvalue())['embedded_package']
                self.assertEqual(inspected['transformer'], selected)
                self.assertEqual(inspected['selection_reason'], 'user-assigned-main')
                calls = []
                with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                        main=lambda tokens: calls.append((tokens, dict(image_package.current()))) or 0)), redirect_stdout(io.StringIO()):
                    self.assertEqual(router.main(['--model-path', str(package), '--prompt', 'kept', '--width', '512',
                        '--height', '512', '--num-images', '1', '--device', 'auto', '--no-default-modifiers',
                        '--no-hires-fix', '--cache-dir', str(self.directory / 'cache')]), 0)
                self.assertEqual(calls[0][1]['transformer'], selected)
                self.assertEqual(calls[0][0][calls[0][0].index('--model-path') + 1], calls[0][1]['prepared_transformer'])
                with redirect_stdout(io.StringIO()) as explicit:
                    self.assertEqual(router.main(['--inspect-model', '--model', str(package), '--package-model',
                                                  'Checkpoint/another.safetensors']), 0)
                inspected = json.loads(explicit.getvalue())['embedded_package']
                self.assertEqual(inspected['transformer'], 'Checkpoint/another.safetensors')
                self.assertEqual(inspected['selection_reason'], 'explicit-request')

    def test_generic_package_rejects_conflicting_user_main_assignments(self):
        package, _ = self.make_generic_package('sdxl', False, ambiguous=True, selected_main=True)
        with package.open('rb') as stream:
            length, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(length))
            payload = stream.read()
        metadata = header['__metadata__']
        manifest = json.loads(metadata['iild_package_manifest'])
        other = next(item for item in manifest['files'] if item['path'] == 'Checkpoint/another.safetensors')
        other['position'], other['position_source'] = 'main', 'user'
        metadata['iild_package_manifest'] = json.dumps(manifest, separators=(',', ':'))
        encoded = json.dumps(header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        package.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            router.main(['--inspect-model', '--model', str(package)])
        self.assertIn('Multiple image checkpoints', errors.getvalue())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(router.main(['--inspect-model', '--model', str(package), '--package-model',
                                          'Checkpoint/another.safetensors']), 0)
        self.assertEqual(json.loads(output.getvalue())['embedded_package']['transformer'], 'Checkpoint/another.safetensors')

    def test_multi_anima_package_selects_model_and_unique_components(self):
        import image_package
        package = self.make_multi_anima_package()
        for selected in ('Checkpoint/one.safetensors', 'Checkpoint/two.safetensors'):
            with self.subTest(selected=selected):
                output = io.StringIO()
                selection = [] if selected.endswith('/one.safetensors') else ['--package-model', selected]
                with redirect_stdout(output):
                    self.assertEqual(router.main(['--inspect-model', '--model', str(package), *selection]), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report['architecture'], 'anima')
                self.assertEqual(report['embedded_package']['transformer'], selected)
                calls = []
                with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                        main=lambda tokens: calls.append((tokens, dict(image_package.current()))) or 0)), redirect_stdout(io.StringIO()):
                    self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', *selection,
                                                  '--cache-dir', str(self.directory / 'cache')]), 0)
                tokens, origin = calls[0]
                components = json.loads(tokens[tokens.index('--components') + 1])
                self.assertEqual(origin['transformer'], selected)
                self.assertEqual(set(components), {'llm', 'vae'})
                self.assertTrue(all(Path(value).is_file() for value in components.values()))

    def test_multi_anima_package_allows_explicit_ambiguous_encoder(self):
        package = self.make_multi_anima_package(multiple_encoders=True)
        mapping = {'llm': 'Encoders/two.safetensors'}
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept',
                '--package-components', json.dumps(mapping), '--cache-dir', str(self.directory / 'cache')]), 0)
        components = json.loads(calls[0][calls[0].index('--components') + 1])
        self.assertTrue(Path(components['llm']).is_file())
        self.assertTrue(Path(components['vae']).is_file())
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--cache-dir', str(self.directory / 'cache')])
        self.assertIn('multiple', errors.getvalue().casefold())

    def test_multi_moss_package_selects_models_and_unique_components(self):
        import image_package
        package = self.make_multi_moss_package()
        for selected in ('Checkpoint/one.safetensors', 'Checkpoint/two.safetensors'):
            with self.subTest(selected=selected):
                selection = [] if selected.endswith('/one.safetensors') else ['--package-model', selected]
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(router.main(['--inspect-model', '--model', str(package), *selection]), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report['architecture'], 'moss-image')
                self.assertEqual(report['embedded_package']['transformer'], selected)
                calls = []
                with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                        main=lambda tokens: calls.append((tokens, dict(image_package.current()))) or 0)), redirect_stdout(io.StringIO()):
                    self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', *selection,
                                                  '--cache-dir', str(self.directory / 'cache')]), 0)
                tokens, origin = calls[0]
                self.assertEqual(origin['transformer'], selected)
                for option in ('--moss-text-encoder', '--vae'):
                    self.assertTrue(Path(tokens[tokens.index(option) + 1]).is_file())

    def test_multi_moss_package_allows_explicit_ambiguous_encoder(self):
        package = self.make_multi_moss_package(multiple_encoders=True)
        mapping = {'llm': 'Encoders/two.safetensors'}
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept',
                '--package-components', json.dumps(mapping), '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertTrue(Path(calls[0][calls[0].index('--moss-text-encoder') + 1]).is_file())
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--cache-dir', str(self.directory / 'cache')])
        self.assertIn('multiple', errors.getvalue().casefold())

    def test_package_rejects_conflicting_model_signatures_in_one_component(self):
        package = self.make_multi_moss_package(conflicting=True)
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            router.main(['--inspect-model', '--model', str(package)])
        self.assertIn('conflicting model signatures', errors.getvalue().casefold())

    def test_generic_package_reaches_native_configuration_with_original_identity(self):
        import native_image
        import standalone_image
        package, _ = self.make_generic_package('sdxl', True)
        reports = []
        def resolve(tokens):
            _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(tokens))
            reports.append((native_image.configuration_values(args), args))
            return 0
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--engine', 'native', '--prompt', 'kept',
                                          '--width', '512', '--height', '512',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        report, args = reports[0]
        self.assertEqual(report['model'], str(package))
        self.assertEqual(args.architecture, 'sdxl')
        self.assertEqual(args.image_package['source'], str(package))

    def test_generic_package_corruption_is_rejected_before_backend(self):
        package, _ = self.make_generic_package('sdxl', True)
        with package.open('r+b') as stream:
            length, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(length))
            key = next(name for name in header if name.endswith('/model.diffusion_model.input_blocks.0.0.weight'))
            stream.seek(8 + length + header[key]['data_offsets'][0])
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append(tokens))), redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--cache-dir', str(self.directory / 'cache')])
        self.assertFalse(calls)

    def test_generic_package_prepares_explicit_split_component_paths(self):
        import native_image
        import standalone_image
        from downloaded_model import inspect_downloaded_model
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        components = {'clip_l': 'Encoders/l.safetensors', 'clip_g': 'Encoders/g.safetensors',
                      'vae': 'VAE/one.safetensors'}
        inspected = inspect_downloaded_model(str(package))
        self.assertTrue(set(components.values()).issubset(inspected['embedded_package']['component_paths']))
        self.assertEqual(set(inspected['backend_plan']['missing_components']), set(components))
        resolved = []
        def resolve(tokens):
            _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(tokens))
            resolved.append((native_image.configuration_values(args), args))
            return 0
        original = package.read_bytes()
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--engine', 'native', '--prompt', 'kept',
                '--package-components', json.dumps(components), '--cache-dir', str(self.directory / 'cache')]), 0)
        report, args = resolved[0]
        self.assertEqual(report['model'], str(package))
        self.assertEqual(args.image_package['selected_components'], components)
        self.assertEqual(set(args.components), set(components))
        self.assertTrue(all(Path(path).is_file() for path in args.components.values()))
        self.assertEqual(package.read_bytes(), original)

        class OpenedImage:
            format = 'PNG'
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def load(self): pass
        class Engine:
            def image(self, args, seed, *, prepare=False, **_):
                return (None, {'weight_storage': 'anonymous'}) if prepare else (object(), {})
        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=lambda _: OpenedImage()))
        def write_image(_image, path, **_): Path(path).write_bytes(b'synthetic test image')
        output = self.directory / 'split-output'
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), patch.dict(sys.modules, {'PIL': fake_pil}), \
             patch.object(native_image, 'write_png', side_effect=write_image), \
             patch.object(native_image, 'NativeEngine', Engine), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--engine', 'native', '--prompt', 'kept',
                '--package-components', json.dumps(components), '--cache-dir', str(self.directory / 'cache'),
                '--output-dir', str(output), '--width', '64', '--height', '64', '--steps', '1', '--no-hires-fix']), 0)
        published = json.loads((output / 'generation.json').read_text())
        self.assertEqual(published['images'][0]['image_package']['selected_components'], components)

    def test_generic_package_auto_selects_only_the_validated_vae(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        calls = []
        # The validator sees metadata for one original component at a time.
        def validate(_model, probe):
            with Path(probe).open('rb') as stream:
                size, = struct.unpack('<Q', stream.read(8))
                names = set(json.loads(stream.read(size)))
            calls.append(names)
            return 'extra.weight' not in names
        received = []
        def receive(tokens):
            received.append((tokens, dict(image_package.current()['selected_components'])))
            return 0
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_vae', side_effect=validate), \
             patch.object(image_package, '_read_probe_header', wraps=image_package._read_probe_header) as read_header, \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=receive)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        read_header.assert_called_once()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('encoder.conv_in.weight' in names and 'decoder.conv_out.weight' in names for names in calls))
        tokens, selected = received[0]
        components = json.loads(tokens[tokens.index('--components') + 1])
        self.assertEqual(set(components), {'vae'})
        self.assertTrue(Path(components['vae']).is_file())
        self.assertEqual(selected, {'vae': 'VAE/one.safetensors'})
        self.assertFalse(list((self.directory / 'cache').rglob('.iild-component-metadata-*')))

    def test_generic_package_vae_ambiguity_and_explicit_override(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        errors = io.StringIO()
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_vae', return_value=True), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda _: 0)), \
             redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                         '--cache-dir', str(self.directory / 'cache')])
        self.assertIn('multiple compatible', errors.getvalue().lower())
        self.assertIn('VAE/one.safetensors', errors.getvalue())
        self.assertIn('VAE/two.safetensors', errors.getvalue())
        calls = []
        with patch.object(image_package, 'inspect_native_vae', side_effect=AssertionError('explicit VAE must win')), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                '--package-components', json.dumps({'vae': 'VAE/one.safetensors'}),
                '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertTrue(Path(json.loads(calls[0][calls[0].index('--components') + 1])['vae']).is_file())
        external = self.directory / 'explicit-vae.safetensors'
        external.write_bytes(b'external')
        for options in (['--vae', str(external)], ['--components', json.dumps({'vae': str(external)})]):
            with self.subTest(options=options), \
                 patch.object(image_package, 'inspect_native_vae', side_effect=AssertionError('explicit VAE must win')), \
                 patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda _: 0)), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                                             '--cache-dir', str(self.directory / 'cache'), *options]), 0)

    def test_auto_selected_component_is_verified_before_backend(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        catalog = image_package.inspect_if_package(str(package))
        output_key = catalog['_vae_candidates']['VAE/one.safetensors']['encoder.conv_in.weight']
        with package.open('r+b') as stream:
            length, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(length))
            stream.seek(8 + length + header[output_key]['data_offsets'][0])
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
        def validate(_model, probe):
            with Path(probe).open('rb') as stream:
                size, = struct.unpack('<Q', stream.read(8))
                return 'extra.weight' not in json.loads(stream.read(size))
        calls = []
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_vae', side_effect=validate), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: calls.append(tokens))), redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                         '--cache-dir', str(self.directory / 'cache')])
        self.assertFalse(calls)

    def test_automatic_request_from_app_prepares_unique_package_vae(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        received = []
        def validate(_model, probe):
            with Path(probe).open('rb') as stream:
                size, = struct.unpack('<Q', stream.read(8))
                return 'extra.weight' not in json.loads(stream.read(size))
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_vae', side_effect=validate), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: received.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model-path', str(package), '--prompt', 'kept',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertIn('--components', received[0])

    def test_generic_package_keeps_embedded_or_explicit_diffusers_vae(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        for engine, state in [('native', 'embedded'), ('native', 'unsupported'), ('diffusers', 'missing')]:
            calls = []
            with self.subTest(engine=engine, state=state), \
                 patch.object(image_package, 'inspect_native_vae', return_value={'state': state}), \
                 patch.object(image_package, 'validate_native_vae', side_effect=AssertionError('unexpected validation')), \
                 patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                     main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
                self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', engine,
                                              '--cache-dir', str(self.directory / 'cache')]), 0)
                self.assertNotIn('--components', calls[0])

    def test_vae_probe_preserves_quant_metadata_and_is_always_removed(self):
        import image_package
        from weight_files import file_signature
        quant = b'{"format":"int8_tensorwise"}'
        names = {'encoded.weight': {'dtype': 'I8', 'shape': [4], 'data_offsets': [0, 4]},
                 'encoded.comfy_quant': {'dtype': 'U8', 'shape': [len(quant)], 'data_offsets': [4, 4 + len(quant)]}}
        raw = json.dumps(names).encode()
        source = self.directory / 'source.safetensors'
        source.write_bytes(struct.pack('<Q', len(raw)) + raw + b'1234' + quant)
        package = {'source': str(source), 'signature': file_signature(source)}
        mapping = {'layer.weight': 'encoded.weight', 'layer.comfy_quant': 'encoded.comfy_quant'}
        probe_path = None
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with image_package._component_metadata_probe(package, mapping, self.directory / 'probe') as probe:
                probe_path = Path(probe)
                with probe_path.open('rb') as stream:
                    size, = struct.unpack('<Q', stream.read(8))
                    header = json.loads(stream.read(size))
                    start, end = header['layer.comfy_quant']['data_offsets']
                    stream.seek(8 + size + start)
                    self.assertEqual(stream.read(end - start), quant)
                raise RuntimeError('stop')
        self.assertFalse(probe_path.exists())

    def test_generic_package_does_not_select_incompatible_vae(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        calls = []
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'incompatible'}), \
             patch.object(image_package, 'validate_native_vae', return_value=False), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertNotIn('--components', calls[0])

    def test_split_clip_components_are_selected_by_the_engine_contract(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True, recognizable_encoders=True)
        def validate(_model, slot, probe):
            with Path(probe).open('rb') as stream:
                size, = struct.unpack('<Q', stream.read(8))
                shape = json.loads(stream.read(size))['text_model.embeddings.token_embedding.weight']['shape']
            return shape[-1] == (2 if slot == 'clip_l' else 3)
        received = []
        def receive(tokens):
            received.append((tokens, dict(image_package.current()['selected_components'])))
            return 0
        with patch.object(image_package, 'inspect_native_text_encoder', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_text_encoder', side_effect=validate), \
             patch.object(image_package, 'choose_native_vae', return_value=None), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=receive)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens, selected = received[0]
        components = json.loads(tokens[tokens.index('--components') + 1])
        self.assertEqual(set(components), {'clip_l', 'clip_g'})
        self.assertTrue(all(Path(path).is_file() for path in components.values()))
        self.assertEqual(selected, {'clip_l': 'Encoders/l.safetensors', 'clip_g': 'Encoders/g.safetensors'})

    def test_split_clip_ambiguity_does_not_override_explicit_or_embedded_slots(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True, recognizable_encoders=True)
        errors = io.StringIO()
        with patch.object(image_package, 'inspect_native_text_encoder', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_text_encoder', return_value=True), \
             patch.object(image_package, 'choose_native_vae', return_value=None), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda _: 0)), \
             redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'kept', '--cache-dir', str(self.directory / 'cache')])
        self.assertIn('multiple compatible clip_l', errors.getvalue().lower())
        received = []
        def state(_model, slot):
            self.assertEqual(slot, 'clip_g')
            return {'state': 'embedded'}
        with patch.object(image_package, 'inspect_native_text_encoder', side_effect=state), \
             patch.object(image_package, 'validate_native_text_encoder', side_effect=AssertionError('no selection needed')), \
             patch.object(image_package, 'choose_native_vae', return_value=None), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: received.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept',
                '--package-components', json.dumps({'clip_l': 'Encoders/l.safetensors'}),
                '--cache-dir', str(self.directory / 'cache')]), 0)
        components = json.loads(received[0][received[0].index('--components') + 1])
        self.assertEqual(set(components), {'clip_l'})

    def test_split_flux_auto_selects_clip_and_t5_without_overriding_external_component(self):
        import image_package
        package, _ = self.make_generic_package('flux', True, separate=True, recognizable_encoders=True)
        external = self.directory / 'external-clip.safetensors'
        external.write_bytes(b'user-specified')
        def inspect(_model, slot):
            self.assertEqual(slot, 't5xxl')
            return {'state': 'missing'}
        def validate(_model, slot, probe):
            self.assertEqual(slot, 't5xxl')
            with Path(probe).open('rb') as stream:
                size, = struct.unpack('<Q', stream.read(8))
                return 'shared.weight' in json.loads(stream.read(size))
        received = []
        def receive(tokens):
            received.append((tokens, dict(image_package.current()['selected_components'])))
            return 0
        with patch.object(image_package, 'inspect_native_text_encoder', side_effect=inspect), \
             patch.object(image_package, 'validate_native_text_encoder', side_effect=validate), \
             patch.object(image_package, 'choose_native_vae', return_value=None), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=receive)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept',
                '--components', json.dumps({'clip_l': str(external)}), '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens, selected = received[0]
        components = json.loads(tokens[tokens.index('--components') + 1])
        self.assertEqual(components['clip_l'], str(external))
        self.assertTrue(Path(components['t5xxl']).is_file())
        self.assertEqual(selected, {'t5xxl': 'Encoders/t5.safetensors'})

    def test_explicit_diffusers_route_does_not_probe_native_text_encoders(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True, recognizable_encoders=True)
        received = []
        with patch.object(image_package, 'inspect_native_text_encoder', side_effect=AssertionError('explicit Diffusers')), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: received.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'diffusers',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertNotIn('--components', received[0])

    def test_vae_probe_skips_automatic_choice_when_sparse_files_are_unavailable(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        calls = []
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, '_prepare_sparse_file', side_effect=image_package._SparseProbeUnavailable()), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                 main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--engine', 'native',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertNotIn('--components', calls[0])
        self.assertFalse(list((self.directory / 'cache').rglob('.iild-component-metadata-*')))

    def test_component_discovery_honors_cancellation_without_probes(self):
        import image_package
        package, _ = self.make_generic_package('sdxl', True, separate=True, recognizable_encoders=True)
        inspected = image_package.inspect_if_package(str(package))
        cache = self.directory / 'canceled-probes'
        with patch.object(image_package, 'inspect_native_vae', return_value={'state': 'missing'}), \
             patch.object(image_package, 'inspect_native_text_encoder', return_value={'state': 'missing'}), \
             patch.object(image_package, 'validate_native_vae', side_effect=AssertionError('canceled')), \
             patch.object(image_package, 'validate_native_text_encoder', side_effect=AssertionError('canceled')):
            with self.assertRaisesRegex(RuntimeError, 'canceled'):
                image_package.choose_native_vae(inspected, 'unused-model', cache, lambda *_: True)
            with self.assertRaisesRegex(RuntimeError, 'canceled'):
                image_package.choose_native_text_encoders(inspected, 'unused-model', cache, ['clip_l'], lambda *_: True)
        self.assertFalse(cache.exists())

    def test_vae_metadata_probe_matches_real_engine_contract(self):
        import image_package
        from weight_files import file_signature
        correct = ROOT / 'resources/vae/sdxl/diffusion_pytorch_model.safetensors'
        wrong = ROOT / 'resources/vae/flux1/diffusion_pytorch_model.safetensors'
        if not correct.is_file() or not wrong.is_file():
            self.skipTest('Native VAE metadata fixtures are unavailable')
        model = self.directory / 'sdxl-fixture.safetensors'
        keys = ['model.diffusion_model.input_blocks.0.0.weight',
                'model.diffusion_model.middle_block.1.proj_in.weight',
                'conditioner.embedders.1.model.positional_embedding']
        header = {name: {'dtype': 'F32', 'shape': [1], 'data_offsets': [4 * i, 4 * (i + 1)]}
                  for i, name in enumerate(keys)}
        raw = json.dumps(header).encode()
        model.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4 * len(keys)))
        try:
            inspection = image_package.inspect_native_vae(model)
        except ValueError as error:
            if 'does not include the native image engine' in str(error):
                self.skipTest('Native image engine is unavailable')
            raise
        if inspection['state'] == 'unsupported':
            self.skipTest('Native image engine is unavailable')
        self.assertEqual(inspection['state'], 'missing')
        self.assertEqual(inspection['vae_family'], 'sdxl-base')
        for source, expected in ((correct, True), (wrong, False)):
            with self.subTest(source=source):
                with source.open('rb') as stream:
                    size, = struct.unpack('<Q', stream.read(8))
                    source_header = json.loads(stream.read(size))
                mapping = {name: name for name in source_header if name != '__metadata__'}
                package = {'source': str(source), 'signature': file_signature(source)}
                with image_package._component_metadata_probe(package, mapping, self.directory / 'probe') as probe:
                    self.assertEqual(image_package.validate_native_vae(model, probe), expected)

    def test_native_text_encoder_contract_uses_real_engine_metadata(self):
        import image_package
        from weight_files import file_signature
        clip = ROOT / 'resources/text_encoders/flux1_clip_l.safetensors'
        t5 = ROOT / 'resources/text_encoders/flux1_t5xxl_fp8_scaled.safetensors'
        if not clip.is_file() or not t5.is_file():
            self.skipTest('Native text encoder metadata fixtures are unavailable')
        model = self.directory / 'flux-fixture.safetensors'
        header = {'model.diffusion_model.double_blocks.0.img_attn.qkv.weight':
                  {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
        raw = json.dumps(header).encode()
        model.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4))
        state = image_package.inspect_native_text_encoder(model, 'clip_l')
        if state['state'] == 'unsupported':
            self.skipTest('Native text encoder inspection is unavailable')
        self.assertEqual(state['state'], 'missing')
        self.assertEqual(image_package.inspect_native_text_encoder(model, 't5xxl')['state'], 'missing')
        self.assertTrue(image_package.validate_native_text_encoder(model, 'clip_l', clip))
        self.assertFalse(image_package.validate_native_text_encoder(model, 'clip_l', t5))
        self.assertTrue(image_package.validate_native_text_encoder(model, 't5xxl', t5))
        self.assertFalse(image_package.validate_native_text_encoder(model, 't5xxl', clip))
        with clip.open('rb') as source:
            size, = struct.unpack('<Q', source.read(8))
            source_header = json.loads(source.read(size))
        mapping = {key: key for key in source_header
                   if key != '__metadata__' and key != 'text_model.encoder.layers.3.layer_norm1.weight'}
        package = {'source': str(clip), 'signature': file_signature(clip)}
        with image_package._component_metadata_probe(package, mapping, self.directory / 'probe') as incomplete:
            self.assertFalse(image_package.validate_native_text_encoder(model, 'clip_l', incomplete))

    def test_generic_package_rejects_unsafe_or_conflicting_component_requests(self):
        package, _ = self.make_generic_package('sdxl', True, separate=True)
        external = self.directory / 'external.safetensors'
        external.write_bytes(b'not used')
        requests = [
            ({'llm': 'Encoders/l.safetensors'}, [], 'native slots'),
            ({'clip_l': '../outside.safetensors'}, [], 'invalid embedded asset path'),
            ({'clip_l': 'Encoders/l.safetensors'}, ['--components', json.dumps({'clip_l': str(external)})], 'both inside and outside'),
            ({'vae': 'VAE/one.safetensors'}, ['--vae', str(external)], 'both inside and outside'),
            ({'clip_l': 'Encoders/l.safetensors'}, ['--engine', 'diffusers'], 'native image engine'),
        ]
        for requested, extra, message in requests:
            with self.subTest(requested=requested, extra=extra):
                calls, errors = [], io.StringIO()
                with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                        main=lambda tokens: calls.append(tokens))), redirect_stdout(io.StringIO()), \
                     redirect_stderr(errors), self.assertRaises(SystemExit):
                    router.main(['--model', str(package), '--prompt', 'kept', '--package-components',
                                 json.dumps(requested), '--cache-dir', str(self.directory / 'cache'), *extra])
                self.assertIn(message.casefold(), errors.getvalue().casefold())
                self.assertFalse(calls)

    def test_nested_anima_package_prepares_native_model_and_components(self):
        import image_package
        package = self.make_nested_anima_package()
        before = package.read_bytes()
        self.assertEqual(self.backend(None, 'auto', '--model', str(package)), 'local')
        calls = []
        def consume(tokens):
            calls.append((tokens, dict(image_package.current())))
            return 0
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=consume)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'keep', '--width', '512',
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens, origin = calls[0]
        value = lambda flag: tokens[tokens.index(flag) + 1]
        self.assertEqual(value('--prompt'), 'keep')
        self.assertNotEqual(value('--model'), str(package))
        self.assertIn(b'llm_adapter.blocks.0.cross_attn', Path(value('--model')).read_bytes())
        components = json.loads(value('--components'))
        self.assertIn(b'model.norm.weight', Path(components['llm']).read_bytes())
        self.assertIn(b'decoder.head.2.weight', Path(components['vae']).read_bytes())
        self.assertEqual(origin['source'], str(package))
        self.assertEqual(package.read_bytes(), before)

    def test_nested_anima_package_keeps_components_from_config(self):
        package = self.make_nested_anima_package()
        encoder = self.directory / 'explicit.safetensors'
        encoder.write_bytes(b'user supplied encoder')
        config = self.directory / 'request.json'
        config.write_text(json.dumps({'model': str(package), 'prompt': 'retained', 'components': {'llm': str(encoder)}}))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--config', str(config), '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens = calls[0]
        self.assertEqual(tokens[tokens.index('--config') + 1], str(config))
        self.assertNotEqual(tokens[tokens.index('--model-path') + 1], str(package))
        components = json.loads(tokens[tokens.index('--components') + 1])
        self.assertEqual(components['llm'], str(encoder))
        self.assertTrue(Path(components['vae']).is_file())

    def test_nested_anima_respects_null_component_values(self):
        package = self.make_nested_anima_package()
        config = self.directory / 'request.json'
        calls = []
        for configured, expected in ((None, 'materialized'), ({'llm': None}, None)):
            config.write_text(json.dumps({'model': str(package), 'prompt': 'kept', 'components': configured}))
            with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(router.main(['--config', str(config), '--cache-dir', str(self.directory / 'cache')]), 0)
            components = json.loads(calls[-1][calls[-1].index('--components') + 1])
            if expected == 'materialized':
                self.assertTrue(Path(components['llm']).is_file())
            else:
                self.assertIsNone(components['llm'])

    def test_nested_anima_native_configuration_keeps_original_package_identity(self):
        import native_image
        import standalone_image
        package = self.make_nested_anima_package()
        reports = []
        def resolve(tokens):
            _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(tokens))
            reports.append((native_image.configuration_values(args), args))
            return 0
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--width', '512',
                                          '--height', '512', '--package-components',
                                          json.dumps({'llm': 'LLM/text.safetensors'}),
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        report, args = reports[0]
        self.assertEqual(report['model'], str(package))
        self.assertEqual(args.image_package['source'], str(package))
        self.assertEqual(args.architecture, 'anima')
        self.assertEqual(set(args.components), {'llm', 'vae'})

    def test_nested_anima_worker_publishes_origin_and_rejects_changed_source(self):
        import native_image
        package = self.make_nested_anima_package()
        changed = [False]
        class OpenedImage:
            format = 'PNG'
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def load(self): pass
        class Engine:
            def image(self, args, seed, *, prepare=False, **_):
                if prepare:
                    return None, {'weight_storage': 'anonymous'}
                if changed[0]:
                    with package.open('ab') as stream:
                        stream.write(b'changed')
                return object(), {}
        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=lambda _: OpenedImage()))
        def write_image(_image, path, **_):
            Path(path).write_bytes(b'synthetic test image')
        common = ['--model', str(package), '--prompt', 'kept', '--width', '64', '--height', '64',
                  '--steps', '1', '--no-hires-fix', '--cache-dir', str(self.directory / 'cache')]
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), \
             patch.dict(sys.modules, {'PIL': fake_pil}), patch.object(native_image, 'write_png', side_effect=write_image), \
             patch.object(native_image, 'NativeEngine', Engine), redirect_stdout(io.StringIO()):
            output = self.directory / 'published'
            self.assertEqual(router.main([*common, '--output-dir', str(output)]), 0)
            report = json.loads((output / 'generation.json').read_text())
            self.assertEqual(report['images'][0]['image_package']['source'], str(package))
            self.assertTrue((output / 'image.png').is_file())
            changed[0] = True
            denied = self.directory / 'changed-output'
            with self.assertRaisesRegex(SystemExit, 'package changed'):
                router.main([*common, '--output-dir', str(denied)])
            self.assertFalse(denied.exists())

    def test_worker_protocol_recovers_after_corrupt_package_and_publishes_next_job(self):
        import native_image
        from inference_worker import serve
        package = self.make_nested_anima_package()
        bad = self.directory / 'damaged.safetensors'
        bad.write_bytes(package.read_bytes())
        with bad.open('r+b') as stream:
            length, = struct.unpack('<Q', stream.read(8)); header = json.loads(stream.read(length))
            key = next(name for name in header if name.endswith('/model.diffusion_model.blocks.0.self_attn.q_proj.weight'))
            stream.seek(8 + length + header[key]['data_offsets'][0]); byte = stream.read(1)
            stream.seek(-1, 1); stream.write(bytes([byte[0] ^ 1]))
        class OpenedImage:
            format = 'PNG'
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def load(self): pass
        class Engine:
            def image(self, args, seed, *, prepare=False, **_):
                return (None, {'weight_storage': 'anonymous'}) if prepare else (object(), {})
        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=lambda _: OpenedImage()))
        def write_image(_image, path, **_): Path(path).write_bytes(b'synthetic test image')
        requests = []
        for label, model in (('bad', bad), ('good', package)):
            arguments = ['--model', str(model), '--prompt', 'test', '--width', '64', '--height', '64',
                         '--steps', '1', '--no-hires-fix', '--cache-dir', str(self.directory / (label + '-cache')),
                         '--output-dir', str(self.directory / (label + '-output'))]
            requests.append({'schema': 'iild-worker-request-v1', 'id': label, 'arguments': arguments})
        stream = io.BytesIO(''.join(json.dumps(item) + '\n' for item in requests).encode())
        output = io.StringIO()
        with patch.dict('os.environ', {'IILD_MODEL_HASH_CACHE': 'off'}), patch.dict(sys.modules, {'PIL': fake_pil}), \
             patch.object(native_image, 'write_png', side_effect=write_image), \
             patch.object(native_image, 'NativeEngine', Engine), patch('sys.stdout', output):
            self.assertEqual(serve(router.main, stream, defer_foreground=router.defer_unsupported_foreground), 0)
        lines = output.getvalue().splitlines()
        results = [json.loads(line.removeprefix('IILD_RESULT ')) for line in lines if line.startswith('IILD_RESULT ')]
        self.assertEqual([result['id'] for result in results], ['bad', 'good'])
        self.assertFalse(results[0]['ok'])
        self.assertIn('checksum', results[0]['error'].lower())
        self.assertTrue(results[1]['ok'], results[1])
        self.assertTrue(any('preparing-model' in line for line in lines if line.startswith('IILD_NATIVE_PROGRESS ')))
        self.assertFalse((self.directory / 'bad-output').exists())
        report = json.loads((self.directory / 'good-output/generation.json').read_text())
        self.assertEqual(report['images'][0]['image_package']['source'], str(package))

    def test_nested_moss_package_routes_and_prepares_components_for_the_worker(self):
        package = self.make_nested_moss_package()
        before = package.read_bytes()
        self.assertEqual(self.backend(None, 'auto', '--model-path', str(package)), 'diffusers')
        import image_package
        calls = []
        def consume(tokens):
            calls.append((tokens, dict(image_package.current())))
            return 0
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=consume)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model-path', str(package), '--prompt', 'keep', '--num-images', '2',
                                          '--cache-dir', str(self.directory / 'cache'), '--no-hires-fix']), 0)
        tokens, origin = calls[0]
        def value(flag): return tokens[tokens.index(flag) + 1]
        self.assertEqual(value('--prompt'), 'keep')
        self.assertEqual(value('--num-images'), '2')
        self.assertEqual(value('--model-config-preset'), 'moss-image-sft')
        self.assertNotEqual(value('--model-path'), str(package))
        self.assertIn(b'layers.0.proj.qkv.weight', Path(value('--model-path')).read_bytes())
        self.assertIn(b'model.language_model.embed_tokens.weight', Path(value('--moss-text-encoder')).read_bytes())
        self.assertIn(b'decoder.conv_out.weight', Path(value('--vae')).read_bytes())
        self.assertEqual(origin['source'], str(package))
        self.assertEqual(origin['prepared_transformer'], value('--model-path'))
        self.assertIsNone(image_package.current())
        self.assertEqual(package.read_bytes(), before)

    def test_model_aliases_route_the_last_explicit_value(self):
        package = self.make_nested_moss_package()
        other = self.directory / 'different.safetensors'
        other.write_bytes(b'other model')
        self.assertEqual(self.backend(None, 'auto', '--model-path', str(package), '--model', str(other)), 'local')
        self.assertEqual(self.backend(None, 'auto', '--model', str(other), '--model-path', str(package)), 'diffusers')

    def test_nested_moss_package_respects_component_overrides_and_explicit_diffusers(self):
        import image_package
        package = self.make_nested_moss_package()
        self.assertEqual(image_package.inspect_if_package(str(package))['architecture'], 'moss-image')
        encoder, vae = self.directory / 'custom-encoder.safetensors', self.directory / 'custom-vae.safetensors'
        encoder.write_bytes(b'caller encoder'); vae.write_bytes(b'caller vae')
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--backend', 'diffusers', '--model', str(package), '--prompt', 'kept',
                                          '--moss-text-encoder', str(encoder), '--vae', str(vae),
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        tokens = calls[0]
        self.assertEqual(tokens[tokens.index('--moss-text-encoder') + 1], str(encoder))
        self.assertEqual(tokens[tokens.index('--vae') + 1], str(vae))
        self.assertEqual(tokens[tokens.index('--model-config-preset') + 1], 'moss-image-sft')
        self.assertEqual(len(list((self.directory / 'cache/embedded-models').glob('*.safetensors'))), 1)

    def test_nested_moss_explicit_diffusers_keeps_configuration_overrides(self):
        import generate_any as generic
        package = self.make_nested_moss_package()
        encoder, vae = self.directory / 'custom-encoder.safetensors', self.directory / 'custom-vae.safetensors'
        encoder.write_bytes(b'caller encoder'); vae.write_bytes(b'caller vae')
        config = self.directory / 'moss-request.json'
        config.write_text(json.dumps({'model': str(package), 'prompt': 'kept', 'moss_text_encoder': str(encoder),
                                      'vae': str(vae), 'model_config_preset': 'moss-image-sft', 'cache_dir': 'cache'}))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--backend', 'diffusers', '--config', str(config)]), 0)
        tokens = calls[0]
        self.assertEqual(tokens[tokens.index('--config') + 1], str(config))
        self.assertNotIn('--moss-text-encoder', tokens)
        self.assertNotIn('--vae', tokens)
        self.assertNotIn('--model-config-preset', tokens)
        self.assertTrue(Path(tokens[tokens.index('--model-path') + 1]).is_file())
        parsed = generic.build_parser().parse_args(tokens)
        self.assertEqual(Path(parsed.moss_text_encoder), encoder)
        self.assertEqual(Path(parsed.vae), vae)
        self.assertEqual(len(list((self.directory / 'cache/embedded-models').glob('*.safetensors'))), 1)

    def test_nested_moss_generic_configuration_keeps_original_package_identity(self):
        import generate_any as generic
        import moss_runtime
        package = self.make_nested_moss_package()
        config = self.directory / 'moss-config'
        config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                             **moss_runtime.COMPONENTS}))
        reports = []
        def resolve(tokens):
            args = generic.resolve_arguments(generic.build_parser().parse_args(tokens))
            reports.append(generic.configuration(args))
            return 0
        with patch.object(generic.model_config_defaults, 'resolve', return_value=config), \
             patch.object(generic.moss_text_encoder, 'resolve', return_value=None), \
             patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
             patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
             patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--model', str(package), '--prompt', 'kept', '--width', '512',
                                          '--height', '512', '--package-components',
                                          json.dumps({'llm': 'LLM/text.safetensors'}),
                                          '--cache-dir', str(self.directory / 'cache')]), 0)
        report = reports[0]
        self.assertEqual(report['model'], str(package))
        self.assertEqual(report['image_package']['source'], str(package))
        self.assertEqual(report['image_package']['transformer'], 'Checkpoint/model.safetensors.index.json')
        self.assertEqual(report['image_package']['selected_components'], {'llm': 'LLM/text.safetensors'})
        self.assertTrue(Path(report['image_package']['prepared_transformer']).is_file())

    def test_package_source_change_is_detected_in_execution_context(self):
        import image_package
        package = self.make_nested_moss_package()
        selected = image_package.inspect_if_package(str(package))
        with image_package.use(selected):
            image_package.verify_source()
            with package.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                image_package.verify_source()

    def test_nested_moss_package_corruption_stops_before_the_worker(self):
        package = self.make_nested_moss_package()
        with package.open('r+b') as stream:
            length, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(length))
            key = next(name for name in header if name.endswith('/layers.0.proj.qkv.weight'))
            stream.seek(8 + length + header[key]['data_offsets'][0])
            byte = stream.read(1); stream.seek(-1, 1); stream.write(bytes([byte[0] ^ 1]))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--prompt', 'keep', '--cache-dir', str(self.directory / 'cache')])
        self.assertEqual(calls, [])
        self.assertFalse(list((self.directory / 'cache/embedded-models').glob('*.safetensors')))

    def test_nested_moss_package_does_not_materialize_during_print_config(self):
        package = self.make_nested_moss_package()
        with self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--print-config', '--cache-dir', str(self.directory / 'cache')])
        self.assertFalse((self.directory / 'cache').exists())

    def test_inspect_model_reports_nested_moss_without_materializing(self):
        package = self.make_nested_moss_package()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(router.main(['--inspect-model', '--model', str(package)]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['architecture'], 'moss-image')
        self.assertEqual(report['role'], 'checkpoint')
        self.assertEqual(report['pipeline_class'], 'MOSSImagePipeline')
        self.assertEqual(report['embedded_package']['transformer'], 'Checkpoint/model.safetensors.index.json')
        self.assertEqual(report['embedded_package']['text_encoder'], 'LLM/text.safetensors')
        self.assertEqual(report['embedded_package']['vae'], 'VAE/vae.safetensors')
        self.assertFalse((self.directory / 'cache').exists())

    def test_inspect_model_reports_nested_anima_without_materializing(self):
        package = self.make_nested_anima_package()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(router.main(['--inspect-model', '--model', str(package)]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual((report['architecture'], report['role']), ('anima', 'checkpoint'))
        self.assertEqual(report['embedded_package']['transformer'], 'Checkpoint/model.safetensors.index.json')
        self.assertEqual(report['embedded_package']['text_encoder'], 'LLM/text.safetensors')
        self.assertEqual(report['embedded_package']['vae'], 'VAE/vae.safetensors')
        self.assertFalse((self.directory / 'cache').exists())

    def test_nested_anima_validate_only_does_not_materialize(self):
        package = self.make_nested_anima_package()
        with self.assertRaises(SystemExit):
            router.main(['--model', str(package), '--validate-only', '--cache-dir', str(self.directory / 'cache')])
        self.assertFalse((self.directory / 'cache').exists())

    def test_unknown_hosted_and_cross_family_presets_are_rejected(self):
        for name in ("new-model-typo", "OpenAI"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.backend(name, "comfyui", "--workflow", "x.json")
        with self.assertRaisesRegex(SystemExit, "incompatible"):
            generate.resolve_arguments(local_parser(generate.build_parser).parse_args(
                ["--base-model", "Illustrious", "--preset", "flux1-dev"]))

    def test_base_model_selects_matching_default_and_retains_identity(self):
        for name, expected in (("Illustrious", "illustrious"), ("NoobAI", "noobai"),
                               ("Flux.1 Krea", "flux1-krea-dev")):
            with self.subTest(name=name):
                preset, args = generate.resolve_arguments(local_parser(generate.build_parser).parse_args(["--base-model", name]))
                self.assertEqual(preset.name, expected)
                self.assertEqual(args.base_model, name)

    def test_community_base_categories_use_compatible_contracts(self):
        for name, expected in (("SD 1.5", "sd15-compatible"), ("SDXL 1.0", "sdxl"),
                               ("Flux.1 S", "flux1-schnell-compatible")):
            preset, args = generate.resolve_arguments(local_parser(generate.build_parser).parse_args(["--base-model", name]))
            self.assertEqual(preset.name, expected)
            self.assertFalse(preset.strict_contract)

    def test_noobai_vpred_explicit_variant_is_retained(self):
        preset, args = generate.resolve_arguments(local_parser(generate.build_parser).parse_args(
            ["--base-model", "NoobAI", "--preset", "noobai-v-pred"]))
        self.assertEqual(preset.name, "noobai-v-pred")
        self.assertIn(("prediction_type", "v_prediction"), preset.scheduler_defaults)

    def test_same_family_cannot_silently_substitute_a_different_base(self):
        for base, preset in (("Flux.1 Krea", "flux1-schnell"),
                             ("Illustrious", "noobai"), ("NoobAI", "illustrious")):
            with self.subTest(base=base), self.assertRaisesRegex(SystemExit, "identities"):
                generate.resolve_arguments(local_parser(generate.build_parser).parse_args(
                    ["--base-model", base, "--preset", preset]))

    def test_guidance_zero_is_required_only_for_schnell(self):
        for name in ("flux1-dev", "flux1-krea-dev"):
            preset, args = generate.resolve_arguments(local_parser(generate.build_parser).parse_args(["--preset", name]))
            generate.validate_generation_arguments(preset, args)
            call = generate.build_pipeline_call_arguments(preset, args, "generator")
            self.assertEqual(call["guidance_scale"], preset.guidance_scale)
            self.assertEqual(call["max_sequence_length"], 512)
        with self.assertRaisesRegex(SystemExit, "guidance scale"):
            preset, args = generate.resolve_arguments(local_parser(generate.build_parser).parse_args(
                ["--preset", "flux1-schnell", "--guidance-scale", "3"]))
            generate.validate_generation_arguments(preset, args)

    def test_pony_requires_explicit_weights(self):
        with self.assertRaisesRegex(SystemExit, "explicit --model"):
            generate.resolve_arguments(generate.build_parser().parse_args(["--base-model", "Pony"]))

    def test_frontdoor_print_config_runs_offline(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"),
                                 "--base-model", "Illustrious", "--preset", "illustrious", "--model", MODEL, "--print-config"],
                                capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertEqual(output["preset"], "illustrious")
        self.assertEqual(output["base_model"], "Illustrious")

    def test_catalog_command_includes_every_snapshot_entry(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--list-base-models"],
                                capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertEqual(len(output["base_models"]), 104)
        self.assertEqual(len(output["source"]["commit"]), 40)

    def test_existing_downloaded_weight_files_route_to_local_runtime(self):
        for suffix in (".safetensors", ".ckpt", ".gguf"):
            path = self.directory / ("download" + suffix)
            path.write_bytes(b"model")
            with self.subTest(suffix=suffix):
                self.assertEqual(self.backend(None, "auto", "--model", str(path)), "local")
                self.assertEqual(self.backend("Illustrious", "auto", "--model=" + str(path)), "local")

    def test_safetensors_with_bin_extension_does_not_require_legacy_inspection_runtime(self):
        path = self.directory / "tensor-data.bin"
        raw = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'x')
        with patch.dict(sys.modules, {"local_image": None}):
            self.assertIsNone(router.legacy_inspection_python(str(path), []))

    def test_renamed_legacy_container_uses_the_requested_safe_inspection_runtime(self):
        path = self.directory / 'legacy.safetensors'
        path.write_bytes(b'PK\x03\x04' + bytes(20))
        interpreter = self.directory / 'managed-python'
        interpreter.write_bytes(b'fixture')
        self.assertEqual(router.legacy_inspection_python(str(path), ['--runtime-python', str(interpreter)]), interpreter)

    def test_auto_routes_recognized_moss_by_tensor_layout_not_filename(self):
        path = self.directory / 'unrelated-name.safetensors'
        header = {
            'video_embedder.weight': {'dtype': 'BF16', 'shape': [4, 2], 'data_offsets': [0, 16]},
            'final_linear_video.weight': {'dtype': 'BF16', 'shape': [2, 4], 'data_offsets': [16, 32]},
            'layers.0.proj.qkv.weight': {'dtype': 'I8', 'shape': [12, 4], 'data_offsets': [32, 80]},
            'layers.0.proj.qkv.comfy_quant': {'dtype': 'U8', 'shape': [2], 'data_offsets': [80, 82]},
        }
        data = json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(82))
        self.assertEqual(self.backend(None, 'auto', '--model', str(path)), 'diffusers')
        self.assertEqual(self.backend(None, 'local', '--model', str(path)), 'local')
        config = self.directory / 'moss-request.json'
        config.write_text(json.dumps({'model': str(path), 'prompt': 'retained'}))
        self.assertEqual(self.backend(None, 'auto', '--config', str(config)), 'diffusers')

        header['layers.0.proj.qkv.weight']['shape'] = [11, 4]
        header['layers.0.proj.qkv.weight']['data_offsets'] = [32, 76]
        header['layers.0.proj.qkv.comfy_quant']['data_offsets'] = [76, 78]
        data = json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(78))
        self.assertEqual(self.backend(None, 'auto', '--model', str(path)), 'local')

    def test_renamed_moss_routes_by_container_and_keeps_request(self):
        header = {
            'video_embedder.weight': {'dtype': 'BF16', 'shape': [4, 2], 'data_offsets': [0, 16]},
            'final_linear_video.weight': {'dtype': 'BF16', 'shape': [2, 4], 'data_offsets': [16, 32]},
            'layers.0.proj.qkv.weight': {'dtype': 'BF16', 'shape': [12, 4], 'data_offsets': [32, 128]},
        }
        data = json.dumps(header, separators=(',', ':')).encode()
        for suffix in ('.bin', '.data', '.gguf'):
            with self.subTest(suffix=suffix):
                path = self.directory / ('renamed' + suffix)
                path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(128))
                self.assertEqual(self.backend(None, 'auto', '--model-path', str(path)), 'diffusers')
                self.assertEqual(self.backend(None, 'local', '--model-path', str(path)), 'local')
                self.assertTrue(router.defer_unsupported_foreground(['--model', str(path), '--backend', 'local']))
                calls = []
                with patch.object(router.importlib, 'import_module',
                                  return_value=SimpleNamespace(main=lambda args: calls.append(args) or 0)):
                    self.assertEqual(router.main(['--model-path', str(path), '--prompt', 'keep',
                                                  '--num-images', '2', '--no-hires-fix']), 0)
                self.assertEqual(calls[0][calls[0].index('--model-config-preset') + 1], 'moss-image-sft')
                self.assertEqual(calls[0][calls[0].index('--model-path') + 1], str(path))
                self.assertEqual(calls[0][calls[0].index('--prompt') + 1], 'keep')
                import generate_any
                parsed = generate_any.build_parser().parse_args(calls[0])
                self.assertEqual(parsed.num_images, 2)
                self.assertFalse(parsed.hires_fix)
        onnx = self.directory / 'renamed.onnx'
        onnx.write_bytes(struct.pack('<Q', len(data)) + data + bytes(128))
        self.assertFalse(router._is_moss_checkpoint(str(onnx)))

    def test_sharded_moss_routes_to_pinned_diffusers_without_repacking(self):
        import vae_defaults
        def shard(filename, entries):
            header, offset = {}, 0
            for name, shape in entries:
                count = 2
                for dimension in shape:
                    count *= dimension
                header[name] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [offset, offset + count]}
                offset += count
            raw = json.dumps(header, separators=(',', ':')).encode()
            (self.directory / filename).write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(offset))
            return {name: filename for name, _ in entries}
        mapping = shard('moss-1.safetensors', [('video_embedder.weight', [4, 2])])
        mapping.update(shard('moss-2.safetensors', [('final_linear_video.weight', [2, 4]),
                                                    ('layers.0.proj.qkv.weight', [12, 4])]))
        path = self.directory / 'model.safetensors.index.json'
        path.write_text(json.dumps({'weight_map': mapping}))
        self.assertEqual(self.backend(None, 'auto', '--model', str(path)), 'diffusers')
        calls = []
        with patch.object(router.importlib, 'import_module',
                          return_value=SimpleNamespace(main=lambda tokens: calls.append(tokens) or 0)):
            self.assertEqual(router.main(['--model', str(path), '--prompt', 'retained']), 0)
        self.assertIn('moss-image-sft', calls[0])
        self.assertEqual(calls[0][calls[0].index('--model') + 1], str(path))

        import generate_any as generic
        import moss_runtime
        config = self.directory / 'moss-config'
        config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                              **moss_runtime.COMPONENTS}))
        with patch.object(generic.model_config_defaults, 'resolve', return_value=config), \
             patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
             patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
             patch.object(vae_defaults, 'select_fallback_vae', return_value=None):
            parsed = generic.build_parser().parse_args(['--model', str(path), '--model-config-preset', 'moss-image-sft',
                                                        '--prompt', 'retained', '--print-config'])
            selected = generic.resolve_arguments(parsed)
        self.assertEqual(selected.model, str(path))
        identity = generic.model_identity(config, str(path))
        generic.verify_identity(identity)
        self.assertEqual(len(identity['single_file_shards']), 2)
        with (self.directory / 'moss-2.safetensors').open('r+b') as output:
            output.seek(-1, 2)
            output.write(b'X')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.verify_identity(identity)

    def test_moss_auto_forwards_a_verified_configuration_without_discarding_inputs(self):
        path = self.directory / 'model.safetensors'
        header = {
            'video_embedder.weight': {'dtype': 'BF16', 'shape': [4, 2], 'data_offsets': [0, 16]},
            'final_linear_video.weight': {'dtype': 'BF16', 'shape': [2, 4], 'data_offsets': [16, 32]},
            'layers.0.proj.qkv.weight': {'dtype': 'BF16', 'shape': [12, 4], 'data_offsets': [32, 128]},
        }
        data = json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(128))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=lambda args: calls.append(args) or 0)):
            self.assertEqual(router.main(['--model', str(path), '--prompt', 'retained', '--steps', '7']), 0)
        self.assertEqual(calls[0][calls[0].index('--model-config-preset') + 1], 'moss-image-sft')
        self.assertEqual(calls[0][calls[0].index('--prompt') + 1], 'retained')
        self.assertEqual(calls[0][calls[0].index('--steps') + 1], '7')

    def test_moss_auto_preserves_json_configuration_and_does_not_override_model_config(self):
        import generate_any as generic
        import moss_runtime
        package = self.make_nested_moss_package()
        model_config = self.directory / 'custom-config'
        model_config.mkdir()
        (model_config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                    **moss_runtime.COMPONENTS}))
        encoder = self.directory / 'external-text.safetensors'
        encoder.write_bytes(b'caller supplied encoder')
        config = self.directory / 'moss-auto.json'
        config.write_text(json.dumps({'model': str(package), 'prompt': 'retained', 'width': 512, 'height': 512,
            'model_config': str(model_config), 'moss_text_encoder': str(encoder)}))
        calls = []
        with patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(
                main=lambda tokens: calls.append(tokens) or 0)), redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--config', str(config), '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertNotIn('--model-config-preset', calls[0])
        parsed = generic.build_parser().parse_args(calls[0])
        self.assertEqual(parsed.prompt, 'retained')
        self.assertEqual((parsed.width, parsed.height), (512, 512))
        self.assertEqual(Path(parsed.model_config), model_config)
        self.assertEqual(Path(parsed.moss_text_encoder), encoder)
        self.assertTrue(Path(parsed.model).is_file())
        self.assertNotEqual(Path(parsed.model), package)

        resolved = []
        def resolve(tokens):
            args = generic.resolve_arguments(generic.build_parser().parse_args(tokens))
            resolved.append(generic.configuration(args))
            return 0
        with patch.object(generic.moss_text_encoder, 'resolve', return_value=None), \
             patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
             patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
             patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')), \
             patch.object(router.importlib, 'import_module', return_value=SimpleNamespace(main=resolve)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(router.main(['--config', str(config), '--cache-dir', str(self.directory / 'cache')]), 0)
        self.assertEqual(resolved[0]['model'], str(package))
        self.assertEqual(resolved[0]['pipeline_inputs']['prompt'], 'retained')
        self.assertEqual(resolved[0]['model_config'], str(model_config))

        for values, message in (({'model_config_preset': 'krea2-turbo'}, 'conflicts'),
                                ({'engine': 'native'}, 'native-only')):
            with self.subTest(values=values):
                config.write_text(json.dumps({'model': str(package), 'prompt': 'retained', **values}))
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit):
                    router.main(['--config', str(config), '--cache-dir', str(self.directory / 'cache')])
                self.assertIn(message, errors.getvalue())

    def test_moss_frontdoor_accepts_dreamscapes_batch_and_hires_contract(self):
        import generate_any as generic
        import moss_runtime
        path = self.directory / 'moss-renamed.data'
        header = {
            'video_embedder.weight': {'dtype': 'BF16', 'shape': [4, 2], 'data_offsets': [0, 16]},
            'final_linear_video.weight': {'dtype': 'BF16', 'shape': [2, 4], 'data_offsets': [16, 32]},
            'layers.0.proj.qkv.weight': {'dtype': 'BF16', 'shape': [12, 4], 'data_offsets': [32, 128]},
        }
        data = json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(128))
        config = self.directory / 'moss-config'
        config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                              **moss_runtime.COMPONENTS}))
        for hires in (False, True):
            output = io.StringIO()
            with self.subTest(hires=hires), \
                 patch.object(generic.model_config_defaults, 'resolve', return_value=config), \
                 patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
                 patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
                 patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')), \
                 redirect_stdout(output):
                self.assertEqual(router.main(['--model-path', str(path), '--prompt', 'test',
                    '--negative-prompt', 'blur', '--width', '512', '--height', '768', '--num-images', '2',
                    '--device', 'auto', '--output-dir', str(self.directory / 'output'),
                    '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers',
                    '--hires-fix' if hires else '--no-hires-fix', '--steps', '12',
                    '--guidance-scale', '3', '--print-config']), 0)
            config_report = json.loads(output.getvalue())
            self.assertEqual(config_report['runner'], 'generic-diffusers')
            self.assertEqual(config_report['model'], str(path))
            self.assertEqual(config_report['pipeline_inputs']['num_images_per_prompt'], 2)
            self.assertEqual(config_report['pipeline_inputs']['num_inference_steps'], 12)
            self.assertEqual(config_report['hires']['base_size'] if hires else config_report['hires'],
                             [256, 384] if hires else None)

    def test_explicit_runtime_choices_keep_precedence_over_existing_file(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        for extra, expected in ((["--preset", "sdxl"], "preset"),
                                (["--pipeline-class", "StableDiffusionXLPipeline"], "diffusers"),
                                (["--workflow", "flow.json"], "comfyui")):
            with self.subTest(extra=extra):
                self.assertEqual(self.backend(None, "auto", "--model", str(path), *extra), expected)
        self.assertEqual(self.backend(None, "diffusers", "--model", str(path)), "diffusers")
        self.assertEqual(self.backend(None, "local", "--model", str(path)), "local")

    def test_local_component_and_runtime_options_route_before_loading_model(self):
        for flag in ("--model-info", "--components", "--text-encoder", "--text-encoder-2", "--runtime-source", "--runtime-python", "--model-type", "--decoder", "--model-negative", "--embedded-guidance", "--sampling-shift"):
            with self.subTest(flag=flag):
                self.assertEqual(self.backend(None, "auto", flag, "value"), "local")

    def test_local_model_config_selects_generic_loader_before_local_file(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        self.assertEqual(self.backend("Kolors", "auto", "--model", str(path), "--model-config", str(self.directory)), "diffusers")
        self.assertEqual(self.backend(None, "auto", "--model=" + str(path), "--model-config=" + str(self.directory)), "diffusers")
        self.assertEqual(self.backend("Illustrious", "auto", "--model", str(path), "--model-config", str(self.directory), "--preset", "illustrious"), "preset")
        self.assertEqual(self.backend(None, "local", "--model", str(path), "--model-config", str(self.directory)), "local")

    def test_generic_media_options_select_generic_loader(self):
        for flag in ("--audio-sample-rate", "--video-layout", "--tensor-outputs", "--generation-architecture"):
            with self.subTest(flag=flag):
                self.assertEqual(self.backend(None, "auto", flag, "value"), "diffusers")

    def test_generic_control_options_route_single_files_to_diffusers(self):
        path = self.directory / 'download.safetensors'
        path.write_bytes(b'model')
        options = (
            ('--controlnet-format', 'krea2-color-depth'),
            ('--control-reference-resize', 'comfy-1mp'),
            ('--control-preprocess', 'canny'), ('--control-preprocessor-library', 'lib'),
            ('--control-depth-model', 'depth'), ('--control-depth-dtype', 'float16'), ('--control-map-preprocess', 'training'),
            ('--control-color-image', 'color.png'), ('--control-color-end', '0.5'),
            ('--control-depth-end', '0.5'), ('--control-inpaint-image', 'image.png'),
            ('--control-mask-image', 'mask.png'), ('--krea2-edit-reference-attention', 'isolated'),
            ('--krea2-fp8-storage', 'cpu-stream'),
            ('--moss-transformer-storage', 'cpu-stream'),
            ('--moss-rapid-adapter', 'adapter'), ('--moss-regions', 'regions.json'), ('--moss-text-encoder', 'encoder.safetensors'),
            ('--flux-clip-encoder', 'encoder.safetensors'), ('--flux-t5-encoder', 'encoder.safetensors'),
            ('--flux-t5-fp8-storage', 'cpu-stream'),
            ('--flux-transformer-storage', 'cpu-stream'),
            ('--krea2-regions', 'regions.json'))
        for flag, value in options:
            with self.subTest(flag=flag):
                self.assertEqual(self.backend(None, 'auto', '--model', str(path), flag, value), 'diffusers')
                self.assertEqual(self.backend(None, 'auto', '--model', str(path), flag + '=' + value), 'diffusers')
        self.assertEqual(self.backend(None, 'auto', '--model', str(path), '--krea2-control-auto-bucket'), 'diffusers')
        self.assertEqual(self.backend(None, 'auto', '--model', str(path), '--controlnet', 'adapter'), 'local')
        self.assertEqual(self.backend(None, 'auto', '--model', str(path), '--preset', 'sdxl',
                                      '--control-preprocess', 'canny'), 'preset')
        self.assertEqual(self.backend(None, 'local', '--model', str(path), '--control-preprocess', 'canny'), 'local')

    def test_generic_control_single_file_reports_missing_configuration(self):
        path = self.directory / 'download.safetensors'
        path.write_bytes(b'model')
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'reference/generate.py'),
                                 '--model', str(path), '--controlnet-format', 'krea2-color-depth',
                                 '--print-config'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('single-file model requires --model-config', result.stderr)

    def test_editing_pipeline_and_inputs_override_downloaded_file_auto_route(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        for pipeline in ("StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline", "KolorsImg2ImgPipeline"):
            with self.subTest(pipeline=pipeline):
                self.assertEqual(self.backend(None, "auto", "--model", str(path), "--pipeline-class", pipeline,
                                              "--pipeline-inputs", '{"image":{"image_path":"/input.png"}}'), "diffusers")

    def test_top_level_model_config_dispatches_generic_print_config_offline(self):
        path = self.directory / "download.safetensors"
        header = json.dumps({"model.diffusion_model.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        config = self.directory / "config"
        config.mkdir()
        (config / "model_index.json").write_text('{"_class_name":"StableDiffusionXLPipeline"}')
        result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/generate.py"),
                                 "--model", str(path), "--model-config", str(config), "--print-config"],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["runner"], "generic-diffusers")
        self.assertEqual(report["model_config"], str(config))

    def test_diffusers_directory_routes_by_model_index(self):
        directory = self.directory / "pipeline"
        directory.mkdir()
        self.assertEqual(self.backend(None, "auto", "--model", str(directory)), "preset")
        (directory / "model_index.json").write_text("{}")
        self.assertEqual(self.backend(None, "auto", "--model", str(directory)), "diffusers")

    def test_krea2_control_uses_existing_tokenizer_through_frontdoor(self):
        model = self.directory / 'krea2'
        model.mkdir()
        (model / 'model_index.json').write_text(json.dumps({'_class_name': 'Krea2Pipeline',
            'is_distilled': True, 'text_encoder': ['transformers', 'Qwen3VLModel'],
            'vae': ['diffusers', 'AutoencoderKLQwenImage']}))
        for folder, name in (('text_encoder', 'model.safetensors'), ('vae', 'diffusion_pytorch_model.safetensors')):
            (model / folder).mkdir()
            (model / folder / name).write_bytes(b'fixture')
        adapter = self.directory / 'adapter.safetensors'
        hint = self.directory / 'hint.png'
        adapter.write_bytes(b'fixture')
        hint.write_bytes(b'fixture')
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'reference/generate.py'), '--model', str(model),
            '--controlnet', str(adapter), '--control-image', str(hint), '--seed', '7',
            '--no-default-modifiers', '--print-config'], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['runner'], 'generic-diffusers')
        self.assertEqual(report['controlnet']['kind'], 'krea2-expanded-input-control-lora')
        self.assertEqual(report['tokenizer']['status'], 'fallback')
        self.assertEqual(len(report['tokenizer']['files']), 3)

    def test_flux_and_moss_control_use_generic_frontdoor_offline(self):
        for family in ('flux', 'moss'):
            with self.subTest(family=family):
                model = self.directory / family
                model.mkdir()
                index = {'_class_name': 'FluxPipeline' if family == 'flux' else 'MOSSImagePipeline',
                         'vae': ['diffusers', 'AutoencoderKL' if family == 'flux' else 'AutoencoderKLFlux2']}
                if family == 'moss':
                    index.update({'transformer': ['pipeline_moss_image', 'MOSSImageTransformer'],
                                  'scheduler': ['pipeline_moss_image', 'MOSSImageScheduler'],
                                  'text_encoder': ['transformers', 'Qwen3_5Model'],
                                  'tokenizer': ['transformers', 'Qwen2Tokenizer']})
                (model / 'model_index.json').write_text(json.dumps(index))
                if family == 'moss':
                    (model / 'text_encoder').mkdir()
                    (model / 'text_encoder/model.safetensors').write_bytes(b'fixture')
                (model / 'vae').mkdir()
                (model / 'vae' / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
                hint = model / 'hint.png'
                hint.write_bytes(b'fixture')
                if family == 'flux':
                    adapter = model / 'control'
                    adapter.mkdir()
                    (adapter / 'config.json').write_text('{"_class_name":"FluxControlNetModel"}')
                    (adapter / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
                else:
                    adapter = model / 'control.safetensors'
                    adapter.write_bytes(b'fixture')
                result = subprocess.run([sys.executable, '-S', str(ROOT / 'reference/generate.py'),
                    '--model', str(model), '--controlnet', str(adapter), '--control-image', str(hint),
                    '--control-preprocess', 'canny', '--seed', '7', '--no-default-modifiers', '--print-config'],
                    capture_output=True, text=True, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(report['runner'], 'generic-diffusers')
                self.assertEqual(report['control_preprocess'], 'canny')
                if family == 'flux':
                    self.assertEqual(report['controlnet']['component_class'], 'FluxControlNetModel')
                else:
                    self.assertEqual(report['controlnet']['kind'], 'moss-image-color-depth-control')
                    self.assertEqual(report['moss_tokenizer']['status'], 'fallback')

    def test_klein_reference_control_uses_generic_frontdoor_offline(self):
        model = self.directory / 'klein'
        (model / 'vae').mkdir(parents=True)
        (model / 'model_index.json').write_text(json.dumps({'_class_name': 'Flux2KleinPipeline',
            'vae': ['diffusers', 'AutoencoderKLFlux2']}))
        (model / 'vae' / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        adapter, hint, reference = (model / name for name in ('control.safetensors', 'hint.png', 'reference.png'))
        for path in (adapter, hint, reference):
            path.write_bytes(b'fixture')
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'reference/generate.py'),
            '--model', str(model), '--controlnet', str(adapter), '--controlnet-format', 'flux2-klein-reference',
            '--control-image', str(hint), '--control-preprocess', 'canny',
            '--control-reference-resize', 'comfy-1mp',
            '--pipeline-inputs', json.dumps({'image': {'image_path': str(reference)}}),
            '--seed', '7', '--no-default-modifiers', '--print-config'], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['runner'], 'generic-diffusers')
        self.assertEqual(report['controlnet']['kind'], 'flux2-klein-reference-lora')
        self.assertEqual(report['controlnet']['reference_order'], ['control', 'reference'])
        self.assertEqual(report['controlnet']['reference_resize'], 'comfy-1mp')
        self.assertEqual(report['control_preprocess'], 'canny')

    def test_nonexistent_model_print_config_keeps_original_preset_contract(self):
        self.assertEqual(self.backend("Illustrious", "auto", "--model", str(self.directory / "not-downloaded.safetensors"), "--print-config"), "preset")
        self.assertEqual(self.backend("NoobAI", "auto", "--prediction-type", "v_prediction", "--clip-skip", "2"), "preset")

    def test_local_backend_dispatch_preserves_model_and_base_arguments(self):
        called = []
        backend = SimpleNamespace(main=lambda arguments: called.append(arguments) or 0)
        with patch.object(router.importlib, "import_module", return_value=backend) as imported:
            result = router.main(["--backend", "local", "--base-model", "Illustrious", "--model", "download.safetensors", "--print-config"])
        self.assertEqual(result, 0)
        imported.assert_called_once_with("standalone_image")
        self.assertEqual(called, [["--base-model", "Illustrious", "--model", "download.safetensors", "--print-config"]])

    def test_inspect_model_outputs_tensor_identity_without_generation(self):
        path = self.directory / "not-a-name-guess.safetensors"
        header = {
            "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F32", "shape": [1, 4, 1, 1], "data_offsets": [0, 16]},
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": {"dtype": "F32", "shape": [1, 768], "data_offsets": [16, 3088]},
        }
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(3088))
        output = io.StringIO()
        with redirect_stdout(output):
            result = router.main(["--inspect-model", "--model", str(path)])
        found = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(found["architecture"], "sd1")
        self.assertEqual(found["preset"], "sd15-compatible")

    def test_inspect_model_requires_a_real_model(self):
        for tokens in (["--inspect-model"], ["--inspect-model", "--model"],
                       ["--inspect-model", "--model", str(self.directory / "missing.safetensors")]):
            with self.subTest(tokens=tokens), self.assertRaises(SystemExit) as error:
                router.main(tokens)
            self.assertEqual(error.exception.code, 2)

    def test_legacy_inspection_reuses_managed_python_and_preserves_options(self):
        interpreter = self.directory / "managed-python"
        tokens = ["--inspect-model", "--model", "download.ckpt", "--model-info", "saved.json"]
        with patch.object(router, "legacy_inspection_python", return_value=interpreter), \
                patch.object(router.subprocess, "call", return_value=7) as called:
            self.assertEqual(router.main(tokens), 7)
        called.assert_called_once_with([str(interpreter), str(ROOT / "reference/generate.py"), *tokens])

    def test_legacy_inspection_same_interpreter_does_not_reexecute(self):
        self.assertIsNone(router.legacy_inspection_python("download.ckpt", ["--runtime-python", sys.executable]))

    def test_safetensors_and_gguf_inspection_stay_dependency_free(self):
        for model in ("download.safetensors", "download.gguf"):
            with self.subTest(model=model):
                self.assertIsNone(router.legacy_inspection_python(model, ["--runtime-python", "missing-python"]))

    def test_explicit_missing_legacy_inspection_python_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime Python does not exist"):
            router.legacy_inspection_python("download.ckpt", ["--runtime-python", str(self.directory / "missing-python")])


if __name__ == "__main__":
    unittest.main()
