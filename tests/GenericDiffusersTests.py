#!/usr/bin/env python3
"""Dependency-free contracts for the generic installed Diffusers runner."""

from __future__ import annotations
import resource_fixture

import json
import hashlib
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from _file_provider import provider
import generate_any as generic


class FakePipeline:
    __module__ = "diffusers.pipelines.test_builtin"

    def __call__(self, prompt=None, image=None, width=32, num_inference_steps=1,
                 guidance_scale=1.0, generator=None, output_type="pil", **kwargs):
        pass


class FakeImage:
    is_animated = False
    mode = "RGB"
    size = (32, 32)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load(self):
        pass

    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, path, **kwargs):
        Path(path).write_bytes(b"fake-png-contract-only")


class GenericDiffusersTests(unittest.TestCase):
    def test_json_configuration_preserves_relative_paths_booleans_and_cli_precedence(self):
        config = self.directory / 'request.json'
        config.write_text(json.dumps({'model': 'model', 'prompt': 'configured', 'steps': 4,
            'overwrite': True, 'default_modifiers': False, 'cache_dir': 'cache',
            'moss_text_encoder': 'encoder.safetensors'}))
        args = generic.build_parser().parse_args(['--config', str(config), '--steps', '9'])
        self.assertEqual(Path(args.model), self.model)
        self.assertEqual(args.prompt, 'configured')
        self.assertEqual(args.steps, 9)
        self.assertTrue(args.overwrite)
        self.assertFalse(args.default_modifiers)
        self.assertEqual(args.cache_dir, self.directory / 'cache')
        self.assertEqual(Path(args.moss_text_encoder), self.directory / 'encoder.safetensors')

    def test_json_configuration_accepts_typed_pipeline_inputs_and_sigmas(self):
        config = self.directory / 'typed-request.json'
        config.write_text(json.dumps({'model': 'model', 'pipeline_inputs': {'height': 512, 'strength': 0.6},
            'tensor_outputs': {'latents': {'semantic': 'latent'}}, 'sigmas': [1.0, 0.0]}))
        args = generic.build_parser().parse_args(['--config', str(config)])
        self.assertEqual(json.loads(args.pipeline_inputs), {'height': 512, 'strength': 0.6})
        self.assertEqual(json.loads(args.tensor_outputs), {'latents': {'semantic': 'latent'}})
        self.assertEqual(args.sigmas, [1.0, 0.0])
        config.write_text(json.dumps({'model': 'model', 'pipeline_inputs': '@inputs.json'}))
        args = generic.build_parser().parse_args(['--config', str(config)])
        self.assertEqual(args.pipeline_inputs, '@' + str(self.directory / 'inputs.json'))

    def test_json_configuration_resolves_missing_gguf_and_pth_as_local_paths(self):
        config = self.directory / 'local-request.json'
        for name in ('missing.gguf', 'missing.pth'):
            with self.subTest(name=name):
                config.write_text(json.dumps({'model': name}))
                parsed = generic.build_parser().parse_args(['--config', str(config)])
                self.assertEqual(parsed.model, str(self.directory / name))

    def test_json_configuration_rejects_unrepresentable_or_invalid_booleans(self):
        import contextlib
        import io
        config = self.directory / 'invalid-request.json'
        for values in ({'overwrite': 'yes'}, {'local_files_only': False}):
            with self.subTest(values=values):
                config.write_text(json.dumps({'model': 'model', **values}))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    generic.build_parser().parse_args(['--config', str(config)])

    def test_moss_loading_progress_uses_the_existing_model_progress_protocol(self):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            generic.report_moss_loading_progress('text_encoder', 12, 40)
        prefix, payload = output.getvalue().strip().split(' ', 1)
        self.assertEqual(prefix, 'IILD_MODEL_PROGRESS')
        self.assertEqual(json.loads(payload), {'schema': 'iild-model-progress-v1',
                                               'component': 'text_encoder',
                                               'completed_bytes': 12, 'total_bytes': 40})

    def test_moss_progress_event_is_machine_readable_and_bounded(self):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            generic.report_moss_progress(2, 7)
        prefix, payload = output.getvalue().strip().split(' ', 1)
        self.assertEqual(prefix, 'IILD_DIFFUSERS_PROGRESS')
        self.assertEqual(json.loads(payload), {'schema': 'iild-diffusers-progress-v1',
                                               'stage': 'denoising', 'step': 2, 'total': 7})

    def test_flux_transformer_storage_rejects_other_model_sources(self):
        with self.assertRaisesRegex(ValueError, 'requires a FLUX.1 single-file checkpoint'):
            self.args('--flux-transformer-storage', 'cpu-stream')

    def test_flux_transformer_storage_rejects_offload(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        checkpoint = self.directory / 'flux.safetensors'
        checkpoint.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'FLUX transformer cpu-stream storage requires --offload none'):
            self.args('--model', str(checkpoint), '--model-config', str(self.model),
                      '--device', 'mps', '--offload', 'model', '--flux-transformer-storage', 'cpu-stream')

    def test_flux_transformer_storage_rejects_lora(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        checkpoint = self.directory / 'flux.safetensors'
        checkpoint.write_bytes(b'fixture')
        with patch.object(generic, 'resolve_lora_selection', return_value=object()):
            with self.assertRaisesRegex(ValueError, 'FLUX transformer cpu-stream storage does not support LoRA'):
                self.args('--model', str(checkpoint), '--model-config', str(self.model), '--no-default-modifiers',
                          '--flux-transformer-storage', 'cpu-stream')

    def test_moss_streamed_transformer_rejects_accelerate_offload(self):
        pipeline = SimpleNamespace(transformer=SimpleNamespace(_iild_cpu_stream=True))
        for mode in ('model', 'sequential'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'requires --offload none'):
                generic.prepare_execution(pipeline, 'mps', mode)

    def test_moss_transformer_storage_rejects_other_model_sources(self):
        with self.assertRaisesRegex(ValueError, 'requires a MOSS single-file checkpoint'):
            self.args('--moss-transformer-storage', 'cpu-stream')

    def test_moss_cpu_int8_rejects_accelerate_offload(self):
        pipeline = SimpleNamespace(text_encoder=SimpleNamespace(_iild_cpu_int8_convrot=True))
        for mode in ('model', 'sequential'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'requires --offload none'):
                generic.prepare_execution(pipeline, 'mps', mode)

    def test_moss_fallback_rejects_offload_before_loading(self):
        import moss_runtime
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                 **moss_runtime.COMPONENTS}))
        with patch.object(generic.moss_text_encoder, 'resolve_fallback',
                          return_value=(object(), 'fallback-int8-convrot')):
            with self.assertRaisesRegex(ValueError, 'requires --offload none'):
                self.args('--device', 'mps', '--offload', 'model')

    def test_moss_missing_encoder_is_reported_before_loading(self):
        import moss_runtime
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                 **moss_runtime.COMPONENTS}))
        with patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'unavailable')):
            with self.assertRaisesRegex(ValueError, 'MOSS.*Qwen3.5.*text encoder'):
                self.args('--no-default-modifiers')

    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = resource_fixture.TemporaryDirectory(prefix="generic-tests-")
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model"
        self.model.mkdir()
        self.index = {"_class_name": "FakePipeline", "unet": ["diffusers", "UNet2DModel"]}
        (self.model / "model_index.json").write_text(json.dumps(self.index))
        self.fake_images = SimpleNamespace(Image=FakeImage, open=Mock(return_value=FakeImage()))
        orientation = patch.object(generic, 'orient_image', side_effect=lambda image: image)
        orientation.start()
        self.addCleanup(orientation.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *tokens):
        return generic.resolve_arguments(generic.build_parser().parse_args([
            "--model", str(self.model), "--output-dir", str(self.directory / "output"), *tokens]))

    def provide_stub_moss_encoder(self):
        folder = self.model / 'text_encoder'
        folder.mkdir(exist_ok=True)
        (folder / 'model.safetensors').write_bytes(b'fixture')

    def test_moss_single_text_encoder_rejects_other_pipeline_families(self):
        with self.assertRaisesRegex(ValueError, 'requires MOSSImagePipeline'):
            self.args('--moss-text-encoder', str(self.directory / 'encoder.safetensors'))

    def fake_diffusers(self):
        class Base:
            pass
        class Builtin(Base, FakePipeline):
            __module__ = "diffusers.pipelines.test_builtin"
        Builtin.__name__ = "FakePipeline"
        Builtin.from_pretrained = Mock(side_effect=lambda *args, **kwargs: Builtin())
        Builtin.from_single_file = Mock(side_effect=lambda *args, **kwargs: Builtin())
        return SimpleNamespace(DiffusionPipeline=Base, FakePipeline=Builtin, pipelines=SimpleNamespace())

    def fake_gguf_diffusers(self):
        runtime = self.fake_diffusers()
        class Model: pass
        class Original: pass
        class Transformer(Model, Original):
            def named_parameters(self): return []
            def named_buffers(self): return []
        Transformer.from_single_file = Mock(side_effect=lambda *_args, **_kwargs: Transformer())
        runtime.ModelMixin, runtime.FromOriginalModelMixin, runtime.Transformer = Model, Original, Transformer
        runtime.GGUFQuantizationConfig = Mock(return_value=object())
        runtime.FakePipeline.__init__ = lambda self, transformer=None: None
        return runtime

    def test_omission_preserves_pipeline_defaults_without_injecting_prompt_or_seed(self):
        args = self.args()
        self.assertEqual(args.inputs, {})
        self.assertIsNone(args.seed)
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.offload, "none")

    def test_dreamscapes_image_count_and_disabled_hires_reach_the_pipeline(self):
        args = self.args('--num-images', '3', '--no-hires-fix', '--width', '512', '--height', '768')
        self.assertEqual(args.inputs['num_images_per_prompt'], 3)
        self.assertEqual((args.inputs['width'], args.inputs['height']), (512, 768))
        self.assertEqual(generic.configuration(args)['num_images'], 3)
        self.assertFalse(generic.configuration(args)['hires_fix'])
        single = self.args('--num-images', '1', '--no-hires-fix')
        self.assertNotIn('num_images_per_prompt', single.inputs)
        override = self.args('--pipeline-inputs', '{"num_images_per_prompt":4}', '--num-images', '1')
        self.assertEqual(override.inputs['num_images_per_prompt'], 1)
        with self.assertRaisesRegex(ValueError, 'single prompt'):
            self.args('--pipeline-inputs', '{"prompt":["one","two"]}', '--num-images', '2')
        def accepts_count(prompt=None, width=None, height=None, num_images_per_prompt=1):
            pass
        generic.validate_call_arguments(accepts_count, args.inputs, seed=None)
        with self.assertRaisesRegex(ValueError, 'num_images_per_prompt'):
            generic.validate_call_arguments(FakePipeline.__call__, args.inputs, seed=None)
        for count in ('0', '1001'):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'num-images'):
                self.args('--num-images', count, '--no-hires-fix')

    def test_legacy_checkpoint_converts_for_generic_loader_and_retains_original(self):
        import checkpoint_conversion
        original = self.directory / 'legacy.safetensors'
        original.write_bytes(b'PK\x03\x04' + bytes(20))
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        converted = self.directory / 'converted.safetensors'
        converted.write_bytes(struct.pack('<Q', len(header)) + header + bytes(4))
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(original),
                      'output': checkpoint_conversion._identity(converted)}
        tokens = ('--model', str(original), '--model-config', str(self.model),
                  '--num-images', '2', '--no-hires-fix', '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as materialize:
            args = self.args(*tokens)
            self.assertEqual(args.model, str(converted))
            self.assertEqual(generic.configuration(args)['model'], str(original))
            self.assertEqual(args.model_conversion, conversion)
            self.assertEqual(args.inputs['num_images_per_prompt'], 2)
            materialize.assert_called_once_with(str(original), args.cache_dir / 'checkpoint-conversions')
        with patch.object(checkpoint_conversion, 'materialize_safetensors') as materialize:
            with self.assertRaisesRegex(ValueError, 'inspect-model'):
                self.args(*tokens, '--print-config')
            with self.assertRaisesRegex(ValueError, 'model-config'):
                self.args('--model', str(original), '--model-config', str(self.directory / 'missing'))
            output = self.directory / 'output'
            output.mkdir()
            (output / 'existing.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'not empty'):
                self.args(*tokens)
            materialize.assert_not_called()

    def test_generic_conversion_rejects_changed_source_or_converted_weights(self):
        import checkpoint_conversion
        original = self.directory / 'legacy.ckpt'
        converted = self.directory / 'converted.safetensors'
        for changed in ('original', 'converted'):
            with self.subTest(changed=changed):
                original.write_bytes(b'PK\x03\x04' + bytes(20))
                converted.write_bytes(b'converted fixture')
                conversion = {'converted': True, 'converted_path': str(converted),
                              'original': checkpoint_conversion._identity(original),
                              'output': checkpoint_conversion._identity(converted)}
                def materialize(*_args):
                    (original if changed == 'original' else converted).write_bytes(b'changed weights')
                    return conversion
                with patch.object(checkpoint_conversion, 'materialize_safetensors', side_effect=materialize):
                    with self.assertRaisesRegex(RuntimeError, '(Source|Converted) checkpoint changed'):
                        self.args('--model', str(original), '--model-config', str(self.model),
                                  '--no-hires-fix', '--no-default-modifiers')

    def test_explicit_pytorch_dequantization_reaches_checkpoint_conversion(self):
        import checkpoint_conversion
        original = self.directory / 'quantized.pt';original.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'converted fixture')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(original),
                      'output': checkpoint_conversion._identity(converted)}
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as materialize:
            args = self.args('--model', str(original), '--model-config', str(self.model),
                             '--dequantize-pytorch', '--no-default-modifiers')
        self.assertTrue(generic.configuration(args)['dequantize_pytorch'])
        materialize.assert_called_once_with(str(original), args.cache_dir / 'checkpoint-conversions',
                                            quantized_tensors='dequantize')

    def test_explicit_pytorch_dequantization_reaches_directory_component_conversion(self):
        import checkpoint_conversion
        import directory_checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.bin';source.write_bytes(b'legacy weights')
        safe = self.directory / 'converted.safetensors';safe.write_bytes(b'safe weights')
        record = {'converted': True, 'converted_path': str(safe),
                  'original': checkpoint_conversion._identity(source),
                  'output': checkpoint_conversion._identity(safe)}
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=record) as materialize:
            with directory_checkpoint_conversion.prepare(self.model, self.index, self.directory / 'cache',
                    self.directory / 'output', None, quantized_tensors='dequantize') as prepared:
                self.assertEqual((prepared.path / 'unet' / 'diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
        materialize.assert_called_once_with(source, self.directory / 'cache/checkpoint-conversions',
                                            quantized_tensors='dequantize')

    def test_dequantization_configuration_accepts_boolean_and_cli_override(self):
        config = self.directory / 'quantized-request.json'
        config.write_text(json.dumps({'model': str(self.model), 'dequantize_pytorch': True}))
        parser = generic.build_parser()
        self.assertTrue(parser.parse_args(['--config', str(config)]).dequantize_pytorch)
        self.assertFalse(parser.parse_args(['--config', str(config), '--no-dequantize-pytorch']).dequantize_pytorch)
        self.assertTrue(parser.parse_values({'model': str(self.model), 'dequantize_pytorch': True}).dequantize_pytorch)

    def test_dequantization_mode_separates_retained_pipeline_cache(self):
        args = self.args('--no-default-modifiers')
        pipeline = object()
        keys = []
        def cached(key, *_args):
            keys.append(key)
            return (pipeline, {'identity': {}}, None, None), False
        with patch.object(generic, 'load_model_index', return_value=({'_class_name': 'StableDiffusionPipeline'}, self.model)), \
             patch.object(generic, 'cached_pipeline', side_effect=cached), \
             patch.object(generic, 'cached_placement', return_value=((pipeline, 'none'), False)), \
             patch.object(generic, 'verify_identity'), patch.object(generic, 'validate_base_model', return_value={}), \
             patch.object(generic, 'validate_lora_activation'):
            for enabled in (False, True):
                args.dequantize_pytorch = enabled
                generic.prepared_pipeline(args, self.fake_diffusers(), 'float32', 'cpu')
        self.assertEqual(len(keys), 2)
        self.assertIsNotNone(keys[0])
        self.assertNotEqual(keys[0], keys[1])


    def test_generic_publication_rechecks_original_legacy_source(self):
        import checkpoint_conversion
        original = self.directory / 'legacy.ckpt'
        original.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.directory / 'converted.safetensors'
        converted.write_bytes(b'converted fixture')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(original),
                      'output': checkpoint_conversion._identity(converted)}
        args = SimpleNamespace(output_dir=self.directory / 'out', overwrite=False, model_conversion=conversion)
        def mutate(*_args):
            original.write_bytes(b'changed during publication')
            return []
        with patch.object(generic, 'save_outputs', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'Source checkpoint changed'):
                generic.publish_generation(SimpleNamespace(), SimpleNamespace(), args, self.fake_images, None, {})
        self.assertFalse((args.output_dir / 'generation.json').exists())

    def test_generic_output_directory_cannot_contain_converted_model_sources(self):
        import checkpoint_conversion
        output = self.directory / 'protected-output'
        output.mkdir()
        source, shard, converted = (output / 'index.json', output / 'image-0001.png',
                                    self.directory / 'converted.safetensors')
        for path in (source, shard, converted): path.write_bytes(b'original weights')
        conversion = {'original': checkpoint_conversion._identity(source),
                      'original_shards': [checkpoint_conversion._identity(shard)],
                      'converted_path': str(converted), 'output': checkpoint_conversion._identity(converted)}
        args = SimpleNamespace(output_dir=output, overwrite=True, model_conversion=conversion)
        with patch.object(generic, 'save_outputs') as save:
            with self.assertRaisesRegex(ValueError, 'model input'):
                generic.publish_generation(SimpleNamespace(), SimpleNamespace(), args, self.fake_images, None, {})
            save.assert_not_called()
        self.assertEqual(source.read_bytes(), b'original weights')
        self.assertEqual(shard.read_bytes(), b'original weights')

    def test_generic_hires_is_rejected_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'HiRes.*--no-hires-fix'):
            self.args('--hires-fix', '--hires-upscaler', 'lanczos', '--hires-strength', '0.35')
        with self.assertRaisesRegex(ValueError, 'HiRes'):
            self.args('--no-hires-fix', '--hires-strength', '0.35')

    def test_moss_hires_plans_app_target_and_rejects_invalid_settings(self):
        import moss_runtime
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                                 **moss_runtime.COMPONENTS}))
        self.provide_stub_moss_encoder()
        args = self.args('--hires-fix', '--width', '512', '--height', '768', '--steps', '10',
                         '--num-images', '2', '--hires-upscaler', 'bicubic', '--hires-strength', '0.4',
                         '--no-default-modifiers')
        self.assertEqual(args.moss_hires.base_size, (256, 384))
        self.assertEqual(args.moss_hires.target_size, (512, 768))
        self.assertEqual(args.moss_hires.strength, .4)
        self.assertEqual(args.moss_hires.upscaler, 'bicubic')
        self.assertEqual(generic.configuration(args)['hires']['target_size'], [512, 768])
        for tokens in (('--width', '30', '--height', '64'),
                       ('--width', '64', '--height', '64', '--hires-strength', '0'),
                       ('--width', '64', '--height', '64', '--hires-strength', 'nan'),
                       ('--width', '64', '--height', '64', '--pipeline-inputs', '{"output_type":"np"}')):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, 'HiRes'):
                self.args('--hires-fix', '--no-default-modifiers', *tokens)

    def test_moss_hires_runs_base_and_image_refinement_with_requested_dimensions(self):
        import moss_hires
        from unittest.mock import Mock
        class ImageFixture:
            def __init__(self, size):
                self.size = size
            def resize(self, size, resample):
                return ImageFixture(size)
        Image = SimpleNamespace(Image=ImageFixture, Resampling=SimpleNamespace(
            NEAREST=0, BILINEAR=1, BICUBIC=2, LANCZOS=3))
        first = ImageFixture((32, 32))
        final = ImageFixture((64, 64))
        def execute(**values):
            values['progress_callback'](1, 1)
            return SimpleNamespace(images=[first])
        pipeline = Mock(side_effect=execute)
        plan = moss_hires.Plan((32, 32), (64, 64), .35, 'lanczos')
        def refine(_pipeline, images, values, _plan, _torch):
            self.assertEqual(images[0].size, (64, 64))
            values['progress_callback'](1, 1)
            return SimpleNamespace(images=[final])
        with patch.object(moss_hires, 'refine', side_effect=refine):
            result, report = moss_hires.run(pipeline, {'prompt': 'example', 'width': 64, 'height': 64},
                                            plan, expected_count=1, Image=Image, torch=None)
        self.assertIs(result.images[0], final)
        base = pipeline.call_args.kwargs
        self.assertEqual((base['width'], base['height']), (32, 32))
        self.assertEqual(report['completed_passes'], 1)
        def wrong_size(_pipeline, _images, values, _plan, _torch):
            values['progress_callback'](1, 1)
            return SimpleNamespace(images=[first])
        with patch.object(moss_hires, 'refine', side_effect=wrong_size):
            with self.assertRaisesRegex(RuntimeError, 'HiRes'):
                moss_hires.run(pipeline, {}, plan, expected_count=1, Image=Image, torch=None)
        no_steps = Mock(return_value=SimpleNamespace(images=[first]))
        with self.assertRaisesRegex(RuntimeError, 'denoising sequence'):
            moss_hires.run(no_steps, {}, plan, expected_count=1, Image=Image, torch=None)

    def test_moss_hires_does_not_drop_incompatible_settings(self):
        import moss_hires
        args = SimpleNamespace(hires_fix=True, hires_upscaler=None, hires_strength=None,
                               offload='none', generic_controlnet=None, moss_regions=None, moss_rapid=None,
                               inputs={'prompt': 'scene', 'width': 64, 'height': 64}, tensor_outputs={})
        for name, value in (('offload', 'model'), ('generic_controlnet', object()),
                            ('moss_regions', object()), ('moss_rapid', object())):
            with self.subTest(name=name), patch.object(args, name, value):
                with self.assertRaisesRegex(ValueError, 'MOSS HiRes'):
                    moss_hires.resolve(args, 'MOSSImagePipeline')
        for name in ('initial_image', 'latents', 'prompt_embeds'):
            with self.subTest(name=name), patch.dict(args.inputs, {name: 'unsupported'}):
                with self.assertRaisesRegex(ValueError, 'MOSS HiRes'):
                    moss_hires.resolve(args, 'MOSSImagePipeline')

    def test_requested_image_count_is_checked_before_publication(self):
        result = SimpleNamespace(images=[FakeImage(), FakeImage()])
        generic.validate_requested_image_count(result, 2, self.fake_images)
        with self.assertRaisesRegex(ValueError, 'image count'):
            generic.validate_requested_image_count(result, 3, self.fake_images)
        with self.assertRaisesRegex(ValueError, 'image count'):
            generic.validate_requested_image_count(SimpleNamespace(images=None), 1, self.fake_images)

    def test_klein_comfy_vae_rejects_missing_normalization_before_override(self):
        import klein_reference_resize
        pipeline = SimpleNamespace(vae=SimpleNamespace(bn=SimpleNamespace(), config=SimpleNamespace(batch_norm_eps=1e-4)),
                                   _encode_vae_image=lambda **kwargs: None, _patchify_latents=lambda value: value)
        original = pipeline._encode_vae_image
        with self.assertRaisesRegex(ValueError, 'VAE normalization'):
            with klein_reference_resize.comfy_vae_execution(pipeline, None):
                pass
        self.assertIs(pipeline._encode_vae_image, original)

    def test_flux2_klein_reference_control_resolves_lora_and_ordered_images(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Flux2KleinPipeline'}))
        adapter, hint, reference = (self.directory / name for name in ('control.safetensors', 'depth.png', 'reference.png'))
        for path in (adapter, hint, reference):
            path.write_bytes(b'fixture')
        inputs = json.dumps({'image': {'image_path': str(reference)}})
        args = self.args('--controlnet', str(adapter), '--controlnet-format', 'flux2-klein-reference',
                         '--control-image', str(hint), '--controlnet-scale', '.8',
                         '--pipeline-inputs', inputs, '--no-default-modifiers')
        self.assertEqual(args.generic_controlnet.metadata()['kind'], 'flux2-klein-reference-lora')
        self.assertEqual(args.lora_selection.local_file.path, str(adapter))
        self.assertEqual(args.lora_selection.scale, .8)
        self.assertEqual(args.inputs['image'], [{'image_path': str(hint), 'mode': 'RGB'},
                                                {'image_path': str(reference)}])
        resized = self.args('--controlnet', str(adapter), '--controlnet-format', 'flux2-klein-reference',
                            '--control-image', str(hint), '--pipeline-inputs', inputs,
                            '--control-reference-resize', 'comfy-1mp', '--no-default-modifiers')
        self.assertEqual(resized.generic_controlnet.metadata()['reference_resize'], 'comfy-1mp')

        for extra, message in ((('--control-guidance-end', '.5'), 'full denoising'),
                               (('--controlnet-scale', '-1'), 'scale'),
                               (('--lora', str(adapter)), 'another LoRA')):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.args('--controlnet', str(adapter), '--controlnet-format', 'flux2-klein-reference',
                          '--control-image', str(hint), '--pipeline-inputs', inputs,
                          '--no-default-modifiers', *extra)
        with self.assertRaisesRegex(ValueError, 'one RGB image_path'):
            self.args('--controlnet', str(adapter), '--controlnet-format', 'flux2-klein-reference',
                      '--control-image', str(hint), '--no-default-modifiers')
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--control-reference-resize', 'comfy-1mp')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        with self.assertRaisesRegex(ValueError, 'requires flux2-klein-reference'):
            self.args('--controlnet', str(adapter), '--control-image', str(hint),
                      '--control-reference-resize', 'comfy-1mp')

    def test_explicit_cli_inputs_override_json_without_discarding_task_values(self):
        args = self.args("--pipeline-inputs", '{"prompt":"old","num_frames":17,"guidance_scale":3}',
                         "--prompt", "", "--guidance-scale", "0", "--steps", "1")
        self.assertEqual(args.inputs, {"prompt": "", "num_frames": 17, "guidance_scale": 0.0,
                                      "num_inference_steps": 1})

    def test_generic_flux_controlnet_resolves_existing_prepared_image_and_options(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        control = self.directory / 'control'
        control.mkdir()
        (control / 'config.json').write_text('{"_class_name":"FluxControlNetModel"}')
        (control / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        image = self.directory / 'hint.png'
        image.write_bytes(b'fixture')
        args = self.args('--controlnet', str(control), '--control-image', str(image),
                         '--controlnet-scale', '0.7', '--control-guidance-start', '0.2',
                         '--control-guidance-end', '0.8')
        self.assertEqual(args.generic_controlnet.target_pipeline, 'FluxControlNetPipeline')
        self.assertEqual(args.inputs['control_image'], {'image_path': str(image), 'mode': 'RGB'})
        self.assertEqual(args.inputs['controlnet_conditioning_scale'], .7)
        self.assertEqual((args.inputs['control_guidance_start'], args.inputs['control_guidance_end']), (.2, .8))
        self.assertEqual(generic.configuration(args)['controlnet']['component_class'], 'FluxControlNetModel')

    def test_sd3_controlnet_resolves_and_checks_transformer_contract(self):
        import generic_controlnet as control
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'StableDiffusion3Pipeline'}))
        folder = self.directory / 'sd3-control'
        folder.mkdir()
        (folder / 'config.json').write_text('{"_class_name":"SD3ControlNetModel"}')
        (folder / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        image = self.directory / 'hint.png'
        image.write_bytes(b'fixture')
        args = self.args('--controlnet', str(folder), '--control-image', str(image), '--control-preprocess', 'canny')
        selected = args.generic_controlnet
        self.assertEqual(selected.target_pipeline, 'StableDiffusion3ControlNetPipeline')
        self.assertEqual(selected.component_class, 'SD3ControlNetModel')
        self.assertEqual(args.inputs['control_image']['image_path'], str(image))
        cfg = dict(patch_size=2, in_channels=16, attention_head_dim=64, num_attention_heads=24,
                   joint_attention_dim=4096, caption_projection_dim=1536, pooled_projection_dim=2048)
        model = type('SD3ControlNetModel', (), {})()
        model.config = SimpleNamespace(**cfg, num_layers=6, extra_conditioning_channels=0)
        model.pos_embed = model.context_embedder = object()
        base = SimpleNamespace(transformer=SimpleNamespace(config=SimpleNamespace(**cfg, num_layers=24)),
                               vae=SimpleNamespace(config=SimpleNamespace(latent_channels=16)))
        control.validate_component(model, base, selected, args)
        model.config.in_channels = 17
        with self.assertRaisesRegex(ValueError, 'in_channels'):
            control.validate_component(model, base, selected, args)
        model.config.in_channels = 16
        model.config.extra_conditioning_channels = 1
        with self.assertRaisesRegex(ValueError, 'conditioning'):
            control.validate_component(model, base, selected, args)
        model.config.extra_conditioning_channels = 0
        model.config.num_layers = 23
        with self.assertRaisesRegex(ValueError, 'blocks'):
            control.validate_component(model, base, selected, args)
        model.config.num_layers = 6
        model.pos_embed = None
        with self.assertRaisesRegex(ValueError, 'embedding'):
            control.validate_component(model, base, selected, args)

    def test_sd3_control_vae_sampling_uses_an_independent_seed_and_restores_encode(self):
        import generic_controlnet as control
        generator = SimpleNamespace(manual_seed=Mock())
        generator.manual_seed.return_value = generator
        torch = SimpleNamespace(Generator=Mock(return_value=generator))
        sample = Mock(return_value='latent')
        distribution = SimpleNamespace(sample=sample)
        class Vae:
            def encode(self, pixels):
                return SimpleNamespace(latent_dist=distribution)
        vae = Vae()
        selection = SimpleNamespace(family='sd3')
        derived = object()
        with patch.object(control, 'derive_generator', return_value=derived) as derive, \
             self.assertRaisesRegex(RuntimeError, 'request failed'):
            with control.sd3_control_seed(SimpleNamespace(vae=vae), selection, 123, torch):
                result = vae.encode('pixels')
                self.assertEqual(result.latent_dist.sample(), 'latent')
                sample.assert_called_once_with(generator=derived)
                raise RuntimeError('request failed')
        self.assertNotIn('encode', vars(vae))
        generator.manual_seed.assert_called_once_with(123)
        derive.assert_called_once_with(generator, b'iild-sd3-control-vae-v1\0', torch)

    def test_depth_preprocessing_resolves_with_flux_control_and_requires_image(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        control = self.directory / 'control'
        control.mkdir()
        (control / 'config.json').write_text('{"_class_name":"FluxControlNetModel"}')
        (control / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        image = self.directory / 'input.png'
        image.write_bytes(b'fixture')
        depth = self.directory / 'depth'
        depth.mkdir()
        (depth / 'config.json').write_text('{"model_type":"depth_anything"}')
        (depth / 'preprocessor_config.json').write_text('{"image_processor_type":"DPTImageProcessor"}')
        (depth / 'model.safetensors').write_bytes(b'fixture')
        tokens = ('--controlnet', str(control), '--control-preprocess', 'depth', '--control-depth-model', str(depth))
        args = self.args(*tokens, '--control-image', str(image), '--control-depth-dtype', 'float16')
        self.assertEqual(generic.configuration(args)['control_depth_model'], str(depth.resolve()))
        self.assertEqual(generic.configuration(args)['control_depth_dtype'], 'float16')
        with self.assertRaisesRegex(ValueError, 'requires --control-image'):
            self.args(*tokens)
        with patch.object(generic, 'is_preparing', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Foreground preparation.*ControlNet'):
                self.args(*tokens, '--control-image', str(image))

    def test_flux2_fun_control_accepts_an_independent_style_lora(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Flux2Pipeline',
            'vae': ['diffusers', 'AutoencoderKLFlux2']}))
        (self.model / 'vae').mkdir()
        (self.model / 'vae/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        control = self.directory / 'control.safetensors'
        style = self.directory / 'style.safetensors'
        image = self.directory / 'hint.png'
        for path in (control, style, image):
            path.write_bytes(b'fixture')
        args = self.args('--controlnet', str(control), '--control-image', str(image), '--seed', '7',
                         '--lora', str(style), '--no-default-modifiers')
        self.assertEqual(args.generic_controlnet.target_pipeline, 'Flux2Pipeline')
        self.assertEqual(args.lora_selection.local_file.path, str(style))

    def test_qwen_image_controlnet_uses_matching_component_and_preprocessor(self):
        import generic_controlnet
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'QwenImagePipeline',
                                                                  'vae': ['diffusers', 'AutoencoderKLQwenImage']}))
        (self.model / 'vae').mkdir()
        (self.model / 'vae/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        folder = self.directory / 'control'
        folder.mkdir()
        (folder / 'config.json').write_text('{"_class_name":"QwenImageControlNetModel"}')
        (folder / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        image = self.directory / 'hint.png'
        image.write_bytes(b'image')
        args = self.args('--controlnet', str(folder), '--control-image', str(image), '--control-preprocess', 'canny',
                         '--control-guidance-start', '0.2', '--control-guidance-end', '0.8')
        self.assertEqual(args.generic_controlnet.target_pipeline, 'QwenImageControlNetPipeline')
        self.assertEqual(args.inputs['control_image']['image_path'], str(image))
        single = self.args('--controlnet', str(folder / 'diffusion_pytorch_model.safetensors'),
                           '--control-image', str(image))
        self.assertEqual(single.generic_controlnet.weight.path, str(folder / 'diffusion_pytorch_model.safetensors'))
        for extra in (('--guess-mode',), ('--control-mode', '1')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args('--controlnet', str(folder), '--control-image', str(image), *extra)
        common = dict(patch_size=2, in_channels=64, attention_head_dim=8, num_attention_heads=2,
                      joint_attention_dim=32, axes_dims_rope=(2, 2, 4))
        vae = type('AutoencoderKLQwenImage', (), {'config': SimpleNamespace(z_dim=16),
                                                   'spatial_compression_ratio': 8})()
        base = SimpleNamespace(transformer=SimpleNamespace(config=SimpleNamespace(**common, num_layers=4)), vae=vae)
        model = type('QwenImageControlNetModel', (), {})()
        model.config = SimpleNamespace(**common, num_layers=2, extra_condition_channels=0)
        generic_controlnet.validate_component(model, base, args.generic_controlnet, args)
        for field, value in (('joint_attention_dim', 16), ('extra_condition_channels', 4), ('num_layers', 0),
                             ('num_layers', 3)):
            saved = getattr(model.config, field)
            setattr(model.config, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                generic_controlnet.validate_component(model, base, args.generic_controlnet, args)
            setattr(model.config, field, saved)
        vae.config.input_channels = 4
        with self.assertRaisesRegex(ValueError, 'RGB'):
            generic_controlnet.validate_component(model, base, args.generic_controlnet, args)

    def test_qwen_published_union_configuration_matches_base_contract(self):
        import generic_controlnet
        common = dict(patch_size=2, in_channels=64, attention_head_dim=128, num_attention_heads=24,
                      joint_attention_dim=3584, axes_dims_rope=(16, 56, 56))
        vae = type('AutoencoderKLQwenImage', (), {'config': SimpleNamespace(z_dim=16, input_channels=3),
                                                   'spatial_compression_ratio': 8})()
        base = SimpleNamespace(transformer=SimpleNamespace(config=SimpleNamespace(**common, num_layers=60)), vae=vae)
        model = type('QwenImageControlNetModel', (), {})()
        model.config = SimpleNamespace(**common, num_layers=5, extra_condition_channels=0)
        selection = generic_controlnet.Selection('', 'QwenImagePipeline', 'QwenImageControlNetPipeline',
                                                  'QwenImageControlNetModel', 'qwen-image', 'control_image', None)
        generic_controlnet.validate_component(model, base, selection, SimpleNamespace())

    def test_generic_controlnet_rejects_cross_family_and_ambiguous_inputs(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        control = self.directory / 'control'
        control.mkdir()
        config = control / 'config.json'
        config.write_text('{"_class_name":"ControlNetModel"}')
        (control / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        image = self.directory / 'hint.png'
        image.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'FluxControlNetModel'):
            self.args('--controlnet', str(control), '--control-image', str(image))
        config.write_text('{"_class_name":"FluxControlNetModel"}')
        for extra in (('--controlnet-scale', '-1'), ('--control-guidance-start', '1'),
                      ('--pipeline-inputs', '{"control_image":"duplicate"}')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args('--controlnet', str(control), '--control-image', str(image), *extra)
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--control-image', str(image))
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--controlnet-config', str(control))

    def test_generic_flux_accepts_a_single_file_and_separate_configuration(self):
        import generic_controlnet
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        control = self.directory / 'flux-control.safetensors'
        control.write_bytes(b'control fixture')
        config = self.directory / 'control-config'
        config.mkdir()
        (config / 'config.json').write_text('{"_class_name":"FluxControlNetModel"}')
        image = self.directory / 'hint.png'
        image.write_bytes(b'image')
        args = self.args('--controlnet', str(control), '--controlnet-config', str(config), '--control-image', str(image))
        self.assertEqual(args.generic_controlnet.weight.path, str(control))
        self.assertEqual(args.generic_controlnet.config.path, str(config / 'config.json'))
        self.assertEqual(generic.configuration(args)['controlnet']['weight']['sha256'], hashlib.sha256(control.read_bytes()).hexdigest())
        generic_controlnet.verify_selection(args.generic_controlnet)
        (config / 'config.json').write_text('{"_class_name":"ControlNetModel"}')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic_controlnet.verify_selection(args.generic_controlnet)

    def test_single_file_control_configuration_and_variant_are_unambiguous(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        control = self.directory / 'flux-control.safetensors'
        control.write_bytes(b'control fixture')
        image = self.directory / 'hint.png'
        image.write_bytes(b'image')
        with self.assertRaisesRegex(ValueError, 'config'):
            self.args('--controlnet', str(control), '--control-image', str(image))
        (self.directory / 'config.json').write_text('{"_class_name":"FluxControlNetModel"}')
        args = self.args('--controlnet', str(control), '--control-image', str(image))
        self.assertEqual(args.generic_controlnet.config.path, str(self.directory / 'config.json'))
        with self.assertRaisesRegex(ValueError, 'variant'):
            self.args('--controlnet', str(control), '--control-image', str(image), '--controlnet-variant', 'fp16')
        with self.assertRaisesRegex(ValueError, 'single-file'):
            self.args('--controlnet', str(self.directory), '--control-image', str(image),
                      '--controlnet-config', str(self.directory))

    def test_original_sd_file_uses_existing_converter_and_flux_rejects_it(self):
        import generic_controlnet
        weight = self.directory / 'legacy.safetensors'
        weight.write_bytes(b'control fixture')
        config = self.directory / 'config.json'
        config.write_text('{"_class_name":"ControlNetModel"}')
        image = self.directory / 'hint.png'
        image.write_bytes(b'image')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'StableDiffusionPipeline'}))
        args = self.args('--controlnet', str(weight), '--control-image', str(image))
        args.cache_dir = self.directory / 'cache'
        wrapper = type('StableDiffusionControlNetPipeline', (), {})
        wrapper.from_pipe = Mock(return_value=wrapper())
        diffusers = SimpleNamespace(ControlNetModel=object(), StableDiffusionControlNetPipeline=wrapper)
        component = object()
        with patch('generic_controlnet.read_weight_keys', return_value={'control_model.input_blocks.0'}), \
                patch('generic_controlnet._load_original_controlnet', return_value=component) as converter, \
                patch('generic_controlnet.require_materialized_component'), \
                patch('generic_controlnet.validate_controlnet_contract'):
            generic_controlnet.attach(object(), args.generic_controlnet, args, diffusers, 'float32')
        converter.assert_called_once()
        wrapper.from_pipe.assert_called_once()
        config.write_text('{"_class_name":"FluxControlNetModel"}')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline'}))
        args = self.args('--controlnet', str(weight), '--control-image', str(image))
        args.cache_dir = self.directory / 'cache'
        diffusers = SimpleNamespace(FluxControlNetModel=object(), FluxControlNetPipeline=object())
        with patch('generic_controlnet.read_weight_keys', return_value={'control_model.input_blocks.0'}):
            with self.assertRaisesRegex(ValueError, 'native Diffusers'):
                generic_controlnet.attach(object(), args.generic_controlnet, args, diffusers, 'float32')

    def test_generic_flux_attaches_checked_controlnet_before_device_placement(self):
        from generic_controlnet import Selection, attach
        selection = Selection(str(self.directory), 'FluxPipeline', 'FluxControlNetPipeline',
                              'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
        values = dict(in_channels=64, patch_size=1, attention_head_dim=16, num_attention_heads=4,
                      joint_attention_dim=32, pooled_projection_dim=16, axes_dims_rope=(2, 2, 2))
        component = type('FluxControlNetModel', (), {
            'config': SimpleNamespace(**values), 'union': False,
            'named_parameters': lambda self: [], 'named_buffers': lambda self: []})()
        base = SimpleNamespace(transformer=SimpleNamespace(config=SimpleNamespace(**values)))
        wrapper = type('FluxControlNetPipeline', (), {})
        wrapper.from_pipe = Mock(return_value=wrapper())
        loader = object()
        diffusers = SimpleNamespace(FluxControlNetModel=loader, FluxControlNetPipeline=wrapper)
        with patch('generic_controlnet._load_package', return_value=component) as load:
            result = attach(base, selection, SimpleNamespace(control_mode=None), diffusers, 'f32')
        self.assertIs(type(result), wrapper)
        self.assertTrue(load.call_args.args[2]['local_files_only'])
        wrapper.from_pipe.assert_called_once_with(base, dtype='f32', controlnet=component)
        component.config.in_channels = 32
        with patch('generic_controlnet._load_package', return_value=component):
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                attach(base, selection, SimpleNamespace(control_mode=None), diffusers, 'f32')

    def test_generic_controlnet_unknown_family_fails_instead_of_silently_dropping_control(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'OtherPipeline'}))
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            self.args('--controlnet', str(self.directory), '--control-image', str(self.model / 'model_index.json'))
        with self.assertRaisesRegex(ValueError, 'requires Krea2Pipeline'):
            self.args('--controlnet', str(self.directory), '--control-image', str(self.model / 'model_index.json'),
                      '--krea2-control-auto-bucket')
        with self.assertRaisesRegex(ValueError, 'requires Krea2Pipeline'):
            self.args('--controlnet', str(self.directory), '--controlnet-format', 'krea2-color-depth')

    def test_flux_t5_streaming_requires_a_local_encoder_and_safe_composition(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'FluxPipeline',
            'text_encoder_2': ['transformers', 'T5EncoderModel']}))
        encoder = self.directory / 't5.safetensors'
        encoder.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'requires --offload none'):
            self.args('--flux-t5-encoder', str(encoder), '--flux-t5-fp8-storage', 'cpu-stream',
                      '--offload', 'model', '--no-default-modifiers')
        import struct
        lora = self.directory / 'style.safetensors'
        header = json.dumps({'text_encoder_2.encoder.block.0.q.lora_A.weight': {}}).encode()
        lora.write_bytes(struct.pack('<Q', len(header)) + header)
        with self.assertRaisesRegex(ValueError, 'T5 and ambiguous keys are unsupported'):
            self.args('--flux-t5-encoder', str(encoder), '--flux-t5-fp8-storage', 'cpu-stream',
                      '--lora', str(lora), '--no-default-modifiers')
        header = json.dumps({'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': {}}).encode()
        lora.write_bytes(struct.pack('<Q', len(header)) + header)
        args = self.args('--flux-t5-encoder', str(encoder), '--flux-t5-fp8-storage', 'cpu-stream',
                         '--lora', str(lora), '--no-default-modifiers')
        self.assertIsNotNone(args.lora_selection)

    def test_krea2_control_uses_a_seeded_single_file_without_forwarding_hint_to_pipeline(self):
        (self.model / 'text_encoder').mkdir()
        (self.model / 'text_encoder/model.safetensors').write_bytes(b'fixture')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Krea2Pipeline', 'is_distilled': True,
                                                                 'text_encoder': ['transformers', 'Qwen3VLModel']}))
        with self.assertRaisesRegex(ValueError, 'single-file checkpoint'):
            self.args('--krea2-fp8-storage', 'cpu-stream')
        fake_base = self.directory / 'base.safetensors'
        fake_base.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'requires --offload none'):
            self.args('--model', str(fake_base), '--model-config', str(self.model),
                      '--krea2-fp8-storage', 'cpu-stream', '--offload', 'model')
        weight = self.directory / 'control.safetensors'
        weight.write_bytes(b'local-control-fixture')
        hint = self.directory / 'depth.png'
        hint.write_bytes(b'local-image-fixture')
        with self.assertRaisesRegex(ValueError, 'requires --seed'):
            self.args('--controlnet', str(weight), '--control-image', str(hint))
        args = self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '17')
        self.assertEqual(args.generic_controlnet.weight.path, str(weight))
        self.assertNotIn('control_image', args.inputs)
        self.assertEqual(generic.configuration(args)['controlnet']['kind'], 'krea2-expanded-input-control-lora')
        style = self.directory / 'style.safetensors'
        style.write_bytes(b'local-style-fixture')
        composed = self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '17',
                             '--lora', str(style), '--no-default-modifiers')
        self.assertEqual(composed.lora_selection.local_file.path, str(style))
        auto = ('--controlnet', str(weight), '--control-image', str(hint), '--seed', '17', '--krea2-control-auto-bucket')
        selected = self.args(*auto)
        self.assertTrue(selected.generic_controlnet.auto_bucket)
        config = generic.configuration(selected)
        self.assertNotIn('width', config['pipeline_inputs'])
        self.assertNotIn('height', config['pipeline_inputs'])
        self.assertEqual(config['krea2']['control_resolution']['status'], 'pending-input-decode')
        for dimensions in (('--width', '1024'), ('--height', '1024'), ('--pipeline-inputs', '{"height":1024}')):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, 'width|height'):
                self.args(*auto, *dimensions)
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--krea2-control-auto-bucket')
        color = self.directory / 'color.png'
        color.write_bytes(b'local-color-fixture')
        both = self.args('--controlnet', str(weight), '--controlnet-format', 'krea2-color-depth',
                         '--control-image', str(hint), '--control-color-image', str(color), '--seed', '17',
                         '--control-color-end', '0.5', '--control-depth-end', '0.75')
        self.assertEqual(generic.configuration(both)['controlnet']['kind'], 'krea2-color-depth-control')
        self.assertEqual((both.generic_controlnet.color_end, both.generic_controlnet.depth_end), (0.5, 0.75))
        styled_color = self.args('--controlnet', str(weight), '--controlnet-format', 'krea2-color-depth',
                                 '--control-image', str(hint), '--seed', '17', '--lora', str(style),
                                 '--no-default-modifiers')
        self.assertEqual(styled_color.lora_selection.local_file.path, str(style))
        color_only = self.args('--controlnet', str(weight), '--controlnet-format', 'krea2-color-depth',
                               '--control-color-image', str(color), '--seed', '17')
        self.assertIsNone(color_only.generic_controlnet.depth)
        for mode in ('canny', 'depth'):
            with self.subTest(color_only_preprocessor=mode), self.assertRaisesRegex(ValueError, 'control.image'):
                self.args('--controlnet', str(weight), '--controlnet-format', 'krea2-color-depth',
                          '--control-color-image', str(color), '--seed', '17', '--control-preprocess', mode)
        with self.assertRaisesRegex(ValueError, 'valid control intervals'):
            self.args('--controlnet', str(weight), '--controlnet-format', 'krea2-color-depth',
                      '--control-color-image', str(color), '--seed', '17', '--control-color-end', '2')
        with self.assertRaises(ValueError):
            self.args('--controlnet', str(weight), '--control-color-image', str(color), '--seed', '17')
        for extra in (('--control-guidance-end', '0.5'), ('--offload', 'model'), ('--guess-mode',),
                      ('--controlnet-config', str(self.directory))):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '17', *extra)

    def test_krea2_regions_resolve_manifest_control_and_preprocessor(self):
        (self.model / 'text_encoder').mkdir()
        (self.model / 'text_encoder/model.safetensors').write_bytes(b'fixture')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Krea2Pipeline', 'is_distilled': True,
                                                                 'text_encoder': ['transformers', 'Qwen3VLModel']}))
        for name in ('map.png', 'depth.png', 'control.safetensors'):
            (self.directory / name).write_bytes(b'fixture')
        manifest = self.directory / 'regions.json'
        manifest.write_text(json.dumps({'map': 'map.png', 'regions': [{'color': '#ff0000', 'prompt': 'person'}],
            'control': {'adapter': 'control.safetensors', 'color_map': 'map.png', 'depth': 'depth.png', 'depth_end': .5}}))
        args = self.args('--krea2-regions', str(manifest), '--prompt', 'scene', '--seed', '17',
                         '--control-preprocess', 'canny')
        self.assertEqual(args.controlnet_format, 'krea2-color-depth')
        self.assertEqual(args.generic_controlnet.depth, str(self.directory / 'depth.png'))
        self.assertEqual(args.generic_controlnet.color, str(self.directory / 'map.png'))
        self.assertEqual(args.generic_controlnet.depth_end, .5)
        self.assertEqual(args.control_preprocess, 'canny')
        self.assertEqual(generic.configuration(args)['krea2_regions']['regions'][0]['prompt'], 'person')
        bucketed = self.args('--krea2-regions', str(manifest), '--prompt', 'scene', '--seed', '17',
                             '--krea2-control-auto-bucket')
        self.assertTrue(bucketed.generic_controlnet.auto_bucket)
        self.assertEqual(bucketed.krea2['control_resolution']['status'], 'pending-input-decode')

    def test_krea2_ostris_pose_selects_reference_adapter_without_forwarding_image(self):
        (self.model / 'text_encoder').mkdir()
        (self.model / 'text_encoder/model.safetensors').write_bytes(b'fixture')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Krea2Pipeline', 'is_distilled': True,
                                                                 'text_encoder': ['transformers', 'Qwen3VLModel']}))
        adapter, hint = self.directory / 'pose.safetensors', self.directory / 'pose.png'
        adapter.write_bytes(b'fixture')
        hint.write_bytes(b'fixture')
        selected = self.args('--controlnet-format', 'krea2-ostris-edit', '--controlnet', str(adapter),
                             '--control-image', str(hint), '--prompt', 'a dancer', '--seed', '9')
        self.assertEqual(selected.generic_controlnet.image, str(hint))
        self.assertEqual(selected.generic_controlnet.reference_attention, 'isolated')
        self.assertEqual(generic.configuration(selected)['controlnet']['kind'], 'krea2-ostris-edit')
        self.assertNotIn('control_image', selected.inputs)
        joint = self.args('--controlnet-format', 'krea2-ostris-edit', '--controlnet', str(adapter),
                          '--control-image', str(hint), '--prompt', 'a dancer', '--seed', '9',
                          '--krea2-edit-reference-attention', 'joint')
        self.assertEqual(joint.generic_controlnet.reference_attention, 'joint')
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--krea2-edit-reference-attention', 'joint')
        for extra in (('--control-guidance-end', '.5'), ('--offload', 'model'),
                      ('--control-color-image', str(hint))):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args('--controlnet-format', 'krea2-ostris-edit', '--controlnet', str(adapter),
                          '--control-image', str(hint), '--prompt', 'a dancer', '--seed', '9', *extra)

    def test_flux2_control_resolves_single_file_and_rejects_incompatible_inputs(self):
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Flux2Pipeline'}))
        vae = self.model / 'vae'
        vae.mkdir()
        (vae / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        weight = self.directory / 'control.safetensors'
        hint = self.directory / 'hint.png'
        weight.write_bytes(b'fixture')
        hint.write_bytes(b'fixture')
        args = self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12',
                         '--control-preprocess', 'canny', '--control-guidance-end', '0.8')
        self.assertEqual(args.generic_controlnet.image, str(hint))
        self.assertEqual(args.generic_controlnet.end, .8)
        self.assertNotIn('control_image', args.inputs)
        self.assertEqual(generic.configuration(args)['controlnet']['kind'], 'flux2-fun-controlnet')
        reference_input = json.dumps({'image': {'image_path': str(hint), 'mode': 'RGB'}})
        reference = self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12',
                              '--width', '64', '--height', '64', '--pipeline-inputs', reference_input)
        self.assertIn('image', reference.inputs)
        masked = self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12',
                           '--control-inpaint-image', str(hint), '--control-mask-image', str(hint))
        self.assertEqual(masked.generic_controlnet.inpaint_image, str(hint))
        self.assertEqual(masked.generic_controlnet.mask_image, str(hint))
        self.assertNotIn('inpaint_image', masked.inputs)
        self.assertNotIn('mask_image', masked.inputs)
        with self.assertRaisesRegex(ValueError, 'explicit options'):
            self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12',
                      '--pipeline-inputs', json.dumps({'mask_image': {'image_path': str(hint)}}))
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--control-mask-image', str(hint))
        with self.assertRaisesRegex(ValueError, 'width and height'):
            self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12',
                      '--pipeline-inputs', reference_input)
        for extra in (('--offload', 'model'), ('--control-mode', '1'),
                      ('--pipeline-inputs', '{"image":"other"}')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args('--controlnet', str(weight), '--control-image', str(hint), '--seed', '12', *extra)

    def test_moss_control_resolves_local_depth_and_color_as_separate_inputs(self):
        import moss_runtime
        index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        self.provide_stub_moss_encoder()
        vae = self.model / 'vae'
        vae.mkdir()
        (vae / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        adapter = self.directory / 'moss-control.safetensors'
        depth = self.directory / 'depth.png'
        color = self.directory / 'color.png'
        for path in (adapter, depth, color): path.write_bytes(b'fixture')
        args = self.args('--controlnet', str(adapter), '--control-image', str(depth),
                         '--control-color-image', str(color), '--seed', '23',
                         '--control-depth-end', '0.4', '--control-color-end', '0.8')
        self.assertEqual(args.generic_controlnet.depth, str(depth))
        self.assertEqual(args.generic_controlnet.color, str(color))
        self.assertEqual(args.generic_controlnet.depth_end, .4)
        self.assertEqual(args.generic_controlnet.color_end, .8)
        self.assertNotIn('control_image', args.inputs)
        self.assertEqual(generic.configuration(args)['controlnet']['kind'], 'moss-image-color-depth-control')
        training = self.args('--controlnet', str(adapter), '--control-image', str(depth),
                             '--control-color-image', str(color), '--seed', '23',
                             '--control-map-preprocess', 'training')
        self.assertEqual(training.generic_controlnet.map_preprocess, 'training')
        self.assertEqual(generic.configuration(training)['controlnet']['map_preprocess'], 'training')
        only_color = self.args('--controlnet', str(adapter), '--control-color-image', str(color), '--seed', '23')
        self.assertIsNone(only_color.generic_controlnet.depth)
        for mode in ('canny', 'depth'):
            with self.subTest(color_only_preprocessor=mode), self.assertRaisesRegex(ValueError, 'control.image'):
                self.args('--controlnet', str(adapter), '--control-color-image', str(color), '--seed', '23',
                          '--control-preprocess', mode)
        with self.assertRaisesRegex(ValueError, 'and/or'):
            self.args('--controlnet', str(adapter), '--seed', '23')
        with self.assertRaisesRegex(ValueError, 'intervals'):
            self.args('--controlnet', str(adapter), '--control-image', str(depth), '--seed', '23',
                      '--control-guidance-end', '0.5', '--control-depth-end', '0.8')

    def test_moss_region_manifest_routes_control_and_rapid_through_cli(self):
        import moss_runtime
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}))
        self.provide_stub_moss_encoder()
        vae = self.model / 'vae'
        vae.mkdir()
        (vae / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        for name in ('map.png', 'depth.png', 'control.safetensors'):
            (self.directory / name).write_bytes(b'fixture')
        rapid = self.directory / 'rapid'
        rapid.mkdir()
        for name in ('lora.safetensors', 'time_embedding.safetensors', 'dataspace.safetensors'):
            (rapid / name).write_bytes(b'fixture')
        region = self.directory / 'regions.json'
        region.write_text(json.dumps({'map': 'map.png', 'regions': [{'color': '#ff0000', 'prompt': 'person'}],
            'control': {'adapter': 'control.safetensors', 'color_map': 'map.png', 'depth': 'depth.png', 'depth_end': .5}}))
        args = self.args('--moss-regions', str(region), '--moss-rapid-adapter', str(rapid), '--seed', '5', '--prompt', 'scene')
        self.assertEqual(args.generic_controlnet.depth, str(self.directory / 'depth.png'))
        self.assertEqual(args.generic_controlnet.color, str(self.directory / 'map.png'))
        self.assertEqual(args.generic_controlnet.depth_end, .5)
        self.assertEqual(args.inputs['num_inference_steps'], 4)
        self.assertEqual(generic.configuration(args)['moss_regions']['regions'][0]['prompt'], 'person')

    def test_control_input_mutation_is_detected_before_output_publication(self):
        path = self.directory / 'control.png'
        path.write_bytes(b'initial bytes')
        _, identities = generic.prepare_inputs({'control_image': {'image_path': str(path)}}, self.fake_images)
        generic.verify_input_files(identities)
        path.write_bytes(b'changed bytes')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.verify_input_files(identities)

    def test_native_canny_uses_owned_rgb_buffer_and_native_defaults(self):
        import native_control_preprocessing as preprocessing
        class Image:
            mode = 'RGB'
            size = (2, 1)
            def tobytes(self): return bytes([0, 1, 2, 3, 4, 5])
        class Images:
            @staticmethod
            def frombytes(mode, size, data): return (mode, size, data)
        class Canny:
            def __call__(self, image, high, low, weak, strong, inverse):
                self.values = (image.width, image.height, image.channel, high, low, weak, strong, inverse)
                image.data[0] = 255
                return True
        canny = Canny()
        lib = SimpleNamespace(preprocess_canny=canny)
        result = preprocessing.apply_canny(Image(), Images, lib)
        self.assertEqual(result, ('RGB', (2, 1), bytes([255, 1, 2, 3, 4, 5])))
        self.assertEqual(canny.values, (2, 1, 3, .08, .08, .8, 1., False))

    def test_native_canny_discovers_its_installed_prefix_before_home(self):
        import native_control_preprocessing as preprocessing
        prefix = self.directory / 'installed'
        module = prefix / 'share/iiLocalDiffusion/reference/diffusers/native_control_preprocessing.py'
        module.parent.mkdir(parents=True)
        module.touch()
        name = ('iiLocalDiffusion.dll' if sys.platform == 'win32' else
                'libiiLocalDiffusion.dylib' if sys.platform == 'darwin' else 'libiiLocalDiffusion.so')
        own = prefix / ('bin' if sys.platform == 'win32' else 'lib') / name
        own.parent.mkdir(parents=True)
        own.touch()
        home = self.directory / 'home'
        other = home / '.local/lib' / name
        other.parent.mkdir(parents=True)
        other.touch()
        with patch.object(preprocessing, '__file__', str(module)), \
             patch.object(preprocessing.Path, 'home', return_value=home), \
             patch.dict('os.environ', {}, clear=True):
            self.assertEqual(preprocessing.library_path(), own.resolve())
            self.assertEqual(preprocessing.library_path(str(other)), other.resolve())
            with self.assertRaisesRegex(ValueError, 'Native Canny'):
                preprocessing.library_path(str(self.directory / 'missing'))

    def test_native_canny_windows_loader_temporarily_exposes_dependency_directory(self):
        import native_control_preprocessing as preprocessing
        from contextlib import contextmanager
        library = self.directory / 'bin' / 'iiLocalDiffusion.dll'
        library.parent.mkdir()
        library.touch()
        events = []
        @contextmanager
        def directory(path):
            events.append(('open', path))
            try:
                yield
            finally:
                events.append(('close', path))
        with patch.object(preprocessing.sys, 'platform', 'win32'), \
             patch.object(preprocessing.os, 'add_dll_directory', create=True, side_effect=directory), \
             patch.object(preprocessing.ctypes, 'CDLL') as load, \
             patch.dict('os.environ', {'PATH': ''}, clear=True):
            preprocessing.load_library(library)
        self.assertEqual(events, [('open', str(library.parent)), ('close', str(library.parent))])
        load.assert_called_once_with(str(library))

    def test_native_canny_other_packaged_library_layouts(self):
        import native_control_preprocessing as preprocessing
        for platform, folder, filename in (('linux', 'lib64', 'libiiLocalDiffusion.so'),
                                            ('win32', 'bin', 'libiiLocalDiffusion.dll')):
            prefix = self.directory / platform
            module = prefix / 'share/iiLocalDiffusion/reference/diffusers/native_control_preprocessing.py'
            module.parent.mkdir(parents=True)
            module.touch()
            library = prefix / folder / filename
            library.parent.mkdir(parents=True)
            library.touch()
            if platform == 'win32':
                other = self.directory / '.local/bin/iiLocalDiffusion.dll'
                other.parent.mkdir(parents=True)
                other.touch()
            with self.subTest(platform=platform), patch.object(preprocessing, '__file__', str(module)), \
                 patch.object(preprocessing.sys, 'platform', platform), \
                 patch.object(preprocessing.Path, 'home', return_value=self.directory), \
                 patch.dict('os.environ', {}, clear=True):
                self.assertEqual(preprocessing.library_path(), library.resolve())

    @unittest.skipIf(sys.platform == 'win32', 'Creating symlinks may require Windows privileges')
    def test_installed_canny_path_loads_the_built_library(self):
        import native_control_preprocessing as preprocessing
        filename = 'libiiLocalDiffusion.dylib' if sys.platform == 'darwin' else 'libiiLocalDiffusion.so'
        built = ROOT / 'build' / filename
        if not built.is_file():
            self.skipTest('The native shared library is not built')
        prefix = self.directory / 'prefix'
        module = prefix / 'share/iiLocalDiffusion/reference/diffusers/native_control_preprocessing.py'
        module.parent.mkdir(parents=True)
        module.touch()
        library = prefix / 'lib' / filename
        library.parent.mkdir(parents=True)
        library.symlink_to(built.resolve())
        class Image:
            mode = 'RGB'
            size = (8, 8)
            def tobytes(self): return bytes(8 * 8 * 3)
        class Images:
            @staticmethod
            def frombytes(mode, size, data): return data
        with patch.object(preprocessing, '__file__', str(module)), patch.dict('os.environ', {}, clear=True):
            loaded = preprocessing.load_library(preprocessing.library_path())
            self.assertEqual(preprocessing.apply_canny(Image(), Images, loaded), bytes(8 * 8 * 3))

    def test_control_preprocessor_requires_controlnet_and_known_mode(self):
        with self.assertRaisesRegex(ValueError, 'require --controlnet'):
            self.args('--control-preprocess', 'canny')

    def test_native_canny_shared_library_processes_black_image(self):
        import native_control_preprocessing as preprocessing
        try:
            path = preprocessing.library_path()
        except ValueError:
            self.skipTest('iiLocalDiffusion shared library is not built or installed')
        class Image:
            mode = 'RGB'
            size = (8, 8)
            def tobytes(self): return bytes(8 * 8 * 3)
        class Images:
            @staticmethod
            def frombytes(mode, size, data): return data
        result = preprocessing.apply_canny(Image(), Images, preprocessing.load_library(path))
        self.assertEqual(result, bytes(8 * 8 * 3))

    def test_native_canny_shared_library_produces_sparse_edges_from_color_image(self):
        import native_control_preprocessing as preprocessing
        try:
            path = preprocessing.library_path()
        except ValueError:
            self.skipTest('iiLocalDiffusion shared library is not built or installed')
        width, height = 64, 48
        original = bytearray(width * height * 3)
        for y in range(12, 36):
            for x in range(16, 48):
                offset = (y * width + x) * 3
                original[offset:offset + 3] = b'\xff\xff\xff'
        class Image:
            mode = 'RGB'
            size = (width, height)
            def tobytes(self): return bytes(original)
        class Images:
            @staticmethod
            def frombytes(mode, size, data): return data
        before = bytes(original)
        lib = preprocessing.load_library(path)
        result = preprocessing.apply_canny(Image(), Images, lib)
        self.assertEqual(bytes(original), before)
        self.assertNotEqual(result, before)
        self.assertGreater(result.count(255), 0)
        self.assertLess(result.count(255), 32 * 24 * 3)
        self.assertTrue(all(result[i] == result[i + 1] == result[i + 2] and result[i] in (0, 255)
                            for i in range(0, len(result), 3)))
        self.assertEqual(preprocessing.apply_canny(Image(), Images, lib), result)

    def test_inputs_json_file(self):
        path = self.directory / "inputs.json"
        path.write_text('{"prompt_2":"secondary","strength":0.2}')
        self.assertEqual(self.args("--pipeline-inputs", "@" + str(path)).inputs,
                         {"prompt_2": "secondary", "strength": 0.2})

    def test_rejects_non_object_duplicates_nonfinite_and_untyped_latent_requests(self):
        for value in ('[]', '{"x":NaN}', '{"nested":{"x":1e999}}', '{"return_dict":null}', '{"x":1,"x":2}', '{"return_dict":false}',
                      '{"output_type":"latent"}', '{"generator":1}', '{"_private":true}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.args("--pipeline-inputs", value)

    def test_explicit_latent_output_contract_and_architecture_are_preserved(self):
        spec = {"images": {"semantic": "latents", "layout": "BSC",
                            "representation_space": "test-packed-vae-v1"}}
        args = self.args("--pipeline-inputs", '{"output_type":"latent"}',
                         "--tensor-outputs", json.dumps(spec),
                         "--generation-architecture", "rectified-flow")
        self.assertEqual(args.tensor_outputs, spec)
        self.assertEqual(generic.configuration(args)["generation_architecture"], "rectified-flow")
        self.assertEqual(generic.configuration(self.args())["generation_architecture"], "unspecified")

    def test_untyped_latent_media_cannot_be_misrepresented_as_pixels(self):
        args = self.args("--pipeline-inputs", '{"output_type":"latent"}',
                         "--tensor-outputs", '{"latents":{"semantic":"latents","layout":"BCHW","representation_space":"vae-v1"}}')
        with self.assertRaisesRegex(ValueError, "Latent output.*images"):
            generic.save_outputs({"images": [FakeImage()]}, None, args, self.fake_images, None)

    def test_single_text_and_text_batch_are_exported_with_media(self):
        outputs = generic.save_outputs({"images": [FakeImage()], "text": "한 문장", "texts": ["a", "b"]},
                                       None, self.args(), self.fake_images, None)
        self.assertEqual([item["kind"] for item in outputs].count("text"), 3)
        text_outputs = [item for item in outputs if item["kind"] == "text"]
        self.assertEqual([Path(item["path"]).read_text(encoding="utf-8") for item in text_outputs], ["한 문장", "a", "b"])

    def test_remote_models_are_rejected_even_with_an_immutable_revision(self):
        for revision in (None, 'main', 'abcdef', 'A' * 40, 'a' * 40):
            tokens = ['--model', 'org/repository'] + (['--revision', revision] if revision else [])
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, 'local'):
                generic.resolve_arguments(generic.build_parser().parse_args(tokens))

    def test_missing_local_files_are_not_reinterpreted_as_hub_repositories(self):
        for model in ("/missing/model", "./missing", "weights/missing.safetensors", "weights/model.gguf"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                generic.resolve_arguments(generic.build_parser().parse_args([
                    "--model", model, "--revision", "a" * 40]))

    def test_local_revision_and_single_file_config_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "remote"):
            self.args("--revision", "a" * 40)
        with self.assertRaisesRegex(ValueError, "only valid"):
            self.args("--model-config", str(self.model))

    def test_single_file_requires_supported_container_and_complete_local_config(self):
        unsupported = self.directory / 'model.gguf'
        unsupported.write_bytes(b'GGUF' + bytes(20))
        with self.assertRaisesRegex(ValueError, 'model-config'):
            self.args('--model', str(unsupported))
        legacy = self.directory / 'model.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        with self.assertRaisesRegex(ValueError, 'model-config'):
            self.args('--model', str(legacy))
        for name in ("model.safetensors", "model.safetensor", "model.SAFETENSORS"):
            with self.subTest(name=name):
                path = self.directory / name
                path.write_bytes(b"weights")
                with self.assertRaisesRegex(ValueError, "model-config"):
                    self.args("--model", str(path))
                args = self.args("--model", str(path), "--model-config", str(self.model))
                self.assertEqual(args.source_kind, "single-file")

    def test_invalid_seeds_sizes_guidance_and_cpu_offload_fail_preflight(self):
        for tokens in (("--seed", "-1"), ("--seed", str(2**63)), ("--width", "0"),
                       ("--height", "-2"), ("--steps", "0"), ("--guidance-scale", "nan"),
                       ("--audio-sample-rate", "0"), ("--device", "cpu", "--offload", "model")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.args(*tokens)

    def test_print_config_requires_no_third_party_python_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/diffusers/generate_any.py"),
                                 "--model", str(self.model), "--print-config"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["pipeline_inputs"], {})

    def test_model_index_custom_code_and_paths_are_rejected(self):
        for value in ({"_class_name": ["evil", "Evil"]}, {"_class_name": "mod.Evil"},
                      {"_class_name": "Good", "unet": ["os.path", "Any"]},
                      {"_class_name": "Good", "../unet": ["diffusers", "Any"]},
                      {"_class_name": "Good", "unet": ["diffusers"]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generic.validate_model_index(value)

    def test_arbitrary_installed_modules_are_not_allowed_as_components(self):
        index = generic.validate_model_index({"_class_name": "FakePipeline", "unet": ["os", "system"]})
        with self.assertRaisesRegex(ValueError, "non-built-in"):
            generic.validate_components(self.fake_diffusers(), index)

    def test_krea2_cli_schedule_and_variant_are_resolved(self):
        (self.model / 'text_encoder').mkdir()
        (self.model / 'text_encoder/model.safetensors').write_bytes(b'fixture')
        (self.model / 'model_index.json').write_text(json.dumps({'_class_name': 'Krea2Pipeline', 'is_distilled': True,
                                                                 'text_encoder': ['transformers', 'Qwen3VLModel']}))
        args = self.args('--sigmas', '[1,0.5,0.1]', '--krea2-mu', '0.8')
        self.assertEqual(args.inputs['num_inference_steps'], 3)
        self.assertEqual(args.inputs['sigmas'], [1, .5, .1])
        self.assertEqual(args.krea2['variant'], 'turbo')
        self.assertEqual(args.krea2['resolved_mu'], .8)
        with self.assertRaises(ValueError): self.args('--krea2-variant', 'raw')

    def test_krea2_layer_selection_is_configuration_not_a_component(self):
        index = {"_class_name": "Krea2Pipeline", "text_encoder_select_layers": [0, 2, 5]}
        self.assertEqual(generic.validate_model_index(index), index)
        generic.validate_components(self.fake_diffusers(), index)
        for value in ([True], [-1], [1.5], ["os", "system"], [], list(range(257))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generic.validate_model_index(dict(index, text_encoder_select_layers=value))

    def test_selected_text_encoder_is_passed_before_pipeline_loading(self):
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, text_encoder=None: None
        args = self.args()
        args.text_encoder_selection = object()
        sentinel = object()
        with patch('krea2_text_encoder.load', return_value=sentinel) as load, \
                patch('krea2_text_encoder.verify_selection') as verify:
            generic.load_pipeline(args, runtime, 'float32')
        load.assert_called_once_with(args.text_encoder_selection, 'float32')
        self.assertIs(runtime.FakePipeline.from_pretrained.call_args.kwargs['text_encoder'], sentinel)
        verify.assert_called_once_with(args.text_encoder_selection)

    def test_selected_tokenizer_is_passed_before_pipeline_loading(self):
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, tokenizer=None: None
        args = self.args()
        args.tokenizer_selection = object()
        sentinel = object()
        with patch('krea2_tokenizer.load', return_value=sentinel) as load, \
                patch('krea2_tokenizer.verify_selection') as verify:
            generic.load_pipeline(args, runtime, 'float32')
        load.assert_called_once_with(args.tokenizer_selection)
        self.assertIs(runtime.FakePipeline.from_pretrained.call_args.kwargs['tokenizer'], sentinel)
        verify.assert_called_once_with(args.tokenizer_selection)

    def test_moss_fallback_vae_is_loaded_only_by_the_moss_loader(self):
        class MOSSImagePipeline:
            def __init__(self, vae=None): pass
            def __call__(self): pass
        args = self.args()
        args.vae_selection = object()
        index = {'_class_name': 'MOSSImagePipeline'}
        pipeline = MOSSImagePipeline()
        with patch.object(generic, 'load_model_index', return_value=(index, self.model)), \
             patch.object(generic, 'validate_components'), \
             patch.object(generic, 'builtin_pipeline', return_value=MOSSImagePipeline), \
             patch.object(generic, 'validate_base_model', return_value={}), \
             patch.object(generic, 'verify_vae_selection'), \
             patch.object(generic.moss_runtime, 'load', return_value=pipeline) as load_moss, \
             patch.object(generic, 'load_selected_vae') as load_generic:
            loaded, _ = generic.load_pipeline(args, self.fake_diffusers(), 'float32')
        self.assertIs(loaded, pipeline)
        load_generic.assert_not_called()
        load_moss.assert_called_once_with(self.model, 'float32', args.vae_selection, unittest.mock.ANY,
                                          progress=generic.report_moss_loading_progress)

    def test_moss_file_vae_receives_the_selected_conversion_cache(self):
        class MOSSImagePipeline:
            def __init__(self, vae=None): pass
            def __call__(self): pass
        args = self.args()
        args.vae_selection = SimpleNamespace(source_kind='file', container_kind='pytorch')
        index = {'_class_name': 'MOSSImagePipeline'}
        pipeline = MOSSImagePipeline()
        with patch.object(generic, 'load_model_index', return_value=(index, self.model)), \
             patch.object(generic, 'validate_components'), \
             patch.object(generic, 'builtin_pipeline', return_value=MOSSImagePipeline), \
             patch.object(generic, 'validate_base_model', return_value={}), \
             patch.object(generic, 'verify_vae_selection'), \
             patch.object(generic.moss_runtime, 'load', return_value=pipeline) as load_moss:
            generic.load_pipeline(args, self.fake_diffusers(), 'float32')
        load_moss.assert_called_once_with(self.model, 'float32', args.vae_selection, unittest.mock.ANY,
            progress=generic.report_moss_loading_progress, vae_cache_dir=args.cache_dir)

    def test_custom_local_component_code_is_not_executed(self):
        directory = self.model / "unet"
        directory.mkdir()
        (directory / "diffusers.py").write_text("raise RuntimeError('must not execute')")
        with self.assertRaisesRegex(ValueError, "Custom local component"):
            generic.validate_components(self.fake_diffusers(), self.index, self.model)

    def test_moss_index_accepts_only_the_vetted_component_contract(self):
        index = {'_class_name': 'MOSSImagePipeline',
                 'transformer': ['pipeline_moss_image', 'MOSSImageTransformer'],
                 'scheduler': ['pipeline_moss_image', 'MOSSImageScheduler'],
                 'text_encoder': ['transformers', 'Qwen3_5Model'],
                 'tokenizer': ['transformers', 'Qwen2Tokenizer'],
                 'vae': ['diffusers', 'AutoencoderKLFlux2']}
        for name in ('transformer', 'scheduler'):
            (self.model / name).mkdir()
            (self.model / name / 'pipeline_moss_image.py').write_text("raise RuntimeError('never import')")
        generic.validate_components(self.fake_diffusers(), generic.validate_model_index(index), self.model)
        for key, value in (('transformer', ['pipeline_moss_image', 'Evil']),
                           ('scheduler', ['other', 'MOSSImageScheduler']),
                           ('text_encoder', ['transformers', 'Evil'])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                generic.validate_components(self.fake_diffusers(), dict(index, **{key: value}), self.model)

    def test_moss_vendor_source_and_license_are_pinned(self):
        vendor = ROOT / 'reference/diffusers/vendor_moss'
        self.assertEqual(hashlib.sha256((vendor / 'pipeline.py').read_bytes()).hexdigest(),
                         'd1db06abe3682a86f4fa3e36b454d4f17cc8d9d25eb4f7c5a776ba277af84e9c')
        self.assertIn('Apache License', (vendor / 'LICENSE.txt').read_text())

    def test_moss_loader_rejects_partial_component_weights(self):
        import moss_runtime
        component = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [], eval=lambda: component)
        loader = SimpleNamespace(from_pretrained=Mock(return_value=(component, {'missing_keys': []})))
        self.assertIs(moss_runtime.strict_load(loader, self.model, 'transformer', local_files_only=True), component)
        loader.from_pretrained.return_value = (component, {'missing_keys': ['weight']})
        with self.assertRaisesRegex(RuntimeError, 'do not exactly match'):
            moss_runtime.strict_load(loader, self.model, 'transformer', local_files_only=True)

    def test_moss_generic_loader_dispatches_to_pinned_implementation(self):
        import moss_runtime
        index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        self.provide_stub_moss_encoder()
        (self.model / 'vae').mkdir()
        (self.model / 'vae/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        for name in ('transformer', 'scheduler'):
            (self.model / name).mkdir()
            (self.model / name / 'pipeline_moss_image.py').write_text("raise RuntimeError('must not execute')")
        diffusers = self.fake_diffusers()
        class MOSSImagePipeline(diffusers.DiffusionPipeline):
            def __call__(self, prompt=None): pass
        instance = MOSSImagePipeline()
        args = self.args('--prompt', 'test')
        with patch.object(moss_runtime, 'pipeline_class', return_value=MOSSImagePipeline), \
                patch.object(moss_runtime, 'load', return_value=instance) as load:
            pipeline, metadata = generic.load_pipeline(args, diffusers, 'float32')
        self.assertIs(pipeline, instance)
        self.assertEqual(metadata['actual_class'], 'MOSSImagePipeline')
        self.assertEqual(load.call_args.args[0], self.model)

    def test_moss_directory_variant_is_selected_and_forwarded(self):
        import moss_runtime
        index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        self.provide_stub_moss_encoder()
        for name in ('transformer', 'vae', 'tokenizer'):
            (self.model / name).mkdir()
        (self.model / 'transformer/diffusion_pytorch_model.fp16.safetensors').write_bytes(b'variant')
        (self.model / 'vae/diffusion_pytorch_model.safetensors').write_bytes(b'default')
        (self.model / 'tokenizer/vocab.json').write_text('{}')
        args = self.args('--no-default-modifiers')
        self.assertEqual((args.variant, args.variant_selection), ('fp16', 'automatic'))
        explicit = self.args('--variant', 'fp16', '--no-default-modifiers')
        self.assertEqual(explicit.variant_selection, 'explicit')
        runtime = self.fake_diffusers()
        class MOSSImagePipeline(runtime.DiffusionPipeline):
            def __call__(self): pass
        instance = MOSSImagePipeline()
        with patch.object(moss_runtime, 'pipeline_class', return_value=MOSSImagePipeline), \
             patch.object(moss_runtime, 'load', return_value=instance) as load:
            generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(load.call_args.kwargs['variant'], 'fp16')

    def test_moss_variant_indexes_are_not_replaced_by_fallback_components(self):
        import moss_runtime
        import moss_text_encoder
        import vae_defaults
        index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        for name in ('transformer', 'text_encoder', 'vae', 'tokenizer'):
            (self.model / name).mkdir()
        (self.model / 'transformer/diffusion_pytorch_model.safetensors').write_bytes(b'default')
        (self.model / 'text_encoder/model.safetensors.index.fp16.json').write_text('{}')
        (self.model / 'vae/diffusion_pytorch_model.safetensors.index.fp16.json').write_text('{}')
        (self.model / 'tokenizer/vocab.json').write_text('{}')
        with patch.object(moss_text_encoder, 'resource_catalogs', side_effect=AssertionError('encoder fallback')), \
             patch.object(vae_defaults, 'select_fallback_vae', side_effect=AssertionError('VAE fallback')):
            args = self.args('--no-default-modifiers')
        self.assertEqual((args.variant, args.moss_text_encoder_status, args.vae_status), ('fp16', 'model', 'model'))

    def test_krea_and_flux_detect_variant_indexes_before_fallback(self):
        import flux_text_encoder
        import krea2_text_encoder
        encoder = self.model / 'text_encoder';encoder.mkdir()
        (encoder / 'model.safetensors.index.fp16.json').write_text('{}')
        args = SimpleNamespace(pipeline_class=None, source_kind='directory', inputs={}, generation_resources=None)
        krea = {'_class_name': 'Krea2Pipeline', 'text_encoder': ['transformers', 'Qwen3VLModel']}
        flux = {'_class_name': 'FluxPipeline', 'text_encoder': ['transformers', 'CLIPTextModel']}
        with patch.object(krea2_text_encoder, 'read_defaults', side_effect=AssertionError('Krea fallback')), \
             patch.object(flux_text_encoder, 'read_defaults', side_effect=AssertionError('FLUX fallback')):
            self.assertEqual(krea2_text_encoder.resolve(args, krea, self.model), (None, 'model'))
            self.assertEqual(flux_text_encoder.resolve_fallback(args, flux, self.model, 'text_encoder'), (None, 'model'))

    def test_overridden_vae_does_not_constrain_automatic_variant(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        for name in ('unet', 'vae'):
            (self.model / name).mkdir()
        (self.model / 'unet/diffusion_pytorch_model.fp16.safetensors').write_bytes(b'variant')
        (self.model / 'vae/diffusion_pytorch_model.bf16.bin').write_bytes(b'unused')
        override = object()
        with patch.object(generic, 'resolve_vae_selection', return_value=(override, 'explicit')):
            args = self.args('--no-default-modifiers')
        self.assertIs(args.vae_selection, override)
        self.assertEqual((args.variant, args.variant_selection), ('fp16', 'automatic'))

    def test_overridden_encoder_does_not_convert_unused_pickle_weights(self):
        import checkpoint_conversion
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        encoder = self.model / 'text_encoder';encoder.mkdir()
        (encoder / 'pytorch_model.bin').write_bytes(b'unused invalid pickle')
        args = self.args('--no-default-modifiers')
        args.text_encoder_selection = object()
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, text_encoder=None: None
        with patch.object(generic.krea2_text_encoder, 'load', return_value=object()), \
             patch.object(generic.krea2_text_encoder, 'verify_selection'), \
             patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            generic.load_pipeline(args, runtime, 'float32')
        convert.assert_not_called()
        self.assertEqual(runtime.FakePipeline.from_pretrained.call_args.args[0], str(self.model))

    def test_moss_runtime_uses_component_specific_variant_and_rejects_missing_variant(self):
        import moss_runtime
        transformer = self.model / 'transformer';transformer.mkdir()
        encoder = self.model / 'text_encoder';encoder.mkdir()
        vae = self.model / 'vae';vae.mkdir()
        (transformer / 'diffusion_pytorch_model.fp16.safetensors').write_bytes(b'variant')
        (encoder / 'model.safetensors').write_bytes(b'default')
        (vae / 'diffusion_pytorch_model.fp16.safetensors').write_bytes(b'variant')
        fake_transformer = SimpleNamespace(config=SimpleNamespace(latent_channels=8))
        fake_encoder = type('Qwen3_5Model', (), {})()
        fake_vae = SimpleNamespace(config=SimpleNamespace(patch_size=(2, 2), latent_channels=2),
                                   bn=SimpleNamespace(num_features=8))
        components = {'transformer': fake_transformer, 'text encoder': fake_encoder, 'vae': fake_vae}
        runtime = SimpleNamespace(AutoencoderKLFlux2=object())
        modules = {'torch': SimpleNamespace(float32='float32'),
                   'transformers': SimpleNamespace(AutoModel=object(), AutoTokenizer=SimpleNamespace(from_pretrained=Mock())),
                   'vendor_moss.pipeline': SimpleNamespace(MOSSImageTransformer=object(),
                       MOSSImagePipeline=lambda **kwargs: SimpleNamespace(**kwargs))}
        calls = {}
        def load(_class, _path, role, **kwargs):
            calls[role] = kwargs
            return components[role]
        with patch.dict(sys.modules, modules), patch.object(moss_runtime, 'load_scheduler', return_value=object()), \
             patch.object(moss_runtime, 'strict_load', side_effect=load) as strict:
            moss_runtime.load(self.model, 'float32', None, runtime, variant='fp16')
            self.assertEqual(calls['transformer']['variant'], 'fp16')
            self.assertNotIn('variant', calls['text encoder'])
            self.assertEqual(calls['vae']['variant'], 'fp16')
            strict.reset_mock()
            with self.assertRaisesRegex(ValueError, 'variant.*bf16'):
                moss_runtime.load(self.model, 'float32', None, runtime, variant='bf16')
            strict.assert_not_called()

    def test_only_public_installed_diffusers_pipeline_classes_can_be_selected(self):
        diffusers = self.fake_diffusers()
        self.assertEqual(generic.builtin_pipeline(diffusers, "FakePipeline"), diffusers.FakePipeline)
        diffusers.Evil = FakePipeline
        diffusers.NotPipeline = str
        for name in ("Evil", "NotPipeline", "Missing", "DiffusionPipeline"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                generic.builtin_pipeline(diffusers, name)

    def test_civitai_labels_reject_cross_architecture_models(self):
        diffusers = self.fake_diffusers()
        cases = (("SD 1.5", "StableDiffusionPipeline"), ("Illustrious", "StableDiffusionXLPipeline"),
                 ("SD 3.5", "StableDiffusion3Pipeline"), ("Flux.1 Krea", "FluxPipeline"),
                 ("Flux.2 D", "Flux2Pipeline"), ("Qwen", "QwenImagePipeline"))
        classes = {}
        for label, name in cases:
            classes[label] = type(name, (diffusers.DiffusionPipeline,),
                                  {"__module__": "diffusers.pipelines.test_builtin"})
            setattr(diffusers, name, classes[label])
        for label, _ in cases:
            for other, _ in cases:
                with self.subTest(label=label, actual=other):
                    if label == other:
                        result = generic.validate_base_model(label, classes[other], diffusers)
                        self.assertEqual(result["status"], "pipeline-architecture-verified")
                    else:
                        with self.assertRaisesRegex(ValueError, "requires architecture"):
                            generic.validate_base_model(label, classes[other], diffusers)

    def test_task_siblings_remain_compatible_across_diffusers_namespaces(self):
        diffusers = self.fake_diffusers()
        for label, expected, sibling in (
                ("NoobAI", "StableDiffusionXLPipeline", "StableDiffusionXLInpaintPipeline"),
                ("SD 3", "StableDiffusion3Pipeline", "StableDiffusion3ControlNetPipeline"),
                ("Flux.1 D", "FluxPipeline", "FluxKontextPipeline"),
                ("Qwen", "QwenImagePipeline", "QwenImageEditPipeline")):
            setattr(diffusers, expected, type(expected, (diffusers.DiffusionPipeline,),
                    {"__module__": "diffusers.pipelines.base_namespace"}))
            actual = type(sibling, (diffusers.DiffusionPipeline,),
                          {"__module__": "diffusers.pipelines.task_namespace"})
            with self.subTest(label=label):
                result = generic.validate_base_model(label, actual, diffusers)
                self.assertEqual(result["status"], "pipeline-architecture-verified")

    def test_kolors_task_siblings_share_the_builtin_kolors_namespace(self):
        diffusers = self.fake_diffusers()
        diffusers.KolorsPipeline = type("KolorsPipeline", (diffusers.DiffusionPipeline,),
                                        {"__module__": "diffusers.pipelines.kolors.pipeline_kolors"})
        image_pipeline = type("KolorsImg2ImgPipeline", (diffusers.DiffusionPipeline,),
                              {"__module__": "diffusers.pipelines.kolors.pipeline_kolors_img2img"})
        report = generic.validate_base_model("Kolors", image_pipeline, diffusers)
        self.assertEqual(report["status"], "pipeline-architecture-verified")
        self.assertEqual(report["actual_architecture"], "diffusers:kolors")

    def test_single_file_cannot_fall_back_to_directory_loader_when_class_has_no_loader(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"fixture")
        args = self.args("--model", str(path), "--model-config", str(self.model))
        diffusers = self.fake_diffusers()
        delattr(diffusers.FakePipeline, "from_single_file")
        with self.assertRaisesRegex(ValueError, "no built-in single-file loader"):
            generic.load_pipeline(args, diffusers, "float32")
        diffusers.FakePipeline.from_pretrained.assert_not_called()

    def test_workflow_family_without_known_class_is_explicitly_declared_only(self):
        result = generic.validate_base_model("Anima", FakePipeline, self.fake_diffusers())
        self.assertEqual(result["status"], "declared-only")
        with self.assertRaisesRegex(ValueError, "Unknown Civitai"):
            generic.validate_base_model("invented unknown label", FakePipeline, self.fake_diffusers())

    def test_unknown_kwargs_are_rejected_even_when_pipeline_accepts_arbitrary_kwargs(self):
        with self.assertRaisesRegex(ValueError, "typo"):
            generic.validate_call_arguments(FakePipeline().__call__, {"typo": 1}, seed=None)
        generic.validate_call_arguments(FakePipeline().__call__, {"prompt": "hi"}, seed=0)

    def test_required_inputs_and_unsupported_seed_are_explicit(self):
        def requires_image(image, *, strength=0.8):
            pass
        with self.assertRaisesRegex(ValueError, "requires input.*image"):
            generic.validate_call_arguments(requires_image, {}, seed=None)
        with self.assertRaisesRegex(ValueError, "generator"):
            generic.validate_call_arguments(requires_image, {"image": "file"}, seed=1)

    def test_directory_load_is_local_safe_and_records_actual_pipeline(self):
        diffusers = self.fake_diffusers()
        pipeline, report = generic.load_pipeline(self.args(), diffusers, "float32")
        self.assertIsInstance(pipeline, diffusers.FakePipeline)
        kwargs = diffusers.FakePipeline.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["local_files_only"], True)
        self.assertEqual(kwargs["use_safetensors"], True)
        self.assertEqual(kwargs["trust_remote_code"], False)
        self.assertEqual(report["actual_class"], "FakePipeline")
        self.assertTrue(report["identity"]["files"][0]["sha256"])

    def test_directory_converts_declared_pytorch_components_in_temporary_overlay(self):
        import checkpoint_conversion
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        unet, encoder = self.model / 'unet', self.model / 'text_encoder'
        unet.mkdir(); encoder.mkdir()
        single, index, shard = (unet / 'diffusion_pytorch_model.bin',
                                encoder / 'pytorch_model.bin.index.json', encoder / 'part.bin')
        single.write_bytes(b'first source'); shard.write_bytes(b'second source')
        index.write_text(json.dumps({'weight_map': {'weight': shard.name}}))
        before = {p: p.read_bytes() for p in (single, index, shard)}
        converted = {}
        def convert(source, _cache):
            source = Path(source)
            result = self.directory / (source.parent.name + '.safetensors')
            result.write_bytes(b'converted ' + source.parent.name.encode())
            conversion = {'converted': True, 'converted_path': str(result),
                          'original': checkpoint_conversion._identity(source),
                          'output': checkpoint_conversion._identity(result)}
            if source == index:
                conversion['original_shards'] = [checkpoint_conversion._identity(shard)]
            converted[str(source)] = conversion
            return conversion
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        observed = []
        def load(folder, **kwargs):
            overlay = Path(folder); observed.append(overlay)
            self.assertNotEqual(overlay, self.model)
            self.assertEqual((overlay / 'model_index.json').read_bytes(), (self.model / 'model_index.json').read_bytes())
            self.assertEqual((overlay / 'unet' / 'diffusion_pytorch_model.safetensors').read_bytes(), b'converted unet')
            self.assertEqual((overlay / 'text_encoder' / 'model.safetensors').read_bytes(), b'converted text_encoder')
            self.assertTrue(kwargs['use_safetensors'])
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        with patch.object(checkpoint_conversion, 'materialize_safetensors', side_effect=convert):
            _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(set(converted), {str(single), str(index)})
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertFalse(observed[0].exists())
        self.assertEqual(generic.configuration(args)['model'], str(self.model))
        self.assertEqual(len(report['identity']['directory_conversions']), 2)
        shard.write_bytes(b'changed source')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.verify_identity(report['identity'])

    def test_directory_prefers_existing_safetensors_without_conversion(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.safetensors').write_bytes(b'existing safe weights')
        (unet / 'diffusion_pytorch_model.bin').write_bytes(b'legacy weights')
        runtime = self.fake_diffusers()
        with patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            generic.load_pipeline(self.args('--no-default-modifiers'), runtime, 'float32')
        convert.assert_not_called()
        self.assertEqual(runtime.FakePipeline.from_pretrained.call_args.args[0], str(self.model))

    def test_directory_converts_pytorch_disguised_as_safetensors(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.safetensors'
        original = b'PK\x03\x04' + bytes(20);source.write_bytes(original)
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();seen = []
        def load(folder, **_kwargs):
            folder = Path(folder);seen.append(folder)
            self.assertEqual((folder / 'unet/diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as convert:
            generic.load_pipeline(args, runtime, 'float32')
        convert.assert_called_once()
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(seen[0].exists())

    def test_directory_converts_standard_component_with_pth_extension(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.pth';source.write_bytes(b'legacy weights')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();seen = []
        def load(folder, **_kwargs):
            folder = Path(folder);seen.append(folder)
            self.assertEqual((folder / 'unet/diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as convert:
            generic.load_pipeline(args, runtime, 'float32')
        convert.assert_called_once()
        self.assertEqual(source.read_bytes(), b'legacy weights')
        self.assertFalse(seen[0].exists())

    def test_directory_detects_alternate_extension_variant_and_refuses_ambiguous_weights(self):
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.fp16.pt').write_bytes(b'weights')
        args = self.args('--no-default-modifiers')
        self.assertEqual(args.variant, 'fp16')
        (unet / 'diffusion_pytorch_model.fp16.pth').write_bytes(b'other weights')
        with self.assertRaisesRegex(ValueError, '(ambiguous|multiple).*weight'):
            generic.load_pipeline(args, self.fake_diffusers(), 'float32')

    def test_directory_converts_pytorch_index_disguised_as_safetensors_index(self):
        import checkpoint_conversion
        import downloaded_model
        unet = self.model / 'unet';unet.mkdir()
        index = unet / 'diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'weight': 'piece.bin'}}))
        shard = unet / 'piece.bin';shard.write_bytes(b'legacy shard')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(index),
                      'original_shards': [checkpoint_conversion._identity(shard)],
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();seen = []
        def load(folder, **_kwargs):
            folder = Path(folder);seen.append(folder)
            self.assertEqual((folder / 'unet/diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
            self.assertFalse((folder / 'unet/diffusion_pytorch_model.safetensors.index.json').exists())
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(downloaded_model, 'model_container_format', return_value='pytorch-index'), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as convert:
            generic.load_pipeline(args, runtime, 'float32')
        convert.assert_called_once()
        self.assertEqual(shard.read_bytes(), b'legacy shard')
        self.assertFalse(seen[0].exists())

    def test_directory_variant_is_converted_and_forwarded_to_loader(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.fp16.bin'
        source.write_bytes(b'variant weights')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();observed = []
        def load(folder, **kwargs):
            overlay = Path(folder);observed.append(overlay)
            self.assertEqual(kwargs['variant'], 'fp16')
            self.assertEqual((overlay / 'unet' / 'diffusion_pytorch_model.fp16.safetensors').read_bytes(), b'safe weights')
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--variant', 'fp16', '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            generic.load_pipeline(args, runtime, 'float32')
        self.assertFalse(observed[0].exists())
        self.assertEqual(generic.configuration(args)['variant'], 'fp16')

    def test_directory_source_change_during_loading_cleans_overlay(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.bin';source.write_bytes(b'original weights')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'converted weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();observed = []
        def load(folder, **_kwargs):
            observed.append(Path(folder));source.write_bytes(b'modified weights')
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                generic.load_pipeline(args, runtime, 'float32')
        self.assertFalse(observed[0].exists())

    def test_directory_conversion_refuses_output_or_cache_inside_model(self):
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.bin').write_bytes(b'legacy weights')
        for tokens in (('--cache-dir', str(self.model / 'cache')),
                       ('--output-dir', str(self.model / 'output'), '--cache-dir', str(self.directory / 'cache')),
                       ('--output-dir', str(self.directory), '--cache-dir', str(self.directory / 'cache'))):
            with self.subTest(tokens=tokens):
                args = self.args(*tokens, '--no-default-modifiers')
                with self.assertRaisesRegex(ValueError, '(cache|output).*directory'):
                    generic.load_pipeline(args, self.fake_diffusers(), 'float32')

    def test_directory_variant_validation_and_configuration_are_side_effect_free(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.fp16.bin').write_bytes(b'legacy weights')
        with patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            args = self.args('--variant', 'fp16', '--print-config', '--no-default-modifiers')
            self.assertEqual(generic.configuration(args)['variant'], 'fp16')
            convert.assert_not_called()
        for variant in ('../fp16', 'bad.name', ''):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'variant'):
                self.args('--variant', variant)

    def test_directory_selects_only_unambiguous_required_variant(self):
        unet = self.model / 'unet';unet.mkdir()
        variant = unet / 'diffusion_pytorch_model.fp16.bin'
        variant.write_bytes(b'legacy weights')
        args = self.args('--no-default-modifiers')
        self.assertEqual((args.variant, generic.configuration(args)['variant']), ('fp16', 'fp16'))
        self.assertEqual(generic.configuration(args)['variant_selection'], 'automatic')
        (unet / 'diffusion_pytorch_model.safetensors').write_bytes(b'default safe')
        args = self.args('--no-default-modifiers')
        self.assertIsNone(args.variant)
        self.assertEqual(generic.configuration(args)['variant_selection'], 'default')

    def test_directory_does_not_guess_between_variant_choices(self):
        unet = self.model / 'unet';unet.mkdir()
        for name in ('fp16', 'bf16'):
            (unet / f'diffusion_pytorch_model.{name}.bin').write_bytes(b'weights')
        with self.assertRaisesRegex(ValueError, 'variant.*(bf16|fp16)'):
            self.args('--no-default-modifiers')
        args = self.args('--variant', 'fp16', '--no-default-modifiers')
        self.assertEqual(args.variant, 'fp16')
        self.assertEqual(generic.configuration(args)['variant_selection'], 'explicit')

    def test_directory_rejects_incompatible_required_component_variants(self):
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        unet, encoder = self.model / 'unet', self.model / 'text_encoder'
        unet.mkdir();encoder.mkdir()
        (unet / 'diffusion_pytorch_model.bin.index.fp16.json').write_text('{}')
        (encoder / 'model.bf16.safetensors').write_bytes(b'weights')
        with self.assertRaisesRegex(ValueError, 'variant.*(unet|text_encoder)'):
            self.args('--no-default-modifiers')

    def test_directory_auto_selects_a_safe_sharded_variant(self):
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.safetensors.index.fp16.json').write_text('{}')
        for number in (1, 2):
            (unet / f'diffusion_pytorch_model.fp16-{number:05d}-of-00002.safetensors').write_bytes(b'shard')
        args = self.args('--no-default-modifiers')
        self.assertEqual(args.variant, 'fp16')

    def test_directory_auto_selects_pytorch_shard_index_without_guessing_shard_names(self):
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.bin.index.fp16.json').write_text('{}')
        for number in (1, 2):
            (unet / f'diffusion_pytorch_model.fp16-{number:05d}-of-00002.bin').write_bytes(b'shard')
        self.assertEqual(self.args('--no-default-modifiers').variant, 'fp16')

    def test_existing_safe_symlinked_component_needs_no_overlay(self):
        other = self.directory / 'unet-target';other.mkdir()
        (other / 'diffusion_pytorch_model.safetensors').write_bytes(b'safe weights')
        try:
            (self.model / 'unet').symlink_to(other, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlinks are unavailable on this platform')
        runtime = self.fake_diffusers()
        generic.load_pipeline(self.args('--no-default-modifiers'), runtime, 'float32')
        self.assertEqual(runtime.FakePipeline.from_pretrained.call_args.args[0], str(self.model))

    def test_symlinked_legacy_component_is_converted_without_changing_its_target(self):
        import checkpoint_conversion
        other = self.directory / 'unet-target';other.mkdir()
        source = other / 'diffusion_pytorch_model.bin';source.write_bytes(b'legacy weights')
        try:
            (self.model / 'unet').symlink_to(other, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlinks are unavailable on this platform')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        logical = self.model / 'unet' / source.name
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(logical),
                      'output': checkpoint_conversion._identity(converted)}
        runtime = self.fake_diffusers();observed = []
        def load(folder, **_kwargs):
            overlay = Path(folder);observed.append(overlay)
            self.assertEqual((overlay / 'unet' / 'diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
            return runtime.FakePipeline()
        runtime.FakePipeline.from_pretrained = Mock(side_effect=load)
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(source.read_bytes(), b'legacy weights')
        self.assertFalse(observed[0].exists())
        (other / 'added-config.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, '(changed|directory)'):
            generic.verify_identity(report['identity'])

    def test_cyclic_directory_symlink_is_rejected_before_conversion(self):
        import checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        (unet / 'diffusion_pytorch_model.bin').write_bytes(b'legacy weights')
        try:
            (self.model / 'cycle').symlink_to(self.model, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlinks are unavailable on this platform')
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        with patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            with self.assertRaisesRegex(ValueError, '(cycle|cyclic|circular)'):
                generic.load_pipeline(args, self.fake_diffusers(), 'float32')
            convert.assert_not_called()

    def test_symlinked_component_cannot_contain_the_cache_or_output(self):
        import checkpoint_conversion
        other = self.directory / 'shared-unet';other.mkdir()
        (other / 'diffusion_pytorch_model.bin').write_bytes(b'legacy weights')
        try:
            (self.model / 'unet').symlink_to(other, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlinks are unavailable on this platform')
        for tokens in (('--cache-dir', str(other / 'cache')),
                       ('--cache-dir', str(self.directory / 'cache'), '--output-dir', str(other / 'output'))):
            with self.subTest(tokens=tokens):
                args = self.args(*tokens, '--no-default-modifiers')
                with patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
                    with self.assertRaisesRegex(ValueError, '(cache|output).*directory'):
                        generic.load_pipeline(args, self.fake_diffusers(), 'float32')
                    convert.assert_not_called()

    def test_changed_directory_invalidates_retained_pipeline(self):
        import inference_session
        args = self.args('--no-default-modifiers')
        cached = (object(), {'identity': {'directory_conversions': []}}, Mock(), None)
        with inference_session.InferenceSession() as session:
            session.value = object()
            with patch.object(generic, 'cached_pipeline', return_value=(cached, True)), \
                 patch.object(generic, 'verify_identity', side_effect=RuntimeError('source changed')):
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    generic.prepared_pipeline(args, self.fake_diffusers(), 'float32', 'cpu')
            self.assertIsNone(session.value)

    def test_changed_directory_at_publication_invalidates_retained_pipeline(self):
        import directory_checkpoint_conversion
        import inference_session
        args = SimpleNamespace(output_dir=self.directory / 'output', overwrite=False,
                               model_directory_sources={'directories': [], 'files': []})
        with inference_session.InferenceSession() as session:
            session.value = object()
            with patch.object(directory_checkpoint_conversion, 'verify_source_snapshot',
                              side_effect=RuntimeError('source changed')):
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    generic.publish_generation(SimpleNamespace(), SimpleNamespace(), args, self.fake_images, None, {})
            self.assertIsNone(session.value)

    def test_directory_overlay_copies_if_symlinks_are_unavailable(self):
        import checkpoint_conversion
        import directory_checkpoint_conversion
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.bin';source.write_bytes(b'legacy weights')
        converted = self.directory / 'converted.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        original = source.stat()
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion), \
             patch.object(directory_checkpoint_conversion.os, 'symlink', side_effect=OSError('unavailable')):
            with directory_checkpoint_conversion.prepare(self.model, self.index, self.directory / 'cache',
                    self.directory / 'output', None) as prepared:
                target = prepared.path / 'unet' / 'diffusion_pytorch_model.safetensors'
                self.assertFalse(target.is_symlink())
                self.assertEqual(target.read_bytes(), b'safe weights')
        after = source.stat()
        self.assertEqual((original.st_ino, original.st_size, original.st_mtime_ns, original.st_ctime_ns),
                         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns))

    def test_flux_single_file_repairs_embedded_t5_before_returning_pipeline(self):
        diffusers = self.fake_diffusers()
        class FluxPipeline(diffusers.DiffusionPipeline):
            def __init__(self):
                self.text_encoder_2 = object()
                self.text_encoder = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
            def __call__(self):
                pass
        instance = FluxPipeline()
        FluxPipeline.from_single_file = Mock(return_value=instance)
        args = self.args()
        checkpoint = self.directory / 'flux.safetensors'
        checkpoint.write_bytes(b'fixture')
        args.source_kind, args.model, args.model_config = 'single-file', str(checkpoint), str(self.model)
        args.flux_text_encoder_status = {'text_encoder': 'model', 'text_encoder_2': 'model'}
        with patch.object(generic, 'load_model_index', return_value=({'_class_name': 'FluxPipeline'}, self.model)), \
                patch.object(generic, 'validate_components'), \
                patch.object(generic, 'builtin_pipeline', return_value=FluxPipeline), \
                patch.object(generic, 'validate_base_model', return_value={}), \
                patch.object(generic.flux_tokenizer, 'bound_clip_length', return_value=None), \
                patch.object(generic.flux_text_encoder, 'repair_embedded_t5', return_value='repaired-shared-alias') as repair:
            pipeline, report = generic.load_pipeline(args, diffusers, 'float32')
            instance.text_encoder = SimpleNamespace(
                named_parameters=lambda: [('missing', SimpleNamespace(is_meta=True))], named_buffers=lambda: [])
            with self.assertRaisesRegex(RuntimeError, 'FLUX.text_encoder.missing'):
                generic.load_pipeline(args, diffusers, 'float32')
        self.assertIs(pipeline, instance)
        repair.assert_called_once_with(instance.text_encoder_2)
        self.assertEqual(report['flux_t5_embedded_loading'], 'repaired-shared-alias')

    def test_flux_single_file_injects_streamed_embedded_t5(self):
        diffusers = self.fake_diffusers()
        class FluxPipeline(diffusers.DiffusionPipeline):
            def __init__(self, text_encoder_2=None):
                self.text_encoder_2 = text_encoder_2
            def __call__(self):
                pass
        instance = FluxPipeline()
        FluxPipeline.from_single_file = Mock(return_value=instance)
        args = self.args()
        checkpoint = self.directory / 'flux.safetensors'
        checkpoint.write_bytes(b'fixture')
        args.source_kind, args.model, args.model_config = 'single-file', str(checkpoint), str(self.model)
        selection = object()
        args.flux_text_encoders = {'text_encoder_2': selection}
        args.flux_text_encoder_status = {'text_encoder_2': 'embedded'}
        args.flux_t5_fp8_storage = 'cpu-stream'
        encoder = SimpleNamespace(_iild_flux_text_encoder_loading={'runtime_weights': 'cpu-stream-fp8'})
        with patch.object(generic, 'load_model_index', return_value=({'_class_name': 'FluxPipeline'}, self.model)), \
                patch.object(generic, 'validate_components'), \
                patch.object(generic, 'builtin_pipeline', return_value=FluxPipeline), \
                patch.object(generic, 'validate_base_model', return_value={}), \
                patch.object(generic.flux_tokenizer, 'bound_clip_length', return_value=None), \
                patch.object(generic.flux_text_encoder, 'load', return_value=encoder) as load, \
                patch.object(generic.flux_text_encoder, 'verify_selection'), \
                patch.object(generic.flux_text_encoder, 'repair_embedded_t5') as repair:
            pipeline, report = generic.load_pipeline(args, diffusers, 'float32')
        self.assertIs(pipeline, instance)
        load.assert_called_once_with(selection, 'float32', storage='cpu-stream')
        repair.assert_not_called()
        self.assertIs(FluxPipeline.from_single_file.call_args.kwargs['text_encoder_2'], encoder)
        self.assertEqual(report['flux_text_encoder_loading']['text_encoder_2']['runtime_weights'], 'cpu-stream-fp8')

    def test_invalid_kwargs_fail_before_weight_loading(self):
        diffusers = self.fake_diffusers()
        with self.assertRaisesRegex(ValueError, "misspelled"):
            generic.load_pipeline(self.args("--pipeline-inputs", '{"misspelled":1}'), diffusers, "dtype")
        diffusers.FakePipeline.from_pretrained.assert_not_called()

    def test_missing_lora_loader_is_rejected_before_base_allocation(self):
        diffusers = self.fake_diffusers()
        adapter = self.directory / "style.safetensors"
        adapter.write_bytes(b"local adapter identity")
        with self.assertRaisesRegex(ValueError, "does not support verified LoRA loading"):
            generic.load_pipeline(self.args("--lora", str(adapter)), diffusers, "float32")
        diffusers.FakePipeline.from_pretrained.assert_not_called()

    def test_explicit_null_components_in_local_index_reach_constructor(self):
        diffusers = self.fake_diffusers()
        diffusers.FakePipeline.__init__ = lambda self, text_encoder_3=None: None
        self.index["text_encoder_3"] = [None, None]
        (self.model / "model_index.json").write_text(json.dumps(self.index))
        generic.load_pipeline(self.args(), diffusers, "float32")
        self.assertIn("text_encoder_3", diffusers.FakePipeline.from_pretrained.call_args.kwargs)
        self.assertIsNone(diffusers.FakePipeline.from_pretrained.call_args.kwargs["text_encoder_3"])

    def test_single_file_passes_explicit_local_config_and_refuses_pickle_extras(self):
        diffusers = self.fake_diffusers()
        path = self.directory / "model.safetensors"
        path.write_bytes(b"safe fixture")
        args = self.args("--model", str(path), "--model-config", str(self.model))
        _, report = generic.load_pipeline(args, diffusers, "dtype")
        self.assertEqual(diffusers.FakePipeline.from_single_file.call_args.kwargs["config"], str(self.model))
        self.assertEqual(report["identity"]["single_file"]["size_bytes"], len(b"safe fixture"))
        (self.model / "pytorch_model.bin").write_bytes(b"unsafe fixture")
        with self.assertRaisesRegex(ValueError, "pickle"):
            generic.load_pipeline(args, diffusers, "dtype")

    def test_single_file_configuration_converts_declared_legacy_components(self):
        import checkpoint_conversion
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        encoder = self.model / 'text_encoder';encoder.mkdir()
        source = encoder / 'pytorch_model.bin';source.write_bytes(b'legacy weights')
        converted = self.directory / 'encoder.safetensors';converted.write_bytes(b'safe weights')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        main = self.directory / 'checkpoint.safetensors';main.write_bytes(b'safe fixture')
        args = self.args('--model', str(main), '--model-config', str(self.model),
                         '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers();seen = []
        def load(_checkpoint, **kwargs):
            folder = Path(kwargs['config']);seen.append(folder)
            self.assertNotEqual(folder, self.model)
            self.assertEqual((folder / 'text_encoder/model.safetensors').read_bytes(), b'safe weights')
            self.assertEqual((folder / 'text_encoder/diffusion_pytorch_model.safetensors').read_bytes(), b'safe weights')
            self.assertFalse((folder / 'text_encoder/pytorch_model.bin').exists())
            return runtime.FakePipeline()
        runtime.FakePipeline.from_single_file = Mock(side_effect=load)
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertFalse(seen[0].exists())
        self.assertEqual(source.read_bytes(), b'legacy weights')
        self.assertEqual(len(report['identity']['directory_conversions']), 1)
        generic.verify_identity(report['identity'])

    def test_gguf_transformer_uses_builtin_component_and_verified_alias(self):
        root = self.model / 'transformer';root.mkdir()
        (root / 'config.json').write_text('{}')
        self.index['transformer'] = ['diffusers', 'TestTransformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        source = self.directory / 'weights.data';source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--model', str(source), '--model-config', str(self.model),
                         '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        self.assertTrue(args.single_file_component_only)
        runtime = self.fake_diffusers()
        class Model: pass
        class Original: pass
        class TestTransformer(Model, Original):
            @staticmethod
            def named_parameters(): return []
            @staticmethod
            def named_buffers(): return []
        transformer = TestTransformer()
        def load(path, **kwargs):
            self.assertEqual(Path(path).suffix, '.gguf')
            self.assertEqual(Path(path).read_bytes(), source.read_bytes())
            self.assertEqual(kwargs['subfolder'], 'transformer')
            self.assertTrue(kwargs['local_files_only'])
            self.assertTrue(kwargs['low_cpu_mem_usage'])
            return transformer
        TestTransformer.from_single_file = Mock(side_effect=load)
        runtime.ModelMixin, runtime.FromOriginalModelMixin, runtime.TestTransformer = Model, Original, TestTransformer
        runtime.GGUFQuantizationConfig = Mock(return_value=object())
        runtime.FakePipeline.__init__ = lambda self, transformer=None: None
        pipeline, report = generic.load_pipeline(args, runtime, 'dtype')
        self.assertIs(pipeline.__class__, runtime.FakePipeline)
        self.assertIs(runtime.FakePipeline.from_pretrained.call_args.kwargs['transformer'], transformer)
        runtime.FakePipeline.from_single_file.assert_not_called()
        runtime.GGUFQuantizationConfig.assert_called_once_with(compute_dtype='dtype')
        self.assertEqual(report['single_file_loading']['format'], 'gguf')
        self.assertEqual(report['single_file_loading']['component'], 'transformer')
        self.assertEqual(source.read_bytes(), b'GGUF' + bytes(64))

    def test_pipeline_loads_explicit_standalone_vae_file_with_its_configuration(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        folder = self.model / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source = self.directory / 'external-vae.data'
        header = json.dumps({'decoder.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
        header += b' ' * (-len(header) % 8)
        source.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        args = self.args('--vae', str(source), '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, vae=None: None
        vae = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        seen = []
        def load(directory, **_kwargs):
            seen.append(Path(directory))
            self.assertEqual((Path(directory) / 'diffusion_pytorch_model.safetensors').read_bytes(), source.read_bytes())
            return vae, {}
        runtime.AutoencoderKL = SimpleNamespace(from_pretrained=Mock(side_effect=load))
        generic.load_pipeline(args, runtime, 'float32')
        self.assertIs(runtime.FakePipeline.from_pretrained.call_args.kwargs['vae'], vae)
        self.assertFalse(seen[0].exists())
        self.assertTrue(source.exists())

    def test_pytorch_vae_conversion_cache_cannot_overlap_output(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        folder = self.model / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source = self.directory / 'external-vae.pt';source.write_bytes(b'PK\x03\x04synthetic weights')
        cache = self.directory / 'cache'
        args = self.args('--vae', str(source), '--cache-dir', str(cache),
                         '--output-dir', str(cache / 'vae-checkpoint-conversions' / 'output'), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, vae=None: None
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            with self.assertRaisesRegex(ValueError, 'output.*conversion cache'):
                generic.load_pipeline(args, runtime, 'float32')
            convert.assert_not_called()
        self.assertEqual(source.read_bytes(), b'PK\x03\x04synthetic weights')

    def test_pipeline_loads_converted_pytorch_vae_and_preserves_source(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        folder = self.model / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source = self.directory / 'external-vae.pt';source.write_bytes(b'PK\x03\x04synthetic weights')
        safe = self.directory / 'converted.safetensors'
        header = json.dumps({'decoder.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
        header += b' ' * (-len(header) % 8)
        safe.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        args = self.args('--vae', str(source), '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, vae=None: setattr(self, 'vae', vae)
        runtime.FakePipeline.from_pretrained.side_effect = lambda *_args, **kwargs: runtime.FakePipeline(vae=kwargs['vae'])
        vae = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        def load(directory, **_kwargs):
            self.assertEqual((Path(directory) / 'diffusion_pytorch_model.safetensors').read_bytes(), safe.read_bytes())
            return vae, {}
        runtime.AutoencoderKL = SimpleNamespace(from_pretrained=Mock(side_effect=load))
        conversion = {'converted': True, 'converted_path': str(safe), 'cache_hit': False,
                      'manifest_path': 'conversion.json', 'manifest': {
                          'conversion_policy': {'quantized_tensors': 'dequantize-float32-v1'},
                          'quantized_tensors': {'count': 1}}}
        with patch('checkpoint_conversion.materialize_safetensors', return_value=conversion) as convert, \
             patch('checkpoint_conversion.verify_conversion_source') as verify:
            _, report = generic.load_pipeline(args, runtime, 'float32')
        convert.assert_called_once_with(str(source), args.cache_dir / 'vae-checkpoint-conversions',
                                        quantized_tensors='dequantize')
        self.assertGreaterEqual(verify.call_count, 2)
        self.assertIs(runtime.FakePipeline.from_pretrained.call_args.kwargs['vae'], vae)
        self.assertEqual(report['vae_conversion']['quantized_tensors']['count'], 1)
        self.assertEqual(source.read_bytes(), b'PK\x03\x04synthetic weights')

    def test_pytorch_vae_print_config_never_starts_conversion(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        folder = self.model / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source = self.directory / 'external-vae.pt';source.write_bytes(b'PK\x03\x04synthetic weights')
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            args = self.args('--vae', str(source), '--cache-dir', str(self.directory / 'cache'),
                             '--print-config', '--no-default-modifiers')
            report = generic.configuration(args)
            self.assertEqual(report['vae']['selection']['container_kind'], 'pytorch')
            convert.assert_not_called()
        self.assertFalse(args.cache_dir.exists())

    def test_pytorch_vae_conversion_cache_resolves_symlinks_before_overlap_check(self):
        self.index['vae'] = ['diffusers', 'AutoencoderKL']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        folder = self.model / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source = self.directory / 'external-vae.pt';source.write_bytes(b'PK\x03\x04synthetic weights')
        cache, output = self.directory / 'cache', self.directory / 'output'
        cache.mkdir();output.mkdir()
        try:
            (cache / 'vae-checkpoint-conversions').symlink_to(output, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlinks are unavailable')
        args = self.args('--vae', str(source), '--cache-dir', str(cache), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        runtime.FakePipeline.__init__ = lambda self, vae=None: None
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            with self.assertRaisesRegex(ValueError, 'output.*conversion cache'):
                generic.load_pipeline(args, runtime, 'float32')
            convert.assert_not_called()

    def test_directory_gguf_component_is_injected_and_verified(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'diffusion_pytorch_model.bin';source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        self.assertEqual(generic.configuration(args)['directory_gguf_components']['transformer']['path'], str(source))
        runtime = self.fake_gguf_diffusers()
        pipeline, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertIs(pipeline.__class__, runtime.FakePipeline)
        self.assertIsInstance(runtime.FakePipeline.from_pretrained.call_args.kwargs['transformer'], runtime.Transformer)
        self.assertEqual(report['directory_gguf_loading']['transformer']['format'], 'gguf')
        alias = Path(runtime.Transformer.from_single_file.call_args.args[0])
        self.assertEqual(alias.suffix, '.gguf')
        self.assertFalse(alias.exists())
        self.assertEqual(source.read_bytes(), b'GGUF' + bytes(64))
        source.write_bytes(b'GGUF changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.verify_identity(report['identity'])

    def test_directory_gguf_selection_honors_standard_weights_and_explicit_variant(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        alternate = component / 'other.gguf';alternate.write_bytes(b'GGUF' + bytes(64))
        standard = component / 'diffusion_pytorch_model.safetensors';standard.write_bytes(b'safe weights')
        args = self.args('--no-default-modifiers')
        runtime = self.fake_gguf_diffusers()
        generic.load_pipeline(args, runtime, 'float32')
        runtime.Transformer.from_single_file.assert_not_called()
        self.assertNotIn('transformer', runtime.FakePipeline.from_pretrained.call_args.kwargs)
        selected = component / 'diffusion_pytorch_model.q4.gguf';selected.write_bytes(b'GGUF' + bytes(72))
        args = self.args('--variant', 'q4', '--no-default-modifiers')
        runtime = self.fake_gguf_diffusers()
        _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(Path(runtime.Transformer.from_single_file.call_args.args[0]).resolve(), selected.resolve())
        self.assertEqual(standard.read_bytes(), b'safe weights')
        standard.unlink()
        with self.assertRaisesRegex(ValueError, '[Aa]mbiguous.*GGUF'):
            self.args('--no-default-modifiers')
        args = self.args('--variant', 'q4', '--no-default-modifiers')
        runtime = self.fake_gguf_diffusers()
        _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(Path(runtime.Transformer.from_single_file.call_args.args[0]).resolve(), selected.resolve())
        self.assertEqual(report['directory_gguf_loading']['transformer']['format'], 'gguf')
        self.assertEqual(alternate.read_bytes(), b'GGUF' + bytes(64))

    def test_gguf_component_discovers_its_parent_pipeline_configuration(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'model.gguf';source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--model', str(source), '--no-default-modifiers')
        self.assertEqual(Path(args.model_config), self.model)
        self.assertEqual(args.model_config_status, 'adjacent')
        runtime = self.fake_gguf_diffusers()
        _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(report['single_file_loading']['component'], 'transformer')
        self.assertIsInstance(runtime.FakePipeline.from_pretrained.call_args.kwargs['transformer'], runtime.Transformer)
        other = self.directory / 'explicit-config';other.mkdir()
        (other / 'model_index.json').write_text(json.dumps(self.index))
        (other / 'transformer').mkdir();(other / 'transformer/config.json').write_text('{}')
        explicit = self.args('--model', str(source), '--model-config', str(other), '--no-default-modifiers')
        self.assertEqual(Path(explicit.model_config), other)
        self.assertEqual(explicit.model_config_status, 'explicit')

    def test_gguf_disguised_as_canonical_safetensors_is_selected_by_content(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'diffusion_pytorch_model.safetensors';source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--no-default-modifiers')
        runtime = self.fake_gguf_diffusers()
        _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(report['directory_gguf_loading']['transformer']['format'], 'gguf')
        self.assertEqual(Path(runtime.Transformer.from_single_file.call_args.args[0]).suffix, '.gguf')
        self.assertEqual(source.read_bytes(), b'GGUF' + bytes(64))

    def test_directory_gguf_with_nonstandard_filename_is_detected_by_content(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'artist-export.weights';source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--no-default-modifiers')
        runtime = self.fake_gguf_diffusers()
        _, report = generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual(report['directory_gguf_loading']['transformer']['format'], 'gguf')
        self.assertEqual(args.directory_gguf_components['transformer'].path, str(source))

    def test_directory_gguf_rejects_invalid_or_changed_content_before_loading(self):
        self.index['transformer'] = ['diffusers', 'Transformer']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        component = self.model / 'transformer';component.mkdir()
        (component / 'config.json').write_text('{}')
        source = component / 'broken.gguf';source.write_bytes(b'not gguf')
        with self.assertRaisesRegex(ValueError, 'invalid or changed'):
            self.args('--no-default-modifiers')
        source.write_bytes(b'GGUF' + bytes(64))
        args = self.args('--no-default-modifiers')
        source.write_bytes(b'GGUF modified')
        runtime = self.fake_gguf_diffusers()
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.load_pipeline(args, runtime, 'float32')
        runtime.Transformer.from_single_file.assert_not_called()

    def test_gguf_component_refuses_ambiguous_missing_and_meta_models(self):
        import gguf_component
        class Model: pass
        class Original: pass
        class Candidate(Model, Original):
            from_single_file = Mock()
        class Pipeline:
            def __init__(self, transformer=None, unet=None): pass
        runtime = SimpleNamespace(ModelMixin=Model, FromOriginalModelMixin=Original, Candidate=Candidate)
        declaration = ['diffusers', 'Candidate']
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            gguf_component.select({'transformer': declaration, 'unet': declaration}, Pipeline, runtime, self.model)
        with self.assertRaisesRegex(ValueError, 'config.json'):
            gguf_component.select({'transformer': declaration}, Pipeline, runtime, self.model)
        (self.model / 'transformer').mkdir()
        (self.model / 'transformer/config.json').write_text('{}')
        selection = gguf_component.select({'transformer': declaration}, Pipeline, runtime, self.model)
        source = self.directory / 'model.gguf';source.write_bytes(b'GGUF fixture')
        from weight_files import LocalWeightFile, cached_model_sha256
        weight = LocalWeightFile(str(source), str(source.resolve()), cached_model_sha256(source), source.stat().st_size)
        model = Candidate()
        model.named_parameters = lambda: [('weight', SimpleNamespace(is_meta=True))]
        model.named_buffers = lambda: []
        Candidate.from_single_file = Mock(return_value=model)
        runtime.GGUFQuantizationConfig = Mock(return_value=object())
        with self.assertRaisesRegex(RuntimeError, 'meta device'):
            gguf_component.load(selection, weight, self.model, self.directory / 'cache', 'dtype', runtime)

    def test_gguf_flux_component_does_not_treat_gguf_as_embedded_vae_or_encoder(self):
        self.index.update({'_class_name': 'FluxPipeline', 'vae': ['diffusers', 'AutoencoderKL'],
                           'text_encoder': ['transformers', 'CLIPTextModel'],
                           'text_encoder_2': ['transformers', 'T5EncoderModel']})
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        for name in ('vae', 'text_encoder', 'text_encoder_2'):
            folder = self.model / name;folder.mkdir()
            (folder / 'model.safetensors').write_bytes(b'existing component')
        source = self.directory / 'transformer.gguf';source.write_bytes(b'GGUF' + bytes(64))
        with patch.object(generic.flux_tokenizer, 'resolve', return_value=({}, {})):
            args = self.args('--model', str(source), '--model-config', str(self.model), '--no-default-modifiers')
        self.assertTrue(args.single_file_component_only)
        self.assertEqual(args.vae_status, 'model')
        self.assertEqual(args.flux_text_encoder_status, {'text_encoder': 'model', 'text_encoder_2': 'model'})

    def test_generic_single_file_accepts_safe_shard_index_via_verified_merge(self):
        import checkpoint_conversion
        import downloaded_model
        source = self.directory / 'checkpoint.safetensors.index.json';source.write_text('{}')
        merged = self.directory / 'merged.safetensors';merged.write_bytes(b'safe fixture')
        conversion = {'converted': True, 'converted_path': str(merged),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(merged), 'original_shards': []}
        with patch.object(downloaded_model, 'model_container_format',
                          side_effect=lambda path: 'safetensors-index' if Path(path) == source else 'safetensors'), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as merge:
            args = self.args('--model', str(source), '--model-config', str(self.model),
                             '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
            self.assertEqual(args.model, str(merged))
            self.assertEqual(args.original_model_path, str(source))
            self.assertEqual(generic.configuration(args)['model'], str(source))
            merge.assert_called_once()
            runtime = self.fake_diffusers()
            generic.load_pipeline(args, runtime, 'float32')
            self.assertEqual(Path(runtime.FakePipeline.from_single_file.call_args.args[0]).read_bytes(), b'safe fixture')
            merge.reset_mock()
            with self.assertRaisesRegex(ValueError, 'deferred'):
                self.args('--model', str(source), '--model-config', str(self.model), '--print-config',
                          '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
            merge.assert_not_called()
        self.assertEqual(source.read_text(), '{}')

    def test_single_file_configuration_accepts_component_variant(self):
        import downloaded_model
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        encoder = self.model / 'text_encoder';encoder.mkdir()
        (encoder / 'model.safetensors').write_bytes(b'default')
        (encoder / 'model.fp16.safetensors').write_bytes(b'fp16')
        main = self.directory / 'checkpoint.safetensors';main.write_bytes(b'safe fixture')
        with patch.object(downloaded_model, 'model_container_format', return_value='safetensors'):
            args = self.args('--model', str(main), '--model-config', str(self.model), '--variant', 'fp16',
                             '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
            runtime = self.fake_diffusers()
            def load(_checkpoint, **kwargs):
                self.assertNotIn('variant', kwargs)
                folder = Path(kwargs['config'])
                self.assertEqual((folder / 'text_encoder/model.safetensors').read_bytes(), b'fp16')
                return runtime.FakePipeline()
            runtime.FakePipeline.from_single_file = Mock(side_effect=load)
            generic.load_pipeline(args, runtime, 'float32')
        self.assertEqual((encoder / 'model.safetensors').read_bytes(), b'default')

    def test_single_file_diffusers_shards_are_merged_only_when_a_fallback_needs_them(self):
        import checkpoint_conversion
        import directory_checkpoint_conversion as directory
        import downloaded_model
        unet = self.model / 'unet';unet.mkdir()
        source = unet / 'diffusion_pytorch_model.safetensors.index.fp16.json'
        source.write_text('{}')
        converted = self.directory / 'merged.safetensors';converted.write_bytes(b'merged shards')
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(source),
                      'output': checkpoint_conversion._identity(converted)}
        with patch.object(downloaded_model, 'model_container_format', return_value='safetensors-index'), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as materialize:
            with directory.prepare(self.model, self.index, self.directory / 'cache', self.directory / 'output',
                    'fp16', strip_legacy=True, merge_diffusers_components={'unet'}) as prepared:
                self.assertEqual((prepared.path / 'unet/diffusion_pytorch_model.safetensors').read_bytes(),
                                 b'merged shards')
                self.assertFalse((prepared.path / 'unet/diffusion_pytorch_model.safetensors.index.fp16.json').exists())
            materialize.assert_called_once_with(source, self.directory / 'cache/checkpoint-conversions')
            with directory.prepare(self.model, self.index, self.directory / 'no-merge-cache',
                    self.directory / 'no-merge-output', 'fp16', strip_legacy=True) as prepared:
                self.assertEqual(prepared.path, self.model)
        self.assertEqual(source.read_text(), '{}')

    def test_single_file_fallback_components_exclude_checkpoint_aware_models(self):
        class Model: pass
        class FromOriginal: pass
        class Auxiliary(Model): pass
        class Embedded(Model, FromOriginal): pass
        runtime = SimpleNamespace(ModelMixin=Model, FromOriginalModelMixin=FromOriginal,
                                  Auxiliary=Auxiliary, Embedded=Embedded)
        index = {'aux': ['diffusers', 'Auxiliary'], 'embedded': ['diffusers', 'Embedded'],
                 'encoder': ['transformers', 'Auxiliary'], 'missing': ['diffusers', 'Missing']}
        self.assertEqual(generic.single_file_diffusers_fallback_components(runtime, index), {'aux'})

    def test_single_file_transformer_markers_preserve_sources_and_selected_variant(self):
        import directory_checkpoint_conversion as directory
        import downloaded_model
        import image_package
        index = {'_class_name': 'FakePipeline', 'text_encoder': ['transformers', 'CLIPTextModel']}
        for kind in ('default', 'variant', 'sharded-variant'):
            with self.subTest(kind=kind):
                root = self.directory / kind
                encoder = root / 'text_encoder';encoder.mkdir(parents=True)
                (encoder / 'model.safetensors').write_bytes(b'default weights')
                variant = None
                if kind == 'variant':
                    variant = 'fp16'
                    (encoder / 'model.fp16.safetensors').write_bytes(b'variant weights')
                elif kind == 'sharded-variant':
                    variant = 'fp16'
                    (encoder / 'model.safetensors.index.fp16.json').write_text('{}')
                    (encoder / 'model.fp16-00001-of-00002.safetensors').write_bytes(b'variant shard')
                before = {item.name: item.read_bytes() for item in encoder.iterdir()}
                def container(path):
                    return 'safetensors-index' if str(path).endswith('.json') else 'safetensors'
                report = {'shards': [{'relative_path': 'model.fp16-00001-of-00002.safetensors'}]}
                with patch.object(downloaded_model, 'model_container_format', side_effect=container), \
                     patch.object(image_package, 'inspect_sharded_index', return_value=report):
                    with directory.prepare(root, index, self.directory / (kind + '-cache'),
                            self.directory / (kind + '-output'), variant, strip_legacy=True,
                            transformer_markers=True) as prepared:
                        folder = prepared.path / 'text_encoder'
                        expected = {'default': b'default weights', 'variant': b'variant weights',
                                    'sharded-variant': b'variant shard'}[kind]
                        self.assertEqual((folder / 'diffusion_pytorch_model.safetensors').read_bytes(), expected)
                        if kind == 'variant':
                            self.assertEqual((folder / 'model.safetensors').read_bytes(), expected)
                        elif kind == 'sharded-variant':
                            self.assertFalse((folder / 'model.safetensors').exists())
                            self.assertTrue((folder / 'model.safetensors.index.json').is_file())
                        with directory.prepare(root, index, self.directory / (kind + '-excluded-cache'),
                                self.directory / (kind + '-excluded-output'), variant, {'text_encoder'},
                                strip_legacy=True, transformer_markers=True) as excluded:
                            self.assertEqual(excluded.path, root)
                self.assertEqual({item.name: item.read_bytes() for item in encoder.iterdir()}, before)

    def test_single_file_configuration_hides_unused_overridden_pickle(self):
        import checkpoint_conversion
        self.index['text_encoder'] = ['transformers', 'CLIPTextModel']
        (self.model / 'model_index.json').write_text(json.dumps(self.index))
        encoder = self.model / 'text_encoder';encoder.mkdir()
        source = encoder / 'pytorch_model.bin';source.write_bytes(b'unused invalid pickle')
        main = self.directory / 'checkpoint.safetensors';main.write_bytes(b'safe fixture')
        args = self.args('--model', str(main), '--model-config', str(self.model),
                         '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        args.text_encoder_selection = object()
        runtime = self.fake_diffusers();runtime.FakePipeline.__init__ = lambda self, text_encoder=None: None
        seen = []
        def load(_checkpoint, **kwargs):
            folder = Path(kwargs['config']);seen.append(folder)
            self.assertFalse((folder / 'text_encoder/pytorch_model.bin').exists())
            self.assertIsNotNone(kwargs['text_encoder'])
            return runtime.FakePipeline()
        runtime.FakePipeline.from_single_file = Mock(side_effect=load)
        with patch.object(generic.krea2_text_encoder, 'load', return_value=object()), \
             patch.object(generic.krea2_text_encoder, 'verify_selection'), \
             patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            _, report = generic.load_pipeline(args, runtime, 'float32')
        convert.assert_not_called()
        self.assertFalse(seen[0].exists())
        self.assertEqual(source.read_bytes(), b'unused invalid pickle')
        self.assertIn('directory_sources', report['identity'])

    def test_single_file_moss_keeps_primary_transformer_out_of_auxiliary_conversion(self):
        import checkpoint_conversion
        import moss_runtime
        index = {'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}
        (self.model / 'model_index.json').write_text(json.dumps(index))
        self.provide_stub_moss_encoder()
        for name in ('transformer', 'vae', 'tokenizer'):
            (self.model / name).mkdir()
        legacy = self.model / 'transformer/diffusion_pytorch_model.bin'
        legacy.write_bytes(b'unused invalid pickle')
        (self.model / 'vae/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        (self.model / 'tokenizer/vocab.json').write_text('{}')
        main = self.directory / 'checkpoint.safetensors';main.write_bytes(b'safe fixture')
        with patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')):
            args = self.args('--model', str(main), '--model-config', str(self.model),
                             '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers()
        class MOSSImagePipeline(runtime.DiffusionPipeline):
            def __init__(self): self.transformer = SimpleNamespace(_iild_single_file_loading={})
            def __call__(self): pass
        instance = MOSSImagePipeline();seen = []
        def load(folder, _dtype, _vae, _diffusers, **kwargs):
            folder = Path(folder);seen.append(folder)
            self.assertFalse((folder / 'transformer/diffusion_pytorch_model.bin').exists())
            self.assertEqual(kwargs['transformer_file'], str(main))
            return instance
        with patch.object(moss_runtime, 'pipeline_class', return_value=MOSSImagePipeline), \
             patch.object(moss_runtime, 'load', side_effect=load), \
             patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
            loaded, _ = generic.load_pipeline(args, runtime, 'float32')
        self.assertIs(loaded, instance)
        self.assertFalse(seen[0].exists())
        self.assertEqual(legacy.read_bytes(), b'unused invalid pickle')
        convert.assert_not_called()

    def test_single_file_primary_inside_config_is_not_treated_as_unknown_pickle(self):
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        main = self.model / 'checkpoint.bin'
        original = struct.pack('<Q', len(header)) + header + b'x'
        main.write_bytes(original)
        args = self.args('--model', str(main), '--model-config', str(self.model),
                         '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers')
        runtime = self.fake_diffusers();seen = []
        def load(checkpoint, **kwargs):
            folder = Path(kwargs['config']);seen.append(folder)
            self.assertEqual(Path(checkpoint).read_bytes(), original)
            self.assertFalse((folder / 'checkpoint.bin').exists())
            return runtime.FakePipeline()
        runtime.FakePipeline.from_single_file = Mock(side_effect=load)
        generic.load_pipeline(args, runtime, 'float32')
        self.assertFalse(seen[0].exists())
        self.assertEqual(main.read_bytes(), original)

    def test_renamed_safetensors_reach_the_safe_single_file_loader(self):
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        source_bytes = struct.pack('<Q', len(header)) + header + b'x'
        for suffix in ('.bin', '.data', '.gguf', '.safetensor'):
            with self.subTest(suffix=suffix):
                path = self.directory / ('checkpoint' + suffix)
                path.write_bytes(source_bytes)
                args = self.args('--model', str(path), '--model-config', str(self.model),
                                 '--cache-dir', str(self.directory / 'cache'))
                diffusers = self.fake_diffusers()
                observed = []
                def load(alias, **kwargs):
                    alias = Path(alias)
                    self.assertEqual(alias.suffix, '.safetensors')
                    self.assertEqual(alias.resolve(), path.resolve())
                    self.assertEqual(alias.read_bytes(), source_bytes)
                    observed.append(alias)
                    return diffusers.FakePipeline()
                diffusers.FakePipeline.from_single_file = Mock(side_effect=load)
                _, report = generic.load_pipeline(args, diffusers, 'dtype')
                self.assertEqual(report['identity']['single_file']['path'], str(path))
                self.assertEqual(path.read_bytes(), source_bytes)
                self.assertEqual(len(observed), 1)
                self.assertFalse(observed[0].exists())

    def test_single_file_recognizes_gguf_content_and_rejects_onnx(self):
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        path = self.directory / 'fake.safetensors'
        path.write_bytes(b'GGUF' + bytes(20))
        args = self.args('--model', str(path), '--model-config', str(self.model))
        self.assertTrue(args.single_file_component_only)
        path = self.directory / 'disguised.onnx'
        path.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        with self.assertRaisesRegex(ValueError, 'safetensors'):
            self.args('--model', str(path), '--model-config', str(self.model))

    def test_disguised_pytorch_is_only_sent_through_safe_conversion(self):
        import checkpoint_conversion
        for name in ('unsafe.safetensors', 'pickle.bin'):
            with self.subTest(name=name):
                path = self.directory / name
                path.write_bytes(b'PK\x03\x04' + bytes(20))
                with patch.object(checkpoint_conversion, 'materialize_safetensors',
                                  side_effect=ValueError('unsafe pickle retry is disabled')) as materialize:
                    with self.assertRaisesRegex(ValueError, 'unsafe pickle retry is disabled'):
                        self.args('--model', str(path), '--model-config', str(self.model))
                    materialize.assert_called_once()

    def test_renamed_single_file_change_during_load_is_rejected_and_alias_is_removed(self):
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        path = self.directory / 'source.bin'
        path.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        args = self.args('--model', str(path), '--model-config', str(self.model),
                         '--cache-dir', str(self.directory / 'cache'))
        diffusers = self.fake_diffusers()
        seen = []
        def change_source(alias, **kwargs):
            seen.append(Path(alias))
            path.write_bytes(path.read_bytes()[:-1] + b'y')
            return diffusers.FakePipeline()
        diffusers.FakePipeline.from_single_file = Mock(side_effect=change_source)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            generic.load_pipeline(args, diffusers, 'dtype')
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())

    def test_model_identity_detects_replacement_with_equal_byte_size(self):
        path = self.model / "weights.safetensors"
        path.write_bytes(b"old")
        identity = generic.model_identity(self.model)
        path.write_bytes(b"new")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            generic.verify_identity(identity)

    def test_image_paths_decode_only_in_controlled_input_fields_and_preserve_batch_order(self):
        path = self.directory / "image.png"
        path.write_bytes(b"image fixture")
        inputs, files = generic.prepare_inputs({"image": [{"image_path": str(path), "mode": "L"}]}, self.fake_images)
        self.assertIsInstance(inputs["image"][0], FakeImage)
        self.assertEqual(files[0]["mode"], "L")
        self.assertEqual(files[0]["input"], "image")
        for value in ({"prompt": {"image_path": str(path)}},
                      {"image": {"image_path": str(path), "other": 1}},
                      {"image": {"image_path": str(path), "mode": "CMYK"}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generic.prepare_inputs(value, self.fake_images)

    def test_animated_images_require_explicit_frame_list(self):
        path = self.directory / "image.gif"
        path.write_bytes(b"fixture")
        self.fake_images.open.return_value.is_animated = True
        with self.assertRaisesRegex(ValueError, "Animated"):
            generic.prepare_inputs({"video": {"image_path": str(path)}}, self.fake_images)

    def test_image_and_video_batches_export_all_frames_with_provenance(self):
        args = self.args()
        outputs = generic.save_outputs(SimpleNamespace(images=[FakeImage(), FakeImage()],
                                        frames=[[FakeImage(), FakeImage()], [FakeImage()]]),
                                       None, args, self.fake_images, None)
        self.assertEqual(len(outputs), 5)
        self.assertEqual(sum(item["kind"] == "video-frame" for item in outputs), 3)
        self.assertTrue(all(Path(item["path"]).is_file() and item["sha256"] for item in outputs))

    def test_video_layout_preserves_short_frame_counts_without_channel_guessing(self):
        class Array:
            def __init__(self, shape):
                self.shape, self.ndim = shape, len(shape)
            def transpose(self, *axes):
                return Array(tuple(self.shape[axis] for axis in axes))
            def __len__(self):
                return self.shape[0]
            def __iter__(self):
                for _ in range(self.shape[0]):
                    yield [FakeImage() for _ in range(self.shape[1])]
        class Tensor(Array):
            def detach(self):
                return self
            def cpu(self):
                return self
            def float(self):
                return self
            def is_floating_point(self):
                return True
            def numpy(self):
                return Array(self.shape)
        np = SimpleNamespace(asarray=lambda value: value)
        for value, layout in ((Tensor((1, 4, 3, 16, 16)), "auto"),
                              (Array((1, 4, 16, 16, 3)), "auto"),
                              (Tensor((1, 3, 4, 16, 16)), "bcfhw")):
            with self.subTest(layout=layout, shape=value.shape):
                outputs = generic.save_outputs({"frames": value}, None,
                    self.args("--overwrite", "--video-layout", layout), self.fake_images, np)
                self.assertEqual(len(outputs), 4)
        with self.assertRaisesRegex(ValueError, "invalid channel axis"):
            generic.save_outputs({"frames": Array((1, 4, 3, 16, 16))}, None,
                                 self.args(), self.fake_images, np)

    def test_unknown_outputs_do_not_claim_success(self):
        with self.assertRaisesRegex(ValueError, "no supported media"):
            generic.save_outputs(SimpleNamespace(meshes=["mesh"]), None, self.args(), self.fake_images, None)

    def test_empty_declared_media_is_not_accepted_as_a_complete_batch(self):
        for result in ({"images": []}, {"frames": []}, {"frames": [[]]},
                       {"images": [FakeImage()], "frames": [[]]}):
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "empty"):
                generic.save_outputs(result, None, self.args("--overwrite"), self.fake_images, None)

    def test_overwrite_failure_archives_old_report_before_changing_artifacts(self):
        class FailingImage(FakeImage):
            def save(self, path, **kwargs):
                raise RuntimeError("second image encoding failed")
        args = self.args("--overwrite")
        args.output_dir.mkdir()
        prior = '{"old":"report remains preserved"}'
        (args.output_dir / "generation.json").write_text(prior)
        (args.output_dir / "image-0001.png").write_bytes(b"old pixels")
        (args.output_dir / "unrelated.txt").write_text("preserve")
        with self.assertRaisesRegex(RuntimeError, "second image"):
            generic.publish_generation({"images": [FakeImage(), FailingImage()]}, None, args,
                                       self.fake_images, None, {"request": "new"})
        self.assertFalse((args.output_dir / "generation.json").exists())
        archives = list(args.output_dir.glob("generation.previous-*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), prior)
        self.assertEqual((args.output_dir / "image-0001.png").read_bytes(), b"fake-png-contract-only")
        self.assertEqual((args.output_dir / "unrelated.txt").read_text(), "preserve")

    def test_completed_overwrite_publishes_new_report_with_actual_safety_flags(self):
        args = self.args("--overwrite")
        args.output_dir.mkdir()
        (args.output_dir / "generation.json").write_text('{"old":true}')
        old_archive = args.output_dir / "generation.previous-existing.json"
        old_archive.write_text('{"older":true}')
        path, outputs = generic.publish_generation(
            {"images": [FakeImage(), FakeImage()], "nsfw_content_detected": [False, True]},
            SimpleNamespace(safety_checker=object()), args, self.fake_images, None, {"request": "new"})
        report = json.loads(path.read_text())
        self.assertEqual(report["request"], "new")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(report["safety"], {"checker_present": True,
                         "pipeline_report": {"nsfw_content_detected": [False, True]}})
        self.assertTrue(Path(report["previous_report"]).is_file())
        self.assertEqual(old_archive.read_text(), '{"older":true}')

    def test_misaligned_safety_flags_leave_no_success_report(self):
        args = self.args()
        with self.assertRaisesRegex(ValueError, "flags do not match"):
            generic.publish_generation({"images": [FakeImage()], "nsfw_content_detected": [True, False]},
                                       None, args, self.fake_images, None, {})
        self.assertFalse((args.output_dir / "generation.json").exists())

    def test_absent_safety_checker_is_not_reported_as_successful_screening(self):
        report = generic.safety_metadata({}, None, [{"kind": "image"}])
        self.assertEqual(report, {"checker_present": False, "pipeline_report": {}})

    def test_atomic_output_collision_does_not_overwrite_existing_content(self):
        path = self.directory / "image.png"
        path.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            generic.atomic_write(path, lambda temporary: temporary.write_bytes(b"new"), False)
        self.assertEqual(path.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])
        generic.atomic_write(path, lambda temporary: temporary.write_bytes(b"new"), True)
        self.assertEqual(path.read_bytes(), b"new")

    def test_failed_writer_cleans_its_temporary_file(self):
        with self.assertRaises(RuntimeError):
            generic.atomic_write(self.directory / "failed.png", Mock(side_effect=RuntimeError("encoding failed")), False)
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])

    def test_audio_rate_is_explicit_or_read_from_the_actual_component(self):
        pipeline = SimpleNamespace(vocoder=SimpleNamespace(config=SimpleNamespace(sampling_rate=16000)))
        self.assertEqual(generic.infer_sample_rate(pipeline, None), 16000)
        self.assertEqual(generic.infer_sample_rate(pipeline, 48000), 48000)
        with self.assertRaisesRegex(ValueError, "audio-sample-rate"):
            generic.infer_sample_rate(SimpleNamespace(), None)

    def test_hardware_policy_checks_offloaded_execution_device(self):
        pipeline = Mock(_execution_device=SimpleNamespace(type="mps"))
        generic.prepare_execution(pipeline, "mps", "model")
        pipeline.enable_model_cpu_offload.assert_called_once_with(device="mps")
        pipeline.to.assert_not_called()
        pipeline._execution_device.type = "cpu"
        with self.assertRaisesRegex(RuntimeError, "selected hardware"):
            generic.prepare_execution(pipeline, "mps", "sequential")


if __name__ == "__main__":
    unittest.main()
