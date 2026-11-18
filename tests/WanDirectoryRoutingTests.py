"""Wan Diffusers directory discovery and dispatch using synthetic weights only."""
import ctypes as C
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
from _file_provider import provider
import image_package


class WanDirectoryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = provider.TemporaryDirectory(prefix='wan-directory-', dir=ROOT / 'resources')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'pipeline'
        self.model.mkdir()
        self.manifest = {
            '_class_name': 'WanPipeline',
            'scheduler': ['diffusers', 'UniPCMultistepScheduler'],
            'text_encoder': ['transformers', 'UMT5EncoderModel'],
            'tokenizer': ['transformers', 'T5TokenizerFast'],
            'transformer': ['diffusers', 'WanTransformer3DModel'],
            'vae': ['diffusers', 'AutoencoderKLWan'],
        }
        self.write_manifest()
        self.weight('transformer', 'diffusion_pytorch_model.safetensors', 'patch_embedding.weight')
        self.weight('text_encoder', 'model.safetensors', 'shared.weight')
        self.weight('vae', 'diffusion_pytorch_model.safetensors', 'decoder.conv1.weight')

    def write_manifest(self):
        (self.model / 'model_index.json').write_text(json.dumps(self.manifest))

    def weight(self, component, filename, tensor):
        directory = self.model / component
        directory.mkdir(exist_ok=True)
        header = json.dumps({tensor: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        path = directory / filename
        path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(4))
        return path

    def inspect(self):
        return image_package.inspect_wan_directory(self.model)

    def test_single_expert_resolves_components_by_content(self):
        original = self.model / 'text_encoder/model.safetensors'
        renamed = original.with_name('weights.bin')
        original.rename(renamed)
        route = self.inspect()
        self.assertEqual(route['schema'], 'iild-wan-directory-v1')
        self.assertEqual(Path(route['model']), (self.model / 'transformer/diffusion_pytorch_model.safetensors').resolve())
        self.assertEqual(Path(route['text_encoder']), renamed.resolve())
        self.assertEqual(Path(route['video_vae']), (self.model / 'vae/diffusion_pytorch_model.safetensors').resolve())
        self.assertIsNone(route['high_noise_model'])
        self.assertIsNone(route['moe_boundary'])
        self.assertEqual(route['scheduler'], 'UniPCMultistepScheduler')

    def test_image_to_video_directory_resolves_declared_vision_encoder(self):
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
            'image_processor': ['transformers', 'CLIPProcessor'],
        })
        self.write_manifest()
        vision = self.weight('image_encoder', 'model.safetensors', 'vision_model.embeddings.patch_embedding.weight')
        route = self.inspect()
        self.assertEqual(route['pipeline'], 'WanImageToVideoPipeline')
        self.assertEqual(route['vision_encoder'], str(vision.resolve()))
        self.assertIn(str(vision), route['dependencies'])

    def test_image_to_video_dual_expert_allows_declared_null_vision(self):
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': [None, None], 'image_processor': [None, None],
            'transformer_2': ['diffusers', 'WanTransformer3DModel'], 'boundary_ratio': 0.9,
        })
        self.write_manifest()
        low = self.weight('transformer_2', 'low.safetensors', 'patch_embedding.weight')
        route = self.inspect()
        self.assertEqual(route['pipeline'], 'WanImageToVideoPipeline')
        self.assertIsNone(route['vision_encoder'])
        self.assertEqual(route['model'], str(low.resolve()))
        self.assertEqual(route['moe_boundary'], 0.9)

    def test_image_to_video_rejects_inconsistent_vision_declarations(self):
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
            'image_processor': [None, None],
        })
        self.write_manifest()
        self.weight('image_encoder', 'model.safetensors', 'vision_model.embeddings.patch_embedding.weight')
        with self.assertRaisesRegex(ValueError, 'image_(encoder|processor)|vision'):
            self.inspect()

    def test_image_to_video_dispatch_requires_first_frame_and_preserves_vision(self):
        import generate_video
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
            'image_processor': ['transformers', 'CLIPProcessor'],
        })
        self.write_manifest()
        vision = self.weight('image_encoder', 'model.safetensors', 'vision_model.embeddings.patch_embedding.weight')
        first = self.root / 'first.png'
        first.write_bytes(b'not decoded in this test')
        arguments = ['--model', str(self.model), '--frames', '9', '--fps', '12', '--width', '32',
                     '--height', '32', '--output', str(self.root / 'result.mp4')]
        with patch('native_video.generate_video') as native:
            with self.assertRaises(SystemExit):
                generate_video.main(arguments)
            native.assert_not_called()
            with self.assertRaises(SystemExit):
                generate_video.main(arguments + ['--last-frame', str(first)])
            native.assert_not_called()
            self.assertEqual(generate_video.main(arguments + ['--first-frame', str(first)]), 0)
            args = native.call_args.args[0]
            self.assertEqual(args.vision_encoder, vision.resolve())
            self.assertEqual(args.shots[0]['conditions'][0]['frame'], 0)
            self.assertEqual(args._native_directory_source['declared_pipeline'], 'WanImageToVideoPipeline')
            args._native_source_verifier()
            vision.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                args._native_source_verifier()

    def test_image_to_video_first_frame_and_vision_reach_native_abi(self):
        from types import SimpleNamespace
        from native_video import Components, NativeVideoEngine, Request
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
            'image_processor': ['transformers', 'CLIPProcessor'],
        })
        self.write_manifest()
        vision = self.weight('image_encoder', 'model.safetensors', 'vision_model.embeddings.patch_embedding.weight')
        first = self.root / 'first.png'
        first.write_bytes(b'keyframe fixture')
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--first-frame', str(first),
            '--output', str(self.root / 'result.mp4')])))
        frames = self.root / 'frames'
        frames.mkdir()
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        observed = {}
        def generate(request, components, progress, frame, user):
            req = C.cast(request, C.POINTER(Request)).contents
            values = C.cast(components, C.POINTER(Components)).contents
            shot = req.shots[0]
            observed.update(vision=values.vision_encoder.decode(),
                            first=C.string_at(shot.first_rgb, shot.first_size), last_size=shot.last_size)
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for index in range(9):
                frame(0, index, 32, 32, data, 32 * 32 * 3, None)
            return 1
        engine.generate_with_components = generate
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
        engine.free = lambda _: None
        keyframe = bytes([27]) * (32 * 32 * 3)
        image = SimpleNamespace(tobytes=lambda: keyframe)
        fake_image = SimpleNamespace(frombytes=lambda *unused: object())
        with patch('native_video.provider.save_image', side_effect=lambda image, path, **kwargs: path.write_bytes(b'frame')):
            engine.render(args, SimpleNamespace(frames=frames), fake_image, {str(first.resolve()): image})
        self.assertEqual(observed, {'vision': str(vision.resolve()), 'first': keyframe, 'last_size': 0})

    def test_image_to_video_linked_vision_index_retains_its_shard_base(self):
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
            'image_processor': ['transformers', 'CLIPProcessor'],
        })
        self.write_manifest()
        shard = self.weight('image_encoder', 'part.safetensors', 'vision_model.embeddings.patch_embedding.weight')
        blobs = self.root / 'vision-blobs'
        blobs.mkdir()
        weight_blob = blobs / 'weights'
        shard.rename(weight_blob)
        index_blob = blobs / 'index'
        index_blob.write_text(json.dumps({'weight_map': {'vision_model.embeddings.patch_embedding.weight': shard.name}}))
        index = shard.with_name('model.safetensors.index.json')
        try:
            shard.symlink_to(weight_blob)
            index.symlink_to(index_blob)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')
        first = self.root / 'first.png'
        first.write_bytes(b'keyframe fixture')
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--first-frame', str(first),
            '--output', str(self.root / 'result.mp4')])))
        self.assertEqual(args.vision_encoder, index)
        self.assertIn(str(shard), self.inspect()['dependencies'])
        args._native_source_verifier()
        replacement = blobs / 'replacement'
        replacement.write_bytes(weight_blob.read_bytes())
        shard.unlink()
        shard.symlink_to(replacement)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            args._native_source_verifier()

    def test_image_to_video_without_vision_dispatches_both_experts(self):
        import generate_video
        self.manifest.update({
            '_class_name': 'WanImageToVideoPipeline',
            'image_encoder': [None, None], 'image_processor': [None, None],
            'transformer_2': ['diffusers', 'WanTransformer3DModel'], 'boundary_ratio': 0.9,
        })
        self.write_manifest()
        low = self.weight('transformer_2', 'low.safetensors', 'patch_embedding.weight')
        first = self.root / 'first.png'
        first.write_bytes(b'keyframe fixture')
        with patch('native_video.generate_video') as native:
            self.assertEqual(generate_video.main(['--model', str(self.model), '--frames', '9', '--fps', '12',
                '--first-frame', str(first), '--output', str(self.root / 'result.mp4')]), 0)
            args = native.call_args.args[0]
            self.assertEqual(args.model, str(low.resolve()))
            self.assertEqual(args.high_noise_model, (self.model / 'transformer/diffusion_pytorch_model.safetensors').resolve())
            self.assertIsNone(args.vision_encoder)
            self.assertEqual(args.moe_boundary, 0.9)

    def test_older_sdk_text_report_remains_usable(self):
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        report = self.inspect()
        report.pop('pipeline')
        report.pop('vision_encoder')
        args = resolve_options(build_parser().parse_args(['--model', str(self.model), '--frames', '9',
            '--fps', '12', '--output', str(self.root / 'result.mp4')]))
        with patch('wan_directory.inspect_wan_directory', return_value=report):
            adapted = prepare(args)
        self.assertEqual(adapted._native_directory_source['declared_pipeline'], 'WanPipeline')
        self.assertIsNone(adapted.vision_encoder)

    def test_incomplete_sdk_image_report_is_rejected(self):
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        self.manifest.update({'_class_name': 'WanImageToVideoPipeline',
            'image_encoder': [None, None], 'image_processor': [None, None]})
        self.write_manifest()
        report = self.inspect()
        report.pop('vision_encoder')
        first = self.root / 'first.png'
        first.write_bytes(b'keyframe fixture')
        args = resolve_options(build_parser().parse_args(['--model', str(self.model), '--frames', '9',
            '--fps', '12', '--first-frame', str(first), '--output', str(self.root / 'result.mp4')]))
        with patch('wan_directory.inspect_wan_directory', return_value=report), \
             self.assertRaisesRegex(ValueError, 'runtime|SDK'):
            prepare(args)

    def test_dual_expert_and_validated_shards(self):
        self.manifest['transformer_2'] = ['diffusers', 'WanTransformer3DModel']
        self.manifest['boundary_ratio'] = 0.875
        self.write_manifest()
        part = self.weight('transformer_2', 'part.safetensors', 'patch_embedding.weight')
        index = part.with_name('diffusion_pytorch_model.safetensors.index.json')
        index.write_text(json.dumps({'weight_map': {'patch_embedding.weight': part.name}}))
        route = self.inspect()
        self.assertEqual(Path(route['model']), index.resolve())
        self.assertEqual(Path(route['high_noise_model']), (self.model / 'transformer/diffusion_pytorch_model.safetensors').resolve())
        self.assertEqual(route['moe_boundary'], 0.875)
        self.assertIn(str(part.resolve()), route['dependencies'])
        part.write_bytes(b'invalid')
        with self.assertRaises(ValueError):
            self.inspect()

    def test_renamed_index_is_selected_as_one_wan_component(self):
        original = self.model / 'transformer/diffusion_pytorch_model.safetensors'
        part = original.with_name('shard.bin')
        original.rename(part)
        index = original.with_name('component-index.data')
        index.write_text(json.dumps({'weight_map': {'patch_embedding.weight': part.name}}))
        route = self.inspect()
        self.assertEqual(Path(route['model']), index)
        self.assertIn(str(index), route['dependencies'])
        self.assertIn(str(part), route['dependencies'])

    def test_ambiguous_missing_and_wrong_components_fail_closed(self):
        self.weight('vae', 'another.safetensors', 'decoder.other.weight')
        with self.assertRaisesRegex(ValueError, 'vae.*(multiple|ambiguous)'):
            self.inspect()
        (self.model / 'vae/another.safetensors').unlink()
        (self.model / 'vae/diffusion_pytorch_model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'vae'):
            self.inspect()
        self.weight('vae', 'diffusion_pytorch_model.safetensors', 'decoder.conv1.weight')
        self.manifest['text_encoder'] = ['custom_module', 'UMT5EncoderModel']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'text_encoder'):
            self.inspect()

    def test_duplicate_json_and_invalid_expert_boundary_fail_closed(self):
        (self.model / 'model_index.json').write_text('{"_class_name":"WanPipeline","_class_name":"Other"}')
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
            self.inspect()
        self.manifest['transformer_2'] = ['diffusers', 'WanTransformer3DModel']
        self.manifest['boundary_ratio'] = 1.5
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'boundary'):
            self.inspect()

    def test_index_must_not_escape_its_component_directory(self):
        original = self.model / 'transformer/diffusion_pytorch_model.safetensors'
        original.rename(self.root / 'outside.safetensors')
        index = self.model / 'transformer/diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'patch_embedding.weight': '../../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError, '[Uu]nsafe'):
            self.inspect()

    def test_retargeted_component_link_invalidates_a_prepared_request(self):
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        component = self.model / 'vae/diffusion_pytorch_model.safetensors'
        target_one = self.root / 'first.safetensors'
        target_two = self.root / 'second.safetensors'
        component.rename(target_one)
        target_two.write_bytes(target_one.read_bytes())
        try:
            component.symlink_to(target_one)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--output', str(self.root / 'result.mp4')])))
        args._native_source_verifier()
        component.unlink()
        component.symlink_to(target_two)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            args._native_source_verifier()

    def test_output_cannot_replace_a_component_behind_a_link(self):
        import generate_video
        component = self.model / 'vae/diffusion_pytorch_model.safetensors'
        target = self.root / 'would-overwrite.mp4'
        component.rename(target)
        original = target.read_bytes()
        try:
            component.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')
        with patch('native_video.generate_video') as native:
            with self.assertRaises(SystemExit):
                generate_video.main(['--model', str(self.model), '--width', '32', '--height', '32',
                    '--frames', '9', '--fps', '12', '--output', str(target)])
            native.assert_not_called()
        self.assertEqual(target.read_bytes(), original)

    def test_index_link_target_does_not_need_an_index_suffix(self):
        part = self.model / 'transformer/diffusion_pytorch_model.safetensors'
        target = part.with_name('index-content.json')
        target.write_text(json.dumps({'weight_map': {'patch_embedding.weight': part.name}}))
        link = part.with_name('diffusion_pytorch_model.safetensors.index.json')
        try:
            link.symlink_to(target.name)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')
        route = self.inspect()
        self.assertEqual(route['model'], str(link))
        self.assertIn(str(part), route['dependencies'])

    def test_huggingface_snapshot_links_retain_their_index_base(self):
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        blobs = self.root / 'blobs'
        blobs.mkdir()
        selected = {}
        for name, tensor in (('transformer', 'patch_embedding.weight'), ('text_encoder', 'shared.weight'),
                             ('vae', 'decoder.conv1.weight')):
            folder = self.model / name
            source = next(folder.glob('*.safetensors'))
            blob = blobs / (name + '-weights')
            source.rename(blob)
            index_blob = blobs / (name + '-index')
            index_blob.write_text(json.dumps({'weight_map': {tensor: source.name}}))
            link = folder / 'model.safetensors.index.json'
            try:
                source.symlink_to(blob)
                link.symlink_to(index_blob)
            except (OSError, NotImplementedError):
                self.skipTest('Symbolic links are unavailable')
            selected[name] = (link, source, blob)
        route = self.inspect()
        self.assertEqual(route['model'], str(selected['transformer'][0]))
        self.assertEqual(route['text_encoder'], str(selected['text_encoder'][0]))
        self.assertEqual(route['video_vae'], str(selected['vae'][0]))
        self.assertIn(str(selected['transformer'][1]), route['dependencies'])
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--output', str(self.root / 'result.mp4')])))
        self.assertEqual(args.model, str(selected['transformer'][0]))
        args._native_source_verifier()
        link, source, blob = selected['vae']
        alternate = blobs / 'alternate-vae-weights'
        alternate.write_bytes(blob.read_bytes())
        source.unlink()
        source.symlink_to(alternate)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            args._native_source_verifier()

    def test_directory_dispatch_preserves_configuration_and_rejects_output_inside_model(self):
        import generate_video
        arguments = ['--model', str(self.model), '--frames', '9', '--fps', '12', '--width', '32',
                     '--height', '32', '--decode-timestep', '0.05', '--decode-noise-scale', '0.025',
                     '--image-cond-noise-scale', '0', '--output', str(self.root / 'result.mp4')]
        with patch('native_video.generate_video') as native, patch('video_runtime.generate_video') as legacy:
            self.assertEqual(generate_video.main(arguments), 0)
            args = native.call_args.args[0]
            self.assertEqual(Path(args.model), (self.model / 'transformer/diffusion_pytorch_model.safetensors').resolve())
            self.assertEqual(args._native_original_configuration['model'], str(self.model))
            self.assertEqual(Path(args.video_vae), (self.model / 'vae/diffusion_pytorch_model.safetensors').resolve())
            args._native_source_verifier()
            (self.model / 'model_index.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                args._native_source_verifier()
            legacy.assert_not_called()
        self.write_manifest()
        with patch('native_video.generate_video') as native:
            with self.assertRaises(SystemExit):
                generate_video.main(arguments[:-1] + [str(self.model / 'output.mp4')])
            native.assert_not_called()

    def test_nondefault_ltx_specific_controls_are_not_silently_ignored(self):
        import generate_video
        with patch('native_video.generate_video') as native:
            with self.assertRaises(SystemExit):
                generate_video.main(['--model', str(self.model), '--frames', '9', '--fps', '12',
                    '--decode-timestep', '0.4', '--output', str(self.root / 'result.mp4')])
            native.assert_not_called()

    def test_directory_rejects_explicit_component_overrides(self):
        import generate_video
        component = self.model / 'vae/diffusion_pytorch_model.safetensors'
        arguments = ['--model', str(self.model), '--frames', '9', '--fps', '12',
                     '--output', str(self.root / 'result.mp4')]
        with patch('native_video.generate_video') as native:
            for option in ('--text-encoder', '--video-vae', '--high-noise-model', '--vision-encoder'):
                with self.subTest(option=option), self.assertRaises(SystemExit):
                    generate_video.main(arguments + [option, str(component)])
            native.assert_not_called()

    def test_desktop_worker_entry_routes_the_same_directory(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('iild_wan_entry', ROOT / 'reference/generate.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with patch('native_video.generate_video') as native:
            self.assertEqual(entry.main(['--backend', 'video', '--model-path', str(self.model), '--width', '32',
                '--height', '32', '--frames', '9', '--fps', '12', '--decode-timestep', '0.05',
                '--decode-noise-scale', '0.025', '--image-cond-noise-scale', '0',
                '--output', str(self.root / 'result.mp4')]), 0)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(native.call_args.args[0]._native_original_configuration['model'], str(self.model))

    def test_text_only_directory_rejects_image_conditioning_before_engine(self):
        import generate_video
        image = self.root / 'keyframe.png'
        image.write_bytes(b'not decoded in this test')
        with patch('native_video.generate_video') as native:
            with self.assertRaises(SystemExit):
                generate_video.main(['--model', str(self.model), '--frames', '9', '--fps', '12',
                    '--first-frame', str(image), '--output', str(self.root / 'result.mp4')])
            native.assert_not_called()

    def test_oversized_directory_manifest_is_rejected_before_dispatch(self):
        import generate_video
        (self.model / 'model_index.json').write_bytes(b' ' * (1024 * 1024 + 1))
        with patch('native_video.generate_video') as native, patch('video_runtime.generate_video') as legacy:
            with self.assertRaises(SystemExit):
                generate_video.main(['--model', str(self.model), '--frames', '9', '--fps', '12',
                    '--output', str(self.root / 'result.mp4')])
            native.assert_not_called()
            legacy.assert_not_called()

    def test_packaging_python_api_uses_native_directory_inspector(self):
        fallback = ROOT / 'build' / ('iild-model-package.exe' if sys.platform == 'win32' else 'iild-model-package')
        tool = Path(os.environ.get('IILD_MODEL_PACKAGING_TOOL', str(fallback)))
        if not tool.is_file():
            self.skipTest('The ModelPackaging CLI is not built')
        sys.path.insert(0, str(ROOT / 'src/ModelPackaging/python'))
        self.addCleanup(sys.path.remove, str(ROOT / 'src/ModelPackaging/python'))
        import iild_model_packaging
        report = iild_model_packaging.inspect_wan_directory(self.model, executable=tool)
        self.assertEqual(report['model'], self.inspect()['model'])
        self.assertEqual(report['scheduler'], 'UniPCMultistepScheduler')

    def test_dual_expert_paths_reach_native_component_abi(self):
        from types import SimpleNamespace
        from native_video import Components, NativeVideoEngine, Request
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        self.manifest['transformer_2'] = ['diffusers', 'WanTransformer3DModel']
        self.manifest['boundary_ratio'] = 0.875
        self.write_manifest()
        low = self.weight('transformer_2', 'low.safetensors', 'patch_embedding.weight')
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--output', str(self.root / 'result.mp4')])))
        output = SimpleNamespace(frames=self.root / 'frames')
        output.frames.mkdir()
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        observed = {}
        def generate(request, components, progress, frame, user):
            req = C.cast(request, C.POINTER(Request)).contents
            values = C.cast(components, C.POINTER(Components)).contents
            observed.update(model=req.model.decode(), text_encoder=req.text_encoder.decode(),
                vae=values.video_vae.decode(), high=values.high_noise_model.decode(), boundary=values.moe_boundary)
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for index in range(9):
                frame(0, index, 32, 32, data, 32 * 32 * 3, None)
            return 1
        engine.generate_with_components = generate
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
        engine.free = lambda _: None
        fake_image = SimpleNamespace(frombytes=lambda *unused: object())
        with patch('native_video.provider.save_image', side_effect=lambda image, path, **kwargs: path.write_bytes(b'frame')):
            engine.render(args, output, fake_image, {})
        self.assertEqual(observed['model'], str(low.resolve()))
        self.assertEqual(observed['high'], str((self.model / 'transformer/diffusion_pytorch_model.safetensors').resolve()))
        self.assertEqual(observed['text_encoder'], str((self.model / 'text_encoder/model.safetensors').resolve()))
        self.assertEqual(observed['vae'], str((self.model / 'vae/diffusion_pytorch_model.safetensors').resolve()))
        self.assertEqual(observed['boundary'], 0.875)

    def test_report_keeps_original_request_and_manifest_changes_prevent_commit(self):
        import native_video
        from video_options import build_parser, resolve_options
        from wan_directory import prepare
        args = prepare(resolve_options(build_parser().parse_args(['--model', str(self.model), '--width', '32',
            '--height', '32', '--frames', '9', '--fps', '12', '--output', str(self.root / 'result.mp4')])))

        class Output:
            frames = self.root / 'frames'
            video = self.root / 'temporary.mp4'
            commit = Mock()
            def __enter__(self): return self
            def __exit__(self, *unused): return False

        output = Output()
        with patch('native_video.preflight_animation', return_value={'Image': object(), 'versions': {}}), \
             patch('native_video.preflight_interpolator'), patch('native_video.load_keyframes', return_value=({}, {})), \
             patch('native_video.AnimationOutput', return_value=output), patch('native_video.NativeVideoEngine') as engine, \
             patch('native_video.encode_video', return_value={}), patch('native_video.report_progress'):
            engine.return_value.render.return_value = ([{'index': 0}], {'frame_counts': [9]})
            report = native_video.generate_video(args)
            self.assertEqual(report['configuration']['model'], str(self.model))
            self.assertEqual(report['model']['directory_source']['declared_scheduler'], 'UniPCMultistepScheduler')
            self.assertEqual(report['model']['directory_source']['execution_scheduler'], 'native-engine-default')
            self.assertEqual(report['model']['directory_source']['components']['video_vae'],
                             str((self.model / 'vae/diffusion_pytorch_model.safetensors').resolve()))
            output.commit.reset_mock()
            def mutate(*unused):
                (self.model / 'model_index.json').write_text('{}')
                return ([{'index': 0}], {'frame_counts': [9]})
            engine.return_value.render.side_effect = mutate
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                native_video.generate_video(args)
            output.commit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
