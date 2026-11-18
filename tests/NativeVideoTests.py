"""Native file routing, timeline sampling, RGB ABI and transactional failures."""
import ctypes as C
import json
import os
import pickle
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from _file_provider import provider
from native_video import NativeVideoEngine, Request, Shot, Components, Progress, Frame, native_shot_plan, model_metadata
from video_options import build_parser, resolve_options


class NativeVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = provider.TemporaryDirectory(prefix="native-video-", dir=ROOT / "resources")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "renamed.safetensors"
        header = json.dumps({"__metadata__": {"model_version": "2.3.0"}}).encode()
        self.model.write_bytes(struct.pack("<Q", len(header)) + header)

    def options(self, *extra):
        return resolve_options(build_parser().parse_args([
            "--model", str(self.model), "--output", str(self.root / "video.mp4"),
            "--width", "32", "--height", "32", "--frames", "9", "--fps", "12", *extra]))

    def test_raw_package_is_a_local_video_input(self):
        args = self.options()
        self.assertEqual(args.model_input.kind, "local")
        self.assertEqual(args.model, str(self.model))
        self.assertIsNone(args.text_encoder)
        self.assertEqual(model_metadata(self.model)["model_version"], "2.3.0")

    def test_standalone_shard_index_is_a_validated_video_input(self):
        mapping = {}
        for name, filename in (('video.first', 'part-1.safetensors'), ('video.second', 'part-2.safetensors')):
            raw = json.dumps({name: {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
            (self.root / filename).write_bytes(struct.pack('<Q', len(raw)) + raw + b'x')
            mapping[name] = filename
        index = self.root / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': mapping}))
        args = resolve_options(build_parser().parse_args(['--model', str(index), '--output', str(self.root / 'video.mp4'),
            '--width', '32', '--height', '32', '--frames', '9', '--fps', '12']))
        self.assertEqual(args.model, str(index))
        report = model_metadata(index)
        self.assertEqual(len(report['shards']), 2)
        self.assertEqual(report['size_bytes'], index.stat().st_size + sum((self.root / name).stat().st_size
                                                                        for name in mapping.values()))

    def test_legacy_video_preparation_preserves_replay_and_rechecks_shards(self):
        import checkpoint_conversion
        import video_checkpoint_conversion
        shard = self.root / 'part.bin'
        shard.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture')
        index = self.root / 'torch-index.json'
        index.write_text(json.dumps({'weight_map': {'fixture.weight': shard.name}}))
        args = self.options('--model', str(index), '--text-encoder', str(index))
        original = checkpoint_conversion._identity(index)
        conversion = {'converted': True, 'converted_path': str(self.model),
                      'original': original, 'original_shards': [checkpoint_conversion._identity(shard)],
                      'output': checkpoint_conversion._identity(self.model)}
        with patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            video_checkpoint_conversion.prepare(args)
        self.assertEqual((args.model, args.text_encoder), (str(self.model), self.model))
        self.assertEqual(args._native_original_configuration['model'], str(index))
        video_checkpoint_conversion.verify(args)
        shard.write_bytes(shard.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            video_checkpoint_conversion.verify(args)

    def test_low_vram_flag_sets_execution_controls(self):
        args = self.options("--low-vram", "--no-cpu-text-encoding", "--no-vae-tiling")
        self.assertEqual(args.offload, "sequential")
        self.assertTrue(args.cpu_text_encoding)
        self.assertTrue(args.vae_tiling)

    def test_python_request_layout_reaches_the_real_cpp_adapter(self):
        library = Path(os.environ["IILD_NATIVE_LIBRARY"]) if os.environ.get("IILD_NATIVE_LIBRARY") else (
            ROOT / "build" / ("iiLocalDiffusion.dll" if sys.platform == "win32" else
                              "libiiLocalDiffusion.dylib" if sys.platform == "darwin" else "libiiLocalDiffusion.so"))
        if not library.is_file():
            self.skipTest("Build the native SDK to check the real C ABI")
        with patch("native_video.library_path", return_value=library):
            engine = NativeVideoEngine()
        shots = (Shot * 1)(Shot(b"A scene", b"noise", 42, 9, 24, None, None, 0, 0, 1, 0))
        request = Request(C.sizeof(Request), str(self.root / "missing.safetensors").encode(), None, None,
                          b"auto", b"auto", 32, 32, 2, 0, 3, 0, 1, 0, shots, 1)
        callback, observer = Progress(lambda *args: 0), Frame(lambda *args: 0)
        handle = engine.generate(C.byref(request), callback, observer, None)
        self.assertTrue(handle)
        try:
            result = json.loads(engine.metadata(handle))
        finally:
            engine.free(handle)
        # Size/layout validation passed; file validation stopped before loading.
        self.assertIn("local", result["error"])
        self.assertNotIn("ABI", result["error"])
        self.assertEqual(result["frame_counts"], [])
        self.assertIsNotNone(engine.generate_with_components)
        components = Components(C.sizeof(Components), None, None, None, -1, -1, -1)
        handle = engine.generate_with_components(C.byref(request), C.byref(components), callback, observer, None)
        try:
            result = json.loads(engine.metadata(handle))
        finally:
            engine.free(handle)
        self.assertIn("local", result["error"])
        self.assertNotIn("ABI", result["error"])
        components.size = 0
        handle = engine.generate_with_components(C.byref(request), C.byref(components), callback, observer, None)
        try:
            result = json.loads(engine.metadata(handle))
        finally:
            engine.free(handle)
        self.assertIn("components ABI", result["error"])

    def test_native_padding_preserves_source_frame_endpoints(self):
        args = self.options("--frames", "24", "--fps", "24")
        plan = native_shot_plan(args.shots[0], args.fps)
        self.assertEqual(plan["indices"][0], 0)
        self.assertEqual(plan["indices"][-1], plan["samples"] - 1)
        self.assertEqual(len(set(plan["indices"])), args.shots[0]["ltx_frames"])
        self.assertEqual(args.max_frames, 24)

    def test_unsupported_interior_keyframes_fail_before_materializing_weights(self):
        shot = {**self.options().shots[0], "conditions": [{"frame": 4, "image": "ignored", "strength": 1.0}]}
        with self.assertRaisesRegex(ValueError, "interior"):
            native_shot_plan(shot, 12)

    def test_native_rgb_frames_are_written_and_all_shots_verified(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required")
        args = self.options()
        output = SimpleNamespace(frames=self.root / "frames")
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        released = []
        def generate(request, progress, frame, user):
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for step in range(2048):
                progress(1, step, 2048, None)
            progress(3, 1, 2, None)
            for i in range(9):
                self.assertEqual(frame(0, i, 32, 32, data, 32 * 32 * 3, None), 0)
            return 1
        engine.generate = generate
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
        engine.free = released.append
        with patch("native_video.report_progress") as progress:
            frames, result = engine.render(args, output, Image, {})
        self.assertLess(progress.call_count, 20)
        self.assertEqual(len(frames), 9)
        self.assertEqual(released, [1])
        self.assertEqual(result["frame_counts"], [9])
        self.assertEqual(result["shot_sampling"][0]["native_frames"], 9)
        with Image.open(output.frames / "frame-000008.png") as image:
            self.assertEqual(image.getpixel((0, 0)), (42, 42, 42))

    def test_incomplete_native_result_is_never_accepted(self):
        from PIL import Image
        args = self.options()
        output = SimpleNamespace(frames=self.root / "frames")
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        engine.generate = lambda *args: 1
        engine.metadata = lambda _: b'{"error":"","frame_counts":[9]}'
        freed = []
        engine.free = freed.append
        with self.assertRaisesRegex(RuntimeError, "complete"):
            engine.render(args, output, Image, {})
        self.assertEqual(freed, [1])

    def test_sequential_offload_selects_low_vram_entrypoint(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required")
        args = self.options("--offload", "sequential")
        output = SimpleNamespace(frames=self.root / "frames")
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        engine.generate = lambda *unused: self.fail("Standard entry point was used for Low VRAM")
        calls = []
        def generate_low(request, progress, frame, user):
            calls.append(True)
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for index in range(9):
                frame(0, index, 32, 32, data, 32 * 32 * 3, None)
            return 1
        engine.generate_low_vram = generate_low
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9],"memory_mode":"low-vram"}'
        engine.free = lambda _: None
        _, result = engine.render(args, output, Image, {})
        self.assertEqual(calls, [True])
        self.assertEqual(result["memory_mode"], "low-vram")

    def test_old_runtime_rejects_low_vram_without_silent_fallback(self):
        args = self.options("--offload", "sequential")
        output = SimpleNamespace(frames=self.root / "frames")
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        engine.generate = lambda *unused: self.fail("Low VRAM must not silently use the standard entry point")
        engine.generate_low_vram = None
        with self.assertRaisesRegex(RuntimeError, "Low VRAM.*runtime"):
            engine.render(args, output, None, {})

    def test_native_components_use_extension_and_reject_old_runtime(self):
        from PIL import Image
        vae = self.root / 'vae.safetensors'
        high = self.root / 'high.safetensors'
        vision = self.root / 'vision.safetensors'
        for path in (vae, high, vision):
            path.write_bytes(self.model.read_bytes())
        for low in (False, True):
            args = self.options('--video-vae', str(vae), '--high-noise-model', str(high),
                                '--vision-encoder', str(vision), '--moe-boundary', '0.75',
                                '--high-noise-steps', '7', '--high-noise-guidance', '3.5',
                                *(['--low-vram'] if low else []))
            output = SimpleNamespace(frames=self.root / ('frames-low' if low else 'frames-normal'))
            provider.mkdir(output.frames)
            engine = NativeVideoEngine.__new__(NativeVideoEngine)
            engine.generate = lambda *unused: self.fail('Components must not be silently ignored')
            engine.generate_low_vram = engine.generate
            calls = []
            def generate(request, components, progress, frame, user):
                value = C.cast(components, C.POINTER(Components)).contents
                self.assertEqual(value.size, C.sizeof(Components))
                self.assertEqual((value.video_vae, value.high_noise_model, value.vision_encoder),
                                 tuple(str(path).encode() for path in (vae, high, vision)))
                self.assertEqual((value.moe_boundary, value.high_noise_steps, value.high_noise_guidance), (0.75, 7, 3.5))
                data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
                for index in range(9): frame(0, index, 32, 32, data, 32 * 32 * 3, None)
                calls.append(True)
                return 1
            engine.generate_with_components = generate if not low else None
            engine.generate_low_vram_with_components = generate if low else None
            engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
            engine.free = lambda _: None
            engine.render(args, output, Image, {})
            self.assertEqual(calls, [True])
            engine.generate_with_components = engine.generate_low_vram_with_components = None
            with self.assertRaisesRegex(RuntimeError, 'components.*runtime'):
                engine.render(args, output, Image, {})

    def test_packaged_expert_controls_reach_native_component_validation(self):
        from PIL import Image
        args = self.options('--moe-boundary', '0.75')
        output = SimpleNamespace(frames=self.root / 'frames-packaged-expert')
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        def generate(request, components, progress, frame, user):
            value = C.cast(components, C.POINTER(Components)).contents
            self.assertIsNone(value.high_noise_model)
            self.assertEqual(value.moe_boundary, 0.75)
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for index in range(9): frame(0, index, 32, 32, data, 32 * 32 * 3, None)
            return 1
        engine.generate_with_components = generate
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
        engine.free = lambda _: None
        engine.render(args, output, Image, {})

    def test_preparing_counts_and_completed_graph_batches_reach_worker_protocol(self):
        from PIL import Image
        args = self.options()
        output = SimpleNamespace(frames=self.root / "frames")
        provider.mkdir(output.frames)
        engine = NativeVideoEngine.__new__(NativeVideoEngine)
        def generate(request, progress, frame, user):
            progress(5, 0, 6963, None)
            progress(5, 6963, 6963, None)
            progress(6, 1, 0, None)
            data = C.create_string_buffer(bytes([42]) * (32 * 32 * 3))
            for index in range(9):
                frame(0, index, 32, 32, data, 32 * 32 * 3, None)
            return 1
        engine.generate = generate
        engine.metadata = lambda _: b'{"error":"","cancelled":false,"frame_counts":[9]}'
        engine.free = lambda _: None
        from contextlib import redirect_stdout
        from io import StringIO
        protocol = StringIO()
        with patch("native_video.report_progress") as progress, patch.dict("os.environ", {"IILD_WORKER_PROGRESS": "1"}), redirect_stdout(protocol):
            engine.render(args, output, Image, {})
        progress.assert_any_call("preparing-model", 6963, 6963)
        progress.assert_any_call("computing", 1, 0)
        events = [json.loads(line.removeprefix("IILD_NATIVE_PROGRESS ")) for line in protocol.getvalue().splitlines()]
        self.assertTrue(any(event["stage"] == "preparing-model" and event["step"] == 6963 for event in events))
        self.assertTrue(any(event["stage"] == "computing" and event["step"] == 1 and event["total"] == 0 for event in events))

    def test_file_worker_dispatch_does_not_call_the_legacy_directory_loader(self):
        import generate_video
        with patch("native_video.generate_video") as native, patch("video_runtime.generate_video") as legacy:
            self.assertEqual(generate_video.main(["--model", str(self.model), "--frames", "9", "--fps", "12",
                                                  "--output", str(self.root / "video.mp4")]), 0)
            native.assert_called_once()
            legacy.assert_not_called()

    def test_native_video_report_does_not_claim_an_ltx_architecture(self):
        import native_video
        args = self.options()
        class Output:
            frames = self.root / 'frames'
            video = self.root / 'temporary.mp4'
            def __enter__(self): return self
            def __exit__(self, *unused): return False
            def commit(self, report): self.report = report
        output = Output()
        with patch('native_video.preflight_animation', return_value={'Image': object(), 'versions': {}}), \
             patch('native_video.preflight_interpolator'), \
             patch('native_video.load_keyframes', return_value=({}, {})), \
             patch('native_video.AnimationOutput', return_value=output), \
             patch('native_video.NativeVideoEngine') as engine, \
             patch('native_video.encode_video', return_value={}), \
             patch('native_video.report_progress'):
            engine.return_value.render.return_value = ([{'index': 0}], {'frame_counts': [9]})
            report = native_video.generate_video(args)
        self.assertEqual(report['stages'][0]['name'], 'Native video')
        self.assertIs(output.report, report)

    def test_changed_converted_source_cannot_publish_a_video(self):
        import checkpoint_conversion
        import native_video
        args = self.options()
        original = self.root / 'legacy-source.bin'
        original.write_bytes(b'legacy fixture')
        args._native_conversions = {'model': {'original': checkpoint_conversion._identity(original),
            'converted_path': str(self.model), 'output': checkpoint_conversion._identity(self.model)}}
        class Output:
            frames = self.root / 'frames-changed'
            video = self.root / 'temporary-changed.mp4'
            report = None
            def __enter__(self): return self
            def __exit__(self, *unused): return False
            def commit(self, report): self.report = report
        output = Output()
        def render(*unused):
            original.write_bytes(b'changed legacy fixture')
            return [{'index': 0}], {'frame_counts': [9]}
        with patch('native_video.preflight_animation', return_value={'Image': object(), 'versions': {}}), \
             patch('native_video.preflight_interpolator'), patch('native_video.load_keyframes', return_value=({}, {})), \
             patch('native_video.AnimationOutput', return_value=output), patch('native_video.NativeVideoEngine') as engine, \
             patch('native_video.encode_video') as encode, patch('native_video.report_progress'):
            engine.return_value.render.side_effect = render
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                native_video.generate_video(args)
        self.assertIsNone(output.report)
        encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
