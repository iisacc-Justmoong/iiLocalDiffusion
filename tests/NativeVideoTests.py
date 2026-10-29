"""Native file routing, timeline sampling, RGB ABI and transactional failures."""
import ctypes as C
import json
import os
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
from native_video import NativeVideoEngine, Request, Shot, Progress, Frame, native_shot_plan, model_metadata
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


if __name__ == "__main__":
    unittest.main()
