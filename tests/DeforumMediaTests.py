#!/usr/bin/env python3
"""Real OpenCV pixels and FFmpeg encoding; run with the optional media runtime."""

import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from local_model_fixture import local_request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from deforum_options import frame_request
from deforum_video import encode_video, preflight_animation, prepare_frame

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "PIL"))


@unittest.skipUnless(AVAILABLE, "Optional OpenCV/NumPy/Pillow runtime is not installed in this interpreter")
class DeforumMediaTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        from PIL import Image
        self.np = np
        self.environment = {"cv2": cv2, "np": np, "Image": Image}
        self.preset, self.args = local_request({"animation_mode": "2D", "max_frames": 4,
                                                           "fps": 8, "width": 32, "height": 32})
        grid = np.zeros((32, 32, 3), dtype=np.uint8)
        grid[6:14, 6:14] = [255, 128, 64]
        self.image = Image.fromarray(grid)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="deforum-media-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def transform(self, **values):
        request = frame_request(self.preset, self.args, 1)
        for name, value in values.items():
            setattr(request, name, value)
        return prepare_frame(self.image, None, request, self.environment, warp=True)

    def test_identity_and_translation_have_expected_pixels(self):
        self.assertEqual(self.transform().tobytes(), self.image.tobytes())
        pixels = self.np.asarray(self.transform(translation_x=4, translation_y=2))
        self.assertEqual(pixels[8, 10].tolist(), [255, 128, 64])
        self.assertEqual(pixels[6, 6].tolist(), [0, 0, 0])

    def test_seeded_noise_does_not_modify_global_numpy_rng(self):
        state = self.np.random.get_state()
        left = self.transform(noise_schedule=0.05)
        right = self.transform(noise_schedule=0.05)
        self.assertEqual(left.tobytes(), right.tobytes())
        self.assertNotEqual(left.tobytes(), self.image.tobytes())
        self.np.testing.assert_array_equal(state[1], self.np.random.get_state()[1])

    def test_wrap_rotation_zoom_and_rgb_coherence_are_finite_and_sized(self):
        for border in ("replicate", "reflect", "wrap"):
            request = frame_request(self.preset, self.args, 1)
            request.border, request.angle, request.zoom = border, 15, 1.2
            request.color_coherence = "RGB"
            transformed = prepare_frame(self.image, self.image, request, self.environment, warp=True)
            self.assertEqual(transformed.size, (32, 32))
            self.assertTrue(self.np.isfinite(self.np.asarray(transformed)).all())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_real_mp4_has_exact_frames_size_and_rate(self):
        self.args.output = self.directory / "movie.mp4"
        environment = preflight_animation(self.args)
        frames = self.directory / "frames"
        frames.mkdir()
        for index in range(4):
            self.transform(translation_x=index).save(frames / f"frame-{index:06d}.png")
        metadata = encode_video(frames, self.args.output, self.args, environment)
        self.assertEqual((metadata["frame_count"], metadata["fps"], metadata["size"]), (4, 8, [32, 32]))
        self.assertEqual(metadata["duration_seconds"], 0.5)
        self.assertTrue(metadata["verified_decode"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_truncated_frame_sequence_is_rejected(self):
        self.args.output = self.directory / "movie.mp4"
        environment = preflight_animation(self.args)
        self.image.save(self.directory / "frame-000000.png")
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            encode_video(self.directory, self.args.output, self.args, environment)


if __name__ == "__main__":
    unittest.main()
