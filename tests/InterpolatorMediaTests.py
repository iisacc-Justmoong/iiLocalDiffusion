#!/usr/bin/env python3
"""Interpolator video publication without the optional Deforum/OpenCV runtime."""

import builtins
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from local_model_fixture import local_request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from animation_video import AnimationOutput, SCHEMAS, encode_video, output_targets, preflight_animation


class InterpolatorPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="interpolator-media-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        _, self.args = local_request({"animation_mode": "Interpolator", "max_frames": 3,
                                                 "fps": 12, "width": 32, "height": 32,
                                                 "output": str(self.directory / "movie.mp4")})

    def stage(self, output):
        output.video.write_bytes(b"fixture")
        (output.frames / "frame-000000.png").write_bytes(b"frame")

    def test_marker_and_rollback_keep_the_previous_interpolator_bundle(self):
        with AnimationOutput(self.args) as output:
            self.stage(output)
            output.commit({"status": "old"})
        video, report, frames = output_targets(self.args.output)
        self.assertEqual(json.loads((frames / "manifest.json").read_text())["schema"], SCHEMAS["Interpolator"])
        self.args.overwrite = True
        with self.assertRaises(OSError):
            with AnimationOutput(self.args) as output:
                self.stage(output)
                with patch("animation_video.publish_file", side_effect=OSError("disk full")):
                    output.commit({"status": "new"})
        self.assertEqual(json.loads(report.read_text())["status"], "old")
        self.assertEqual(video.read_bytes(), b"fixture")
        self.assertFalse(list(self.directory.glob(".*.lock")))

    def test_deforum_and_interpolator_share_an_exclusive_output_lock(self):
        with AnimationOutput(self.args):
            self.args.animation_mode = "2D"
            with self.assertRaises(FileExistsError):
                with AnimationOutput(self.args):
                    pass
        self.assertEqual(list(self.directory.iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("PIL") and shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "Pillow and FFmpeg runtime are required")
    def test_real_verified_mp4_without_opencv_import(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ("cv2", "deforum_video", "deforum_runtime"):
                raise AssertionError("Interpolator must not depend on OpenCV/Deforum")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            environment = preflight_animation(self.args)
        with AnimationOutput(self.args) as output:
            for index in range(3):
                image = environment["Image"].new("RGB", (32, 32), (index * 50, 128, 64))
                image.save(output.frames / f"frame-{index:06d}.png")
            video = encode_video(output.frames, output.video, self.args, environment)
            output.commit({"status": "complete", "output": video})
        self.assertEqual((video["frame_count"], video["fps"], video["duration_seconds"]), (3, 12, .25))
        self.assertTrue(video["verified_decode"])

    @unittest.skipUnless(importlib.util.find_spec("PIL") and shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "Pillow and FFmpeg runtime are required")
    def test_gif_preserves_frames_and_quantized_timing_above_12_fps(self):
        for mode, fps, count in (("2D", 24, 7), ("Interpolator", 30, 6),
                                 ("Interpolator", 100, 3), ("2D", 12, 3), ("2D", 24, 1)):
            with self.subTest(mode=mode, fps=fps):
                _, args = local_request({"animation_mode": mode, "fps": fps, "max_frames": count,
                                         "width": 32, "height": 32,
                                         "output": str(self.directory / f"{mode}-{fps}-{count}.GIF")})
                environment = preflight_animation(args)
                with AnimationOutput(args) as output:
                    self.assertEqual(output.video.suffix, ".gif")
                    for index in range(count):
                        image = environment["Image"].new("RGB", (32, 32), (0 if fps == 100 else index * 30, 128, 64))
                        image.save(output.frames / f"frame-{index:06d}.png")
                    video = encode_video(output.frames, output.video, args, environment)
                    output.commit({"status": "complete", "output": video})
                self.assertEqual((video["codec"], video["frame_count"]), ("gif", count))
                self.assertTrue(video["verified_decode"])
                self.assertAlmostEqual(video["duration_seconds"], count / fps, delta=.01)
                with environment["Image"].open(args.output) as image:
                    self.assertEqual((image.format, image.n_frames, image.info["loop"]), ("GIF", count, 0))

    @unittest.skipUnless(importlib.util.find_spec("PIL") and shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "Pillow and FFmpeg runtime are required")
    def test_truncated_gif_frames_are_rejected(self):
        self.args.output = self.directory / "truncated.gif"
        environment = preflight_animation(self.args)
        with AnimationOutput(self.args) as output:
            environment["Image"].new("RGB", (32, 32)).save(output.frames / "frame-000000.png")
            with self.assertRaisesRegex(RuntimeError, "frame count"):
                encode_video(output.frames, output.video, self.args, environment)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
