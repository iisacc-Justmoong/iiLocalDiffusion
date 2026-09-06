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
        _, self.args = generate.resolve_request({"animation_mode": "Interpolator", "max_frames": 3,
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


if __name__ == "__main__":
    unittest.main()
