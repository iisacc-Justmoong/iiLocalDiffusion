#!/usr/bin/env python3
"""LTX source timelines and the second-stage motion interpolator."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_model_fixture import local_parser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from video_options import build_parser, configuration, resolve_options


class VideoInterpolationPlanTests(unittest.TestCase):
    def request(self, *tokens):
        return resolve_options(local_parser(build_parser).parse_args(list(tokens)))

    def test_default_two_stages_keep_final_duration_and_reduce_ltx_frames(self):
        args = self.request()
        shot = args.shots[0]
        self.assertTrue(args.interpolation_enabled)
        self.assertEqual((args.fps, args.interpolation_factor, args.max_frames), (24, 2, 120))
        self.assertEqual((shot["ltx_frames"], shot["sample_frames"]), (61, 65))
        self.assertEqual(shot["source_positions"], [*range(0, 119, 2), 119])
        self.assertAlmostEqual((shot["ltx_frames"] - 1) / shot["sampling_fps"], 119 / 24)

    def test_low_fps_ltx_has_no_interpolator_stage(self):
        for fps in (8, 12):
            args = self.request("--fps", str(fps), "--frames", "9")
            self.assertFalse(args.interpolation_enabled)
            self.assertEqual(args.shots[0]["source_positions"], list(range(9)))
            self.assertEqual(args.shots[0]["sampling_fps"], fps)

    def test_keyframes_are_exact_source_anchors_even_between_regular_samples(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="interpolation-plan-") as temporary:
            image = Path(temporary) / "keyframe.png"
            image.write_bytes(b"decoded at runtime")
            storyboard = Path(temporary) / "shots.json"
            storyboard.write_text(json.dumps({"shots": [{"frames": 10, "conditions": [
                {"image": "keyframe.png", "frame": 3}, {"image": "keyframe.png", "frame": 4}]}]}))
            args = self.request("--storyboard", str(storyboard), "--interpolation-factor", "3")
            shot = args.shots[0]
            self.assertEqual(shot["source_positions"], [0, 3, 4, 6, 9])
            self.assertEqual([c["frame"] for c in shot["conditions"]], [3, 4])
            self.assertEqual([c["source_frame"] for c in shot["conditions"]], [1, 2])

    def test_factor_and_fractional_fps_are_replayable(self):
        args = self.request("--fps", "29.97", "--frames", "17", "--interpolation-factor", "4")
        self.assertEqual(args.shots[0]["source_positions"], [0, 4, 8, 12, 16])
        replay = resolve_options(local_parser(build_parser).parse_values(configuration(args)))
        self.assertEqual(replay.shots, args.shots)
        for factor in (0, 1, 9):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                self.request("--interpolation-factor", str(factor))


AVAILABLE = importlib.util.find_spec("PIL") and shutil.which("ffmpeg") and shutil.which("ffprobe")


@unittest.skipUnless(AVAILABLE, "Pillow and FFmpeg/FFprobe are required")
class VideoInterpolationMediaTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="video-interpolation-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def request(self, *tokens):
        return resolve_options(local_parser(build_parser).parse_args([
            "--frames", "9", "--width", "96", "--height", "64", "--output", str(self.directory / "movie.mp4"),
            *tokens]))

    def sources(self, args, output, *, cut=False):
        from weight_files import file_sha256
        records = []
        for shot in args.shots:
            folder = output.frames / "ltx" / f"shot-{shot['index']:04d}"
            folder.mkdir(parents=True)
            for index, position in enumerate(shot["source_positions"]):
                image = self.Image.new("RGB", (args.width, args.height),
                                       (255, 0, 0) if shot["index"] == 0 else (0, 0, 255)) if cut else self.Image.new("RGB", (args.width, args.height))
                if not cut:
                    self.ImageDraw.Draw(image).rectangle((16 + 2 * position, 20, 31 + 2 * position, 35), fill="white")
                path = folder / f"frame-{index:06d}.png"
                image.save(path)
                records.append({"index": shot["start_frame"] + position, "shot": shot["index"],
                                "file": str(path.relative_to(output.frames)), "sha256": file_sha256(path)})
        return records

    def test_real_motion_interpolation_keeps_anchors_and_inserts_distinct_frames(self):
        from animation_video import AnimationOutput, preflight_animation, encode_video
        from video_interpolator import interpolate_video, preflight_interpolator
        from weight_files import file_sha256
        args = self.request()
        environment = preflight_animation(args)
        preflight_interpolator(args, environment)
        with AnimationOutput(args) as output:
            sources = self.sources(args, output)
            frames, stage = interpolate_video(args, args.shots, sources, output, environment)
            self.assertEqual((len(frames), stage["inserted_frames"]), (9, 4))
            self.assertEqual(stage["engine"], "ffmpeg-minterpolate")
            self.assertTrue(stage["verified"])
            for source in sources:
                self.assertEqual(file_sha256(output.frames / f"frame-{source['index']:06d}.png"), source["sha256"])
            middle = (output.frames / "frame-000001.png").read_bytes()
            self.assertNotEqual(middle, (output.frames / sources[0]["file"]).read_bytes())
            self.assertNotEqual(middle, (output.frames / sources[1]["file"]).read_bytes())
            with environment["Image"].open(output.frames / "frame-000001.png") as image:
                weights = [(x, image.getpixel((x, y))[0]) for y in range(args.height) for x in range(args.width)]
                center = sum(x * value for x, value in weights) / sum(value for _, value in weights)
                self.assertAlmostEqual(center, 25.5, delta=1)
            video = encode_video(output.frames, output.video, args, environment)
            self.assertEqual((video["frame_count"], video["fps"]), (9, 24))
            output.commit({"status": "complete", "interpolation": stage, "video": video})
        self.assertTrue((args.output.with_name("movie-frames") / sources[0]["file"]).exists())

    def test_short_sequences_and_fractional_fps_keep_exact_output_count(self):
        from animation_video import AnimationOutput, preflight_animation
        from video_interpolator import interpolate_video
        for count in (2, 3, 8):
            with self.subTest(count=count):
                args = self.request("--frames", str(count), "--fps", "29.97", "--interpolation-factor", "3")
                environment = preflight_animation(args)
                with AnimationOutput(args) as output:
                    sources = self.sources(args, output)
                    frames, _ = interpolate_video(args, args.shots, sources, output, environment)
                    self.assertEqual([f["index"] for f in frames], list(range(count)))

    def test_shot_cuts_are_not_blended_by_interpolation(self):
        from animation_video import AnimationOutput, preflight_animation
        from video_interpolator import interpolate_video
        storyboard = self.directory / "storyboard.json"
        storyboard.write_text(json.dumps({"shots": [{"frames": 5}, {"frames": 6}]}))
        args = self.request("--storyboard", str(storyboard))
        environment = preflight_animation(args)
        with AnimationOutput(args) as output:
            sources = self.sources(args, output, cut=True)
            frames, stage = interpolate_video(args, args.shots, sources, output, environment)
            self.assertEqual(stage["shot_boundaries"], "preserved-no-cross-shot-interpolation")
            self.assertEqual(len(frames), 11)
            for frame in frames:
                with self.Image.open(output.frames / frame["file"]) as image:
                    r, _, b = image.getpixel((0, 0))
                    self.assertTrue(r > 240 and b < 10 if frame["shot"] == 0 else b > 240 and r < 10)

    def test_unavailable_filter_is_an_explicit_preflight_failure(self):
        from video_interpolator import preflight_interpolator
        with patch("video_interpolator.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="Unknown filter", stderr="")):
            with self.assertRaisesRegex(ValueError, "minterpolate"):
                preflight_interpolator(self.request(), {"ffmpeg": "ffmpeg"})

    def test_failed_second_stage_keeps_the_previous_bundle(self):
        from animation_video import AnimationOutput, preflight_animation
        from video_interpolator import interpolate_video
        args = self.request()
        with AnimationOutput(args) as output:
            output.video.write_bytes(b"previous")
            output.commit({"status": "previous"})
        args.overwrite = True
        environment = preflight_animation(args)
        with self.assertRaises(RuntimeError):
            with AnimationOutput(args) as output:
                sources = self.sources(args, output)
                with patch("video_interpolator.subprocess.run", return_value=SimpleNamespace(returncode=1, stderr="interpolation failed")):
                    interpolate_video(args, args.shots, sources, output, environment)
        self.assertEqual(args.output.read_bytes(), b"previous")
        self.assertEqual(json.loads(args.output.with_suffix(".json").read_text())["status"], "previous")


if __name__ == "__main__":
    unittest.main()
