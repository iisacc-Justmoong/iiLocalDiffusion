#!/usr/bin/env python3
"""Cinematic video planning and front-door contracts, without model downloads."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from local_model_fixture import local_parser, MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from video_options import build_parser, resolve_options, configuration, CAMERA_MOTIONS


class VideoOptionsTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="video-options-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def request(self, *tokens):
        return resolve_options(local_parser(build_parser).parse_args(list(tokens)))

    def document(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data))
        return path

    def test_defaults_make_a_five_second_temporal_generation_plan(self):
        args = self.request()
        self.assertEqual(args.max_frames, 120)
        self.assertEqual(args.shots[0]["sample_frames"], 65)
        self.assertEqual(args.shots[0]["ltx_frames"], 61)
        self.assertEqual(args.shots[0]["frames"], 120)
        self.assertEqual(args.fps, 24)
        self.assertEqual(args.animation_mode, "Video")
        self.assertIsNone(args.revision)
        self.assertTrue(args.local_files_only)
        self.assertEqual(args.shots[0]["camera"], ["none"])

    def test_duration_and_camera_mix_are_compiled_into_explicit_shots(self):
        args = self.request("--duration", "3", "--fps", "25", "--camera", "dolly-in", "pan-left")
        shot = args.shots[0]
        self.assertEqual((shot["frames"], shot["sample_frames"]), (75, 41))
        self.assertIn("forward", shot["effective_prompt"])
        self.assertIn("left", shot["effective_prompt"])
        self.assertEqual(len(CAMERA_MOTIONS), len(set(CAMERA_MOTIONS)))

    def test_invalid_parameters_fail_before_loading_runtime(self):
        cases = [("--duration", "nan"), ("--duration", "0"), ("--fps", "0"),
                 ("--frames", "1"), ("--frames", "-2"), ("--width", "321"),
                 ("--height", "0"), ("--steps", "0"), ("--guidance-scale", "nan"),
                 ("--seed", "-1"), ("--frames", "9", "--duration", "1"),
                 ("--camera", "static", "dolly-in"), ("--camera", "none", "pan-left"),
                 ("--camera", "pan-left", "pan-right"), ("--output", "bad.png"),
                 ("--device", "cpu", "--offload", "model"),
                 ("--model", "owner/model"), ("--revision", "main")]
        for tokens in cases:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.request(*tokens)

    def test_storyboard_has_exact_cut_positions_and_relative_image_paths(self):
        image = self.directory / "keyframe.png"
        image.write_bytes(b"image is decoded during media preflight")
        storyboard = self.document("shots.json", {"shots": [
            {"prompt": "A glass bottle on a table.", "frames": 9, "first_frame": "keyframe.png"},
            {"prompt": "The bottle turns.", "frames": 17, "continue_previous": True,
             "camera": ["orbit-left"]}]})
        args = self.request("--storyboard", str(storyboard), "--seed", "7")
        self.assertEqual(args.max_frames, 26)
        self.assertEqual([s["start_frame"] for s in args.shots], [0, 9])
        self.assertEqual([s["seed"] for s in args.shots], [7, 8])
        self.assertEqual(args.shots[0]["conditions"][0]["image"], str(image.resolve()))
        self.assertTrue(args.shots[1]["continue_previous"])

    def test_first_and_last_keyframes_condition_the_output_timeline(self):
        image = self.directory / "image.png"
        image.write_bytes(b"fixture")
        args = self.request("--frames", "16", "--first-frame", str(image), "--last-frame", str(image))
        self.assertEqual([c["frame"] for c in args.shots[0]["conditions"]], [0, 15])
        self.assertEqual(args.shots[0]["sample_frames"], 9)

    def test_storyboard_rejects_ambiguous_or_unusable_conditions(self):
        bad = [{"shots": []}, {"shots": [{"prompt": "x", "unknown": 1}]},
               {"shots": [{"prompt": "x", "continue_previous": True}]},
               {"shots": [{"prompt": "x", "frames": True}]},
               {"shots": [{"prompt": "x", "camera": "orbit-left"}]},
               {"shots": [{"prompt": "x", "conditions": [{"image": "missing.png", "frame": 0}]}]}]
        for data in bad:
            path = self.document("bad.json", data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.request("--storyboard", str(path))

    def test_config_cli_override_and_replay_preserve_plan(self):
        config = self.document("request.json", {"backend": "video", "prompt": "A river flows.",
                                                "duration": 2, "camera": ["pan-left"],
                                                "output": "river.mp4"})
        args = self.request("--config", str(config), "--prompt", "A waterfall flows.")
        self.assertEqual(args.prompt, "A waterfall flows.")
        self.assertEqual(args.output, self.directory / "river.mp4")
        replay = self.document("replay.json", configuration(args))
        self.assertEqual(args.shots, self.request("--config", str(replay)).shots)

    def test_frontdoor_video_and_json_routes_work_without_ml_imports(self):
        config = self.document("video.json", {"backend": "video", "frames": 9})
        for tokens in (["--backend", "video"], ["--config", str(config)]):
            result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--model", MODEL,
                                     *tokens, "--print-config"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["backend"], "video")

    def test_output_cannot_destroy_its_storyboard(self):
        path = self.document("movie.json", {"shots": [{"frames": 9}]})
        with self.assertRaisesRegex(ValueError, "must not replace"):
            self.request("--storyboard", str(path), "--output", str(path.with_suffix(".mp4")), "--overwrite")


if __name__ == "__main__":
    unittest.main()
