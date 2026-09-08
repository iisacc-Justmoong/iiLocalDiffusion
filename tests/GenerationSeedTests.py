#!/usr/bin/env python3
"""Per-request random defaults and explicit seed replay across generation routes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
import local_image
import video_options
from generation_config import configuration_values
from local_model_fixture import local_parser, local_request, MODEL
from ComfyUIImageWorkflowTests import object_info, classes
from comfyui_image_workflow import build_workflow


class GenerationSeedTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="generation-seed-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_reused_image_parser_resolves_fresh_seed_per_request(self):
        parser = local_parser(generate.build_parser)
        values = parser.parse_args([])
        self.assertIsNone(values.seed)
        with patch("secrets.randbits", side_effect=[101, 202]) as random:
            _, first = generate.resolve_arguments(values)
            _, second = generate.resolve_arguments(values)
        self.assertEqual((first.seed, second.seed), (101, 202))
        self.assertEqual(random.call_count, 2)
        self.assertIsNone(values.seed)

    def test_image_configuration_records_seed_and_replays_without_randomizing(self):
        with patch("secrets.randbits", return_value=314159) as random:
            _, args = local_request({"seed": None, "num_images": 3, "hires_fix": True})
        random.assert_called_once_with(32)
        self.assertEqual(args.hires_seed, args.seed)
        recorded = configuration_values(args)
        self.assertEqual(recorded["seed"], 314159)
        with patch("secrets.randbits", side_effect=AssertionError("Explicit seeds must be retained")):
            _, replay = generate.resolve_request(recorded)
        self.assertEqual((replay.seed, replay.seed_stride, replay.num_images), (314159, 1, 3))

    def test_explicit_zero_and_config_cli_precedence_preserve_seed(self):
        path = self.directory / "request.json"
        path.write_text(json.dumps({"model": MODEL, "seed": 42}))
        with patch("secrets.randbits", side_effect=AssertionError("Explicit seeds must be retained")):
            _, configured = generate.resolve_arguments(generate.build_parser().parse_args(["--config", str(path)]))
            _, override = generate.resolve_arguments(generate.build_parser().parse_args([
                "--config", str(path), "--seed", "0"]))
        self.assertEqual((configured.seed, override.seed), (42, 0))

    def test_animation_uses_one_random_base_seed_and_keeps_frame_policy(self):
        from deforum_options import frame_request
        with patch("secrets.randbits", return_value=77) as random:
            preset, args = local_request({"animation_mode": "2D", "max_frames": 3})
            seeds = [frame_request(preset, args, index).seed for index in range(3)]
        random.assert_called_once_with(32)
        self.assertEqual(seeds, [77, 77, 77])
        with patch("secrets.randbits", return_value=88) as random:
            _, args = local_request({"animation_mode": "Interpolator", "max_frames": 3})
        random.assert_called_once_with(32)
        self.assertEqual((args.seed, args.end_seed), (88, 88))

    def test_temporal_video_randomizes_base_and_preserves_explicit_shot_seeds(self):
        storyboard = self.directory / "shots.json"
        storyboard.write_text(json.dumps({"shots": [{}, {}, {"seed": 0}]}))
        parser = local_parser(video_options.build_parser)
        with patch("secrets.randbits", side_effect=[901, 902]) as random:
            first = video_options.resolve_options(parser.parse_args(["--storyboard", str(storyboard)]))
            second = video_options.resolve_options(parser.parse_args([]))
        self.assertEqual(random.call_count, 2)
        self.assertEqual([shot["seed"] for shot in first.shots], [901, 902, 0])
        self.assertEqual(video_options.configuration(first)["seed"], 901)
        self.assertEqual(second.seed, 902)
        with patch("secrets.randbits", side_effect=AssertionError("Explicit video seed")):
            explicit = video_options.resolve_options(parser.parse_args(["--seed", "0"]))
        self.assertEqual(explicit.shots[0]["seed"], 0)

    def test_remote_image_seed_is_resolved_without_calling_an_endpoint(self):
        from remote_generation import image_configuration
        with patch("secrets.randbits", side_effect=[11, 22]):
            _, first = generate.resolve_request({"model_api": "http://127.0.0.1:9000/infer"})
            _, second = generate.resolve_request({"model_api": "http://127.0.0.1:9000/infer"})
        self.assertEqual((image_configuration(first)["seed"], second.seed), (11, 22))

    def test_managed_image_records_resolved_seed_before_starting_runtime(self):
        model = self.directory / "model.safetensors"
        model.write_bytes(b"fixture")
        parser = local_image.build_parser()
        inspection = {"role": "checkpoint", "weights_role": "checkpoint", "architecture": "sdxl"}
        with patch("downloaded_model.inspect_downloaded_model", return_value=inspection), \
                patch("secrets.randbits", side_effect=[33, 44]) as random:
            first = local_image.resolved_request(parser.parse_args(["--model", str(model)]))
            second = local_image.resolved_request(parser.parse_args(["--model", str(model)]))
            explicit = local_image.resolved_request(parser.parse_args(["--model", str(model), "--seed", "0"]))
        self.assertEqual(random.call_count, 2)
        self.assertEqual((first["seed"], second["seed"], explicit["seed"]), (33, 44, 0))

    def test_workflow_builder_randomizes_once_and_reuses_seed_for_refinement(self):
        with patch("secrets.randbits", side_effect=[55, 66]) as random:
            first = build_workflow("SDXL 1.0", "checkpoint.safetensors", {}, "prompt", object_info(),
                                   hires_fix=True, hires_passes=2)
            second = build_workflow("SDXL 1.0", "checkpoint.safetensors", {}, "prompt", object_info())
            explicit = build_workflow("SDXL 1.0", "checkpoint.safetensors", {}, "prompt", object_info(), seed=0)
        self.assertEqual(random.call_count, 2)
        self.assertEqual(classes(first, "KSampler")[0][1]["seed"], 55)
        self.assertEqual([inputs["noise_seed"] for _, inputs in classes(first, "RandomNoise")], [55, 55])
        self.assertEqual(classes(second, "KSampler")[0][1]["seed"], 66)
        self.assertEqual(classes(explicit, "KSampler")[0][1]["seed"], 0)


if __name__ == "__main__":
    unittest.main()
