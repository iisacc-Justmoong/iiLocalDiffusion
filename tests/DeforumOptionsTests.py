#!/usr/bin/env python3
"""Offline contracts for Deforum schedules, request validation and CLI routing."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from GenerationRouterTests import router
from deforum_schedules import NumericSchedule, PromptSchedule
from deforum_options import frame_request


class ScheduleTests(unittest.TestCase):
    def test_linear_interpolation_and_endpoint_hold(self):
        schedule = NumericSchedule("2:(1), 6:(3)", max_frames=9, fps=24)
        self.assertEqual([schedule.at(i) for i in range(9)], [1, 1, 1, 1.5, 2, 2.5, 3, 3, 3])

    def test_nested_math_and_frame_variables(self):
        schedule = NumericSchedule("0:(1 + 0.1*sin(2*pi*t/fps)), max_f:(2)", max_frames=25, fps=24)
        self.assertAlmostEqual(schedule.at(6), 1.1)
        self.assertEqual(schedule.at(24), 2)
        self.assertEqual(NumericSchedule("0:(max(1, cos(t)))", max_frames=3, fps=24).at(2), 1)

    def test_expressions_cannot_execute_python(self):
        for text in ("0:(__import__('os'))", "0:(t.real)", "0:([1][0])", "0:(2**1000000)",
                     "0:(float('inf'))", "0:(1/0)", "0:(True)", "0:(nan)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                NumericSchedule(text, max_frames=3, fps=24).values()

    def test_malformed_or_duplicate_keyframes_are_rejected(self):
        for text in ("", "garbage 0:(1)", "0:(1),", "0:(1),0:(2)", "-1:(2)",
                     "3:(2)", "1.5:(2)", "0:(1) trailing", "0:(1", "0:(1), 1+0:(2), 1:(3)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                NumericSchedule(text, max_frames=3, fps=24)

    def test_prompt_keyframes_are_held_and_sorted(self):
        schedule = PromptSchedule({"6": "forest", "0": "city"}, max_frames=8)
        self.assertEqual([schedule.at(i) for i in (0, 5, 6, 7)], ["city", "city", "forest", "forest"])
        for value in ({"1": "missing zero"}, {"0": 12}, {"0": "ok", "00": "duplicate"}, {"0": "a", "8": "b"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PromptSchedule(value, max_frames=8)


class DeforumOptionsTests(unittest.TestCase):
    def resolve(self, **values):
        return generate.resolve_request({"animation_mode": "2D", "max_frames": 5, **values})

    def test_defaults_and_movie_extension(self):
        preset, args = self.resolve()
        self.assertEqual((args.fps, args.seed_behavior, args.border), (24, "fixed", "replicate"))
        self.assertEqual(args.output.suffix, ".mp4")
        request = frame_request(preset, args, 1)
        self.assertEqual((request.zoom, request.angle, request.translation_x, request.translation_y), (1, 0, 0, 0))
        self.assertEqual(request.denoising_strength, 0.35)

    def test_prompt_schedule_reaches_both_sdxl_encoders_and_seeds(self):
        preset, args = self.resolve(preset="sdxl-base", seed=9, seed_behavior="iter", seed_stride=2,
                                    animation_prompts={"0": "city", "2": "forest"},
                                    animation_negative_prompts={"0": "blur", "3": "rain"},
                                    cfg_scale_schedule="0:(2), 4:(6)")
        request = frame_request(preset, args, 3)
        self.assertEqual((request.prompt, request.prompt_2), ("forest", "forest"))
        self.assertEqual((request.negative_prompt, request.negative_prompt_2), ("rain", "rain"))
        self.assertEqual((request.seed, request.guidance_scale), (15, 5))

    def test_configuration_roundtrip_preserves_scheduled_fallbacks(self):
        preset, args = self.resolve(preset="sdxl-base", animation_prompts={"0": "a", "2": "b"})
        values = generate.configuration_values(args)
        replay_preset, replay = generate.resolve_request(values)
        self.assertEqual(frame_request(replay_preset, replay, 3).prompt_2, "b")
        self.assertEqual(generate.configuration_values(replay), values)

    def test_random_seeds_are_reproducible_and_independent_of_access_order(self):
        preset, args = self.resolve(seed_behavior="random")
        seeds = [frame_request(preset, args, index).seed for index in (4, 0, 2, 4)]
        self.assertEqual(seeds[0], seeds[3])
        self.assertEqual(len(set(seeds)), 3)

    def test_invalid_and_incompatible_requests_fail_before_model_loading(self):
        cases = [dict(max_frames=0), dict(fps=0), dict(fps=1e-15), dict(fps=float("nan")), dict(num_images=2),
                 dict(hires_fix=True), dict(output="bad.png"), dict(strength_schedule="0:(1.1)"),
                 dict(strength_schedule="0:(0.01)", steps=2), dict(zoom="0:(0)"),
                 dict(noise_schedule="0:(-1)"), dict(contrast_schedule="0:(-1)"),
                 dict(seed_behavior="iter", seed=2**64-1), dict(video_crf=52),
                 dict(preset="flux1-schnell", cfg_scale_schedule="0:(7)"),
                 dict(preset="flux1-schnell", animation_negative_prompts={"0": "blur"}),
                 dict(guidance_rescale=0.1),
                 dict(timesteps=[999, 500, 0]), dict(denoising_end=0.5, preset="sdxl-base")]
        for values in cases:
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve(**values)

    def test_animation_options_cannot_be_silently_ignored_for_images(self):
        with self.assertRaises(SystemExit):
            generate.resolve_request({"max_frames": 5})

    def test_configured_animation_wins_over_local_checkpoint_auto_route(self):
        import argparse
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="deforum-route-") as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"animation_mode":"2D"}')
            model = Path(directory) / "model.safetensors"
            model.write_bytes(b"fixture")
            args = argparse.Namespace(backend="auto", base_model=None)
            flags = ["--model", str(model), "--config", str(path)]
            self.assertEqual(router.select_backend(args, flags), "deforum")
            self.assertEqual(router.select_backend(args, [*flags, "--animation-mode", "none"]), "local")

    def test_router_and_json_configuration_are_offline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="deforum-options-") as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"animation_mode": "2D", "max_frames": 3, "fps": 12,
                                        "animation_prompts": {"0": "a city"}}))
            for flags in (["--backend", "deforum"], ["--animation-mode", "2D"]):
                result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"),
                                         *flags, "--config", str(path), "--fps", "8", "--print-config"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = json.loads(result.stdout)
                self.assertEqual((values["animation_mode"], values["fps"]), ("2D", 8))


if __name__ == "__main__":
    unittest.main()
