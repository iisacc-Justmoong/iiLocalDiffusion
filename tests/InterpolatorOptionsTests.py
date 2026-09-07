#!/usr/bin/env python3
"""Offline Interpolator request, configuration and routing contracts."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from local_model_fixture import local_request, MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from GenerationRouterTests import router


class InterpolatorOptionsTests(unittest.TestCase):
    def resolve(self, **values):
        return local_request({"animation_mode": "Interpolator", "max_frames": 5, **values})

    def test_defaults_preserve_start_values_and_use_video_output(self):
        _, args = self.resolve(prompt="city", negative_prompt="blur", seed=17)
        self.assertEqual((args.end_prompt, args.end_negative_prompt, args.end_seed), ("city", "blur", 17))
        self.assertTrue(args.cpu_text_encoding)
        self.assertEqual((args.fps, args.output.suffix), (12, ".mp4"))
        self.assertIn("interpolator", args.output.stem)
        self.assertIsNone(args.zoom)

    def test_secondary_end_prompts_follow_explicit_or_primary_values(self):
        _, args = self.resolve(preset="sdxl-base", prompt="city", end_prompt="forest",
                               negative_prompt="rain", end_negative_prompt="blur")
        self.assertEqual((args.end_prompt_2, args.end_negative_prompt_2), ("forest", "blur"))
        _, args = self.resolve(preset="sdxl-base", prompt_2="detail", negative_prompt_2="grain",
                               end_prompt="forest", end_prompt_2="leaves", end_negative_prompt_2="")
        self.assertEqual((args.end_prompt_2, args.end_negative_prompt_2), ("leaves", ""))
        _, args = self.resolve(preset="sdxl-base", prompt_2="detail", end_prompt="forest")
        self.assertEqual(args.end_prompt_2, "detail")

    def test_configuration_roundtrip_preserves_all_endpoints(self):
        _, args = self.resolve(preset="sdxl-base", end_prompt="forest", end_negative_prompt="",
                               end_seed=2**64-1)
        values = generate.configuration_values(args)
        _, replay = local_request(values)
        self.assertEqual(generate.configuration_values(replay), values)

    def test_incompatible_requests_fail_without_runtime(self):
        cases = [dict(max_frames=1), dict(fps=0), dict(fps=1e-15), dict(video_crf=52),
                 dict(encoding_timeout=0), dict(num_images=2), dict(hires_fix=True),
                 dict(end_seed=2**64), dict(end_seed=-(2**63)-1), dict(end_prompt_2="detail"),
                 dict(output="movie.png"), dict(cpu_text_encoding=False), dict(zoom="1.1"),
                 dict(animation_prompts={"0": "a"}), dict(seed_behavior="iter"),
                 dict(preset="flux1-schnell", end_negative_prompt="blur"),
                 dict(preset="sdxl-base", denoising_end=0.5)]
        for values in cases:
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve(**values)

    def test_endpoints_cannot_be_silently_ignored_in_other_modes(self):
        for values in ({"end_prompt": "b"}, {"animation_mode": "2D", "end_seed": 43}):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                local_request(values)

    def test_auto_route_wins_over_local_model(self):
        import argparse
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="interpolator-route-") as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"animation_mode":"Interpolator"}')
            model = Path(directory) / "model.safetensors"
            model.write_bytes(b"fixture")
            args = argparse.Namespace(backend="auto", base_model=None)
            self.assertEqual(router.select_backend(args, ["--config", str(path), "--model", str(model)]),
                             "interpolator")
            self.assertEqual(router.select_backend(args, ["--animation-mode", "Interpolator"]), "interpolator")

    def test_cli_and_json_work_without_importing_runtime(self):
        for flags in (["--backend", "interpolator"], ["--animation-mode", "Interpolator"]):
            result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--model", MODEL, *flags,
                                     "--end-prompt", "forest", "--end-seed", "43", "--max-frames", "3",
                                     "--print-config"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = json.loads(result.stdout)
            self.assertEqual((values["animation_mode"], values["end_prompt"], values["end_seed"]),
                             ("Interpolator", "forest", 43))

    def test_explicit_backend_rejects_conflicting_mode(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--model", MODEL,
                                 "--backend", "interpolator", "--animation-mode", "2D", "--print-config"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --animation-mode Interpolator", result.stderr)


if __name__ == "__main__":
    unittest.main()
