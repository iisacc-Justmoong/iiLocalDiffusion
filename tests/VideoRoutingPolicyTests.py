#!/usr/bin/env python3
"""Video defaults and the low-FPS/GIF image-animation boundary."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from local_model_fixture import local_request, MODEL
from GenerationRouterTests import router

ROOT = Path(__file__).resolve().parents[1]


class VideoRoutingPolicyTests(unittest.TestCase):
    def backend(self, *tokens, backend="auto"):
        return router.select_backend(argparse.Namespace(base_model=None, backend=backend), list(tokens))

    def test_mp4_defaults_to_ltx_and_boundary_is_inclusive(self):
        self.assertEqual(self.backend("--output", "movie.mp4"), "video")
        for fps, expected in ((8, "deforum"), (12, "deforum"), (12.001, "video"), (24, "video")):
            with self.subTest(fps=fps):
                self.assertEqual(self.backend("--fps", str(fps), "--output", "movie.mp4"), expected)

    def test_gif_and_interpolation_endpoints_select_image_animation(self):
        self.assertEqual(self.backend("--fps", "24", "--output", "movie.GIF"), "deforum")
        self.assertEqual(self.backend("--fps", "12", "--end-prompt", "forest"), "interpolator")
        self.assertEqual(self.backend("--output", "movie.gif", "--end-seed", "43"), "interpolator")

    def test_explicit_animation_cannot_bypass_policy_in_router_or_python(self):
        for backend, mode in (("deforum", "2D"), ("interpolator", "Interpolator")):
            for fps in (12.001, 24):
                with self.subTest(backend=backend, fps=fps):
                    with self.assertRaisesRegex(ValueError, "12.*GIF.*LTX"):
                        self.backend("--fps", str(fps), backend=backend)
                    with self.assertRaisesRegex(ValueError, "12.*GIF.*LTX"):
                        self.backend("--fps", str(fps), "--animation-mode", mode)
                    with self.assertRaisesRegex(SystemExit, "12.*GIF.*LTX"):
                        local_request({"animation_mode": mode, "fps": fps})
            for fps, output in ((12, "movie.mp4"), (24, "movie.GIF")):
                with self.subTest(backend=backend, fps=fps, output=output):
                    _, args = local_request({"animation_mode": mode, "fps": fps, "output": output})
                    self.assertEqual(args.fps, fps)
            _, args = local_request({"animation_mode": mode})
            self.assertEqual(args.fps, 12)

    def test_temporal_requests_and_explicit_ltx_stay_temporal(self):
        self.assertEqual(self.backend("--fps", "8", backend="video"), "video")
        self.assertEqual(self.backend("--fps", "8", "--camera", "dolly-in"), "video")
        self.assertEqual(self.backend("--frames", "25"), "video")
        self.assertEqual(self.backend("--duration", "2"), "video")
        self.assertEqual(self.backend("--interpolation-factor", "3"), "video")
        self.assertEqual(self.backend("--model", MODEL, "--output", "image.png"), "diffusers")

    def test_explicit_generic_pipelines_and_workflows_keep_their_contracts(self):
        self.assertEqual(self.backend("--workflow", "flow.json", "--fps", "24", "--output", "movie.mp4"), "comfyui")
        self.assertEqual(self.backend("--pipeline-class", "LTXPipeline", "--fps", "24"), "diffusers")

    def test_malformed_routing_values_fail_as_input_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="video-policy-invalid-") as directory:
            config = Path(directory) / "request.json"
            for value in ({"output": 12}, {"fps": {}}, {"fps": True}):
                with self.subTest(value=value):
                    config.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        self.backend("--config", str(config))

    def test_config_and_cli_precedence_choose_the_same_route_without_ml(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="video-policy-") as directory:
            config = Path(directory) / "request.json"
            for values, extra, expected in (
                ({"fps": 12, "frames": 3, "output": "movie.mp4"}, [], "2D"),
                ({"fps": 12, "frames": 3, "output": "movie.mp4"}, ["--fps", "24"], "video"),
                ({"fps": 24, "duration": .25, "output": "movie.gif", "end_prompt": "forest"}, [], "Interpolator"),
                ({"fps": 24, "output": "movie.gif"}, ["--output", str(Path(directory) / "movie.mp4")], "video"),
            ):
                with self.subTest(values=values, extra=extra):
                    config.write_text(json.dumps({"model": MODEL, **values}))
                    result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/generate.py"),
                                             "--config", str(config), *extra, "--print-config"],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report.get("animation_mode", report.get("backend")), expected)

    def test_frames_alias_and_duration_resolve_to_a_replayable_animation(self):
        import generate
        for values, expected in (({"frames": 7}, 7), ({"duration": .5, "fps": 12}, 6)):
            _, args = local_request({"animation_mode": "2D", **values})
            self.assertEqual(args.max_frames, expected)
            saved = generate.configuration_values(args)
            _, replay = local_request(saved)
            self.assertEqual(generate.configuration_values(replay), saved)
        with self.assertRaises(SystemExit):
            local_request({"animation_mode": "2D", "frames": 4, "max_frames": 5})
        with self.assertRaises(SystemExit):
            local_request({"animation_mode": "2D", "frames": 4, "duration": 1})

    def test_invalid_gif_timing_is_rejected_before_loading_models(self):
        for fps in (0, float("nan"), 101, .0001):
            with self.subTest(fps=fps), self.assertRaises(SystemExit):
                local_request({"animation_mode": "2D", "output": "movie.gif", "fps": fps})


if __name__ == "__main__":
    unittest.main()
