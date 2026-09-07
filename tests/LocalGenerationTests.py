#!/usr/bin/env python3
"""Generation requires caller-owned local models, including auxiliary weights."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
import generate_any
import video_options


class LocalGenerationTests(unittest.TestCase):
    model = str(ROOT / "tests/fixtures/sd-v1-manifest")

    def test_image_and_animation_require_an_explicit_local_model(self):
        for mode in ("none", "2D", "Interpolator"):
            with self.subTest(mode=mode), self.assertRaisesRegex(SystemExit, "--model"):
                generate.resolve_request({"animation_mode": mode})

    def test_hub_ids_are_rejected_even_with_pinned_revisions(self):
        for name in ("model", "lora", "controlnet"):
            values = {"model": self.model, name: "owner/model",
                      "revision" if name == "model" else name + "_revision": "a" * 40}
            if name == "lora":
                values["lora_weight_name"] = "adapter.safetensors"
            with self.subTest(name=name), self.assertRaisesRegex(SystemExit, "local|Local"):
                generate.resolve_request(values)
        with self.assertRaisesRegex(ValueError, "local|Local"):
            generate_any.resolve_arguments(generate_any.build_parser().parse_args(
                ["--model", "owner/model", "--revision", "a" * 40]))

    def test_local_image_and_animation_requests_are_always_offline(self):
        for mode in ("none", "2D", "Interpolator"):
            _, args = generate.resolve_request({"model": self.model, "animation_mode": mode})
            self.assertTrue(args.local_files_only)
            self.assertTrue(args.model_selection.is_local)
            self.assertIsNone(args.model_selection.requested_revision)
        with self.assertRaisesRegex(SystemExit, "local|Local|offline"):
            generate.resolve_request({"model": self.model, "local_files_only": False})

    def test_video_requires_a_local_model_and_has_no_implicit_download(self):
        for tokens in ([], ["--model", "owner/model", "--revision", "a" * 40]):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "local|Local|--model"):
                video_options.resolve_options(video_options.build_parser().parse_args(tokens))
        args = video_options.resolve_options(video_options.build_parser().parse_args([
            "--model", self.model]))
        self.assertEqual(args.model, self.model)
        self.assertTrue(args.local_files_only)


if __name__ == "__main__":
    unittest.main()
