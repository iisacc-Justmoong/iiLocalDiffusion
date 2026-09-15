#!/usr/bin/env python3
"""Public request defaults, replacement precedence, and packaged asset identity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from generation_defaults import append_default_tokens, checked_resource, read_defaults
from generation_config import configuration_values


class GenerationDefaultsTests(unittest.TestCase):
    def request(self, preset="sdxl-base", **values):
        model = ROOT / "tests/fixtures" / {"sdxl-base": "sdxl-base-manifest", "sd15": "sd-v1-manifest",
                                         "flux1-schnell": "flux1-schnell-manifest"}[preset]
        return generate.resolve_request({"preset": preset, "model": str(model), **values})[1]

    def test_sdxl_uses_all_seven_embeddings_and_full_strength_fallback(self):
        args = self.request()
        self.assertTrue(args.lora_selection.weight_name.startswith("addDetailAesthetic_v20_32"))
        self.assertEqual(args.lora_scale, 1.0)
        self.assertEqual(len(args.text_embedding_selections), 7)
        for item in args.text_embedding_selections:
            self.assertIn(item.token, args.negative_prompt)
            self.assertTrue(Path(item.file.path).is_file())
        self.assertEqual(len(args.default_modifier_metadata["negative_tokens"]), 7)

    def test_custom_lora_replaces_fallback_but_keeps_default_negatives(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "custom.safetensors"
            path.write_bytes(b"local adapter identity fixture")
            args = self.request(lora=str(path), lora_scale=0.35, negative_prompt="blurred text")
            self.assertEqual(args.lora_selection.weight_name, path.name)
            self.assertEqual(args.lora_scale, 0.35)
            self.assertTrue(args.negative_prompt.startswith("blurred text, "))
            self.assertEqual(len(args.text_embedding_selections), 7)

    def test_secondary_negative_keeps_custom_text_and_defaults(self):
        args = self.request(negative_prompt="artifact", negative_prompt_2="bad anatomy")
        for item in args.text_embedding_selections:
            self.assertIn(item.token, args.negative_prompt_2)
        self.assertTrue(args.negative_prompt_2.startswith("bad anatomy, "))

    def test_resolved_configuration_replay_does_not_duplicate_tokens(self):
        first = self.request()
        _, second = generate.resolve_request(configuration_values(first))
        self.assertEqual(first.negative_prompt, second.negative_prompt)
        self.assertEqual(first.text_embedding_selections, second.text_embedding_selections)

    def test_incompatible_families_never_receive_sdxl_lora(self):
        sd15 = self.request("sd15")
        self.assertIsNone(sd15.lora_selection)
        self.assertEqual(len(sd15.text_embedding_selections), 3)
        flux = self.request("flux1-schnell")
        self.assertIsNone(flux.lora_selection)
        self.assertEqual(flux.text_embedding_selections, [])

    def test_explicit_baseline_is_available_for_controlled_comparisons(self):
        args = self.request(default_modifiers=False)
        self.assertIsNone(args.lora_selection)
        self.assertEqual(args.text_embedding_selections, [])
        self.assertEqual(args.negative_prompt, "")

    def test_missing_default_package_fails_instead_of_silent_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            with self.assertRaisesRegex(SystemExit, "generation-defaults"):
                self.request(generation_resources=directory)

    def test_all_manifest_identities_match_actual_files(self):
        root, manifest = read_defaults()
        for item in [manifest["fallback_lora"], *manifest["negative_embeddings"]]:
            file = checked_resource(root, item)
            self.assertEqual(file.sha256, item["sha256"])
        checked_resource(root, manifest["fallback_vae"])
        with self.assertRaisesRegex(ValueError, "differs"):
            checked_resource(root, {**manifest["fallback_lora"], "sha256": "0" * 64})

    def test_token_deduplication_does_not_confuse_substrings(self):
        self.assertEqual(append_default_tokens("x_token_extra", ["x_token"]), "x_token_extra, x_token")
        self.assertEqual(append_default_tokens("(x_token:0.8)", ["x_token"]), "(x_token:0.8)")


if __name__ == "__main__":
    unittest.main()
