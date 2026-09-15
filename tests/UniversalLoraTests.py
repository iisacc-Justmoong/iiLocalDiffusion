#!/usr/bin/env python3
"""LoRA routing and family defaults without importing an inference runtime."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate_any
from generation_defaults import read_defaults


class UniversalLoraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="universal-lora-")
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model"
        self.model.mkdir()
        self.adapter = self.directory / "adapter.safetensors"
        self.adapter.write_bytes(b"adapter identity fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, pipeline, *tokens, hidden_size=None):
        index = {"_class_name": pipeline}
        # Keep these adapter-only fixtures independent from the required decoder.
        from vae_defaults import pipeline_vae_family, VAE_CLASSES
        family = pipeline_vae_family(pipeline)
        if family:
            index["vae"] = ["diffusers", VAE_CLASSES[family]]
            (self.model / "vae").mkdir(exist_ok=True)
            (self.model / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"embedded VAE fixture")
        (self.model / "model_index.json").write_text(json.dumps(index))
        if hidden_size is not None:
            encoder = self.model / "text_encoder"
            encoder.mkdir(exist_ok=True)
            (encoder / "config.json").write_text(json.dumps({"hidden_size": hidden_size}))
        return generate_any.resolve_arguments(generate_any.build_parser().parse_args([
            "--model", str(self.model), *tokens]))

    def registry(self, families, scale=0.7):
        item = {"file": self.adapter.name, "families": families, "scale": scale,
                "sha256": hashlib.sha256(self.adapter.read_bytes()).hexdigest(),
                "size": self.adapter.stat().st_size}
        manifest = {"version": 1, "fallback_loras": [item], "negative_embeddings": []}
        (self.directory / "generation-defaults.json").write_text(json.dumps(manifest))
        return manifest

    def test_explicit_lora_reaches_every_installed_pipeline_route(self):
        for name in ("StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusion3Pipeline",
                     "FluxPipeline", "Flux2Pipeline", "QwenImagePipeline", "HunyuanDiTPipeline"):
            with self.subTest(pipeline=name):
                args = self.args(name, "--lora", str(self.adapter), "--lora-scale", "0.35")
                self.assertEqual(args.lora_selection.scale, 0.35)
                self.assertEqual(args.lora_selection.local_file.path, str(self.adapter))
                config = generate_any.configuration(args)
                self.assertEqual(config["lora"], str(self.adapter))
                self.assertEqual(config["lora_scale"], 0.35)

    def test_family_registry_selects_sd1_sd2_sd3_flux_and_other_defaults(self):
        for name, family, width in (("StableDiffusionPipeline", "sd15", 768),
                                    ("StableDiffusionPipeline", "sd2", 1024),
                                    ("StableDiffusion3Pipeline", "sd3", None),
                                    ("FluxPipeline", "flux1", None),
                                    ("Flux2Pipeline", "flux2", None),
                                    ("QwenImagePipeline", "qwen-image", None)):
            with self.subTest(family=family):
                self.registry([family])
                args = self.args(name, "--generation-resources", str(self.directory), hidden_size=width)
                self.assertEqual(args.lora_selection.scale, 0.7)
                self.assertEqual(args.default_modifier_metadata["fallback_lora"], self.adapter.name)
                self.assertEqual(args.default_modifier_metadata["family"], family)

    def test_unknown_sd_version_does_not_guess_sd1(self):
        self.registry(["sd15"])
        args = self.args("StableDiffusionPipeline", "--generation-resources", str(self.directory))
        self.assertIsNone(args.lora_selection)
        self.assertEqual(args.default_modifier_metadata["fallback_lora_status"], "not-configured-for-family")

    def test_sdxl_fallback_is_shared_with_generic_images_and_editing(self):
        for name in ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline",
                     "StableDiffusionXLInpaintPipeline"):
            with self.subTest(pipeline=name):
                args = self.args(name)
                self.assertEqual(args.lora_selection.weight_name, "addDetailAesthetic_v20_32.safetensors")
                self.assertEqual(args.lora_selection.scale, 1.0)

    def test_explicit_lora_and_disabled_defaults_take_precedence(self):
        self.registry(["flux1"], 0.7)
        explicit = self.args("FluxPipeline", "--generation-resources", str(self.directory),
                             "--lora", str(self.adapter), "--lora-scale", "0.2")
        self.assertEqual(explicit.lora_selection.scale, 0.2)
        baseline = self.args("FluxPipeline", "--generation-resources", str(self.directory),
                             "--no-default-modifiers")
        self.assertIsNone(baseline.lora_selection)

    def test_overlapping_family_entries_and_zero_fallback_are_rejected(self):
        from generation_defaults import fallback_loras
        manifest = self.registry(["flux1"])
        manifest["fallback_lora"] = {**manifest["fallback_loras"][0], "families": ["flux1-schnell"]}
        with self.assertRaisesRegex(ValueError, "Duplicate.*family"):
            fallback_loras(manifest)
        manifest = self.registry(["sd2"], 0)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            fallback_loras(manifest)

    def test_existing_single_fallback_manifest_remains_supported(self):
        from generation_defaults import fallback_loras
        _, manifest = read_defaults()
        self.assertEqual(fallback_loras(manifest), [manifest["fallback_lora"]])

    def test_relative_registry_paths_cannot_escape_or_use_absolute_files(self):
        from generation_defaults import checked_resource
        item = self.registry(["sd2"])["fallback_loras"][0]
        for file in (str(self.adapter), "../adapter.safetensors"):
            with self.subTest(file=file), self.assertRaisesRegex(ValueError, "escapes"):
                checked_resource(self.directory.relative_to(ROOT), {**item, "file": file})


if __name__ == "__main__":
    unittest.main()
