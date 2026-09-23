#!/usr/bin/env python3
"""VAE omission, embedded precedence and latent-family safety contracts."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from vae_defaults import resolve_vae_selection, verify_vae_selection, load_selected_vae


class VaeDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(vae=None, pipeline_class=None, model=str(self.root),
                                    source_kind="directory", generation_resources=None)
        self.index = {"_class_name": "QwenImagePipeline", "vae": ["diffusers", "AutoencoderKLQwenImage"]}

    def resolve(self):
        return resolve_vae_selection(self.args, self.index, self.root)

    def test_metadata_mode_loads_fallback_without_checksum_scan(self):
        with patch.dict(os.environ, {"IILD_MODEL_VALIDATION": "metadata"}), \
                patch("weight_files.cached_model_sha256", side_effect=AssertionError("Full model hash")):
            for name in ("QwenImagePipeline", "StableDiffusionXLPipeline", "FluxPipeline"):
                self.index = {"_class_name": name}
                selection, status = self.resolve()
                self.assertEqual(status, "fallback")
                self.assertIsNone(selection.weight.sha256)
                verify_vae_selection(selection)

    def test_omission_loads_verified_official_qwen_vae(self):
        selection, status = self.resolve()
        self.assertEqual(status, "fallback")
        self.assertEqual(selection.weight.sha256, "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344")
        self.assertEqual(selection.class_name, "AutoencoderKLQwenImage")
        verify_vae_selection(selection)

    def test_explicitly_null_or_omitted_component_gets_fallback(self):
        for value in ([None, None], None):
            self.index["vae"] = value
            self.assertEqual(self.resolve()[1], "fallback")

    def test_fallback_is_independent_of_optional_style_modifiers(self):
        self.args.default_modifiers = False
        self.assertEqual(self.resolve()[1], "fallback")

    def test_embedded_vae_is_preserved_even_if_defaults_are_unavailable(self):
        vae = self.root / "vae"
        vae.mkdir()
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"present")
        self.args.generation_resources = self.root / "absent"
        self.assertEqual(self.resolve(), (None, "model"))

    def test_model_owned_alternative_vae_architecture_stays_with_its_loader(self):
        self.index = {"_class_name": "StableDiffusionXLPipeline", "vae": ["diffusers", "AutoencoderTiny"]}
        (self.root / "vae").mkdir()
        (self.root / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"embedded")
        self.args.generation_resources = self.root / "absent"
        self.assertEqual(self.resolve(), (None, "model"))

    def test_embedded_single_file_vae_is_preserved(self):
        model = self.root / "checkpoint.safetensors"
        header = json.dumps({"first_stage_model.decoder.conv_in.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        model.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.args.source_kind, self.args.model = "single-file", str(model)
        self.assertEqual(self.resolve(), (None, "model"))

    def test_other_latent_families_and_rgba_never_receive_qwen_rgb_weights(self):
        for name in ("StableDiffusionPipeline", "StableDiffusion3Pipeline",
                     "WanPipeline", "QwenImageLayeredPipeline", "QwenImage2Pipeline"):
            self.index["_class_name"] = name
            self.assertEqual(self.resolve(), (None, "not-configured-for-pipeline"))

    def test_sdxl_and_flux_select_their_own_latent_space_without_vae_argument(self):
        for name, cls, family, channels in (
                ("StableDiffusionXLPipeline", "AutoencoderKL", "sdxl", 4),
                ("FluxPipeline", "AutoencoderKL", "flux1", 16),
                ("Flux2Pipeline", "AutoencoderKLFlux2", "flux2", 32),
                ("Flux2KleinPipeline", "AutoencoderKLFlux2", "flux2", 32)):
            for component in (None, [None, None], ["diffusers", cls]):
                with self.subTest(pipeline=name, component=component):
                    self.index = {"_class_name": name, "vae": component}
                    selected, status = self.resolve()
                    self.assertEqual(status, "fallback")
                    self.assertEqual(selected.class_name, cls)
                    self.assertIn("/vae/" + family + "/", selected.directory + "/")
                    self.assertEqual(json.loads(Path(selected.config_path).read_text())["latent_channels"], channels)

    def test_flux1_vae_is_not_accepted_by_sdxl_or_flux2(self):
        self.index = {"_class_name": "FluxPipeline"}
        self.args.vae = self.resolve()[0].directory
        for name in ("StableDiffusionXLPipeline", "Flux2Pipeline"):
            self.index = {"_class_name": name}
            with self.assertRaisesRegex(ValueError, "VAE"):
                self.resolve()

    def test_duplicate_family_defaults_fail(self):
        from generation_defaults import read_defaults
        root, manifest = read_defaults()
        manifest["fallback_vaes"].append(manifest["fallback_vaes"][0])
        self.index = {"_class_name": "StableDiffusionXLPipeline"}
        with patch("vae_defaults.read_defaults", return_value=(root, manifest)):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.resolve()

    def test_missing_fallback_fails_with_an_actionable_error(self):
        self.args.generation_resources = self.root
        (self.root / "generation-defaults.json").write_text('{"version":1}')
        with self.assertRaisesRegex(ValueError, "fallback_vae"):
            self.resolve()

    def test_explicit_vae_directory_precedes_embedded_and_fallback(self):
        bundled, _ = self.resolve()
        self.args.vae = bundled.directory
        self.args.generation_resources = self.root / "absent"
        selected, status = self.resolve()
        self.assertEqual(status, "explicit")
        self.assertEqual(selected.weight, bundled.weight)

    def test_wrong_vae_class_or_latent_contract_is_rejected(self):
        vae = self.root / "override"
        vae.mkdir()
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
        self.args.vae = str(vae)
        for config in ({"_class_name": "AutoencoderKL", "latent_channels": 4},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 32},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 16, "input_channels": 4},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 16, "out_channels": 4}):
            (vae / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "VAE"):
                self.resolve()

    def test_loader_passes_the_component_before_pipeline_construction(self):
        selected, _ = self.resolve()
        vae = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        cls = SimpleNamespace(from_pretrained=Mock(return_value=vae))
        runtime = SimpleNamespace(AutoencoderKLQwenImage=cls)
        self.assertIs(load_selected_vae(selected, runtime, "float32"), vae)
        self.assertTrue(cls.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(cls.from_pretrained.call_args.kwargs["use_safetensors"])

    def test_changed_vae_config_invalidates_selection(self):
        import shutil
        selected, _ = self.resolve()
        destination = self.root / "copy"
        shutil.copytree(selected.directory, destination)
        self.args.vae = str(destination)
        selected, _ = self.resolve()
        config = destination / "config.json"
        config.write_text(config.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "VAE changed"):
            verify_vae_selection(selected)

    def test_config_only_model_uses_fallback_and_partial_weights_do_not(self):
        directory = self.root / "vae"
        directory.mkdir()
        (directory / "config.json").write_text('{"_class_name":"AutoencoderKLQwenImage"}')
        self.assertEqual(self.resolve()[1], "fallback")
        (directory / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
        self.assertEqual(self.resolve(), (None, "model"))

    def test_single_file_missing_vae_uses_the_same_fallback(self):
        model = self.root / "checkpoint.safetensors"
        header = json.dumps({"transformer.img_in.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        model.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.args.source_kind, self.args.model = "single-file", str(model)
        self.assertEqual(self.resolve()[1], "fallback")

    def test_manifest_config_path_and_hash_are_enforced(self):
        from generation_defaults import read_defaults
        root, original = read_defaults()
        for field, value in (("file", str(root / original["fallback_vae"]["config"]["file"])),
                             ("file", "../config.json"), ("sha256", "0" * 64)):
            manifest = json.loads(json.dumps(original))
            manifest["fallback_vae"]["config"][field] = value
            with patch("vae_defaults.read_defaults", return_value=(root, manifest)):
                with self.assertRaisesRegex(ValueError, "VAE configuration"):
                    self.resolve()


if __name__ == "__main__":
    unittest.main()
