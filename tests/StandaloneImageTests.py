#!/usr/bin/env python3
"""Standalone checkpoint routing, offline resources and artifact publication."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import DownloadedModelTests as fixtures
import standalone_image
import checkpoint_config
import generate


class StandaloneImageTests(unittest.TestCase):
    setUp = fixtures.DownloadedModelTests.setUp
    safetensors = fixtures.DownloadedModelTests.safetensors
    sd = fixtures.DownloadedModelTests.sd

    def checkpoint(self, context=2048, **kwargs):
        return self.safetensors({
            "model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, context],
            "first_stage_model.encoder.conv_in.weight": [1],
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": [1],
        }, **kwargs)

    def test_checkpoint_uses_bundled_config_without_comfy_or_network(self):
        for context, preset, family in ((768, "sd15-compatible", "sd1"), (2048, "sdxl", "sdxl")):
            model = self.checkpoint(context)
            args = standalone_image.build_parser().parse_args(["--model-path", str(model)])
            with patch("subprocess.Popen", side_effect=AssertionError("No server")):
                selected, resolved = standalone_image.resolve_arguments(args)
            self.assertEqual(selected.name, preset)
            self.assertEqual(Path(resolved.config_selection.source), checkpoint_config.CONFIGS / family)
            self.assertTrue(resolved.local_files_only)

    def test_adapter_split_and_cross_family_are_rejected_before_loading(self):
        cases = [(self.sd(2048, name="split.safetensors"), [], "components"),
                 (self.checkpoint(name="full.safetensors"), ["--base-model", "SD 1.5"], "conflicts"),
                 (self.safetensors({"lora_unet.foo.lora_up.weight": [1, 1]}, name="lora.safetensors"), [], "lora")]
        for model, extra, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((ValueError, SystemExit), message):
                standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
                    ["--model", str(model), *extra]))

    def test_explicit_config_and_prediction_metadata_are_preserved(self):
        model = self.checkpoint(metadata={"prediction_type": "v_prediction"})
        config = checkpoint_config.CONFIGS / "sdxl"
        _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
            ["--model", str(model), "--model-config", str(config)]))
        self.assertEqual(args.config_selection.source, str(config))
        self.assertEqual(args.prediction_type, "v_prediction")

    def test_bundled_resources_match_manifest_and_contain_no_weights(self):
        manifest = json.loads((checkpoint_config.CONFIGS / "manifest.json").read_text())
        for family, record in manifest["families"].items():
            self.assertEqual(len(record["revision"]), 40)
            for name, identity in record["files"].items():
                data = (checkpoint_config.CONFIGS / family / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"], f"{family}/{name}: SHA-256")
                self.assertEqual(len(data), identity["size_bytes"], f"{family}/{name}: size")
                self.assertNotIn(Path(name).suffix, (".safetensors", ".bin", ".ckpt"))
            self.assertTrue((checkpoint_config.CONFIGS / family / "tokenizer/merges.txt").is_file())

    def test_animation_single_file_uses_same_offline_configuration(self):
        for mode in ("2D", "Interpolator"):
            _, args = generate.resolve_request({"model": str(self.checkpoint()),
                                               "animation_mode": mode})
            self.assertEqual(Path(args.config_selection.source), checkpoint_config.CONFIGS / "sdxl")

    def test_uppercase_singular_suffix_preserves_original_and_safe_alias(self):
        from weight_files import checked_safetensors_path
        model = self.checkpoint(name="MODEL.SAFETENSOR")
        _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(["--model", str(model)]))
        self.assertEqual(args.model_selection.source, str(model))
        with checked_safetensors_path(args.model_selection.single_file, self.directory / "aliases", "model") as path:
            self.assertEqual(path.suffix, ".safetensors")
            self.assertEqual(path.resolve(), model)

    def test_output_failure_preserves_existing_and_never_publishes_completion(self):
        output = self.directory / "result"
        output.mkdir()
        with patch("generate.run", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(SystemExit, "inference failed"):
                standalone_image.main(["--model", str(self.checkpoint()), "--output-dir", str(output)])
        self.assertEqual(list(output.iterdir()), [])
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(SystemExit, "empty"):
            standalone_image.main(["--model", str(self.checkpoint()), "--output-dir", str(output)])
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for real PNG validation")
    def test_success_publishes_batch_and_final_paths_together(self):
        from PIL import Image
        output = self.directory / "result"
        def render(preset, args):
            for index in range(args.num_images):
                path = args.output.with_stem(f"image-{index:04d}")
                Image.new("RGB", (64, 64), "red").save(path)
                path.with_suffix(".json").write_text(json.dumps({"fixture": {"seed": args.seed + index}, "output": {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}))
            return 0
        with patch("generate.run", side_effect=render), patch("secrets.randbits", return_value=456) as random:
            self.assertEqual(standalone_image.main(["--model", str(self.checkpoint()),
                "--num-images", "2", "--output-dir", str(output)]), 0)
        manifest = json.loads((output / "generation.json").read_text())
        random.assert_called_once_with(32)
        self.assertEqual([record["fixture"]["seed"] for record in manifest["images"]], [456, 457])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["outputs"]), 2)
        for record in manifest["outputs"]:
            self.assertEqual(Path(record["path"]).parent, output)
            self.assertTrue(Path(record["path"]).is_file())
        self.assertEqual(list(self.directory.glob(".iild-generation-*")), [])

    def test_model_changed_during_inference_cannot_publish_success(self):
        model = self.checkpoint()
        output = self.directory / "changed-result"
        def mutate(preset, args):
            model.write_bytes(b"changed")
            return 0
        with patch("generate.run", side_effect=mutate), self.assertRaisesRegex(SystemExit, "changed"):
            standalone_image.main(["--model", str(model), "--output-dir", str(output)])
        self.assertEqual(list(output.iterdir()), [])

    def test_router_local_is_standalone_and_comfy_is_explicit(self):
        spec = importlib.util.spec_from_file_location("standalone_router", ROOT / "reference/generate.py")
        router = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(router)
        with patch.object(standalone_image, "main", return_value=0) as run:
            router.main(["--model-path", str(self.checkpoint()), "--work-dir", str(self.directory / "work")])
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
