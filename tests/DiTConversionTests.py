#!/usr/bin/env python3
"""DiT-standard conversion and merge integration regressions."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))

from dit_conversion import convert_to_dit, inspect_dit_conversion, main
from model_merge import merge_models


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "safetensors")),
                     "Requires the existing Torch/safetensors model runtime")
class DiTConversionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        self.torch, self.load, self.save, self.safe_open = torch, load_file, save_file, safe_open
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="dit-conversion-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.template = self.root / "tsubaki.safetensors"
        self.output = self.root / "converted.safetensors"
        self.save({
            "transformer_blocks.0.attn.to_q.weight": torch.full((3, 2), 0.25),
            "transformer_blocks.0.ff.net.0.weight": torch.full((4, 3), 0.5),
            "transformer_blocks.0.norm.weight": torch.ones(3),
            "position_ids": torch.arange(3, dtype=torch.int64),
        }, self.template, metadata={"modelspec.architecture": "TsubakiDiT", "format": "pt"})

    def source(self, family, path, offset=0.0):
        t = self.torch
        if family == "sd15":
            tensors = {
                "model.diffusion_model.input_blocks.0.0.weight": t.arange(8, dtype=t.float32).reshape(2, 4) + offset,
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": t.arange(6, dtype=t.float32).reshape(2, 3) + offset,
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.weight": t.arange(12, dtype=t.float32).reshape(3, 4) + offset,
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": t.ones(2, 768),
            }
        elif family == "sdxl":
            tensors = {
                "model.diffusion_model.input_blocks.0.0.weight": t.ones(4, 4) + offset,
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": t.arange(6, dtype=t.float32).reshape(2, 3) + offset,
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": t.ones(2, 2048),
            }
        elif family == "dit":
            tensors = {
                "joint_blocks.0.context_block.adaLN_modulation.1.bias": t.ones(3) + offset,
                "joint_blocks.0.x_block.attn.qkv.weight": t.arange(6, dtype=t.float32).reshape(3, 2) + offset,
            }
        elif family == "reference-pro":
            tensors = {
                "reference_encoder.blocks.0.attn.to_q.weight": t.arange(6, dtype=t.float32).reshape(2, 3) + offset,
                "reference_encoder.blocks.0.ff.net.0.weight": t.arange(12, dtype=t.float32).reshape(3, 4) + offset,
            }
        else:
            raise AssertionError(family)
        self.save(tensors, path)
        return path

    def test_all_requested_ecosystems_emit_the_tsubaki_tensor_contract(self):
        aliases = (("sd15", "auto"), ("sdxl", "haruka"), ("sdxl", "hoshino"),
                   ("sdxl", "illustrious"), ("dit", "tsubaki"),
                   ("reference-pro", "reference-pro"))
        target = self.load(self.template)
        for index, (fixture, family) in enumerate(aliases):
            with self.subTest(family=family):
                source = self.source(fixture, self.root / f"{family}-{index}.safetensors")
                output = self.root / f"output-{index}.safetensors"
                report = convert_to_dit(source, self.template, output, source_family=family)
                actual = self.load(output)
                self.assertEqual(actual.keys(), target.keys())
                for key in target:
                    self.assertEqual(actual[key].shape, target[key].shape)
                    self.assertEqual(actual[key].dtype, target[key].dtype)
                self.assertEqual(report["standard"], "iild-dit-standard-v1")
                self.assertEqual(report["target_family"], "tsubaki")
                self.assertFalse(report["semantic_equivalence"])
                self.assertGreater(report["transferred_tensor_count"], 0)

    def test_auto_detection_distinguishes_sd15_sdxl_and_dit(self):
        for family in ("sd15", "sdxl", "dit"):
            source = self.source(family, self.root / f"auto-{family}.safetensors")
            report = inspect_dit_conversion(source, self.template)
            self.assertEqual(report["source_family"], family)
            self.assertFalse(self.output.exists())

    def test_conversion_is_deterministic_preserves_sources_and_records_mapping(self):
        source = self.source("sd15", self.root / "source.safetensors")
        before = source.read_bytes(), self.template.read_bytes()
        first = convert_to_dit(source, self.template, self.output)
        second_path = self.root / "second.safetensors"
        second = convert_to_dit(source, self.template, second_path)
        first_tensors, second_tensors = self.load(self.output), self.load(second_path)
        self.assertEqual(first_tensors.keys(), second_tensors.keys())
        for key in first_tensors:
            self.torch.testing.assert_close(first_tensors[key], second_tensors[key], rtol=0, atol=0)
        self.assertEqual(before, (source.read_bytes(), self.template.read_bytes()))
        self.assertEqual(first["mappings"], second["mappings"])
        self.assertTrue(all(mapping["policy"] == "role-depth-stat-match-template-fill-v1"
                            for mapping in first["mappings"]))
        with self.safe_open(self.output, framework="pt") as reader:
            metadata = reader.metadata()
        embedded = json.loads(metadata["iild_dit_conversion"])
        self.assertEqual(embedded["source_sha256"], first["source_sha256"])
        self.assertEqual(embedded["target_template_sha256"], first["target_template_sha256"])
        self.assertEqual(metadata["modelspec.architecture"], "TsubakiDiT")

    def test_converted_outputs_are_directly_mergeable_as_dit(self):
        first_source = self.source("sd15", self.root / "sd15.safetensors")
        second_source = self.source("sdxl", self.root / "illustrious.safetensors", offset=2)
        first = self.root / "first.safetensors"
        second = self.root / "second.safetensors"
        first_report = convert_to_dit(first_source, self.template, first)
        convert_to_dit(second_source, self.template, second, source_family="illustrious")
        merged = self.root / "merged.safetensors"
        report = merge_models(first, second, weights=0.5, output=merged)
        self.assertEqual(report["mode"], "weighted-sum")
        self.assertEqual(self.load(merged).keys(), self.load(self.template).keys())
        with self.safe_open(merged, framework="pt") as reader:
            self.assertEqual(reader.metadata()["modelspec.architecture"], "TsubakiDiT")
            self.assertEqual(reader.metadata()["iild.dit.standard"], "iild-dit-standard-v1")
            self.assertEqual(reader.metadata()["iild.model_family"], "tsubaki")
            self.assertEqual(reader.metadata()["iild.dit.template_sha256"],
                             first_report["target_template_sha256"])

    def test_strict_merge_rejects_converted_models_from_different_dit_templates(self):
        source = self.source("sd15", self.root / "sd15.safetensors")
        first = self.root / "first.safetensors"
        second = self.root / "second.safetensors"
        convert_to_dit(source, self.template, first)
        alternate = self.root / "alternate-tsubaki.safetensors"
        state = self.load(self.template)
        state["transformer_blocks.0.attn.to_q.weight"] += 1
        self.save(state, alternate, metadata={"modelspec.architecture": "TsubakiDiT"})
        convert_to_dit(source, alternate, second)
        with self.assertRaisesRegex(ValueError, "architecture metadata differs.*template_sha256"):
            merge_models(first, second, checkpoint_policy="strict",
                         output=self.root / "must-not-exist.safetensors")

    def test_nonfinite_unknown_and_existing_output_fail_without_publication(self):
        source = self.source("sd15", self.root / "source.safetensors")
        tensors = self.load(source)
        for value in tensors.values():
            if value.is_floating_point():
                value.reshape(-1)[0] = float("nan")
        self.save(tensors, source)
        with self.assertRaisesRegex(ValueError, "finite"):
            convert_to_dit(source, self.template, self.output)
        self.assertFalse(self.output.exists())
        unknown = self.root / "unknown.safetensors"
        self.save({"mystery.weight": self.torch.ones(2, 2)}, unknown)
        with self.assertRaisesRegex(ValueError, "source family"):
            inspect_dit_conversion(unknown, self.template)
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            convert_to_dit(unknown, self.template, self.output, source_family="reference-pro")
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_cli_inspection_and_build(self):
        source = self.source("reference-pro", self.root / "reference-pro.safetensors")
        arguments = ["--source", str(source), "--target-dit", str(self.template),
                     "--source-family", "reference-pro", "--output", str(self.output)]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(arguments + ["--inspect"]), 0)
        self.assertEqual(json.loads(output.getvalue())["output"], str(self.output.absolute()))
        self.assertFalse(self.output.exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        self.assertTrue(self.output.is_file())


if __name__ == "__main__":
    unittest.main()
