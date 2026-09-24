"""Real tensor regressions for cross-family composition and prediction markers."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from model_merge import merge_models, inspect_merge_request
from model_merge_options import resolve_merge_request
from iild_package import materialize_archive, write_archive


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "Requires the SDK tensor runtime")
class UnifiedModelMergeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.torch, self.save = torch, save_file
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="unified-merge-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "sdxl.safetensors"
        self.anima = self.root / "anima.safetensors"
        self.lora = self.root / "anima-lora.safetensors"
        self.save({"unet.weight": torch.ones(2, 2)}, str(self.base),
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        self.save({"model.diffusion_model.blocks.0.weight": torch.ones(3, 2)}, str(self.anima),
                  metadata={"modelspec.architecture": "anima"})
        self.save({"diffusion_model.blocks.0.lora_A.weight": torch.ones(1, 2),
                   "diffusion_model.blocks.0.lora_B.weight": torch.ones(3, 1)}, str(self.lora),
                  metadata={"modelspec.architecture": "anima"})

    def compatible_checkpoint(self, name="sdxl-compatible.safetensors", value=2.0):
        path = self.root / name
        self.save({"unet.weight": self.torch.full((2, 2), value)}, str(path),
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        return path

    def compatible_lora(self):
        path = self.root / "sdxl-lora.safetensors"
        self.save({"unet.lora_A.weight": self.torch.ones(1, 2),
                   "unet.lora_B.weight": self.torch.ones(2, 1)}, str(path),
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        return path

    def test_archive_output_is_verified_after_writing(self):
        output = self.root / "verified.iildmodel"
        report = merge_models(self.base, self.compatible_checkpoint(), output=output, mode="unified")
        self.assertEqual(report["output_verification"]["status"], "passed")
        self.assertEqual(report["output_verification"]["stage_count"], 2)
        self.assertEqual(report["preflight"]["output_contract"], "independent-network-cascade")

    def test_wrong_archive_member_is_never_published(self):
        output = self.root / "wrong.iildmodel"
        def corrupt(staging, destination):
            self.save({"unet.weight": self.torch.zeros(2, 2)}, str(staging / "members/000/model.safetensors"))
            write_archive(staging, destination)
        with patch("model_merge_unified.write_archive", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "member.*(changed|hash)"):
                merge_models(self.base, self.compatible_checkpoint(), output=output, mode="unified")
        self.assertFalse(output.exists())

    def test_prediction_mismatch_is_detected_before_hashing(self):
        extra = self.root / "vpred.safetensors"
        self.save({"unet.weight": self.torch.ones(2, 2), "v_pred": self.torch.empty(0),
                   "ztsnr": self.torch.empty(0)}, str(extra),
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        with patch("model_merge_files.cached_model_sha256", side_effect=AssertionError("Expensive hash before preflight")):
            with self.assertRaisesRegex(ValueError, "Prediction settings differ"):
                merge_models(self.base, extra, output=self.root / "invalid.safetensors",
                             checkpoint_policy="strict")
        self.assertFalse((self.root / "invalid.safetensors").exists())

    def test_same_prediction_markers_are_preserved(self):
        from safetensors import safe_open
        extra = self.root / "vpred.safetensors"
        state = {"unet.weight": self.torch.ones(2, 2), "v_pred": self.torch.empty(0), "ztsnr": self.torch.empty(0)}
        self.save(state, str(self.base)); self.save(state, str(extra))
        output = self.root / "merged.safetensors"
        merge_models(self.base, extra, output=output)
        with safe_open(output, framework="pt") as reader:
            self.assertEqual(set(reader.keys()), set(state))

    def test_unified_preserves_families_and_fuses_lora_only_into_compatible_member(self):
        from safetensors.torch import load_file
        compatible = self.compatible_checkpoint()
        lora = self.compatible_lora()
        before = [p.read_bytes() for p in (self.base, compatible, self.anima, lora)]
        output = self.root / "combined.iildmodel"
        report = merge_models(self.base, compatible, additional_models=[self.anima, lora], mode="unified", output=output)
        package = materialize_archive(output, self.root / "cache")
        self.assertEqual(report["composition"], "ordered-image-refinement")
        self.assertEqual(report["stages"][1]["loras"], [{"source_index": 2, "strength": 1.0}])
        self.assertEqual(report["stages"][1]["strength"], 0.35)
        self.assertEqual(report["excluded_material_count"], 1)
        self.assertEqual(report["excluded_sources"][0]["path"], str(self.anima))
        self.assertTrue(output.is_file())
        self.assertTrue(self.torch.equal(load_file(package / "members/001/model.safetensors")["unet.weight"], self.torch.full((2, 2), 3.0)))
        self.assertEqual([p.read_bytes() for p in (self.base, compatible, self.anima, lora)], before)
        manifest = json.loads((package / "model_index.json").read_text())
        self.assertEqual(manifest["schema"], "iild-unified-model-v1")
        self.assertEqual(len(manifest["stages"]), 2)

    def test_unmatched_lora_and_invalid_strength_leave_no_output(self):
        output = self.root / "combined.iildmodel"
        with self.assertRaisesRegex(ValueError, "No merge material is compatible"):
            merge_models(self.base, self.lora, mode="unified", output=output, compatibility_models=[])
        with self.assertRaisesRegex(ValueError, "strengths must be"):
            merge_models(self.base, self.compatible_checkpoint(), mode="unified", weights=1.1, output=output)
        self.assertFalse(output.exists())

    def test_inspection_does_not_hash_or_publish(self):
        request = resolve_merge_request(self.base, self.compatible_checkpoint(), mode="unified", output=self.root / "result.iildmodel")
        with patch("model_merge_files.cached_model_sha256", side_effect=AssertionError("Unexpected hash")):
            report = inspect_merge_request(request)
        self.assertEqual(len(report["stages"]), 2)
        self.assertFalse(request.output.exists())

    def test_single_stage_packaged_file_is_a_weighted_merge_input(self):
        staging = self.root / "single-package"
        staging.mkdir()
        member = staging / "model.safetensors"
        member.write_bytes(self.base.read_bytes())
        manifest = {"schema": "iild-unified-model-v1", "_class_name": "IILDUnifiedCascade",
                    "container": "zip-stored-v1", "composition": "ordered-image-refinement",
                    "stages": [{"model": member.name, "strength": 1, "size_bytes": member.stat().st_size,
                                "sha256": hashlib.sha256(member.read_bytes()).hexdigest()}]}
        (staging / "model_index.json").write_text(json.dumps(manifest))
        package = self.root / "single.iildmodel"
        write_archive(staging, package)
        output = self.root / "remerged.safetensors"
        report = merge_models(package, self.base, weights=0.5, output=output,
                              cache_dir=self.root / "merge-cache")
        self.assertTrue(output.is_file())
        self.assertEqual(report["sources"][0]["format"], "iildmodel")
        self.assertEqual(report["sources"][0]["path"], str(package))

    def test_unmodified_members_preserve_even_nonfinite_source_bytes(self):
        compatible = self.compatible_checkpoint()
        self.save({"unet.weight": self.torch.full((2, 2), float("nan"))}, str(compatible),
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        output = self.root / "combined.iildmodel"
        before = compatible.read_bytes()
        report = merge_models(self.base, compatible, weights=0, mode="unified", output=output)
        package = materialize_archive(output, self.root / "cache-nonfinite")
        self.assertEqual((package / "members/001/model.safetensors").read_bytes(), before)
        self.assertIn("not certified", report["tensor_validation"]["copied_members"])
        self.assertEqual(compatible.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            merge_models(self.base, compatible, mode="unified", output=output)

    def test_nonfinite_arithmetic_still_fails_without_publication(self):
        compatible = self.compatible_checkpoint(value=float("nan"))
        lora = self.compatible_lora()
        output = self.root / "combined.iildmodel"
        with self.assertRaisesRegex(ValueError, "finite"):
            merge_models(self.base, compatible, additional_models=[lora], mode="unified", output=output,
                         checkpoint_policy="strict")
        self.assertFalse(output.exists())

    def test_custom_architecture_and_marker_storage_are_preserved(self):
        self.save({"model.diffusion_model.input_blocks.0.0.weight": self.torch.ones(1, 4, 1, 1),
                   "model.diffusion_model.input_blocks.1.attn2.to_k.weight": self.torch.ones(1, 768),
                   "model.diffusion_model.input_blocks.2.attn2.to_k.weight": self.torch.ones(1, 1024),
                   "v_pred": self.torch.tensor(1)}, str(self.anima))
        output = self.root / "custom.iildmodel"
        before = self.anima.read_bytes()
        with self.assertRaisesRegex(ValueError, "No merge material is compatible"):
            merge_models(self.base, self.anima, mode="unified", output=output)
        self.assertFalse(output.exists())
        self.assertEqual(self.anima.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
