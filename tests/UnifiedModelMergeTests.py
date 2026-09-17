"""Real tensor regressions for cross-family composition and prediction markers."""
import importlib.util
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
        self.save({"unet.weight": torch.ones(2, 2)}, str(self.base))
        self.save({"model.diffusion_model.blocks.0.weight": torch.ones(3, 2)}, str(self.anima))
        self.save({"diffusion_model.blocks.0.lora_A.weight": torch.ones(1, 2),
                   "diffusion_model.blocks.0.lora_B.weight": torch.ones(3, 1)}, str(self.lora))

    def test_prediction_mismatch_is_detected_before_hashing(self):
        extra = self.root / "vpred.safetensors"
        self.save({"unet.weight": self.torch.ones(2, 2), "v_pred": self.torch.empty(0),
                   "ztsnr": self.torch.empty(0)}, str(extra))
        with patch("model_merge_files.cached_model_sha256", side_effect=AssertionError("Expensive hash before preflight")):
            with self.assertRaisesRegex(ValueError, "Prediction settings differ"):
                merge_models(self.base, extra, output=self.root / "invalid.safetensors")
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
        before = [p.read_bytes() for p in (self.base, self.anima, self.lora)]
        output = self.root / "combined.iildmodel"
        report = merge_models(self.base, self.anima, additional_models=[self.lora], mode="unified", output=output)
        self.assertEqual(report["composition"], "ordered-image-refinement")
        self.assertEqual(report["stages"][1]["loras"], [{"source_index": 2, "strength": 1.0}])
        self.assertEqual(report["stages"][1]["strength"], 0.35)
        self.assertTrue(self.torch.equal(load_file(output / "members/001/model.safetensors")["model.diffusion_model.blocks.0.weight"], self.torch.full((3, 2), 2.0)))
        self.assertEqual([p.read_bytes() for p in (self.base, self.anima, self.lora)], before)
        manifest = json.loads((output / "model_index.json").read_text())
        self.assertEqual(manifest["schema"], "iild-unified-model-v1")
        self.assertEqual(len(manifest["stages"]), 2)

    def test_unmatched_lora_and_invalid_strength_leave_no_output(self):
        output = self.root / "combined.iildmodel"
        with self.assertRaisesRegex(ValueError, "no compatible checkpoint"):
            merge_models(self.base, self.lora, mode="unified", output=output, compatibility_models=[])
        with self.assertRaisesRegex(ValueError, "strengths must be"):
            merge_models(self.base, self.anima, mode="unified", weights=1.1, output=output)
        self.assertFalse(output.exists())

    def test_inspection_does_not_hash_or_publish(self):
        request = resolve_merge_request(self.base, self.anima, mode="unified", output=self.root / "result.iildmodel")
        with patch("model_merge_files.cached_model_sha256", side_effect=AssertionError("Unexpected hash")):
            report = inspect_merge_request(request)
        self.assertEqual(len(report["stages"]), 2)
        self.assertFalse(request.output.exists())

    def test_unmodified_members_preserve_even_nonfinite_source_bytes(self):
        self.save({"bad.weight": self.torch.tensor([float("nan")])}, str(self.anima))
        output = self.root / "combined.iildmodel"
        before = self.anima.read_bytes()
        report = merge_models(self.base, self.anima, weights=0, mode="unified", output=output)
        self.assertEqual((output / "members/001/model.safetensors").read_bytes(), before)
        self.assertIn("not certified", report["tensor_validation"]["copied_members"])
        self.assertEqual(self.anima.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            merge_models(self.base, self.anima, mode="unified", output=output)

    def test_nonfinite_arithmetic_still_fails_without_publication(self):
        self.save({"model.diffusion_model.blocks.0.weight": self.torch.full((3, 2), float("nan"))}, str(self.anima))
        output = self.root / "combined.iildmodel"
        with self.assertRaisesRegex(ValueError, "finite"):
            merge_models(self.base, self.anima, additional_models=[self.lora], mode="unified", output=output)
        self.assertFalse(output.exists())

    def test_custom_architecture_and_marker_storage_are_preserved(self):
        self.save({"model.diffusion_model.input_blocks.0.0.weight": self.torch.ones(1, 4, 1, 1),
                   "model.diffusion_model.input_blocks.1.attn2.to_k.weight": self.torch.ones(1, 768),
                   "model.diffusion_model.input_blocks.2.attn2.to_k.weight": self.torch.ones(1, 1024),
                   "v_pred": self.torch.tensor(1)}, str(self.anima))
        output = self.root / "custom.iildmodel"
        before = self.anima.read_bytes()
        report = merge_models(self.base, self.anima, mode="unified", output=output)
        self.assertEqual(report["stages"][1]["architecture"], "unknown")
        self.assertIn("architecture_note", report["stages"][1])
        self.assertEqual((output / "members/001/model.safetensors").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
