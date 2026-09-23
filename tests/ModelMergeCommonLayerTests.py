"""Cross-architecture base-layout projection and explicit instability contract."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from model_merge import inspect_merge_request, merge_models
from model_merge_options import build_parser, resolve_merge_request


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "Requires the SDK tensor runtime")
class ModelMergeCommonLayerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.torch, self.save = torch, save_file
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="common-layer-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "flux.safetensors"
        self.krea = self.root / "krea-int8.safetensors"
        self.output = self.root / "merged.safetensors"
        save_file({
            "transformer_blocks.0.attn.to_q.weight": torch.zeros((2, 3), dtype=torch.float32),
            "transformer_blocks.0.attn.to_q.bias": torch.tensor([4.0, 8.0]),
            "step_counter": torch.tensor(7, dtype=torch.int64),
        }, self.base, metadata={"modelspec.architecture": "flux2"})
        save_file({
            "blocks.4.attn.wq.weight": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int8),
            "blocks.4.attn.wq.weight_scale": torch.tensor([[2.0], [3.0], [4.0]]),
            "blocks.4.attn.wq.comfy_quant": torch.ones(8, dtype=torch.uint8),
        }, self.krea, metadata={"modelspec.architecture": "krea2-turbo"})

    def test_strict_rejects_but_common_layer_projects_int8_and_completes(self):
        with self.assertRaisesRegex(ValueError, "architecture|keys"):
            merge_models(self.base, self.krea, output=self.output, checkpoint_policy="strict")
        report = merge_models(self.base, self.krea, output=self.output,
                              checkpoint_policy="common-layer", weights=0.5)
        from safetensors.torch import load_file
        tensors = load_file(self.output)
        self.torch.testing.assert_close(
            tensors["transformer_blocks.0.attn.to_q.weight"],
            self.torch.tensor([[1.0, 2.0, 4.5], [6.0, 10.0, 12.0]]))
        self.torch.testing.assert_close(
            tensors["transformer_blocks.0.attn.to_q.bias"], self.torch.tensor([4.0, 8.0]))
        self.assertEqual(tensors["step_counter"].item(), 7)
        layer = report["common_layers"]["1"]
        self.assertEqual(layer["policy"], "base-layout-role-depth-crop-pad-v1")
        self.assertFalse(layer["semantic_equivalence"])
        self.assertGreater(layer["projected_tensors"], 0)
        self.assertEqual(report["checkpoint_policy"], "common-layer")

    def test_inspection_reports_mapping_without_loading_or_writing_output(self):
        request = resolve_merge_request(self.base, self.krea, output=self.output,
                                        checkpoint_policy="common-layer")
        report = inspect_merge_request(request)
        self.assertEqual(report["common_layers"]["1"]["source_tensors"], 3)
        self.assertFalse(self.output.exists())

    def test_cli_contract_is_explicit_and_invalid_files_still_fail(self):
        parsed = build_parser().parse_args([
            "--base-model", str(self.base), "--additional-model", str(self.krea),
            "--checkpoint-policy", "common-layer"])
        self.assertEqual(parsed.checkpoint_policy, "common-layer")
        broken = self.root / "broken.safetensors"
        broken.write_bytes(b"not a safetensors file")
        with self.assertRaisesRegex(ValueError, "Cannot read safetensors"):
            merge_models(self.base, broken, output=self.output, checkpoint_policy="common-layer")

    def test_report_embedded_in_output_declares_unstable_projection(self):
        merge_models(self.base, self.krea, output=self.output, checkpoint_policy="common-layer")
        from safetensors import safe_open
        with safe_open(self.output, framework="pt") as reader:
            recipe = json.loads(reader.metadata()["iild_merge"])
        self.assertEqual(recipe["checkpoint_policy"], "common-layer")
        self.assertFalse(recipe["common_layers"]["1"]["semantic_equivalence"])


if __name__ == "__main__":
    unittest.main()
