"""Architecture bridge regressions with real adapter arithmetic and publication."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from model_merge import inspect_merge_request, main, merge_models
from model_merge_options import resolve_merge_request
from iild_package import materialize_archive


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "safetensors")), "Requires tensor runtime")
class ModelMergeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import load_file, save_file
        self.t, self.save, self.load = torch, save_file, load_file
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="compatibility-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base, self.lora = self.root / "base.safetensors", self.root / "style.safetensors"
        self.bridge = self.root / "complete.safetensors"
        self.output = self.root / "result.iildmodel"
        self.module = "blocks.0.adaln_modulation_cross_attn.1"
        self.key = "model.diffusion_model.net." + self.module + ".weight"
        self.save({"unet.weight": torch.ones(2, 2)}, self.base)
        self.save({"diffusion_model." + self.module + ".lora_A.weight": torch.tensor([[1., 2.]]),
                   "diffusion_model." + self.module + ".lora_B.weight": torch.tensor([[1.], [2.], [3.]])}, self.lora)
        self.checkpoint(self.bridge, complete=True)

    def checkpoint(self, path, *, complete=False, shape=(3, 2), prefix="model.diffusion_model.net."):
        state = {prefix + self.module + ".weight": self.t.ones(*shape),
                 prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight": self.t.ones(2, 2)}
        if complete:
            state.update({"text_encoders.llm.model.embed_tokens.weight": self.t.ones(2, 2),
                          "vae.encoder.conv_in.weight": self.t.ones(2, 2),
                          "vae.decoder.conv_out.weight": self.t.ones(2, 2)})
        self.save(state, path)

    def request(self, **options):
        return resolve_merge_request(self.base, self.lora, mode="unified", output=self.output, **options)

    def test_registered_bridge_discovery_and_inspection_do_not_write_models(self):
        import model_merge_compatibility as module
        # The source location selects only the installation-local registry.
        location = self.root / "reference/diffusers/model_merge_compatibility.py"
        location.parent.mkdir(parents=True)
        (location.parent.parent / "merge-compatibility.json").write_text(json.dumps({"checkpoints": [str(self.bridge)]}))
        wrapped = self.root / "base.iildmodel"
        wrapped.mkdir()
        inner = wrapped / "model.safetensors"
        inner.write_bytes(self.base.read_bytes())
        with patch.object(module, "__file__", str(location)):
            discovered = module.LoraCompatibilityBridge.discover([inner], exclude=[self.base])
            self.assertIn(self.bridge, discovered)
            report = inspect_merge_request(resolve_merge_request(inner, self.lora,
                mode="unified", output=self.output))
        self.assertEqual(report["stages"][-1]["compatibility_bridge"]["checkpoint"], str(self.bridge))
        self.assertFalse(self.output.exists())

    def test_nested_anima_namespace_is_equivalent_for_direct_fusion(self):
        output = self.root / "fused.safetensors"
        merge_models(self.bridge, self.lora, output=output)
        self.t.testing.assert_close(self.load(output)[self.key], self.t.tensor([[2., 3.], [3., 5.], [4., 7.]]))

    def test_public_compatibility_object_reports_structural_gap(self):
        from model_merge_compatibility import LoraCompatibility
        compatible = LoraCompatibility.inspect(self.bridge, self.lora)
        self.assertTrue(compatible.compatible)
        self.assertEqual(compatible.target_count, 1)
        self.assertEqual(compatible.architecture, "anima")
        self.assertEqual(compatible.embedded_components, ("text_encoder", "vae"))
        mismatch = LoraCompatibility.inspect(self.base, self.lora)
        self.assertFalse(mismatch.compatible)
        self.assertIn(self.module, mismatch.reason)

    def test_discovery_prefers_complete_checkpoint_and_never_hashes_in_inspection(self):
        self.checkpoint(self.root / "partial.safetensors")
        with patch("model_merge_files.cached_model_sha256", side_effect=AssertionError("Unexpected hash")):
            report = inspect_merge_request(self.request())
        stage = report["stages"][1]
        self.assertEqual(stage["compatibility_bridge"]["checkpoint"], str(self.bridge))
        self.assertEqual(stage["compatibility_bridge"]["selection_reason"], "complete-runtime-components")
        self.assertEqual(stage["loras"], [{"source_index": 1, "strength": 1.0}])
        self.assertEqual(stage["strength"], 0.35)
        self.assertEqual(report["weights"], [1.0])
        self.assertFalse(self.output.exists())

    def test_bridge_fuses_all_targets_preserves_inputs_and_records_source(self):
        before = {p: p.read_bytes() for p in (self.base, self.lora, self.bridge)}
        report = merge_models(self.base, self.lora, mode="unified", output=self.output, compatibility_strength=0.2)
        self.assertEqual(len(report["stages"]), 2)
        self.assertEqual(len(report["sources"]), 3)
        self.assertEqual(report["sources"][2]["path"], str(self.bridge))
        stage = report["stages"][1]
        self.assertEqual(stage["source_index"], 2)
        self.assertEqual(stage["strength"], 0.2)
        self.assertEqual(report["compatibility_bridge_count"], 1)
        package = materialize_archive(self.output, self.root / "package-cache")
        self.t.testing.assert_close(self.load(package / stage["model"])[self.key],
                                   self.t.tensor([[2., 3.], [3., 5.], [4., 7.]]))
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)
        from unified_image import inspect_package
        inspect_package(self.output, hashes=True)

    def test_multiple_loras_share_bridge_without_changing_requested_strengths(self):
        other = self.root / "other-lora.safetensors"
        other.write_bytes(self.lora.read_bytes())
        report = merge_models(self.base, self.lora, additional_models=[other], weights=[0.5, 2.0],
                              mode="unified", output=self.output)
        self.assertEqual(report["compatibility_bridge_count"], 1)
        self.assertEqual(len(report["stages"]), 2)
        self.assertEqual(report["weights"], [0.5, 2.0])
        self.assertEqual(len(report["stages"][1]["loras"]), 2)
        package = materialize_archive(self.output, self.root / "package-cache")
        self.t.testing.assert_close(self.load(package / report["stages"][1]["model"])[self.key],
                                   1 + 2.5 * self.t.tensor([[1., 2.], [2., 4.], [3., 6.]]))

    def test_bridge_keeps_material_order_and_explicit_checkpoints_take_priority(self):
        later = self.root / "later.safetensors"
        self.save({"different.weight": self.t.ones(2, 2)}, later)
        report = inspect_merge_request(self.request(additional_models=[later]))
        self.assertEqual([s["source_index"] for s in report["stages"]], [0, 3, 2])
        report = inspect_merge_request(self.request(additional_models=[self.bridge]))
        self.assertEqual(len(report["stages"]), 2)
        self.assertNotIn("compatibility_bridge", report["stages"][1])

    def test_ambiguous_candidates_require_explicit_choice_and_publish_nothing(self):
        other = self.root / "equally-complete.safetensors"
        self.checkpoint(other, complete=True)
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous.*compatibility"):
            merge_models(self.base, self.lora, mode="unified", output=self.output)
        self.assertFalse(self.output.exists())
        report = merge_models(self.base, self.lora, mode="unified", output=self.output, compatibility_models=[self.bridge])
        self.assertEqual(report["stages"][1]["compatibility_bridge"]["checkpoint"], str(self.bridge))

    def test_all_shapes_must_match_and_strict_mode_disables_discovery(self):
        with self.assertRaisesRegex(ValueError, "no compatible checkpoint"):
            inspect_merge_request(self.request(compatibility_models=[]))
        self.checkpoint(self.bridge, complete=True, shape=(4, 2))
        with self.assertRaisesRegex(ValueError, "no compatible checkpoint"):
            merge_models(self.base, self.lora, mode="unified", output=self.output)
        self.assertFalse(self.output.exists())

    def test_bad_adapter_and_nonfinite_bridge_cannot_publish(self):
        state = self.load(self.bridge)
        state[self.key][0, 0] = float("nan")
        self.save(state, self.bridge)
        with self.assertRaisesRegex(ValueError, "finite"):
            merge_models(self.base, self.lora, mode="unified", output=self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".result.iildmodel-*")))

    def test_unrelated_corrupt_file_does_not_block_discovery(self):
        (self.root / "broken.safetensors").write_text("not a checkpoint")
        report = inspect_merge_request(self.request())
        self.assertEqual(report["stages"][1]["compatibility_bridge"]["checkpoint"], str(self.bridge))

    def test_ambiguous_export_aliases_are_not_chosen_by_shape(self):
        state = self.load(self.bridge)
        state["diffusion_model." + self.module + ".weight"] = state[self.key].clone()
        self.save(state, self.bridge)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            merge_models(self.base, self.lora, mode="unified", output=self.output)
        self.assertFalse(self.output.exists())

    def test_changed_bridge_source_is_revalidated_before_publication(self):
        import model_merge_files
        original_verify = model_merge_files.MergeModelFiles.verify
        def change_bridge(model):
            if model.root == self.bridge:
                self.checkpoint(self.bridge, complete=True, shape=(4, 2))
            return original_verify(model)
        with patch.object(model_merge_files.MergeModelFiles, "verify", change_bridge):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                merge_models(self.base, self.lora, mode="unified", output=self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".result.iildmodel-*")))

    def test_lora_strength_zero_preserves_checkpoint_and_still_validates_shapes(self):
        report = merge_models(self.base, self.lora, mode="unified", weights=0, output=self.output)
        stage = report["stages"][1]
        self.assertEqual(stage["loras"][0]["strength"], 0)
        package = materialize_archive(self.output, self.root / "package-cache")
        self.t.testing.assert_close(self.load(package / stage["model"])[self.key], self.load(self.bridge)[self.key])

    def test_discovery_does_not_recurse_into_packaged_members_or_follow_symlinks(self):
        candidate = self.root / "nested.iildmodel" / "members" / "model.safetensors"
        candidate.parent.mkdir(parents=True)
        self.bridge.rename(candidate)
        self.bridge.symlink_to(candidate)
        with self.assertRaisesRegex(ValueError, "no compatible checkpoint"):
            inspect_merge_request(self.request())
        report = inspect_merge_request(self.request(compatibility_models=[candidate]))
        self.assertEqual(len(report["stages"]), 2)

    def test_cli_explicit_bridge_and_strict_policy(self):
        args = ["--base-model", str(self.base), "--additional-model", str(self.lora),
                "--mode", "unified", "--output", str(self.output), "--inspect"]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(args + ["--compatibility-model", str(self.bridge), "--compatibility-strength", "0.4"]), 0)
        self.assertEqual(json.loads(output.getvalue())["stages"][1]["strength"], 0.4)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args + ["--no-auto-compatibility"])

    def test_invalid_bridge_options_are_rejected_without_tensor_loading(self):
        for value in (-0.1, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.request(compatibility_strength=value)
        with self.assertRaises((TypeError, ValueError)):
            self.request(compatibility_models=str(self.bridge))
        with self.assertRaisesRegex(ValueError, "unified"):
            resolve_merge_request(self.base, self.lora, compatibility_models=[self.bridge])


if __name__ == "__main__":
    unittest.main()
