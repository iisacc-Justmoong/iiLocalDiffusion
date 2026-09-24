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
            self.torch.tensor([[1.0, 4.5, 10.0], [2.0, 6.0, 12.0]]))
        self.torch.testing.assert_close(
            tensors["transformer_blocks.0.attn.to_q.bias"], self.torch.tensor([4.0, 8.0]))
        self.assertEqual(tensors["step_counter"].item(), 7)
        layer = report["common_layers"]["1"]
        self.assertEqual(layer["policy"], "base-layout-normalize-project-flatten-v3")
        self.assertFalse(layer["semantic_equivalence"])
        self.assertGreater(layer["projected_tensors"], 0)
        self.assertEqual(report["checkpoint_policy"], "common-layer")

    def test_float8_base_is_checked_in_accumulation_dtype_and_preserves_storage_dtype(self):
        self.save({
            "transformer_blocks.0.attn.to_q.weight": self.torch.zeros(
                (2, 3), dtype=self.torch.float8_e4m3fn),
        }, self.base, metadata={"modelspec.architecture": "flux2-fp8"})
        report = merge_models(self.base, self.krea, output=self.output,
                              checkpoint_policy="common-layer", weights=0.5)
        from safetensors.torch import load_file
        tensor = load_file(self.output)["transformer_blocks.0.attn.to_q.weight"]
        self.assertEqual(tensor.dtype, self.torch.float8_e4m3fn)
        self.torch.testing.assert_close(
            tensor.float(), self.torch.tensor([[1.0, 4.5, 10.0], [2.0, 6.0, 12.0]]))
        self.assertEqual(report["merged_tensor_count"], 1)
        self.assertEqual(report["changed_tensor_count"], 1)

    def test_inspection_reports_mapping_without_loading_or_writing_output(self):
        request = resolve_merge_request(self.base, self.krea, output=self.output,
                                        checkpoint_policy="common-layer")
        report = inspect_merge_request(request)
        self.assertEqual(report["common_layers"]["1"]["source_tensors"], 3)
        self.assertFalse(self.output.exists())

    def test_inspection_distinguishes_projection_and_lists_components(self):
        report = inspect_merge_request(resolve_merge_request(self.base, self.krea, output=self.output))
        self.assertEqual(report["resource_compatibility"][0]["status"], "conditional")
        self.assertEqual(report["base_profile"]["components"]["dit"]["tensor_count"], 2)
        self.assertEqual(report["preflight"]["output_contract"], "base-tensor-layout")
        self.assertTrue(report["preflight"]["risks"])

    def test_component_ownership_and_architecture_evidence(self):
        for family, width in (("sd1", 768), ("sd2", 1024), ("sdxl", 2048)):
            with self.subTest(family=family):
                tensors = {
                    "model.diffusion_model.input_blocks.0.transformer_blocks.0.attn2.to_k.weight": self.torch.ones((2, width)),
                    "first_stage_model.encoder.conv.weight": self.torch.ones((2, 2)),
                    "conditioner.embedders.0.transformer.text_model.layer.weight": self.torch.ones((2, 2)),
                    "conditioner.embedders.1.transformer.text_model.layer.weight": self.torch.ones((2, 2)),
                    "model.diffusion_model.label_emb.weight": self.torch.ones((2, 2)),
                }
                self.save(tensors, self.base)
                self.save(tensors, self.krea)
                report = inspect_merge_request(resolve_merge_request(self.base, self.krea, output=self.output))
                self.assertEqual(report["base_profile"]["ecosystems"], [family])
                groups = report["base_profile"]["components"]
                self.assertEqual([groups[name]["tensor_count"] for name in ("unet", "dit", "vae", "text_encoder")], [2, 0, 1, 2])
                self.assertEqual(report["resource_compatibility"][0]["status"], "compatible")

    def test_output_is_reopened_and_all_tensor_values_verified(self):
        report = merge_models(self.base, self.krea, output=self.output)
        verification = report["output_verification"]
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(verification["tensor_count"], 3)
        self.assertEqual(verification["changed_tensor_count"], report["changed_tensor_count"])
        self.assertEqual(verification["method"], "reopen-all-tensors-exact-expected-values")

    def test_dit_family_signatures_do_not_require_loading_large_tensors(self):
        from model_merge_diagnostics import model_profile
        class HeaderSlice:
            def __init__(self, shape): self.shape = shape
            def get_shape(self): return list(self.shape)
            def get_dtype(self): return "F16"
        class HeaderReader:
            def __init__(self, shapes): self.shapes = shapes
            def metadata(self): return {}
            def get_slice(self, key): return HeaderSlice(self.shapes[key])
        cases = [
            ("flux1", {"double_blocks.0.img_attn.qkv.weight": (9216, 3072)}),
            ("flux2", {"double_blocks.0.img_attn.qkv.weight": (12288, 4096)}),
            ("flux2", {"blocks.0.attn.wq.weight": (4096, 4096), "tproj.weight": (4096, 256)}),
            ("anima", {"llm_adapter.blocks.0.cross_attn.to_q.weight": (1024, 1024)}),
            ("sd3", {"joint_blocks.0.attn.qkv.weight": (3072, 1024)}),
        ]
        for family, shapes in cases:
            with self.subTest(family=family):
                profile = model_profile({"header": HeaderReader(shapes)}, {(".", key): "header" for key in shapes})
                self.assertEqual(profile["ecosystems"], [family])
                self.assertEqual(profile["components"]["dit"]["tensor_count"], len(shapes))

    def test_corrupt_serializer_result_is_not_published(self):
        from unittest.mock import patch
        def corrupt(tensors, filename, **kwargs):
            wrong = {key: value.clone() for key, value in tensors.items()}
            wrong["transformer_blocks.0.attn.to_q.weight"].zero_()
            self.save(wrong, filename, **kwargs)
        with patch("safetensors.torch.save_file", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "Saved tensor values differ"):
                merge_models(self.base, self.krea, output=self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_quantization_metadata_does_not_stop_valid_material(self):
        # A selected material with an empty scale must fall back without
        # publishing invalid values or claiming an effective blend.
        self.save({"blocks.4.attn.wq.weight": self.torch.ones((3, 2), dtype=self.torch.int8),
                   "blocks.4.attn.wq.weight_scale": self.torch.empty(0)}, self.krea,
                  metadata={"modelspec.architecture": "krea2"})
        report = merge_models(self.base, self.krea, output=self.output)
        self.assertEqual(report["result_kind"], "base-fallback")
        self.assertEqual(report["output_verification"]["status"], "passed")

    def test_cli_contract_is_explicit_and_invalid_files_still_fail(self):
        defaulted = build_parser().parse_args([
            "--base-model", str(self.base), "--additional-model", str(self.krea)])
        self.assertEqual(defaulted.checkpoint_policy, "common-layer")
        parsed = build_parser().parse_args([
            "--base-model", str(self.base), "--additional-model", str(self.krea),
            "--checkpoint-policy", "common-layer"])
        self.assertEqual(parsed.checkpoint_policy, "common-layer")
        broken = self.root / "broken.safetensors"
        broken.write_bytes(b"not a safetensors file")
        with self.assertRaisesRegex(ValueError, "Cannot read safetensors"):
            merge_models(self.base, broken, output=self.output, checkpoint_policy="strict")

    def test_missing_schedule_coordinate_preserves_base_by_default(self):
        schedule_base = self.root / "schedule-base.safetensors"
        schedule_material = self.root / "schedule-material.safetensors"
        self.save({
            "denoiser.sigmas": self.torch.tensor([1.0, 0.5, 0.0]),
            "transformer_blocks.0.attn.to_q.weight": self.torch.zeros((2, 3)),
        }, schedule_base, metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"})
        self.save({
            "transformer_blocks.0.attn.to_q.weight": self.torch.ones((2, 3)),
            "conditioner.embedders.1.model.attn.in_proj_bias": self.torch.arange(8.0),
        }, schedule_material, metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"})
        report = merge_models(schedule_base, schedule_material, output=self.output, weights=0.5)
        from safetensors.torch import load_file
        tensors = load_file(self.output)
        self.torch.testing.assert_close(
            tensors["denoiser.sigmas"], self.torch.tensor([1.0, 0.5, 0.0]))
        self.torch.testing.assert_close(
            tensors["transformer_blocks.0.attn.to_q.weight"], self.torch.full((2, 3), 0.5))
        layer = report["common_layers"]["1"]
        self.assertEqual(layer["base_preserved_tensors"], 1)
        self.assertEqual(layer["mapping_examples"][0]["selection"],
                         "base-preserved-no-semantic-source")

    def test_report_embedded_in_output_declares_unstable_projection(self):
        merge_models(self.base, self.krea, output=self.output, checkpoint_policy="common-layer")
        from safetensors import safe_open
        with safe_open(self.output, framework="pt") as reader:
            recipe = json.loads(reader.metadata()["iild_merge"])
        self.assertEqual(recipe["checkpoint_policy"], "common-layer")
        self.assertFalse(recipe["common_layers"]["1"]["semantic_equivalence"])

    def test_base_ecosystem_keeps_compatible_materials_and_reports_exclusions(self):
        compatible = self.root / "flux-compatible.safetensors"
        incompatible = self.root / "sdxl-incompatible.safetensors"
        self.save({
            "transformer_blocks.0.attn.to_q.weight": self.torch.ones((2, 3)),
            "transformer_blocks.0.attn.to_q.bias": self.torch.tensor([4.0, 8.0]),
            "step_counter": self.torch.tensor(7, dtype=self.torch.int64),
        }, compatible, metadata={"modelspec.architecture": "flux2"})
        self.save({"transformer_blocks.0.attn.to_q.weight": self.torch.full((2, 3), 99.0)},
                  incompatible, metadata={"modelspec.architecture": "stable-diffusion-xl"})
        inspection = inspect_merge_request(resolve_merge_request(
            self.base, compatible, additional_models=[incompatible], weights=[0.25, 0.25],
            output=self.output))
        self.assertEqual(inspection["included_material_count"], 1)
        self.assertEqual(inspection["excluded_material_count"], 1)
        self.assertEqual(inspection["weights"], [0.25])
        self.assertEqual(inspection["base_weight"], 0.75)
        self.assertFalse(inspection["resource_compatibility"][1]["compatible"])
        report = merge_models(self.base, compatible, additional_models=[incompatible],
                              weights=[0.25, 0.25], output=self.output)
        from safetensors.torch import load_file
        self.torch.testing.assert_close(
            load_file(self.output)["transformer_blocks.0.attn.to_q.weight"],
            self.torch.full((2, 3), 0.25))
        self.assertEqual(report["excluded_sources"][0]["path"], str(incompatible))

    def test_strict_merge_stops_when_every_material_is_incompatible(self):
        incompatible = self.root / "sdxl-only.safetensors"
        self.save({"foreign.weight": self.torch.ones(2)}, incompatible,
                  metadata={"modelspec.architecture": "stable-diffusion-xl"})
        with self.assertRaisesRegex(ValueError, "No merge material is compatible"):
            merge_models(self.base, incompatible, output=self.output, checkpoint_policy="strict")
        self.assertFalse(self.output.exists())

    def merge_fixture(self, base, material, **kwargs):
        from safetensors.torch import load_file
        metadata = {"modelspec.architecture": "flux2"}
        self.save(base, self.base, metadata=metadata)
        self.save(material, self.krea, metadata=metadata)
        report = merge_models(self.base, self.krea, output=self.output, **kwargs)
        return load_file(self.output), report

    def test_transposed_matrix_is_aligned_before_sampling(self):
        tensors, report = self.merge_fixture(
            {"attn.to_q.weight": self.torch.zeros(2, 3)},
            {"attn.to_q.weight": self.torch.arange(6.0).reshape(3, 2)})
        self.torch.testing.assert_close(tensors["attn.to_q.weight"],
                                       self.torch.tensor([[0., 1., 2.], [.5, 1.5, 2.5]]))
        self.assertEqual(report["common_layers"]["1"]["transform_counts"]["transpose"], 1)

    def test_rank_change_uses_flattened_coordinates(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.zeros(2, 2)},
            {"layer.weight": self.torch.tensor([2., 4., 6.])})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([[1., 2.], [2., 3.]]))
        self.assertEqual(report["common_layers"]["1"]["transform_counts"]["flatten-resample"], 1)

    def test_equal_element_rank_change_reshapes_without_repeating_values(self):
        tensors, _ = self.merge_fixture(
            {"layer.weight": self.torch.zeros(2, 2)},
            {"layer.weight": self.torch.arange(4.0).reshape(1, 4, 1)})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.arange(4.0).reshape(2, 2) / 2)

    def test_scalar_and_singleton_coordinates(self):
        tensors, _ = self.merge_fixture(
            {"scale.weight": self.torch.tensor(0.), "layer.weight": self.torch.zeros(2, 3)},
            {"scale.weight": self.torch.tensor([2., 4.]), "layer.weight": self.torch.tensor(6.)})
        self.torch.testing.assert_close(tensors["scale.weight"], self.torch.tensor(1.))
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.full((2, 3), 3.))

    def test_empty_source_and_missing_tensor_do_not_subtract_base(self):
        tensors, report = self.merge_fixture(
            {"attn.to_q.weight": self.torch.full((2, 2), 8.),
             "attn.to_k.bias": self.torch.tensor([3.]), "v_pred": self.torch.empty(0)},
            {"attn.to_q.weight": self.torch.empty(0, 2)}, mode="weighted-difference")
        self.torch.testing.assert_close(tensors["attn.to_q.weight"], self.torch.full((2, 2), 8.))
        self.torch.testing.assert_close(tensors["attn.to_k.bias"], self.torch.tensor([3.]))
        self.assertEqual(report["changed_tensor_count"], 0)
        self.assertEqual(report["merged_tensor_count"], 0)

    def test_schedule_and_quantization_metadata_are_preserved_even_when_present(self):
        tensors, _ = self.merge_fixture(
            {"denoiser.sigmas": self.torch.tensor([1., .5, 0.]),
             "attn.to_q.weight": self.torch.ones(2, 2)},
            {"denoiser.sigmas": self.torch.tensor([5., 3., 1.]),
             "attn.to_q.weight": self.torch.ones(2, 2)}, mode="weighted-difference")
        self.torch.testing.assert_close(tensors["denoiser.sigmas"], self.torch.tensor([1., .5, 0.]))

    def test_normalized_namespace_wins_over_shape_only_candidate(self):
        tensors, report = self.merge_fixture(
            {"model.diffusion_model.blocks.0.attn.to_q.weight": self.torch.zeros(2, 2)},
            {"module.blocks.0.attn.to_q.weight": self.torch.full((2, 2), 4.),
             "other.blocks.0.attn.to_q.weight": self.torch.full((2, 2), 99.)})
        self.torch.testing.assert_close(next(iter(tensors.values())), self.torch.full((2, 2), 2.))
        self.assertEqual(report["common_layers"]["1"]["mapping_examples"][0]["selection"], "normalized-name")

    def test_text_weights_are_not_borrowed_for_missing_denoiser_weights(self):
        tensors, _ = self.merge_fixture(
            {"attn.to_q.weight": self.torch.full((2, 2), 7.)},
            {"text_encoder.attn.to_q.weight": self.torch.full((2, 2), 99.)})
        self.torch.testing.assert_close(tensors["attn.to_q.weight"], self.torch.full((2, 2), 7.))

    def test_depth_is_normalized_between_different_block_counts(self):
        tensors, _ = self.merge_fixture(
            {f"blocks.{i}.attn.to_q.weight": self.torch.zeros(1) for i in (0, 1, 2)},
            {f"layers.{i}.attn.wq.weight": self.torch.tensor([float(i)]) for i in (0, 2, 4)})
        self.torch.testing.assert_close(tensors["blocks.1.attn.to_q.weight"], self.torch.tensor([1.]))
        self.torch.testing.assert_close(tensors["blocks.2.attn.to_q.weight"], self.torch.tensor([2.]))

    def test_float8_material_scale_is_applied_in_float32(self):
        tensors, _ = self.merge_fixture(
            {"layer.weight": self.torch.zeros(2, 2)},
            {"layer.weight": self.torch.full((2, 2), 2., dtype=self.torch.float8_e4m3fn),
             "layer.weight_scale": self.torch.tensor([3., 4.])})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([[3., 3.], [4., 4.]]))

    def test_quantized_base_is_dequantized_and_reencoded_with_base_scale(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([[5, 9], [5, 9]], dtype=self.torch.uint8),
             "layer.weight_scale": self.torch.tensor([2., 4.]),
             "layer.weight_zero_point": self.torch.tensor(1.)},
            {"layer.weight": self.torch.zeros(2, 2)})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([[3, 5], [3, 5]], dtype=self.torch.uint8))
        self.torch.testing.assert_close(tensors["layer.weight_scale"], self.torch.tensor([2., 4.]))
        self.assertEqual(report["changed_tensor_count"], 1)

    def test_block_quantization_uses_contiguous_scale_groups(self):
        tensors, _ = self.merge_fixture(
            {"layer.weight": self.torch.zeros(2, 4)},
            {"layer.weight": self.torch.ones((2, 4), dtype=self.torch.int8),
             "layer.weight_scale": self.torch.tensor([[2., 4.], [6., 8.]])})
        self.torch.testing.assert_close(tensors["layer.weight"],
                                       self.torch.tensor([[1., 1., 2., 2.], [3., 3., 4., 4.]]))

    def test_missing_difference_material_is_neutral_with_another_active_material(self):
        self.save({"attn.to_k.weight": self.torch.ones(2, 2)}, self.root / "third.safetensors",
                  metadata={"modelspec.architecture": "flux2"})
        tensors, _ = self.merge_fixture(
            {"attn.to_q.weight": self.torch.full((2, 2), 8.)},
            {"attn.to_q.weight": self.torch.full((2, 2), 2.)},
            additional_models=[self.root / "third.safetensors"], weights=[.5, .5], mode="weighted-difference")
        self.torch.testing.assert_close(tensors["attn.to_q.weight"], self.torch.full((2, 2), 7.))

    def test_invalid_material_quantizer_is_preserved_and_reported(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.full((2, 2), 8.)},
            {"layer.weight": self.torch.ones((2, 2), dtype=self.torch.int8),
             "layer.weight_scale": self.torch.tensor(0.)}, mode="weighted-difference")
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.full((2, 2), 8.))
        self.assertEqual(report["common_layers"]["1"]["runtime_preserved_tensors"], 1)
        self.assertIn("positive", report["common_layers"]["1"]["runtime_events"][0]["reason"])

    def test_invalid_base_quantizer_is_preserved_and_reported(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.ones((2, 2), dtype=self.torch.int8),
             "layer.weight_scale": self.torch.tensor(0.)},
            {"layer.weight": self.torch.zeros(2, 2)})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.ones((2, 2), dtype=self.torch.int8))
        self.assertEqual(report["numeric_normalization"]["invalid_base_quantization_tensors"], 1)

    def test_same_rank_axis_resampling_is_deterministic(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.zeros(4, 2)},
            {"layer.weight": self.torch.arange(15.0).reshape(3, 5)})
        self.torch.testing.assert_close(tensors["layer.weight"],
                                       self.torch.tensor([[0., 4.], [5., 9.], [5., 9.], [10., 14.]]) / 2)
        self.assertEqual(report["common_layers"]["1"]["transform_counts"]["axis-resample"], 1)

    def test_float64_material_is_not_rounded_before_accumulation(self):
        tensors, _ = self.merge_fixture(
            {"layer.weight": self.torch.ones(1, dtype=self.torch.float64)},
            {"layer.weight": self.torch.tensor([1. + 1e-12], dtype=self.torch.float64)})
        self.assertEqual(tensors["layer.weight"].item(), 1. + 5e-13)

    def test_shape_projection_matrix_is_finite_deterministic_and_base_shaped(self):
        from model_merge_tensor import project_tensor
        shapes = [(), (1,), (3,), (2, 3), (3, 2), (4, 1), (1, 2, 3), (2, 2, 2, 1)]
        import math
        for source in shapes:
            value = self.torch.arange(math.prod(source), dtype=self.torch.float32).reshape(source)
            for target in shapes:
                with self.subTest(source=source, target=target):
                    result = project_tensor(self.torch, value, target)
                    self.assertEqual(tuple(result.shape), target)
                    self.assertTrue(self.torch.isfinite(result).all())
                    self.assertTrue(self.torch.equal(result, project_tensor(self.torch, value, target)))

    def test_flatten_resampling_across_chunk_boundaries(self):
        from unittest.mock import patch
        from model_merge_tensor import project_tensor
        with patch("model_merge_tensor._COORDINATE_CHUNK", 3):
            value = project_tensor(self.torch, self.torch.tensor([0., 2., 4.]), (2, 4))
        self.torch.testing.assert_close(value, self.torch.tensor([[0., 0., 2., 2.], [2., 2., 4., 4.]]))

    def test_package_stage_names_cannot_be_cross_mapped(self):
        from contextlib import ExitStack
        from safetensors import safe_open
        from model_merge_common import project_checkpoint
        with ExitStack() as stack:
            left = stack.enter_context(safe_open(self.base, framework="pt"))
            right = stack.enter_context(safe_open(self.krea, framework="pt"))
            _, _, report = project_checkpoint(
                {"material": right}, {("stage2", key): "material" for key in right.keys()},
                {"base": left}, {("stage1", key): "base" for key in left.keys()})
        self.assertEqual(report["base_preserved_tensors"], 3)

    def test_nonfinite_material_coordinates_are_neutral_in_both_modes(self):
        from safetensors import safe_open
        for mode, expected in (("weighted-sum", [2., 4., 6., 10.]),
                               ("weighted-difference", [2., 4., 6., 2.])):
            with self.subTest(mode=mode):
                self.output = self.root / (mode + ".safetensors")
                tensors, report = self.merge_fixture(
                    {"layer.weight": self.torch.tensor([2., 4., 6., 8.])},
                    {"layer.weight": self.torch.tensor([float("nan"), float("inf"), -float("inf"), 12.])},
                    mode=mode, weights=.5)
                self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor(expected), rtol=0, atol=0)
                info = report["numeric_normalization"]
                self.assertEqual(info["nonfinite_material_values"], 3)
                self.assertEqual(info["nonfinite_material_tensors"], 1)
                example = info["nonfinite_material_examples"][0]
                self.assertEqual(example["source_index"], 1)
                self.assertEqual(example["nan_values"], 1)
                self.assertEqual(example["positive_infinity_values"], 1)
                self.assertEqual(example["negative_infinity_values"], 1)
                self.assertEqual(example["coordinates"], [[0], [1], [2]])
                with safe_open(self.output, framework="pt") as reader:
                    embedded = json.loads(reader.metadata()["iild_merge"])
                self.assertEqual(embedded["numeric_normalization"], info)
                with safe_open(self.krea, framework="pt") as reader:
                    self.assertTrue(self.torch.isnan(reader.get_tensor("layer.weight")[0]))

    def test_wholly_nonfinite_material_preserves_base_without_counting_a_merge(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.full((2, 2), 8.)},
            {"layer.weight": self.torch.full((2, 2), float("nan"))}, weights=1.)
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.full((2, 2), 8.))
        self.assertEqual(report["merged_tensor_count"], 0)
        self.assertEqual(report["changed_tensor_count"], 0)
        self.assertEqual(report["numeric_normalization"]["nonfinite_material_values"], 4)

    def test_repair_does_not_discard_other_material_contributions(self):
        third = self.root / "third.safetensors"
        self.save({"layer.weight": self.torch.tensor([12., 12.])}, third,
                  metadata={"modelspec.architecture": "flux2"})
        tensors, _ = self.merge_fixture(
            {"layer.weight": self.torch.tensor([8., 8.])},
            {"layer.weight": self.torch.tensor([float("nan"), 4.])},
            additional_models=[third], weights=[.25, .25])
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([9., 8.]))

    def test_projected_nonfinite_values_use_base_coordinates(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.full((2, 2), 8.)},
            {"layer.weight": self.torch.tensor([2., float("nan"), 6.])})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([[5., 8.], [8., 7.]]))
        example = report["numeric_normalization"]["nonfinite_material_examples"][0]
        self.assertEqual(example["coordinates"], [[0, 1], [1, 0]])

    def test_float8_nonfinite_material_repair_preserves_output_storage(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([2., 2.], dtype=self.torch.float8_e4m3fn)},
            {"layer.weight": self.torch.tensor([float("nan"), 4.], dtype=self.torch.float8_e4m3fn)})
        self.assertEqual(tensors["layer.weight"].dtype, self.torch.float8_e4m3fn)
        self.torch.testing.assert_close(tensors["layer.weight"].float(), self.torch.tensor([2., 3.]))
        self.assertEqual(report["numeric_normalization"]["nonfinite_material_values"], 1)

    def test_bad_coordinate_examples_are_bounded(self):
        _, report = self.merge_fixture(
            {"layer.weight": self.torch.ones(131080)},
            {"layer.weight": self.torch.full((131080,), float("nan"))})
        info = report["numeric_normalization"]
        self.assertEqual(info["nonfinite_material_values"], 131080)
        self.assertEqual(len(info["nonfinite_material_examples"][0]["coordinates"]), 8)

    def test_nonfinite_base_is_zero_filled_before_merging(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([float("nan"), float("inf"), -float("inf"), 8.])},
            {"layer.weight": self.torch.tensor([2., 4., 6., 12.])})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([1., 2., 3., 10.]))
        self.assertEqual(report["numeric_normalization"]["base_nonfinite_values"], 3)

    def test_nonfinite_base_with_no_usable_material_is_repaired_base_output(self):
        self.save({"foreign.weight": self.torch.ones(2)}, self.krea,
                  metadata={"modelspec.architecture": "sdxl"})
        self.save({"layer.weight": self.torch.tensor([float("nan"), 8.])}, self.base,
                  metadata={"modelspec.architecture": "flux2"})
        report = merge_models(self.base, self.krea, output=self.output)
        from safetensors.torch import load_file
        self.torch.testing.assert_close(load_file(self.output)["layer.weight"], self.torch.tensor([0., 8.]))
        self.assertEqual(report["result_kind"], "repaired-base")

    def test_storage_overflow_saturates_instead_of_aborting(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([60000., -60000.], dtype=self.torch.float16)},
            {"layer.weight": self.torch.tensor([-60000., 60000.], dtype=self.torch.float16)},
            mode="weighted-difference", weights=1.)
        self.torch.testing.assert_close(tensors["layer.weight"].float(), self.torch.tensor([65504., -65504.]))
        self.assertEqual(report["numeric_normalization"]["output_clipped_values"], 2)

    def test_arithmetic_overflow_has_finite_output(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([1., 2.], dtype=self.torch.float64)},
            {"layer.weight": self.torch.tensor([1e308, -1e308], dtype=self.torch.float64)},
            mode="weighted-difference", weights=1e308)
        self.assertTrue(self.torch.isfinite(tensors["layer.weight"]).all())
        self.assertEqual(report["numeric_normalization"]["output_nonfinite_values"], 2)

    def test_all_unusable_material_files_become_reported_base_fallback(self):
        from safetensors.torch import load_file
        for index, contents in enumerate((b"", b"not a checkpoint", None)):
            with self.subTest(contents=contents):
                material = self.root / f"unusable-{index}.safetensors"
                if contents is not None:
                    material.write_bytes(contents)
                output = self.root / f"fallback-{index}.safetensors"
                report = merge_models(self.base, material, output=output)
                self.assertEqual(report["included_material_count"], 0)
                self.assertEqual(report["excluded_material_count"], 1)
                self.assertEqual(report["result_kind"], "base-fallback")
                self.assertTrue(report["no_effect"])
                self.assertEqual(report["weights"], [])
                self.torch.testing.assert_close(load_file(output)["transformer_blocks.0.attn.to_q.weight"],
                                               self.torch.zeros(2, 3))

    def test_unusable_material_keeps_valid_material_coefficient(self):
        broken = self.root / "broken.safetensors"
        broken.write_bytes(b"broken")
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([4.])},
            {"layer.weight": self.torch.tensor([8.])},
            additional_models=[broken], weights=[.25, .5])
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([5.]))
        self.assertEqual(report["weights"], [.25])
        self.assertEqual(report["excluded_material_count"], 1)

    def test_oversubscribed_weights_are_ratio_normalized_without_overflow(self):
        third = self.root / "third.safetensors"
        self.save({"layer.weight": self.torch.tensor([10.])}, third, metadata={"modelspec.architecture": "flux2"})
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([100.])},
            {"layer.weight": self.torch.tensor([2.])}, additional_models=[third], weights=[1e308, 1e308])
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.tensor([6.]))
        self.assertEqual(report["weights"], [.5, .5])
        self.assertEqual(report["base_weight"], 0.)
        self.assertEqual(report["weight_normalization"]["requested_weights"], [1e308, 1e308])

    def test_invalid_lora_values_are_zero_filled_without_losing_valid_delta(self):
        t = self.torch
        tensors, report = self.merge_fixture(
            {"layer.weight": t.zeros(2, 2)},
            {"layer.lora_A.weight": t.tensor([[float("nan"), 2.]]),
             "layer.lora_B.weight": t.tensor([[1.], [2.]])})
        t.testing.assert_close(tensors["layer.weight"], t.tensor([[0., 2.], [0., 4.]]))
        self.assertEqual(report["numeric_normalization"]["lora_nonfinite_values"], 1)

    def test_lora_delta_overflow_is_zero_filled(self):
        t = self.torch
        tensors, report = self.merge_fixture(
            {"layer.weight": t.ones(2, 2)},
            {"layer.lora_A.weight": t.full((1, 2), 1e30), "layer.lora_B.weight": t.full((2, 1), 1e30)})
        self.assertTrue(t.isfinite(tensors["layer.weight"]).all())
        self.assertEqual(report["numeric_normalization"]["lora_delta_nonfinite_values"], 4)

    def test_zero_repaired_lora_does_not_claim_effective_material_blend(self):
        t = self.torch
        tensors, report = self.merge_fixture(
            {"layer.weight": t.full((2, 2), float("nan"))},
            {"layer.lora_A.weight": t.full((1, 2), float("nan")), "layer.lora_B.weight": t.ones(2, 1)})
        t.testing.assert_close(tensors["layer.weight"], t.zeros(2, 2))
        self.assertEqual(report["merged_tensor_count"], 0)
        self.assertEqual(report["changed_tensor_count"], 1)
        self.assertEqual(report["result_kind"], "repaired-base")

    def test_preserved_float_state_is_also_finite(self):
        tensors, report = self.merge_fixture(
            {"denoiser.sigmas": self.torch.tensor([float("nan"), 1.]), "layer.weight": self.torch.ones(2)},
            {"layer.weight": self.torch.ones(2)})
        self.torch.testing.assert_close(tensors["denoiser.sigmas"], self.torch.tensor([0., 1.]))
        self.assertEqual(report["numeric_normalization"]["base_nonfinite_values"], 1)

    def test_float8_output_overflow_and_invalid_base_are_repaired(self):
        tensors, report = self.merge_fixture(
            {"layer.weight": self.torch.tensor([float("nan"), 400.], dtype=self.torch.float8_e4m3fn)},
            {"layer.weight": self.torch.tensor([2., -400.])}, mode="weighted-difference", weights=2.)
        self.torch.testing.assert_close(tensors["layer.weight"].float(), self.torch.tensor([-4., 448.]))
        self.assertEqual(report["numeric_normalization"]["base_nonfinite_values"], 1)
        self.assertEqual(report["numeric_normalization"]["output_clipped_values"], 1)

    def test_runtime_lora_failure_is_reported_without_aborting(self):
        from unittest.mock import patch
        with patch("model_merge_lora.LoraDelta.delta", side_effect=RuntimeError("unsupported contraction")):
            tensors, report = self.merge_fixture(
                {"layer.weight": self.torch.ones(2, 2)},
                {"layer.lora_A.weight": self.torch.ones(1, 2), "layer.lora_B.weight": self.torch.ones(2, 1)})
        self.torch.testing.assert_close(tensors["layer.weight"], self.torch.ones(2, 2))
        self.assertEqual(report["numeric_normalization"]["skipped_lora_deltas"], 1)
        self.assertEqual(report["result_kind"], "base-fallback")

    def test_oom_is_not_misreported_as_recoverable_lora_failure(self):
        from unittest.mock import patch
        with patch("model_merge_lora.LoraDelta.delta", side_effect=self.torch.OutOfMemoryError("allocation failed")):
            with self.assertRaises(self.torch.OutOfMemoryError):
                self.merge_fixture(
                    {"layer.weight": self.torch.ones(2, 2)},
                    {"layer.lora_A.weight": self.torch.ones(1, 2), "layer.lora_B.weight": self.torch.ones(2, 1)})
        self.assertFalse(self.output.exists())

    def test_checkpoint_directory_without_manifest_is_omitted(self):
        material = self.root / "missing-manifest"
        material.mkdir()
        self.save({"layer.weight": self.torch.ones(2, 2)}, material / "model.safetensors",
                  metadata={"modelspec.architecture": "flux2"})
        report = merge_models(self.base, material, output=self.output)
        self.assertEqual(report["result_kind"], "base-fallback")
        self.assertEqual(report["excluded_material_count"], 1)
        self.assertIn("model_index.json", report["excluded_sources"][0]["reason"])

    def test_unreadable_material_tensor_has_neutral_contribution(self):
        from unittest.mock import patch
        for mode in ("weighted-sum", "weighted-difference"):
            with self.subTest(mode=mode), patch("model_merge_common.CommonLayerReader.get_tensor", side_effect=OSError("unreadable tensor")):
                self.output = self.root / (mode + ".safetensors")
                tensors, report = self.merge_fixture(
                    {"layer.weight": self.torch.full((2, 2), 8.)},
                    {"layer.weight": self.torch.ones(2, 2)}, mode=mode)
                self.torch.testing.assert_close(tensors["layer.weight"], self.torch.full((2, 2), 8.))
                self.assertEqual(report["numeric_normalization"]["unreadable_material_tensors"], 1)
                self.assertTrue(report["no_effect"])

    def test_inspection_allocation_failure_is_not_an_excluded_material(self):
        from unittest.mock import patch
        from model_merge import inspect_merge_model
        def inspected(path, *args, **kwargs):
            if path == self.krea:
                raise self.torch.OutOfMemoryError("inspection allocation failed")
            return inspect_merge_model(path, *args, **kwargs)
        with patch("model_merge.inspect_merge_model", side_effect=inspected):
            with self.assertRaises(self.torch.OutOfMemoryError):
                merge_models(self.base, self.krea, output=self.output)
        self.assertFalse(self.output.exists())

    def test_nan_output_returns_base_and_infinities_saturate(self):
        from model_merge_tensor import safe_storage
        report = {}
        base = self.torch.tensor([7., 9., 11.], dtype=self.torch.float16)
        value = self.torch.tensor([float("nan"), float("inf"), -float("inf")])
        result = safe_storage(self.torch, value, base, report, (".", "layer.weight"))
        self.torch.testing.assert_close(result.float(), self.torch.tensor([7., 65504., -65504.]))
        self.assertEqual(report["output_nonfinite_values"], 3)
        self.assertEqual(report["output_clipped_values"], 2)


if __name__ == "__main__":
    unittest.main()
