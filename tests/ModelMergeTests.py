#!/usr/bin/env python3
"""Merge argument contracts and real, small Torch/safetensors checkpoint tests."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import model_merge
from model_merge_options import build_parser, resolve_merge_request, resolve_merge_weights


class MergeWorkspace(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="model-merge-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.base = self.directory / "base.safetensors"
        self.extra = self.directory / "extra.safetensors"
        self.third = self.directory / "third.safetensors"
        self.output = self.directory / "merged.safetensors"
        for path in (self.base, self.extra, self.third):
            path.write_bytes(b"argument-only fixture")


class ModelMergeOptionsTests(MergeWorkspace):
    def checkpoint_request(self, *args, **kwargs):
        request = resolve_merge_request(*args, **kwargs)
        return resolve_merge_weights(request, ["checkpoint"] * len(request.additional_models))

    def test_base_and_first_additional_model_are_required_in_api_and_cli(self):
        for args in ((), (self.base,)):
            with self.assertRaises(TypeError):
                resolve_merge_request(*args)
            with self.assertRaises(TypeError):
                model_merge.merge_models(*args)
        for args in ([], ["--base-model", str(self.base)],
                     ["--additional-model", str(self.extra)]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                build_parser().parse_args(args)
            self.assertEqual(error.exception.code, 2)

    def test_two_model_default_is_equal_weighted_sum(self):
        request = self.checkpoint_request(self.base, self.extra)
        self.assertEqual(request.mode, "weighted-sum")
        self.assertEqual(request.weights, (0.5,))
        self.assertEqual(request.base_weight, 0.5)
        self.assertEqual(request.models, (self.base, self.extra))
        self.assertIn("build", request.output.parts)

    def test_multiple_model_defaults_are_equal_and_order_independent(self):
        request = self.checkpoint_request(self.base, self.extra, additional_models=[self.third])
        self.assertEqual(request.weights, (1 / 3, 1 / 3))
        self.assertAlmostEqual(request.base_weight, 1 / 3)
        self.assertEqual(request.models, (self.base, self.extra, self.third))

    def test_explicit_weights_leave_the_remaining_weight_for_the_base(self):
        request = self.checkpoint_request(self.base, self.extra,
                                        additional_models=[self.third], weights=[0.2, 0.3])
        self.assertEqual(request.weights, (0.2, 0.3))
        self.assertEqual(request.base_weight, 0.5)

    def test_scalar_weight_is_broadcast_and_inputs_are_copied(self):
        additional = [self.third]
        request = resolve_merge_request(self.base, self.extra, additional_models=additional, weights=0.25)
        additional.clear()
        self.assertEqual(request.weights, (0.25, 0.25))
        self.assertEqual(len(request.models), 3)

    def test_difference_keeps_base_and_defaults_each_extra_to_half(self):
        request = self.checkpoint_request(self.base, self.extra,
                                        additional_models=[self.third], mode="weighted-difference")
        self.assertEqual(request.base_weight, 1)
        self.assertEqual(request.weights, (0.5, 0.5))
        request = resolve_merge_request(self.base, self.extra, weights=2, mode="weighted-difference")
        self.assertEqual(request.weights, (2,))

    def test_invalid_weights_and_modes_fail_before_loading_runtime(self):
        for weights in (True, "0.5", math.nan, math.inf, -0.1, [], [0.1, 0.2], [None], [False]):
            with self.subTest(weights=weights), self.assertRaises((TypeError, ValueError)):
                resolve_merge_request(self.base, self.extra, weights=weights)
        with self.assertRaisesRegex(ValueError, "sum.*1"):
            self.checkpoint_request(self.base, self.extra, additional_models=[self.third], weights=[0.6, 0.6])
        with self.assertRaisesRegex(ValueError, "mode"):
            resolve_merge_request(self.base, self.extra, mode="add-difference")
        with self.assertRaises((TypeError, ValueError)):
            resolve_merge_request(self.base, self.extra, additional_models=str(self.third))

    def test_missing_empty_and_remote_sources_fail(self):
        for source in ("", "https://example.com/model.safetensors", self.directory / "missing.safetensors"):
            with self.subTest(source=source), self.assertRaises((ValueError, FileNotFoundError)):
                resolve_merge_request(self.base, source)
        self.extra.write_bytes(b"")
        with self.assertRaises(ValueError):
            resolve_merge_request(self.base, self.extra)

    def test_output_cannot_replace_an_input_or_existing_target(self):
        for output in (self.base, self.extra):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                resolve_merge_request(self.base, self.extra, output=output)
        self.output.symlink_to(self.base)
        with self.assertRaises((ValueError, FileExistsError)):
            resolve_merge_request(self.base, self.extra, output=self.output)
        self.output.unlink()
        self.output.symlink_to(self.directory / "missing")
        with self.assertRaises((ValueError, FileExistsError)):
            resolve_merge_request(self.base, self.extra, output=self.output)

    def test_cli_repeated_models_and_weights_are_resolved_without_torch(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(model_merge.main([
                "--base-model", str(self.base), "--additional-model", str(self.extra),
                "--additional-model", str(self.third), "--weights", "0.2", "0.3", "--print-config",
            ]), 0)
        value = json.loads(output.getvalue())
        self.assertIsNone(value["base_weight"])
        self.assertEqual(value["coefficient_resolution"], "after-input-inspection")
        self.assertEqual(value["weights"], [0.2, 0.3])
        self.assertEqual(value["additional_models"], [str(self.extra), str(self.third)])

    def test_import_and_help_work_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/merge.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("weighted-difference", result.stdout)

    def test_missing_runtime_dependency_is_an_actionable_error(self):
        with patch.object(model_merge.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "Torch and safetensors"):
                model_merge.merge_models(self.base, self.extra, output=self.output)
        self.assertFalse(self.output.exists())


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))


@unittest.skipUnless(HAS_RUNTIME, "Real model merging requires the existing Torch/safetensors environment")
class ModelMergeTensorTests(MergeWorkspace):
    @classmethod
    def setUpClass(cls):
        import torch
        import safetensors
        import safetensors.torch
        cls.torch = torch
        cls.safe = safetensors
        cls.save = staticmethod(safetensors.torch.save_file)
        cls.load = staticmethod(safetensors.torch.load_file)

    def setUp(self):
        super().setUp()
        for path, values in ((self.base, [2, 4]), (self.extra, [6, 8]), (self.third, [10, 12])):
            self.write(path, values)

    def write(self, path, values, dtype=None, **more):
        self.save({"weight": self.torch.tensor(values, dtype=dtype or self.torch.float32), **more},
                  str(path), metadata={"format": "pt", "original_label": "base fixture"})

    def merge(self, **kwargs):
        return model_merge.merge_models(self.base, self.extra, output=self.output, **kwargs)

    def test_default_sum_and_original_bytes_are_preserved(self):
        before = [p.read_bytes() for p in (self.base, self.extra)]
        result = self.merge()
        self.assertEqual(self.load(self.output)["weight"].tolist(), [4, 6])
        self.assertEqual(before, [p.read_bytes() for p in (self.base, self.extra)])
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["output"], str(self.output))
        with self.safe.safe_open(self.output, framework="pt") as reader:
            metadata = json.loads(reader.metadata()["iild_merge"])
        self.assertEqual(metadata["weights"], [0.5])
        self.assertEqual(metadata["base_weight"], 0.5)
        self.assertEqual(len(metadata["sources"]), 2)
        self.assertEqual(len(metadata["sources"][0]["files"][0]["sha256"]), 64)

    def test_multiple_sum_uses_one_combination_without_sequential_bias(self):
        self.merge(additional_models=[self.third], weights=[0.25, 0.5])
        self.assertEqual(self.load(self.output)["weight"].tolist(), [7, 9])

    def test_two_model_difference_and_multiple_weighted_subtractions(self):
        self.merge(mode="weighted-difference")
        self.assertEqual(self.load(self.output)["weight"].tolist(), [-1, 0])
        self.output.unlink()
        self.merge(mode="weighted-difference", additional_models=[self.third], weights=[0.25, 0.5])
        self.assertEqual(self.load(self.output)["weight"].tolist(), [-4.5, -4])

    def test_zero_and_unit_weights_select_exact_endpoints(self):
        for weight, expected in ((0, [2, 4]), (1, [6, 8])):
            self.merge(weights=weight)
            self.assertEqual(self.load(self.output)["weight"].tolist(), expected)
            self.output.unlink()

    def test_half_and_bfloat16_accumulate_in_float32_and_keep_base_dtype(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.write(self.base, [60000, -60000], dtype)
                self.write(self.extra, [60000, -60000], torch.float32)
                self.merge()
                tensor = self.load(self.output)["weight"]
                self.assertEqual(tensor.dtype, dtype)
                self.assertTrue(torch.isfinite(tensor).all().item())
                self.output.unlink()

    def test_integer_boolean_buffers_and_empty_tensors_are_preserved(self):
        torch = self.torch
        more = {"ids": torch.tensor([1, 2], dtype=torch.int64), "mask": torch.tensor([True, False]),
                "scalar": torch.tensor(3.0), "empty": torch.empty((0, 3))}
        for path in (self.base, self.extra):
            self.write(path, [1, 2], **more)
        self.merge(mode="weighted-difference")
        saved = self.load(self.output)
        self.assertEqual(saved["ids"].tolist(), [1, 2])
        self.assertEqual(saved["mask"].tolist(), [True, False])
        self.assertEqual(tuple(saved["empty"].shape), (0, 3))
        self.assertEqual(saved["scalar"].item(), 1.5)

    def test_missing_extra_keys_or_shapes_never_publish_output(self):
        for state in ({"other": self.torch.ones(2)},
                      {"weight": self.torch.ones(2), "extra": self.torch.ones(1)},
                      {"weight": self.torch.ones(3)}):
            self.save(state, str(self.extra))
            with self.assertRaisesRegex(ValueError, "keys|shape"):
                self.merge()
            self.assertFalse(self.output.exists())

    def test_incompatible_nonfloating_buffers_fail(self):
        torch = self.torch
        for base, extra in ((torch.tensor([1]), torch.tensor([2])),
                            (torch.tensor([1]), torch.tensor([1.0])),
                            (torch.tensor([1], dtype=torch.int64), torch.tensor([1], dtype=torch.int32))):
            self.write(self.base, [1], ids=base)
            self.write(self.extra, [1], ids=extra)
            with self.assertRaisesRegex(ValueError, "dtype|buffer"):
                self.merge()
            self.assertFalse(self.output.exists())

    def test_nonfinite_inputs_and_cast_overflow_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            self.write(self.extra, [value, 1])
            with self.assertRaisesRegex(ValueError, "finite"):
                self.merge()
            self.assertFalse(self.output.exists())
        self.write(self.base, [60000], self.torch.float16)
        self.write(self.extra, [-60000], self.torch.float16)
        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            self.merge(mode="weighted-difference", weights=1)
        self.assertFalse(self.output.exists())

    def test_corrupt_safetensors_fail_without_publication(self):
        self.extra.write_bytes(b"not safetensors")
        with self.assertRaises(ValueError):
            self.merge()
        self.assertFalse(self.output.exists())

    def test_singular_suffix_and_legacy_tensor_checkpoints_reuse_safe_conversion(self):
        self.extra.rename(self.extra.with_suffix(".safetensor"))
        self.extra = self.extra.with_suffix(".safetensor")
        self.merge()
        self.assertEqual(self.load(self.output)["weight"].tolist(), [4, 6])
        self.output.unlink()
        self.extra = self.extra.with_suffix(".ckpt")
        self.torch.save({"state_dict": {"weight": self.torch.tensor([6.0, 8.0])}}, self.extra)
        self.merge(cache_dir=self.directory / "cache")
        self.assertEqual(self.load(self.output)["weight"].tolist(), [4, 6])

    def test_empty_and_quantized_float_formats_are_rejected(self):
        self.save({}, str(self.extra))
        with self.assertRaisesRegex(ValueError, "keys"):
            self.merge()
        self.write(self.base, [1, 2], self.torch.float8_e4m3fn)
        self.write(self.extra, [1, 2], self.torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.merge()
        self.assertFalse(self.output.exists())

    def test_conflicting_explicit_architecture_metadata_is_rejected(self):
        for path, architecture in ((self.base, "stable-diffusion-v1"), (self.extra, "stable-diffusion-xl")):
            self.save({"weight": self.torch.ones(2)}, str(path), metadata={"modelspec.architecture": architecture})
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.merge()
        self.assertFalse(self.output.exists())

    def test_failed_write_cleans_temporary_files_and_preserves_sources(self):
        before = set(self.directory.iterdir())
        with patch("safetensors.torch.save_file", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.merge()
        self.assertEqual(before, set(self.directory.iterdir()))

    def test_source_change_during_save_prevents_publication(self):
        original_save = self.save
        def change(tensors, filename, metadata=None):
            original_save(tensors, filename, metadata=metadata)
            self.extra.write_bytes(b"changed source")
        with patch("safetensors.torch.save_file", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.merge()
        self.assertFalse(self.output.exists())

    def test_competing_output_is_never_overwritten(self):
        original_save = self.save
        def competing(tensors, filename, metadata=None):
            original_save(tensors, filename, metadata=metadata)
            self.output.write_bytes(b"another job")
        with patch("safetensors.torch.save_file", side_effect=competing):
            with self.assertRaises(FileExistsError):
                self.merge()
        self.assertEqual(self.output.read_bytes(), b"another job")

    def package(self, name, value, *, shards=False):
        root = self.directory / name
        component = root / "unet"
        component.mkdir(parents=True)
        (root / "model_index.json").write_text(json.dumps({"_class_name": "DDPMPipeline",
                                                          "unet": ["diffusers", "UNet2DModel"]}))
        (component / "config.json").write_text(json.dumps({"sample_size": 8, "in_channels": 3}))
        state = {"weight": self.torch.full((2,), float(value)), "bias": self.torch.tensor(float(value))}
        if shards:
            mapping = {}
            for index, (key, tensor) in enumerate(state.items()):
                filename = f"diffusion_pytorch_model-{index + 1:05}-of-00002.safetensors"
                self.save({key: tensor}, str(component / filename))
                mapping[key] = filename
            (component / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 12}, "weight_map": mapping,
            }))
        else:
            self.save(state, str(component / "diffusion_pytorch_model.safetensors"))
        return root

    def test_diffusers_packages_support_different_sharding_and_preserve_base_layout(self):
        base = self.package("base-package", 2, shards=True)
        extra = self.package("extra-package", 6)
        output = self.directory / "merged-package"
        report = model_merge.merge_models(base, extra, output=output)
        self.assertEqual(report["tensor_count"], 2)
        for source in (base / "unet").glob("*.safetensors"):
            saved = self.load(output / "unet" / source.name)
            for value in saved.values():
                self.assertTrue((value == 4).all().item())
        for source in base.rglob("*.json"):
            self.assertEqual(source.read_bytes(), (output / source.relative_to(base)).read_bytes())
        manifest = json.loads((output / "merge.json").read_text())
        self.assertEqual(manifest["weights"], [0.5])

    def test_package_configuration_or_tokenizer_mismatch_fails(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        (extra / "unet/config.json").write_text('{"sample_size":16,"in_channels":3}')
        output = self.directory / "merged-package"
        with self.assertRaisesRegex(ValueError, "configuration|assets"):
            model_merge.merge_models(base, extra, output=output)
        self.assertFalse(output.exists())

    def test_package_allows_runtime_version_and_dtype_metadata_differences(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        (extra / "unet/config.json").write_text(json.dumps({"sample_size": 8, "in_channels": 3,
            "_name_or_path": "elsewhere", "_diffusers_version": "different", "torch_dtype": "float16"}))
        model_merge.merge_models(base, extra, output=self.directory / "merged-package")

    def test_package_allows_component_specific_architecture_hints(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        for root, value in ((base, 2), (extra, 6)):
            for component in ("unet", "vae"):
                folder = root / component
                folder.mkdir(exist_ok=True)
                self.save({"weight": self.torch.full((2,), float(value))},
                          str(folder / "diffusion_pytorch_model.safetensors"),
                          metadata={"modelspec.architecture": f"sd1/{component}"})
        output = self.directory / "merged-package"
        model_merge.merge_models(base, extra, output=output)
        self.assertEqual(self.load(output / "vae/diffusion_pytorch_model.safetensors")["weight"].tolist(), [4, 4])

    def test_package_tokenizer_vocabulary_difference_is_rejected(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        for root, content in ((base, "a b"), (extra, "a c")):
            (root / "tokenizer").mkdir()
            (root / "tokenizer/merges.txt").write_text(content)
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            model_merge.merge_models(base, extra, output=self.directory / "merged-package")

    def test_package_refuses_duplicate_variants_other_weights_and_missing_components(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        variant = extra / "unet/diffusion_pytorch_model.fp16.safetensors"
        self.save({"weight": self.torch.ones(2)}, str(variant))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            model_merge.merge_models(base, extra, output=self.directory / "merged-package")
        variant.unlink()
        (extra / "unet/pytorch_model.bin").write_bytes(b"ambiguous variant")
        with self.assertRaisesRegex(ValueError, "only safetensors"):
            model_merge.merge_models(base, extra, output=self.directory / "merged-package")
        (extra / "unet/pytorch_model.bin").unlink()
        (extra / "unet").rename(extra / "different_component")
        with self.assertRaisesRegex(ValueError, "configuration|keys"):
            model_merge.merge_models(base, extra, output=self.directory / "merged-package")

    def test_package_asset_mutation_and_file_addition_abort_publication(self):
        base = self.package("base-package", 2)
        extra = self.package("extra-package", 6)
        original_save = self.save
        output = self.directory / "merged-package"
        def change(tensors, filename, metadata=None):
            original_save(tensors, filename, metadata=metadata)
            (extra / "new-asset.txt").write_text("added during merge")
        with patch("safetensors.torch.save_file", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                model_merge.merge_models(base, extra, output=output)
        self.assertFalse(output.exists())

    def test_package_rejects_invalid_shard_index_and_nested_outputs(self):
        base = self.package("base-package", 2, shards=True)
        extra = self.package("extra-package", 6)
        with self.assertRaisesRegex(ValueError, "inside|overlap"):
            model_merge.merge_models(base, extra, output=base / "merged")
        index = base / "unet/diffusion_pytorch_model.safetensors.index.json"
        index.write_text('{"weight_map":{"weight":"../outside.safetensors"}}')
        with self.assertRaisesRegex(ValueError, "index|shard"):
            model_merge.merge_models(base, extra, output=self.directory / "bad-package")

    def test_cli_saves_a_reloadable_checkpoint_and_reports_result(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/merge.py"),
            "--base-model", str(self.base), "--additional-model", str(self.extra),
            "--mode", "weighted-difference", "--weights", "0.25", "--output", str(self.output)],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["output"], str(self.output))
        self.assertEqual(self.load(self.output)["weight"].tolist(), [0.5, 2])


if __name__ == "__main__":
    unittest.main()
