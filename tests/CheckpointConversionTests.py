#!/usr/bin/env python3
"""Conversion policy, provenance and atomic cache tests without ML dependencies."""

from __future__ import annotations

import builtins
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import checkpoint_conversion as conversion


class Tensor:
    def __init__(self, values=(1, 2), dtype="float32", layout="strided", *, quantized=False, meta=False):
        self.values = values
        self.dtype = dtype
        self.layout = layout
        self.is_quantized = quantized
        self.is_meta = meta
        self.operations = []

    def detach(self):
        self.operations.append("detach")
        return self

    def cpu(self):
        self.operations.append("cpu")
        return self

    def contiguous(self):
        self.operations.append("contiguous")
        return self

    def clone(self):
        self.operations.append("clone")
        return Tensor(self.values, self.dtype)


class Parameter(Tensor):
    pass


class CheckpointConversionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="checkpoint-conversion-tests-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = self.directory / "cache"
        self.tensor = Tensor()
        self.torch = SimpleNamespace(
            Tensor=Tensor, nn=SimpleNamespace(Parameter=Parameter), strided="strided",
            __version__="2.13.0", load=Mock(return_value={"weight": self.tensor}),
            **{name: name for name in conversion._DTYPE_NAMES},
        )
        self.save_file = Mock(side_effect=self.save)
        self.safetensors = SimpleNamespace(__version__="test-safetensors")
        self.importer = Mock(side_effect=lambda name: {
            "torch": self.torch,
            "safetensors.torch": SimpleNamespace(save_file=self.save_file),
            "safetensors": self.safetensors,
        }[name])
        imported = patch.object(conversion.importlib, "import_module", self.importer)
        imported.start()
        self.addCleanup(imported.stop)

    def save(self, tensors, filename, metadata):
        Path(filename).write_text(json.dumps({
            "tensors": {name: {"values": list(value.values), "dtype": value.dtype}
                        for name, value in tensors.items()},
            "metadata": metadata,
        }, sort_keys=True), encoding="utf-8")

    def checkpoint(self, name="model.ckpt", data=b"mock checkpoint source"):
        path = self.directory / name
        path.write_bytes(data)
        return path

    def test_module_import_and_safetensors_passthrough_do_not_import_ml_runtimes(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "safetensors"):
                raise AssertionError("Implicit ML dependency import")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded):
            importlib.reload(conversion)
        for suffix in (".safetensors", ".safetensor"):
            source = self.checkpoint("existing" + suffix)
            result = conversion.materialize_safetensors(source, self.cache)
            self.assertFalse(result["converted"])
            self.assertEqual(result["converted_path"], str(source))
            self.assertEqual(result["original"], result["output"])
            self.assertIsNone(result["manifest_path"])
        self.importer.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_all_legacy_suffixes_load_cpu_weights_only_and_keep_source_unchanged(self):
        for suffix in conversion.LEGACY_SUFFIXES:
            with self.subTest(suffix=suffix):
                source = self.checkpoint("model" + suffix, suffix.encode())
                before = source.read_bytes()
                result = conversion.materialize_safetensors(source, self.cache)
                self.assertTrue(result["converted"])
                self.assertFalse(result["cache_hit"])
                self.assertEqual(source.read_bytes(), before)
                args, kwargs = self.torch.load.call_args
                self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertTrue(args[0].closed)
                self.assertEqual(result["original"]["sha256"], conversion.file_sha256(source))
                self.assertEqual(result["output"]["sha256"], conversion.file_sha256(Path(result["converted_path"])))
                manifest = json.loads(Path(result["manifest_path"]).read_text())
                self.assertEqual(manifest, result["manifest"])
                self.assertEqual(manifest["conversion_policy"]["weights_only"], True)
                self.assertFalse(manifest["generation_verified"])
                self.assertEqual(manifest["tensor_count"], 1)
                self.assertEqual(manifest["runtime"], {"torch": "2.13.0", "safetensors": "test-safetensors"})
        self.assertEqual(self.torch.load.call_count, len(conversion.LEGACY_SUFFIXES))

    def test_wrapped_state_dict_ignores_training_metadata_and_preserves_tied_keys(self):
        shared = Tensor(values=(9, 7), dtype="float16")
        self.torch.load.return_value = {"epoch": 24, "state_dict": {
            "weight": shared, "tied.weight": shared, "bias": Parameter(dtype="int64"),
            "mask": Tensor(dtype="bool"),
        }}
        result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        saved = self.save_file.call_args.args[0]
        self.assertEqual(set(saved), {"weight", "tied.weight", "bias", "mask"})
        self.assertIsNot(saved["weight"], saved["tied.weight"])
        self.assertEqual(saved["weight"].values, saved["tied.weight"].values)
        self.assertEqual(shared.operations, ["detach", "cpu", "contiguous", "clone"] * 2)
        self.assertTrue(result["manifest"]["state_dict_wrapped"])
        self.assertEqual(result["manifest"]["tensor_dtypes"], {"bool": 1, "float16": 2, "int64": 1})

    def test_content_addressed_cache_reuse_checks_hashes_without_reimporting_torch(self):
        source = self.checkpoint()
        first = conversion.materialize_safetensors(source, self.cache)
        same_content = self.checkpoint("same-content.pt", source.read_bytes())
        self.importer.reset_mock()
        second = conversion.materialize_safetensors(same_content, self.cache)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["converted_path"], second["converted_path"])
        self.assertEqual(second["original"]["path"], str(same_content))
        self.assertEqual(second["manifest"]["original"]["path"], str(source))
        self.importer.assert_not_called()
        self.torch.load.assert_called_once()

    def test_gguf_and_unrecognized_suffixes_reject_before_loading(self):
        for name, message in (("model.gguf", "native GGUF loader"), ("model.pkl", "Unsupported checkpoint")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                conversion.materialize_safetensors(self.checkpoint(name), self.cache)
        self.importer.assert_not_called()

    def test_missing_empty_and_directory_sources_are_rejected(self):
        directory = self.directory / "directory.ckpt"
        directory.mkdir()
        for source in ("", self.directory / "missing.ckpt", self.checkpoint("empty.pt", b""), directory):
            with self.subTest(source=source), self.assertRaises(ValueError):
                conversion.materialize_safetensors(source, self.cache)
        self.importer.assert_not_called()

    def test_weights_only_failures_never_retry_unsafe_loading(self):
        for error in (RuntimeError("unpickling global is not allowed"), TypeError("weights_only unsupported")):
            with self.subTest(error=error):
                self.torch.load.reset_mock()
                self.torch.load.side_effect = error
                with self.assertRaisesRegex(ValueError, "unsafe pickle retry is disabled"):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
                self.torch.load.assert_called_once()
                self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])
        self.save_file.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_missing_runtime_dependencies_fail_without_cache_publication(self):
        self.importer.side_effect = ModuleNotFoundError("torch missing")
        with self.assertRaisesRegex(RuntimeError, "requires installed Torch and safetensors"):
            conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertFalse(self.cache.exists())

    def test_known_vulnerable_unrecognized_and_prerelease_torch_versions_reject_before_load(self):
        for version in ("2.5.1", "2.9.1", "2.10.0rc1", "unknown", None):
            with self.subTest(version=version):
                self.torch.__version__ = version
                with self.assertRaisesRegex(RuntimeError, "requires stable Torch >=2.10.0"):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.torch.load.assert_not_called()
        self.assertFalse(self.cache.exists())
        for version in ("2.10.0", "2.10.1+cu128", "2.13.0", "2.10.0.post1"):
            with self.subTest(version=version):
                self.torch.__version__ = version
                conversion.ensure_safe_torch_version(self.torch)

    def test_a_flat_tensor_named_state_dict_is_preserved(self):
        self.torch.load.return_value = {"state_dict": Tensor()}
        result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertFalse(result["manifest"]["state_dict_wrapped"])
        self.assertEqual(set(self.save_file.call_args.args[0]), {"state_dict"})

    def test_invalid_state_dictionary_entries_are_rejected(self):
        class TensorSubclass(Tensor):
            pass
        invalid = [None, [], {}, {"state_dict": {}}, {"state_dict": []},
                   {1: Tensor()}, {"": Tensor()}, {"weight": 1}, {"weight": object()},
                   {"weight": Tensor(layout="sparse_coo")}, {"weight": Tensor(quantized=True)},
                   {"weight": Tensor(meta=True)}, {"weight": Tensor(dtype="complex64")},
                   {"weight": TensorSubclass()}]
        for state in invalid:
            with self.subTest(state=state):
                self.torch.load.return_value = state
                with self.assertRaises(ValueError):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.save_file.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_source_mutation_during_load_rejects_even_same_size_content(self):
        source = self.checkpoint(data=b"before")
        def mutate(*args, **kwargs):
            source.write_bytes(b"AFTER!")
            return {"weight": Tensor()}
        self.torch.load.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "changed during weights-only loading"):
            conversion.materialize_safetensors(source, self.cache)
        self.save_file.assert_not_called()

    def test_source_mutation_during_save_rejects_and_removes_temporary_output(self):
        source = self.checkpoint(data=b"before")
        def mutate(*args, **kwargs):
            self.save(*args, **kwargs)
            source.write_bytes(b"AFTER!")
        self.save_file.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "Source checkpoint changed"):
            conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_source_symlink_retargeting_is_detected(self):
        source = self.checkpoint()
        link = self.directory / "linked.ckpt"
        link.symlink_to(source)
        other = self.checkpoint("same.pt", source.read_bytes())
        def retarget(*args, **kwargs):
            self.save(*args, **kwargs)
            link.unlink()
            link.symlink_to(other)
        self.save_file.side_effect = retarget
        with self.assertRaisesRegex(RuntimeError, "Source checkpoint changed"):
            conversion.materialize_safetensors(link, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_corrupted_output_is_rejected_and_not_overwritten(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        output = Path(result["converted_path"])
        output.write_bytes(b"corrupted converted output")
        self.importer.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "no longer matches its manifest"):
            conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(output.read_bytes(), b"corrupted converted output")
        self.importer.assert_not_called()

    def test_invalid_or_missing_conversion_policy_is_rejected(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        manifest_path = Path(result["manifest_path"])
        for change in (None, {"weights_only": False}, "invalid"):
            with self.subTest(change=change):
                manifest = dict(result["manifest"])
                if change is None:
                    manifest.pop("conversion_policy")
                else:
                    manifest["conversion_policy"] = change
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "Invalid checkpoint conversion cache"):
                    conversion.materialize_safetensors(source, self.cache)
        self.torch.load.assert_called_once()

    def test_cached_tensor_file_cannot_be_replaced_by_symlink(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        output = Path(result["converted_path"])
        copy = self.directory / "other.safetensors"
        shutil.copyfile(output, copy)
        output.unlink()
        output.symlink_to(copy)
        with self.assertRaisesRegex(RuntimeError, "must not be symlinks"):
            conversion.materialize_safetensors(source, self.cache)

    def test_save_failure_never_publishes_partial_cache(self):
        def fail(tensors, filename, metadata):
            Path(filename).write_bytes(b"partial output")
            raise RuntimeError("save failed")
        self.save_file.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_manifest_and_tensor_file_are_published_together(self):
        rename = conversion.os.rename
        def checked_rename(source, destination):
            self.assertTrue((source / "model.safetensors").is_file())
            self.assertTrue((source / "conversion.json").is_file())
            self.assertFalse(destination.exists())
            rename(source, destination)
        with patch.object(conversion.os, "rename", side_effect=checked_rename) as renamed:
            result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        renamed.assert_called_once()
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertEqual(len(list(self.cache.iterdir())), 1)

    def test_concurrent_cache_winner_is_validated_and_temporary_entry_removed(self):
        def competing_rename(source, destination):
            shutil.copytree(source, destination)
            raise FileExistsError("A concurrent conversion won publication")
        with patch.object(conversion.os, "rename", side_effect=competing_rename):
            result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(len(list(self.cache.iterdir())), 1)


@unittest.skipUnless(os.environ.get("IILD_CHECKPOINT_REAL_RUNTIME_TESTS") == "1",
                     "Explicit local Torch/safetensors smoke is disabled")
class RealCheckpointConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.safetensors = importlib.import_module("safetensors.torch")
        cls.directory = ROOT / "build" / "reference" / "checkpoint-conversion-smoke"
        cls.directory.mkdir(parents=True, exist_ok=True)
        cls.smoke_run = Path(tempfile.mkdtemp(prefix="run-", dir=cls.directory))

    def test_real_zip_and_legacy_serialization_preserve_values_dtypes_and_tied_keys(self):
        torch = self.torch
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        state = {"weight": weight, "tied.weight": weight,
                 "mask": torch.tensor([True, False]), "step": torch.tensor(5, dtype=torch.int64)}
        results = []
        for zip_serialization in (True, False):
            name = "tiny.ckpt" if zip_serialization else "tiny-legacy.pt"
            source = self.smoke_run / name
            torch.save({"state_dict": state, "epoch": 1}, source,
                       _use_new_zipfile_serialization=zip_serialization)
            before = conversion.file_sha256(source)
            result = conversion.materialize_safetensors(source, self.smoke_run / "cache")
            restored = self.safetensors.load_file(result["converted_path"], device="cpu")
            self.assertEqual(set(restored), set(state))
            for key in state:
                self.assertTrue(torch.equal(restored[key], state[key]), key)
                self.assertEqual(restored[key].dtype, state[key].dtype)
                self.assertTrue(restored[key].is_contiguous())
            self.assertEqual(conversion.file_sha256(source), before)
            results.append({"source": result["original"], "output": result["output"],
                            "manifest_path": result["manifest_path"],
                            "zip_serialization": zip_serialization,
                            "tensor_values_equal": True, "source_unchanged": True})
        (self.directory / "tiny-conversion.json").write_text(json.dumps({
            "torch_version": torch.__version__, "cases": results, "generation_verified": False,
        }, indent=2) + "\n", encoding="utf-8")

    def test_malicious_reduce_is_rejected_without_executing_its_marker(self):
        marker = self.smoke_run / "unexpected-code-execution.txt"
        class MaliciousReduce:
            def __reduce__(self):
                return (eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",))
        source = self.smoke_run / "malicious.ckpt"
        self.torch.save({"state_dict": {"weight": MaliciousReduce()}}, source)
        self.assertFalse(marker.exists())
        before = conversion.file_sha256(source)
        with self.assertRaisesRegex(ValueError, "unsafe pickle retry is disabled") as caught:
            conversion.materialize_safetensors(source, self.smoke_run / "malicious-cache")
        self.assertFalse(marker.exists())
        self.assertEqual(conversion.file_sha256(source), before)
        self.assertFalse((self.smoke_run / "malicious-cache").exists())
        (self.directory / "malicious-checkpoint.json").write_text(json.dumps({
            "torch_version": self.torch.__version__, "source": str(source),
            "source_sha256": before, "source_unchanged": True,
            "marker": str(marker), "marker_exists": marker.exists(),
            "error": str(caught.exception), "cause_type": type(caught.exception.__cause__).__name__,
            "unsafe_retry": False, "generation_verified": False,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
