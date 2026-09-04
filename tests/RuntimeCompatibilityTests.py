#!/usr/bin/env python3
"""Mocked runtime imports: no model weights, GPU, ComfyUI or network required."""

from __future__ import annotations

import builtins
import importlib
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import civitai_catalog
import runtime_compatibility as audit


class DiffusionPipeline:
    def __init__(self):
        raise AssertionError("The audit must not instantiate a pipeline")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("The audit must not load model weights")


def pipeline(name, module="diffusers.pipelines.fixture"):
    return type(name, (DiffusionPipeline,), {"__module__": module})


def available_runtime():
    values = {record["pipeline_class"]: pipeline(record["pipeline_class"])
              for record in civitai_catalog.list_base_models() if record["pipeline_class"]}
    return SimpleNamespace(DiffusionPipeline=DiffusionPipeline, **values)


class RuntimeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.metadata = patch.object(audit.importlib.metadata, "version",
                                     side_effect=lambda name: {"diffusers": "0.40.0", "torch": "2.13.0"}[name])
        self.metadata.start()
        self.addCleanup(self.metadata.stop)

    def test_module_import_does_not_import_ml_runtimes(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "diffusers"):
                raise AssertionError("Implicit ML runtime import")
            return original(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded):
            importlib.reload(audit)

    def test_all_rows_have_explicit_non_generation_status(self):
        with patch.object(audit.importlib, "import_module", return_value=available_runtime()) as imported:
            result = audit.inspect_runtime()
        imported.assert_called_once_with("diffusers")
        self.assertEqual(len(result["rows"]), 105)
        self.assertEqual(result["catalog_source"]["commit"], civitai_catalog.CATALOG_SOURCE["commit"])
        self.assertEqual(result["runtime"]["version"], "0.40.0")
        self.assertEqual(result["runtime"]["torch_version"], "2.13.0")
        self.assertTrue(result["runtime"]["import_available"])
        self.assertFalse(result["generation_verified"])
        self.assertFalse(result["weights_verified"])
        for row in result["rows"]:
            with self.subTest(name=row["name"]):
                self.assertFalse(row["generation_verified"])
                self.assertFalse(row["weights_verified"])
                if row["local_status"] != "local":
                    self.assertEqual(row["status"], row["local_status"])
                elif row["preferred_backend"] == "comfyui":
                    self.assertEqual(row["status"], "workflow_required")
                    self.assertFalse(row["runtime_checked"])
                else:
                    self.assertEqual(row["status"], "pipeline_available")
                    self.assertTrue(row["pipeline_available"])

    def test_one_model_normalizes_name_and_checks_its_exact_class(self):
        runtime = SimpleNamespace(DiffusionPipeline=DiffusionPipeline,
                                  FluxPipeline=pipeline("FluxPipeline"))
        with patch.object(audit.importlib, "import_module", return_value=runtime):
            result = audit.inspect_runtime(" flux.1 krea ")
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["name"], "Flux.1 Krea")
        self.assertEqual(row["pipeline_class"], "FluxPipeline")
        self.assertEqual(row["status"], "pipeline_available")

    def test_missing_installation_is_distinct_from_missing_class(self):
        with patch.object(audit.importlib.metadata, "version",
                          side_effect=importlib.metadata.PackageNotFoundError("missing")), \
                patch.object(audit.importlib, "import_module", side_effect=ModuleNotFoundError("diffusers")):
            result = audit.inspect_runtime("Illustrious")
        self.assertIsNone(result["runtime"]["version"])
        self.assertTrue(result["runtime"]["import_attempted"])
        self.assertFalse(result["runtime"]["import_available"])
        self.assertEqual(result["rows"][0]["status"], "runtime_missing")
        self.assertIn("ModuleNotFoundError", result["rows"][0]["error"])
        with patch.object(audit.importlib, "import_module",
                          return_value=SimpleNamespace(DiffusionPipeline=DiffusionPipeline)):
            result = audit.inspect_runtime("Krea 2")
        self.assertTrue(result["runtime"]["import_available"])
        self.assertEqual(result["rows"][0]["status"], "runtime_missing")
        self.assertIn("Krea2Pipeline", result["rows"][0]["error"])

    def test_lazy_optional_dependency_failure_does_not_abort_other_rows(self):
        runtime = available_runtime()
        class LazyRuntime:
            def __getattr__(self, name):
                if name == "Krea2Pipeline":
                    raise ImportError("optional sentencepiece dependency missing")
                return getattr(runtime, name)
        with patch.object(audit.importlib, "import_module", return_value=LazyRuntime()):
            result = audit.inspect_runtime()
        rows = {row["name"]: row for row in result["rows"]}
        self.assertEqual(rows["Illustrious"]["status"], "pipeline_available")
        self.assertEqual(rows["Krea 2"]["status"], "runtime_missing")
        self.assertIn("sentencepiece", rows["Krea 2"]["error"])

    def test_nonclass_foreign_class_and_dummy_exports_are_rejected(self):
        invalid = [object(), Mock(), type("StableDiffusionXLPipeline", (), {}),
                   DiffusionPipeline, pipeline("StableDiffusionXLPipeline", "custom_module"),
                   pipeline("StableDiffusionXLPipeline", "diffusers.utils.dummy_torch_objects")]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                runtime = SimpleNamespace(DiffusionPipeline=DiffusionPipeline,
                                          StableDiffusionXLPipeline=candidate)
                with patch.object(audit.importlib, "import_module", return_value=runtime):
                    row = audit.inspect_runtime("Illustrious")["rows"][0]
                self.assertEqual(row["status"], "runtime_missing")
                self.assertFalse(row["pipeline_available"])

    def test_workflow_hosted_and_unknown_do_not_import_or_contact_runtimes(self):
        for name, status in (("Anima", "workflow_required"), ("OpenAI", "hosted"), ("Other", "unknown")):
            with self.subTest(name=name), \
                    patch.object(audit.importlib, "import_module", side_effect=AssertionError("runtime import")), \
                    patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
                result = audit.inspect_runtime(name)
            self.assertFalse(result["runtime"]["import_attempted"])
            self.assertEqual(result["rows"][0]["status"], status)
            self.assertFalse(result["rows"][0]["runtime_checked"])
        self.assertIn("not checked", audit.inspect_runtime("Anima")["rows"][0]["notes"])

    def test_unknown_name_fails_before_importing_any_runtime(self):
        with patch.object(audit.importlib, "import_module", side_effect=AssertionError("runtime import")):
            with self.assertRaisesRegex(ValueError, "Unknown Civitai"):
                audit.inspect_runtime("Krea typo")

    def test_root_import_failure_is_reported_without_aborting_catalog(self):
        with patch.object(audit.importlib, "import_module", side_effect=RuntimeError("incompatible torchvision")):
            result = audit.inspect_runtime()
        self.assertEqual(len(result["rows"]), 105)
        self.assertIn("torchvision", result["runtime"]["error"])
        rows = {row["name"]: row for row in result["rows"]}
        self.assertEqual(rows["Illustrious"]["status"], "runtime_missing")
        self.assertEqual(rows["Anima"]["status"], "workflow_required")
        self.assertEqual(rows["OpenAI"]["status"], "hosted")


if __name__ == "__main__":
    unittest.main()
