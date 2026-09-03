#!/usr/bin/env python3
"""Conversion argument and plan-policy checks without optional ML packages."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "coreml"))
import convert_linear


class CoreMLConversionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="coreml-conversion-", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = ["--fixture", "--output-dir", str(self.directory / "output")]

    def assert_rejected(self, arguments, message):
        error = io.StringIO()
        with patch.dict(sys.modules, {"coremltools": None, "numpy": None}), redirect_stderr(error):
            with self.assertRaises(SystemExit) as failure:
                convert_linear.main(arguments)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn(message, error.getvalue())

    def test_help_is_dependency_free(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"coremltools": None}), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                convert_linear.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--model", output.getvalue())

    def test_source_must_be_explicit(self):
        self.assert_rejected(["--output-dir", str(self.directory / "output")], "required")
        self.assert_rejected(self.base + ["--model", "checkpoint.safetensors"], "not allowed")
        self.assert_rejected(["--model", "checkpoint.safetensors", "--output-dir",
                              str(self.directory / "output")], "--weight-key")
        self.assert_rejected(self.base + ["--weight-key", "weight"], "--model")

    def test_empty_model_is_not_a_synthetic_fixture(self):
        self.assert_rejected(["--model", "", "--weight-key", "weight", "--output-dir",
                              str(self.directory / "output")], "must not be empty")

    def test_dimensions_are_bounded_positive_integers(self):
        for argument, value in (("--batch-size", "0"), ("--batch-size", "-1"),
                                ("--fixture-width", "0")):
            self.assert_rejected(self.base + [argument, value], "must be positive")

    def test_evidence_must_stay_under_build(self):
        self.assert_rejected(self.base + ["--output-dir", str(ROOT / "outside-build")],
                             "under the repository build/")

    def test_previous_evidence_is_preserved(self):
        output = self.directory / "output"
        output.mkdir()
        evidence = output / "provenance.json"
        evidence.write_text("previous evidence", encoding="utf-8")
        self.assert_rejected(self.base, "not overwritten")
        self.assertEqual(evidence.read_text(), "previous evidence")

    def test_native_test_must_exist_when_requested(self):
        self.assert_rejected(self.base + ["--native-test", str(self.directory / "missing")],
                             "Build the native CoreMLTests")

    def test_wrong_file_extension_is_rejected_before_import(self):
        path = self.directory / "unsafe.pkl"
        path.write_bytes(b"not a model")
        self.assert_rejected(["--model", str(path), "--weight-key", "weight", "--output-dir",
                              str(self.directory / "output")], "safetensors")

    def test_plan_requires_preferred_not_merely_supported_neural_engine(self):
        cpu = [{"preferred_device": "cpu", "supported_devices": ["cpu", "neural-engine"]}]
        with self.assertRaisesRegex(RuntimeError, "does not prefer"):
            convert_linear.check_neural_plan(cpu, allow_cpu_plan=False)
        self.assertEqual(convert_linear.check_neural_plan(cpu, allow_cpu_plan=True), 0)
        neural = cpu + [{"preferred_device": "neural-engine", "supported_devices": ["neural-engine"]}]
        self.assertEqual(convert_linear.check_neural_plan(neural, allow_cpu_plan=False), 1)
        with self.assertRaises(RuntimeError):
            convert_linear.check_neural_plan([], allow_cpu_plan=False)


if __name__ == "__main__":
    unittest.main()
