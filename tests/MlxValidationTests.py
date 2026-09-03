#!/usr/bin/env python3
"""Lightweight validation-harness contracts; no model or GPU download needed."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "validation"))
import check_mlx_linear


class MlxValidationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="mlx-validation-", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = ["--model", "unused.safetensors", "--weight-key", "weight",
                     "--bias-key", "bias", "--output-dir", str(self.directory / "output")]

    def assert_rejected(self, arguments, message):
        error = io.StringIO()
        # All argument checks must happen before optional ML dependencies load.
        with patch.dict(sys.modules, {"torch": None}), redirect_stderr(error):
            with self.assertRaises(SystemExit) as failure:
                check_mlx_linear.main(self.base + arguments)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn(message, error.getvalue())

    def test_help_does_not_import_torch(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": None}), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                check_mlx_linear.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--native-test", output.getvalue())

    def test_batch_must_be_positive(self):
        self.assert_rejected(["--batch-size", "0"], "--batch-size must be positive")

    def test_output_must_be_under_build(self):
        self.assert_rejected(["--output-dir", str(ROOT / "outside-build")], "under the repository")

    def test_existing_evidence_is_preserved(self):
        output = self.directory / "output"
        output.mkdir()
        evidence = output / "provenance.json"
        evidence.write_text("previous evidence", encoding="utf-8")
        self.assert_rejected([], "existing evidence is not overwritten")
        self.assertEqual(evidence.read_text(encoding="utf-8"), "previous evidence")

    def test_output_cannot_be_a_regular_file(self):
        output = self.directory / "output"
        output.write_text("previous file", encoding="utf-8")
        self.assert_rejected([], "new or empty")
        self.assertEqual(output.read_text(encoding="utf-8"), "previous file")

    def test_native_test_binary_is_required(self):
        self.assert_rejected(["--native-test", str(self.directory / "missing")],
                             "Build the native ComputeTests target first")

    def test_native_result_requires_real_partition_and_budget(self):
        result = {"device": "metal", "passed": True, "batch": 4, "input_features": 3,
                  "output_features": 4, "hybrid": {"cpu_output_features": 1,
                  "gpu_output_features": 3, "staged_gpu_weight_bytes": 16,
                  "gpu_weight_budget_bytes": 32, "ram_weight_bytes": 64}}
        check_mlx_linear.validate_native_result(result, 4, 3, 4)
        for name, value in (("cpu_output_features", 0), ("gpu_output_features", 4),
                            ("staged_gpu_weight_bytes", 33), ("ram_weight_bytes", 0)):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "CPU/GPU"):
                check_mlx_linear.validate_native_result(
                    dict(result, hybrid=dict(result["hybrid"], **{name: value})), 4, 3, 4)
        with self.assertRaisesRegex(RuntimeError, "CPU/GPU"):
            check_mlx_linear.validate_native_result(dict(result, hybrid=None), 4, 3, 4)
        with self.assertRaisesRegex(RuntimeError, "shapes"):
            check_mlx_linear.validate_native_result(result, 1, 3, 4)
        with self.assertRaisesRegex(RuntimeError, "on a GPU"):
            check_mlx_linear.validate_native_result(dict(result, device="cpu"), 4, 3, 4)

    def test_one_output_retains_gpu_parity_without_impossible_partition(self):
        check_mlx_linear.validate_native_result({"device": "cuda", "passed": True,
            "batch": 1, "input_features": 3, "output_features": 1, "hybrid": None}, 1, 3, 1)


if __name__ == "__main__":
    unittest.main()
