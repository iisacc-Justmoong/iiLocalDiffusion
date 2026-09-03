#!/usr/bin/env python3
"""Regression tests for the accelerator-required generation policy."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "diffusers"
sys.path.insert(0, str(REFERENCE))

import hardware
import generate


def fake_torch(*, metal=False, cuda=False):
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: metal)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


class HardwarePolicyTests(unittest.TestCase):
    def test_auto_selects_cuda_or_metal_without_cpu_fallback(self):
        self.assertEqual(hardware.select_device(fake_torch(cuda=True), "auto"), "cuda")
        self.assertEqual(hardware.select_device(fake_torch(metal=True), "auto"), "mps")
        with self.assertRaisesRegex(SystemExit, "GPU.*--device cpu"):
            hardware.select_device(fake_torch(), "auto")

    def test_explicit_metal_is_an_mps_alias(self):
        self.assertEqual(hardware.select_device(fake_torch(metal=True), "metal"), "mps")
        self.assertEqual(hardware.select_device(fake_torch(metal=True), "mps"), "mps")
        self.assertEqual(generate.build_parser().parse_args(["--device", "metal"]).device,
                         "metal")

    def test_explicit_accelerators_fail_instead_of_changing_device(self):
        for requested in ("metal", "mps", "cuda"):
            with self.subTest(requested=requested), self.assertRaises(SystemExit):
                hardware.select_device(fake_torch(), requested)
        with self.assertRaises(SystemExit):
            hardware.select_device(fake_torch(metal=True), "mlx")

    def test_cpu_requires_explicit_selection(self):
        self.assertEqual(hardware.select_device(fake_torch(metal=True), "cpu"), "cpu")
        self.assertEqual(hardware.select_device(fake_torch(), "cpu"), "cpu")

    def test_mps_fallback_environment_is_rejected_for_metal(self):
        with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(SystemExit, "--cpu-text-encoding or --device cpu"):
                hardware.select_device(fake_torch(metal=True), "auto")
            self.assertEqual(hardware.select_device(fake_torch(), "cpu"), "cpu")
            self.assertEqual(hardware.select_device(fake_torch(cuda=True), "cuda"), "cuda")

    def test_explicitly_disabled_mps_fallback_is_supported(self):
        with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            self.assertEqual(hardware.select_device(fake_torch(metal=True), "auto"), "mps")

    def test_pipeline_execution_device_must_match_selected_hardware(self):
        for device in ("mps", "cuda", "cpu"):
            with self.subTest(device=device):
                pipeline = SimpleNamespace(_execution_device=SimpleNamespace(type=device))
                self.assertEqual(hardware.validate_execution_device(pipeline, device), device)
        wrong = SimpleNamespace(_execution_device=SimpleNamespace(type="cpu"))
        with self.assertRaisesRegex(RuntimeError, "execution device"):
            hardware.validate_execution_device(wrong, "mps")
        with self.assertRaisesRegex(RuntimeError, "execution device"):
            hardware.validate_execution_device(SimpleNamespace(), "cuda")

    def test_preflight_runs_and_synchronizes_on_selected_gpu(self):
        events = []

        class Tensor:
            device = SimpleNamespace(type="mps")

            def __matmul__(self, other):
                events.append("matmul")
                return self

            def __eq__(self, other):
                self.expected = other
                return self

        tensor = Tensor()
        torch = fake_torch(metal=True)
        torch.ones = Mock(return_value=tensor)
        torch.all = Mock(return_value=SimpleNamespace(item=lambda: True))
        torch.mps = SimpleNamespace(synchronize=lambda: events.append("synchronize"))
        report = hardware.accelerator_preflight(torch, "mps", "fp16")
        self.assertEqual(events, ["matmul", "synchronize"])
        self.assertEqual(torch.ones.call_args.kwargs["device"], "mps")
        self.assertEqual(torch.ones.call_args.kwargs["dtype"], "fp16")
        self.assertEqual(tensor.expected, 16)
        self.assertTrue(report["matmul_verified"])
        self.assertTrue(report["gpu_accelerated"])
        self.assertEqual(report["backend"], "metal")
        self.assertEqual(report["runtime"], "pytorch")
        self.assertFalse(report["cpu_fallback_allowed"])

    def test_preflight_rejects_cpu_results_from_gpu_request(self):
        output = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        tensor = Mock()
        tensor.__matmul__ = Mock(return_value=output)
        torch = fake_torch(metal=True)
        torch.ones = Mock(return_value=tensor)
        with self.assertRaisesRegex(RuntimeError, "preflight.*device"):
            hardware.accelerator_preflight(torch, "mps", "fp16")

    def test_preflight_rejects_incorrect_gpu_result(self):
        tensor = Mock()
        tensor.device = SimpleNamespace(type="cuda")
        tensor.__matmul__ = Mock(return_value=tensor)
        tensor.__eq__ = Mock(return_value=tensor)
        torch = fake_torch(cuda=True)
        torch.ones = Mock(return_value=tensor)
        torch.all = Mock(return_value=SimpleNamespace(item=lambda: False))
        torch.cuda.synchronize = Mock()
        with self.assertRaisesRegex(RuntimeError, "preflight.*result"):
            hardware.accelerator_preflight(torch, "cuda", "fp16")


if __name__ == "__main__":
    unittest.main()
