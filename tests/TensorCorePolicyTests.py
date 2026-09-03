#!/usr/bin/env python3
"""Tensor Core eligibility is separate from measured kernel utilization."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "diffusers"))
import generate
import hardware


def cuda_torch(name="NVIDIA RTX 4090", major=8, minor=9):
    return SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(name=name, major=major, minor=minor)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="ieee")),
            cudnn=SimpleNamespace(conv=SimpleNamespace(fp32_precision="ieee"))),
    )


class TensorCorePolicyTests(unittest.TestCase):
    def test_architecture_and_device_family_are_both_used(self):
        for name, major, minor, expected in (
            ("NVIDIA RTX 4090", 8, 9, "supported"),
            ("NVIDIA A100", 8, 0, "supported"),
            ("NVIDIA Tesla V100", 7, 0, "supported"),
            ("NVIDIA GeForce RTX 2080", 7, 5, "supported"),
            ("Tesla T4", 7, 5, "supported"),
            ("NVIDIA GeForce GTX 1650", 7, 5, "unsupported"),
            ("Unidentified Turing GPU", 7, 5, "unknown"),
            ("NVIDIA GTX 1080", 6, 1, "unsupported"),
        ):
            with self.subTest(name=name):
                value = hardware.tensor_core_capabilities(cuda_torch(name, major, minor), "cuda")
                self.assertEqual(value["support"], expected)
                self.assertFalse(value["usage_verified"])

    def test_auto_enables_tf32_only_for_ampere_or_newer(self):
        for major, minor, expected in ((8, 0, "tf32"), (7, 0, "ieee")):
            torch = cuda_torch("Tesla V100" if major == 7 else "A100", major, minor)
            result = hardware.configure_tensor_cores(torch, "cuda", None)
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, expected)
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, expected)
            self.assertEqual(result["fp32_matmul_precision"], expected)
            self.assertFalse(result["usage_verified"])

    def test_tf32_can_be_disabled_without_claiming_fp16_cores_disabled(self):
        torch = cuda_torch()
        result = hardware.configure_tensor_cores(torch, "cuda", False)
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        self.assertTrue(result["fp16_eligible"])
        self.assertEqual(result["dispatch"], "runtime-auto")

    def test_unsupported_explicit_tf32_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TF32"):
            hardware.configure_tensor_cores(cuda_torch("V100", 7, 0), "cuda", True)
        for device in ("cpu", "mps"):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                hardware.configure_tensor_cores(SimpleNamespace(), device, True)
            result = hardware.configure_tensor_cores(SimpleNamespace(), device, None)
            self.assertEqual(result["support"], "unavailable")

    def test_rocm_is_not_reported_as_nvidia_tensor_cores(self):
        torch = cuda_torch("AMD Instinct MI300", 9, 4)
        torch.version.hip = "6.4"
        self.assertEqual(hardware.tensor_core_capabilities(torch, "cuda")["support"], "unavailable")
        with self.assertRaisesRegex(ValueError, "TF32"):
            hardware.configure_tensor_cores(torch, "cuda", True)

    def test_cli_default_is_auto_and_explicit_disable_is_preserved(self):
        self.assertIsNone(generate.build_parser().parse_args([]).cuda_tf32)
        self.assertFalse(generate.build_parser().parse_args(["--no-cuda-tf32"]).cuda_tf32)
        self.assertTrue(generate.build_parser().parse_args(["--cuda-tf32"]).cuda_tf32)


if __name__ == "__main__":
    unittest.main()
