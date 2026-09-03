#!/usr/bin/env python3
"""AMD device policy tests; simulated devices are not physical ROCm evidence."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import hardware


def rocm_torch(*, available=True, hip="7.2.1"):
    return SimpleNamespace(
        __version__="2.13.0+rocm-test",
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(
            is_available=lambda: available, current_device=lambda: 1,
            device_count=lambda: 2 if available else 0,
            get_device_properties=lambda index: SimpleNamespace(
                name="AMD Radeon RX 7900 XTX", gcnArchName="gfx1100",
                total_memory=24 * 1024**3),
            synchronize=Mock(),
        ),
        # No CUDA/cuDNN precision setters: AMD must never touch them.
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


class RocmPolicyTests(unittest.TestCase):
    def test_explicit_rocm_is_a_valid_cli_device(self):
        args = generate.build_parser().parse_args(["--device", "rocm"])
        self.assertEqual(args.device, "rocm")

    def test_rocm_uses_pytorch_cuda_namespace(self):
        torch = rocm_torch()
        for device in ("rocm", "auto", "cuda"):
            with self.subTest(device=device):
                self.assertEqual(hardware.select_device(torch, device), "cuda")

    def test_explicit_rocm_needs_hip_build_and_usable_gpu(self):
        for torch in (rocm_torch(hip=None), rocm_torch(available=False)):
            with self.assertRaisesRegex(SystemExit, "ROCm"):
                hardware.select_device(torch, "rocm")

    def test_rocm_does_not_fall_back_to_cpu(self):
        with self.assertRaisesRegex(SystemExit, "GPU.*CPU fallback"):
            hardware.select_device(rocm_torch(available=False), "auto")

    def test_amd_inventory_records_actual_runtime_properties(self):
        report = hardware.rocm_capabilities(rocm_torch())
        self.assertTrue(report["available"])
        self.assertEqual(report["hip_version"], "7.2.1")
        self.assertEqual(report["device_count"], 2)
        self.assertEqual(report["device_index"], 1)
        self.assertEqual(report["device_name"], "AMD Radeon RX 7900 XTX")
        self.assertEqual(report["architecture"], "gfx1100")
        self.assertEqual(report["total_memory_bytes"], 24 * 1024**3)
        self.assertFalse(report["kernel_usage_verified"])

    def test_no_hip_is_not_an_amd_gpu(self):
        torch = rocm_torch(hip=None)
        torch.cuda.get_device_properties = Mock(side_effect=AssertionError("not AMD"))
        report = hardware.rocm_capabilities(torch)
        self.assertFalse(report["available"])
        self.assertEqual(report["device_count"], 0)

    def test_hip_build_without_gpu_keeps_version_but_no_device(self):
        report = hardware.rocm_capabilities(rocm_torch(available=False))
        self.assertFalse(report["available"])
        self.assertEqual(report["hip_version"], "7.2.1")
        self.assertNotIn("device_name", report)

    def test_amd_never_uses_nvidia_tf32_settings(self):
        result = hardware.configure_tensor_cores(rocm_torch(), "cuda", None)
        self.assertEqual(result["support"], "unavailable")
        self.assertFalse(result["usage_verified"])
        self.assertNotIn("fp32_matmul_precision", result)
        for setting in (True, False):
            with self.assertRaisesRegex(ValueError, "NVIDIA.*TF32|TF32.*NVIDIA"):
                hardware.configure_tensor_cores(rocm_torch(), "cuda", setting)

    def test_preflight_runs_rocm_math_and_reports_amd_not_nvidia(self):
        torch = rocm_torch()
        tensor = Mock()
        tensor.device = SimpleNamespace(type="cuda")
        tensor.__matmul__ = Mock(return_value=tensor)
        tensor.__eq__ = Mock(return_value=tensor)
        torch.ones = Mock(return_value=tensor)
        torch.all = Mock(return_value=SimpleNamespace(item=lambda: True))
        report = hardware.accelerator_preflight(torch, "cuda", "float16")
        torch.ones.assert_called_once_with((16, 16), device="cuda", dtype="float16")
        torch.cuda.synchronize.assert_called_once()
        self.assertEqual(report["backend"], "rocm")
        self.assertEqual(report["device"], "cuda")
        self.assertEqual(report["rocm"]["architecture"], "gfx1100")
        self.assertTrue(report["matmul_verified"])
        self.assertFalse(report["cpu_fallback_allowed"])

    def test_amd_preflight_failure_is_not_retried_on_cpu(self):
        torch = rocm_torch()
        torch.ones = Mock(side_effect=RuntimeError("no kernel image available"))
        with self.assertRaisesRegex(RuntimeError, "ROCm.*no kernel image"):
            hardware.accelerator_preflight(torch, "cuda", "float16")
        torch.ones.assert_called_once()

    def test_backend_name_is_runtime_specific(self):
        self.assertEqual(hardware.execution_backend(rocm_torch(), "cuda"), "rocm")
        self.assertEqual(hardware.execution_backend(rocm_torch(hip=None), "cuda"), "cuda")
        self.assertEqual(hardware.execution_backend(rocm_torch(), "cpu"), "cpu")
        self.assertEqual(hardware.execution_backend(rocm_torch(), "mps"), "metal")

    def test_rocm_requirements_do_not_pin_the_standard_torch_build(self):
        requirements = ROOT / "reference" / "diffusers"
        standard = (requirements / "requirements.txt").read_text()
        amd = (requirements / "requirements-rocm.txt").read_text()
        common = (requirements / "requirements-common.txt").read_text()
        self.assertIn("torch==2.13.0", standard)
        self.assertIn("-r requirements-common.txt", amd)
        self.assertNotIn("torch==", "\n".join(
            line for line in (amd + common).splitlines() if not line.startswith("#")))


if __name__ == "__main__":
    unittest.main()
