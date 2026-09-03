"""Explicit GPU selection and execution checks for the PyTorch oracle."""

from __future__ import annotations

import os
from typing import Any


def execution_backend(torch: Any, device: str) -> str:
    """HIP intentionally shares PyTorch's CUDA namespace, not NVIDIA hardware."""
    if device == "cuda" and getattr(getattr(torch, "version", None), "hip", None):
        return "rocm"
    return "metal" if device == "mps" else device


def rocm_capabilities(torch: Any) -> dict[str, Any]:
    hip = getattr(getattr(torch, "version", None), "hip", None)
    available = bool(hip and torch.cuda.is_available())
    result: dict[str, Any] = {
        "available": available, "hip_version": hip,
        "device_count": torch.cuda.device_count() if available else 0,
        "kernel_usage_verified": False,
    }
    if available:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "device_index": index, "device_name": properties.name,
            "architecture": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": properties.total_memory,
        })
    return result


def tensor_core_capabilities(torch: Any, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "support": "unavailable", "fp16_eligible": False,
        "bf16_eligible": False, "tf32_eligible": False,
        "dispatch": "runtime-auto", "usage_verified": False,
        "evidence": "architecture-and-device-family; not a kernel profile",
    }
    if (device != "cuda" or getattr(getattr(torch, "version", None), "hip", None)
            or not torch.cuda.is_available()):
        return result
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    major, minor = properties.major, properties.minor
    name = properties.name.upper()
    # Compute capability 7.5 also includes non-Tensor GTX 16 GPUs. Do not
    # equate CUDA availability (or every Turing device) with Tensor Cores.
    support = "unknown"
    if major < 7 or (major == 7 and minor == 5 and "GTX 16" in name):
        support = "unsupported"
    elif (major >= 8 or (major == 7 and minor in (0, 2))
          or (major == 7 and minor == 5 and (
              "RTX" in name or name in ("T4", "TESLA T4", "NVIDIA T4", "NVIDIA TESLA T4")))):
        support = "supported"
    result.update({
        "support": support, "device_name": properties.name, "device_index": index,
        "compute_capability": [major, minor], "fp16_eligible": support == "supported",
        "bf16_eligible": support == "supported" and major >= 8,
        "tf32_eligible": support == "supported" and major >= 8,
    })
    return result


def configure_tensor_cores(torch: Any, device: str, requested_tf32: bool | None) -> dict[str, Any]:
    capabilities = tensor_core_capabilities(torch, device)
    capabilities["requested_tf32"] = requested_tf32
    if execution_backend(torch, device) == "rocm":
        if requested_tf32 is not None:
            raise ValueError("NVIDIA TF32 options do not apply to AMD ROCm execution.")
        # Do not change AMD BLAS/MIOpen policy through NVIDIA-specific controls.
        return capabilities
    if device != "cuda":
        if requested_tf32 is not None:
            raise ValueError("--cuda-tf32/--no-cuda-tf32 requires CUDA execution.")
        return capabilities
    if requested_tf32 and not capabilities["tf32_eligible"]:
        raise ValueError("TF32 requires an eligible NVIDIA Ampere-or-newer GPU.")
    enabled = capabilities["tf32_eligible"] if requested_tf32 is None else requested_tf32
    precision = "tf32" if enabled else "ieee"
    # Use the pinned PyTorch >=2.9 API; never mix it with legacy allow_tf32.
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.conv.fp32_precision = precision
    capabilities["fp32_matmul_precision"] = torch.backends.cuda.matmul.fp32_precision
    capabilities["fp32_conv_precision"] = torch.backends.cudnn.conv.fp32_precision
    if (capabilities["fp32_matmul_precision"] != precision
            or capabilities["fp32_conv_precision"] != precision):
        raise RuntimeError("PyTorch did not apply the requested TF32 precision policy.")
    return capabilities


def select_device(torch: Any, requested: str) -> str:
    device = "mps" if requested == "metal" else requested
    if device == "rocm":
        if not getattr(getattr(torch, "version", None), "hip", None):
            raise SystemExit(
                "ROCm requires an AMD HIP-enabled PyTorch build. Install the build "
                "matching your Radeon GPU, operating system, and AMD driver."
            )
        if not torch.cuda.is_available():
            raise SystemExit("ROCm was requested but no usable AMD GPU is available in PyTorch.")
        device = "cuda"
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            raise SystemExit(
                "No usable GPU is available. Install a CUDA/ROCm/Metal-enabled PyTorch runtime "
                "or explicitly select --device cpu. Automatic CPU fallback is disabled."
            )
    if device not in ("mps", "cuda", "cpu"):
        raise SystemExit(f"Unsupported PyTorch device: {requested}")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Metal (MPS) was requested but is not available in PyTorch.")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise SystemExit(
                "PYTORCH_ENABLE_MPS_FALLBACK=1 permits silent CPU operations. Unset it "
                "or set it to 0 for GPU generation. Select CPU work explicitly with "
                "--cpu-text-encoding or --device cpu, not kernel-error fallback."
            )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in PyTorch.")
    return device


def accelerator_preflight(torch: Any, device: str, dtype: Any) -> dict[str, Any]:
    """Fail before loading model weights if the selected arithmetic cannot execute."""
    backend = execution_backend(torch, device)
    try:
        matrix = torch.ones((16, 16), device=device, dtype=dtype)
        result = matrix @ matrix
        if result.device.type != device:
            raise RuntimeError(
                f"Hardware preflight used device {result.device.type}, expected {device}."
            )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except RuntimeError as error:
        if backend == "rocm":
            raise RuntimeError(
                f"ROCm arithmetic preflight failed: {error}. Check AMD's GPU/OS/runtime "
                "compatibility matrix; no CPU fallback was attempted."
            ) from error
        raise
    if not torch.all(result == 16).item():
        raise RuntimeError("Hardware preflight returned an incorrect matrix result.")
    report = {
        "runtime": "pytorch",
        "backend": backend,
        "device": device,
        "gpu_accelerated": device != "cpu",
        "default_policy": "accelerator-required",
        "cpu_fallback_allowed": False,
        "matmul_verified": True,
    }
    if backend == "rocm":
        report["rocm"] = rocm_capabilities(torch)
    return report


def validate_execution_device(pipeline: Any, selected: str) -> str:
    execution = getattr(pipeline, "_execution_device", None)
    actual = getattr(execution, "type", None)
    if actual != selected:
        raise RuntimeError(
            f"Pipeline execution device is {actual!r}; selected hardware is {selected!r}."
        )
    return actual


def main() -> int:
    """Model-free hardware probe, also usable before downloading any weights."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Verify the selected PyTorch arithmetic without a model.")
    parser.add_argument("--device", choices=("auto", "metal", "mps", "cuda", "rocm", "cpu"),
                        default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install a hardware-compatible PyTorch runtime first.") from error
    device = select_device(torch, args.device)
    tensor_cores = configure_tensor_cores(torch, device, None)
    report = accelerator_preflight(torch, device, getattr(torch, args.dtype))
    report.update({"torch_version": torch.__version__, "requested_device": args.device,
                   "dtype": args.dtype, "tensor_cores": tensor_cores})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
