#!/usr/bin/env python3
"""Compare a checkpoint's linear layer with MLX GPU and CPU/GPU/RAM execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
from weight_files import resolve_weight_file, verify_weight_file, weight_file_metadata


def validate_native_result(result: dict, batch: int, inputs: int, outputs: int) -> None:
    if result.get("device") not in ("metal", "cuda") or result.get("passed") is not True:
        raise RuntimeError("Native parity did not run on a GPU")
    if (result.get("batch"), result.get("input_features"), result.get("output_features")) != (
        batch, inputs, outputs
    ):
        raise RuntimeError("Native parity returned mismatched shapes")
    if outputs < 2:
        return
    hybrid = result.get("hybrid") or {}
    cpu = hybrid.get("cpu_output_features", 0)
    gpu = hybrid.get("gpu_output_features", 0)
    staged = hybrid.get("staged_gpu_weight_bytes", 0)
    budget = hybrid.get("gpu_weight_budget_bytes", 0)
    if (cpu <= 0 or gpu <= 0 or cpu + gpu != outputs or not 0 < staged <= budget
            or hybrid.get("ram_weight_bytes") != outputs * (inputs + 1) * 4):
        raise RuntimeError("Native parity did not verify CPU/GPU partition and bounded RAM staging")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local safetensors checkpoint/component")
    parser.add_argument("--weight-key", required=True)
    parser.add_argument("--bias-key", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-test", type=Path, default=ROOT / "build" / "ComputeTests")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "reference" / "hardware" / "mlx-linear")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    directory = args.output_dir.resolve()
    if not directory.is_relative_to((ROOT / "build").resolve()):
        parser.error("Validation artifacts must be kept under the repository build/ directory")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("--output-dir must be new or empty; existing evidence is not overwritten")
    if not args.native_test.is_file():
        parser.error("Build the native ComputeTests target first")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    model = resolve_weight_file(args.model, "--model")
    with safe_open(model.path, framework="pt", device="cpu") as source:
        weight = source.get_tensor(args.weight_key).contiguous()
        bias = source.get_tensor(args.bias_key).contiguous()
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        parser.error("Select a linear weight [out,in] and matching bias [out]")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or bias.dtype not in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        parser.error("Native linear weights require float16, bfloat16, or float32")
    generator = torch.Generator("cpu").manual_seed(args.seed)
    inputs = torch.randn(args.batch_size, weight.shape[1], generator=generator)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(inputs, weight.float(), bias.float())
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "weights.safetensors": {"weight": weight, "bias": bias},
        "input.safetensors": {"input": inputs},
        "expected.safetensors": {"output": expected},
    }
    for filename, tensors in fixtures.items():
        save_file(tensors, str(directory / filename))
    # MLX enables CUDA TF32 by default. Disable it only in this independent
    # strict-float32 comparison process, before MLX initializes its math mode.
    native_environment = dict(os.environ, MLX_ENABLE_TF32="0")
    process = subprocess.run([str(args.native_test.resolve()), str(directory)],
                             check=False, text=True, capture_output=True, env=native_environment)
    (directory / "native-stdout.log").write_text(process.stdout, encoding="utf-8")
    (directory / "native-stderr.log").write_text(process.stderr, encoding="utf-8")
    print(process.stdout, end="")
    print(process.stderr, end="", file=sys.stderr)
    process.check_returncode()
    result = json.loads((directory / "mlx-parity.json").read_text(encoding="utf-8"))
    validate_native_result(result, args.batch_size, weight.shape[1], weight.shape[0])
    verify_weight_file(model, "oracle model")
    provenance = {
        "model": weight_file_metadata(model),
        "weight_key": args.weight_key,
        "bias_key": args.bias_key,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "oracle_device": "cpu",
        "oracle_dtype": "float32",
        "native_environment": {"MLX_ENABLE_TF32": "0"},
        "native": result,
        "scope": "One named linear layer; not a complete MLX diffusion pipeline.",
        "fixtures": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in fixtures
        },
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evidence: {directory / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
