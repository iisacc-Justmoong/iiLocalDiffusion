#!/usr/bin/env python3
"""Convert one explicitly named safetensors linear component for Core ML/ANE.

This is an offline component converter and numerical oracle, not a diffusion
pipeline or a downloader. A synthetic model is created only with --fixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
from weight_files import file_sha256, resolve_weight_file, verify_weight_file, weight_file_metadata


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def check_neural_plan(operations: list[dict], *, allow_cpu_plan: bool) -> int:
    count = sum(operation["preferred_device"] == "neural-engine" for operation in operations)
    if count == 0 and not allow_cpu_plan:
        raise RuntimeError("Core ML does not prefer Neural Engine for any operation; "
                           "use a suitable model/shape or explicitly --allow-cpu-plan for diagnostics")
    return count


def describe_plan(ct, compiled: Path) -> tuple[list[dict], int]:
    from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice
    from coremltools.models.compute_plan import MLComputePlan

    def label(device):
        if isinstance(device, MLNeuralEngineComputeDevice):
            return "neural-engine"
        if isinstance(device, MLGPUComputeDevice):
            return "gpu"
        if isinstance(device, MLCPUComputeDevice):
            return "cpu"
        return "unknown"

    plan = MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    operations = []

    def visit(block):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            operations.append({
                "operation": operation.operator_name,
                "preferred_device": label(usage.preferred_compute_device) if usage else "unknown",
                "supported_devices": [label(device) for device in usage.supported_compute_devices] if usage else [],
            })
            for child in operation.blocks:
                visit(child)

    visit(plan.model_structure.program.functions["main"].block)
    cores = sum(device.total_core_count for device in ct.models.MLModel.get_available_compute_devices()
                if isinstance(device, MLNeuralEngineComputeDevice))
    return operations, cores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Local .safetensors/.safetensor file; never downloaded")
    source.add_argument("--fixture", action="store_true", help="Explicit synthetic test weights, not a trained model")
    parser.add_argument("--weight-key", help="Exact rank-two [out,in] tensor key")
    parser.add_argument("--bias-key", help="Optional exact [out] tensor key")
    parser.add_argument("--fixture-width", type=positive_integer, default=768)
    parser.add_argument("--batch-size", type=positive_integer, default=512)
    parser.add_argument("--io-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=nonnegative_finite, default=0.005)
    parser.add_argument("--rtol", type=nonnegative_finite, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-cpu-plan", action="store_true")
    parser.add_argument("--native-test", type=Path, help="Also verify this artifact through the C++ CoreMLTests binary")
    args = parser.parse_args(argv)
    if args.model is not None and not args.weight_key:
        parser.error("--model requires --weight-key")
    if args.fixture and (args.weight_key or args.bias_key):
        parser.error("Tensor keys require --model, not --fixture")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    directory = args.output_dir.resolve()
    if not directory.is_relative_to((ROOT / "build").resolve()):
        parser.error("Conversion artifacts must stay under the repository build/ directory")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("--output-dir must be new or empty; existing evidence is not overwritten")
    if args.native_test and not args.native_test.is_file():
        parser.error("Build the native CoreMLTests target first")
    try:
        model_file = resolve_weight_file(args.model, "--model") if args.model is not None else None
    except ValueError as error:
        parser.error(str(error))

    import coremltools as ct
    import ml_dtypes  # Registers NumPy's bfloat16 dtype for safetensors input.
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from safetensors import safe_open

    if model_file:
        with safe_open(model_file.path, framework="np") as tensors:
            weights = tensors.get_tensor(args.weight_key)
            bias = tensors.get_tensor(args.bias_key) if args.bias_key else None
        source_dtype = str(weights.dtype)
        if source_dtype not in ("float16", "bfloat16", "float32") or (bias is not None and
                str(bias.dtype) not in ("float16", "bfloat16", "float32")):
            parser.error("Core ML conversion requires float16, bfloat16, or float32 source weights")
        weights = weights.astype(np.float32)
        bias = bias.astype(np.float32) if bias is not None else None
        source_metadata = dict(weight_file_metadata(model_file), weight_key=args.weight_key,
                               bias_key=args.bias_key, dtype=source_dtype)
    else:
        weights = np.full((args.fixture_width, args.fixture_width), 0.125, dtype=np.float32)
        bias = np.full(args.fixture_width, 0.5, dtype=np.float32)
        source_metadata = {"format": "synthetic-test-fixture", "trained_model": False}
    if weights.ndim != 2 or min(weights.shape) <= 0 or (bias is not None and bias.shape != (weights.shape[0],)):
        parser.error("Select weight [out,in] and optional matching bias [out]")
    for value in (weights, bias):
        if value is not None and (not np.isfinite(value).all() or (np.abs(value) > 65504).any()):
            parser.error("Weights must be finite and within float16's representable range")
    output_features, input_features = weights.shape
    io_dtype = np.float16 if args.io_precision == "fp16" else np.float32
    inputs = (np.random.default_rng(args.seed).standard_normal(
        (args.batch_size, input_features), dtype=np.float32) * 0.125).astype(io_dtype).astype(np.float32)
    expected = inputs @ weights.T
    if bias is not None:
        expected += bias
    packed_inputs = np.ascontiguousarray(inputs.T[None, :, None, :])
    packed_expected = np.ascontiguousarray(expected.T[None, :, None, :])
    # CPU-only Core ML is a diagnostic path for this float16 graph, not the
    # independent float32 oracle. A zero-input/bias case is exactly representable
    # and avoids relaxing ANE's numerical acceptance for CPU half accumulation.
    cpu_inputs = np.zeros_like(packed_inputs)
    cpu_expected = np.zeros_like(packed_expected)
    if bias is not None:
        cpu_expected[:] = bias.astype(np.float16).astype(np.float32)[None, :, None, None]
    converted_weights = weights.astype(io_dtype)[:, :, None, None]
    converted_bias = bias.astype(io_dtype) if bias is not None else None

    @mb.program(input_specs=[mb.TensorSpec(shape=packed_inputs.shape,
                dtype=types.fp16 if args.io_precision == "fp16" else types.fp32)],
                opset_version=ct.target.macOS13)
    def component(input):
        return mb.conv(x=input, weight=converted_weights, bias=converted_bias, name="output")

    directory.mkdir(parents=True, exist_ok=True)
    model = ct.convert(component, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True,
                       outputs=[ct.TensorType(name="output", dtype=io_dtype)])
    package = directory / "model.mlpackage"
    compiled = directory / "model.mlmodelc"
    model.save(str(package))
    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    operations, neural_cores = describe_plan(ct, compiled)
    plan_evidence = {"compute_units": "cpu+neural-engine", "neural_engine_cores": neural_cores,
                     "operations": operations, "hardware_usage_verified": False,
                     "evidence": "Core ML anticipated dispatch, not measured hardware counters"}
    (directory / "plan.json").write_text(json.dumps(plan_evidence, indent=2) + "\n", encoding="utf-8")
    neural_operations = check_neural_plan(operations, allow_cpu_plan=args.allow_cpu_plan)
    # Python only converts and creates the NumPy oracle. Predictions run in the
    # native process. This also avoids coremltools 9.0's Python buffer-deallocation
    # path on Core ML's asynchronous reset queue (observed on macOS 27/Python 3.13).
    oracle = {"inputs": {"input": packed_inputs.ravel().tolist()},
              "outputs": {"output": packed_expected.ravel().tolist()},
              "cpu_inputs": {"input": cpu_inputs.ravel().tolist()},
              "cpu_outputs": {"output": cpu_expected.ravel().tolist()},
              "atol": args.atol, "rtol": args.rtol,
              "require_neural_engine_plan": not args.allow_cpu_plan}
    with (directory / "oracle.json").open("w", encoding="utf-8") as stream:
        json.dump(oracle, stream, separators=(",", ":"), allow_nan=False)
    native_evidence = {"verified": False}
    if args.native_test:
        command = [str(args.native_test.resolve()), "--fixture", str(directory)]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        (directory / "native-stdout.log").write_text(process.stdout, encoding="utf-8")
        (directory / "native-stderr.log").write_text(process.stderr, encoding="utf-8")
        print(process.stdout, end="")
        print(process.stderr, end="", file=sys.stderr)
        process.check_returncode()
        reports = [json.loads(line.removeprefix("COREML_RESULT ")) for line in process.stdout.splitlines()
                   if line.startswith("COREML_RESULT ")]
        if len(reports) != 1 or reports[0].get("passed") is not True or reports[0].get("compute_units") != "cpu+neural-engine":
            raise RuntimeError("Native validation did not report a successful Core ML prediction")
        native_evidence = {"verified": True, "binary": str(args.native_test.resolve()),
                           "binary_sha256": file_sha256(args.native_test.resolve()), "result": reports[0]}
    if model_file:
        verify_weight_file(model_file, "Core ML source component")
    artifacts = {}
    for root in (package, compiled):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(directory))] = file_sha256(path)
    provenance = {
        "source": source_metadata, "seed": args.seed, "batch_size": args.batch_size,
        "input_features": input_features, "output_features": output_features,
        "io_precision": args.io_precision, "compute_precision": "float16",
        "host": platform.platform(), "coremltools_version": ct.__version__,
        "numpy_version": np.__version__, "ml_dtypes_version": ml_dtypes.__version__,
        "plan": plan_evidence, "neural_engine_preferred_operations": neural_operations,
        "native": native_evidence,
        "oracle": {"runtime": "NumPy", "dtype": "float32", "atol": args.atol, "rtol": args.rtol,
                   "max_abs_error": native_evidence.get("result", {}).get("max_abs_error"),
                   "file_sha256": file_sha256(directory / "oracle.json")},
        "artifacts": artifacts,
        "scope": "One named linear component; no whole diffusion pipeline, LoRA merge, or measured engine counters.",
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Core ML component: {compiled}")
    print("Native numerical validation:", "passed" if native_evidence["verified"] else "not requested")
    print(f"Neural Engine cores: {neural_cores}; preferred operations: {neural_operations}")
    print(f"Evidence: {directory / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
