"""Normalize heterogeneous diffusion checkpoints to one explicit DiT tensor contract.

The target DiT checkpoint is the executable architecture.  Legacy tensors are
deterministically transplanted into that contract for merge bootstrapping; this
is not a learned architecture distillation and does not claim semantic parity.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from downloaded_model import _tensor_identity
from weight_files import (LocalWeightFile, cached_model_sha256, file_signature,
                          resolve_weight_file, verify_weight_file, file_sha256)


STANDARD = "iild-dit-standard-v1"
POLICY = "role-depth-stat-match-template-fill-v1"
SOURCE_FAMILIES = ("auto", "sd15", "sdxl", "haruka", "hoshino", "illustrious",
                   "dit", "tsubaki", "reference-pro")
TARGET_FAMILIES = ("auto", "tsubaki", "dit")
_DIT_ARCHITECTURES = frozenset({"anima", "sd3", "flux1", "flux1-dev", "flux1-schnell"})
_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_ALIASES = {"haruka": "sdxl", "hoshino": "sdxl", "illustrious": "sdxl",
            "tsubaki": "dit"}


def _path(value, role):
    if not isinstance(value, (str, Path)) or not str(value).strip() or "://" in str(value):
        raise ValueError(f"{role} requires a nonempty local path.")
    return Path(value).expanduser().absolute()


def _open(path, role, stack, safetensors, *, hash_content):
    selected = _path(path, role)
    if hash_content:
        weight = resolve_weight_file(str(selected), role)
    else:
        if selected.suffix.lower() not in (".safetensors", ".safetensor"):
            raise ValueError(f"{role} requires a .safetensors or .safetensor file.")
        if not selected.is_file() or selected.stat().st_size == 0:
            raise ValueError(f"{role} file is missing, empty, or not a regular file: {selected}")
        signature = file_signature(selected)
        weight = LocalWeightFile(str(selected), signature[0], None, signature[3], signature)
    try:
        reader = stack.enter_context(safetensors.safe_open(weight.resolved_file, framework="pt", device="cpu"))
    except Exception as error:
        raise ValueError(f"Cannot read {role} safetensors: {error}") from error
    keys = list(reader.keys())
    if not keys:
        raise ValueError(f"{role} contains no tensors.")
    return weight, reader, {key: tuple(reader.get_slice(key).get_shape()) for key in keys}


def _explicit_family(value, *, target=False):
    choices = TARGET_FAMILIES if target else SOURCE_FAMILIES
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{'Target' if target else 'Source'} family must be one of: {', '.join(choices)}.")
    return _ALIASES.get(value, value)


def _source_family(value, shapes):
    family = _explicit_family(value)
    architecture, role, _, task = _tensor_identity(shapes)
    if role in ("lora", "embedding", "vae", "controlnet"):
        raise ValueError("Source must be a full checkpoint, not an adapter or standalone component.")
    if role != "checkpoint" and family == "auto":
        raise ValueError("Could not determine source family from a full checkpoint; choose --source-family explicitly.")
    if family != "auto":
        return family, architecture, role, task
    if architecture == "sd1":
        return "sd15", architecture, role, task
    if architecture and architecture.startswith("sdxl"):
        return "sdxl", architecture, role, task
    if architecture in _DIT_ARCHITECTURES:
        return "dit", architecture, role, task
    raise ValueError("Could not determine source family; select sd15, sdxl/Haruka/Hoshino/Illustrious, DiT/Tsubaki, or Reference Pro explicitly.")


def _target_family(value, shapes, metadata):
    family = _explicit_family(value, target=True)
    architecture, role, _, _ = _tensor_identity(shapes)
    declaration = " ".join(str(metadata.get(key, "")) for key in
                           ("modelspec.architecture", "modelspec.implementation", "iild.model_family")).casefold()
    declared_dit = any(token in declaration for token in ("dit", "tsubaki", "transformer"))
    if family == "auto" and not (architecture in _DIT_ARCHITECTURES or declared_dit):
        raise ValueError("Target checkpoint is not recognizably DiT; use a Tsubaki/DiT template or declare --target-family tsubaki.")
    if family == "auto":
        family = "tsubaki" if "tsubaki" in declaration else "dit"
    if role in ("lora", "embedding", "vae", "controlnet"):
        raise ValueError("Target DiT must be a full checkpoint, not an adapter or component.")
    return family, architecture


def _identity(name):
    lowered = name.casefold().replace("_", ".")
    component = ("vae" if any(token in lowered for token in ("vae", "first.stage", "decoder", "encoder.conv"))
                 else "text" if any(token in lowered for token in ("text", "cond.stage", "clip", "t5", "llm"))
                 else "denoiser")
    roles = (
        ("q", ("to.q", "q.proj", ".q.weight")),
        ("k", ("to.k", "k.proj", ".k.weight")),
        ("v", ("to.v", "v.proj", ".v.weight")),
        ("out", ("to.out", "out.proj", "proj.out")),
        ("qkv", ("qkv", "in.proj")),
        ("ff-up", ("ff.net.0", "fc1", "up.proj")),
        ("ff-down", ("ff.net.2", "fc2", "down.proj")),
        ("norm", ("norm", "adaln")),
        ("embed", ("embed", "position", "pos.")),
    )
    role = next((label for label, tokens in roles if any(token in lowered for token in tokens)),
                "conv" if "conv" in lowered else "bias" if lowered.endswith(".bias") else "other")
    numbers = re.findall(r"(?:blocks?|layers?)\.(\d+)", lowered)
    return component, role, int(numbers[0]) if numbers else 0


def _score(source, target_key, target_shape):
    key, shape, identity = source
    target_identity = _identity(target_key)
    source_count, target_count = math.prod(shape), math.prod(target_shape)
    size_distance = abs(math.log2(max(source_count, 1) / max(target_count, 1)))
    first_distance = abs(math.log2(max(shape[0], 1) / max(target_shape[0], 1)))
    return (identity[0] != target_identity[0], identity[1] != target_identity[1],
            len(shape) != len(target_shape), abs(identity[2] - target_identity[2]),
            first_distance + size_distance, key)


def _mapping(source_shapes, source_reader, target_shapes, target_reader):
    floating_sources = []
    for key, shape in source_shapes.items():
        value = source_reader.get_slice(key)
        if shape and value.get_dtype() in _FLOAT_DTYPES and key not in ("v_pred", "ztsnr"):
            floating_sources.append((key, shape, _identity(key)))
    mappings = {}
    reports = []
    for key, target_shape in target_shapes.items():
        target_slice = target_reader.get_slice(key)
        if not target_shape or target_slice.get_dtype() not in _FLOAT_DTYPES:
            continue
        exact = next((entry for entry in floating_sources if entry[0] == key), None)
        selected = exact or (min(floating_sources, key=lambda item: _score(item, key, target_shape))
                             if floating_sources else None)
        if selected is None:
            continue
        source_key, source_shape, _ = selected
        retained = min(source_shape[0], target_shape[0]) * min(
            math.prod(source_shape[1:]), math.prod(target_shape[1:]))
        report = {"policy": POLICY, "semantic_equivalence": False, "source_tensor": source_key,
                  "target_tensor": key, "selection": "exact-name" if exact else "role-depth-shape-nearest",
                  "source_shape": list(source_shape), "target_shape": list(target_shape),
                  "retained_values": retained, "template_filled_values": math.prod(target_shape) - retained,
                  "cropped_values": math.prod(source_shape) - retained}
        mappings[key] = report
        reports.append(report)
    return mappings, reports


def _plan(source, target_dit, output, source_family, target_family, transfer_strength, *, hash_content):
    if (isinstance(transfer_strength, bool) or not isinstance(transfer_strength, (int, float))
            or not math.isfinite(transfer_strength) or not 0 <= transfer_strength <= 1):
        raise ValueError("Transfer strength must be finite and in [0, 1].")
    destination = _path(output, "Output")
    if destination.suffix.lower() not in (".safetensors", ".safetensor"):
        raise ValueError("DiT conversion output must end in .safetensors or .safetensor.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Conversion output already exists: {destination}")
    import safetensors
    with ExitStack() as stack:
        source_weight, source_reader, source_shapes = _open(
            source, "Source checkpoint", stack, safetensors, hash_content=hash_content)
        target_weight, target_reader, target_shapes = _open(
            target_dit, "Target DiT template", stack, safetensors, hash_content=hash_content)
        if Path(source_weight.resolved_file) == Path(target_weight.resolved_file):
            raise ValueError("Source and target DiT template must be different files.")
        if destination.resolve() in (Path(source_weight.resolved_file), Path(target_weight.resolved_file)):
            raise ValueError("Conversion output must not replace an input.")
        family, source_architecture, source_role, source_task = _source_family(source_family, source_shapes)
        target_name, target_architecture = _target_family(target_family, target_shapes, target_reader.metadata() or {})
        mappings, reports = _mapping(source_shapes, source_reader, target_shapes, target_reader)
        if not mappings:
            raise ValueError("Source and target contain no transferable floating tensors.")
        report = {"schema": "iild-dit-conversion-v1", "standard": STANDARD,
                  "policy": POLICY, "semantic_equivalence": False,
                  "source": str(Path(source_weight.path).absolute()), "source_family": family,
                  "source_architecture": source_architecture, "source_role": source_role,
                  "source_task": source_task, "target_template": str(Path(target_weight.path).absolute()),
                  "target_family": target_name, "target_architecture": target_architecture,
                  "output": str(destination), "transfer_strength": float(transfer_strength),
                  "target_tensor_count": len(target_shapes), "transferred_tensor_count": len(mappings),
                  "preserved_tensor_count": len(target_shapes) - len(mappings), "mappings": reports,
                  "inspection": "structure-only" if not hash_content else "complete"}
        return report, source_weight, target_weight, mappings


def inspect_dit_conversion(source, target_dit, output=None, *, source_family="auto",
                           target_family="auto", transfer_strength=1.0):
    destination = output if output is not None else Path(source).expanduser().absolute().with_name(
        Path(source).stem + "-dit.safetensors")
    return _plan(source, target_dit, destination, source_family, target_family,
                 transfer_strength, hash_content=False)[0]


def _adapt(source, template, target_shape, torch):
    source_matrix = source.to(dtype=torch.float64).reshape(source.shape[0], -1)
    template_matrix = template.to(dtype=torch.float64).reshape(template.shape[0], -1)
    rows = min(source_matrix.shape[0], template_matrix.shape[0])
    columns = min(source_matrix.shape[1], template_matrix.shape[1])
    selected = source_matrix[:rows, :columns]
    target_selected = template_matrix[:rows, :columns]
    source_rms = torch.sqrt(torch.mean(selected.square())) if selected.numel() else torch.tensor(0.)
    target_rms = torch.sqrt(torch.mean(target_selected.square())) if target_selected.numel() else torch.tensor(0.)
    scale = target_rms / source_rms if source_rms.item() > 0 and target_rms.item() > 0 else 1.0
    transplanted = template_matrix.clone()
    transplanted[:rows, :columns] = selected * scale
    return transplanted.reshape(target_shape)


def convert_to_dit(source, target_dit, output, *, source_family="auto", target_family="auto",
                   transfer_strength=1.0):
    try:
        torch = importlib.import_module("torch")
        safetensors = importlib.import_module("safetensors")
        serializer = importlib.import_module("safetensors.torch")
    except ImportError as error:
        raise RuntimeError("DiT conversion requires the existing Torch and safetensors environment.") from error
    report, source_weight, target_weight, mappings = _plan(
        source, target_dit, output, source_family, target_family, transfer_strength, hash_content=True)
    report["source_sha256"] = cached_model_sha256(Path(source_weight.resolved_file))
    report["target_template_sha256"] = cached_model_sha256(Path(target_weight.resolved_file))
    destination = Path(report["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, torch.no_grad():
        source_reader = stack.enter_context(safetensors.safe_open(source_weight.resolved_file, framework="pt", device="cpu"))
        target_reader = stack.enter_context(safetensors.safe_open(target_weight.resolved_file, framework="pt", device="cpu"))
        tensors = {}
        for key in target_reader.keys():
            template = target_reader.get_tensor(key)
            if template.is_floating_point() and not torch.isfinite(template).all().item():
                raise ValueError(f"Target DiT template tensor must be finite: {key}")
            mapping = mappings.get(key)
            if mapping is None:
                tensors[key] = template.clone().contiguous()
                continue
            source_tensor = source_reader.get_tensor(mapping["source_tensor"])
            if not torch.isfinite(source_tensor).all().item():
                raise ValueError(f"Source tensor must be finite: {mapping['source_tensor']}")
            transplanted = _adapt(source_tensor, template, tuple(mapping["target_shape"]), torch)
            result = template.to(dtype=torch.float64).lerp(transplanted, report["transfer_strength"])
            result = result.to(dtype=template.dtype).contiguous()
            if not torch.isfinite(result).all().item():
                raise ValueError(f"Converted tensor is not finite: {key}")
            tensors[key] = result
        metadata = dict(sorted((target_reader.metadata() or {}).items()))
        embedded = {key: value for key, value in report.items()
                    if key not in ("output", "mappings", "inspection")}
        metadata.update({"format": metadata.get("format", "pt"), "iild.dit.standard": STANDARD,
                         "iild.model_family": report["target_family"],
                         "iild.dit.template_sha256": report["target_template_sha256"],
                         "iild_dit_conversion": json.dumps(embedded, sort_keys=True)})
        metadata = dict(sorted(metadata.items()))
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
            staging = Path(temporary) / destination.name
            serializer.save_file(tensors, str(staging), metadata=metadata)
            del tensors
            verify_weight_file(source_weight, "conversion source")
            verify_weight_file(target_weight, "DiT template")
            if cached_model_sha256(Path(source_weight.resolved_file)) != report["source_sha256"]:
                raise RuntimeError("Conversion source changed while building the output.")
            if cached_model_sha256(Path(target_weight.resolved_file)) != report["target_template_sha256"]:
                raise RuntimeError("DiT template changed while building the output.")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Conversion output already exists: {destination}")
            os.link(staging, destination)
    report["output_sha256"] = file_sha256(destination)
    report["output_size_bytes"] = destination.stat().st_size
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Convert SD/SDXL/DiT/editing checkpoints to a Tsubaki DiT tensor contract for merging.")
    parser.add_argument("--source", required=True, help="Source single-file safetensors checkpoint.")
    parser.add_argument("--target-dit", required=True, help="Tsubaki/DiT checkpoint defining the output tensor contract.")
    parser.add_argument("--output", required=True, help="New .safetensors output; never overwritten.")
    parser.add_argument("--source-family", choices=SOURCE_FAMILIES, default="auto",
                        help="Auto, SD 1.5, SDXL aliases (Haruka/Hoshino/Illustrious), DiT/Tsubaki, or Reference Pro.")
    parser.add_argument("--target-family", choices=TARGET_FAMILIES, default="auto",
                        help="Declare Tsubaki only when the template metadata does not identify its DiT architecture.")
    parser.add_argument("--transfer-strength", type=float, default=1.0,
                        help="Blend of transplanted values into the executable DiT template, in [0,1].")
    parser.add_argument("--inspect", action="store_true", help="Validate and print the structural mapping without hashing or writing output.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        function = inspect_dit_conversion if args.inspect else convert_to_dit
        report = function(args.source, args.target_dit, args.output, source_family=args.source_family,
                          target_family=args.target_family, transfer_strength=args.transfer_strength)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"DiT conversion failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
