"""Local model merging with existing PyTorch arithmetic and safetensors I/O."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from model_merge_files import inspect_merge_model, validate_package_assets, validate_shard_indexes
from model_merge_lora import is_lora_key, prepare_lora
from model_merge_lora_targets import lora_target_aliases
from model_merge_options import MergeRequest, build_parser, resolve_merge_request, resolve_merge_weights
from weight_files import file_sha256

_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_BUFFER_DTYPES = frozenset({"BOOL", "U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"})
_ARCHITECTURE_HINTS = ("modelspec.architecture", "modelspec.implementation", "modelspec.prediction_type")


def _open_models(models, stack, safe_open):
    readers = []
    layouts = []
    kinds = []
    known_hints = {}
    for model in models:
        opened = {}
        layout = {}
        for name, weight in sorted(model.weights.items()):
            try:
                reader = stack.enter_context(safe_open(weight.resolved_file, framework="pt", device="cpu"))
            except Exception as error:
                raise ValueError(f"Cannot read safetensors merge input {weight.path}: {error}") from error
            opened[name] = reader
            component = str(PurePosixPath(name).parent) if model.root.is_dir() else "."
            if not reader.keys():
                raise ValueError(f"Weight file contains no tensor keys: {weight.path}")
            for key in reader.keys():
                address = (component, key)
                if address in layout:
                    raise ValueError(f"Duplicate tensor keys/weight variants in model: {address}")
                layout[address] = name
        if not layout:
            raise ValueError(f"Model contains no tensor keys: {model.root}")
        kind = "lora" if any(is_lora_key(key) for _, key in layout) else "checkpoint"
        if not layouts and kind != "checkpoint":
            raise ValueError("Base model must be a full checkpoint, not a LoRA adapter.")
        if kind == "checkpoint":
            if model.root.is_dir() != models[0].root.is_dir():
                raise ValueError("All checkpoints must use the same format: single files or Diffusers directories.")
            if model.root.is_dir() and "model_index.json" not in model.assets:
                raise ValueError("A full checkpoint directory requires model_index.json.")
            for name, reader in opened.items():
                component = str(PurePosixPath(name).parent) if model.root.is_dir() else "."
                for hint in _ARCHITECTURE_HINTS:
                    value = (reader.metadata() or {}).get(hint)
                    if value:
                        identity = (component, hint)
                        if identity in known_hints and value != known_hints[identity]:
                            raise ValueError(f"Model architecture metadata differs for {hint}: {model.root}")
                        known_hints[identity] = value
        elif "model_index.json" in model.assets:
            raise ValueError("LoRA material must be an adapter export, not a pipeline with embedded adapter keys.")
        if layouts and kind == "checkpoint" and layout.keys() != layouts[0].keys():
            missing = sorted(layouts[0].keys() - layout.keys())[:5]
            extra = sorted(layout.keys() - layouts[0].keys())[:5]
            raise ValueError(f"Model tensor keys differ from the base: missing={missing}, extra={extra}; {model.root}")
        validate_shard_indexes(model, layout)
        readers.append(opened)
        layouts.append(layout)
        kinds.append(kind)
    checkpoint_indices = [index for index, kind in enumerate(kinds) if kind == "checkpoint"]
    validate_package_assets([models[index] for index in checkpoint_indices])
    for address in layouts[0]:
        slices = [readers[index][layouts[index][address]].get_slice(address[1]) for index in checkpoint_indices]
        shape = slices[0].get_shape()
        dtype = slices[0].get_dtype()
        for value in slices:
            if value.get_shape() != shape:
                raise ValueError(f"Tensor shape mismatch: {address}")
            other = value.get_dtype()
            if dtype in _FLOAT_DTYPES and other in _FLOAT_DTYPES:
                continue
            if dtype not in _BUFFER_DTYPES or dtype != other:
                raise ValueError(f"Unsupported or incompatible tensor dtype for {address}: {dtype}, {other}")
    return readers, layouts, kinds


def _merge_tensor(torch, readers, layouts, address, request, checkpoint_indices, loras):
    # Only one additional tensor is materialized at a time. Output shards are
    # saved separately using the official serializer, with no custom tensor I/O.
    key = address[1]
    base = readers[0][layouts[0][address]].get_tensor(key)
    if not base.is_floating_point():
        for index in checkpoint_indices[1:]:
            if not torch.equal(base, readers[index][layouts[index][address]].get_tensor(key)):
                raise ValueError(f"Nonfloating buffer values differ from the base: {address}")
        return base.clone().contiguous(), False
    dtypes = [readers[index][layouts[index][address]].get_slice(key).get_dtype() for index in checkpoint_indices]
    for _, delta in loras:
        dtypes.extend(reader.get_slice(name).get_dtype() for reader, name in (delta.down, delta.up))
    accumulation_dtype = torch.float64 if ("F64" in dtypes or max(request.weights) > 3.4e38
                                         or any(delta.scale > 3.4e38 for _, delta in loras)) else torch.float32
    if not torch.isfinite(base).all().item():
        raise ValueError(f"Base tensor must contain finite values: {address}")
    result = base.to(dtype=accumulation_dtype).mul(request.base_weight)
    for index in checkpoint_indices[1:]:
        additional = readers[index][layouts[index][address]].get_tensor(key)
        if not torch.isfinite(additional).all().item():
            raise ValueError(f"Additional tensor must contain finite values: {address}")
        weight = request.weights[index - 1]
        coefficient = weight if request.mode == "weighted-sum" else -weight
        result.add_(additional.to(dtype=accumulation_dtype), alpha=coefficient)
    for index, delta in loras:
        value = delta.delta(torch, accumulation_dtype)
        weight = request.weights[index - 1]
        coefficient = weight if request.mode == "weighted-sum" else -weight
        target = result[slice(*delta.target.rows)] if delta.target.rows else result
        target.add_(value, alpha=coefficient)
    result = result.to(dtype=base.dtype).contiguous()
    if not torch.isfinite(result).all().item():
        raise ValueError(f"Merge result is not finite or overflows the base dtype: {address}")
    return result, True


def _execute_merge(request: MergeRequest) -> dict:
    try:
        torch = importlib.import_module("torch")
        safetensors = importlib.import_module("safetensors")
        serializer = importlib.import_module("safetensors.torch")
    except ImportError as error:
        raise RuntimeError("Model merging requires the existing Torch and safetensors environment.") from error
    models = [inspect_merge_model(source, request.cache_dir) for source in request.models]
    with ExitStack() as stack, torch.no_grad():
        readers, layouts, kinds = _open_models(models, stack, safetensors.safe_open)
        request = resolve_merge_weights(request, kinds[1:])
        checkpoint_indices = [index for index, kind in enumerate(kinds) if kind == "checkpoint"]
        loras, adapter_reports = {}, {}
        if "lora" in kinds:
            try:
                aliases = lora_target_aliases(layouts[0], readers[0])
                for index, kind in enumerate(kinds):
                    if kind != "lora":
                        continue
                    deltas = prepare_lora(models[index], readers[index], aliases, readers[0], layouts[0], models[0], torch)
                    adapter_reports[index] = [{"module": delta.module, "component": delta.target.address[0],
                        "tensor": delta.target.address[1], "rows": delta.target.rows,
                        "alpha_rank_scale": delta.scale} for delta in deltas]
                    for delta in deltas:
                        loras.setdefault(delta.target.address, []).append((index, delta))
            except ImportError as error:
                raise RuntimeError("SD/Kohya LoRA name conversion requires the existing Diffusers environment.") from error
        recipe = {
            "schema": "iild-model-merge-v1", "mode": request.mode,
            "base_weight": request.base_weight, "weights": list(request.weights),
            "sources": [{**model.provenance(), "kind": kinds[index],
                         **({"targets": adapter_reports[index]} if index in adapter_reports else {})}
                        for index, model in enumerate(models)],
            "lora_target_count": len(loras),
            "lora_policy": "add-or-subtract-delta-after-checkpoint-blend",
            "tensor_count": len(layouts[0]),
            "arithmetic": "cpu-fp32-or-fp64", "output_dtype": "base",
            "nonfloating_buffers": "require-equal-preserve-base",
            "runtime": {"torch": torch.__version__, "safetensors": safetensors.__version__},
        }
        request.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{request.output.name}-", dir=request.output.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            directory = request.base_model.is_dir()
            for name, asset in models[0].assets.items():
                if name == "merge.json":
                    continue
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset.resolved_file, target)
            output_files = []
            merged_count = 0
            for name, reader in readers[0].items():
                component = str(PurePosixPath(name).parent) if directory else "."
                tensors = {}
                for key in reader.keys():
                    address = (component, key)
                    tensor, merged = _merge_tensor(torch, readers, layouts, address, request,
                                                   checkpoint_indices, loras.get(address, ()))
                    tensors[key] = tensor
                    merged_count += int(merged)
                relative = name if directory else request.output.name
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                metadata = {"format": "pt", "iild_merge": json.dumps(recipe, sort_keys=True),
                            "iild_merge_base_metadata": json.dumps(reader.metadata() or {}, sort_keys=True)}
                # Retain architecture hints used by downstream checkpoint loaders.
                for key in _ARCHITECTURE_HINTS:
                    if key in (reader.metadata() or {}):
                        metadata[key] = reader.metadata()[key]
                serializer.save_file(tensors, str(target), metadata=metadata)
                del tensors, tensor
                output_files.append({"name": relative, "sha256": file_sha256(target),
                                     "size_bytes": target.stat().st_size})
            report = {**recipe, "output": str(request.output), "output_files": output_files,
                      "merged_tensor_count": merged_count,
                      "preserved_buffer_count": len(layouts[0]) - merged_count}
            if directory:
                (staging / "merge.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for model in models:
                model.verify()
            if request.output.exists() or request.output.is_symlink():
                raise FileExistsError(f"Merge output already exists: {request.output}")
            if directory:
                # Completed nonempty directories are published with one rename.
                # A competing completed merge cannot be replaced by os.rename.
                os.rename(staging, request.output)
            else:
                # Atomic no-clobber publication, including concurrent writers.
                os.link(staging / request.output.name, request.output)
            return report


def merge_models(
    base_model: str | Path,
    additional_model: str | Path,
    *,
    additional_models: Sequence[str | Path] = (),
    weights: float | Sequence[float] | None = None,
    mode: str = "weighted-sum",
    output: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Merge checkpoints and/or LoRAs; base and first material are required.

    Weighted sum: (1 - sum(checkpoint weights)) * A + checkpoint sum + LoRA deltas.
    Weighted difference: A - checkpoint sum - LoRA deltas.
    Single files produce safetensors; Diffusers directories retain the base's
    configuration and shard layout. Returns output hashes and source provenance.
    See docs/model-merging.md for defaults, compatibility and memory requirements.
    """
    return _execute_merge(resolve_merge_request(
        base_model, additional_model, additional_models=additional_models, weights=weights,
        mode=mode, output=output, cache_dir=cache_dir))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = resolve_merge_request(args.base_model, args.additional_models[0],
            additional_models=args.additional_models[1:], weights=args.weights, mode=args.mode,
            output=args.output, cache_dir=args.cache_dir)
        result = request.as_dict() if args.print_config else _execute_merge(request)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"Model merge failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
