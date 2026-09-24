"""Local model merging with existing PyTorch arithmetic and safetensors I/O."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from model_merge_files import align_anima_checkpoint, inspect_merge_model, open_merge_weights, validate_package_assets
from model_merge_common import project_checkpoint, preserves_base_tensor, POLICY
from model_merge_tensor import (QuantizationError, decode_tensor, repair_nonfinite_material,
                                record_numeric_repair, zero_nonfinite, safe_storage)
from model_merge_ecosystem import exact_layout_compatible, identify_ecosystem
from model_merge_diagnostics import model_profile, preflight_summary
from model_merge_verify import verify_saved_tensors
from model_merge_lora import is_lora_key, prepare_lora
from model_merge_lora_targets import lora_target_aliases
from model_merge_options import MergeRequest, build_parser, resolve_merge_request, resolve_merge_weights
from weight_files import file_sha256

_FLOAT_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"})
_BUFFER_DTYPES = frozenset({"BOOL", "U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"})
_ARCHITECTURE_HINTS = ("modelspec.architecture", "modelspec.implementation", "modelspec.prediction_type",
                       "iild.dit.standard", "iild.model_family", "iild.dit.template_sha256")


def _isfinite(torch, tensor):
    """Check low-precision storage tensors through a supported arithmetic dtype."""
    candidate = tensor.to(dtype=torch.float32) if "float8" in str(tensor.dtype) else tensor
    return torch.isfinite(candidate).all().item()


def _equal(torch, left, right):
    if "float8" in str(left.dtype) or "float8" in str(right.dtype):
        return torch.equal(left.float(), right.float())
    return torch.equal(left, right)


def _open_models(models, stack, safe_open, *, compatible=True, checkpoint_policy="common-layer"):
    readers = []
    layouts = []
    kinds = []
    known_hints = {}
    prediction_markers = None
    common_layers = {}
    for model in models:
        opened, layout = open_merge_weights(model, stack, safe_open)
        kind = "lora" if any(is_lora_key(key) for _, key in layout) else "checkpoint"
        if not layouts and kind != "checkpoint":
            raise ValueError("Base model must be a full checkpoint, not a LoRA adapter.")
        if kind == "checkpoint":
            markers = frozenset(key for component, key in layout if component == "." and key in ("v_pred", "ztsnr"))
            for key in markers if compatible and checkpoint_policy == "strict" else ():
                shape = opened[layout[(".", key)]].get_slice(key).get_shape()
                if shape != [0]:
                    raise ValueError(f"Prediction marker {key} must be an empty tensor: {model.root}")
            if compatible and checkpoint_policy == "strict" and prediction_markers is not None and markers != prediction_markers:
                raise ValueError("Prediction settings differ (v_pred / ztsnr). These are sampling markers, not merge weights. "
                                 "Use --mode unified with an .iildmodel output to retain each model's prediction settings.")
            prediction_markers = markers
            if compatible and checkpoint_policy == "strict" and model.root.is_dir() != models[0].root.is_dir():
                raise ValueError("All checkpoints must use the same format: single files or Diffusers directories.")
            if model.root.is_dir() and "model_index.json" not in model.assets:
                raise ValueError("A full checkpoint directory requires model_index.json.")
            for name, reader in opened.items():
                component = str(PurePosixPath(name).parent) if model.root.is_dir() else "."
                for hint in _ARCHITECTURE_HINTS:
                    value = (reader.metadata() or {}).get(hint)
                    if value:
                        identity = (component, hint)
                        if compatible and checkpoint_policy == "strict" and identity in known_hints and value != known_hints[identity]:
                            raise ValueError(f"Model architecture metadata differs for {hint}: {model.root}")
                        known_hints[identity] = value
        elif "model_index.json" in model.assets:
            raise ValueError("LoRA material must be an adapter export, not a pipeline with embedded adapter keys.")
        if compatible and layouts and kind == "checkpoint":
            if checkpoint_policy == "common-layer":
                opened, layout, common_layers[len(layouts)] = project_checkpoint(
                    opened, layout, readers[0], layouts[0])
            elif not model.root.is_dir():
                opened, layout = align_anima_checkpoint(opened, layout, layouts[0])
        if compatible and checkpoint_policy == "strict" and layouts and kind == "checkpoint" and layout.keys() != layouts[0].keys():
            missing = sorted(layouts[0].keys() - layout.keys())[:5]
            extra = sorted(layout.keys() - layouts[0].keys())[:5]
            raise ValueError(f"Model tensor keys differ from the base: missing={missing}, extra={extra}; {model.root}. "
                             "Use --mode unified for different architectures.")
        readers.append(opened)
        layouts.append(layout)
        kinds.append(kind)
    checkpoint_indices = [index for index, kind in enumerate(kinds) if kind == "checkpoint"]
    if not compatible:
        return readers, layouts, kinds, common_layers
    if checkpoint_policy == "common-layer":
        return readers, layouts, kinds, common_layers
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
    return readers, layouts, kinds, common_layers


def _inspect_requested_models(request, torch, *, hash_content=True):
    models, failures = [], {}
    for index, path in enumerate(request.models):
        try:
            models.append(inspect_merge_model(path, request.cache_dir, hash_content=hash_content))
        except (ValueError, OSError, RuntimeError) as error:
            if index == 0 or not request.automatic_repair or isinstance(error, torch.OutOfMemoryError):
                raise
            models.append(None)
            failures[index] = str(error)
    return models, failures


def _filter_compatible_materials(request, models, safe_open, torch, failures=None):
    """Keep only materials that can be applied to the selected base model.

    Ecosystem metadata/tensor evidence is the primary checkpoint boundary. An
    otherwise unclassified checkpoint is accepted only when its entire tensor
    layout is an exact arithmetic match. LoRAs must resolve real targets in the
    base; synthetic projection is deliberately not a compatibility signal.
    """
    failures = dict(failures or {})
    with ExitStack() as stack:
        opened = []
        for index, model in enumerate(models):
            try:
                opened.append(open_merge_weights(model, stack, safe_open) if model is not None else ({}, {}))
            except (ValueError, OSError, RuntimeError) as error:
                if index == 0 or not request.automatic_repair or isinstance(error, torch.OutOfMemoryError):
                    raise
                failures[index] = str(error)
                opened.append(({}, {}))
        readers = [value[0] for value in opened]
        layouts = [value[1] for value in opened]
        kinds = ["lora" if any(is_lora_key(key) for _, key in layout) else "checkpoint"
                 for layout in layouts]
        if kinds[0] != "checkpoint":
            raise ValueError("Base model must be a full checkpoint, not a LoRA adapter.")
        base_identity = identify_ecosystem(readers[0], layouts[0])
        aliases = lora_target_aliases(layouts[0], readers[0])
        included = []
        compatibility = []
        for index in range(1, len(models)):
            identity = identify_ecosystem(readers[index], layouts[index])
            shared = sorted(set(base_identity.families) & set(identity.families))
            compatible = False
            reason = ""
            targets = []
            if index in failures:
                reason = f"Unusable material omitted: {failures[index]}"
            elif kinds[index] == "checkpoint" and models[index].root.is_dir() and "model_index.json" not in models[index].assets:
                reason = "Unusable material omitted: a checkpoint directory requires model_index.json."
            elif kinds[index] == "lora" and "model_index.json" in models[index].assets:
                reason = "Unusable material omitted: a pipeline with embedded adapter keys is not a LoRA export."
            elif kinds[index] == "lora":
                try:
                    deltas = prepare_lora(models[index], readers[index], aliases, readers[0], layouts[0],
                                          models[0], torch, policy="strict")
                    compatible = bool(deltas)
                    targets = [{"module": delta.module, "component": delta.target.address[0],
                                "tensor": delta.target.address[1], "rows": delta.target.rows,
                                "scale": delta.scale} for delta in deltas]
                    reason = (f"{len(deltas)} LoRA target(s) match the base model."
                              if compatible else "The LoRA has no applicable target in the base model.")
                except (ValueError, RuntimeError, OSError, ImportError) as error:
                    if isinstance(error, torch.OutOfMemoryError):
                        raise
                    reason = f"LoRA targets do not match the base model: {error}"
            elif shared:
                compatible = True
                reason = f"Shared model ecosystem: {', '.join(shared)}."
            elif not (base_identity.families and identity.families) and exact_layout_compatible(
                    readers[0], layouts[0], readers[index], layouts[index]):
                compatible = True
                reason = "The complete tensor layout and dtypes match the base model."
            elif base_identity.families and identity.families:
                reason = (f"Model ecosystem differs from the base ({identity.label} versus "
                          f"{base_identity.label}).")
            elif not base_identity.families:
                reason = "The base model ecosystem is unknown and tensor keys, shapes, or dtypes differ."
            else:
                reason = "The material ecosystem is unknown and tensor keys, shapes, or dtypes differ from the base."
            entry = {
                "source_index": index,
                "path": str(request.models[index]),
                "kind": kinds[index],
                "base_ecosystems": list(base_identity.families),
                "material_ecosystems": list(identity.families),
                "compatible": compatible,
                "reason": reason,
                "status": ("incompatible" if not compatible else "compatible" if kinds[index] == "lora" else
                           "compatible" if request.mode != "unified" and exact_layout_compatible(
                               readers[0], layouts[0], readers[index], layouts[index]) else "conditional"),
                "profile": model_profile(readers[index], layouts[index]),
                "lora_targets": targets,
                "lora_target_count": len(targets),
                "numeric_values_checked": False,
            }
            compatibility.append(entry)
            if compatible:
                included.append(index)
        if not included and not request.automatic_repair:
            reasons = "; ".join(entry["reason"] for entry in compatibility[:3])
            raise ValueError(f"No merge material is compatible with the selected base model. {reasons}")
        filtered_weights = (tuple(request.weights[index - 1] for index in included)
                            if request.weights is not None else None)
        filtered_request = replace(
            request,
            additional_models=tuple(request.additional_models[index - 1] for index in included),
            weights=filtered_weights,
            base_weight=None,
        )
        filtered_models = [models[0], *(models[index] for index in included)]
        excluded = [entry for entry in compatibility if not entry["compatible"]]
        return filtered_request, filtered_models, compatibility, excluded


def _merge_tensor(torch, readers, layouts, address, request, checkpoint_indices, loras, normalization=None):
    # Only one additional tensor is materialized at a time. Output shards are
    # saved separately using the official serializer, with no custom tensor I/O.
    key = address[1]
    base = readers[0][layouts[0][address]].get_tensor(key)
    original_base = base
    common = request.automatic_repair
    if common and base.is_floating_point():
        base = zero_nonfinite(torch, base, normalization, address, "base_nonfinite_values")
    base_changed = base is not original_base
    base_dtype = readers[0][layouts[0][address]].get_slice(key).get_dtype()
    if common and preserves_base_tensor(address, base_dtype, base.shape, layouts[0]):
        return base.clone().contiguous(), False, base_changed
    if not common and not base.is_floating_point():
        for index in checkpoint_indices[1:]:
            if not torch.equal(base, readers[index][layouts[index][address]].get_tensor(key)):
                raise ValueError(f"Nonfloating buffer values differ from the base: {address}")
        return base.clone().contiguous(), False, base_changed
    scale = zero = None
    try:
        numeric, scale, zero = decode_tensor(torch, readers[0], layouts[0], address, base) if common else (base, None, None)
    except QuantizationError as error:
        if normalization is not None:
            normalization["invalid_base_quantization_tensors"] += 1
            if len(normalization["examples"]) < 64:
                normalization["examples"].append({"tensor": list(address), "action": "preserve-base", "reason": str(error)})
        return base.clone().contiguous(), False, base_changed
    dtypes = [readers[index][layouts[index][address]].get_slice(key).get_dtype() for index in checkpoint_indices]
    for _, delta in loras:
        dtypes.extend(reader.get_slice(name).get_dtype() for reader, name in (delta.down, delta.up))
    accumulation_dtype = torch.float64 if ("F64" in dtypes or max(request.weights, default=0) > 1
                                         or any(delta.scale > 3.4e38 for _, delta in loras)) else torch.float32
    if not _isfinite(torch, numeric):
        if not common:
            raise ValueError(f"Base tensor must contain finite values: {address}")
        numeric = zero_nonfinite(torch, numeric, normalization, address, "base_decoded_nonfinite_values")
    result = numeric.to(dtype=accumulation_dtype).mul(request.base_weight)
    applied = False
    for index in checkpoint_indices[1:]:
        reader = readers[index][layouts[index][address]]
        weight = request.weights[index - 1]
        if weight == 0:
            continue
        available = not hasattr(reader, "contributes") or reader.contributes(key)
        try:
            additional = reader.get_tensor(key) if available else None
        except (ValueError, OSError, RuntimeError) as error:
            if not common or isinstance(error, torch.OutOfMemoryError):
                raise
            additional, available = None, False
            record_numeric_repair(normalization, "unreadable_material_tensors", address,
                                  source_index=index, reason=str(error))
        available = available and (not hasattr(reader, "contributes") or reader.contributes(key))
        if not available:
            # Sum returns the missing share to A. Difference has a neutral zero
            # contribution, not a second copy of A to subtract.
            if request.mode == "weighted-sum":
                result.add_(numeric.to(dtype=result.dtype), alpha=weight)
            continue
        valid_contribution = True
        if not _isfinite(torch, additional):
            if not common:
                raise ValueError(f"Additional tensor must contain finite values: {address}")
            replacement = numeric if request.mode == "weighted-sum" else additional.new_zeros(())
            additional, repair = repair_nonfinite_material(torch, additional, replacement)
            valid_contribution = repair["nonfinite_values"] < additional.numel()
            if normalization is not None:
                normalization["nonfinite_material_values"] += repair["nonfinite_values"]
                normalization["nonfinite_material_tensors"] += 1
                if len(normalization["nonfinite_material_examples"]) < 64:
                    source_address = reader.mappings.get(address) if hasattr(reader, "mappings") else address
                    normalization["nonfinite_material_examples"].append({
                        "source_index": index, "tensor": list(address),
                        "source_tensor": list(source_address) if source_address else None,
                        "replacement": "base" if request.mode == "weighted-sum" else "zero",
                        **repair})
        if additional.dtype == torch.float64 and result.dtype != torch.float64:
            result = result.double()
        coefficient = weight if request.mode == "weighted-sum" else -weight
        result.add_(additional.to(dtype=result.dtype), alpha=coefficient)
        applied = applied or valid_contribution
    for index, delta in loras:
        weight = request.weights[index - 1]
        try:
            value = (delta.delta(torch, result.dtype, repair=True, normalization=normalization, source_index=index)
                     if common else delta.delta(torch, result.dtype))
        except (ValueError, OSError, RuntimeError) as error:
            if not common or isinstance(error, torch.OutOfMemoryError):
                raise
            record_numeric_repair(normalization, "skipped_lora_deltas", address, source_index=index, reason=str(error))
            continue
        if weight == 0:
            continue
        if common and not torch.count_nonzero(value).item():
            # An empty/fully repaired LoRA delta must not label a repaired base
            # as an effective material blend.
            continue
        coefficient = weight if request.mode == "weighted-sum" else -weight
        target = result[slice(*delta.target.rows)] if delta.target.rows else result
        target.add_(value, alpha=coefficient)
        applied = True
    if not applied:
        return base.clone().contiguous(), False, base_changed
    if not common and not torch.isfinite(result).all().item():
        raise ValueError(f"Merge result is not finite or overflows the base dtype: {address}")
    if scale is not None:
        result = result / scale + zero
        if normalization is not None:
            normalization["requantized_tensors"] += 1
    if common:
        result = safe_storage(torch, result, base, normalization, address)
        return result, True, not _equal(torch, result, original_base)
    if not base.is_floating_point():
        limits = torch.iinfo(base.dtype)
        result = result.round().clamp(limits.min, limits.max)
    result = result.to(dtype=base.dtype).contiguous()
    if not _isfinite(torch, result):
        raise ValueError(f"Merge result is not finite or overflows the base dtype: {address}")
    return result, True, not _equal(torch, result, base)


def _refresh_legacy_package_manifest(staging, output_files):
    """Keep legacy directory-shaped .iildmodel stage identities executable."""
    manifest_path = staging / "model_index.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("A legacy .iildmodel package requires a valid model_index.json.") from error
    stages = manifest.get("stages") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get("schema") != "iild-unified-model-v1"
            or not isinstance(stages, list) or not stages):
        raise ValueError("A legacy .iildmodel package requires a nonempty unified stage manifest.")
    identities = {entry["name"]: entry for entry in output_files}
    for stage in stages:
        name = stage.get("model") if isinstance(stage, dict) else None
        identity = identities.get(name)
        if identity is None:
            raise ValueError(f"A legacy .iildmodel stage is missing from the weighted output: {name}")
        stage["sha256"] = identity["sha256"]
        stage["size_bytes"] = identity["size_bytes"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _execute_merge(request: MergeRequest) -> dict:
    try:
        torch = importlib.import_module("torch")
        safetensors = importlib.import_module("safetensors")
        serializer = importlib.import_module("safetensors.torch")
    except ImportError as error:
        raise RuntimeError("Model merging requires the existing Torch and safetensors environment.") from error
    inspection = inspect_merge_request(request)
    if request.mode == "unified":
        from model_merge_unified import execute_unified
        return execute_unified(request)
    all_models, failures = _inspect_requested_models(request, torch)
    request, models, resource_compatibility, excluded_sources = _filter_compatible_materials(
        request, all_models, safetensors.safe_open, torch, failures)
    with ExitStack() as stack, torch.no_grad():
        readers, layouts, kinds, common_layers = _open_models(
            models, stack, safetensors.safe_open, checkpoint_policy=request.checkpoint_policy)
        request = resolve_merge_weights(request, kinds[1:])
        checkpoint_indices = [index for index, kind in enumerate(kinds) if kind == "checkpoint"]
        loras, adapter_reports = {}, {}
        if "lora" in kinds:
            try:
                aliases = lora_target_aliases(layouts[0], readers[0])
                for index, kind in enumerate(kinds):
                    if kind != "lora":
                        continue
                    deltas = prepare_lora(models[index], readers[index], aliases, readers[0], layouts[0], models[0], torch, policy=request.lora_policy)
                    adapter_reports[index] = [{"module": delta.module, "component": delta.target.address[0],
                        "tensor": delta.target.address[1], "rows": delta.target.rows,
                        "alpha_rank_scale": delta.scale, "adaptation": delta.adaptation} for delta in deltas]
                    for delta in deltas:
                        loras.setdefault(delta.target.address, []).append((index, delta))
            except ImportError as error:
                raise RuntimeError("SD/Kohya LoRA name conversion requires the existing Diffusers environment.") from error
        recipe = {
            "schema": "iild-model-merge-v1", "mode": request.mode,
            "base_weight": request.base_weight, "weights": list(request.weights),
            "weight_normalization": request.weight_normalization,
            "repair_policy": "best-effort-v1" if request.automatic_repair else "strict",
            "base_profile": inspection["base_profile"], "preflight": inspection["preflight"],
            "sources": [{**model.provenance(), "kind": kinds[index],
                         **({"targets": adapter_reports[index]} if index in adapter_reports else {})}
                        for index, model in enumerate(models)],
            "lora_target_count": len(loras),
            "lora_policy": request.lora_policy,
            "checkpoint_policy": request.checkpoint_policy,
            "common_layers": {str(index): report for index, report in common_layers.items()},
            "resource_compatibility": resource_compatibility,
            "excluded_sources": excluded_sources,
            "included_material_count": len(models) - 1,
            "excluded_material_count": len(excluded_sources),
            "lora_alias_policy": "identical-projections-once; distinct-projections-additive",
            "tensor_count": len(layouts[0]),
            "checkpoint_key_policy": (POLICY
                                      if request.checkpoint_policy == "common-layer" else
                                      "exact-or-equivalent-anima-namespace; preserve-base-names"),
            "arithmetic": "cpu-fp32-or-fp64", "output_dtype": "base",
            "nonfloating_buffers": "preserve-base; scaled-int8-weights-requantized" if request.checkpoint_policy == "common-layer" else "require-equal-preserve-base",
            "nonfinite_material_policy": "base-for-sum;zero-for-difference" if request.checkpoint_policy == "common-layer" else "reject",
            "numeric_normalization": {"requantized_tensors": 0, "invalid_base_quantization_tensors": 0, "examples": [],
                                      "nonfinite_material_values": 0, "nonfinite_material_tensors": 0,
                                      "nonfinite_material_examples": []},
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
            changed_count = 0
            verification = {"status": "passed", "method": "reopen-all-tensors-exact-expected-values",
                            "tensor_count": 0, "changed_tensor_count": 0, "finite_tensor_count": 0,
                            "quality_guaranteed": False}
            for name, reader in readers[0].items():
                component = str(PurePosixPath(name).parent) if directory else "."
                tensors = {}
                for key in reader.keys():
                    address = (component, key)
                    tensor, merged, changed = _merge_tensor(torch, readers, layouts, address, request,
                                                            checkpoint_indices, loras.get(address, ()), recipe["numeric_normalization"])
                    tensors[key] = tensor
                    merged_count += int(merged)
                    changed_count += int(changed)
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
                verified = verify_saved_tensors(target, tensors, reader, safetensors.safe_open, torch)
                for field, count in verified.items():
                    verification[field] += count
                del tensors, tensor
                output_files.append({"name": relative, "sha256": file_sha256(target),
                                     "size_bytes": target.stat().st_size})
            if directory and request.base_model.suffix.lower() == ".iildmodel":
                _refresh_legacy_package_manifest(staging, output_files)
            report = {**recipe, "output": str(request.output), "output_files": output_files,
                      "merged_tensor_count": merged_count,
                      "changed_tensor_count": changed_count,
                      "output_verification": verification,
                      "preserved_buffer_count": len(layouts[0]) - merged_count,
                      "result_kind": "base-fallback" if not changed_count else "merged" if merged_count else "repaired-base",
                      "no_effect": changed_count == 0}
            if directory:
                (staging / "merge.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if verification["changed_tensor_count"] != changed_count:
                raise ValueError("Saved output change count differs from the computed merge.")
            for model in all_models:
                if model is not None:
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
    compatibility_models: Sequence[str | Path] | None = None,
    compatibility_strength: float = 0.35,
    lora_policy: str = "strict",
    checkpoint_policy: str = "common-layer",
) -> dict:
    """Merge checkpoints and/or LoRAs; base and first material are required.

    Weighted sum: (1 - sum(checkpoint weights)) * A + checkpoint sum + LoRA deltas.
    Weighted difference: A - checkpoint sum - LoRA deltas.
    Weight modes preserve the base format and tensor layout. Unified mode builds
    an ordered image-refinement package with checkpoint strengths in [0,1].
    Returns output hashes and source provenance.
    See docs/model-merging.md for defaults, compatibility and memory requirements.
    """
    return _execute_merge(resolve_merge_request(
        base_model, additional_model, additional_models=additional_models, weights=weights,
        mode=mode, output=output, cache_dir=cache_dir,
        compatibility_models=compatibility_models, compatibility_strength=compatibility_strength,
        lora_policy=lora_policy, checkpoint_policy=checkpoint_policy))


def inspect_merge_request(request: MergeRequest) -> dict:
    """Fail on structural conflicts before hashing gigabytes of checkpoint data."""
    import safetensors
    import torch
    all_models, failures = _inspect_requested_models(request, torch, hash_content=False)
    request, models, resource_compatibility, excluded_sources = _filter_compatible_materials(
        request, all_models, safetensors.safe_open, torch, failures)
    with ExitStack() as stack:
        readers, layouts, kinds, common_layers = _open_models(
            models, stack, safetensors.safe_open, compatible=request.mode != "unified",
            checkpoint_policy=request.checkpoint_policy)
        request = resolve_merge_weights(request, kinds[1:])
        adaptations = {}
        if request.mode == "unified":
            from model_merge_unified import plan_stages
            stages = plan_stages(request, models, readers, layouts, kinds, stack)
        else:
            stages = []
            if "lora" in kinds:
                import torch
                aliases = lora_target_aliases(layouts[0], readers[0])
                for index, kind in enumerate(kinds):
                    if kind == "lora":
                        deltas = prepare_lora(models[index], readers[index], aliases, readers[0], layouts[0], models[0], torch, policy=request.lora_policy)
                        adaptations[index] = [delta.adaptation for delta in deltas if delta.adaptation]
        return {**request.as_dict(), "synthetic_adaptations": adaptations,
                "base_profile": model_profile(readers[0], layouts[0]),
                "preflight": preflight_summary(request, resource_compatibility, common_layers, stages),
                "common_layers": {str(index): report for index, report in common_layers.items()},
                "resource_compatibility": resource_compatibility,
                "excluded_sources": excluded_sources,
                "included_material_count": len(models) - 1,
                "excluded_material_count": len(excluded_sources),
                "inspection": "structure-only", "kinds": kinds, "stages": stages,
                "lora_alias_policy": "identical-projections-once; distinct-projections-additive",
                "resolved_sources": [str(model.root) for model in models],
                "data_validation": ("source hashes and finite arithmetic are checked when building; copied members preserve source bytes"
                                    if request.mode == "unified" else
                                    "best-effort arithmetic repairs nonfinite base/material/LoRA values and saturates output; unusable materials are omitted; source integrity remains required"
                                    if request.checkpoint_policy == "common-layer" else
                                    "finite values and source hashes are checked when building the output")}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = resolve_merge_request(args.base_model, args.additional_models[0],
            additional_models=args.additional_models[1:], weights=args.weights, mode=args.mode,
            output=args.output, cache_dir=args.cache_dir,
            compatibility_models=args.compatibility_models, compatibility_strength=args.compatibility_strength,
            lora_policy=args.lora_policy, checkpoint_policy=args.checkpoint_policy)
        result = request.as_dict() if args.print_config else inspect_merge_request(request) if args.inspect else _execute_merge(request)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"Model merge failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
