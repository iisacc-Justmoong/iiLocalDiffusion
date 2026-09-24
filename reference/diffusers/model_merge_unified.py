"""Portable, ordered image-space composition of independent model architectures.

The package is a cascade, never a fictitious cross-architecture weight average.
Strict LoRAs require matching targets; synthetic policy explicitly adapts unmatched deltas.
"""
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from model_merge_files import inspect_merge_model, open_merge_weights
from model_merge_compatibility import LoraCompatibility, LoraCompatibilityBridge
from model_merge_lora import prepare_lora
from model_merge_lora_targets import lora_target_aliases
from model_merge_options import resolve_merge_weights
from weight_files import file_sha256
from iild_package import CONTAINER, write_archive

SCHEMA = "iild-unified-model-v1"


def plan_stages(request, models, readers, layouts, kinds, stack, *, hash_content=False):
    import torch
    import safetensors
    checkpoints = [i for i, kind in enumerate(kinds) if kind == "checkpoint"]
    if len(checkpoints) > 64:
        raise ValueError("Unified models support at most 64 checkpoint stages.")
    if any(models[i].root.is_dir() for i in checkpoints):
        raise ValueError("Unified native cascades require single-file checkpoints; export Diffusers checkpoints before combining them.")
    stages = []
    positions = {index: index for index in checkpoints}
    from downloaded_model import _tensor_identity

    def make_stage(index, strength):
        stage = {"source_index": index, "strength": strength, "loras": []}
        shapes = {key: tuple(readers[index][filename].get_slice(key).get_shape())
                  for (_, key), filename in layouts[index].items()}
        try:
            stage["architecture"] = _tensor_identity(shapes)[0] or "unknown"
        except ValueError as error:
            # Classification is evidence, not permission to preserve a member.
            # A custom hybrid may intentionally use several context widths.
            stage["architecture"] = "unknown"
            stage["architecture_note"] = str(error)
        stage["prediction_markers"] = [key for key in ("v_pred", "ztsnr") if key in shapes]
        return stage

    for index in checkpoints:
        stages.append(make_stage(index, 1.0 if index == 0 else request.weights[index - 1]))
    aliases = {}
    candidates = request.compatibility_models
    # Selected bridge sources are appended to the inventory; original material
    # indexes and coefficients must remain unchanged while planning them.
    for i, kind in enumerate(tuple(kinds)):
        if kind != "lora":
            continue
        matches, failures = [], []
        for checkpoint in checkpoints:
            try:
                if checkpoint not in aliases:
                    aliases[checkpoint] = lora_target_aliases(layouts[checkpoint], readers[checkpoint])
                deltas = prepare_lora(models[i], readers[i], aliases[checkpoint], readers[checkpoint],
                                      layouts[checkpoint], models[checkpoint], torch)
                matches.append(checkpoint)
            except ValueError as error:
                failures.append(str(error))
        if not matches and request.lora_policy == "synthetic":
            target = max((checkpoint for checkpoint in checkpoints if positions[checkpoint] < i), key=positions.get)
            deltas = prepare_lora(models[i], readers[i], aliases[target], readers[target],
                                  layouts[target], models[target], torch, policy="synthetic")
            stage = next(stage for stage in stages if stage["source_index"] == target)
            stage["loras"].append({"source_index": i, "strength": request.weights[i - 1],
                                   "synthetic_adaptations": [delta.adaptation for delta in deltas if delta.adaptation]})
            continue
        if not matches:
            if candidates is None:
                candidates = LoraCompatibilityBridge.discover(
                    [models[index].root for index in checkpoints], exclude=request.models)
            try:
                bridge = LoraCompatibilityBridge.resolve(models[i].root, candidates,
                    refinement_strength=request.compatibility_strength, cache_dir=request.cache_dir)
            except ValueError as error:
                raise ValueError(f"LoRA material {i}: {error} Input mismatch: " + "; ".join(failures[:2])) from error
            if len(stages) >= 64:
                raise ValueError("Unified models support at most 64 checkpoint stages, including compatibility bridges.")
            checkpoint = inspect_merge_model(bridge.checkpoint, request.cache_dir, hash_content=hash_content)
            opened, layout = open_merge_weights(checkpoint, stack, safetensors.safe_open)
            validation = LoraCompatibility.from_open(checkpoint, opened, layout, models[i], readers[i])
            if not validation.compatible:
                raise ValueError(f"Compatibility checkpoint changed during planning: {validation.reason}")
            target = len(models)
            models.append(checkpoint); readers.append(opened); layouts.append(layout); kinds.append("checkpoint")
            checkpoints.append(target)
            positions[target] = i
            stage = make_stage(target, bridge.refinement_strength)
            stage["compatibility_bridge"] = bridge.as_dict()
            stage["loras"].append({"source_index": i, "strength": request.weights[i - 1]})
            stages.append(stage)
            continue
        preceding = [checkpoint for checkpoint in matches if positions[checkpoint] < i]
        if not preceding and len(matches) > 1:
            raise ValueError(f"LoRA material {i} matches several later checkpoints. Place it after the intended checkpoint.")
        target = max(preceding, key=positions.get) if preceding else matches[0]
        stage = next(stage for stage in stages if stage["source_index"] == target)
        stage["loras"].append({"source_index": i, "strength": request.weights[i - 1]})
    return sorted(stages, key=lambda stage: positions[stage["source_index"]])


def _copy(source, target):
    # APFS clones are independent copy-on-write files, not hard links to sources.
    if sys.platform == "darwin":
        clone = ctypes.CDLL(None, use_errno=True).clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(target), 0) == 0:
            return
    shutil.copyfile(source, target)


def execute_unified(request):
    import safetensors
    import torch
    from model_merge import _filter_compatible_materials, _open_models, merge_models, inspect_merge_request
    from iild_package import inspect_archive
    inspection = inspect_merge_request(request)
    all_models = [inspect_merge_model(source, request.cache_dir) for source in request.models]
    request, models, resource_compatibility, excluded_sources = _filter_compatible_materials(
        request, all_models, safetensors.safe_open, torch)
    with ExitStack() as stack:
        readers, layouts, kinds, _ = _open_models(models, stack, safetensors.safe_open, compatible=False)
        request = resolve_merge_weights(request, kinds[1:])
        planned = plan_stages(request, models, readers, layouts, kinds, stack, hash_content=True)
        request.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{request.output.name}-", dir=request.output.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            stages, files = [], []
            for number, stage in enumerate(planned):
                index = stage["source_index"]
                relative = f"members/{number:03d}/model.safetensors"
                target = staging / relative
                target.parent.mkdir(parents=True)
                if stage["loras"]:
                    adapters = stage["loras"]
                    merged = merge_models(models[index].root, models[adapters[0]["source_index"]].root,
                                          additional_models=[models[a["source_index"]].root for a in adapters[1:]],
                                          weights=[a["strength"] for a in adapters], output=target, cache_dir=request.cache_dir,
                                          lora_policy=request.lora_policy, checkpoint_policy=request.checkpoint_policy)
                    digest = merged["output_files"][0]["sha256"]
                    stage = {**stage, "output_verification": merged["output_verification"]}
                else:
                    source = next(iter(models[index].weights.values()))
                    _copy(source.resolved_file, target)
                    # copy/clone completes before return, and every source is
                    # revalidated below. Reuse its digest instead of rereading
                    # an identical multi-gigabyte clone a second time.
                    digest = source.sha256
                size = target.stat().st_size
                stages.append({**stage, "model": relative, "sha256": digest, "size_bytes": size})
                files.append({"name": relative, "sha256": digest, "size_bytes": size})
            manifest = {"_class_name": "IILDUnifiedCascade", "schema": SCHEMA, "container": CONTAINER,
                        "composition": "ordered-image-refinement", "stages": stages}
            manifest_path = staging / "model_index.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            files.append({"name": "model_index.json", "sha256": file_sha256(manifest_path),
                          "size_bytes": manifest_path.stat().st_size})
            report = {"schema": "iild-model-merge-v1", "mode": "unified", "output": str(request.output),
                      "composition": manifest["composition"], "stages": stages, "output_files": files,
                      "base_profile": inspection["base_profile"], "preflight": inspection["preflight"],
                      "sources": [{**model.provenance(), "kind": kind} for model, kind in zip(models, kinds)],
                      "resource_compatibility": resource_compatibility,
                      "excluded_sources": excluded_sources,
                      "included_material_count": len(models) - 1,
                      "excluded_material_count": len(excluded_sources),
                      "weights": list(request.weights), "base_weight": None, "merged_tensor_count": 0,
                      "tensor_validation": {"copied_members": "byte-preserved; numeric finiteness is not certified",
                                            "fused_members": "reopened tensors match computed finite output; common-layer repairs invalid inputs"},
                      "lora_routing": "nearest-preceding-compatible-checkpoint; synthetic-fallback-if-enabled",
                      "lora_policy": request.lora_policy,
                      "compatibility_bridge_count": sum("compatibility_bridge" in stage for stage in stages),
                      "compatibility_policy": request.as_dict()["compatibility_policy"],
                      "note": "Independent models, sequential image refinement; not a single-network weight merge."}
            (staging / "merge.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for model in all_models:
                model.verify()
            if request.output.exists() or request.output.is_symlink():
                raise FileExistsError(f"Merge output already exists: {request.output}")
            archive = Path(temporary) / request.output.name
            write_archive(staging, archive)
            _, verified_manifest, _ = inspect_archive(archive, hashes=True)
            if verified_manifest != manifest:
                raise ValueError("Saved unified manifest differs from planned stages.")
            report["output_verification"] = {"status": "passed", "method": "reopen-all-package-member-hashes",
                                              "stage_count": len(stages), "quality_guaranteed": False,
                                              "copied_values": "byte-preserved; not numerically repaired"}
            if request.output.exists() or request.output.is_symlink():
                raise FileExistsError(f"Merge output already exists: {request.output}")
            os.link(archive, request.output)
            report["package"] = {"container": CONTAINER, "sha256": file_sha256(request.output),
                                 "size_bytes": request.output.stat().st_size}
            return report
