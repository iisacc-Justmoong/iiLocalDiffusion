#!/usr/bin/env python3
"""Load a pinned Diffusers pipeline and save normalized component metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from pipeline_loading import load_pipeline
from presets import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    SD15_PRESET,
    PipelinePreset,
    resolve_model_selection,
    validate_pipeline_contract,
)


MODEL_ID = SD15_PRESET.model_id
MODEL_REVISION = SD15_PRESET.revision
CORE_COMPONENTS = tuple(name for name, _ in SD15_PRESET.expected_components)
EXPECTED_CLASSES = dict(SD15_PRESET.expected_components)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIRECTORY = REPOSITORY_ROOT / "build" / "reference" / "huggingface"
DEFAULT_XET_CACHE_DIRECTORY = REPOSITORY_ROOT / "build" / "reference" / "huggingface-xet"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build" / "reference" / SD15_PRESET.inspection_filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a pinned Diffusers pipeline and persist normalized metadata."
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET_NAME)
    parser.add_argument("--model", default=None, help="Hub model ID or local Diffusers directory")
    parser.add_argument("--revision", default=None, help="Immutable Hub commit revision")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--xet-cache-dir", type=Path, default=DEFAULT_XET_CACHE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_arguments(args: argparse.Namespace) -> tuple[PipelinePreset, argparse.Namespace]:
    preset = PRESETS[args.preset]
    try:
        selection = resolve_model_selection(preset, args.model, args.revision)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    resolved = argparse.Namespace(**vars(args))
    resolved.model_selection = selection
    resolved.output = args.output or (
        REPOSITORY_ROOT / "build" / "reference" / preset.inspection_filename
    )
    return preset, resolved


def load_dependencies() -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        from diffusers import FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
    except ImportError as error:
        raise SystemExit(
            "Reference dependencies are missing. Install reference/diffusers/requirements.txt "
            "into reference/diffusers/.venv first."
        ) from error
    return torch, {
        "FluxPipeline": FluxPipeline,
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    }


def normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): normalize(member)
            for key, member in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(member) for member in value]
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    return str(value)


def component_config(component: Any) -> dict[str, Any]:
    if hasattr(component, "config"):
        config = component.config
        if hasattr(config, "to_dict"):
            return normalize(config.to_dict())
        if isinstance(config, Mapping):
            return normalize(dict(config))
        if hasattr(config, "items"):
            return normalize(dict(config.items()))
        return {"value": normalize(config)}
    if hasattr(component, "init_kwargs"):
        return normalize(dict(component.init_kwargs))
    return {}


def package_versions() -> dict[str, str]:
    names = (
        "accelerate",
        "diffusers",
        "huggingface-hub",
        "hf-xet",
        "numpy",
        "Pillow",
        "safetensors",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    preset, args = resolve_arguments(build_parser().parse_args())
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing inspection: {args.output}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.xet_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(args.xet_cache_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch, pipeline_classes = load_dependencies()

    dtype = getattr(torch, preset.runtime.accelerator_dtype)
    load_arguments: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "dtype": dtype,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    if args.model_selection.requested_revision is not None:
        load_arguments["revision"] = args.model_selection.requested_revision
    if preset.runtime.weight_variant is not None:
        load_arguments["variant"] = preset.runtime.weight_variant
    if preset.family == "sdxl-base":
        load_arguments["add_watermarker"] = False

    pipeline_class = pipeline_classes[preset.pipeline_class]
    pipeline = load_pipeline(
        pipeline_class,
        args.model_selection.source,
        load_arguments,
    )
    validate_pipeline_contract(pipeline, preset)

    components: dict[str, dict[str, Any]] = {}
    for name, component in sorted(pipeline.components.items()):
        components[name] = {
            "class": None if component is None else type(component).__name__,
            "config": {} if component is None else component_config(component),
            "module": None if component is None else type(component).__module__,
            "present": component is not None,
        }

    inspection: dict[str, Any] = {
        "components": components,
        "model": {
            "id_or_path": args.model_selection.source,
            "preset": preset.name,
            "requested_revision": args.model_selection.requested_revision,
        },
        "pipeline": {
            "class": type(pipeline).__name__,
            "config": normalize(dict(pipeline.config.items())),
            "module": type(pipeline).__module__,
        },
        "runtime": {
            "load_dtype": str(dtype),
            "packages": package_versions(),
            "platform": platform.platform(),
            "python": sys.version,
            "xet_cache": os.environ.get("HF_XET_CACHE"),
        },
    }
    write_json_atomically(args.output, inspection)

    print(f"Pipeline: {type(pipeline).__name__}")
    for name, _ in preset.expected_components:
        print(f"{name}: {components[name]['class']}")
    print(f"Inspection: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
