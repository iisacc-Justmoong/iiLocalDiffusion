#!/usr/bin/env python3
"""Generate a pinned iiLocalDiffusion Diffusers reference image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

from pipeline_loading import _is_gated_repository_error, load_pipeline
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
DEFAULT_PROMPT = "a red cube on a white table"
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SEED = 42
DEFAULT_WIDTH = SD15_PRESET.width
DEFAULT_HEIGHT = SD15_PRESET.height
DEFAULT_STEPS = SD15_PRESET.steps
DEFAULT_GUIDANCE_SCALE = SD15_PRESET.guidance_scale
LORA_ADAPTER_NAME = "iild_lora"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIRECTORY = REPOSITORY_ROOT / "build" / "reference" / "huggingface"
DEFAULT_XET_CACHE_DIRECTORY = REPOSITORY_ROOT / "build" / "reference" / "huggingface-xet"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build" / "reference" / SD15_PRESET.generation_filename


@dataclass(frozen=True)
class LoraSelection:
    source: str
    weight_name: str
    requested_revision: str | None
    is_local: bool
    scale: float
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class LoraActivation:
    active_adapters: tuple[str, ...]
    registered_components: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a pinned Diffusers reference fixture."
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET_NAME)
    parser.add_argument("--model", default=None, help="Hub model ID or local Diffusers directory")
    parser.add_argument("--revision", default=None, help="Immutable Hub commit revision")
    parser.add_argument(
        "--lora",
        default=None,
        help="Local LoRA safetensors file/directory or Hugging Face repository ID",
    )
    parser.add_argument(
        "--lora-revision",
        default=None,
        help="Immutable 40-character commit revision for a remote LoRA",
    )
    parser.add_argument(
        "--lora-weight-name",
        default=None,
        help="Exact .safetensors filename for a LoRA directory or repository",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA adapter scale; defaults to 1.0",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--xet-cache-dir", type=Path, default=DEFAULT_XET_CACHE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--attention-slicing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the preset and device-specific attention-slicing policy.",
    )
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
    resolved.lora_selection = resolve_lora_selection(resolved)
    resolved.width = preset.width if args.width is None else args.width
    resolved.height = preset.height if args.height is None else args.height
    resolved.steps = preset.steps if args.steps is None else args.steps
    resolved.guidance_scale = (
        preset.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    generation_filename = preset.generation_filename
    if resolved.lora_selection is not None:
        filename = Path(generation_filename)
        generation_filename = f"{filename.stem}-lora{filename.suffix}"
    resolved.output = args.output or REPOSITORY_ROOT / "build" / "reference" / generation_filename
    return preset, resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_lora_weight_name(weight_name: str) -> None:
    if (
        not weight_name
        or "/" in weight_name
        or "\\" in weight_name
        or Path(weight_name).suffix != ".safetensors"
    ):
        raise SystemExit(
            "--lora-weight-name must be one .safetensors filename without directories."
        )


def resolve_lora_selection(args: argparse.Namespace) -> LoraSelection | None:
    if args.lora is None:
        if args.lora_revision is not None or args.lora_weight_name is not None:
            raise SystemExit(
                "Using --lora-revision or --lora-weight-name requires --lora."
            )
        if args.lora_scale is not None:
            raise SystemExit("--lora-scale requires --lora.")
        return None

    scale = 1.0 if args.lora_scale is None else args.lora_scale
    if not math.isfinite(scale):
        raise SystemExit("--lora-scale must be finite.")

    source = args.lora
    if not source:
        raise SystemExit("--lora must not be empty.")
    candidate = Path(source).expanduser()
    looks_local = (
        candidate.exists()
        or candidate.is_absolute()
        or source.startswith((".", "~"))
        or source.lower().endswith(".safetensors")
    )
    if looks_local and not candidate.exists():
        raise SystemExit(f"Local LoRA path does not exist: {candidate}")

    if candidate.exists():
        if args.lora_revision is not None:
            raise SystemExit("A local LoRA does not accept --lora-revision.")
        if candidate.is_file():
            if args.lora_weight_name is not None:
                raise SystemExit(
                    "A direct LoRA file does not accept --lora-weight-name."
                )
            resolved_file = candidate.resolve()
        elif candidate.is_dir():
            if args.lora_weight_name is None:
                raise SystemExit(
                    "A LoRA directory requires --lora-weight-name."
                )
            _validate_lora_weight_name(args.lora_weight_name)
            resolved_file = (candidate / args.lora_weight_name).resolve()
        else:
            raise SystemExit(f"Local LoRA path is not a regular file or directory: {candidate}")

        _validate_lora_weight_name(resolved_file.name)
        if not resolved_file.is_file() or resolved_file.stat().st_size == 0:
            raise SystemExit(f"LoRA safetensors file is missing or empty: {resolved_file}")
        return LoraSelection(
            source=str(resolved_file.parent),
            weight_name=resolved_file.name,
            requested_revision=None,
            is_local=True,
            scale=scale,
            sha256=_file_sha256(resolved_file),
            size_bytes=resolved_file.stat().st_size,
        )

    if args.lora_revision is None:
        raise SystemExit("A remote LoRA requires --lora-revision with an immutable commit SHA.")
    if re.fullmatch(r"[0-9a-f]{40}", args.lora_revision) is None:
        raise SystemExit("--lora-revision must be a 40-character lowercase commit SHA.")
    if args.lora_weight_name is None:
        raise SystemExit("A remote LoRA requires --lora-weight-name.")
    _validate_lora_weight_name(args.lora_weight_name)
    return LoraSelection(
        source=source,
        weight_name=args.lora_weight_name,
        requested_revision=args.lora_revision,
        is_local=False,
        scale=scale,
        sha256=None,
        size_bytes=None,
    )


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


def select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available in this PyTorch environment.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch environment.")
    return requested


def package_versions(require_lora: bool) -> dict[str, str]:
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
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit(
                f"Reference dependency {name} is missing; reinstall "
                "reference/diffusers/requirements.txt."
            ) from None
    try:
        versions["peft"] = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        if require_lora:
            raise SystemExit(
                "LoRA generation requires PEFT; reinstall "
                "reference/diffusers/requirements.txt."
            ) from None
    return versions


def _verify_local_lora_identity(selection: LoraSelection) -> None:
    if not selection.is_local:
        return
    path = Path(selection.source) / selection.weight_name
    if (
        not path.is_file()
        or path.stat().st_size != selection.size_bytes
        or _file_sha256(path) != selection.sha256
    ):
        raise RuntimeError(
            f"Local LoRA changed after argument resolution: {path}"
        )


def apply_lora(
    pipeline: Any,
    selection: LoraSelection | None,
    cache_directory: Path,
    local_files_only: bool,
) -> LoraActivation | None:
    if selection is None:
        return None

    _verify_local_lora_identity(selection)

    load_arguments: dict[str, Any] = {
        "adapter_name": LORA_ADAPTER_NAME,
        "cache_dir": cache_directory,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "weight_name": selection.weight_name,
    }
    if selection.requested_revision is not None:
        load_arguments["revision"] = selection.requested_revision

    try:
        pipeline.load_lora_weights(selection.source, **load_arguments)
        _verify_local_lora_identity(selection)
        pipeline.set_adapters(LORA_ADAPTER_NAME, adapter_weights=selection.scale)
        adapters_by_component = pipeline.get_list_adapters()
    except Exception as error:
        if _is_gated_repository_error(error):
            raise SystemExit(
                f"Cannot access gated LoRA {selection.source}. Accept its terms on "
                "Hugging Face and authenticate this environment with `hf auth login`, "
                "then retry."
            ) from None
        raise RuntimeError(
            f"Could not load and activate LoRA {selection.source}/{selection.weight_name}: "
            f"{error}"
        ) from error

    registered_components = tuple(
        sorted(
            component
            for component, adapter_names in adapters_by_component.items()
            if LORA_ADAPTER_NAME in adapter_names
        )
    )
    if not registered_components:
        raise RuntimeError(
            f"LoRA adapter {LORA_ADAPTER_NAME} was not registered and activated."
        )

    active_adapters: set[str] = set()
    for component in registered_components:
        model_component = getattr(pipeline, component, None)
        active_adapter_query = getattr(model_component, "active_adapters", None)
        if not callable(active_adapter_query):
            raise RuntimeError(
                f"LoRA component {component} cannot report its active adapters."
            )
        component_active_adapters = tuple(active_adapter_query())
        if LORA_ADAPTER_NAME not in component_active_adapters:
            raise RuntimeError(
                f"LoRA adapter {LORA_ADAPTER_NAME} is not active on {component}."
            )
        active_adapters.update(component_active_adapters)

    return LoraActivation(tuple(sorted(active_adapters)), registered_components)


def prepare_pipeline_with_adapters(
    pipeline: Any,
    preset: PipelinePreset,
    args: argparse.Namespace,
    device: str,
    attention_slicing: bool,
) -> tuple[Any, dict[str, Any], LoraActivation | None]:
    validate_pipeline_contract(pipeline, preset)
    activation = apply_lora(
        pipeline,
        args.lora_selection,
        args.cache_dir,
        args.local_files_only,
    )
    pipeline, optimization = prepare_pipeline_for_execution(
        pipeline,
        preset,
        device,
        attention_slicing,
    )
    return pipeline, optimization, activation


def lora_metadata(
    selection: LoraSelection | None,
    activation: LoraActivation | None,
) -> dict[str, Any] | None:
    if selection is None:
        return None
    if activation is None:
        raise RuntimeError("LoRA metadata requires a verified activation.")
    return {
        "active_adapters": list(activation.active_adapters),
        "adapter_name": LORA_ADAPTER_NAME,
        "format": "safetensors",
        "fused": False,
        "is_local": selection.is_local,
        "registered_components": list(activation.registered_components),
        "requested_revision": selection.requested_revision,
        "resolved_file": (
            str(Path(selection.source) / selection.weight_name)
            if selection.is_local
            else None
        ),
        "scale": selection.scale,
        "sha256": selection.sha256,
        "size_bytes": selection.size_bytes,
        "source": selection.source,
        "type": "lora",
        "weight_name": selection.weight_name,
    }


def write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safety_metadata(pipeline: Any, result: Any) -> dict[str, Any]:
    checker = getattr(pipeline, "safety_checker", None)
    watermarker = getattr(pipeline, "watermark", None)
    return {
        "checker_present": checker is not None,
        "nsfw_content_detected": getattr(result, "nsfw_content_detected", None),
        "watermarker_present": watermarker is not None,
    }


def component_dtypes(pipeline: Any, preset: PipelinePreset) -> dict[str, str]:
    return {
        name: str(component.dtype)
        for name, _ in preset.expected_components
        if (component := pipeline.components.get(name)) is not None
        and hasattr(component, "dtype")
    }


def validate_rendered_image(image: Any, width: int, height: int) -> tuple[tuple[int, int], ...]:
    if image.size != (width, height):
        raise RuntimeError(f"Unexpected image size: {image.size}")
    if image.mode != "RGB":
        raise RuntimeError(f"Rendered image must be RGB before validation, found {image.mode}")
    extrema = image.getextrema()
    if len(extrema) != 3 or all(low == high for low, high in extrema):
        raise RuntimeError(f"Rendered image is uniform or invalid: channel extrema {extrema}")
    return extrema


def build_load_arguments(
    preset: PipelinePreset,
    args: argparse.Namespace,
    dtype: Any,
    use_weight_variant: bool,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "dtype": dtype,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    if args.model_selection.requested_revision is not None:
        arguments["revision"] = args.model_selection.requested_revision
    if use_weight_variant and preset.runtime.weight_variant is not None:
        arguments["variant"] = preset.runtime.weight_variant
    if preset.name == "sdxl-base":
        arguments["add_watermarker"] = False
    return arguments


def build_pipeline_call_arguments(
    preset: PipelinePreset,
    args: argparse.Namespace,
    generator: Any,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if preset.runtime.passes_negative_prompt:
        arguments["negative_prompt"] = args.negative_prompt
    if preset.runtime.max_sequence_length is not None:
        arguments["max_sequence_length"] = preset.runtime.max_sequence_length
        arguments["output_type"] = "pil"
    return arguments


def resolve_attention_slicing(
    preset: PipelinePreset,
    device: str,
    requested: bool | None,
) -> bool:
    if requested and not preset.runtime.supports_attention_slicing:
        raise SystemExit(f"Attention slicing is not supported by preset {preset.name}.")
    if requested is None:
        return device == "mps" and preset.runtime.supports_attention_slicing
    return requested


def prepare_pipeline_for_execution(
    pipeline: Any,
    preset: PipelinePreset,
    device: str,
    attention_slicing: bool,
) -> tuple[Any, dict[str, Any]]:
    optimization = {
        "attention_slicing": False,
        "offload_policy": "none",
        "vae_slicing_enabled": False,
        "vae_tiling_enabled": False,
    }

    execution = preset.runtime.accelerator_execution
    if execution not in ("resident", "sequential-cpu-offload"):
        raise RuntimeError(f"Unknown accelerator execution policy: {execution}")
    if device in ("mps", "cuda") and execution == "sequential-cpu-offload":
        if preset.runtime.accelerator_vae_slicing:
            pipeline.vae.enable_slicing()
            optimization["vae_slicing_enabled"] = True
        if preset.runtime.accelerator_vae_tiling:
            pipeline.vae.enable_tiling()
            optimization["vae_tiling_enabled"] = True
        pipeline.enable_sequential_cpu_offload(device=device)
        optimization["offload_policy"] = "sequential-cpu"
    else:
        pipeline = pipeline.to(device)

    if attention_slicing:
        pipeline.enable_attention_slicing()
        optimization["attention_slicing"] = True
    return pipeline, optimization


def validate_generation_arguments(preset: PipelinePreset, args: argparse.Namespace) -> None:
    dimension_multiple = preset.runtime.dimension_multiple
    if (
        args.width <= 0
        or args.height <= 0
        or args.width % dimension_multiple
        or args.height % dimension_multiple
    ):
        raise SystemExit(
            f"Width and height must be positive multiples of {dimension_multiple} "
            f"for preset {preset.name}."
        )
    if args.steps <= 0:
        raise SystemExit("Inference steps must be positive.")
    if preset.name == "flux1-schnell" and args.guidance_scale != 0.0:
        raise SystemExit("FLUX.1-schnell guidance scale must be 0.0.")
    if (
        not preset.runtime.passes_negative_prompt
        and args.negative_prompt != DEFAULT_NEGATIVE_PROMPT
    ):
        raise SystemExit(f"Preset {preset.name} does not use --negative-prompt.")
    if args.output.suffix.lower() != ".png":
        raise SystemExit("Reference output must use the .png extension.")


def main() -> int:
    preset, args = resolve_arguments(build_parser().parse_args())
    validate_generation_arguments(preset, args)

    metadata_path = args.output.with_suffix(".json")
    existing = [path for path in (args.output, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing reference output: {paths}")

    installed_package_versions = package_versions(args.lora_selection is not None)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.xet_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(args.xet_cache_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch, pipeline_classes = load_dependencies()
    device = select_device(torch, args.device)
    dtype_name = (
        preset.runtime.cpu_dtype if device == "cpu" else preset.runtime.accelerator_dtype
    )
    dtype = getattr(torch, dtype_name)
    attention_slicing = resolve_attention_slicing(
        preset, device, args.attention_slicing
    )
    load_arguments = build_load_arguments(
        preset, args, dtype, use_weight_variant=device != "cpu"
    )

    pipeline_class = pipeline_classes[preset.pipeline_class]
    pipeline = load_pipeline(
        pipeline_class,
        args.model_selection.source,
        load_arguments,
    )
    pipeline, optimization, lora_activation = prepare_pipeline_with_adapters(
        pipeline,
        preset,
        args,
        device,
        attention_slicing,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        result = pipeline(**build_pipeline_call_arguments(preset, args, generator))

    image = result.images[0]
    if image.mode != "RGB":
        image = image.convert("RGB")
    channel_extrema = validate_rendered_image(image, args.width, args.height)

    temporary_image = args.output.with_name(args.output.name + ".tmp")
    image.save(temporary_image, format="PNG")
    os.replace(temporary_image, args.output)
    image_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()

    metadata: dict[str, Any] = {
        "adapters": (
            []
            if args.lora_selection is None
            else [lora_metadata(args.lora_selection, lora_activation)]
        ),
        "fixture": {
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "max_sequence_length": preset.runtime.max_sequence_length,
            "negative_prompt": (
                args.negative_prompt if preset.runtime.passes_negative_prompt else None
            ),
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": args.steps,
            "width": args.width,
        },
        "model": {
            "id_or_path": args.model_selection.source,
            "preset": preset.name,
            "requested_revision": args.model_selection.requested_revision,
            "pipeline_class": type(pipeline).__name__,
            "scheduler_class": type(pipeline.scheduler).__name__,
        },
        "output": {
            "mode": image.mode,
            "path": str(args.output.resolve()),
            "sha256": image_sha256,
            "size": [image.width, image.height],
            "channel_extrema": channel_extrema,
        },
        "runtime": {
            "device": device,
            "execution_device": (
                f"{device}:0"
                if optimization["offload_policy"] != "none"
                else device
            ),
            "component_dtypes": component_dtypes(pipeline, preset),
            "dtype": str(dtype),
            "optimization": optimization,
            "xet_cache": os.environ.get("HF_XET_CACHE"),
            "packages": installed_package_versions,
            "platform": platform.platform(),
            "python": sys.version,
        },
        "safety": safety_metadata(pipeline, result),
        "reproducibility": {
            "generator_device": "cpu",
            "note": (
                "A seed does not guarantee pixel identity across hardware "
                "or runtime releases."
            ),
        },
    }
    write_json_atomically(metadata_path, metadata)

    print(f"Image: {args.output.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"SHA-256: {image_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
