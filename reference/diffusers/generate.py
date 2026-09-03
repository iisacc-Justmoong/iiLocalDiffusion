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

from cpu_conditioning import CpuConditioning, encode_cpu_prompt
from hardware import (
    accelerator_preflight, configure_tensor_cores, select_device, validate_execution_device,
)
from model_loading import load_generation_pipeline, selection_metadata
from pipeline_loading import _is_gated_repository_error
from presets import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    SD15_PRESET,
    PipelinePreset,
    resolve_model_selection,
    validate_pipeline_contract,
)
from weight_files import (
    LocalWeightFile,
    SAFETENSORS_SUFFIXES,
    checked_safetensors_path,
    file_sha256,
    resolve_weight_file,
    verify_weight_file,
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
    local_file: LocalWeightFile | None = None


@dataclass(frozen=True)
class LoraActivation:
    active_adapters: tuple[str, ...]
    registered_components: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a pinned Diffusers reference fixture."
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET_NAME)
    parser.add_argument(
        "--model",
        default=None,
        help="Hub model ID, local Diffusers directory, or local safetensors model file",
    )
    parser.add_argument("--revision", default=None, help="Immutable Hub commit revision")
    parser.add_argument(
        "--model-config",
        default=None,
        help="Diffusers directory/Hub ID supplying a single-file model's configuration and extras",
    )
    parser.add_argument(
        "--model-config-revision",
        default=None,
        help="Immutable commit for --model-config; defaults to the selected preset revision",
    )
    parser.add_argument(
        "--vae", default=None, help="Replacement local VAE .safetensors or .safetensor file"
    )
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
    parser.add_argument(
        "--device", choices=("auto", "metal", "mps", "cuda", "rocm", "cpu"), default="auto",
        help="GPU required by default; rocm requires AMD HIP PyTorch, metal aliases MPS; CPU is explicit",
    )
    parser.add_argument(
        "--cpu-text-encoding", action="store_true",
        help="Compute prompt embeddings on CPU before GPU denoising and decoding",
    )
    parser.add_argument(
        "--cuda-tf32", action=argparse.BooleanOptionalAction, default=None,
        help="Allow TF32 Tensor Core math on CUDA (auto enables it on eligible GPUs); "
             "disabling TF32 does not disable FP16/BF16 Tensor Core kernels",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=None,
        help="Positive PyTorch CPU intra-op thread count; otherwise retain the runtime default",
    )
    parser.add_argument(
        "--offload", choices=("auto", "none", "model", "sequential"), default="auto",
        help="RAM weight offload: auto uses the preset, or model offload for CPU text encoding",
    )
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
        selection = resolve_model_selection(
            preset, args.model, args.revision, allow_single_file=True
        )
        config_selection = None
        if selection.single_file is not None:
            config_selection = resolve_model_selection(
                preset,
                args.model_config,
                args.model_config_revision,
                model_argument="--model-config",
                revision_argument="--model-config-revision",
            )
        elif args.model_config is not None or args.model_config_revision is not None:
            raise ValueError("--model-config options require --model to be a single file.")
        vae_file = None if args.vae is None else resolve_weight_file(args.vae, "--vae")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    resolved = argparse.Namespace(**vars(args))
    resolved.model_selection = selection
    resolved.config_selection = config_selection
    resolved.vae_file = vae_file
    resolved.lora_selection = resolve_lora_selection(resolved)
    resolved.width = preset.width if args.width is None else args.width
    resolved.height = preset.height if args.height is None else args.height
    resolved.steps = preset.steps if args.steps is None else args.steps
    resolved.guidance_scale = (
        preset.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    filename = Path(preset.generation_filename)
    modifiers = []
    if args.model is not None or args.revision is not None:
        modifiers.append("custom")
    if vae_file is not None:
        modifiers.append("vae")
    if resolved.lora_selection is not None:
        modifiers.append("lora")
    suffix = "" if not modifiers else "-" + "-".join(modifiers)
    generation_filename = f"{filename.stem}{suffix}{filename.suffix}"
    resolved.output = args.output or REPOSITORY_ROOT / "build" / "reference" / generation_filename
    return preset, resolved


def _validate_lora_weight_name(weight_name: str, *, local: bool = False) -> None:
    suffixes = SAFETENSORS_SUFFIXES if local else (".safetensors",)
    if (
        not weight_name
        or "/" in weight_name
        or "\\" in weight_name
        or Path(weight_name).suffix not in suffixes
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
        or source.lower().endswith(SAFETENSORS_SUFFIXES)
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
            resolved_file = candidate.absolute()
        elif candidate.is_dir():
            if args.lora_weight_name is None:
                raise SystemExit(
                    "A LoRA directory requires --lora-weight-name."
                )
            _validate_lora_weight_name(args.lora_weight_name, local=True)
            resolved_file = (candidate / args.lora_weight_name).absolute()
        else:
            raise SystemExit(f"Local LoRA path is not a regular file or directory: {candidate}")

        _validate_lora_weight_name(resolved_file.name, local=True)
        if not resolved_file.is_file() or resolved_file.stat().st_size == 0:
            raise SystemExit(f"LoRA safetensors file is missing or empty: {resolved_file}")
        try:
            local_file = resolve_weight_file(str(resolved_file), "--lora")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        return LoraSelection(
            source=str(resolved_file.parent),
            weight_name=resolved_file.name,
            requested_revision=None,
            is_local=True,
            scale=scale,
            sha256=local_file.sha256,
            size_bytes=local_file.size_bytes,
            local_file=local_file,
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
        from diffusers import (
            AutoencoderKL,
            FluxPipeline,
            FluxTransformer2DModel,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
        )
        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
    except ImportError as error:
        raise SystemExit(
            "Reference dependencies are missing. Install reference/diffusers/requirements.txt "
            "into reference/diffusers/.venv first."
        ) from error
    return torch, {
        "AutoencoderKL": AutoencoderKL,
        "FluxPipeline": FluxPipeline,
        "FluxTransformer2DModel": FluxTransformer2DModel,
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "StableDiffusionSafetyChecker": StableDiffusionSafetyChecker,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
        "UNet2DConditionModel": UNet2DConditionModel,
    }


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
    if selection.local_file is not None:
        verify_weight_file(selection.local_file, "LoRA")
        return
    path = Path(selection.source) / selection.weight_name
    if (
        not path.is_file()
        or path.stat().st_size != selection.size_bytes
        or file_sha256(path) != selection.sha256
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
        if selection.local_file is not None:
            with checked_safetensors_path(
                selection.local_file, cache_directory / "single-file-aliases", "LoRA"
            ) as path:
                load_arguments["weight_name"] = path.name
                pipeline.load_lora_weights(str(path.parent), **load_arguments)
        else:
            _verify_local_lora_identity(selection)
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
    torch: Any = None,
) -> tuple[Any, dict[str, Any], LoraActivation | None, CpuConditioning | None]:
    validate_pipeline_contract(pipeline, preset)
    activation = apply_lora(
        pipeline,
        args.lora_selection,
        args.cache_dir,
        args.local_files_only,
    )
    conditioning = None
    requested_offload = getattr(args, "offload", "auto")
    offload = requested_offload
    if getattr(args, "cpu_text_encoding", False):
        if torch is None:
            raise ValueError("CPU text encoding requires the PyTorch runtime.")
        conditioning = encode_cpu_prompt(pipeline, preset, args, torch)
        if (offload == "auto" and device != "cpu"
                and preset.runtime.accelerator_execution == "resident"):
            offload = "model"
    pipeline, optimization = prepare_pipeline_for_execution(
        pipeline,
        preset,
        device,
        attention_slicing,
        offload=offload,
    )
    optimization["requested_offload"] = requested_offload
    return pipeline, optimization, activation, conditioning


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
            (
                selection.local_file.resolved_file
                if selection.local_file is not None
                else str(Path(selection.source) / selection.weight_name)
            )
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
    variant = preset.runtime.weight_variant
    source = args.config_selection or args.model_selection
    if use_weight_variant and variant is not None:
        has_variant = source.source == preset.model_id
        if source.is_local:
            has_variant = any(Path(source.source).glob(f"*/*.{variant}*.safetensors"))
        if has_variant:
            arguments["variant"] = variant
    if preset.name == "sdxl-base":
        arguments["add_watermarker"] = False
    return arguments


def build_pipeline_call_arguments(
    preset: PipelinePreset,
    args: argparse.Namespace,
    generator: Any,
    conditioning: CpuConditioning | None = None,
    device: str | None = None,
    dtype: Any = None,
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
    if conditioning is not None:
        if device is None or dtype is None:
            raise ValueError("CPU conditioning requires the target device and dtype.")
        arguments.pop("prompt")
        arguments.pop("negative_prompt", None)
        arguments.update(conditioning.for_device(device, dtype))
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
    *,
    offload: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    if offload not in ("auto", "none", "model", "sequential"):
        raise ValueError(f"Unknown RAM offload policy: {offload}")
    if offload in ("model", "sequential") and device not in ("mps", "cuda"):
        raise ValueError("RAM weight offload requires a GPU execution device.")
    optimization = {
        "attention_slicing": False,
        "offload_policy": "none",
        "vae_slicing_enabled": False,
        "vae_tiling_enabled": False,
        "weight_storage": "ram" if device == "cpu" else "device",
    }

    execution = preset.runtime.accelerator_execution
    if execution not in ("resident", "sequential-cpu-offload"):
        raise RuntimeError(f"Unknown accelerator execution policy: {execution}")
    if offload == "auto":
        offload = ("sequential" if device in ("mps", "cuda")
                   and execution == "sequential-cpu-offload" else "none")
    if device in ("mps", "cuda"):
        if preset.runtime.accelerator_vae_slicing:
            pipeline.vae.enable_slicing()
            optimization["vae_slicing_enabled"] = True
        if preset.runtime.accelerator_vae_tiling:
            pipeline.vae.enable_tiling()
            optimization["vae_tiling_enabled"] = True
    if offload == "sequential":
        pipeline.enable_sequential_cpu_offload(device=device)
        optimization["offload_policy"] = "sequential-cpu"
        optimization["weight_storage"] = "ram"
    elif offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
        optimization["offload_policy"] = "model-cpu"
        optimization["weight_storage"] = "ram"
    else:
        pipeline = pipeline.to(device)

    if attention_slicing:
        pipeline.enable_attention_slicing()
        optimization["attention_slicing"] = True
    return pipeline, optimization


def validate_generation_arguments(preset: PipelinePreset, args: argparse.Namespace) -> None:
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise SystemExit("CPU threads must be positive.")
    if args.device == "cpu" and args.offload in ("model", "sequential"):
        raise SystemExit("RAM weight offload requires a GPU execution device.")
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
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = select_device(torch, args.device)
    dtype_name = (
        preset.runtime.cpu_dtype if device == "cpu" else preset.runtime.accelerator_dtype
    )
    dtype = getattr(torch, dtype_name)
    tensor_cores = configure_tensor_cores(torch, device, args.cuda_tf32)
    hardware = accelerator_preflight(torch, device, dtype)
    hardware["requested_device"] = args.device
    hardware["tensor_cores"] = tensor_cores
    print(f"Compute: {hardware['runtime']} / {hardware['backend']} ({dtype_name})", flush=True)
    attention_slicing = resolve_attention_slicing(
        preset, device, args.attention_slicing
    )
    load_arguments = build_load_arguments(
        preset, args, dtype, use_weight_variant=device != "cpu"
    )

    pipeline_class = pipeline_classes[preset.pipeline_class]
    print(f"Model: {args.model_selection.source}", flush=True)
    if args.config_selection is not None:
        print(f"Configuration / auxiliary models: {args.config_selection.source}", flush=True)
    if args.vae_file is not None:
        print(f"VAE override: {args.vae_file.path}", flush=True)
    if args.lora_selection is not None:
        print(f"LoRA: {args.lora_selection.source}/{args.lora_selection.weight_name}", flush=True)

    pipeline, loading_metadata = load_generation_pipeline(
        pipeline_class,
        preset,
        args.model_selection,
        args.config_selection,
        args.vae_file,
        load_arguments,
        pipeline_classes,
    )
    pipeline, optimization, lora_activation, conditioning = prepare_pipeline_with_adapters(
        pipeline,
        preset,
        args,
        device,
        attention_slicing,
        torch,
    )
    validate_execution_device(pipeline, device)
    hardware["participating_devices"] = (
        ["cpu", device] if conditioning is not None and device != "cpu" else [device]
    )
    print(f"Resources: {', '.join(hardware['participating_devices'])}; "
          f"weight storage={optimization['weight_storage']}; "
          f"offload={optimization['offload_policy']}", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        result = pipeline(**build_pipeline_call_arguments(
            preset, args, generator, conditioning, device, dtype))

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
            **selection_metadata(args.model_selection),
            "preset": preset.name,
            "pipeline_class": type(pipeline).__name__,
            "scheduler_class": type(pipeline.scheduler).__name__,
            "loading": loading_metadata,
        },
        "vae": {
            "overridden": args.vae_file is not None,
            "source": loading_metadata["component_sources"]["vae"],
            "file": loading_metadata["vae_override"],
            "latent_channels": pipeline.vae.config.latent_channels,
            "scaling_factor": pipeline.vae.config.scaling_factor,
            "shift_factor": getattr(pipeline.vae.config, "shift_factor", None),
            "spatial_downsample_factor": 2 ** (len(pipeline.vae.config.block_out_channels) - 1),
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
            "execution_device": str(pipeline._execution_device),
            "hardware": hardware,
            "cpu_threads": torch.get_num_threads(),
            "cpu_conditioning": (
                {"enabled": False} if conditioning is None else conditioning.metadata
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
