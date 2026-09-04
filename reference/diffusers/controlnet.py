"""Optional, independently selected ControlNet weights and conditioning images."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from model_loading import require_materialized_component, selection_metadata, read_weight_keys
from pipeline_loading import _is_gated_repository_error
from presets import ModelSelection, PipelinePreset, resolve_model_selection, validate_pipeline_contract
from weight_files import (
    LocalWeightFile, checked_safetensors_path, file_sha256, verify_weight_file,
)


PIPELINES = {
    "sd15": ("StableDiffusionControlNetPipeline", "ControlNetModel"),
    "sdxl-base": ("StableDiffusionXLControlNetPipeline", "ControlNetModel"),
    "flux1-schnell": ("FluxControlNetPipeline", "FluxControlNetModel"),
}
DEPENDENT_OPTIONS = (
    "controlnet_revision", "controlnet_config", "controlnet_config_revision",
    "controlnet_variant", "control_image", "controlnet_scale",
    "control_guidance_start", "control_guidance_end", "guess_mode", "control_mode",
)


def add_controlnet_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controlnet", default=None,
                        help="ControlNet Diffusers directory, pinned Hub ID, or local safetensors file")
    parser.add_argument("--controlnet-revision", default=None,
                        help="Immutable 40-character lowercase Hub commit for ControlNet")
    parser.add_argument("--controlnet-config", default=None,
                        help="Component config directory/Hub ID for a ControlNet file; defaults to its sibling config.json")
    parser.add_argument("--controlnet-config-revision", default=None,
                        help="Immutable Hub commit for --controlnet-config")
    parser.add_argument("--controlnet-variant", default=None,
                        help="ControlNet package weight variant, for example fp16; independent of base weights")
    parser.add_argument("--control-image", type=Path, default=None,
                        help="Local conditioning image prepared for the selected ControlNet (no automatic detector)")
    parser.add_argument("--controlnet-scale", "--controlnet-conditioning-scale", type=float, default=None,
                        help="Non-negative conditioning strength; selected ControlNet defaults to 1.0")
    parser.add_argument("--control-guidance-start", type=float, default=None,
                        help="ControlNet start fraction of denoising; default 0.0")
    parser.add_argument("--control-guidance-end", type=float, default=None,
                        help="ControlNet end fraction of denoising; default 1.0")
    parser.add_argument("--guess-mode", action=argparse.BooleanOptionalAction, default=None,
                        help="SD/SDXL ControlNet guess mode; default false")
    parser.add_argument("--control-mode", type=int, default=None,
                        help="Model-defined FLUX Union conditioning mode; required only by a Union model")


def _image_file(path: Path) -> LocalWeightFile:
    path = path.expanduser().absolute()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"--control-image must be a non-empty local image file: {path}")
    resolved = path.resolve()
    return LocalWeightFile(str(path), str(resolved), file_sha256(resolved), resolved.stat().st_size)


def resolve_controlnet_options(preset: PipelinePreset, args: Any) -> None:
    args.controlnet_selection = None
    args.controlnet_config_selection = None
    args.control_image_file = None
    if args.controlnet is None:
        if any(getattr(args, name) is not None for name in DEPENDENT_OPTIONS):
            raise ValueError("ControlNet options require --controlnet.")
        return
    if not args.controlnet:
        raise ValueError("--controlnet must not be empty.")
    args.controlnet_selection = resolve_model_selection(
        preset, args.controlnet, args.controlnet_revision, allow_single_file=True,
        model_argument="--controlnet", revision_argument="--controlnet-revision")
    if not args.controlnet_selection.is_local and args.controlnet_revision is None:
        raise ValueError("A remote ControlNet requires an explicit --controlnet-revision.")
    if args.control_image is None:
        raise ValueError("--controlnet requires --control-image.")
    args.control_image_file = _image_file(args.control_image)
    args.control_image = Path(args.control_image_file.path)
    for name, default in (("controlnet_scale", 1.0), ("control_guidance_start", 0.0),
                          ("control_guidance_end", 1.0), ("guess_mode", False)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not math.isfinite(args.controlnet_scale) or args.controlnet_scale < 0:
        raise ValueError("--controlnet-scale must be finite and non-negative.")
    start, end = args.control_guidance_start, args.control_guidance_end
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= 1):
        raise ValueError("Control guidance requires 0 <= --control-guidance-start < --control-guidance-end <= 1.")
    if args.guidance_rescale != 0:
        raise ValueError("--guidance-rescale is not supported by ControlNet pipelines.")
    if preset.family == "flux1-schnell" and args.guess_mode:
        raise ValueError("--guess-mode requires SD/SDXL ControlNet.")
    if args.control_mode is not None and (preset.family != "flux1-schnell" or args.control_mode < 0):
        raise ValueError("--control-mode requires FLUX ControlNet and a non-negative mode index.")
    if args.controlnet_variant is not None and (
        not args.controlnet_variant or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.controlnet_variant)
    ):
        raise ValueError("--controlnet-variant must be a non-empty filename variant.")
    single_file = args.controlnet_selection.single_file
    if single_file is None:
        if args.controlnet_config is not None or args.controlnet_config_revision is not None:
            raise ValueError("--controlnet-config options require a single-file --controlnet.")
    else:
        if args.controlnet_variant is not None:
            raise ValueError("--controlnet-variant requires a ControlNet directory or Hub repository.")
        config_source = args.controlnet_config
        if config_source is None:
            sibling = Path(single_file.path).parent
            if not (sibling / "config.json").is_file():
                raise ValueError("A single-file --controlnet requires --controlnet-config or a sibling config.json.")
            config_source = str(sibling)
        args.controlnet_config_selection = resolve_model_selection(
            preset, config_source, args.controlnet_config_revision,
            model_argument="--controlnet-config", revision_argument="--controlnet-config-revision")
        if not args.controlnet_config_selection.is_local and args.controlnet_config_revision is None:
            raise ValueError("A remote ControlNet configuration requires an explicit --controlnet-config-revision.")


def controlnet_preset(preset: PipelinePreset) -> PipelinePreset:
    pipeline_class, model_class = PIPELINES[preset.family]
    return replace(preset, pipeline_class=pipeline_class,
                   expected_components=(*preset.expected_components, ("controlnet", model_class)))


def load_control_image(args: Any) -> Any:
    if args.control_image_file is None:
        return None
    from PIL import Image, ImageOps

    identity = args.control_image_file
    verify_weight_file(identity, "control image")
    try:
        with Image.open(identity.path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Animated control images are not supported; select a single frame.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not decode --control-image: {error}") from error
    verify_weight_file(identity, "control image")
    args.controlnet_image = image
    return image


def _snapshot(selection: ModelSelection, args: Any, *, config_only: bool = False) -> Path:
    if selection.is_local:
        root = Path(selection.source)
        if not root.is_dir():
            raise ValueError(f"ControlNet package directory is missing: {root}")
        return root
    from huggingface_hub import snapshot_download

    variant = args.controlnet_variant
    suffix = f".{variant}" if variant else ""
    indexes = list(dict.fromkeys((f"diffusion_pytorch_model.safetensors.index{suffix}.json",
                                 f"diffusion_pytorch_model.safetensors{suffix}.index.json")))
    download_arguments = dict(
        revision=selection.requested_revision, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only)
    root = Path(snapshot_download(
        selection.source, **download_arguments,
        allow_patterns=["config.json"] if config_only else ["config.json", *indexes]))
    if config_only:
        return root
    # Read the selected index before fetching tensors. Variant indexes can refer
    # to historical shard names, so prefix globs both miss files and overfetch.
    weights = [str(path.relative_to(root).as_posix()) for path in _package_files(root, variant)
               if path.suffix == ".safetensors"]
    return Path(snapshot_download(selection.source, **download_arguments, allow_patterns=weights))


def _package_files(root: Path, variant: str | None) -> list[Path]:
    """Select the same index or single weight file as Diffusers, without loading tensors."""
    if not root.is_dir():
        raise ValueError(f"ControlNet package directory is missing: {root}")
    suffix = f".{variant}" if variant else ""
    indexes = (root / f"diffusion_pytorch_model.safetensors.index{suffix}.json",
               root / f"diffusion_pytorch_model.safetensors{suffix}.index.json")
    index = next((path for path in indexes if path.exists()), None)
    if index is None:
        return [root / f"diffusion_pytorch_model{suffix}.safetensors"]
    try:
        values = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Could not read ControlNet safetensors index {index}: {error}") from error
    weight_map = values.get("weight_map") if isinstance(values, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"ControlNet safetensors index requires a non-empty weight_map object: {index}")
    shards = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"ControlNet safetensors index contains an invalid tensor name: {index}")
        if (not isinstance(filename, str) or not filename.endswith(".safetensors")
                or any(character in filename for character in ("\\", ":", "\0", "*", "?", "[", "]"))
                or any(part in ("", ".", "..") for part in filename.split("/"))):
            raise ValueError(f"ControlNet safetensors index contains an unsafe shard filename: {index}")
        shards.add(filename)
    return [index, *(root / filename for filename in sorted(shards))]


def _identities(paths: list[Path]) -> list[LocalWeightFile]:
    result = []
    for path in sorted(set(paths)):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"ControlNet package file is missing or empty: {path}")
        result.append(LocalWeightFile(str(path.absolute()), str(path.resolve()), file_sha256(path), path.stat().st_size))
    return result


def _file_metadata(identity: LocalWeightFile) -> dict[str, Any]:
    return dict(path=identity.path, resolved_file=identity.resolved_file,
                sha256=identity.sha256, size_bytes=identity.size_bytes)


def validate_controlnet_contract(controlnet: Any, pipeline: Any, preset: PipelinePreset, args: Any) -> None:
    expected = PIPELINES[preset.family][1]
    if type(controlnet).__name__ != expected:
        raise RuntimeError(f"ControlNet must be {expected} for {preset.name}.")
    base = pipeline.transformer if preset.family == "flux1-schnell" else pipeline.unet
    fields = (
        ("in_channels", "patch_size", "attention_head_dim", "num_attention_heads",
         "joint_attention_dim", "pooled_projection_dim", "axes_dims_rope")
        if preset.family == "flux1-schnell" else
        ("in_channels", "cross_attention_dim", "block_out_channels", "down_block_types",
         "layers_per_block", "addition_embed_type", "addition_time_embed_dim",
         "projection_class_embeddings_input_dim")
    )
    for field in fields:
        actual, target = getattr(controlnet.config, field, None), getattr(base.config, field, None)
        if isinstance(actual, (list, tuple)) and isinstance(target, (list, tuple)):
            actual, target = tuple(actual), tuple(target)
        if actual != target:
            raise RuntimeError(f"ControlNet {field} is incompatible with {preset.name}: {actual} != {target}")
    if preset.family != "flux1-schnell":
        if getattr(controlnet.config, "conditioning_channels", 3) != 3:
            raise RuntimeError("ControlNet must accept a three-channel RGB conditioning image.")
    else:
        mode_count = getattr(controlnet.config, "num_mode", None)
        if getattr(controlnet, "union", False):
            if args.control_mode is None or not 0 <= args.control_mode < mode_count:
                raise ValueError(f"This FLUX Union ControlNet requires --control-mode in [0,{mode_count}).")
        elif args.control_mode is not None:
            raise ValueError("--control-mode requires a FLUX Union ControlNet.")


def attach_controlnet(pipeline: Any, preset: PipelinePreset, args: Any,
                      classes: dict[str, Any], dtype: Any) -> tuple[Any, dict[str, Any] | None]:
    selection = args.controlnet_selection
    if selection is None:
        return pipeline, None
    # Preserve the strict base-model contract before changing the pipeline type.
    validate_pipeline_contract(pipeline, preset)
    model_class = classes[PIPELINES[preset.family][1]]
    try:
        config_selection = args.controlnet_config_selection
        root = _snapshot(config_selection or selection, args, config_only=selection.single_file is not None)
        config_path = root / "config.json"
        identities = _identities([config_path])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("_class_name") != PIPELINES[preset.family][1]:
            raise ValueError(f"ControlNet config.json must describe {PIPELINES[preset.family][1]}.")
        load_args = dict(dtype=dtype, local_files_only=True, use_safetensors=True,
                         low_cpu_mem_usage=args.low_cpu_mem_usage)
        if selection.single_file is not None:
            weight = selection.single_file
            with checked_safetensors_path(weight, args.cache_dir / "single-file-aliases", "ControlNet") as path:
                keys = read_weight_keys(path)
                if any(key.startswith(("control_model.", "input_blocks.")) for key in keys):
                    if preset.family == "flux1-schnell":
                        raise ValueError("FLUX ControlNet files require native Diffusers tensor names.")
                    controlnet = _load_original_controlnet(model_class, path, config, root, load_args)
                else:
                    # Diffusers' normal package loader retains low-memory loading and
                    # strict missing-weight reporting for native component state dicts.
                    with tempfile.TemporaryDirectory(prefix="controlnet-", dir=args.cache_dir) as temporary:
                        staging = Path(temporary)
                        (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
                        (staging / "diffusion_pytorch_model.safetensors").symlink_to(path.resolve())
                        controlnet = _load_package(model_class, staging, load_args)
            identities.append(weight)
        else:
            identities.extend(_identities(_package_files(root, args.controlnet_variant)))
            if args.controlnet_variant is not None:
                load_args["variant"] = args.controlnet_variant
            controlnet = _load_package(model_class, root, load_args)
        require_materialized_component(controlnet, "controlnet")
        validate_controlnet_contract(controlnet, pipeline, preset, args)
        for identity in identities:
            verify_weight_file(identity, "ControlNet")
        extras = {"controlnet": controlnet}
        if preset.family == "sdxl-base":
            extras["add_watermarker"] = bool(args.watermark)
        pipeline = classes[PIPELINES[preset.family][0]].from_pipe(pipeline, dtype=dtype, **extras)
        validate_pipeline_contract(pipeline, controlnet_preset(preset))
    except Exception as error:
        if _is_gated_repository_error(error):
            raise SystemExit("Cannot access gated ControlNet/configuration weights; accept their terms and authenticate with `hf auth login`.") from None
        raise RuntimeError(f"Could not assemble ControlNet: {error}") from error
    return pipeline, {
        **selection_metadata(selection),
        "configuration": None if config_selection is None else selection_metadata(config_selection),
        "model_class": type(controlnet).__name__, "files": [_file_metadata(item) for item in identities],
        "variant": args.controlnet_variant, "scale": args.controlnet_scale,
        "guidance_start": args.control_guidance_start, "guidance_end": args.control_guidance_end,
        "guess_mode": bool(args.guess_mode), "control_mode": args.control_mode,
        "image": {**_file_metadata(args.control_image_file), "mode": "RGB",
                  "oriented_size": list(args.controlnet_image.size),
                  "generation_size": [args.width, args.height],
                  "preprocessing": "EXIF transpose, RGB conversion; Diffusers resize; no detector"},
    }


def _load_original_controlnet(model_class: Any, path: Path, config: dict[str, Any],
                             root: Path, load_args: dict[str, Any]) -> Any:
    from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint
    from safetensors.torch import load_file

    # The upstream loader drops missing_keys under low_cpu_mem_usage=False.
    # Retain its conversion but verify the complete native state dictionary.
    converted = convert_controlnet_checkpoint(load_file(str(path)), config=config)
    model = model_class.from_single_file(converted, config=str(root), **load_args)
    expected, supplied = set(model.state_dict()), set(converted)
    missing, unexpected = sorted(expected - supplied), sorted(supplied - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"ControlNet converted weights do not match their configuration: "
            f"missing={missing}, unexpected={unexpected}")
    return model


def _load_package(model_class: Any, root: Path, load_args: dict[str, Any]) -> Any:
    model, info = model_class.from_pretrained(str(root), output_loading_info=True, **load_args)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"ControlNet weights do not match their configuration: {info}")
    return model


def controlnet_call_arguments(preset: PipelinePreset, args: Any) -> dict[str, Any]:
    if getattr(args, "controlnet_selection", None) is None:
        return {}
    if getattr(args, "controlnet_image", None) is None:
        raise ValueError("ControlNet generation requires a decoded --control-image; call load_control_image first.")
    values = {
        "control_image" if preset.family == "flux1-schnell" else "image": args.controlnet_image,
        "controlnet_conditioning_scale": args.controlnet_scale,
        "control_guidance_start": args.control_guidance_start,
        "control_guidance_end": args.control_guidance_end,
    }
    if preset.family == "flux1-schnell":
        if args.control_mode is not None:
            values["control_mode"] = args.control_mode
    else:
        values["guess_mode"] = bool(args.guess_mode)
    return values
