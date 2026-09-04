"""Validated, replayable options for repeated image-to-image HiRes Fix passes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from typing import Any

from generation_config import json_object
from generation_options import SDXL_SIZES
from presets import PipelinePreset


HIRES_UPSCALERS = ("nearest", "bilinear", "bicubic", "lanczos")
HIRES_OPTION_NAMES = (
    "hires_passes", "hires_scale", "hires_width", "hires_height", "hires_upscaler",
    "hires_denoising_strength", "hires_steps", "hires_seed", "hires_guidance_scale",
    "hires_true_cfg_scale", "hires_scheduler", "hires_scheduler_config", "hires_save_base",
)


def add_hires_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hires-fix", action=argparse.BooleanOptionalAction, default=False,
                        help="Upscale generated images and refine them with repeated diffusion passes")
    parser.add_argument("--hires-passes", type=int, default=None,
                        help="Number of additional HiRes refinements (default when enabled: 1)")
    parser.add_argument("--hires-scale", type=float, default=None,
                        help="Per-pass HiRes multiplier above 1; defaults to 2 when no first target size is given")
    parser.add_argument("--hires-width", type=int, default=None,
                        help="First refinement width; subsequent passes repeat the resulting width ratio")
    parser.add_argument("--hires-height", type=int, default=None,
                        help="First refinement height; subsequent passes repeat the resulting height ratio")
    parser.add_argument("--hires-upscaler", choices=HIRES_UPSCALERS, default=None,
                        help="Image interpolation before refinement (default with HiRes Fix: lanczos)")
    parser.add_argument("--hires-denoising-strength", "--hires-strength", type=float, default=None,
                        help="Each refinement's denoising strength in (0,1] (default with HiRes Fix: 0.35)")
    parser.add_argument("--hires-steps", type=int, default=None,
                        help="Each refinement's schedule length; defaults to --steps, reduced by strength")
    parser.add_argument("--hires-seed", type=int, default=None,
                        help="Each refinement's first image seed; defaults to --seed and reuses --seed-stride")
    parser.add_argument("--hires-guidance-scale", type=float, default=None,
                        help="Each refinement's guidance; defaults to --guidance-scale (FLUX schnell requires 0)")
    parser.add_argument("--hires-true-cfg-scale", type=float, default=None,
                        help="FLUX refinement true CFG, at least 1; defaults to --true-cfg-scale")
    parser.add_argument("--hires-scheduler", default=None,
                        help="Each refinement's compatible Diffusers scheduler; auto inherits the first pass")
    parser.add_argument("--hires-scheduler-config", type=json_object, default=None,
                        help="Each refinement's scheduler constructor overrides as JSON (default: {})")
    parser.add_argument("--hires-save-base", action=argparse.BooleanOptionalAction, default=None,
                        help="Also save the first-pass images (default with HiRes Fix: disabled)")


def _option(name: str) -> str:
    return "--" + name.replace("_", "-")


def _finite_number(value: Any, name: str, *, lower: float, inclusive: bool = True,
                   upper: float | None = None) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < lower or (not inclusive and value == lower)
            or (upper is not None and value > upper)):
        interval = "[" if inclusive else "("
        interval += f"{lower},{'infinity' if upper is None else upper}"
        interval += ")" if upper is None else "]"
        raise SystemExit(f"{_option(name)} must be finite and in {interval}.")


def _integer(value: Any, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise SystemExit(f"{_option(name)} must be {'a positive' if positive else 'an'} integer.")


def _nearest_multiple(value: float, multiple: int) -> int:
    if not math.isfinite(value):
        raise SystemExit("The requested HiRes dimensions are too large to represent.")
    return math.floor(value / multiple + 0.5) * multiple


def _scaled_dimension(value: int, scale: float, multiple: int) -> int:
    try:
        return _nearest_multiple(value * scale, multiple)
    except OverflowError as error:
        raise SystemExit("The requested HiRes dimensions are too large to represent.") from error


def _proportional_dimension(target: int, original_axis: int, other_axis: int, multiple: int) -> int:
    # Integer arithmetic makes exact halfway rounding predictable even for large dimensions.
    numerator = target * other_axis
    denominator = original_axis * multiple
    return ((2 * numerator + denominator) // (2 * denominator)) * multiple


def resolve_hires_stage_sizes(width: int, height: int, passes: int, *, scale: float | None = None,
                              target_width: int | None = None, target_height: int | None = None,
                              dimension_multiple: int = 8) -> list[list[int]]:
    """Scale each preceding rounded size, using explicit first targets as ratios.

    The only size bound is Pillow's signed C-int dimension representation. There
    is no arbitrary pass-count cap; every stage must make a representable enlargement.
    """
    maximum = 2**31 - 1
    try:
        for name, value in (("width", width), ("height", height), ("hires_passes", passes),
                            ("dimension_multiple", dimension_multiple)):
            _integer(value, name, positive=True)
        if max(width, height) > maximum:
            raise SystemExit("The requested HiRes dimensions are too large to represent as signed 32-bit image sizes.")
        explicit = target_width is not None or target_height is not None
        if explicit and scale is not None:
            raise SystemExit("--hires-scale cannot be combined with --hires-width or --hires-height.")
        for name, value in (("hires_width", target_width), ("hires_height", target_height)):
            if value is not None:
                _integer(value, name, positive=True)
                if value % dimension_multiple:
                    raise SystemExit(f"{_option(name)} must be a positive multiple of {dimension_multiple}.")
        if explicit:
            first_width = target_width if target_width is not None else _proportional_dimension(target_height, height, width, dimension_multiple)
            first_height = target_height if target_height is not None else _proportional_dimension(target_width, width, height, dimension_multiple)
        else:
            scale = 2.0 if scale is None else scale
            _finite_number(scale, "hires_scale", lower=1, inclusive=False)
            first_width = _scaled_dimension(width, scale, dimension_multiple)
            first_height = _scaled_dimension(height, scale, dimension_multiple)
        stages = []
        previous_width, previous_height = width, height
        for index in range(passes):
            if index == 0:
                next_width, next_height = first_width, first_height
            elif explicit:
                next_width = _proportional_dimension(previous_width, width, first_width, dimension_multiple)
                next_height = _proportional_dimension(previous_height, height, first_height, dimension_multiple)
            else:
                next_width = _scaled_dimension(previous_width, scale, dimension_multiple)
                next_height = _scaled_dimension(previous_height, scale, dimension_multiple)
            if max(next_width, next_height) > maximum:
                raise SystemExit("The requested HiRes dimensions are too large to represent as signed 32-bit image sizes.")
            if (next_width < previous_width or next_height < previous_height
                    or (next_width == previous_width and next_height == previous_height)):
                raise SystemExit("HiRes dimensions must retain or enlarge both axes and enlarge at least one axis.")
            # Growth never shrinks under a fixed positive multiplier. This lower
            # bound rejects absurd counts before constructing a huge stage list.
            if index == 0 and (width + (next_width - width) * passes > maximum
                               or height + (next_height - height) * passes > maximum):
                raise SystemExit("Repeated HiRes dimensions are too large to represent as signed 32-bit image sizes.")
            stages.append([next_width, next_height])
            previous_width, previous_height = next_width, next_height
        return stages
    except (SystemExit, OverflowError) as error:
        raise ValueError(str(error)) from error


def resolve_hires_options(preset: PipelinePreset, args: argparse.Namespace) -> None:
    """Validate options and record target dimensions without changing their replayable inputs."""
    args.hires_fix = getattr(args, "hires_fix", False)
    for name in HIRES_OPTION_NAMES:
        if not hasattr(args, name):
            setattr(args, name, None)
    args.hires_target_width = None
    args.hires_target_height = None
    args.hires_stage_sizes = None
    if not args.hires_fix:
        provided = [name for name in HIRES_OPTION_NAMES if getattr(args, name) is not None]
        if provided:
            raise SystemExit(f"{', '.join(_option(name) for name in provided)} require --hires-fix.")
        return

    if args.hires_passes is None:
        args.hires_passes = 1
    explicit_size = args.hires_width is not None or args.hires_height is not None
    if not explicit_size and args.hires_scale is None:
        args.hires_scale = 2.0
    try:
        args.hires_stage_sizes = resolve_hires_stage_sizes(
            args.width, args.height, args.hires_passes, scale=args.hires_scale,
            target_width=args.hires_width, target_height=args.hires_height,
            dimension_multiple=preset.runtime.dimension_multiple)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.hires_target_width, args.hires_target_height = args.hires_stage_sizes[-1]

    defaults = {
        "hires_upscaler": "lanczos", "hires_denoising_strength": 0.35,
        "hires_steps": args.steps, "hires_seed": args.seed,
        "hires_guidance_scale": args.guidance_scale, "hires_true_cfg_scale": args.true_cfg_scale,
        "hires_scheduler": "auto", "hires_scheduler_config": {}, "hires_save_base": False,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.hires_upscaler not in HIRES_UPSCALERS:
        raise SystemExit(f"--hires-upscaler must be one of {', '.join(HIRES_UPSCALERS)}.")
    _finite_number(args.hires_denoising_strength, "hires_denoising_strength", lower=0, inclusive=False, upper=1)
    _integer(args.hires_steps, "hires_steps", positive=True)
    try:
        initial_steps = min(args.hires_steps * args.hires_denoising_strength, args.hires_steps)
        if preset.family == "flux1-schnell":
            # Match FluxImg2ImgPipeline.get_timesteps: FLUX rounds its starting
            # offset, while SD/SDXL round the number of retained steps instead.
            effective_steps = args.hires_steps - int(max(args.hires_steps - initial_steps, 0))
        else:
            effective_steps = int(initial_steps)
    except (OverflowError, ValueError) as error:
        raise SystemExit("--hires-steps is too large to represent a diffusion schedule.") from error
    if effective_steps < 1:
        raise SystemExit("HiRes Fix needs at least one denoising step; increase --hires-steps or --hires-strength.")
    _integer(args.hires_seed, "hires_seed")
    seeds = (args.hires_seed, args.hires_seed + (args.num_images - 1) * args.seed_stride)
    if any(seed < -(2**63) or seed >= 2**64 for seed in seeds):
        raise SystemExit("Every HiRes image seed must be in [-2^63, 2^64-1]; check --hires-seed and --seed-stride.")
    _finite_number(args.hires_guidance_scale, "hires_guidance_scale", lower=0)
    _finite_number(args.hires_true_cfg_scale, "hires_true_cfg_scale", lower=1)
    if preset.family == "flux1-schnell" and not preset.requires_guidance_embeds and args.hires_guidance_scale != 0:
        raise SystemExit("FLUX schnell requires --hires-guidance-scale 0.")
    if preset.family != "flux1-schnell" and args.hires_true_cfg_scale != 1:
        raise SystemExit("--hires-true-cfg-scale values other than 1 require FLUX.")
    if (not isinstance(args.hires_scheduler, str) or not args.hires_scheduler
            or args.hires_scheduler.startswith("_")):
        raise SystemExit("--hires-scheduler must be auto or a compatible Diffusers scheduler class.")
    try:
        if not isinstance(args.hires_scheduler_config, dict):
            raise ValueError("Expected a JSON object.")
        json_object(json.dumps(args.hires_scheduler_config, allow_nan=False))
    except (TypeError, ValueError, argparse.ArgumentTypeError, RecursionError) as error:
        raise SystemExit(f"--hires-scheduler-config must be a finite JSON object: {error}") from error
    if not isinstance(args.hires_save_base, bool):
        raise SystemExit("--hires-save-base must be a boolean.")


def hires_request(preset: PipelinePreset, args: argparse.Namespace, pass_index: int = 0) -> argparse.Namespace:
    """Copy shared settings for the indexed refinement, keeping the base request unchanged."""
    if not getattr(args, "hires_fix", False):
        raise ValueError("A second-pass request requires --hires-fix.")
    if getattr(args, "hires_target_width", None) is None or getattr(args, "hires_target_height", None) is None:
        raise ValueError("Resolve --hires-fix options before constructing the second-pass request.")
    sizes = getattr(args, "hires_stage_sizes", None)
    if (not isinstance(sizes, list) or isinstance(pass_index, bool) or not isinstance(pass_index, int)
            or not 0 <= pass_index < len(sizes)):
        raise ValueError("pass_index must identify a resolved HiRes stage.")
    request = argparse.Namespace(**vars(args))
    request.hires_stage_sizes = deepcopy(sizes)
    request.width, request.height = sizes[pass_index]
    request.steps, request.seed = args.hires_steps, args.hires_seed
    request.guidance_scale, request.true_cfg_scale = args.hires_guidance_scale, args.hires_true_cfg_scale
    request.scheduler = args.hires_scheduler
    request.scheduler_config = deepcopy(args.hires_scheduler_config)
    request.timesteps = None
    request.sigmas = None
    request.denoising_end = None
    request.latents_file = None
    request.tensor_inputs = None
    request.guidance_rescale = 0.0
    if preset.family == "sdxl-base":
        for name in SDXL_SIZES:
            setattr(request, name, (request.height, request.width))
    return request
