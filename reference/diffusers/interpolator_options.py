"""Two-endpoint prompt and noise interpolation request contract."""

from copy import copy
from typing import Any

PROMPTS = ("prompt", "negative_prompt", "prompt_2", "negative_prompt_2")
OPTION_NAMES = (*(f"end_{name}" for name in PROMPTS), "end_seed")


def add_interpolator_options(parser) -> None:
    group = parser.add_argument_group("Interpolator animation")
    for name in PROMPTS:
        group.add_argument("--end-" + name.replace("_", "-"), default=None,
                           help="Text at the final frame; embeddings interpolate from the starting text")
    group.add_argument("--end-seed", type=int, default=None,
                       help="Final noise seed; defaults to --seed (prompt-only interpolation)")


def resolve_interpolator_options(preset: Any, args: Any) -> None:
    if args.animation_mode != "Interpolator":
        if any(getattr(args, name) is not None for name in OPTION_NAMES):
            raise ValueError("End prompt/seed options require --animation-mode Interpolator or --backend interpolator.")
        return
    if args.max_frames < 2:
        raise ValueError("Interpolator requires --max-frames >= 2, including both endpoints.")
    if "cpu_text_encoding" in args._provided and not args.cpu_text_encoding:
        raise ValueError("Interpolator encodes both endpoints once on CPU before GPU/offload hooks; "
                         "--no-cpu-text-encoding is unsupported.")
    args.cpu_text_encoding = True
    for name in ("latents", "embeddings", "denoising_end"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"Interpolator does not support --{name.replace('_', '-')}.")
    if args.end_seed is None:
        args.end_seed = args.seed
    if not -(2**63) <= args.end_seed < 2**64:
        raise ValueError("--end-seed must be in [-2^63, 2^64-1].")
    for field in PROMPTS:
        name = "end_" + field
        value = getattr(args, name)
        if field.endswith("_2") and preset.family == "sd15":
            if value is not None:
                raise ValueError("Secondary end prompts require SDXL or FLUX.")
            continue
        if value is None:
            if field.endswith("_2") and field not in args._provided:
                value = getattr(args, name.removesuffix("_2"))
            else:
                value = getattr(args, field)
            setattr(args, name, value)
    if preset.family == "flux1-schnell" and args.true_cfg_scale == 1:
        if args.end_negative_prompt or args.end_negative_prompt_2:
            raise ValueError("FLUX Interpolator negative prompts require --true-cfg-scale greater than 1.")
    if args.controlnet_selection is not None and args.guidance_rescale != 0:
        raise ValueError("Interpolator ControlNet does not support --guidance-rescale.")


def endpoint_request(args: Any) -> Any:
    request = copy(args)
    for field in PROMPTS:
        setattr(request, field, getattr(args, "end_" + field))
    request.seed = args.end_seed
    return request
