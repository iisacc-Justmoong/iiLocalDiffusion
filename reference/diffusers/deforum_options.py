"""Omission-safe animation inputs shared by the CLI, JSON and Python API."""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
from pathlib import Path
from typing import Any

from deforum_schedules import NumericSchedule, PromptSchedule
from generation_config import json_object


SCHEDULE_DEFAULTS = {"zoom": "0:(1)", "angle": "0:(0)", "translation_x": "0:(0)",
                     "translation_y": "0:(0)", "strength_schedule": "0:(0.35)",
                     "noise_schedule": "0:(0)", "contrast_schedule": "0:(1)"}
PROMPTS = {"animation_prompts": "prompt", "animation_negative_prompts": "negative_prompt",
           "animation_prompts_2": "prompt_2", "animation_negative_prompts_2": "negative_prompt_2"}
DEFAULTS = {"seed_behavior": "fixed", "border": "replicate", "color_coherence": "none"}
OPTION_NAMES = (*SCHEDULE_DEFAULTS, *PROMPTS, *DEFAULTS, "cfg_scale_schedule", "init_image")


def add_deforum_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Deforum 2D animation")
    group.add_argument("--init-image", type=Path, default=None, help="Optional local first-frame image")
    for name in PROMPTS:
        group.add_argument("--" + name.replace("_", "-"), type=json_object, default=None,
                           help="JSON frame-to-text keyframes, including frame 0; held until the next key")
    for name in (*SCHEDULE_DEFAULTS, "cfg_scale_schedule"):
        group.add_argument("--" + name.replace("_", "-"), default=None,
                           help="Numeric keyframes, e.g. '0:(1), 60:(1.02)'; arithmetic may use t, fps, max_f")
    group.add_argument("--seed-behavior", choices=("fixed", "iter", "random"), default=None,
                       help="Per-frame seed policy; iter uses --seed-stride; random is reproducible")
    group.add_argument("--border", choices=("replicate", "reflect", "wrap"), default=None)
    group.add_argument("--color-coherence", choices=("none", "RGB"), default=None,
                       help="RGB matches the first frame's channel histograms before diffusion")


def resolve_deforum_options(preset: Any, args: Any) -> None:
    if args.animation_mode != "2D":
        if any(getattr(args, name) is not None for name in OPTION_NAMES):
            raise ValueError("Animation options require --animation-mode 2D or --backend deforum.")
        return
    for name, default in {**DEFAULTS, **SCHEDULE_DEFAULTS}.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.guidance_rescale != 0 and (preset.family == "sd15" or args.controlnet_selection is not None):
        raise ValueError("This Deforum image-to-image pipeline does not support --guidance-rescale.")
    for name in ("latents", "embeddings", "timesteps", "sigmas", "denoising_end"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"Deforum does not support --{name.replace('_', '-')}.")
    if args.cfg_scale_schedule is None:
        args.cfg_scale_schedule = f"0:({args.guidance_scale})"
    args.deforum_schedules = {
        name: NumericSchedule(getattr(args, name), max_frames=args.max_frames, fps=args.fps)
        for name in (*SCHEDULE_DEFAULTS, "cfg_scale_schedule")}
    args.deforum_prompts = {}
    for name, field in PROMPTS.items():
        values = getattr(args, name)
        if field.endswith("_2") and preset.family == "sd15":
            if values is not None:
                raise ValueError("Secondary animation prompts require SDXL or FLUX.")
            continue
        if values is None:
            if field.endswith("_2") and field not in args._provided:
                values = dict(getattr(args, name.removesuffix("_2")))
            else:
                values = {"0": getattr(args, field)}
            setattr(args, name, values)
        args.deforum_prompts[field] = PromptSchedule(values, max_frames=args.max_frames)
    if preset.family == "flux1-schnell" and args.true_cfg_scale == 1:
        if any(value for field in ("negative_prompt", "negative_prompt_2")
               for value in args.deforum_prompts[field].keyframes.values()):
            raise ValueError("FLUX animation negative prompts require --true-cfg-scale greater than 1.")
    if args.seed_behavior == "iter":
        last = args.seed + (args.max_frames - 1) * args.seed_stride
        if not -(2**63) <= last < 2**64:
            raise ValueError("Animation seeds exceed [-2^63, 2^64-1]; check seed and seed stride.")
    # Validate the entire timeline before importing ML or loading weights, including
    # expressions that become invalid only on a later frame.
    for frame in range(args.max_frames):
        values = {name: schedule.at(frame) for name, schedule in args.deforum_schedules.items()}
        strength = values["strength_schedule"]
        if not 0 <= strength <= 1:
            raise ValueError(f"strength_schedule must be in [0,1] at frame {frame}.")
        if (frame > 0 or args.init_image is not None) and strength > 0 and int(args.steps * strength) < 1:
            raise ValueError(f"strength_schedule produces zero denoising steps at frame {frame}; increase --steps or strength.")
        if not 0 < values["zoom"] <= 100:
            raise ValueError(f"zoom must be in (0,100] at frame {frame}.")
        if not 0 <= values["noise_schedule"] <= 1:
            raise ValueError(f"noise_schedule must be in [0,1] at frame {frame}.")
        if values["contrast_schedule"] < 0 or values["cfg_scale_schedule"] < 0:
            raise ValueError(f"Contrast and CFG schedules must be non-negative at frame {frame}.")
        if (preset.family == "flux1-schnell" and not preset.requires_guidance_embeds
                and values["cfg_scale_schedule"] != 0):
            raise ValueError("FLUX.1-schnell cfg_scale_schedule must remain zero.")
    if args.init_image is not None:
        args.init_image = args.init_image.expanduser().resolve()
        if not args.init_image.is_file() or args.init_image.stat().st_size == 0:
            raise ValueError("--init-image must be a non-empty local image file.")


def frame_request(preset: Any, args: Any, index: int) -> argparse.Namespace:
    request = copy(args)
    request.frame_index = index
    for field, schedule in args.deforum_prompts.items():
        setattr(request, field, schedule.at(index))
    for name, schedule in args.deforum_schedules.items():
        setattr(request, name, schedule.at(index))
    request.guidance_scale = request.cfg_scale_schedule
    request.denoising_strength = request.strength_schedule
    # The existing FLUX ControlNet img2img compatibility path consumes this field.
    request.hires_denoising_strength = request.denoising_strength
    if args.seed_behavior == "iter":
        request.seed = args.seed + index * args.seed_stride
    elif args.seed_behavior == "random":
        identity = f"iild-deforum:{args.seed}:{index}".encode("ascii")
        request.seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return request
