"""Family-aware generation values; defaults are policy, not guessed model structure."""

from __future__ import annotations

import argparse
import math
import re
from typing import Any

from generation_config import json_object
from presets import PipelinePreset


DTYPES = ("auto", "float32", "float16", "bfloat16")
SDXL_SIZES = ("original_size", "target_size", "negative_original_size", "negative_target_size")
SDXL_CROPS = ("crops_coords_top_left", "negative_crops_coords_top_left")


def add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-images", "--num-images-per-prompt", type=int, default=1,
                        help="Images for this prompt; every image is saved (default: 1)")
    parser.add_argument("--seed-stride", type=int, default=1,
                        help="Per-image seed increment; 0 reuses the same seed (default: 1)")
    parser.add_argument("--generator-device", choices=("cpu", "execution"), default="cpu",
                        help="Noise generator device; execution requires CUDA/ROCm (default: cpu)")
    parser.add_argument("--dtype", choices=DTYPES, default="auto",
                        help="Model compute/storage dtype; auto uses the preset's CPU/GPU policy")
    parser.add_argument("--cpu-text-dtype", choices=DTYPES, default="auto",
                        help="Opt-in CPU prompt encoder dtype; auto uses the preset's CPU policy")
    parser.add_argument("--device-index", type=int, default=0,
                        help="CUDA/ROCm device index; CPU and Metal require 0")
    parser.add_argument("--weight-variant", default="auto",
                        help="Weight filename variant: auto, none, or an explicit variant such as fp16")
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True,
                        help="Memory-efficient model/VAE/LoRA loading (default: enabled)")
    parser.add_argument("--prompt-2", default=None, help="SDXL/FLUX second encoder text; defaults to --prompt")
    parser.add_argument("--negative-prompt-2", default=None,
                        help="SDXL/FLUX second negative text; defaults to --negative-prompt")
    parser.add_argument("--guidance-rescale", type=float, default=0.0,
                        help="SD/SDXL CFG rescale in [0,1]; 0 disables extra rescaling")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="SD/SDXL DDIM stochasticity; 0 is deterministic")
    parser.add_argument("--clip-skip", type=int, default=None,
                        help="SD/SDXL CLIP layer skip; omitted keeps the pipeline's native selection")
    parser.add_argument("--max-sequence-length", type=int, default=None,
                        help="FLUX T5 token limit in [1,512]; schnell defaults to 256")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0,
                        help="FLUX two-pass CFG; 1 disables it, >1 enables negative prompts")
    parser.add_argument("--scheduler", default="auto",
                        help="auto retains the model scheduler; otherwise a compatible Diffusers class name")
    parser.add_argument("--scheduler-config", type=json_object, default={},
                        help='Scheduler constructor overrides as JSON, e.g. {"timestep_spacing":"trailing"}')
    parser.add_argument("--timesteps", nargs="+", type=int, default=None,
                        help="Descending SD/SDXL custom timesteps, when the scheduler supports them")
    parser.add_argument("--sigmas", nargs="+", type=float, default=None,
                        help="Descending custom noise sigmas; cannot be combined with --timesteps")
    parser.add_argument("--denoising-end", type=float, default=None,
                        help="SDXL early denoising stop in (0,1); omitted runs the full schedule")
    for name in SDXL_SIZES:
        parser.add_argument("--" + name.replace("_", "-"), nargs=2, type=int, default=None,
                            metavar=("HEIGHT", "WIDTH"), help="SDXL size conditioning; defaults to output dimensions")
    for name in SDXL_CROPS:
        parser.add_argument("--" + name.replace("_", "-"), nargs=2, type=int, default=None,
                            metavar=("TOP", "LEFT"), help="SDXL crop conditioning; defaults to 0 0")
    parser.add_argument("--vae-slicing", action=argparse.BooleanOptionalAction, default=None,
                        help="VAE batch slicing; omitted preserves the preset/device policy")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=None,
                        help="VAE spatial tiling; omitted preserves the preset/device policy")
    parser.add_argument("--attention-slice-size", default="auto",
                        help="Attention slice size: auto, max, or a positive integer")
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=None,
                        help="SDXL invisible watermark; defaults to disabled, requires its optional dependency")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Display generation progress (default: enabled)")
    parser.add_argument("--png-compress-level", type=int, default=6,
                        help="Lossless PNG compression in [0,9] (default: 6)")
    parser.add_argument("--png-optimize", action=argparse.BooleanOptionalAction, default=False,
                        help="Optimize PNG storage without changing pixels (default: disabled)")
    parser.add_argument("--latents", default=None,
                        help="Optional local safetensors initial noise; omitted uses the seeded generator")
    parser.add_argument("--latents-key", default="latents", help="Tensor key in --latents (default: latents)")
    parser.add_argument("--embeddings", default=None,
                        help="Optional local safetensors prompt/pooled/negative tensors; omitted encodes the text")
    parser.add_argument("--embedding-keys", type=json_object, default={},
                        help="JSON map from standard embedding argument names to file tensor keys")
    parser.add_argument("--cross-attention-kwargs", type=json_object, default={},
                        help="SD/SDXL portable attention options as JSON; supports call-time LoRA scale")
    parser.add_argument("--joint-attention-kwargs", type=json_object, default={},
                        help="FLUX portable attention options as JSON; supports call-time LoRA scale")


def resolve_generation_options(preset: PipelinePreset, args: argparse.Namespace) -> None:
    if preset.name in ("sdxl-base", "flux1-schnell"):
        if args.prompt_2 is None:
            args.prompt_2 = args.prompt
        if args.negative_prompt_2 is None:
            args.negative_prompt_2 = args.negative_prompt
    if args.max_sequence_length is None:
        args.max_sequence_length = preset.runtime.max_sequence_length
    if preset.name == "sdxl-base":
        for name in SDXL_SIZES:
            value = getattr(args, name)
            setattr(args, name, (args.height, args.width) if value is None else tuple(value))
        for name in SDXL_CROPS:
            value = getattr(args, name)
            setattr(args, name, (0, 0) if value is None else tuple(value))
        if args.watermark is None:
            args.watermark = False
    if args.attention_slice_size not in ("auto", "max"):
        try:
            args.attention_slice_size = int(args.attention_slice_size)
        except (ValueError, TypeError) as error:
            raise SystemExit("--attention-slice-size must be auto, max, or a positive integer.") from error


def with_generation_defaults(preset: PipelinePreset, args: Any) -> argparse.Namespace:
    """Keep older programmatic callers valid when new optional values are added."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_generation_options(parser)
    resolved = parser.parse_args([], namespace=argparse.Namespace(**vars(args)))
    resolve_generation_options(preset, resolved)
    return resolved


def uses_negative_prompt(preset: PipelinePreset, args: Any) -> bool:
    return preset.runtime.passes_negative_prompt or (
        preset.name == "flux1-schnell" and getattr(args, "true_cfg_scale", 1.0) > 1.0
    )


def validate_generation_options(preset: PipelinePreset, args: argparse.Namespace) -> None:
    for name, lower, upper in (
        ("guidance_scale", 0, None), ("eta", 0, None), ("guidance_rescale", 0, 1),
        ("true_cfg_scale", 1, None),
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and in [{lower},{upper or 'infinity'}].")
    if args.num_images <= 0:
        raise SystemExit("--num-images must be positive.")
    seeds = (args.seed, args.seed + (args.num_images - 1) * args.seed_stride)
    if any(seed < -(2**63) or seed >= 2**64 for seed in seeds):
        raise SystemExit("Every image seed must be in [-2^63, 2^64-1]; check --seed and --seed-stride.")
    if args.device_index < 0 or (args.device in ("cpu", "metal", "mps") and args.device_index != 0):
        raise SystemExit("--device-index must be non-negative; CPU and Metal require index 0.")
    if args.generator_device == "execution" and args.device in ("cpu", "metal", "mps"):
        raise SystemExit("--generator-device execution requires CUDA/ROCm.")
    if args.cpu_text_dtype != "auto" and not args.cpu_text_encoding:
        raise SystemExit("--cpu-text-dtype requires --cpu-text-encoding.")
    if args.clip_skip is not None and args.clip_skip < 0:
        raise SystemExit("--clip-skip must be non-negative.")
    if not 0 <= args.png_compress_level <= 9:
        raise SystemExit("--png-compress-level must be in [0,9].")
    if isinstance(args.attention_slice_size, int) and args.attention_slice_size <= 0:
        raise SystemExit("--attention-slice-size must be positive.")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.weight_variant) is None:
        raise SystemExit("--weight-variant must be auto, none, or a filename variant without directories.")
    if not args.scheduler or args.scheduler.startswith("_"):
        raise SystemExit("--scheduler must be auto or a compatible Diffusers scheduler class.")
    if not args.latents_key or (args.latents is None and args.latents_key != "latents"):
        raise SystemExit("--latents-key must be non-empty and requires --latents.")
    allowed_embeddings = {"prompt_embeds", "negative_prompt_embeds"}
    if preset.name != "sd15":
        allowed_embeddings.update(("pooled_prompt_embeds", "negative_pooled_prompt_embeds"))
    if any(name not in allowed_embeddings or not isinstance(value, str) or not value
           for name, value in args.embedding_keys.items()):
        raise SystemExit("--embedding-keys must map supported embedding names to non-empty tensor keys.")
    if args.embedding_keys and args.embeddings is None:
        raise SystemExit("--embedding-keys requires --embeddings.")
    if args.embeddings is not None and args.cpu_text_encoding:
        raise SystemExit("--embeddings cannot be combined with --cpu-text-encoding.")
    if args.embeddings is not None and args.clip_skip is not None:
        raise SystemExit("--clip-skip applies to text encoding, not supplied --embeddings.")
    for name in ("cross_attention_kwargs", "joint_attention_kwargs"):
        options = getattr(args, name)
        is_flux = preset.name == "flux1-schnell"
        if options and (name == "joint_attention_kwargs") != is_flux:
            raise SystemExit(f"--{name.replace('_', '-')} does not apply to this preset.")
        if options.keys() - {"scale"}:
            raise SystemExit("The supported attention processors expose only the portable LoRA scale value.")
        if "scale" in options:
            scale = options["scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise SystemExit("Attention LoRA scale must be a finite number.")
            if args.lora_selection is None:
                raise SystemExit("Attention LoRA scale requires --lora; it is not a base-model multiplier.")
    if args.timesteps is not None and args.sigmas is not None:
        raise SystemExit("--timesteps and --sigmas are mutually exclusive.")
    for name in ("timesteps", "sigmas"):
        values = getattr(args, name)
        if values is not None and (
            not values or any(not math.isfinite(value) or value < 0 for value in values)
            or any(left <= right for left, right in zip(values, values[1:]))
        ):
            raise SystemExit(f"--{name} must be a non-empty, finite, non-negative, strictly descending list.")
    if args.sigmas is not None and args.sigmas[0] == 0:
        raise SystemExit("--sigmas must start above zero.")
    if preset.name == "sd15":
        if args.prompt_2 is not None or args.negative_prompt_2 is not None:
            raise SystemExit("--prompt-2 and --negative-prompt-2 require SDXL or FLUX.")
    if preset.name != "sdxl-base":
        if any(getattr(args, name) is not None for name in (*SDXL_SIZES, *SDXL_CROPS, "denoising_end")):
            raise SystemExit("Size/crop conditioning and --denoising-end require SDXL.")
        if args.watermark is not None:
            raise SystemExit("--watermark requires SDXL.")
    else:
        for name in SDXL_SIZES:
            if any(value <= 0 for value in getattr(args, name)):
                raise SystemExit(f"--{name.replace('_', '-')} dimensions must be positive.")
        for name in SDXL_CROPS:
            if any(value < 0 for value in getattr(args, name)):
                raise SystemExit(f"--{name.replace('_', '-')} coordinates must be non-negative.")
        if args.denoising_end is not None and (
            not math.isfinite(args.denoising_end) or not 0 < args.denoising_end < 1
        ):
            raise SystemExit("--denoising-end must be finite and in (0,1).")
    if preset.name != "flux1-schnell":
        if args.max_sequence_length is not None or args.true_cfg_scale != 1.0:
            raise SystemExit("--max-sequence-length and --true-cfg-scale require FLUX.")
    else:
        if args.eta != 0 or args.guidance_rescale != 0 or args.clip_skip is not None:
            raise SystemExit("--eta, --guidance-rescale and --clip-skip require SD/SDXL.")
        if args.timesteps is not None:
            raise SystemExit("FLUX accepts --sigmas, not --timesteps.")
        if not 1 <= args.max_sequence_length <= 512:
            raise SystemExit("--max-sequence-length must be in [1,512].")
        if args.true_cfg_scale == 1 and (args.negative_prompt or args.negative_prompt_2):
            raise SystemExit("This preset does not use --negative-prompt unless --true-cfg-scale is greater than 1.")


def pipeline_generation_values(preset: PipelinePreset, args: Any) -> dict[str, Any]:
    values = {"num_images_per_prompt": args.num_images, "output_type": "pil", "return_dict": True}
    for name in ("timesteps", "sigmas"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    if preset.name in ("sdxl-base", "flux1-schnell"):
        values["prompt_2"] = args.prompt_2
    if uses_negative_prompt(preset, args):
        values["negative_prompt"] = args.negative_prompt
        if preset.name in ("sdxl-base", "flux1-schnell"):
            values["negative_prompt_2"] = args.negative_prompt_2
    if preset.name == "flux1-schnell":
        values.update(max_sequence_length=args.max_sequence_length, true_cfg_scale=args.true_cfg_scale)
    else:
        values.update(eta=args.eta, guidance_rescale=args.guidance_rescale, clip_skip=args.clip_skip)
    if preset.name == "sdxl-base":
        values.update({name: getattr(args, name) for name in (*SDXL_SIZES, *SDXL_CROPS, "denoising_end")})
    attention_name = "joint_attention_kwargs" if preset.name == "flux1-schnell" else "cross_attention_kwargs"
    if getattr(args, attention_name):
        values[attention_name] = dict(getattr(args, attention_name))
    return values


def resolved_dtype(preset: PipelinePreset, args: Any, device: str) -> str:
    if args.dtype != "auto":
        return args.dtype
    return preset.runtime.cpu_dtype if device == "cpu" else preset.runtime.accelerator_dtype


def select_device_index(torch: Any, device: str, index: int) -> None:
    if index < 0:
        raise ValueError("Device index must be non-negative.")
    if device != "cuda":
        if index != 0:
            raise ValueError("CPU and Metal only support device index 0.")
        return
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA/ROCm device index {index} is not visible.")
    torch.cuda.set_device(index)


def build_generators(torch: Any, args: Any, device: str) -> tuple[Any, list[int], str]:
    generator_device = "cpu"
    if args.generator_device == "execution":
        if device != "cuda":
            raise ValueError("--generator-device execution requires a CUDA/ROCm execution device.")
        generator_device = f"cuda:{args.device_index}"
    seeds = [args.seed + index * args.seed_stride for index in range(args.num_images)]
    generators = [torch.Generator(device=generator_device).manual_seed(seed) for seed in seeds]
    return generators[0] if len(generators) == 1 else generators, seeds, generator_device
