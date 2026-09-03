"""Pixel upscaling followed by a verified Diffusers image-to-image refinement."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from typing import Any, Callable

from cpu_conditioning import encode_cpu_prompt
from encoder_compatibility import clip_skip_compatibility
from generation_options import build_generators
from generation_scheduler import configure_scheduler, validate_scheduler_values
from generation_tensor_inputs import load_tensor_inputs
from hardware import validate_execution_device
from hires_options import hires_request
from text_embeddings import text_embedding_prompt_context, validate_text_embeddings


PIPELINES = {
    ("sd15", False): "StableDiffusionImg2ImgPipeline",
    ("sd15", True): "StableDiffusionControlNetImg2ImgPipeline",
    ("sdxl-base", False): "StableDiffusionXLImg2ImgPipeline",
    ("sdxl-base", True): "StableDiffusionXLControlNetImg2ImgPipeline",
    ("flux1-schnell", False): "FluxImg2ImgPipeline",
    # The img2img class lacks the negative conditioning supported by our public
    # FLUX ControlNet path. The compatibility helper retains its denoising loop.
    ("flux1-schnell", True): "FluxControlNetPipeline",
}


class DenoisingAudit:
    """Observe actual sampling and reject non-finite latents before decoding."""

    def __init__(self, torch: Any):
        self.torch = torch
        self.timesteps: list[float] = []

    def __call__(self, pipeline: Any, step: int, timestep: Any, values: dict[str, Any]):
        latents = values.get("latents")
        if latents is None or not self.torch.isfinite(latents).all().item():
            raise RuntimeError(f"Non-finite or missing denoising latents at step {step}.")
        self.timesteps.append(float(timestep))
        return values

    def metadata(self) -> dict[str, Any]:
        if not self.timesteps:
            raise RuntimeError("HiRes Fix requires an actual non-empty denoising pass.")
        return {"executed_steps": len(self.timesteps), "executed_timesteps": self.timesteps,
                "finite_latents": True}


def image_metadata(image: Any) -> dict[str, Any]:
    return {"size": list(image.size), "mode": image.mode,
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "channel_extrema": image.getextrema()}


def validate_stage_images(result: Any, count: int, width: int, height: int) -> list[Any]:
    if len(result.images) != count:
        raise RuntimeError(f"Expected {count} stage images, received {len(result.images)}.")
    images = []
    for image in result.images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (width, height):
            raise RuntimeError(f"Unexpected stage image size: {image.size}; expected {(width, height)}.")
        extrema = image.getextrema()
        if len(extrema) != 3 or all(low == high for low, high in extrema):
            raise RuntimeError(f"Stage image is uniform or invalid: {extrema}")
        images.append(image)
    return images


def upscale_images(images: list[Any], width: int, height: int, method: str) -> list[Any]:
    from PIL import Image

    filters = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
               "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
    if method not in filters:
        raise ValueError(f"Unsupported HiRes upscaler: {method}")
    return [image.resize((width, height), resample=filters[method]) for image in images]


def refinement_call_arguments(preset: Any, request: Any, arguments: dict[str, Any],
                              images: list[Any]) -> dict[str, Any]:
    """Separate the generated source images from optional ControlNet guidance."""
    arguments = dict(arguments)
    for name in ("latents", "timesteps", "sigmas", "denoising_end", "guidance_rescale"):
        arguments.pop(name, None)
    has_controlnet = request.controlnet_selection is not None
    if has_controlnet and preset.name != "flux1-schnell":
        arguments["control_image"] = arguments.pop("image")
    if not has_controlnet and preset.name in ("sd15", "sdxl-base"):
        arguments.pop("width", None)
        arguments.pop("height", None)
    if not (has_controlnet and preset.name == "flux1-schnell"):
        arguments["image"] = images
        arguments["strength"] = request.hires_denoising_strength
    return arguments


def _validate_preserved_adapters(pipeline: Any, activation: Any) -> None:
    if activation is None:
        return
    for name in activation.registered_components:
        component = getattr(pipeline, name)
        if not set(activation.active_adapters).issubset(component.active_adapters()):
            raise RuntimeError(f"HiRes refinement lost active LoRA adapters on {name}.")


@dataclass
class HiresResult:
    pipeline: Any
    result: Any
    request: Any
    metadata: dict[str, Any]
    optimization: dict[str, Any]
    scheduler_metadata: dict[str, Any]
    conditioning: Any
    seeds: list[int]


def run_hires_fix(
    pipeline: Any, preset: Any, args: Any, images: list[Any], torch: Any,
    device: str, dtype: Any, attention_slicing: bool, activation: Any,
    *, build_call: Callable, prepare_execution: Callable,
    pipeline_classes: dict[str, Any] | None = None,
) -> HiresResult:
    request = hires_request(preset, args)
    upscaled = upscale_images(images, request.width, request.height, args.hires_upscaler)
    class_name = PIPELINES[(preset.name, args.controlnet_selection is not None)]
    if pipeline_classes is None:
        import diffusers
        selected_class = getattr(diffusers, class_name)
    else:
        selected_class = pipeline_classes[class_name]

    # Sequential offload may leave parameters on meta. Its public removal API
    # first restores their stored weights; only then is moving/conversion safe.
    pipeline.remove_all_hooks()
    pipeline.to("cpu")
    scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
    overrides = {"scheduler": scheduler, "dtype": dtype}
    if preset.name == "sdxl-base":
        overrides["add_watermarker"] = bool(args.watermark)
    refined = selected_class.from_pipe(pipeline, **overrides)
    _validate_preserved_adapters(refined, activation)
    validate_text_embeddings(refined, getattr(args, "text_embedding_activation", None), torch)
    scheduler_metadata = configure_scheduler(refined, request)
    validate_scheduler_values(refined, request)

    conditioning = None
    if request.embeddings_file is not None:
        # Stage-two CFG can require negative embeddings even when stage one
        # did not. Reload the same verified file against the second request.
        request.tensor_inputs = load_tensor_inputs(refined, preset, request, torch, dtype)
        conditioning = request.tensor_inputs.conditioning
    if request.cpu_text_encoding:
        with text_embedding_prompt_context(refined, preset, request) as prompt_args:
            conditioning = encode_cpu_prompt(refined, preset, prompt_args, torch)
    offload = request.offload
    if (offload == "auto" and request.cpu_text_encoding and device != "cpu"
            and preset.runtime.accelerator_execution == "resident"):
        offload = "model"
    refined, optimization = prepare_execution(
        refined, preset, device, attention_slicing, offload=offload,
        vae_slicing=request.vae_slicing, vae_tiling=request.vae_tiling,
        attention_slice_size=request.attention_slice_size, device_index=request.device_index)
    optimization["requested_offload"] = request.offload
    validate_execution_device(refined, device)
    if device == "cuda" and refined._execution_device.index != request.device_index:
        raise RuntimeError("HiRes refinement did not retain the requested GPU device index.")
    refined.set_progress_bar_config(disable=not args.progress)
    generator, seeds, generator_device = build_generators(torch, request, device)
    audit = DenoisingAudit(torch)
    clip_context = (clip_skip_compatibility(refined, preset, request.clip_skip)
                    if conditioning is None else nullcontext(False))
    compatibility = None
    with (torch.inference_mode(), clip_context as clip_compatibility,
          text_embedding_prompt_context(refined, preset, request) as prompt_args):
        call = refinement_call_arguments(
            preset, request, build_call(preset, prompt_args, generator, conditioning, device, dtype), upscaled)
        call["callback_on_step_end"] = audit
        if preset.name == "flux1-schnell" and request.controlnet_selection is not None:
            from hires_flux_controlnet import refine_flux_controlnet
            result, compatibility = refine_flux_controlnet(
                refined, request, upscaled, call, torch, generator, device, dtype)
        else:
            result = refined(**call)
    # Validate every output before the caller publishes any final/base files.
    result.images = validate_stage_images(result, request.num_images, request.width, request.height)
    metadata = {
        "enabled": True,
        "upscale": {"method": args.hires_upscaler, "requested_scale": args.hires_scale,
                    "source_size": [args.width, args.height],
                    "target_size": [request.width, request.height],
                    "actual_scale": [request.width / args.width, request.height / args.height],
                    "images": [image_metadata(image) for image in upscaled]},
        "refinement": {
            "pipeline_class": type(refined).__name__,
            "requested_steps": request.steps, "denoising_strength": args.hires_denoising_strength,
            "guidance_scale": request.guidance_scale, "true_cfg_scale": request.true_cfg_scale,
            "seeds": seeds, "generator_device": generator_device,
            "scheduler": scheduler_metadata, **audit.metadata(),
            "schedule": refined.scheduler.timesteps.detach().cpu().tolist(),
            "optimization": optimization,
            "cpu_conditioning": {"enabled": False} if conditioning is None else conditioning.metadata,
            "clip_skip_layout_compatibility": clip_compatibility,
            "compatibility": compatibility,
        },
    }
    return HiresResult(refined, result, request, metadata, optimization,
                       scheduler_metadata, conditioning, seeds)
