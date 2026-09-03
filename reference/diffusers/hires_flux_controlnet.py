"""Image-conditioned FLUX ControlNet refinement with upstream true CFG support.

Diffusers 0.40.0's ControlNet img2img call omits negative conditioning. Reuse
its public img2img latent/schedule helpers and the existing ControlNet text
pipeline's denoising loop, retaining the full transformed scheduler schedule.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
from typing import Any


def _dependencies():
    import numpy as np
    from diffusers import FluxImg2ImgPipeline
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    return FluxImg2ImgPipeline, calculate_shift, np


@contextmanager
def _selected_schedule(scheduler: Any, timesteps: Any):
    """Let the upstream call consume an already initialized img2img schedule.

    Sigmas and the solver's begin index stay untouched. Reinitializing them
    here would reset the strength offset or transform the sigmas twice.
    The pipeline instance is used sequentially, as required by Diffusers'
    mutable scheduler and execution/offload state.
    """
    absent = object()
    instance_method = vars(scheduler).get("set_timesteps", absent)
    full_timesteps = scheduler.timesteps

    def keep_schedule(num_inference_steps=None, device=None, sigmas=None,
                      mu=None, timesteps=None):
        # Explicit sigmas parameter is required by Diffusers retrieve_timesteps.
        return None

    scheduler.timesteps = timesteps
    scheduler.set_timesteps = keep_schedule
    try:
        yield
    finally:
        scheduler.timesteps = full_timesteps
        if instance_method is absent:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = instance_method


def refine_flux_controlnet(
    pipeline: Any,
    request: Any,
    images: list[Any],
    call_arguments: dict[str, Any],
    torch: Any,
    generator: Any,
    device: str,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """Refine upscaled RGB images without dropping negative prompts or CFG.

    ``pipeline`` is the prepared FluxControlNetPipeline. ``call_arguments``
    supplies the second stage's dimensions, step count, and conditioning.
    The complete new schedule is transformed once before strength selects
    its tail, exactly as in upstream FluxImg2ImgPipeline.get_timesteps.
    """
    strength = request.hires_denoising_strength
    steps = call_arguments["num_inference_steps"]
    if (not isinstance(strength, (int, float)) or isinstance(strength, bool)
            or not math.isfinite(strength) or not 0 < strength <= 1):
        raise ValueError("FLUX ControlNet refinement requires denoising strength in (0, 1].")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("FLUX ControlNet refinement requires a positive step count.")
    if not images:
        raise ValueError("FLUX ControlNet refinement requires an upscaled image batch.")

    image_class, calculate_shift, np = _dependencies()
    accepted = inspect.signature(image_class.__init__).parameters
    image_pipeline = image_class(**{
        name: component for name, component in pipeline.components.items() if name in accepted
    })
    scheduler = pipeline.scheduler
    height, width = call_arguments["height"], call_arguments["width"]
    image_seq_len = ((height // image_pipeline.vae_scale_factor // 2)
                     * (width // image_pipeline.vae_scale_factor // 2))
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    raw_sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
    scheduler.set_timesteps(sigmas=raw_sigmas, device=device, mu=mu)
    timesteps, effective_steps = image_pipeline.get_timesteps(steps, strength, device)
    if effective_steps < 1 or len(timesteps) < 1:
        raise ValueError("FLUX ControlNet refinement requires at least one denoising step.")
    metadata = {
        "pipeline": type(pipeline).__name__,
        "conditioning": "img2img-latents-with-controlnet-and-true-cfg",
        "requested_steps": steps,
        "effective_steps": effective_steps,
        "denoising_strength": strength,
        "raw_sigmas": raw_sigmas,
        "full_sigmas": scheduler.sigmas.tolist(),
        "denoising_timesteps": timesteps.tolist(),
    }
    with torch.inference_mode():
        initial_image = image_pipeline.image_processor.preprocess(images, height=height, width=width)
        latents, _ = image_pipeline.prepare_latents(
            initial_image, timesteps[:1].repeat(len(images)), len(images),
            pipeline.transformer.config.in_channels // 4, height, width,
            dtype, device, generator,
        )
        arguments = {**call_arguments, "latents": latents, "generator": generator,
                     "sigmas": raw_sigmas}
        with _selected_schedule(scheduler, timesteps):
            result = pipeline(**arguments)
    return result, metadata
