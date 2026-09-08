"""Opt-in, per-step VAE previews for local latent image diffusion."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from generation_output import write_png


def add_preview_options(parser):
    parser.add_argument("--preview-dir", type=Path,
                        help="Empty temporary directory for live VAE previews; emits IILD_PREVIEW JSON lines after every denoising step")


def validate_preview_location(directory, output_directory):
    if directory is not None and Path(directory).expanduser().resolve().is_relative_to(Path(output_directory).expanduser().resolve()):
        raise ValueError("Keep the temporary preview directory outside the final output directory.")


class DenoisingPreview:
    """Decode a detached copy; never replace the sampler's latents or random state."""

    def __init__(self, directory, torch, width, height):
        self.directory = Path(directory).expanduser().absolute()
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise ValueError("The preview directory must not be redirected.")
        if self.directory.exists() and (not self.directory.is_dir() or any(self.directory.iterdir())):
            raise ValueError("The preview directory must be empty.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.torch, self.width, self.height = torch, width, height

    def __call__(self, pipeline, step, timestep, values):
        torch = self.torch
        latents = values["latents"][:1].detach().clone()
        if not torch.isfinite(latents).all().item():
            raise RuntimeError("Cannot preview non-finite denoising latents.")
        if latents.ndim == 3 and type(pipeline).__name__.startswith("Flux"):
            default_extent = pipeline.default_sample_size * pipeline.vae_scale_factor if hasattr(pipeline, "default_sample_size") else None
            latents = pipeline._unpack_latents(latents, self.height or default_extent, self.width or default_extent, pipeline.vae_scale_factor)
        if latents.ndim != 4:
            raise ValueError("Live image previews require spatial image latents.")
        vae = pipeline.vae
        original_dtype = vae.dtype
        upcast = original_dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
        try:
            if upcast:
                vae.to(dtype=torch.float32)
            latents = latents.to(dtype=vae.dtype)
            mean, std = getattr(vae.config, "latents_mean", None), getattr(vae.config, "latents_std", None)
            if mean is not None and std is not None:
                mean = torch.tensor(mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
                std = torch.tensor(std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
                latents = latents * std / vae.config.scaling_factor + mean
            else:
                latents = latents / vae.config.scaling_factor + (getattr(vae.config, "shift_factor", None) or 0)
            # Decoding can execute offload hooks that replace cached VAE weights.
            with torch.no_grad():
                decoded = vae.decode(latents, return_dict=False)[0]
            if not torch.isfinite(decoded).all().item():
                raise RuntimeError("The preview VAE produced non-finite pixels.")
            image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")
        finally:
            if upcast:
                vae.to(dtype=original_dtype)
        # Preview storage and Qt decoding stay bounded even for larger final images.
        image.thumbnail((512, 512))
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise RuntimeError("The preview directory was redirected during generation.")
        name = f"step-{step + 1:06d}.png"
        write_png(image, self.directory / name, compress_level=1, optimize=False, overwrite=False)
        print("IILD_PREVIEW " + json.dumps({"schema": "iild-preview-v1", "step": step + 1,
              "total_steps": len(pipeline.scheduler.timesteps), "image": name}), flush=True)
        return values


def attach_preview(pipeline, arguments, directory, torch, width, height):
    if directory is None:
        return
    parameters = inspect.signature(pipeline.__call__).parameters
    if ("callback_on_step_end" not in parameters or "latents" not in getattr(pipeline, "_callback_tensor_inputs", ())
            or not callable(getattr(getattr(pipeline, "vae", None), "decode", None))):
        raise ValueError("This pipeline does not expose latent image denoising callbacks for live previews.")
    if "callback_on_step_end" in arguments:
        raise ValueError("Live previews cannot replace an existing denoising callback.")
    arguments["callback_on_step_end"] = DenoisingPreview(directory, torch, width, height)
    arguments["callback_on_step_end_tensor_inputs"] = ["latents"]
