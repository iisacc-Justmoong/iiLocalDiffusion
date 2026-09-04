"""Optional caller-supplied latent and text tensors through the existing safe format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cpu_conditioning import CpuConditioning
from presets import PipelinePreset
from weight_files import LocalWeightFile, checked_safetensors_path, weight_file_metadata


EMBEDDING_KEYS = (
    "prompt_embeds", "negative_prompt_embeds",
    "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
)


@dataclass(frozen=True)
class GenerationTensorInputs:
    latents: Any = None
    conditioning: CpuConditioning | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_selected_tensors(
    weight: LocalWeightFile, keys: dict[str, str], required: set[str], staging: Path,
) -> dict[str, Any]:
    from safetensors import safe_open

    with checked_safetensors_path(weight, staging, "generation tensor input") as path:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            available = set(reader.keys())
            missing = {keys[name] for name in required if keys[name] not in available}
            if missing:
                raise ValueError("Missing generation tensor keys: " + ", ".join(sorted(missing)))
            # No pickle, remote code, arbitrary object construction, or whole-file tensor loading.
            return {name: reader.get_tensor(key).clone() for name, key in keys.items() if key in available}


def _owned_tensor(tensor: Any, torch: Any, dtype: Any, name: str) -> Any:
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise ValueError(f"Generation tensor {name} must contain finite floating-point values.")
    result = tensor.to(device="cpu", dtype=dtype, copy=True)
    if not torch.isfinite(result).all().item():
        raise ValueError(f"Generation tensor {name} overflows the requested dtype.")
    return result


def _check_shape(name: str, tensor: Any, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Generation tensor {name} shape is {tuple(tensor.shape)}, expected {expected}.")


def load_tensor_inputs(
    pipeline: Any, preset: PipelinePreset, args: Any, torch: Any, dtype: Any,
) -> GenerationTensorInputs:
    latent_file = getattr(args, "latents_file", None)
    embedding_file = getattr(args, "embeddings_file", None)
    if latent_file is None and embedding_file is None:
        return GenerationTensorInputs()
    staging = args.cache_dir / "single-file-aliases"
    metadata: dict[str, Any] = {}
    latents = None
    if latent_file is not None:
        raw = read_selected_tensors(latent_file, {"latents": args.latents_key}, {"latents"}, staging)
        latents = _owned_tensor(raw["latents"], torch, dtype, "latents")
        factor = pipeline.vae_scale_factor
        if preset.family == "flux1-schnell":
            expected = (args.num_images, (args.height // (2 * factor)) * (args.width // (2 * factor)),
                        pipeline.transformer.config.in_channels)
        else:
            expected = (args.num_images, pipeline.unet.config.in_channels,
                        args.height // factor, args.width // factor)
        _check_shape("latents", latents, expected)
        metadata["latents"] = {
            "file": weight_file_metadata(latent_file), "key": args.latents_key,
            "shape": list(latents.shape), "source_dtype": str(raw["latents"].dtype), "dtype": str(dtype),
        }
    conditioning = None
    if embedding_file is not None:
        has_pool = preset.family != "sd15"
        has_cfg = args.true_cfg_scale > 1 if preset.family == "flux1-schnell" else args.guidance_scale > 1
        supported = EMBEDDING_KEYS if has_pool else EMBEDDING_KEYS[:2]
        keys = {name: args.embedding_keys.get(name, name) for name in supported}
        required = {"prompt_embeds"}
        if has_pool:
            required.add("pooled_prompt_embeds")
        if has_cfg:
            required.add("negative_prompt_embeds")
            if has_pool:
                required.add("negative_pooled_prompt_embeds")
        raw = read_selected_tensors(embedding_file, keys, required, staging)
        if required - raw.keys():
            raise ValueError("Missing required prompt/pooled/negative embeddings.")
        tensors = {name: _owned_tensor(tensor, torch, dtype, name) for name, tensor in raw.items()}
        prompt = tensors["prompt_embeds"]
        if len(prompt.shape) != 3 or prompt.shape[0] not in (1, args.num_images) or prompt.shape[1] <= 0:
            raise ValueError("Prompt embeddings require shape [1 or num_images, tokens, embedding_dimension].")
        batch, tokens, _ = prompt.shape
        if preset.family == "flux1-schnell":
            dimension = pipeline.transformer.config.joint_attention_dim
            pool_dimension = pipeline.transformer.config.pooled_projection_dim
            if tokens > 512:
                raise ValueError("FLUX embedding token count must not exceed 512.")
        else:
            dimension = pipeline.unet.config.cross_attention_dim
            pool_dimension = pipeline.text_encoder_2.config.projection_dim if has_pool else None
        for name, tensor in tensors.items():
            expected = (batch, pool_dimension) if "pooled" in name else (batch, tokens, dimension)
            _check_shape(name, tensor, expected)
        source_shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
        expanded = batch == args.num_images and args.num_images > 1
        if preset.family == "flux1-schnell":
            if batch == 1 and args.num_images > 1:
                tensors = {name: tensor.repeat_interleave(args.num_images, dim=0) for name, tensor in tensors.items()}
            expanded = True
        conditioning = CpuConditioning(tensors, {
            "enabled": False, "provided_embeddings": True, "batch_expanded": expanded,
            "execution_device": "cpu", "dtype": str(dtype),
            "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        })
        metadata["embeddings"] = {
            "file": weight_file_metadata(embedding_file),
            "keys": {name: keys[name] for name in raw},
            "source_shapes": source_shapes, "source_dtypes": {name: str(tensor.dtype) for name, tensor in raw.items()},
            "execution_shapes": conditioning.metadata["shapes"],
            "dtype": str(dtype), "batch_expanded": expanded,
            "text_prompts_used": False,
        }
    return GenerationTensorInputs(latents, conditioning, metadata)
