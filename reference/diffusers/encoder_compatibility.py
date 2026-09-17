"""Scoped Diffusers/Transformers layout compatibility, without replacing neural math."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from presets import PipelinePreset


@contextmanager
def lora_encoder_compatibility(pipeline: Any) -> Iterator[None]:
    """Match legacy CLIP LoRA names to the encoder actually being loaded."""
    load = getattr(pipeline, "load_lora_into_text_encoder", None)
    if not callable(load):
        yield
        return
    missing = object()
    previous = vars(pipeline).get("load_lora_into_text_encoder", missing)

    def compatible(state_dict, network_alphas=None, text_encoder=None, prefix=None, **kwargs):
        if (text_encoder is not None and not hasattr(text_encoder, "text_model")
                and hasattr(text_encoder, "encoder") and hasattr(text_encoder, "embeddings")):
            component = prefix or getattr(pipeline, "text_encoder_name", "text_encoder")
            old, new = component + ".text_model.", component + "."
            def remap(values):
                if values is None:
                    return None
                result = {}
                for key, value in values.items():
                    target = new + key[len(old):] if key.startswith(old) else key
                    if target in result:
                        raise ValueError("LoRA contains conflicting nested and flat CLIP keys.")
                    result[target] = value
                return result
            state_dict, network_alphas = remap(state_dict), remap(network_alphas)
        return load(state_dict, network_alphas=network_alphas, text_encoder=text_encoder, prefix=prefix, **kwargs)

    # Only this pipeline's loader is adapted, not global Diffusers state or the
    # neural module tree. Preserve text_encoder_2's still-nested projection model.
    pipeline.load_lora_into_text_encoder = compatible
    try:
        yield
    finally:
        if previous is missing:
            del pipeline.load_lora_into_text_encoder
        else:
            pipeline.load_lora_into_text_encoder = previous


@contextmanager
def clip_skip_compatibility(pipeline: Any, preset: PipelinePreset, skip: int | None) -> Iterator[bool]:
    if preset.family != "sd15" or skip is None:
        yield False
        return
    encoder = pipeline.text_encoder
    if hasattr(encoder, "text_model"):
        if not callable(getattr(encoder.text_model, "final_layer_norm", None)):
            raise ValueError("Unsupported CLIP skip normalization layout.")
        yield False
        return
    normalization = getattr(encoder, "final_layer_norm", None)
    if not callable(normalization):
        raise ValueError("Unsupported CLIP skip normalization layout.")
    # Transformers 5.16's CLIP is flat; Diffusers 0.40's SD clip-skip path
    # still accesses text_model.final_layer_norm. A non-Module view avoids
    # duplicate registration, cyclic children and altered state_dict keys.
    alias = SimpleNamespace(final_layer_norm=normalization)
    encoder.text_model = alias
    try:
        yield True
    finally:
        if getattr(encoder, "text_model", None) is alias:
            del encoder.text_model
