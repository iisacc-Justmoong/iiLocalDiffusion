"""Scoped Diffusers/Transformers layout compatibility, without replacing neural math."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from presets import PipelinePreset


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
