"""Verified learned-token embeddings using Diffusers' Textual Inversion loader."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Any

from weight_files import checked_safetensors_path, weight_file_metadata


COMPONENT_KEYS = {
    "text_encoder": "text_encoder", "text_encoder_2": "text_encoder_2",
    "clip_l": "text_encoder", "clip_g": "text_encoder_2", "t5xxl": "text_encoder_2",
}
PROMPT_TENSOR_KEYS = {
    "prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
}


@dataclass
class TextEmbeddingActivation:
    metadata: list[dict[str, Any]]
    registrations: list[dict[str, Any]]


class _EmbeddingEncoder:
    """Delegate registration while preserving unused padded vocabulary rows."""

    def __init__(self, encoder: Any):
        self.encoder = encoder

    @property
    def device(self):
        return self.encoder.device

    @property
    def dtype(self):
        return self.encoder.dtype

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def resize_token_embeddings(self, size: int):
        size = max(size, self.get_input_embeddings().weight.shape[0])
        # All new token rows receive verified vectors immediately afterwards.
        return self.encoder.resize_token_embeddings(size, mean_resizing=False)


def _valid_token(token: str) -> None:
    if (not isinstance(token, str) or not token
            or any(c.isspace() or unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in token)):
        raise ValueError("Text embedding tokens must be non-empty and contain no whitespace or control characters.")


def _components(pipeline: Any, preset: Any) -> dict[str, tuple[Any, Any]]:
    names = ("text_encoder",) if preset.family == "sd15" else ("text_encoder", "text_encoder_2")
    result = {}
    for name in names:
        tokenizer_name = "tokenizer" if name == "text_encoder" else "tokenizer_2"
        encoder, tokenizer = getattr(pipeline, name, None), getattr(pipeline, tokenizer_name, None)
        if encoder is None or tokenizer is None:
            raise ValueError(f"Text embeddings require {name} and {tokenizer_name}.")
        if encoder.device.type != "cpu" or getattr(encoder, "_hf_hook", None) is not None:
            raise ValueError("Load text embeddings on CPU before installing device/offload hooks.")
        rows = encoder.get_input_embeddings().weight.shape[0]
        if any(index < 0 or index >= rows for index in tokenizer.get_vocab().values()):
            raise ValueError(f"Existing {tokenizer_name} token IDs exceed {name}'s embedding table.")
        result[name] = (encoder, tokenizer)
    return result


def _read_file(selection: Any, staging: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    with checked_safetensors_path(selection.file, staging, "text embedding") as path:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            keys = list(reader.keys())
            if not keys or PROMPT_TENSOR_KEYS.intersection(keys):
                raise ValueError("Use --embeddings for complete prompt tensors; --text-embedding needs learned token vectors.")
            if len(keys) > 1 and not set(keys).issubset(COMPONENT_KEYS):
                raise ValueError("A text embedding file needs one token tensor or named encoder tensors.")
            return {key: reader.get_tensor(key).clone() for key in keys}, reader.metadata() or {}


def _prepare_entries(selection: Any, raw: dict[str, Any], file_metadata: dict[str, str],
                     components: dict[str, tuple[Any, Any]], preset: Any, torch: Any) -> list[dict[str, Any]]:
    entries = []
    generic = len(raw) == 1 and next(iter(raw)) not in COMPONENT_KEYS
    if generic:
        key = next(iter(raw))
        default_token = key if key != "emb_params" else Path(selection.file.path).stem
    else:
        default_token = Path(selection.file.path).stem
    token = selection.token or file_metadata.get("token") or file_metadata.get("name") or default_token
    _valid_token(token)
    for key, source in raw.items():
        if (not source.is_floating_point() or source.ndim not in (1, 2)
                or any(size <= 0 for size in source.shape) or not torch.isfinite(source).all().item()):
            raise ValueError(f"Text embedding {key} requires finite vectors shaped [dimension] or [vectors, dimension].")
        vectors = source.unsqueeze(0) if source.ndim == 1 else source
        component = COMPONENT_KEYS.get(key)
        if (key == "clip_g" and preset.family != "sdxl-base") or (key == "t5xxl" and preset.family != "flux1-schnell"):
            raise ValueError(f"Text embedding key {key} is incompatible with {preset.name}.")
        if selection.encoder != "auto":
            if component is not None and component != selection.encoder:
                raise ValueError(f"Text embedding key {key} targets {component}, not {selection.encoder}.")
            component = selection.encoder
        if component is None:
            matches = [name for name, (encoder, _) in components.items()
                       if encoder.get_input_embeddings().weight.shape[-1] == vectors.shape[-1]]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely match the text embedding dimension; select --text-embedding-encoder.")
            component = matches[0]
        if component not in components:
            raise ValueError(f"Text embedding targets unavailable {component}.")
        encoder, tokenizer = components[component]
        weight = encoder.get_input_embeddings().weight
        if weight.shape[-1] != vectors.shape[-1]:
            raise ValueError(f"Text embedding {key} dimension {vectors.shape[-1]} does not match {component} ({weight.shape[-1]}).")
        if vectors.shape[0] > getattr(tokenizer, "model_max_length", 512):
            raise ValueError(f"Text embedding {key} has more vectors than the tokenizer's context length.")
        expected = vectors.to(device="cpu", dtype=weight.dtype, copy=True)
        if not torch.isfinite(expected).all().item():
            raise ValueError(f"Text embedding {key} overflows {component}'s dtype.")
        tokens = [token] + [f"{token}_{i}" for i in range(1, vectors.shape[0])]
        entries.append({"component": component, "token": token, "tokens": tokens, "vectors": expected,
                        "source_key": key, "source_dtype": str(source.dtype), "shape": list(vectors.shape),
                        "dtype": str(expected.dtype), "vector_count": vectors.shape[0]})
    return entries


def apply_text_embeddings(pipeline: Any, preset: Any, args: Any, torch: Any) -> TextEmbeddingActivation | None:
    selections = getattr(args, "text_embedding_selections", ())
    if not selections:
        return None
    if torch is None:
        raise ValueError("Text embeddings require the PyTorch runtime.")
    from diffusers.loaders import TextualInversionLoaderMixin

    components = _components(pipeline, preset)
    planned = []
    occupied = {name: set(tokenizer.get_vocab()) for name, (_, tokenizer) in components.items()}
    for selection in selections:
        raw, metadata = _read_file(selection, args.cache_dir / "single-file-aliases")
        entries = _prepare_entries(selection, raw, metadata, components, preset, torch)
        for entry in entries:
            names = occupied[entry["component"]]
            conflicts = names.intersection(entry["tokens"])
            # Existing suffixes would also make the upstream converter consume
            # vectors that do not belong to this embedding.
            if conflicts or entry["token"] + "_1" in names:
                raise ValueError(f"Text embedding token collision on {entry['component']}: {entry['token']}")
            names.update(entry["tokens"])
        planned.append((selection, entries))

    # Validate every file/component/token before mutating any vocabulary.
    registrations, metadata = [], []
    for selection, entries in planned:
        entry_metadata = []
        for entry in entries:
            encoder, tokenizer = components[entry["component"]]
            entry["embedding_rows_before"] = encoder.get_input_embeddings().weight.shape[0]
            loader = TextualInversionLoaderMixin()
            loader.hf_device_map = None
            loader.components = {"text_encoder": encoder, "tokenizer": tokenizer}
            loader.load_textual_inversion({entry["token"]: entry["vectors"]},
                                          token=entry["token"], tokenizer=tokenizer,
                                          text_encoder=_EmbeddingEncoder(encoder))
            entry["embedding_rows_after"] = encoder.get_input_embeddings().weight.shape[0]
            entry["token_ids"] = [tokenizer.convert_tokens_to_ids(token) for token in entry["tokens"]]
            registrations.append(entry)
            entry_metadata.append({name: value for name, value in entry.items() if name != "vectors"})
        metadata.append({"file": weight_file_metadata(selection.file), "requested_token": selection.token,
                         "requested_encoder": selection.encoder, "registrations": entry_metadata})
    activation = TextEmbeddingActivation(metadata, registrations)
    validate_text_embeddings(pipeline, activation, torch)
    return activation


def validate_text_embeddings(pipeline: Any, activation: TextEmbeddingActivation | None, torch: Any) -> None:
    if activation is None:
        return
    for entry in activation.registrations:
        name = entry["component"]
        tokenizer = getattr(pipeline, "tokenizer" if name == "text_encoder" else "tokenizer_2")
        ids = [tokenizer.convert_tokens_to_ids(token) for token in entry["tokens"]]
        if (ids != entry["token_ids"] or len(set(ids)) != len(ids)
                or any(tokenizer.get_vocab().get(token) != token_id for token, token_id in zip(entry["tokens"], ids))):
            raise RuntimeError(f"Text embedding token IDs changed on {name}.")
        weight = getattr(pipeline, name).get_input_embeddings().weight
        actual = weight.detach()[ids].to(device="cpu")
        expected = entry["vectors"].to(dtype=actual.dtype)
        if not torch.isfinite(actual).all().item() or not torch.equal(actual, expected):
            raise RuntimeError(f"Text embedding vectors changed on {name}.")


def _expand_prompt(prompt: str, registrations: list[dict[str, Any]], component: str) -> str:
    replacements = {}
    for entry in registrations:
        if entry["component"] == component:
            replacements.update({token: token for token in entry["tokens"]})
            replacements[entry["token"]] = " ".join(entry["tokens"])
    if not replacements:
        return prompt
    # One longest-match pass preserves explicit suffix tokens and avoids
    # replacing a prefix inside another loaded token or re-expanding inserts.
    pattern = "|".join(re.escape(token) for token in sorted(replacements, key=len, reverse=True))
    return re.sub(pattern, lambda match: replacements[match.group()], prompt)


@contextmanager
def text_embedding_prompt_context(pipeline: Any, preset: Any, args: Any):
    """Expand each encoder's learned tokens once, including FLUX compatibility paths."""
    activation = getattr(args, "text_embedding_activation", None)
    if activation is None:
        yield args
        return
    request = argparse.Namespace(**vars(args))
    for name, fallback, component in (
        ("prompt", "prompt", "text_encoder"),
        ("negative_prompt", "negative_prompt", "text_encoder"),
        ("prompt_2", "prompt", "text_encoder_2"),
        ("negative_prompt_2", "negative_prompt", "text_encoder_2"),
    ):
        if component == "text_encoder_2" and preset.family == "sd15":
            continue
        original = getattr(args, name, None)
        # Diffusers treats an empty secondary prompt like an omitted one.
        # Resolve this before expansion so it uses the second encoder's tokens.
        if original is None or (name.endswith("_2") and original == ""):
            original = getattr(args, fallback)
        setattr(request, name, _expand_prompt(original, activation.registrations, component))
    absent = object()
    original_conversion = vars(pipeline).get("maybe_convert_prompt", absent)
    pipeline.maybe_convert_prompt = lambda prompt, tokenizer: prompt
    try:
        yield request
    finally:
        if original_conversion is absent:
            del pipeline.maybe_convert_prompt
        else:
            pipeline.maybe_convert_prompt = original_conversion
