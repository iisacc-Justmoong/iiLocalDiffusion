"""One installed resource manifest for the SDK's default image modifiers."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from text_embedding_options import TextEmbeddingSelection
from weight_files import resolve_weight_file


def resource_directory():
    override = os.environ.get("IILD_GENERATION_RESOURCES")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2] / "resources"


def read_defaults(directory=None):
    root = (Path(directory) if directory else resource_directory()).expanduser().resolve()
    path = root / "generation-defaults.json"
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Generation defaults manifest is too large.")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported generation-defaults resource manifest.")
    return root, manifest


def checked_resource(root, item):
    relative = Path(item["file"])
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("A generation-defaults resource escapes its package.")
    file = resolve_weight_file(str(path), "generation default")
    if (file.sha256 is not None and file.sha256 != item["sha256"]) or (file.sha256 is not None and file.size_bytes != item["size"]):
        raise ValueError(f"Bundled generation resource differs from its manifest: {path}")
    return file


def append_default_tokens(prompt, tokens):
    # Token aliases have no whitespace or attention syntax. Preserve custom text
    # and make resolved JSON requests safe to replay without duplicate defaults.
    import re
    missing = [token for token in tokens if not re.search(r"(?<![\w])" + re.escape(token) + r"(?![\w])", prompt)]
    return ", ".join(part for part in (prompt, ", ".join(missing)) if part)


def canonical_lora_family(family):
    return {"sd1": "sd15", "sdxl": "sdxl-base", "flux1-schnell": "flux1",
            "flux1-dev": "flux1"}.get(family, family)


def fallback_loras(manifest):
    """Keep the original single entry readable while allowing per-family defaults."""
    entries = manifest.get("fallback_loras", [])
    if not isinstance(entries, list):
        raise ValueError("fallback_loras must be an array.")
    entries = ([manifest["fallback_lora"]] if "fallback_lora" in manifest else []) + entries
    if not entries or len(entries) > 64:
        raise ValueError("The defaults manifest requires 1..64 fallback LoRAs.")
    seen = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid fallback LoRA entry.")
        scale = item.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale == 0:
            raise ValueError("A fallback LoRA must have a finite nonzero strength.")
        families = item.get("families")
        if not isinstance(families, list) or not families:
            raise ValueError("A fallback LoRA requires explicit model families.")
        for family in families:
            if not isinstance(family, str) or not family or family in ("*", "other", "sd1-or-sd2"):
                raise ValueError("A fallback LoRA requires an unambiguous model family.")
            family = canonical_lora_family(family)
            if family in seen:
                raise ValueError(f"Duplicate fallback LoRA family: {family}")
            seen.add(family)
    return entries


def pipeline_lora_family(name, folder):
    for prefix, family in (("StableDiffusionXL", "sdxl-base"), ("StableDiffusion3", "sd3"),
                           ("Flux2", "flux2"), ("Flux", "flux1"), ("QwenImage", "qwen-image"),
                           ("HunyuanDiT", "hunyuan-dit"), ("Chroma", "chroma"), ("ZImage", "z-image")):
        if name.startswith(prefix):
            return family
    if name.startswith("StableDiffusion"):
        # SD 1.x and SD 2.x share a pipeline class but not their CLIP width.
        path = Path(folder) / "text_encoder/config.json"
        config = json.loads(path.read_text()) if path.is_file() else {}
        return {768: "sd15", 1024: "sd2"}.get(config.get("hidden_size"), "sd1-or-sd2")
    return "pipeline:" + name


def resolve_default_lora(preset, args):
    resolve_family_default_lora(preset.family, args)


def resolve_family_default_lora(family, args):
    args.default_modifier_metadata = {"enabled": args.default_modifiers, "family": family}
    if not args.default_modifiers:
        args.default_modifier_metadata["fallback_lora_status"] = "disabled"
        return
    root, manifest = read_defaults(getattr(args, "generation_resources", None))
    choices = fallback_loras(manifest)
    if args.lora is not None:
        args.default_modifier_metadata["fallback_lora_status"] = "explicit-override"
        return
    fallback = next((item for item in choices if canonical_lora_family(family) in
                     [canonical_lora_family(value) for value in item["families"]]), None)
    if fallback is not None:
        if args.lora_revision is not None or args.lora_weight_name is not None:
            raise ValueError("An explicit --lora is required with revision or weight-name overrides.")
        args.lora = checked_resource(root, fallback).path
        if args.lora_scale is None:
            args.lora_scale = fallback["scale"]
        args.default_modifier_metadata["fallback_lora"] = fallback["file"]
        args.default_modifier_metadata["fallback_lora_status"] = "selected"
    else:
        args.default_modifier_metadata["fallback_lora_status"] = "not-configured-for-family"


def resolve_default_embeddings(preset, args):
    if not args.default_modifiers:
        return
    root, manifest = read_defaults(getattr(args, "generation_resources", None))
    selected = [item for item in manifest["negative_embeddings"] if preset.family in item["families"]]
    if not selected:
        return
    if getattr(args, "embeddings_file", None) is not None:
        raise ValueError("Precomputed --embeddings require --no-default-modifiers because default negatives must be encoded.")
    tokens = []
    by_file = {Path(item.file.resolved_file): item for item in args.text_embedding_selections}
    for item in selected:
        file = checked_resource(root, item)
        explicit = by_file.get(Path(file.resolved_file))
        if explicit is not None:
            if explicit.token not in (None, item["token"]):
                raise ValueError("A bundled negative embedding cannot be registered under two token names.")
            args.text_embedding_selections.remove(explicit)
        args.text_embedding_selections.append(TextEmbeddingSelection(file, item["token"], "auto"))
        tokens.append(item["token"])
    args.negative_prompt = append_default_tokens(args.negative_prompt, tokens)
    if args.negative_prompt_2:
        args.negative_prompt_2 = append_default_tokens(args.negative_prompt_2, tokens)
    args.default_modifier_metadata["negative_embeddings"] = [item["file"] for item in selected]
    args.default_modifier_metadata["negative_tokens"] = tokens
