"""Bounded model-ecosystem evidence for base-scoped merge filtering."""

from __future__ import annotations

from dataclasses import dataclass
import re


_HINTS = ("modelspec.architecture", "modelspec.implementation", "iild.model_family",
          "ss_base_model_version", "base_model", "base_model_version", "architecture")


def _family_hint(value):
    value = str(value or "").lower().replace("_", "-")
    if "krea2" in value or "krea-2" in value or "flux2" in value or "flux-2" in value:
        return "flux2"
    if "anima" in value or "cosmos-predict2" in value:
        return "anima"
    if any(name in value for name in ("illustrious", "pony", "noobai", "noob-ai", "stable-diffusion-xl", "sdxl")):
        return "sdxl"
    if "stable-diffusion-3" in value or re.search(r"(?:^|\W)sd3(?:\W|$)", value):
        return "sd3"
    if "stable-diffusion-2" in value or re.search(r"(?:^|\W)sd2(?:\W|$)", value):
        return "sd2"
    if any(name in value for name in ("stable-diffusion-1", "sd-1.5", "sd15", "sd1")):
        return "sd1"
    if "flux" in value:
        return "flux1"
    return None


@dataclass(frozen=True)
class EcosystemIdentity:
    families: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def label(self):
        return ", ".join(self.families) if self.families else "unknown"


def identify_ecosystem(readers, layout):
    families, evidence = set(), []
    keys, shapes = set(), {}
    for address, filename in layout.items():
        key = address[1]
        keys.add(key)
        shapes[key] = tuple(readers[filename].get_slice(key).get_shape())
    for reader in readers.values():
        metadata = reader.metadata() or {}
        for name in _HINTS:
            family = _family_hint(metadata.get(name))
            if family:
                families.add(family)
                evidence.append(f"metadata:{name}")
    normalized = {}
    for key, shape in shapes.items():
        value = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "unet.", "transformer."):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        normalized[value] = shape
    lowered = "\n".join(keys).lower().replace("_", ".")
    if "llm.adapter.blocks.0.cross.attn" in lowered or "anima" in lowered:
        families.add("anima"); evidence.append("tensor:anima-adapter")
    if (any(key.startswith("joint_blocks.") for key in normalized)
            or any("transformer_blocks.0.attn.add_q_proj" in key for key in normalized)):
        families.add("sd3"); evidence.append("tensor:joint-attention")
    flux_shapes = [shape for key, shape in normalized.items()
                   if key.startswith(("double_blocks.", "single_blocks.")) and len(shape) == 2]
    if flux_shapes:
        width = max(min(shape) for shape in flux_shapes)
        families.add("flux2" if width >= 4096 else "flux1")
        evidence.append(f"tensor:flux-width-{width}")
    if (any(key.startswith("blocks.0.") for key in normalized)
            and any(key.startswith(("tproj.", "txtfusion.")) for key in normalized)):
        families.add("flux2"); evidence.append("tensor:krea2-transformer")
    cross = {shape[-1] for key, shape in normalized.items()
             if key.endswith("attn2.to_k.weight") and len(shape) == 2}
    if len(cross) == 1:
        family = {768: "sd1", 1024: "sd2", 2048: "sdxl", 1280: "sdxl"}.get(next(iter(cross)))
        if family:
            families.add(family); evidence.append(f"tensor:cross-attention-{next(iter(cross))}")
    if not families:
        if any(token in lowered for token in ("double.blocks", "single.blocks", "krea2", "flux2")):
            families.add("flux2"); evidence.append("adapter:flux2-targets")
        elif "llm.adapter" in lowered or "t.embedding" in lowered:
            families.add("anima"); evidence.append("adapter:anima-targets")
    return EcosystemIdentity(tuple(sorted(families)), tuple(dict.fromkeys(evidence)))


def exact_layout_compatible(base_readers, base_layout, readers, layout):
    if base_layout.keys() != layout.keys():
        return False
    floating = ("F8_", "F16", "BF16", "F32", "F64")
    for address, filename in base_layout.items():
        left = base_readers[filename].get_slice(address[1])
        right = readers[layout[address]].get_slice(address[1])
        if left.get_shape() != right.get_shape():
            return False
        if left.get_dtype() != right.get_dtype() and not (
                left.get_dtype().startswith(floating) and right.get_dtype().startswith(floating)):
            return False
    return True
