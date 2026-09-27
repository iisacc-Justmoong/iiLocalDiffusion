"""Architecture contracts for the in-process native image backend.

Detection is separate from execution: a plan identifies missing components but
does not claim that a header-only inspection validates every weight tensor.
The pinned engine performs that validation when the context is prepared.
"""
from pathlib import Path


# (text encoders, VAE family, default steps, CFG, distilled guidance, image size)
PROFILES = {
    "sd1": (("clip_l",), "sd15", 20, 7.0, 3.5, 512),
    "sd2": (("clip_l",), "sd2", 20, 7.0, 3.5, 768),
    "sdxl": (("clip_l", "clip_g"), "sdxl-base", 25, 7.0, 3.5, 1024),
    "sd3": (("clip_l", "clip_g", "t5xxl"), "sd3", 28, 4.5, 3.5, 1024),
    "flux1": (("clip_l", "t5xxl"), "flux1", 28, 1.0, 3.5, 1024),
    "flux1-dev": (("clip_l", "t5xxl"), "flux1", 28, 1.0, 3.5, 1024),
    "flux1-schnell": (("clip_l", "t5xxl"), "flux1", 4, 1.0, 0.0, 1024),
    "flux2": (("llm",), "flux2", 28, 1.0, 4.0, 1024),
    "flux2-klein": (("llm",), "flux2", 20, 4.0, 1.0, 1024),
    "z-image": (("llm",), "flux1", 28, 5.0, 1.0, 1024),
    "qwen-image": (("llm",), "qwen-image", 30, 4.0, 1.0, 1024),
    "chroma": (("t5xxl",), "flux1", 28, 4.0, 1.0, 1024),
    "krea2": (("llm",), "wan", 52, 4.5, 1.0, 1024),
    "anima": (("llm",), "qwen-image", 20, 7.0, 3.5, 1024),
}
COMPONENTS = frozenset({"clip_l", "clip_g", "t5xxl", "llm", "vae"})


def inspect_pipeline_package(directory):
    """Read a local pipeline identity without importing or executing its code."""
    directory = Path(directory).expanduser().resolve(strict=True)
    index = directory / "model_index.json"
    if not index.is_file() or index.stat().st_size > 1024 * 1024:
        raise ValueError("A pipeline package requires a bounded local model_index.json.")
    from generation_config import json_object
    config = json_object(index.read_text(encoding="utf-8"))
    name = config.get("_class_name")
    if not isinstance(name, str) or not name.isidentifier() or not name.endswith("Pipeline"):
        raise ValueError("Pipeline package has no valid built-in pipeline class identity.")
    architecture = None
    for prefix, identity in (("StableDiffusionXL", "sdxl"), ("StableDiffusion3", "sd3"),
            ("StableDiffusion", "sd1-or-sd2"), ("Flux2Klein", "flux2-klein"), ("Flux2", "flux2"),
            ("Flux", "flux1"), ("ZImage", "z-image"), ("Krea2", "krea2"), ("QwenImage", "qwen-image"),
            ("Chroma", "chroma"), ("Wan", "wan"), ("LTX", "ltx")):
        if name.startswith(prefix):
            architecture = identity
            break
    return {"path": str(directory), "format": "diffusers", "role": "pipeline", "architecture": architecture,
            "pipeline_class": name, "confidence": "configuration", "backend": "diffusers",
            "validation": "configuration-only; installed built-in class, component weights and output contract are validated on load",
            "components": {key: value for key, value in config.items()
                           if isinstance(value, list) and len(value) == 2 and all(v is None or isinstance(v, str) for v in value)}}


def family(architecture):
    if architecture and architecture.startswith("flux1"):
        return "flux1"
    if architecture and architecture.startswith("sdxl"):
        return "sdxl"
    return architecture


def detect_transformer(shapes):
    """Match original and Diffusers layouts before generic Flux/UNet tests."""
    if any(k in shapes for k in ("txtfusion.projector.weight", "text_fusion.projector.weight")):
        return "krea2"
    if "cap_embedder.0.weight" in shapes and any(k.startswith(("layers.", "noise_refiner.")) for k in shapes):
        return "z-image"
    if "transformer_blocks.0.img_mod.1.weight" in shapes:
        # Layered/RGBA and Mage Flow have distinct decoder contracts.
        if "time_text_embed.addition_t_embedding.weight" in shapes:
            return "qwen-image-layered"
        image_shape = shapes.get("img_in.weight", ())
        if len(image_shape) == 2 and image_shape[1] == 128:
            return "mage-flow"
        return "qwen-image"
    if any(k.startswith("distilled_guidance_layer.") for k in shapes):
        return "chroma"
    if any(k.startswith("double_stream_modulation_img.") for k in shapes):
        return "flux2" if any(k.startswith(("single_blocks.47.", "single_transformer_blocks.47.")) for k in shapes) else "flux2-klein"
    return None


def embedded_slots(shapes, architecture):
    slots = []
    prefixes = {
        "vae": ("first_stage_model.encoder.", "vae.encoder."),
        "clip_l": ("text_encoders.clip_l.", "text_encoder.text_model.", "conditioner.embedders.0.transformer."),
        "clip_g": ("text_encoders.clip_g.", "conditioner.embedders.1.model."),
        "t5xxl": ("text_encoders.t5xxl.", "text_encoders.t5.", "text_encoder_3.encoder."),
        "llm": ("text_encoders.llm.", "text_encoders.qwen3.", "text_encoders.qwen2_5_vl."),
    }
    if architecture == "krea2":
        prefixes["llm"] += ("text_encoders.qwen3vl_4b.transformer.",)
    for slot, starts in prefixes.items():
        if any(k.startswith(starts) for k in shapes):
            slots.append(slot)
    if architecture in ("sd1", "sd2") and any(k.startswith("cond_stage_model.") for k in shapes):
        slots.append("clip_l")
    if family(architecture) == "flux1" and any(k.startswith("text_encoder_2.encoder.") for k in shapes):
        slots.append("t5xxl")
    return sorted(set(slots))


def plan_native(inspection, components=None):
    architecture = inspection.get("architecture")
    if inspection.get("role") != "checkpoint" or architecture not in PROFILES:
        raise ValueError(f"No native image backend contract for {architecture or inspection.get('role')}. Use a matching Diffusers pipeline package.")
    if inspection.get("task") != "text-to-image":
        raise ValueError(f"{inspection.get('task')} requires image/mask inputs and its matching pipeline; it is not a text-to-image request.")
    if inspection.get("format") not in ("safetensors", "gguf"):
        raise ValueError("Native inference requires safetensors or GGUF weights; convert legacy pickle checkpoints first.")
    if components is not None and not isinstance(components, dict):
        raise ValueError("--components must be an object of native component names and local file paths.")
    components = dict(components or {})
    unknown = set(components) - COMPONENTS
    if unknown:
        raise ValueError("Unknown native components: " + ", ".join(sorted(unknown)))
    encoders, vae, steps, cfg, guidance, size = PROFILES[architecture]
    extra = set(components) - {*encoders, "vae"}
    if extra:
        raise ValueError(f"Components do not belong to {architecture}: " + ", ".join(sorted(extra)))
    for slot, path in components.items():
        if not isinstance(path, str) or not path or "\0" in path or not Path(path).expanduser().is_file():
            raise ValueError(f"Component {slot} requires an existing local weight file.")
        components[slot] = str(Path(path).expanduser().resolve())
    present = set(inspection.get("component_slots", [])) | set(components)
    required = ["vae", *encoders]
    missing = [slot for slot in required if slot not in present]
    base = (inspection.get("base_model") or "").casefold()
    if architecture == "z-image" and "turbo" in base:
        steps, cfg = 9, 1.0
    if architecture == "flux2-klein" and base and "base" not in base:
        steps, cfg = 4, 1.0
    return {"schema": "iild-backend-plan-v1", "backend": "native", "architecture": architecture,
            "task": "text-to-image", "output_kind": "image", "vae_family": vae,
            "components": components, "embedded_components": sorted(set(inspection.get("component_slots", []))),
            "missing_components": missing, "ready_for_loading": not missing,
            "validation": "headers-only; full tensor compatibility is checked during model preparation",
            "defaults": {"steps": steps, "guidance_scale": cfg, "embedded_guidance": guidance, "size": size}}
