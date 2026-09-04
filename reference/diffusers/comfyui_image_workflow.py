"""Build local text-to-image API graphs from the running ComfyUI node schema.

This module performs no model I/O and imports no ML runtime. Model/component
names must already be registered in the server inventory. Architectural recipes
follow ComfyUI e80c1570 and official workflow_templates; tensor compatibility is
ultimately checked by the loaders when the submitted graph executes.
"""

from __future__ import annotations

from copy import deepcopy
import math

from civitai_catalog import lookup_base_model


# (latent node, text-loader type, text-encoder count, resolution, steps, CFG,
#  sampler, scheduler). None text-loader means a bundled checkpoint is required.
_RECIPES = {
    "sd1": ("EmptyLatentImage", "stable_diffusion", 1, 512, 20, 7.0, "euler", "normal"),
    "sd2": ("EmptyLatentImage", "stable_diffusion", 1, 512, 20, 7.0, "euler", "normal"),
    "sdxl": ("EmptyLatentImage", "sdxl", 2, 1024, 25, 7.0, "euler", "normal"),
    "sd3": ("EmptySD3LatentImage", "sd3", 3, 1024, 28, 5.0, "euler", "sgm_uniform"),
    "flux1": ("EmptySD3LatentImage", "flux", 2, 1024, 20, 1.0, "euler", "simple"),
    "flux2": ("EmptyFlux2LatentImage", "flux2", 1, 1024, 20, 1.0, "euler", "flux2"),
    "flux2-klein": ("EmptyFlux2LatentImage", "flux2", 1, 1024, 4, 1.0, "euler", "flux2"),
    "auraflow": ("EmptyLatentImage", "stable_diffusion", 1, 1024, 30, 3.5, "euler", "simple"),
    "chroma": ("EmptySD3LatentImage", "chroma", 1, 1024, 26, 3.5, "euler", "beta"),
    "pixart-alpha": ("EmptyLatentImage", "pixart", 1, 1024, 30, 4.5, "euler", "normal"),
    "pixart-sigma": ("EmptyLatentImage", "pixart", 1, 1024, 30, 4.5, "euler", "normal"),
    "hunyuan-dit": ("EmptyLatentImage", None, 0, 1024, 30, 6.0, "euler", "normal"),
    "hidream-i1": ("EmptySD3LatentImage", "hidream", 4, 1024, 50, 5.0, "uni_pc", "simple"),
    "hidream-o1": ("EmptyHiDreamO1LatentImage", None, 0, 2048, 40, 5.0, "dpmpp_2m_sde", "normal"),
    "qwen-image": ("EmptySD3LatentImage", "qwen_image", 1, 1328, 20, 4.0, "euler", "simple"),
    "lumina": ("EmptySD3LatentImage", "lumina2", 1, 1024, 30, 4.0, "euler", "simple"),
    "z-image": ("EmptySD3LatentImage", "lumina2", 1, 1024, 25, 4.0, "res_multistep", "simple"),
    "anima": ("EmptySD3LatentImage", "stable_diffusion", 1, 1024, 30, 4.0, "euler", "simple"),
    "ernie-image": ("EmptyFlux2LatentImage", "flux2", 1, 1024, 20, 4.0, "euler", "simple"),
    "boogu": ("EmptySD3LatentImage", "boogu", 1, 1024, 4, 1.0, "lcm", "sgm_uniform"),
    "krea2": ("EmptySD3LatentImage", "krea2", 1, 1024, 8, 1.0, "euler", "simple"),
    "lens": ("EmptyFlux2LatentImage", "lens", 1, 1440, 20, 5.0, "euler", "simple"),
    "mageflow": (None, "mage", 1, 1024, 30, 5.0, "euler", "simple"),
    "ideogram4": ("EmptyFlux2LatentImage", "ideogram4", 1, 1024, 20, 7.0, "euler", "ideogram4"),
    "stable-cascade": ("StableCascade_EmptyLatentImage", "stable_cascade", 1, 1024, 20, 4.0, "euler_ancestral", "simple"),
}
_TEXT_KEYS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
_COMPONENT_KEYS = {"vae", "model_negative", "decoder", *_TEXT_KEYS}


def workflow_requirements(base_model: str) -> dict:
    """Describe the explicit recipe and component roles, without contacting a server."""
    base = lookup_base_model(base_model)
    if base["local_status"] == "hosted":
        raise ValueError(f"{base['name']} is hosted; automatic workflows require local models.")
    if "text-to-image" not in base["task"].split(","):
        raise ValueError(f"{base['name']} requires {base['task']}; it is not a text-to-image model.")
    family = base["family"]
    if family not in _RECIPES:
        raise ValueError(f"No automatic local image recipe for {base['name']}; use its complete "
                         "Diffusers pipeline or an explicit matching ComfyUI workflow.")
    latent, clip_type, count, size, steps, cfg, sampler, scheduler = _RECIPES[family]
    name = base["name"]
    if name.endswith(" 768"):
        size = 768
    if name == "Flux.1 S":
        steps = 4
    if name.startswith("Flux.2 Klein") and name.endswith("-base"):
        steps, cfg = 20, 5.0
    if "LCM" in name:
        steps, cfg, sampler, scheduler = 4, 1.0, "lcm", "sgm_uniform"
    if name == "SDXL Turbo":
        size, steps, cfg, sampler, scheduler = 512, 1, 1.0, "euler_ancestral", "sgm_uniform"
    if name == "SD 3.5 Large Turbo":
        steps, cfg = 4, 1.0
    if name == "ZImageTurbo":
        steps, cfg = 8, 1.0
    return {"base_model": name, "family": family, "latent_node": latent,
            "clip_type": clip_type, "text_encoder_count": count,
            "split_components": (["vae", *_TEXT_KEYS[:count]] if clip_type else None),
            "extra_components": (["model_negative"] if family == "ideogram4" else ["decoder"]
                                 if family == "stable-cascade" else []),
            "width": size, "height": size, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler}


def _dimension_multiple(family: str) -> int:
    if family == "hidream-o1":
        return 32
    return 8 if family in ("sd1", "sd2", "sdxl", "auraflow", "pixart-alpha", "pixart-sigma", "hunyuan-dit") else 16


def resolve_workflow_hires(base_model: str, *, width=None, height=None, steps=None,
                          hires_fix=False, hires_passes=None, hires_scale=None,
                          hires_strength=None, hires_steps=None, hires_upscaler=None) -> dict:
    """Resolve repeatable image refinement before starting the managed server.

    Passes count additional refinements. Each pass scales the preceding image;
    native SplitSigmasDenoise retains round(schedule steps * strength) steps.
    """
    supplied = {"hires_passes": hires_passes, "hires_scale": hires_scale,
                "hires_strength": hires_strength, "hires_steps": hires_steps,
                "hires_upscaler": hires_upscaler}
    if not isinstance(hires_fix, bool):
        raise ValueError("hires_fix must be a boolean.")
    if not hires_fix:
        if any(value is not None for value in supplied.values()):
            raise ValueError("HiRes refinement options require --hires-fix.")
        return {"enabled": False, "passes": 0, "stages": []}
    from hires_options import resolve_hires_stage_sizes
    recipe = workflow_requirements(base_model)
    width = recipe["width"] if width is None else width
    height = recipe["height"] if height is None else height
    passes = 1 if hires_passes is None else hires_passes
    scale = 2.0 if hires_scale is None else hires_scale
    strength = 0.35 if hires_strength is None else hires_strength
    schedule_steps = recipe["steps"] if steps is None else steps
    schedule_steps = schedule_steps if hires_steps is None else hires_steps
    upscaler = "lanczos" if hires_upscaler is None else hires_upscaler
    if (isinstance(schedule_steps, bool) or not isinstance(schedule_steps, int)
            or not 1 <= schedule_steps <= 4096):
        raise ValueError("--hires-steps must be an integer in [1, 4096].")
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 < strength <= 1):
        raise ValueError("--hires-strength must be finite and in (0, 1].")
    effective_steps = round(schedule_steps * strength)
    if effective_steps < 1:
        raise ValueError("HiRes refinement needs at least one denoising step; increase --hires-steps or --hires-strength.")
    if upscaler not in ("nearest", "bilinear", "bicubic", "lanczos"):
        raise ValueError("--hires-upscaler must be nearest, bilinear, bicubic, or lanczos.")
    dimensions = resolve_hires_stage_sizes(width, height, passes, scale=scale,
                                           dimension_multiple=_dimension_multiple(recipe["family"]))
    if any(max(size) > 16384 for size in dimensions):
        raise ValueError("HiRes image dimensions exceed the native ComfyUI ImageScale limit of 16384.")
    return {"enabled": True, "passes": passes, "scale": scale, "strength": strength,
            "steps": schedule_steps, "effective_steps": effective_steps, "upscaler": upscaler,
            "base_size": [width, height], "stages": [
                {"pass_index": index + 1, "width": size[0], "height": size[1]}
                for index, size in enumerate(dimensions)]}


def _schema(objects: dict, class_name: str) -> dict:
    specification = objects.get(class_name)
    if not isinstance(specification, dict):
        raise ValueError(f"ComfyUI is missing required image node {class_name}.")
    if specification.get("api_node") or str(specification.get("python_module", "")).startswith("comfy_api_nodes"):
        raise ValueError(f"Local image generation cannot use hosted API node {class_name}.")
    return specification


def _check_scalar(value, schema, location: str) -> None:
    kind = schema[0] if schema else None
    settings = schema[1] if len(schema) > 1 and isinstance(schema[1], dict) else {}
    if kind == "COMBO":
        kind = settings.get("options")
        if not isinstance(kind, list):
            raise ValueError(f"{location} has no enumerable installed COMBO choices.")
    if isinstance(kind, list):
        if value not in kind:
            raise ValueError(f"Unavailable model or unsupported choice at {location}: {value!r}")
    elif kind == "INT":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{location} requires an integer.")
    elif kind == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{location} requires a finite number.")
    elif kind == "BOOLEAN" and not isinstance(value, bool):
        raise ValueError(f"{location} requires a boolean.")
    elif kind == "STRING" and not isinstance(value, str):
        raise ValueError(f"{location} requires text.")
    elif kind not in ("INT", "FLOAT", "BOOLEAN", "STRING") and not isinstance(kind, list):
        raise ValueError(f"{location} requires a {kind} node link.")
    if kind in ("INT", "FLOAT"):
        if "min" in settings and value < settings["min"] or "max" in settings and value > settings["max"]:
            raise ValueError(f"{location} is outside the installed node's range.")


class _Graph:
    def __init__(self, objects):
        self.objects, self.nodes = objects, {}

    def add(self, class_name: str, **values) -> list:
        specification = _schema(self.objects, class_name)
        inputs = specification.get("input", {})
        required = inputs.get("required", {})
        available = {**required, **inputs.get("optional", {})}
        for name in values:
            if name not in available:
                raise ValueError(f"Installed ComfyUI node does not accept {class_name}.{name}.")
        for name, schema in required.items():
            if name not in values:
                settings = schema[1] if len(schema) > 1 and isinstance(schema[1], dict) else {}
                if schema[0] == "COMFY_AUTOGROW_V3" and settings.get("template", {}).get("min") == 0:
                    # ComfyUI expands these into optional numbered inputs; no
                    # image input is correct for a text-only MageFlow request.
                    continue
                if "default" not in settings:
                    raise ValueError(f"Missing required ComfyUI input {class_name}.{name}.")
                values[name] = deepcopy(settings["default"])
        for name, value in values.items():
            schema = available[name]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                source, slot = value
                if source not in self.nodes or isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                    raise ValueError(f"Invalid node link at {class_name}.{name}.")
                source_class = self.nodes[source]["class_type"]
                outputs = _schema(self.objects, source_class).get("output", [])
                if slot >= len(outputs) or outputs[slot] != schema[0]:
                    raise ValueError(f"Incompatible node link at {class_name}.{name}.")
            else:
                _check_scalar(value, schema, f"{class_name}.{name}")
        key = str(len(self.nodes) + 1)
        self.nodes[key] = {"class_type": class_name, "inputs": values}
        return [key, 0]


def _model_inventory(objects: dict, class_name: str, input_name: str) -> list:
    schema = objects.get(class_name, {}).get("input", {}).get("required", {}).get(input_name, [])
    if schema and isinstance(schema[0], list):
        return schema[0]
    if schema and schema[0] == "COMBO" and len(schema) > 1:
        choices = schema[1].get("options", [])
        return choices if isinstance(choices, list) else []
    return []


def _denoiser(graph: _Graph, name: str):
    if name.lower().endswith(".gguf"):
        return graph.add("UnetLoaderGGUF", unet_name=name)
    return graph.add("UNETLoader", unet_name=name, weight_dtype="default")


def _text_loader(graph: _Graph, recipe: dict, components: dict):
    values = [components.get(key) for key in _TEXT_KEYS]
    supplied = [value for value in values if value is not None]
    if not supplied:
        return None
    if values[:len(supplied)] != supplied:
        raise ValueError("Text encoders must be consecutive: text_encoder, text_encoder_2, ...")
    count = len(supplied)
    family = recipe["family"]
    if recipe["clip_type"] is None:
        raise ValueError(f"{recipe['base_model']} needs its bundled checkpoint text encoder; "
                         "the installed built-in CLIP loaders do not expose this encoder type.")
    if count != recipe["text_encoder_count"] and not (family == "sd3" and count in (1, 2, 3)
                                                     or family == "hidream-i1" and count in (1, 2, 4)):
        raise ValueError(f"{recipe['base_model']} requires {recipe['text_encoder_count']} text encoder(s), got {count}.")
    gguf = any(name.lower().endswith(".gguf") for name in supplied)
    if count == 1:
        return graph.add("CLIPLoaderGGUF" if gguf else "CLIPLoader",
                         clip_name=supplied[0], type=recipe["clip_type"])
    if count == 2:
        return graph.add("DualCLIPLoaderGGUF" if gguf else "DualCLIPLoader",
                         clip_name1=supplied[0], clip_name2=supplied[1], type=recipe["clip_type"])
    class_name = "TripleCLIPLoader" if count == 3 else "QuadrupleCLIPLoader"
    if gguf:
        class_name += "GGUF"
    return graph.add(class_name, **{f"clip_name{i + 1}": value for i, value in enumerate(supplied)})


def build_workflow(base_model: str, model_name: str, components: dict, prompt: str,
                   object_info: dict, *, negative_prompt: str = "", width: int | None = None,
                   height: int | None = None, seed: int = 0, steps: int | None = None,
                   cfg: float | None = None, sampler_name: str | None = None,
                   scheduler: str | None = None, batch_size: int = 1,
                   prediction_type: str | None = None, model_type: str = "auto",
                   guidance: float | None = None, sampling_shift: float | None = None,
                   zsnr: bool = False, clip_skip: int | None = None,
                   hires_fix: bool = False, hires_passes: int | None = None,
                   hires_scale: float | None = None, hires_strength: float | None = None,
                   hires_steps: int | None = None, hires_upscaler: str | None = None) -> dict:
    """Return a fully connected API graph, validated against live /object_info.

    Components are server inventory names under vae, text_encoder through
    text_encoder_4; Ideogram4 requires model_negative and Stable Cascade requires
    its stage-B denoiser as decoder. A checkpoint
    provides MODEL/CLIP/VAE; explicit components replace those roles. A split
    denoiser requires its VAE and text encoder files. GGUF requires corresponding
    local GGUF loader nodes. No filename guessing or implicit downloads occur.
    """
    recipe = workflow_requirements(base_model)
    if not isinstance(components, dict) or set(components) - _COMPONENT_KEYS:
        raise ValueError(f"Components must use these keys: {', '.join(sorted(_COMPONENT_KEYS))}.")
    for name, value in {"model": model_name, **components}.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty server model inventory name.")
    if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
        raise ValueError("Prompts must be strings.")
    if not isinstance(zsnr, bool):
        raise ValueError("zsnr must be a boolean.")
    if model_type not in ("auto", "checkpoint", "diffusion_model"):
        raise ValueError("model_type must be auto, checkpoint, or diffusion_model.")
    family = recipe["family"]
    if family == "ideogram4" and negative_prompt:
        raise ValueError("Ideogram4 uses a separate unconditional denoiser, not a negative text prompt.")
    width = recipe["width"] if width is None else width
    height = recipe["height"] if height is None else height
    dimension_multiple = _dimension_multiple(family)
    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % dimension_multiple:
            raise ValueError(f"{name} must be a positive multiple of {dimension_multiple} for {base_model}.")
    steps = recipe["steps"] if steps is None else steps
    cfg = recipe["cfg"] if cfg is None else cfg
    sampler_name = recipe["sampler_name"] if sampler_name is None else sampler_name
    scheduler = recipe["scheduler"] if scheduler is None else scheduler
    hires = resolve_workflow_hires(base_model, width=width, height=height, steps=steps,
                                  hires_fix=hires_fix, hires_passes=hires_passes, hires_scale=hires_scale,
                                  hires_strength=hires_strength, hires_steps=hires_steps, hires_upscaler=hires_upscaler)
    if guidance is not None and family not in ("flux1", "flux2"):
        raise ValueError("Embedded guidance applies only to Flux.1 dev/Krea and Flux.2 dev.")
    if guidance is not None and recipe["base_model"] == "Flux.1 S":
        raise ValueError("Flux.1 Schnell does not use embedded guidance.")
    if zsnr and prediction_type is None:
        raise ValueError("zsnr requires an explicit prediction_type.")
    graph = _Graph(object_info)
    if model_type == "auto":
        checkpoint = model_name in _model_inventory(object_info, "CheckpointLoaderSimple", "ckpt_name")
        split = (model_name in _model_inventory(object_info, "UNETLoader", "unet_name")
                 or model_name in _model_inventory(object_info, "UnetLoaderGGUF", "unet_name"))
        if checkpoint and split:
            raise ValueError("Model exists in both checkpoint and diffusion_model inventories; specify model_type.")
        if not checkpoint and not split:
            raise ValueError(f"Model is absent from the local ComfyUI checkpoint/denoiser inventory: {model_name}")
        model_type = "checkpoint" if checkpoint else "diffusion_model"
    if model_type == "checkpoint":
        model = graph.add("CheckpointLoaderSimple", ckpt_name=model_name)
        clip, vae = [model[0], 1], [model[0], 2]
    else:
        if recipe["split_components"] is None:
            raise ValueError(f"{base_model} requires a bundled checkpoint for automatic image generation.")
        if "vae" not in components or "text_encoder" not in components:
            raise ValueError(f"Split {base_model} denoiser requires vae and text_encoder components.")
        model, clip, vae = _denoiser(graph, model_name), None, None
    clip = _text_loader(graph, recipe, components) or clip
    if "vae" in components:
        vae = graph.add("VAELoader", vae_name=components["vae"])
    if "model_negative" in components and family != "ideogram4":
        raise ValueError("model_negative is an Ideogram4 unconditional denoiser component.")
    if "decoder" in components and family != "stable-cascade":
        raise ValueError("decoder is a Stable Cascade stage-B denoiser component.")
    if clip_skip is not None:
        if family not in ("sd1", "sd2", "sdxl") or isinstance(clip_skip, bool) or not isinstance(clip_skip, int) or clip_skip < 0:
            raise ValueError("clip_skip must be a nonnegative integer for SD1/2/SDXL.")
        clip = graph.add("CLIPSetLastLayer", clip=clip, stop_at_clip_layer=-(clip_skip + 1))
    if prediction_type is not None:
        if family not in ("sd1", "sd2", "sdxl"):
            raise ValueError("prediction_type overrides apply only to SD1/2/SDXL discrete diffusion.")
        prediction = {"epsilon": "eps", "v_prediction": "v_prediction", "sample": "x0", "lcm": "lcm"}.get(prediction_type)
        if prediction is None:
            raise ValueError("prediction_type must be epsilon, v_prediction, sample, or lcm.")
        model = graph.add("ModelSamplingDiscrete", model=model, sampling=prediction, zsnr=zsnr)
    elif "LCM" in recipe["base_model"]:
        model = graph.add("ModelSamplingDiscrete", model=model, sampling="lcm", zsnr=False)
    if sampling_shift is not None:
        if family not in ("sd3", "hidream-i1", "auraflow", "chroma", "qwen-image", "lumina", "z-image", "anima", "mageflow"):
            raise ValueError(f"sampling_shift is not supported by the {base_model} image recipe.")
        patch = "ModelSamplingSD3" if family in ("sd3", "hidream-i1") else "ModelSamplingAuraFlow"
        model = graph.add(patch, model=model, shift=sampling_shift)
    elif family in ("qwen-image", "chroma"):
        model = graph.add("ModelSamplingAuraFlow", model=model, shift=3.1 if family == "qwen-image" else 1.0)
    if family == "chroma":
        clip = graph.add("T5TokenizerOptions", clip=clip, min_padding=0, min_length=0)
    if family == "lens":
        model = graph.add("ModelSamplingFlux", model=model, max_shift=1.15, base_shift=0.5, width=width, height=height)
        model = graph.add("CFGNorm", model=model, strength=1.0, pre_cfg=True)

    def encode(text, target_width=width, target_height=height):
        if family == "pixart-alpha":
            return graph.add("CLIPTextEncodePixArtAlpha", clip=clip, text=text, width=target_width, height=target_height)
        if family == "hunyuan-dit":
            return graph.add("CLIPTextEncodeHunyuanDiT", clip=clip, bert=text, mt5xl=text)
        if family == "lumina":
            return graph.add("CLIPTextEncodeLumina2", clip=clip, user_prompt=text, system_prompt="superior")
        return graph.add("CLIPTextEncode", clip=clip, text=text)

    if family == "mageflow":
        encoded = graph.add("TextEncodeMageFlowEdit", clip=clip, vae=vae, prompt=prompt,
                            negative_prompt=negative_prompt, width=width, height=height, batch_size=batch_size)
        positive, negative, latent = encoded, [encoded[0], 1], [encoded[0], 2]
    else:
        positive, negative = encode(prompt), encode(negative_prompt)
        latent = graph.add(recipe["latent_node"], width=width, height=height, batch_size=batch_size)
    if family == "ideogram4":
        negative = graph.add("ConditioningZeroOut", conditioning=positive)
    if family in ("flux1", "flux2") and recipe["base_model"] != "Flux.1 S":
        positive = graph.add("FluxGuidance", conditioning=positive,
                             guidance=(4.0 if family == "flux2" else 3.5) if guidance is None else guidance)

    if family in ("flux2", "flux2-klein", "ideogram4"):
        if scheduler != recipe["scheduler"]:
            raise ValueError(f"{base_model} requires its {recipe['scheduler']} schedule; got {scheduler}.")
        if family == "ideogram4":
            if "model_negative" not in components:
                raise ValueError("Ideogram4 requires its separate model_negative unconditional denoiser.")
            negative_model = _denoiser(graph, components["model_negative"])
            guider = graph.add("DualModelGuider", model=model, model_negative=negative_model,
                               positive=positive, negative=negative, cfg=cfg)
            sigmas = graph.add("Ideogram4Scheduler", steps=steps, width=width, height=height, mu=0.5, std=1.75)
        else:
            guider = graph.add("CFGGuider", model=model, positive=positive, negative=negative, cfg=cfg)
            sigmas = graph.add("Flux2Scheduler", steps=steps, width=width, height=height)
        noise = graph.add("RandomNoise", noise_seed=seed)
        sampler = graph.add("KSamplerSelect", sampler_name=sampler_name)
        samples = graph.add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                            sigmas=sigmas, latent_image=latent)
    else:
        samples = graph.add("KSampler", model=model, positive=positive, negative=negative, latent_image=latent,
                            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, denoise=1.0)
        if family == "stable-cascade":
            if "decoder" not in components:
                raise ValueError("Stable Cascade requires its stage-B denoiser as the decoder component.")
            decoder = _denoiser(graph, components["decoder"])
            decoder_positive = graph.add("StableCascade_StageB_Conditioning", conditioning=positive, stage_c=samples)
            decoder_negative = graph.add("ConditioningZeroOut", conditioning=negative)
            samples = graph.add("KSampler", model=decoder, positive=decoder_positive, negative=decoder_negative,
                                latent_image=[latent[0], 1], seed=seed, steps=10, cfg=1.0,
                                sampler_name="euler_ancestral", scheduler="simple", denoise=1.0)
    decoded = graph.add("VAEDecode", samples=samples, vae=vae)
    for stage in hires["stages"]:
        target_width, target_height = stage["width"], stage["height"]
        upscaled = graph.add("ImageScale", image=decoded, width=target_width, height=target_height,
                             upscale_method="nearest-exact" if hires["upscaler"] == "nearest" else hires["upscaler"],
                             crop="disabled")
        encoded_image = graph.add("VAEEncode", pixels=upscaled, vae=vae)
        refinement_model, refinement_positive, refinement_negative = model, positive, negative
        refinement_sampler, refinement_scheduler, refinement_cfg = sampler_name, scheduler, cfg
        if family == "stable-cascade":
            # The final stage-A image is encoded into stage-B latents. Refining
            # that decoder preserves the original stage-C semantic prior.
            refinement_model, refinement_positive, refinement_negative = decoder, decoder_positive, decoder_negative
            refinement_sampler, refinement_scheduler, refinement_cfg = "euler_ancestral", "simple", 1.0
        elif family == "pixart-alpha":
            refinement_positive = encode(prompt, target_width, target_height)
            refinement_negative = encode(negative_prompt, target_width, target_height)
        elif family == "lens":
            refinement_model = graph.add("ModelSamplingFlux", model=model, max_shift=1.15, base_shift=0.5,
                                         width=target_width, height=target_height)
        if family == "ideogram4":
            refinement_guider = graph.add("DualModelGuider", model=refinement_model, model_negative=negative_model,
                                          positive=refinement_positive, negative=refinement_negative, cfg=refinement_cfg)
            schedule = graph.add("Ideogram4Scheduler", steps=hires["steps"], width=target_width,
                                 height=target_height, mu=0.5, std=1.75)
        else:
            refinement_guider = graph.add("CFGGuider", model=refinement_model, positive=refinement_positive,
                                          negative=refinement_negative, cfg=refinement_cfg)
            if family in ("flux2", "flux2-klein"):
                schedule = graph.add("Flux2Scheduler", steps=hires["steps"], width=target_width, height=target_height)
            else:
                schedule = graph.add("BasicScheduler", model=refinement_model, scheduler=refinement_scheduler,
                                     steps=hires["steps"], denoise=1.0)
        split = graph.add("SplitSigmasDenoise", sigmas=schedule, denoise=hires["strength"])
        refinement_noise = graph.add("RandomNoise", noise_seed=seed)
        selected_sampler = graph.add("KSamplerSelect", sampler_name=refinement_sampler)
        refined = graph.add("SamplerCustomAdvanced", noise=refinement_noise, guider=refinement_guider,
                            sampler=selected_sampler, sigmas=[split[0], 1], latent_image=encoded_image)
        decoded = graph.add("VAEDecode", samples=refined, vae=vae)
    graph.add("SaveImage", images=decoded, filename_prefix="iiLocalDiffusion")
    return graph.nodes
