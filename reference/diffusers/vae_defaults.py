"""Local VAE component selection before Diffusers constructs a pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

from generation_defaults import checked_resource, read_defaults, canonical_lora_family
from weight_files import LocalWeightFile, cached_model_sha256, resolve_weight_file, file_signature

# RGB Qwen Image 1.x pipelines share the published 16-channel latent space.
# Layered requires an RGBA VAE; similarly sized latents from other families are
# not compatible. Do not infer compatibility from a class-name prefix.
QWEN_RGB_PIPELINES = frozenset({
    "QwenImagePipeline", "QwenImageImg2ImgPipeline", "QwenImageInpaintPipeline",
    "QwenImageEditPipeline", "QwenImageEditPlusPipeline", "QwenImageEditInpaintPipeline",
    "QwenImageControlNetPipeline", "QwenImageControlNetInpaintPipeline",
})

VAE_PIPELINES = {
    "qwen-image": QWEN_RGB_PIPELINES,
    "sdxl-base": frozenset({
        "StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLInstructPix2PixPipeline", "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline", "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLAdapterPipeline", "StableDiffusionXLControlNetUnionPipeline",
        "StableDiffusionXLControlNetUnionImg2ImgPipeline", "StableDiffusionXLControlNetUnionInpaintPipeline",
    }),
    "flux1": frozenset({
        "FluxPipeline", "FluxImg2ImgPipeline", "FluxInpaintPipeline", "FluxFillPipeline",
        "FluxKontextPipeline", "FluxKontextInpaintPipeline", "FluxControlPipeline",
        "FluxControlImg2ImgPipeline", "FluxControlInpaintPipeline", "FluxControlNetPipeline",
        "FluxControlNetImg2ImgPipeline", "FluxControlNetInpaintPipeline",
    }),
    "flux2": frozenset({"Flux2Pipeline", "Flux2KleinPipeline", "Flux2KleinKVPipeline", "Flux2KleinInpaintPipeline"}),
}
VAE_CLASSES = {"qwen-image": "AutoencoderKLQwenImage", "sdxl-base": "AutoencoderKL",
               "flux1": "AutoencoderKL", "flux2": "AutoencoderKLFlux2"}


def pipeline_vae_family(name):
    return next((family for family, names in VAE_PIPELINES.items() if name in names), None)


@dataclass(frozen=True)
class VaeSelection:
    directory: str
    class_name: str
    weight: LocalWeightFile
    config_path: str
    config_sha256: str


def single_file_has_vae(path):
    with Path(path).open("rb") as source:
        size = source.read(8)
        if len(size) != 8:
            raise ValueError("Invalid safetensors model header while checking its VAE.")
        length = struct.unpack("<Q", size)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > Path(path).stat().st_size:
            raise ValueError("Invalid safetensors model header while checking its VAE.")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors model header while checking its VAE.")
    return any(key.startswith(("first_stage_model.", "vae.")) for key in header)


def _select_directory(directory, expected_class, family):
    directory = Path(directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--vae requires a local Diffusers VAE directory with config.json and safetensors weights.")
    config_path = directory / "config.json"
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("VAE configuration is too large.")
    before = file_signature(config_path)
    config = json.loads(config_path.read_text())
    digest = cached_model_sha256(config_path)
    if file_signature(config_path) != before:
        raise RuntimeError("VAE configuration changed while reading.")
    if not isinstance(config, dict) or config.get("_class_name") != expected_class:
        raise ValueError(f"VAE must use the pipeline's {expected_class} architecture.")
    if family == "qwen-image" and (config.get("z_dim") != 16 or config.get("input_channels", 3) != 3 or config.get("in_channels", 3) != 3
                     or config.get("out_channels", 3) != 3):
        raise ValueError("Qwen Image RGB VAE requires z_dim=16 and three image channels.")
    if family in ("sdxl-base", "flux1", "flux2"):
        channels = {"sdxl-base": 4, "flux1": 16, "flux2": 32}[family]
        if (config.get("latent_channels") != channels or config.get("in_channels") != 3
                or config.get("out_channels") != 3 or len(config.get("block_out_channels", [])) != 4):
            raise ValueError(f"VAE does not match the {family} RGB latent space ({channels} channels, 8x downsampling).")
        if family in ("sdxl-base", "flux1"):
            scale, shift = (0.13025, 0.0) if family == "sdxl-base" else (0.3611, 0.1159)
            if (not math.isclose(config.get("scaling_factor", 0), scale, rel_tol=1e-6)
                    or not math.isclose(config.get("shift_factor") or 0, shift, abs_tol=1e-6)):
                raise ValueError(f"VAE scaling/shift does not match {family}.")
        elif config.get("patch_size") != [2, 2]:
            raise ValueError("FLUX.2 VAE requires 2x2 latent patches and its batch normalization.")
    weight = resolve_weight_file(str(directory / "diffusion_pytorch_model.safetensors"), "VAE")
    return VaeSelection(str(directory), expected_class, weight, str(config_path), digest)


def resolve_vae_selection(args, index, folder):
    name = args.pipeline_class or index["_class_name"]
    family = pipeline_vae_family(name)
    component = index.get("vae")
    expected_class = VAE_CLASSES[family] if family else (
        component[1] if isinstance(component, list) and len(component) == 2 and component[0] == "diffusers" else None)
    if getattr(args, "vae", None) is not None:
        if expected_class is None or name == "QwenImageLayeredPipeline":
            raise ValueError("This pipeline has no supported explicit VAE override contract.")
        return _select_directory(args.vae, expected_class, family), "explicit"
    if not family:
        return None, "not-configured-for-pipeline"
    # A partial/damaged embedded component must fail in its own loader instead
    # of being silently replaced. Config-only exports genuinely lack weights.
    if args.source_kind == "single-file":
        if single_file_has_vae(args.model):
            return None, "model"
    vae_folder = Path(folder) / "vae"
    if component not in (None, [None, None]) and any(
            path.is_file() and (path.suffix in (".safetensors", ".bin", ".pt", ".ckpt")
                                or path.name.endswith(".index.json")) for path in vae_folder.glob("*")):
        return None, "model"
    if component not in (None, [None, None], ["diffusers", expected_class]):
        raise ValueError(f"The {family} model declares an incompatible VAE architecture.")
    return select_fallback_vae(family, getattr(args, "generation_resources", None)), "fallback"


def select_fallback_vae(family, directory=None):
    family = canonical_lora_family(family)
    root, manifest = read_defaults(directory)
    entries = manifest.get("fallback_vaes", [])
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("Invalid fallback_vaes registry.")
    entries = ([manifest["fallback_vae"]] if "fallback_vae" in manifest else []) + entries
    registry = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("families"), list) or not item["families"]:
            raise ValueError("Each fallback_vae requires explicit compatible families.")
        for member in item["families"]:
            if not isinstance(member, str):
                raise ValueError("Invalid fallback_vae family.")
            member = canonical_lora_family(member)
            if member not in VAE_CLASSES or item.get("class_name") != VAE_CLASSES[member]:
                raise ValueError("Incompatible fallback_vae architecture or family.")
            if member in registry:
                raise ValueError("Duplicate fallback_vae family: " + member)
            registry[member] = item
    if family not in registry:
        raise ValueError(f"Missing fallback_vae for {family}; install the iiLocalDiffusion VAE resources.")
    item = registry[family]
    weight = checked_resource(root, item)
    config = item["config"]
    relative = Path(config["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Bundled VAE configuration must use a relative package path.")
    config_path = (root / relative).resolve(strict=True)
    if not config_path.is_relative_to(root) or config_path.stat().st_size != config["size"] \
            or cached_model_sha256(config_path) != config["sha256"]:
        raise ValueError("Bundled VAE configuration differs from its manifest.")
    selection = _select_directory(Path(weight.path).parent, item["class_name"], family)
    if selection.config_path != str(config_path):
        raise ValueError("Bundled VAE weights and configuration must share a directory.")
    return selection


def verify_vae_selection(selection):
    if selection is None:
        return
    if (resolve_weight_file(selection.weight.path, "VAE") != selection.weight
            or cached_model_sha256(Path(selection.config_path)) != selection.config_sha256):
        raise RuntimeError("The selected VAE changed while loading or generating.")


def resolve_preset_vae(preset, args):
    if args.vae_file is not None:
        return None, "explicit"
    if pipeline_vae_family(preset.pipeline_class) is None:
        return None, "not-configured-for-pipeline"
    source = args.config_selection or args.model_selection
    folder = Path(source.source)
    index_path = folder / "model_index.json"
    # Preset requests already supply a fixed architecture; full model validation
    # remains at the loading boundary, including for print-config requests.
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"_class_name": preset.pipeline_class}
    request = SimpleNamespace(vae=None, pipeline_class=preset.pipeline_class,
        model=args.model_selection.source,
        source_kind="single-file" if args.model_selection.single_file else "directory",
        generation_resources=getattr(args, "generation_resources", None))
    selection, status = resolve_vae_selection(request, index, folder)
    if (request.source_kind == "single-file" and selection is None and status == "model"
            and not single_file_has_vae(request.model)):
        family = pipeline_vae_family(preset.pipeline_class)
        return _select_directory(folder / "vae", VAE_CLASSES[family], family), "model-config"
    return selection, status


def load_selected_vae(selection, diffusers, dtype):
    verify_vae_selection(selection)
    component_class = getattr(diffusers, selection.class_name)
    component = component_class.from_pretrained(selection.directory, dtype=dtype,
        local_files_only=True, use_safetensors=True, trust_remote_code=False)
    from model_loading import require_materialized_component
    require_materialized_component(component, "vae")
    verify_vae_selection(selection)
    return component


def vae_metadata(selection, status):
    return {"status": status, "selection": asdict(selection) if selection else None}
