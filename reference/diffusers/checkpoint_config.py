"""Resolve offline checkpoint configurations shipped with the SDK, without a server."""

from pathlib import Path

from civitai_catalog import lookup_base_model
from downloaded_model import inspect_downloaded_model
from presets import PRESETS

CONFIGS = Path(__file__).resolve().parent / "configs"
FAMILIES = {"sd15": "sd1", "sdxl-base": "sdxl", "flux1-schnell": "flux1"}


def inspect_checkpoint(model, base_model=None, model_info=None):
    inspection = inspect_downloaded_model(model, model_info)
    if inspection["role"] != "checkpoint":
        raise ValueError(f"This model is a {inspection['role']}. {inspection['role_guidance']}")
    if inspection.get("task") != "text-to-image":
        raise ValueError("This checkpoint requires image conditioning; select its built-in Diffusers pipeline and inputs.")
    selected = inspection.get("preset")
    if base_model:
        record = lookup_base_model(base_model)
        architecture = inspection.get("architecture", "") or ""
        family = "flux1" if architecture.startswith("flux1") else architecture
        if family != record["family"] or (inspection.get("base_model") and inspection["base_model"] != record["name"]):
            raise ValueError("--base-model conflicts with the downloaded tensor architecture or metadata.")
        selected = record["preset"]
    if selected not in PRESETS:
        raise ValueError("This checkpoint needs --backend diffusers with a local --model-config and compatible pipeline. "
                         "Automatic standalone checkpoints support SD 1.x, SDXL and configured FLUX.1.")
    return selected, inspection


def bundled_configuration(model, preset):
    family = FAMILIES.get(preset.family)
    if family not in ("sd1", "sdxl"):
        raise ValueError("This single-file model requires --model-config with local tokenizers and companion weights; "
                         "no model weights are downloaded automatically.")
    inspection = inspect_downloaded_model(model)
    if inspection.get("architecture") != family or inspection.get("role") != "checkpoint":
        raise ValueError("Checkpoint tensor architecture conflicts with the selected preset configuration.")
    folder = CONFIGS / family
    if not (folder / "model_index.json").is_file() or not (folder / "tokenizer/merges.txt").is_file():
        raise ValueError("Bundled checkpoint configuration is missing. Reinstall iiLocalDiffusion or provide --model-config.")
    return str(folder)
