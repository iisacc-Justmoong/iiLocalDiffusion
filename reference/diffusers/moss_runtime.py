"""Offline loader for the pinned OpenMOSS implementation, never package code."""
from __future__ import annotations

from pathlib import Path


COMPONENTS = {
    "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
    "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
    "text_encoder": ["transformers", "Qwen3_5Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def validate_index(index):
    if index.get("_class_name") != "MOSSImagePipeline":
        raise ValueError("The vetted MOSS loader requires MOSSImagePipeline.")
    for name, expected in COMPONENTS.items():
        if index.get(name) != expected:
            raise ValueError(f"MOSS component {name} must be {expected}.")
    extra = {name for name, value in index.items()
             if not name.startswith("_") and isinstance(value, (tuple, list)) and name not in COMPONENTS}
    if extra:
        raise ValueError("Unexpected MOSS pipeline components: " + ", ".join(sorted(extra)))


def pipeline_class():
    from vendor_moss.pipeline import MOSSImagePipeline
    return MOSSImagePipeline


def strict_load(component_class, path, role, **kwargs):
    model, info = component_class.from_pretrained(str(path), output_loading_info=True, **kwargs)
    if not isinstance(info, dict) or any(info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"MOSS {role} weights do not exactly match their configuration: {info}")
    from model_loading import require_materialized_component
    require_materialized_component(model, role)
    return model.eval()


def load_scheduler(folder):
    """Load only schedules compatible with the pinned MOSS Euler implementation."""
    from vendor_moss.pipeline import MOSSImageScheduler
    scheduler = MOSSImageScheduler.from_pretrained(str(folder / 'scheduler'), local_files_only=True)
    # Diffusers retains unexpected config keys even when it warns that they
    # are ignored; the inherited Euler scheduler can still act on those keys.
    for key in ('use_dynamic_shifting', 'invert_sigmas', 'shift_terminal',
                'use_karras_sigmas', 'use_exponential_sigmas', 'use_beta_sigmas',
                'stochastic_sampling'):
        if getattr(scheduler.config, key, None):
            raise ValueError(f'MOSS scheduler {key} is incompatible with the pinned flow schedule.')
    return scheduler


def load(folder: Path, dtype, vae_selection, diffusers, *, transformer_file=None, text_encoder_file=None,
         tokenizer_selection=None, text_config_selection=None, transformer_storage='dense', progress=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from vae_defaults import verify_vae_selection
    from vendor_moss.pipeline import MOSSImageTransformer, MOSSImagePipeline

    if transformer_file is None and transformer_storage != 'dense':
        raise ValueError('MOSS transformer cpu-stream storage requires a single-file checkpoint.')
    scheduler = load_scheduler(folder)
    common = {"local_files_only": True, "trust_remote_code": False}
    if transformer_file is not None:
        from moss_single_file import load as load_transformer
        transformer = load_transformer(transformer_file, folder / 'transformer', dtype,
                                       storage=transformer_storage, progress=progress)
    else:
        transformer = strict_load(MOSSImageTransformer, folder / "transformer", "transformer", dtype=dtype,
                                  local_files_only=True, use_safetensors=True)
    if tokenizer_selection is not None:
        from moss_tokenizer import load as load_tokenizer
        tokenizer = load_tokenizer(tokenizer_selection)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"), **common)
    if text_encoder_file is not None:
        from moss_text_encoder import load as load_text_encoder, verify_config
        verify_config(text_config_selection)
        text_encoder = load_text_encoder(text_encoder_file, text_config_selection.path if text_config_selection else folder / 'text_encoder',
                                         dtype, progress=progress)
        verify_config(text_config_selection)
    else:
        text_encoder = strict_load(AutoModel, folder / "text_encoder", "text encoder", dtype=dtype,
                                   use_safetensors=True, **common)
        if type(text_encoder).__name__ != "Qwen3_5Model":
            raise ValueError("MOSS requires the Qwen3_5Model text encoder declared by its package.")
    if vae_selection:
        verify_vae_selection(vae_selection)
    vae = strict_load(diffusers.AutoencoderKLFlux2, vae_selection.directory if vae_selection else folder / "vae",
                      "vae", dtype=torch.float32, local_files_only=True, use_safetensors=True)
    if vae_selection:
        verify_vae_selection(vae_selection)
    if (tuple(vae.config.patch_size) != (2, 2) or
            vae.config.latent_channels * 4 != transformer.config.latent_channels or
            vae.bn.num_features != transformer.config.latent_channels):
        raise ValueError("MOSS transformer and FLUX.2 VAE packed latent channels do not match.")
    return MOSSImagePipeline(transformer=transformer, scheduler=scheduler,
                             tokenizer=tokenizer, text_encoder=text_encoder, vae=vae)
