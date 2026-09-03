"""Compose selected checkpoint, denoiser and VAE weights through Diffusers APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipeline_loading import _is_gated_repository_error, load_pipeline
from presets import ModelSelection, PipelinePreset, validate_vae_contract
from weight_files import LocalWeightFile, checked_safetensors_path, weight_file_metadata


CONFIG_PATTERNS = ("*.json", "**/*.json", "*.txt", "**/*.txt", "**/*.model")


def selection_metadata(selection: ModelSelection) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "id_or_path": selection.source,
        "is_local": selection.is_local,
        "requested_revision": selection.requested_revision,
        "format": "diffusers",
    }
    if selection.single_file is not None:
        metadata.update(weight_file_metadata(selection.single_file))
    return metadata


def download_configuration(
    selection: ModelSelection, cache_directory: Path, local_files_only: bool
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        selection.source,
        revision=selection.requested_revision,
        cache_dir=cache_directory,
        local_files_only=local_files_only,
        allow_patterns=list(CONFIG_PATTERNS),
    )


def resolve_configuration_directory(
    selection: ModelSelection,
    preset: PipelinePreset,
    cache_directory: Path,
    local_files_only: bool,
) -> Path:
    root = Path(
        selection.source
        if selection.is_local
        else download_configuration(selection, cache_directory, local_files_only)
    ).resolve()
    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise ValueError(f"Model configuration requires a local model_index.json: {root}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("_class_name") != preset.pipeline_class:
        raise ValueError(f"Model configuration must describe {preset.pipeline_class}: {root}")

    allowed: dict[str, tuple[str, ...]] = {}
    for name, class_name in preset.expected_components:
        library = (
            "transformers"
            if name.startswith(("text_encoder", "tokenizer"))
            else "diffusers"
        )
        allowed[name] = (library, class_name)
        expected = [library, class_name]
        actual = index.get(name)
        if class_name == "T5TokenizerFast" and actual == [library, "T5Tokenizer"]:
            continue
        if actual != expected:
            raise ValueError(f"Unexpected {name} in model configuration: {actual}")

    allowed.update(
        safety_checker=("stable_diffusion", "StableDiffusionSafetyChecker"),
        feature_extractor=("transformers", "CLIPImageProcessor"),
    )
    for name, value in index.items():
        if not isinstance(value, list) or len(value) != 2 or value == [None, None]:
            continue
        if not all(isinstance(member, str) for member in value):
            continue
        if name not in allowed or tuple(value) != allowed[name]:
            if name == "tokenizer_2" and value == ["transformers", "T5Tokenizer"]:
                continue
            raise ValueError(f"Unsupported component declaration in model configuration: {name}")
    return root


def read_weight_keys(path: Path) -> set[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        return set(weights.keys())


def classify_model_weights(preset: PipelinePreset, keys: set[str]) -> str:
    if preset.name == "flux1-schnell":
        if any(key.endswith(("img_in.weight", "x_embedder.weight")) for key in keys):
            return "transformer"
    else:
        if "conv_in.weight" in keys and any(key.startswith("down_blocks.") for key in keys):
            return "unet"
        if any(key.startswith("model.diffusion_model.") for key in keys):
            if any(
                key.startswith(("cond_stage_model.", "conditioner.embedders.")) for key in keys
            ):
                return "checkpoint"
            return "unet"
    raise ValueError(
        f"--model does not contain recognized {preset.name} checkpoint/denoiser weights; "
        "VAE and LoRA files belong in --vae and --lora."
    )


def require_checkpoint_components(
    preset: PipelinePreset, keys: set[str], has_vae_override: bool
) -> None:
    text_encoder_keys = (
        ("cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",)
        if preset.name == "sd15"
        else (
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
            "conditioner.embedders.1.model.positional_embedding",
        )
    )
    # Diffusers uses these tensors to recognize embedded CLIP weights. Without them
    # it can silently load a different text encoder from the configuration source.
    for key in text_encoder_keys:
        if key not in keys:
            raise ValueError(f"Single-file checkpoint is missing text encoder weights: {key}")
    if not has_vae_override and not any(key.startswith("first_stage_model.") for key in keys):
        raise ValueError("Single-file checkpoint is missing VAE weights; provide --vae.")


def require_materialized_component(component: Any, name: str) -> None:
    for tensor_name, tensor in (
        *component.named_parameters(),
        *component.named_buffers(),
    ):
        if tensor.is_meta:
            raise RuntimeError(f"Missing weights left {name}.{tensor_name} on the meta device.")


def load_single_component(
    component_class: Any,
    weights: LocalWeightFile,
    configuration_directory: Path,
    component_name: str,
    load_arguments: Mapping[str, Any],
) -> Any:
    cache_directory = Path(load_arguments["cache_dir"])
    with checked_safetensors_path(
        weights, cache_directory / "single-file-aliases", component_name
    ) as path:
        component = component_class.from_single_file(
            str(path),
            config=str(configuration_directory),
            subfolder=component_name,
            dtype=load_arguments["dtype"],
            cache_dir=cache_directory,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        require_materialized_component(component, component_name)
    return component


def _arguments_for_source(
    source: ModelSelection, load_arguments: Mapping[str, Any]
) -> dict[str, Any]:
    arguments = dict(load_arguments)
    arguments.pop("revision", None)
    if source.requested_revision is not None:
        arguments["revision"] = source.requested_revision
    return arguments


def _load_checkpoint_pipeline(
    pipeline_class: Any,
    preset: PipelinePreset,
    selection: ModelSelection,
    config_selection: ModelSelection,
    config_directory: Path,
    overrides: dict[str, Any],
    load_arguments: Mapping[str, Any],
    component_classes: Mapping[str, Any],
) -> Any:
    if preset.name == "sd15":
        index = json.loads((config_directory / "model_index.json").read_text(encoding="utf-8"))
        if index.get("safety_checker") not in (None, [None, None]):
            safety_arguments = _arguments_for_source(config_selection, load_arguments)
            safety_arguments.pop("add_watermarker", None)
            variant = safety_arguments.get("variant")
            if variant is not None and config_selection.is_local:
                # Direct component loads do not perform the pipeline loader's
                # per-component variant selection for mixed local packages.
                checker_directory = config_directory / "safety_checker"
                if not any(checker_directory.glob(f"*.{variant}*.safetensors")):
                    safety_arguments.pop("variant")
            overrides["safety_checker"] = component_classes[
                "StableDiffusionSafetyChecker"
            ].from_pretrained(
                config_selection.source, subfolder="safety_checker", **safety_arguments
            )

    arguments = {
        "config": str(config_directory),
        "cache_dir": load_arguments["cache_dir"],
        "dtype": load_arguments["dtype"],
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "trust_remote_code": False,
        **overrides,
    }
    if preset.name == "sdxl-base":
        arguments["add_watermarker"] = False
    with checked_safetensors_path(
        selection.single_file,
        Path(load_arguments["cache_dir"]) / "single-file-aliases",
        "model",
    ) as path:
        pipeline = pipeline_class.from_single_file(str(path), **arguments)
        for name, component in pipeline.components.items():
            if hasattr(component, "named_parameters"):
                require_materialized_component(component, name)
    return pipeline


def load_generation_pipeline(
    pipeline_class: Any,
    preset: PipelinePreset,
    selection: ModelSelection,
    config_selection: ModelSelection | None,
    vae_file: LocalWeightFile | None,
    load_arguments: Mapping[str, Any],
    component_classes: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    kind = "diffusers"
    configuration = None
    config_directory = None
    overrides: dict[str, Any] = {}
    try:
        if selection.single_file is not None:
            keys = read_weight_keys(Path(selection.single_file.path))
            kind = classify_model_weights(preset, keys)
            if kind == "checkpoint":
                require_checkpoint_components(preset, keys, vae_file is not None)
            if config_selection is None:
                raise ValueError("A single-file model requires a resolved configuration source.")
            configuration = config_selection
        elif vae_file is not None:
            configuration = selection

        if configuration is not None:
            config_directory = resolve_configuration_directory(
                configuration,
                preset,
                Path(load_arguments["cache_dir"]),
                load_arguments["local_files_only"],
            )
        if vae_file is not None:
            overrides["vae"] = load_single_component(
                component_classes["AutoencoderKL"],
                vae_file,
                config_directory,
                "vae",
                load_arguments,
            )
            validate_vae_contract(overrides["vae"], preset)

        if kind == "checkpoint":
            pipeline = _load_checkpoint_pipeline(
                pipeline_class,
                preset,
                selection,
                configuration,
                config_directory,
                overrides,
                load_arguments,
                component_classes,
            )
        else:
            source = selection
            if kind in ("unet", "transformer"):
                component_class = component_classes[
                    "UNet2DConditionModel" if kind == "unet" else "FluxTransformer2DModel"
                ]
                overrides[kind] = load_single_component(
                    component_class,
                    selection.single_file,
                    config_directory,
                    kind,
                    load_arguments,
                )
                source = configuration
            pipeline = load_pipeline(
                pipeline_class,
                source.source,
                {**_arguments_for_source(source, load_arguments), **overrides},
            )
    except Exception as error:
        if _is_gated_repository_error(error):
            raise SystemExit(
                "Cannot access a gated model/configuration source. Accept its terms and "
                "authenticate with `hf auth login`, then retry."
            ) from None
        raise RuntimeError(f"Could not assemble the selected model/VAE: {error}") from error

    origins = {name: "model" for name, value in pipeline.components.items() if value is not None}
    if kind in ("unet", "transformer"):
        origins = {name: "model_config" for name in origins}
        origins[kind] = "model"
    elif kind == "checkpoint":
        embedded = {"unet", "text_encoder", "text_encoder_2", "vae"}
        origins = {name: "model" if name in embedded else "model_config" for name in origins}
    if vae_file is not None:
        origins["vae"] = "vae_override"

    return pipeline, {
        "weights_role": kind,
        "configuration": None if configuration is None else selection_metadata(configuration),
        "configuration_directory": None if config_directory is None else str(config_directory),
        "component_sources": origins,
        "vae_override": None if vae_file is None else weight_file_metadata(vae_file),
    }
