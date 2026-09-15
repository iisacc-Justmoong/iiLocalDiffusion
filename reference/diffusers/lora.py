"""Shared local LoRA selection, activation and provenance for image pipelines."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
from weight_files import (LocalWeightFile, SAFETENSORS_SUFFIXES, cached_model_sha256,
                          checked_safetensors_path, resolve_weight_file, verify_weight_file)

LORA_ADAPTER_NAME = "iild_lora"


@dataclass(frozen=True)
class LoraSelection:
    source: str
    weight_name: str
    requested_revision: str | None
    is_local: bool
    scale: float
    sha256: str | None
    size_bytes: int | None
    local_file: LocalWeightFile | None = None


@dataclass(frozen=True)
class LoraActivation:
    active_adapters: tuple[str, ...]
    registered_components: tuple[str, ...]


def validate_lora_support(pipeline):
    missing = [name for name in ("load_lora_weights", "set_adapters", "get_list_adapters")
               if not callable(getattr(pipeline, name, None))]
    if missing:
        name = getattr(pipeline, "__name__", type(pipeline).__name__)
        raise ValueError(f"{name} does not support verified LoRA loading in the installed Diffusers runtime: "
                         + ", ".join(missing))


def validate_lora_activation(pipeline, activation, stage="Generation"):
    if activation is None:
        return
    for name in activation.registered_components:
        query = getattr(getattr(pipeline, name, None), "active_adapters", None)
        if not callable(query) or not set(activation.active_adapters).issubset(query()):
            raise RuntimeError(f"{stage} lost active LoRA adapters on {name}.")


def verify_lora_identity(selection):
    if selection is not None:
        _verify_local_lora_identity(selection)


def add_lora_options(parser):
    parser.add_argument(
        "--lora",
        default=None,
        help="Local LoRA safetensors file or directory",
    )
    parser.add_argument(
        "--lora-revision",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lora-weight-name",
        default=None,
        help="Exact .safetensors filename for a local LoRA directory",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA adapter scale; defaults to 1.0",
    )


def _validate_lora_weight_name(weight_name: str, *, local: bool = False) -> None:
    suffixes = SAFETENSORS_SUFFIXES if local else (".safetensors",)
    if (
        not weight_name
        or "/" in weight_name
        or "\\" in weight_name
        or Path(weight_name).suffix not in suffixes
    ):
        raise SystemExit(
            "--lora-weight-name must be one .safetensors filename without directories."
        )


def resolve_lora_selection(args: argparse.Namespace) -> LoraSelection | None:
    if args.lora is None:
        if args.lora_revision is not None or args.lora_weight_name is not None:
            raise SystemExit(
                "Using --lora-revision or --lora-weight-name requires --lora."
            )
        if args.lora_scale is not None:
            raise SystemExit("--lora-scale requires --lora.")
        return None

    scale = 1.0 if args.lora_scale is None else args.lora_scale
    if not math.isfinite(scale):
        raise SystemExit("--lora-scale must be finite.")

    source = args.lora
    if not source:
        raise SystemExit("--lora must not be empty.")
    candidate = Path(source).expanduser()
    looks_local = (
        candidate.exists()
        or candidate.is_absolute()
        or source.startswith((".", "~"))
        or source.lower().endswith(SAFETENSORS_SUFFIXES)
    )
    if looks_local and not candidate.exists():
        raise SystemExit(f"Local LoRA path does not exist: {candidate}")

    if candidate.exists():
        if args.lora_revision is not None:
            raise SystemExit("A local LoRA does not accept --lora-revision.")
        if candidate.is_file():
            if args.lora_weight_name is not None:
                raise SystemExit(
                    "A direct LoRA file does not accept --lora-weight-name."
                )
            resolved_file = candidate.absolute()
        elif candidate.is_dir():
            if args.lora_weight_name is None:
                raise SystemExit(
                    "A LoRA directory requires --lora-weight-name."
                )
            _validate_lora_weight_name(args.lora_weight_name, local=True)
            resolved_file = (candidate / args.lora_weight_name).absolute()
        else:
            raise SystemExit(f"Local LoRA path is not a regular file or directory: {candidate}")

        _validate_lora_weight_name(resolved_file.name, local=True)
        if not resolved_file.is_file() or resolved_file.stat().st_size == 0:
            raise SystemExit(f"LoRA safetensors file is missing or empty: {resolved_file}")
        try:
            local_file = resolve_weight_file(str(resolved_file), "--lora")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        return LoraSelection(
            source=str(resolved_file.parent),
            weight_name=resolved_file.name,
            requested_revision=None,
            is_local=True,
            scale=scale,
            sha256=local_file.sha256,
            size_bytes=local_file.size_bytes,
            local_file=local_file,
        )

    raise SystemExit(f"--lora must be an existing local file or directory: {source}")


def _verify_local_lora_identity(selection: LoraSelection) -> None:
    if not selection.is_local:
        return
    if selection.local_file is not None:
        verify_weight_file(selection.local_file, "LoRA")
        return
    path = Path(selection.source) / selection.weight_name
    if (
        not path.is_file()
        or path.stat().st_size != selection.size_bytes
        or cached_model_sha256(path) != selection.sha256
    ):
        raise RuntimeError(
            f"Local LoRA changed after argument resolution: {path}"
        )


def apply_lora(
    pipeline: Any,
    selection: LoraSelection | None,
    cache_directory: Path,
    local_files_only: bool,
    *,
    low_cpu_mem_usage: bool = True,
) -> LoraActivation | None:
    if selection is None:
        return None

    if not selection.is_local:
        raise ValueError("LoRA generation requires local weights.")
    load_arguments: dict[str, Any] = {
        "adapter_name": LORA_ADAPTER_NAME,
        "cache_dir": cache_directory,
        "local_files_only": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "use_safetensors": True,
        "weight_name": selection.weight_name,
    }

    try:
        if selection.local_file is not None:
            with checked_safetensors_path(
                selection.local_file, cache_directory / "single-file-aliases", "LoRA"
            ) as path:
                load_arguments["weight_name"] = path.name
                pipeline.load_lora_weights(str(path.parent), **load_arguments)
        else:
            _verify_local_lora_identity(selection)
            pipeline.load_lora_weights(selection.source, **load_arguments)
            _verify_local_lora_identity(selection)
        pipeline.set_adapters(LORA_ADAPTER_NAME, adapter_weights=selection.scale)
        adapters_by_component = pipeline.get_list_adapters()
    except Exception as error:
        raise RuntimeError(
            f"Could not load and activate LoRA {selection.source}/{selection.weight_name}: "
            f"{error}"
        ) from error

    registered_components = tuple(
        sorted(
            component
            for component, adapter_names in adapters_by_component.items()
            if LORA_ADAPTER_NAME in adapter_names
        )
    )
    if not registered_components:
        raise RuntimeError(
            f"LoRA adapter {LORA_ADAPTER_NAME} was not registered and activated."
        )

    active_adapters: set[str] = set()
    for component in registered_components:
        model_component = getattr(pipeline, component, None)
        active_adapter_query = getattr(model_component, "active_adapters", None)
        if not callable(active_adapter_query):
            raise RuntimeError(
                f"LoRA component {component} cannot report its active adapters."
            )
        component_active_adapters = tuple(active_adapter_query())
        if LORA_ADAPTER_NAME not in component_active_adapters:
            raise RuntimeError(
                f"LoRA adapter {LORA_ADAPTER_NAME} is not active on {component}."
            )
        active_adapters.update(component_active_adapters)

    return LoraActivation(tuple(sorted(active_adapters)), registered_components)


def lora_metadata(
    selection: LoraSelection | None,
    activation: LoraActivation | None,
) -> dict[str, Any] | None:
    if selection is None:
        return None
    if activation is None:
        raise RuntimeError("LoRA metadata requires a verified activation.")
    return {
        "active_adapters": list(activation.active_adapters),
        "adapter_name": LORA_ADAPTER_NAME,
        "format": "safetensors",
        "fused": False,
        "is_local": selection.is_local,
        "registered_components": list(activation.registered_components),
        "requested_revision": selection.requested_revision,
        "resolved_file": (
            (
                selection.local_file.resolved_file
                if selection.local_file is not None
                else str(Path(selection.source) / selection.weight_name)
            )
            if selection.is_local
            else None
        ),
        "scale": selection.scale,
        "sha256": selection.sha256,
        "size_bytes": selection.size_bytes,
        "source": selection.source,
        "type": "lora",
        "weight_name": selection.weight_name,
    }
