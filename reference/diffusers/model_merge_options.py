"""Lightweight argument contract for local checkpoint arithmetic; no ML imports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from numbers import Real
from pathlib import Path

from checkpoint_conversion import LEGACY_SUFFIXES, SAFETENSORS_SUFFIXES

MODES = ("weighted-sum", "weighted-difference")
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "build/reference"


@dataclass(frozen=True)
class MergeRequest:
    base_model: Path
    additional_models: tuple[Path, ...]
    mode: str
    weights: tuple[float, ...] | None
    base_weight: float | None
    output: Path
    cache_dir: Path

    @property
    def models(self) -> tuple[Path, ...]:
        return (self.base_model, *self.additional_models)

    def as_dict(self) -> dict:
        return {
            "base_model": str(self.base_model),
            "additional_models": [str(path) for path in self.additional_models],
            "mode": self.mode,
            "weights": list(self.weights) if self.weights is not None else None,
            "base_weight": self.base_weight,
            "coefficient_resolution": "resolved" if self.base_weight is not None else "after-input-inspection",
            "output": str(self.output),
            "cache_dir": str(self.cache_dir),
        }


def _path(value: str | Path, role: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "://" in str(value):
        raise ValueError(f"{role} requires a nonempty local path.")
    return Path(value).expanduser().absolute()


def _model(value: str | Path, *, base: bool = False) -> Path:
    path = _path(value, "Model")
    if path.is_dir():
        if not (path / "model_index.json").is_file() and (base or not (
                (path / "adapter_config.json").is_file()
                or any(path.glob("*.safetensors")) or any(path.glob("*.safetensor")))):
            raise ValueError(f"A base directory requires model_index.json; an adapter directory requires LoRA weights: {path}")
    elif (not path.is_file() or path.stat().st_size == 0
          or path.suffix.lower() not in (*SAFETENSORS_SUFFIXES, *LEGACY_SUFFIXES)):
        raise ValueError(f"Model must be a nonempty safetensors/ckpt/pt/pth/bin file or Diffusers directory: {path}")
    return path


def resolve_merge_request(
    base_model: str | Path,
    additional_model: str | Path,
    *,
    additional_models: Sequence[str | Path] = (),
    weights: float | Sequence[float] | None = None,
    mode: str = "weighted-sum",
    output: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> MergeRequest:
    """Validate paths and requested strengths without reading model tensors.

    A scalar (or one-element sequence) broadcasts to every additional model.
    Effective defaults and the base coefficient are resolved after checkpoint
    versus LoRA inspection. Inputs are never modified.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown merge mode {mode!r}; choose one of {MODES}.")
    if isinstance(additional_models, (str, bytes, Path)) or not isinstance(additional_models, Sequence):
        raise TypeError("additional_models must be a sequence of local model paths.")
    base = _model(base_model, base=True)
    additional = tuple(_model(value) for value in (additional_model, *additional_models))
    directory = base.is_dir()
    count = len(additional)
    if weights is None:
        values = None
    else:
        if isinstance(weights, Real):
            values = (weights,) * count
        elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
            values = tuple(weights)
            if len(values) == 1:
                values *= count
        else:
            raise TypeError("weights must be a finite number or a sequence of finite numbers.")
        if len(values) != count:
            raise ValueError("Supply one weight to broadcast or exactly one weight per additional model.")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Each weight must be a finite nonnegative number, not a boolean.")
    values = tuple(float(value) for value in values) if values is not None else None
    default_name = f"{base.stem if not directory else base.name}-{mode}"
    destination = _path(output, "Output") if output is not None else (
        DEFAULT_DIRECTORY / "merged" / (default_name if directory else default_name + ".safetensors"))
    if not directory and destination.suffix.lower() not in SAFETENSORS_SUFFIXES:
        raise ValueError("A merged checkpoint output must end in .safetensors or .safetensor.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Merge output already exists; choose a new path: {destination}")
    cache = _path(cache_dir, "Conversion cache") if cache_dir is not None else DEFAULT_DIRECTORY / "model-merge-cache"
    for source in (base, *additional):
        resolved = source.resolve()
        if destination.resolve() == resolved or (source.is_dir() and destination.resolve().is_relative_to(resolved)):
            raise ValueError("Merge output must not overlap or be inside an input model.")
        if source.is_dir() and cache.resolve().is_relative_to(resolved):
            raise ValueError("Conversion cache must not be inside an input model.")
    return MergeRequest(base, additional, mode, values, None, destination, cache)


def resolve_merge_weights(request: MergeRequest, kinds: Sequence[str]) -> MergeRequest:
    """LoRAs add deltas; only full checkpoints take a share of the base."""
    if len(kinds) != len(request.additional_models) or any(kind not in ("checkpoint", "lora") for kind in kinds):
        raise ValueError("Expected one checkpoint/LoRA kind per additional model.")
    count = kinds.count("checkpoint")
    default = 1 / (count + 1) if request.mode == "weighted-sum" else 0.5
    values = request.weights or tuple(1.0 if kind == "lora" else default for kind in kinds)
    base_weight = 1.0
    if request.mode == "weighted-sum":
        try:
            total = math.fsum(value for value, kind in zip(values, kinds) if kind == "checkpoint")
        except OverflowError as error:
            raise ValueError("The sum of additional checkpoint weights must be <= 1.") from error
        if total > 1.0 + 1e-12:
            raise ValueError("The sum of additional checkpoint weights must be <= 1.")
        base_weight = max(0.0, 1.0 - total)
    return replace(request, weights=values, base_weight=base_weight)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge local model weights with a weighted sum or subtraction.")
    parser.add_argument("--base-model", required=True, help="Base checkpoint or Diffusers directory (A).")
    parser.add_argument("--additional-model", required=True, action="append", dest="additional_models",
                        help="Checkpoint or LoRA file/directory; repeat for more materials (at least one required).")
    parser.add_argument("--mode", choices=MODES, default="weighted-sum")
    parser.add_argument("--weights", "--weight", nargs="+", type=float,
                        help="One strength for all materials, or one per material. Default LoRA strength: 1.")
    parser.add_argument("--output", help="New output file/directory; default: build/reference/merged/.")
    parser.add_argument("--cache-dir", help="Cache for existing safe legacy-checkpoint conversion.")
    parser.add_argument("--print-config", action="store_true", help="Validate arguments without loading tensors.")
    return parser
