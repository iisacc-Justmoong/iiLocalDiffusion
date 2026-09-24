"""Lightweight argument contract for local checkpoint arithmetic; no ML imports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from numbers import Real
from pathlib import Path

from checkpoint_conversion import LEGACY_SUFFIXES, SAFETENSORS_SUFFIXES

MODES = ("weighted-sum", "weighted-difference", "unified")
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
    compatibility_models: tuple[Path, ...] | None = None
    compatibility_strength: float = 0.35
    lora_policy: str = "strict"
    checkpoint_policy: str = "common-layer"
    weight_normalization: dict | None = None

    @property
    def automatic_repair(self) -> bool:
        return self.checkpoint_policy == "common-layer" and self.mode != "unified"

    @property
    def models(self) -> tuple[Path, ...]:
        return (self.base_model, *self.additional_models)

    def as_dict(self) -> dict:
        return {
            "lora_policy": self.lora_policy,
            "checkpoint_policy": self.checkpoint_policy,
            "base_model": str(self.base_model),
            "additional_models": [str(path) for path in self.additional_models],
            "mode": self.mode,
            "weight_semantics": "checkpoint-refinement-strength/lora-delta-scale" if self.mode == "unified" else "tensor-coefficient",
            "weights": list(self.weights) if self.weights is not None else None,
            "base_weight": self.base_weight,
            "weight_normalization": self.weight_normalization,
            "repair_policy": "best-effort-v1" if self.automatic_repair else "strict",
            "coefficient_resolution": "resolved" if self.base_weight is not None else "after-input-inspection",
            "output": str(self.output),
            "cache_dir": str(self.cache_dir),
            "compatibility_models": ([str(path) for path in self.compatibility_models]
                                     if self.compatibility_models is not None else None),
            "compatibility_policy": ("nearby-checkpoints" if self.compatibility_models is None else "explicit-candidates")
                                    if self.mode == "unified" else
                                    ("base-layout-common-layer" if self.checkpoint_policy == "common-layer"
                                     else "strict-weight-compatibility"),
            "compatibility_strength": self.compatibility_strength,
        }


def _path(value: str | Path, role: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "://" in str(value):
        raise ValueError(f"{role} requires a nonempty local path.")
    return Path(value).expanduser().absolute()


def _model(value: str | Path, *, base: bool = False, allow_unusable=False) -> Path:
    path = _path(value, "Model")
    if allow_unusable and not base:
        # Selected local materials may become empty, unreadable or disappear.
        # Their failures are reported by inspection, not hidden by argument parsing.
        return path
    if path.is_dir():
        if not (path / "model_index.json").is_file() and (base or not (
                (path / "adapter_config.json").is_file()
                or any(path.glob("*.safetensors")) or any(path.glob("*.safetensor")))):
            raise ValueError(f"A base directory requires model_index.json; an adapter directory requires LoRA weights: {path}")
    elif (not path.is_file() or path.stat().st_size == 0
          or path.suffix.lower() not in (*SAFETENSORS_SUFFIXES, *LEGACY_SUFFIXES, ".iildmodel")):
        raise ValueError(f"Model must be a nonempty safetensors/ckpt/pt/pth/bin/.iildmodel file or Diffusers directory: {path}")
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
    compatibility_models: Sequence[str | Path] | None = None,
    compatibility_strength: float = 0.35,
    lora_policy: str = "strict",
    checkpoint_policy: str = "common-layer",
) -> MergeRequest:
    """Validate paths and requested strengths without reading model tensors.

    A scalar (or one-element sequence) broadcasts to every additional model.
    Effective defaults and the base coefficient are resolved after checkpoint
    versus LoRA inspection. Inputs are never modified.
    """
    if lora_policy not in ("strict", "synthetic"):
        raise ValueError("LoRA policy must be strict or synthetic.")
    if checkpoint_policy not in ("strict", "common-layer"):
        raise ValueError("Checkpoint policy must be strict or common-layer.")
    if mode not in MODES:
        raise ValueError(f"Unknown merge mode {mode!r}; choose one of {MODES}.")
    if isinstance(additional_models, (str, bytes, Path)) or not isinstance(additional_models, Sequence):
        raise TypeError("additional_models must be a sequence of local model paths.")
    base = _model(base_model, base=True)
    additional = tuple(_model(value, allow_unusable=checkpoint_policy == "common-layer" and mode != "unified")
                       for value in (additional_model, *additional_models))
    if (isinstance(compatibility_strength, bool) or not isinstance(compatibility_strength, Real)
            or not math.isfinite(compatibility_strength) or not 0 <= compatibility_strength <= 1):
        raise ValueError("Compatibility refinement strength must be finite and in [0, 1].")
    candidates = None
    if compatibility_models is not None:
        if mode != "unified":
            raise ValueError("Compatibility checkpoints require --mode unified; weight arithmetic cannot bridge architectures.")
        if isinstance(compatibility_models, (str, bytes, Path)) or not isinstance(compatibility_models, Sequence):
            raise TypeError("compatibility_models must be a sequence of local checkpoint paths.")
        candidates = tuple(dict.fromkeys(_model(value, base=True) for value in compatibility_models))
        if any(path.is_dir() or path.suffix.lower() not in SAFETENSORS_SUFFIXES for path in candidates):
            raise ValueError("Compatibility checkpoints must be single-file safetensors exports.")
    directory = base.is_dir()
    package_directory = directory and base.suffix.lower() == ".iildmodel"
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
    default_name = f"{base.stem if not directory else base.name.removesuffix('.iildmodel')}-{mode}"
    destination = _path(output, "Output") if output is not None else (
        DEFAULT_DIRECTORY / "merged" / (default_name + ".iildmodel" if package_directory
                                          else default_name if directory else default_name + ".safetensors"))
    if mode == "unified":
        destination = _path(output, "Output") if output is not None else DEFAULT_DIRECTORY / "merged" / (default_name + ".iildmodel")
        if destination.suffix.lower() != ".iildmodel":
            raise ValueError("A unified model output must be a new .iildmodel package file.")
    elif package_directory and destination.suffix.lower() != ".iildmodel":
        raise ValueError("A weighted legacy .iildmodel package output must end in .iildmodel.")
    elif not directory and destination.suffix.lower() not in SAFETENSORS_SUFFIXES:
        raise ValueError("A merged checkpoint output must end in .safetensors or .safetensor.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Merge output already exists; choose a new path: {destination}")
    cache = _path(cache_dir, "Conversion cache") if cache_dir is not None else DEFAULT_DIRECTORY / "model-merge-cache"
    for source in (base, *additional, *(candidates or ())):
        resolved = source.resolve()
        if destination.resolve() == resolved or (source.is_dir() and destination.resolve().is_relative_to(resolved)):
            raise ValueError("Merge output must not overlap or be inside an input model.")
        if source.is_dir() and cache.resolve().is_relative_to(resolved):
            raise ValueError("Conversion cache must not be inside an input model.")
    return MergeRequest(base, additional, mode, values, None, destination, cache, candidates,
                        float(compatibility_strength), lora_policy, checkpoint_policy)


def resolve_merge_weights(request: MergeRequest, kinds: Sequence[str]) -> MergeRequest:
    """LoRAs add deltas; only full checkpoints take a share of the base."""
    if len(kinds) != len(request.additional_models) or any(kind not in ("checkpoint", "lora") for kind in kinds):
        raise ValueError("Expected one checkpoint/LoRA kind per additional model.")
    count = kinds.count("checkpoint")
    default = 0.35 if request.mode == "unified" else 1 / (count + 1) if request.mode == "weighted-sum" else 0.5
    values = request.weights or tuple(1.0 if kind == "lora" else default for kind in kinds)
    base_weight = 1.0
    normalization = None
    if request.mode == "unified" and any(value > 1 for value, kind in zip(values, kinds) if kind == "checkpoint"):
        raise ValueError("Unified checkpoint refinement strengths must be in [0, 1]; LoRA strengths may exceed 1.")
    if request.mode == "weighted-sum":
        try:
            total = math.fsum(value for value, kind in zip(values, kinds) if kind == "checkpoint")
        except OverflowError:
            total = math.inf
        if total > 1.0 + 1e-12:
            if not request.automatic_repair:
                raise ValueError("The sum of additional checkpoint weights must be <= 1.")
            largest = max(value for value, kind in zip(values, kinds) if kind == "checkpoint")
            denominator = math.fsum(value / largest for value, kind in zip(values, kinds) if kind == "checkpoint")
            normalized = tuple(value / largest / denominator if kind == "checkpoint" else value
                               for value, kind in zip(values, kinds))
            normalization = {"policy": "checkpoint-ratios-total-one", "requested_weights": list(values),
                             "effective_weights": list(normalized)}
            values, total = normalized, 1.0
        base_weight = max(0.0, 1.0 - total)
    return replace(request, weights=values, base_weight=base_weight, weight_normalization=normalization)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge compatible model weights or build an ordered unified model cascade.")
    parser.add_argument("--base-model", required=True, help="Base checkpoint or Diffusers directory (A).")
    parser.add_argument("--additional-model", required=True, action="append", dest="additional_models",
                        help="Checkpoint or LoRA file/directory; repeat for more materials (at least one required).")
    parser.add_argument("--mode", choices=MODES, default="weighted-sum", help="Unified mode preserves independent architectures and refines images in order.")
    parser.add_argument("--weights", "--weight", nargs="+", type=float,
                        help="One strength for all materials, or one per material. Default LoRA strength: 1.")
    parser.add_argument("--output", help="New output file/directory; default: build/reference/merged/.")
    parser.add_argument("--cache-dir", help="Cache for existing safe legacy-checkpoint conversion.")
    compatibility = parser.add_mutually_exclusive_group()
    compatibility.add_argument("--compatibility-model", action="append", dest="compatibility_models",
                               help="Local checkpoint candidate for otherwise unmatched LoRAs in unified mode; repeat as needed. Default: nearby checkpoint files.")
    compatibility.add_argument("--no-auto-compatibility", action="store_const", const=[], dest="compatibility_models",
                               help="Require every LoRA to match an explicitly supplied material checkpoint.")
    parser.add_argument("--compatibility-strength", type=float, default=0.35,
                        help="Image refinement strength of automatically inserted compatibility stages, in [0,1] (default: 0.35).")
    parser.add_argument("--lora-policy", choices=("strict", "synthetic"), default="strict",
                        help="Applied only after base compatibility selection; unmatched LoRAs are excluded before either policy runs.")
    parser.add_argument("--checkpoint-policy", choices=("strict", "common-layer"), default="common-layer",
                        help="Within the base ecosystem, common-layer automatically resamples differing tensor coordinates onto the base layout (default); strict requires an exact layout.")
    parser.add_argument("--print-config", action="store_true", help="Validate arguments without loading tensors.")
    parser.add_argument("--inspect", action="store_true", help="Inspect structural compatibility and LoRA routing before hashing or writing output.")
    return parser
