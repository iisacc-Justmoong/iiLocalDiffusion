"""Resolve a LoRA's structural gap through a real, locally available checkpoint.

Equivalent export namespaces are mapped by the existing target resolver. A
different network is connected through the unified RGB cascade, never by
reshaping, truncating or guessing unrelated weights.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from model_merge_files import inspect_merge_model, open_merge_weights
from model_merge_lora import is_lora_key, prepare_lora
from model_merge_lora_targets import lora_target_aliases
from model_merge_options import DEFAULT_DIRECTORY
from weight_files import SAFETENSORS_SUFFIXES


@dataclass(frozen=True)
class LoraCompatibility:
    checkpoint: Path
    compatible: bool
    architecture: str
    target_count: int
    embedded_components: tuple[str, ...]
    reason: str = ""

    @classmethod
    def from_open(cls, checkpoint, readers, layout, adapter, adapter_readers):
        """Check every target, shape, rank and alpha without loading full tensors."""
        import torch
        from downloaded_model import _tensor_identity
        shapes = {key: tuple(readers[name].get_slice(key).get_shape()) for (_, key), name in layout.items()}
        try:
            architecture = _tensor_identity(shapes)[0] or "unknown"
        except ValueError:
            architecture = "unknown"
        components = []
        # Presence is structural evidence only, not a runtime/quality certificate.
        if any(key.startswith("text_encoders.llm.") for key in shapes):
            components.append("text_encoder")
        if (any(key.startswith(("vae.encoder.", "first_stage_model.encoder.")) for key in shapes)
                and any(key.startswith(("vae.decoder.", "first_stage_model.decoder.")) for key in shapes)):
            components.append("vae")
        try:
            if any(is_lora_key(key) for key in shapes):
                raise ValueError("Compatibility candidate is an adapter, not a checkpoint.")
            deltas = prepare_lora(adapter, adapter_readers, lora_target_aliases(layout, readers),
                                  readers, layout, checkpoint, torch)
        except ValueError as error:
            return cls(checkpoint.root, False, architecture, 0, tuple(components), str(error))
        return cls(checkpoint.root, True, architecture, len(deltas), tuple(components))

    @classmethod
    def inspect(cls, checkpoint, adapter, *, cache_dir=None):
        """Public path-based preflight; no checkpoint hashes, output or downloads."""
        import safetensors
        cache = Path(cache_dir) if cache_dir is not None else DEFAULT_DIRECTORY / "model-merge-cache"
        checkpoint = inspect_merge_model(Path(checkpoint).expanduser().absolute(), cache, hash_content=False)
        adapter = inspect_merge_model(Path(adapter).expanduser().absolute(), cache, hash_content=False)
        with ExitStack() as stack:
            readers, layout = open_merge_weights(checkpoint, stack, safetensors.safe_open)
            adapter_readers, _ = open_merge_weights(adapter, stack, safetensors.safe_open)
            return cls.from_open(checkpoint, readers, layout, adapter, adapter_readers)

    @property
    def has_runtime_components(self):
        # Anima exports are often denoiser-only. Prefer an available complete
        # export over an equally compatible backbone requiring separate assets.
        return self.architecture == "anima" and set(self.embedded_components) == {"text_encoder", "vae"}


@dataclass(frozen=True)
class LoraCompatibilityBridge:
    compatibility: LoraCompatibility
    refinement_strength: float
    selection_reason: str

    @property
    def checkpoint(self):
        return self.compatibility.checkpoint

    def as_dict(self):
        return {"method": "compatible-checkpoint-image-refinement", "checkpoint": str(self.checkpoint),
                "architecture": self.compatibility.architecture, "target_count": self.compatibility.target_count,
                "embedded_components": list(self.compatibility.embedded_components),
                "refinement_strength": self.refinement_strength, "selection_reason": self.selection_reason}

    @staticmethod
    def discover(checkpoints, *, exclude=()):
        """Inspect only direct sibling safetensors, never a disk-wide search."""
        excluded = {Path(path).resolve() for path in exclude}
        found = {}
        for directory in sorted({Path(path).parent for path in checkpoints}):
            for path in sorted(directory.iterdir()):
                if (not path.name.startswith(".") and not path.is_symlink() and path.is_file()
                        and path.suffix.lower() in SAFETENSORS_SUFFIXES and path.resolve() not in excluded):
                    found.setdefault(path.resolve(), path)
        return tuple(found.values())

    @classmethod
    def resolve(cls, adapter, checkpoints, *, refinement_strength=0.35, cache_dir=None):
        """Select a fully matching local bridge or explain the missing/ambiguous input.

        Candidate order and filenames never break ties. Supplying one explicit
        candidate or making it an ordinary merge material resolves ambiguity.
        """
        import math
        from numbers import Real
        import safetensors
        if (isinstance(refinement_strength, bool) or not isinstance(refinement_strength, Real)
                or not math.isfinite(refinement_strength) or not 0 <= refinement_strength <= 1):
            raise ValueError("Compatibility refinement strength must be finite and in [0, 1].")
        if isinstance(checkpoints, (str, bytes, Path)):
            raise TypeError("Compatibility checkpoints must be a sequence of local paths.")
        cache = Path(cache_dir) if cache_dir is not None else DEFAULT_DIRECTORY / "model-merge-cache"
        adapter = inspect_merge_model(Path(adapter).expanduser().absolute(), cache, hash_content=False)
        matches, failures = [], []
        with ExitStack() as stack:
            adapter_readers, _ = open_merge_weights(adapter, stack, safetensors.safe_open)
            for path in dict.fromkeys(Path(p).expanduser().resolve() for p in checkpoints):
                try:
                    if not path.is_file() or path.suffix.lower() not in SAFETENSORS_SUFFIXES:
                        raise ValueError("Bridge requires a single-file safetensors checkpoint.")
                    model = inspect_merge_model(path, cache, hash_content=False)
                    with ExitStack() as candidate_stack:
                        readers, layout = open_merge_weights(model, candidate_stack, safetensors.safe_open)
                        result = LoraCompatibility.from_open(model, readers, layout, adapter, adapter_readers)
                    if result.compatible:
                        matches.append(result)
                    else:
                        failures.append(f"{path.name}: {result.reason}")
                except (ValueError, OSError) as error:
                    failures.append(f"{path.name}: {error}")
        if not matches:
            raise ValueError("LoRA has no compatible checkpoint for a compatibility bridge. "
                             "Add its matching full checkpoint as a material or --compatibility-model. "
                             "A LoRA alone cannot reconstruct its missing base network. " + "; ".join(failures[:3]))
        complete = [result for result in matches if result.has_runtime_components]
        candidates = complete or matches
        if len(candidates) != 1:
            raise ValueError("Ambiguous LoRA compatibility checkpoints: " + ", ".join(str(r.checkpoint) for r in candidates)
                             + ". Add the intended checkpoint as a material or supply a single --compatibility-model.")
        reason = "complete-runtime-components" if complete and len(matches) > 1 else "only-compatible-checkpoint"
        return cls(candidates[0], float(refinement_strength), reason)
