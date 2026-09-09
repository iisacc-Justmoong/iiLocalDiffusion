"""Checkpoint inventory, provenance and Diffusers package compatibility for merging."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath

from checkpoint_conversion import LEGACY_SUFFIXES, materialize_safetensors
from weight_files import (SAFETENSORS_SUFFIXES, LocalWeightFile, cached_model_sha256,
                          file_signature, resolve_weight_file, verify_weight_file)

_IGNORED_CONFIG_KEYS = frozenset({"_name_or_path", "_diffusers_version", "transformers_version",
                                "_commit_hash", "_use_default_values", "torch_dtype"})
_OTHER_WEIGHT_SUFFIXES = (*LEGACY_SUFFIXES, ".gguf", ".onnx", ".h5", ".msgpack")


@dataclass
class MergeModelFiles:
    root: Path
    weights: dict[str, LocalWeightFile]
    assets: dict[str, LocalWeightFile]
    originals: dict[str, LocalWeightFile]

    def provenance(self) -> dict:
        return {"path": str(self.root), "format": ("diffusers" if "model_index.json" in self.assets else "adapter")
                if self.root.is_dir() else "checkpoint",
                "files": [{"name": name, "sha256": value.sha256, "size_bytes": value.size_bytes}
                          for name, value in sorted(self.originals.items())]}

    def verify(self) -> None:
        if self.root.is_dir() and set(_files(self.root)) != set(self.originals):
            raise RuntimeError(f"Input model files changed during merging: {self.root}")
        for value in (*self.originals.values(), *self.weights.values()):
            verify_weight_file(value, "merge input")


def _identity(path: Path) -> LocalWeightFile:
    before = file_signature(path)
    digest = cached_model_sha256(path)
    if before != file_signature(path):
        raise RuntimeError(f"Input changed while inspecting it: {path}")
    return LocalWeightFile(str(path), before[0], digest, before[3])


def _files(root: Path) -> dict[str, Path]:
    result = {}
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Diffusers merge inputs must not contain directory symlinks.")
        for name in sorted(names):
            if name.startswith("."):
                continue
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"Model asset must be a regular file: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def inspect_merge_model(source: Path, cache: Path) -> MergeModelFiles:
    if source.is_file():
        original = _identity(source)
        converted = source
        if source.suffix.lower() in LEGACY_SUFFIXES:
            converted = Path(materialize_safetensors(source, cache)["converted_path"])
        weight = resolve_weight_file(str(converted), "Merge input")
        originals, assets = {source.name: original}, {}
        config = source.parent / "adapter_config.json"
        if source.stem == "adapter_model" and config.is_file():
            assets[config.name] = originals[config.name] = _identity(config)
        return MergeModelFiles(source, {source.name: weight}, assets, originals)
    originals = {name: _identity(path) for name, path in _files(source).items()}
    weights = {}
    assets = {}
    for name, identity in originals.items():
        suffix = Path(name).suffix.lower()
        if suffix in SAFETENSORS_SUFFIXES:
            weights[name] = identity
        elif suffix in _OTHER_WEIGHT_SUFFIXES:
            raise ValueError(f"Diffusers merge packages require only safetensors weights; remove other variants: {name}")
        else:
            assets[name] = identity
    if not weights:
        raise ValueError(f"No safetensors model weights found in {source}")
    return MergeModelFiles(source, weights, assets, originals)


def _config(value):
    if isinstance(value, dict):
        return {key: _config(item) for key, item in value.items() if key not in _IGNORED_CONFIG_KEYS}
    if isinstance(value, list):
        return [_config(item) for item in value]
    return value


def _runtime_assets(model: MergeModelFiles) -> dict:
    result = {}
    for name, asset in model.assets.items():
        path = Path(name)
        if name == "merge.json" or name.endswith(".safetensors.index.json"):
            continue
        if path.suffix.lower() in (".json", ".yaml", ".yml") or any(
                part.startswith("tokenizer") for part in path.parts[:-1]):
            if path.suffix.lower() == ".json":
                try:
                    result[name] = _config(json.loads(Path(asset.path).read_text(encoding="utf-8")))
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"Invalid model configuration: {asset.path}") from error
            else:
                result[name] = asset.sha256
    return result


def validate_package_assets(models: list[MergeModelFiles]) -> None:
    expected = _runtime_assets(models[0])
    for model in models[1:]:
        if _runtime_assets(model) != expected:
            raise ValueError(f"Model configuration/tokenizer assets do not match the base: {model.root}")


def validate_shard_indexes(model: MergeModelFiles, key_files: dict[tuple[str, str], str]) -> None:
    for name, asset in model.assets.items():
        if not name.endswith(".safetensors.index.json"):
            continue
        try:
            value = json.loads(Path(asset.path).read_text(encoding="utf-8"))
            mapping = value["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("missing weight_map")
            component = PurePosixPath(name).parent
            actual = {key: PurePosixPath(filename).name for (parent, key), filename in key_files.items()
                      if parent == str(component)}
            for key, filename in mapping.items():
                if (not isinstance(key, str) or not isinstance(filename, str)
                        or PurePosixPath(filename).name != filename or "\\" in filename):
                    raise ValueError("invalid shard filename")
            if mapping != actual:
                raise ValueError("weight_map does not match the component's tensor keys and shard files")
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid safetensors shard index {asset.path}: {error}") from error
