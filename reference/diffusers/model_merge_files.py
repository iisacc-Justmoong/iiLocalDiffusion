"""Checkpoint inventory, provenance and Diffusers package compatibility for merging."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath

from checkpoint_conversion import LEGACY_SUFFIXES, materialize_safetensors
from weight_files import (SAFETENSORS_SUFFIXES, LocalWeightFile, cached_model_sha256,
                          file_signature, resolve_weight_file, verify_weight_file)
from iild_package import materialize_archive

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
                if self.root.is_dir() else "iildmodel" if self.root.suffix.lower() == ".iildmodel" else "checkpoint",
                "files": [{"name": name, "sha256": value.sha256, "size_bytes": value.size_bytes}
                          for name, value in sorted(self.originals.items())]}

    def verify(self) -> None:
        if self.root.is_dir() and set(_files(self.root)) != set(self.originals):
            raise RuntimeError(f"Input model files changed during merging: {self.root}")
        for value in (*self.originals.values(), *self.weights.values()):
            verify_weight_file(value, "merge input")


def _identity(path: Path, hash_content: bool = True) -> LocalWeightFile:
    before = file_signature(path)
    digest = cached_model_sha256(path) if hash_content else ""
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


def inspect_merge_model(source: Path, cache: Path, *, hash_content: bool = True) -> MergeModelFiles:
    if source.is_file():
        original = _identity(source, hash_content)
        if source.suffix.lower() == ".iildmodel":
            root = materialize_archive(source, cache / "iildmodel")
            manifest = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
            stages = manifest.get("stages") if isinstance(manifest, dict) else None
            if (not isinstance(manifest, dict) or manifest.get("schema") != "iild-unified-model-v1"
                    or manifest.get("_class_name") != "IILDUnifiedCascade"
                    or manifest.get("composition") != "ordered-image-refinement"
                    or not isinstance(stages, list) or len(stages) != 1 or stages[0].get("strength") != 1
                    or stages[0].get("loras")):
                raise ValueError("Merge inputs require a single-stage .iildmodel package without pending LoRAs.")
            relative = stages[0].get("model")
            if not isinstance(relative, str):
                raise ValueError("The .iildmodel package has no checkpoint member.")
            member = root / relative
            nested = inspect_merge_model(member, cache, hash_content=hash_content)
            return MergeModelFiles(source, {source.name: next(iter(nested.weights.values()))}, {},
                                   {source.name: original})
        converted = source
        if source.suffix.lower() in LEGACY_SUFFIXES:
            converted = Path(materialize_safetensors(source, cache)["converted_path"])
        weight = resolve_weight_file(str(converted), "Merge input") if hash_content else _identity(converted, False)
        originals, assets = {source.name: original}, {}
        config = source.parent / "adapter_config.json"
        if source.stem == "adapter_model" and config.is_file():
            assets[config.name] = originals[config.name] = _identity(config, hash_content)
        return MergeModelFiles(source, {source.name: weight}, assets, originals)
    originals = {name: _identity(path, hash_content) for name, path in _files(source).items()}
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


def open_merge_weights(model, stack, safe_open):
    """Open validated tensor headers in the caller's lifetime, without loading weights."""
    readers, layout = {}, {}
    for name, weight in sorted(model.weights.items()):
        try:
            reader = stack.enter_context(safe_open(weight.resolved_file, framework="pt", device="cpu"))
        except Exception as error:
            raise ValueError(f"Cannot read safetensors merge input {weight.path}: {error}") from error
        readers[name] = reader
        component = str(PurePosixPath(name).parent) if model.root.is_dir() else "."
        if not reader.keys():
            raise ValueError(f"Weight file contains no tensor keys: {weight.path}")
        for key in reader.keys():
            address = (component, key)
            if address in layout:
                raise ValueError(f"Duplicate tensor keys/weight variants in model: {address}")
            layout[address] = name
    if not layout:
        raise ValueError(f"Model contains no tensor keys: {model.root}")
    validate_shard_indexes(model, layout)
    return readers, layout


def _anima_tensor_names(layout):
    names = {}
    for component, key in layout:
        if component != ".":
            return None
        local = key
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if local.startswith(prefix):
                local = local[len(prefix):]
                break
        local = local.removeprefix("net.")
        if not local.startswith(("blocks.", "llm_adapter.", "x_embedder.", "final_layer.",
                                 "t_embedder.", "t_embedding_norm.")):
            local = key  # Embedded encoders, VAE and prediction markers remain exact.
        if local in names:
            return None  # Never collapse two tensors onto the same target.
        names[local] = (component, key)
    return names if "llm_adapter.blocks.0.cross_attn.q_proj.weight" in names else None


class _AlignedCheckpointReader:
    """Read equivalent source keys through the base checkpoint's namespace."""
    def __init__(self, reader, names):
        self.reader, self.names = reader, names

    def keys(self):
        return list(self.names)

    def metadata(self):
        return self.reader.metadata()

    def get_slice(self, key):
        return self.reader.get_slice(self.names[key])

    def get_tensor(self, key):
        return self.reader.get_tensor(self.names[key])


def align_anima_checkpoint(readers, layout, base_layout):
    """Align only complete, unambiguous single-file Anima tensor inventories.

    Shapes, dtypes and prediction settings still pass the ordinary merge checks;
    this view changes neither source bytes nor the base output layout.
    """
    if len(readers) != 1 or layout.keys() == base_layout.keys():
        return readers, layout
    base, source = _anima_tensor_names(base_layout), _anima_tensor_names(layout)
    if base is None or source is None or base.keys() != source.keys():
        return readers, layout
    name, reader = next(iter(readers.items()))
    aliases = {address[1]: source[local][1] for local, address in base.items()}
    return {name: _AlignedCheckpointReader(reader, aliases)}, {address: name for address in base_layout}


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
                result[name] = asset.sha256 or cached_model_sha256(Path(asset.resolved_file))
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
