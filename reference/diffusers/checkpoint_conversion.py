"""Convert explicitly selected local tensor checkpoints without unsafe pickle fallback.

Importing this module does not import Torch or safetensors. Callers should place
``cache_dir`` below their build directory. A conversion checks serialization and
source identity only; it does not prove a checkpoint's model family or quality.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import BinaryIO

from weight_files import file_sha256


LEGACY_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")
SAFETENSORS_SUFFIXES = (".safetensors", ".safetensor")
CONVERSION_VERSION = 1
MINIMUM_TORCH_VERSION = (2, 10, 0)
CONVERSION_POLICY = {
    "loader": "torch.load",
    "map_location": "cpu",
    "weights_only": True,
    "tensor_layout": "dense-contiguous",
    "tensor_storage": "independent",
    "minimum_torch_version": "2.10.0",
}
_DTYPE_NAMES = (
    "bool", "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64",
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
    "float8_e8m0fnu",
)


def _identity(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Checkpoint must be a nonempty regular file: {path}")
    resolved = path.resolve()
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or path.resolve() != resolved:
        raise RuntimeError(f"Checkpoint changed while computing its identity: {path}")
    return {
        "path": str(path),
        "resolved_file": str(resolved),
        "sha256": digest,
        "size_bytes": after.st_size,
    }


def _verify_original(original: dict) -> None:
    try:
        matches = _identity(Path(original["path"])) == original
    except (OSError, ValueError):
        matches = False
    if not matches:
        raise RuntimeError(f"Source checkpoint changed during conversion: {original['path']}")


def _stream_sha256(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def ensure_safe_torch_version(torch) -> None:
    """Reject releases with known weights-only unpickler vulnerabilities."""
    version = str(getattr(torch, "__version__", ""))
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9._-]+)?", version)
    if match is None or tuple(int(part) for part in match.groups()) < MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"Legacy checkpoint conversion requires stable Torch >=2.10.0; found {version!r}. "
            "Older weights-only unpicklers are affected by CVE-2026-24747."
        )


def _tensor_state(loaded, torch) -> tuple[dict, bool]:
    if not isinstance(loaded, Mapping):
        raise ValueError("Checkpoint must contain a tensor state dictionary, not a serialized model.")
    wrapped = isinstance(loaded.get("state_dict"), Mapping)
    state = loaded["state_dict"] if wrapped else loaded
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict must be a nonempty mapping of names to tensors.")
    accepted_types = [torch.Tensor]
    parameter = getattr(getattr(torch, "nn", None), "Parameter", None)
    if parameter is not None:
        accepted_types.append(parameter)
    dtypes = {getattr(torch, name) for name in _DTYPE_NAMES if hasattr(torch, name)}
    result = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint tensor names must be nonempty strings.")
        if type(value) not in accepted_types:
            raise ValueError(f"Checkpoint entry {name!r} is not a plain Torch tensor.")
        if (value.layout != torch.strided or value.is_quantized or value.is_meta
                or value.dtype not in dtypes):
            raise ValueError(f"Checkpoint entry {name!r} requires a dense, non-quantized numeric tensor.")
        # Independent storage preserves every key, including tied/shared tensors,
        # without safetensors' shared-storage rejection or lossy key deletion.
        result[name] = value.detach().cpu().contiguous().clone()
    return result, wrapped


def _cached_result(directory: Path, original: dict, *, cache_hit: bool) -> dict:
    manifest_path = directory / "conversion.json"
    output_path = directory / "model.safetensors"
    try:
        if directory.is_symlink() or manifest_path.is_symlink() or output_path.is_symlink():
            raise ValueError("Conversion cache entries must not be symlinks.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["schema"] != "iild-checkpoint-conversion"
                or manifest["schema_version"] != CONVERSION_VERSION
                or manifest["conversion_policy"] != CONVERSION_POLICY
                or manifest["original"]["sha256"] != original["sha256"]
                or manifest["original"]["size_bytes"] != original["size_bytes"]
                or manifest["output"]["filename"] != output_path.name):
            raise ValueError("Conversion manifest does not match this source or conversion policy.")
        output = _identity(output_path)
        if (output["sha256"] != manifest["output"]["sha256"]
                or output["size_bytes"] != manifest["output"]["size_bytes"]):
            raise ValueError("Converted tensor file no longer matches its manifest.")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid checkpoint conversion cache at {directory}: {error}") from error
    _verify_original(original)
    return {
        "converted_path": str(output_path),
        "converted": True,
        "cache_hit": cache_hit,
        "original": original,
        "output": output,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def materialize_safetensors(path: str | Path, cache_dir: str | Path) -> dict:
    """Return a verified local safetensors path and original-file provenance.

    Legacy files use only ``torch.load(map_location='cpu', weights_only=True)``.
    Their tensor dictionary is copied to an atomic, content-addressed directory
    beneath ``cache_dir``. Existing valid conversions require no ML imports.
    ``.gguf`` requires a native loader and is never converted by this function.
    """
    if not str(path).strip():
        raise ValueError("Checkpoint path must not be empty.")
    source = Path(path).expanduser().absolute()
    suffix = source.suffix.lower()
    if suffix == ".gguf":
        raise ValueError("GGUF is not converted to safetensors; select a native GGUF loader/workflow.")
    if suffix not in (*SAFETENSORS_SUFFIXES, *LEGACY_SUFFIXES):
        raise ValueError(f"Unsupported checkpoint suffix {suffix!r}; use safetensors, ckpt, pt, pth, or bin.")
    original = _identity(source)
    if suffix in SAFETENSORS_SUFFIXES:
        return {
            "converted_path": str(source),
            "converted": False,
            "cache_hit": False,
            "original": original,
            "output": dict(original),
            "manifest_path": None,
            "manifest": None,
        }
    if not str(cache_dir).strip():
        raise ValueError("Conversion cache directory must not be empty.")
    cache = Path(cache_dir).expanduser().resolve()
    directory = cache / f"v{CONVERSION_VERSION}-{original['sha256']}"
    if directory.exists() or directory.is_symlink():
        return _cached_result(directory, original, cache_hit=True)
    try:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")
        safetensors = importlib.import_module("safetensors")
    except ImportError as error:
        raise RuntimeError("Legacy checkpoint conversion requires installed Torch and safetensors.") from error
    ensure_safe_torch_version(torch)
    # The open descriptor pins the file being loaded. Hash it on both sides of
    # deserialization, and separately check that the caller's source still agrees.
    with Path(original["resolved_file"]).open("rb") as stream:
        if _stream_sha256(stream) != original["sha256"]:
            raise RuntimeError("Source checkpoint changed before weights-only loading.")
        try:
            loaded = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(
                "Weights-only checkpoint loading failed; unsafe pickle retry is disabled. "
                "Use a tensor-only checkpoint or obtain safetensors from its publisher."
            ) from error
        if _stream_sha256(stream) != original["sha256"]:
            raise RuntimeError("Source checkpoint changed during weights-only loading.")
    tensors, wrapped = _tensor_state(loaded, torch)
    del loaded
    _verify_original(original)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=cache))
    try:
        converted_path = temporary / "model.safetensors"
        safetensors_torch.save_file(tensors, str(converted_path), metadata={
            "original_sha256": original["sha256"],
            "conversion": "iild-weights-only-v1",
        })
        if not converted_path.is_file() or converted_path.stat().st_size == 0:
            raise RuntimeError("Safetensors conversion did not produce a nonempty tensor file.")
        output = {
            "filename": "model.safetensors",
            "sha256": file_sha256(converted_path),
            "size_bytes": converted_path.stat().st_size,
        }
        manifest = {
            "schema": "iild-checkpoint-conversion",
            "schema_version": CONVERSION_VERSION,
            "original": original,
            "output": output,
            "conversion_policy": dict(CONVERSION_POLICY),
            "runtime": {"torch": getattr(torch, "__version__", None),
                        "safetensors": getattr(safetensors, "__version__", None)},
            "state_dict_wrapped": wrapped,
            "tensor_count": len(tensors),
            "tensor_dtypes": dict(sorted(Counter(str(tensor.dtype) for tensor in tensors.values()).items())),
            "generation_verified": False,
        }
        (temporary / "conversion.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify_original(original)
        try:
            os.rename(temporary, directory)
        except OSError:
            # Another conversion of the same content may have won the rename.
            # Reuse it only after checking both its manifest and output hash.
            if not directory.exists():
                raise
            return _cached_result(directory, original, cache_hit=True)
        return _cached_result(directory, original, cache_hit=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
