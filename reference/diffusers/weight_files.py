"""Identity and safe loader paths for explicitly selected local tensor files."""

from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock
from typing import Iterator


SAFETENSORS_SUFFIXES = (".safetensors", ".safetensor")
_MODEL_HASH_LIMIT = 1024
_model_hashes: OrderedDict[str, tuple[tuple, str]] = OrderedDict()
_hash_lock = RLock()
_hash_stats = {"model_hashes": 0, "model_hash_hits": 0, "model_bytes_hashed": 0}


@dataclass(frozen=True)
class LocalWeightFile:
    path: str
    resolved_file: str
    sha256: str
    size_bytes: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows_change_time(path: Path) -> int:
    # Python's Windows st_ctime is creation time, not the metadata-change clock.
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class BasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in
                    ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")]
        _fields_.append(("FileAttributes", wintypes.DWORD))

    query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    info = BasicInfo()
    with path.open("rb") as source:
        if not query(msvcrt.get_osfhandle(source.fileno()), 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
    return info.ChangeTime


def file_signature(path: Path) -> tuple:
    """Cheap identity check; no tensor bytes are read on an unchanged file."""
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Model input is not a regular file: {path}")
    change_time = _windows_change_time(resolved) if os.name == "nt" else info.st_ctime_ns
    return (str(resolved), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, change_time)


def cached_model_sha256(path: Path) -> str:
    """Process-local bounded cache. Outputs continue to use uncached file_sha256."""
    with _hash_lock:
        before = file_signature(path)
        key = before[0]
        cached = _model_hashes.get(key)
        if cached is not None and cached[0] == before:
            _model_hashes.move_to_end(key)
            _hash_stats["model_hash_hits"] += 1
            return cached[1]
        _model_hashes.pop(key, None)
        digest = file_sha256(path)
        _hash_stats["model_hashes"] += 1
        _hash_stats["model_bytes_hashed"] += before[3]
        if file_signature(path) != before:
            raise RuntimeError(f"Model input changed while hashing: {path}")
        _model_hashes[key] = (before, digest)
        if len(_model_hashes) > _MODEL_HASH_LIMIT:
            _model_hashes.popitem(last=False)
        return digest


def clear_model_hash_cache() -> None:
    with _hash_lock:
        _model_hashes.clear()
        _hash_stats.update({name: 0 for name in _hash_stats})


def model_hash_statistics() -> dict[str, int]:
    with _hash_lock:
        return dict(_hash_stats)


def resolve_weight_file(source: str, argument: str) -> LocalWeightFile:
    if not source:
        raise ValueError(f"{argument} must not be empty.")
    path = Path(source).expanduser().absolute()
    if path.suffix.lower() not in SAFETENSORS_SUFFIXES:
        raise ValueError(f"{argument} requires a .safetensors or .safetensor file.")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{argument} file is missing, empty, or not a regular file: {path}")
    resolved = path.resolve()
    return LocalWeightFile(
        path=str(path),
        resolved_file=str(resolved),
        sha256=cached_model_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def verify_weight_file(weight: LocalWeightFile, role: str) -> None:
    path = Path(weight.path)
    try:
        matches = (
            path.is_file()
            and str(path.resolve()) == weight.resolved_file
            and path.stat().st_size == weight.size_bytes
            and cached_model_sha256(path) == weight.sha256
        )
    except (OSError, ValueError):
        matches = False
    if not matches:
        raise RuntimeError(f"Local {role} changed after argument resolution: {weight.path}")


@contextmanager
def checked_safetensors_path(
    weight: LocalWeightFile,
    staging_directory: Path,
    role: str,
) -> Iterator[Path]:
    """Keep Diffusers on its safetensors branch, including for the singular suffix."""
    verify_weight_file(weight, role)
    path = Path(weight.path)
    if path.suffix == ".safetensors":
        yield path
    else:
        staging_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="weights-", dir=staging_directory) as temporary:
            alias = Path(temporary) / "weights.safetensors"
            try:
                alias.symlink_to(weight.resolved_file)
            except OSError as error:
                raise RuntimeError(
                    "The .safetensor alias requires symlink support; use a file named "
                    "with the standard .safetensors suffix on this filesystem."
                ) from error
            yield alias
    verify_weight_file(weight, role)


def weight_file_metadata(weight: LocalWeightFile) -> dict[str, object]:
    return {
        "format": "safetensors",
        "path": weight.path,
        "resolved_file": weight.resolved_file,
        "sha256": weight.sha256,
        "size_bytes": weight.size_bytes,
    }
