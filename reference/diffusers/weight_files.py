"""Identity and safe loader paths for explicitly selected local tensor files."""

from __future__ import annotations

from contextlib import closing, contextmanager
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sqlite3
import sys
import tempfile
import time
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
    sha256: str | None
    size_bytes: int
    signature: tuple | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    report = os.environ.get("IILD_WORKER_PROGRESS") == "1" and path.suffix.lower() in SAFETENSORS_SUFFIXES
    total, completed, last = path.stat().st_size if report else 0, 0, 0.0
    with path.open("rb") as source:
        buffer = bytearray(8 * 1024 * 1024)
        view = memoryview(buffer)
        while count := source.readinto(buffer):
            digest.update(view[:count])
            if report:
                completed += count
                now = time.monotonic()
                if completed == total or now - last >= 0.5:
                    print("IILD_MODEL_PROGRESS " + json.dumps({"schema": "iild-model-progress-v1",
                          "completed_bytes": completed, "total_bytes": total}), flush=True)
                    last = now
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



def _persistent_model_hash(signature: tuple, digest: str | None = None) -> str | None:
    """Optional bounded local cache; never use publisher-supplied hashes as proof."""
    configured = os.environ.get("IILD_MODEL_HASH_CACHE")
    if configured == "off":
        return None
    root = (Path(os.environ["XDG_CACHE_HOME"]) if os.environ.get("XDG_CACHE_HOME") else
            Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache"))
    path = Path(configured).expanduser() if configured else root / "iiLocalDiffusion/model-hashes.sqlite3"
    identity = json.dumps(signature, separators=(",", ":"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A busy or unavailable cache must not delay generation or bypass hashing.
        with closing(sqlite3.connect(path, timeout=0.05)) as database, database:
            database.execute("CREATE TABLE IF NOT EXISTS hashes_v1 (path TEXT PRIMARY KEY, identity TEXT NOT NULL, digest TEXT NOT NULL, touched INTEGER NOT NULL)")
            if digest is None:
                row = database.execute("SELECT digest FROM hashes_v1 WHERE path=? AND identity=?", (signature[0], identity)).fetchone()
                if row and isinstance(row[0], str) and len(row[0]) == 64 and all(c in "0123456789abcdef" for c in row[0]):
                    return row[0]
            else:
                database.execute("INSERT OR REPLACE INTO hashes_v1 VALUES (?, ?, ?, ?)",
                                 (signature[0], identity, digest, time.time_ns()))
                database.execute("DELETE FROM hashes_v1 WHERE path NOT IN (SELECT path FROM hashes_v1 ORDER BY touched DESC LIMIT ?)", (_MODEL_HASH_LIMIT,))
    except (OSError, sqlite3.Error):
        pass
    return None


def cached_model_sha256(path: Path) -> str:
    """Reuse verified digests across workers; outputs use uncached file_sha256."""
    with _hash_lock:
        before = file_signature(path)
        key = before[0]
        cached = _model_hashes.get(key)
        if cached is not None and cached[0] == before:
            _model_hashes.move_to_end(key)
            _hash_stats["model_hash_hits"] += 1
            return cached[1]
        _model_hashes.pop(key, None)
        digest = _persistent_model_hash(before)
        persisted = digest is not None
        if not persisted:
            digest = file_sha256(path)
            _hash_stats["model_hashes"] += 1
            _hash_stats["model_bytes_hashed"] += before[3]
        else:
            _hash_stats["model_hash_hits"] += 1
        if file_signature(path) != before:
            raise RuntimeError(f"Model input changed while hashing: {path}")
        if not persisted:
            _persistent_model_hash(before, digest)
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


def metadata_model_validation() -> bool:
    """Interactive generation checks paths and lets the actual loader read weights."""
    return os.environ.get("IILD_MODEL_VALIDATION") == "metadata"


def model_content_sha256(path: Path) -> str | None:
    if metadata_model_validation():
        file_signature(path)
        return None
    return cached_model_sha256(path)


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
        sha256=model_content_sha256(resolved),
        size_bytes=resolved.stat().st_size,
        signature=file_signature(resolved) if metadata_model_validation() else None,
    )


def verify_weight_file(weight: LocalWeightFile, role: str) -> None:
    path = Path(weight.path)
    try:
        matches = (
            path.is_file()
            and str(path.resolve()) == weight.resolved_file
            and path.stat().st_size == weight.size_bytes
            and (weight.signature is None or file_signature(path) == weight.signature)
            and (weight.sha256 is None or cached_model_sha256(path) == weight.sha256)
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
        **({"validation": "metadata"} if weight.sha256 is None else {}),
        "format": "safetensors",
        "path": weight.path,
        "resolved_file": weight.resolved_file,
        "sha256": weight.sha256,
        "size_bytes": weight.size_bytes,
    }
