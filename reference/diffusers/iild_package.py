"""Single-file .iildmodel archive creation, inspection and materialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from weight_files import cached_model_sha256, file_signature


CONTAINER = "zip-stored-v1"
MAX_ENTRIES = 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024**4


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\0" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ValueError("Invalid .iildmodel archive member path.")
    return path


def inspect_archive(path: str | Path, *, hashes=False):
    source = Path(path).expanduser().absolute()
    if (not source.is_file() or source.is_symlink() or source.resolve() != source
            or source.suffix.lower() != ".iildmodel" or source.stat().st_size <= 0):
        raise ValueError("Choose an available canonical .iildmodel package file.")
    before = file_signature(source)
    entries = {}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            if not 1 <= len(infos) <= MAX_ENTRIES:
                raise ValueError("An .iildmodel package requires 1 to 1024 file entries.")
            total = 0
            for info in infos:
                relative = _relative(info.filename)
                mode = info.external_attr >> 16
                if (info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1
                        or info.compress_type != zipfile.ZIP_STORED or info.file_size <= 0
                        or info.compress_size != info.file_size or relative.as_posix() in entries):
                    raise ValueError("The .iildmodel package contains an unsupported, redirected, duplicate or compressed entry.")
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("The .iildmodel package exceeds the 16 TiB extraction limit.")
                entries[relative.as_posix()] = info
            manifest_info = entries.get("model_index.json")
            if manifest_info is None or manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("The .iildmodel manifest is missing or oversized.")
            try:
                manifest = json.loads(archive.read(manifest_info))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid .iildmodel manifest: {error}") from error
            if not isinstance(manifest, dict) or manifest.get("container") != CONTAINER:
                raise ValueError(f"An .iildmodel package must declare container {CONTAINER}.")
            if hashes:
                for stage in manifest.get("stages", ()) if isinstance(manifest, dict) else ():
                    name = stage.get("model") if isinstance(stage, dict) else None
                    info = entries.get(name)
                    if info is None or info.file_size != stage.get("size_bytes"):
                        raise ValueError("Unified model member is missing or changed.")
                    digest = hashlib.sha256()
                    with archive.open(info) as member:
                        while block := member.read(8 * 1024 * 1024):
                            digest.update(block)
                    if digest.hexdigest() != stage.get("sha256"):
                        raise ValueError("Unified model member hash differs from the published object.")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Invalid .iildmodel ZIP64 package: {error}") from error
    if file_signature(source) != before:
        raise RuntimeError("The .iildmodel package changed during inspection.")
    return source, manifest, entries


def materialize_archive(path: str | Path, cache_directory: str | Path | None = None) -> Path:
    source, manifest, entries = inspect_archive(path)
    signature = file_signature(source)
    key = hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()
    cache = (Path(cache_directory).expanduser().absolute() if cache_directory is not None
             else Path(tempfile.gettempdir()) / "iiLocalDiffusion/iildmodel-v1")
    destination = cache / key

    def valid(root):
        if not root.is_dir() or root.is_symlink():
            return False
        try:
            marker = json.loads((root / ".iildmodel-source.json").read_text())
            if marker != {"signature": list(signature)}:
                return False
            for name, info in entries.items():
                file = root / name
                if not file.is_file() or file.is_symlink() or file.stat().st_size != info.file_size:
                    return False
            for stage in manifest.get("stages", ()):
                file = root / stage["model"]
                if cached_model_sha256(file) != stage.get("sha256"):
                    return False
            return True
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    if valid(destination):
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=cache) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        with zipfile.ZipFile(source, "r") as archive:
            for name, info in entries.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise RuntimeError("An .iildmodel member changed while extracting.")
        (staging / ".iildmodel-source.json").write_text(json.dumps({"signature": list(signature)}))
        if file_signature(source) != signature:
            raise RuntimeError("The .iildmodel package changed while extracting.")
        if destination.exists():
            if valid(destination):
                return destination
            raise RuntimeError("The existing .iildmodel extraction cache is invalid.")
        os.rename(staging, destination)
    if not valid(destination):
        raise RuntimeError("The extracted .iildmodel package failed validation.")
    return destination


def write_archive(source_directory: str | Path, output: str | Path) -> None:
    root = Path(source_directory)
    destination = Path(output)
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files or len(files) > MAX_ENTRIES:
        raise ValueError("An .iildmodel package requires 1 to 1024 regular files.")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            relative = _relative(path.relative_to(root).as_posix()).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.flag_bits |= 0x800
            with path.open("rb") as incoming, archive.open(info, "w", force_zip64=True) as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
