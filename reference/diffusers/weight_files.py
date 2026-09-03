"""Identity and safe loader paths for explicitly selected local tensor files."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Iterator


SAFETENSORS_SUFFIXES = (".safetensors", ".safetensor")


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


def resolve_weight_file(source: str, argument: str) -> LocalWeightFile:
    if not source:
        raise ValueError(f"{argument} must not be empty.")
    path = Path(source).expanduser().absolute()
    if path.suffix not in SAFETENSORS_SUFFIXES:
        raise ValueError(f"{argument} requires a .safetensors or .safetensor file.")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{argument} file is missing, empty, or not a regular file: {path}")
    resolved = path.resolve()
    return LocalWeightFile(
        path=str(path),
        resolved_file=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def verify_weight_file(weight: LocalWeightFile, role: str) -> None:
    path = Path(weight.path)
    try:
        matches = (
            path.is_file()
            and str(path.resolve()) == weight.resolved_file
            and path.stat().st_size == weight.size_bytes
            and file_sha256(path) == weight.sha256
        )
    except OSError:
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
