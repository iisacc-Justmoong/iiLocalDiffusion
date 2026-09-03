"""Lossless output publication and non-destructive defaults for every batch image."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any


def _paths(base: Path, count: int) -> list[Path]:
    return [base] if count == 1 else [
        base.with_stem(f"{base.stem}-{index + 1:04d}") for index in range(count)
    ]


def _existing(paths: list[Path]) -> list[Path]:
    return [
        candidate for path in paths for candidate in (path, path.with_suffix(".json"))
        if candidate.exists() or candidate.is_symlink()
    ]


def hires_base_paths(paths: list[Path]) -> list[Path]:
    return [path.with_stem(path.stem + "-base") for path in paths]


def resolve_output_paths(args: Any) -> list[Path]:
    base = args.output.expanduser()
    paths = _paths(base, args.num_images)
    save_base = getattr(args, "hires_fix", False) and getattr(args, "hires_save_base", False)
    def artifacts(final_paths):
        return final_paths + hires_base_paths(final_paths) if save_base else final_paths
    if args.output_was_default and not args.overwrite:
        run = 2
        while _existing(artifacts(paths)):
            candidate = base.with_stem(f"{base.stem}-run-{run:04d}")
            paths = _paths(candidate, args.num_images)
            run += 1
        args.output = base if run == 2 else candidate
    else:
        existing = _existing(artifacts(paths))
        if existing and not args.overwrite:
            raise SystemExit("Refusing to overwrite existing reference output: " + ", ".join(map(str, existing)))
        if any(path.is_dir() for path in existing):
            raise SystemExit("Output targets must be files, not directories.")
    return paths


def publish_file(temporary: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, target)
    else:
        # Both paths are in the same directory. Hard-link publication is atomic
        # and fails if another generator creates this output after preflight.
        os.link(temporary, target)


def write_png(image: Any, path: Path, *, compress_level: int, optimize: bool, overwrite: bool) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        image.save(temporary, format="PNG", compress_level=compress_level, optimize=optimize)
        publish_file(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
