"""Lossless output publication and non-destructive defaults for every batch image."""

from __future__ import annotations

from _file_provider import provider

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


def _converted_model_sources(args: Any) -> set[Path]:
    conversions = list(getattr(args, 'model_directory_conversions', ()))
    if getattr(args, 'model_conversion', None):
        conversions.append(args.model_conversion)
    sources = []
    for conversion in conversions:
        records = [conversion.get("original", {}), *conversion.get("original_shards", ())]
        sources.extend(record.get("path") for record in records)
        sources.append(conversion.get("converted_path"))
    if getattr(args, 'source_kind', None) == 'single-file':
        sources.extend((getattr(args, 'model', None), getattr(args, 'original_model_path', None)))
    sources.extend(weight.path for weight in getattr(args, 'directory_gguf_components', {}).values())
    vae = getattr(args, 'vae_selection', None)
    if vae is not None:
        sources.extend((vae.weight.path, vae.config_path))
        sources.extend(shard.path for shard in getattr(vae.weight, 'shards', ()))
    return {Path(source).expanduser().resolve() for source in sources if source}


def protect_model_inputs(args: Any, paths: list[Path]) -> None:
    """Refuse publication over a converted checkpoint or any of its sources."""
    protected = _converted_model_sources(args)
    for path in paths:
        for candidate in (path, path.with_suffix(".json")):
            if candidate.expanduser().resolve() in protected:
                raise ValueError(f"Output would overwrite a model input: {candidate}")


def protect_model_directory(args: Any) -> None:
    """Keep dynamic output names away from all converted model inputs."""
    directory = Path(args.output_dir).expanduser().resolve()
    if getattr(args, 'source_kind', None) == 'directory' and getattr(args, 'model', None):
        model_root = Path(args.model).expanduser().resolve()
        if directory == model_root or model_root in directory.parents or directory in model_root.parents:
            raise ValueError(f"Output directory overlaps a model input: {model_root}")
    for source in _converted_model_sources(args):
        if source == directory or directory in source.parents:
            raise ValueError(f"Output directory contains a model input: {source}")
    source_directories = (getattr(args, 'model_directory_sources', None) or {}).get('directories', ())
    for entry in source_directories:
        source = Path(entry['resolved'])
        if source == directory or source in directory.parents or directory in source.parents:
            raise ValueError(f"Output directory overlaps a model input: {source}")


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
    protect_model_inputs(args, artifacts(paths))
    return paths


def publish_file(temporary: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        provider.replace(temporary, target)
    else:
        # Both paths are in the same directory. Hard-link publication is atomic
        # and fails if another generator creates this output after preflight.
        provider.link(temporary, target)


def write_png(image: Any, path: Path, *, compress_level: int, optimize: bool, overwrite: bool) -> None:
    with provider.staged_path(path, overwrite=overwrite) as temporary:
        image.save(temporary, format="PNG", compress_level=compress_level, optimize=optimize)
