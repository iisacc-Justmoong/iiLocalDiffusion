"""Local Textual Inversion file selections without importing model runtimes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

from presets import PipelinePreset
from weight_files import LocalWeightFile, resolve_weight_file


TEXT_EMBEDDING_ENCODERS = ("auto", "text_encoder", "text_encoder_2")


@dataclass(frozen=True)
class TextEmbeddingSelection:
    file: LocalWeightFile
    token: str | None
    encoder: str


def add_text_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text-embedding", "--textual-inversion", type=Path, nargs="+", default=None,
                        metavar="FILE", help="Local Textual Inversion safetensors files to register before text encoding")
    parser.add_argument("--text-embedding-token", nargs="+", default=None, metavar="TOKEN",
                        help="One literal token per file; omitted uses each file's stored token")
    parser.add_argument("--text-embedding-encoder", nargs="+", choices=TEXT_EMBEDDING_ENCODERS, default=None,
                        metavar="ENCODER", help="One target encoder per file; auto uses compatible model dimensions")


def resolve_text_embedding_options(preset: PipelinePreset, args: Any) -> None:
    """Validate each requested file and retain a replayable ordered argument mapping."""
    for name in ("text_embedding", "text_embedding_token", "text_embedding_encoder"):
        if not hasattr(args, name):
            setattr(args, name, None)
    args.text_embedding_selections = []
    if args.text_embedding is None:
        if args.text_embedding_token is not None or args.text_embedding_encoder is not None:
            raise ValueError("--text-embedding-token and --text-embedding-encoder require --text-embedding.")
        return
    if not isinstance(args.text_embedding, (list, tuple)) or not args.text_embedding:
        raise ValueError("--text-embedding requires a non-empty list of local safetensors files.")
    if any(not isinstance(path, (Path, str)) for path in args.text_embedding):
        raise ValueError("--text-embedding requires local file paths.")
    if getattr(args, "embeddings", None) is not None or getattr(args, "embeddings_file", None) is not None:
        raise ValueError("--text-embedding cannot be combined with precomputed --embeddings.")

    count = len(args.text_embedding)
    for name in ("text_embedding_token", "text_embedding_encoder"):
        values = getattr(args, name)
        if values is not None and (not isinstance(values, (list, tuple)) or len(values) != count):
            raise ValueError(f"--{name.replace('_', '-')} count must match the --text-embedding file count.")
    tokens = [None] * count if args.text_embedding_token is None else list(args.text_embedding_token)
    encoders = ["auto"] * count if args.text_embedding_encoder is None else list(args.text_embedding_encoder)
    if args.text_embedding_token is not None:
        for token in tokens:
            if (not isinstance(token, str) or not token
                    or any(character.isspace() or unicodedata.category(character) in ("Cc", "Cf", "Cs")
                           for character in token)):
                raise ValueError("Each --text-embedding-token must be non-empty without whitespace or control characters.")
    for encoder in encoders:
        if encoder not in TEXT_EMBEDDING_ENCODERS:
            raise ValueError(f"--text-embedding-encoder must be one of {', '.join(TEXT_EMBEDDING_ENCODERS)}.")
        if preset.family == "sd15" and encoder == "text_encoder_2":
            raise ValueError("SD 1.5 has no text_encoder_2; select auto or text_encoder.")

    selections = [
        TextEmbeddingSelection(resolve_weight_file(str(path), "--text-embedding"), token, encoder)
        for path, token, encoder in zip(args.text_embedding, tokens, encoders)
    ]
    args.text_embedding = [Path(selection.file.path) for selection in selections]
    args.text_embedding_encoder = encoders
    if args.text_embedding_token is not None:
        args.text_embedding_token = tokens
    args.text_embedding_selections = selections
