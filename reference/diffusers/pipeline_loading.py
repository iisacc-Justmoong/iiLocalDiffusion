"""Shared, user-facing loading boundary for Diffusers reference tools."""

from __future__ import annotations

from typing import Any, Mapping


def _is_gated_repository_error(error: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        gated_error_type = None
    else:
        gated_error_type = GatedRepoError

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            (gated_error_type is not None and isinstance(current, gated_error_type))
            or type(current).__name__ == "GatedRepoError"
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_pipeline(
    pipeline_class: Any,
    source: str,
    load_arguments: Mapping[str, Any],
) -> Any:
    try:
        return pipeline_class.from_pretrained(source, **dict(load_arguments))
    except Exception as error:
        if not _is_gated_repository_error(error):
            raise
        raise SystemExit(
            f"Cannot access gated model {source}. Accept its terms on Hugging Face "
            "and authenticate this environment with `hf auth login`, then retry."
        ) from None
