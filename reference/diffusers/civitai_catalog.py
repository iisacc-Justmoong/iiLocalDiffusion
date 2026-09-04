"""Versioned Civitai base-model routing metadata, without importing ML runtimes.

The catalog describes upstream architecture availability, not a guarantee that
arbitrary checkpoint contents, optional adapters or installed runtimes work.
Loading it never downloads weights or executes third-party pipeline code.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


_SNAPSHOT = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
CATALOG_SOURCE = deepcopy(_SNAPSHOT["source"])
UPSTREAM_SOURCES = deepcopy(_SNAPSHOT["upstream_sources"])
BASE_MODELS = tuple(deepcopy(_SNAPSHOT["base_models"]))
_BY_NAME = {record["name"].casefold(): record for record in _SNAPSHOT["base_models"]}
if len(_BY_NAME) != len(BASE_MODELS):
    raise ValueError("Civitai catalog contains duplicate base-model names")


def lookup_base_model(name: str) -> dict:
    """Return an independent record for an exact Civitai name (case insensitive).

    Do not interpret an unknown name as SDXL, Other or a similarly named model:
    such a fallback could silently select an incompatible denoiser or scheduler.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A nonempty Civitai base-model name is required")
    try:
        return deepcopy(_BY_NAME[name.strip().casefold()])
    except KeyError:
        raise ValueError(f"Unknown Civitai base model: {name!r}") from None


def list_base_models() -> list[dict]:
    """Return independent rows in the upstream snapshot's order."""
    return deepcopy(_SNAPSHOT["base_models"])
