"""Explicit, offline audit of installed pipeline classes against the catalog.

Importing this module reads metadata only. ``inspect_runtime`` may import the
installed Diffusers/Torch runtime but never instantiates a pipeline, loads model
weights, connects to ComfyUI or downloads anything.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.metadata
import inspect

from civitai_catalog import CATALOG_SOURCE, list_base_models, lookup_base_model


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _pipeline_available(runtime, name: str) -> tuple[bool, str | None]:
    """Reject placeholder/dummy classes as well as non-pipeline exports."""
    try:
        candidate = getattr(runtime, name, None)
        base = getattr(runtime, "DiffusionPipeline", None)
        if (not inspect.isclass(base) or not inspect.isclass(candidate)
                or candidate is base or not issubclass(candidate, base)
                or not candidate.__module__.startswith("diffusers.pipelines.")):
            return False, f"{name} is not a built-in pipeline class in the installed Diffusers runtime."
        return True, None
    except Exception as error:
        # Diffusers imports classes lazily; a class may be exported while its
        # optional Transformers/sentencepiece/video dependency cannot load.
        return False, _error_text(error)


def inspect_runtime(base_name: str | None = None) -> dict:
    """Inspect all catalog rows or one exact name without loading any weights.

    ``pipeline_available`` proves only that the installed package can import the
    suggested built-in class. It does not check a model's components, hardware,
    scheduler, adapters, task-specific arguments, output validity or generation.
    ComfyUI rows explicitly remain ``workflow_required``: this audit does not
    inspect its installation or make even a loopback network request.
    """
    records = list_base_models() if base_name is None else [lookup_base_model(base_name)]
    runtime_info = {
        "package": "diffusers",
        "version": _package_version("diffusers"),
        "torch_version": _package_version("torch"),
        "import_attempted": False,
        "import_available": False,
        "error": None,
    }
    needs_pipeline = any(record["local_status"] == "local"
                         and record["preferred_backend"] in ("preset", "diffusers")
                         for record in records)
    runtime = None
    if needs_pipeline:
        runtime_info["import_attempted"] = True
        try:
            runtime = importlib.import_module("diffusers")
            runtime_info["import_available"] = True
        except Exception as error:
            runtime_info["error"] = _error_text(error)

    probes: dict[str, tuple[bool, str | None]] = {}
    rows = []
    for record in records:
        row = {name: record[name] for name in
               ("name", "local_status", "preferred_backend", "pipeline_class")}
        row.update(status="unknown", pipeline_available=False, runtime_checked=False,
                   generation_verified=False, weights_verified=False, error=None)
        if record["local_status"] in ("hosted", "unknown"):
            row["status"] = record["local_status"]
        elif record["preferred_backend"] == "comfyui":
            row["status"] = "workflow_required"
            row["notes"] = "ComfyUI installation, server, nodes, model files and workflow execution were not checked."
        else:
            row["runtime_checked"] = True
            name = record["pipeline_class"]
            if runtime is None:
                row["error"] = runtime_info["error"]
            elif not name:
                row["error"] = "The catalog does not identify a built-in pipeline class."
            else:
                if name not in probes:
                    probes[name] = _pipeline_available(runtime, name)
                row["pipeline_available"], row["error"] = probes[name]
            row["status"] = "pipeline_available" if row["pipeline_available"] else "runtime_missing"
        rows.append(row)
    return {"catalog_source": deepcopy(CATALOG_SOURCE), "runtime": runtime_info, "rows": rows,
            "generation_verified": False, "weights_verified": False}
