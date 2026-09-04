#!/usr/bin/env python3
"""Route Civitai base families to an installed local generation runtime."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

REFERENCE = Path(__file__).resolve().parent / "diffusers"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from civitai_catalog import CATALOG_SOURCE, list_base_models, lookup_base_model


def option_value(tokens: list[str], name: str) -> str | None:
    """Read a forwarded option without consuming it from the selected backend."""
    found = None
    for index, token in enumerate(tokens):
        if token.startswith(name + "="):
            found = token.split("=", 1)[1]
        elif token == name:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"{name} requires a value")
            found = tokens[index + 1]
    return found


def legacy_inspection_python(model: str, remaining: list[str]) -> Path | None:
    """Reuse the managed safe Torch runtime without importing ML on the host."""
    if Path(model).suffix.casefold() not in {".ckpt", ".pt", ".pth", ".bin"}:
        return None
    from local_image import DEFAULT_PYTHON
    requested = option_value(remaining, "--runtime-python")
    interpreter = Path(requested).expanduser().absolute() if requested else DEFAULT_PYTHON.absolute()
    if requested and not interpreter.is_file():
        raise ValueError(f"Inspection runtime Python does not exist: {interpreter}")
    # Do not resolve venv symlinks: two environments may use the same binary but
    # expose different dependencies. Re-execution uses this exact absolute path.
    if interpreter.is_file() and interpreter != Path(sys.executable).absolute():
        return interpreter
    return None


def select_backend(args, remaining: list[str]) -> str:
    base = lookup_base_model(args.base_model) if args.base_model else None
    if base is not None and base["local_status"] == "hosted":
        raise ValueError(f"{base['name']} is hosted-only in the catalog and has no local generation route.")
    if args.backend != "auto":
        if args.backend == "preset" and base is not None and base["preset"] is None:
            raise ValueError("This model family requires the diffusers or comfyui backend.")
        return args.backend
    flags = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    if "--workflow" in flags:
        return "comfyui"
    if "--pipeline-class" in flags or "--pipeline-inputs" in flags:
        return "diffusers"
    if "--preset" in flags:
        if base is not None and base["preset"] is None:
            raise ValueError("--preset cannot represent this base model; choose a pipeline or workflow.")
        return "preset"
    if flags & {"--model-config", "--audio-sample-rate", "--video-layout"}:
        return "diffusers"
    if (flags & {"--model-info", "--components", "--model-type", "--decoder", "--model-negative",
                 "--embedded-guidance", "--sampling-shift", "--zsnr",
                 "--sampler", "--startup-timeout"}
            or any(flag.startswith(("--runtime-", "--text-encoder")) for flag in flags)):
        return "local"
    model = option_value(remaining, "--model")
    if model:
        local_path = Path(model).expanduser()
        if local_path.is_file():
            return "local"
        if local_path.is_dir() and (local_path / "model_index.json").is_file():
            return "diffusers"
    if base is None or base["preset"]:
        return "preset"
    if base["preferred_backend"] == "diffusers":
        return "diffusers"
    raise ValueError(f"{base['name']} requires an explicit local --workflow with --backend comfyui, "
                     "or --backend diffusers with a supported built-in pipeline and --model.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--backend", choices=("auto", "local", "preset", "diffusers", "comfyui"), default="auto")
    parser.add_argument("--list-base-models", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--inspect-model", action="store_true")
    tokens = list(sys.argv[1:] if argv is None else argv)
    args, remaining = parser.parse_known_args(tokens)
    if args.list_base_models:
        print(json.dumps({"source": CATALOG_SOURCE, "base_models": list_base_models()}, indent=2))
        return 0
    if args.check_runtime:
        from runtime_compatibility import inspect_runtime
        try:
            report = inspect_runtime(args.base_model)
        except ValueError as error:
            parser.exit(2, str(error) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if args.inspect_model:
        from downloaded_model import inspect_downloaded_model
        try:
            model = option_value(remaining, "--model")
            if not model:
                raise ValueError("--inspect-model requires --model with a local weight file")
            interpreter = legacy_inspection_python(model, remaining)
            if interpreter is not None:
                return subprocess.call([str(interpreter), str(Path(__file__).absolute()), *tokens])
            report = inspect_downloaded_model(model, option_value(remaining, "--model-info"))
        except (ValueError, OSError) as error:
            parser.exit(2, str(error) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if tokens in ([], ["--help"], ["-h"]):
        print(__doc__ + "\n\n"
              "--list-base-models                 List the pinned Civitai compatibility catalog\n"
              "--check-runtime                    Audit installed pipelines without downloading weights\n"
              "--inspect-model --model PATH        Inspect a local model and its Civitai metadata\n"
              "--base-model NAME                  Select a Civitai base identity\n"
              "--backend auto|local|preset|diffusers|comfyui\n\n"
              "Auto routes existing local weight files through the local image runtime, and\n"
              "complete model_index.json directories or --model-config through Diffusers.\n"
              "Explicit backend, preset, pipeline and workflow choices win.\n\n"
              "Remaining arguments go to the selected backend. For full backend help:\n"
              "  --backend local --help\n  --backend preset --help\n  --backend diffusers --help\n  --backend comfyui --help\n\n"
              "Examples:\n"
              "  --model /path/download.safetensors --prompt 'a red cube'\n"
              "  --inspect-model --model /path/download.safetensors\n"
              "  --base-model Illustrious --model /path/model.safetensors --print-config\n"
              "  --base-model NoobAI --preset noobai-v-pred --model /path/model.safetensors\n"
              "  --backend diffusers --model /path/diffusers --prompt 'a red cube'\n"
              "  --backend comfyui --workflow /path/api.json --validate-only")
        return 0
    try:
        backend = select_backend(args, remaining)
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    if args.base_model:
        remaining = ["--base-model", args.base_model, *remaining]
    module = importlib.import_module({"preset": "generate", "diffusers": "generate_any",
                                      "comfyui": "comfyui_runtime", "local": "local_image"}[backend])
    if backend == "preset":
        # Preserve the original CLI entry point and Python API.
        previous = sys.argv
        try:
            sys.argv = [str(REFERENCE / "generate.py"), *remaining]
            return module.main()
        finally:
            sys.argv = previous
    return module.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
