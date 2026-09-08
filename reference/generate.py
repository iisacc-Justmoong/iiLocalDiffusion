#!/usr/bin/env python3
"""Generate images and videos from local paths, API endpoints or cloud model IDs."""

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
from animation_options import DEFAULTS as ANIMATION_DEFAULTS, allows_image_animation, validate_animation_output


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
    flags = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    config = {}
    configuration = option_value(remaining, "--config")
    if configuration is not None:
        from generation_config import json_object
        path = Path(configuration).expanduser()
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Configuration exceeds the 1 MiB limit.")
            config = json_object(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"--config {path}: {error}") from error

    def value(name):
        override = option_value(remaining, "--" + name.replace("_", "-"))
        return config.get(name) if override is None else override

    remote = value("model_api") is not None or value("model_cloud") is not None
    if base is not None and base["local_status"] == "hosted" and not remote:
        raise ValueError(f"{base['name']} is hosted-only in the catalog and has no local generation route.")
    if remote and base is not None:
        raise ValueError("Remote model identity comes from --model-api/--model-cloud; omit the local --base-model catalog selection.")

    animation_mode = value("animation_mode")
    output = value("output")
    fps = value("fps")
    if output is not None and not isinstance(output, str):
        raise ValueError("Video output must be a path string.")
    if fps is not None:
        if isinstance(fps, bool) or not isinstance(fps, (str, int, float)):
            raise ValueError("--fps requires a number.")
        fps = float(fps)

    def animation(backend):
        validate_animation_output(float(ANIMATION_DEFAULTS["fps"] if fps is None else fps), output)
        if base is not None and base["preset"] is None:
            raise ValueError("Animation currently requires an SD, SDXL or FLUX.1 preset family.")
        return backend

    def temporal():
        if base is not None:
            raise ValueError("The video backend requires a local LTX model, not a Civitai base family.")
        return "video"

    if args.backend != "auto":
        if args.backend in ("deforum", "interpolator"):
            return animation(args.backend)
        if args.backend == "video":
            return temporal()
        if args.backend == "preset" and base is not None and base["preset"] is None:
            raise ValueError("This model family requires the diffusers or comfyui backend.")
        return args.backend
    if config.get("backend") == "video":
        return temporal()
    if ("--list-camera-motions" in flags
            or any(value(name) is not None for name in ("storyboard", "camera", "first_frame", "last_frame", "interpolation_factor"))):
        return temporal()
    if animation_mode in ("2D", "Interpolator"):
        return animation("deforum" if animation_mode == "2D" else "interpolator")
    if "--workflow" in flags:
        return "comfyui"
    if "--pipeline-class" in flags or "--pipeline-inputs" in flags:
        return "diffusers"
    endpoints = any(value(name) is not None for name in (
        "end_prompt", "end_prompt_2", "end_negative_prompt", "end_negative_prompt_2", "end_seed"))
    video_request = (output is not None and Path(output).suffix.lower() in (".mp4", ".gif")
                     or any(value(name) is not None for name in ("fps", "frames", "max_frames", "duration"))
                     or endpoints)
    if animation_mode != "none" and video_request:
        if allows_image_animation(float(24 if fps is None else fps), output):
            return animation("interpolator" if endpoints else "deforum")
        if endpoints:
            # Endpoint interpolation is an explicit image-animation request;
            # LTX cannot silently consume those image-model controls or weights.
            validate_animation_output(float(24 if fps is None else fps), output)
        return temporal()
    if "--preset" in flags:
        if base is not None and base["preset"] is None:
            raise ValueError("--preset cannot represent this base model; choose a pipeline or workflow.")
        return "preset"
    if flags & {"--model-config", "--audio-sample-rate", "--video-layout",
                "--tensor-outputs", "--generation-architecture"}:
        return "diffusers"
    if (flags & {"--model-info", "--components", "--model-type", "--decoder", "--model-negative",
                 "--embedded-guidance", "--sampling-shift", "--zsnr",
                 "--sampler", "--startup-timeout"}
            or any(flag.startswith(("--runtime-", "--text-encoder")) for flag in flags)):
        return "local"
    model = option_value(remaining, "--model-path") or option_value(remaining, "--model")
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
    parser.add_argument("--backend", choices=("auto", "local", "preset", "diffusers", "comfyui", "comfyui-local", "deforum", "interpolator", "video"), default="auto")
    parser.add_argument("--list-base-models", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--inspect-model", action="store_true")
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens == ["--worker"]:
        from inference_worker import serve
        return serve(main)
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
            model = option_value(remaining, "--model-path") or option_value(remaining, "--model")
            if option_value(remaining, "--model-api") or option_value(remaining, "--model-cloud"):
                raise ValueError("--inspect-model requires a local --model-path.")
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
              "--worker                           Serve sequential NDJSON requests with memory caches and foreground-residency\n"
              "--inspect-model --model PATH        Inspect a local model and its Civitai metadata\n"
              "--base-model NAME                  Select a Civitai base identity\n"
              "--backend auto|local|preset|diffusers|comfyui|comfyui-local|deforum|interpolator|video\n\n"
              "Local checkpoints use standalone Diffusers/PyTorch; ComfyUI is optional.\n"
              "--backend comfyui-local            Explicit opt-in to the legacy managed ComfyUI runtime\n\n"
              "--backend video                    Generate LTX video, then interpolate its frames\n"
              "--backend deforum                  Generate a Deforum 2D MP4/GIF animation\n"
              "--backend interpolator             Generate an interpolated prompt/seed MP4/GIF\n\n"
              "Video defaults to local LTX at 24 FPS. FPS <= 12 or .gif output selects\n"
              "Deforum; end-prompt/end-seed inputs select Interpolator instead.\n"
              "Explicit image animation defaults to 12 FPS and rejects higher-FPS MP4.\n"
              "Select one model location:\n"
              "  --model-path PATH                 Local weights (--model remains an alias)\n"
              "  --model-api HTTPS_URL             Direct inference endpoint\n"
              "  --model-cloud OWNER/MODEL         Remote model ID, with --model-provider NAME\n"
              "Remote LTX video requires --model-family ltx. Tokens use --model-token-env NAME.\n"
              "Deforum/prompt Interpolator and generic tensor/workflow runtimes require local weights.\n\n"
              "LTX output above 12 FPS uses a second frame-Interpolator stage (default 2x).\n"
              "--fps is the final rate; --interpolation-factor controls source-frame spacing.\n\n"
              "Auto routes existing local weight files through the local image runtime, and\n"
              "complete model_index.json directories or --model-config through Diffusers.\n"
              "Explicit backends, pipelines and workflows retain their contracts.\n"
              "Image presets remain subject to the animation FPS/GIF policy.\n\n"
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
    if backend in ("deforum", "interpolator"):
        mode = "2D" if backend == "deforum" else "Interpolator"
        if option_value(remaining, "--animation-mode") not in (None, mode):
            parser.exit(2, f"--backend {backend} requires --animation-mode {mode}.\n")
        remaining = [*remaining, "--animation-mode", mode]
        backend = "preset"
    from inference_session import is_preparing
    if is_preparing() and backend not in ("local", "preset", "diffusers"):
        parser.exit(2, "Foreground preparation requires a local image pipeline.\n")
    module = importlib.import_module({"preset": "generate", "diffusers": "generate_any", "video": "generate_video",
                                      "comfyui": "comfyui_runtime", "comfyui-local": "local_image",
                                      "local": "standalone_image"}[backend])
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
