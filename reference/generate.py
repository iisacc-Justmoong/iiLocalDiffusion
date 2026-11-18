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


# These options exist only in the generic Diffusers runner. Shared flags such as
# --controlnet intentionally retain the existing preset/native routing policy.
GENERIC_CONTROL_OPTIONS = frozenset({
    '--controlnet-format', '--control-preprocess', '--control-preprocessor-library', '--control-reference-resize',
    '--control-depth-model', '--control-depth-dtype', '--control-color-image', '--control-map-preprocess',
    '--control-color-end', '--control-depth-end', '--control-inpaint-image', '--control-mask-image',
    '--krea2-control-auto-bucket', '--krea2-edit-reference-attention', '--krea2-fp8-storage', '--moss-transformer-storage',
    '--moss-rapid-adapter', '--moss-regions', '--moss-text-encoder', '--krea2-regions',
    '--flux-clip-encoder', '--flux-t5-encoder', '--flux-t5-fp8-storage', '--flux-transformer-storage', '--variant',
    '--dequantize-pytorch', '--no-dequantize-pytorch',
})


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


def model_value(tokens: list[str], config: dict | None = None) -> str | None:
    """Follow argparse's last-value rule across the --model aliases."""
    found = None
    for index, token in enumerate(tokens):
        if token in ('--model', '--model-path'):
            if index + 1 >= len(tokens) or tokens[index + 1].startswith('--'):
                raise ValueError(f'{token} requires a value')
            found = tokens[index + 1]
        elif token.startswith(('--model=', '--model-path=')):
            found = token.split('=', 1)[1]
    if found is not None:
        return found
    config = config or {}
    return config.get('model') if 'model' in config else config.get('model_path')


def replace_option(tokens: list[str], names: tuple[str, ...], value: str) -> list[str]:
    """Replace equivalent explicit options, or append a canonical override."""
    result, index, found = [], 0, False
    while index < len(tokens):
        token = tokens[index]
        if token in names:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith('--'):
                raise ValueError(f'{token} requires a value')
            result.extend((token, value)); index += 2; found = True
        elif any(token.startswith(name + '=') for name in names):
            result.append(token.split('=', 1)[0] + '=' + value); index += 1; found = True
        else:
            result.append(token); index += 1
    if not found:
        result.extend((names[0], value))
    return result


def legacy_inspection_python(model: str, remaining: list[str]) -> Path | None:
    """Reuse the managed safe Torch runtime without importing ML on the host."""
    path = Path(model).expanduser()
    if path.is_file():
        from downloaded_model import model_container_format
        if model_container_format(path) not in ("pytorch", "pytorch-index"):
            return None
    elif path.suffix.casefold() not in {".ckpt", ".pt", ".pth", ".bin"}:
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


def _is_moss_checkpoint(model: str | None) -> bool:
    """Recognize MOSS from validated safetensors or weights-only PyTorch metadata."""
    if not model:
        return False
    path = Path(model).expanduser()
    try:
        from downloaded_model import (_safetensors, _tensor_identity, inspect_downloaded_model,
                                      model_container_format)
        if not path.is_file() or path.suffix.casefold() == '.onnx':
            return False
        container = model_container_format(path)
        if container in ('pytorch', 'pytorch-index', 'safetensors-index'):
            report = inspect_downloaded_model(path)
            return report['architecture'] == 'moss-image' and report['role'] == 'checkpoint'
        if container != 'safetensors':
            return False
        shapes, _, _ = _safetensors(path)
        architecture, role, _, _ = _tensor_identity(shapes)
        return architecture == 'moss-image' and role == 'checkpoint'
    except (OSError, ValueError, RuntimeError):
        # Unknown/invalid files keep their existing engine validation path.
        return False


def defer_unsupported_foreground(arguments: list[str]) -> bool:
    """Do not convert legacy checkpoints or start unsupported MOSS warmups."""
    backend = option_value(arguments, '--backend')
    model = model_value(arguments)
    if backend in (None, 'auto', 'local', 'diffusers') and model:
        from downloaded_model import model_container_format
        path = Path(model).expanduser()
        try:
            if path.is_file() and path.suffix.casefold() != '.onnx' and model_container_format(path) in ('pytorch', 'pytorch-index'):
                return True
        except (OSError, ValueError):
            pass
        try:
            from image_package import inspect_if_package
            package = inspect_if_package(model)
            if package and package.get('role') == 'checkpoint':
                return True
        except (OSError, ValueError, RuntimeError):
            # A broken package must be diagnosed by the actual request, not
            # converted or marked ready during foreground notification.
            return True
    return backend == 'local' and _is_moss_checkpoint(model)


def select_backend(args, remaining: list[str]) -> str:
    args._auto_moss = None
    args._image_package = None
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
            if 'model' in config and 'model_path' in config:
                raise ValueError('Duplicate configuration value for model (including aliases).')
            config_root = path.expanduser().resolve().parent
            for key in ('model', 'model_path', 'cache_dir'):
                item = config.get(key)
                if not isinstance(item, str) or not item:
                    continue
                candidate = Path(item).expanduser()
                local = (key != 'model' or candidate.is_absolute() or item.startswith(('.', '~'))
                         or (config_root / candidate).exists()
                         or candidate.suffix.casefold() in ('.safetensors', '.safetensor', '.ckpt', '.bin', '.pt', '.pth', '.gguf'))
                if local and not candidate.is_absolute():
                    config[key] = str((config_root / candidate).resolve())
        except (OSError, UnicodeError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"--config {path}: {error}") from error
    args._configuration = config

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
    unified_path = model_value(remaining, config)
    if unified_path:
        selected = Path(unified_path).expanduser()
        if selected.is_file() and selected.suffix.lower() == ".iildmodel":
            return "unified"
        index = selected / "model_index.json"
        if index.is_file() and index.stat().st_size <= 1024 * 1024:
            package = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(package, dict) and package.get("schema") == "iild-unified-model-v1":
                return "unified"
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
    model = model_value(remaining, config)
    if isinstance(model, str):
        from image_package import inspect_if_package
        args._image_package = inspect_if_package(model, getattr(args, 'package_model', None))
        args._auto_moss = (args._image_package.get('architecture') == 'moss-image'
                           if args._image_package else _is_moss_checkpoint(model))
    if args._auto_moss:
        native_flags = {"--engine", "--model-info", "--components", "--model-type", "--decoder",
                        "--model-negative", "--embedded-guidance", "--sampling-shift", "--zsnr",
                        "--sampler", "--startup-timeout"}
        configured_flags = {'--' + key.replace('_', '-') for key, setting in config.items() if setting is not None}
        unsupported = sorted(flag for flag in flags | configured_flags if flag in native_flags
                             or flag.startswith(("--runtime-", "--text-encoder")))
        if unsupported:
            raise ValueError("The MOSS Image backend does not accept native-only options: " + ", ".join(unsupported))
        if base is not None and base.get("family") != "moss-image":
            raise ValueError("--base-model conflicts with the detected MOSS Image tensor architecture.")
        if value('model_config_preset') not in (None, 'moss-image-sft'):
            raise ValueError('The requested model configuration conflicts with the detected MOSS transformer.')
        if value('pipeline_class') not in (None, 'MOSSImagePipeline'):
            raise ValueError('The requested pipeline conflicts with the detected MOSS transformer.')
        return "diffusers"
    if (value('model_config') is not None or value('model_config_preset') is not None
            or config.get('dequantize_pytorch') is not None
            or flags & ({"--model-config", "--model-config-preset", "--audio-sample-rate", "--video-layout",
                         "--tensor-outputs", "--generation-architecture"} | GENERIC_CONTROL_OPTIONS)):
        return "diffusers"
    if (value("engine") is not None or value("components") is not None
            or flags & {"--engine", "--model-info", "--components", "--model-type", "--decoder", "--model-negative",
                 "--embedded-guidance", "--sampling-shift", "--zsnr",
                 "--sampler", "--startup-timeout"}
            or any(flag.startswith(("--runtime-", "--text-encoder")) for flag in flags)):
        return "local"
    if model:
        local_path = Path(model).expanduser()
        if local_path.is_file():
            from gguf_component import nearby_config
            if nearby_config(local_path) is not None:
                return 'diffusers'
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
    parser.add_argument("--backend", choices=("auto", "local", "preset", "diffusers", "comfyui", "comfyui-local", "deforum", "interpolator", "video", "unified"), default="auto")
    parser.add_argument("--list-base-models", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--inspect-model", action="store_true")
    parser.add_argument("--package-model", help="Original checkpoint or index path inside an image package")
    parser.add_argument("--package-components", help="JSON mapping of native component slots to original package paths")
    parser.add_argument("--list-backends", action="store_true", help="List architecture contracts without loading inference runtimes")
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens == ["--worker"]:
        from inference_worker import serve
        return serve(main, defer_foreground=defer_unsupported_foreground)
    args, remaining = parser.parse_known_args(tokens)
    if args.list_backends:
        from backend_registry import PROFILES
        print(json.dumps({"schema": "iild-backends-v1", "native_image": {
            name: {"components": ["vae", *values[0]], "vae_family": values[1], "formats": ["safetensors", "gguf"]}
            for name, values in PROFILES.items()}, "pipeline_packages": {
                "backend": "diffusers", "configuration": "model_index.json",
                "outputs": ["image", "video", "audio", "tensor"],
                "capability": "determined by the installed built-in pipeline and its input/output contract"}}, indent=2))
        return 0
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
            if args.package_components is not None:
                raise ValueError('--package-components prepares weights during generation; omit it for metadata inspection.')
            model = model_value(remaining)
            if option_value(remaining, "--model-api") or option_value(remaining, "--model-cloud"):
                raise ValueError("--inspect-model requires a local --model-path.")
            if not model:
                raise ValueError("--inspect-model requires --model with a local weight file")
            interpreter = legacy_inspection_python(model, remaining)
            if interpreter is not None:
                return subprocess.call([str(interpreter), str(Path(__file__).absolute()), *tokens])
            if Path(model).expanduser().is_dir():
                if args.package_model is not None:
                    raise ValueError('--package-model requires an embedded image package file.')
                from backend_registry import inspect_pipeline_package
                report = inspect_pipeline_package(model)
            else:
                report = inspect_downloaded_model(model, option_value(remaining, "--model-info"), args.package_model)
            component_json = option_value(remaining, "--components")
            if component_json is not None:
                from backend_registry import plan_native
                from generation_config import json_object
                report["backend_plan"] = plan_native(report, json_object(component_json))
        except (ValueError, OSError) as error:
            parser.exit(2, str(error) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if tokens in ([], ["--help"], ["-h"]):
        print(__doc__ + "\n\n"
              "--list-base-models                 List the pinned Civitai compatibility catalog\n"
              "--list-backends                    List native architecture and pipeline output contracts\n"
              "--check-runtime                    Audit installed pipelines without downloading weights\n"
              "--worker                           Serve sequential NDJSON requests with memory caches and foreground-residency\n"
              "--inspect-model --model PATH        Inspect a local model and its Civitai metadata\n"
              "--package-model PATH              Select an original checkpoint inside a model package\n"
              "--package-components JSON         Map native slots to original paths inside a model package\n"
              "--base-model NAME                  Select a Civitai base identity\n"
              "--backend auto|local|preset|diffusers|comfyui|comfyui-local|deforum|interpolator|video|unified\n\n"
              "Local checkpoints select native or Diffusers inference by architecture; ComfyUI is optional.\n"
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
              "complete model_index.json directories, --model-config, or --model-config-preset through Diffusers.\n"
              "Unified .iildmodel objects use the native ordered image-refinement cascade.\n"
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
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
    model = model_value(remaining, args._configuration)
    if backend in ('local', 'diffusers') and model and args._image_package is None:
        try:
            from image_package import inspect_if_package
            args._image_package = inspect_if_package(model, args.package_model)
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(2, str(error) + '\n')
    package = args._image_package
    if args.package_model is not None and (backend not in ('local', 'diffusers') or not package):
        parser.exit(2, '--package-model requires an embedded image package on the local or Diffusers route.\n')
    packaged_moss = bool(package and package.get('architecture') == 'moss-image' and backend == 'diffusers')
    packaged_anima = bool(package and package.get('architecture') == 'anima' and backend == 'local')
    from backend_registry import PROFILES
    packaged_generic = bool(package and package.get('architecture') in PROFILES
                            and package.get('architecture') != 'anima' and backend in ('local', 'diffusers'))
    if args.package_components is not None and not (packaged_moss or packaged_anima or
                                                    (packaged_generic and backend == 'local')):
        parser.exit(2, '--package-components requires a recognized native image or MOSS package route.\n')
    from inference_session import is_preparing
    if (packaged_moss or packaged_anima or packaged_generic) and is_preparing():
        parser.exit(2, 'Embedded package preparation is deferred until generation.\n')
    if packaged_moss:
        args._auto_moss = True
    if args.backend == 'auto' and backend == 'diffusers' and not any(
            token.split('=', 1)[0] in ('--model-config', '--model-config-preset') for token in remaining) and not any(
            args._configuration.get(key) is not None for key in ('model_config', 'model_config_preset')):
        detected = args._auto_moss if args._auto_moss is not None else _is_moss_checkpoint(model)
        if detected:
            remaining = [*remaining, '--model-config-preset', 'moss-image-sft']
    if packaged_moss:
        try:
            if '--print-config' in remaining or '--validate-only' in remaining:
                raise ValueError('Package weight preparation is deferred until generation; use --inspect-model for metadata.')
            def configured(flag, key):
                explicit = option_value(remaining, flag)
                return args._configuration.get(key) if explicit is None else explicit
            preset = configured('--model-config-preset', 'model_config_preset')
            if preset is not None and preset != 'moss-image-sft':
                raise ValueError('The requested model configuration conflicts with the packaged MOSS transformer.')
            pipeline = configured('--pipeline-class', 'pipeline_class')
            if pipeline is not None and pipeline != 'MOSSImagePipeline':
                raise ValueError('The requested pipeline conflicts with the packaged MOSS transformer.')
            if configured('--model-config', 'model_config') is None and preset is None:
                remaining = [*remaining, '--model-config-preset', 'moss-image-sft']
            from generation_config import json_object
            package_components = json_object(args.package_components) if args.package_components is not None else {}
            flags = {'llm': ('--moss-text-encoder', 'moss_text_encoder'), 'vae': ('--vae', 'vae')}
            for slot, path in package_components.items():
                if slot not in flags or not isinstance(path, str) or not path:
                    raise ValueError('MOSS package components must map llm or vae to an original path.')
                if configured(*flags[slot]) is not None:
                    raise ValueError(f'Component {slot} was supplied both inside and outside the package.')
            for slot in package.get('ambiguous_components', {}):
                if slot not in package_components and configured(*flags[slot]) is None:
                    raise ValueError(f'Multiple MOSS {slot} components in the package; choose one explicitly.')
            from image_package import materialize, progress_reporter, verify_source
            from resource_paths import CACHE_DIRECTORY
            cache_option = configured('--cache-dir', 'cache_dir')
            if cache_option is not None and not cache_option:
                raise ValueError('--cache-dir must not be empty.')
            cache = Path(CACHE_DIRECTORY if cache_option is None else cache_option).expanduser().absolute() / 'embedded-models'
            progress = progress_reporter()
            transformer = materialize(package, cache, package['transformer'], progress)
            from downloaded_model import model_container_format
            for slot, source in package_components.items():
                prepared = materialize(package, cache, source, progress)
                if model_container_format(Path(prepared)) not in ('safetensors', 'gguf'):
                    raise ValueError(f'Package component {slot} must be Safetensors or GGUF weights.')
                remaining = [*remaining, flags[slot][0], prepared]
            if package.get('text_encoder') and configured('--moss-text-encoder', 'moss_text_encoder') is None:
                remaining = [*remaining, '--moss-text-encoder', materialize(package, cache, package['text_encoder'], progress)]
            if package.get('vae') and configured('--vae', 'vae') is None:
                remaining = [*remaining, '--vae', materialize(package, cache, package['vae'], progress)]
            remaining = replace_option(remaining, ('--model-path', '--model'), transformer)
            verify_source(package)
            package = {**package, 'prepared_transformer': transformer,
                       'selected_components': dict(package_components)}
        except (ValueError, OSError, RuntimeError, argparse.ArgumentTypeError) as error:
            parser.exit(2, str(error) + '\n')
    if packaged_anima:
        try:
            if '--print-config' in remaining or '--validate-only' in remaining:
                raise ValueError('Package weight preparation is deferred until generation; use --inspect-model for metadata.')
            if (option_value(remaining, '--engine') or args._configuration.get('engine')) == 'diffusers':
                raise ValueError('The embedded Anima package needs the native engine for automatic component preparation.')
            from generation_config import json_object
            selected = option_value(remaining, '--components')
            components = json_object(selected) if selected is not None else args._configuration.get('components')
            if components is None:
                components = {}
            if not isinstance(components, dict):
                raise ValueError('Native components must be a JSON object.')
            components = dict(components)
            package_components = json_object(args.package_components) if args.package_components is not None else {}
            explicit_vae = option_value(remaining, '--vae') is not None or args._configuration.get('vae') is not None
            for slot, path in package_components.items():
                if slot not in ('llm', 'vae') or not isinstance(path, str) or not path:
                    raise ValueError('Anima package components must map llm or vae to an original path.')
                if slot in components or (slot == 'vae' and explicit_vae):
                    raise ValueError(f'Component {slot} was supplied both inside and outside the package.')
            for slot in package.get('ambiguous_components', {}):
                if slot not in components and slot not in package_components and not (slot == 'vae' and explicit_vae):
                    raise ValueError(f'Multiple Anima {slot} components in the package; choose one explicitly.')
            from image_package import materialize, progress_reporter, verify_source
            from resource_paths import CACHE_DIRECTORY
            cache_option = option_value(remaining, '--cache-dir')
            if cache_option is None:
                cache_option = args._configuration.get('cache_dir')
            if cache_option is None:
                cache_option = CACHE_DIRECTORY
            if not cache_option:
                raise ValueError('--cache-dir must not be empty.')
            cache = Path(cache_option)
            cache = cache.expanduser().absolute() / 'embedded-models'
            progress = progress_reporter()
            transformer = materialize(package, cache, package['transformer'], progress)
            from downloaded_model import model_container_format
            for slot, source in package_components.items():
                prepared = materialize(package, cache, source, progress)
                if model_container_format(Path(prepared)) not in ('safetensors', 'gguf'):
                    raise ValueError(f'Package component {slot} must be Safetensors or GGUF weights.')
                components[slot] = prepared
            if package.get('text_encoder') and 'llm' not in components:
                components['llm'] = materialize(package, cache, package['text_encoder'], progress)
            if package.get('vae') and 'vae' not in components and option_value(remaining, '--vae') is None and args._configuration.get('vae') is None:
                components['vae'] = materialize(package, cache, package['vae'], progress)
            remaining = replace_option(remaining, ('--model-path', '--model'), transformer)
            if components:
                remaining = replace_option(remaining, ('--components',), json.dumps(components, ensure_ascii=False, allow_nan=False))
            verify_source(package)
            package = {**package, 'prepared_transformer': transformer,
                       'selected_components': dict(package_components)}
        except (ValueError, OSError, RuntimeError, argparse.ArgumentTypeError) as error:
            parser.exit(2, str(error) + '\n')
    if packaged_generic:
        try:
            if '--print-config' in remaining or '--validate-only' in remaining:
                raise ValueError('Package weight preparation is deferred until generation; use --inspect-model for metadata.')
            from image_package import choose_native_text_encoders, choose_native_vae, materialize, progress_reporter, verify_source
            from resource_paths import CACHE_DIRECTORY
            from generation_config import json_object
            selected = option_value(remaining, '--components')
            components = json_object(selected) if selected is not None else args._configuration.get('components')
            if components is None:
                components = {}
            if not isinstance(components, dict):
                raise ValueError('Native components must be a JSON object.')
            components = dict(components)
            package_components = json_object(args.package_components) if args.package_components is not None else {}
            allowed = {*PROFILES[package['architecture']][0], 'vae'}
            for slot, path in package_components.items():
                if slot not in allowed or not isinstance(path, str) or not path:
                    raise ValueError('Package components must map native slots for the detected architecture to original paths.')
                if slot in components or (slot == 'vae' and (option_value(remaining, '--vae') is not None
                                                               or args._configuration.get('vae') is not None)):
                    raise ValueError(f'Component {slot} was supplied both inside and outside the package.')
            if package_components and (option_value(remaining, '--engine') or args._configuration.get('engine')) == 'diffusers':
                raise ValueError('--package-components requires the native image engine.')
            cache_option = option_value(remaining, '--cache-dir')
            if cache_option is None:
                cache_option = args._configuration.get('cache_dir')
            if cache_option is None:
                cache_option = CACHE_DIRECTORY
            if not cache_option:
                raise ValueError('--cache-dir must not be empty.')
            cache = Path(cache_option).expanduser().absolute() / 'embedded-models'
            progress = progress_reporter()
            transformer = materialize(package, cache, package['transformer'], progress)
            from downloaded_model import model_container_format
            if backend == 'local' and (option_value(remaining, '--engine') or args._configuration.get('engine')) != 'diffusers':
                if ('vae' not in components and 'vae' not in package_components
                        and option_value(remaining, '--vae') is None and args._configuration.get('vae') is None):
                    chosen_vae = choose_native_vae(package, transformer, cache, progress)
                    if chosen_vae is not None:
                        package_components['vae'] = chosen_vae
                missing_slots = [slot for slot in PROFILES[package['architecture']][0]
                                 if slot in ('clip_l', 'clip_g', 't5xxl')
                                 and slot not in components and slot not in package_components]
                package_components.update(choose_native_text_encoders(package, transformer, cache, missing_slots, progress))
            for slot, source in package_components.items():
                prepared = materialize(package, cache, source, progress)
                if model_container_format(Path(prepared)) not in ('safetensors', 'gguf'):
                    raise ValueError(f'Package component {slot} must be Safetensors or GGUF weights.')
                components[slot] = prepared
            remaining = replace_option(remaining, ('--model-path', '--model'), transformer)
            if package_components:
                remaining = replace_option(remaining, ('--components',), json.dumps(components, ensure_ascii=False, allow_nan=False))
            verify_source(package)
            package = {**package, 'prepared_transformer': transformer,
                       'selected_components': dict(package_components)}
        except (ValueError, OSError, RuntimeError, argparse.ArgumentTypeError) as error:
            parser.exit(2, str(error) + '\n')
    if args.base_model:
        remaining = ["--base-model", args.base_model, *remaining]
    if backend in ("deforum", "interpolator"):
        mode = "2D" if backend == "deforum" else "Interpolator"
        if option_value(remaining, "--animation-mode") not in (None, mode):
            parser.exit(2, f"--backend {backend} requires --animation-mode {mode}.\n")
        remaining = [*remaining, "--animation-mode", mode]
        backend = "preset"
    if is_preparing() and backend not in ("local", "preset", "diffusers", "unified"):
        parser.exit(2, "Foreground preparation requires a local image pipeline.\n")
    module = importlib.import_module({"preset": "generate", "diffusers": "generate_any", "video": "generate_video",
                                      "comfyui": "comfyui_runtime", "comfyui-local": "local_image",
                                      "local": "standalone_image", "unified": "unified_image"}[backend])
    if backend == "preset":
        # Preserve the original CLI entry point and Python API.
        previous = sys.argv
        try:
            sys.argv = [str(REFERENCE / "generate.py"), *remaining]
            return module.main()
        finally:
            sys.argv = previous
    if packaged_moss or packaged_anima or packaged_generic:
        from image_package import use
        with use(package):
            return module.main(remaining)
    return module.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
