#!/usr/bin/env python3
"""Run an installed Diffusers pipeline without assuming an SD1/SDXL/FLUX contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import wave
from typing import Any

from generation_output import publish_file
from hardware import accelerator_preflight, select_device, validate_execution_device
from weight_files import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "build/reference/huggingface"
IMAGE_INPUTS = frozenset({
    "image", "images", "mask_image", "control_image", "control_images", "reference_image",
    "reference_images", "ip_adapter_image", "conditioning_image", "video", "frames",
    "last_image", "start_image", "end_image", "image_2", "image_3", "image_4",
})
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
UNSAFE_WEIGHTS = frozenset({".ckpt", ".pt", ".pth", ".pkl", ".pickle", ".bin"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True,
                        help="Local Diffusers directory, .safetensors file, or pinned Hub repository")
    parser.add_argument("--revision", help="Required immutable 40-character commit for a Hub repository")
    parser.add_argument("--pipeline-class", help="Installed diffusers pipeline class; otherwise use model_index.json")
    parser.add_argument("--model-config", help="Local complete Diffusers configuration/extras for a single file")
    parser.add_argument("--base-model", help="Civitai identity checked against the selected pipeline architecture")
    parser.add_argument("--pipeline-inputs", default="{}",
                        help="JSON object or @JSON-file containing explicit pipeline __call__ arguments")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "metal", "rocm"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="none")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/reference/generic-output")
    parser.add_argument("--audio-sample-rate", type=int,
                        help="WAV rate; otherwise infer from the pipeline's vocoder/VAE/config")
    parser.add_argument("--video-layout", choices=("auto", "bfhwc", "bfchw", "bcfhw"), default="auto",
                        help="Array video layout; auto uses BFHWC for NumPy and Diffusers BFCHW for Torch")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not supported: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number is not supported: {value}")
    return parsed


def read_json(text: str) -> Any:
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def read_inputs(source: str) -> dict[str, Any]:
    value = read_json(Path(source[1:]).expanduser().read_text() if source.startswith("@") else source)
    if not isinstance(value, dict):
        raise ValueError("--pipeline-inputs must contain a JSON object.")
    for name in value:
        if not IDENTIFIER.fullmatch(name):
            raise ValueError(f"Invalid pipeline argument name: {name!r}")
    if "return_dict" in value and value["return_dict"] is not True:
        raise ValueError("return_dict must be true when supplied; named outputs are required for export.")
    if value.get("output_type") in ("latent", "latents"):
        raise ValueError("Latent outputs cannot be exported as generated media.")
    if "generator" in value:
        raise ValueError("Use --seed to construct a torch.Generator; JSON generator objects are unsupported.")
    return value


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    if not args.model.strip():
        raise ValueError("--model must not be empty.")
    source = Path(args.model).expanduser()
    if source.exists():
        source = source.absolute()
        result.model = str(source)
        result.source_kind = "directory" if source.is_dir() else "single-file"
        if args.revision is not None:
            raise ValueError("--revision applies only to a remote Hub repository, not local files.")
        if not source.is_dir() and (not source.is_file() or source.suffix != ".safetensors" or not source.stat().st_size):
            raise ValueError("Local model files must be non-empty .safetensors files; use a Diffusers directory for other layouts.")
    else:
        if (args.model.startswith(("/", "~", ".")) or "\\" in args.model
                or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.model)
                or source.suffix in UNSAFE_WEIGHTS | {".safetensors", ".gguf"}):
            raise ValueError(f"Local model does not exist, or invalid Hub repository: {args.model}")
        if not args.revision or not IMMUTABLE_REVISION.fullmatch(args.revision):
            raise ValueError("A remote --model requires --revision with an immutable lowercase 40-character commit SHA.")
        result.source_kind = "hub"
    if result.source_kind == "single-file":
        if not args.model_config:
            raise ValueError("A single-file model requires --model-config with a local Diffusers configuration directory and required extras.")
        config = Path(args.model_config).expanduser().absolute()
        if not config.is_dir() or not (config / "model_index.json").is_file():
            raise ValueError("--model-config must be a local directory containing model_index.json.")
        result.model_config = str(config)
    elif args.model_config:
        raise ValueError("--model-config is only valid for a single-file --model.")
    if args.pipeline_class and not IDENTIFIER.fullmatch(args.pipeline_class):
        raise ValueError("--pipeline-class must be a public installed Diffusers class name, not a module or script path.")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - 1:
        raise ValueError("--seed must be in the inclusive range 0..2^63-1.")
    for name in ("width", "height", "steps", "audio_sample_rate"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.guidance_scale is not None and not math.isfinite(args.guidance_scale):
        raise ValueError("--guidance-scale must be finite.")
    if args.device == "cpu" and args.offload != "none":
        raise ValueError("CPU execution requires --offload none.")
    result.inputs = read_inputs(args.pipeline_inputs)
    for option, key in (("prompt", "prompt"), ("negative_prompt", "negative_prompt"),
                        ("width", "width"), ("height", "height"),
                        ("steps", "num_inference_steps"), ("guidance_scale", "guidance_scale")):
        value = getattr(args, option)
        if value is not None:
            result.inputs[key] = value
    result.cache_dir = args.cache_dir.expanduser().absolute()
    result.output_dir = args.output_dir.expanduser().absolute()
    return result


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner": "generic-diffusers", "model": args.model, "source_kind": args.source_kind,
        "revision": args.revision, "model_config": args.model_config, "base_model": args.base_model,
        "pipeline_class": args.pipeline_class, "pipeline_inputs": args.inputs, "seed": args.seed,
        "device": args.device, "dtype": args.dtype, "offload": args.offload,
        "local_files_only": args.local_files_only, "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir), "audio_sample_rate": args.audio_sample_rate,
        "video_layout": args.video_layout,
        "overwrite": args.overwrite,
    }


def validate_model_index(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model_index.json must contain an object.")
    name = value.get("_class_name")
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ValueError("model_index.json must identify a built-in pipeline class; custom pipeline scripts are unsupported.")
    for component, selection in value.items():
        if component.startswith("_") or not isinstance(selection, (list, tuple)):
            continue
        if not IDENTIFIER.fullmatch(component) or len(selection) != 2:
            raise ValueError(f"Invalid model component declaration: {component}")
        library, class_name = selection
        if library is None and class_name is None:
            continue
        if (not isinstance(library, str) or not IDENTIFIER.fullmatch(library)
                or not isinstance(class_name, str) or not IDENTIFIER.fullmatch(class_name)):
            raise ValueError(f"Custom or malformed component declaration: {component}")
    return value


def builtin_pipeline(diffusers: Any, name: str) -> Any:
    candidate = getattr(diffusers, name, None)
    if (not inspect.isclass(candidate) or not issubclass(candidate, diffusers.DiffusionPipeline)
            or candidate is diffusers.DiffusionPipeline
            or not candidate.__module__.startswith("diffusers.pipelines.")):
        raise ValueError(f"{name!r} is not a built-in pipeline in the installed Diffusers runtime. Use a compatible installed runtime or the workflow backend.")
    return candidate


def architecture_group(pipeline_class: Any) -> str:
    # These task siblings live in multiple namespaces (notably ControlNet), so
    # namespace equality alone incorrectly rejects compatible editing pipelines.
    name = pipeline_class.__name__
    for prefix, group in (("StableDiffusionXL", "sdxl"), ("StableDiffusion3", "sd3"),
                          ("StableDiffusion", "sd1-or-sd2"), ("Flux2", "flux2"),
                          ("Flux", "flux1"), ("QwenImage", "qwen-image"),
                          ("HunyuanDiT", "hunyuan-dit")):
        if name.startswith(prefix):
            return group
    parts = pipeline_class.__module__.split(".")
    if len(parts) < 3 or parts[:2] != ["diffusers", "pipelines"]:
        raise ValueError(f"Cannot identify the architecture namespace for {name}.")
    return "diffusers:" + parts[2]


def validate_base_model(label: str | None, pipeline_class: Any, diffusers: Any) -> dict[str, Any]:
    actual = architecture_group(pipeline_class)
    if label is None:
        return {"status": "not-declared", "actual_architecture": actual}
    from civitai_catalog import lookup_base_model
    record = lookup_base_model(label)
    if record["local_status"] == "hosted":
        raise ValueError(f"{record['name']} is a hosted-only Civitai base model, not a local pipeline.")
    expected_name = record["pipeline_class"]
    if expected_name is None:
        return {"status": "declared-only", "catalog_name": record["name"],
                "actual_architecture": actual,
                "reason": "The catalog has no conventional built-in pipeline class for this label."}
    expected = architecture_group(builtin_pipeline(diffusers, expected_name))
    if actual != expected:
        raise ValueError(f"Civitai base model {record['name']!r} requires architecture {expected!r} "
                         f"({expected_name}), but {pipeline_class.__name__} uses {actual!r}.")
    return {"status": "pipeline-architecture-verified", "catalog_name": record["name"],
            "catalog_pipeline_class": expected_name, "actual_architecture": actual,
            "scope": "Pipeline architecture only; checkpoint identity, SD1 versus SD2, and variants within one family are not inferred."}


def validate_components(diffusers: Any, index: dict[str, Any], folder: Path | None = None) -> None:
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, (list, tuple)) or value == [None, None]:
            continue
        library, _ = value
        if library not in ("diffusers", "transformers"):
            module = getattr(diffusers.pipelines, library, None)
            if module is None or not getattr(module, "__name__", "").startswith("diffusers.pipelines."):
                raise ValueError(f"Component {name} requests non-built-in library {library!r}.")
        if folder is not None and (folder / name / f"{library}.py").exists():
            raise ValueError(f"Custom local component code is unsupported: {name}/{library}.py")


def load_model_index(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    if args.source_kind == "hub":
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(args.model, "model_index.json", revision=args.revision,
                                   cache_dir=str(args.cache_dir), local_files_only=args.local_files_only))
        folder = None
    else:
        folder = Path(args.model_config if args.source_kind == "single-file" else args.model)
        path = folder / "model_index.json"
    return validate_model_index(read_json(path.read_text())), folder


def file_identity(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {"path": str(path.relative_to(relative_to)) if relative_to else str(path),
            "resolved_path": str(path.resolve()), "size_bytes": after.st_size, "sha256": digest}


def model_identity(folder: Path, single_file: str | None = None) -> dict[str, Any]:
    files = [file_identity(path, folder) for path in sorted(folder.rglob("*"))
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(folder).parts)
             and path.suffix not in UNSAFE_WEIGHTS | {".py", ".pyc"}]
    result: dict[str, Any] = {"directory": str(folder), "files": files}
    if single_file:
        result["single_file"] = file_identity(Path(single_file))
    return result


def verify_identity(identity: dict[str, Any]) -> None:
    entries = list(identity["files"])
    if identity.get("single_file"):
        entries.append(identity["single_file"])
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = Path(identity["directory"]) / path
        if (not path.is_file() or str(path.resolve()) != entry["resolved_path"]
                or path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]):
            raise RuntimeError(f"Model file changed while loading/generating: {path}")


def load_pipeline(args: argparse.Namespace, diffusers: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    index, folder = load_model_index(args)
    validate_components(diffusers, index, folder)
    pipeline_class = builtin_pipeline(diffusers, args.pipeline_class or index["_class_name"])
    validate_base_model(args.base_model, pipeline_class, diffusers)
    # Validate user input names before allocating model weights, including **kwargs pipelines.
    validate_call_arguments(pipeline_class.__call__, args.inputs, seed=args.seed)
    if args.source_kind == "hub":
        folder = Path(pipeline_class.download(args.model, revision=args.revision,
                      cache_dir=str(args.cache_dir), local_files_only=args.local_files_only,
                      use_safetensors=True, trust_remote_code=False))
        local_index = validate_model_index(read_json((folder / "model_index.json").read_text()))
        if local_index != index:
            raise RuntimeError("Downloaded model_index.json differs from the pinned index.")
        validate_components(diffusers, local_index, folder)
    assert folder is not None
    if args.source_kind == "single-file":
        if not hasattr(pipeline_class, "from_single_file"):
            raise ValueError(f"{pipeline_class.__name__} has no built-in single-file loader. Provide a complete Diffusers directory or use the workflow backend.")
        unsafe = [str(path.relative_to(folder)) for path in folder.rglob("*")
                  if path.is_file() and path.suffix in UNSAFE_WEIGHTS]
        if unsafe:
            raise ValueError("Single-file configuration extras must contain safetensors weights only; remove/convert pickle weights: " + ", ".join(unsafe[:5]))
    identity = model_identity(folder, args.model if args.source_kind == "single-file" else None)
    kwargs = {"dtype": dtype, "local_files_only": True, "cache_dir": str(args.cache_dir),
              "use_safetensors": True, "trust_remote_code": False}
    if args.source_kind == "single-file":
        pipeline = pipeline_class.from_single_file(args.model, config=str(folder), **kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(str(folder), **kwargs)
    if pipeline.__class__ is not pipeline_class:
        raise RuntimeError(f"Requested {pipeline_class.__name__}, but loader instantiated {pipeline.__class__.__name__}.")
    verify_identity(identity)
    return pipeline, {"identity": identity, "index_class": index["_class_name"],
                      "actual_class": pipeline.__class__.__name__,
                      "base_model_validation": validate_base_model(args.base_model, pipeline.__class__, diffusers)}


def validate_call_arguments(call: Any, inputs: dict[str, Any], *, seed: int | None) -> None:
    signature = inspect.signature(call)
    allowed = {name for name, parameter in signature.parameters.items()
               if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
    allowed.discard("self")
    requested = set(inputs) | ({"generator"} if seed is not None else set())
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError("Pipeline does not explicitly support input(s): " + ", ".join(unknown)
                         + ". Unknown **kwargs are refused rather than silently ignored.")
    required = {name for name, parameter in signature.parameters.items()
                if name != "self" and parameter.default is inspect.Parameter.empty
                and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,
                                       parameter.KEYWORD_ONLY)}
    missing = sorted(required - requested)
    if missing:
        raise ValueError("Pipeline requires input(s): " + ", ".join(missing))


def prepare_inputs(inputs: dict[str, Any], image_module: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identities: list[dict[str, Any]] = []

    def convert(value: Any, root_key: str) -> Any:
        if isinstance(value, list):
            return [convert(item, root_key) for item in value]
        if isinstance(value, dict):
            if "image_path" in value:
                if set(value) - {"image_path", "mode"} or root_key not in IMAGE_INPUTS:
                    raise ValueError(f"image_path is only accepted for supported image/video inputs, not {root_key!r}.")
                if not isinstance(value["image_path"], str):
                    raise ValueError("image_path must be a local file path string.")
                mode = value.get("mode", "RGB")
                if mode not in ("RGB", "RGBA", "L"):
                    raise ValueError("Image mode must be RGB, RGBA, or L.")
                path = Path(value["image_path"]).expanduser().absolute()
                identity = file_identity(path)
                with image_module.open(path) as opened:
                    if getattr(opened, "is_animated", False):
                        raise ValueError("Animated input files are unsupported; pass an explicit list of frame image_path objects.")
                    opened.load()
                    image = opened.convert(mode)
                if file_sha256(path) != identity["sha256"]:
                    raise RuntimeError(f"Image input changed while decoding: {path}")
                identities.append({**identity, "input": root_key, "mode": mode, "size": list(image.size)})
                return image
            return {key: convert(item, root_key) for key, item in value.items()}
        return value

    return {name: convert(value, name) for name, value in inputs.items()}, identities


def prepare_execution(pipeline: Any, device: str, offload: str) -> Any:
    if offload == "none":
        pipeline = pipeline.to(device)
    elif offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.enable_sequential_cpu_offload(device=device)
    validate_execution_device(pipeline, device)
    return pipeline


def output_value(result: Any, key: str) -> Any:
    return result.get(key) if isinstance(result, dict) else getattr(result, key, None)


def as_array(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def as_image(value: Any, image_module: Any, np: Any) -> Any:
    if isinstance(value, image_module.Image):
        return value
    tensor = hasattr(value, "detach")
    array = as_array(value, np)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(f"Generated frame requires three dimensions, got {array.shape}.")
    if tensor:
        if array.shape[0] not in (1, 3, 4):
            raise ValueError(f"Generated Torch images require CHW layout, got {array.shape}.")
        array = array.transpose(1, 2, 0)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported generated frame channel layout: {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.size == 0 or array.min() < 0 or array.max() > 1:
            raise ValueError("Generated floating frames must be finite decoded pixels in [0, 1].")
        array = (array * 255).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError("Generated frames must contain uint8 or decoded floating pixels.")
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    return image_module.fromarray(array)


def atomic_write(path: Path, writer: Any, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        publish_file(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def infer_sample_rate(pipeline: Any, requested: int | None) -> int:
    if requested is not None:
        return requested
    for component in (getattr(pipeline, "vocoder", None), getattr(pipeline, "vae", None), pipeline):
        config = getattr(component, "config", None)
        for key in ("sampling_rate", "sample_rate"):
            value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError("Audio output requires --audio-sample-rate because the pipeline does not expose its sampling rate.")


def save_outputs(result: Any, pipeline: Any, args: argparse.Namespace, image_module: Any,
                 np: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    def save_image(value: Any, filename: str, kind: str) -> None:
        image = as_image(value, image_module, np)
        path = args.output_dir / filename
        atomic_write(path, lambda temporary: image.save(temporary, format="PNG"), args.overwrite)
        outputs.append({**file_identity(path), "kind": kind, "size": list(image.size), "mode": image.mode})

    images = output_value(result, "images")
    frames = output_value(result, "frames")
    audio = output_value(result, "audios")
    if audio is None:
        audio = output_value(result, "audio")
    if images is not None:
        if isinstance(images, image_module.Image):
            images = [images]
        elif hasattr(images, "shape") and len(images.shape) == 3:
            images = [images]
        if len(images) == 0:
            raise ValueError("The pipeline returned an empty images batch.")
        for index, image in enumerate(images):
            save_image(image, f"image-{index + 1:04d}.png", "image")
    if frames is not None:
        if isinstance(frames, list) and frames and isinstance(frames[0], image_module.Image):
            frames = [frames]
        elif hasattr(frames, "shape"):
            tensor = hasattr(frames, "detach")
            frames = as_array(frames, np)
            if frames.ndim == 4:
                frames = frames[None, ...]
            if frames.ndim != 5:
                raise ValueError("Video output requires a batch of frame sequences.")
            layout = args.video_layout
            if layout == "auto":
                layout = "bfchw" if tensor else "bfhwc"
            if layout == "bfchw":
                if frames.shape[2] not in (1, 3, 4):
                    raise ValueError("BFCHW video output has an invalid channel axis; select the actual --video-layout.")
                frames = frames.transpose(0, 1, 3, 4, 2)
            elif layout == "bcfhw":
                if frames.shape[1] not in (1, 3, 4):
                    raise ValueError("BCFHW video output has an invalid channel axis.")
                frames = frames.transpose(0, 2, 3, 4, 1)
            elif frames.shape[-1] not in (1, 3, 4):
                raise ValueError("BFHWC video output has an invalid channel axis; select the actual --video-layout.")
        if len(frames) == 0:
            raise ValueError("The pipeline returned an empty video batch.")
        for batch, sequence in enumerate(frames):
            if len(sequence) == 0:
                raise ValueError("The pipeline returned an empty video frame sequence.")
            for index, frame in enumerate(sequence):
                save_image(frame, f"video-{batch + 1:04d}/frame-{index + 1:06d}.png", "video-frame")
    if audio is not None:
        array = as_array(audio, np)
        if array.ndim == 1:
            array = array[None, ...]
        if array.ndim not in (2, 3) or not np.isfinite(array).all() or array.size == 0:
            raise ValueError("Audio output must be a finite non-empty [batch, samples] or [batch, channels, samples] array.")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Audio output must contain floating amplitudes in [-1, 1], not integer/encoded samples.")
        rate = infer_sample_rate(pipeline, args.audio_sample_rate)
        for index, sample in enumerate(array):
            if sample.ndim == 1:
                sample = sample[:, None]
            elif sample.shape[0] <= 8:
                sample = sample.T
            if sample.shape[-1] > 8:
                raise ValueError("Audio output channel layout is ambiguous or exceeds 8 channels.")
            pcm = (np.clip(sample, -1, 1) * 32767).round().astype("<i2")
            path = args.output_dir / f"audio-{index + 1:04d}.wav"

            def write_wave(temporary: Path) -> None:
                with wave.open(str(temporary), "wb") as stream:
                    stream.setnchannels(pcm.shape[1])
                    stream.setsampwidth(2)
                    stream.setframerate(rate)
                    stream.writeframes(pcm.tobytes())

            atomic_write(path, write_wave, args.overwrite)
            outputs.append({**file_identity(path), "kind": "audio", "sample_rate": rate,
                            "channels": pcm.shape[1], "samples": pcm.shape[0], "encoding": "PCM16"})
    if not outputs:
        raise ValueError("The pipeline returned no supported media. Expected images, frames, or audios/audio; use the workflow backend for other output types.")
    return outputs


def retire_previous_report(directory: Path) -> Path | None:
    """Keep prior provenance, but never leave it claiming a partially replaced run."""
    previous = directory / "generation.json"
    if not previous.exists():
        return None
    descriptor, name = tempfile.mkstemp(prefix="generation.previous-", suffix=".json", dir=directory)
    os.close(descriptor)
    archive = Path(name)
    try:
        # The archive name was exclusively reserved by this process. Replace only
        # that empty reservation; no unrelated existing report can be clobbered.
        os.replace(previous, archive)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive


def safety_metadata(result: Any, pipeline: Any, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "checker_present": getattr(pipeline, "safety_checker", None) is not None,
        "pipeline_report": {},
    }
    missing = object()
    for key in ("nsfw_content_detected", "nsfw_detected", "has_nsfw_concept", "applied_safety_concept"):
        value = result.get(key, missing) if isinstance(result, dict) else getattr(result, key, missing)
        if value is missing:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if key == "nsfw_content_detected" and isinstance(value, (list, tuple)):
            image_count = sum(item["kind"] == "image" for item in outputs)
            if image_count and len(value) != image_count:
                raise ValueError("The pipeline's nsfw_content_detected flags do not match its image batch.")
        # Keep actual values, including null, instead of inferring that an absent
        # checker or absent result means the generated content was checked.
        json.dumps(value, allow_nan=False)
        metadata["pipeline_report"][key] = value
    return metadata


def publish_generation(result: Any, pipeline: Any, args: argparse.Namespace, image_module: Any,
                       np: Any, report: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    previous = retire_previous_report(args.output_dir) if args.overwrite else None
    outputs = save_outputs(result, pipeline, args, image_module, np)
    completed = {**report, "outputs": outputs, "safety": safety_metadata(result, pipeline, outputs)}
    if previous is not None:
        completed["previous_report"] = str(previous)
    report_path = args.output_dir / "generation.json"
    atomic_write(report_path,
                 lambda path: path.write_text(json.dumps(completed, indent=2, ensure_ascii=False,
                                                        allow_nan=False) + "\n"),
                 args.overwrite)
    return report_path, outputs


def main(argv: list[str] | None = None) -> int:
    try:
        args = resolve_arguments(build_parser().parse_args(argv))
        if args.print_config:
            print(json.dumps(configuration(args), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.output_dir.exists() and not args.overwrite and any(args.output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output_dir}; choose a fresh directory or pass --overwrite.")
        os.environ.setdefault("HF_XET_CACHE", str(ROOT / "build/reference/huggingface-xet"))
        import torch
        import diffusers
        import numpy as np
        from PIL import Image
        device = select_device(torch, args.device)
        dtype = getattr(torch, args.dtype)
        hardware = accelerator_preflight(torch, device, dtype)
        inputs, input_files = prepare_inputs(args.inputs, Image)
        pipeline, model = load_pipeline(args, diffusers, dtype)
        pipeline = prepare_execution(pipeline, device, args.offload)
        validate_call_arguments(pipeline.__call__, inputs, seed=args.seed)
        if args.seed is not None:
            inputs["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.inference_mode():
            result = pipeline(**inputs)
        validate_execution_device(pipeline, device)
        verify_identity(model["identity"])
        versions = {}
        for package in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "sentencepiece", "numpy", "Pillow"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report = {"schema_version": 1, "request": configuration(args), "model": model,
                  "input_files": input_files,
                  "runtime": {"packages": versions, "hardware": hardware,
                              "execution_device": device, "dtype": str(dtype), "offload": args.offload},
                  "validation_scope": "Successful execution of these exact files and inputs; not universal family compatibility, visual quality, or model-license clearance."}
        report_path, outputs = publish_generation(result, pipeline, args, Image, np, report)
        print(json.dumps({"report": str(report_path), "outputs": [item["path"] for item in outputs]}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
        raise SystemExit(f"Generic Diffusers generation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
