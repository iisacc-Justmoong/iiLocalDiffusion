"""Complete Anima checkpoints through the installed SDK's native engine.

The persistent Python worker owns output publication; its loaded native library
owns one reusable inference context. No server, model conversion or VAE override.
"""
from dataclasses import asdict
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from generation_output import resolve_output_paths, write_png
from generate import write_json_atomically
from generation_seed import resolve_seed
from inference_session import cached_pipeline, is_preparing, record_execution, verify_pipeline_sources
from lora import resolve_lora_selection
from weight_files import file_sha256, resolve_weight_file, verify_weight_file


class LoRA(C.Structure):
    _fields_ = [("path", C.c_char_p), ("strength", C.c_float)]


class Request(C.Structure):
    _fields_ = [("size", C.c_size_t), ("model", C.c_char_p), ("prompt", C.c_char_p),
                ("negative_prompt", C.c_char_p), ("resources", C.c_char_p),
                ("width", C.c_int32), ("height", C.c_int32), ("steps", C.c_int32),
                ("seed", C.c_int64), ("default_modifiers", C.c_int32), ("prepare_only", C.c_int32),
                ("timeout_milliseconds", C.c_int32),
                ("loras", C.POINTER(LoRA)), ("lora_count", C.c_size_t)]


Progress = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p)


def library_path():
    explicit = os.environ.get("IILD_NATIVE_LIBRARY")
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    here = Path(__file__).resolve()
    name = "iiLocalDiffusion.dll" if os.name == "nt" else "libiiLocalDiffusion" + (".dylib" if sys.platform == "darwin" else ".so")
    # Installed reference is <prefix>/share/iiLocalDiffusion/reference/diffusers.
    # A source checkout uses its existing build/, never another user's SDK.
    prefix = here.parents[4] if here.parents[2].name == "iiLocalDiffusion" else None
    candidates = ([prefix / ("bin" if os.name == "nt" else "lib") / name] if prefix else [])
    candidates.append(here.parents[2] / "build" / name)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("The native iiLocalDiffusion library is missing. Reinstall the SDK with native inference enabled.")


class NativeEngine:
    def __init__(self):
        self.library = C.CDLL(str(library_path()))
        declarations = {
            "available": ([], C.c_int),
            "generate": ([C.POINTER(Request), Progress, C.c_void_p], C.c_void_p),
            "metadata": ([C.c_void_p], C.c_char_p),
            "rgb": ([C.c_void_p, C.POINTER(C.c_size_t)], C.c_void_p),
            "free": ([C.c_void_p], None), "release": ([], None),
        }
        try:
            for name, (arguments, result) in declarations.items():
                function = getattr(self.library, "iild_native_" + name + "_v1")
                function.argtypes, function.restype = arguments, result
                setattr(self, name, function)
            if not self.available():
                raise RuntimeError("This SDK was built without native image inference.")
        except AttributeError as error:
            raise RuntimeError("The native SDK is older than its Python worker. Reinstall iiLocalDiffusion.") from error

    def __del__(self):
        if hasattr(self, "release"):
            self.release()

    def image(self, args, seed, *, prepare=False):
        last = [-1, 0.0]
        def progress(stage, step, total, _):
            now = time.monotonic()
            if args.progress and (stage != last[0] or stage == 3 or step == total or now - last[1] >= 0.2):
                phases = ("waiting", "loading", "encoding", "denoising", "decoding", "preparing-model")
                print("IILD_NATIVE_PROGRESS " + json.dumps({"schema": "iild-native-progress-v1",
                      "stage": phases[stage], "step": step, "total": total}), flush=True)
                last[:] = [stage, now]
            return 0
        callback = Progress(progress)
        adapters = (LoRA * len(args.native_loras))(*[
            LoRA(str(path).encode(), scale) for path, scale in args.native_loras])
        request = Request(C.sizeof(Request), args.model.encode(), args.prompt.encode(),
                          args.negative_prompt.encode(), str(args.generation_resources).encode() if args.generation_resources else None,
                          args.width, args.height, args.steps, seed, args.default_modifiers, prepare, 0,
                          adapters, len(adapters))
        handle = self.generate(C.byref(request), callback, None)
        if not handle:
            raise RuntimeError("Native inference could not allocate a result.")
        try:
            metadata = json.loads(self.metadata(handle))
            if metadata["error"] or metadata["cancelled"]:
                raise RuntimeError(metadata["error"] or "Native image generation was cancelled.")
            if prepare:
                return None, metadata
            size = C.c_size_t()
            pixels = self.rgb(handle, C.byref(size))
            if (metadata["width"], metadata["height"]) != (args.width, args.height) or not pixels or size.value != args.width * args.height * 3:
                raise RuntimeError("Native inference returned invalid RGB dimensions or byte count.")
            from PIL import Image
            return Image.frombytes("RGB", (args.width, args.height), C.string_at(pixels, size.value)), metadata
        finally:
            self.free(handle)


def resolve_arguments(args, inspection):
    if inspection["role"] != "checkpoint" or inspection["architecture"] != "anima":
        raise ValueError("Native standalone routing requires an identified Anima checkpoint.")
    if inspection["format"] != "safetensors":
        raise ValueError("Native standalone Anima requires a complete safetensors checkpoint.")
    if inspection["missing_components"]:
        raise ValueError("Anima checkpoint is missing components: " + ", ".join(inspection["missing_components"])
                         + ". Select a complete Anima checkpoint with its text encoder and VAE.")
    supported = {"config", "print_config", "model", "model_info", "base_model", "output", "output_dir", "work_dir",
                 "cache_dir", "preview_dir", "validate_only", "device", "prompt", "negative_prompt", "width", "height",
                 "steps", "seed", "num_images", "seed_stride", "default_modifiers", "generation_resources",
                 "lora", "lora_scale", "lora_weight_name", "progress", "png_compress_level", "png_optimize", "overwrite",
                 "local_files_only"}
    unsupported = set(args._provided) - supported
    if unsupported:
        raise ValueError("Native Anima does not support these overrides: " + ", ".join("--" + name.replace("_", "-") for name in sorted(unsupported)))
    if args.base_model and args.base_model != "Anima":
        raise ValueError("--base-model conflicts with the Anima tensor architecture.")
    if args.device != "auto":
        raise ValueError("Native Anima uses --device auto with SDK-managed GPU/CPU placement.")
    if not args.local_files_only:
        raise ValueError("Native Anima requires local model files.")
    args.width = args.width if args.width is not None else 1024
    args.height = args.height if args.height is not None else 1024
    args.steps = args.steps if args.steps is not None else 20
    for name in ("width", "height"):
        value = getattr(args, name)
        if not 64 <= value <= 2048 or value % 8:
            raise ValueError("Native Anima dimensions must be multiples of 8 in [64,2048].")
    if not 1 <= args.steps <= 1000 or not 1 <= args.num_images <= 1000:
        raise ValueError("Native Anima steps and image count must be in [1,1000].")
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError("PNG compression must be in [0,9].")
    args.seed = resolve_seed(args.seed)
    if not all(0 <= seed < 2**63 for seed in (args.seed, args.seed + (args.num_images - 1) * args.seed_stride)):
        raise ValueError("Native Anima seeds must be in [0,2^63).")
    if "prompt" not in args._provided:
        args.prompt = "A landscape"
    if "negative_prompt" not in args._provided:
        args.negative_prompt = ""
    if not args.prompt.strip() or any("\0" in value or len(value.encode()) > 128000 for value in (args.prompt, args.negative_prompt)):
        raise ValueError("Native Anima requires a nonempty prompt and bounded UTF-8 text without NUL.")
    model_file = resolve_weight_file(args.model, "--model")
    args.model = model_file.resolved_file
    args.model_selection = SimpleNamespace(single_file=model_file)
    args.vae_file = None
    args.native_loras = []
    selection = resolve_lora_selection(args)
    if selection:
        if not selection.local_file or not math.isfinite(selection.scale):
            raise ValueError("Native Anima requires a finite scale and local LoRA file.")
        args.native_loras.append((selection.local_file.resolved_file, selection.scale))
    args.output_was_default = args.output is None
    args.output = args.output or Path(__file__).resolve().parents[2] / "build/reference/anima/image.png"
    args.engine = "native"
    return None, args


def run(_preset, args):
    model = args.model_selection.single_file
    sources = [model.resolved_file, *[path for path, _ in args.native_loras]]
    key = ("native-anima", model.resolved_file, str(args.generation_resources), args.default_modifiers, tuple(args.native_loras))
    engine, _ = cached_pipeline(key, sources, NativeEngine)
    if is_preparing():
        engine.image(args, args.seed, prepare=True)
        record_execution("native-auto", "managed", args.model)
        return 0
    if args.preview_dir:
        print("Native Anima reports generation stages; per-step latent previews are unavailable.", flush=True)
    paths = resolve_output_paths(args)
    for index, path in enumerate(paths):
        seed = args.seed + index * args.seed_stride
        image, performance = engine.image(args, seed)
        verify_weight_file(model, "model")
        verify_pipeline_sources()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(image, path, compress_level=args.png_compress_level, optimize=args.png_optimize, overwrite=args.overwrite)
        report = {"backend": "native", "architecture": "anima", "model": asdict(model),
                  "fixture": {"prompt": args.prompt, "negative_prompt": args.negative_prompt, "width": args.width,
                              "height": args.height, "steps": args.steps, "seed": seed},
                  "vae": {"source": "checkpoint", "override": None}, "loras": args.native_loras,
                  "default_modifiers": args.default_modifiers, "performance": performance,
                  "output": {"path": str(path.resolve()), "sha256": file_sha256(path),
                             "width": args.width, "height": args.height}, "batch": {"index": index, "count": len(paths)}}
        write_json_atomically(path.with_suffix(".json"), report, overwrite=args.overwrite)
        print(f"Image: {path}", flush=True)
    record_execution("native-auto", "managed", args.model)
    return 0


def configuration_values(args):
    """Replay only native inputs; Diffusers' preset defaults are not applicable."""
    names = ("model", "model_info", "base_model", "output_dir", "work_dir", "cache_dir", "preview_dir",
             "device", "prompt", "negative_prompt", "width", "height", "steps", "seed", "num_images",
             "seed_stride", "default_modifiers", "generation_resources", "lora", "lora_scale", "lora_weight_name",
             "progress", "png_compress_level", "png_optimize", "overwrite", "local_files_only")
    values = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if not args.output_was_default:
        values["output"] = args.output
    return json.loads(json.dumps(values, default=lambda value: str(value.expanduser().resolve())))
