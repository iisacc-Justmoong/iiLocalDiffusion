"""Architecture-aware checkpoints through the installed SDK's native engine.

The persistent Python worker owns output publication; its loaded native library
owns one reusable inference context with explicit companion weight bindings.
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
from weight_files import file_sha256, verify_weight_file
from backend_registry import plan_native, family


class LoRA(C.Structure):
    _fields_ = [("path", C.c_char_p), ("strength", C.c_float)]


class Request(C.Structure):
    _fields_ = [("size", C.c_size_t), ("model", C.c_char_p), ("prompt", C.c_char_p),
                ("negative_prompt", C.c_char_p), ("resources", C.c_char_p),
                ("width", C.c_int32), ("height", C.c_int32), ("steps", C.c_int32),
                ("seed", C.c_int64), ("default_modifiers", C.c_int32), ("prepare_only", C.c_int32),
                ("timeout_milliseconds", C.c_int32),
                ("loras", C.POINTER(LoRA)), ("lora_count", C.c_size_t)]


class RequestV2(C.Structure):
    _fields_ = [("size", C.c_size_t), ("image", Request),
                *[(name, C.c_char_p) for name in ("clip_l", "clip_g", "t5xxl", "llm", "vae")],
                ("guidance_scale", C.c_float), ("distilled_guidance", C.c_float),
                ("hires", C.c_int32), ("cpu", C.c_int32), ("prediction", C.c_int32)]


class RequestV3(C.Structure):
    _fields_ = [("size", C.c_size_t), ("image", RequestV2), ("sampler", C.c_int32),
                ("flow_shift", C.c_float), ("sigmas", C.POINTER(C.c_float)), ("sigma_count", C.c_size_t)]


Progress = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p)
Preview = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                     C.c_void_p, C.c_size_t, C.c_void_p)


class NativePreviewWriter:
    """Publish bounded projections of actual engine latents across both passes."""
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().absolute()
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise ValueError("The preview directory must not be redirected.")
        if self.directory.exists() and (not self.directory.is_dir() or any(self.directory.iterdir())):
            raise ValueError("The preview directory must be empty.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    def __call__(self, step, total, width, height, pixels):
        if not (1 <= step <= total <= 10000 and 1 <= width <= 512 and 1 <= height <= 512
                and len(pixels) == width * height * 3):
            raise ValueError("Invalid native preview dimensions or progress.")
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise RuntimeError("The preview directory was redirected during generation.")
        from PIL import Image
        image = Image.frombytes("RGB", (width, height), pixels)
        name = f"step-{self.sequence + 1:06d}.png"
        write_png(image, self.directory / name, compress_level=1, optimize=False, overwrite=False)
        self.sequence += 1
        print("IILD_PREVIEW " + json.dumps({"schema": "iild-preview-v1", "step": step,
              "total_steps": total, "sequence": self.sequence, "image": name}), flush=True)


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
    if os.name == "nt":
        candidates.extend(path.with_name("libiiLocalDiffusion.dll") for path in list(candidates))
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
            "generate_with_preview": ([C.POINTER(Request), Progress, Preview, C.c_void_p], C.c_void_p),
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
            from inference_session import register_runtime_release
            register_runtime_release(str(self.library._name), self.release)
        except AttributeError as error:
            raise RuntimeError("The native SDK is older than its Python worker. Reinstall iiLocalDiffusion.") from error

    # The native runtime, not this lightweight ctypes adapter, owns residency.
    # Replacing a request/session adapter must not evict model source memory.
    def image(self, args, seed, *, prepare=False):
        last = [-1, 0.0]
        def progress(stage, step, total, _):
            now = time.monotonic()
            if args.progress and (stage != last[0] or stage == 3 or step == total or now - last[1] >= 0.2):
                phases = ("waiting", "loading", "encoding", "denoising", "decoding", "preparing-model", "computing")
                print("IILD_NATIVE_PROGRESS " + json.dumps({"schema": "iild-native-progress-v1",
                      "stage": phases[stage], "step": step, "total": total}), flush=True)
                last[:] = [stage, now]
            return 0
        callback = Progress(progress)
        preview_errors = []
        writer = getattr(args, "_native_preview", None) if not prepare else None
        def preview(sequence, step, total, width, height, pixels, size, _):
            try:
                if not pixels or not (1 <= width <= 512 and 1 <= height <= 512) or size != width * height * 3:
                    raise ValueError("Invalid native preview buffer.")
                writer(step, total, width, height, C.string_at(pixels, size))
                return 0
            except Exception as error:
                preview_errors.append(error)
                return 1
        preview_callback = Preview(preview) if writer else Preview()
        adapters = (LoRA * len(args.native_loras))(*[
            LoRA(str(path).encode(), scale) for path, scale in args.native_loras])
        request = Request(C.sizeof(Request), args.model.encode(), args.prompt.encode(),
                          args.negative_prompt.encode(), str(args.generation_resources).encode() if args.generation_resources else None,
                          args.width, args.height, args.steps, seed, args.default_modifiers, prepare, 0,
                          adapters, len(adapters))
        if args.native_v2:
            try:
                function = self.library.iild_native_generate_v2
            except AttributeError as error:
                raise RuntimeError("This model requires the component-aware native SDK (V2). Rebuild/reinstall iiLocalDiffusion.") from error
            function.argtypes = [C.POINTER(RequestV2), Progress, Preview, C.c_void_p]
            function.restype = C.c_void_p
            paths = [args.components.get(slot, "").encode() or None for slot in ("clip_l", "clip_g", "t5xxl", "llm", "vae")]
            extended = RequestV2(C.sizeof(RequestV2), request, *paths, args.guidance_scale,
                                 args.embedded_guidance, 0, args.device == "cpu",
                                 {"auto": 0, "epsilon": 1, "v_prediction": 2}[args.prediction_type])
            if getattr(args, "native_v3", False):
                try:
                    function = self.library.iild_native_generate_v3
                except AttributeError as error:
                    raise RuntimeError("These sampling controls require native SDK V3. Reinstall iiLocalDiffusion.") from error
                function.argtypes = [C.POINTER(RequestV3), Progress, Preview, C.c_void_p]
                function.restype = C.c_void_p
                sigmas = (C.c_float * len(args.native_sigmas))(*args.native_sigmas)
                controlled = RequestV3(C.sizeof(RequestV3), extended,
                    {"auto": 0, "euler": 1, "heun": 2}[args.native_sampler], args.native_flow_shift, sigmas, len(sigmas))
                handle = function(C.byref(controlled), callback, preview_callback, None)
            else:
                handle = function(C.byref(extended), callback, preview_callback, None)
        else:
            handle = self.generate_with_preview(C.byref(request), callback, preview_callback, None)
        if not handle:
            raise RuntimeError("Native inference could not allocate a result.")
        try:
            if preview_errors:
                raise preview_errors[0]
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


def native_weight(source, argument):
    from downloaded_model import _safetensors, _gguf
    from weight_files import LocalWeightFile, model_content_sha256, file_signature
    path = Path(source).expanduser().resolve(strict=True)
    if path.suffix.casefold() in (".safetensors", ".safetensor"):
        _safetensors(path)
    elif path.suffix.casefold() == ".gguf":
        _gguf(path)
    else:
        raise ValueError(f"{argument} requires safetensors or GGUF weights.")
    return LocalWeightFile(str(Path(source).expanduser().absolute()), str(path), model_content_sha256(path),
                           path.stat().st_size, file_signature(path))


def resolve_arguments(args, inspection):
    if args.base_model:
        from civitai_catalog import lookup_base_model
        record = lookup_base_model(args.base_model)
        if record["family"] != family(inspection.get("architecture")):
            raise ValueError("--base-model conflicts with the tensor architecture.")
        inspection = dict(inspection, base_model=record["name"])
    components = dict(args.components)
    if args.vae:
        if "vae" in components and Path(components["vae"]).expanduser().resolve() != Path(args.vae).expanduser().resolve():
            raise ValueError("Conflicting --vae and --components VAE paths.")
        components["vae"] = str(args.vae)
    args.native_plan = plan_native(inspection, components)
    if args.native_plan["missing_components"]:
        raise ValueError(f"{inspection['architecture']} checkpoint is missing components: "
                         + ", ".join(args.native_plan["missing_components"])
                         + ". Supply matching local weight files through --components.")
    args.components = args.native_plan["components"]
    args.architecture = inspection["architecture"]
    if args.prediction_type == "auto" and inspection.get("prediction_type") in ("epsilon", "v_prediction"):
        args.prediction_type = inspection["prediction_type"]
    if args.prediction_type not in ("auto", "epsilon", "v_prediction"):
        raise ValueError("Native prediction type must be auto, epsilon or v_prediction.")
    if args.prediction_type != "auto" and args.architecture not in ("sd1", "sd2", "sdxl"):
        raise ValueError("Flow backends require their architecture's automatic prediction contract.")
    args.native_v2 = (args.architecture != "anima" or bool(components) or args.engine == "native"
                      or any(name in args._provided for name in ("guidance_scale", "embedded_guidance")) or args.device == "cpu")
    supported = {"config", "print_config", "model", "model_info", "base_model", "output", "output_dir", "work_dir",
                 "cache_dir", "preview_dir", "validate_only", "device", "prompt", "negative_prompt", "width", "height",
                 "steps", "seed", "num_images", "seed_stride", "default_modifiers", "generation_resources",
                 "lora", "lora_scale", "lora_weight_name", "progress", "png_compress_level", "png_optimize", "overwrite",
                 "local_files_only", "engine", "components", "vae", "guidance_scale", "embedded_guidance", "prediction_type",
                 "krea2_variant", "krea2_mu", "native_sampler", "sigmas"}
    unsupported = set(args._provided) - supported
    if unsupported:
        raise ValueError("Native image inference does not support these overrides: " + ", ".join("--" + name.replace("_", "-") for name in sorted(unsupported)))
    if args.device not in ("auto", "cpu"):
        raise ValueError("Native inference uses --device auto or cpu.")
    if not args.local_files_only:
        raise ValueError("Native inference requires local model files.")
    defaults = dict(args.native_plan["defaults"])
    args.krea2 = None
    args.native_sigmas = []
    args.native_flow_shift = float('inf')
    args.native_v3 = args.architecture == 'krea2' or args.native_sampler != 'auto'
    if args.native_v3:
        args.native_v2 = True
    if args.architecture == 'krea2':
        from krea2_contract import resolve
        inputs = {key: value for key, value in (("width", args.width), ("height", args.height),
                  ("num_inference_steps", args.steps), ("sigmas", args.sigmas)) if value is not None}
        # The native ABI retains the requested RGB size, aligns its internal
        # canvas to 64 pixels, then center-crops without resampling. Resolve the
        # Krea schedule against that canvas, not the final QuickGenerate crop.
        output_size = [args.width if args.width is not None else defaults['size'],
                       args.height if args.height is not None else defaults['size']]
        if any(type(value) is not int or not 64 <= value <= 2048 or value % 8
               for value in output_size):
            raise ValueError('Native image dimensions must be multiples of 8 in [64,2048].')
        canvas_size = [((value + 63) // 64) * 64 for value in output_size]
        inputs.update(zip(('width', 'height'), canvas_size))
        if args.guidance_scale is not None:
            inputs['guidance_scale'] = args.guidance_scale - 1
        # A bare checkpoint has no Diffusers is_distilled configuration. Use the
        # quality sampling preset for desktop auto mode, without claiming that
        # this detects the checkpoint's training variant. Turbo remains explicit.
        requested_variant = args.krea2_variant
        control = SimpleNamespace(inputs=inputs, dtype='source',
            krea2_variant='raw' if requested_variant == 'auto' else requested_variant,
            krea2_mu=args.krea2_mu)
        args.krea2 = resolve(control, {'_class_name': 'Krea2Pipeline'})
        args.krea2['variant_source'] = 'native-quality-default' if requested_variant == 'auto' else 'explicit'
        args.krea2_variant = args.krea2['variant']
        # Native upstream does not expose every latent for finite-value checks.
        args.krea2['finite_latents_required'] = False
        args.krea2['guidance_convention'] = 'native CFG = Krea guidance + 1'
        args.krea2.update(output_size=output_size, canvas_size=canvas_size,
                          output_transform='center-crop-no-resampling')
        defaults.update(steps=inputs['num_inference_steps'], guidance_scale=inputs['guidance_scale'] + 1)
        args.native_plan['defaults'] = dict(defaults)
        args.native_plan['krea2_variant'] = args.krea2_variant
        args.native_flow_shift = args.krea2['resolved_mu']
        if args.native_sampler == 'auto':
            args.native_sampler = 'euler'
        # Pin the published Euler schedule rather than inherit an upstream
        # discrete scheduler whose endpoint/grid may differ across versions.
        steps = inputs['num_inference_steps']
        sigmas = args.sigmas if args.sigmas is not None else [(steps - i) / steps for i in range(steps)]
        factor = math.exp(args.native_flow_shift)
        args.native_sigmas = [factor * s / (1 + (factor - 1) * s) for s in sigmas] + [0.0]
    elif args.krea2_variant != 'auto' or args.krea2_mu is not None or args.sigmas is not None:
        raise ValueError('Krea 2 variant, mu and native custom sigma controls require Krea 2 weights.')
    args.width = args.width if args.width is not None else defaults["size"]
    args.height = args.height if args.height is not None else defaults["size"]
    args.steps = args.steps if args.steps is not None else defaults["steps"]
    for name in ("guidance_scale", "embedded_guidance"):
        value = getattr(args, name)
        if value is None:
            value = defaults[name]
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Native {name} must be finite and in [0,100].")
        setattr(args, name, value)
    for name in ("width", "height"):
        value = getattr(args, name)
        if not 64 <= value <= 2048 or value % 8:
            raise ValueError("Native image dimensions must be multiples of 8 in [64,2048].")
    if not 1 <= args.steps <= 1000 or not 1 <= args.num_images <= 1000:
        raise ValueError("Native image steps and image count must be in [1,1000].")
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError("PNG compression must be in [0,9].")
    args.seed = resolve_seed(args.seed)
    if not all(0 <= seed < 2**63 for seed in (args.seed, args.seed + (args.num_images - 1) * args.seed_stride)):
        raise ValueError("Native image seeds must be in [0,2^63).")
    if "prompt" not in args._provided:
        args.prompt = "A landscape"
    if "negative_prompt" not in args._provided:
        args.negative_prompt = ""
    if not args.prompt.strip() or any("\0" in value or len(value.encode()) > 128000 for value in (args.prompt, args.negative_prompt)):
        raise ValueError("Native inference requires a nonempty prompt and bounded UTF-8 text without NUL.")
    model_file = native_weight(args.model, "--model")
    args.model = model_file.resolved_file
    args.model_selection = SimpleNamespace(single_file=model_file)
    args.native_component_files = {slot: native_weight(path, slot) for slot, path in args.components.items()}
    args.vae_file = args.native_component_files.get("vae")
    args.native_loras = []
    selection = resolve_lora_selection(args)
    if selection:
        if not selection.local_file or not math.isfinite(selection.scale):
            raise ValueError("Native inference requires a finite scale and local LoRA file.")
        args.native_loras.append((selection.local_file.resolved_file, selection.scale))
    args.output_was_default = args.output is None
    args.output = args.output or Path(__file__).resolve().parents[2] / "build/reference/native/image.png"
    args.engine = "native"
    return None, args


def publication_event(performance, event, started, error=""):
    """Continue the native trace through source validation and PNG publication."""
    trace = performance.get("telemetry_path")
    if not trace:
        return
    elapsed = (time.monotonic() - started) * 1000
    record = {"schema": "iild-native-telemetry-v1", "event": event,
              "trace_path": trace, "phase": "postprocess", "operation": "publish-image",
              "backend": "CPU", "backend_scope": "output-publication", "pid": os.getpid(),
              "phase_elapsed_ms": elapsed, "unix_ms": time.time() * 1000,
              "elapsed_ms": performance.get("telemetry_elapsed_ms", 0) + elapsed}
    if error:
        record["detail"] = str(error)[:1500]
    try:
        with Path(trace).open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        print("IILD_NATIVE_TELEMETRY " + json.dumps(record), flush=True)
    except OSError as failure:
        print(f"Native telemetry publication write failed: {failure}", file=sys.stderr, flush=True)


def run(_preset, args):
    model = args.model_selection.single_file
    sources = [model.resolved_file, *args.components.values(), *[path for path, _ in args.native_loras]]
    key = ("native", args.architecture, args.device, args.prediction_type, model.resolved_file, tuple(sorted(args.components.items())),
           str(args.generation_resources), args.default_modifiers, tuple(args.native_loras))
    engine, _ = cached_pipeline(key, sources, NativeEngine)
    # Every queued batch has an explicit preload barrier, even without an idle
    # foreground warmup. A cached context reuses its already resident sources.
    # Fail closed with an older SDK instead of silently streaming from disk.
    _, preparation = engine.image(args, args.seed, prepare=True)
    if preparation.get("weight_storage") != "anonymous":
        raise RuntimeError("The native SDK did not preload model sources into anonymous memory. Update iiLocalDiffusion before generation.")
    if is_preparing():
        record_execution("native-" + args.device, "managed", args.model)
        return 0
    if args.preview_dir:
        args._native_preview = NativePreviewWriter(args.preview_dir)
    paths = resolve_output_paths(args)
    for index, path in enumerate(paths):
        seed = args.seed + index * args.seed_stride
        image, performance = engine.image(args, seed)
        performance["preparation"] = preparation
        publication_started = time.monotonic()
        publication_event(performance, "publication-start", publication_started)
        try:
            verify_weight_file(model, "model")
            for slot, weight in args.native_component_files.items():
                verify_weight_file(weight, slot)
            verify_pipeline_sources()
            path.parent.mkdir(parents=True, exist_ok=True)
            write_png(image, path, compress_level=args.png_compress_level, optimize=args.png_optimize, overwrite=args.overwrite)
        except Exception as error:
            publication_event(performance, "publication-failed", publication_started, error)
            raise
        performance["publication_ms"] = (time.monotonic() - publication_started) * 1000
        publication_event(performance, "output-published", publication_started)
        report = {"backend": "native", "architecture": args.architecture, "model": asdict(model),
                  "backend_plan": args.native_plan, "components": {k: asdict(v) for k, v in args.native_component_files.items()},
                  "fixture": {"prompt": args.prompt, "negative_prompt": args.negative_prompt, "width": args.width,
                              "height": args.height, "steps": args.steps, "seed": seed,
                              "guidance_scale": args.guidance_scale, "embedded_guidance": args.embedded_guidance,
                              "prediction_type": args.prediction_type,
                              "sampling": {"sampler": args.native_sampler, "sigmas": args.native_sigmas,
                                           "flow_shift": args.native_flow_shift if math.isfinite(args.native_flow_shift) else None},
                              "krea2": args.krea2,
                              "hires": not args.native_v2},
                  "vae": {"source": "explicit" if args.vae_file else "checkpoint", "override": args.components.get("vae")}, "loras": args.native_loras,
                  "default_modifiers": args.default_modifiers, "performance": performance,
                  "output": {"path": str(path.resolve()), "sha256": file_sha256(path), "kind": "image",
                             "size": [args.width, args.height], "mode": "RGB",
                             "width": args.width, "height": args.height}, "batch": {"index": index, "count": len(paths)}}
        write_json_atomically(path.with_suffix(".json"), report, overwrite=args.overwrite)
        print(f"Image: {path}", flush=True)
    record_execution("native-" + args.device, "managed", args.model)
    return 0


def configuration_values(args):
    """Replay only native inputs; Diffusers' preset defaults are not applicable."""
    names = ("model", "model_info", "base_model", "output_dir", "work_dir", "cache_dir", "preview_dir",
             "device", "prompt", "negative_prompt", "width", "height", "steps", "seed", "num_images",
             "seed_stride", "default_modifiers", "generation_resources", "lora", "lora_scale", "lora_weight_name",
             "progress", "png_compress_level", "png_optimize", "overwrite", "local_files_only")
    values = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.native_v2:
        values.update(engine="native", components=args.components, guidance_scale=args.guidance_scale,
                      embedded_guidance=args.embedded_guidance, prediction_type=args.prediction_type)
    if args.native_v3:
        values.update(native_sampler=args.native_sampler)
    if args.krea2 is not None:
        values.update(krea2_variant=args.krea2_variant, krea2_mu=args.krea2_mu, sigmas=args.sigmas)
    if not args.output_was_default:
        values["output"] = args.output
    return json.loads(json.dumps(values, default=lambda value: str(value.expanduser().resolve())))
