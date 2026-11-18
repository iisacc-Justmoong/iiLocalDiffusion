"""Architecture-aware checkpoints through the installed SDK's native engine.

The persistent Python worker owns output publication; its loaded native library
owns one reusable inference context with explicit companion weight bindings.
"""

from _file_provider import provider
from dataclasses import asdict
from contextlib import ExitStack
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from generation_output import protect_model_inputs, resolve_output_paths, write_png
from generate import write_json_atomically
from generation_seed import resolve_seed
from hires_options import hires_enabled
from inference_session import cached_pipeline, is_preparing, record_execution, verify_pipeline_sources
from lora import resolve_lora_selection
from weight_files import file_sha256, verify_weight_file
from backend_registry import plan_native, family
from resource_paths import OUTPUT_DIRECTORY
from native_upscaler import NativeUpscaler, Options as UpscaleOptions, _RequestV3 as UpscaleRequestV3

DEFAULT_OUTPUT = OUTPUT_DIRECTORY / "native/image.png"


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


class RequestHires(C.Structure):
    _fields_ = [("size", C.c_size_t), ("image", RequestV3), ("upscaler", UpscaleRequestV3),
                ("denoise_strength", C.c_float)]


Progress = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p)
Preview = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                     C.c_void_p, C.c_size_t, C.c_void_p)


class NativePreviewWriter:
    """Publish bounded projections of actual engine latents across both passes."""
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().absolute()
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise ValueError("The preview directory must not be redirected.")
        if self.directory.exists() and (not self.directory.is_dir() or any(provider.iterdir(self.directory))):
            raise ValueError("The preview directory must be empty.")
        provider.mkdir(self.directory, parents=True, exist_ok=True)
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


def loaded_library_path(library):
    """Return the Mach-O that actually owns the SDK entry point, not the dlopen argument."""
    class DlInfo(C.Structure):
        _fields_ = [("filename", C.c_char_p), ("base", C.c_void_p),
                    ("symbol", C.c_char_p), ("address", C.c_void_p)]

    try:
        entry = library.iild_native_available_v1
    except AttributeError:
        # NativeEngine gives older or incomplete libraries a more useful error.
        return None
    dladdr = C.CDLL(None).dladdr
    dladdr.argtypes = (C.c_void_p, C.POINTER(DlInfo))
    dladdr.restype = C.c_int
    info = DlInfo()
    if not dladdr(C.cast(entry, C.c_void_p), C.byref(info)) or not info.filename:
        raise RuntimeError("Cannot determine which native iiLocalDiffusion library was loaded.")
    return Path(os.fsdecode(info.filename))


def load_library(path):
    """Load the configured SDK and reject macOS loader redirection to another copy."""
    path = Path(path).resolve()
    with ExitStack() as directories:
        if os.name == "nt":
            seen = set()
            for directory in (path.parent, *map(Path, os.environ.get("PATH", "").split(os.pathsep))):
                key = os.path.normcase(str(directory.resolve()))
                if key not in seen and directory.is_dir():
                    seen.add(key)
                    directories.enter_context(os.add_dll_directory(str(directory.resolve())))
        library = C.CDLL(str(path))
        if sys.platform == "darwin":
            actual = loaded_library_path(library)
            if actual is not None:
                try:
                    matches = os.path.samefile(path, actual)
                except OSError:
                    matches = path == actual.resolve()
                if not matches:
                    raise RuntimeError(
                        f"The macOS loader selected a different iiLocalDiffusion library: {actual}. "
                        f"The worker requested: {path}. Restart the worker without a conflicting "
                        "DYLD_LIBRARY_PATH or DYLD_FRAMEWORK_PATH, or select the intended SDK library.")
        return library


class NativeEngine:
    def __init__(self):
        self.library = load_library(library_path())
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
    def image(self, args, seed, *, prepare=False, hires_upscaler=None, hires_options=None,
              hires_denoise_strength=0.35):
        use_hires = hires_enabled(getattr(args, "hires_fix", None))
        if not use_hires and not args.native_v2:
            raise ValueError("Disabling Hires Fix requires component-aware native generation.")
        if hires_upscaler is not None and not use_hires:
            raise ValueError("Hires upscaler requires Hires Fix to be enabled.")
        if hires_upscaler is None and hires_options is not None:
            raise ValueError("Hires options require an upscaler.")
        if hires_upscaler is not None:
            if not isinstance(hires_upscaler, NativeUpscaler):
                raise TypeError("Hires requires a native upscaler provider.")
            if prepare or not args.native_v2:
                raise ValueError("Hires provider requires component-aware generation, not preparation.")
            if not math.isfinite(hires_denoise_strength) or not 0 < hires_denoise_strength <= 1:
                raise ValueError("Hires denoise strength must be in (0, 1].")
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
                                 args.embedded_guidance, use_hires, args.device == "cpu",
                                 {"auto": 0, "epsilon": 1, "v_prediction": 2}[args.prediction_type])
            if getattr(args, "native_v3", False) or hires_upscaler is not None:
                try:
                    function = (self.library.iild_native_generate_hires_v1 if hires_upscaler is not None
                                else self.library.iild_native_generate_v3)
                except AttributeError as error:
                    raise RuntimeError("These generation controls require an updated native SDK. Reinstall iiLocalDiffusion.") from error
                function.argtypes = [C.POINTER(RequestHires if hires_upscaler is not None else RequestV3),
                                     Progress, Preview, C.c_void_p]
                function.restype = C.c_void_p
                explicit_sampling = getattr(args, "native_v3", False)
                values = args.native_sigmas if explicit_sampling else []
                sigmas = (C.c_float * len(values))(*values)
                controlled = RequestV3(C.sizeof(RequestV3), extended,
                    {"auto": 0, "euler": 1, "heun": 2}[args.native_sampler] if explicit_sampling else 0,
                    args.native_flow_shift if explicit_sampling else math.inf, sigmas, len(sigmas))
                if hires_upscaler is not None:
                    chosen = hires_upscaler._bridge_request(hires_options if hires_options is not None else UpscaleOptions())
                    hires = RequestHires(C.sizeof(RequestHires), controlled, chosen, hires_denoise_strength)
                    handle = function(C.byref(hires), callback, preview_callback, None)
                else:
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
    from downloaded_model import _safetensors, _gguf, model_container_format
    from weight_files import IndexedWeightFile, LocalWeightFile, model_content_sha256, file_signature
    from model_sources import local_model_path
    logical = Path(os.path.abspath(os.path.expanduser(os.fspath(source))))
    path = local_model_path(logical)
    container = model_container_format(logical)
    if container == "safetensors":
        _safetensors(path)
    elif container == "gguf":
        _gguf(path)
    elif container == 'safetensors-index':
        from image_package import inspect_sharded_index, sharded_index_paths
        initial = file_signature(path)
        paths = [Path(item) for item in sharded_index_paths(path)]
        signatures = [file_signature(item) for item in paths]
        report = inspect_sharded_index(path)
        if file_signature(path) != initial or [file_signature(item) for item in paths] != signatures:
            raise RuntimeError(f'Local {argument} shard index changed during inspection.')
        reported = report['shards']
        resolved_index = logical.resolve(strict=True)
        if (Path(report['path']) != logical or Path(report['resolved_path']) != resolved_index or len(reported) != len(paths)
                or any(logical.parent / item['relative_path'] != shard
                       or item['resolved_path'] != before[0] or item['size_bytes'] != before[3]
                       for item, shard, before in zip(reported, paths, signatures))):
            raise RuntimeError('The Safetensors shard inventory changed during inspection.')
        shards = []
        for shard, before in zip(paths, signatures):
            resolved_shard = shard.resolve(strict=True)
            digest = model_content_sha256(resolved_shard)
            if file_signature(shard) != before:
                raise RuntimeError(f'Local {argument} shard changed during inspection: {shard}')
            shards.append(LocalWeightFile(str(shard), str(resolved_shard), digest, resolved_shard.stat().st_size, before))
        digest = model_content_sha256(resolved_index)
        if file_signature(path) != initial or [file_signature(item) for item in paths] != signatures:
            raise RuntimeError(f'Local {argument} shard index changed during inspection.')
        return IndexedWeightFile(str(logical), str(resolved_index), digest,
                                 resolved_index.stat().st_size, initial, tuple(shards))
    else:
        raise ValueError(f"{argument} requires safetensors or GGUF weights.")
    return LocalWeightFile(str(logical), str(path), model_content_sha256(path),
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
        from model_sources import local_model_path
        if "vae" in components and local_model_path(components["vae"]) != local_model_path(args.vae):
            raise ValueError("Conflicting --vae and --components VAE paths.")
        components["vae"] = str(args.vae)
    args.native_plan = plan_native(inspection, components)
    if inspection.get("architecture") == "krea2" and args.native_plan["missing_components"]:
        from generation_defaults import resolve_krea2_components
        components = resolve_krea2_components(inspection, components, args.generation_resources)
        args.native_plan = plan_native(inspection, components)
    if inspection.get("architecture") == "anima" and args.native_plan["missing_components"]:
        from generation_defaults import resolve_anima_components
        components = resolve_anima_components(inspection, components, args.generation_resources)
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
                      or any(name in args._provided for name in ("guidance_scale", "embedded_guidance", "hires_fix",
                                                                  "hires_upscaler", "hires_denoising_strength"))
                      or args.device == "cpu")
    supported = {"config", "print_config", "model", "model_info", "base_model", "output", "output_dir", "work_dir",
                 "cache_dir", "preview_dir", "validate_only", "device", "prompt", "negative_prompt", "width", "height",
                 "steps", "seed", "num_images", "seed_stride", "default_modifiers", "generation_resources",
                 "lora", "lora_scale", "lora_weight_name", "progress", "png_compress_level", "png_optimize", "overwrite",
                 "local_files_only", "engine", "components", "vae", "guidance_scale", "embedded_guidance", "prediction_type",
                 "krea2_variant", "krea2_mu", "native_sampler", "sigmas", "hires_fix", "hires_upscaler", "hires_denoising_strength"}
    unsupported = set(args._provided) - supported
    if unsupported:
        raise ValueError("Native image inference does not support these overrides: " + ", ".join("--" + name.replace("_", "-") for name in sorted(unsupported)))
    args.hires_fix = hires_enabled(getattr(args, "hires_fix", None))
    if not args.hires_fix and (args.hires_upscaler is not None or args.hires_denoising_strength is not None):
        raise ValueError("Hires upscaler and strength require Hires Fix to be enabled.")
    if args.hires_upscaler is not None and args.hires_upscaler not in ("nearest", "bilinear", "bicubic", "lanczos"):
        raise ValueError("Unsupported native Hires upscaler.")
    if args.hires_denoising_strength is not None and (isinstance(args.hires_denoising_strength, bool)
            or not isinstance(args.hires_denoising_strength, (int, float))
            or not math.isfinite(args.hires_denoising_strength) or not 0 < args.hires_denoising_strength <= 1):
        raise ValueError("Native Hires strength must be finite and in (0,1].")
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
        if any(type(value) is not int or not 64 <= value <= 4096 or value % 8
               for value in output_size):
            raise ValueError('Native image dimensions must be multiples of 8 in [64,4096].')
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
        if not 64 <= value <= 4096 or value % 8:
            raise ValueError("Native image dimensions must be multiples of 8 in [64,4096].")
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
    args.model = model_file.execution_path
    args.model_selection = SimpleNamespace(single_file=model_file)
    args.native_component_files = {slot: native_weight(path, slot) for slot, path in args.components.items()}
    args.components = {slot: weight.execution_path for slot, weight in args.native_component_files.items()}
    args.vae_file = args.native_component_files.get("vae")
    args.native_loras = []
    selection = resolve_lora_selection(args)
    if selection:
        if not selection.local_file or not math.isfinite(selection.scale):
            raise ValueError("Native inference requires a finite scale and local LoRA file.")
        args.native_loras.append((selection.local_file.resolved_file, selection.scale))
    args.output_was_default = args.output is None
    args.output = args.output or DEFAULT_OUTPUT
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
        with provider.open(Path(trace), "a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        print("IILD_NATIVE_TELEMETRY " + json.dumps(record), flush=True)
    except OSError as failure:
        print(f"Native telemetry publication write failed: {failure}", file=sys.stderr, flush=True)


def run(_preset, args):
    model = args.model_selection.single_file
    sources = [model.execution_path, *[shard.path for shard in getattr(model, 'shards', ())],
               *args.components.values(), *[path for path, _ in args.native_loras]]
    if getattr(args, 'image_package', None):
        sources.append(args.image_package['source'])
    key = ("native", args.architecture, args.device, args.prediction_type, model.execution_path, tuple(sorted(args.components.items())),
           str(args.generation_resources), args.default_modifiers, tuple(args.native_loras))
    engine, _ = cached_pipeline(key, sources, NativeEngine)
    # Every queued batch has an explicit preload barrier, even without an idle
    # foreground warmup. A cached context reuses its already resident sources.
    # Fail closed with an older SDK instead of silently streaming from disk.
    _, preparation = engine.image(args, args.seed, prepare=True)
    if preparation.get("weight_storage") != "anonymous":
        raise RuntimeError("The native SDK did not preload model sources into anonymous memory. Update iiLocalDiffusion before generation.")
    if is_preparing():
        from checkpoint_conversion import verify_conversion_source
        verify_conversion_source(getattr(args, "model_conversion", None))
        from image_package import verify_source as verify_package_source
        verify_package_source(getattr(args, 'image_package', None))
        record_execution("native-" + args.device, "managed", args.model)
        return 0
    if args.preview_dir:
        args._native_preview = NativePreviewWriter(args.preview_dir)
    paths = resolve_output_paths(args)
    hires_provider = None
    hires_options = None
    if args.hires_fix and (args.hires_upscaler is not None or args.hires_denoising_strength is not None):
        kind = args.hires_upscaler or "lanczos"
        hires_provider = NativeUpscaler("lanczos3" if kind == "lanczos" else kind, library=engine.library._name)
        hires_options = UpscaleOptions(target_width=((args.width + 63) // 64) * 64,
                                      target_height=((args.height + 63) // 64) * 64)
    for index, path in enumerate(paths):
        seed = args.seed + index * args.seed_stride
        if hires_provider is None:
            image, performance = engine.image(args, seed)
        else:
            image, performance = engine.image(args, seed, hires_upscaler=hires_provider, hires_options=hires_options,
                hires_denoise_strength=args.hires_denoising_strength if args.hires_denoising_strength is not None else .35)
        performance["preparation"] = preparation
        publication_started = time.monotonic()
        publication_event(performance, "publication-start", publication_started)
        try:
            verify_weight_file(model, "model")
            from checkpoint_conversion import verify_conversion_source
            verify_conversion_source(getattr(args, "model_conversion", None))
            for slot, weight in args.native_component_files.items():
                verify_weight_file(weight, slot)
            verify_pipeline_sources()
            from image_package import verify_source as verify_package_source
            verify_package_source(getattr(args, 'image_package', None))
            protect_model_inputs(args, [path])
            provider.mkdir(path.parent, parents=True, exist_ok=True)
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
                              "hires": args.hires_fix},
                  "vae": {"source": "explicit" if args.vae_file else "checkpoint", "override": args.components.get("vae")}, "loras": args.native_loras,
                  "default_modifiers": args.default_modifiers, "performance": performance,
                  "output": {"path": str(path.resolve()), "sha256": file_sha256(path), "kind": "image",
                             "size": [args.width, args.height], "mode": "RGB",
                             "width": args.width, "height": args.height}, "batch": {"index": index, "count": len(paths)}}
        if getattr(args, "model_conversion", None):
            report["model_conversion"] = args.model_conversion
        if getattr(args, 'image_package', None):
            report['image_package'] = {key: args.image_package[key] for key in
                ('source', 'architecture', 'transformer', 'text_encoder', 'vae', 'prepared_transformer')}
            if args.image_package.get('selected_components'):
                report['image_package']['selected_components'] = dict(args.image_package['selected_components'])
        protect_model_inputs(args, [path])
        write_json_atomically(path.with_suffix(".json"), report, overwrite=args.overwrite)
        print(f"Image: {path}", flush=True)
    record_execution("native-" + args.device, "managed", args.model)
    return 0


def configuration_values(args):
    """Replay only native inputs; Diffusers' preset defaults are not applicable."""
    names = ("model", "model_info", "base_model", "output_dir", "work_dir", "cache_dir", "preview_dir",
             "device", "prompt", "negative_prompt", "width", "height", "steps", "seed", "num_images",
             "seed_stride", "default_modifiers", "generation_resources", "lora", "lora_scale", "lora_weight_name",
             "progress", "png_compress_level", "png_optimize", "overwrite", "local_files_only",
             "hires_fix", "hires_upscaler", "hires_denoising_strength")
    values = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if getattr(args, "original_model_path", None):
        values["model"] = args.original_model_path
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
