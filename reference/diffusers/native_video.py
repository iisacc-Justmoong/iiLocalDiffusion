"""Native video checkpoints through the Qt-free C++ SDK and verified MP4 publication."""
from _file_provider import provider
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from animation_video import AnimationOutput, encode_video, preflight_animation
from native_image import library_path, load_library, Progress
from video_options import configuration
from video_runtime import load_keyframes, report_progress, verify_model_files
from video_interpolator import interpolate_video, preflight_interpolator
from weight_files import file_sha256


class Shot(C.Structure):
    _fields_ = [("prompt", C.c_char_p), ("negative_prompt", C.c_char_p), ("seed", C.c_int64),
                ("frames", C.c_int32), ("fps", C.c_int32), ("first_rgb", C.c_void_p), ("last_rgb", C.c_void_p),
                ("first_size", C.c_size_t), ("last_size", C.c_size_t), ("conditioning_strength", C.c_float),
                ("continue_previous", C.c_int32)]


class Request(C.Structure):
    _fields_ = [("size", C.c_size_t), *[(name, C.c_char_p) for name in
                ("model", "text_encoder", "asset_cache", "backend", "precision")],
                *[(name, C.c_int32) for name in ("width", "height", "steps", "threads")],
                ("guidance_scale", C.c_float), ("cpu", C.c_int32), ("vae_tiling", C.c_int32),
                ("cpu_text_encoding", C.c_int32), ("shots", C.POINTER(Shot)), ("shot_count", C.c_size_t)]


class Components(C.Structure):
    _fields_ = [("size", C.c_size_t), ("video_vae", C.c_char_p), ("high_noise_model", C.c_char_p),
                ("vision_encoder", C.c_char_p), ("moe_boundary", C.c_float),
                ("high_noise_steps", C.c_int32), ("high_noise_guidance", C.c_float)]


Frame = C.CFUNCTYPE(C.c_int, C.c_size_t, C.c_int32, C.c_int32, C.c_int32,
                   C.c_void_p, C.c_size_t, C.c_void_p)


def native_shot_plan(shot, output_fps):
    """Native VAE padding stays inside inference; publish exactly the shot timeline."""
    for condition in shot["conditions"]:
        if condition["frame"] not in (0, shot["frames"] - 1):
            raise ValueError("Native video supports first/last keyframes; interior image conditions require a compatible Diffusers route.")
    strengths = {condition["strength"] for condition in shot["conditions"]}
    if len(strengths) > 1 or (shot["continue_previous"] and strengths and strengths != {1.0}):
        raise ValueError("Native first/last keyframes must use the same conditioning strength.")
    samples = shot["sample_frames"]
    count = shot["ltx_frames"]
    indices = [round(i * (samples - 1) / (count - 1)) for i in range(count)]
    return {"samples": samples, "fps": max(1, min(120, round(output_fps * (samples - 1) / (shot["frames"] - 1)))),
            "indices": indices, "strength": next(iter(strengths), 1.0)}


def model_metadata(path, indexed_weight=None):
    stat = path.stat()
    from downloaded_model import model_container_format
    if model_container_format(path) == 'safetensors-index':
        from image_package import inspect_sharded_index
        from native_image import native_weight
        from weight_files import verify_weight_file
        weight = indexed_weight or native_weight(path, 'native video model')
        report = inspect_sharded_index(path)
        data = path.read_bytes()
        verify_weight_file(weight, 'native video model')
        return {'source': str(path), 'size_bytes': stat.st_size + sum(shard.size_bytes for shard in weight.shards),
                'mtime_ns': stat.st_mtime_ns, 'header_sha256': hashlib.sha256(data).hexdigest(),
                'model_version': report['metadata'].get('model_version', ''),
                'execution_pipeline': 'iiLocalDiffusion.NativeVideo',
                'shards': [{'path': shard.path, 'size_bytes': shard.size_bytes} for shard in weight.shards]}
    with provider.open(path, "rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("Incomplete native video checkpoint header.")
        size, = struct.unpack("<Q", prefix)
        # GGUF models are inspected by the native engine without a safetensors header.
        if prefix[:4] == b"GGUF":
            header = b""
            metadata = {}
        else:
            if not 2 <= size <= min(64 * 1024**2, stat.st_size - 8):
                raise ValueError("Invalid or oversized video checkpoint header.")
            header = file.read(size)
            metadata = json.loads(header).get("__metadata__", {})
    return {"source": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "header_sha256": hashlib.sha256(prefix + header).hexdigest(),
            "model_version": metadata.get("model_version", ""), "execution_pipeline": "iiLocalDiffusion.NativeVideo"}


class NativeVideoEngine:
    def __init__(self):
        self.library = load_library(library_path())
        try:
            self.generate = self.library.iild_native_video_generate_v1
            self.generate.argtypes, self.generate.restype = [C.POINTER(Request), Progress, Frame, C.c_void_p], C.c_void_p
            self.generate_low_vram = getattr(self.library, "iild_native_video_generate_low_vram_v1", None)
            if self.generate_low_vram is not None:
                self.generate_low_vram.argtypes, self.generate_low_vram.restype = self.generate.argtypes, C.c_void_p
            self.generate_with_components = getattr(self.library, "iild_native_video_generate_with_components_v1", None)
            self.generate_low_vram_with_components = getattr(self.library, "iild_native_video_generate_low_vram_with_components_v1", None)
            for function in (self.generate_with_components, self.generate_low_vram_with_components):
                if function is not None:
                    function.argtypes = [C.POINTER(Request), C.POINTER(Components), Progress, Frame, C.c_void_p]
                    function.restype = C.c_void_p
            self.metadata = self.library.iild_native_video_metadata_v1
            self.metadata.argtypes, self.metadata.restype = [C.c_void_p], C.c_char_p
            self.free = self.library.iild_native_video_free_v1
            self.free.argtypes, self.free.restype = [C.c_void_p], None
        except AttributeError as error:
            raise RuntimeError("This video checkpoint needs the current native iiLocalDiffusion SDK. Rebuild the SDK runtime.") from error

    def render(self, args, output, image_module, keyframes):
        low_vram = args.offload == "sequential"
        fields = ("video_vae", "high_noise_model", "vision_encoder", "moe_boundary", "high_noise_steps", "high_noise_guidance")
        has_components = any(getattr(args, name, None) is not None for name in fields)
        if has_components:
            name = "generate_low_vram_with_components" if low_vram else "generate_with_components"
            generate = getattr(self, name, None)
            if generate is None:
                raise RuntimeError("Native video components require an updated iiLocalDiffusion runtime; rebuild and reinstall the SDK.")
        else:
            generate = getattr(self, "generate_low_vram", None) if low_vram else self.generate
        if generate is None:
            raise RuntimeError("Low VRAM requires an updated iiLocalDiffusion runtime; rebuild and reinstall the SDK.")
        plans = [native_shot_plan(shot, args.fps) for shot in args.shots]
        buffers, values = [], []
        for shot, plan in zip(args.shots, plans):
            endpoints = {condition["frame"]: keyframes[condition["image"]].tobytes() for condition in shot["conditions"]}
            def image(frame):
                data = endpoints.get(frame, b"")
                if not data:
                    return None, 0
                buffer = C.create_string_buffer(data)
                buffers.append(buffer)
                return C.cast(buffer, C.c_void_p), len(data)
            first, first_size = image(0)
            last, last_size = image(shot["frames"] - 1)
            values.append(Shot(shot["effective_prompt"].encode(), shot["negative_prompt"].encode(), shot["seed"],
                               plan["samples"], plan["fps"], first, last, first_size, last_size,
                               plan["strength"], shot["continue_previous"]))
        requests = (Shot * len(values))(*values)
        records, errors, delivered = [], [], [set() for _ in values]
        folders = []
        for shot in args.shots:
            folder = output.frames / "ltx" / f"shot-{shot['index']:04d}" if args.interpolation_enabled else output.frames
            provider.mkdir(folder, parents=True, exist_ok=True)
            folders.append(folder)
        last_progress = [-1, 0.0]
        def progress(stage, step, total, _):
            phases = ("waiting", "loading", "encoding", "denoising", "decoding", "preparing-model", "computing")
            now = time.monotonic()
            if (stage != last_progress[0] or stage == 3 or (total > 0 and step == total)
                    or now - last_progress[1] >= 0.2):
                report_progress(phases[stage] if 0 <= stage < len(phases) else "computing", step, total)
                # Existing desktop workers already understand native preparation
                # and completed CPU batches, including their unknown totals.
                if stage in (5, 6) and os.environ.get("IILD_WORKER_PROGRESS") == "1":
                    print("IILD_NATIVE_PROGRESS " + json.dumps({"schema": "iild-native-progress-v1",
                        "stage": phases[stage], "step": step, "total": total}), flush=True)
                last_progress[:] = [stage, now]
            return bool(errors)
        def frame(shot_index, index, width, height, rgb, size, _):
            try:
                if (shot_index >= len(args.shots) or width != args.width or height != args.height
                        or not rgb or size != width * height * 3):
                    raise RuntimeError("Native video returned an invalid RGB frame buffer.")
                plan, shot = plans[shot_index], args.shots[shot_index]
                if index not in plan["indices"]:
                    return 0
                local = plan["indices"].index(index)
                if local in delivered[shot_index]:
                    raise RuntimeError("Native video returned duplicate frames.")
                absolute = shot["start_frame"] + shot["source_positions"][local]
                path = folders[shot_index] / f"frame-{local if args.interpolation_enabled else absolute:06d}.png"
                image = image_module.frombytes("RGB", (width, height), C.string_at(rgb, size))
                provider.save_image(image, path, format="PNG")
                records.append({"index": absolute, "shot": shot_index, "source_index": local, "native_sample_index": index,
                                "file": str(path.relative_to(output.frames)), "sha256": file_sha256(path)})
                delivered[shot_index].add(local)
                return 0
            except Exception as error:
                errors.append(error)
                return 1
        callback, observer = Progress(progress), Frame(frame)
        request = Request(C.sizeof(Request), args.model.encode(),
                          str(args.text_encoder).encode() if args.text_encoder else None,
                          str(args.cache_dir / "native-video-assets").encode(),
                          ("metal" if args.device == "mps" else args.device).encode(), args.dtype.encode(),
                          args.width, args.height, args.steps, 0, args.guidance_scale,
                          args.device == "cpu", args.vae_tiling, args.cpu_text_encoding, requests, len(values))
        if has_components:
            def component_path(name):
                value = getattr(args, name, None)
                return str(value).encode() if value is not None else None
            def optional_number(name):
                value = getattr(args, name, None)
                return value if value is not None else -1
            components = Components(C.sizeof(Components), *(component_path(name) for name in fields[:3]),
                                    optional_number("moe_boundary"), optional_number("high_noise_steps"),
                                    optional_number("high_noise_guidance"))
            handle = generate(C.byref(request), C.byref(components), callback, observer, None)
        else:
            handle = generate(C.byref(request), callback, observer, None)
        if not handle:
            raise RuntimeError("Native video did not return a result.")
        try:
            result = json.loads(self.metadata(handle))
        finally:
            self.free(handle)
        if errors:
            raise errors[0]
        if result.get("error") or result.get("cancelled"):
            raise RuntimeError(result.get("error") or "Native video generation was cancelled.")
        if result.get("frame_counts") != [plan["samples"] for plan in plans] or any(
                len(frames) != shot["ltx_frames"] for frames, shot in zip(delivered, args.shots)):
            raise RuntimeError("Native video did not complete all requested shots and frames.")
        result["shot_sampling"] = [{"native_frames": plan["samples"], "native_fps": plan["fps"],
                                    "selected_sample_indices": plan["indices"]} for plan in plans]
        return records, result


def generate_video(args):
    report_progress("loading")
    from video_checkpoint_conversion import verify as verify_conversions
    verify_conversions(args)
    verify_directory = getattr(args, '_native_source_verifier', lambda: None)
    verify_directory()
    # Reject unsupported conditioning before any multi-GB asset is materialized.
    for shot in args.shots:
        native_shot_plan(shot, args.fps)
    environment = preflight_animation(args)
    preflight_interpolator(args, environment)
    keyframes, keyframe_metadata = load_keyframes(args, environment["Image"])
    path = Path(args.model)
    from downloaded_model import model_container_format
    from native_image import native_weight
    from weight_files import verify_weight_file
    indexed = native_weight(path, 'native video model') if model_container_format(path) == 'safetensors-index' else None
    encoder_indexed = (native_weight(args.text_encoder, 'native video text encoder') if args.text_encoder
                       and model_container_format(args.text_encoder) == 'safetensors-index' else None)
    model = model_metadata(path, indexed)
    if getattr(args, '_native_directory_source', None):
        model['directory_source'] = args._native_directory_source
    stat = path.stat()
    stamps = {path: (stat.st_size, stat.st_mtime_ns)}
    components = [(name, getattr(args, name, None)) for name in ("text_encoder", "video_vae", "high_noise_model", "vision_encoder")]
    component_indexes = []
    for name, component in components:
        if component:
            stat = component.stat()
            stamps[component] = (stat.st_size, stat.st_mtime_ns)
            if name != "text_encoder" and model_container_format(component) == 'safetensors-index':
                component_indexes.append((name, native_weight(component, f'native video {name}')))
    started = time.monotonic()
    with AnimationOutput(args) as output:
        source_frames, execution = NativeVideoEngine().render(args, output, environment["Image"], keyframes)
        verify_conversions(args)
        verify_directory()
        verify_model_files(stamps)
        if indexed:
            verify_weight_file(indexed, 'native video model')
        if encoder_indexed:
            verify_weight_file(encoder_indexed, 'native video text encoder')
        for name, weight in component_indexes:
            verify_weight_file(weight, f'native video {name}')
        shots = [{**shot, "conditioning_source": "previous-shot-final-frame" if shot["continue_previous"] else "keyframes-or-text"}
                 for shot in args.shots]
        interpolation = {"enabled": False, "reason": "fps-at-most-12"}
        frames = source_frames
        if args.interpolation_enabled:
            report_progress("interpolating")
            frames, interpolation = interpolate_video(args, shots, source_frames, output, environment)
        report_progress("encoding-video")
        encoded = encode_video(output.frames, output.video, args, environment)
        verify_conversions(args)
        verify_directory()
        verify_model_files(stamps)
        if indexed:
            verify_weight_file(indexed, 'native video model')
        if encoder_indexed:
            verify_weight_file(encoder_indexed, 'native video text encoder')
        for name, weight in component_indexes:
            verify_weight_file(weight, f'native video {name}')
        report = {"schema": "iild-temporal-video-v1", "status": "complete", "model": model,
                  "method": "native-joint-spatiotemporal-diffusion" + ("+motion-interpolation" if args.interpolation_enabled else ""),
                  "camera_control": "text-conditioning", "shot_composition": "cut", "execution": execution,
                  "configuration": getattr(args, '_native_original_configuration', None) or configuration(args),
                  "shots": shots, "frames": frames, "source_frames": source_frames,
                  "stages": [{"name": "Native video", "output_frames": len(source_frames), "verified_rgb_frames": True}],
                  "interpolation": interpolation, "keyframes": list(keyframe_metadata.values()), "video": encoded,
                  "elapsed_seconds": time.monotonic() - started, "versions": environment["versions"]}
        conversions = getattr(args, '_native_conversions', {})
        if conversions:
            report['checkpoint_conversions'] = {name: {key: value[key] for key in ('original', 'original_shards', 'output') if key in value}
                                                for name, value in conversions.items()}
        output.commit(report)
    report_progress("complete", args.max_frames, args.max_frames)
    return report
