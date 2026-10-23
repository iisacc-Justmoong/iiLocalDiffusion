"""LTX 2 packages through the Qt-free C++ SDK; Python publishes verified MP4s."""
from _file_provider import provider
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from animation_video import AnimationOutput, encode_video, preflight_animation
from native_image import library_path, Progress
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


Frame = C.CFUNCTYPE(C.c_int, C.c_size_t, C.c_int32, C.c_int32, C.c_int32,
                   C.c_void_p, C.c_size_t, C.c_void_p)


def native_shot_plan(shot, output_fps):
    """Native VAE padding stays inside inference; publish exactly the shot timeline."""
    for condition in shot["conditions"]:
        if condition["frame"] not in (0, shot["frames"] - 1):
            raise ValueError("Native LTX video supports first/last keyframes; interior image conditions require the Diffusers route.")
    strengths = {condition["strength"] for condition in shot["conditions"]}
    if len(strengths) > 1 or (shot["continue_previous"] and strengths and strengths != {1.0}):
        raise ValueError("Native first/last keyframes must use the same conditioning strength.")
    samples = shot["sample_frames"]
    count = shot["ltx_frames"]
    indices = [round(i * (samples - 1) / (count - 1)) for i in range(count)]
    return {"samples": samples, "fps": max(1, min(120, round(output_fps * (samples - 1) / (shot["frames"] - 1)))),
            "indices": indices, "strength": next(iter(strengths), 1.0)}


def model_metadata(path):
    stat = path.stat()
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
        self.library = C.CDLL(str(library_path()))
        try:
            self.generate = self.library.iild_native_video_generate_v1
            self.generate.argtypes, self.generate.restype = [C.POINTER(Request), Progress, Frame, C.c_void_p], C.c_void_p
            self.metadata = self.library.iild_native_video_metadata_v1
            self.metadata.argtypes, self.metadata.restype = [C.c_void_p], C.c_char_p
            self.free = self.library.iild_native_video_free_v1
            self.free.argtypes, self.free.restype = [C.c_void_p], None
        except AttributeError as error:
            raise RuntimeError("This video checkpoint needs the current native iiLocalDiffusion SDK. Rebuild the SDK runtime.") from error

    def render(self, args, output, image_module, keyframes):
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
        handle = self.generate(C.byref(request), callback, observer, None)
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
    # Reject unsupported conditioning before any multi-GB asset is materialized.
    for shot in args.shots:
        native_shot_plan(shot, args.fps)
    environment = preflight_animation(args)
    preflight_interpolator(args, environment)
    keyframes, keyframe_metadata = load_keyframes(args, environment["Image"])
    path = Path(args.model)
    model = model_metadata(path)
    stat = path.stat()
    stamps = {path: (stat.st_size, stat.st_mtime_ns)}
    if args.text_encoder:
        stat = args.text_encoder.stat()
        stamps[args.text_encoder] = (stat.st_size, stat.st_mtime_ns)
    started = time.monotonic()
    with AnimationOutput(args) as output:
        source_frames, execution = NativeVideoEngine().render(args, output, environment["Image"], keyframes)
        verify_model_files(stamps)
        shots = [{**shot, "conditioning_source": "previous-shot-final-frame" if shot["continue_previous"] else "keyframes-or-text"}
                 for shot in args.shots]
        interpolation = {"enabled": False, "reason": "fps-at-most-12"}
        frames = source_frames
        if args.interpolation_enabled:
            report_progress("interpolating")
            frames, interpolation = interpolate_video(args, shots, source_frames, output, environment)
        report_progress("encoding-video")
        encoded = encode_video(output.frames, output.video, args, environment)
        verify_model_files(stamps)
        report = {"schema": "iild-temporal-video-v1", "status": "complete", "model": model,
                  "method": "native-joint-spatiotemporal-diffusion" + ("+motion-interpolation" if args.interpolation_enabled else ""),
                  "camera_control": "text-conditioning", "shot_composition": "cut", "execution": execution,
                  "configuration": configuration(args), "shots": shots, "frames": frames, "source_frames": source_frames,
                  "stages": [{"name": "LTX 2 native", "output_frames": len(source_frames), "verified_rgb_frames": True}],
                  "interpolation": interpolation, "keyframes": list(keyframe_metadata.values()), "video": encoded,
                  "elapsed_seconds": time.monotonic() - started, "versions": environment["versions"]}
        output.commit(report)
    report_progress("complete", args.max_frames, args.max_frames)
    return report
