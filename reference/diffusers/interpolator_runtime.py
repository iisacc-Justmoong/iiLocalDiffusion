"""Generate each frame from interpolated text embeddings and initial noise."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

from animation_video import AnimationOutput, SCHEMAS, encode_video, output_targets
from cpu_conditioning import CpuConditioning, encode_cpu_prompt
from generation_options import build_generators
from generation_output import write_png
from hires import DenoisingAudit, image_metadata, validate_stage_images, _validate_preserved_adapters
from interpolator_options import endpoint_request
from text_embeddings import text_embedding_prompt_context
from weight_files import file_sha256

SCHEMA = SCHEMAS["Interpolator"]


def _fraction(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Interpolation fraction must be finite and in [0,1].")


def _pair(a: Any, b: Any, torch: Any) -> None:
    if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point():
        raise ValueError("Interpolation endpoints must have matching floating-point tensor shapes.")
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise ValueError("Interpolation endpoints must be finite.")


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    # A canonical float32 byte representation also supports bfloat16 tensors.
    data = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "hash_format": "float32-le",
            "sha256": hashlib.sha256(data.tobytes()).hexdigest()}


def interpolate_conditioning(a: CpuConditioning, b: CpuConditioning, fraction: float,
                            torch: Any) -> CpuConditioning:
    _fraction(fraction)
    if a.tensors.keys() != b.tensors.keys() or a.tensors.get("prompt_embeds") is None:
        raise ValueError("Interpolation endpoints must expose the same prompt conditioning tensors.")
    result = {}
    for name, left in a.tensors.items():
        right = b.tensors[name]
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"Interpolation endpoint {name} is missing on one side.")
            result[name] = None
            continue
        _pair(left, right, torch)
        left, right = left.detach().cpu().float(), right.detach().cpu().float()
        # Explicit endpoint clones prevent pipeline in-place operations from
        # changing later frames and preserve the exact endpoint values.
        result[name] = (left.clone() if fraction == 0 else right.clone() if fraction == 1
                        else torch.lerp(left, right, fraction))
        if not torch.isfinite(result[name]).all().item():
            raise ValueError(f"Interpolated {name} is non-finite.")
    return CpuConditioning(result, {"enabled": True, "execution_device": "cpu", "fraction": fraction,
                                    "method": "linear", "shapes": {
                                        key: None if value is None else list(value.shape)
                                        for key, value in result.items()}})


def interpolate_noise(a: Any, b: Any, fraction: float, torch: Any) -> Any:
    _fraction(fraction)
    _pair(a, b, torch)
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    if fraction == 0 or torch.equal(a, b):
        return a.clone()
    if fraction == 1:
        return b.clone()
    # Independent unit Gaussian endpoints retain unit expected variance.
    # Equal noise (including equivalent signed seeds) must stay unchanged.
    result = math.sqrt(1 - fraction) * a + math.sqrt(fraction) * b
    if not torch.isfinite(result).all().item():
        raise ValueError("Interpolated noise is non-finite.")
    return result


def prepare_endpoints(pipeline: Any, preset: Any, args: Any, torch: Any,
                      start: CpuConditioning) -> tuple[CpuConditioning, CpuConditioning]:
    request = endpoint_request(args)
    with text_embedding_prompt_context(pipeline, preset, request) as prompt_args:
        end = encode_cpu_prompt(pipeline, preset, prompt_args, torch)
    # Validate the entire pair before accelerator hooks and the first frame.
    interpolate_conditioning(start, end, 0.5, torch)
    return start, end


def prepare_noise(pipeline: Any, preset: Any, args: Any, torch: Any, device: str,
                  dtype: Any) -> tuple[Any, Any, dict[str, Any]]:
    height, width = args.height // pipeline.vae_scale_factor, args.width // pipeline.vae_scale_factor
    flux = preset.family == "flux1-schnell"
    channels = pipeline.transformer.config.in_channels // 4 if flux else pipeline.unet.config.in_channels
    if flux and (height % 2 or width % 2):
        raise ValueError("FLUX interpolation requires even latent dimensions for 2x2 packing.")
    shape = (1, channels, height, width)
    endpoints = []
    for request in (args, endpoint_request(args)):
        generator, _, generator_device = build_generators(torch, request, device)
        latent = torch.randn(shape, generator=generator, device=generator_device, dtype=dtype)
        if flux:
            latent = pipeline._pack_latents(latent, 1, channels, height, width)
        # These are unscaled input latents. The pipeline owns scheduler sigma
        # scaling; doing it here too would corrupt the denoising trajectory.
        endpoints.append(latent.detach().cpu())
    _pair(*endpoints, torch)
    return *endpoints, {"seeds": [args.seed, args.end_seed], "generator_device": generator_device,
                        "method": "sqrt-variance-preserving", "equal_noise": torch.equal(*endpoints),
                        "start": tensor_metadata(endpoints[0]), "end": tensor_metadata(endpoints[1])}


def render_interpolator_frames(
    pipeline: Any, preset: Any, args: Any, torch: Any, device: str, dtype: Any,
    *, build_call: Callable, write_frame: Callable, activation: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start, end = args.interpolator_conditioning
    a, b, noise = prepare_noise(pipeline, preset, args, torch, device, dtype)
    info = {"prompt_method": "linear", "noise": noise,
            "conditioning": {"start": start.metadata, "end": end.metadata},
            "sampler_noise": "reset start-seed generator independently for every frame"}
    target_device = f"cuda:{args.device_index}" if device == "cuda" else device
    frames = []
    pipeline.set_progress_bar_config(disable=not args.progress)
    for index in range(args.max_frames):
        fraction = index / (args.max_frames - 1)
        print(f"Interpolator frame {index + 1}/{args.max_frames}; fraction={fraction:.6f}", flush=True)
        pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
        _validate_preserved_adapters(pipeline, activation)
        # Textual-inversion weights were verified before endpoint encoding.
        # The frame loop consumes cached tensors and must not read encoder
        # weights after sequential offload has replaced them with meta tensors.
        conditioning = interpolate_conditioning(start, end, fraction, torch)
        generator, _, generator_device = build_generators(torch, args, device)
        audit = DenoisingAudit(torch)
        with torch.inference_mode():
            call = build_call(preset, args, generator, conditioning, target_device, dtype)
            call["latents"] = interpolate_noise(a, b, fraction, torch).to(device=target_device, dtype=dtype)
            for name in (*conditioning.tensors, "latents"):
                if call.get(name) is not None and not torch.isfinite(call[name]).all().item():
                    raise ValueError(f"Interpolator {name} is non-finite after execution dtype conversion.")
            inputs = {name: None if call[name] is None else tensor_metadata(call[name])
                      for name in conditioning.tensors}
            latents = tensor_metadata(call["latents"])
            call["callback_on_step_end"] = audit
            result = pipeline(**call)
        sampling = audit.metadata()
        image, = validate_stage_images(result, 1, args.width, args.height)
        saved = write_frame(index, image)
        frames.append({"index": index, "time_seconds": index / args.fps, "fraction": fraction,
                       "method": "interpolated-text-to-image", "prompt_weights": [1 - fraction, fraction],
                       "sampler_seed": args.seed, "generator_device": generator_device,
                       "conditioning": inputs, "latents": latents, "sampling": sampling,
                       "pipeline_class": type(pipeline).__name__,
                       "output": {**image_metadata(image), **saved},
                       "safety": {"checker_present": getattr(pipeline, "safety_checker", None) is not None,
                                  "nsfw_content_detected": getattr(result, "nsfw_content_detected", None),
                                  "watermarker_present": getattr(pipeline, "watermark", None) is not None}})
    return frames, info


def run_interpolator_animation(
    pipeline: Any, preset: Any, args: Any, torch: Any, device: str, dtype: Any,
    attention_slicing: bool, activation: Any, *, build_call: Callable,
    prepare_execution: Callable, environment: dict[str, Any], metadata: dict[str, Any],
) -> int:
    video_path, report_path, frames_path = output_targets(args.output)
    with AnimationOutput(args) as output:
        def write_frame(index, image):
            filename = f"frame-{index:06d}.png"
            path = output.frames / filename
            write_png(image, path, compress_level=args.png_compress_level,
                      optimize=args.png_optimize, overwrite=False)
            return {"path": str((frames_path / filename).absolute()), "sha256": file_sha256(path)}

        frames, interpolation = render_interpolator_frames(
            pipeline, preset, args, torch, device, dtype, build_call=build_call,
            write_frame=write_frame, activation=activation)
        video = encode_video(output.frames, output.video, args, environment)
        output.commit({**metadata, "schema": SCHEMA, "status": "complete", "animation_mode": "Interpolator",
                       "frames": frames, "completed_frames": len(frames), "interpolation": interpolation,
                       "output": {**video, "path": str(video_path.absolute())},
                       "media_runtime": environment["versions"],
                       "reproducibility": {"note": "Seeds do not guarantee identical pixels across hardware or runtime versions."}})
    print(f"Video: {video_path.absolute()}\nFrames: {frames_path.absolute()}\nMetadata: {report_path.absolute()}")
    return 0
