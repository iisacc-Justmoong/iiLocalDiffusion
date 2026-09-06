"""Sequential Deforum 2D feedback using the existing Diffusers model runtime."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from types import SimpleNamespace
from typing import Any, Callable

from cpu_conditioning import encode_cpu_prompt
from deforum_options import frame_request
from deforum_video import AnimationOutput, SCHEMA, encode_video, output_targets, prepare_frame
from encoder_compatibility import clip_skip_compatibility
from generation_options import build_generators
from generation_output import write_png
from hardware import validate_execution_device
from hires import DenoisingAudit, PIPELINES, image_metadata, validate_stage_images, _validate_preserved_adapters
from text_embeddings import text_embedding_prompt_context, validate_text_embeddings
from weight_files import file_sha256


def image_call_arguments(pipeline: Any, preset: Any, request: Any,
                         call: dict[str, Any], image: Any) -> dict[str, Any]:
    call = dict(call)
    has_controlnet = request.controlnet_selection is not None
    if has_controlnet and preset.family != "flux1-schnell":
        call["control_image"] = call.pop("image")
    if not has_controlnet and preset.family in ("sd15", "sdxl-base"):
        call.pop("width", None)
        call.pop("height", None)
    if not (has_controlnet and preset.family == "flux1-schnell"):
        call["image"] = image
        call["strength"] = request.denoising_strength
    # SD 1.5 img2img does not expose guidance_rescale in pinned Diffusers.
    # Omission of its neutral default is safe; a requested value must not vanish.
    parameters = inspect.signature(pipeline.__call__).parameters
    if "guidance_rescale" not in parameters:
        if call.get("guidance_rescale", 0) != 0:
            raise ValueError("This image-to-image pipeline does not support --guidance-rescale.")
        call.pop("guidance_rescale", None)
    return call


def _prepare_frame_pipeline(pipeline: Any, preset: Any, request: Any, torch: Any,
                            device: str, dtype: Any, attention_slicing: bool, activation: Any,
                            image_to_image: bool, converted: bool, prepare_execution: Callable,
                            pipeline_classes: dict[str, Any] | None):
    move = request.cpu_text_encoding or (image_to_image and not converted)
    if move:
        pipeline.remove_all_hooks()
        pipeline.to("cpu")
    # Schedulers contain step indices/history; never carry sampling state to a
    # new frame, even when a fixed seed and the same pipeline are reused.
    scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
    if image_to_image and not converted:
        name = PIPELINES[(preset.family, request.controlnet_selection is not None)]
        if pipeline_classes is None:
            import diffusers
            selected = getattr(diffusers, name)
        else:
            selected = pipeline_classes[name]
        overrides = {"scheduler": scheduler, "dtype": dtype}
        if preset.family == "sdxl-base":
            overrides["add_watermarker"] = bool(request.watermark)
        pipeline = selected.from_pipe(pipeline, **overrides)
        _validate_preserved_adapters(pipeline, activation)
        validate_text_embeddings(pipeline, getattr(request, "text_embedding_activation", None), torch)
        converted = True
    else:
        pipeline.scheduler = scheduler
    conditioning = None
    if request.cpu_text_encoding:
        with text_embedding_prompt_context(pipeline, preset, request) as prompt_args:
            conditioning = encode_cpu_prompt(pipeline, preset, prompt_args, torch)
    if move:
        offload = request.offload
        if (offload == "auto" and request.cpu_text_encoding and device != "cpu"
                and preset.runtime.accelerator_execution == "resident"):
            offload = "model"
        pipeline, _ = prepare_execution(
            pipeline, preset, device, attention_slicing, offload=offload,
            vae_slicing=request.vae_slicing, vae_tiling=request.vae_tiling,
            attention_slice_size=request.attention_slice_size, device_index=request.device_index)
        validate_execution_device(pipeline, device)
        if device == "cuda" and pipeline._execution_device.index != request.device_index:
            raise RuntimeError("Deforum pipeline did not retain the requested GPU device index.")
    pipeline.set_progress_bar_config(disable=not request.progress)
    return pipeline, conditioning, converted


def render_deforum_frames(
    pipeline: Any, preset: Any, args: Any, torch: Any, device: str, dtype: Any,
    attention_slicing: bool, activation: Any, *, build_call: Callable,
    prepare_execution: Callable, environment: dict[str, Any], write_frame: Callable,
    pipeline_classes: dict[str, Any] | None = None, transform: Callable = prepare_frame,
) -> list[dict[str, Any]]:
    """Retain only the previous and reference images; emit each validated frame."""
    previous = environment.get("initial_image")
    reference = None
    converted = False
    frames = []
    for index in range(args.max_frames):
        request = frame_request(preset, args, index)
        source_metadata = None if previous is None else image_metadata(previous)
        initial = (None if previous is None else
                   transform(previous, reference, request, environment, warp=index > 0))
        initial_metadata = None if initial is None else image_metadata(initial)
        print(f"Deforum frame {index + 1}/{args.max_frames}; seed={request.seed}", flush=True)
        safety, conditioning = None, None
        generator_device = None
        compatibility = None
        if initial is not None and request.denoising_strength == 0:
            image = initial
            sampling = {"executed_steps": 0, "executed_timesteps": [], "finite_latents": None}
            method = "warp-only"
        else:
            pipeline, conditioning, converted = _prepare_frame_pipeline(
                pipeline, preset, request, torch, device, dtype, attention_slicing, activation,
                initial is not None, converted, prepare_execution, pipeline_classes)
            generator, _, generator_device = build_generators(torch, request, device)
            audit = DenoisingAudit(torch)
            clip_context = (clip_skip_compatibility(pipeline, preset, request.clip_skip)
                            if conditioning is None else nullcontext(False))
            with (torch.inference_mode(), clip_context,
                  text_embedding_prompt_context(pipeline, preset, request) as prompt_args):
                call = build_call(preset, prompt_args, generator, conditioning, device, dtype)
                if initial is not None:
                    call = image_call_arguments(pipeline, preset, request, call, initial)
                call["callback_on_step_end"] = audit
                if initial is not None and preset.family == "flux1-schnell" and request.controlnet_selection is not None:
                    from hires_flux_controlnet import refine_flux_controlnet
                    result, compatibility = refine_flux_controlnet(
                        pipeline, request, [initial], call, torch, generator, device, dtype)
                else:
                    result = pipeline(**call)
            sampling = audit.metadata()
            image, = validate_stage_images(result, 1, args.width, args.height)
            method = "text-to-image" if initial is None else "image-to-image"
            safety = {"checker_present": getattr(pipeline, "safety_checker", None) is not None,
                      "nsfw_content_detected": getattr(result, "nsfw_content_detected", None),
                      "watermarker_present": getattr(pipeline, "watermark", None) is not None}
        # The same validation applies to zero-strength frames and generated frames.
        image, = validate_stage_images(SimpleNamespace(images=[image]), 1, args.width, args.height)
        output = write_frame(index, image)
        frames.append({
            "index": index, "time_seconds": index / args.fps, "method": method,
            "seed": request.seed, "generator_device": generator_device,
            "prompt": request.prompt, "negative_prompt": request.negative_prompt,
            "prompt_2": request.prompt_2, "negative_prompt_2": request.negative_prompt_2,
            "guidance_scale": request.guidance_scale, "denoising_strength": request.denoising_strength,
            "camera": {name: getattr(request, name) for name in ("zoom", "angle", "translation_x", "translation_y")},
            "camera_applied": initial is not None and index > 0,
            "preprocessing_applied": initial is not None,
            "noise": request.noise_schedule, "contrast": request.contrast_schedule,
            "previous_frame": None if index == 0 else index - 1,
            "source": source_metadata, "init": initial_metadata,
            "output": {**image_metadata(image), **output}, "sampling": sampling,
            "pipeline_class": type(pipeline).__name__, "safety": safety,
            "cpu_conditioning": None if conditioning is None else conditioning.metadata,
            "compatibility": compatibility,
        })
        previous = image
        if reference is None:
            reference = image.copy()
    return frames


def run_deforum_animation(
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

        frames = render_deforum_frames(
            pipeline, preset, args, torch, device, dtype, attention_slicing, activation,
            build_call=build_call, prepare_execution=prepare_execution, environment=environment,
            write_frame=write_frame)
        video = encode_video(output.frames, output.video, args, environment)
        output.commit({**metadata, "schema": SCHEMA, "status": "complete", "animation_mode": "2D",
                       "frames": frames, "completed_frames": len(frames),
                       "initial_image": environment.get("initial_image_identity"),
                       "output": {**video, "path": str(video_path.absolute())},
                       "media_runtime": environment["versions"],
                       "reproducibility": {"seed_behavior": args.seed_behavior,
                                           "note": "Seeds do not guarantee identical pixels across hardware or runtime versions."}})
    print(f"Video: {video_path.absolute()}\nFrames: {frames_path.absolute()}\nMetadata: {report_path.absolute()}")
    return 0
