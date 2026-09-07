"""LTX temporal denoising, keyframe conditioning and verified video publication."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from animation_video import AnimationOutput, encode_video, preflight_animation
from hardware import accelerator_preflight, select_device
from video_options import configuration
from video_interpolator import interpolate_video, preflight_interpolator
from weight_files import file_sha256


def model_contract(directory):
    """Only built-in LTX components may be loaded; model packages cannot supply code."""
    try:
        index = json.loads((directory / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("A video model needs a valid Diffusers model_index.json.") from error
    expected = {"scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "transformer": ["diffusers", "LTXVideoTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLLTXVideo"]}
    if index.get("_class_name") not in ("LTXPipeline", "LTXConditionPipeline", "LTXImageToVideoPipeline"):
        raise ValueError("The video backend requires an LTX temporal diffusion model package.")
    for name, component in expected.items():
        if index.get(name) != component:
            raise ValueError(f"Unsupported video model component: {name}.")
    if index.get("tokenizer") not in (["transformers", "T5Tokenizer"], ["transformers", "T5TokenizerFast"],
                                      ["transformers", "PreTrainedTokenizerFast"], ["transformers", "TokenizersBackend"]):
        raise ValueError("The video backend requires a built-in T5-compatible tokenizer.")
    if any(not key.startswith("_") and key not in {*expected, "tokenizer"} for key in index):
        raise ValueError("The video model declares unsupported additional components.")
    return index


def load_tokenizer(directory, index):
    if (directory / "tokenizer/spiece.model").is_file():
        try:
            import google.protobuf  # noqa: F401; required by Transformers' SentencePiece conversion
        except ImportError as error:
            raise ImportError("LTX SentencePiece tokenizers require protobuf; install "
                              "reference/diffusers/requirements-video.txt in the generation environment.") from error
    import transformers
    tokenizer_class = getattr(transformers, index["tokenizer"][1])
    return tokenizer_class.from_pretrained(str(directory / "tokenizer"), local_files_only=True)


def load_pipeline(args, torch, dtype):
    directory = Path(args.model).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Video generation requires an existing local model directory.")
    from diffusers import LTXConditionPipeline
    index = model_contract(directory)
    # Tokenizer conversion must succeed before hashing/loading multi-GB weights.
    tokenizer = load_tokenizer(directory, index)
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and (path.suffix in (".safetensors", ".json", ".model", ".txt") or path.name == "README.md")
                   and ".cache" not in path.relative_to(directory).parts)
    identity, stamps = [], {}
    for path in files:
        stat = path.stat()
        if not stat.st_size:
            raise ValueError(f"Empty video model resource: {path}")
        stamps[path] = (stat.st_size, stat.st_mtime_ns)
        identity.append({"file": str(path.relative_to(directory)), "sha256": file_sha256(path),
                         "size_bytes": stat.st_size})
    if not any(record["file"].startswith("transformer/") and record["file"].endswith(".safetensors")
               for record in identity):
        raise ValueError("The video package does not contain safe transformer weights.")
    pipeline = LTXConditionPipeline.from_pretrained(str(directory), dtype=dtype,
                                                   local_files_only=True, use_safetensors=True,
                                                   low_cpu_mem_usage=True, tokenizer=tokenizer)
    verify_model_files(stamps)
    if (pipeline.vae_spatial_compression_ratio != 32 or pipeline.vae_temporal_compression_ratio != 8):
        raise ValueError("This video backend requires the LTX spatial-32/temporal-8 VAE contract.")
    metadata = {"source": args.model, "revision": args.revision, "directory": str(directory.resolve()),
                "declared_pipeline": index["_class_name"], "execution_pipeline": type(pipeline).__name__,
                "files": identity, "spatial_compression": 32, "temporal_compression": 8}
    return pipeline, metadata, stamps


def verify_model_files(stamps):
    for path, previous in stamps.items():
        stat = path.stat()
        if previous != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Video model resource changed during generation: {path}")


def load_keyframes(args, image_module):
    from PIL import ImageOps
    loaded, metadata = {}, {}
    for shot in args.shots:
        for condition in shot["conditions"]:
            path = condition["image"]
            if path in loaded:
                continue
            digest = file_sha256(Path(path))
            with image_module.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Keyframes must be static images.")
                image.load()
                original_size = list(image.size)
                prepared = ImageOps.exif_transpose(image).convert("RGB")
                prepared = ImageOps.fit(prepared, (args.width, args.height),
                                        method=image_module.Resampling.LANCZOS)
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Keyframe changed while being read: {path}")
            loaded[path] = prepared
            metadata[path] = {"path": path, "sha256": digest, "size_bytes": Path(path).stat().st_size,
                              "original_size": original_size, "size": [args.width, args.height],
                              "fit": "center-crop", "exif_orientation_applied": True}
    return loaded, metadata


def prepare_execution(pipeline, args, torch, device, dtype):
    embeddings = []
    if args.cpu_text_encoding:
        # Encode before accelerator offload hooks. Releasing T5 afterwards keeps
        # the memory budget available for temporal attention and VAE decoding.
        for shot in args.shots:
            with torch.inference_mode():
                values = pipeline.encode_prompt(prompt=shot["effective_prompt"],
                                                 negative_prompt=shot["negative_prompt"],
                                                 do_classifier_free_guidance=args.guidance_scale > 1,
                                                 max_sequence_length=args.max_sequence_length,
                                                 device="cpu")
            names = ("prompt_embeds", "prompt_attention_mask", "negative_prompt_embeds",
                     "negative_prompt_attention_mask")
            result = {}
            for name, value in zip(names, values):
                if value is not None:
                    if not torch.isfinite(value).all().item():
                        raise RuntimeError("Video text conditioning contains non-finite values.")
                    result[name] = value.detach().cpu()
            embeddings.append(result)
        pipeline.register_modules(text_encoder=None)
        gc.collect()
    if args.vae_tiling:
        pipeline.vae.enable_tiling()
    offload = ("none" if device == "cpu" else "model") if args.offload == "auto" else args.offload
    if offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload(device=device)
    else:
        pipeline.to(device)
    return embeddings, {"offload": offload, "cpu_text_encoding": args.cpu_text_encoding,
                        "vae_tiling": args.vae_tiling, "dtype": str(dtype)}


def validate_captions(pipeline, args):
    for shot in args.shots:
        captions = [shot["effective_prompt"]]
        if args.guidance_scale > 1:
            captions.append(shot["negative_prompt"])
        for caption in captions:
            tokens = pipeline.tokenizer(caption, truncation=False, add_special_tokens=True)["input_ids"]
            if len(tokens) > args.max_sequence_length:
                raise ValueError(f"Shot {shot['index'] + 1} needs {len(tokens)} text tokens including camera motion; "
                                 "increase --max-sequence-length or shorten the prompt. Silent truncation is disabled.")


class VideoDenoisingAudit:
    def __init__(self, torch, expected_device):
        self.torch = torch
        self.expected_device = expected_device
        self.steps = []

    def __call__(self, pipeline, step, timestep, values):
        latents = values["latents"]
        if not self.torch.isfinite(latents).all().item():
            raise RuntimeError(f"Video denoising produced non-finite latents at step {step}.")
        if latents.device.type != self.expected_device:
            raise RuntimeError(f"Video denoising ran on {latents.device.type}, expected {self.expected_device}.")
        self.steps.append({"step": step, "timestep": float(timestep), "shape": list(latents.shape),
                           "device": str(latents.device), "finite": True})
        return values


def render_shots(pipeline, args, torch, device, embeddings, keyframes, output, image_module):
    import numpy as np
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    records, frame_records, previous = [], [], None
    for shot in args.shots:
        started = time.monotonic()
        print(f"Video shot {shot['index'] + 1}/{len(args.shots)}: "
              f"{shot['ltx_frames']} LTX source frames for {shot['frames']} output frames", flush=True)
        conditions = [LTXVideoCondition(image=keyframes[condition["image"]], frame_index=condition["source_frame"],
                                        strength=condition["strength"]) for condition in shot["conditions"]]
        if shot["continue_previous"]:
            conditions.insert(0, LTXVideoCondition(image=previous, frame_index=0, strength=1.0))
        audit = VideoDenoisingAudit(torch, device)
        call = {"conditions": conditions or None, "height": args.height, "width": args.width,
                "num_frames": shot["sample_frames"], "frame_rate": shot["sampling_fps"], "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale, "max_sequence_length": args.max_sequence_length,
                "generator": torch.Generator(device="cpu").manual_seed(shot["seed"]),
                "decode_timestep": args.decode_timestep, "decode_noise_scale": args.decode_noise_scale,
                "image_cond_noise_scale": args.image_cond_noise_scale, "output_type": "np",
                "callback_on_step_end": audit, "callback_on_step_end_tensor_inputs": ["latents"]}
        if embeddings:
            # LTX's encode_prompt leaves caller-provided tensors where they
            # are. Move both embeddings and masks before the temporal loop.
            call.update({name: value.to(device=device) for name, value in embeddings[shot["index"]].items()})
        else:
            call.update(prompt=shot["effective_prompt"], negative_prompt=shot["negative_prompt"])
        with torch.inference_mode():
            frames = np.asarray(pipeline(**call).frames)
        expected = (1, shot["sample_frames"], args.height, args.width, 3)
        if frames.shape != expected or not np.issubdtype(frames.dtype, np.floating):
            raise RuntimeError(f"Unexpected decoded video shape/dtype: {frames.shape}, {frames.dtype}; expected {expected}.")
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise RuntimeError("Decoded video must contain finite RGB values in [0,1].")
        if len(audit.steps) != args.steps:
            raise RuntimeError("Video generation did not complete all requested temporal denoising steps.")
        folder = output.frames
        if args.interpolation_enabled:
            folder = folder / "ltx" / f"shot-{shot['index']:04d}"
            folder.mkdir(parents=True)
        for local_index, position in enumerate(shot["source_positions"]):
            index = shot["start_frame"] + position
            image = image_module.fromarray(np.rint(frames[0, local_index] * 255).astype(np.uint8))
            path = folder / f"frame-{local_index if args.interpolation_enabled else index:06d}.png"
            image.save(path, format="PNG")
            frame_records.append({"index": index, "shot": shot["index"], "source_index": local_index,
                                  "file": str(path.relative_to(output.frames)),
                                  "sha256": file_sha256(path)})
            previous = image
        records.append({**shot, "denoising": audit.steps, "elapsed_seconds": time.monotonic() - started,
                        "conditioning_source": "previous-shot-final-frame" if shot["continue_previous"] else "keyframes-or-text"})
        del frames
    return records, frame_records


def generate_video(args):
    environment = preflight_animation(args)
    preflight_interpolator(args, environment)
    keyframes, keyframe_metadata = load_keyframes(args, environment["Image"])
    import torch
    import diffusers
    device = select_device(torch, args.device)
    dtype_name = ("float32" if device == "cpu" else "bfloat16") if args.dtype == "auto" else args.dtype
    dtype = getattr(torch, dtype_name)
    hardware = accelerator_preflight(torch, device, dtype)
    print(f"Video runtime: {hardware['backend']} / {dtype_name}", flush=True)
    # The shared animation lock covers model loading too, preventing duplicate
    # jobs from consuming memory before one fails at publication.
    with AnimationOutput(args) as output:
        pipeline, model, stamps = load_pipeline(args, torch, dtype)
        validate_captions(pipeline, args)
        embeddings, execution = prepare_execution(pipeline, args, torch, device, dtype)
        shots, source_frames = render_shots(pipeline, args, torch, device, embeddings, keyframes,
                                            output, environment["Image"])
        verify_model_files(stamps)
        del pipeline, embeddings
        gc.collect()
        stages = [{"name": "LTX", "method": "joint-spatiotemporal-diffusion",
                   "output_frames": len(source_frames), "verified_finite_denoising": True}]
        interpolation = {"enabled": False, "reason": "fps-at-most-12"}
        frames = source_frames
        if args.interpolation_enabled:
            frames, interpolation = interpolate_video(args, shots, source_frames, output, environment)
            stages.append({"name": "Interpolator", **interpolation})
        verify_model_files(stamps)
        encoded = encode_video(output.frames, output.video, args, environment)
        report = {"schema": "iild-temporal-video-v1", "status": "complete", "model": model,
                  "method": ("joint-spatiotemporal-diffusion+motion-interpolation" if args.interpolation_enabled
                             else "joint-spatiotemporal-diffusion"), "camera_control": "text-conditioning",
                  "shot_composition": "cut", "hardware": hardware, "execution": execution,
                  "configuration": configuration(args), "shots": shots, "frames": frames,
                  "stages": stages, "interpolation": interpolation, "source_frames": source_frames,
                  "keyframes": list(keyframe_metadata.values()), "video": encoded,
                  "versions": {**environment["versions"], "torch": torch.__version__,
                               "diffusers": diffusers.__version__}}
        output.commit(report)
    return report
