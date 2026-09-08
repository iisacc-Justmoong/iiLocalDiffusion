"""Remote model evaluation with local validation, publication and LTX interpolation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from model_sources import resolve_model_input
from generation_seed import resolve_seed
from remote_inference import RemoteInference
from weight_files import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FIELDS = {"model", "model_api", "model_cloud", "model_provider", "model_token_env", "model_family", "model_timeout"}
IMAGE_FIELDS = SOURCE_FIELDS | {"prompt", "negative_prompt", "width", "height", "steps", "guidance_scale",
                               "seed", "num_images", "seed_stride", "output", "overwrite",
                               "png_compress_level", "png_optimize"}


def resolve_image_options(args):
    from video_options import _integer, _number, _text
    args.seed = resolve_seed(args.seed)
    args.model_input = resolve_model_input(args)
    if args.model_input.kind == "local":
        raise ValueError("Remote image execution requires --model-api or --model-cloud.")
    if args.animation_mode != "none":
        raise ValueError("Deforum and prompt Interpolator require --model-path: remote image APIs do not expose their latent controls.")
    unsupported = set(args._provided) - IMAGE_FIELDS - {"config", "print_config"}
    if unsupported:
        raise ValueError("Unsupported remote image options: " + ", ".join(sorted(unsupported)))
    if args.model_input.family not in (None, "image"):
        raise ValueError("Remote image generation requires an image model.")
    for name, default in (("width", 512), ("height", 512), ("steps", 30), ("guidance_scale", 7.5)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    for name in ("width", "height"):
        _integer(getattr(args, name), name, 1, 4096)
    _integer(args.steps, "steps", 1, 1000)
    _integer(args.num_images, "num_images", 1, 64)
    _integer(args.seed_stride, "seed_stride", 0, 2**63 - 1)
    _integer(args.seed, "seed", 0, 2**63 - 1)
    _integer(args.seed + (args.num_images - 1) * args.seed_stride, "last image seed", 0, 2**63 - 1)
    _integer(args.png_compress_level, "png_compress_level", 0, 9)
    _number(args.guidance_scale, "guidance_scale", 0, 100)
    _text(args.prompt, "prompt")
    _text(args.negative_prompt, "negative_prompt", empty=True)
    args.output_was_default = args.output is None
    args.output = (args.output or ROOT / "build/reference/remote-image.png").expanduser().absolute()
    if args.output.suffix.lower() != ".png":
        raise ValueError("Remote image output must use the .png extension.")
    if args.config and args.config.resolve() in (args.output.resolve(), args.output.with_suffix(".json").resolve()):
        raise ValueError("Image output must not replace the request configuration.")
    return args


def image_configuration(args):
    return {name: str(value) if isinstance(value, Path) else value
            for name in sorted(IMAGE_FIELDS) for value in (getattr(args, name),)
            if value is not None}


def generate_image(args):
    from generation_output import resolve_output_paths, publish_file
    paths = resolve_output_paths(args)
    if args.config and any(args.config.resolve() in (path.resolve(), path.with_suffix(".json").resolve())
                           for path in paths):
        raise ValueError("Image batch output must not replace the request configuration.")
    remote = RemoteInference(args.model_input)
    reports = []
    # No output is published until every image in this request has been decoded.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_name("." + args.output.name + ".remote.lock")
    with lock.open("x"):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix=".iild-remote-", dir=args.output.parent) as temporary:
            root = Path(temporary)
            staged = []
            for index, path in enumerate(paths):
                seed = args.seed + index * args.seed_stride
                image = remote.image(args, seed)
                image.load()
                if image.size != (args.width, args.height):
                    raise RuntimeError("Remote image dimensions differ from the request.")
                target = root / f"image-{index}.png"
                image.convert("RGB").save(target, format="PNG", compress_level=args.png_compress_level,
                                           optimize=args.png_optimize)
                report = {"schema": "iild-remote-image-v1", "status": "complete",
                          "model_input": args.model_input.metadata(), "configuration": image_configuration(args),
                          "execution": "remote", "weights_verified_locally": False,
                          "image": {"file": str(path), "seed": seed, "size": list(image.size),
                                    "sha256": file_sha256(target), "verified_decode": True}}
                sidecar = root / f"image-{index}.json"
                sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                staged.extend(((target, path), (sidecar, path.with_suffix(".json"))))
                reports.append(report)
            backups, published = [], []
            try:
                for index, (source, target) in enumerate(staged):
                    if target.is_symlink() or target.is_dir():
                        raise ValueError("Remote image targets cannot be symlinks or directories.")
                    if target.exists() and args.overwrite:
                        backup = root / f"backup-{index}"
                        target.rename(backup)
                        backups.append((backup, target))
                    publish_file(source, target, overwrite=False)
                    published.append(target)
            except BaseException:
                for target in reversed(published):
                    target.unlink()
                for backup, target in reversed(backups):
                    backup.rename(target)
                raise
    finally:
        lock.unlink(missing_ok=True)
    return reports


def validate_remote_video(args):
    # These values affect local tensor evaluation and cannot be approximated by
    # a provider's text-to-video API. Reject them before any paid request.
    controls = {"device", "dtype", "offload", "cpu_text_encoding", "vae_tiling", "max_sequence_length",
                "decode_timestep", "decode_noise_scale", "image_cond_noise_scale", "cache_dir", "local_files_only"}
    supplied = set(args._provided) & controls
    if supplied:
        raise ValueError("Local execution options cannot control a remote model: " + ", ".join(sorted(supplied)))
    if any(shot["conditions"] or shot["continue_previous"] for shot in args.shots):
        raise ValueError("Remote text-to-video supports independent text shots; keyframes/continuity require --model-path.")


def _decode_source(args, shot, clip, folder, environment):
    result = subprocess.run([environment["ffprobe"], "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(clip)],
                            capture_output=True, text=True, timeout=args.encoding_timeout)
    try:
        stream, = json.loads(result.stdout)["streams"]
        valid = (not result.returncode and not result.stderr.strip()
                 and (stream["width"], stream["height"]) == (args.width, args.height)
                 and int(stream["nb_read_frames"]) == shot["sample_frames"])
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("Remote video dimensions or decoded frame count differ from the LTX request.")
    result = subprocess.run([environment["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                             "-i", str(clip), "-map", "0:v:0", "-an", "-frames:v", str(shot["ltx_frames"]),
                             "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-start_number", "0",
                             str(folder / "frame-%06d.png")], capture_output=True, text=True,
                            timeout=args.encoding_timeout)
    if result.returncode:
        raise RuntimeError("Could not decode the remote LTX source frames.")
    records = []
    for index, position in enumerate(shot["source_positions"]):
        path = folder / f"frame-{index:06d}.png"
        with environment["Image"].open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (args.width, args.height):
                raise RuntimeError("Remote video produced an invalid source frame.")
        records.append({"index": shot["start_frame"] + position, "shot": shot["index"],
                        "source_index": index, "file": f"ltx/shot-{shot['index']:04d}/{path.name}",
                        "sha256": file_sha256(path)})
    return records


def generate_video(args):
    import shutil
    from animation_video import AnimationOutput, encode_video, preflight_animation
    from video_interpolator import interpolate_video, preflight_interpolator
    from video_options import configuration
    validate_remote_video(args)
    environment = preflight_animation(args)
    preflight_interpolator(args, environment)
    remote = RemoteInference(args.model_input)
    with AnimationOutput(args) as output:
        source_frames, clips = [], []
        for shot in args.shots:
            folder = output.frames / "ltx" / f"shot-{shot['index']:04d}"
            folder.mkdir(parents=True)
            clip = folder / "source.mp4"
            clip.write_bytes(remote.video(args, shot))
            source_frames.extend(_decode_source(args, shot, clip, folder, environment))
            clips.append({"shot": shot["index"], "file": str(clip.relative_to(output.frames)),
                          "sha256": file_sha256(clip), "decoded_frames": shot["sample_frames"]})
        stages = [{"name": "LTX", "execution": "remote", "architecture_validation": "caller-declared",
                   "weights_verified_locally": False, "output_frames": len(source_frames)}]
        if args.interpolation_enabled:
            frames, interpolation = interpolate_video(args, args.shots, source_frames, output, environment)
            stages.append({"name": "Interpolator", **interpolation})
        else:
            interpolation = {"enabled": False, "reason": "fps-at-most-12"}
            frames = []
            for source in source_frames:
                path = output.frames / f"frame-{source['index']:06d}.png"
                shutil.copyfile(output.frames / source["file"], path)
                frames.append({**source, "file": path.name})
        encoded = encode_video(output.frames, output.video, args, environment)
        report = {"schema": "iild-temporal-video-v1", "status": "complete",
                  "model_input": args.model_input.metadata(), "configuration": configuration(args),
                  "shots": args.shots, "source_clips": clips, "source_frames": source_frames,
                  "frames": frames, "stages": stages, "interpolation": interpolation, "video": encoded,
                  "source_timing": "decoded frames placed on the requested LTX source timeline",
                  "versions": environment["versions"]}
        output.commit(report)
    return report
