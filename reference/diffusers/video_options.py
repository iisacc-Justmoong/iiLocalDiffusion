"""Offline configuration and shot planning for temporal video diffusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from generation_config import ConfigurationArgumentParser, json_object

ROOT = Path(__file__).resolve().parents[2]
CAMERA_MOTIONS = {
    "none": "",
    "static": "The camera remains stationary on a locked tripod.",
    "dolly-in": "The camera moves slowly forward toward the subject, revealing depth and parallax.",
    "dolly-out": "The camera moves backward away from the subject, revealing the surrounding scene.",
    "pan-left": "The camera rotates smoothly to the left across the scene.",
    "pan-right": "The camera rotates smoothly to the right across the scene.",
    "tilt-up": "The camera tilts upward from the lower part of the scene to the sky.",
    "tilt-down": "The camera tilts downward from above toward the subject.",
    "orbit-left": "The camera circles to the left around the subject, revealing its sides and the background.",
    "orbit-right": "The camera circles to the right around the subject, revealing its sides and the background.",
    "crane-up": "The camera rises vertically above the subject, gradually revealing an elevated view.",
    "crane-down": "The camera descends vertically toward the subject from an elevated view.",
    "zoom-in": "The lens zooms in smoothly, bringing the subject into a close view.",
    "zoom-out": "The lens zooms out smoothly, widening the view of the subject and surroundings.",
    "handheld": "The camera follows the action with subtle, natural handheld movement.",
    "tracking": "The camera tracks alongside the moving subject, keeping it in the frame.",
    "dolly-zoom": "The camera moves backward while the lens zooms in, keeping the subject's size steady as the background perspective changes.",
}


def build_parser():
    parser = ConfigurationArgumentParser(description="Generate locally with a temporal video diffusion model.",
                                        allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("video",), default="video")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--list-camera-motions", action="store_true")
    from model_sources import add_model_arguments
    add_model_arguments(parser, local_help="Local LTX Diffusers model directory")
    parser.add_argument("--revision", help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "build/reference/huggingface")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Local model loading is mandatory; disabling it is rejected")
    parser.add_argument("--prompt", default="A red cube rotates slowly on a white table in a bright studio.")
    parser.add_argument("--negative-prompt", default="blurry, distorted, inconsistent motion, flicker")
    parser.add_argument("--camera", nargs="+", choices=tuple(CAMERA_MOTIONS), default=["none"])
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--storyboard", type=Path, help="JSON shot list; each shot may have image keyframes")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", "--max-frames", type=int, help="Output frames per shot; exclusive with --duration")
    parser.add_argument("--duration", type=float, help="Seconds per shot (default: 5); rounded to output frames")
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--interpolation-factor", type=int, default=2,
                        help="LTX source-frame spacing in output frames, 2..8 (default: 2); "
                             "FPS > 12 uses the frame Interpolator after LTX")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "metal", "cuda", "rocm"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--offload", choices=("auto", "none", "model", "sequential"), default="auto")
    parser.add_argument("--cpu-text-encoding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-timestep", type=float, default=0.05)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--image-cond-noise-scale", type=float, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                                                 "medium", "slow", "slower", "veryslow"), default="medium")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--encoding-timeout", type=float, default=300)
    return parser


def _integer(value, name, minimum, maximum):
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}].")
    return value


def _number(value, name, minimum, maximum):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite and in [{minimum},{maximum}].")
    return value


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or len(value) > 32_000 or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'a' if empty else 'a non-empty'} string of at most 32000 characters.")
    return value


def _image(value, root):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("A keyframe image must be a non-empty local path.")
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError(f"Keyframe image must be a non-empty local file of at most 256 MiB: {path}")
    return str(path)


def camera_prompt(camera):
    if not isinstance(camera, list) or not 1 <= len(camera) <= 3:
        raise ValueError("camera must be an array of one to three motion names.")
    if any(not isinstance(name, str) or name not in CAMERA_MOTIONS for name in camera):
        raise ValueError("Unknown camera motion.")
    if len(set(camera)) != len(camera) or (len(camera) > 1 and set(camera) & {"none", "static"}):
        raise ValueError("A neutral or static camera cannot be mixed with other camera motions.")
    for pair in ({"pan-left", "pan-right"}, {"tilt-up", "tilt-down"}, {"dolly-in", "dolly-out"},
                 {"orbit-left", "orbit-right"}, {"crane-up", "crane-down"}, {"zoom-in", "zoom-out"}):
        if pair <= set(camera):
            raise ValueError("Opposing camera motions cannot be applied simultaneously.")
    return " ".join(CAMERA_MOTIONS[name] for name in camera if CAMERA_MOTIONS[name])


def _frame_count(frames, duration, fps):
    if frames is not None and duration is not None:
        raise ValueError("frames and duration are mutually exclusive.")
    if frames is not None:
        return _integer(frames, "frames", 2, 4097)
    duration = 5 if duration is None else _number(duration, "duration", 0.001, 600)
    return _integer(int(math.floor(duration * fps + 0.5)), "duration * fps", 2, 4097)


def plan_shots(args):
    fields = {"prompt", "negative_prompt", "camera", "frames", "duration", "seed",
              "first_frame", "last_frame", "conditions", "continue_previous"}
    root = Path.cwd()
    if args.storyboard:
        path = args.storyboard.expanduser().resolve()
        root = path.parent
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Storyboard exceeds the 1 MiB limit.")
            document = json_object(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"Cannot read storyboard: {error}") from error
        if set(document) != {"shots"}:
            raise ValueError("Storyboard must contain only a shots array.")
        raw = document["shots"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= 256:
            raise ValueError("Storyboard must have 1 to 256 shots.")
    else:
        raw = [{}]
    result, offset = [], 0
    for index, source in enumerate(raw):
        if not isinstance(source, dict) or set(source) - fields:
            raise ValueError(f"Shot {index + 1} has unknown fields or is not an object.")
        frames = _frame_count(source.get("frames", None if "duration" in source else args.frames),
                              source.get("duration", None if "frames" in source else args.duration), args.fps)
        camera = source.get("camera", args.camera)
        motion = camera_prompt(camera)
        prompt = _text(source.get("prompt", args.prompt), "prompt")
        negative = _text(source.get("negative_prompt", args.negative_prompt), "negative_prompt", empty=True)
        seed = _integer(source.get("seed", (args.seed + index) % (2**63)), "seed", 0, 2**63 - 1)
        continuity = source.get("continue_previous", False)
        if not isinstance(continuity, bool) or (continuity and index == 0):
            raise ValueError("continue_previous requires a preceding shot and a boolean value.")
        conditions = []
        first = source.get("first_frame", args.first_frame)
        last = source.get("last_frame", args.last_frame)
        for image, frame in ((first, 0), (last, frames - 1)):
            if image is not None:
                conditions.append({"image": _image(image, root), "frame": frame, "strength": 1.0})
        extra = source.get("conditions", [])
        if not isinstance(extra, list) or len(extra) > 32:
            raise ValueError("conditions must be an array of at most 32 keyframes.")
        for condition in extra:
            if (not isinstance(condition, dict) or set(condition) - {"image", "frame", "strength"}
                    or not {"image", "frame"} <= set(condition)):
                raise ValueError("Each condition requires image, frame and optional strength.")
            conditions.append({"image": _image(condition["image"], root),
                               "frame": _integer(condition["frame"], "condition.frame", 0, frames - 1),
                               "strength": _number(condition.get("strength", 1), "condition.strength", 0.001, 1)})
        indices = [condition["frame"] for condition in conditions]
        if len(set(indices)) != len(indices) or (continuity and 0 in indices):
            raise ValueError("A frame may have only one image condition, including continue_previous.")
        # Preserve the exact output timeline, including off-grid image anchors
        # and the last frame. Only intervening frames belong to the interpolator.
        spacing = args.interpolation_factor if args.interpolation_enabled else 1
        positions = sorted({*range(0, frames, spacing), frames - 1, *indices})
        for condition in conditions:
            condition["source_frame"] = positions.index(condition["frame"])
        source_count = len(positions)
        result.append({"index": index, "start_frame": offset, "frames": frames,
                       "source_positions": positions, "ltx_frames": source_count,
                       "sampling_fps": args.fps * (source_count - 1) / (frames - 1),
                       "sample_frames": ((source_count - 1 + 7) // 8) * 8 + 1,
                       "seed": seed, "prompt": prompt, "negative_prompt": negative, "camera": list(camera),
                       "effective_prompt": (motion + " " + prompt.strip()).strip(),
                       "conditions": sorted(conditions, key=lambda condition: condition["frame"]),
                       "continue_previous": continuity})
        offset += frames
    return result


def resolve_options(args):
    for name in ("width", "height"):
        value = _integer(getattr(args, name), name, 32, 4096)
        if value % 32:
            raise ValueError(f"{name} must be divisible by 32 for the LTX Video VAE.")
    _number(args.fps, "fps", 0.01, 240)
    _integer(args.interpolation_factor, "interpolation_factor", 2, 8)
    _integer(args.steps, "steps", 1, 1000)
    _integer(args.seed, "seed", 0, 2**63 - 1)
    _integer(args.max_sequence_length, "max_sequence_length", 16, 512)
    _number(args.guidance_scale, "guidance_scale", 0, 100)
    _integer(args.video_crf, "video_crf", 0, 51)
    _number(args.encoding_timeout, "encoding_timeout", 0.01, 86400)
    for name in ("decode_timestep", "decode_noise_scale", "image_cond_noise_scale"):
        _number(getattr(args, name), name, 0, 1)
    if args.device == "cpu" and args.offload in ("model", "sequential"):
        raise ValueError("RAM offload requires a GPU execution device.")
    from model_sources import resolve_model_input
    args.model_input = resolve_model_input(args)
    if args.model_input.kind == "local":
        model = Path(args.model_input.location)
        if not model.is_dir():
            raise ValueError(f"--model must be an existing local LTX Diffusers directory: {args.model}")
        args.model = str(model)
    elif args.model_input.family != "ltx":
        raise ValueError("Remote video requires --model-family ltx and an endpoint/provider serving LTX weights.")
    if args.revision is not None:
        raise ValueError("Generation model inputs do not accept a Hub revision.")
    if not args.local_files_only:
        raise ValueError("Video generation requires local model files; --no-local-files-only is unsupported.")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    for name in ("storyboard", "first_frame", "last_frame"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_was_default = args.output is None
    args.output = (args.output or ROOT / "build/reference/video.mp4").expanduser().absolute()
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension.")
    args.interpolation_enabled = args.fps > 12
    args.shots = plan_shots(args)
    from animation_video import output_targets
    targets = [path.resolve() for path in output_targets(args.output)]
    inputs = [condition["image"] for shot in args.shots for condition in shot["conditions"]]
    inputs.extend(path for path in (args.config, args.storyboard) if path is not None)
    for source in inputs:
        path = Path(source).resolve()
        if path in targets or path.is_relative_to(targets[2]):
            raise ValueError("Video output must not replace a configuration, storyboard or keyframe input.")
    args.max_frames = sum(shot["frames"] for shot in args.shots)
    args.animation_mode = "Video"
    return args


def configuration(args):
    # Keep the input vocabulary replayable; the generated shot plan belongs in
    # the output sidecar, not in a config that the parser could not load again.
    remote = args.model_input.kind != "local"
    local_controls = {"device", "dtype", "offload", "cpu_text_encoding", "vae_tiling", "max_sequence_length",
                      "decode_timestep", "decode_noise_scale", "image_cond_noise_scale", "cache_dir", "local_files_only"}
    return {name: str(value) if isinstance(value, Path) else value
            for name in args._argument_names
            if name != "list_camera_motions"
            if not (remote and name in local_controls)
            for value in (getattr(args, name),)}
