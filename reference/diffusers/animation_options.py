"""Video settings shared by Deforum and Interpolator, without media imports."""

import argparse
from fractions import Fraction
import math
from pathlib import Path
from typing import Any

DEFAULTS = {"max_frames": 120, "fps": 12.0, "video_crf": 18, "video_preset": "medium",
            "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "encoding_timeout": 300.0}


def allows_image_animation(fps: float, output: Path | str | None) -> bool:
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("--fps must be finite and in (0,240].")
    return fps <= 12 or (output is not None and Path(output).suffix.lower() == ".gif")


def validate_animation_output(fps: float, output: Path | str | None) -> None:
    suffix = Path(output).suffix.lower() if output is not None else ".mp4"
    if suffix not in (".mp4", ".gif"):
        raise ValueError("Animation output must use the .mp4 or .gif extension.")
    if not allows_image_animation(fps, output):
        raise ValueError("Deforum and Interpolator require FPS <= 12 or GIF output; "
                         "use the LTX video backend with a local LTX model for other videos.")
    if suffix == ".gif" and not 100 / 65535 <= fps <= 100:
        raise ValueError("GIF frame delays must fit 1..65535 centiseconds; FPS must be in [100/65535,100].")


def add_animation_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Animation output")
    group.add_argument("--animation-mode", choices=("none", "2D", "Interpolator"), default="none",
                       help="2D uses frame feedback; Interpolator blends prompt and seed endpoints")
    group.add_argument("--max-frames", "--frames", type=int, default=None, help="Total frames, including endpoints (default: 120)")
    group.add_argument("--duration", type=float, default=None, help="Seconds, rounded to frames; exclusive with --frames/--max-frames")
    group.add_argument("--fps", type=float, default=None, help="Default: 12; FPS above 12 requires GIF output")
    group.add_argument("--video-crf", type=int, default=None, help="H.264 quality in [0,51] (default: 18)")
    group.add_argument("--video-preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                                                "medium", "slow", "slower", "veryslow"), default=None)
    group.add_argument("--ffmpeg", default=None, help="FFmpeg executable with libx264 for MP4 or GIF/palette filters for GIF")
    group.add_argument("--ffprobe", default=None, help="FFprobe executable name or path")
    group.add_argument("--encoding-timeout", type=float, default=None,
                       help="Encoding/verification timeout in seconds (default: 300)")


def resolve_animation_options(args: Any) -> None:
    if args.animation_mode == "none":
        if args.duration is not None or any(getattr(args, name) is not None for name in DEFAULTS):
            raise ValueError("Animation options require --animation-mode 2D or Interpolator.")
        return
    if args.fps is None:
        args.fps = DEFAULTS["fps"]
    validate_animation_output(args.fps, args.output)
    if args.duration is not None:
        if args.max_frames is not None:
            raise ValueError("--duration and --frames/--max-frames are mutually exclusive.")
        if not math.isfinite(args.duration) or not 0 < args.duration * args.fps <= 1_000_000:
            raise ValueError("--duration must be positive and produce at most 1000000 frames.")
        args.max_frames = int(math.floor(args.duration * args.fps + 0.5))
        args.duration = None  # Canonical replay uses the resolved integer frame count.
    for name, value in DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 1 <= args.max_frames <= 1_000_000:
        raise ValueError("--max-frames must be in [1,1000000].")
    if Fraction(str(args.fps)).limit_denominator(1_000_000) == 0:
        raise ValueError("--fps is too small to represent as a video time base.")
    if not math.isfinite(args.encoding_timeout) or args.encoding_timeout <= 0:
        raise ValueError("--encoding-timeout must be finite and positive.")
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be in [0,51].")
    if args.num_images != 1:
        raise ValueError("Animation requires --num-images 1; --max-frames controls video length.")
    if args.hires_fix:
        raise ValueError("Animation does not combine with --hires-fix; choose the video resolution directly.")
