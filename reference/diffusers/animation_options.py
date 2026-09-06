"""Video settings shared by Deforum and Interpolator, without media imports."""

import argparse
from fractions import Fraction
import math
from typing import Any

DEFAULTS = {"max_frames": 120, "fps": 24.0, "video_crf": 18, "video_preset": "medium",
            "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "encoding_timeout": 300.0}


def add_animation_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Animation output")
    group.add_argument("--animation-mode", choices=("none", "2D", "Interpolator"), default="none",
                       help="2D uses frame feedback; Interpolator blends prompt and seed endpoints")
    group.add_argument("--max-frames", type=int, default=None, help="Total frames, including endpoints (default: 120)")
    group.add_argument("--fps", type=float, default=None, help="Video frames per second (default: 24)")
    group.add_argument("--video-crf", type=int, default=None, help="H.264 quality in [0,51] (default: 18)")
    group.add_argument("--video-preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                                                "medium", "slow", "slower", "veryslow"), default=None)
    group.add_argument("--ffmpeg", default=None, help="FFmpeg executable name or path, with libx264")
    group.add_argument("--ffprobe", default=None, help="FFprobe executable name or path")
    group.add_argument("--encoding-timeout", type=float, default=None,
                       help="Encoding/verification timeout in seconds (default: 300)")


def resolve_animation_options(args: Any) -> None:
    if args.animation_mode == "none":
        if any(getattr(args, name) is not None for name in DEFAULTS):
            raise ValueError("Animation options require --animation-mode 2D or Interpolator.")
        return
    for name, value in DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 1 <= args.max_frames <= 1_000_000:
        raise ValueError("--max-frames must be in [1,1000000].")
    if not math.isfinite(args.fps) or not 0 < args.fps <= 240:
        raise ValueError("--fps must be finite and in (0,240].")
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
