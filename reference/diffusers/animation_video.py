"""Transactional PNG-sequence/MP4 publication shared by animation generators."""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from generation_output import publish_file
from weight_files import file_sha256

SCHEMAS = {"2D": "iild-deforum-2d-v1", "Interpolator": "iild-interpolator-v1"}


def output_targets(output: Path) -> tuple[Path, Path, Path]:
    return output, output.with_suffix(".json"), output.with_name(output.stem + "-frames")


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def check_targets(args: Any) -> None:
    video, report, frames = output_targets(args.output)
    existing = [path for path in (video, report, frames) if _exists(path)]
    if existing and not args.overwrite:
        raise ValueError("Refusing to overwrite animation output: " + ", ".join(map(str, existing)))
    if any(path.is_symlink() for path in existing) or any(path.is_dir() for path in (video, report)):
        raise ValueError("Animation file targets cannot be directories or symlinks.")
    if frames in existing:
        try:
            marker = json.loads((frames / "manifest.json").read_text(encoding="utf-8"))
            if (marker.get("schema") != SCHEMAS[getattr(args, "animation_mode", "2D")]
                    or marker.get("video") != str(video.absolute())):
                raise ValueError("Unrecognized frames directory.")
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError("Refusing to replace an unmanaged animation frames directory.") from error


def _executable(value: str, name: str) -> str:
    result = shutil.which(str(Path(value).expanduser()))
    if not result:
        raise ValueError(f"Animation requires {name}; install FFmpeg or set --{name} to its executable.")
    return str(Path(result).absolute())


def preflight_animation(args: Any) -> dict[str, Any]:
    base = args.output
    if args.output_was_default and not args.overwrite:
        run = 2
        while any(_exists(path) for path in output_targets(args.output)):
            args.output = base.with_stem(f"{base.stem}-run-{run:04d}")
            run += 1
    check_targets(args)
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Animation requires Pillow from the Diffusers runtime dependencies.") from error
    environment = {"Image": Image, "ffmpeg": _executable(args.ffmpeg, "ffmpeg"),
                   "ffprobe": _executable(args.ffprobe, "ffprobe")}
    encoder = subprocess.run([environment["ffmpeg"], "-hide_banner", "-h", "encoder=libx264"],
                             capture_output=True, text=True, timeout=30, check=True)
    if "Encoder libx264" not in encoder.stdout:
        raise ValueError("The selected FFmpeg does not provide the libx264 H.264 encoder.")
    environment["versions"] = {"pillow": Image.__version__}
    for name in ("ffmpeg", "ffprobe"):
        version = subprocess.run([environment[name], "-version"], capture_output=True,
                                 text=True, timeout=30, check=True)
        environment["versions"][name] = version.stdout.splitlines()[0]
    return environment


def encode_video(frames: Path, output: Path, args: Any, environment: dict[str, Any]) -> dict[str, Any]:
    rate = Fraction(str(args.fps)).limit_denominator(1_000_000)
    command = [environment["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-framerate", str(rate), "-start_number", "0", "-i", str(frames / "frame-%06d.png"),
               "-frames:v", str(args.max_frames), "-an", "-c:v", "libx264", "-crf", str(args.video_crf),
               "-preset", args.video_preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.encoding_timeout)
    if result.returncode:
        raise RuntimeError("FFmpeg animation encoding failed: " + result.stderr[-4000:])
    probe = subprocess.run([environment["ffprobe"], "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries",
                            "stream=codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate,duration",
                            "-of", "json", str(output)], capture_output=True, text=True,
                           timeout=args.encoding_timeout)
    if probe.returncode or probe.stderr.strip():
        raise RuntimeError("FFprobe could not decode the animation: " + probe.stderr[-4000:])
    try:
        stream, = json.loads(probe.stdout)["streams"]
        valid = (stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                 and (stream["width"], stream["height"]) == (args.width, args.height)
                 and int(stream["nb_read_frames"]) == args.max_frames
                 and abs(float(Fraction(stream["avg_frame_rate"]) - rate)) < 1e-6
                 and abs(float(stream["duration"]) - args.max_frames / float(rate)) <= 1 / float(rate))
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        raise RuntimeError("FFprobe returned an invalid animation stream report.") from error
    if not valid:
        raise RuntimeError("Encoded video does not match the requested frame count, size, codec or frame rate.")
    return {"codec": "h264", "pixel_format": "yuv420p", "frame_count": args.max_frames,
            "fps": float(rate), "size": [args.width, args.height], "duration_seconds": float(stream["duration"]),
            "sha256": file_sha256(output), "size_bytes": output.stat().st_size, "verified_decode": True}


class AnimationOutput:
    """Keep old output intact on sampling/encoding failure; publish the report last."""

    def __init__(self, args: Any):
        self.args = args
        self.targets = output_targets(args.output)
        self.lock = args.output.with_name("." + args.output.name + ".animation.lock")
        self.temporary = None

    def __enter__(self):
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        try:
            check_targets(self.args)
            self.temporary = tempfile.TemporaryDirectory(prefix=".iild-animation-", dir=self.args.output.parent)
            self.directory = Path(self.temporary.name)
            self.frames = self.directory / "frames"
            self.frames.mkdir()
            self.video = self.directory / "video.mp4"
        except BaseException:
            self.lock.unlink()
            raise
        return self

    def commit(self, metadata: dict[str, Any]) -> None:
        video, report, frames = self.targets
        marker = {"schema": SCHEMAS[getattr(self.args, "animation_mode", "2D")],
                  "video": str(video.absolute()), "frame_count": self.args.max_frames}
        (self.frames / "manifest.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        staged_report = self.directory / "report.json"
        staged_report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        check_targets(self.args)
        backups, published = [], []
        try:
            # Retire the old completion record first, then replace its artifacts.
            for index, target in enumerate((report, video, frames)):
                if _exists(target):
                    backup = self.directory / f"previous-{index}"
                    os.replace(target, backup)
                    backups.append((backup, target))
            frames.mkdir()  # Exclusive creation also protects against a competing producer.
            published.append(frames)
            for source in self.frames.iterdir():
                os.replace(source, frames / source.name)
            for source, target in ((self.video, video), (staged_report, report)):
                publish_file(source, target, overwrite=False)
                published.append(target)
        except BaseException:
            for target in reversed(published):
                if target == frames:
                    shutil.rmtree(target)
                else:
                    target.unlink()
            for backup, target in reversed(backups):
                os.replace(backup, target)
            raise

    def __exit__(self, *exception):
        try:
            if self.temporary is not None:
                self.temporary.cleanup()
        finally:
            self.lock.unlink(missing_ok=True)
