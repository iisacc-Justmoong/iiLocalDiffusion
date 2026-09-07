"""Motion-compensated frame interpolation after local LTX inference."""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import shutil
import subprocess
import time

from weight_files import file_sha256

MOTION_OPTIONS = "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:scd=fdiff:scd_threshold=10"


def preflight_interpolator(args, environment):
    if not args.interpolation_enabled:
        return
    for name in ("minterpolate", "tpad"):
        result = subprocess.run([environment["ffmpeg"], "-hide_banner", "-h", f"filter={name}"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or f"Filter {name}" not in result.stdout:
            raise ValueError(f"LTX's second-stage Interpolator requires the FFmpeg {name} filter.")


def _timeline(folder, positions, fps):
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    # Round absolute timestamps, then subtract, to avoid accumulating per-frame
    # microsecond rounding at fractional rates. All filenames are generated here.
    times = [int(position * 1_000_000 / rate + Fraction(1, 2))
             for position in [*positions, positions[-1] + 1]]
    lines = ["ffconcat version 1.0"]
    for index in range(len(positions)):
        lines.extend([f"file frame-{index:06d}.png", f"option framerate {rate}",
                      f"duration {(times[index + 1] - times[index]) / 1_000_000:.6f}"])
    path = folder / "timeline.ffconcat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, rate


def interpolate_video(args, shots, source_frames, output, environment):
    if not args.interpolation_enabled:
        return source_frames, {"enabled": False, "reason": "fps-at-most-12"}
    started = time.monotonic()
    records, shot_reports = [], []
    for shot in shots:
        shot_started = time.monotonic()
        positions = shot["source_positions"]
        sources = sorted((frame for frame in source_frames if frame["shot"] == shot["index"]),
                         key=lambda frame: frame["index"])
        if [frame["index"] - shot["start_frame"] for frame in sources] != positions:
            raise RuntimeError("Interpolator source frame count or anchor positions do not match the LTX plan.")
        folder = output.frames / "ltx" / f"shot-{shot['index']:04d}"
        for index, source in enumerate(sources):
            path = folder / f"frame-{index:06d}.png"
            if output.frames / source["file"] != path or file_sha256(path) != source["sha256"]:
                raise RuntimeError("An LTX source frame changed before interpolation.")
        timeline, rate = _timeline(folder, positions, args.fps)
        # Boundary clones supply motion-estimation lookahead. They are trimmed
        # away; only the caller's exact final-frame timeline is published.
        filters = (f"format=yuv444p,tpad=start_mode=clone:start=2:stop_mode=clone:stop=4,"
                   f"minterpolate=fps={rate}:{MOTION_OPTIONS},"
                   f"trim=start_frame=2:end_frame={shot['frames'] + 2},setpts=PTS-STARTPTS")
        command = [environment["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(timeline), "-vf", filters,
                   "-an", "-frames:v", str(shot["frames"]), "-fps_mode", "passthrough",
                   "-c:v", "png", "-pix_fmt", "rgb24", "-start_number", str(shot["start_frame"]),
                   str(output.frames / "frame-%06d.png")]
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.encoding_timeout)
        if result.returncode:
            raise RuntimeError("LTX frame interpolation failed: " + result.stderr[-4000:])
        anchors = {source["index"]: source for source in sources}
        for local_index in range(shot["frames"]):
            index = shot["start_frame"] + local_index
            path = output.frames / f"frame-{index:06d}.png"
            if not path.is_file():
                raise RuntimeError("Interpolator did not produce the requested output frame count.")
            if index in anchors:
                # Keep actual LTX pixels at every anchor, without a YUV/RGB
                # round-trip or changes to endpoint/keyframe provenance.
                shutil.copyfile(output.frames / anchors[index]["file"], path)
            with environment["Image"].open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB" or image.size != (args.width, args.height):
                    raise RuntimeError("Interpolator output must be an RGB PNG with the requested dimensions.")
            digest = file_sha256(path)
            record = {"index": index, "shot": shot["index"], "file": path.name, "sha256": digest}
            if index in anchors:
                if digest != anchors[index]["sha256"]:
                    raise RuntimeError("An LTX anchor changed during interpolation.")
                record.update(method="ltx-anchor", source_frames=[index])
            else:
                right = bisect_right(positions, local_index)
                left_position, right_position = positions[right - 1:right + 1]
                record.update(method="motion-interpolated",
                              source_frames=[shot["start_frame"] + left_position, shot["start_frame"] + right_position],
                              fraction=(local_index - left_position) / (right_position - left_position))
            records.append(record)
        shot_reports.append({"shot": shot["index"], "input_frames": len(sources), "output_frames": shot["frames"],
                             "inserted_frames": shot["frames"] - len(sources), "filter": filters,
                             "elapsed_seconds": time.monotonic() - shot_started})
    if len(records) != args.max_frames:
        raise RuntimeError("Interpolator output length does not match the requested video timeline.")
    return records, {"enabled": True, "engine": "ffmpeg-minterpolate", "device": "cpu",
                     "method": "motion-compensated-frame-interpolation", "factor": args.interpolation_factor,
                     "input_frames": len(source_frames), "output_frames": len(records),
                     "inserted_frames": len(records) - len(source_frames), "fps": args.fps,
                     "duration_seconds": len(records) / args.fps, "anchors_preserved": True,
                     "shot_boundaries": "preserved-no-cross-shot-interpolation", "shots": shot_reports,
                     "elapsed_seconds": time.monotonic() - started, "verified": True}
