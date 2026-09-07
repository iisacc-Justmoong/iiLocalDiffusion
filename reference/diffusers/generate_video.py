#!/usr/bin/env python3
"""Local cinematic video generation from text, keyframes and a shot plan."""

import json
import os
import subprocess

from video_options import build_parser, configuration, resolve_options, CAMERA_MOTIONS


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_camera_motions:
        print(json.dumps(CAMERA_MOTIONS, indent=2))
        return 0
    try:
        args = resolve_options(args)
        if args.model_input.kind != "local":
            from remote_generation import validate_remote_video
            validate_remote_video(args)
        if args.print_config:
            print(json.dumps(configuration(args), indent=2, allow_nan=False))
            return 0
        if args.model_input.kind == "local":
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            from video_runtime import generate_video
        else:
            from remote_generation import generate_video
        generate_video(args)
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Video generation failed: {error}\n")
    print(f"Video: {args.output}\nReport: {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
