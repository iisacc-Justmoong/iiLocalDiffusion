#!/usr/bin/env python3
"""Real offline SD/SDXL Interpolator inference with small random test weights."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from DeforumDiffusersSmoke import create_fixture

ROOT = Path(__file__).resolve().parents[1]


def create_interpolator_fixture(directory, family):
    create_fixture(directory, family)
    from transformers import CLIPTokenizer
    # Transformers 5 accepts vocab/merges, not the old vocab_file/merges_file
    # constructor arguments. Keep a real content token distinct from padding.
    tokenizer = CLIPTokenizer(vocab={"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2},
                              merges=[], model_max_length=77)
    assert tokenizer("a")["input_ids"] != tokenizer("")["input_ids"]
    tokenizer.save_pretrained(directory / "model/tokenizer")
    if family == "sdxl":
        tokenizer.save_pretrained(directory / "model/tokenizer_2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    directory = ROOT / "build/interpolator-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from PIL import Image
    control = directory / "control.png"
    image = Image.new("RGB", (64, 64), "black")
    for index in range(64):
        image.putpixel((index, index), (255, 128, 64))
    image.save(control)
    reports = []
    for family in ("sd15", "sdxl"):
        fixture = directory / family
        if not (fixture / "model/model_index.json").is_file():
            create_interpolator_fixture(fixture, family)
        for use_controlnet in (False, True):
            name = f"{family}{'-controlnet' if use_controlnet else ''}-{args.device}"
            output = directory / (name + ".mp4")
            command = [sys.executable, str(ROOT / "reference/generate.py"), "--backend", "interpolator",
                       "--preset", "sd15-compatible" if family == "sd15" else "sdxl",
                       "--model", str(fixture / "model"), "--device", args.device,
                       "--dtype", "float32", "--cpu-threads", "2", "--width", "64", "--height", "64",
                       "--steps", "4", "--max-frames", "3", "--fps", "8",
                       # The tiny vocabulary intentionally has only the content token 'a'.
                       "--prompt", "a", "--end-prompt", "a a a", "--end-seed", "43",
                       "--negative-prompt", "a a", "--end-negative-prompt", "",
                       "--local-files-only", "--no-progress", "--overwrite", "--output", str(output)]
            if use_controlnet:
                command.extend(["--controlnet", str(fixture / "controlnet"), "--control-image", str(control)])
            with (directory / (name + ".log")).open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Smoke failed: {name}; inspect {directory / (name + '.log')}")
            report = json.loads(output.with_suffix(".json").read_text())
            frames = report["frames"]
            assert report["status"] == "complete" and report["completed_frames"] == 3
            assert report["schema"] == "iild-interpolator-v1"
            assert [frame["fraction"] for frame in frames] == [0, .5, 1]
            assert all(frame["sampling"]["finite_latents"] for frame in frames)
            assert [frame["sampling"]["executed_steps"] for frame in frames] == [4, 4, 4]
            assert report["output"]["verified_decode"]
            assert frames[0]["latents"]["sha256"] == report["interpolation"]["noise"]["start"]["sha256"]
            assert frames[-1]["latents"]["sha256"] == report["interpolation"]["noise"]["end"]["sha256"]
            for field in ("latents", "output"):
                assert len({frame[field]["sha256"] for frame in frames}) == 3
            assert len({frame["conditioning"]["prompt_embeds"]["sha256"] for frame in frames}) == 3
            if family == "sdxl":
                assert all(frame["conditioning"]["pooled_prompt_embeds"] for frame in frames)
                assert all(frame["conditioning"]["negative_pooled_prompt_embeds"] for frame in frames)
            reports.append({"name": name, "video": str(output), "sampling_steps": [
                frame["sampling"]["executed_steps"] for frame in frames]})
            print(json.dumps(reports[-1]), flush=True)
    (directory / f"verification-{args.device}.json").write_text(json.dumps({
        "fixture": "random small compatible weights; execution validation only", "runs": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
