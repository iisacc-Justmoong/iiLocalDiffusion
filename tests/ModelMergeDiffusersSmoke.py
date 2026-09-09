#!/usr/bin/env python3
"""Offline merge -> reload -> SDK inference with locally initialized tiny models."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-entry", type=Path, default=ROOT / "reference/merge.py")
    parser.add_argument("--generate-entry", type=Path, default=ROOT / "reference/generate.py")
    args = parser.parse_args()
    args.merge_entry = args.merge_entry.resolve(strict=True)
    args.generate_entry = args.generate_entry.resolve(strict=True)
    import torch
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    from PIL import Image

    destination = ROOT / "build/reference/model-merge-smoke"
    destination.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=destination))
    pipelines = []
    for index, name in enumerate(("base", "additional")):
        torch.manual_seed(100 + index)
        unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
                          down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
                          block_out_channels=(8,), layers_per_block=1, norm_num_groups=4)
        pipeline = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10))
        pipeline.save_pretrained(run / name, safe_serialization=True)
        pipelines.append(pipeline)
    environment = {**os.environ, "IILD_PYTHON_EXECUTABLE": sys.executable,
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONPYCACHEPREFIX": str(ROOT / "build/pycache")}
    cases = []
    for mode in ("weighted-sum", "weighted-difference"):
        merged = run / mode
        completed = subprocess.run([sys.executable, str(args.merge_entry),
            "--base-model", str(run / "base"), "--additional-model", str(run / "additional"),
            "--mode", mode, "--weights", "0.25", "--output", str(merged)],
            env=environment, capture_output=True, text=True, check=True)
        merge_report = json.loads(completed.stdout)
        loaded = DDPMPipeline.from_pretrained(merged, local_files_only=True)
        base_state, additional_state = [pipeline.unet.state_dict() for pipeline in pipelines]
        for key, actual in loaded.unet.state_dict().items():
            expected = (base_state[key] * 0.75 + additional_state[key] * 0.25
                        if mode == "weighted-sum" else base_state[key] - additional_state[key] * 0.25)
            torch.testing.assert_close(actual, expected)
        generated = run / f"{mode}-images"
        subprocess.run([sys.executable, str(args.generate_entry), "--backend", "diffusers",
            "--model-path", str(merged), "--device", "cpu", "--steps", "2", "--seed", "42",
            "--output-dir", str(generated)], env=environment, capture_output=True, text=True, check=True)
        images = sorted(generated.rglob("*.png"))
        if len(images) != 1:
            raise AssertionError(f"Expected one generated image, found {images}")
        with Image.open(images[0]) as image:
            image.load()
            if image.size != (8, 8):
                raise AssertionError(f"Unexpected output size: {image.size}")
        cases.append({"mode": mode, "tensor_values_verified": True, "pipeline_reload_verified": True,
                      "image": str(images[0]), "merge": merge_report,
                      "generation_report": str(generated / "generation.json")})
    report = {"schema": "iild-model-merge-smoke-v1", "offline": True,
              "weights": "locally initialized tiny models; no image-quality claim",
              "merge_entry": str(args.merge_entry), "generate_entry": str(args.generate_entry), "cases": cases}
    report_path = run / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stdout or "")
        sys.stderr.write(error.stderr or "")
        raise
