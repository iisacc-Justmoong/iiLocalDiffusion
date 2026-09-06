#!/usr/bin/env python3
"""Offline Interpolator with LoRA, learned tokens, ControlNet and sequential GPU offload."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from InterpolatorDiffusersSmoke import create_interpolator_fixture

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="sequential")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from safetensors.torch import save_file
    directory = ROOT / "build/interpolator-smoke"
    fixture = directory / "sd15"
    if not (fixture / "model/model_index.json").is_file():
        create_interpolator_fixture(fixture, "sd15")
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    unet = UNet2DConditionModel.from_pretrained(fixture / "model/unet", local_files_only=True)
    unet.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]))
    with torch.no_grad():
        for name, parameter in unet.named_parameters():
            if "lora_" in name:
                parameter.normal_(0, .01)
    StableDiffusionPipeline.save_lora_weights(
        directory / "adapter", unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        safe_serialization=True)
    embedding = directory / "learned-token.safetensors"
    save_file({"clip_l": torch.randn((2, 768)) * .02}, str(embedding), metadata={"token": "iildtest"})
    control = directory / "adapter-control.png"
    image = Image.new("RGB", (64, 64), "black")
    for index in range(64):
        image.putpixel((index, index), (255, 128, 64))
    image.save(control)
    name = f"adapters-{args.device}-{args.dtype}-{args.offload}"
    output = directory / (name + ".mp4")
    command = [sys.executable, str(ROOT / "reference/generate.py"), "--backend", "interpolator",
               "--preset", "sd15-compatible", "--model", str(fixture / "model"),
               "--device", args.device, "--dtype", args.dtype, "--offload", args.offload,
               "--cpu-threads", "2", "--width", "64", "--height", "64", "--steps", "4",
               "--max-frames", "3", "--fps", "8", "--prompt", "a iildtest", "--end-prompt", "a a iildtest",
               "--seed", "42", "--end-seed", "43", "--lora", str(directory / "adapter"),
               "--lora-weight-name", "pytorch_lora_weights.safetensors",
               "--text-embedding", str(embedding), "--controlnet", str(fixture / "controlnet"),
               "--control-image", str(control), "--local-files-only", "--no-progress",
               "--overwrite", "--output", str(output)]
    log = directory / (name + ".log")
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Adapter smoke failed; inspect {log}")
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["status"] == "complete" and report["output"]["verified_decode"]
    assert report["adapters"][0]["active_adapters"] == ["iild_lora"]
    assert report["text_embeddings"][0]["registrations"][0]["vector_count"] == 2
    expected_offload = {"none": "none", "model": "model-cpu", "sequential": "sequential-cpu"}[args.offload]
    assert report["runtime"]["optimization"]["offload_policy"] == expected_offload
    assert report["runtime"]["dtype"] == "torch." + args.dtype
    assert all(frame["sampling"]["executed_steps"] == 4 for frame in report["frames"])
    assert len({frame["conditioning"]["prompt_embeds"]["sha256"] for frame in report["frames"]}) == 3
    print(json.dumps({"video": str(output), "verified": True,
                      "features": ["LoRA", "Textual Inversion", "ControlNet", args.dtype, args.offload]}))


if __name__ == "__main__":
    main()
