#!/usr/bin/env python3
"""Offline real-inference smoke with small random weights; not a visual-quality test.

Run with the Diffusers venv and optional requirements-deforum.txt. All model
fixtures, videos and logs are written to build/deforum-smoke. No weights are
downloaded, and production model contracts are not relaxed for this fixture.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def create_fixture(directory, family):
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                           StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel)
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    torch.manual_seed(2026)
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = directory / "vocab.json"
    vocabulary.write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}))
    merges = directory / "merges.txt"
    merges.write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab_file=str(vocabulary), merges_file=str(merges), model_max_length=77)
    config = CLIPTextConfig(vocab_size=3, hidden_size=768, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=12, max_position_embeddings=77, projection_dim=768,
                            bos_token_id=0, eos_token_id=1, pad_token_id=1)
    text_encoder = CLIPTextModel(config)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(32,) * 4,
                        layers_per_block=1, norm_num_groups=32, sample_size=64,
                        scaling_factor=0.18215 if family == "sd15" else 0.13025)
    unet_arguments = dict(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                          block_out_channels=(32, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                          up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), norm_num_groups=32,
                          cross_attention_dim=768 if family == "sd15" else 2048, attention_head_dim=4)
    if family == "sdxl":
        unet_arguments.update(addition_embed_type="text_time", addition_time_embed_dim=256,
                              projection_class_embeddings_input_dim=2816, use_linear_projection=True)
    unet = UNet2DConditionModel(**unet_arguments)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
                              steps_offset=1)
    if family == "sd15":
        pipeline = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                           unet=unet, scheduler=scheduler, safety_checker=None,
                                           feature_extractor=None, requires_safety_checker=False)
    else:
        config_2 = CLIPTextConfig(vocab_size=3, hidden_size=1280, intermediate_size=32, num_hidden_layers=2,
                                  num_attention_heads=20, max_position_embeddings=77, projection_dim=1280,
                                  bos_token_id=0, eos_token_id=1, pad_token_id=1)
        pipeline = StableDiffusionXLPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                             text_encoder_2=CLIPTextModelWithProjection(config_2),
                                             tokenizer_2=tokenizer, unet=unet, scheduler=scheduler,
                                             add_watermarker=False)
    pipeline.save_pretrained(directory / "model", safe_serialization=True)
    controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(8, 16, 16, 32))
    controlnet.save_pretrained(directory / "controlnet", safe_serialization=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--families", nargs="+", choices=("sd15", "sdxl"), default=["sd15", "sdxl"])
    args = parser.parse_args()
    directory = ROOT / "build/deforum-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from PIL import Image
    image = Image.new("RGB", (64, 64), "black")
    for index in range(64):
        image.putpixel((index, index), (255, 128, 64))
    control = directory / "control.png"
    image.save(control)
    reports = []
    for family in args.families:
        fixture = directory / family
        if not (fixture / "model/model_index.json").is_file():
            create_fixture(fixture, family)
        for use_controlnet in (False, True):
            name = f"{family}{'-controlnet' if use_controlnet else ''}-{args.device}"
            output = directory / (name + ".mp4")
            command = [sys.executable, str(ROOT / "reference/generate.py"), "--backend", "deforum",
                       "--preset", "sd15-compatible" if family == "sd15" else "sdxl",
                       "--model", str(fixture / "model"), "--device", args.device,
                       "--dtype", "float32", "--cpu-threads", "2", "--width", "64", "--height", "64",
                       "--steps", "4", "--max-frames", "4", "--fps", "8", "--strength-schedule", "0:(0.5)",
                       "--animation-prompts", '{"0":"a city","2":"a forest"}',
                       "--zoom", "0:(1.02)", "--angle", "0:(1)", "--seed-behavior", "iter",
                       "--local-files-only", "--no-progress", "--overwrite", "--output", str(output)]
            if use_controlnet:
                command.extend(["--controlnet", str(fixture / "controlnet"), "--control-image", str(control)])
            with (directory / (name + ".log")).open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Smoke failed: {name}; inspect {directory / (name + '.log')}")
            report = json.loads(output.with_suffix(".json").read_text())
            assert report["status"] == "complete" and report["completed_frames"] == 4
            assert [frame["method"] for frame in report["frames"]] == ["text-to-image"] + ["image-to-image"] * 3
            assert all(frame["sampling"]["finite_latents"] for frame in report["frames"])
            assert report["output"]["verified_decode"]
            assert [frame["seed"] for frame in report["frames"]] == [42, 43, 44, 45]
            for previous, current in zip(report["frames"], report["frames"][1:]):
                assert current["source"]["pixel_sha256"] == previous["output"]["pixel_sha256"]
            reports.append({"name": name, "video": str(output), "sampling_steps": [
                frame["sampling"]["executed_steps"] for frame in report["frames"]]})
            print(json.dumps(reports[-1]), flush=True)
    (directory / f"verification-{args.device}.json").write_text(json.dumps({
        "fixture": "random small compatible weights; execution validation only", "runs": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
