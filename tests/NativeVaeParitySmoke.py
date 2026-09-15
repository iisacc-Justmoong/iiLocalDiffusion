#!/usr/bin/env python3
"""Offline native SDXL decoder parity using the checkpoint's actual embedded VAE.

Requires the existing reference/diffusers environment; never downloads weights.
The deterministic latent fixture tests numerical decoding, not prompt quality.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 32, 48, 64])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from PIL import Image

    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).resolve().parents[1] / "resources/vae/sdxl/config.json").read_text())
    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:
        state = {k: checkpoint.get_tensor(k).float() for k in checkpoint.keys()
                 if k.startswith(("first_stage_model.", "vae."))}
    assert state and all(torch.isfinite(v).all() for v in state.values()), "Invalid embedded VAE weights"
    vae = AutoencoderKL.from_config(config).eval()
    vae.load_state_dict(convert_ldm_vae_checkpoint(state, config), strict=True)
    generator = torch.Generator().manual_seed(20260914)
    latents = torch.randn(1, 4, args.height // 8, args.width // 8, generator=generator) * 0.5
    latents.numpy().tofile(output / "latents.f32")
    with torch.inference_mode():
        expected = (vae.decode(latents / config["scaling_factor"]).sample * 0.5 + 0.5).clamp(0, 1).numpy()[0]
    assert np.isfinite(expected).all()
    expected.tofile(output / "reference.f32")

    def png(values, path):
        Image.fromarray((values.transpose(1, 2, 0).clip(0, 1) * 255).round().astype("uint8")).save(path)

    png(expected, output / "reference.png")
    results = []
    environment = dict(os.environ)
    for name in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        environment.pop(name, None)
    for tile in args.tiles:
        destination = output / f"tile-{tile}.f32"
        with (output / f"tile-{tile}.log").open("w") as log:
            subprocess.run([str(args.probe.resolve()), str(args.model.resolve()), str(output / "latents.f32"),
                            str(args.width), str(args.height), str(tile), str(destination)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        actual = np.fromfile(destination, dtype="float32").reshape(expected.shape)
        assert np.isfinite(actual).all()
        error = np.abs(actual - expected)
        result = {"tile": tile, "mae": float(error.mean()), "max_error": float(error.max()),
                  "rmse": float(np.sqrt((error ** 2).mean()))}
        results.append(result)
        png(actual, output / f"tile-{tile}.png")
        print(json.dumps(result), flush=True)
        # No spatial truncation: fp16 ggml operations must stay close to float32.
        if tile == 0 or tile >= max(args.width, args.height) // 8:
            assert result["mae"] < 0.01, "Native decoder differs from independent float32 reference"
    by_tile = {result["tile"]: result for result in results}
    if args.width == args.height == 512 and 32 in by_tile and 40 in by_tile:
        assert by_tile[40]["mae"] < by_tile[32]["mae"] * 0.75, "Larger mobile tiles lost their context advantage"
    (output / "verification.json").write_text(json.dumps({"model": str(args.model),
        "width": args.width, "height": args.height, "vae_tensors": len(state), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
