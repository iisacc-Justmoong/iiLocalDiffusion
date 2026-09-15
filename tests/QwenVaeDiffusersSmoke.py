#!/usr/bin/env python3
"""Actual pretrained Qwen VAE decoding through the public offline generic runner.

The tiny random transformer verifies wiring, not pretrained model image quality.
Run with the Diffusers virtual environment; outputs remain under build/.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(ROOT / "reference/diffusers"))


def main():
    import torch
    from diffusers import QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import save_file
    from vae_defaults import resolve_vae_selection
    import generate_any as generic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/qwen-vae/smoke")
    parser.add_argument("--launcher", type=Path, default=ROOT / "reference/generate.py")
    options = parser.parse_args()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = output / "model-without-vae"
    model.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(135)
    transformer = QwenImageTransformer2DModel(patch_size=2, in_channels=64, out_channels=16,
        num_layers=1, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6))
    transformer.save_pretrained(model / "transformer", safe_serialization=True)
    FlowMatchEulerDiscreteScheduler().save_pretrained(model / "scheduler")
    index = {"_class_name": "QwenImagePipeline", "_diffusers_version": "0.40.0",
             "transformer": ["diffusers", "QwenImageTransformer2DModel"],
             "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
             "vae": ["diffusers", "AutoencoderKLQwenImage"],
             "text_encoder": [None, None], "tokenizer": [None, None]}
    (model / "model_index.json").write_text(json.dumps(index, indent=2))
    tensors = output / "conditioning.safetensors"
    save_file({"embeds": torch.randn(1, 4, 16), "mask": torch.ones(1, 4, dtype=torch.int64)}, str(tensors))
    def descriptor(key, semantic, layout):
        return {"tensor_path": str(tensors), "key": key, "semantic": semantic, "layout": layout,
                "representation_space": "test-qwen-tiny-conditioning"}
    inputs = {"prompt_embeds": descriptor("embeds", "embeddings", "BSC"),
              "prompt_embeds_mask": descriptor("mask", "attention-mask", "BS"),
              "true_cfg_scale": 1.0}
    # Retain the same seed and exact conditioning for fallback/explicit parity.
    arguments = ["--backend", "diffusers", "--model", str(model), "--no-default-modifiers",
                 "--pipeline-inputs", json.dumps(inputs), "--seed", "42", "--width", "64", "--height", "64",
                 "--steps", "1", "--device", "cpu", "--dtype", "float32"]
    args = generic.resolve_arguments(generic.build_parser().parse_args(arguments[2:]))
    selection, status = resolve_vae_selection(args, index, model)
    assert status == "fallback" and not (model / "vae").exists()
    del transformer
    gc.collect()
    runs = []
    for name, extra in (("fallback", []), ("explicit", ["--vae", selection.directory])):
        destination = output / name
        command = [sys.executable, str(options.launcher), *arguments, "--output-dir", str(destination), "--overwrite", *extra]
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "IILD_PYTHON_EXECUTABLE": sys.executable,
                                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        report = json.loads((destination / "generation.json").read_text())
        assert report["request"]["vae"]["status"] == name
        assert report["request"]["vae"]["selection"]["weight"]["sha256"] == selection.weight.sha256
        image = report["outputs"][0]
        from PIL import Image
        import numpy as np
        with Image.open(image["path"]) as opened:
            pixels = np.asarray(opened)
            assert opened.size == (64, 64) and pixels.shape == (64, 64, 3)
            assert float(pixels.std()) > 0
        runs.append({"mode": name, "image": image["path"], "sha256": image["sha256"],
                     "pixel_std": float(pixels.std())})
    assert runs[0]["sha256"] == runs[1]["sha256"], "Omitted and explicit VAE produced different pixels"
    result = {"pretrained_vae_sha256": selection.weight.sha256, "vae_omitted": True,
              "model_contains_vae": False, "fallback_explicit_pixel_parity": True, "runs": runs,
              "scope": "Real pretrained VAE decode; tiny random transformer and synthetic conditioning, not image quality validation."}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
