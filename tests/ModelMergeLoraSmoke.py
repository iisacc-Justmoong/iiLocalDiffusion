#!/usr/bin/env python3
"""Offline PEFT oracle -> checkpoint/LoRA merge -> reload -> SDK image inference."""

import argparse
import copy
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
    args.merge_entry = args.merge_entry.absolute()
    args.generate_entry = args.generate_entry.absolute()
    import torch
    import peft
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.tuners.lora import LoraLayer
    from PIL import Image

    torch.set_num_threads(2)
    destination = ROOT / "build/reference/model-merge-lora-smoke"
    destination.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=destination))
    models = []
    for index, name in enumerate(("base", "checkpoint")):
        torch.manual_seed(31 + index)
        unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
                          down_block_types=("AttnDownBlock2D",), up_block_types=("AttnUpBlock2D",),
                          block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, attention_head_dim=4)
        DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10)).save_pretrained(run / name)
        models.append(unet)
    adapters = []
    for index, (rank, alpha) in enumerate(((2, 1), (3, 6))):
        torch.manual_seed(51 + index)
        adapted = get_peft_model(copy.deepcopy(models[0]), LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=["to_q", "to_v", "conv_in"], lora_dropout=0.0))
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(std=0.03)
        path = run / f"lora-{index}"
        adapted.save_pretrained(path)
        adapters.append(path)
    environment = {**os.environ, "IILD_PYTHON_EXECUTABLE": sys.executable,
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONPYCACHEPREFIX": str(ROOT / "build/pycache")}
    cases = []
    for mode in ("weighted-sum", "weighted-difference"):
        merged = run / mode
        completed = subprocess.run([sys.executable, str(args.merge_entry),
            "--base-model", str(run / "base"), "--additional-model", str(adapters[0]),
            "--additional-model", str(run / "checkpoint"), "--additional-model", str(adapters[1]),
            "--mode", mode, "--weights", "0.8", "0.25", "0.3", "--output", str(merged)],
            env=environment, capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
        loaded = DDPMPipeline.from_pretrained(merged, local_files_only=True)
        oracle_base = copy.deepcopy(models[0])
        base, checkpoint = models[0].state_dict(), models[1].state_dict()
        sign = 1 if mode == "weighted-sum" else -1
        oracle_base.load_state_dict({key: value * (0.75 if sign == 1 else 1) + sign * 0.25 * checkpoint[key]
                                     for key, value in base.items()})
        oracle = PeftModel.from_pretrained(oracle_base, adapters[0], adapter_name="first", local_files_only=True)
        oracle.load_adapter(adapters[1], adapter_name="second", local_files_only=True)
        oracle.base_model.set_adapter(["first", "second"])
        for module in oracle.modules():
            if isinstance(module, LoraLayer):
                module.set_scale("first", sign * 0.8)
                module.set_scale("second", sign * 0.3)
        oracle.eval()
        torch.manual_seed(70)
        sample = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            expected_forward = oracle(sample, 1).sample
            actual_forward = loaded.unet(sample, 1).sample
        torch.testing.assert_close(actual_forward, expected_forward, atol=2e-6, rtol=2e-5)
        fused = oracle.merge_and_unload(safe_merge=True, adapter_names=["first", "second"])
        for key, actual in loaded.unet.state_dict().items():
            torch.testing.assert_close(actual, fused.state_dict()[key], atol=1e-7, rtol=1e-6)
        generated = run / f"{mode}-images"
        subprocess.run([sys.executable, str(args.generate_entry), "--backend", "diffusers",
            "--model-path", str(merged), "--device", "cpu", "--steps", "2", "--seed", "42",
            "--output-dir", str(generated)], env=environment, capture_output=True, text=True, check=True)
        images = sorted(generated.rglob("*.png"))
        if len(images) != 1:
            raise AssertionError(f"Expected one generated image: {images}")
        with Image.open(images[0]) as image:
            image.load()
            if image.size != (8, 8):
                raise AssertionError(f"Unexpected output size: {image.size}")
        cases.append({"mode": mode, "peft_fused_weights_verified": len(fused.state_dict()),
                      "peft_unfused_forward_verified": True, "image": str(images[0]), "merge": report})
    report = {"schema": "iild-model-merge-lora-smoke-v1", "offline": True, "peft": peft.__version__,
              "inputs": "locally initialized tiny DDPM checkpoints and two nonzero LoRAs with ranks 2 and 3",
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
