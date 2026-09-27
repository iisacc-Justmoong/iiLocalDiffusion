#!/usr/bin/env python3
"""Offline real Diffusers architecture/CLI/output smoke with tiny random weights.

Exercises computation, not pretrained quality. Does not download any model.
Artifacts and the machine-readable result stay in build/.
"""
import argparse
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(ROOT / "reference/diffusers"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["sd3", "flux1", "flux2", "z-image", "qwen-image", "krea2"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/backend-runtime-smoke")
    args = parser.parse_args()
    import torch
    import diffusers as d
    import numpy as np
    from PIL import Image
    from safetensors.torch import save_file
    from backend_registry import inspect_pipeline_package
    from downloaded_model import inspect_downloaded_model
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("backend_smoke_router", ROOT / "reference/generate.py")
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    results_path = args.output_dir / "results.json"
    records = [record for record in json.loads(results_path.read_text()) if record["family"] not in args.families] if results_path.exists() else []
    for family in args.families:
        torch.manual_seed(42)
        folder = args.output_dir.resolve() / family
        model = folder / "model"
        model.mkdir(parents=True, exist_ok=True)
        inputs = {"prompt_embeds": torch.randn(1, 16, 32)}
        vae = d.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16,
            block_out_channels=(32, 32, 32, 32), layers_per_block=1,
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            norm_num_groups=32, scaling_factor=0.3611, shift_factor=0.1159)
        if family == "sd3":
            name = "StableDiffusion3Pipeline"
            transformer = d.SD3Transformer2DModel(sample_size=8, in_channels=16, out_channels=16,
                num_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=32, caption_projection_dim=32, pooled_projection_dim=16,
                pos_embed_max_size=16)
            inputs["pooled_prompt_embeds"] = torch.randn(1, 16)
        elif family == "flux1":
            name = "FluxPipeline"
            transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
                num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
                axes_dims_rope=(4, 6, 6))
            inputs["pooled_prompt_embeds"] = torch.randn(1, 16)
        elif family == "flux2":
            name = "Flux2KleinPipeline"
            transformer = d.Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
                num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=32, guidance_embeds=False, axes_dims_rope=(4, 4, 4, 4))
            vae = d.AutoencoderKLFlux2(in_channels=3, out_channels=3, latent_channels=32,
                block_out_channels=(32, 32, 32, 32), layers_per_block=1,
                down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                norm_num_groups=32)
        elif family == "z-image":
            name = "ZImagePipeline"
            transformer = d.ZImageTransformer2DModel(in_channels=16, dim=32, n_layers=1,
                n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=32,
                axes_dims=[4, 6, 6], axes_lens=[64, 64, 64])
            inputs["prompt_embeds"] = torch.randn(16, 32)
        elif family == "qwen-image":
            name = "QwenImagePipeline"
            transformer = d.QwenImageTransformer2DModel(in_channels=64, out_channels=16,
                num_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=32, axes_dims_rope=(4, 6, 6))
            vae = d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1)
            inputs["prompt_embeds_mask"] = torch.ones(1, 16, dtype=torch.int64)
        elif family == "krea2":
            name = "Krea2Pipeline"
            transformer = d.Krea2Transformer2DModel(in_channels=64, num_layers=1, attention_head_dim=16,
                num_attention_heads=2, num_key_value_heads=2, intermediate_size=64, timestep_embed_dim=32,
                text_hidden_dim=32, num_text_layers=1, text_num_attention_heads=2, text_num_key_value_heads=2,
                text_intermediate_size=64, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                axes_dims_rope=(4, 6, 6))
            vae = d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1)
            inputs["prompt_embeds"] = torch.randn(1, 16, 1, 32)
            inputs["prompt_embeds_mask"] = torch.ones(1, 16, dtype=torch.bool)
        else:
            raise ValueError(family)
        scheduler = d.FlowMatchEulerDiscreteScheduler()
        if family == "krea2":
            scheduler = d.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=.5,
                max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
        index = {"_class_name": name, "transformer": ["diffusers", type(transformer).__name__],
                 "vae": ["diffusers", type(vae).__name__], "scheduler": ["diffusers", type(scheduler).__name__]}
        for field in inspect.signature(getattr(d, name).__init__).parameters:
            if (field.startswith(("text_encoder", "tokenizer")) and field != "text_encoder_select_layers") or field == "processor":
                index[field] = [None, None]
        if family == "krea2":
            index["text_encoder_select_layers"] = [0]
            index["is_distilled"] = True
        if family == "flux2":
            index["is_distilled"] = True
        (model / "model_index.json").write_text(json.dumps(index))
        transformer.save_pretrained(model / "transformer", safe_serialization=True)
        vae.save_pretrained(model / "vae", safe_serialization=True)
        scheduler.save_pretrained(model / "scheduler")
        tensor_file = folder / "conditioning.safetensors"
        save_file(inputs, str(tensor_file))
        descriptors = {key: {"tensor_path": str(tensor_file), "key": key,
            "semantic": "attention-mask" if key.endswith("mask") else "embeddings",
            "layout": "SC" if family == "z-image" else {2: "BC", 3: "BSC", 4: "BSLC"}[value.ndim],
            "representation_space": "tiny-random-" + family} for key, value in inputs.items()}
        if family == "z-image":
            descriptors["prompt_embeds"] = [descriptors["prompt_embeds"]]
        if family == "qwen-image":
            descriptors["true_cfg_scale"] = 1.0
        inspected = inspect_pipeline_package(model)
        assert inspected["backend"] == "diffusers"
        checkpoint = inspect_downloaded_model(model / "transformer/diffusion_pytorch_model.safetensors")
        expected = {"flux1": "flux1-schnell", "flux2": "flux2-klein"}.get(family, family)
        assert checkpoint["architecture"] == expected, (family, checkpoint)
        del transformer, vae
        gc.collect()
        output = folder / "output"
        argv = ["--model", str(model), "--pipeline-inputs", json.dumps(descriptors),
            "--width", "64", "--height", "64", "--steps", "1", "--seed", "42",
            "--guidance-scale", "0" if family in ("krea2", "z-image") else "1", "--device", "cpu",
            "--dtype", "float32", "--no-default-modifiers", "--output-dir", str(output), "--overwrite"]
        assert router.main(argv) == 0, family
        report = json.loads((output / "generation.json").read_text())
        with Image.open(report["outputs"][0]["path"]) as image:
            pixels = np.asarray(image)
            assert image.size == (64, 64) and pixels.shape == (64, 64, 3) and float(pixels.std()) > 0
        records.append({"family": family, "pipeline": name, "tensor_architecture": checkpoint["architecture"], "status": "passed", "weights": "tiny-random",
                        "output": report["outputs"][0]})
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        print("SMOKE_PASS " + family, flush=True)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
