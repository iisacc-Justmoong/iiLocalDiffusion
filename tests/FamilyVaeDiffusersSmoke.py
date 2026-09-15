#!/usr/bin/env python3
"""Offline family fallback wiring with real pretrained VAEs and tiny random denoisers.

Checks public CLI omission/explicit parity and the preset composition boundary.
This does not evaluate the image quality of a complete pretrained denoiser.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(ROOT / "reference/diffusers"))


def fixture(directory, family):
    import torch
    import diffusers as d
    from safetensors.torch import save_file
    torch.manual_seed(135)
    model = directory / "model-without-vae"
    model.mkdir(parents=True, exist_ok=True)
    inputs = {"prompt_embeds": torch.randn(1, 4, 32)}
    index = {"text_encoder": [None, None], "tokenizer": [None, None]}
    if family == "sdxl-base":
        denoiser = d.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            layers_per_block=1, block_out_channels=(32, 32),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), norm_num_groups=32,
            cross_attention_dim=32, attention_head_dim=4, addition_embed_type="text_time",
            addition_time_embed_dim=8, projection_class_embeddings_input_dim=64)
        scheduler = d.DDIMScheduler(clip_sample=False)
        index.update(_class_name="StableDiffusionXLPipeline", unet=["diffusers", "UNet2DConditionModel"],
            text_encoder_2=[None, None], tokenizer_2=[None, None], force_zeros_for_empty_prompt=True)
        component = "unet"
        inputs["pooled_prompt_embeds"] = torch.randn(1, 16)
    elif family == "flux1":
        denoiser = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6))
        scheduler = d.FlowMatchEulerDiscreteScheduler()
        index.update(_class_name="FluxPipeline", transformer=["diffusers", "FluxTransformer2DModel"],
            text_encoder_2=[None, None], tokenizer_2=[None, None])
        component = "transformer"
        inputs["pooled_prompt_embeds"] = torch.randn(1, 16)
    else:
        denoiser = d.Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, guidance_embeds=False, axes_dims_rope=(4, 4, 4, 4))
        scheduler = d.FlowMatchEulerDiscreteScheduler()
        index.update(_class_name="Flux2KleinPipeline", transformer=["diffusers", "Flux2Transformer2DModel"],
                     is_distilled=True)
        component = "transformer"
    # Omit the VAE component declaration and directory entirely.
    index["scheduler"] = ["diffusers", type(scheduler).__name__]
    (model / "model_index.json").write_text(json.dumps(index, indent=2))
    denoiser.save_pretrained(model / component, safe_serialization=True)
    scheduler.save_pretrained(model / "scheduler")
    tensor_file = directory / "conditioning.safetensors"
    save_file(inputs, str(tensor_file))
    descriptors = {name: {"tensor_path": str(tensor_file), "key": name, "semantic": "embeddings",
        "layout": "BSC" if value.ndim == 3 else "BC", "representation_space": "tiny-family-test"}
        for name, value in inputs.items()}
    return model, index, inputs, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/family-vae/smoke")
    parser.add_argument("--launcher", type=Path, default=ROOT / "reference/generate.py")
    parser.add_argument("--families", nargs="+", default=["sdxl-base", "flux1", "flux2"])
    options = parser.parse_args()
    import torch
    import diffusers as d
    import numpy as np
    from PIL import Image
    from vae_defaults import resolve_vae_selection, resolve_preset_vae
    from presets import PRESETS, ModelSelection
    from model_loading import load_generation_pipeline
    torch.set_num_threads(4)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for family in options.families:
        directory = output / family
        model, index, tensors, inputs = fixture(directory, family)
        args = SimpleNamespace(vae=None, pipeline_class=None, model=str(model), source_kind="directory", generation_resources=None)
        selection, status = resolve_vae_selection(args, index, model)
        assert status == "fallback" and not (model / "vae").exists()
        runs = []
        for name, extra in (("fallback", []), ("explicit", ["--vae", selection.directory])):
            destination = directory / name
            command = [sys.executable, str(options.launcher), "--backend", "diffusers", "--model", str(model),
                "--no-default-modifiers", "--pipeline-inputs", json.dumps(inputs), "--seed", "42",
                "--width", "64", "--height", "64", "--steps", "1", "--device", "cpu", "--dtype", "float32",
                "--guidance-scale", "1", "--output-dir", str(destination), "--overwrite", *extra]
            with (directory / (name + ".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "IILD_PYTHON_EXECUTABLE": sys.executable,
                         "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            report = json.loads((destination / "generation.json").read_text())
            assert report["request"]["vae"]["status"] == name
            assert report["request"]["vae"]["selection"]["weight"]["sha256"] == selection.weight.sha256
            image = report["outputs"][0]
            with Image.open(image["path"]) as opened:
                pixels = np.asarray(opened)
                assert opened.size == (64, 64) and pixels.shape == (64, 64, 3) and float(pixels.std()) > 0
            runs.append({"mode": name, "image": image["path"], "sha256": image["sha256"]})
        assert runs[0]["sha256"] == runs[1]["sha256"], family + " fallback/explicit pixels differ"
        preset_checked = False
        if family in ("sdxl-base", "flux1"):
            preset = PRESETS["sdxl" if family == "sdxl-base" else "flux1-schnell-compatible"]
            request = SimpleNamespace(vae_file=None, model_selection=ModelSelection(str(model), None, True),
                                      config_selection=None, generation_resources=None)
            preset_vae, preset_status = resolve_preset_vae(preset, request)
            assert preset_vae == selection and preset_status == "fallback"
            pipeline, metadata = load_generation_pipeline(getattr(d, index["_class_name"]), preset,
                request.model_selection, None, None,
                {"dtype": torch.float32, "cache_dir": directory / "cache",
                 **({"add_watermarker": False} if family == "sdxl-base" else {}),
                 **{name: None for name, value in index.items() if value == [None, None]}},
                {"AutoencoderKL": d.AutoencoderKL}, vae_selection=preset_vae, vae_status=preset_status)
            assert metadata["component_sources"]["vae"] == "vae_fallback"
            pipeline.set_progress_bar_config(disable=True)
            image = pipeline(**tensors, width=64, height=64, num_inference_steps=1, guidance_scale=1,
                             generator=torch.Generator().manual_seed(42)).images[0]
            with Image.open(runs[0]["image"]) as expected:
                assert np.array_equal(np.asarray(image), np.asarray(expected)), family + " preset composition differs"
            preset_checked = True
            del pipeline
            gc.collect()
        results.append({"family": family, "vae_sha256": selection.weight.sha256,
            "model_contains_vae": False, "fallback_explicit_pixel_parity": True,
            "preset_composition_parity": preset_checked, "runs": runs})
        print(f"Verified {family}: actual pretrained VAE, CLI fallback/explicit parity, preset={preset_checked}", flush=True)
    (output / "verification.json").write_text(json.dumps({"results": results,
        "scope": "Actual pretrained VAE decoding with tiny random denoisers and synthetic conditioning; not pretrained image quality."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
