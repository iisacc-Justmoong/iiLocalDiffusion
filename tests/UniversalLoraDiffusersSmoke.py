#!/usr/bin/env python3
"""Real offline SD1/SD2/SDXL/SD3/FLUX LoRA and Deforum execution on random tiny models.

This proves adapter effects and retention, not pretrained-model visual quality.
All files stay in build/universal-lora-smoke. No downloads or paid services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))


def fixture(directory, family):
    import torch
    import diffusers as d
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPTokenizer, PreTrainedTokenizerFast, T5Config, T5EncoderModel)
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    torch.manual_seed(120)
    directory.mkdir(parents=True)
    (directory / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab_file=str(directory / "vocab.json"),
                              merges_file=str(directory / "merges.txt"), model_max_length=77)

    def clip(width, projected=False):
        config = CLIPTextConfig(vocab_size=3, hidden_size=width, intermediate_size=32, num_hidden_layers=2,
                                num_attention_heads=width // 64 if width >= 64 else 4,
                                max_position_embeddings=77, projection_dim=width,
                                bos_token_id=0, eos_token_id=1, pad_token_id=1)
        return (CLIPTextModelWithProjection if projected else CLIPTextModel)(config)

    vae = d.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16 if family == "flux1" else 4,
                          down_block_types=("DownEncoderBlock2D",) * 4,
                          up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(32,) * 4,
                          layers_per_block=1, norm_num_groups=32, sample_size=64,
                          scaling_factor={"sdxl-base": 0.13025, "flux1": 0.3611}.get(family, 0.18215),
                          shift_factor=0.1159 if family == "flux1" else 0.0)
    if family in ("sd15", "sd2", "sdxl-base"):
        width = {"sd15": 768, "sd2": 1024, "sdxl-base": 2048}[family]
        unet_kwargs = dict(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                           block_out_channels=(32, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                           up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), norm_num_groups=32,
                           cross_attention_dim=width, attention_head_dim=4)
        if family == "sdxl-base":
            unet_kwargs.update(addition_embed_type="text_time", addition_time_embed_dim=256,
                               projection_class_embeddings_input_dim=2816, use_linear_projection=True)
        unet = d.UNet2DConditionModel(**unet_kwargs)
        scheduler = d.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                    beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
                                    steps_offset=1)
        if family == "sdxl-base":
            pipeline = d.StableDiffusionXLPipeline(vae=vae, unet=unet, scheduler=scheduler,
                text_encoder=clip(768), text_encoder_2=clip(1280, True),
                tokenizer=tokenizer, tokenizer_2=tokenizer, add_watermarker=False)
        else:
            pipeline = d.StableDiffusionPipeline(vae=vae, unet=unet, scheduler=scheduler,
                text_encoder=clip(width), tokenizer=tokenizer, safety_checker=None,
                feature_extractor=None, requires_safety_checker=False)
        component = "unet"
    elif family == "sd3":
        transformer = d.SD3Transformer2DModel(sample_size=8, patch_size=2, in_channels=4, out_channels=4,
            num_layers=1, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=64,
            caption_projection_dim=32, pooled_projection_dim=64, pos_embed_max_size=16)
        pipeline = d.StableDiffusion3Pipeline(transformer=transformer, vae=vae,
            scheduler=d.FlowMatchEulerDiscreteScheduler(), text_encoder=clip(32, True),
            text_encoder_2=clip(32, True), text_encoder_3=None,
            tokenizer=tokenizer, tokenizer_2=tokenizer, tokenizer_3=None)
        component = "transformer"
    else:
        words = ["[PAD]", "[EOS]", "[UNK]", "a"]
        fast = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        fast.pre_tokenizer = pre_tokenizers.Whitespace()
        t5_tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast, pad_token="[PAD]",
                                              eos_token="[EOS]", unk_token="[UNK]", model_max_length=512)
        encoder = T5EncoderModel(T5Config(vocab_size=4, d_model=4096, d_ff=32, num_layers=1,
                                          num_heads=2, d_kv=8, pad_token_id=0, eos_token_id=1))
        transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=4096, pooled_projection_dim=768, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6))
        pipeline = d.FluxPipeline(transformer=transformer, vae=vae,
            scheduler=d.FlowMatchEulerDiscreteScheduler(), text_encoder=clip(768),
            text_encoder_2=encoder, tokenizer=tokenizer, tokenizer_2=t5_tokenizer)
        component = "transformer"
    pipeline.save_pretrained(directory / "model", safe_serialization=True)
    model = getattr(pipeline, component)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    pipeline.text_encoder.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for target in (model, pipeline.text_encoder):
            for name, parameter in target.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(std=0.15)
    state = {component + "." + key: value for key, value in get_peft_model_state_dict(model).items()}
    state.update({"text_encoder." + key: value for key, value in get_peft_model_state_dict(pipeline.text_encoder).items()})
    adapter = directory / "adapter.safetensors"
    save_file(state, adapter)
    return {"file": str(adapter.relative_to(directory.parent)), "families": [family], "scale": 1.0,
            "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(), "size": adapter.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("sd15", "sd2", "sdxl-base", "sd3", "flux1"),
                        default=["sd15", "sd2", "sdxl-base", "sd3", "flux1"])
    parser.add_argument("--skip-deforum", action="store_true")
    options = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import numpy as np
    from PIL import Image
    import generate_any
    import generate
    from inference_session import InferenceSession
    torch.set_num_threads(2)
    base = ROOT / "build/universal-lora-smoke"
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    print(f"LoRA smoke artifacts: {directory}", flush=True)
    entries = [fixture(directory / family, family) for family in options.families]
    (directory / "generation-defaults.json").write_text(json.dumps({
        "version": 1, "fallback_loras": entries, "negative_embeddings": []}, indent=2))
    runs = []
    for family in options.families:
        images, reports = {}, {}
        for mode in ("base", "zero", "default", "repeat"):
            output = directory / family / mode
            tokens = ["--model", str(directory / family / "model"), "--generation-resources", str(directory),
                      "--device", "cpu", "--steps", "2", "--seed", "42", "--width", "64", "--height", "64",
                      "--prompt", "a", "--guidance-scale", "1" if family == "flux1" else "3",
                      "--output-dir", str(output)]
            if family in ("sd3", "flux1"):
                tokens.extend(["--pipeline-inputs", '{"max_sequence_length":77}'])
            if mode == "base":
                tokens.append("--no-default-modifiers")
            elif mode == "zero":
                tokens.extend(["--lora", str(directory / family / "adapter.safetensors"), "--lora-scale", "0"])
            # A retained session also verifies that a default adapter is not loaded twice.
            if mode == "base":
                session = InferenceSession()
                session.__enter__()
            assert generate_any.main(tokens) == 0
            report = json.loads((output / "generation.json").read_text())
            reports[mode] = report
            image_path = next(Path(item["path"]) for item in report["outputs"] if item["kind"] == "image")
            with Image.open(image_path) as image:
                images[mode] = np.array(image)
            if mode != "base":
                adapter = report["adapters"]["lora"]
                assert adapter["active_adapters"] == ["iild_lora"]
                assert set(adapter["registered_components"]) == {"text_encoder", "transformer" if family in ("flux1", "sd3") else "unet"}
        session.__exit__(None, None, None)
        assert np.array_equal(images["base"], images["zero"]), family + " zero strength changed base output"
        assert np.array_equal(images["default"], images["repeat"]), family + " repeat changed output"
        changed = int(np.count_nonzero(images["base"] != images["default"]))
        assert changed > 0, family + " LoRA did not affect output"
        if family in ("sd15", "sd2", "sdxl-base", "flux1"):
            assert reports["repeat"]["runtime"]["pipeline_cache_hit"]
        runs.append({"family": family, "changed_channels": changed, "zero_matches_base": True,
                     "repeat_matches": True, "default_selection": reports["default"]["request"]["generation_defaults"],
                     "adapter": reports["default"]["adapters"]["lora"]})
        print(f"Verified LoRA image effect: {family}, changed channels={changed}", flush=True)
        if not options.skip_deforum and family in ("sd15", "sdxl-base", "flux1"):
            videos = []
            for mode in ("zero", "default"):
                output = directory / family / f"deforum-{mode}.mp4"
                values = {"model": str(directory / family / "model"),
                          "preset": {"sd15": "sd15-compatible", "sdxl-base": "sdxl",
                                     "flux1": "flux1-schnell-compatible"}[family],
                          "generation_resources": str(directory), "device": "cpu", "dtype": "float32",
                          "offload": "none", "cpu_threads": 2, "steps": 4, "width": 64, "height": 64,
                          "prompt": "a", "guidance_scale": 0 if family == "flux1" else 3,
                          "seed": 42, "animation_mode": "2D", "max_frames": 3, "fps": 8,
                          "strength_schedule": "0:(0.5)", "progress": False, "output": str(output)}
                if family == "flux1":
                    values["max_sequence_length"] = 77
                if mode == "zero":
                    values.update(lora=str(directory / family / "adapter.safetensors"), lora_scale=0)
                preset, args = generate.resolve_request(values)
                if family == "flux1":
                    # The preset intentionally requires full-size FLUX encoders.
                    # Use the generic verified loader for this small fixture and
                    # exercise the unchanged production Deforum loop directly.
                    import diffusers
                    from deforum_runtime import run_deforum_animation
                    from deforum_video import preflight_animation
                    generic_args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([
                        "--model", values["model"], "--generation-resources", str(directory),
                        "--lora", str(directory / family / "adapter.safetensors"),
                        "--lora-scale", "0" if mode == "zero" else "1", "--prompt", "a"]))
                    pipeline, _, _ = generate_any.prepared_pipeline(generic_args, diffusers, torch.float32, "cpu")
                    args.text_embedding_activation = None
                    assert run_deforum_animation(pipeline, preset, args, torch, "cpu", torch.float32,
                        False, generic_args.lora_activation, build_call=generate.build_pipeline_call_arguments,
                        prepare_execution=generate.prepare_pipeline_for_execution,
                        environment=preflight_animation(args), metadata={"fixture": "tiny FLUX via generic loader"}) == 0
                else:
                    assert generate.run(preset, args) == 0
                report = json.loads(output.with_suffix(".json").read_text())
                assert report["output"]["verified_decode"] and report["completed_frames"] == 3
                assert [f["method"] for f in report["frames"]] == ["text-to-image", "image-to-image", "image-to-image"]
                assert all(f["lora"]["applied"] and f["sampling"]["finite_latents"] for f in report["frames"])
                assert all(f["lora"]["active_adapters"] == ["iild_lora"] for f in report["frames"])
                videos.append(report)
            assert all(a["output"]["pixel_sha256"] != b["output"]["pixel_sha256"]
                       for a, b in zip(videos[0]["frames"], videos[1]["frames"]))
            runs[-1]["deforum"] = {"frames": 3, "all_frames_affected": True,
                                    "verified_decode": True, "video": videos[1]["output"]["path"],
                                    "entry": "generic loader + production Deforum loop" if family == "flux1" else "preset runner"}
            print(f"Verified LoRA on every Deforum frame: {family}", flush=True)
    report = directory / "verification.json"
    report.write_text(json.dumps({"scope": "Real local random-weight inference; no trained-model quality claim",
                                  "runs": runs}, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
