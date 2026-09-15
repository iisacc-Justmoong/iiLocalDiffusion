#!/usr/bin/env python3
"""Offline real CLIP/UNet smoke: every bundled vector reaches the denoiser.

Uses the actual bundled learned vectors with small randomly initialized models;
this checks conditioning and execution, not image quality or a trained base.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))


def main():
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    import generate
    from text_embeddings import apply_text_embeddings, text_embedding_prompt_context

    torch.set_num_threads(4)
    torch.manual_seed(12)
    tokenizer = CLIPTokenizer.from_pretrained(ROOT / "reference/diffusers/configs/sdxl/tokenizer", local_files_only=True)
    tokenizer2 = CLIPTokenizer.from_pretrained(ROOT / "reference/diffusers/configs/sdxl/tokenizer_2", local_files_only=True)
    def config(width, heads, projection=None):
        return CLIPTextConfig(vocab_size=len(tokenizer), hidden_size=width, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=heads, max_position_embeddings=77,
                              projection_dim=projection or width, bos_token_id=49406, eos_token_id=49407)
    encoder = CLIPTextModel(config(768, 12))
    encoder2 = CLIPTextModelWithProjection(config(1280, 20))
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                               block_out_channels=(32, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                               up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=2048,
                               attention_head_dim=(4, 4), addition_embed_type="text_time",
                               addition_time_embed_dim=8, projection_class_embeddings_input_dim=1328)
    vae = AutoencoderKL(block_out_channels=(32,), in_channels=3, out_channels=3, latent_channels=4)
    pipeline = StableDiffusionXLPipeline(vae=vae, text_encoder=encoder, text_encoder_2=encoder2,
                                        tokenizer=tokenizer, tokenizer_2=tokenizer2, unet=unet,
                                        scheduler=EulerDiscreteScheduler())
    preset, args = generate.resolve_request({"preset": "sdxl-base", "model": str(ROOT / "tests/fixtures/sdxl-base-manifest"),
                                             "prompt": "a white cockatoo", "device": "cpu"})
    args.text_embedding_activation = apply_text_embeddings(pipeline, preset, args, torch)
    seen = {"text_encoder": set(), "text_encoder_2": set()}
    hooks = []
    for name in seen:
        hooks.append(getattr(pipeline, name).register_forward_pre_hook(
            lambda module, inputs, key=name: seen[key].update(inputs[0].flatten().tolist())))
    def encode():
        with text_embedding_prompt_context(pipeline, preset, args) as request:
            return pipeline.encode_prompt(prompt=request.prompt, prompt_2=request.prompt_2,
                                          negative_prompt=request.negative_prompt, negative_prompt_2=request.negative_prompt_2,
                                          device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=True)
    with torch.inference_mode():
        positive, negative, pooled, negative_pooled = encode()
        assert positive.shape == negative.shape == (1, 154, 2048), positive.shape
        for registration in args.text_embedding_activation.registrations:
            assert set(registration["token_ids"]).issubset(seen[registration["component"]]), registration["token"]
        latents = torch.randn((1, 4, 8, 8))
        time_ids = torch.tensor([[64, 64, 0, 0, 64, 64]], dtype=torch.float32)
        def denoise(condition, pool):
            return unet(latents, 1, encoder_hidden_states=condition,
                        added_cond_kwargs={"text_embeds": pool, "time_ids": time_ids}).sample
        before = denoise(negative, negative_pooled)
        last = next(item for item in args.text_embedding_activation.registrations
                    if item["token"] == "iild_negative_realisticvision" and item["component"] == "text_encoder")
        encoder.get_input_embeddings().weight[last["token_ids"][-1]].add_(0.25)
        _, changed_negative, _, changed_pool = encode()
        after = denoise(changed_negative, changed_pool)
        assert not torch.equal(negative, changed_negative), "A late learned vector was truncated"
        assert not torch.equal(before, after), "Late learned vectors did not reach the UNet"
        assert torch.isfinite(after).all()
    for hook in hooks:
        hook.remove()
    report = {"scope": "real bundled vectors, small random CLIP/UNet; not a quality test",
              "embeddings": len(args.text_embedding_selections), "learned_vectors": 128,
              "sequence_length": negative.shape[1], "all_registered_tokens_encoded": True,
              "last_vector_changes_conditioning_and_denoiser": True,
              "negative_shape": list(negative.shape), "maximum_denoiser_difference": float((before-after).abs().max())}
    output = ROOT / "build/default-modifiers-diffusers.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
