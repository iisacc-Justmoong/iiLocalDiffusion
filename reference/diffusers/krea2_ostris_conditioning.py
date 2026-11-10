"""Qwen3-VL image conditioning for AI Toolkit Krea 2 edit adapters."""
import math


def reference_latents(pipeline, image, torch, np):
    """Encode one reference at up to 1 MP and assign the first reference RoPE index."""
    pixels = np.array(image.convert('RGB'), copy=True)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError('Krea 2 reference conditioning needs a nonempty RGB image.')
    height, width = pixels.shape[:2]
    value = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div(255.)
    scale = min(1., math.sqrt((1024 * 1024) / (height * width)))
    out_h = max(round(height * scale / 16) * 16, 16)
    out_w = max(round(width * scale / 16) * 16, 16)
    if (out_h, out_w) != (height, width):
        value = torch.nn.functional.interpolate(value, size=(out_h, out_w), mode='area')
    parameter = next(pipeline.vae.parameters())
    value = value.mul(2.).sub(1.).unsqueeze(2).to(device=parameter.device, dtype=parameter.dtype)
    latent = pipeline.vae.encode(value).latent_dist.mode()
    mean = torch.as_tensor(pipeline.vae.config.latents_mean, dtype=torch.float32, device=latent.device).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(pipeline.vae.config.latents_std, dtype=torch.float32, device=latent.device).view(1, -1, 1, 1, 1)
    latent = ((latent.float() - mean) / std).squeeze(2)
    if not bool(torch.isfinite(latent).all()):
        raise ValueError('Krea 2 reference VAE produced non-finite latents.')
    batch, channels, latent_h, latent_w = latent.shape
    packed = pipeline._pack_latents(latent, batch, channels, latent_h, latent_w)
    h, w = latent_h // 2, latent_w // 2
    positions = torch.zeros((h, w, 3), dtype=torch.float32, device=latent.device)
    positions[..., 0] = 1
    positions[..., 1] = torch.arange(h, device=latent.device)[:, None]
    positions[..., 2] = torch.arange(w, device=latent.device)[None, :]
    return packed, positions.reshape(-1, 3)


def vision_patches(image, torch, np):
    """Prepare one RGB image using the Ostris/Comfy Qwen3-VL image contract."""
    pixels = np.array(image.convert('RGB'), copy=True)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError('Krea 2 visual conditioning needs a nonempty RGB image.')
    height, width = pixels.shape[:2]
    value = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div(255.)
    scale = min(1., math.sqrt((384 * 384) / (height * width)))
    target_h, target_w = max(round(height * scale), 1), max(round(width * scale), 1)
    if (target_h, target_w) != (height, width):
        value = torch.nn.functional.interpolate(value, size=(target_h, target_w), mode='area')
    factor = 32  # 16-pixel patches and a 2x2 spatial merger.
    out_h, out_w = round(target_h / factor) * factor, round(target_w / factor) * factor
    if out_h * out_w > 12845056:
        shrink = math.sqrt(target_h * target_w / 12845056)
        out_h = max(factor, math.floor(target_h / shrink / factor) * factor)
        out_w = max(factor, math.floor(target_w / shrink / factor) * factor)
    elif out_h * out_w < 3136:
        grow = math.sqrt(3136 / (target_h * target_w))
        out_h = math.ceil(target_h * grow / factor) * factor
        out_w = math.ceil(target_w * grow / factor) * factor
    value = torch.nn.functional.interpolate(value, size=(out_h, out_w), mode='bilinear', align_corners=False)
    value = value.squeeze(0).mul(2.).sub(1.)
    grid_h, grid_w = out_h // 16, out_w // 16
    # Each image is repeated over Qwen's two temporal slots; spatial patches
    # are ordered by merger cell, then by the 2x2 patches within that cell.
    value = value.unsqueeze(0).repeat(2, 1, 1, 1)
    value = value.reshape(1, 2, 3, grid_h // 2, 2, 16, grid_w // 2, 2, 16)
    patches = value.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_h * grid_w, 3 * 2 * 16 * 16)
    return patches, torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)


def encode_prompt(pipeline, prompt, image, torch, np, *, max_tokens=8192):
    """Produce Krea 2 tapped text states conditioned on a single visual image."""
    if not isinstance(prompt, str) or type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ValueError('Krea 2 visual conditioning requires a string prompt and bounded token count.')
    model, tokenizer = pipeline.text_encoder, pipeline.tokenizer
    if model is None or not hasattr(model, 'visual') or tokenizer is None:
        raise ValueError('Krea 2 visual conditioning requires the full Qwen3-VL model and tokenizer.')
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if image_id != model.config.image_token_id:
        raise ValueError('Krea 2 tokenizer and Qwen3-VL image token IDs differ.')
    prefix, suffix = pipeline.prompt_template_encode_prefix, pipeline.prompt_template_encode_suffix
    prefix_length = pipeline.prompt_template_encode_start_idx
    if len(tokenizer(prefix, add_special_tokens=False)['input_ids']) != prefix_length:
        raise ValueError('Krea 2 tokenizer does not match the expected text prefix.')
    text = prefix + 'Picture 1: <|vision_start|><|image_pad|><|vision_end|>' + prompt + suffix
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if token_ids.count(image_id) != 1:
        raise ValueError('Krea 2 prompt contains an ambiguous image placeholder.')
    patches, grid = vision_patches(image, torch, np)
    image_tokens = int(grid.prod().item()) // 4
    position = token_ids.index(image_id)
    token_ids[position:position + 1] = [image_id] * image_tokens
    if len(token_ids) - prefix_length > max_tokens:
        raise ValueError('Krea 2 visual prompt exceeds its token limit.')
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    outputs = model(input_ids=ids, attention_mask=mask, pixel_values=patches.to(device),
                    image_grid_thw=grid.to(device), mm_token_type_ids=(ids == image_id).long(),
                    output_hidden_states=True, use_cache=False)
    hidden = torch.stack([outputs.hidden_states[index] for index in pipeline.text_encoder_select_layers], dim=2)
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError('Krea 2 visual prompt encoder produced non-finite states.')
    return hidden[:, prefix_length:], mask[:, prefix_length:]
