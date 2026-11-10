"""Reproduce the public Klein reference workflow's image-area resize."""
from contextlib import contextmanager
import math

import control_hint_identity


def prepare(images, torch, np, Image):
    if len(images) != 2:
        raise ValueError('Klein reference resize requires control and reference images.')
    results, descriptions = [], []
    for image in images:
        if image.mode != 'RGB' or min(image.size) < 1:
            raise ValueError('Klein reference resize requires nonempty RGB images.')
        width, height = image.size
        factor = math.sqrt((1024 * 1024) / (width * height))
        output_width, output_height = round(width * factor), round(height * factor)
        if min(output_width, output_height) < 1 or max(output_width, output_height) > 8192:
            raise ValueError('Klein reference resize produced unsupported image dimensions.')
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        scaled = torch.nn.functional.interpolate(pixels, size=(output_height, output_width), mode='nearest-exact')
        result = Image.fromarray(scaled.squeeze(0).permute(1, 2, 0).contiguous().numpy(), mode='RGB')
        results.append(result)
        descriptions.append({'input_size': [width, height], 'output_size': [output_width, output_height],
                             'output_identity': control_hint_identity.fingerprint(result, np)})
    return results, {'mode': 'comfy-1mp', 'target_pixels': 1024 * 1024,
                     'resampling': 'nearest-exact', 'images': descriptions}


@contextmanager
def execution(pipeline):
    """Prevent Diffusers from rescaling a rounded result a second time."""
    processor = pipeline.image_processor
    if not callable(getattr(processor, '_resize_to_target_area', None)):
        raise ValueError('Installed Klein image processor cannot preserve the requested resize.')
    had_override = '_resize_to_target_area' in vars(processor)
    original = vars(processor).get('_resize_to_target_area')
    processor._resize_to_target_area = lambda image, target_area=1024 * 1024: image
    try:
        yield
    finally:
        if had_override:
            processor._resize_to_target_area = original
        else:
            del processor._resize_to_target_area


@contextmanager
def comfy_vae_execution(pipeline, torch):
    """Match the existing Flux.2 VAE's batch-normalization arithmetic."""
    vae = getattr(pipeline, 'vae', None)
    normalization = getattr(vae, 'bn', None)
    if (not callable(getattr(pipeline, '_encode_vae_image', None))
            or not callable(getattr(pipeline, '_patchify_latents', None))
            or getattr(normalization, 'running_mean', None) is None
            or getattr(normalization, 'running_var', None) is None
            or getattr(getattr(vae, 'config', None), 'batch_norm_eps', None) is None):
        raise ValueError('Installed Klein pipeline lacks the required Flux.2 VAE normalization.')
    had_override = '_encode_vae_image' in vars(pipeline)
    original = vars(pipeline).get('_encode_vae_image')

    def encode(image, generator):
        if image.ndim != 4:
            raise ValueError('Klein reference VAE expects a four-dimensional image tensor.')
        latent = vae.encode(image).latent_dist.mode()
        latent = pipeline._patchify_latents(latent)
        mean = normalization.running_mean.to(device=latent.device, dtype=latent.dtype)
        variance = normalization.running_var.to(device=latent.device, dtype=latent.dtype)
        return torch.nn.functional.batch_norm(latent, mean, variance, training=False,
                                              eps=vae.config.batch_norm_eps)

    pipeline._encode_vae_image = encode
    try:
        yield
    finally:
        if had_override:
            pipeline._encode_vae_image = original
        else:
            del pipeline._encode_vae_image
