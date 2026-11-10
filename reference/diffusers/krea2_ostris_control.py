"""Prepared-image control selection for AI Toolkit Krea 2 edit LoRAs."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

import krea2_ostris_conditioning
import krea2_ostris_lora
import krea2_ostris_reference
from weight_files import resolve_weight_file


@dataclass(frozen=True)
class Selection:
    weight: object
    image: str
    scale: float
    reference_attention: str = 'isolated'
    target_pipeline: str = 'Krea2Pipeline'
    image_argument: str = 'control_image'

    def metadata(self):
        return {'kind': 'krea2-ostris-edit', 'path': self.weight.path, 'sha256': self.weight.sha256,
                'size_bytes': self.weight.size_bytes, 'image': self.image, 'scale': self.scale,
                'reference_mode': 'qwen3-vl-and-vae-timestep-zero', 'reference_attention': self.reference_attention}


def resolve(args):
    if (args.controlnet_variant is not None or args.controlnet_config is not None
            or args.guess_mode is not None or args.control_mode is not None
            or args.control_color_image is not None or args.control_color_end is not None
            or args.control_depth_end is not None or getattr(args, 'krea2_control_auto_bucket', False)):
        raise ValueError('Krea 2 edit control does not support variant, color/depth, Union, or auto-bucket options.')
    if args.control_guidance_start not in (None, 0.) or args.control_guidance_end not in (None, 1.):
        raise ValueError('Krea 2 edit control requires the full sampling interval.')
    if args.offload != 'none' or args.seed is None:
        raise ValueError('Krea 2 edit control requires --offload none and --seed.')
    if not isinstance(args.inputs.get('prompt'), str) or any(key in args.inputs for key in (
            'prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask')):
        raise ValueError('Krea 2 edit control requires one string prompt and no precomputed embeddings.')
    if args.inputs.get('negative_prompt') is not None and not isinstance(args.inputs['negative_prompt'], str):
        raise ValueError('Krea 2 edit control requires a single string negative prompt.')
    if getattr(args, 'krea2_regions', None):
        raise ValueError('Krea 2 edit control cannot yet be combined with regional prompting.')
    if args.control_preprocessor_library and args.control_preprocess != 'canny':
        raise ValueError('--control-preprocessor-library requires --control-preprocess canny.')
    image = Path(args.control_image).expanduser().absolute() if args.control_image else None
    if image is None or not image.is_file() or not image.stat().st_size:
        raise ValueError('Krea 2 edit control requires a nonempty local --control-image.')
    scale = 1. if args.controlnet_scale is None else args.controlnet_scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Krea 2 edit control scale must be finite and nonnegative.')
    attention = getattr(args, 'krea2_edit_reference_attention', None) or 'isolated'
    return Selection(resolve_weight_file(args.controlnet, 'Krea 2 edit LoRA'), str(image), scale, attention)


@contextmanager
def execution(pipeline, image, inputs, selection, torch, np):
    maximum = inputs.get('max_sequence_length', 512)
    positive, positive_mask = krea2_ostris_conditioning.encode_prompt(
        pipeline, inputs['prompt'], image, torch, np, max_tokens=maximum)
    config = pipeline.transformer.config
    if (positive.ndim != 4 or tuple(positive.shape[2:]) != (config.num_text_layers, config.text_hidden_dim)
            or positive.shape[0] != 1 or positive_mask.dtype != torch.bool
            or tuple(positive_mask.shape) != tuple(positive.shape[:2]) or not bool(positive_mask.any())):
        raise ValueError('Krea 2 edit prompt embeddings do not match the transformer contract.')
    guidance = inputs['guidance_scale'] > 0
    negative = negative_mask = None
    if guidance:
        negative, negative_mask = krea2_ostris_conditioning.encode_prompt(
            pipeline, inputs.get('negative_prompt') or '', image, torch, np, max_tokens=maximum)
        if (negative.ndim != 4 or tuple(negative.shape[2:]) != tuple(positive.shape[2:]) or negative_mask.dtype != torch.bool
                or tuple(negative_mask.shape) != tuple(negative.shape[:2])
                or negative.shape[0] != 1 or not bool(negative_mask.any())
                or not bool(torch.isfinite(negative).all())):
            raise ValueError('Krea 2 edit negative prompt embeddings must be finite and have a valid mask.')
        negative = negative[:, negative_mask[0]]
        negative_mask = torch.ones(negative.shape[:2], dtype=torch.bool, device=negative.device)
    length = max(positive.shape[1], negative.shape[1] if guidance else 0)
    def pad(value, mask):
        size = length - value.shape[1]
        if size:
            value = torch.cat((value, value.new_zeros((value.shape[0], size, *value.shape[2:]))), dim=1)
            mask = torch.cat((mask, torch.zeros((mask.shape[0], size), dtype=torch.bool, device=mask.device)), dim=1)
        return value, mask
    positive, positive_mask = pad(positive, positive_mask)
    if guidance:
        negative, negative_mask = pad(negative, negative_mask)
    reference, positions = krea2_ostris_conditioning.reference_latents(pipeline, image, torch, np)
    old_inputs = {key: inputs[key] for key in ('prompt', 'negative_prompt') if key in inputs}
    try:
        inputs.pop('prompt', None)
        inputs.pop('negative_prompt', None)
        inputs['prompt_embeds'], inputs['prompt_embeds_mask'] = positive, positive_mask
        if guidance:
            inputs['negative_prompt_embeds'], inputs['negative_prompt_embeds_mask'] = negative, negative_mask
        with ExitStack() as stack:
            lora = stack.enter_context(krea2_ostris_lora.execution(pipeline.transformer, selection.weight, selection.scale, torch))
            refs = stack.enter_context(krea2_ostris_reference.execution(pipeline, reference, positions, guidance, torch,
                                                                         attention=selection.reference_attention))
            yield {'adapter': lora, 'reference': refs, 'positive_tokens': int(positive_mask.sum().item()),
                   'negative_tokens': int(negative_mask.sum().item()) if guidance else 0}
    finally:
        for key in ('prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask'):
            inputs.pop(key, None)
        inputs.update(old_inputs)
