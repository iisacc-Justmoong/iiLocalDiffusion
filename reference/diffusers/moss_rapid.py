"""Local rapid-moss data-space sampling and unmerged FP32 adapter support."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

from weight_files import resolve_weight_file, verify_weight_file

TARGETS = ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down')
FILES = ('lora.safetensors', 'time_embedding.safetensors', 'dataspace.safetensors')
TIME_KEYS = ('time_embedding.0.weight', 'time_embedding.0.bias',
             'time_embedding.2.weight', 'time_embedding.2.bias',
             'shared_block_modulation.bias', 'video_head_modulation.bias')


@dataclass(frozen=True)
class Selection:
    weights: tuple

    def metadata(self):
        return {'kind': 'moss-data-space', 'files': [
            {'path': w.path, 'sha256': w.sha256, 'size_bytes': w.size_bytes} for w in self.weights]}


def resolve(args, pipeline_name):
    if args.moss_rapid_adapter is None:
        return None
    if pipeline_name != 'MOSSImagePipeline':
        raise ValueError('The MOSS few-step adapter requires MOSSImagePipeline.')
    if args.offload != 'none' or args.seed is None or args.dtype not in ('float32', 'bfloat16'):
        raise ValueError('MOSS few-step sampling requires --offload none, a seed and float32 or bfloat16.')
    if args.preview_dir:
        raise ValueError('MOSS few-step sampling does not support step previews.')
    values = args.inputs
    if values.get('negative_prompt') not in (None, '') or any(
            key in values for key in ('negative_prompt_embeds', 'negative_prompt_attention_mask',
                                      'latents', 'slg_layers', 'slg_scale', 'slg_start', 'slg_stop', 'sigmas')):
        raise ValueError('MOSS few-step sampling does not accept negative prompts, supplied latents or skip guidance.')
    if values.get('uncond_skip_layers') not in (None, [], ()):
        raise ValueError('MOSS few-step sampling does not support unconditional skip layers.')
    values.setdefault('num_inference_steps', 4)
    values.setdefault('guidance_scale', 1.0)
    if type(values['num_inference_steps']) is not int or not 1 <= values['num_inference_steps'] <= 4:
        raise ValueError('MOSS few-step sampling requires 1 to 4 steps.')
    if values['guidance_scale'] != 1.0:
        raise ValueError('MOSS few-step sampling requires guidance scale 1.')
    folder = Path(args.moss_rapid_adapter).expanduser().absolute()
    if not folder.is_dir():
        raise ValueError('MOSS few-step adapter must be a local directory.')
    return Selection(tuple(resolve_weight_file(str(folder / name), 'MOSS few-step adapter') for name in FILES))


def plan_weights(lora, time, data, modules, time_shapes, channels, layers):
    targets = [f'layers.{i}.proj.{name}' for i in range(layers) for name in TARGETS]
    if not targets or set(time) != set(TIME_KEYS) or any(tuple(time[k]) != tuple(time_shapes.get(k, ())) for k in TIME_KEYS):
        raise ValueError('MOSS few-step time path does not match the transformer.')
    if set(data) != {'latent_mix.Q', 'step_codes'} or tuple(data['latent_mix.Q']) != (channels, channels) or tuple(data['step_codes']) != (4,):
        raise ValueError('MOSS few-step data-space tensors have incompatible shapes.')
    expected = set()
    for target in targets:
        key = 'diffusion_model.' + target
        down, up, alpha = key + '.lora_down.weight', key + '.lora_up.weight', key + '.alpha'
        expected.update((down, up, alpha))
        shape, a, b = modules.get(target), lora.get(down), lora.get(up)
        if (shape is None or len(shape) != 2 or a is None or len(a) != 2 or a[0] < 1
                or tuple(a) != (a[0], shape[1]) or tuple(b or ()) != (shape[0], a[0])
                or tuple(lora.get(alpha, (None,))) not in ((), (1,))):
            raise ValueError(f'MOSS few-step LoRA tensors do not match {target}.')
    if set(lora) != expected:
        raise ValueError('MOSS few-step LoRA has missing or unexpected tensors.')
    return targets


@contextmanager
def execution(pipeline, selection, torch):
    import copy
    from safetensors import safe_open
    transformer = pipeline.transformer
    modules = dict(transformer.named_modules())
    module_shapes = {n: tuple(m.weight.shape) for n, m in modules.items()
                     if getattr(m, 'weight', None) is not None and m.weight.ndim == 2}
    time_shapes = {k: tuple(v.shape) for k, v in transformer.state_dict().items() if k in TIME_KEYS}
    for weight in selection.weights:
        verify_weight_file(weight, 'MOSS few-step adapter')
    device = transformer.video_embedder.weight.device
    dtype = transformer.video_embedder.weight.dtype
    if device.type == 'meta' or dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('MOSS few-step sampling requires resident FP32 or BF16 transformer weights.')
    def read(source, name):
        value = source.get_tensor(name)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f'MOSS few-step tensor must be finite floating point: {name}')
        return value.to(device=device, dtype=torch.float32)
    with safe_open(selection.weights[0].path, framework='pt', device='cpu') as lora, \
         safe_open(selection.weights[1].path, framework='pt', device='cpu') as time, \
         safe_open(selection.weights[2].path, framework='pt', device='cpu') as data:
        shapes = lambda source: {n: tuple(source.get_slice(n).get_shape()) for n in source.keys()}
        targets = plan_weights(shapes(lora), shapes(time), shapes(data), module_shapes, time_shapes,
                               transformer.config.latent_channels, len(transformer.layers))
        pairs = []
        for target in targets:
            key = 'diffusion_model.' + target
            a, b, alpha = (read(lora, key + suffix) for suffix in
                           ('.lora_down.weight', '.lora_up.weight', '.alpha'))
            scale = alpha.item() / a.shape[0]
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('MOSS few-step LoRA alpha must be finite and positive.')
            pairs.append((modules[target], a, b, scale))
        replacements = {name: copy.deepcopy(getattr(transformer, name)).float() for name in
                        ('time_embedding', 'shared_block_modulation', 'video_head_modulation')}
        for key in TIME_KEYS:
            root, member = key.split('.', 1)
            replacements[root].state_dict()[member].copy_(read(time, key))
        q, codes = read(data, 'latent_mix.Q'), read(data, 'step_codes')
    for weight in selection.weights:
        verify_weight_file(weight, 'MOSS few-step adapter')
    old = {name: getattr(transformer, name) for name in replacements}
    handles = []
    try:
        for name, value in replacements.items():
            setattr(transformer, name, value)
        for module, a, b, scale in pairs:
            def add(module, args, output, a=a, b=b, scale=scale):
                delta = torch.nn.functional.linear(torch.nn.functional.linear(args[0].float(), a), b)
                return output + (scale * delta).to(output.dtype)
            handles.append(module.register_forward_hook(add))
        yield {'Q': q, 'codes': codes}
    finally:
        for handle in reversed(handles):
            handle.remove()
        for name, value in old.items():
            setattr(transformer, name, value)


def generate(pipeline, inputs, adapter, control_trace, torch):
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
    width, height = inputs.get('width', 1024), inputs.get('height', 1024)
    if any(type(x) is not int or not 16 <= x <= 8192 or x % 16 for x in (width, height)):
        raise ValueError('MOSS few-step dimensions must be multiples of 16 in [16,8192].')
    prompt, embeddings = inputs.get('prompt'), inputs.get('prompt_embeds')
    if (prompt is None) == (embeddings is None):
        raise ValueError('Provide exactly one prompt or prompt_embeds.')
    maximum = inputs.get('max_sequence_length', 4096)
    if type(maximum) is not int or not 1 <= maximum <= 4096:
        raise ValueError('max_sequence_length must be between 1 and 4096.')
    if embeddings is None:
        if inputs.get('prompt_attention_mask') is not None:
            raise ValueError('prompt_attention_mask requires prompt_embeds.')
        embeddings, mask = pipeline.encode_prompt(prompt, maximum)
    else:
        mask = inputs.get('prompt_attention_mask')
    device = pipeline._execution_device
    if embeddings.ndim != 3 or embeddings.shape[0] < 1:
        raise ValueError('MOSS prompt embeddings must be a nonempty batch of sequences.')
    if mask is None:
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
    if mask.shape != embeddings.shape[:2]:
        raise ValueError('MOSS prompt mask must match prompt embeddings.')
    count = inputs.get('num_images_per_prompt', 1)
    if type(count) is not int or count < 1:
        raise ValueError('num_images_per_prompt must be a positive integer.')
    embeddings = embeddings.to(device).repeat_interleave(count, 0)
    mask = mask.to(device).repeat_interleave(count, 0)
    shape = (embeddings.shape[0], pipeline.transformer.config.latent_channels, height // 16, width // 16)
    noise = torch.randn(shape, generator=inputs['generator'], device='cpu', dtype=torch.float32).to(device)
    output = torch.zeros_like(noise)
    steps = inputs['num_inference_steps']
    progress = control_trace['completed_steps'] if control_trace else None
    if progress is not None:
        progress['total'] = steps
    for index in range(steps):
        if progress is not None:
            progress['step'] = index
        value = noise + torch.einsum('oc,bchw->bohw', adapter['Q'], output)
        region = getattr(pipeline, '_iild_regions', None)
        region_masks = region.masks(index, steps) if region is not None else None
        region_kwargs = {'region_attention': region_masks} if region_masks is not None else {}
        output = noise - pipeline.transformer(value, adapter['codes'][index], embeddings, mask, **region_kwargs).sample.float()
        if inputs.get('progress_callback') is not None:
            inputs['progress_callback'](index + 1, steps)
    if progress is not None:
        progress['step'] = steps
    kind = inputs.get('output_type', 'pil')
    if kind == 'latent':
        images = output
    else:
        if kind not in ('pil', 'np', 'pt') or pipeline.vae is None:
            raise ValueError('MOSS decoded output requires a VAE and pil, np, or pt output.')
        vae = pipeline.vae
        latent = output.to(dtype=vae.dtype)
        latent = latent * (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(latent).view(1, -1, 1, 1) + vae.bn.running_mean.to(latent).view(1, -1, 1, 1)
        b, c, h, w = latent.shape
        latent = latent.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2*h, 2*w)
        images = pipeline.image_processor.postprocess(vae.decode(latent, return_dict=False)[0].clamp(-1, 1), output_type=kind)
    pipeline.maybe_free_model_hooks()
    return ImagePipelineOutput(images=images)
