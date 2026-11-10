"""FLUX.2 Fun ControlNet using the resident Diffusers transformer and VAE.

The control branch follows Alibaba VideoX-Fun's Apache-2.0 FLUX.2 control
architecture, adapted to Diffusers' independently loaded pipeline and hooks.
The checkpoint itself is supplied by the caller and has its own license.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

from weight_files import LocalWeightFile, resolve_weight_file, verify_weight_file

LAYERS = (0, 2, 4, 6)


def control_active(step, total, start, end):
    """Use the same complete-step interval convention as Diffusers ControlNet."""
    return total > 0 and 0 <= step < total and step / total >= start and (step + 1) / total <= end


@dataclass(frozen=True)
class Selection:
    weight: LocalWeightFile
    image: str
    scale: float
    start: float
    end: float
    target_pipeline: str = 'Flux2Pipeline'
    image_argument: str = 'control_image'
    inpaint_image: str | None = None
    mask_image: str | None = None

    def metadata(self):
        return {'kind': 'flux2-fun-controlnet', 'path': self.weight.path,
                'sha256': self.weight.sha256, 'size_bytes': self.weight.size_bytes,
                'image': self.image, 'scale': self.scale, 'start': self.start,
                'end': self.end, 'control_layers': list(LAYERS),
                'inpaint_image': self.inpaint_image, 'mask_image': self.mask_image}


def resolve(args):
    if any(value is not None for value in (args.controlnet_variant, args.guess_mode, args.control_mode,
                                            args.control_color_image, args.control_color_end, args.control_depth_end)):
        raise ValueError('FLUX.2 Fun control does not support variant, guess, Union mode, or MOSS controls.')
    if args.offload != 'none' or args.seed is None:
        raise ValueError('FLUX.2 Fun control requires --offload none and an explicit --seed.')
    if args.control_preprocessor_library and args.control_preprocess != 'canny':
        raise ValueError('--control-preprocessor-library requires --control-preprocess canny.')
    image = Path(args.control_image or '').expanduser().absolute()
    if not args.control_image or not image.is_file() or not image.stat().st_size:
        raise ValueError('FLUX.2 Fun control requires a nonempty local --control-image.')
    if any(key in args.inputs for key in ('control_image', 'inpaint_image', 'mask_image', 'latents')):
        raise ValueError('FLUX.2 Fun control requires its control, inpaint and mask inputs through their explicit options and does not accept supplied latents.')
    if args.inputs.get('image') is not None and any(key not in args.inputs for key in ('width', 'height')):
        raise ValueError('FLUX.2 Fun control with a reference image requires explicit width and height.')
    for key in ('width', 'height'):
        value = args.inputs.get(key, 1024)
        if type(value) is not int or not 16 <= value <= 8192 or value % 16:
            raise ValueError('FLUX.2 control dimensions must be multiples of 16 in [16,8192].')
    scale = 1. if args.controlnet_scale is None else args.controlnet_scale
    start = 0. if args.control_guidance_start is None else args.control_guidance_start
    end = 1. if args.control_guidance_end is None else args.control_guidance_end
    if not all(map(math.isfinite, (scale, start, end))) or scale < 0 or not 0 <= start < end <= 1:
        raise ValueError('FLUX.2 control requires nonnegative finite scale and 0 <= start < end <= 1.')
    def optional_image(value):
        if value is None:
            return None
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('FLUX.2 Fun inpaint and mask inputs must be nonempty local images.')
        return str(path)
    return Selection(resolve_weight_file(args.controlnet, 'FLUX.2 Fun control adapter'), str(image), scale, start, end,
                     inpaint_image=optional_image(getattr(args, 'control_inpaint_image', None)),
                     mask_image=optional_image(getattr(args, 'control_mask_image', None)))


def validate_shapes(shapes, base_shapes, *, hidden, latent_channels):
    expected = {'control_img_in.weight': (hidden, 2 * latent_channels + 4),
                'control_img_in.bias': (hidden,)}
    for index in range(len(LAYERS)):
        prefix = f'control_transformer_blocks.{index}.'
        expected.update({prefix + key: tuple(shape) for key, shape in base_shapes.items()})
        expected[prefix + 'after_proj.weight'] = (hidden, hidden)
        expected[prefix + 'after_proj.bias'] = (hidden,)
    expected['control_transformer_blocks.0.before_proj.weight'] = (hidden, hidden)
    expected['control_transformer_blocks.0.before_proj.bias'] = (hidden,)
    if set(shapes) != set(expected):
        raise ValueError('FLUX.2 control checkpoint has missing or unexpected tensors.')
    for name, shape in expected.items():
        if tuple(shapes[name]) != shape:
            raise ValueError(f'FLUX.2 control tensor shape mismatch: {name}.')


def _control_latents(pipeline, image, inputs, torch, np, *, inpaint_image=None, mask_image=None):
    from diffusers.image_processor import VaeImageProcessor
    width, height = inputs.get('width', 1024), inputs.get('height', 1024)
    vae = pipeline.vae
    param = next(vae.parameters())
    # VideoX-Fun normalizes the RGB control image in FP32 before casting it to
    # the VAE's dtype. Reuse that same Diffusers image processor and ordering.
    pixels = VaeImageProcessor(vae_scale_factor=16).preprocess(image.convert('RGB'), height=height, width=width)
    pixels = pixels.to(device=param.device, dtype=param.dtype)
    raw = vae.encode(pixels).latent_dist.mode()
    def patchify(z):
        batch, channels, latent_height, latent_width = z.shape
        if latent_height % 2 or latent_width % 2:
            raise ValueError('FLUX.2 VAE returned unpatchable control latents.')
        z = z.reshape(batch, channels, latent_height // 2, 2, latent_width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return z.reshape(batch, channels * 4, latent_height // 2, latent_width // 2)
    z = patchify(raw)
    mean = vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    if z.shape[1] != mean.shape[1]:
        raise ValueError('FLUX.2 control requires the matching VAE normalization.')
    control = (z - mean) / std
    if mask_image is None:
        mask = None
        condition_mask = z.new_zeros((z.shape[0], 4, z.shape[2], z.shape[3]))
    else:
        processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                      do_convert_grayscale=True)
        mask = processor.preprocess(mask_image, height=height, width=width).to(device=param.device, dtype=param.dtype)
        mask = (mask >= 0.5).to(dtype=param.dtype)
        condition_mask = torch.nn.functional.interpolate(1 - mask, size=raw.shape[-2:], mode='nearest')
        condition_mask = patchify(condition_mask)
    if inpaint_image is None:
        inpaint = torch.zeros_like(control)
    else:
        source = VaeImageProcessor(vae_scale_factor=16).preprocess(inpaint_image.convert('RGB'), height=height, width=width)
        source = source.to(device=param.device, dtype=param.dtype)
        source = torch.zeros_like(source) if mask is None else source * (mask < 0.5)
        inpaint = (patchify(vae.encode(source).latent_dist.mode()) - mean) / std
    if not all(bool(torch.isfinite(value).all()) for value in (control, condition_mask, inpaint)):
        raise ValueError('FLUX.2 control VAE produced non-finite latents.')
    return torch.cat(tuple(value.flatten(2).transpose(1, 2) for value in (control, condition_mask, inpaint)), dim=-1)


@contextmanager
def execution(pipeline, image, inputs, selection, torch, np, *, inpaint_image=None, mask_image=None):
    from safetensors import safe_open
    transformer = pipeline.transformer
    config = transformer.config
    if len(transformer.transformer_blocks) != 8:
        raise ValueError('FLUX.2 Fun control requires eight double-stream transformer blocks.')
    base = transformer.transformer_blocks[0]
    parameter = next(transformer.parameters())
    if parameter.device.type == 'meta':
        raise ValueError('FLUX.2 Fun control requires resident model weights.')
    hidden = config.num_attention_heads * config.attention_head_dim
    with torch.device('meta'):
        blocks = torch.nn.ModuleList([type(base)(dim=hidden, num_attention_heads=config.num_attention_heads,
                            attention_head_dim=config.attention_head_dim, mlp_ratio=config.mlp_ratio,
                            eps=config.eps) for _ in LAYERS])
        control_in = torch.nn.Linear(2 * config.in_channels + 4, hidden)
        before = torch.nn.Linear(hidden, hidden)
        after = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in LAYERS])
    base_shapes = {name: tuple(value.shape) for name, value in blocks[0].state_dict().items()}
    verify_weight_file(selection.weight, 'FLUX.2 Fun control adapter')
    with safe_open(selection.weight.path, framework='pt', device='cpu') as source:
        shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        validate_shapes(shapes, base_shapes, hidden=hidden, latent_channels=config.in_channels)
        def tensor(name):
            value = source.get_tensor(name)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'FLUX.2 control tensor is non-floating or non-finite: {name}')
            value = value.to(device=parameter.device, dtype=parameter.dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'FLUX.2 control tensor overflows selected precision: {name}')
            return value
        def load(module, prefix):
            module.load_state_dict({name: tensor(prefix + name) for name in module.state_dict()}, strict=True, assign=True)
        load(control_in, 'control_img_in.')
        load(before, 'control_transformer_blocks.0.before_proj.')
        for i, block in enumerate(blocks):
            prefix = f'control_transformer_blocks.{i}.'
            load(block, prefix)
            load(after[i], prefix + 'after_proj.')
    verify_weight_file(selection.weight, 'FLUX.2 Fun control adapter')
    context = _control_latents(pipeline, image, inputs, torch, np,
                               inpaint_image=inpaint_image, mask_image=mask_image).to(device=parameter.device, dtype=parameter.dtype)
    handles = []
    scheduler = pipeline.scheduler
    had_step = 'step' in vars(scheduler)
    previous_instance_step = vars(scheduler).get('step')
    original_step = scheduler.step
    state = {'step': 0, 'hints': ()}
    try:
        def compute_hints(module, args, kwargs):
            if not control_active(state['step'], len(scheduler.timesteps), selection.start, selection.end) or not selection.scale:
                state['hints'] = ()
                return
            x = kwargs['hidden_states']
            reference_tokens = x.shape[1] - context.shape[1]
            if reference_tokens < 0 or (reference_tokens and inputs.get('image') is None):
                raise ValueError('FLUX.2 control and generated image token counts differ.')
            control_context = context.expand(x.shape[0], -1, -1)
            if reference_tokens:
                # VideoX-Fun appends zero control tokens for reference images;
                # only the generated image receives the control hint.
                control_context = torch.cat((control_context, control_context.new_zeros(
                    (x.shape[0], reference_tokens, control_context.shape[-1]))), dim=1)
            c = control_in(control_context)
            c = before(c) + x
            text = kwargs['encoder_hidden_states']
            hints = []
            for block, projection in zip(blocks, after):
                text, c = block(hidden_states=c, encoder_hidden_states=text,
                                temb_mod_img=kwargs['temb_mod_img'], temb_mod_txt=kwargs['temb_mod_txt'],
                                image_rotary_emb=kwargs.get('image_rotary_emb'),
                                joint_attention_kwargs=kwargs.get('joint_attention_kwargs'))
                hints.append(projection(c))
            state['hints'] = tuple(hints)
        handles.append(base.register_forward_pre_hook(compute_hints, with_kwargs=True))
        for index, layer in enumerate(LAYERS):
            def inject(module, args, output, index=index):
                if not state['hints']:
                    return output
                text, image_output = output
                return text, image_output + state['hints'][index] * selection.scale
            handles.append(transformer.transformer_blocks[layer].register_forward_hook(inject))
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            state['step'] += 1
            return result
        scheduler.step = step
        yield {'kind': 'flux2-fun-controlnet', 'control_layers': list(LAYERS),
               'control_context': 'normalized control latent, packed inverse mask and inpaint latent',
               'inpaint_image': inpaint_image is not None, 'mask_image': mask_image is not None,
               'reference_context': ('zero control tokens appended for reference images'
                                     if inputs.get('image') is not None else None)}
    finally:
        for handle in handles:
            handle.remove()
        if had_step:
            scheduler.step = previous_instance_step
        else:
            vars(scheduler).pop('step', None)
