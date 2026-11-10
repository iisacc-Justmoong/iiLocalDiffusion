"""Krea 2 Diffusers-format color/depth control adapter."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from weight_files import resolve_weight_file, verify_weight_file


@dataclass(frozen=True)
class Selection:
    weight: object
    depth: str | None
    color: str | None
    scale: float
    start: float
    depth_end: float
    color_end: float
    auto_bucket: bool = False
    target_pipeline: str = 'Krea2Pipeline'
    image_argument: str = 'control_image'

    def metadata(self):
        return {'kind': 'krea2-color-depth-control', 'path': self.weight.path,
                'sha256': self.weight.sha256, 'size_bytes': self.weight.size_bytes,
                'depth': self.depth, 'color': self.color, 'scale': self.scale,
                'start': self.start, 'depth_end': self.depth_end, 'color_end': self.color_end,
                'resolution_mode': 'input-aspect-bucket' if self.auto_bucket else 'explicit-or-default',
                'scale_scope': 'LoRA layers; the trained input projection remains fully active'}


def resolve(args):
    if getattr(args, 'krea2_control_auto_bucket', False) and getattr(args, 'krea2_control_size_explicit', False):
        raise ValueError('--krea2-control-auto-bucket cannot be combined with an explicit width or height.')
    if any(value is not None for value in (args.controlnet_variant, args.guess_mode, args.control_mode)):
        raise ValueError('Krea 2 color/depth control does not support a variant, guess mode, or Union mode.')
    if args.offload != 'none' or args.seed is None:
        raise ValueError('Krea 2 color/depth control requires --offload none and an explicit --seed.')
    if args.control_preprocessor_library and args.control_preprocess != 'canny':
        raise ValueError('--control-preprocessor-library requires --control-preprocess canny.')
    if args.control_preprocess == 'canny' and not args.control_image:
        raise ValueError('Native Canny requires a depth/control image.')
    if not args.control_image and not args.control_color_image:
        raise ValueError('Krea 2 color/depth control requires a depth and/or color image.')
    if 'control_image' in args.inputs or 'control_color_image' in args.inputs:
        raise ValueError('Do not also provide control images through --pipeline-inputs.')
    def image_path(value):
        if value is None:
            return None
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Krea 2 control images must be nonempty local files.')
        return str(path)
    scale = 1. if args.controlnet_scale is None else args.controlnet_scale
    start = 0. if args.control_guidance_start is None else args.control_guidance_start
    end = 1. if args.control_guidance_end is None else args.control_guidance_end
    depth_end = end if args.control_depth_end is None else args.control_depth_end
    color_end = end if args.control_color_end is None else args.control_color_end
    if (not all(map(math.isfinite, (scale, start, end, depth_end, color_end))) or scale < 0
            or not 0 <= start < end <= 1 or not 0 <= depth_end <= end or not 0 <= color_end <= end):
        raise ValueError('Krea 2 color/depth control requires nonnegative scale and valid control intervals.')
    return Selection(resolve_weight_file(args.controlnet, 'Krea 2 color/depth adapter'),
                     image_path(args.control_image), image_path(args.control_color_image),
                     scale, start, depth_end, color_end, getattr(args, 'krea2_control_auto_bucket', False))


def plan_weights(shapes, module_shapes, projection_shape):
    if len(projection_shape) != 2:
        raise ValueError('Krea 2 color/depth control requires a valid input projection.')
    outputs, inputs = projection_shape
    if tuple(shapes.get('img_in.weight', ())) != (outputs, 3 * inputs) or tuple(shapes.get('img_in.bias', ())) != (outputs,):
        raise ValueError('Krea 2 color/depth input projection does not match the base transformer.')
    if not module_shapes:
        raise ValueError('Krea 2 color/depth control requires transformer targets.')
    expected = {'img_in.weight', 'img_in.bias'}
    rank = None
    for target, shape in module_shapes.items():
        a, b = target + '.lora_A.weight', target + '.lora_B.weight'
        expected.update((a, b))
        left, right = shapes.get(a), shapes.get(b)
        if (len(shape) != 2 or left is None or right is None or len(left) != 2 or len(right) != 2
                or left[0] < 1 or tuple(left[1:]) != (shape[1],) or tuple(right) != (shape[0], left[0])):
            raise ValueError(f'Krea 2 color/depth LoRA tensors do not match {target}.')
        if rank is None:
            rank = left[0]
        elif left[0] != rank:
            raise ValueError('Krea 2 color/depth LoRA ranks must be consistent.')
    if set(shapes) != expected:
        raise ValueError('Krea 2 color/depth control has missing or unexpected tensors.')
    return sorted(module_shapes)


def _encode_hint(pipeline, image, inputs, torch, np, *, color, output_identity=None):
    from PIL import Image
    width, height = inputs['width'], inputs['height']
    mode = Image.Resampling.NEAREST if color else Image.Resampling.BILINEAR
    image = (image.convert('RGB') if color else image.convert('L').convert('RGB')).resize((width, height), mode)
    if output_identity is not None:
        from control_hint_identity import fingerprint
        output_identity.update(fingerprint(image, np))
    vae = pipeline.vae
    parameter = next(vae.parameters())
    pixels = torch.from_numpy(np.array(image, copy=True)).to(device=parameter.device, dtype=torch.float32)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).div(127.5).sub(1).to(dtype=parameter.dtype)
    latent = vae.encode(pixels).latent_dist.mode()
    mean = torch.as_tensor(vae.config.latents_mean, device=latent.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=latent.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent = ((latent.float() - mean) / std).squeeze(2)
    if not bool(torch.isfinite(latent).all()):
        raise ValueError('Krea 2 color/depth control VAE produced non-finite latents.')
    batch, channels, height, width = latent.shape
    return pipeline._pack_latents(latent, batch, channels, height, width)


@contextmanager
def execution(pipeline, images, inputs, selection, torch, np):
    from safetensors import safe_open
    transformer = pipeline.transformer
    # Treat a PEFT wrapper and its base layer as one logical model layer.
    # Traversal is parent-first, so adapter internals do not become targets.
    modules, wrapped = {}, []
    for name, module in transformer.named_modules():
        if any(name.startswith(prefix + '.') for prefix in wrapped):
            continue
        base = module.get_base_layer() if callable(getattr(module, 'get_base_layer', None)) else module
        if isinstance(base, torch.nn.Linear):
            modules[name] = module
            if base is not module:
                wrapped.append(name)
    projection = modules.get('img_in')
    base_projection = (projection.get_base_layer() if callable(getattr(projection, 'get_base_layer', None))
                       else projection)
    if not isinstance(base_projection, torch.nn.Linear):
        raise ValueError('Krea 2 transformer has no compatible input projection.')
    targets = {name: tuple(module.weight.shape) for name, module in modules.items()
               if name != 'img_in'}
    device, dtype = base_projection.weight.device, base_projection.weight.dtype
    if device.type == 'meta':
        raise ValueError('Krea 2 color/depth control requires resident model weights.')
    verify_weight_file(selection.weight, 'Krea 2 color/depth adapter')
    with safe_open(selection.weight.path, framework='pt', device='cpu') as source:
        shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        plan = plan_weights(shapes, targets, tuple(base_projection.weight.shape))
        def tensor(name):
            value = source.get_tensor(name)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Krea 2 color/depth tensor is invalid: {name}')
            value = value.to(device=device, dtype=dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Krea 2 color/depth tensor overflows selected precision: {name}')
            return value
        first, bias = tensor('img_in.weight'), tensor('img_in.bias')
        pairs = [(modules[name], tensor(name + '.lora_A.weight'), tensor(name + '.lora_B.weight')) for name in plan]
    verify_weight_file(selection.weight, 'Krea 2 color/depth adapter')
    hints = {}
    color_id, depth_id = {}, {}
    color = _encode_hint(pipeline, images['color'], inputs, torch, np, color=True, output_identity=color_id).to(device=device, dtype=dtype) if images.get('color') is not None else None
    depth = _encode_hint(pipeline, images['depth'], inputs, torch, np, color=False, output_identity=depth_id).to(device=device, dtype=dtype) if images.get('depth') is not None else None
    if color is not None:
        hints['color'] = color_id
    if depth is not None:
        hints['depth'] = depth_id
    scheduler = pipeline.scheduler
    had_step = 'step' in vars(scheduler)
    old_instance = vars(scheduler).get('step')
    old_step = scheduler.step
    state, activity, handles = {'step': 0}, {'color_steps': [], 'depth_steps': []}, []
    try:
        def project(module, args, output):
            value = args[0]
            total = len(scheduler.timesteps)
            fraction = state['step'] / total if total else 0
            def selected(hint, end, name):
                if hint is None or not selection.start <= fraction < end:
                    return torch.zeros_like(value)
                if hint.shape[1:] != value.shape[1:]:
                    raise ValueError('Krea 2 color/depth control latent does not match the generated image tokens.')
                if not activity[name] or activity[name][-1] != state['step']:
                    activity[name].append(state['step'])
                return hint.expand(value.shape[0], -1, -1).to(device=value.device, dtype=value.dtype)
            return torch.nn.functional.linear(torch.cat((value, selected(color, selection.color_end, 'color_steps'),
                                                         selected(depth, selection.depth_end, 'depth_steps')), dim=-1), first, bias)
        # Leave the wrapper's style residual in place after control projection.
        handles.append(base_projection.register_forward_hook(project))
        for module, a, b in pairs:
            def add_adapter(module, args, output, a=a, b=b):
                return output + selection.scale * torch.nn.functional.linear(torch.nn.functional.linear(args[0], a), b)
            handles.append(module.register_forward_hook(add_adapter))
        def step(*args, **kwargs):
            result = old_step(*args, **kwargs)
            state['step'] += 1
            return result
        scheduler.step = step
        yield {'adapter': 'krea2-color-depth-control', 'control_activity': activity,
               'vae_input_hints': hints,
               'control_size': [inputs['width'], inputs['height']],
               'vae_latents': 'posterior mean, 2x2 packing and Qwen Image latent normalization',
               'scale_scope': 'LoRA layers; the trained input projection remains fully active'}
    finally:
        if had_step:
            scheduler.step = old_instance
        else:
            vars(scheduler).pop('step', None)
        for handle in reversed(handles):
            handle.remove()
