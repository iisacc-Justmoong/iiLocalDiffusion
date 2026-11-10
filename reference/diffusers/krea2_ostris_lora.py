"""Strict AI Toolkit Krea 2 edit LoRA mapping; no reference-image execution."""
from contextlib import contextmanager
import math
import re

from krea2_controlnet import BLOCK_TARGETS
from weight_files import verify_weight_file


def target_name(source):
    match = re.fullmatch(r'diffusion_model\.(?:blocks\.(\d+)|txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+))\.(attn\.(?:wq|wk|wv|wo|gate)|mlp\.(?:gate|up|down))', source)
    if match is None:
        raise ValueError(f'Unsupported Krea 2 edit LoRA target: {source}')
    block, group, index, suffix = match.groups()
    base = f'transformer_blocks.{block}' if block is not None else f'text_fusion.{group}.{index}'
    return base + '.' + BLOCK_TARGETS[suffix]


def plan_weights(shapes, modules):
    suffix_a, suffix_b = '.lora_A.weight', '.lora_B.weight'
    if not shapes:
        raise ValueError('Krea 2 edit LoRA has no tensors.')
    stems = {name[:-len(suffix_a)] for name in shapes if name.endswith(suffix_a)}
    expected = {stem + suffix for stem in stems for suffix in (suffix_a, suffix_b)}
    if set(shapes) != expected:
        raise ValueError('Krea 2 edit LoRA has missing, extra or unsupported tensors.')
    plan = []
    seen = set()
    for stem in sorted(stems):
        target = target_name(stem)
        a, b, shape = shapes[stem + suffix_a], shapes[stem + suffix_b], modules.get(target)
        if (target in seen or shape is None or len(shape) != 2 or len(a) != 2 or len(b) != 2
                or a[0] < 1 or tuple(a) != (a[0], shape[1]) or tuple(b) != (shape[0], a[0])):
            raise ValueError(f'Krea 2 edit LoRA tensor dimensions do not match {target}.')
        seen.add(target)
        plan.append((target, stem + suffix_a, stem + suffix_b))
    return plan


@contextmanager
def execution(transformer, weight, scale, torch):
    """Temporarily apply unmerged residuals to a resident Diffusers transformer."""
    from safetensors import safe_open
    from krea2_single_file import streamed_linear_type
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Krea 2 edit LoRA scale must be finite and nonnegative.')
    modules = dict(transformer.named_modules())
    shapes = {name: tuple(module.weight.shape) for name, module in modules.items()
              if isinstance(module, torch.nn.Linear)}
    verify_weight_file(weight, 'Krea 2 edit LoRA')
    with safe_open(weight.path, framework='pt', device='cpu') as source:
        plan = plan_weights({name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}, shapes)
        pairs = []
        for target, a_name, b_name in plan:
            module = modules[target]
            if module.weight.is_meta:
                raise ValueError('Krea 2 edit LoRA requires resident model weights.')
            compute_weight = transformer.img_in.weight if isinstance(module, streamed_linear_type()) else module.weight
            values = []
            for name in (a_name, b_name):
                value = source.get_tensor(name)
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Krea 2 edit LoRA tensor must be finite floating point: {name}')
                value = value.to(device=compute_weight.device, dtype=compute_weight.dtype)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Krea 2 edit LoRA tensor overflows selected precision: {name}')
                values.append(value)
            pairs.append((module, *values))
    verify_weight_file(weight, 'Krea 2 edit LoRA')
    handles = []
    try:
        for module, a, b in pairs:
            def apply(module, args, output, a=a, b=b):
                delta = torch.nn.functional.linear(torch.nn.functional.linear(args[0].to(a.dtype), a), b)
                return output + scale * delta.to(output.dtype)
            handles.append(module.register_forward_hook(apply))
        yield {'adapter': 'krea2-ai-toolkit-edit-lora', 'targets': len(plan), 'scale': scale}
    finally:
        for handle in reversed(handles):
            handle.remove()
    verify_weight_file(weight, 'Krea 2 edit LoRA')
