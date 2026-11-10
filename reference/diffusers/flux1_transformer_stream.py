"""Strict original FLUX.1 transformer loading with CPU-resident linear weights."""
from __future__ import annotations

import json
from pathlib import Path

from flux1_single_file import PREFIX, ROOTS, inspect_header
from streamed_linear import streamed_linear_type
from weight_files import file_signature


def plan(shapes, expected, num_layers, num_single_layers):
    """Plan the original FLUX mapping, including fused QKV and final modulation."""
    mapping, seen = {}, set()

    def add(source, targets, mode='direct'):
        if source not in shapes or any(name not in expected or name in seen for name in targets):
            raise ValueError(f'FLUX transformer has a missing or duplicate tensor: {source}.')
        shape = tuple(shapes[source])
        output = [tuple(expected[name]) for name in targets]
        if mode == 'split':
            valid = bool(shape) and all(value and value[1:] == shape[1:] for value in output)
            valid = valid and sum(value[0] for value in output) == shape[0]
        else:
            valid = len(output) == 1 and output[0] == shape
            if mode == 'swap':
                valid = valid and bool(shape) and shape[0] % 2 == 0
        if not valid:
            raise ValueError(f'FLUX transformer tensor shape does not match its configuration: {source}.')
        mapping[source] = (tuple(targets), mode)
        seen.update(targets)

    def pair(source, target):
        for suffix in ('.weight', '.bias'):
            add(source + suffix, [target + suffix])

    for original, converted in (
        ('time_in', 'time_text_embed.timestep_embedder'),
        ('vector_in', 'time_text_embed.text_embedder'),
        ('guidance_in', 'time_text_embed.guidance_embedder'),
    ):
        if original == 'guidance_in' and converted + '.linear_1.weight' not in expected:
            continue
        pair(original + '.in_layer', converted + '.linear_1')
        pair(original + '.out_layer', converted + '.linear_2')
    pair('txt_in', 'context_embedder')
    pair('img_in', 'x_embedder')
    for index in range(num_layers):
        src, dst = f'double_blocks.{index}.', f'transformer_blocks.{index}.'
        for original, converted in (
            ('img_mod.lin', 'norm1.linear'), ('txt_mod.lin', 'norm1_context.linear'),
            ('img_mlp.0', 'ff.net.0.proj'), ('img_mlp.2', 'ff.net.2'),
            ('txt_mlp.0', 'ff_context.net.0.proj'), ('txt_mlp.2', 'ff_context.net.2'),
            ('img_attn.proj', 'attn.to_out.0'), ('txt_attn.proj', 'attn.to_add_out'),
        ):
            pair(src + original, dst + converted)
        for side, names in (('img', ('to_q', 'to_k', 'to_v')),
                            ('txt', ('add_q_proj', 'add_k_proj', 'add_v_proj'))):
            for suffix in ('.weight', '.bias'):
                add(src + side + '_attn.qkv' + suffix,
                    [dst + 'attn.' + name + suffix for name in names], 'split')
        for side, role in (('img', ''), ('txt', 'added_')):
            for original, target in (('query_norm', 'q'), ('key_norm', 'k')):
                add(src + side + '_attn.norm.' + original + '.scale',
                    [dst + 'attn.norm_' + role + target + '.weight'])
    for index in range(num_single_layers):
        src, dst = f'single_blocks.{index}.', f'single_transformer_blocks.{index}.'
        pair(src + 'modulation.lin', dst + 'norm.linear')
        pair(src + 'linear2', dst + 'proj_out')
        for suffix in ('.weight', '.bias'):
            add(src + 'linear1' + suffix,
                [dst + name + suffix for name in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'proj_mlp')], 'split')
        for original, target in (('query_norm', 'q'), ('key_norm', 'k')):
            add(src + 'norm.' + original + '.scale', [dst + 'attn.norm_' + target + '.weight'])
    pair('final_layer.linear', 'proj_out')
    for suffix in ('.weight', '.bias'):
        add('final_layer.adaLN_modulation.1' + suffix, ['norm_out.linear' + suffix], 'swap')
    if set(mapping) != set(shapes):
        raise ValueError(f'FLUX transformer has unexpected tensors: {sorted(set(shapes) - set(mapping))[:5]}.')
    if seen != set(expected):
        raise ValueError(f'FLUX transformer has missing tensors: {sorted(set(expected) - seen)[:5]}.')
    return mapping


def load(checkpoint, configuration, dtype):
    import torch
    from diffusers import FluxTransformer2DModel
    from safetensors import safe_open
    from model_loading import require_materialized_component

    checkpoint, configuration = Path(checkpoint), Path(configuration)
    config_path = configuration / 'transformer' / 'config.json'
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError('FLUX transformer streaming requires a local transformer/config.json.')
    before_config, before_weight = file_signature(config_path), file_signature(checkpoint)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('_class_name') != 'FluxTransformer2DModel' or 'auto_map' in config:
        raise ValueError('FLUX transformer streaming requires a built-in transformer configuration.')
    header = inspect_header(checkpoint)
    prefixed = {name[len(PREFIX):]: name for name in header if name.startswith(PREFIX)}
    plain = {name: name for name in header if name.split('.', 1)[0] in ROOTS}
    if prefixed and plain:
        raise ValueError('FLUX transformer mixes original tensor namespaces.')
    names = prefixed or plain
    metadata = header.get('__metadata__', {})
    if (not isinstance(metadata, dict) or not names or 'scaled_fp8' in header
            or any('quant' in key.lower() or 'scaled_fp8' in key.lower() for key in metadata)
            or any(name.endswith(('.weight_scale', '.scale_weight')) for name in names)):
        raise ValueError('FLUX transformer streaming requires unscaled original tensors.')
    with torch.device('meta'):
        model = FluxTransformer2DModel.from_config(config)
    model.requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    shapes = {name: tuple(header[full]['shape']) for name, full in names.items()}
    mapping = plan(shapes, expected, int(model.config.num_layers), int(model.config.num_single_layers))
    streamed, values = {}, {}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for name, (targets, mode) in mapping.items():
            tensor = source.get_tensor(names[name])
            if tensor.dtype not in (torch.float8_e4m3fn, torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f'FLUX transformer tensor has an unsupported dtype: {name}.')
            converted = tensor.to(dtype=dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f'FLUX transformer tensor is non-finite or overflows selected precision: {name}.')
            if mode == 'split':
                sizes = [expected[target][0] for target in targets]
                parts = tensor.split(sizes, dim=0)
                converted_parts = converted.split(sizes, dim=0)
            elif mode == 'swap':
                first, second = tensor.chunk(2, dim=0)
                part = torch.cat((second, first), dim=0)
                parts = (part,)
                converted_parts = (part.to(dtype=dtype),)
            else:
                parts, converted_parts = (tensor,), (converted,)
            for target, part, dense_part in zip(targets, parts, converted_parts):
                eligible = (target.endswith('.weight') and
                            isinstance(model.get_submodule(target.removesuffix('.weight')), torch.nn.Linear))
                values[target] = part if eligible else dense_part
                if eligible:
                    streamed[target] = (part.numel(), part.element_size())
        model.load_state_dict(values, strict=True, assign=True)
    if not streamed:
        raise ValueError('FLUX transformer streaming requires linear weights.')
    for target in streamed:
        module_name = target.removesuffix('.weight')
        parent_name, _, leaf = module_name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, streamed_linear_type()(model.get_submodule(module_name), dtype))
    require_materialized_component(model, 'FLUX.transformer')
    if file_signature(config_path) != before_config or file_signature(checkpoint) != before_weight:
        raise RuntimeError('FLUX transformer checkpoint or configuration changed while loading.')
    model._iild_cpu_stream = True
    model._iild_single_file_loading = {
        'format': 'original-unscaled', 'runtime_weights': 'cpu-stream', 'streamed_layers': len(streamed),
        'streamed_source_bytes': sum(count * size for count, size in streamed.values()),
        'largest_expanded_layer_bytes': max(count * torch.empty((), dtype=dtype).element_size()
                                             for count, _ in streamed.values()),
    }
    return model.eval()
