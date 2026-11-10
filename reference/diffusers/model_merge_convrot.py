"""Validated LoRA updates for Comfy tensorwise INT8 ConvRot weights."""
from __future__ import annotations

import json
import math


def tensor_spec(readers, layout, address):
    """Resolve bounded quantization companions without reading weight payloads."""
    component, key = address
    if not key.endswith('.weight'):
        raise ValueError('ConvRot LoRA target must be a weight matrix.')
    stem = key[:-len('.weight')]
    names = (key, stem + '.weight_scale', stem + '.comfy_quant')
    header = {}
    for name in names:
        item = (component, name)
        if item not in layout:
            raise ValueError(f'Missing ConvRot quantization companion: {name}.')
        value = readers[layout[item]].get_slice(name)
        header[name] = {'shape': value.get_shape(), 'dtype': value.get_dtype()}
    info = header[names[2]]
    if info['dtype'] != 'U8' or len(info['shape']) != 1 or not 1 <= info['shape'][0] <= 4096:
        raise ValueError('Invalid ConvRot quantization metadata descriptor.')
    try:
        raw = readers[layout[(component, names[2])]].get_tensor(names[2])
        metadata = json.loads(bytes(raw.tolist()))
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        raise ValueError('Invalid ConvRot quantization metadata.') from error
    from moss_int8_format import tensor_spec as validate
    scale, _, group = validate(header, key, {names[2]: metadata})
    return (component, scale), group


def requantize(torch, rotated, base, scale, affected=None):
    """Store a finite rotated matrix with its new scalar or row-wise scales."""
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError('Invalid ConvRot scale values.')
    if not bool(torch.isfinite(rotated).all()):
        raise ValueError('ConvRot merged values are not finite.')
    maxima = (rotated.abs().amax().reshape(scale.shape) if scale.numel() == 1
              else rotated.abs().amax(dim=1, keepdim=True))
    computed_scale = (maxima.clamp_min(1e-30) / 127).to(dtype=scale.dtype)
    if not bool(torch.isfinite(computed_scale).all()) or bool((computed_scale <= 0).any()):
        raise ValueError('ConvRot requantization scale is not representable.')
    if scale.numel() == 1:
        result_scale = computed_scale
        result = (rotated / result_scale.to(dtype=rotated.dtype)).round().clamp(-128, 127).to(torch.int8)
    else:
        if affected is None:
            affected = torch.ones(base.shape[0], dtype=torch.bool)
        result_scale = scale.clone()
        result_scale[affected] = computed_scale[affected]
        result = base.clone()
        result[affected] = (rotated[affected] / result_scale[affected].to(dtype=rotated.dtype)).round().clamp(-128, 127).to(torch.int8)
    return result.contiguous(), result_scale.contiguous()


def merge(torch, readers, layout, address, base, loras, request):
    """Add exact low-rank deltas in the stored rotation, then requantize."""
    if base.dtype != torch.int8 or base.ndim != 2:
        raise ValueError('ConvRot LoRA requires a two-dimensional INT8 weight.')
    scale_address, group = tensor_spec(readers, layout, address)
    scale = readers[layout[scale_address]].get_tensor(scale_address[1])
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError(f'Invalid ConvRot scale values: {scale_address[1]}.')
    use_double = any(reader.get_slice(name).get_dtype() == 'F64'
                     for _, delta in loras for reader, name in (delta.down, delta.up))
    dtype = torch.float64 if use_double else torch.float32
    rotated = base.to(dtype=dtype) * scale.to(dtype=dtype)
    affected = torch.zeros(base.shape[0], dtype=torch.bool)
    from moss_convrot import rotate
    applied = False
    for index, delta in loras:
        coefficient = request.weights[index - 1] * delta.scale
        if request.mode == 'weighted-difference':
            coefficient = -coefficient
        if not math.isfinite(coefficient):
            raise ValueError(f'Nonfinite LoRA coefficient: {delta.module}.')
        if coefficient == 0:
            continue
        if delta.adaptation is not None:
            raise ValueError('Synthetic LoRA adaptation is not supported for quantized targets.')
        down = delta.down[0].get_tensor(delta.down[1]).to(dtype=dtype)
        up = delta.up[0].get_tensor(delta.up[1]).to(dtype=dtype)
        if down.ndim != 2 or up.ndim != 2 or not bool(torch.isfinite(down).all()) or not bool(torch.isfinite(up).all()):
            raise ValueError(f'ConvRot LoRA projections must be finite matrices: {delta.module}.')
        if not bool(torch.count_nonzero(down)) or not bool(torch.count_nonzero(up)):
            continue
        left, right = (down.T, up.T) if delta.fan_in_fan_out != delta.target.transpose else (up, down)
        output = rotated[slice(*delta.target.rows)] if delta.target.rows else rotated
        if left.shape[0] != output.shape[0] or right.shape[1] != output.shape[1] or left.shape[1] != right.shape[0]:
            raise ValueError(f'ConvRot LoRA dimensions differ from the target: {delta.module}.')
        output.addmm_(left, rotate(right, group), alpha=coefficient)
        if delta.target.rows:
            affected[slice(*delta.target.rows)] = True
        else:
            affected.fill_(True)
        applied = True
    if not applied:
        return base.clone().contiguous(), scale.clone().contiguous(), scale_address, False
    result, result_scale = requantize(torch, rotated, base, scale, affected)
    return result, result_scale, scale_address, True
