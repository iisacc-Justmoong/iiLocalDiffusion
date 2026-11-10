"""Validate the published MOSS Qwen3.5 ConvRot checkpoint layout.

This module only plans the conversion; it does not implement inference.
"""
from __future__ import annotations

from dataclasses import dataclass


PREFIX = 'model.language_model.'
IGNORED = ('model.visual.', 'lm_head.', 'mtp.')


@dataclass(frozen=True)
class QuantizedTensor:
    weight: str
    target: str
    scale: str
    metadata: str
    group_size: int


def tensor_spec(header, name, metadata):
    """Validate one Comfy tensorwise INT8 ConvRot weight and its companions."""
    info = header[name]
    if info.get('dtype') != 'I8' or not name.endswith('.weight') or len(info.get('shape', ())) != 2:
        raise ValueError(f'Unsupported MOSS quantized tensor: {name}.')
    stem = name[:-len('.weight')]
    scale_name, meta_name = stem + '.weight_scale', stem + '.comfy_quant'
    scale, meta_info = header.get(scale_name), header.get(meta_name)
    spec = metadata.get(meta_name)
    if (not isinstance(scale, dict) or scale.get('dtype') != 'F32'
            or tuple(scale.get('shape', ())) not in ((), (1,), (info['shape'][0], 1))):
        raise ValueError(f'Invalid MOSS quantized scale: {scale_name}.')
    if (not isinstance(meta_info, dict) or meta_info.get('dtype') != 'U8'
            or len(meta_info.get('shape', ())) != 1 or not 1 <= meta_info['shape'][0] <= 4096
            or not isinstance(spec, dict)
            or set(spec) not in ({'format', 'convrot', 'convrot_groupsize'},
                                 {'format', 'convrot', 'convrot_groupsize', 'per_row'})
            or ('per_row' in spec and (spec['per_row'] is not True
                                       or tuple(scale['shape']) != (info['shape'][0], 1)))
            or spec.get('format') != 'int8_tensorwise' or spec.get('convrot') is not True):
        raise ValueError(f'Invalid MOSS quantization metadata: {meta_name}.')
    group = spec.get('convrot_groupsize')
    if (type(group) is not int or group < 4 or group > 4096 or group & (group - 1)
            or (group.bit_length() - 1) % 2 or info['shape'][1] % group):
        raise ValueError(f'Invalid MOSS ConvRot group size: {meta_name}.')
    return scale_name, meta_name, group


def plan(header, expected, metadata):
    """Check every language tensor and return floating and INT8 tensor mappings.

    `header` entries have safetensors `dtype` and `shape`; `metadata` contains
    decoded, bounded JSON payloads keyed by their `.comfy_quant` tensor name.
    """
    floating = {}
    quantized = {}
    consumed = set()
    for name, info in header.items():
        if name == '__metadata__' or name.startswith(IGNORED):
            continue
        if not name.startswith(PREFIX):
            raise ValueError(f'Unsupported MOSS Qwen3.5 encoder tensor: {name}.')
        if name.endswith(('.comfy_quant', '.weight_scale')):
            continue
        target = name[len(PREFIX):]
        if target not in expected or tuple(info['shape']) != tuple(expected[target]):
            raise ValueError(f'MOSS Qwen3.5 encoder tensor does not match configuration: {name}.')
        if info['dtype'] in ('F16', 'BF16', 'F32'):
            floating[name] = target
            consumed.add(name)
            continue
        scale_name, meta_name, group = tensor_spec(header, name, metadata)
        quantized[name] = QuantizedTensor(name, target, scale_name, meta_name, group)
        consumed.update((name, scale_name, meta_name))
    language = {name for name in header if name.startswith(PREFIX)}
    if consumed != language:
        raise ValueError(f'Unexpected or unpaired MOSS quantization tensors: {sorted(language - consumed)[:5]}.')
    targets = set(floating.values()) | {entry.target for entry in quantized.values()}
    if targets != set(expected):
        raise ValueError(f'MOSS Qwen3.5 encoder has missing text tensors: {sorted(set(expected) - targets)[:5]}.')
    return floating, quantized
