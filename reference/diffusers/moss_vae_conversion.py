"""Stage FLUX.2 VAE keys for MOSS without altering tensor payloads."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import struct


def _header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid MOSS VAE safetensors header.')
        length, = struct.unpack('<Q', prefix)
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > size:
            raise ValueError('Invalid MOSS VAE safetensors header size.')
        try:
            header = json.loads(source.read(length))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError('Invalid MOSS VAE safetensors header.') from error
    if not isinstance(header, dict):
        raise ValueError('Invalid MOSS VAE safetensors header.')
    return header, length + 8


def _diffusers_name(name):
    name = name.replace('.nin_shortcut.', '.conv_shortcut.')
    name = name.replace('encoder.norm_out.', 'encoder.conv_norm_out.')
    name = name.replace('decoder.norm_out.', 'decoder.conv_norm_out.')
    name = name.replace('encoder.quant_conv.', 'quant_conv.')
    name = name.replace('decoder.post_quant_conv.', 'post_quant_conv.')
    name = re.sub(r'encoder\.down\.(\d+)\.block\.(\d+)\.', r'encoder.down_blocks.\1.resnets.\2.', name)
    name = re.sub(r'encoder\.down\.(\d+)\.downsample\.conv\.', r'encoder.down_blocks.\1.downsamplers.0.conv.', name)
    name = re.sub(r'decoder\.up\.(\d+)\.block\.(\d+)\.',
                  lambda m: f'decoder.up_blocks.{3-int(m[1])}.resnets.{m[2]}.', name)
    name = re.sub(r'decoder\.up\.(\d+)\.upsample\.conv\.',
                  lambda m: f'decoder.up_blocks.{3-int(m[1])}.upsamplers.0.conv.', name)
    name = re.sub(r'(encoder|decoder)\.mid\.block_(\d+)\.',
                  lambda m: f'{m[1]}.mid_block.resnets.{int(m[2])-1}.', name)
    for old, new in (('norm', 'group_norm'), ('q', 'to_q'), ('k', 'to_k'),
                     ('v', 'to_v'), ('proj_out', 'to_out.0')):
        name = name.replace(f'.mid.attn_1.{old}.', f'.mid_block.attentions.0.{new}.')
    return name


def stage_moss_vae_weights(source, destination, reference):
    """Map a legacy file against a verified reference; return whether rewritten."""
    source, destination = Path(source), Path(destination)
    original, data_offset = _header(source)
    expected, _ = _header(reference)
    metadata = original.pop('__metadata__', None)
    expected.pop('__metadata__', None)
    mapped = {}
    ranges = []
    byte_width = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8,
                  'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    for key, value in original.items():
        if not isinstance(value, dict) or value.get('dtype') not in byte_width:
            raise ValueError('Unsupported MOSS VAE tensor descriptor.')
        shape, offsets = value.get('shape'), value.get('data_offsets')
        if (not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(n) is int and n >= 0 for n in offsets) or offsets[1] < offsets[0]):
            raise ValueError('Invalid MOSS VAE tensor descriptor.')
        count = 1
        for dimension in shape: count *= dimension
        if offsets[1] - offsets[0] != count * byte_width[value['dtype']]:
            raise ValueError('Invalid MOSS VAE tensor size.')
        name = key if key in expected else _diffusers_name(key)
        if name in mapped or name not in expected:
            raise ValueError(f'MOSS VAE has an unexpected or duplicate tensor: {key}')
        shape = list(shape)
        if shape != expected[name].get('shape') and len(shape) == 4 and shape[-2:] == [1, 1]:
            shape = shape[:2]
        if shape != expected[name].get('shape'):
            raise ValueError(f'MOSS VAE tensor shape does not match FLUX.2: {key}')
        mapped[name] = {**value, 'shape': shape}
        ranges.append(tuple(offsets))
    if set(mapped) != set(expected):
        raise ValueError('MOSS VAE is missing tensors required by FLUX.2.')
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor: raise ValueError('MOSS VAE tensor offsets are incomplete or overlap.')
        cursor = end
    if cursor != source.stat().st_size - data_offset:
        raise ValueError('MOSS VAE tensor payload is incomplete.')
    converted = any(key not in expected or original[key]['shape'] != mapped[key]['shape'] for key in original if key in mapped)
    converted = converted or set(original) != set(mapped)
    if not converted:
        # A hard link changes the source inode's change time, invalidating the
        # identity token used by metadata-only model validation.
        shutil.copyfile(source, destination)
        return False
    if metadata is not None: mapped['__metadata__'] = metadata
    header = json.dumps(mapped, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    header += b' ' * (-len(header) % 8)
    with source.open('rb') as input_file, destination.open('xb') as output_file:
        output_file.write(struct.pack('<Q', len(header)))
        output_file.write(header)
        input_file.seek(data_offset)
        shutil.copyfileobj(input_file, output_file, 1024 * 1024)
    return True
