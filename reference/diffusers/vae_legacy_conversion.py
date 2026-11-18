"""Stage original LDM VAE tensor names without modifying tensor payloads."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import struct

from weight_files import IndexedWeightFile


def _header(path):
    from downloaded_model import MAX_HEADER_BYTES, _json_object
    with Path(path).open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid standalone VAE Safetensors header.')
        length, = struct.unpack('<Q', prefix)
        if length < 2 or length > MAX_HEADER_BYTES or length + 8 > Path(path).stat().st_size:
            raise ValueError('Invalid standalone VAE Safetensors header size.')
        return _json_object(source.read(length), 'VAE Safetensors header'), 8 + length


def _legacy_names(names):
    return any(re.search(r'^(?:encoder\.down\.\d+\.|decoder\.up\.\d+\.|'
                         r'(?:encoder|decoder)\.mid\.(?:block_\d+|attn_\d+)\.)', name)
               for name in names)


def _conversion_map(selection, headers, runtime):
    entries = {name: descriptor for header, _ in headers for name, descriptor in header.items()
               if name != '__metadata__'}
    if sum(len(header) - ('__metadata__' in header) for header, _ in headers) != len(entries):
        raise ValueError('Standalone VAE contains duplicate tensor names across shards.')
    prefixes = {prefix for name in entries for prefix in ('vae.', 'first_stage_model.') if name.startswith(prefix)}
    if len(prefixes) > 1:
        raise ValueError('Standalone VAE mixes tensor namespaces.')
    prefix = next(iter(prefixes), '')
    if prefix and any(not name.startswith(prefix) for name in entries):
        raise ValueError('Standalone VAE mixes prefixed and unprefixed tensor names.')
    bare = {name[len(prefix):]: name for name in entries}
    legacy = _legacy_names(bare)
    if not prefix and not legacy:
        return None
    if legacy and selection.class_name not in ('AutoencoderKL', 'AutoencoderKLFlux2'):
        raise ValueError('Legacy LDM VAE names require an AutoencoderKL or AutoencoderKLFlux2 architecture.')

    # Validate offsets, dtypes and complete payload extents before copying.
    from downloaded_model import _safetensors
    paths = (selection.weight.shards if isinstance(selection.weight, IndexedWeightFile)
             else (selection.weight,))
    for source in paths:
        _safetensors(Path(source.path))

    import torch
    try:
        config = json.loads(Path(selection.config_path).read_text())
        component = getattr(runtime, selection.class_name)
        with torch.device('meta'):
            expected = {name: tuple(tensor.shape) for name, tensor in component.from_config(config).state_dict().items()}
        input_tensors = {name: torch.empty(entries[original]['shape'], device='meta')
                         for name, original in bare.items()}
        if legacy and selection.class_name == 'AutoencoderKL':
            from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
            converted = convert_ldm_vae_checkpoint(input_tensors, config)
        elif legacy:
            from moss_vae_conversion import _diffusers_name
            converted = {}
            for source_name, tensor in input_tensors.items():
                name = _diffusers_name(source_name)
                if name in converted:
                    raise ValueError(f'Duplicate converted VAE tensor: {name}.')
                target_shape = expected.get(name)
                while target_shape is not None and tensor.ndim > len(target_shape) and tensor.shape[-1] == 1:
                    tensor = tensor[..., 0]
                converted[name] = tensor
        else:
            converted = input_tensors
    except (KeyError, IndexError, TypeError, RuntimeError, OverflowError, ValueError) as error:
        raise ValueError(f'Standalone VAE names cannot be converted: {error}') from error
    missing = sorted(expected.keys() - converted.keys())
    unexpected = sorted(converted.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(f'Standalone VAE conversion does not match its configuration: missing={missing[:5]}, unexpected={unexpected[:5]}.')
    sources = {id(tensor): bare[name] for name, tensor in input_tensors.items()}
    mapping = {}
    for name, tensor in converted.items():
        if tensor is None or tuple(tensor.shape) != expected[name]:
            raise ValueError(f'Standalone VAE tensor has an incompatible converted shape: {name}.')
        base = tensor
        while getattr(base, '_base', None) is not None:
            base = base._base
        original = sources.get(id(base))
        if (original is None or original in mapping or not tensor.is_contiguous()
                or tensor.storage_offset() != 0 or tensor.numel() != base.numel()):
            raise ValueError(f'Standalone VAE tensor conversion is not a one-to-one byte-preserving view: {name}.')
        mapping[original] = (name, list(tensor.shape))
    if set(mapping) != set(entries):
        raise ValueError('Standalone VAE has additional or unmapped source tensors.')
    return mapping


def _write_converted_shard(source, destination, header, offset, mapping):
    converted = {}
    for name, descriptor in header.items():
        if name == '__metadata__':
            converted[name] = descriptor
        else:
            target, shape = mapping[name]
            if target in converted:
                raise ValueError('Standalone VAE conversion produced duplicate tensor names.')
            converted[target] = {**descriptor, 'shape': shape}
    encoded = json.dumps(converted, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as original, destination.open('xb') as output:
        output.write(struct.pack('<Q', len(encoded)))
        output.write(encoded)
        original.seek(offset)
        shutil.copyfileobj(original, output, 1024 * 1024)


def stage_legacy_vae(selection, directory, runtime):
    """Stage shards and convert legacy VAE names in a private overlay.

    Returns false for an ordinary single file, which can be linked directly.
    Source identity is checked by the caller before and after staging/loading.
    """
    root = Path(directory)
    indexed = isinstance(selection.weight, IndexedWeightFile)
    sources = selection.weight.shards if indexed else (selection.weight,)
    headers = [_header(item.path) for item in sources]
    mapping = _conversion_map(selection, headers, runtime)
    if indexed:
        from directory_checkpoint_conversion import _link_or_copy
        source_index = Path(selection.weight.path)
        index = json.loads(source_index.read_text())
        metadata = index.setdefault('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('Standalone VAE shard index metadata must be an object.')
        names = {}
        for number, item in enumerate(sources):
            relative = Path(item.path).relative_to(source_index.parent)
            if not relative.parts or '..' in relative.parts or relative in names:
                raise ValueError('Standalone VAE shard paths are invalid or duplicated.')
            names[relative] = f'vae-shard-{number:05d}.safetensors'
        index['weight_map'] = {
            (mapping[name][0] if mapping is not None else name): names[Path(path)]
            for name, path in index['weight_map'].items()
        }
        (root / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(index))
        for item, (header, offset) in zip(sources, headers):
            relative = Path(item.path).relative_to(source_index.parent)
            target = root / names[relative]
            if mapping is None:
                _link_or_copy(Path(item.path), target)
            else:
                _write_converted_shard(item.path, target, header, offset, mapping)
    else:
        if mapping is None:
            return False
        header, offset = headers[0]
        _write_converted_shard(sources[0].path, root / 'diffusion_pytorch_model.safetensors', header, offset, mapping)
    return True
