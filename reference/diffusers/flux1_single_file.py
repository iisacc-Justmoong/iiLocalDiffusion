"""Strict header check for the bundled FLUX.1 Schnell single-file preset."""
import json
from pathlib import Path
import struct

from weight_files import file_signature

PREFIX = 'model.diffusion_model.'
ROOTS = frozenset({'double_blocks', 'single_blocks', 'final_layer', 'time_in', 'vector_in',
                   'img_in', 'txt_in', 'guidance_in'})
ALLOWED_DTYPES = frozenset({'F16', 'BF16', 'F32', 'F8_E4M3'})


def inspect_header(path):
    path = Path(path)
    before = file_signature(path)
    with path.open('rb') as source:
        size = source.read(8)
        if len(size) != 8:
            raise ValueError('Invalid FLUX single-file safetensors header.')
        length = struct.unpack('<Q', size)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid FLUX single-file safetensors header.')
        header = json.loads(source.read(length))
    if not isinstance(header, dict) or file_signature(path) != before:
        raise ValueError('Invalid or changed FLUX single-file safetensors header.')
    return header


def validate_schnell(path, config_folder):
    """Reject missing, mixed-namespace, differently shaped or scaled transformers."""
    header = inspect_header(path)
    expected = json.loads((Path(config_folder) / 'transformer-source-shapes.json').read_text())
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Invalid bundled FLUX transformer shape contract.')
    prefixed = {name[len(PREFIX):]: value for name, value in header.items() if name.startswith(PREFIX)}
    unprefixed = {name: value for name, value in header.items()
                  if name.split('.', 1)[0] in ROOTS}
    if prefixed and unprefixed:
        raise ValueError('FLUX Schnell checkpoint mixes transformer namespaces.')
    transformer = prefixed or unprefixed
    if 'scaled_fp8' in header or any(name.endswith(('.scale_weight', '.weight_scale')) for name in transformer):
        raise ValueError('The FLUX Schnell preset does not support separately scaled FP8 transformer weights.')
    if set(transformer) != set(expected):
        raise ValueError('FLUX Schnell transformer tensor names do not match the bundled configuration.')
    for name, shape in expected.items():
        item = transformer[name]
        if (not isinstance(item, dict) or item.get('dtype') not in ALLOWED_DTYPES
                or item.get('shape') != shape):
            raise ValueError(f'FLUX Schnell transformer tensor format or shape is unsupported: {name}.')


def validate_embedded_vae(path, config_folder, diffusers):
    """Check original FLUX VAE conversion against its configured meta model."""
    import torch
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    config_path = Path(config_folder) / 'vae' / 'config.json'
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError('FLUX embedded VAE requires a local vae/config.json.')
    before = file_signature(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (not isinstance(config, dict) or config.get('_class_name') != 'AutoencoderKL'
            or 'auto_map' in config or file_signature(config_path) != before):
        raise ValueError('FLUX embedded VAE configuration is invalid or changed.')
    header = inspect_header(path)
    if (any(name.startswith('vae.') for name in header)
            and any(name.startswith('first_stage_model.') for name in header)):
        raise ValueError('FLUX embedded VAE mixes original tensor namespaces.')
    tensors = {}
    for name, item in header.items():
        if not name.startswith(('vae.', 'first_stage_model.')):
            continue
        shape = item.get('shape') if isinstance(item, dict) else None
        if (not isinstance(shape, list) or len(shape) > 8
                or any(type(dimension) is not int or dimension < 0 or dimension > 2**31 - 1
                       for dimension in shape)):
            raise ValueError(f'FLUX embedded VAE has an invalid tensor shape: {name}.')
        try:
            tensors[name] = torch.empty(shape, device='meta')
        except (RuntimeError, OverflowError) as error:
            raise ValueError(f'FLUX embedded VAE has an invalid tensor shape: {name}.') from error
    if not tensors:
        raise ValueError('FLUX embedded VAE has no tensors.')
    try:
        with torch.device('meta'):
            model = diffusers.AutoencoderKL.from_config(config)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        converted = convert_ldm_vae_checkpoint(tensors, config)
    except (KeyError, IndexError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f'FLUX embedded VAE cannot be converted: {error}') from error
    missing = sorted(set(expected) - set(converted))
    if missing:
        raise ValueError(f'FLUX embedded VAE is missing converted weights: {missing[:5]}.')
    unexpected = sorted(set(converted) - set(expected))
    if unexpected:
        raise ValueError(f'FLUX embedded VAE has unexpected converted weights: {unexpected[:5]}.')
    for name, value in converted.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f'FLUX embedded VAE has an incompatible converted shape: {name}.')
