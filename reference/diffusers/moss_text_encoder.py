"""Load the language branch of a local Qwen3.5 single-file encoder for MOSS."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from weight_files import cached_model_sha256, file_sha256, file_signature, resolve_weight_file, verify_weight_file
from generation_defaults import checked_resource, resource_catalogs

PREFIX = 'model.language_model.'
IGNORED = ('model.visual.', 'lm_head.', 'mtp.')


@dataclass(frozen=True)
class ConfigSelection:
    path: str
    size: int
    sha256: str
    status: str


def verify_config(selection):
    if selection is None:
        return
    path = Path(selection.path)
    if not path.is_file() or path.stat().st_size != selection.size or file_sha256(path) != selection.sha256:
        raise RuntimeError('MOSS text encoder configuration changed during generation.')


def resolve_config(folder, resources=None):
    path = Path(folder) / 'text_encoder/config.json'
    status = 'model'
    expected = None
    if not path.is_file():
        for root, manifest in resource_catalogs(resources):
            expected = manifest.get('fallback_moss_text_config')
            if expected is not None:
                break
        if not isinstance(expected, dict) or not isinstance(expected.get('file'), str):
            raise ValueError('Install the verified MOSS Qwen3.5 configuration.')
        relative = Path(expected['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('MOSS text encoder configuration escapes the resource package.')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('MOSS text encoder configuration escapes the resource package.')
        status = 'fallback'
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError('MOSS text encoder configuration is missing or too large.')
    before = file_signature(path)
    digest = file_sha256(path)
    size = path.stat().st_size
    config = json.loads(path.read_text(encoding='utf-8'))
    if (file_signature(path) != before or not isinstance(config, dict)
            or config.get('model_type') != 'qwen3_5' or 'auto_map' in config):
        raise ValueError('MOSS text encoder requires an unchanged Qwen3.5 configuration.')
    if expected and (size != expected.get('size') or digest != expected.get('sha256')):
        raise ValueError('MOSS text encoder configuration differs from its manifest.')
    return ConfigSelection(str(path), size, digest, status)


def resolve(path, pipeline_class):
    if path is None:
        return None
    if pipeline_class != 'MOSSImagePipeline':
        raise ValueError('--moss-text-encoder requires MOSSImagePipeline.')
    return resolve_weight_file(path, 'MOSS Qwen3.5 text encoder')


def resolve_fallback(args, index, folder):
    """Use a pinned local INT8 resource only when the model has no encoder weights."""
    if (args.pipeline_class or index['_class_name']) != 'MOSSImagePipeline':
        return None, 'not-configured-for-pipeline'
    if index.get('text_encoder') != ['transformers', 'Qwen3_5Model']:
        raise ValueError('MOSS requires a Qwen3_5Model text encoder declaration.')
    component = Path(folder) / 'text_encoder'
    from directory_checkpoint_conversion import component_has_weights
    if component_has_weights(component):
        return None, 'model'
    item = None
    for root, manifest in resource_catalogs(getattr(args, 'generation_resources', None)):
        item = manifest.get('fallback_moss_text_weights')
        if item is not None:
            break
    if item is None:
        return None, 'unavailable'
    if (not isinstance(item, dict) or item.get('format') != 'int8-tensorwise-convrot'
            or not isinstance(item.get('file'), str) or type(item.get('size')) is not int
            or item['size'] <= 0 or not isinstance(item.get('sha256'), str)
            or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256'])):
        raise ValueError('Invalid MOSS text encoder weight catalog.')
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts or not (root / relative).resolve().is_relative_to(root):
        raise ValueError('MOSS text encoder weight escapes the resource package.')
    if not (root / relative).is_file():
        return None, 'unavailable'
    selected = checked_resource(root, item)
    if selected.sha256 is None:
        digest = cached_model_sha256(Path(selected.resolved_file))
        if digest != item['sha256']:
            raise ValueError('Bundled generation resource differs from its manifest.')
        selected = replace(selected, sha256=digest)
    return selected, 'fallback-int8-convrot'


def plan(shapes, expected):
    mapping = {}
    for source, shape in shapes.items():
        if source.startswith(IGNORED):
            continue
        if not source.startswith(PREFIX):
            raise ValueError(f'Unsupported MOSS Qwen3.5 encoder tensor: {source}.')
        target = source[len(PREFIX):]
        if target not in expected or tuple(shape) != tuple(expected[target]):
            raise ValueError(f'MOSS Qwen3.5 encoder tensor does not match configuration: {source}.')
        mapping[source] = target
    if set(mapping.values()) != set(expected):
        raise ValueError(f'MOSS Qwen3.5 encoder has missing text tensors: {sorted(set(expected) - set(mapping.values()))[:5]}.')
    return mapping


def load_convrot(source, header, model, expected, dtype, progress=None):
    import torch
    import moss_int8_format
    from moss_convrot import module_types

    metadata = {}
    for name, info in header.items():
        if name.startswith(PREFIX) and name.endswith('.comfy_quant'):
            if info['dtype'] != 'U8' or len(info['shape']) != 1 or not 1 <= info['shape'][0] <= 4096:
                raise ValueError(f'Invalid MOSS quantization metadata: {name}.')
            try:
                metadata[name] = json.loads(bytes(source.get_tensor(name).tolist()))
            except (ValueError, UnicodeDecodeError, TypeError) as error:
                raise ValueError(f'Invalid MOSS quantization metadata: {name}.') from error
    floating, quantized = moss_int8_format.plan(header, expected, metadata)
    if not quantized:
        raise ValueError('MOSS ConvRot checkpoint contains no quantized language tensors.')
    counter = None
    if progress is not None:
        from moss_loading_progress import TensorLoadingProgress
        names = list(floating) + [name for item in quantized.values() for name in (item.weight, item.scale)]
        counter = TensorLoadingProgress(header, names, 'text_encoder', progress)
    values = {}
    for original, target in floating.items():
        value = source.get_tensor(original)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'MOSS text encoder tensor is non-finite: {original}.')
        value = value.to(dtype=dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'MOSS text encoder tensor overflows selected precision: {original}.')
        values[target] = value
        if counter is not None:
            counter.completed(original)
    missing, unexpected = model.load_state_dict(values, strict=False, assign=True)
    if set(missing) != {item.target for item in quantized.values()} or unexpected:
        raise ValueError('MOSS quantized text encoder state does not match its configuration.')
    linear_class, embedding_class = module_types()
    for item in quantized.values():
        module_name = item.target[:-len('.weight')]
        original_module = model.get_submodule(module_name)
        if type(original_module) is torch.nn.Linear:
            replacement_class = linear_class
        elif type(original_module) is torch.nn.Embedding:
            replacement_class = embedding_class
        else:
            raise ValueError(f'Unsupported MOSS INT8 module: {module_name}.')
        weight = source.get_tensor(item.weight)
        scale = source.get_tensor(item.scale)
        if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
            raise ValueError(f'Invalid MOSS quantized scale values: {item.scale}.')
        model.set_submodule(module_name, replacement_class(original_module, weight, scale, item.group_size, dtype))
        if counter is not None:
            counter.completed(item.weight, item.scale)
    model._iild_cpu_int8_convrot = True


def load(selection, configuration, dtype, *, progress=None):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5TextModel
    from model_loading import require_materialized_component

    config_path = Path(configuration)
    if config_path.is_dir():
        config_path /= 'config.json'
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError('MOSS single-file text encoder requires a bounded text_encoder/config.json.')
    before = file_signature(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('model_type') != 'qwen3_5' or 'auto_map' in config:
        raise ValueError('MOSS single-file text encoder requires a Qwen3.5 configuration.')
    verify_weight_file(selection, 'MOSS Qwen3.5 text encoder')
    with torch.device('meta'):
        model = Qwen3_5TextModel(Qwen3_5Config.from_dict(config).text_config)
    model.requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(selection.path, framework='pt', device='cpu') as source:
        header = {name: {'shape': source.get_slice(name).get_shape(), 'dtype': source.get_slice(name).get_dtype()}
                  for name in source.keys()}
        if any(name.startswith(PREFIX) and name.endswith('.comfy_quant') for name in header):
            load_convrot(source, header, model, expected, dtype, progress=progress)
        else:
            mapping = plan({name: info['shape'] for name, info in header.items()}, expected)
            counter = None
            if progress is not None:
                from moss_loading_progress import TensorLoadingProgress
                counter = TensorLoadingProgress(header, mapping, 'text_encoder', progress)
            values = {}
            for original, target in mapping.items():
                tensor = source.get_tensor(original)
                if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f'MOSS text encoder requires unquantized floating weights: {original}.')
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'MOSS text encoder tensor is non-finite: {original}.')
                tensor = tensor.to(dtype=dtype)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'MOSS text encoder tensor overflows selected precision: {original}.')
                values[target] = tensor
                if counter is not None:
                    counter.completed(original)
            model.load_state_dict(values, strict=True, assign=True)
    rotary = model.rotary_emb
    if rotary.inv_freq.is_meta:
        with torch.device('cpu'):
            model.rotary_emb = type(rotary)(model.config)
    require_materialized_component(model, 'MOSS.text_encoder')
    verify_weight_file(selection, 'MOSS Qwen3.5 text encoder')
    if file_signature(config_path) != before:
        raise RuntimeError('MOSS text encoder configuration changed while loading.')
    if progress is not None and getattr(model, '_iild_cpu_int8_convrot', False):
        from moss_loading_progress import attach_quantized_forward_progress
        attach_quantized_forward_progress(model, 'text_encoder_compute', progress)
    return model.eval()
