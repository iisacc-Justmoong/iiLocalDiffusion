"""Load a local MOSS transformer, splitting Comfy-Org's fused QKV tensors."""
from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import re
import warnings

from weight_files import file_signature
from streamed_linear import streamed_linear_type

FUSED_QKV = re.compile(r'layers\.[0-9]+\.proj\.qkv\.weight\Z')
ATTENTION_CONFIG = re.compile(r'layers\.[0-9]+\.comfy_attention\.config\Z')


class _IndexedSlice:
    def __init__(self, tensor):
        self._tensor = tensor

    def get_shape(self):
        return self._tensor['shape']

    def get_dtype(self):
        return self._tensor['dtype']


class _IndexedSource:
    """Present validated shard metadata through the safetensors reader interface."""
    def __init__(self, report, paths, opener):
        self._metadata = report['metadata']
        self._tensors = {item['name']: item for item in report['tensors']}
        self._paths = paths
        self._opener = opener
        self._open = OrderedDict()

    def keys(self):
        return self._tensors.keys()

    def metadata(self):
        return self._metadata

    def get_slice(self, name):
        return _IndexedSlice(self._tensors[name])

    def get_tensor(self, name):
        shard = self._tensors[name]['shard']
        if shard not in self._open:
            # Bound open descriptors even for models with thousands of shards.
            if len(self._open) == 8:
                _, (manager, _) = self._open.popitem(last=False)
                manager.__exit__(None, None, None)
            manager = self._opener(str(self._paths[shard]), framework='pt', device='cpu')
            reader = manager.__enter__()
            self._open[shard] = manager, reader
        self._open.move_to_end(shard)
        return self._open[shard][1].get_tensor(name)

    def close(self):
        while self._open:
            _, (manager, _) = self._open.popitem(last=False)
            manager.__exit__(None, None, None)


@contextmanager
def open_checkpoint(checkpoint):
    """Open a single model or validated indexed shards without combining files."""
    from safetensors import safe_open
    from downloaded_model import model_container_format
    if model_container_format(checkpoint) != 'safetensors-index':
        with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
            yield source
        return

    from image_package import inspect_sharded_index
    from native_image import native_weight
    from weight_files import verify_weight_file
    weight = native_weight(str(checkpoint), 'MOSS transformer')
    report = inspect_sharded_index(checkpoint)
    verify_weight_file(weight, 'MOSS transformer')
    paths = [Path(item.path) for item in weight.shards]
    if (len(paths) != len(report['shards'])
            or any(str(path.resolve()) != item['resolved_path']
                   for path, item in zip(paths, report['shards']))):
        raise RuntimeError('The MOSS transformer shard inventory changed during inspection.')
    source = _IndexedSource(report, paths, safe_open)
    try:
        yield source
    finally:
        source.close()
    verify_weight_file(weight, 'MOSS transformer')


def merge_history_diagnostics(metadata):
    """Report recorded synthetic LoRA remaps, without asserting their quality."""
    raw = metadata.get('iild_merge') if isinstance(metadata, dict) else None
    if not isinstance(raw, str) or len(raw) > 8 * 1024 * 1024:
        return None
    try:
        history = json.loads(raw)
    except (ValueError, TypeError):
        return None
    sources = history.get('sources') if isinstance(history, dict) else None
    if not isinstance(sources, list):
        return None
    count, examples = 0, []
    for source in sources:
        if not isinstance(source, dict) or source.get('kind') != 'lora':
            continue
        targets = source.get('targets')
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, dict):
                continue
            adaptation = target.get('adaptation')
            if (not isinstance(adaptation, dict) or adaptation.get('synthetic') is not True
                    or adaptation.get('selection') != 'role-depth-shape-nearest'):
                continue
            count += 1
            if len(examples) < 5:
                examples.append({'module': str(target.get('module', ''))[:256],
                                 'target': str(target.get('tensor', ''))[:256]})
    if not count:
        return None
    return {'synthetic_lora_targets': count, 'examples': examples, 'quality_verified': False,
            'message': 'The checkpoint records synthetic LoRA remapping to approximate targets. '
                       'This may alter unrelated weights; image quality and causation are not verified.'}


def _unprefix(name):
    for prefix in ('model.diffusion_model.', 'diffusion_model.', 'transformer.'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def plan(shapes, expected):
    """Validate a complete model before reading any weight payloads."""
    result, seen = {}, set()
    for source, shape in shapes.items():
        name = _unprefix(source)
        if FUSED_QKV.fullmatch(name):
            names = tuple(name.replace('.qkv.weight', f'.{part}.weight') for part in 'qkv')
            dimensions = [tuple(expected.get(target, ())) for target in names]
            valid = (all(len(value) == 2 for value in dimensions)
                     and len(set(dimensions)) == 1
                     and tuple(shape) == (3 * dimensions[0][0], dimensions[0][1]))
        else:
            names = (name,)
            valid = name in expected and tuple(shape) == tuple(expected[name])
        if any(name in seen for name in names):
            raise ValueError(f'MOSS single-file checkpoint has duplicate targets: {source}.')
        if not valid:
            raise ValueError(f'MOSS single-file checkpoint has an unexpected tensor or shape: {source}.')
        seen.update(names)
        result[source] = names
    if seen != set(expected):
        raise ValueError(f'MOSS single-file checkpoint has missing tensors: {sorted(set(expected) - seen)[:5]}.')
    return result


def attention_preferences(header, expected, metadata):
    """Validate bounded optional Comfy attention preferences, never weight tensors."""
    result, seen = {}, set()
    for source, info in header.items():
        if not source.endswith('.comfy_attention.config'):
            continue
        name = _unprefix(source)
        target = name.removesuffix('.comfy_attention.config')
        if (not ATTENTION_CONFIG.fullmatch(name) or target in seen
                or target + '.proj.q.weight' not in expected
                or info.get('dtype') != 'U8' or len(info.get('shape', ())) != 1
                or not 1 <= info['shape'][0] <= 4096):
            raise ValueError(f'Invalid MOSS attention preference tensor: {source}.')
        value = metadata.get(source)
        choices = value if isinstance(value, list) else [value]
        if len(choices) > 32 or any(not isinstance(item, dict)
                or not isinstance(item.get('attention'), str) or not 0 < len(item['attention']) <= 128
                for item in choices):
            raise ValueError(f'Invalid MOSS attention preferences: {source}.')
        result[source] = [item['attention'] for item in choices]
        seen.add(target)
    return result


def quantized_plan(header, expected, metadata):
    """Validate all transformer weights and quantization companions before loading."""
    from moss_int8_format import tensor_spec

    attention = attention_preferences(header, expected, metadata)
    weights = {name: tuple(info['shape']) for name, info in header.items()
               if not name.endswith(('.comfy_quant', '.weight_scale', '.comfy_attention.config'))}
    mapping = plan(weights, expected)
    quantized, consumed = {}, set(attention)
    for name, targets in mapping.items():
        dtype = header[name]['dtype']
        if dtype in ('F16', 'BF16', 'F32'):
            consumed.add(name)
            continue
        scale, meta, group = tensor_spec(header, name, metadata)
        quantized[name] = (targets, scale, meta, group)
        consumed.update((name, scale, meta))
    if consumed != set(header):
        raise ValueError(f'Unexpected or unpaired MOSS quantization tensors: {sorted(set(header) - consumed)[:5]}.')
    return mapping, quantized


def load(checkpoint, configuration, dtype, *, storage='dense', progress=None):
    """Read strict floating and Comfy INT8 ConvRot weights into the pinned transformer."""
    import torch
    from model_loading import require_materialized_component
    from vendor_moss.pipeline import MOSSImageTransformer

    if storage not in ('dense', 'cpu-stream'):
        raise ValueError('MOSS transformer storage must be dense or cpu-stream.')
    checkpoint, configuration = Path(checkpoint), Path(configuration)
    config_path = configuration / 'config.json'
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError('MOSS single-file loading requires a bounded transformer/config.json.')
    before_config, before_weight = file_signature(config_path), file_signature(checkpoint)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('_class_name') != 'MOSSImageTransformer' or 'auto_map' in config:
        raise ValueError('MOSS transformer configuration must declare the pinned model class.')
    with torch.device('meta'):
        model = MOSSImageTransformer.from_config(config)
    model.requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with open_checkpoint(checkpoint) as source:
        merge_provenance = merge_history_diagnostics(source.metadata())
        if merge_provenance:
            warnings.warn(f"MOSS checkpoint records {merge_provenance['synthetic_lora_targets']} synthetic LoRA "
                          f"remaps. {merge_provenance['message']}", RuntimeWarning)
        header = {name: {'shape': tuple(source.get_slice(name).get_shape()),
                         'dtype': source.get_slice(name).get_dtype()} for name in source.keys()}
        metadata = {}
        for name, info in header.items():
            if not name.endswith(('.comfy_quant', '.comfy_attention.config')):
                continue
            if info['dtype'] != 'U8' or len(info['shape']) != 1 or not 1 <= info['shape'][0] <= 4096:
                raise ValueError(f'Invalid MOSS quantization or attention metadata: {name}.')
            try:
                metadata[name] = json.loads(bytes(source.get_tensor(name).tolist()))
            except (ValueError, UnicodeDecodeError, TypeError) as error:
                raise ValueError(f'Invalid MOSS quantization or attention metadata: {name}.') from error
        mapping, quantized = quantized_plan(header, expected, metadata)
        attention = attention_preferences(header, expected, metadata)
        counter = None
        if progress is not None:
            from moss_loading_progress import TensorLoadingProgress
            counter = TensorLoadingProgress(header, list(mapping) + [entry[1] for entry in quantized.values()],
                                            'transformer', progress)
        values, streamed = {}, {}
        for original, targets in mapping.items():
            if original in quantized:
                continue
            tensor = source.get_tensor(original)
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f'MOSS single-file tensor requires unquantized floating weights: {original}.')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'MOSS single-file tensor is non-finite: {original}.')
            converted = tensor.to(dtype=dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f'MOSS single-file tensor overflows selected precision: {original}.')
            parts = tensor.chunk(3, dim=0) if len(targets) == 3 else (tensor,)
            converted_parts = converted.chunk(3, dim=0) if len(targets) == 3 else (converted,)
            for target, part, converted_part in zip(targets, parts, converted_parts):
                eligible = (storage == 'cpu-stream' and target.endswith('.weight')
                            and target.startswith(('layers.', 'text_embedder.blocks.'))
                            and isinstance(model.get_submodule(target.removesuffix('.weight')), torch.nn.Linear))
                values[target] = part if eligible else converted_part
                if eligible:
                    streamed[target] = (part.numel(), part.element_size())
            if counter is not None:
                counter.completed(original)
        missing, unexpected = model.load_state_dict(values, strict=False, assign=True)
        missing_weights = {target for targets, _, _, _ in quantized.values() for target in targets}
        if set(missing) != missing_weights or unexpected:
            raise ValueError('MOSS transformer tensors do not match their configured targets.')
        if quantized:
            from moss_convrot import module_types
            linear_class, _ = module_types()
            for original, (targets, scale_name, _, group) in quantized.items():
                weight = source.get_tensor(original)
                scale = source.get_tensor(scale_name)
                if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
                    raise ValueError(f'Invalid MOSS quantized scale values: {scale_name}.')
                weights = weight.chunk(len(targets), dim=0)
                scales = scale.chunk(len(targets), dim=0) if scale.ndim == 2 else (scale,) * len(targets)
                for target, part, factor in zip(targets, weights, scales):
                    module_name = target.removesuffix('.weight')
                    module = model.get_submodule(module_name)
                    if type(module) is not torch.nn.Linear:
                        raise ValueError(f'Unsupported MOSS INT8 transformer module: {module_name}.')
                    model.set_submodule(module_name, linear_class(module, part.contiguous(), factor.contiguous(), group, dtype))
                if counter is not None:
                    counter.completed(original, scale_name)
    if storage == 'cpu-stream' and not streamed and not quantized:
        raise ValueError('MOSS cpu-stream storage requires transformer linear weights.')
    for target in streamed:
        module_name = target.removesuffix('.weight')
        parent_name, _, leaf = module_name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, streamed_linear_type()(model.get_submodule(module_name), dtype))
    require_materialized_component(model, 'MOSS.transformer')
    if file_signature(config_path) != before_config or file_signature(checkpoint) != before_weight:
        raise RuntimeError('MOSS single-file model or transformer configuration changed while loading.')
    model._iild_single_file_loading = {
        'format': 'comfy-int8-convrot' if quantized else 'original-dense',
        'runtime_weights': ('cpu-stream+int8-convrot' if storage == 'cpu-stream' else 'int8-convrot')
                           if quantized else ('cpu-stream' if storage == 'cpu-stream' else 'dense'),
        'streamed_layers': len(streamed),
        'quantized_layers': sum(len(targets) for targets, _, _, _ in quantized.values()),
        'streamed_source_bytes': sum(count * size for count, size in streamed.values()),
        'largest_expanded_layer_bytes': max((count * torch.empty((), dtype=dtype).element_size()
                                             for count, _ in streamed.values()), default=0),
    }
    if merge_provenance:
        model._iild_single_file_loading['merge_provenance'] = merge_provenance
    if attention:
        methods = sorted({method for options in attention.values() for method in options})
        model._iild_single_file_loading['attention'] = {
            'preferred': methods, 'layers': len(attention), 'selected': 'pytorch-sdpa',
            'reason': 'stored Comfy attention kernels are unavailable in this runtime',
        }
        warnings.warn('MOSS attention preferences use PyTorch SDPA in this runtime; '
                      'results can differ from the preferred Comfy kernel.', RuntimeWarning)
    model._iild_cpu_int8_convrot = bool(quantized)
    model._iild_cpu_stream = storage == 'cpu-stream' or bool(quantized)
    if progress is not None and quantized:
        from moss_loading_progress import attach_quantized_forward_progress
        attach_quantized_forward_progress(model, 'transformer_compute', progress)
    return model.eval()
