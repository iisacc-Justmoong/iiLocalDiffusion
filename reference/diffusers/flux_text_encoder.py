"""Load local FLUX.1 CLIP and T5 safetensors without executing package code."""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import struct

from flux_tokenizer import FLUX1_CLIP_PIPELINES, uses_precomputed
from generation_defaults import checked_resource, read_defaults
from weight_files import cached_model_sha256, file_signature, resolve_weight_file, verify_weight_file

ROLES = {'text_encoder': ('CLIPTextModel', 'clip_text_model'), 'text_encoder_2': ('T5EncoderModel', 't5')}


@dataclass(frozen=True)
class Selection:
    role: str
    weight: object
    config_path: str
    config_sha256: str
    config_size: int
    config_status: str
    weight_status: str = 'explicit'
    namespace: str = ''

    def metadata(self):
        return {'mode': 'embedded-single-file' if self.namespace else 'single-file',
                'status': self.weight_status, 'class': ROLES[self.role][0], 'path': self.weight.path,
                'sha256': self.weight.sha256, 'size_bytes': self.weight.size_bytes,
                'validation': 'sha256' if self.weight.sha256 is not None else 'metadata',
                'config': {'path': self.config_path, 'sha256': self.config_sha256,
                           'size_bytes': self.config_size, 'status': self.config_status}}


def _read_config(path, role):
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError(f'FLUX {role} requires a bounded local configuration.')
    before = file_signature(path)
    content = path.read_bytes()
    if file_signature(path) != before:
        raise RuntimeError('FLUX text encoder configuration changed while reading.')
    config = json.loads(content)
    if not isinstance(config, dict) or config.get('model_type') != ROLES[role][1] or 'auto_map' in config:
        raise ValueError(f'FLUX {role} has an incompatible configuration.')
    import hashlib
    return config, len(content), hashlib.sha256(content).hexdigest()


def resolve_file(path, role, pipeline_class, folder, resources, *, weight_status='explicit'):
    if path is None:
        return None
    if role not in ROLES or pipeline_class not in FLUX1_CLIP_PIPELINES:
        raise ValueError('FLUX single-file text encoders require a FLUX.1 pipeline.')
    weight = resolve_weight_file(path, f'FLUX {role}')
    config_path = Path(folder) / role / 'config.json'
    status, item = 'model', None
    if not config_path.is_file():
        root, manifest = read_defaults(resources)
        entries = manifest.get('fallback_flux_text_configs', [])
        if not isinstance(entries, list) or len(entries) > 8 or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError('Invalid FLUX text encoder configuration catalog.')
        matches = [entry for entry in entries if entry.get('component') == role]
        if len(matches) != 1 or matches[0].get('class_name') != ROLES[role][0] or not isinstance(matches[0].get('file'), str):
            raise ValueError(f'Install the verified FLUX {role} configuration.')
        item = matches[0]
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('FLUX text encoder configuration escapes the resource package.')
        config_path = (root / relative).resolve(strict=True)
        if not config_path.is_relative_to(root):
            raise ValueError('FLUX text encoder configuration escapes the resource package.')
        status = 'fallback'
    _, size, digest = _read_config(config_path, role)
    if item and (size != item.get('size') or digest != item.get('sha256')):
        raise ValueError('FLUX text encoder configuration differs from its manifest.')
    return Selection(role, weight, str(config_path), digest, size, status, weight_status)


def resolve_embedded_t5(path, pipeline_class, folder, resources):
    selected = resolve_file(path, 'text_encoder_2', pipeline_class, folder, resources, weight_status='embedded')
    prefix = 'text_encoders.t5xxl.transformer.'
    with Path(selected.weight.path).open('rb') as source:
        size = source.read(8)
        if len(size) != 8:
            raise ValueError('Invalid embedded FLUX T5 safetensors header.')
        length = struct.unpack('<Q', size)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > Path(selected.weight.path).stat().st_size:
            raise ValueError('Invalid embedded FLUX T5 safetensors header.')
        header = json.loads(source.read(length))
    if not isinstance(header, dict) or prefix + 'shared.weight' not in header:
        raise ValueError('Embedded FLUX T5 CPU streaming requires the Comfy T5 tensor namespace.')
    unsupported = [name for name in header if name.startswith('text_encoders.t5xxl.')
                   and not name.startswith(prefix) and name != 'text_encoders.t5xxl.logit_scale']
    if unsupported or 'scaled_fp8' in header:
        raise ValueError('Embedded FLUX T5 has ambiguous quantization metadata outside its tensor namespace.')
    verify_weight_file(selected.weight, 'FLUX embedded T5')
    return replace(selected, namespace=prefix)


def _single_file_has_encoder(path, role):
    path = Path(path)
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid safetensors header while checking FLUX text encoders.')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header while checking FLUX text encoders.')
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors header while checking FLUX text encoders.')
    prefixes = (('text_encoders.clip_l.transformer.', 'text_encoder.', 'conditioner.embedders.0.')
                if role == 'text_encoder' else ('text_encoders.t5xxl.transformer.', 'text_encoder_2.'))
    return any(key.startswith(prefixes) for key in header)


def resolve_fallback(args, index, folder, role):
    name = args.pipeline_class or index['_class_name']
    if name not in FLUX1_CLIP_PIPELINES or role not in ROLES:
        return None, 'not-configured-for-pipeline'
    component = index.get(role)
    if component in (None, [None, None]):
        return None, 'not-declared'
    if component != ['transformers', ROLES[role][0]]:
        raise ValueError(f'FLUX {role} declares an incompatible encoder.')
    if (args.source_kind == 'single-file' and not getattr(args, 'single_file_component_only', False)
            and _single_file_has_encoder(args.model, role)):
        return None, 'model'
    component_dir = Path(folder) / role
    from directory_checkpoint_conversion import component_has_weights
    if component_has_weights(component_dir):
        return None, 'model'
    if uses_precomputed(args.inputs):
        return None, 'precomputed-embeddings'
    root, manifest = read_defaults(getattr(args, 'generation_resources', None))
    entries = manifest.get('fallback_flux_text_weights', [])
    if not isinstance(entries, list) or len(entries) > 8 or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('Invalid FLUX text encoder weight catalog.')
    matches = [entry for entry in entries if entry.get('component') == role]
    if len(matches) != 1 or matches[0].get('class_name') != ROLES[role][0] or not isinstance(matches[0].get('file'), str):
        raise ValueError(f'Missing FLUX {role} fallback catalog entry.')
    item = matches[0]
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts or not (root / relative).resolve().is_relative_to(root):
        raise ValueError('FLUX text encoder weight escapes the resource package.')
    if not (root / relative).is_file():
        return None, 'unavailable'
    weight = checked_resource(root, item)
    if weight.sha256 is None:
        # Catalogued fallback weights represent exact published resources, even
        # when user-selected models use the cheaper metadata-only policy.
        digest = cached_model_sha256(Path(weight.resolved_file))
        if digest != item.get('sha256'):
            raise ValueError('Bundled generation resource differs from its manifest.')
        weight = replace(weight, sha256=digest)
    selected = resolve_file(weight.path, role, name, folder, root, weight_status='fallback')
    return replace(selected, weight=weight), 'fallback'


def verify_selection(selection):
    if selection is None:
        return
    verify_weight_file(selection.weight, f'FLUX {selection.role}')
    _, size, digest = _read_config(Path(selection.config_path), selection.role)
    if size != selection.config_size or digest != selection.config_sha256:
        raise RuntimeError('FLUX text encoder configuration changed during generation.')


def repair_embedded_t5(model):
    """Restore T5's shared embedding alias after Diffusers single-file loading."""
    from model_loading import require_materialized_component
    if type(model).__name__ != 'T5EncoderModel':
        raise ValueError('Embedded FLUX T5 must be a T5EncoderModel.')
    shared = getattr(model, 'shared', None)
    encoder = getattr(model, 'encoder', None)
    embedded = getattr(encoder, 'embed_tokens', None)
    if shared is None or embedded is None or shared.weight.shape != embedded.weight.shape:
        raise ValueError('Embedded FLUX T5 has an invalid shared embedding contract.')
    repaired = bool(embedded.weight.is_meta and not shared.weight.is_meta)
    if repaired:
        encoder.set_input_embeddings(shared)
    require_materialized_component(model, 'FLUX.text_encoder_2')
    return 'repaired-shared-alias' if repaired else 'already-materialized'


def plan(shapes, expected, role):
    mapping = {}
    for source in shapes:
        target = source
        if role == 'text_encoder' and source.startswith('text_model.') and source not in expected:
            target = source[len('text_model.'):]
        if target in mapping:
            raise ValueError(f'FLUX {role} has duplicate tensors for {target}.')
        mapping[target] = source
    missing, unexpected = set(expected) - set(mapping), set(mapping) - set(expected)
    if missing or unexpected:
        raise ValueError(f'FLUX {role} has missing or unexpected tensors: {sorted(missing)[:5]}, {sorted(unexpected)[:5]}.')
    for key, shape in expected.items():
        if tuple(shapes[mapping[key]]) != tuple(shape):
            raise ValueError(f'FLUX {role} tensor shape mismatch: {key}.')
    return mapping


def plan_scaling(shapes, dtypes, role):
    """Recognize the public legacy ComfyUI scalar-weight FP8 contract only."""
    marker = 'scaled_fp8'
    scale_names = {name for name in shapes if name.endswith('.scale_weight')}
    if marker not in shapes:
        if scale_names:
            raise ValueError('FLUX FP8 weight scales require the scaled_fp8 marker.')
        return shapes, {}
    if role != 'text_encoder_2' or tuple(shapes[marker]) != (0,) or dtypes.get(marker) != 'F8_E4M3':
        raise ValueError('Unsupported FLUX scaled_fp8 marker.')
    if not scale_names:
        raise ValueError('FLUX scaled_fp8 requires scalar weight scales.')
    scales = {}
    for name in scale_names:
        weight = name[:-len('.scale_weight')] + '.weight'
        if (weight not in shapes or tuple(shapes[name]) not in ((), (1,))
                or dtypes.get(name) != 'F32' or dtypes.get(weight) != 'F8_E4M3'):
            raise ValueError(f'Invalid FLUX FP8 weight scale: {name}.')
        scales[weight] = name
    for name, dtype in dtypes.items():
        if name != marker and dtype == 'F8_E4M3' and name not in scales:
            raise ValueError(f'FLUX FP8 tensor requires a scalar weight scale: {name}.')
    return {name: shape for name, shape in shapes.items() if name != marker and name not in scale_names}, scales


def load(selection, dtype, *, storage='dense'):
    import torch
    from safetensors import safe_open
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
    from model_loading import require_materialized_component

    if storage not in ('dense', 'cpu-stream') or (storage == 'cpu-stream' and selection.role != 'text_encoder_2'):
        raise ValueError('FLUX T5 FP8 storage must be dense or cpu-stream for the T5 encoder.')
    verify_selection(selection)
    config, _, _ = _read_config(Path(selection.config_path), selection.role)
    config_cls, model_cls = ((CLIPTextConfig, CLIPTextModel) if selection.role == 'text_encoder'
                             else (T5Config, T5EncoderModel))
    with torch.device('meta'):
        model = model_cls(config_cls.from_dict(config))
    model.requires_grad_(False)
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(selection.weight.path, framework='pt', device='cpu') as source:
        namespace = selection.namespace
        keys = {key[len(namespace):]: key for key in source.keys() if not namespace or key.startswith(namespace)}
        shapes = {key: tuple(source.get_slice(original).get_shape()) for key, original in keys.items()}
        dtypes = {key: source.get_slice(original).get_dtype() for key, original in keys.items()}
        core_shapes, scales = plan_scaling(shapes, dtypes, selection.role)
        if storage == 'cpu-stream' and not scales and not (namespace and 'F8_E4M3' in dtypes.values()):
            raise ValueError('FLUX T5 cpu-stream storage requires a scaled FP8 checkpoint.')
        if namespace and selection.role == 'text_encoder_2' and 'encoder.embed_tokens.weight' not in core_shapes:
            core_shapes = {**core_shapes, 'encoder.embed_tokens.weight': core_shapes['shared.weight']}
            keys['encoder.embed_tokens.weight'] = keys['shared.weight']
        mapping = plan(core_shapes, expected, selection.role)
        if selection.role == 'text_encoder_2':
            shared_key, embedded_key = keys[mapping['shared.weight']], keys[mapping['encoder.embed_tokens.weight']]
            if shared_key != embedded_key:
                shared = source.get_tensor(shared_key)
                embedded = source.get_tensor(embedded_key)
                if shared.dtype != embedded.dtype or not torch.equal(shared.view(torch.uint8), embedded.view(torch.uint8)):
                    raise ValueError('FLUX T5 shared and encoder embedding weights differ.')
                del shared, embedded
        values, streamed_scales = {}, {}
        for key in expected:
            if key == 'encoder.embed_tokens.weight' and 'shared.weight' in values:
                values[key] = values['shared.weight']
                continue
            tensor = source.get_tensor(keys[mapping[key]])
            encoded = tensor
            allowed = (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn)
            if tensor.dtype not in allowed:
                raise ValueError(f'FLUX text encoder requires F16/BF16/F32 or unscaled E4M3: {key}.')
            tensor = tensor.to(dtype=torch.float32)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'FLUX text encoder tensor is non-finite: {key}.')
            if mapping[key] in scales:
                scale = source.get_tensor(keys[scales[mapping[key]]])
                if scale.dtype != torch.float32 or not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
                    raise ValueError(f'FLUX FP8 weight scale is invalid: {key}.')
                scale = scale.to(dtype=dtype)
                if not bool(torch.isfinite(scale).all()):
                    raise ValueError(f'FLUX FP8 weight scale overflows selected precision: {key}.')
                # Match Comfy's eager dequantization at the selected computation precision.
                tensor = tensor.to(dtype=dtype) * scale
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'FLUX scaled FP8 tensor is non-finite: {key}.')
                if storage == 'cpu-stream':
                    streamed_scales[key] = source.get_tensor(keys[scales[mapping[key]]]).detach().clone()
            elif (storage == 'cpu-stream' and namespace and encoded.dtype == torch.float8_e4m3fn
                  and key.startswith('encoder.block.')
                  and key.endswith('.weight')):
                module = model.get_submodule(key.removesuffix('.weight'))
                if isinstance(module, torch.nn.Linear):
                    streamed_scales[key] = torch.ones((), dtype=torch.float32)
            tensor = tensor.to(dtype=dtype)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'FLUX text encoder tensor overflows selected precision: {key}.')
            values[key] = encoded if key in streamed_scales else tensor
        if selection.role == 'text_encoder_2':
            values['encoder.embed_tokens.weight'] = values['shared.weight']
        if storage == 'cpu-stream' and not streamed_scales:
            raise ValueError('FLUX T5 cpu-stream storage requires at least one FP8 linear weight.')
        model.load_state_dict(values, strict=True, assign=True)
    if streamed_scales:
        from flux_t5_stream import linear_type
        for key, scale in streamed_scales.items():
            if not key.startswith('encoder.block.') or not key.endswith('.weight'):
                raise ValueError(f'FLUX T5 streamed FP8 target is unsupported: {key}.')
            name = key.removesuffix('.weight')
            layer = model.get_submodule(name)
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError(f'FLUX T5 streamed FP8 target is not linear: {name}.')
            parent_name, _, leaf = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, linear_type()(layer, scale, dtype))
    if selection.role == 'text_encoder_2':
        model.tie_weights()
    else:
        embeddings = getattr(getattr(model, 'text_model', model), 'embeddings', None)
        position_ids = getattr(embeddings, 'position_ids', None)
        if position_ids is not None and position_ids.is_meta:
            embeddings.register_buffer('position_ids', torch.arange(model.config.max_position_embeddings).expand((1, -1)),
                                       persistent=False)
    require_materialized_component(model, f'FLUX.{selection.role}')
    verify_selection(selection)
    model._iild_flux_text_encoder_loading = {
        'format': ('comfy-scalar-fp8-e4m3fn' if scales else
                   'unscaled-fp8-e4m3fn' if 'F8_E4M3' in dtypes.values() else 'dense'),
        'scaled_layers': len(scales), 'runtime_weights': 'cpu-stream-fp8' if streamed_scales else 'dense',
        'native_fp8_matmul': False,
        'quantized_source_bytes': sum(math.prod(shapes[mapping[name]]) for name in streamed_scales) if streamed_scales
                                  else sum(math.prod(shapes[name]) for name in scales),
        'largest_expanded_layer_bytes': max((math.prod(shapes[mapping[name]]) * torch.empty((), dtype=dtype).element_size()
                                             for name in streamed_scales), default=0) if streamed_scales else
                                        max((math.prod(shapes[name]) * torch.empty((), dtype=dtype).element_size()
                                             for name in scales), default=0),
    }
    return model.eval()
