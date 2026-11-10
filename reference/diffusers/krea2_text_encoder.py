"""Load the SDK's Qwen3-VL language and optional vision encoder resources."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

from generation_defaults import checked_resource, read_defaults
from weight_files import LocalWeightFile, file_sha256, file_signature, verify_weight_file

PREFIX = 'model.language_model.'
EXPECTED_CONFIG = {
    'model_type': 'qwen3_vl',
    'text_config': {'model_type': 'qwen3_vl_text', 'vocab_size': 151936, 'hidden_size': 2560,
                    'intermediate_size': 9728, 'num_hidden_layers': 36, 'num_attention_heads': 32,
                    'num_key_value_heads': 8, 'head_dim': 128, 'rope_theta': 5000000,
                    'rope_scaling': {'mrope_interleaved': True, 'mrope_section': [24, 20, 20],
                                     'rope_type': 'default'}},
}


@dataclass(frozen=True)
class Selection:
    weight: LocalWeightFile
    config_path: str
    config_sha256: str

    def metadata(self):
        return {'status': 'fallback', 'class': 'Qwen3VLTextModel', 'weight': self.weight.path,
                'sha256': self.weight.sha256, 'size_bytes': self.weight.size_bytes,
                'validation': 'sha256' if self.weight.sha256 is not None else 'metadata',
                'config': self.config_path, 'config_sha256': self.config_sha256}


def _contains_expected(value, expected):
    return (isinstance(value, dict) and all(key in value and
            (_contains_expected(value[key], item) if isinstance(item, dict) else value[key] == item)
            for key, item in expected.items()))


def _read_config(path):
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Krea 2 fallback text encoder requires a bounded local configuration.')
    before = file_signature(path)
    content = path.read_bytes()
    digest = file_sha256(path)
    if file_signature(path) != before:
        raise RuntimeError('Krea 2 text encoder configuration changed while reading.')
    config = json.loads(content)
    if not _contains_expected(config, EXPECTED_CONFIG):
        raise ValueError('Krea 2 fallback configuration does not match Qwen3-VL-4B.')
    return config, digest


def _single_file_has_encoder(path):
    path = Path(path)
    with path.open('rb') as source:
        size = source.read(8)
        if len(size) != 8:
            raise ValueError('Invalid safetensors header while checking the Krea 2 encoder.')
        length = struct.unpack('<Q', size)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header while checking the Krea 2 encoder.')
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors header while checking the Krea 2 encoder.')
    return any(name.startswith(('text_encoders.qwen3vl_4b.transformer.', 'text_encoder.', PREFIX))
               for name in header)


def resolve(args, index, folder):
    if (args.pipeline_class or index['_class_name']) != 'Krea2Pipeline':
        return None, 'not-configured-for-pipeline'
    component = index.get('text_encoder')
    if args.source_kind == 'single-file' and _single_file_has_encoder(args.model):
        return None, 'model'
    component_dir = Path(folder) / 'text_encoder'
    if component not in (None, [None, None]) and any(
            path.is_file() and (path.suffix in ('.safetensors', '.bin', '.pt', '.ckpt')
                                or path.name.endswith('.index.json')) for path in component_dir.glob('*')):
        return None, 'model'
    if component not in (None, [None, None], ['transformers', 'Qwen3VLModel']):
        raise ValueError('Krea 2 fallback requires a Qwen3VLModel text encoder declaration.')
    inputs = getattr(args, 'inputs', {})
    if ('prompt_embeds' in inputs and (inputs.get('guidance_scale', 0) == 0
                                      or 'negative_prompt_embeds' in inputs)):
        return None, 'precomputed-embeddings'
    root, manifest = read_defaults(getattr(args, 'generation_resources', None))
    entries = manifest.get('fallback_llms', [])
    if not isinstance(entries, list) or len(entries) > 64 or any(not isinstance(item, dict) for item in entries):
        raise ValueError('Invalid fallback text encoder catalog.')
    found = [item for item in entries if isinstance(item.get('families'), list) and 'krea2' in item['families']]
    if len(found) != 1 or found[0].get('class_name') != 'Qwen3VLForConditionalGeneration':
        raise ValueError('Add or sync the verified Qwen3-VL-4B generation resources for Krea2.')
    item = found[0]
    weight = checked_resource(root, item)
    config_item = item.get('config')
    if not isinstance(config_item, dict) or not isinstance(config_item.get('file'), str):
        raise ValueError('The Krea 2 fallback catalog requires a local encoder configuration.')
    relative = Path(config_item['file'])
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError('The Krea 2 text encoder configuration escapes the resource package.')
    _, digest = _read_config(path)
    if path.stat().st_size != config_item.get('size') or digest != config_item.get('sha256'):
        raise ValueError('Krea 2 fallback text encoder configuration differs from its manifest.')
    return Selection(weight, str(path), digest), 'fallback'


def verify_selection(selection):
    if selection is None:
        return
    verify_weight_file(selection.weight, 'Krea 2 fallback text encoder')
    path = Path(selection.config_path)
    try:
        _, digest = _read_config(path)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError('Krea 2 fallback text encoder configuration changed.') from error
    if digest != selection.config_sha256:
        raise RuntimeError('Krea 2 fallback text encoder configuration changed.')


def load_language_weights(model, weight, dtype, torch):
    """Load the exact language namespace, ignoring the separate vision branch."""
    from safetensors import safe_open
    from model_loading import require_materialized_component
    verify_weight_file(weight, 'Krea 2 fallback text encoder')
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(weight.path, framework='pt', device='cpu') as source:
        actual = {name[len(PREFIX):]: name for name in source.keys() if name.startswith(PREFIX)}
        if set(actual) != set(expected):
            raise ValueError('Krea 2 text encoder has missing or unexpected language tensors.')
        for name, shape in expected.items():
            if tuple(source.get_slice(actual[name]).get_shape()) != shape:
                raise ValueError(f'Krea 2 text encoder tensor shape mismatch: {name}.')
        values = {}
        for name, key in actual.items():
            tensor = source.get_tensor(key)
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Krea 2 text encoder tensor is non-floating or non-finite: {name}.')
            tensor = tensor.to(dtype=dtype)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Krea 2 text encoder tensor overflows selected precision: {name}.')
            values[name] = tensor
        model.load_state_dict(values, strict=True, assign=True)
    # Rotary inverse frequencies are deterministic non-persistent buffers, so
    # they do not appear in the checkpoint and must be rebuilt after meta init.
    rotary = getattr(model, 'rotary_emb', None)
    if rotary is not None and getattr(getattr(rotary, 'inv_freq', None), 'is_meta', False):
        with torch.device('cpu'):
            model.rotary_emb = type(rotary)(model.config)
    require_materialized_component(model, 'Krea2.text_encoder')
    verify_weight_file(weight, 'Krea 2 fallback text encoder')
    model.eval()
    return model


def load_full_weights(model, weight, dtype, torch):
    """Strictly load both Qwen3-VL branches from the existing split resource."""
    from safetensors import safe_open
    from model_loading import require_materialized_component
    verify_weight_file(weight, 'Krea 2 fallback multimodal encoder')
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(weight.path, framework='pt', device='cpu') as source:
        prefixes = ('model.language_model.', 'model.visual.')
        actual = {name[len('model.'):]: name for name in source.keys() if name.startswith(prefixes)}
        if set(actual) != set(expected):
            raise ValueError('Krea 2 multimodal encoder has missing or unexpected language or vision tensors.')
        for name, shape in expected.items():
            if tuple(source.get_slice(actual[name]).get_shape()) != shape:
                raise ValueError(f'Krea 2 multimodal encoder tensor shape mismatch: {name}.')
        values = {}
        for name, key in actual.items():
            value = source.get_tensor(key)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Krea 2 multimodal encoder tensor is non-floating or non-finite: {name}.')
            value = value.to(dtype=dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Krea 2 multimodal encoder tensor overflows selected precision: {name}.')
            values[name] = value
        model.load_state_dict(values, strict=True, assign=True)
    # Both deterministic non-persistent rotary buffers are absent from the file.
    with torch.device('cpu'):
        text = model.language_model.rotary_emb
        if text.inv_freq.is_meta:
            model.language_model.rotary_emb = type(text)(model.language_model.config)
        vision = model.visual.rotary_pos_emb
        if vision.inv_freq.is_meta:
            model.visual.rotary_pos_emb = type(vision)(vision.dim, vision.theta)
    require_materialized_component(model, 'Krea2.multimodal_encoder')
    verify_weight_file(weight, 'Krea 2 fallback multimodal encoder')
    model.eval()
    return model


def load_full(selection, dtype):
    """Load the resident Qwen3-VL visual and text models without remote code."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLModel
    verify_selection(selection)
    config, _ = _read_config(Path(selection.config_path))
    if not isinstance(config.get('vision_config'), dict):
        raise ValueError('Krea 2 fallback configuration lacks Qwen3-VL vision settings.')
    with torch.device('meta'):
        model = Qwen3VLModel(Qwen3VLConfig.from_dict(config))
    load_full_weights(model, selection.weight, dtype, torch)
    verify_selection(selection)
    return model


def load(selection, dtype):
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLTextModel
    verify_selection(selection)
    config, _ = _read_config(Path(selection.config_path))
    config = Qwen3VLConfig.from_dict(config).text_config
    with torch.device('meta'):
        model = Qwen3VLTextModel(config)
    load_language_weights(model, selection.weight, dtype, torch)
    verify_selection(selection)
    return model
