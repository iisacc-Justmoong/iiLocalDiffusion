"""Load original Krea 2 transformer safetensors into Diffusers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar

from weight_files import file_signature

# Diffusers/PEFT internally inspect a model state dict during adapter loading.
# This context is local to the calling execution context and must not escape it.
_PEFT_STATE_INSPECTION = ContextVar('iild_krea2_peft_state_inspection', default=frozenset())

# Original Krea 2 and Comfy-Org names; Diffusers' conversion contract is
# documented in its Apache-2.0 single_file_utils at revision 86ef420b1736.
PREFIXES = {
    'first.': 'img_in.', 'tmlp.0.': 'time_embed.linear_1.', 'tmlp.2.': 'time_embed.linear_2.',
    'tproj.1.': 'time_mod_proj.', 'txtmlp.0.scale': 'txt_in.norm.weight',
    'txtmlp.1.': 'txt_in.linear_1.', 'txtmlp.3.': 'txt_in.linear_2.',
    'txtfusion.': 'text_fusion.', 'blocks.': 'transformer_blocks.',
    'last.linear.': 'final_layer.linear.', 'last.norm.scale': 'final_layer.norm.weight',
    'last.modulation.lin': 'final_layer.scale_shift_table',
}
BLOCK_NAMES = {
    '.attn.wq.': '.attn.to_q.', '.attn.wk.': '.attn.to_k.', '.attn.wv.': '.attn.to_v.',
    '.attn.wo.': '.attn.to_out.0.', '.attn.gate.': '.attn.to_gate.',
    '.attn.qknorm.qnorm.scale': '.attn.norm_q.weight',
    '.attn.qknorm.knorm.scale': '.attn.norm_k.weight', '.mlp.': '.ff.',
    '.prenorm.scale': '.norm1.weight', '.postnorm.scale': '.norm2.weight',
    '.mod.lin': '.scale_shift_table',
}


def quantized_layers(shapes, metadata, *, matmul_policy=None):
    """Validate the scalar E4M3 metadata used by published Comfy checkpoints."""
    encoded = (metadata or {}).get('_quantization_metadata')
    if encoded is None:
        if any(name.endswith('.weight_scale') for name in shapes):
            raise ValueError('Krea 2 scaled weights require quantization metadata.')
        return {}
    if not isinstance(encoded, str) or len(encoded) > 1024 * 1024:
        raise ValueError('Krea 2 quantization metadata is invalid or too large.')
    try:
        description = json.loads(encoded)
    except ValueError as error:
        raise ValueError('Krea 2 quantization metadata is invalid.') from error
    if not isinstance(description, dict) or set(description) != {'layers'} or not isinstance(description['layers'], dict):
        raise ValueError('Krea 2 quantization metadata must contain a layer map.')
    result = {}
    required = 0
    for name, options in description['layers'].items():
        if (not isinstance(name, str) or not name or not isinstance(options, dict)
                or options.get('format') != 'float8_e4m3fn'
                or set(options) - {'format', 'full_precision_matrix_mult'}
                or ('full_precision_matrix_mult' in options and type(options['full_precision_matrix_mult']) is not bool)):
            raise ValueError('Krea 2 quantization metadata uses an unsupported layer format.')
        weight, scale = name + '.weight', name + '.weight_scale'
        if weight not in shapes or scale not in shapes or tuple(shapes[scale]) not in ((), (1,)):
            raise ValueError(f'Krea 2 FP8 layer requires a scalar weight scale: {name}.')
        result[weight] = scale
        required += options.get('full_precision_matrix_mult', False)
    if not result:
        raise ValueError('Krea 2 quantization metadata contains no layers.')
    if matmul_policy is not None:
        matmul_policy.update(required=required, not_required=len(result) - required)
    return result


def plan(shapes, expected, *, metadata=None):
    """Map names and shapes before reading any tensor payloads."""
    quantized = quantized_layers(shapes, metadata)
    scale_names = set(quantized.values())
    result, targets = {}, set()
    for source, shape in shapes.items():
        if source in scale_names:
            continue
        name = source.removeprefix('model.diffusion_model.')
        for old, new in PREFIXES.items():
            if name.startswith(old):
                name = new + name[len(old):]
                break
        for old, new in BLOCK_NAMES.items():
            name = name.replace(old, new)
        if name in targets:
            raise ValueError(f'Krea 2 checkpoint has duplicate target {name}.')
        targets.add(name)
        reshape = name.startswith('transformer_blocks.') and name.endswith('.scale_shift_table')
        target_shape = tuple(expected.get(name, ()))
        source_shape = tuple(shape)
        valid = (len(source_shape) == 1 and len(target_shape) == 2 and target_shape[0] == 6
                 and source_shape[0] == target_shape[0] * target_shape[1]) if reshape else source_shape == target_shape
        if name not in expected or not valid:
            raise ValueError(f'Krea 2 checkpoint has unexpected tensor or shape: {source}.')
        result[source] = (name, target_shape if reshape else None)
    if targets != set(expected):
        missing = sorted(set(expected) - targets)
        raise ValueError(f'Krea 2 checkpoint has missing tensors: {missing[:5]}.')
    return result


@lru_cache(maxsize=1)
def streamed_linear_type():
    """A linear layer retaining FP8 on CPU and materializing one weight per call."""
    import torch

    class StreamedLinear(torch.nn.Linear):
        def __init__(self, source, scale):
            torch.nn.Module.__init__(self)
            self.in_features = source.in_features
            self.out_features = source.out_features
            self.weight = source.weight
            self.bias = source.bias
            self.register_buffer('_iild_fp8_scale', scale.detach().clone(), persistent=False)

        def _apply(self, fn, recurse=True):
            # Keep compressed CPU storage intact when the enclosing pipeline
            # moves to an accelerator; ordinary bias/adapter tensors still move.
            weight = self._parameters.pop('weight')
            scale = self._buffers.pop('_iild_fp8_scale')
            try:
                return super()._apply(fn, recurse=recurse)
            finally:
                self._parameters['weight'] = weight
                self._buffers['_iild_fp8_scale'] = scale

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if self not in _PEFT_STATE_INSPECTION.get():
                raise RuntimeError('A Krea 2 streamed FP8 transformer cannot be serialized as dense weights.')
            super()._save_to_state_dict(destination, prefix, keep_vars)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            from streamed_linear import reject_weight_reload
            reject_weight_reload(state_dict, prefix)
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                          missing_keys, unexpected_keys, error_msgs)

        def forward(self, inputs):
            weight = self.weight.to(dtype=inputs.dtype)
            weight = weight * self._iild_fp8_scale.to(dtype=inputs.dtype)
            return torch.nn.functional.linear(inputs, weight.to(device=inputs.device), self.bias)

    return StreamedLinear


@contextmanager
def adapter_loading(model, selection, dtype):
    """Give PEFT a temporary compute-dtype view without expanding FP8 weights."""
    if selection is None:
        yield
        return
    import torch
    from safetensors import safe_open

    layers = [layer for layer in model.modules() if isinstance(layer, streamed_linear_type())]
    if not layers:
        yield
        return
    local_file = getattr(selection, 'local_file', None)
    if local_file is None:
        raise ValueError('Krea 2 streamed FP8 requires a local ordinary LoRA file.')
    with safe_open(local_file.path, framework='pt', device='cpu') as source:
        names = source.keys()
        suffixes = ('.lora_A.weight', '.lora_B.weight', '.lora_down.weight', '.lora_up.weight', '.alpha')
        if not names or any(not name.endswith(suffixes) for name in names):
            raise ValueError('Krea 2 streamed FP8 supports ordinary LoRA tensors only.')
    originals = []
    try:
        for layer in layers:
            weight = layer._parameters['weight']
            # PEFT uses this dtype and shape for adapter initialization. The
            # zero-stride view allocates one scalar and is never used to infer.
            view = torch.empty((), dtype=dtype, device='cpu').expand(weight.shape)
            layer._parameters['weight'] = torch.nn.Parameter(view, requires_grad=False)
            originals.append((layer, weight))
        token = _PEFT_STATE_INSPECTION.set(frozenset(layers))
        try:
            yield
        finally:
            _PEFT_STATE_INSPECTION.reset(token)
    finally:
        for layer, weight in originals:
            layer._parameters['weight'] = weight


def load(checkpoint, configuration, dtype, *, storage='dense'):
    """Strictly load dense or scalar-scaled FP8 weights without writing a copy."""
    import torch
    from diffusers import Krea2Transformer2DModel
    from safetensors import safe_open
    from model_loading import require_materialized_component

    if storage not in ('dense', 'cpu-stream'):
        raise ValueError('Krea 2 FP8 storage must be dense or cpu-stream.')
    checkpoint, configuration = Path(checkpoint), Path(configuration)
    config_path = configuration / 'config.json'
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError('Krea 2 single-file loading requires a bounded transformer/config.json.')
    before_config, before_weight = file_signature(config_path), file_signature(checkpoint)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('_class_name') != 'Krea2Transformer2DModel' or 'auto_map' in config:
        raise ValueError('Krea 2 transformer configuration must declare the built-in model class.')
    with torch.device('meta'):
        model = Krea2Transformer2DModel.from_config(config)
    model.requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        metadata = source.metadata() or {}
        matmul_policy = {}
        quantized = quantized_layers(shapes, metadata, matmul_policy=matmul_policy)
        if storage == 'cpu-stream' and not quantized:
            raise ValueError('Krea 2 cpu-stream storage requires a scaled FP8 checkpoint.')
        mapping = plan(shapes, expected, metadata=metadata)
        values, streamed_scales = {}, {}
        for source_name, (target, shape) in mapping.items():
            tensor = source.get_tensor(source_name)
            if source_name in quantized:
                if tensor.dtype != torch.float8_e4m3fn:
                    raise ValueError(f'Krea 2 FP8 metadata does not match tensor dtype: {source_name}.')
                scale = source.get_tensor(quantized[source_name])
                if scale.dtype != torch.float32 or not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
                    raise ValueError(f'Krea 2 FP8 weight scale is invalid: {source_name}.')
                # Comfy's eager FP8 path casts both operands to compute dtype
                # before multiplying. This deliberately materializes dense weights.
                expanded = tensor.to(dtype=dtype) * scale.to(dtype=dtype)
                if not bool(torch.isfinite(expanded).all()):
                    raise ValueError(f'Krea 2 single-file tensor overflows selected precision: {source_name}.')
                if storage == 'cpu-stream':
                    streamed_scales[target] = scale.detach().clone()
                else:
                    tensor = expanded
            elif tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f'Krea 2 single-file tensor requires unquantized floating weights: {source_name}.')
            if source_name not in quantized and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Krea 2 single-file tensor is non-finite: {source_name}.')
            if source_name not in quantized:
                tensor = tensor.to(dtype=dtype)
            if source_name not in quantized and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Krea 2 single-file tensor overflows selected precision: {source_name}.')
            values[target] = tensor.reshape(shape) if shape else tensor
        model.load_state_dict(values, strict=True, assign=True)
    for target, scale in streamed_scales.items():
        if not target.endswith('.weight'):
            raise ValueError(f'Krea 2 streamed FP8 target is not a linear weight: {target}.')
        module_name = target.removesuffix('.weight')
        if not module_name.startswith(('transformer_blocks.', 'text_fusion.')):
            raise ValueError(f'Krea 2 streamed FP8 does not support this layer: {module_name}.')
        layer = model.get_submodule(module_name)
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f'Krea 2 streamed FP8 target is not a linear layer: {module_name}.')
        parent_name, _, leaf = module_name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, streamed_linear_type()(layer, scale))
    require_materialized_component(model, 'Krea2.transformer')
    if file_signature(config_path) != before_config or file_signature(checkpoint) != before_weight:
        raise RuntimeError('Krea 2 single-file model or transformer configuration changed while loading.')
    model._iild_single_file_loading = {
        'format': 'comfy-scalar-fp8-e4m3fn' if quantized else 'original-dense',
        'quantized_layers': len(quantized),
        'runtime_weights': 'cpu-stream-fp8' if storage == 'cpu-stream' else 'dense',
        'native_fp8_matmul': False,
        'checkpoint_full_precision_matmul_required_layers': matmul_policy.get('required', 0),
        'checkpoint_full_precision_matmul_not_required_layers': matmul_policy.get('not_required', 0),
        'quantized_matmul_execution': 'dequantized-runtime-dtype' if quantized else None,
        'quantized_source_bytes': sum(math.prod(shapes[name]) for name in quantized),
        'largest_expanded_layer_bytes': max((math.prod(shapes[name]) * torch.empty((), dtype=dtype).element_size()
                                             for name in quantized), default=0),
        'dequantization': 'weight and scalar cast to runtime dtype before multiplication' if quantized else None,
    }
    return model.eval()
