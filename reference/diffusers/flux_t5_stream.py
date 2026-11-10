"""CPU-resident scalar FP8 T5 linear layers for bounded accelerator memory."""
from functools import lru_cache
import json
from pathlib import Path
import struct


def validate_lora(selection):
    """Allow only explicitly component-prefixed FLUX LoRA tensors."""
    if selection is None:
        return
    from lora import verify_lora_identity
    if not selection.is_local or selection.local_file is None:
        raise ValueError('FLUX T5 streaming requires a local, component-prefixed LoRA file.')
    verify_lora_identity(selection)
    path = Path(selection.local_file.path)
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid FLUX LoRA safetensors header.')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid FLUX LoRA safetensors header.')
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError('Invalid FLUX LoRA safetensors header.')
    keys = [key for key in header if key != '__metadata__']
    if not keys or any(not key.startswith(('transformer.', 'text_encoder.')) for key in keys):
        raise ValueError('FLUX T5 streaming requires LoRA keys explicitly targeting transformer or text_encoder; T5 and ambiguous keys are unsupported.')
    verify_lora_identity(selection)


def validate_lora_pipeline(pipeline):
    components = getattr(pipeline, '_lora_loadable_modules', None)
    if not isinstance(components, (tuple, list)) or not components or not set(components).issubset({'transformer', 'text_encoder'}):
        raise ValueError('This FLUX LoRA loader may target the streamed T5 encoder and is unsupported.')


@lru_cache(maxsize=1)
def linear_type():
    import torch

    class StreamedT5Linear(torch.nn.Linear):
        def __init__(self, source, scale, compute_dtype):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = source.in_features, source.out_features
            self.bias = source.bias
            # T5 examines wo.weight.dtype before calling the layer. This
            # zero-stride placeholder exposes the compute dtype without a full
            # dense allocation; forward never reads its value.
            self.weight = torch.nn.Parameter(torch.empty((), dtype=compute_dtype).expand(source.weight.shape),
                                             requires_grad=False)
            self.register_buffer('_iild_fp8_weight', source.weight.detach(), persistent=False)
            self.register_buffer('_iild_fp8_scale', scale.detach().clone(), persistent=False)

        def _apply(self, fn, recurse=True):
            weight = self._parameters.pop('weight')
            encoded = self._buffers.pop('_iild_fp8_weight')
            scale = self._buffers.pop('_iild_fp8_scale')
            try:
                super()._apply(fn, recurse=recurse)
                scalar = fn(weight[0, 0])
                weight = torch.nn.Parameter(scalar.expand((self.out_features, self.in_features)), requires_grad=False)
                return self
            finally:
                self._parameters['weight'] = weight
                self._buffers['_iild_fp8_weight'] = encoded
                self._buffers['_iild_fp8_scale'] = scale

        def forward(self, inputs):
            weight = self._iild_fp8_weight.to(dtype=inputs.dtype) * self._iild_fp8_scale.to(dtype=inputs.dtype)
            return torch.nn.functional.linear(inputs, weight.to(device=inputs.device), self.bias)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            raise RuntimeError('A streamed FLUX T5 layer cannot be serialized as a dense checkpoint.')

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            from streamed_linear import reject_weight_reload
            reject_weight_reload(state_dict, prefix)
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                          missing_keys, unexpected_keys, error_msgs)

    return StreamedT5Linear
