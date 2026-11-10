"""Portable inference for CPU-resident Comfy tensorwise INT8 ConvRot weights."""
from functools import lru_cache
import warnings


@lru_cache(maxsize=32)
def hadamard(size, device, dtype):
    import torch
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError('ConvRot group size must be a power of four.')
    base = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                         [1, -1, 1, 1], [-1, 1, 1, 1]], device=device, dtype=dtype)
    result = base
    while result.shape[0] < size:
        result = torch.kron(result, base)
    return result / size ** 0.5


def rotate(value, group):
    original = value.shape
    if original[-1] % group:
        raise ValueError('ConvRot group size must divide the feature dimension.')
    matrix = hadamard(group, str(value.device), value.dtype)
    return (value.reshape(-1, group) @ matrix).reshape(original)


def embedding(weight, scale, indices, group, dtype):
    import torch
    indices_cpu = indices.to(device='cpu')
    values = torch.nn.functional.embedding(indices_cpu, weight).float()
    factors = (torch.nn.functional.embedding(indices_cpu, scale).float()
               if scale.ndim >= 2 else scale.float())
    return rotate(values * factors, group).to(device=indices.device, dtype=dtype)


def mps_int8_mm(activations, weight):
    """Compute signed INT8 products with exact FP32 partials and INT32 sums."""
    import torch
    if (activations.device.type != 'mps' or activations.dtype != torch.int8
            or weight.dtype != torch.int8 or activations.ndim != 2 or weight.ndim != 2
            or activations.shape[1] != weight.shape[1]):
        raise ValueError('MPS INT8 matrix inputs must have compatible two-dimensional shapes.')
    result = torch.zeros((activations.shape[0], weight.shape[0]), dtype=torch.int32, device=activations.device)
    # Each INT8 product is at most 16384. A 1024-term sum is at most 2**24,
    # so every FP32 product and partial sum remains an exactly representable integer.
    for start in range(0, activations.shape[1], 1024):
        end = min(start + 1024, activations.shape[1])
        rows = activations[:, start:end].to(dtype=torch.float32)
        columns = weight[:, start:end].to(device=activations.device, dtype=torch.float32).T
        result.add_(torch.mm(rows, columns).to(dtype=torch.int32))
    return result


def linear(inputs, weight, scale, bias, group):
    import torch
    rotated = rotate(inputs, group)
    shape = rotated.shape
    rows = rotated.reshape(-1, shape[-1])
    if rows.shape[0] == 0:
        return inputs.new_empty((*shape[:-1], weight.shape[0]))
    row_scale = (rows.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    divisor = row_scale.to(rows.dtype)
    divisor = torch.where(divisor == 0, torch.full_like(divisor, torch.finfo(rows.dtype).tiny), divisor)
    quantized = (rows / divisor).round().clamp(-128, 127).to(torch.int8)
    accum = None
    if inputs.device.type == 'mps':
        try:
            accum = mps_int8_mm(quantized, weight).to('cpu').float()
        except RuntimeError as error:
            warnings.warn(f'MPS INT8 acceleration is unavailable; using the CPU reference: {error}', RuntimeWarning)
    if accum is None:
        # Keep the existing reference path for CPU, other devices and MPS fallback.
        quantized = quantized.to('cpu').contiguous()
        transposed = weight.T.contiguous()
        pad_k = (-quantized.shape[1]) % 8
        pad_n = (-transposed.shape[1]) % 8
        if pad_k or pad_n:
            quantized = torch.nn.functional.pad(quantized, (0, pad_k))
            transposed = torch.nn.functional.pad(transposed, (0, pad_n, 0, pad_k))
        accum = torch._int_mm(quantized, transposed)[:, :weight.shape[0]].float()
    factors = row_scale.to('cpu') * scale.reshape(1, -1)
    result = (accum * factors).to(device=inputs.device, dtype=inputs.dtype)
    if bias is not None:
        result = result + bias.to(device=inputs.device, dtype=inputs.dtype)
    return result.reshape(*shape[:-1], weight.shape[0])


@lru_cache(maxsize=1)
def module_types():
    import torch

    class CpuInt8Mixin:
        def _set_quant(self, weight, scale, group, dtype):
            self.group_size = group
            self.weight = torch.nn.Parameter(torch.empty((), dtype=dtype).expand(weight.shape), requires_grad=False)
            self.register_buffer('_iild_int8', weight, persistent=False)
            self.register_buffer('_iild_scale', scale, persistent=False)

        def _apply(self, fn, recurse=True):
            weight = self._parameters.pop('weight')
            encoded = self._buffers.pop('_iild_int8')
            scale = self._buffers.pop('_iild_scale')
            try:
                super()._apply(fn, recurse=recurse)
                scalar = fn(weight[0, 0])
                weight = torch.nn.Parameter(scalar.expand(encoded.shape), requires_grad=False)
                return self
            finally:
                self._parameters['weight'] = weight
                self._buffers['_iild_int8'] = encoded
                self._buffers['_iild_scale'] = scale

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            raise RuntimeError('A streamed MOSS INT8 layer cannot be serialized as a dense checkpoint.')

    class CpuInt8Linear(CpuInt8Mixin, torch.nn.Linear):
        def __init__(self, source, weight, scale, group, dtype):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = source.in_features, source.out_features
            self.bias = source.bias
            self._set_quant(weight, scale, group, dtype)

        def forward(self, inputs):
            return linear(inputs, self._iild_int8, self._iild_scale, self.bias, self.group_size)

    class CpuInt8Embedding(CpuInt8Mixin, torch.nn.Embedding):
        def __init__(self, source, weight, scale, group, dtype):
            torch.nn.Module.__init__(self)
            self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
            self.padding_idx = source.padding_idx
            self.max_norm, self.norm_type = source.max_norm, source.norm_type
            self.scale_grad_by_freq, self.sparse = source.scale_grad_by_freq, source.sparse
            self._set_quant(weight, scale, group, dtype)

        def forward(self, indices):
            return embedding(self._iild_int8, self._iild_scale, indices, self.group_size, self.weight.dtype)

    return CpuInt8Linear, CpuInt8Embedding
