"""Linear weights kept on CPU and transferred only for their active call."""
from functools import lru_cache


def reject_weight_reload(state_dict, prefix):
    """Do not accept a dense replacement while forward reads retained weights."""
    if prefix + 'weight' in state_dict:
        raise RuntimeError('cannot replace a streamed weight via load_state_dict; reload the original checkpoint.')


@lru_cache(maxsize=1)
def streamed_linear_type():
    import torch

    class StreamedLinear(torch.nn.Linear):
        def __init__(self, source, dtype):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = source.in_features, source.out_features
            self.bias = source.bias
            self.register_buffer('_iild_streamed_weight', source.weight.detach(), persistent=False)
            self.weight = torch.nn.Parameter(torch.empty((), dtype=dtype).expand(source.weight.shape),
                                             requires_grad=False)

        def _apply(self, fn, recurse=True):
            weight = self._parameters.pop('weight')
            streamed = self._buffers.pop('_iild_streamed_weight')
            try:
                super()._apply(fn, recurse=recurse)
                weight = torch.nn.Parameter(fn(weight[0, 0]).expand(streamed.shape), requires_grad=False)
                return self
            finally:
                self._parameters['weight'] = weight
                self._buffers['_iild_streamed_weight'] = streamed

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            raise RuntimeError('A streamed transformer cannot be serialized as dense weights.')

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            reject_weight_reload(state_dict, prefix)
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                          missing_keys, unexpected_keys, error_msgs)

        def forward(self, inputs):
            weight = self._iild_streamed_weight.to(device=inputs.device, dtype=inputs.dtype)
            return torch.nn.functional.linear(inputs, weight, self.bias)

    return StreamedLinear
