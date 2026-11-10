"""Report bounded progress only after MOSS tensor work has completed."""
from __future__ import annotations

import math
import time


_BYTE_WIDTH = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2,
    'U32': 4, 'I32': 4, 'F32': 4,
    'U64': 8, 'I64': 8, 'F64': 8,
}


class TensorLoadingProgress:
    """Count planned tensor bytes once each, after their processing succeeds."""

    def __init__(self, header, names, component, report, *, clock=time.monotonic):
        self._remaining = {}
        for name in set(names):
            entry = header[name]
            width = _BYTE_WIDTH.get(entry['dtype'])
            if width is None:
                raise ValueError(f'Unsupported MOSS tensor dtype: {entry["dtype"]}.')
            self._remaining[name] = math.prod(entry['shape']) * width
        self._total = sum(self._remaining.values())
        self._done = 0
        self._component = component
        self._report = report
        self._clock = clock
        self._last_report = None

    def completed(self, *names):
        if not names or len(set(names)) != len(names) or any(name not in self._remaining for name in names):
            raise ValueError('MOSS tensor progress must refer to unfinished planned tensors.')
        advance = sum(self._remaining.pop(name) for name in names)
        if not advance:
            return
        self._done += advance
        now = self._clock()
        if self._last_report is None or now - self._last_report >= 0.5 or self._done == self._total:
            self._last_report = now
            self._report(self._component, self._done, self._total)

    def is_pending(self, name):
        return name in self._remaining


def attach_quantized_forward_progress(model, component, report):
    """Count each INT8 module once per model forward, only after it returns."""
    modules = [(name, module) for name, module in model.named_modules()
               if getattr(module, '_iild_int8', None) is not None]
    if not modules:
        return
    header = {name: {'shape': tuple(module._iild_int8.shape), 'dtype': 'I8'}
              for name, module in modules}
    current = [None]

    def reset(_module, _inputs):
        current[0] = TensorLoadingProgress(header, header, component, report)

    def callback(name):
        def done(_module, _inputs, _output):
            counter = current[0]
            if counter is not None and counter.is_pending(name):
                counter.completed(name)
        return done

    model.register_forward_pre_hook(reset)
    for name, module in modules:
        module.register_forward_hook(callback(name))
