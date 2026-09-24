"""Base-coordinate projection and explicit scale/zero-point normalization.

No tensor file parsing or family routing lives here. These operations are
deterministic approximations, not learned architecture conversion.
"""

from __future__ import annotations

import math

_COORDINATE_CHUNK = 1 << 20


class QuantizationError(ValueError):
    """An available quantization parameter cannot describe this tensor."""


def is_quantization_auxiliary(key):
    return key.endswith(("_scale", ".scale_weight", ".scale_input", "_scale_inv",
                         "_zero_point", ".zero_point", ".comfy_quant", ".quant_state"))


def _parameter_address(layout, address, suffixes):
    component, key = address
    for suffix in suffixes:
        candidate = (component, key + suffix)
        if candidate in layout:
            return candidate
    if key.endswith(".weight"):
        candidate = (component, key[:-7] + ".scale_weight")
        if candidate in layout and suffixes[0] == "_scale":
            return candidate
    return None


def has_quantization_scale(layout, address):
    return _parameter_address(layout, address, ("_scale",)) is not None


def _expand_parameter(torch, parameter, shape):
    """Scalar, output-channel, broadcast, then contiguous block scales."""
    if not parameter.numel() or not torch.isfinite(parameter).all().item():
        raise QuantizationError("empty or nonfinite quantization parameter")
    if parameter.numel() == 1:
        return parameter.reshape(())
    if parameter.ndim == 1 and shape and parameter.numel() == shape[0]:
        return parameter.reshape(shape[0], *([1] * (len(shape) - 1)))
    try:
        if tuple(torch.broadcast_shapes(parameter.shape, shape)) == tuple(shape):
            return parameter
    except RuntimeError:
        pass
    if parameter.ndim == len(shape) and all(s > 0 and t % s == 0 for s, t in zip(parameter.shape, shape)):
        for axis, target in enumerate(shape):
            parameter = parameter.repeat_interleave(target // parameter.shape[axis], dim=axis)
        return parameter
    count = math.prod(shape)
    if count and count % parameter.numel() == 0:
        return parameter.reshape(-1).repeat_interleave(count // parameter.numel()).reshape(shape)
    raise QuantizationError("quantization blocks do not divide the tensor")


def decode_tensor(torch, readers, layout, address, value=None):
    """Return arithmetic values and the reversible base storage parameters.

    Floating storage, including FP8, can have scales too. Packed INT4/opaque
    formats are intentionally not guessed from arbitrary metadata blobs.
    """
    reader = readers[layout[address]]
    value = reader.get_tensor(address[1]) if value is None else value
    value = value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
    scale_address = _parameter_address(layout, address, ("_scale",))
    if scale_address is None:
        return value, None, None
    scale = readers[layout[scale_address]].get_tensor(scale_address[1]).to(value.dtype)
    scale = _expand_parameter(torch, scale, value.shape)
    if (scale <= 0).any().item():
        raise QuantizationError("quantization scale must be positive")
    zero_address = _parameter_address(layout, address, ("_zero_point", ".zero_point"))
    zero = (readers[layout[zero_address]].get_tensor(zero_address[1]).to(value.dtype)
            if zero_address else value.new_zeros(()))
    zero = _expand_parameter(torch, zero, value.shape)
    return (value - zero) * scale, scale, zero


def projection_transform(source_shape, target_shape):
    source, target = tuple(source_shape), tuple(target_shape)
    if not math.prod(source) or not math.prod(target):
        return "empty-preserve"
    if source == target:
        return "identity"
    if len(source) == len(target) == 2 and source == target[::-1]:
        return "transpose"
    if math.prod(source) == math.prod(target):
        return "reshape"
    if len(source) != len(target):
        return "flatten-resample"
    return "axis-resample"


def _coordinates(torch, source_size, target_size, device):
    # Float64 avoids the loss of integer coordinates for large flattened tensors.
    return torch.linspace(0, source_size - 1, steps=target_size,
                          device=device, dtype=torch.float64).round().long()


def project_tensor(torch, value, target_shape):
    """Names are resolved by the caller; here only coordinates are normalized."""
    shape = tuple(int(size) for size in target_shape)
    transform = projection_transform(value.shape, shape)
    # CPU indexing is not implemented for every low-precision storage dtype.
    if "float8" in str(value.dtype):
        value = value.float()
    if transform == "empty-preserve":
        raise ValueError("Empty source/target tensors must be preserved by the caller.")
    if transform == "transpose":
        return value.t().contiguous()
    if transform in ("identity", "reshape"):
        return value.reshape(shape).contiguous()
    if transform == "flatten-resample":
        flattened = value.reshape(-1)
        count = math.prod(shape)
        result = value.new_empty(count)
        step = (flattened.numel() - 1) / max(count - 1, 1)
        for start in range(0, count, _COORDINATE_CHUNK):
            end = min(count, start + _COORDINATE_CHUNK)
            coordinates = (torch.arange(start, end, device=value.device, dtype=torch.float64) * step).round().long()
            result[start:end] = flattened.index_select(0, coordinates)
        return result.reshape(shape)
    # Shrink before expanding, so intermediate tensors cannot exceed both the
    # source and destination simply because the leading axis grows first.
    axes = sorted(range(len(shape)), key=lambda axis: shape[axis] / value.shape[axis])
    for axis in axes:
        if value.shape[axis] != shape[axis]:
            value = value.index_select(axis, _coordinates(torch, value.shape[axis], shape[axis], value.device))
    return value.contiguous()


def repair_nonfinite_material(torch, value, replacement):
    """Replace invalid material coordinates without changing valid coordinates.

    The caller supplies the neutral value for its arithmetic: base coordinates
    for a sum, zero for a difference. Positions refer to the projected base
    shape. Bound diagnostic indexes even for a tensor containing only NaNs.
    """
    invalid = ~torch.isfinite(value)
    count = int(invalid.sum().item())
    flat = invalid.reshape(-1)
    coordinates = []
    for start in range(0, flat.numel(), 65536):
        indexes = flat[start:start + 65536].nonzero().flatten()[:8 - len(coordinates)].tolist()
        for index in indexes:
            offset = start + index
            coordinate = []
            for size in reversed(value.shape):
                coordinate.append(offset % size)
                offset //= size
            coordinates.append(list(reversed(coordinate)))
        if len(coordinates) == 8:
            break
    details = {"nonfinite_values": count,
               "nan_values": int(torch.isnan(value).sum().item()),
               "positive_infinity_values": int(torch.isposinf(value).sum().item()),
               "negative_infinity_values": int(torch.isneginf(value).sum().item()),
               "coordinates": coordinates}
    return torch.where(invalid, replacement, value), details


def record_numeric_repair(report, kind, address, count=1, *, source_index=None, reason=None):
    if report is None or count == 0:
        return
    report[kind] = report.get(kind, 0) + count
    events = report.setdefault("repair_events", [])
    if len(events) < 64:
        events.append({"kind": kind, "tensor": list(address), "count": count,
                       "source_index": source_index, "reason": reason})


def zero_nonfinite(torch, value, report, address, kind, *, source_index=None):
    arithmetic = value.float() if "float8" in str(value.dtype) else value
    count = int((~torch.isfinite(arithmetic)).sum().item())
    if not count:
        return value
    record_numeric_repair(report, kind, address, count, source_index=source_index)
    return torch.nan_to_num(arithmetic, nan=0., posinf=0., neginf=0.).to(value.dtype)


def safe_storage(torch, value, base, report, address):
    """Make even overflowing arithmetic representable in the base storage."""
    invalid = ~torch.isfinite(value)
    count = int(invalid.sum().item())
    if count:
        record_numeric_repair(report, "output_nonfinite_values", address, count)
        # NaN has no direction: preserve the already-sanitized base coordinate.
        value = torch.where(torch.isnan(value), base.to(value.dtype), value)
    limits = torch.finfo(base.dtype) if base.is_floating_point() else torch.iinfo(base.dtype)
    clipped = int(((value > limits.max) | (value < limits.min)).sum().item())
    record_numeric_repair(report, "output_clipped_values", address, clipped)
    value = value.clamp(limits.min, limits.max)
    if not base.is_floating_point():
        value = value.round()
    return value.to(base.dtype).contiguous()
