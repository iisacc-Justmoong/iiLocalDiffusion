"""Deterministic cross-architecture checkpoint projection onto the base layout."""

from __future__ import annotations

import math
import re


POLICY = "base-layout-role-depth-crop-pad-v1"
_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})


def _identity(address):
    component, key = address
    name = f"{component}.{key}".lower().replace("_", ".")
    group = ("vae" if any(value in name for value in ("vae", "first.stage", "decoder", "encoder.conv")) else
             "text" if any(value in name for value in ("text", "conditioner", "cond.stage", "clip", "t5")) else
             "denoiser")
    roles = (
        ("q", (".wq.", ".to.q.", ".q.proj.", ".q.projection.")),
        ("k", (".wk.", ".to.k.", ".k.proj.", ".k.projection.")),
        ("v", (".wv.", ".to.v.", ".v.proj.", ".v.projection.")),
        ("out", (".wo.", ".to.out.", ".out.proj.", ".output.proj.")),
        ("qkv", ("qkv", "in.proj")),
        ("gate", (".gate.", "gate.proj")),
        ("ff-up", ("ff.net.0", "mlp.fc1", "up.proj")),
        ("ff-down", ("ff.net.2", "mlp.fc2", "down.proj")),
        ("norm", ("norm", "qknorm")),
        ("embed", ("embed", "proj.in", "input")),
        ("output", ("proj.out", "final.layer", "output")),
    )
    role = next((role for role, tokens in roles if any(token in name for token in tokens)),
                "conv" if "conv" in name else "other")
    numbers = [int(value) for value in re.findall(r"(?:blocks?|layers?)\.(\d+)", name)]
    depth = numbers[0] if numbers else 0
    kind = "bias" if key.endswith(".bias") else "weight" if key.endswith(".weight") else "other"
    return group, role, depth, kind


def _shape_distance(base, source):
    base_size, source_size = math.prod(base), math.prod(source)
    return (abs(len(base) - len(source)), abs(math.log2(max(base_size, 1) / max(source_size, 1))),
            sum(abs(math.log2(max(a, 1) / max(b, 1))) for a, b in zip(base, source)))


class _ProjectedSlice:
    def __init__(self, shape, dtype):
        self._shape, self._dtype = list(shape), dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype


class CommonLayerReader:
    """Expose a foreign checkpoint through the base checkpoint's tensor layout."""

    def __init__(self, source_readers, source_layout, base_readers, base_layout, mappings, addresses):
        self.source_readers = source_readers
        self.source_layout = source_layout
        self.base_readers = base_readers
        self.base_layout = base_layout
        self.mappings = mappings
        self.addresses = {address[1]: address for address in addresses}

    def metadata(self):
        return {"iild.checkpoint_projection": POLICY}

    def keys(self):
        return list(self.addresses)

    def get_slice(self, key):
        address = self.addresses[key]
        base = self.base_readers[self.base_layout[address]].get_slice(key)
        return _ProjectedSlice(base.get_shape(), base.get_dtype())

    def get_tensor(self, key):
        import torch
        address = self.addresses[key]
        base = self.base_readers[self.base_layout[address]].get_tensor(key)
        source_address = self.mappings.get(address)
        if source_address is None or not base.is_floating_point():
            return base.clone().contiguous()
        source_reader = self.source_readers[self.source_layout[source_address]]
        value = source_reader.get_tensor(source_address[1])
        if not value.is_floating_point():
            value = value.to(torch.float32)
            scale_address = (source_address[0], source_address[1] + "_scale")
            if scale_address in self.source_layout:
                scale_reader = self.source_readers[self.source_layout[scale_address]]
                scale = scale_reader.get_tensor(scale_address[1]).to(torch.float32)
                try:
                    value = value * scale
                except RuntimeError:
                    value = value.reshape(-1) * scale.reshape(-1).repeat_interleave(
                        max(1, math.ceil(value.numel() / max(scale.numel(), 1))))[:value.numel()]
                    value = value.reshape(source_reader.get_slice(source_address[1]).get_shape())
        flat = value.reshape(-1)
        target = torch.zeros(base.numel(), dtype=value.dtype, device=value.device)
        count = min(flat.numel(), target.numel())
        target[:count] = flat[:count]
        return target.reshape(base.shape)


def project_checkpoint(source_readers, source_layout, base_readers, base_layout):
    """Map every usable foreign tensor to the closest semantic base tensor.

    Unmapped and nonfloating base tensors preserve the base value. This policy
    intentionally guarantees an executable base-layout checkpoint rather than
    architecture equivalence or useful image quality.
    """
    candidates = []
    for address, filename in source_layout.items():
        value = source_readers[filename].get_slice(address[1])
        if address[1].endswith((".comfy_quant", ".weight_scale")):
            continue
        candidates.append((address, value.get_shape(), value.get_dtype(), _identity(address)))
    mappings, details = {}, []
    exact = projected = preserved = 0
    for address, filename in base_layout.items():
        base = base_readers[filename].get_slice(address[1])
        base_shape, base_dtype = base.get_shape(), base.get_dtype()
        chosen = address if address in source_layout else None
        selection = "exact"
        if chosen is None and base_dtype in _FLOAT_DTYPES and candidates:
            identity = _identity(address)
            eligible = [item for item in candidates if item[3][3] == identity[3]]
            chosen = min(eligible, key=lambda item: (
                item[3][0] != identity[0], item[3][1] != identity[1], item[3][3] != identity[3],
                abs(item[3][2] - identity[2]), *_shape_distance(base_shape, item[1]), item[0]))[0] if eligible else None
            selection = "role-depth-shape-nearest" if chosen else "base-preserved-no-common-kind"
        mappings[address] = chosen
        if chosen is None:
            preserved += 1
        elif chosen == address and source_readers[source_layout[chosen]].get_slice(chosen[1]).get_shape() == base_shape:
            exact += 1
        else:
            projected += 1
        if len(details) < 256 and selection != "exact":
            source_shape = source_readers[source_layout[chosen]].get_slice(chosen[1]).get_shape() if chosen else None
            details.append({"target": list(address), "source": list(chosen) if chosen else None,
                            "target_shape": list(base_shape), "source_shape": list(source_shape) if source_shape else None,
                            "selection": selection})
    projected_readers = {}
    for filename in set(base_layout.values()):
        addresses = [address for address, owner in base_layout.items() if owner == filename]
        projected_readers[filename] = CommonLayerReader(
            source_readers, source_layout, base_readers, base_layout, mappings, addresses)
    report = {"policy": POLICY, "synthetic": True, "semantic_equivalence": False,
              "base_layout_tensors": len(base_layout), "source_tensors": len(source_layout),
              "exact_tensors": exact, "projected_tensors": projected,
              "base_preserved_tensors": preserved, "mapping_examples": details}
    return projected_readers, dict(base_layout), report
