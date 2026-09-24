"""Deterministic checkpoint mapping onto the selected base layout."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from model_merge_tensor import (QuantizationError, decode_tensor, has_quantization_scale,
                                is_quantization_auxiliary, project_tensor, projection_transform)


POLICY = "base-layout-normalize-project-flatten-v3"
_FLOAT_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"})
_QUANT_DTYPES = frozenset({"I8", "U8"})


def _identity(address):
    component, key = address
    name = f"{component}.{key}".lower().replace("_", ".")
    # Text encoders also contain decoder blocks; resolve their ownership first.
    group = ("text" if any(value in name for value in ("text", "conditioner", "cond.stage", "clip", "t5")) else
             "vae" if any(value in name for value in ("vae", "first.stage", "encoder.conv", "decoder")) else
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
    kind = ("bias" if name.endswith(".bias") else "weight" if name.endswith(".weight") else "other")
    return group, role, depth, kind


def _canonical(address):
    key = address[1].lower()
    prefixes = ("module.", "_orig_mod.", "model.diffusion_model.", "diffusion_model.", "model.")
    while any(key.startswith(prefix) for prefix in prefixes):
        key = next(key[len(prefix):] for prefix in prefixes if key.startswith(prefix))
    return _identity(address)[0], key.replace("_", ".")


def preserves_base_tensor(address, dtype, shape, layout):
    key = address[1]
    if not math.prod(shape) or is_quantization_auxiliary(key):
        return True
    # Schedules, prediction flags and running statistics are runtime state, not
    # trainable parameters. A timestep embedding .weight remains a parameter.
    tail = key.rsplit(".", 1)[-1]
    if tail in {"sigmas", "sigma", "alphas", "betas", "alphas_cumprod", "v_pred", "ztsnr",
                "log_sigmas", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                "timesteps", "running_mean", "running_var", "num_batches_tracked"}:
        return True
    return dtype not in _FLOAT_DTYPES and not (
        dtype in _QUANT_DTYPES and _identity(address)[3] == "weight" and has_quantization_scale(layout, address))


def _shape_distance(base, source):
    return (abs(len(base) - len(source)),
            abs(math.log2(max(math.prod(base), 1) / max(math.prod(source), 1))),
            sum(abs(math.log2(max(a, 1) / max(b, 1))) for a, b in zip(base, source)))


def _depth_coordinates(items):
    groups = defaultdict(set)
    for address, _, _, identity in items:
        groups[(address[0], identity[0], identity[1], identity[3])].add(identity[2])
    coordinates = {}
    for address, _, _, identity in items:
        depths = sorted(groups[(address[0], identity[0], identity[1], identity[3])])
        coordinates[address] = depths.index(identity[2]) / max(len(depths) - 1, 1)
    return coordinates


class CommonLayerReader:
    """Expose source arithmetic values through the base's inventory and shapes."""

    def __init__(self, source_readers, source_layout, base_readers, base_layout, mappings, addresses, report):
        self.source_readers, self.source_layout = source_readers, source_layout
        self.base_readers, self.base_layout = base_readers, base_layout
        self.mappings, self.report = mappings, report
        self.addresses = {address[1]: address for address in addresses}

    def metadata(self):
        return {"iild.checkpoint_projection": POLICY}

    def keys(self):
        return list(self.addresses)

    def get_slice(self, key):
        return self.base_readers[self.base_layout[self.addresses[key]]].get_slice(key)

    def contributes(self, key):
        return self.mappings[self.addresses[key]] is not None

    def get_tensor(self, key):
        import torch
        address = self.addresses[key]
        source_address = self.mappings[address]
        if source_address is None:
            return self.base_readers[self.base_layout[address]].get_tensor(key).clone().contiguous()
        try:
            value, scale, _ = decode_tensor(torch, self.source_readers, self.source_layout, source_address)
        except QuantizationError as error:
            self.mappings[address] = None
            self.report["runtime_preserved_tensors"] += 1
            if len(self.report["runtime_events"]) < 64:
                self.report["runtime_events"].append({"target": list(address), "reason": str(error)})
            return self.base_readers[self.base_layout[address]].get_tensor(key).clone().contiguous()
        if scale is not None:
            self.report["dequantized_tensors"] += 1
        return project_tensor(torch, value, self.get_slice(key).get_shape())


def project_checkpoint(source_readers, source_layout, base_readers, base_layout):
    """Exact name, normalized name, then same-component/role depth mapping.

    Missing semantic roles are preserved, never filled with an unrelated VAE,
    text encoder, bias, schedule, or quantizer. Shape fitting is independent of
    selection, so every selected pair follows the same coordinate contract.
    """
    candidates, base_items = [], []
    for layout, readers, items in ((source_layout, source_readers, candidates),
                                   (base_layout, base_readers, base_items)):
        for address, filename in sorted(layout.items()):
            value = readers[filename].get_slice(address[1])
            shape, dtype = value.get_shape(), value.get_dtype()
            if preserves_base_tensor(address, dtype, shape, layout):
                continue
            items.append((address, shape, dtype, _identity(address)))
    source_items = {item[0]: item for item in candidates}
    canonical = defaultdict(list)
    role_candidates = defaultdict(list)
    for item in candidates:
        canonical[(_canonical(item[0]), item[0][0])].append(item)
        role_candidates[(item[3][0], item[3][1], item[3][3])].append(item)
    source_depth, base_depth = _depth_coordinates(candidates), _depth_coordinates(base_items)
    mappings, details, transforms = {}, [], Counter()
    exact = projected = preserved = 0
    for address, filename in sorted(base_layout.items()):
        base = base_readers[filename].get_slice(address[1])
        shape, dtype, identity = base.get_shape(), base.get_dtype(), _identity(address)
        chosen, selection = None, "base-preserved-no-semantic-source"
        named = canonical[(_canonical(address), address[0])]
        if not preserves_base_tensor(address, dtype, shape, base_layout):
            if address in source_items:
                chosen, selection = address, "exact"
            elif len(named) == 1:
                chosen, selection = named[0][0], "normalized-name"
            else:
                eligible = role_candidates[(identity[0], identity[1], identity[3])]
                # Keep package stages and separate text encoders in their own
                # components when both exports use a directory layout.
                if address[0] != "." and any(item[0][0] != "." for item in candidates):
                    eligible = [item for item in eligible if item[0][0] == address[0]]
                if eligible:
                    chosen = min(eligible, key=lambda item: (
                        abs(source_depth[item[0]] - base_depth[address]),
                        *_shape_distance(shape, item[1]), item[0]))[0]
                    selection = "semantic-normalized-depth"
        mappings[address] = chosen
        transform = projection_transform(source_items[chosen][1], shape) if chosen else "base-preserved"
        transforms[transform] += 1
        if chosen is None:
            preserved += 1
        elif chosen == address and transform == "identity":
            exact += 1
        else:
            projected += 1
        if len(details) < 256 and (selection != "exact" or transform != "identity"):
            details.append({"target": list(address), "source": list(chosen) if chosen else None,
                            "target_shape": list(shape), "source_shape": list(source_items[chosen][1]) if chosen else None,
                            "selection": selection, "transform": transform})
    report = {"policy": POLICY, "synthetic": True, "semantic_equivalence": False,
              "base_layout_tensors": len(base_layout), "source_tensors": len(source_layout),
              "exact_tensors": exact, "projected_tensors": projected, "base_preserved_tensors": preserved,
              "transform_counts": dict(transforms), "mapping_examples": details,
              "dequantized_tensors": 0, "runtime_preserved_tensors": 0, "runtime_events": []}
    projected_readers = {}
    for filename in sorted(set(base_layout.values())):
        addresses = [address for address, owner in base_layout.items() if owner == filename]
        projected_readers[filename] = CommonLayerReader(
            source_readers, source_layout, base_readers, base_layout, mappings, addresses, report)
    return projected_readers, dict(base_layout), report
