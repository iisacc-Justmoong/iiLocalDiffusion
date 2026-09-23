"""Deterministic, explicitly untrained LoRA target/shape adaptation."""
import math
import re

from model_merge_lora_targets import LoraTarget

POLICY = "role-depth-zero-pad-crop-v1"


def _identity(name):
    name = name.lower().replace("_", ".")
    group = ("vae" if any(v in name for v in ("vae", "first.stage", "decoder", "encoder.conv")) else
             "text" if any(v in name for v in ("text", "lora.te", "cond.stage")) else "denoiser")
    roles = (("q", ("to.q", "q.proj", "q.projection")),
             ("k", ("to.k", "k.proj", "k.projection")),
             ("v", ("to.v", "v.proj", "v.projection")),
             ("out", ("to.out", "out.proj", "output.proj")),
             ("qkv", ("qkv", "in.proj")),
             ("ff-up", ("ff.net.0", "fc1", "up.proj")),
             ("ff-down", ("ff.net.2", "fc2", "down.proj")))
    role = next((role for role, tokens in roles if any(token in name for token in tokens)), "conv" if "conv" in name else "other")
    numbers = re.findall(r"(?:blocks?|layers?)\.(\d+)", name)
    return group, role, int(numbers[0]) if numbers else 0


class SyntheticTargets:
    def __init__(self, readers, layout):
        self.entries = []
        for address, filename in sorted(layout.items()):
            value = readers[filename].get_slice(address[1])
            shape = value.get_shape()
            if address[1].endswith(".weight") and 2 <= len(shape) <= 5 and value.get_dtype() in ("F16", "BF16", "F32", "F64"):
                self.entries.append((address, shape, _identity(".".join(address))))

    def resolve(self, module, shape):
        if not self.entries:
            raise ValueError("Synthetic LoRA adaptation requires a floating matrix/convolution target.")
        group, role, depth = _identity(module)
        def score(entry):
            address, target, identity = entry
            distance = abs(math.log2(target[0] / shape[0])) + abs(math.log2(math.prod(target[1:]) / math.prod(shape[1:])))
            return (identity[0] != group, identity[1] != role, abs(identity[2] - depth), distance, address)
        address, _, _ = min(self.entries, key=score)
        return LoraTarget(address)


def adaptation_report(module, target, source_shape, target_shape, remapped):
    retained = min(source_shape[0], target_shape[0]) * min(math.prod(source_shape[1:]), math.prod(target_shape[1:]))
    return {"policy": POLICY, "synthetic": True, "semantic_equivalence": False,
            "source_module": module, "target": list(target.address), "rows": target.rows,
            "selection": "role-depth-shape-nearest" if remapped else "exact-alias-shape-adaptation",
            "source_shape": list(source_shape), "target_shape": list(target_shape),
            "retained_values": retained, "zero_filled_values": math.prod(target_shape) - retained,
            "cropped_values": math.prod(source_shape) - retained}


def adapt_delta(value, report, torch):
    shape = report["target_shape"]
    source = value.reshape(value.shape[0], -1)
    target = torch.zeros((shape[0], math.prod(shape[1:])), dtype=value.dtype, device=value.device)
    rows, columns = min(source.shape[0], target.shape[0]), min(source.shape[1], target.shape[1])
    target[:rows, :columns] = source[:rows, :columns]
    return target.reshape(shape)
