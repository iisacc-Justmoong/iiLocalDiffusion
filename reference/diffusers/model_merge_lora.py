"""Standard LoRA delta preparation using PyTorch, with strict material validation."""

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import re

from model_merge_lora_targets import LoraTarget, normalize_module, resolve_lora_target

_PAIR = re.compile(r"^(.*)\.(lora_A|lora_B)(?:\.([^.]+))?\.weight$")
_LEGACY_PAIR = re.compile(r"^(.*)\.(lora_down|lora_up|lora\.down|lora\.up|lora_linear_layer\.down|lora_linear_layer\.up)\.weight$")
_PROCESSOR_PAIR = re.compile(r"^(.*_lora)\.(down|up)\.weight$")
_FLOAT_DTYPES = {"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"}


def is_lora_key(key):
    return any(token in key for token in ("lora_", "_lora.", ".lora.", "dora_", "hada_", "lokr_", "oft_", "boft_"))


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"LoRA {label} must be finite and nonnegative.")
    return float(value)


def _config(model):
    assets = [asset for name, asset in model.assets.items() if Path(name).name == "adapter_config.json"]
    if not assets:
        return {}
    if len(assets) != 1:
        raise ValueError("A LoRA material must contain exactly one adapter configuration.")
    try:
        config = json.loads(Path(assets[0].resolved_file).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid LoRA adapter_config.json") from error
    if not isinstance(config, dict) or config.get("peft_type", "LORA") != "LORA":
        raise ValueError("Only standard PEFT LORA configurations are supported.")
    unsupported = ("use_dora", "lora_bias", "modules_to_save", "target_parameters", "layer_replication",
                   "trainable_token_indices", "alora_invocation_tokens", "use_qalora")
    if any(config.get(key) for key in unsupported) or config.get("bias", "none") != "none":
        raise ValueError("Unsupported LoRA variant or extra trained parameters in adapter_config.json.")
    if config.get("init_lora_weights") in ("olora", "corda") or str(config.get("init_lora_weights", "")).startswith("pissa"):
        raise ValueError("Convert adapters with modified base initialization to standard LoRA before merging.")
    for name in ("alpha_pattern", "rank_pattern"):
        if not isinstance(config.get(name, {}), dict):
            raise ValueError(f"LoRA {name} must be an object.")
    for name in ("use_rslora", "fan_in_fan_out"):
        if name in config and type(config[name]) is not bool:
            raise ValueError(f"LoRA {name} must be boolean.")
    return config


def _pattern_value(config, name, module, default):
    # Match PEFT's ordered suffix patterns, including its optional module prefix.
    for pattern, value in config.get(name, {}).items():
        try:
            if re.match(rf"(.*\.)?({pattern})$", module):
                return value
        except re.error as error:
            raise ValueError(f"Invalid LoRA {name} pattern: {pattern}") from error
    return default


@dataclass
class LoraDelta:
    target: LoraTarget
    down: tuple[object, str]
    up: tuple[object, str]
    scale: float
    fan_in_fan_out: bool
    module: str
    adaptation: dict | None = None

    def delta(self, torch, dtype, *, repair=False, normalization=None, source_index=None):
        down = self.down[0].get_tensor(self.down[1]).to(dtype=dtype)
        up = self.up[0].get_tensor(self.up[1]).to(dtype=dtype)
        if repair:
            from model_merge_tensor import zero_nonfinite
            down = zero_nonfinite(torch, down, normalization, self.target.address, "lora_nonfinite_values", source_index=source_index)
            up = zero_nonfinite(torch, up, normalization, self.target.address, "lora_nonfinite_values", source_index=source_index)
        if not torch.isfinite(down).all().item() or not torch.isfinite(up).all().item():
            raise ValueError(f"LoRA tensors must be finite: {self.module}")
        if down.ndim == 2:
            result = up @ down
            if self.fan_in_fan_out:
                result = result.T
        else:
            # Standard ungrouped convolution LoRA has a spatial down projection
            # and a 1x1 up projection. PyTorch performs the tensor contraction.
            result = (up.flatten(1) @ down.flatten(1)).reshape(up.shape[0], *down.shape[1:])
        result = result.mul(self.scale)
        if self.target.transpose:
            result = result.T
        if repair:
            result = zero_nonfinite(torch, result, normalization, self.target.address, "lora_delta_nonfinite_values", source_index=source_index)
        if not torch.isfinite(result).all().item():
            raise ValueError(f"LoRA delta is not finite: {self.module}")
        if self.adaptation:
            from model_merge_synthetic import adapt_delta
            result = adapt_delta(result, self.adaptation, torch)
        return result


def prepare_lora(model, readers, aliases, base_readers, base_layout, base_model, torch, policy="strict"):
    if policy not in ("strict", "synthetic"):
        raise ValueError(f"Unknown LoRA adaptation policy: {policy}")
    from model_merge_synthetic import SyntheticTargets, adaptation_report
    synthetic_targets = SyntheticTargets(base_readers, base_layout) if policy == "synthetic" else None
    config = _config(model)
    pairs, alphas, adapter_names = {}, {}, set()
    for reader in readers.values():
        for key in reader.keys():
            match = _PAIR.fullmatch(key)
            legacy = _LEGACY_PAIR.fullmatch(key) or _PROCESSOR_PAIR.fullmatch(key)
            if match or legacy:
                module, side = (match or legacy).group(1, 2)
                if match:
                    adapter_names.add(match.group(3))
                # Old attention processors spell to_q_lora.down.weight, etc.
                if module.endswith("_lora"):
                    module = module[:-5]
                    if module.endswith("to_out"):
                        module += ".0"
                side = "down" if side in ("down", "lora_A", "lora_down", "lora.down", "lora_linear_layer.down") else "up"
                pair = pairs.setdefault(module, {})
                if side in pair:
                    raise ValueError(f"Duplicate LoRA projection: {key}")
                pair[side] = (reader, key)
            elif key.endswith(".alpha"):
                module = key.removesuffix(".alpha")
                if module in alphas:
                    raise ValueError(f"Duplicate LoRA alpha: {key}")
                value = reader.get_tensor(key)
                if value.numel() != 1 or value.dtype == torch.bool:
                    raise ValueError(f"LoRA alpha must be scalar: {key}")
                alphas[module] = _number(value.item(), "alpha")
            else:
                raise ValueError(f"Unsupported LoRA tensor/variant (DoRA/LyCORIS or extra trained weights): {key}")
    if not pairs or len(adapter_names) > 1 or alphas.keys() - pairs.keys():
        raise ValueError("LoRA material has no projection pairs, mixed named adapters or unmatched alpha keys.")
    deltas = []
    for module, pair in pairs.items():
        if pair.keys() != {"down", "up"}:
            raise ValueError(f"LoRA requires paired down/up projections: {module}")
        slices = [pair[side][0].get_slice(pair[side][1]) for side in ("down", "up")]
        shapes = [value.get_shape() for value in slices]
        down, up = shapes
        if (any(value.get_dtype() not in _FLOAT_DTYPES for value in slices)
                or len(down) not in (2, 3, 4, 5) or len(down) != len(up)
                or any(n <= 0 for n in (*down, *up)) or down[0] != up[1]
                or (len(up) > 2 and any(n != 1 for n in up[2:]))):
            raise ValueError(f"Unsupported LoRA dtype, rank or convolution shape: {module}")
        rank = down[0]
        normalized = normalize_module(module)
        declared_rank = _number(_pattern_value(config, "rank_pattern", normalized, config.get("r", rank)), "rank")
        if declared_rank != rank:
            raise ValueError(f"LoRA rank differs from adapter configuration: {module}")
        alpha = alphas.get(module)
        if alpha is None:
            alpha = _number(_pattern_value(config, "alpha_pattern", normalized, config.get("lora_alpha", rank)), "alpha")
        scaling = alpha / (math.sqrt(rank) if config.get("use_rslora", False) else rank)
        fan = config.get("fan_in_fan_out", False)
        expected = [up[0], *down[1:]]
        if fan:
            if len(expected) != 2:
                raise ValueError("fan_in_fan_out is supported only for linear LoRA targets.")
            expected.reverse()
        remapped = False
        try:
            target = resolve_lora_target(module, aliases, base_model)
        except ValueError:
            if synthetic_targets is None:
                raise
            target = synthetic_targets.resolve(module, expected)
            remapped = True
        base = base_readers[base_layout[target.address]].get_slice(target.address[1])
        shape = list(base.get_shape())
        if base.get_dtype() not in _FLOAT_DTYPES or not 2 <= len(shape) <= 5:
            raise ValueError(f"Unsupported LoRA target dtype/rank: {module}")
        if target.rows:
            shape[0] = target.rows[1] - target.rows[0]
        if target.transpose:
            expected.reverse()
        if shape != expected and policy == "strict":
            raise ValueError(f"LoRA delta shape {expected} differs from base target {shape}: {module}")
        delta = LoraDelta(target, pair["down"], pair["up"], scaling, fan, module,
                          adaptation_report(module, target, expected, shape, remapped)
                          if remapped or shape != expected else None)
        for prior in deltas:
            overlap = prior.target.address == target.address and (prior.target.rows is None or target.rows is None
                or max(prior.target.rows[0], target.rows[0]) < min(prior.target.rows[1], target.rows[1]))
            if not overlap:
                continue
            # Some exports include both SGM and Diffusers aliases. Coalesce only
            # equal projection pairs with identical scaling/orientation, so the
            # same adapter is applied once. Distinct pairs remain separate
            # additive deltas on that target, preserving every learned branch.
            identical = prior.target == target and prior.scale == scaling and prior.fan_in_fan_out == fan and prior.adaptation == delta.adaptation
            if identical:
                for first, second in ((prior.down, delta.down), (prior.up, delta.up)):
                    left, right = first[0].get_tensor(first[1]), second[0].get_tensor(second[1])
                    comparable_left = left.float() if "float8" in str(left.dtype) else left
                    comparable_right = right.float() if "float8" in str(right.dtype) else right
                    if left.dtype != right.dtype or not torch.equal(comparable_left, comparable_right):
                        identical = False
                        break
            if identical:
                break
        else:
            deltas.append(delta)
    return deltas
