"""Adapters for already-loaded Diffusers and Transformers generation runtimes.

The owner selects/loads/places models. Calls do not download weights, enable
remote code, move models, change dtypes, or fall back to another device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Any, Mapping


def _mapping(value: Mapping) -> Mapping:
    return MappingProxyType(dict(value))


def _arguments(inputs: Mapping[str, Any], parameters: Mapping[str, Any], device: str) -> dict[str, Any]:
    import numpy as np
    import torch
    duplicates = set(inputs) & set(parameters)
    if duplicates:
        raise ValueError(f"Arguments occur in both parameters and bindings: {sorted(duplicates)}")
    def move(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy()).to(device=device)
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device).clone()
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    return {key: move(value) for key, value in {**parameters, **inputs}.items()}


def _host(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    return value


def _outputs(result: Any, fields: Mapping[str, str], *, autoregressive: bool) -> dict[str, Any]:
    import torch
    outputs = {}
    for name, key in fields.items():
        value = result.get(key) if isinstance(result, Mapping) else getattr(result, key, None)
        if value is None:
            raise ValueError(f"Generation did not return declared output field {key!r}")
        if autoregressive and key == "logits":
            if not isinstance(value, (list, tuple)) or not value or not all(isinstance(item, torch.Tensor) for item in value):
                raise ValueError("Transformers logits must be a nonempty sequence of per-step tensors")
            # Transformers supplies one [batch, vocabulary] tensor per step.
            if any(item.ndim != 2 for item in value):
                raise ValueError("Transformers per-step logits must have batch/vocabulary axes")
            value = torch.stack(value, dim=1)
        outputs[name] = _host(value)
    return outputs


@dataclass(frozen=True)
class DiffusersAdapter:
    pipeline: Any
    outputs: Mapping[str, str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def __post_init__(self):
        if self.parameters.get("return_dict", True) is not True:
            raise ValueError("DiffusersAdapter requires return_dict=True")
        if not self.outputs or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                                   for key, value in self.outputs.items()):
            raise ValueError("Adapters require named output fields")
        object.__setattr__(self, "outputs", _mapping(self.outputs))
        object.__setattr__(self, "parameters", _mapping({**self.parameters, "return_dict": True}))

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch
        arguments = _arguments(inputs, self.parameters, self.device)
        # Built-in signatures describe accepted inputs. Silently ignored kwargs
        # would make a linked stage look successful while dropping conditioning.
        signature = inspect.signature(self.pipeline.__call__)
        allowed = {name for name, param in signature.parameters.items()
                   if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)}
        if set(arguments) - allowed:
            raise ValueError(f"Diffusers pipeline does not explicitly accept {sorted(set(arguments) - allowed)}")
        signature.bind(**arguments)
        with torch.inference_mode():
            result = self.pipeline(**arguments)
        return _outputs(result, self.outputs, autoregressive=False)


@dataclass(frozen=True)
class TransformersAdapter:
    model: Any
    outputs: Mapping[str, str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def __post_init__(self):
        if not self.outputs or any(not isinstance(key, str) or not key or value not in {"sequences", "logits"}
                                   for key, value in self.outputs.items()):
            raise ValueError("TransformersAdapter exports sequences or raw logits; model-specific caches are not portable")
        if self.parameters.get("return_dict_in_generate", True) is not True:
            raise ValueError("TransformersAdapter requires return_dict_in_generate=True")
        if "logits" in self.outputs.values() and self.parameters.get("output_logits", True) is not True:
            raise ValueError("Declared logits require output_logits=True")
        self._validate_beams(self.parameters)
        parameters = {**self.parameters, "return_dict_in_generate": True}
        if "logits" in self.outputs.values():
            parameters["output_logits"] = True
        object.__setattr__(self, "outputs", _mapping(self.outputs))
        object.__setattr__(self, "parameters", _mapping(parameters))

    def _validate_beams(self, arguments: Mapping[str, Any]) -> None:
        config = arguments.get("generation_config", getattr(self.model, "generation_config", None))
        beams = arguments.get("num_beams", getattr(config, "num_beams", 1))
        if "logits" in self.outputs.values() and beams not in (None, 1):
            raise ValueError("Raw beam logits do not align with final sequences; use num_beams=1 for logits "
                             "or supply an explicit beam ancestry adapter")

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch
        if getattr(self.model, "training", False):
            raise ValueError("Transformers model must be in eval() mode before generation")
        arguments = _arguments(inputs, self.parameters, self.device)
        self._validate_beams(arguments)
        # GenerationMixin validates model kwargs and owns autoregressive decoding.
        with torch.inference_mode():
            result = self.model.generate(**arguments)
        return _outputs(result, self.outputs, autoregressive=True)


@dataclass(frozen=True)
class TokenDecodeAdapter:
    tokenizer: Any
    skip_special_tokens: bool = True

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(inputs) != {"tokens"}:
            raise ValueError("TokenDecodeAdapter requires exactly the tokens port")
        return {"text": self.tokenizer.batch_decode(inputs["tokens"],
                    skip_special_tokens=self.skip_special_tokens, clean_up_tokenization_spaces=False)}
