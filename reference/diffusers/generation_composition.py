"""Named host-value contracts for heterogeneous local generation stages.

This module orchestrates existing callables; it implements no tensor arithmetic,
sampler, tokenizer, model loader, or inference backend. Metadata is caller-declared.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

ARCHITECTURES = frozenset({"diffusion", "rectified-flow", "flow-matching", "autoregressive", "hybrid", "adapter"})
DTYPES = frozenset({"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"})
SEMANTICS = frozenset({"sample", "latent", "noise", "epsilon", "v_prediction", "velocity",
                       "embedding", "pixels", "mask", "token_ids", "logits", "timestep", "tensor"})


@dataclass(frozen=True)
class Port:
    kind: str
    dtype: str
    shape: tuple[int | None, ...]
    layout: str
    semantic: str
    representation_space: str

    @staticmethod
    def text(*, batch: bool = False) -> Port:
        return Port("text", "utf8", (None,) if batch else (), "B" if batch else "", "text", "utf8")

    @staticmethod
    def tensor(dtype: str, shape: tuple[int | None, ...], layout: str,
               semantic: str, representation_space: str) -> Port:
        return Port("tensor", dtype, tuple(shape), layout, semantic, representation_space)

    @staticmethod
    def from_tensor_descriptor(descriptor: Mapping[str, Any]) -> Port:
        """Interpret the concrete tensor_input descriptor from a generic output report.

        Use generic_io.load_tensor_input separately to verify and load its file.
        This method does not authenticate a file or invent an unspecified space.
        """
        from generic_io import validate_input_descriptor
        descriptor = validate_input_descriptor(dict(descriptor))
        if "dtype" not in descriptor or "shape" not in descriptor:
            raise ValueError("A concrete port descriptor requires dtype and shape")
        semantics = {"latents": "latent", "embeddings": "embedding", "token-ids": "token_ids",
                     "continuous-state": "sample", "attention-mask": "mask", "v-prediction": "v_prediction"}
        semantic = semantics.get(descriptor["semantic"], descriptor["semantic"])
        return Port.tensor(descriptor["dtype"], tuple(descriptor["shape"]), descriptor["layout"],
                           semantic, descriptor["representation_space"])

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(
                dim is not None and (type(dim) is not int or dim <= 0) for dim in self.shape):
            raise ValueError("Port shape must contain positive integers or None for dynamic dimensions")
        if (not isinstance(self.layout, str) or len(self.layout) != len(self.shape)
                or any(axis < "A" or axis > "Z" for axis in self.layout)):
            raise ValueError("Port layout must name each dimension with one axis character")
        if (not isinstance(self.representation_space, str) or not self.representation_space.strip()
                or self.representation_space in {"unspecified", "unknown", "auto"}):
            raise ValueError("Port requires an explicit representation_space")
        if self.kind == "text":
            if (self.dtype, self.semantic, self.representation_space) != ("utf8", "text", "utf8") or (
                    self.shape, self.layout) not in (((), ""), ((None,), "B")):
                raise ValueError("Text ports must be scalar UTF-8 text or a dynamic batch of text")
        elif self.kind != "tensor" or self.dtype not in DTYPES or self.semantic not in SEMANTICS:
            raise ValueError("Unsupported port kind, dtype, or semantic")
        elif self.semantic == "token_ids" and (not self.dtype.startswith(("int", "uint")) or self.layout not in ("S", "BS")):
            raise ValueError("token_ids require an integer dtype and S or BS layout")
        elif self.semantic in {"sample", "latent", "noise", "epsilon", "v_prediction", "velocity", "embedding", "logits"} and not (
                self.dtype.startswith("float") or self.dtype == "bfloat16"):
            raise ValueError(f"{self.semantic} requires a floating dtype")

    def require_compatible(self, destination: Port) -> None:
        for field in ("kind", "dtype", "layout", "semantic", "representation_space"):
            if getattr(self, field) != getattr(destination, field):
                raise ValueError(f"Incompatible {field}: {getattr(self, field)!r} -> {getattr(destination, field)!r}; "
                                 "provide an explicit conversion stage")
        if len(self.shape) != len(destination.shape) or any(
                a is not None and b is not None and a != b for a, b in zip(self.shape, destination.shape)):
            raise ValueError(f"Incompatible shape: {self.shape} -> {destination.shape}")

    def validate(self, value: Any, label: str) -> None:
        if self.kind == "text":
            valid = isinstance(value, str) if not self.shape else (
                isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(item, str) for item in value))
            if not valid:
                raise ValueError(f"{label}: expected {'batched' if self.shape else 'scalar'} text")
            return
        # Only real supported runtime arrays are accepted, not shape/dtype duck types.
        import numpy as np
        if isinstance(value, np.ndarray):
            dtype = str(value.dtype)
            shape = value.shape
            finite = bool(np.isfinite(value).all()) if value.dtype.kind in "fiub" else False
            negative = bool((value < 0).any()) if self.semantic == "token_ids" and value.dtype.kind in "iu" else False
        else:
            import torch
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError(f"{label}: expected an ordinary CPU Torch tensor or NumPy array")
            dtype = str(value.dtype).removeprefix("torch.")
            shape = tuple(value.shape)
            finite = bool(torch.isfinite(value).all().item())
            negative = bool((value < 0).any().item()) if self.semantic == "token_ids" else False
        if dtype != self.dtype:
            raise ValueError(f"{label}: dtype {dtype} does not match {self.dtype}")
        if len(shape) != len(self.shape) or any(
                actual <= 0 or (expected is not None and expected != actual)
                for actual, expected in zip(shape, self.shape)):
            raise ValueError(f"{label}: shape {shape} does not match {self.shape}")
        if not finite or negative:
            raise ValueError(f"{label}: values must be finite and token IDs must be nonnegative")


@dataclass(frozen=True)
class Source:
    stage: str | None
    port: str

    @staticmethod
    def input(name: str) -> Source:
        return Source(None, name)


@dataclass(frozen=True)
class Stage:
    name: str
    architecture: str
    inputs: Mapping[str, Port]
    outputs: Mapping[str, Port]
    bindings: Mapping[str, Source]
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    def __post_init__(self):
        for field in ("inputs", "outputs", "bindings"):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))


@dataclass(frozen=True)
class GenerationResult:
    outputs: Mapping[str, Any]
    architectures: tuple[str, ...]
    executed_stages: tuple[str, ...]

    @property
    def hybrid(self) -> bool:
        return "hybrid" in self.architectures or len(set(self.architectures)) > 1


def _ports(ports: Mapping[str, Port], label: str) -> None:
    for name, port in ports.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(port, Port):
            raise ValueError(f"{label}: names must be nonempty strings and values must be Ports")


def _keys(values: Mapping, expected: Mapping, label: str) -> None:
    if not isinstance(values, Mapping) or set(values) != set(expected):
        raise ValueError(f"{label}: named values must exactly match declared ports {list(expected)}")


def _snapshot(value: Any) -> Any:
    """Own validated host values so callback writes cannot corrupt another edge."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(value)  # Validated text batches contain immutable strings.
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.copy()
    return value.detach().clone()  # Port.validate already requires an actual Torch tensor.


class GenerationPipeline:
    """Preflight an ordered stage list and validate concrete values at each boundary."""

    def __init__(self, inputs: Mapping[str, Port], stages: list[Stage], outputs: Mapping[str, Source]):
        self.inputs = MappingProxyType(dict(inputs))
        self.stages = tuple(stages)
        self.outputs = MappingProxyType(dict(outputs))
        self.validate()

    def validate(self) -> None:
        _ports(self.inputs, "pipeline inputs")
        if not self.stages or not self.outputs:
            raise ValueError("A pipeline requires stages and named outputs")
        known = {None: self.inputs}
        for stage in self.stages:
            if not isinstance(stage.name, str) or not stage.name.strip() or stage.name in known:
                raise ValueError("Stage names must be nonempty and unique")
            if stage.architecture not in ARCHITECTURES or not callable(stage.execute):
                raise ValueError(f"{stage.name}: unsupported architecture or missing executor")
            _ports(stage.inputs, stage.name + " inputs")
            _ports(stage.outputs, stage.name + " outputs")
            if not stage.outputs:
                raise ValueError(f"{stage.name}: a stage requires outputs")
            _keys(stage.bindings, stage.inputs, stage.name + " bindings")
            for name, source in stage.bindings.items():
                self._resolve(known, source).require_compatible(stage.inputs[name])
            known[stage.name] = stage.outputs
        for name, source in self.outputs.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Pipeline output names must be nonempty strings")
            self._resolve(known, source)

    @staticmethod
    def _resolve(known: Mapping, source: Source):
        if not isinstance(source, Source) or source.stage not in known or source.port not in known[source.stage]:
            raise ValueError(f"Unknown or forward source binding: {source}")
        return known[source.stage][source.port]

    def run(self, inputs: Mapping[str, Any]) -> GenerationResult:
        self.validate()
        _keys(inputs, self.inputs, "pipeline inputs")
        for name, port in self.inputs.items():
            port.validate(inputs[name], "input." + name)
        values = {None: {name: _snapshot(value) for name, value in inputs.items()}}
        for stage in self.stages:
            arguments = {name: _snapshot(self._resolve(values, source)) for name, source in stage.bindings.items()}
            for name, port in stage.inputs.items():
                port.validate(arguments[name], stage.name + "." + name)
            produced = stage.execute(MappingProxyType(arguments))
            _keys(produced, stage.outputs, stage.name + " outputs")
            for name, port in stage.outputs.items():
                port.validate(produced[name], stage.name + "." + name)
            values[stage.name] = {name: _snapshot(value) for name, value in produced.items()}
        specs = {None: self.inputs, **{stage.name: stage.outputs for stage in self.stages}}
        outputs = {}
        for name, source in self.outputs.items():
            value = self._resolve(values, source)
            self._resolve(specs, source).validate(value, "output." + name)
            outputs[name] = _snapshot(value)
        return GenerationResult(
            MappingProxyType(outputs),
            tuple(stage.architecture for stage in self.stages if stage.architecture != "adapter"),
            tuple(stage.name for stage in self.stages))
