"""Safe, lossless tensor interchange for installed generation pipelines.

Representation labels are caller declarations. This layer never equates latent
spaces, casts token IDs to floating point, or serializes opaque model caches.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

ARCHITECTURES = ("auto", "unspecified", "diffusion", "rectified-flow", "flow-matching",
                 "autoregressive", "hybrid")
SEMANTICS = frozenset({"latents", "embeddings", "token-ids", "logits", "continuous-state",
                       "noise", "epsilon", "v-prediction", "velocity", "attention-mask", "tensor"})
SPEC_KEYS = frozenset({"semantic", "layout", "representation_space"})
DTYPES = frozenset({"float64", "float32", "float16", "bfloat16", "int64", "int32",
                   "int16", "int8", "uint8", "bool"})
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def validate_spec(spec: Any) -> dict[str, str]:
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError("Tensor output contracts require exactly semantic, layout, and representation_space.")
    if not isinstance(spec["semantic"], str) or spec["semantic"] not in SEMANTICS:
        raise ValueError("Unsupported tensor semantic; choose " + ", ".join(sorted(SEMANTICS)))
    if not isinstance(spec["layout"], str) or not re.fullmatch(r"[A-Z]{1,16}", spec["layout"]):
        raise ValueError("Tensor layout requires one uppercase axis letter per dimension (for example BCHW or BSC).")
    space = spec["representation_space"]
    if not isinstance(space, str) or not space.strip() or any(ord(c) < 32 for c in space):
        raise ValueError("Tensor representation_space must be a non-empty explicit label.")
    return dict(spec)


def validate_output_specs(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("--tensor-outputs must contain a JSON object.")
    result = {}
    for name, spec in value.items():
        if (not isinstance(name, str) or not IDENTIFIER.fullmatch(name)
                or name in {"past_key_values", "cache", "mems"} or name.endswith("_cache")):
            raise ValueError(f"Unsupported tensor output field {name!r}; opaque model caches cannot be serialized.")
        result[name] = validate_spec(spec)
        if name in {"sequences", "token_ids"} and spec["semantic"] != "token-ids":
            raise ValueError(f"Output field {name!r} requires token-ids semantic.")
    return result


def validate_input_descriptor(value: Any) -> dict[str, Any]:
    required = SPEC_KEYS | {"tensor_path", "key"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - required - {"sha256", "shape", "dtype"}):
        raise ValueError("Tensor inputs require tensor_path, exact key, semantic, layout, and representation_space; only sha256, shape, and dtype are optional.")
    path = value["tensor_path"]
    if not isinstance(path, str) or not path.strip() or Path(path).suffix != ".safetensors":
        raise ValueError("tensor_path must name a local .safetensors file.")
    if not isinstance(value["key"], str) or not value["key"]:
        raise ValueError("Tensor input key must be an exact non-empty safetensors key.")
    validate_spec({name: value[name] for name in SPEC_KEYS})
    if "sha256" in value and (not isinstance(value["sha256"], str)
                              or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Tensor input sha256 must be 64 lowercase hexadecimal characters.")
    if "dtype" in value and (not isinstance(value["dtype"], str) or value["dtype"] not in DTYPES):
        raise ValueError("Unsupported tensor input dtype.")
    if "shape" in value:
        shape = value["shape"]
        if (not isinstance(shape, list) or not shape or len(shape) != len(value["layout"])
                or any(type(size) is not int or size <= 0 for size in shape)):
            raise ValueError("Tensor input shape requires positive integer dimensions matching layout.")
    return dict(value)


def tensor_value(value: Any, spec: dict[str, str]) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("A declared tensor output must be a numeric tensor or array.") from error
    dtype = str(value.dtype).removeprefix("torch.")
    if dtype not in DTYPES or value.layout != torch.strided:
        raise ValueError(f"Unsupported tensor dtype or storage layout: {value.dtype}, {value.layout}.")
    if value.ndim != len(spec["layout"]) or value.numel() == 0:
        raise ValueError("Tensor shape must be non-empty and match its declared layout rank.")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError("Generation tensors must contain finite floating-point values.")
    if spec["semantic"] == "token-ids":
        if (value.is_floating_point() or value.dtype == torch.bool or value.ndim not in (1, 2)
                or spec["layout"] != ("S" if value.ndim == 1 else "BS")):
            raise ValueError("Token IDs require an integer S or BS tensor, preserving the integer dtype.")
        if (value < 0).any().item():
            raise ValueError("Token IDs must be non-negative.")
    if spec["semantic"] in {"latents", "embeddings", "logits", "continuous-state", "noise", "epsilon", "v-prediction", "velocity"}:
        if not value.is_floating_point():
            raise ValueError(f"Tensor semantic {spec['semantic']} requires floating-point values.")
    # Clone isolates ownership from pipeline views and safetensors mmap storage.
    return value.detach().to(device="cpu").contiguous().clone()


def load_tensor_input(descriptor: dict[str, Any], identity_reader: Any) -> tuple[Any, dict[str, Any]]:
    descriptor = validate_input_descriptor(descriptor)
    path = Path(descriptor["tensor_path"]).expanduser().absolute()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Tensor input must be a non-empty local file: {path}")
    before = identity_reader(path)
    if "sha256" in descriptor and before["sha256"] != descriptor["sha256"]:
        raise ValueError(f"Tensor input sha256 does not match: {path}")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as reader:
        if descriptor["key"] not in reader.keys():
            raise ValueError(f"Missing exact tensor input key: {descriptor['key']}")
        metadata = reader.metadata() or {}
        if "iild_generation_io" in metadata or ("schema_version" in metadata and "field" in metadata):
            if metadata.get("schema_version") != "1" or metadata.get("iild_generation_io", "1") != "1":
                raise ValueError("Unsupported iiLocalDiffusion tensor metadata schema version.")
            for name, expected in {"field": descriptor["key"], **{key: descriptor[key] for key in SPEC_KEYS}}.items():
                if metadata.get(name) != expected:
                    raise ValueError(f"Tensor input {name} conflicts with its stored iiLocalDiffusion metadata.")
        value = tensor_value(reader.get_tensor(descriptor["key"]), descriptor)
    after = identity_reader(path)
    if before != after:
        raise RuntimeError(f"Tensor input changed while loading: {path}")
    dtype, shape = str(value.dtype).removeprefix("torch."), list(value.shape)
    if "dtype" in descriptor and descriptor["dtype"] != dtype:
        raise ValueError(f"Tensor input dtype is {dtype}, expected {descriptor['dtype']}.")
    if "shape" in descriptor and descriptor["shape"] != shape:
        raise ValueError(f"Tensor input shape is {shape}, expected {descriptor['shape']}.")
    return value, {**after, "key": descriptor["key"], "dtype": dtype, "shape": shape,
                   **{key: descriptor[key] for key in SPEC_KEYS}, "kind": "tensor"}


def write_tensor_output(value: Any, name: str, spec: dict[str, str], directory: Path,
                        overwrite: bool, atomic_writer: Any, identity_reader: Any) -> dict[str, Any]:
    from safetensors.torch import save_file

    value = tensor_value(value, spec)
    path = directory / f"{name}.safetensors"
    metadata = {**spec, "field": name, "schema_version": "1", "iild_generation_io": "1"}
    atomic_writer(path, lambda temporary: save_file({name: value}, str(temporary), metadata=metadata), overwrite)
    identity = identity_reader(path)
    descriptor = {"tensor_path": str(path), "key": name, "sha256": identity["sha256"],
                  "dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape), **spec}
    kind = {"latents": "latent", "token-ids": "token-ids"}.get(spec["semantic"], "tensor")
    return {**identity, "kind": kind, "field": name, "format": "safetensors", "tensor_input": descriptor}


def automatic_token_spec(value: Any) -> dict[str, str]:
    import torch

    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Token output must be an integer tensor or rectangular array.") from error
    if tensor.ndim not in (1, 2):
        raise ValueError("Token output requires S or BS layout.")
    return {"semantic": "token-ids", "layout": "S" if tensor.ndim == 1 else "BS",
            "representation_space": "unspecified"}
