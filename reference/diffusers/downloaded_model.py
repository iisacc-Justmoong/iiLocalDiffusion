"""Inspect local model identity without decoding tensors or executing pickle code.

Architecture evidence deliberately comes from tensors and bounded metadata, never
from a filename. This is routing evidence, not proof that a complete pipeline and
all of its companion encoders are present or that its license permits a use.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

from civitai_catalog import lookup_base_model


MAX_HEADER_BYTES = 100_000_000
MAX_INFO_BYTES = 8_000_000
MAX_ITEMS = 200_000
_DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
    "F16": 16, "BF16": 16, "I32": 32, "U32": 32, "F32": 32,
    "F64": 64, "I64": 64, "U64": 64, "F8_E4M3": 8,
    "F8_E5M2": 8, "F8_E8M0": 8, "F4": 4, "F6_E2M3": 6,
    "F6_E3M2": 6,
}
_ROLES = {
    "checkpoint": "checkpoint", "lora": "lora", "locon": "lora",
    "lycoris": "lora", "textualinversion": "embedding", "embedding": "embedding",
    "vae": "vae", "controlnet": "controlnet", "hypernetwork": "hypernetwork",
    "upscaler": "upscaler", "motionmodule": "motion-module",
    "aestheticgradient": "aesthetic-gradient", "poses": "poses",
    "wildcards": "wildcards", "detection": "detection", "other": "unknown",
}
_ROUTE = {
    "sd1": ("sd15-compatible", "StableDiffusionPipeline"),
    "sd2": (None, "StableDiffusionPipeline"),
    "sdxl": ("sdxl", "StableDiffusionXLPipeline"),
    "sdxl-refiner": (None, "StableDiffusionXLImg2ImgPipeline"),
    "flux1-dev": ("flux1-dev", "FluxPipeline"),
    "flux1-schnell": ("flux1-schnell-compatible", "FluxPipeline"),
    "flux1": (None, "FluxPipeline"),
    "sd3": (None, "StableDiffusion3Pipeline"),
}
_GUIDANCE = {
    "lora": "Attach this adapter with --lora to a matching base checkpoint.",
    "vae": "Attach this autoencoder with --vae to a matching base checkpoint.",
    "embedding": "Attach this embedding with --text-embedding to a matching base checkpoint.",
    "controlnet": "Attach this ControlNet with --controlnet and its conditioning image.",
    "checkpoint": "Select this file as --model; companion components may still be required.",
    "unknown": "Provide --model-info or a complete local pipeline configuration; do not guess a checkpoint family.",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_object(raw, description):
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"Invalid {description}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{description} must be a JSON object")
    return result


def _read_exact(source, count):
    result = source.read(count)
    if len(result) != count:
        raise ValueError("Truncated model header")
    return result


def _safetensors(path):
    size = path.stat().st_size
    with path.open("rb") as source:
        length = struct.unpack("<Q", _read_exact(source, 8))[0]
        if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
            raise ValueError("Invalid or oversized safetensors header length")
        raw = _read_exact(source, length)
    if not raw.startswith(b"{"):
        raise ValueError("Safetensors header must begin with a JSON object")
    header = _json_object(raw, "safetensors header")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Safetensors metadata must contain string values")
    if not header or len(header) > MAX_ITEMS:
        raise ValueError("Safetensors header has no tensors or too many tensors")
    shapes, intervals = {}, []
    data_size = size - 8 - length
    for key, descriptor in header.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"Invalid tensor descriptor: {key}")
        shape, offsets = descriptor.get("shape"), descriptor.get("data_offsets")
        dtype = descriptor.get("dtype")
        if (not isinstance(shape, list) or len(shape) > 16
                or any(type(n) is not int or n < 0 or n > 2**63 - 1 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_size):
            raise ValueError(f"Invalid tensor shape or offsets: {key}")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            raise ValueError(f"Unknown safetensors dtype {dtype!r}; update the format reader")
        bits = math.prod(shape) * _DTYPE_BITS[dtype]
        if bits % 8 or bits // 8 != offsets[1] - offsets[0]:
            raise ValueError(f"Tensor shape does not match byte extent: {key}")
        shapes[key] = tuple(shape)
        intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Safetensors data contains overlapping tensors or unindexed bytes")
        cursor = end
    if cursor != data_size:
        raise ValueError("Safetensors file has trailing unindexed bytes")
    return shapes, metadata, [f"Validated {len(shapes)} safetensors tensor descriptors without reading tensor data"]


def _gguf(path):
    """Read only standard GGUF descriptors; quantized tensor decoding belongs to gguf/backend."""
    size = path.stat().st_size
    with path.open("rb") as source:
        def read(count):
            if count < 0 or source.tell() + count > min(size, MAX_HEADER_BYTES):
                raise ValueError("GGUF header exceeds bounded metadata inspection limits")
            return _read_exact(source, count)

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

        def string():
            length = number("Q")
            if length > MAX_INFO_BYTES:
                raise ValueError("GGUF string is too large")
            try:
                return read(length).decode("utf-8")
            except UnicodeError as error:
                raise ValueError("Invalid GGUF UTF-8 string") from error

        primitives = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind, depth=0):
            if kind in primitives:
                return number(primitives[kind])
            if kind == 8:
                return string()
            if kind == 9 and depth < 8:
                element, count = number("I"), number("Q")
                if count > MAX_ITEMS:
                    raise ValueError("GGUF metadata array is too large")
                # Only scalar metadata affects routing; do not retain token arrays.
                for _ in range(count):
                    value(element, depth + 1)
                return None
            raise ValueError("Unsupported GGUF metadata value type or nesting")

        if read(4) != b"GGUF":
            raise ValueError("Invalid GGUF magic; big-endian GGUF requires a compatible backend")
        version, count, metadata_count = number("I"), number("Q"), number("Q")
        if version not in (2, 3) or count > MAX_ITEMS or metadata_count > MAX_ITEMS or count == 0:
            raise ValueError("Unsupported GGUF version or tensor/metadata count")
        metadata = {}
        for _ in range(metadata_count):
            key = string()
            if key in metadata:
                raise ValueError(f"Duplicate GGUF metadata key: {key}")
            metadata[key] = value(number("I"))
        shapes, offsets = {}, []
        for _ in range(count):
            name, rank = string(), number("I")
            if not 1 <= rank <= 16 or name in shapes:
                raise ValueError("Invalid GGUF tensor rank or duplicate name")
            dimensions = tuple(number("Q") for _ in range(rank))
            if any(n == 0 or n > 2**63 - 1 for n in dimensions):
                raise ValueError("Invalid GGUF tensor dimension")
            number("I")  # GGML quantization type; the maintained backend validates/decodes it.
            offsets.append(number("Q"))
            shapes[name] = tuple(reversed(dimensions))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF alignment")
        data_start = (source.tell() + alignment - 1) // alignment * alignment
        if any(data_start + offset >= size or offset % alignment for offset in offsets):
            raise ValueError("GGUF tensor data offset is outside the file or misaligned")
    return shapes, metadata, [f"Read GGUF v{version} metadata and {count} tensor descriptors; quantized payload requires backend validation"]


def _torch_checkpoint(path):
    try:
        import torch
    except ImportError:
        return {}, {}, ["PyTorch is unavailable; pickle checkpoint was not opened. Supply --model-info or inspect in the Diffusers environment."]
    from checkpoint_conversion import ensure_safe_torch_version
    try:
        ensure_safe_torch_version(torch)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    try:
        state = torch.load(str(path), map_location="meta", weights_only=True)
    except Exception as error:
        raise ValueError("Checkpoint cannot be inspected with torch.load(weights_only=True); unsafe pickle loading is not permitted") from error
    for _ in range(8):
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        else:
            break
    if not isinstance(state, dict) or len(state) > MAX_ITEMS:
        raise ValueError("Checkpoint has no bounded tensor state dictionary")
    shapes = {key: tuple(value.shape) for key, value in state.items()
              if isinstance(key, str) and isinstance(value, torch.Tensor)}
    if not shapes:
        # A1111 textual inversion saves its embedding under string_to_param.
        nested = state.get("string_to_param")
        if isinstance(nested, dict) and any(isinstance(v, torch.Tensor) for v in nested.values()):
            shapes = {"emb_params": tuple(next(v for v in nested.values() if isinstance(v, torch.Tensor)).shape)}
    if not shapes:
        raise ValueError("Checkpoint contains no tensor weights")
    return shapes, {}, [f"Inspected {len(shapes)} tensor shapes using torch weights_only=True on the meta device"]


def _tensor_identity(shapes):
    keys = set(shapes)
    # Adapters are checked first, even if a malicious sidecar calls them checkpoints.
    if any(re.search(r"(?:lora_[AB]|lora_(?:up|down)|hada_w[12]_[ab]|lokr_w[12])(?:\.|$)", k) for k in keys):
        return None, "lora", ["Adapter tensor keys are present"], None
    if any(k.startswith(("control_model.", "controlnet_cond_embedding.", "controlnet_down_blocks.")) for k in keys):
        return None, "controlnet", ["ControlNet component tensor keys are present"], None
    if keys and (keys <= {"emb_params", "clip_l", "clip_g", "string_to_param"}
                 or (len(keys) == 1 and len(next(iter(shapes.values()))) == 2
                     and next(iter(shapes.values()))[-1] in (768, 1024, 1280))):
        return None, "embedding", ["Standalone learned embedding tensors are present"], None
    normalized = {}
    for key, shape in shapes.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "unet.", "transformer."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = shape
    if any(k.startswith(("encoder.", "first_stage_model.encoder.", "vae.encoder.")) for k in keys) and any(
            k.startswith(("decoder.", "first_stage_model.decoder.", "vae.decoder.")) for k in keys) and not any(
            k.startswith(("input_blocks.", "down_blocks.", "double_blocks.", "joint_blocks.", "transformer_blocks.")) for k in normalized):
        return None, "vae", ["Standalone encoder and decoder weights are present"], None
    if ("joint_blocks.0.context_block.adaLN_modulation.1.bias" in normalized
            or ("pos_embed.proj.weight" in normalized and "transformer_blocks.0.attn.add_q_proj.weight" in normalized
                and not any(k.startswith("single_transformer_blocks.") for k in normalized))):
        return "sd3", "checkpoint", ["SD3 joint attention transformer tensor signature"], None
    flux_key = any(k in normalized for k in ("double_blocks.0.img_attn.norm.key_norm.scale", "double_blocks.0.img_attn.norm.key_norm.weight"))
    flux_diffusers = "x_embedder.weight" in normalized and any(k.startswith("single_transformer_blocks.") for k in normalized)
    if (flux_key and "img_in.weight" in normalized) or flux_diffusers:
        # Chroma, Flux2 and editing variants share many Flux1 keys.
        if any(k.startswith(("distilled_guidance_layer.", "double_stream_modulation_img.", "single_stream_modulation.")) for k in normalized):
            return None, "checkpoint", ["Flux-related transformer requires its own base-model metadata"], None
        input_key = "img_in.weight" if flux_key else "x_embedder.weight"
        channels = normalized[input_key][1] if len(normalized[input_key]) == 2 else None
        if channels != 64:
            return "flux1", "checkpoint", [f"Flux transformer has non-text-to-image input channels: {channels}"], "image-to-image"
        guidance = any(k.startswith(("guidance_in.", "time_text_embed.guidance_embedder.")) for k in normalized)
        architecture = "flux1-dev" if guidance else "flux1-schnell"
        return architecture, "checkpoint", [f"Flux1 transformer with guidance embeddings: {guidance}"], None
    cross = {shape[-1] for key, shape in normalized.items()
             if key.endswith("attn2.to_k.weight") and len(shape) == 2
             and key.startswith(("input_blocks.", "middle_block.", "output_blocks.", "down_blocks.", "mid_block.", "up_blocks."))}
    denoiser = "input_blocks.0.0.weight" in normalized or "conv_in.weight" in normalized
    if denoiser and cross:
        if len(cross) != 1:
            raise ValueError(f"Conflicting denoiser cross-attention dimensions: {sorted(cross)}")
        context = next(iter(cross))
        architecture = {768: "sd1", 1024: "sd2", 2048: "sdxl", 1280: "sdxl-refiner"}.get(context)
        if architecture:
            input_shape = normalized.get("input_blocks.0.0.weight", normalized.get("conv_in.weight"))
            channels = input_shape[1] if len(input_shape) > 1 else None
            task = {4: None, 9: "inpainting", 8: "image-to-image"}.get(channels, "image-to-image")
            return architecture, "checkpoint", [f"UNet cross-attention width {context}, input channels {channels}"], task
    return None, "unknown", ["No supported unambiguous checkpoint tensor signature"], None


def _info(path, info_path):
    candidates = [Path(info_path).expanduser()] if info_path is not None else list(dict.fromkeys(
        candidate for candidate in (path.with_suffix(".civitai.info"), Path(str(path) + ".civitai.info")) if candidate.is_file()))
    if len(candidates) > 1:
        raise ValueError("Multiple Civitai sidecars exist; select one with --model-info")
    if not candidates:
        return {}, None, []
    selected = candidates[0]
    if not selected.is_file():
        raise FileNotFoundError(f"Model metadata file does not exist: {selected}")
    if selected.stat().st_size > MAX_INFO_BYTES:
        raise ValueError("Model metadata exceeds the bounded JSON size")
    info = _json_object(selected.read_bytes(), "model metadata")
    files = info.get("files", [])
    if not isinstance(files, list) or any(not isinstance(f, dict) for f in files):
        raise ValueError("Civitai files metadata must be an array of objects")
    hashes = []
    for file in files:
        values = file.get("hashes", {})
        if not isinstance(values, dict):
            raise ValueError("Civitai file hashes must be an object")
        hashes.extend(v for k, v in values.items() if k.casefold() == "sha256")
    if "sha256" in info:
        hashes.append(info["sha256"])
    evidence = [f"Read model metadata: {selected.absolute()}"]
    if hashes:
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in hashes):
            raise ValueError("Model metadata contains an invalid SHA256")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() not in {value.lower() for value in hashes}:
            raise ValueError("Model metadata SHA256 does not match the selected local file")
        evidence.append(f"Matched Civitai SHA256: {digest.hexdigest()}")
    else:
        evidence.append("Metadata has no SHA256; exact file identity is unverified")
    return info, str(selected.absolute()), evidence


def inspect_downloaded_model(path, info_path=None):
    """Return bounded local routing evidence; reject malformed or conflicting identity.

    ``base_model`` is a validated Civitai category only when metadata names one.
    ``architecture`` can independently route a recognized tensor checkpoint.
    ``confidence`` distinguishes architecture/metadata evidence from a SHA256
    match; none of these values promises complete runtime or generation support.
    """
    selected = Path(path).expanduser().absolute()
    if not selected.is_file():
        raise FileNotFoundError(f"Local model file does not exist: {selected}")
    suffix = selected.suffix.casefold()
    formats = {".safetensors": ("safetensors", _safetensors), ".safetensor": ("safetensors", _safetensors),
               ".gguf": ("gguf", _gguf), ".ckpt": ("pytorch", _torch_checkpoint),
               ".pt": ("pytorch", _torch_checkpoint), ".pth": ("pytorch", _torch_checkpoint), ".bin": ("pytorch", _torch_checkpoint)}
    if suffix not in formats:
        raise ValueError("Unsupported local model container; use safetensors, GGUF, or a weights-only PyTorch checkpoint")
    format_name, reader = formats[suffix]
    shapes, metadata, evidence = reader(selected)
    architecture, role, tensor_evidence, task = _tensor_identity(shapes)
    evidence.extend(tensor_evidence)
    info, resolved_info, info_evidence = _info(selected, info_path)
    evidence.extend(info_evidence)
    base_name = info.get("baseModel", metadata.get("baseModel"))
    if info.get("baseModel") is not None and metadata.get("baseModel") is not None and info["baseModel"] != metadata["baseModel"]:
        raise ValueError("Conflicting Civitai baseModel values in embedded metadata and sidecar")
    record = lookup_base_model(base_name) if base_name is not None else None
    if record and record["local_status"] == "hosted":
        raise ValueError(f"Civitai base model {record['name']} is cloud-only")
    model = info.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Civitai model metadata must be an object")
    claimed_role = model.get("type", info.get("type"))
    if claimed_role is not None:
        if not isinstance(claimed_role, str):
            raise ValueError("Civitai model type must be a string")
        claimed_role = _ROLES.get(re.sub(r"[ _-]", "", claimed_role.casefold()), "unknown")
        if role != "unknown" and claimed_role != "unknown" and role != claimed_role:
            raise ValueError(f"Civitai model role {claimed_role} conflicts with tensor role {role}")
        if role == "unknown":
            role = claimed_role
    if architecture and record:
        family = "flux1" if architecture.startswith("flux1") else "sdxl" if architecture.startswith("sdxl") else architecture
        if family != record["family"]:
            raise ValueError(f"Civitai base model {record['name']} conflicts with tensor architecture {architecture}")
        if architecture == "flux1-schnell" and record.get("preset") in ("flux1-dev", "flux1-krea-dev"):
            raise ValueError("Civitai Flux dev metadata conflicts with absent guidance embeddings")
        if architecture == "flux1-dev" and record.get("preset") == "flux1-schnell-compatible":
            raise ValueError("Civitai Flux schnell metadata conflicts with guidance embeddings")
    container_architecture = {"flux": "flux1", "sd1": "sd1", "sd2": "sd2", "sdxl": "sdxl", "sd3": "sd3"}.get(metadata.get("general.architecture"))
    if container_architecture and architecture:
        tensor_family = "flux1" if architecture.startswith("flux1") else architecture
        if tensor_family != container_architecture:
            raise ValueError("GGUF architecture metadata conflicts with tensor architecture")
    if architecture is None and container_architecture:
        architecture = container_architecture
        if role == "unknown":
            role = "checkpoint"
        evidence.append(f"GGUF declares architecture {container_architecture}")
    if container_architecture and record and container_architecture != record["family"]:
        raise ValueError("Civitai base model conflicts with GGUF architecture metadata")
    preset, pipeline = _ROUTE.get(architecture, (None, None))
    if record:
        preset, pipeline = record.get("preset"), record.get("pipeline_class")
        architecture = architecture or record["family"]
    prediction = info.get("predictionType", metadata.get("modelspec.prediction_type", metadata.get("prediction_type")))
    if metadata.get("ss_v_parameterization", "").lower() == "true":
        if prediction not in (None, "v_prediction"):
            raise ValueError("Conflicting prediction parameterization metadata")
        prediction = "v_prediction"
    if prediction not in (None, "epsilon", "v_prediction", "sample", "flow_prediction"):
        raise ValueError(f"Unknown prediction parameterization: {prediction!r}")
    if record and record["name"] == "NoobAI" and prediction == "v_prediction":
        preset = "noobai-v-pred"
    if task or architecture == "sdxl-refiner":
        preset = None
        if task == "inpainting":
            pipeline = "StableDiffusionXLInpaintPipeline" if architecture == "sdxl" else "StableDiffusionInpaintPipeline"
        elif architecture == "sdxl-refiner":
            pipeline, task = "StableDiffusionXLImg2ImgPipeline", "image-to-image"
    if role != "checkpoint":
        preset, pipeline = None, None
    components = []
    if role == "checkpoint":
        components.append("denoiser")
    if any(key.startswith(("first_stage_model.encoder.", "vae.encoder.")) for key in shapes):
        components.append("vae")
    if any(key.startswith(("cond_stage_model.", "conditioner.embedders.", "text_encoders.", "text_encoder.")) for key in shapes):
        components.append("text_encoder")
    missing = [component for component in ("vae", "text_encoder") if component not in components] if role == "checkpoint" else []
    weights_role = "denoiser" if role == "checkpoint" and missing else role
    confidence = "exact" if any(s.startswith("Matched Civitai SHA256:") for s in evidence) else "metadata" if record else "architecture" if architecture else "unknown"
    return {"path": str(selected), "format": format_name, "base_model": record["name"] if record else None,
            "preset": preset, "pipeline_class": pipeline, "architecture": architecture, "role": role,
            "confidence": confidence, "evidence": evidence, "model_info": resolved_info,
            "weights_role": weights_role, "available_components": components, "missing_components": missing,
            "prediction_type": prediction, "task": task or (record.get("task") if record else "text-to-image"),
            "role_guidance": _GUIDANCE.get(role, "This component needs its matching backend workflow and base model.")}
