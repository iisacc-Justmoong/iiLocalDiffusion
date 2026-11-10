"""Offline relative-depth conditioning using a caller-supplied Depth Anything package."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Selection:
    directory: str
    precision: str = "float32"


@dataclass(frozen=True)
class FloatDepth:
    """Normalized, uncropped-at-consumption relative depth in a float tensor."""
    values: object


def _json_file(path, limit=1024 * 1024):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Depth model requires a bounded local {path.name}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid depth model {path.name}.") from error
    if not isinstance(value, dict) or "auto_map" in value:
        raise ValueError(f"Unsupported depth model {path.name}.")
    return value


def resolve(mode, directory, precision=None):
    if mode != "depth":
        if directory is not None or precision is not None:
            raise ValueError("Depth model options require --control-preprocess depth.")
        return None
    if precision not in (None, "float16", "float32"):
        raise ValueError("Depth model precision must be float16 or float32.")
    if not directory or "://" in directory:
        raise ValueError("--control-preprocess depth requires a local --control-depth-model directory.")
    folder = Path(directory).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Depth model directory does not exist.")
    config = _json_file(folder / "config.json")
    architectures = config.get("architectures")
    if (config.get("model_type") != "depth_anything"
            or config.get("depth_estimation_type", "relative") != "relative"
            or (architectures is not None and architectures != ["DepthAnythingForDepthEstimation"])):
        raise ValueError("Depth preprocessing requires a relative DepthAnythingForDepthEstimation model.")
    processor = _json_file(folder / "preprocessor_config.json")
    if processor.get("image_processor_type") != "DPTImageProcessor":
        raise ValueError("Depth preprocessing requires a DPTImageProcessor.")
    single = folder / "model.safetensors"
    if not single.is_file() or not single.stat().st_size:
        index = _json_file(folder / "model.safetensors.index.json", 8 * 1024 * 1024)
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Depth model requires safetensors weights.")
        for name in weights.values():
            if (not isinstance(name, str) or Path(name).name != name or "\\" in name
                    or not name.endswith(".safetensors") or not (folder / name).is_file()
                    or not (folder / name).stat().st_size):
                raise ValueError("Depth model contains an invalid safetensors shard.")
    return Selection(str(folder), precision or "float32")


def apply(image, selection, torch, np, Image, device, crop_size=None, *, preserve_float=False):
    from transformers import DPTImageProcessor, DepthAnythingForDepthEstimation

    if crop_size:
        width, height = crop_size
        scale = max(width / image.width, height / image.height)
        resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))),
                               Image.Resampling.LANCZOS)
        left, top = (resized.width - width) // 2, (resized.height - height) // 2
        image = resized.crop((left, top, left + width, top + height))
    processor = DPTImageProcessor.from_pretrained(selection.directory, local_files_only=True, trust_remote_code=False)
    model, info = DepthAnythingForDepthEstimation.from_pretrained(
        selection.directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        output_loading_info=True, dtype=getattr(torch, selection.precision))
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Depth model weights do not match the declared model configuration.")
    model = model.to(device).eval()
    # BatchFeature casts floating tensors while preserving integer masks or IDs.
    # Match the selected model precision before handing inputs to the backend.
    features = processor(images=image, return_tensors="pt").to(device=device, dtype=getattr(torch, selection.precision))
    with torch.no_grad():
        predicted = model(**features).predicted_depth
        if predicted.ndim != 3 or predicted.shape[0] != 1 or not torch.isfinite(predicted).all():
            raise ValueError("Depth model returned an invalid prediction.")
        depth = torch.nn.functional.interpolate(predicted.unsqueeze(1).float(), size=(image.height, image.width),
                                                mode="bilinear", align_corners=False)[0, 0]
        if not torch.isfinite(depth).all():
            raise ValueError("Depth model returned a non-finite resized prediction.")
        minimum, maximum = depth.min(), depth.max()
        span = maximum - minimum
        if not torch.isfinite(span):
            # Finite endpoints can still overflow when they have opposite signs.
            # Scale first only in that case, preserving the usual FP32 path.
            magnitude = torch.maximum(minimum.abs(), maximum.abs())
            depth = depth / magnitude
            minimum, maximum = minimum / magnitude, maximum / magnitude
            span = maximum - minimum
            epsilon = 1e-6 / magnitude
        else:
            epsilon = 1e-6
        depth = (depth - minimum) / (span + epsilon)
        if not torch.isfinite(depth).all():
            raise ValueError("Depth model returned an invalid normalized prediction.")
        depth = depth.clamp(0, 1).cpu()
        if preserve_float:
            result = FloatDepth(depth)
        else:
            pixels = (depth.numpy() * 255).astype(np.uint8)
            result = Image.fromarray(pixels, mode="L").convert("RGB")
    return result, {
        "name": "depth-anything-relative", "precision": selection.precision, "near": "white",
        "resampling": "bilinear", "output": "relative-depth-float32" if preserve_float else "rgb8",
        "crop_size": list(crop_size) if crop_size else None}
