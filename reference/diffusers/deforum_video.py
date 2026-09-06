"""OpenCV camera transforms on top of the shared animation video output."""

from __future__ import annotations

import hashlib
from typing import Any

from animation_video import AnimationOutput, encode_video, output_targets, preflight_animation as preflight_video
from weight_files import file_sha256

SCHEMA = "iild-deforum-2d-v1"


def preflight_animation(args: Any) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ValueError("Deforum requires the existing Diffusers dependencies plus "
                         "reference/diffusers/requirements-deforum.txt (OpenCV headless).") from error
    environment = preflight_video(args)
    environment.update({"cv2": cv2, "np": np, "initial_image": None})
    environment["versions"]["opencv"] = cv2.__version__
    if args.init_image is not None:
        path = args.init_image
        identity = {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("--init-image must contain one static image.")
            initial = ImageOps.exif_transpose(source).convert("RGB")
            initial.load()
        if file_sha256(path) != identity["sha256"] or path.stat().st_size != identity["size_bytes"]:
            raise ValueError("--init-image changed while being decoded.")
        environment["initial_image"] = initial.resize((args.width, args.height), Image.Resampling.LANCZOS)
        environment["initial_image_identity"] = {**identity, "source_size": list(initial.size),
                                                 "generation_size": [args.width, args.height]}
    return environment


def prepare_frame(image: Any, reference: Any, request: Any, environment: dict[str, Any], *, warp: bool) -> Any:
    cv2, np, Image = (environment[name] for name in ("cv2", "np", "Image"))
    pixels = np.asarray(image.convert("RGB"))
    if warp:
        matrix = cv2.getRotationMatrix2D(((request.width - 1) / 2, (request.height - 1) / 2),
                                         request.angle, request.zoom)
        matrix[:, 2] += (request.translation_x, request.translation_y)
        border = {"replicate": cv2.BORDER_REPLICATE, "reflect": cv2.BORDER_REFLECT_101,
                  "wrap": cv2.BORDER_WRAP}[request.border]
        pixels = cv2.warpAffine(pixels, matrix, (request.width, request.height),
                               flags=cv2.INTER_LINEAR, borderMode=border)
    if request.color_coherence == "RGB" and reference is not None:
        target = np.asarray(reference)
        matched = np.empty_like(pixels)
        for channel in range(3):
            values, inverse, counts = np.unique(pixels[:, :, channel], return_inverse=True, return_counts=True)
            target_values, target_counts = np.unique(target[:, :, channel], return_counts=True)
            quantiles = counts.cumsum() / counts.sum()
            target_quantiles = target_counts.cumsum() / target_counts.sum()
            mapping = np.interp(quantiles, target_quantiles, target_values)
            matched[:, :, channel] = mapping[inverse].reshape(pixels.shape[:2]).round().astype(np.uint8)
        pixels = matched
    pixels = pixels.astype(np.float32) * request.contrast_schedule
    if request.noise_schedule:
        # Noise has an independent, frame-specific stream; it cannot consume the
        # diffusion generator's state or NumPy's process-global random state.
        identity = f"iild-deforum-noise:{request.seed}:{request.frame_index}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        pixels += np.random.default_rng(seed).normal(0, request.noise_schedule * 255, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).round().astype(np.uint8))
