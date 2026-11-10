"""MOSS-Image color/depth expanded-input control adapter."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

from weight_files import LocalWeightFile, resolve_weight_file

IMAGE_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
TEXT_TARGETS = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")


@dataclass(frozen=True)
class Selection:
    weight: LocalWeightFile
    depth: str | None
    color: str | None
    scale: float
    start: float
    depth_end: float
    color_end: float
    map_preprocess: str = 'semantic'
    target_pipeline: str = "MOSSImagePipeline"
    image_argument: str = "control_image"

    def metadata(self):
        return {"kind": "moss-image-color-depth-control", "path": self.weight.path,
                "sha256": self.weight.sha256, "size_bytes": self.weight.size_bytes,
                "depth": self.depth, "color": self.color, "scale": self.scale,
                "start": self.start, "depth_end": self.depth_end, "color_end": self.color_end,
                "scale_scope": "LoRA layers; the trained input projection remains fully active",
                "map_preprocess": self.map_preprocess}


def resolve(args):
    if args.controlnet_variant is not None or args.guess_mode is not None or args.control_mode is not None:
        raise ValueError("MOSS control does not support a package variant, guess mode, or Union mode.")
    if args.offload != "none" or args.seed is None:
        raise ValueError("MOSS control requires --offload none and an explicit --seed.")
    if args.control_preprocessor_library and args.control_preprocess != "canny":
        raise ValueError("--control-preprocessor-library requires --control-preprocess canny.")
    if args.control_preprocess == "canny" and not args.control_image:
        raise ValueError("Native Canny requires a depth/control image.")
    if not args.control_image and not args.control_color_image:
        raise ValueError("MOSS control requires a depth --control-image and/or --control-color-image.")
    def image_path(value):
        if value is None:
            return None
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("MOSS control images must be nonempty local files.")
        return str(path)
    if "control_image" in args.inputs or "control_color_image" in args.inputs:
        raise ValueError("Do not also provide MOSS control images through --pipeline-inputs.")
    scale = 1.0 if args.controlnet_scale is None else args.controlnet_scale
    start = 0.0 if args.control_guidance_start is None else args.control_guidance_start
    end = 1.0 if args.control_guidance_end is None else args.control_guidance_end
    depth_end = end if args.control_depth_end is None else args.control_depth_end
    color_end = end if args.control_color_end is None else args.control_color_end
    if (not all(map(math.isfinite, (scale, start, end, depth_end, color_end))) or scale < 0
            or not 0 <= start < end <= 1 or not 0 <= depth_end <= end or not 0 <= color_end <= end):
        raise ValueError("MOSS control requires nonnegative scale and valid control intervals within [0,1].")
    for key in ("width", "height"):
        value = args.inputs.get(key, 1024)
        if type(value) is not int or not 16 <= value <= 8192 or value % 16:
            raise ValueError("MOSS control dimensions must be multiples of 16 in [16,8192].")
    return Selection(resolve_weight_file(args.controlnet, "MOSS control adapter"),
                     image_path(args.control_image), image_path(args.control_color_image),
                     scale, start, depth_end, color_end, getattr(args, 'control_map_preprocess', None) or 'semantic')


def plan_weights(shapes, module_shapes, metadata, *, projection_shape, image_layers, text_layers):
    if any(metadata.get(key) != value for key, value in (
            ("hayate_control", "1"),
            ("control_channels", "color,depth"), ("control_preprocess", "vae"))):
        raise ValueError("MOSS control adapter metadata does not match the color/depth contract.")
    if metadata.get('base_model') not in ('moss-base', 'moss-sft'):
        raise ValueError('MOSS control adapter must declare a supported MOSS base or SFT model.')
    try:
        rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MOSS control adapter requires rank and alpha metadata.") from error
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or len(projection_shape) != 2:
        raise ValueError("Invalid MOSS control adapter rank, alpha, or base projection.")
    outputs, inputs = projection_shape
    if (tuple(shapes.get("video_embedder.weight", ())) != (outputs, 3 * inputs)
            or tuple(shapes.get("video_embedder.bias", ())) != (outputs,)):
        raise ValueError("MOSS control input projection does not match the base transformer.")
    targets = [*(f"layers.{i}.proj.{name}" for i in range(image_layers) for name in IMAGE_TARGETS),
               *(f"text_embedder.blocks.{i}.{name}" for i in range(text_layers) for name in TEXT_TARGETS)]
    if not targets:
        raise ValueError("MOSS control requires transformer targets.")
    expected = {"video_embedder.weight", "video_embedder.bias"}
    for target in targets:
        a, b = target + ".lora_A.weight", target + ".lora_B.weight"
        expected.update((a, b))
        shape = module_shapes.get(target)
        if (shape is None or len(shape) != 2 or tuple(shapes.get(a, ())) != (rank, shape[1])
                or tuple(shapes.get(b, ())) != (shape[0], rank)):
            raise ValueError(f"MOSS adapter tensors do not match {target}.")
    if set(shapes) != expected:
        raise ValueError("MOSS control adapter has missing or unexpected tensors.")
    return targets


def _encode_hint(pipeline, image, inputs, torch, np, *, color, preprocess='semantic', output_identity=None):
    from PIL import Image, ImageOps
    width, height = inputs.get("width", 1024), inputs.get("height", 1024)
    if preprocess == 'training':
        # Match the published training cache's load_pixels transform.
        image = ImageOps.fit(image.convert('RGB'), (width, height), method=Image.Resampling.LANCZOS)
    elif preprocess == 'semantic':
        mode = Image.Resampling.NEAREST if color else Image.Resampling.BILINEAR
        image = (image.convert('RGB') if color else image.convert('L').convert('RGB')).resize((width, height), mode)
    else:
        raise ValueError('Unknown MOSS control map preprocessing mode.')
    if output_identity is not None:
        from control_hint_identity import fingerprint
        output_identity.update(fingerprint(image, np))
    vae = pipeline.vae
    parameter = next(vae.parameters())
    # Normalize the original 8-bit values in FP32 before converting to the
    # VAE precision; early BF16 rounding changes intermediate color values.
    pixels = torch.from_numpy(np.array(image, copy=True)).to(device=parameter.device, dtype=torch.float32)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1).to(dtype=parameter.dtype)
    z = vae.encode(pixels).latent_dist.mode()
    batch, channels, height, width = z.shape
    if height % 2 or width % 2:
        raise ValueError("MOSS control VAE returned an unpatchable latent.")
    z = z.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(batch, channels * 4, height // 2, width // 2)
    mean = vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    if z.shape[1] != mean.shape[1]:
        raise ValueError("MOSS control requires the matching FLUX.2 VAE normalization.")
    z = (z - mean) / std
    if not bool(torch.isfinite(z).all()):
        raise ValueError("MOSS control VAE produced non-finite latents.")
    return z.flatten(2).transpose(1, 2)


@contextmanager
def execution(pipeline, images, inputs, selection, torch, np):
    from safetensors import safe_open
    from weight_files import verify_weight_file
    transformer = pipeline.transformer
    modules = dict(transformer.named_modules())
    module_shapes = {name: tuple(module.weight.shape) for name, module in modules.items()
                     if getattr(module, "weight", None) is not None and module.weight.ndim == 2}
    projection = modules.get("video_embedder")
    if projection is None:
        raise ValueError("MOSS transformer has no video input projection.")
    verify_weight_file(selection.weight, "MOSS control adapter")
    with safe_open(selection.weight.path, framework="pt", device="cpu") as source:
        shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        metadata = source.metadata() or {}
        targets = plan_weights(shapes, module_shapes, metadata, projection_shape=tuple(projection.weight.shape),
                               image_layers=len(transformer.layers), text_layers=len(transformer.text_embedder.blocks))
        device, dtype = projection.weight.device, projection.weight.dtype
        if device.type == "meta":
            raise ValueError("MOSS control requires resident model weights.")
        hints = {}
        color_id, depth_id = {}, {}
        color = _encode_hint(pipeline, images["color"], inputs, torch, np, color=True, preprocess=selection.map_preprocess, output_identity=color_id).to(device=device, dtype=dtype) if images.get("color") is not None else None
        depth = _encode_hint(pipeline, images["depth"], inputs, torch, np, color=False, preprocess=selection.map_preprocess, output_identity=depth_id).to(device=device, dtype=dtype) if images.get("depth") is not None else None
        if color is not None:
            hints['color'] = color_id
        if depth is not None:
            hints['depth'] = depth_id
        def tensor(name):
            value = source.get_tensor(name)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"MOSS control tensor is non-floating or non-finite: {name}")
            converted = value.to(device=device, dtype=dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f"MOSS control tensor overflows the selected precision: {name}")
            return converted
        first = tensor("video_embedder.weight")
        bias = tensor("video_embedder.bias")
        pairs = [(modules[name], tensor(name + ".lora_A.weight"), tensor(name + ".lora_B.weight"))
                 for name in targets]
        multiplier = selection.scale * float(metadata["alpha"]) / int(metadata["rank"])
    verify_weight_file(selection.weight, "MOSS control adapter")
    handles = []
    scheduler = pipeline.scheduler
    had_step = "step" in vars(scheduler)
    old_instance_step = vars(scheduler).get("step")
    old_step = scheduler.step
    state = {"step": 0}
    activity = {"color_steps": [], "depth_steps": []}
    try:
        def project(module, args, output):
            value = args[0]
            total = state.get('total', len(scheduler.timesteps))
            fraction = state["step"] / total if total else 0
            def selected(hint, end, name):
                if hint is None or not selection.start <= fraction < end:
                    return torch.zeros_like(value)
                if hint.shape[1:] != value.shape[1:]:
                    raise ValueError("MOSS control latent does not match the generated image tokens.")
                steps = activity[name]
                if not steps or steps[-1] != state["step"]:
                    steps.append(state["step"])
                return hint.expand(value.shape[0], -1, -1).to(device=value.device, dtype=value.dtype)
            return torch.nn.functional.linear(torch.cat((value, selected(color, selection.color_end, "color_steps"),
                                                         selected(depth, selection.depth_end, "depth_steps")), dim=-1), first, bias)
        handles.append(projection.register_forward_hook(project))
        for module, a, b in pairs:
            def add_adapter(module, args, output, a=a, b=b):
                return output + multiplier * torch.nn.functional.linear(torch.nn.functional.linear(args[0], a), b)
            handles.append(module.register_forward_hook(add_adapter))
        def next_step(*args, **kwargs):
            result = old_step(*args, **kwargs)
            state["step"] += 1
            return result
        scheduler.step = next_step
        yield {"adapter": "moss-image-color-depth-control", "completed_steps": state,
               "control_activity": activity,
               "vae_input_hints": hints,
               "trained_base": metadata['base_model'],
               "control_size": [inputs.get("width", 1024), inputs.get("height", 1024)],
               "map_preprocess": selection.map_preprocess,
               "color_resize": "center-fit Lanczos" if selection.map_preprocess == 'training' else "nearest",
               "depth_resize": "center-fit Lanczos" if selection.map_preprocess == 'training' else "bilinear",
               "depth_channels": "RGB preserved" if selection.map_preprocess == 'training' else "grayscale replicated to RGB; near is white",
               "vae_latents": "posterior mean, 2x2 packing and FLUX.2 batch normalization"}
    finally:
        if had_step:
            scheduler.step = old_instance_step
        elif "step" in vars(scheduler):
            del scheduler.step
        for handle in reversed(handles):
            handle.remove()
