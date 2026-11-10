"""Krea 2 expanded-input Control-LoRA tensor contract and execution."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from generation_seed import derive_generator

from weight_files import LocalWeightFile, resolve_weight_file
from depth_preprocessing import FloatDepth


# Original Krea 2 names in Patil/Krea-2-depth-controlnet and their public
# Diffusers transformer equivalents. These are semantic mappings, not a
# conversion of an arbitrary adapter into a ControlNet.
BLOCK_TARGETS = {
    "attn.wq": "attn.to_q", "attn.wk": "attn.to_k", "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0", "attn.gate": "attn.to_gate",
    "mlp.gate": "ff.gate", "mlp.up": "ff.up", "mlp.down": "ff.down",
}

# Ordering and logarithmic aspect-distance match the public depth adapter.
BUCKETS = ((1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832),
           (768, 1344), (1344, 768), (704, 1472), (1472, 704))


def pick_bucket(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Krea 2 control image dimensions must be positive.")
    aspect = math.log(width / height)
    return min(BUCKETS, key=lambda bucket: abs(math.log(bucket[0] / bucket[1]) - aspect))


def configure_resolution(args, inputs, image):
    from krea2_contract import default_mu
    width, height = pick_bucket(*image.size)
    args.inputs.update(width=width, height=height)
    inputs.update(width=width, height=height)
    contract = args.krea2
    contract['resolved_mu'] = (contract['mu_override'] if contract['mu_override'] is not None
                               else default_mu(contract['variant'], width, height))
    contract['control_resolution'] = {'mode': 'input-aspect-bucket', 'status': 'resolved',
                                      'input_size': list(image.size), 'output_size': [width, height]}


@dataclass(frozen=True)
class Selection:
    weight: LocalWeightFile
    image: str
    scale: float
    auto_bucket: bool = False
    target_pipeline: str = "Krea2Pipeline"
    image_argument: str = "control_image"

    def metadata(self):
        return {"kind": "krea2-expanded-input-control-lora", "path": self.weight.path,
                "sha256": self.weight.sha256, "size_bytes": self.weight.size_bytes,
                "image": self.image, "scale": self.scale,
                "resolution_mode": "input-aspect-bucket" if self.auto_bucket else "explicit-or-default",
                "scale_scope": "LoRA layers; the trained input projection remains fully active"}


def resolve(args):
    auto_bucket = getattr(args, 'krea2_control_auto_bucket', False)
    if auto_bucket and getattr(args, 'krea2_control_size_explicit', False):
        raise ValueError("--krea2-control-auto-bucket cannot be combined with an explicit width or height.")
    if any(value is not None for value in (args.control_color_image, args.control_color_end, args.control_depth_end)):
        raise ValueError("Color/depth controls require MOSSImagePipeline.")
    if args.controlnet_variant is not None or args.guess_mode is not None or args.control_mode is not None:
        raise ValueError("Krea 2 Control-LoRA does not support a package variant, guess mode, or Union mode.")
    if args.control_guidance_start not in (None, 0.0) or args.control_guidance_end not in (None, 1.0):
        raise ValueError("Krea 2 Control-LoRA currently requires the full denoising interval.")
    if args.control_preprocessor_library and args.control_preprocess != "canny":
        raise ValueError("--control-preprocessor-library requires --control-preprocess canny.")
    if args.offload != "none":
        raise ValueError("Krea 2 Control-LoRA currently requires --offload none.")
    if args.seed is None:
        raise ValueError("Krea 2 Control-LoRA requires --seed for reproducible VAE sampling.")
    if not args.control_image:
        raise ValueError("--controlnet requires --control-image.")
    image = Path(args.control_image).expanduser().absolute()
    if not image.is_file() or not image.stat().st_size:
        raise ValueError("--control-image requires a nonempty local image.")
    if "control_image" in args.inputs:
        raise ValueError("Do not also supply control_image in --pipeline-inputs.")
    scale = 1.0 if args.controlnet_scale is None else args.controlnet_scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Krea 2 control LoRA scale must be finite and nonnegative.")
    return Selection(resolve_weight_file(args.controlnet, "Krea 2 control adapter"), str(image), scale,
                     auto_bucket=auto_bucket)


def plan_weights(shapes, module_shapes, *, projection_shape, block_count):
    if len(projection_shape) != 2 or block_count < 1:
        raise ValueError("Krea 2 Control-LoRA requires a valid base input projection and transformer blocks.")
    outputs, inputs = projection_shape
    if tuple(shapes.get("first.weight", ())) != (outputs, 2 * inputs) or tuple(shapes.get("first.bias", ())) != (outputs,):
        raise ValueError("Krea 2 Control-LoRA input projection does not match the base model.")
    expected = {"first.weight", "first.bias"}
    plan = {}
    for index in range(block_count):
        for original, destination in BLOCK_TARGETS.items():
            source = f"blocks.{index}.{original}"
            target = f"transformer_blocks.{index}.{destination}"
            a_key, b_key = source + ".A", source + ".B"
            expected.update((a_key, b_key))
            target_shape = module_shapes.get(target)
            a, b = shapes.get(a_key), shapes.get(b_key)
            if (target_shape is None or len(target_shape) != 2 or a is None or b is None
                    or len(a) != 2 or len(b) != 2 or a[0] < 1
                    or tuple(a[1:]) != (target_shape[1],)
                    or tuple(b) != (target_shape[0], a[0])):
                raise ValueError(f"Krea 2 Control-LoRA tensors do not match {target}.")
            plan[source] = target
    if set(shapes) != expected:
        raise ValueError("Krea 2 Control-LoRA has missing or unexpected tensors.")
    return plan


def _control_latents(pipeline, image, inputs, torch, np, *, output_identity=None):
    from PIL import Image
    width, height = inputs["width"], inputs["height"]
    vae = pipeline.vae
    parameter = next(vae.parameters())
    if isinstance(image, FloatDepth):
        depth = image.values
        if (not isinstance(depth, torch.Tensor) or not depth.is_floating_point()
                or tuple(depth.shape) != (height, width) or not bool(torch.isfinite(depth).all())
                or bool((depth < 0).any()) or bool((depth > 1).any())):
            raise ValueError("Krea 2 relative depth must be a finite [0,1] map at the output size.")
        if output_identity is not None:
            from control_hint_identity import fingerprint
            output_identity.update(fingerprint(image, np))
        pixels = depth.to(device=parameter.device, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
        pixels = pixels.mul(2).sub(1).unsqueeze(2).to(dtype=parameter.dtype)
    else:
        factor = max(width / image.width, height / image.height)
        resized = image.convert("RGB").resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)
        left, top = (resized.width - width) // 2, (resized.height - height) // 2
        cropped = resized.crop((left, top, left + width, top + height))
        if output_identity is not None:
            from control_hint_identity import fingerprint
            output_identity.update(fingerprint(cropped, np))
        # Normalize in FP32, then cast once for the VAE.
        pixels = torch.from_numpy(np.array(cropped, copy=True)).to(device=parameter.device, dtype=torch.float32)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).div(127.5).sub(1).to(dtype=parameter.dtype)
    generator = inputs.get("generator")
    if generator is None or isinstance(generator, list):
        raise ValueError("Krea 2 Control-LoRA requires one explicit seeded generator.")
    # Cloning the generation state repeats its very first normal samples in
    # the control VAE and in the initial image noise. Derive a separate stream
    # without consuming either the request generator or the global RNG.
    control_generator = derive_generator(generator, b'iild-krea2-control-vae-v1\0', torch)
    latent = vae.encode(pixels).latent_dist.sample(generator=control_generator)
    mean = torch.as_tensor(vae.config.latents_mean, device=latent.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=latent.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent = ((latent.float() - mean) / std).squeeze(2)
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Krea 2 control VAE produced non-finite latents.")
    batch, channels, latent_height, latent_width = latent.shape
    return pipeline._pack_latents(latent, batch, channels, latent_height, latent_width)


@contextmanager
def execution(pipeline, image, inputs, selection, torch, np):
    """Attach a single request's tensors and always restore the base transformer."""
    from safetensors import safe_open
    from weight_files import verify_weight_file
    transformer = pipeline.transformer
    modules = dict(transformer.named_modules())
    module_shapes = {name: tuple(module.weight.shape) for name, module in modules.items()
                     if getattr(module, "weight", None) is not None and module.weight.ndim == 2}
    projection = modules.get("img_in")
    if projection is None:
        raise ValueError("Krea 2 transformer has no input projection.")
    base_projection = projection.get_base_layer() if callable(getattr(projection, 'get_base_layer', None)) else projection
    if not isinstance(base_projection, torch.nn.Linear):
        raise ValueError("Krea 2 control requires a linear input projection.")
    verify_weight_file(selection.weight, "Krea 2 control adapter")
    with safe_open(selection.weight.path, framework="pt", device="cpu") as source:
        shapes = {name: tuple(source.get_slice(name).get_shape()) for name in source.keys()}
        plan = plan_weights(shapes, module_shapes, projection_shape=tuple(projection.weight.shape),
                            block_count=len(transformer.transformer_blocks))
        device, dtype = projection.weight.device, projection.weight.dtype
        if device.type == "meta":
            raise ValueError("Krea 2 control requires resident model weights.")
        input_identity = {}
        control = _control_latents(pipeline, image, inputs, torch, np, output_identity=input_identity).to(device=device, dtype=dtype)
        def tensor(name):
            value = source.get_tensor(name)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Krea 2 control tensor is non-floating or non-finite: {name}")
            converted = value.to(device=device, dtype=dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f"Krea 2 control tensor overflows the selected precision: {name}")
            return converted
        first = tensor("first.weight")
        bias = tensor("first.bias")
        pairs = [(modules[target], tensor(origin + ".A"), tensor(origin + ".B"))
                 for origin, target in plan.items()]
    verify_weight_file(selection.weight, "Krea 2 control adapter")
    handles = []
    try:
        def project(module, args, output):
            value = args[0]
            if control.shape[1:] != value.shape[1:]:
                raise ValueError("Krea 2 control latent does not match the generated image tokens.")
            conditioning = control.expand(value.shape[0], -1, -1).to(device=value.device, dtype=value.dtype)
            return torch.nn.functional.linear(torch.cat((value, conditioning), dim=-1), first, bias)
        # PEFT wraps the base linear and adds style residuals after its forward.
        # Replace only the base projection so those residuals remain active.
        handles.append(base_projection.register_forward_hook(project))
        for module, a, b in pairs:
            def add_adapter(module, args, output, a=a, b=b):
                return output + selection.scale * torch.nn.functional.linear(
                    torch.nn.functional.linear(args[0], a), b)
            handles.append(module.register_forward_hook(add_adapter))
        yield {"adapter": "krea2-expanded-input-control-lora", "blocks": len(transformer.transformer_blocks),
               "vae_input_hints": {"depth": input_identity},
               "scale": selection.scale, "control_latent_sampling": "independent stream derived from request generator state (v1)",
               "control_normalization": "FP32 pixels and VAE latent normalization before model dtype cast",
               "image_preprocessing": ("relative float depth, already resized and cropped" if isinstance(image, FloatDepth)
                                       else "RGB, Lanczos cover resize and center crop"),
               "control_size": [inputs["width"], inputs["height"]]}
    finally:
        for handle in reversed(handles):
            handle.remove()
