"""Local ControlNet composition for the generic built-in Diffusers runner."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace

from _file_provider import provider
from controlnet import _load_original_controlnet, _load_package, _package_files, validate_controlnet_contract
from model_loading import read_weight_keys, require_materialized_component
from generation_seed import derive_generator
from weight_files import (LocalWeightFile, checked_safetensors_path, file_sha256, file_signature,
                          resolve_weight_file, verify_weight_file, weight_file_metadata)
import krea2_controlnet
import krea2_color_control
import krea2_ostris_control
import moss_controlnet
import flux2_controlnet


# Explicit pairings prevent a same-named input from being mistaken for an
# architecture-compatible ControlNet. New families need their own contract.
FAMILIES = {
    "StableDiffusionPipeline": ("StableDiffusionControlNetPipeline", "ControlNetModel", "sd15", "image"),
    "StableDiffusionXLPipeline": ("StableDiffusionXLControlNetPipeline", "ControlNetModel", "sdxl-base", "image"),
    "StableDiffusion3Pipeline": ("StableDiffusion3ControlNetPipeline", "SD3ControlNetModel", "sd3", "control_image"),
    "FluxPipeline": ("FluxControlNetPipeline", "FluxControlNetModel", "flux1-schnell", "control_image"),
    "QwenImagePipeline": ("QwenImageControlNetPipeline", "QwenImageControlNetModel", "qwen-image", "control_image"),
}


@dataclass(frozen=True)
class Selection:
    directory: str
    base_pipeline: str
    target_pipeline: str
    component_class: str
    family: str
    image_argument: str
    variant: str | None
    weight: LocalWeightFile | None = None
    config: LocalWeightFile | None = None

    def metadata(self):
        return {"directory": self.directory, "base_pipeline": self.base_pipeline,
                "target_pipeline": self.target_pipeline, "component_class": self.component_class,
                "family": self.family, "image_argument": self.image_argument, "variant": self.variant,
                "weight": weight_file_metadata(self.weight) if self.weight else None,
                "config": ({"path": self.config.path, "sha256": self.config.sha256,
                            "size_bytes": self.config.size_bytes} if self.config else None)}


@dataclass(frozen=True)
class KleinReferenceSelection:
    """A reference-conditioning LoRA, not a separate ControlNet network."""
    weight: LocalWeightFile
    image: str
    reference: str
    scale: float
    resize: str = 'none'
    target_pipeline: str = 'Flux2KleinPipeline'
    image_argument: str = 'image'

    def metadata(self):
        return {'kind': 'flux2-klein-reference-lora', 'weight': weight_file_metadata(self.weight),
                'image': self.image, 'reference': self.reference, 'scale': self.scale,
                'reference_order': ['control', 'reference'], 'reference_resize': self.resize}


def resolve_klein_reference(args):
    if any(value is not None for value in (args.controlnet_config, args.controlnet_variant, args.guess_mode,
            args.control_mode, args.control_color_image, args.control_color_end, args.control_depth_end)):
        raise ValueError('FLUX.2 Klein reference control does not support ControlNet configuration, variants, modes, or color/depth controls.')
    if args.control_guidance_start not in (None, 0.) or args.control_guidance_end not in (None, 1.):
        raise ValueError('FLUX.2 Klein reference control requires the full denoising interval.')
    if args.control_preprocessor_library and args.control_preprocess != 'canny':
        raise ValueError('--control-preprocessor-library requires --control-preprocess canny.')
    if args.lora is not None or args.lora_scale is not None or args.lora_weight_name is not None or args.lora_revision is not None:
        raise ValueError('FLUX.2 Klein reference control cannot be combined with another LoRA.')
    reference = args.inputs.get('image')
    if (not isinstance(reference, dict) or set(reference) - {'image_path', 'mode'}
            or not isinstance(reference.get('image_path'), str) or reference.get('mode', 'RGB') != 'RGB'):
        raise ValueError('FLUX.2 Klein reference control requires one RGB image_path as image in --pipeline-inputs.')
    def local_image(value):
        if not isinstance(value, str) or not value:
            raise ValueError('FLUX.2 Klein reference control requires local control and reference images.')
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('FLUX.2 Klein reference control requires nonempty local control and reference images.')
        return str(path)
    image, reference_path = local_image(args.control_image), local_image(reference['image_path'])
    scale = 1. if args.controlnet_scale is None else args.controlnet_scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('FLUX.2 Klein reference control scale must be finite and nonnegative.')
    weight = resolve_weight_file(args.controlnet, 'FLUX.2 Klein reference LoRA')
    args.inputs['image'] = [{'image_path': image, 'mode': 'RGB'}, reference]
    return KleinReferenceSelection(weight, image, reference_path, scale,
                                   getattr(args, 'control_reference_resize', None) or 'none')


def add_options(parser):
    parser.add_argument("--controlnet", help="Local Diffusers ControlNet component directory or safetensors file")
    parser.add_argument("--controlnet-format", choices=("krea2-color-depth", "krea2-ostris-edit", "flux2-klein-reference"),
                        help="Select the Krea 2 or FLUX.2 Klein reference adapter format")
    parser.add_argument('--krea2-edit-reference-attention', choices=('isolated', 'joint'),
                        help='AI Toolkit edit reference attention; defaults to isolated')
    parser.add_argument("--controlnet-config", help="Local configuration directory for a single-file ControlNet")
    parser.add_argument("--control-image", help="Local prepared RGB conditioning image")
    parser.add_argument('--control-reference-resize', choices=('none', 'comfy-1mp'),
                        help='FLUX.2 Klein reference image resize policy')
    parser.add_argument("--control-inpaint-image", help="Local FLUX.2 Fun inpaint source image")
    parser.add_argument("--control-mask-image", help="Local FLUX.2 Fun inpaint mask (white marks repaint area)")
    parser.add_argument("--controlnet-variant")
    parser.add_argument("--controlnet-scale", "--controlnet-conditioning-scale", type=float)
    parser.add_argument("--control-guidance-start", type=float)
    parser.add_argument("--control-guidance-end", type=float)
    parser.add_argument("--guess-mode", action="store_true", default=None)
    parser.add_argument("--control-mode", type=int)
    parser.add_argument("--control-preprocess", choices=("none", "canny", "depth"))
    parser.add_argument("--control-preprocessor-library", help="Local iiLocalDiffusion shared library for native Canny")
    parser.add_argument("--control-depth-model", help="Local relative Depth Anything model directory")
    parser.add_argument("--control-depth-dtype", choices=("float16", "float32"), help="Depth model inference precision (default: float32)")
    parser.add_argument("--control-color-image", help="MOSS or Krea 2 color/region conditioning image")
    parser.add_argument('--control-map-preprocess', choices=('semantic', 'training'),
                        help='MOSS color/depth map spatial preprocessing')
    parser.add_argument("--control-color-end", type=float)
    parser.add_argument("--control-depth-end", type=float)
    parser.add_argument("--krea2-control-auto-bucket", action="store_true",
                        help="Select the public Krea 2 depth adapter's input-aspect resolution bucket")


def resolve(args, index):
    dependent = (args.control_image, args.controlnet_config, args.controlnet_variant, args.controlnet_scale,
                 args.control_guidance_start, args.control_guidance_end, args.guess_mode, args.control_mode,
                 args.control_preprocess, args.control_preprocessor_library, args.control_color_image,
                 getattr(args, 'control_map_preprocess', None),
                 args.control_color_end, args.control_depth_end, getattr(args, "control_depth_model", None),
                 getattr(args, "control_depth_dtype", None), getattr(args, "controlnet_format", None),
                 getattr(args, 'control_reference_resize', None),
                 getattr(args, "control_inpaint_image", None), getattr(args, "control_mask_image", None),
                 getattr(args, 'krea2_edit_reference_attention', None),
                 True if getattr(args, "krea2_control_auto_bucket", False) else None)
    if args.controlnet is None:
        if any(value is not None for value in dependent):
            raise ValueError("ControlNet options require --controlnet.")
        return None
    if args.control_preprocess == "depth" and not args.control_image:
        raise ValueError("--control-preprocess depth requires --control-image.")
    base = args.pipeline_class or index["_class_name"]
    if getattr(args, 'control_map_preprocess', None) is not None and base != 'MOSSImagePipeline':
        raise ValueError('--control-map-preprocess requires MOSSImagePipeline.')
    if base != 'Flux2Pipeline' and any(getattr(args, key, None) is not None
                                      for key in ('control_inpaint_image', 'control_mask_image')):
        raise ValueError('Control inpaint image and mask require Flux2Pipeline.')
    format_name = getattr(args, 'controlnet_format', None)
    if format_name in ('krea2-color-depth', 'krea2-ostris-edit') and base != 'Krea2Pipeline':
        raise ValueError('--controlnet-format requires Krea2Pipeline.')
    if format_name == 'flux2-klein-reference' and base != 'Flux2KleinPipeline':
        raise ValueError('flux2-klein-reference requires Flux2KleinPipeline.')
    if getattr(args, 'control_reference_resize', None) is not None and format_name != 'flux2-klein-reference':
        raise ValueError('--control-reference-resize requires flux2-klein-reference control.')
    if getattr(args, 'krea2_edit_reference_attention', None) and getattr(args, 'controlnet_format', None) != 'krea2-ostris-edit':
        raise ValueError('--krea2-edit-reference-attention requires krea2-ostris-edit control.')
    if getattr(args, "krea2_control_auto_bucket", False) and base != "Krea2Pipeline":
        raise ValueError("--krea2-control-auto-bucket requires Krea2Pipeline.")
    if base in ("Krea2Pipeline", "MOSSImagePipeline", "Flux2Pipeline") and args.controlnet_config is not None:
        raise ValueError("--controlnet-config is not supported for this adapter format.")
    if base == "Krea2Pipeline":
        if getattr(args, 'controlnet_format', None) == 'krea2-ostris-edit':
            return krea2_ostris_control.resolve(args)
        if getattr(args, 'controlnet_format', None) == 'krea2-color-depth':
            return krea2_color_control.resolve(args)
        return krea2_controlnet.resolve(args)
    if base == "MOSSImagePipeline":
        return moss_controlnet.resolve(args)
    if base == "Flux2Pipeline":
        return flux2_controlnet.resolve(args)
    if base == 'Flux2KleinPipeline' and format_name == 'flux2-klein-reference':
        return resolve_klein_reference(args)
    if base not in FAMILIES:
        raise ValueError(f"Generic ControlNet composition is not implemented for {base}.")
    if any(value is not None for value in (args.control_color_image, args.control_color_end, args.control_depth_end)):
        raise ValueError("Color/depth controls require MOSSImagePipeline.")
    target, component, family, image_argument = FAMILIES[base]
    if not args.controlnet:
        raise ValueError("--controlnet must be a local component directory or safetensors file.")
    source = Path(args.controlnet).expanduser().absolute()
    weight = None
    config_identity = None
    if source.is_dir():
        if args.controlnet_config is not None:
            raise ValueError("--controlnet-config requires a single-file ControlNet.")
        folder = source.resolve()
    else:
        weight = resolve_weight_file(str(source), 'ControlNet')
        if args.controlnet_variant is not None:
            raise ValueError("--controlnet-variant requires a ControlNet directory.")
        folder = Path(args.controlnet_config).expanduser().resolve() if args.controlnet_config else source.parent.resolve()
        if not folder.is_dir():
            raise ValueError("--controlnet-config must be a local directory.")
    config = folder / "config.json"
    if not config.is_file() or config.stat().st_size > 1024 * 1024:
        raise ValueError("ControlNet component requires a bounded local config.json.")
    parsed = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or parsed.get("_class_name") != component:
        raise ValueError(f"{base} requires a {component} component.")
    variant = args.controlnet_variant
    if variant is not None and (not variant or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in variant)):
        raise ValueError("Invalid ControlNet package variant.")
    if weight is not None:
        before = file_signature(config)
        digest = file_sha256(config)
        if file_signature(config) != before or json.loads(config.read_text(encoding='utf-8')) != parsed:
            raise RuntimeError('ControlNet configuration changed while reading.')
        config_identity = LocalWeightFile(str(config), str(config.resolve()), digest, config.stat().st_size)
    else:
        for path in _package_files(folder, variant):
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"ControlNet package file is missing or empty: {path}")
    if not args.control_image:
        raise ValueError("--controlnet requires --control-image.")
    if args.control_preprocessor_library and args.control_preprocess != "canny":
        raise ValueError("--control-preprocessor-library requires --control-preprocess canny.")
    image = Path(args.control_image).expanduser().absolute()
    if not image.is_file() or not image.stat().st_size:
        raise ValueError("--control-image requires a nonempty local image.")
    scale = 1.0 if args.controlnet_scale is None else args.controlnet_scale
    start = 0.0 if args.control_guidance_start is None else args.control_guidance_start
    end = 1.0 if args.control_guidance_end is None else args.control_guidance_end
    if not math.isfinite(scale) or scale < 0 or not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= 1:
        raise ValueError("ControlNet requires a finite nonnegative scale and 0 <= start < end <= 1.")
    if family not in ("sd15", "sdxl-base") and args.guess_mode:
        raise ValueError("--guess-mode is only supported for SD/SDXL.")
    if args.control_mode is not None and (family != "flux1-schnell" or args.control_mode < 0):
        raise ValueError("--control-mode requires a nonnegative FLUX Union mode.")
    values = {image_argument: {"image_path": str(image), "mode": "RGB"},
              "controlnet_conditioning_scale": scale,
              "control_guidance_start": start, "control_guidance_end": end}
    if args.guess_mode is not None:
        values["guess_mode"] = args.guess_mode
    if args.control_mode is not None:
        values["control_mode"] = args.control_mode
    duplicate = set(values) & set(args.inputs)
    if duplicate:
        raise ValueError("ControlNet arguments must not also appear in --pipeline-inputs: " + ", ".join(sorted(duplicate)))
    args.inputs.update(values)
    return Selection(str(folder), base, target, component, family, image_argument, variant, weight, config_identity)


def verify_selection(selection):
    if selection.weight is not None:
        verify_weight_file(selection.weight, 'ControlNet')
        verify_weight_file(selection.config, 'ControlNet configuration')


@contextmanager
def sd3_control_seed(pipeline, selection, seed, torch):
    """Give SD3's unseeded control-image VAE sample its own request generator."""
    if getattr(selection, 'family', None) != 'sd3' or seed is None:
        yield
        return
    vae = pipeline.vae
    instance_methods = vars(vae)
    had_override = 'encode' in instance_methods
    previous_override = instance_methods.get('encode')
    original_encode = vae.encode
    request_generator = torch.Generator(device='cpu').manual_seed(seed)
    generator = derive_generator(request_generator, b'iild-sd3-control-vae-v1\0', torch)

    def encode(*args, **kwargs):
        output = original_encode(*args, **kwargs)
        distribution = output.latent_dist
        original_sample = distribution.sample

        def sample(*sample_args, **sample_kwargs):
            if not sample_args and 'generator' not in sample_kwargs:
                sample_kwargs['generator'] = generator
            return original_sample(*sample_args, **sample_kwargs)

        distribution.sample = sample
        return output

    vae.encode = encode
    try:
        yield
    finally:
        if had_override:
            vae.encode = previous_override
        else:
            del vae.encode


def validate_component(control, pipeline, selection, args):
    if selection.family == 'sd3':
        if type(control).__name__ != 'SD3ControlNetModel':
            raise ValueError('Stable Diffusion 3 requires SD3ControlNetModel.')
        config, base = control.config, pipeline.transformer.config
        for field in ('patch_size', 'in_channels', 'attention_head_dim', 'num_attention_heads',
                      'joint_attention_dim', 'caption_projection_dim', 'pooled_projection_dim'):
            actual, expected = getattr(config, field, None), getattr(base, field, None)
            if actual is None or expected is None or actual != expected:
                raise ValueError(f'Stable Diffusion 3 ControlNet {field} does not match the base transformer.')
        if getattr(control, 'pos_embed', None) is None or getattr(control, 'context_embedder', None) is None:
            raise ValueError('Stable Diffusion 3 ControlNet requires latent and text embedding modules.')
        if (getattr(config, 'extra_conditioning_channels', 0) != 0
                or getattr(pipeline.vae.config, 'latent_channels', None) != config.in_channels):
            raise ValueError('Stable Diffusion 3 ControlNet requires VAE latent conditioning without extra channels.')
        layers, total = getattr(config, 'num_layers', None), getattr(base, 'num_layers', None)
        if (type(layers) is not int or type(total) is not int or total < 2
                or not 1 <= layers < total or (total - 2) * layers // total != layers - 1):
            raise ValueError('Stable Diffusion 3 ControlNet blocks must all contribute to the base transformer.')
        return
    if selection.family != 'qwen-image':
        return validate_controlnet_contract(control, pipeline,
                   SimpleNamespace(family=selection.family, name=selection.base_pipeline), args)
    if type(control).__name__ != 'QwenImageControlNetModel':
        raise ValueError('Qwen Image requires QwenImageControlNetModel.')
    config, base = control.config, pipeline.transformer.config
    for field in ('patch_size', 'in_channels', 'attention_head_dim', 'num_attention_heads',
                  'joint_attention_dim', 'axes_dims_rope'):
        actual, expected = getattr(config, field, None), getattr(base, field, None)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            actual, expected = tuple(actual), tuple(expected)
        if actual is None or expected is None or actual != expected:
            raise ValueError(f'Qwen Image ControlNet {field} does not match the base transformer.')
    if (base.patch_size != 2 or base.in_channels != 64 or type(pipeline.vae).__name__ != 'AutoencoderKLQwenImage'
            or getattr(pipeline.vae.config, 'z_dim', None) != 16
            or getattr(pipeline.vae.config, 'input_channels', 3) != 3
            or getattr(pipeline.vae, 'spatial_compression_ratio', None) != 8
            or getattr(config, 'extra_condition_channels', 0) != 0):
        raise ValueError('Qwen Image ControlNet requires the RGB 16-channel VAE and non-inpainting inputs.')
    if any(getattr(base, key, False) for key in ('guidance_embeds', 'zero_cond_t', 'use_additional_t_cond', 'use_layer3d_rope')):
        raise ValueError('Qwen Image ControlNet does not support this base transformer conditioning variant.')
    layers, base_layers = getattr(config, 'num_layers', None), getattr(base, 'num_layers', None)
    if (type(layers) is not int or type(base_layers) is not int or not 1 <= layers <= base_layers
            or (base_layers - 1) // math.ceil(base_layers / layers) + 1 != layers):
        raise ValueError('Qwen Image ControlNet block count would leave control blocks unused.')


def attach(pipeline, selection, args, diffusers, dtype):
    model_class = getattr(diffusers, selection.component_class, None)
    wrapper = getattr(diffusers, selection.target_pipeline, None)
    if model_class is None or wrapper is None:
        raise ValueError("The installed Diffusers runtime does not provide the selected ControlNet classes.")
    kwargs = {"dtype": dtype, "local_files_only": True, "use_safetensors": True, "low_cpu_mem_usage": True}
    if selection.variant:
        kwargs["variant"] = selection.variant
    verify_selection(selection)
    if selection.weight is None:
        control = _load_package(model_class, Path(selection.directory), kwargs)
    else:
        config = json.loads(Path(selection.config.path).read_text(encoding='utf-8'))
        verify_selection(selection)
        provider.mkdir(args.cache_dir, parents=True, exist_ok=True)
        with checked_safetensors_path(selection.weight, Path(args.cache_dir) / 'single-file-aliases', 'ControlNet') as path:
            keys = read_weight_keys(path)
            if any(key.startswith(('control_model.', 'input_blocks.')) for key in keys):
                if selection.family not in ('sd15', 'sdxl-base'):
                    raise ValueError(f'{selection.base_pipeline} ControlNet files require native Diffusers tensor names.')
                control = _load_original_controlnet(model_class, path, config, Path(selection.directory), kwargs)
            else:
                with provider.TemporaryDirectory(prefix='generic-controlnet-', dir=args.cache_dir) as temporary:
                    staging = Path(temporary)
                    provider.write_text(staging / 'config.json', json.dumps(config), encoding='utf-8')
                    (staging / 'diffusion_pytorch_model.safetensors').symlink_to(path.resolve())
                    control = _load_package(model_class, staging, kwargs)
        verify_selection(selection)
    require_materialized_component(control, "controlnet")
    validate_component(control, pipeline, selection, args)
    wrapped = wrapper.from_pipe(pipeline, dtype=dtype, controlnet=control)
    if type(wrapped) is not wrapper:
        raise RuntimeError("ControlNet composition returned an unexpected pipeline class.")
    return wrapped
