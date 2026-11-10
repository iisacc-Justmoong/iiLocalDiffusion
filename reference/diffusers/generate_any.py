#!/usr/bin/env python3
"""Run an installed Diffusers pipeline without assuming an SD1/SDXL/FLUX contract."""

from __future__ import annotations

from _file_provider import provider

import argparse
from contextlib import nullcontext
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import wave
from typing import Any

from generation_output import publish_file
from generation_defaults import pipeline_lora_family, resolve_family_default_lora
from generation_preview import add_preview_options, attach_preview, validate_preview_location
from generic_io import (ARCHITECTURES, automatic_token_spec, load_tensor_input,
                        validate_input_descriptor, validate_output_specs, write_tensor_output)
from hardware import accelerator_preflight, select_device, validate_execution_device
from lora import (add_lora_options, apply_lora, lora_metadata, resolve_lora_selection,
                  validate_lora_activation, validate_lora_support, verify_lora_identity)
from inference_session import (SchedulerConfiguration, cached_configuration, cached_pipeline,
                               cached_placement, is_preparing, record_device_placement, record_execution, verify_pipeline_sources)
from weight_files import model_content_sha256, file_sha256, file_signature
from vae_defaults import resolve_vae_selection, load_selected_vae, verify_vae_selection, vae_metadata, single_file_has_vae
import krea2_contract
import krea2_single_file
import krea2_text_encoder
import krea2_tokenizer
import generic_controlnet
import flux2_controlnet
import flux1_single_file
import flux1_transformer_stream
import flux_tokenizer
import flux_text_encoder
import flux_t5_stream
import klein_reference_resize
import krea2_controlnet
import krea2_color_control
import krea2_regions
import krea2_ostris_control
import moss_runtime
import moss_text_encoder
import model_config_defaults
import moss_tokenizer
import moss_controlnet
import moss_rapid
import moss_regions
import native_control_preprocessing
import depth_preprocessing
import control_hint_identity
from resource_paths import CACHE_DIRECTORY, OUTPUT_DIRECTORY

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = CACHE_DIRECTORY / "huggingface"
DEFAULT_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "generic-output"
IMAGE_INPUTS = frozenset({
    "image", "images", "mask_image", "control_image", "control_images", "control_color_image",
    "control_inpaint_image", "control_mask_image", "region_image", "reference_image",
    "reference_images", "ip_adapter_image", "conditioning_image", "video", "frames",
    "last_image", "start_image", "end_image", "image_2", "image_3", "image_4",
})
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
UNSAFE_WEIGHTS = frozenset({".ckpt", ".pt", ".pth", ".pkl", ".pickle", ".bin"})


def report_moss_progress(step: int, total: int) -> None:
    print('IILD_DIFFUSERS_PROGRESS ' + json.dumps({'schema': 'iild-diffusers-progress-v1',
          'stage': 'denoising', 'step': step, 'total': total}, separators=(',', ':')), flush=True)


def report_moss_loading_progress(component: str, completed: int, total: int) -> None:
    print('IILD_MODEL_PROGRESS ' + json.dumps({'schema': 'iild-model-progress-v1',
          'component': component, 'completed_bytes': completed, 'total_bytes': total},
          separators=(',', ':')), flush=True)


def report_dependency_progress(completed: int, _name: str) -> None:
    print('IILD_DEPENDENCY_PROGRESS ' + json.dumps({'schema': 'iild-dependency-progress-v1',
          'completed_modules': completed}, separators=(',', ':')), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", "--model-path", dest="model", required=True,
                        help="Local Diffusers directory or .safetensors file")
    parser.add_argument("--revision", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-class", help="Installed diffusers pipeline class; otherwise use model_index.json")
    parser.add_argument("--model-config", help="Local complete Diffusers configuration/extras for a single file")
    parser.add_argument('--model-config-preset', choices=('moss-image-sft', 'krea2-raw', 'krea2-turbo', 'flux1-schnell'),
                        help='Use a verified bundled single-file model configuration')
    parser.add_argument("--vae", help="Replacement local Diffusers VAE directory; omitted Qwen RGB VAE uses installed fallback")
    parser.add_argument("--base-model", help="Civitai identity checked against the selected pipeline architecture")
    add_lora_options(parser)
    parser.add_argument("--default-modifiers", action=argparse.BooleanOptionalAction, default=True,
                        help="Select the compatible default LoRA from the installed resource manifest")
    parser.add_argument("--generation-resources", type=Path,
                        help="Directory containing generation-defaults.json and its local adapters")
    parser.add_argument("--pipeline-inputs", default="{}",
                        help="JSON object or @JSON-file containing explicit pipeline __call__ arguments")
    generic_controlnet.add_options(parser)
    parser.add_argument('--moss-rapid-adapter', help='Local rapid-moss adapter directory containing its three safetensors files')
    parser.add_argument('--moss-text-encoder', help='Local Qwen3.5 BF16 or INT8 ConvRot single-file text encoder for MOSS')
    parser.add_argument('--moss-transformer-storage', choices=('dense', 'cpu-stream'),
                        help='Keep MOSS single-file transformer linear weights on CPU and transfer one layer at a time')
    parser.add_argument('--flux-clip-encoder', help='Local FLUX.1 CLIP text encoder safetensors')
    parser.add_argument('--flux-t5-encoder', help='Local FLUX.1 T5 text encoder safetensors')
    parser.add_argument('--flux-t5-fp8-storage', choices=('dense', 'cpu-stream'),
                        help='Keep scaled FP8 T5 linear weights on CPU and expand each layer when used')
    parser.add_argument('--flux-transformer-storage', choices=('dense', 'cpu-stream'),
                        help='Keep original FLUX.1 single-file transformer linear weights on CPU')
    parser.add_argument('--moss-regions', help='Local color-map regional prompt JSON for MOSS')
    parser.add_argument('--krea2-regions', help='Local color-map regional prompt JSON for Krea 2')
    parser.add_argument("--generation-architecture", choices=ARCHITECTURES, default="unspecified",
                        help="Caller-declared generation architecture; auto/unspecified never infer model semantics")
    parser.add_argument("--tensor-outputs", default="{}",
                        help="JSON object or @JSON-file mapping output fields to semantic/layout/representation_space")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--sigmas", type=read_json, help="Custom unshifted sigma array; Krea 2 validates its numerical contract")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "metal", "rocm"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    krea2_contract.add_options(parser)
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="none")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    add_preview_options(parser)
    parser.add_argument("--audio-sample-rate", type=int,
                        help="WAV rate; otherwise infer from the pipeline's vocoder/VAE/config")
    parser.add_argument("--video-layout", choices=("auto", "bfhwc", "bfchw", "bcfhw"), default="auto",
                        help="Array video layout; auto uses BFHWC for NumPy and Diffusers BFCHW for Torch")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not supported: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number is not supported: {value}")
    return parsed


def read_json(text: str) -> Any:
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def read_inputs(source: str) -> dict[str, Any]:
    value = read_json(Path(source[1:]).expanduser().read_text() if source.startswith("@") else source)
    if not isinstance(value, dict):
        raise ValueError("--pipeline-inputs must contain a JSON object.")
    for name in value:
        if not IDENTIFIER.fullmatch(name):
            raise ValueError(f"Invalid pipeline argument name: {name!r}")
    if "return_dict" in value and value["return_dict"] is not True:
        raise ValueError("return_dict must be true when supplied; named outputs are required for export.")
    if "generator" in value:
        raise ValueError("Use --seed to construct a torch.Generator; JSON generator objects are unsupported.")
    return value


def validate_input_descriptors(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            validate_input_descriptors(item)
    elif isinstance(value, dict):
        if "tensor_path" in value:
            validate_input_descriptor(value)
        else:
            for item in value.values():
                validate_input_descriptors(item)


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    if not args.local_files_only:
        raise ValueError("Generation requires local-only model loading.")
    if not args.model.strip():
        raise ValueError("--model must not be empty.")
    source = Path(args.model).expanduser()
    if source.exists():
        source = source.absolute()
        result.model = str(source)
        result.source_kind = "directory" if source.is_dir() else "single-file"
        if args.revision is not None:
            raise ValueError("--revision applies only to a remote Hub repository, not local files.")
        if not source.is_dir() and (not source.is_file() or source.suffix.casefold() not in (".safetensors", ".safetensor") or not source.stat().st_size):
            raise ValueError("Local model files must be non-empty safetensors files; use a Diffusers directory for other layouts.")
    else:
        raise ValueError(f"--model must be an existing local file or directory: {args.model}")
    result.local_files_only = True
    result.model_config_status = 'not-applicable'
    if args.model_config and getattr(args, 'model_config_preset', None):
        raise ValueError('--model-config and --model-config-preset cannot be combined.')
    if result.source_kind == "single-file":
        default_config = (model_config_defaults.resolve(args.pipeline_class, getattr(args, 'krea2_variant', 'auto'),
                                                         getattr(args, 'generation_resources', None),
                                                         preset=getattr(args, 'model_config_preset', None))
                          if not args.model_config else None)
        if not args.model_config and default_config is None:
            raise ValueError("A single-file model requires --model-config with local Diffusers configuration or a compatible --model-config-preset.")
        config = Path(args.model_config).expanduser().absolute() if args.model_config else default_config
        if not config.is_dir() or not (config / "model_index.json").is_file():
            raise ValueError("--model-config must be a local directory containing model_index.json.")
        result.model_config = str(config)
        result.model_config_status = 'explicit' if args.model_config else 'bundled'
        if getattr(args, 'model_config_preset', None) == 'flux1-schnell':
            flux1_single_file.validate_schnell(result.model, config)
    elif args.model_config or getattr(args, 'model_config_preset', None):
        raise ValueError("--model-config and --model-config-preset are only valid for a single-file --model.")
    if args.pipeline_class and not IDENTIFIER.fullmatch(args.pipeline_class):
        raise ValueError("--pipeline-class must be a public installed Diffusers class name, not a module or script path.")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - 1:
        raise ValueError("--seed must be in the inclusive range 0..2^63-1.")
    for name in ("width", "height", "steps", "audio_sample_rate"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.guidance_scale is not None and not math.isfinite(args.guidance_scale):
        raise ValueError("--guidance-scale must be finite.")
    if args.device == "cpu" and args.offload != "none":
        raise ValueError("CPU execution requires --offload none.")
    result.inputs = read_inputs(args.pipeline_inputs)
    if 'progress_callback' in result.inputs:
        raise ValueError('progress_callback is reserved for the generation worker.')
    tensor_source = args.tensor_outputs
    tensor_text = Path(tensor_source[1:]).expanduser().read_text() if tensor_source.startswith("@") else tensor_source
    result.tensor_outputs = validate_output_specs(read_json(tensor_text))
    if (result.inputs.get("output_type") in ("latent", "latents")
            and not any(spec["semantic"] == "latents" for spec in result.tensor_outputs.values())):
        raise ValueError("Latent outputs require --tensor-outputs with explicit latents semantic, layout, and representation_space.")
    validate_input_descriptors(result.inputs)
    for option, key in (("prompt", "prompt"), ("negative_prompt", "negative_prompt"),
                        ("width", "width"), ("height", "height"),
                        ("steps", "num_inference_steps"), ("guidance_scale", "guidance_scale"), ("sigmas", "sigmas")):
        value = getattr(args, option)
        if value is not None:
            result.inputs[key] = value
    result.cache_dir = args.cache_dir.expanduser().absolute()
    result.output_dir = args.output_dir.expanduser().absolute()
    index, folder = load_model_index(result)
    result.moss_text_encoder_selection = moss_text_encoder.resolve(result.moss_text_encoder,
                                                                    result.pipeline_class or index['_class_name'])
    if result.moss_text_encoder_selection:
        result.moss_text_encoder_status = 'explicit'
    else:
        result.moss_text_encoder_selection, result.moss_text_encoder_status = moss_text_encoder.resolve_fallback(
            result, index, folder)
    if result.moss_text_encoder_status == 'unavailable':
        raise ValueError('MOSS Image requires a compatible local Qwen3.5 text encoder. '
                         'Add or sync the verified MOSS encoder to the generation resource catalog, '
                         'or provide --moss-text-encoder explicitly.')
    if result.moss_text_encoder_status == 'fallback-int8-convrot' and result.offload != 'none':
        raise ValueError('MOSS INT8 ConvRot CPU storage requires --offload none.')
    result.moss_text_config_selection = (moss_text_encoder.resolve_config(folder, getattr(result, 'generation_resources', None))
                                         if result.moss_text_encoder_selection else None)
    result.moss_tokenizer_selection, result.moss_tokenizer_status = moss_tokenizer.resolve(result, index, folder)
    if result.moss_transformer_storage is not None and (result.source_kind != 'single-file'
            or index['_class_name'] != 'MOSSImagePipeline'
            or (result.pipeline_class is not None and result.pipeline_class != 'MOSSImagePipeline')):
        raise ValueError('--moss-transformer-storage requires a MOSS single-file checkpoint.')
    if result.moss_transformer_storage == 'cpu-stream' and result.offload != 'none':
        raise ValueError('MOSS transformer cpu-stream storage requires --offload none.')
    if result.krea2_fp8_storage is not None and (result.source_kind != 'single-file'
            or index['_class_name'] != 'Krea2Pipeline'
            or (result.pipeline_class is not None and result.pipeline_class != 'Krea2Pipeline')):
        raise ValueError('--krea2-fp8-storage requires a Krea 2 single-file checkpoint.')
    if result.krea2_fp8_storage == 'cpu-stream' and result.offload != 'none':
        raise ValueError('Krea 2 cpu-stream storage requires --offload none.')
    if result.flux_transformer_storage is not None and (result.source_kind != 'single-file'
            or index['_class_name'] != 'FluxPipeline'
            or (result.pipeline_class is not None and result.pipeline_class != 'FluxPipeline')):
        raise ValueError('--flux-transformer-storage requires a FLUX.1 single-file checkpoint.')
    if result.flux_transformer_storage == 'cpu-stream' and result.offload != 'none':
        raise ValueError('FLUX transformer cpu-stream storage requires --offload none.')
    result.krea2_control_size_explicit = any(key in result.inputs for key in ('width', 'height'))
    result.krea2 = krea2_contract.resolve(result, {**index, "_class_name": result.pipeline_class or index["_class_name"]})
    result.moss_regions = moss_regions.resolve(result, result.pipeline_class or index['_class_name'])
    result.krea2_regions = krea2_regions.resolve(result, result.pipeline_class or index['_class_name'])
    result.generic_controlnet = generic_controlnet.resolve(result, index)
    result.moss_rapid = moss_rapid.resolve(result, result.pipeline_class or index['_class_name'])
    if isinstance(result.generic_controlnet, (krea2_controlnet.Selection, krea2_color_control.Selection)) and result.generic_controlnet.auto_bucket:
        result.krea2['control_resolution'] = {'mode': 'input-aspect-bucket', 'status': 'pending-input-decode'}
        if result.krea2['mu_override'] is None:
            result.krea2['resolved_mu'] = None
    if is_preparing() and (result.generic_controlnet or result.moss_regions or result.krea2_regions):
        raise ValueError("Foreground preparation does not support ControlNet requests.")
    result.depth_preprocessor = depth_preprocessing.resolve(result.control_preprocess, result.control_depth_model,
                                                           result.control_depth_dtype)
    if isinstance(result.generic_controlnet, generic_controlnet.KleinReferenceSelection):
        result.lora = result.generic_controlnet.weight.path
        result.lora_scale = result.generic_controlnet.scale
    family = pipeline_lora_family(result.pipeline_class or index["_class_name"], folder)
    resolve_family_default_lora(family, result)
    result.lora_selection = resolve_lora_selection(result)
    if result.flux_transformer_storage == 'cpu-stream' and result.lora_selection is not None:
        raise ValueError('FLUX transformer cpu-stream storage does not support LoRA.')
    if result.moss_rapid and (result.lora_selection is not None or is_preparing()):
        raise ValueError('MOSS few-step sampling cannot use another LoRA or foreground preparation.')
    if isinstance(result.generic_controlnet, (krea2_ostris_control.Selection, moss_controlnet.Selection)) and result.lora_selection is not None:
        raise ValueError("This control adapter cannot currently be combined with another LoRA.")
    result.vae_selection, result.vae_status = resolve_vae_selection(result, index, folder)
    result.text_encoder_selection, result.text_encoder_status = krea2_text_encoder.resolve(result, index, folder)
    result.tokenizer_selection, result.tokenizer_status = krea2_tokenizer.resolve(
        result, index, folder, result.text_encoder_status)
    result.flux_tokenizer_selections, result.flux_tokenizer_status = flux_tokenizer.resolve(result, index, folder)
    result.flux_text_encoders = {}
    result.flux_text_encoder_status = {}
    for role, path in (('text_encoder', result.flux_clip_encoder), ('text_encoder_2', result.flux_t5_encoder)):
        if path is not None:
            expected = ['transformers', flux_text_encoder.ROLES[role][0]]
            if index.get(role) not in (None, [None, None], expected):
                raise ValueError(f'FLUX {role} declares an incompatible encoder.')
            result.flux_text_encoders[role] = flux_text_encoder.resolve_file(
                path, role, result.pipeline_class or index['_class_name'], folder, result.generation_resources)
            result.flux_text_encoder_status[role] = 'explicit'
        elif (result.pipeline_class or index['_class_name']) in flux_tokenizer.FLUX1_CLIP_PIPELINES:
            selected, status = flux_text_encoder.resolve_fallback(result, index, folder, role)
            if (role == 'text_encoder_2' and status == 'model' and result.source_kind == 'single-file'
                    and result.flux_t5_fp8_storage == 'cpu-stream'
                    and flux_text_encoder._single_file_has_encoder(result.model, role)):
                selected = flux_text_encoder.resolve_embedded_t5(result.model,
                    result.pipeline_class or index['_class_name'], folder, result.generation_resources)
                status = 'embedded'
            result.flux_text_encoder_status[role] = status
            if selected:
                result.flux_text_encoders[role] = selected
    if result.flux_t5_fp8_storage is not None and 'text_encoder_2' not in result.flux_text_encoders:
        raise ValueError('--flux-t5-fp8-storage requires a local FLUX.1 T5 single-file encoder.')
    if result.flux_t5_fp8_storage == 'cpu-stream' and result.offload != 'none':
        raise ValueError('FLUX T5 cpu-stream storage requires --offload none.')
    if result.flux_t5_fp8_storage == 'cpu-stream' and result.lora_selection is not None:
        flux_t5_stream.validate_lora(result.lora_selection)
    validate_preview_location(args.preview_dir, result.output_dir)
    return result


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_inputs = dict(args.inputs)
    if (getattr(args, 'krea2', None) or {}).get('control_resolution', {}).get('status') == 'pending-input-decode':
        pipeline_inputs.pop('width', None)
        pipeline_inputs.pop('height', None)
    return {
        "runner": "generic-diffusers", "model": args.model, "source_kind": args.source_kind,
        "revision": args.revision, "model_config": args.model_config,
        "model_config_status": getattr(args, 'model_config_status', 'not-applicable'),
        "model_config_preset": getattr(args, 'model_config_preset', None), "base_model": args.base_model,
        "pipeline_class": args.pipeline_class, "pipeline_inputs": pipeline_inputs, "seed": args.seed,
        "lora": args.lora, "lora_scale": args.lora_selection.scale if args.lora_selection else None,
        "lora_weight_name": args.lora_weight_name,
        "generation_defaults": args.default_modifier_metadata,
        "vae": vae_metadata(args.vae_selection, args.vae_status),
        "text_encoder": (({**args.text_encoder_selection.metadata(), 'class': 'Qwen3VLModel', 'vision_enabled': True}
                          if isinstance(args.generic_controlnet, krea2_ostris_control.Selection)
                          else args.text_encoder_selection.metadata()) if args.text_encoder_selection
                         else {"status": args.text_encoder_status}),
        "tokenizer": (args.tokenizer_selection.metadata() if args.tokenizer_selection
                      else {"status": args.tokenizer_status}),
        "device": args.device, "dtype": args.dtype, "offload": args.offload,
        "krea2": getattr(args, "krea2", None), "krea2_fp8_storage": getattr(args, 'krea2_fp8_storage', None) or 'dense',
        "moss_transformer_storage": getattr(args, 'moss_transformer_storage', None) or 'dense',
        "flux_transformer_storage": getattr(args, 'flux_transformer_storage', None) or 'dense',
        "moss_text_encoder": ({'mode': 'single-file-language-only', 'path': args.moss_text_encoder_selection.path,
                               'sha256': args.moss_text_encoder_selection.sha256,
                               'size_bytes': args.moss_text_encoder_selection.size_bytes,
                               'status': getattr(args, 'moss_text_encoder_status', 'explicit'),
                               'config': vars(args.moss_text_config_selection)}
                              if getattr(args, 'moss_text_encoder_selection', None) else None),
        "moss_text_encoder_status": getattr(args, 'moss_text_encoder_status', 'not-configured-for-pipeline'),
        "moss_tokenizer": (args.moss_tokenizer_selection.metadata() if getattr(args, 'moss_tokenizer_selection', None)
                           else {'status': getattr(args, 'moss_tokenizer_status', 'not-configured-for-pipeline')}),
        "flux_tokenizers": {role: (args.flux_tokenizer_selections[role].metadata()
                            if role in getattr(args, 'flux_tokenizer_selections', {}) else {'status': status})
                            for role, status in getattr(args, 'flux_tokenizer_status', {}).items()},
        "flux_text_encoders": {role: selected.metadata() for role, selected in getattr(args, 'flux_text_encoders', {}).items()},
        "flux_text_encoder_status": getattr(args, 'flux_text_encoder_status', {}),
        "flux_t5_fp8_storage": getattr(args, 'flux_t5_fp8_storage', None) or 'dense',
        "controlnet": args.generic_controlnet.metadata() if args.generic_controlnet else None,
        "moss_rapid": args.moss_rapid.metadata() if getattr(args, 'moss_rapid', None) else None,
        "moss_regions": args.moss_regions.metadata() if getattr(args, 'moss_regions', None) else None,
        "krea2_regions": args.krea2_regions.metadata() if getattr(args, 'krea2_regions', None) else None,
        "control_preprocess": args.control_preprocess or "none",
        "control_preprocessor_library": args.control_preprocessor_library,
        "control_depth_model": args.depth_preprocessor.directory if args.depth_preprocessor else None,
        "control_depth_dtype": args.depth_preprocessor.precision if args.depth_preprocessor else None,
        "local_files_only": args.local_files_only, "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir), "audio_sample_rate": args.audio_sample_rate,
        "preview_dir": str(args.preview_dir) if args.preview_dir is not None else None,
        "video_layout": args.video_layout,
        "generation_architecture": args.generation_architecture,
        "generation_architecture_status": "caller-declared" if args.generation_architecture not in ("auto", "unspecified") else "not-inferred",
        "tensor_outputs": args.tensor_outputs,
        "overwrite": args.overwrite,
    }


def is_pipeline_configuration(index: dict[str, Any], name: str, value: Any) -> bool:
    # Krea 2 stores a list of tapped text layers in model_index.json. It is
    # configuration, not a [library, class] component to import.
    if index.get("_class_name") != "Krea2Pipeline" or name != "text_encoder_select_layers":
        return False
    if value is not None and (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 256
                              or any(type(layer) is not int or not 0 <= layer <= 1024 for layer in value)):
        raise ValueError("Krea 2 text_encoder_select_layers must be a bounded list of nonnegative layer indices.")
    return True


def validate_model_index(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model_index.json must contain an object.")
    name = value.get("_class_name")
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ValueError("model_index.json must identify a built-in pipeline class; custom pipeline scripts are unsupported.")
    for component, selection in value.items():
        if is_pipeline_configuration(value, component, selection):
            continue
        if component.startswith("_") or not isinstance(selection, (list, tuple)):
            continue
        if not IDENTIFIER.fullmatch(component) or len(selection) != 2:
            raise ValueError(f"Invalid model component declaration: {component}")
        library, class_name = selection
        if library is None and class_name is None:
            continue
        if (not isinstance(library, str) or not IDENTIFIER.fullmatch(library)
                or not isinstance(class_name, str) or not IDENTIFIER.fullmatch(class_name)):
            raise ValueError(f"Custom or malformed component declaration: {component}")
    return value


def builtin_pipeline(diffusers: Any, name: str) -> Any:
    if name == "MOSSImagePipeline":
        candidate = moss_runtime.pipeline_class()
        if not inspect.isclass(candidate) or not issubclass(candidate, diffusers.DiffusionPipeline):
            raise ValueError("The pinned MOSS pipeline is incompatible with the installed Diffusers runtime.")
        return candidate
    candidate = getattr(diffusers, name, None)
    if (not inspect.isclass(candidate) or not issubclass(candidate, diffusers.DiffusionPipeline)
            or candidate is diffusers.DiffusionPipeline
            or not candidate.__module__.startswith("diffusers.pipelines.")):
        raise ValueError(f"{name!r} is not a built-in pipeline in the installed Diffusers runtime. Use a compatible installed runtime or the workflow backend.")
    return candidate


def architecture_group(pipeline_class: Any) -> str:
    # These task siblings live in multiple namespaces (notably ControlNet), so
    # namespace equality alone incorrectly rejects compatible editing pipelines.
    name = pipeline_class.__name__
    if name == "MOSSImagePipeline":
        return "moss-image"
    for prefix, group in (("StableDiffusionXL", "sdxl"), ("StableDiffusion3", "sd3"),
                          ("StableDiffusion", "sd1-or-sd2"), ("Flux2", "flux2"),
                          ("Flux", "flux1"), ("QwenImage", "qwen-image"),
                          ("HunyuanDiT", "hunyuan-dit")):
        if name.startswith(prefix):
            return group
    parts = pipeline_class.__module__.split(".")
    if len(parts) < 3 or parts[:2] != ["diffusers", "pipelines"]:
        raise ValueError(f"Cannot identify the architecture namespace for {name}.")
    return "diffusers:" + parts[2]


def validate_base_model(label: str | None, pipeline_class: Any, diffusers: Any) -> dict[str, Any]:
    actual = architecture_group(pipeline_class)
    if label is None:
        return {"status": "not-declared", "actual_architecture": actual}
    from civitai_catalog import lookup_base_model
    record = lookup_base_model(label)
    if record["local_status"] == "hosted":
        raise ValueError(f"{record['name']} is a hosted-only Civitai base model, not a local pipeline.")
    expected_name = record["pipeline_class"]
    if expected_name is None:
        return {"status": "declared-only", "catalog_name": record["name"],
                "actual_architecture": actual,
                "reason": "The catalog has no conventional built-in pipeline class for this label."}
    expected = architecture_group(builtin_pipeline(diffusers, expected_name))
    if actual != expected:
        raise ValueError(f"Civitai base model {record['name']!r} requires architecture {expected!r} "
                         f"({expected_name}), but {pipeline_class.__name__} uses {actual!r}.")
    return {"status": "pipeline-architecture-verified", "catalog_name": record["name"],
            "catalog_pipeline_class": expected_name, "actual_architecture": actual,
            "scope": "Pipeline architecture only; checkpoint identity, SD1 versus SD2, and variants within one family are not inferred."}


def validate_components(diffusers: Any, index: dict[str, Any], folder: Path | None = None) -> None:
    moss = index.get("_class_name") == "MOSSImagePipeline"
    if moss:
        moss_runtime.validate_index(index)
    for name, value in index.items():
        if is_pipeline_configuration(index, name, value):
            continue
        if name.startswith("_") or not isinstance(value, (list, tuple)) or value == [None, None]:
            continue
        library, _ = value
        if moss and name in ("transformer", "scheduler") and value == moss_runtime.COMPONENTS[name]:
            # The installed, pinned implementation is used; the package's Python file is ignored.
            continue
        if library not in ("diffusers", "transformers"):
            module = getattr(diffusers.pipelines, library, None)
            if module is None or not getattr(module, "__name__", "").startswith("diffusers.pipelines."):
                raise ValueError(f"Component {name} requests non-built-in library {library!r}.")
        if folder is not None and (folder / name / f"{library}.py").exists():
            raise ValueError(f"Custom local component code is unsupported: {name}/{library}.py")


def load_model_index(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    if args.source_kind not in ("directory", "single-file"):
        raise ValueError("Generation requires a local model source.")
    folder = Path(args.model_config if args.source_kind == "single-file" else args.model)
    path = folder / "model_index.json"
    index = cached_configuration(("generic-model-index", str(path)), [path], lambda: read_json(path.read_text()))
    return validate_model_index(index), folder


def file_identity(path: Path, relative_to: Path | None = None, *, model_file: bool = False) -> dict[str, Any]:
    before = file_signature(path)
    digest = model_content_sha256(path) if model_file else file_sha256(path)
    after = path.stat()
    if before != file_signature(path):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {"path": str(path.relative_to(relative_to)) if relative_to else str(path),
            "resolved_path": str(path.resolve()), "size_bytes": after.st_size, "sha256": digest,
            **({"validation": "metadata", "signature": list(before)} if digest is None else {})}


def model_files(folder: Path) -> list[Path]:
    return [path for path in sorted(provider.rglob(folder, "*"))
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(folder).parts)
            and path.suffix not in UNSAFE_WEIGHTS | {".py", ".pyc"}]


def model_identity(folder: Path, single_file: str | None = None) -> dict[str, Any]:
    files = [file_identity(path, folder, model_file=True) for path in model_files(folder)]
    result: dict[str, Any] = {"directory": str(folder), "files": files}
    if single_file:
        result["single_file"] = file_identity(Path(single_file), model_file=True)
    return result


def verify_identity(identity: dict[str, Any]) -> None:
    folder = Path(identity["directory"])
    if [str(path.relative_to(folder)) for path in model_files(folder)] != [entry["path"] for entry in identity["files"]]:
        raise RuntimeError(f"Model directory changed while loading/generating: {folder}")
    entries = list(identity["files"])
    if identity.get("single_file"):
        entries.append(identity["single_file"])
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = Path(identity["directory"]) / path
        if (not path.is_file() or str(path.resolve()) != entry["resolved_path"]
                or path.stat().st_size != entry["size_bytes"]
                or (entry.get("signature") is not None and file_signature(path) != tuple(entry["signature"]))
                or (entry["sha256"] is not None and model_content_sha256(path) != entry["sha256"])):
            raise RuntimeError(f"Model file changed while loading/generating: {path}")


def load_pipeline(args: argparse.Namespace, diffusers: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    index, folder = load_model_index(args)
    validate_components(diffusers, index, folder)
    pipeline_class = builtin_pipeline(diffusers, args.pipeline_class or index["_class_name"])
    if index["_class_name"] == "MOSSImagePipeline" and pipeline_class.__name__ != "MOSSImagePipeline":
        raise ValueError("MOSS packages require the pinned MOSSImagePipeline.")
    validate_base_model(args.base_model, pipeline_class, diffusers)
    if getattr(args, "lora_selection", None) is not None:
        validate_lora_support(pipeline_class)
    # Validate user input names before allocating model weights, including **kwargs pipelines.
    selected_control = getattr(args, "generic_controlnet", None)
    call_class = builtin_pipeline(diffusers, selected_control.target_pipeline) if selected_control else pipeline_class
    validate_call_arguments(call_class.__call__, args.inputs, seed=args.seed)
    assert folder is not None
    krea_single = (args.source_kind == 'single-file' and index['_class_name'] == 'Krea2Pipeline'
                   and pipeline_class.__name__ == 'Krea2Pipeline')
    moss_single = (args.source_kind == 'single-file' and index['_class_name'] == 'MOSSImagePipeline'
                   and pipeline_class.__name__ == 'MOSSImagePipeline')
    if args.source_kind == "single-file":
        if not krea_single and not moss_single and not hasattr(pipeline_class, "from_single_file"):
            raise ValueError(f"{pipeline_class.__name__} has no built-in single-file loader. Provide a complete Diffusers directory or use the workflow backend.")
        unsafe = [str(path.relative_to(folder)) for path in provider.rglob(folder, "*")
                  if path.is_file() and path.suffix in UNSAFE_WEIGHTS]
        if unsafe:
            raise ValueError("Single-file configuration extras must contain safetensors weights only; remove/convert pickle weights: " + ", ".join(unsafe[:5]))
    identity = model_identity(folder, args.model if args.source_kind == "single-file" else None)
    control_identity = (model_identity(Path(selected_control.directory))
                        if isinstance(selected_control, generic_controlnet.Selection) and selected_control.weight is None else None)
    kwargs = {"dtype": dtype, "local_files_only": True, "cache_dir": str(args.cache_dir),
              "use_safetensors": True, "trust_remote_code": False}
    # A saved SD3 pipeline may explicitly omit T5. Diffusers requires those
    # constructor arguments as None even though model_index.json records them.
    parameters = inspect.signature(pipeline_class.__init__).parameters
    kwargs.update({name: None for name, value in index.items()
                   if value == [None, None] and name in parameters})
    selection = getattr(args, "vae_selection", None)
    if selection is not None and pipeline_class.__name__ != "MOSSImagePipeline":
        if "vae" not in parameters:
            raise ValueError("The selected pipeline does not accept a VAE component.")
        kwargs["vae"] = load_selected_vae(selection, diffusers, dtype)
    text_encoder = getattr(args, 'text_encoder_selection', None)
    if text_encoder is not None:
        if 'text_encoder' not in parameters:
            raise ValueError('The selected pipeline does not accept a text encoder component.')
        kwargs['text_encoder'] = (krea2_text_encoder.load_full(text_encoder, dtype)
                                  if isinstance(selected_control, krea2_ostris_control.Selection)
                                  else krea2_text_encoder.load(text_encoder, dtype))
    elif getattr(args, 'text_encoder_status', None) == 'precomputed-embeddings':
        kwargs['text_encoder'] = None
    tokenizer = getattr(args, 'tokenizer_selection', None)
    if tokenizer is not None:
        if 'tokenizer' not in parameters:
            raise ValueError('The selected pipeline does not accept a tokenizer component.')
        kwargs['tokenizer'] = krea2_tokenizer.load(tokenizer)
    elif getattr(args, 'tokenizer_status', None) == 'precomputed-embeddings':
        kwargs['tokenizer'] = None
    flux_selections = getattr(args, 'flux_tokenizer_selections', {})
    for role, status in getattr(args, 'flux_tokenizer_status', {}).items():
        if role in flux_selections:
            if role not in parameters:
                raise ValueError(f'The selected pipeline does not accept {role}.')
            kwargs[role] = flux_tokenizer.load(flux_selections[role])
        elif status == 'precomputed-embeddings' and role in parameters:
            kwargs[role] = None
    flux_encoders = getattr(args, 'flux_text_encoders', {})
    flux_encoder_loading = {}
    for role, status in getattr(args, 'flux_text_encoder_status', {}).items():
        if status == 'precomputed-embeddings' and role in parameters:
            kwargs[role] = None
    for role, selected in flux_encoders.items():
        if role not in parameters:
            raise ValueError(f'The selected pipeline does not accept {role}.')
        storage = (getattr(args, 'flux_t5_fp8_storage', None) or 'dense') if role == 'text_encoder_2' else 'dense'
        kwargs[role] = flux_text_encoder.load(selected, dtype, storage=storage)
        flux_encoder_loading[role] = kwargs[role]._iild_flux_text_encoder_loading
    single_file_loading = None
    if krea_single:
        kwargs['transformer'] = krea2_single_file.load(args.model, folder / 'transformer', dtype,
                                                       storage=getattr(args, 'krea2_fp8_storage', None) or 'dense')
        single_file_loading = kwargs['transformer']._iild_single_file_loading
        pipeline = pipeline_class.from_pretrained(str(folder), **kwargs)
    elif pipeline_class.__name__ == "MOSSImagePipeline":
        moss_kwargs = {'progress': report_moss_loading_progress}
        if moss_single:
            moss_kwargs['transformer_file'] = args.model
            moss_kwargs['transformer_storage'] = getattr(args, 'moss_transformer_storage', None) or 'dense'
        if getattr(args, 'moss_text_encoder_selection', None):
            moss_kwargs['text_encoder_file'] = args.moss_text_encoder_selection
            moss_kwargs['text_config_selection'] = args.moss_text_config_selection
        if getattr(args, 'moss_tokenizer_selection', None):
            moss_kwargs['tokenizer_selection'] = args.moss_tokenizer_selection
        pipeline = moss_runtime.load(folder, dtype, selection, diffusers, **moss_kwargs)
        if moss_single:
            single_file_loading = pipeline.transformer._iild_single_file_loading
    elif args.source_kind == "single-file":
        if (pipeline_class.__name__ == 'FluxPipeline' and getattr(args, 'vae_status', None) == 'model'
                and single_file_has_vae(args.model)):
            flux1_single_file.validate_embedded_vae(args.model, folder, diffusers)
        if pipeline_class.__name__ == 'FluxPipeline' and getattr(args, 'flux_transformer_storage', None) == 'cpu-stream':
            kwargs['transformer'] = flux1_transformer_stream.load(args.model, folder, dtype)
            single_file_loading = kwargs['transformer']._iild_single_file_loading
        pipeline = pipeline_class.from_single_file(args.model, config=str(folder), **kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(str(folder), **kwargs)
    if pipeline.__class__ is not pipeline_class:
        raise RuntimeError(f"Requested {pipeline_class.__name__}, but loader instantiated {pipeline.__class__.__name__}.")
    flux_t5_embedded = None
    if args.source_kind == 'single-file' and pipeline_class.__name__ in flux_tokenizer.FLUX1_CLIP_PIPELINES:
        encoder_status = getattr(args, 'flux_text_encoder_status', {})
        if encoder_status.get('text_encoder') == 'model':
            from model_loading import require_materialized_component
            clip = getattr(pipeline, 'text_encoder', None)
            if clip is None:
                raise RuntimeError('Embedded FLUX CLIP encoder was not loaded.')
            require_materialized_component(clip, 'FLUX.text_encoder')
        if encoder_status.get('text_encoder_2') == 'model':
            flux_t5_embedded = flux_text_encoder.repair_embedded_t5(getattr(pipeline, 'text_encoder_2', None))
    flux_clip_limit = flux_tokenizer.bound_clip_length(pipeline)
    krea2_contract.validate_pipeline(pipeline, getattr(args, "krea2", None))
    if isinstance(selected_control, generic_controlnet.Selection):
        pipeline = generic_controlnet.attach(pipeline, selected_control, args, diffusers, dtype)
        if control_identity:
            verify_identity(control_identity)
        generic_controlnet.verify_selection(selected_control)
    verify_identity(identity)
    verify_vae_selection(selection)
    if getattr(args, 'moss_text_encoder_selection', None):
        from weight_files import verify_weight_file
        verify_weight_file(args.moss_text_encoder_selection, 'MOSS Qwen3.5 text encoder')
        moss_text_encoder.verify_config(args.moss_text_config_selection)
    if getattr(args, 'moss_tokenizer_selection', None):
        krea2_tokenizer.verify_selection(args.moss_tokenizer_selection)
    krea2_text_encoder.verify_selection(text_encoder)
    krea2_tokenizer.verify_selection(tokenizer)
    flux_tokenizer.verify_selections(flux_selections)
    for selected in flux_encoders.values():
        flux_text_encoder.verify_selection(selected)
    return pipeline, {"identity": identity, "controlnet_identity": control_identity, "index_class": index["_class_name"],
                      "actual_class": pipeline.__class__.__name__,
                      "base_model_validation": validate_base_model(args.base_model, pipeline.__class__, diffusers),
                      **({'flux_clip_token_limit': flux_clip_limit} if flux_clip_limit else {}),
                      **({'flux_t5_embedded_loading': flux_t5_embedded} if flux_t5_embedded else {}),
                      **({'flux_text_encoder_loading': flux_encoder_loading} if flux_encoder_loading else {}),
                      **({'single_file_loading': single_file_loading} if single_file_loading is not None else {})}


def validate_call_arguments(call: Any, inputs: dict[str, Any], *, seed: int | None) -> None:
    signature = inspect.signature(call)
    allowed = {name for name, parameter in signature.parameters.items()
               if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
    allowed.discard("self")
    requested = set(inputs) | ({"generator"} if seed is not None else set())
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError("Pipeline does not explicitly support input(s): " + ", ".join(unknown)
                         + ". Unknown **kwargs are refused rather than silently ignored.")
    required = {name for name, parameter in signature.parameters.items()
                if name != "self" and parameter.default is inspect.Parameter.empty
                and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,
                                       parameter.KEYWORD_ONLY)}
    missing = sorted(required - requested)
    if missing:
        raise ValueError("Pipeline requires input(s): " + ", ".join(missing))


def orient_image(image):
    """Apply the same EXIF orientation policy as the preset ControlNet runner."""
    from PIL import ImageOps
    return ImageOps.exif_transpose(image)


def prepare_inputs(inputs: dict[str, Any], image_module: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identities: list[dict[str, Any]] = []

    def convert(value: Any, root_key: str) -> Any:
        if isinstance(value, list):
            return [convert(item, root_key) for item in value]
        if isinstance(value, dict):
            if "tensor_path" in value:
                tensor, identity = load_tensor_input(value, file_identity)
                identities.append({**identity, "input": root_key})
                return tensor
            if "image_path" in value:
                if set(value) - {"image_path", "mode"} or root_key not in IMAGE_INPUTS:
                    raise ValueError(f"image_path is only accepted for supported image/video inputs, not {root_key!r}.")
                if not isinstance(value["image_path"], str):
                    raise ValueError("image_path must be a local file path string.")
                mode = value.get("mode", "RGB")
                if mode not in ("RGB", "RGBA", "L"):
                    raise ValueError("Image mode must be RGB, RGBA, or L.")
                path = Path(value["image_path"]).expanduser().absolute()
                identity = file_identity(path)
                with image_module.open(path) as opened:
                    if getattr(opened, "is_animated", False):
                        raise ValueError("Animated input files are unsupported; pass an explicit list of frame image_path objects.")
                    opened.load()
                    image = orient_image(opened).convert(mode)
                if file_sha256(path) != identity["sha256"]:
                    raise RuntimeError(f"Image input changed while decoding: {path}")
                identities.append({**identity, "input": root_key, "mode": mode, "size": list(image.size),
                                   "exif_transpose": True})
                return image
            return {key: convert(item, root_key) for key, item in value.items()}
        return value

    return {name: convert(value, name) for name, value in inputs.items()}, identities


def verify_input_files(identities: list[dict[str, Any]]) -> None:
    for identity in identities:
        try:
            current = file_identity(Path(identity["path"]))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Pipeline input changed after loading: {identity['path']}") from error
        if any(current[key] != identity[key] for key in ("resolved_path", "size_bytes", "sha256")):
            raise RuntimeError(f"Pipeline input changed after loading: {identity['path']}")


def prepare_execution(pipeline: Any, device: str, offload: str) -> Any:
    if offload != 'none' and getattr(getattr(pipeline, 'text_encoder', None), '_iild_cpu_int8_convrot', False) is True:
        raise ValueError('MOSS INT8 ConvRot CPU storage requires --offload none.')
    if offload != 'none' and getattr(getattr(pipeline, 'transformer', None), '_iild_cpu_stream', False) is True:
        raise ValueError('Streamed transformer storage requires --offload none.')
    record_device_placement()
    if offload == "none":
        pipeline = pipeline.to(device)
    elif offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.enable_sequential_cpu_offload(device=device)
    validate_execution_device(pipeline, device)
    return pipeline


def prepared_pipeline(args, diffusers, dtype, device):
    index, folder = load_model_index(args)
    pipeline_name = args.pipeline_class or index["_class_name"]
    selection = getattr(args, "lora_selection", None)
    verify_lora_identity(selection)
    vae = getattr(args, "vae_selection", None)
    verify_vae_selection(vae)
    text_encoder = getattr(args, 'text_encoder_selection', None)
    krea2_text_encoder.verify_selection(text_encoder)
    tokenizer = getattr(args, 'tokenizer_selection', None)
    krea2_tokenizer.verify_selection(tokenizer)
    flux_tokenizers = getattr(args, 'flux_tokenizer_selections', {})
    flux_tokenizer.verify_selections(flux_tokenizers)
    flux_encoders = getattr(args, 'flux_text_encoders', {})
    for selected in flux_encoders.values():
        flux_text_encoder.verify_selection(selected)
    key = (("diffusers", pipeline_name, str(dtype), repr(selection), repr(vae), repr(text_encoder), repr(tokenizer), repr(flux_tokenizers), repr(flux_encoders),
            (getattr(args, "krea2", None) or {}).get("variant"), getattr(args, 'krea2_fp8_storage', None),
            getattr(args, 'flux_t5_fp8_storage', None), getattr(args, 'flux_transformer_storage', None))
           if not args.generic_controlnet and pipeline_name in ("StableDiffusionPipeline", "StableDiffusionXLPipeline", "FluxPipeline", "Krea2Pipeline") else None)
    if is_preparing() and key is None:
        raise ValueError("Foreground preparation requires a retained local image pipeline.")
    sources = [folder] + ([args.model] if args.source_kind == "single-file" else [])
    if selection is not None:
        sources.append(Path(selection.source) / selection.weight_name)
    if vae is not None:
        sources.extend([vae.weight.path, vae.config_path])
    if text_encoder is not None:
        sources.extend([text_encoder.weight.path, text_encoder.config_path])
    if tokenizer is not None:
        sources.extend(Path(path) for path, _, _ in tokenizer.files)
    for selected in flux_tokenizers.values():
        sources.extend(Path(path) for path, _, _ in selected.files)
    for selected in flux_encoders.values():
        sources.extend((selected.weight.path, selected.config_path))
    if getattr(args, 'moss_text_encoder_selection', None):
        sources.append(args.moss_text_encoder_selection.path)
        sources.append(args.moss_text_config_selection.path)
    if getattr(args, 'moss_tokenizer_selection', None):
        sources.extend(Path(path) for path, _, _ in args.moss_tokenizer_selection.files)
    if isinstance(args.generic_controlnet, generic_controlnet.Selection):
        if args.generic_controlnet.weight is None:
            sources.append(args.generic_controlnet.directory)
        else:
            sources.extend([args.generic_controlnet.weight.path, args.generic_controlnet.config.path])
    elif isinstance(args.generic_controlnet, (krea2_controlnet.Selection, krea2_color_control.Selection, krea2_ostris_control.Selection, moss_controlnet.Selection, flux2_controlnet.Selection, generic_controlnet.KleinReferenceSelection)):
        sources.append(args.generic_controlnet.weight.path)
    if getattr(args, 'moss_rapid', None):
        sources.extend(weight.path for weight in args.moss_rapid.weights)
    def load():
        pipeline, model = load_pipeline(args, diffusers, dtype)
        if selection is not None and getattr(args, 'flux_t5_fp8_storage', None) == 'cpu-stream':
            flux_t5_stream.validate_lora(selection)
            flux_t5_stream.validate_lora_pipeline(pipeline)
        if selection is not None and getattr(args, 'krea2_fp8_storage', None) == 'cpu-stream':
            with krea2_single_file.adapter_loading(pipeline.transformer, selection, dtype):
                activation = apply_lora(pipeline, selection, args.cache_dir, True)
        else:
            activation = (apply_lora(pipeline, selection, args.cache_dir, True)
                          if selection is not None else None)
        return pipeline, model, SchedulerConfiguration(pipeline) if key is not None else None, activation
    (pipeline, model, scheduler, activation), configuration_hit = cached_pipeline(key, sources, load)
    if configuration_hit:
        scheduler.restore(pipeline)
    # The declaration is a per-request validation, independent of loaded weights.
    model = {**model, "base_model_validation": validate_base_model(args.base_model, type(pipeline), diffusers)}
    def place(previous):
        if previous is not None and previous[1] != "none":
            pipeline.remove_all_hooks()
        return prepare_execution(pipeline, device, args.offload), args.offload
    (pipeline, _), placement_hit = cached_placement((device, args.offload) if key is not None else None, place)
    validate_lora_activation(pipeline, activation)
    args.lora_activation = activation
    args.model_configuration_cache_hit, args.device_placement_cache_hit = configuration_hit, placement_hit
    return pipeline, model, configuration_hit and placement_hit


def output_value(result: Any, key: str) -> Any:
    return result.get(key) if isinstance(result, dict) else getattr(result, key, None)


def as_array(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def as_image(value: Any, image_module: Any, np: Any) -> Any:
    if isinstance(value, image_module.Image):
        return value
    tensor = hasattr(value, "detach")
    array = as_array(value, np)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(f"Generated frame requires three dimensions, got {array.shape}.")
    if tensor:
        if array.shape[0] not in (1, 3, 4):
            raise ValueError(f"Generated Torch images require CHW layout, got {array.shape}.")
        array = array.transpose(1, 2, 0)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported generated frame channel layout: {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.size == 0 or array.min() < 0 or array.max() > 1:
            raise ValueError("Generated floating frames must be finite decoded pixels in [0, 1].")
        array = (array * 255).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError("Generated frames must contain uint8 or decoded floating pixels.")
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    return image_module.fromarray(array)


def atomic_write(path: Path, writer: Any, overwrite: bool) -> None:
    provider.mkdir(path.parent, parents=True, exist_ok=True)
    with provider.staged_path(path, overwrite=overwrite) as temporary:
        writer(temporary)


def infer_sample_rate(pipeline: Any, requested: int | None) -> int:
    if requested is not None:
        return requested
    for component in (getattr(pipeline, "vocoder", None), getattr(pipeline, "vae", None), pipeline):
        config = getattr(component, "config", None)
        for key in ("sampling_rate", "sample_rate"):
            value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError("Audio output requires --audio-sample-rate because the pipeline does not expose its sampling rate.")


def save_outputs(result: Any, pipeline: Any, args: argparse.Namespace, image_module: Any,
                 np: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    tensor_specs = validate_output_specs(getattr(args, "tensor_outputs", {}))
    if getattr(args, "inputs", {}).get("output_type") in ("latent", "latents"):
        for name in ("images", "frames", "audios", "audio", "latents"):
            if output_value(result, name) is not None and tensor_specs.get(name, {}).get("semantic") != "latents":
                raise ValueError(f"Latent output field {name!r} requires an explicit latents tensor contract.")
    for name in ("sequences", "token_ids"):
        value = output_value(result, name)
        if value is not None and name not in tensor_specs:
            tensor_specs[name] = automatic_token_spec(value)
    for name, spec in tensor_specs.items():
        value = output_value(result, name)
        if value is None:
            raise ValueError(f"The pipeline did not return declared tensor output {name!r}.")
        outputs.append(write_tensor_output(value, name, spec, args.output_dir, args.overwrite,
                                           atomic_write, file_identity))

    def save_image(value: Any, filename: str, kind: str) -> None:
        image = as_image(value, image_module, np)
        path = args.output_dir / filename
        atomic_write(path, lambda temporary: provider.save_image(image, temporary, format="PNG"), args.overwrite)
        outputs.append({**file_identity(path), "kind": kind, "size": list(image.size), "mode": image.mode})

    images = None if "images" in tensor_specs else output_value(result, "images")
    frames = None if "frames" in tensor_specs else output_value(result, "frames")
    audio = None if "audios" in tensor_specs else output_value(result, "audios")
    if audio is None:
        audio = None if "audio" in tensor_specs else output_value(result, "audio")
    if images is not None:
        if isinstance(images, image_module.Image):
            images = [images]
        elif hasattr(images, "shape") and len(images.shape) == 3:
            images = [images]
        if len(images) == 0:
            raise ValueError("The pipeline returned an empty images batch.")
        for index, image in enumerate(images):
            save_image(image, f"image-{index + 1:04d}.png", "image")
    if frames is not None:
        if isinstance(frames, list) and frames and isinstance(frames[0], image_module.Image):
            frames = [frames]
        elif hasattr(frames, "shape"):
            tensor = hasattr(frames, "detach")
            frames = as_array(frames, np)
            if frames.ndim == 4:
                frames = frames[None, ...]
            if frames.ndim != 5:
                raise ValueError("Video output requires a batch of frame sequences.")
            layout = args.video_layout
            if layout == "auto":
                layout = "bfchw" if tensor else "bfhwc"
            if layout == "bfchw":
                if frames.shape[2] not in (1, 3, 4):
                    raise ValueError("BFCHW video output has an invalid channel axis; select the actual --video-layout.")
                frames = frames.transpose(0, 1, 3, 4, 2)
            elif layout == "bcfhw":
                if frames.shape[1] not in (1, 3, 4):
                    raise ValueError("BCFHW video output has an invalid channel axis.")
                frames = frames.transpose(0, 2, 3, 4, 1)
            elif frames.shape[-1] not in (1, 3, 4):
                raise ValueError("BFHWC video output has an invalid channel axis; select the actual --video-layout.")
        if len(frames) == 0:
            raise ValueError("The pipeline returned an empty video batch.")
        for batch, sequence in enumerate(frames):
            if len(sequence) == 0:
                raise ValueError("The pipeline returned an empty video frame sequence.")
            for index, frame in enumerate(sequence):
                save_image(frame, f"video-{batch + 1:04d}/frame-{index + 1:06d}.png", "video-frame")
    if audio is not None:
        array = as_array(audio, np)
        if array.ndim == 1:
            array = array[None, ...]
        if array.ndim not in (2, 3) or not np.isfinite(array).all() or array.size == 0:
            raise ValueError("Audio output must be a finite non-empty [batch, samples] or [batch, channels, samples] array.")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Audio output must contain floating amplitudes in [-1, 1], not integer/encoded samples.")
        rate = infer_sample_rate(pipeline, args.audio_sample_rate)
        for index, sample in enumerate(array):
            if sample.ndim == 1:
                sample = sample[:, None]
            elif sample.shape[0] <= 8:
                sample = sample.T
            if sample.shape[-1] > 8:
                raise ValueError("Audio output channel layout is ambiguous or exceeds 8 channels.")
            pcm = (np.clip(sample, -1, 1) * 32767).round().astype("<i2")
            path = args.output_dir / f"audio-{index + 1:04d}.wav"

            def write_wave(temporary: Path) -> None:
                with provider.open(temporary, "wb") as device, wave.open(device, "wb") as stream:
                    stream.setnchannels(pcm.shape[1])
                    stream.setsampwidth(2)
                    stream.setframerate(rate)
                    stream.writeframes(pcm.tobytes())

            atomic_write(path, write_wave, args.overwrite)
            outputs.append({**file_identity(path), "kind": "audio", "sample_rate": rate,
                            "channels": pcm.shape[1], "samples": pcm.shape[0], "encoding": "PCM16"})
    for name in ("text", "texts"):
        if name in tensor_specs:
            continue
        value = output_value(result, name)
        if value is None:
            continue
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple)) or not values or any(not isinstance(item, str) for item in values):
            raise ValueError(f"Text output {name!r} must be a string or a non-empty sequence of strings.")
        for index, item in enumerate(values):
            path = args.output_dir / f"{name}-{index + 1:04d}.txt"
            atomic_write(path, lambda temporary, content=item: provider.write_text(temporary, content, encoding="utf-8"), args.overwrite)
            outputs.append({**file_identity(path), "kind": "text", "field": name, "encoding": "UTF-8"})
    if not outputs:
        raise ValueError("The pipeline returned no supported media or typed generation outputs. Expected images, frames, audios/audio, text/texts, sequences/token_ids, or declared tensor outputs.")
    return outputs


def retire_previous_report(directory: Path) -> Path | None:
    """Keep prior provenance, but never leave it claiming a partially replaced run."""
    previous = directory / "generation.json"
    if not previous.exists():
        return None
    descriptor, name = provider.mkstemp(prefix="generation.previous-", suffix=".json", dir=directory)
    os.close(descriptor)
    archive = Path(name)
    try:
        # The archive name was exclusively reserved by this process. Replace only
        # that empty reservation; no unrelated existing report can be clobbered.
        provider.replace(previous, archive)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive


def safety_metadata(result: Any, pipeline: Any, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "checker_present": getattr(pipeline, "safety_checker", None) is not None,
        "pipeline_report": {},
    }
    missing = object()
    for key in ("nsfw_content_detected", "nsfw_detected", "has_nsfw_concept", "applied_safety_concept"):
        value = result.get(key, missing) if isinstance(result, dict) else getattr(result, key, missing)
        if value is missing:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if key == "nsfw_content_detected" and isinstance(value, (list, tuple)):
            image_count = sum(item["kind"] == "image" for item in outputs)
            if image_count and len(value) != image_count:
                raise ValueError("The pipeline's nsfw_content_detected flags do not match its image batch.")
        # Keep actual values, including null, instead of inferring that an absent
        # checker or absent result means the generated content was checked.
        json.dumps(value, allow_nan=False)
        metadata["pipeline_report"][key] = value
    return metadata


def publish_generation(result: Any, pipeline: Any, args: argparse.Namespace, image_module: Any,
                       np: Any, report: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    previous = retire_previous_report(args.output_dir) if args.overwrite else None
    outputs = save_outputs(result, pipeline, args, image_module, np)
    completed = {**report, "outputs": outputs, "safety": safety_metadata(result, pipeline, outputs)}
    if previous is not None:
        completed["previous_report"] = str(previous)
    report_path = args.output_dir / "generation.json"
    atomic_write(report_path,
                 lambda path: provider.write_text(path, json.dumps(completed, indent=2, ensure_ascii=False,
                                                        allow_nan=False) + "\n"),
                 args.overwrite)
    return report_path, outputs


def main(argv: list[str] | None = None) -> int:
    try:
        args = resolve_arguments(build_parser().parse_args(argv))
        if args.print_config:
            print(json.dumps(configuration(args), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if not is_preparing() and args.output_dir.exists() and not args.overwrite and any(provider.iterdir(args.output_dir)):
            raise ValueError(f"Output directory is not empty: {args.output_dir}; choose a fresh directory or pass --overwrite.")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        from dependency_import_progress import DependencyImportProgress
        import_progress = DependencyImportProgress(report_dependency_progress)
        with import_progress:
            import torch
            import diffusers
            import numpy as np
            from PIL import Image
        device = select_device(torch, args.device)
        dtype = getattr(torch, args.dtype)
        hardware = accelerator_preflight(torch, device, dtype)
        if is_preparing() and set(args.inputs) - {"prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale"}:
            raise ValueError("Foreground preparation does not consume task-specific pipeline inputs.")
        inputs, input_files = ({}, []) if is_preparing() else prepare_inputs(args.inputs, Image)
        krea_control_image = None
        flux2_control_image = None
        flux2_inpaint_image = None
        flux2_mask_image = None
        moss_control_images = None
        region_image = None
        krea_region_selection = getattr(args, 'krea2_regions', None)
        region_selection = getattr(args, 'moss_regions', None) or krea_region_selection
        if region_selection:
            prepared, region_files = prepare_inputs({'region_image': {'image_path': region_selection.map_path, 'mode': 'RGB'}}, Image)
            region_image = prepared['region_image']
            input_files.extend(region_files)
        if isinstance(args.generic_controlnet, (krea2_controlnet.Selection, krea2_ostris_control.Selection)):
            prepared, control_files = prepare_inputs({"control_image": {
                "image_path": args.generic_controlnet.image, "mode": "RGB"}}, Image)
            krea_control_image = prepared["control_image"]
            input_files.extend(control_files)
            if isinstance(args.generic_controlnet, krea2_controlnet.Selection) and args.generic_controlnet.auto_bucket:
                krea2_controlnet.configure_resolution(args, inputs, krea_control_image)
        if isinstance(args.generic_controlnet, flux2_controlnet.Selection):
            descriptors = {'control_image': {'image_path': args.generic_controlnet.image, 'mode': 'RGB'}}
            if args.generic_controlnet.inpaint_image:
                descriptors['control_inpaint_image'] = {'image_path': args.generic_controlnet.inpaint_image, 'mode': 'RGB'}
            if args.generic_controlnet.mask_image:
                descriptors['control_mask_image'] = {'image_path': args.generic_controlnet.mask_image, 'mode': 'L'}
            prepared, control_files = prepare_inputs(descriptors, Image)
            flux2_control_image = prepared['control_image']
            flux2_inpaint_image = prepared.get('control_inpaint_image')
            flux2_mask_image = prepared.get('control_mask_image')
            input_files.extend(control_files)
        if isinstance(args.generic_controlnet, (moss_controlnet.Selection, krea2_color_control.Selection)):
            descriptors = {}
            if args.generic_controlnet.depth:
                descriptors["control_image"] = {"image_path": args.generic_controlnet.depth, "mode": "RGB"}
            if args.generic_controlnet.color:
                descriptors["control_color_image"] = {"image_path": args.generic_controlnet.color, "mode": "RGB"}
            prepared, control_files = prepare_inputs(descriptors, Image)
            moss_control_images = {"depth": prepared.get("control_image"), "color": prepared.get("control_color_image")}
            input_files.extend(control_files)
            if isinstance(args.generic_controlnet, krea2_color_control.Selection) and args.generic_controlnet.auto_bucket:
                krea2_controlnet.configure_resolution(args, inputs, moss_control_images['depth'] or moss_control_images['color'])
        preprocessor = None
        depth_identity = None
        depth_report = None
        processed_identity = None
        if args.generic_controlnet and args.control_preprocess in ("canny", "depth"):
            image_key = args.generic_controlnet.image_argument
            original = (krea_control_image if krea_control_image is not None else
                        flux2_control_image if flux2_control_image is not None else
                        moss_control_images["depth"] if moss_control_images is not None else
                        inputs[image_key][0] if isinstance(args.generic_controlnet, generic_controlnet.KleinReferenceSelection)
                        else inputs[image_key])
            if args.control_preprocess == "canny":
                path = native_control_preprocessing.library_path(args.control_preprocessor_library)
                preprocessor = file_identity(path)
                processed = native_control_preprocessing.apply_canny(
                    original, Image, native_control_preprocessing.load_library(path))
                if file_identity(path) != preprocessor:
                    raise RuntimeError("Native Canny library changed while preprocessing.")
            else:
                depth_identity = model_identity(Path(args.depth_preprocessor.directory))
                is_krea_depth = isinstance(args.generic_controlnet, krea2_controlnet.Selection)
                crop = ((inputs["width"], inputs["height"]) if is_krea_depth else None)
                processed, depth_report = depth_preprocessing.apply(original, args.depth_preprocessor, torch, np, Image, device, crop,
                                                                     preserve_float=is_krea_depth)
                verify_identity(depth_identity)
                depth_report["model"] = depth_identity
            processed_identity = control_hint_identity.fingerprint(processed, np)
            if krea_control_image is not None:
                krea_control_image = processed
            elif flux2_control_image is not None:
                flux2_control_image = processed
            elif moss_control_images is not None:
                moss_control_images["depth"] = processed
            elif isinstance(args.generic_controlnet, generic_controlnet.KleinReferenceSelection):
                inputs[image_key][0] = processed
            else:
                inputs[image_key] = processed
        with import_progress:
            pipeline, model, pipeline_cache_hit = prepared_pipeline(args, diffusers, dtype, device)
        record_execution(device, args.offload, args.model)
        if is_preparing():
            verify_identity(model["identity"])
            verify_pipeline_sources()
            return 0
        reference_resize_trace = None
        reference_resize_context = nullcontext()
        reference_vae_context = nullcontext()
        if (isinstance(args.generic_controlnet, generic_controlnet.KleinReferenceSelection)
                and args.generic_controlnet.resize == 'comfy-1mp'):
            inputs['image'], reference_resize_trace = klein_reference_resize.prepare(inputs['image'], torch, np, Image)
            reference_resize_trace['vae_normalization'] = 'batch-norm-inference'
            reference_resize_context = klein_reference_resize.execution(pipeline)
            reference_vae_context = klein_reference_resize.comfy_vae_execution(pipeline, torch)
        validate_call_arguments(pipeline.__call__, inputs, seed=args.seed)
        if args.seed is not None:
            inputs["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
        if type(pipeline).__name__ == 'MOSSImagePipeline':
            inputs['progress_callback'] = report_moss_progress
        attach_preview(pipeline, inputs, args.preview_dir, torch, inputs.get("width"), inputs.get("height"))
        # Retained offload weights must remain versioned across device changes.
        if krea_control_image is not None:
            module = krea2_ostris_control if isinstance(args.generic_controlnet, krea2_ostris_control.Selection) else krea2_controlnet
            control_context = module.execution(pipeline, krea_control_image, inputs, args.generic_controlnet, torch, np)
        elif moss_control_images is not None:
            control_module = krea2_color_control if isinstance(args.generic_controlnet, krea2_color_control.Selection) else moss_controlnet
            control_context = control_module.execution(pipeline, moss_control_images, inputs,
                                                       args.generic_controlnet, torch, np)
        elif flux2_control_image is not None:
            control_context = flux2_controlnet.execution(pipeline, flux2_control_image, inputs,
                                                          args.generic_controlnet, torch, np,
                                                          inpaint_image=flux2_inpaint_image, mask_image=flux2_mask_image)
        else:
            control_context = nullcontext(None)
        rapid = getattr(args, 'moss_rapid', None)
        rapid_context = moss_rapid.execution(pipeline, rapid, torch) if rapid else nullcontext(None)
        region_module = krea2_regions if krea_region_selection else moss_regions
        if region_selection and not krea_region_selection:
            fit = ('center' if isinstance(args.generic_controlnet, moss_controlnet.Selection)
                   and args.generic_controlnet.map_preprocess == 'training' else 'stretch')
            region_context = region_module.execution(pipeline, inputs, region_selection, region_image, torch, np, map_fit=fit)
        elif region_selection:
            region_context = region_module.execution(pipeline, inputs, region_selection, region_image, torch, np)
        else:
            region_context = nullcontext(None)
        with torch.no_grad(), reference_resize_context, reference_vae_context, generic_controlnet.sd3_control_seed(pipeline, args.generic_controlnet, args.seed, torch), krea2_contract.execution(pipeline, inputs, args.krea2, torch) as krea2_trace, control_context as control_trace, region_context as region_trace, rapid_context as rapid_adapter:
            result = moss_rapid.generate(pipeline, inputs, rapid_adapter, control_trace, torch) if rapid else pipeline(**inputs)
        validate_execution_device(pipeline, device)
        validate_lora_activation(pipeline, args.lora_activation)
        verify_lora_identity(args.lora_selection)
        verify_vae_selection(args.vae_selection)
        krea2_text_encoder.verify_selection(args.text_encoder_selection)
        krea2_tokenizer.verify_selection(args.tokenizer_selection)
        flux_tokenizer.verify_selections(getattr(args, 'flux_tokenizer_selections', {}))
        for selected in getattr(args, 'flux_text_encoders', {}).values():
            flux_text_encoder.verify_selection(selected)
        if getattr(args, 'moss_text_encoder_selection', None):
            from weight_files import verify_weight_file
            verify_weight_file(args.moss_text_encoder_selection, 'MOSS Qwen3.5 text encoder')
            moss_text_encoder.verify_config(args.moss_text_config_selection)
        if getattr(args, 'moss_tokenizer_selection', None):
            krea2_tokenizer.verify_selection(args.moss_tokenizer_selection)
        verify_identity(model["identity"])
        if model.get("controlnet_identity"):
            verify_identity(model["controlnet_identity"])
        if isinstance(args.generic_controlnet, generic_controlnet.Selection):
            generic_controlnet.verify_selection(args.generic_controlnet)
        if isinstance(args.generic_controlnet, (krea2_controlnet.Selection, krea2_color_control.Selection, krea2_ostris_control.Selection, moss_controlnet.Selection, flux2_controlnet.Selection, generic_controlnet.KleinReferenceSelection)):
            from weight_files import verify_weight_file
            verify_weight_file(args.generic_controlnet.weight, "Control adapter")
        if rapid:
            from weight_files import verify_weight_file
            for weight in rapid.weights:
                verify_weight_file(weight, 'MOSS few-step adapter')
        if region_selection:
            moss_regions.verify(region_selection)
        if preprocessor and file_identity(Path(preprocessor["path"])) != preprocessor:
            raise RuntimeError("Native Canny library changed during generation.")
        if depth_identity:
            verify_identity(depth_identity)
        verify_input_files(input_files)
        verify_pipeline_sources()
        versions = {}
        for package in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "sentencepiece", "numpy", "Pillow", "peft"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        preprocessor_report = ({"name": "native-canny", "library": preprocessor,
                                "high": 0.08, "low": 0.08, "weak": 0.8,
                                "strong": 1.0, "inverse": False} if preprocessor else depth_report)
        if preprocessor_report is not None:
            preprocessor_report['output_identity'] = processed_identity
        report = {"schema_version": 1, "request": configuration(args), "model": model,
                  "krea2_execution": krea2_trace,
                  "control_execution": control_trace,
                  "control_reference_resize": reference_resize_trace,
                  "regional_execution": region_trace,
                  "adapters": {"lora": lora_metadata(args.lora_selection, args.lora_activation)},
                  "input_files": input_files,
                  "control_preprocessor": preprocessor_report,
                  "runtime": {"packages": versions, "hardware": hardware, "pipeline_cache_hit": pipeline_cache_hit,
                              "model_configuration_cache_hit": args.model_configuration_cache_hit,
                              "device_placement_cache_hit": args.device_placement_cache_hit,
                              "execution_device": device, "dtype": str(dtype), "offload": args.offload},
                  "validation_scope": "Successful execution of these exact files and inputs; not universal family compatibility, visual quality, or model-license clearance."}
        report_path, outputs = publish_generation(result, pipeline, args, Image, np, report)
        print(json.dumps({"report": str(report_path), "outputs": [item["path"] for item in outputs]}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
        raise SystemExit(f"Generic Diffusers generation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
