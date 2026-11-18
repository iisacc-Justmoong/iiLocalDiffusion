"""Local VAE component selection before Diffusers constructs a pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile
from types import SimpleNamespace

from generation_defaults import checked_resource, read_defaults, canonical_lora_family
from moss_vae_conversion import stage_moss_vae_weights
from weight_files import IndexedWeightFile, LocalWeightFile, file_sha256, resolve_weight_file, file_signature, verify_weight_file
from _file_provider import provider

# RGB Qwen Image 1.x pipelines share the published 16-channel latent space.
# Layered requires an RGBA VAE; similarly sized latents from other families are
# not compatible. Do not infer compatibility from a class-name prefix.
QWEN_RGB_PIPELINES = frozenset({
    "QwenImagePipeline", "QwenImageImg2ImgPipeline", "QwenImageInpaintPipeline",
    "QwenImageEditPipeline", "QwenImageEditPlusPipeline", "QwenImageEditInpaintPipeline",
    "QwenImageControlNetPipeline", "QwenImageControlNetInpaintPipeline",
})

VAE_PIPELINES = {
    "qwen-image": QWEN_RGB_PIPELINES | frozenset({"Krea2Pipeline"}),
    "sdxl-base": frozenset({
        "StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLInstructPix2PixPipeline", "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline", "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLAdapterPipeline", "StableDiffusionXLControlNetUnionPipeline",
        "StableDiffusionXLControlNetUnionImg2ImgPipeline", "StableDiffusionXLControlNetUnionInpaintPipeline",
    }),
    "flux1": frozenset({
        "FluxPipeline", "FluxImg2ImgPipeline", "FluxInpaintPipeline", "FluxFillPipeline",
        "FluxKontextPipeline", "FluxKontextInpaintPipeline", "FluxControlPipeline",
        "FluxControlImg2ImgPipeline", "FluxControlInpaintPipeline", "FluxControlNetPipeline",
        "FluxControlNetImg2ImgPipeline", "FluxControlNetInpaintPipeline",
    }),
    "flux2": frozenset({"Flux2Pipeline", "Flux2KleinPipeline", "Flux2KleinKVPipeline", "Flux2KleinInpaintPipeline", "MOSSImagePipeline"}),
}
VAE_CLASSES = {"qwen-image": "AutoencoderKLQwenImage", "sdxl-base": "AutoencoderKL",
               "flux1": "AutoencoderKL", "flux2": "AutoencoderKLFlux2"}
# Shared resources also serve the native/video backend. Validate its known
# entry without routing an unsupported pipeline through an RGB VAE loader.
VAE_RESOURCE_CLASSES = {**VAE_CLASSES, "wan": "AutoencoderKLWan"}


def pipeline_vae_family(name):
    return next((family for family, names in VAE_PIPELINES.items() if name in names), None)


@dataclass(frozen=True)
class VaeSelection:
    directory: str
    class_name: str
    weight: LocalWeightFile
    config_path: str
    config_sha256: str
    source_kind: str = 'directory'
    container_kind: str = 'safetensors'


def single_file_has_vae(path):
    path = Path(path)
    from downloaded_model import model_container_format
    if model_container_format(path) == 'safetensors-index':
        from image_package import inspect_sharded_index
        report = inspect_sharded_index(path)
        return any(item['name'].startswith(('first_stage_model.', 'vae.')) for item in report['tensors'])
    with Path(path).open("rb") as source:
        size = source.read(8)
        if len(size) != 8:
            raise ValueError("Invalid safetensors model header while checking its VAE.")
        length = struct.unpack("<Q", size)[0]
        if length < 2 or length > 16 * 1024 * 1024 or length + 8 > Path(path).stat().st_size:
            raise ValueError("Invalid safetensors model header while checking its VAE.")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors model header while checking its VAE.")
    return any(key.startswith(("first_stage_model.", "vae.")) for key in header)


def _read_vae_config(config_path):
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("VAE configuration is too large.")
    before = file_signature(config_path)
    config = json.loads(config_path.read_text())
    digest = file_sha256(config_path)
    if file_signature(config_path) != before:
        raise RuntimeError("VAE configuration changed while reading.")
    if not isinstance(config, dict):
        raise ValueError('VAE configuration must be a JSON object.')
    return config, digest


def _validate_vae_config(config, expected_class, family):
    if not isinstance(config, dict) or config.get("_class_name") != expected_class:
        raise ValueError(f"VAE must use the pipeline's {expected_class} architecture.")
    if family == "qwen-image" and (config.get("z_dim") != 16 or config.get("input_channels", 3) != 3 or config.get("in_channels", 3) != 3
                     or config.get("out_channels", 3) != 3):
        raise ValueError("Qwen Image RGB VAE requires z_dim=16 and three image channels.")
    if family in ("sdxl-base", "flux1", "flux2"):
        channels = {"sdxl-base": 4, "flux1": 16, "flux2": 32}[family]
        if (config.get("latent_channels") != channels or config.get("in_channels") != 3
                or config.get("out_channels") != 3 or len(config.get("block_out_channels", [])) != 4):
            raise ValueError(f"VAE does not match the {family} RGB latent space ({channels} channels, 8x downsampling).")
        if family in ("sdxl-base", "flux1"):
            scale, shift = (0.13025, 0.0) if family == "sdxl-base" else (0.3611, 0.1159)
            if (not math.isclose(config.get("scaling_factor", 0), scale, rel_tol=1e-6)
                    or not math.isclose(config.get("shift_factor") or 0, shift, abs_tol=1e-6)):
                raise ValueError(f"VAE scaling/shift does not match {family}.")
        elif config.get("patch_size") != [2, 2]:
            raise ValueError("FLUX.2 VAE requires 2x2 latent patches and its batch normalization.")


def _select_directory(directory, expected_class, family, variant=None):
    directory = Path(directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--vae requires a local Diffusers VAE directory with config.json and safetensors weights.")
    config_path = directory / "config.json"
    config, digest = _read_vae_config(config_path)
    _validate_vae_config(config, expected_class, family)
    if variant is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', variant):
        raise ValueError('VAE weight variant must be a simple nonempty name.')
    base = 'diffusion_pytorch_model'
    safe = directory / (base + '.safetensors')
    def choices(suffix, selected):
        stem = f'{base}.{selected}{suffix}' if selected else base + suffix
        names = [stem, stem + '.index.json']
        if selected:
            names.append(base + suffix + '.index.' + selected + '.json')
        return [directory / name for name in names if (directory / name).is_file()]
    def choose(selected):
        preferred = choices('.safetensors', selected)
        if preferred:
            if len(preferred) > 1 and not preferred[0].name.endswith('.safetensors'):
                raise ValueError('Choose a VAE file explicitly; the directory contains ambiguous Safetensors indexes.')
            return preferred[0]
        canonical = choices('.bin', selected)
        if canonical:
            if len(canonical) > 1 and not canonical[0].name.endswith('.bin'):
                raise ValueError('Choose a VAE file explicitly; the directory contains ambiguous PyTorch indexes.')
            return canonical[0]
        alternatives = []
        for suffix in ('.pt', '.pth', '.ckpt'):
            candidates = choices(suffix, selected)
            if len(candidates) > 1 and not candidates[0].name.endswith(suffix):
                raise ValueError('Choose a VAE file explicitly; the directory contains ambiguous PyTorch indexes.')
            if candidates:
                alternatives.append(candidates[0])
        if len(alternatives) > 1:
            raise ValueError('Choose a VAE file explicitly; the directory contains ambiguous weights.')
        return alternatives[0] if alternatives else None
    source = choose(variant) if variant is not None else None
    if source is None:
        source = choose(None)
    if source is None and variant is not None:
        raise ValueError(f'The requested VAE variant {variant!r} and default weights are unavailable.')
    if source is None:
        patterns = (re.compile(r'^diffusion_pytorch_model\.([A-Za-z0-9_-]{1,64})\.(?:safetensors|bin|pt|pth|ckpt)(?:\.index\.json)?$', re.I),
                    re.compile(r'^diffusion_pytorch_model\.(?:safetensors|bin|pt|pth|ckpt)\.index\.([A-Za-z0-9_-]{1,64})\.json$', re.I))
        variants = {match.group(1) for path in directory.iterdir() if path.is_file()
                    for pattern in patterns if (match := pattern.fullmatch(path.name))}
        if len(variants) > 1:
            raise ValueError('Ambiguous VAE weight variants; choose --variant or an explicit file.')
        if variants:
            source = choose(next(iter(variants)))
    if source is None:
        raise ValueError(f'VAE file is missing: {directory}')
    from downloaded_model import model_container_format
    if source == safe and model_container_format(source) == 'safetensors':
        weight = resolve_weight_file(str(source), 'VAE')
        return VaeSelection(str(directory), expected_class, weight, str(config_path), digest)
    return _select_file(source, expected_class, family, directory.parent)


def _checkpoint_weight(path):
    from checkpoint_conversion import _identity
    record = _identity(Path(path))
    signature = record.get('signature')
    return LocalWeightFile(record['path'], record['resolved_file'], record['sha256'], record['size_bytes'],
                           tuple(signature) if signature is not None else None)


def _pytorch_index_weight(source):
    from image_package import inspect_pytorch_index
    before = file_signature(source)
    report = inspect_pytorch_index(source)
    if Path(report['path']) != source or Path(report['resolved_path']) != source.resolve(strict=True):
        raise RuntimeError('The VAE PyTorch shard index changed during inspection.')
    weight = _checkpoint_weight(source)
    shards = []
    for item in report['shards']:
        relative = Path(item['relative_path'])
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ValueError('Invalid VAE PyTorch shard path.')
        shard = _checkpoint_weight(source.parent / relative)
        if shard.resolved_file != item['resolved_path'] or shard.size_bytes != item['size_bytes']:
            raise RuntimeError('A VAE PyTorch shard changed during inspection.')
        shards.append(shard)
    if file_signature(source) != before or inspect_pytorch_index(source) != report:
        raise RuntimeError('The VAE PyTorch shard index changed during inspection.')
    selected = IndexedWeightFile(weight.path, weight.resolved_file, weight.sha256, weight.size_bytes,
                                 weight.signature, tuple(shards))
    verify_weight_file(selected, 'VAE')
    return selected


def _select_file(source, expected_class, family, folder, resources=None):
    source = Path(source).expanduser().absolute()
    from downloaded_model import model_container_format
    container = model_container_format(source)
    if container == 'safetensors':
        weight = resolve_weight_file(str(source), 'VAE')
    elif container == 'safetensors-index':
        from native_image import native_weight
        weight = native_weight(str(source), 'VAE')
    elif container == 'pytorch':
        weight = _checkpoint_weight(source)
    elif container == 'pytorch-index':
        weight = _pytorch_index_weight(source)
    else:
        raise ValueError('A standalone Diffusers VAE requires Safetensors or weights-only PyTorch weights or a shard index.')
    candidates = (source.parent / 'config.json', Path(folder) / 'vae' / 'config.json')
    seen = set()
    for config_path in candidates:
        if config_path in seen or not config_path.is_file() or config_path.resolve() == source.resolve():
            continue
        seen.add(config_path)
        config, digest = _read_vae_config(config_path)
        if config.get('_class_name') != expected_class:
            continue
        _validate_vae_config(config, expected_class, family)
        return VaeSelection(str(config_path.parent), expected_class, weight, str(config_path), digest, 'file', container)
    if not seen and family is not None:
        config_path, digest = _fallback_vae_config(family, resources)
        return VaeSelection(str(config_path.parent), expected_class, weight, str(config_path), digest, 'file', container)
    raise ValueError(f'A standalone VAE requires a local {expected_class} config.json beside the weights or in the pipeline vae directory.')


def resolve_vae_selection(args, index, folder):
    name = args.pipeline_class or index["_class_name"]
    family = pipeline_vae_family(name)
    component = index.get("vae")
    expected_class = VAE_CLASSES[family] if family else (
        component[1] if isinstance(component, list) and len(component) == 2 and component[0] == "diffusers" else None)
    if getattr(args, "vae", None) is not None:
        if expected_class is None or name == "QwenImageLayeredPipeline":
            raise ValueError("This pipeline has no supported explicit VAE override contract.")
        if name == "MOSSImagePipeline" and Path(args.vae).expanduser().is_file():
            from downloaded_model import model_container_format
            if model_container_format(Path(args.vae).expanduser()) == 'safetensors':
                return _select_moss_embedded_vae(args), "explicit"
        if Path(args.vae).expanduser().is_file():
            return _select_file(args.vae, expected_class, family, folder,
                                getattr(args, 'generation_resources', None)), "explicit"
        return _select_directory(args.vae, expected_class, family, getattr(args, 'variant', None)), "explicit"
    if not family:
        return None, "not-configured-for-pipeline"
    # A partial/damaged embedded component must fail in its own loader instead
    # of being silently replaced. Config-only exports genuinely lack weights.
    if args.source_kind == "single-file" and not getattr(args, 'single_file_component_only', False):
        if single_file_has_vae(args.model):
            return None, "model"
    vae_folder = Path(folder) / "vae"
    from directory_checkpoint_conversion import component_has_weights
    if component not in (None, [None, None]) and component_has_weights(vae_folder):
        return None, "model"
    if component not in (None, [None, None], ["diffusers", expected_class]):
        raise ValueError(f"The {family} model declares an incompatible VAE architecture.")
    return select_fallback_vae(family, getattr(args, "generation_resources", None)), "fallback"


def _select_moss_embedded_vae(args):
    """Pair package weights with the verified FLUX.2 config, then load strictly."""
    source = resolve_weight_file(args.vae, "MOSS embedded VAE")
    config = select_fallback_vae("flux2", getattr(args, "generation_resources", None))
    verify_vae_selection(config)
    cache = Path(args.cache_dir).resolve(strict=True)
    if not cache.is_dir():
        raise ValueError("MOSS VAE preparation requires the job cache directory.")
    directory = Path(tempfile.mkdtemp(prefix="moss-vae-", dir=cache))
    try:
        destination = directory / "diffusion_pytorch_model.safetensors"
        converted = stage_moss_vae_weights(source.resolved_file, destination, config.weight.resolved_file)
        verify_weight_file(source, "MOSS embedded VAE")
        shutil.copyfile(config.config_path, directory / "config.json")
        verify_vae_selection(config)
        selected = _select_directory(directory, "AutoencoderKLFlux2", "flux2")
        if not converted and (selected.weight.sha256 != source.sha256 or selected.weight.size_bytes != source.size_bytes):
            raise RuntimeError("The MOSS embedded VAE changed while preparing its configuration.")
        if selected.config_sha256 != config.config_sha256:
            raise RuntimeError("The MOSS VAE configuration changed while preparing the model.")
        return selected
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _fallback_vae_entry(family, directory=None):
    family = canonical_lora_family(family)
    if family not in VAE_CLASSES:
        raise ValueError(f"Unsupported fallback VAE pipeline family: {family}")
    root, manifest = read_defaults(directory)
    entries = manifest.get("fallback_vaes", [])
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("Invalid fallback_vaes registry.")
    entries = ([manifest["fallback_vae"]] if "fallback_vae" in manifest else []) + entries
    registry = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("families"), list) or not item["families"]:
            raise ValueError("Each fallback_vae requires explicit compatible families.")
        for member in item["families"]:
            if not isinstance(member, str):
                raise ValueError("Invalid fallback_vae family.")
            member = canonical_lora_family(member)
            if member not in VAE_RESOURCE_CLASSES or item.get("class_name") != VAE_RESOURCE_CLASSES[member]:
                raise ValueError("Incompatible fallback_vae architecture or family.")
            if member in registry:
                raise ValueError("Duplicate fallback_vae family: " + member)
            registry[member] = item
    if family not in registry:
        raise ValueError(f"Missing fallback_vae for {family}; install the iiLocalDiffusion VAE resources.")
    return root, registry[family]


def _fallback_vae_config(family, directory=None):
    family = canonical_lora_family(family)
    root, item = _fallback_vae_entry(family, directory)
    config = item.get('config')
    if not isinstance(config, dict) or not isinstance(config.get('file'), str):
        raise ValueError('The fallback VAE manifest requires a configuration file.')
    relative = Path(config['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Bundled VAE configuration must use a relative package path.')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Bundled VAE configuration escapes its package.')
    value, digest = _read_vae_config(path)
    if path.stat().st_size != config.get('size') or digest != config.get('sha256'):
        raise ValueError('Bundled VAE configuration differs from its manifest.')
    _validate_vae_config(value, VAE_CLASSES[family], family)
    return path, digest


def select_fallback_vae(family, directory=None):
    family = canonical_lora_family(family)
    root, item = _fallback_vae_entry(family, directory)
    weight = checked_resource(root, item)
    config = item["config"]
    relative = Path(config["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Bundled VAE configuration must use a relative package path.")
    config_path = (root / relative).resolve(strict=True)
    if not config_path.is_relative_to(root):
        raise ValueError("Bundled VAE configuration escapes its package.")
    selection = _select_directory(Path(weight.path).parent, item["class_name"], family)
    if selection.config_path != str(config_path):
        raise ValueError("Bundled VAE weights and configuration must share a directory.")
    if config_path.stat().st_size != config["size"] or selection.config_sha256 != config["sha256"]:
        raise ValueError("Bundled VAE configuration differs from its manifest.")
    return selection


def verify_vae_selection(selection):
    if selection is None:
        return
    config = Path(selection.config_path)
    try:
        before = file_signature(config)
        digest = file_sha256(config)
        config_changed = file_signature(config) != before or digest != selection.config_sha256
    except OSError as error:
        raise RuntimeError("The selected VAE changed while loading or generating.") from error
    if isinstance(selection.weight, IndexedWeightFile) or selection.container_kind == 'pytorch':
        verify_weight_file(selection.weight, 'VAE')
        weight_changed = False
    else:
        try:
            weight_changed = resolve_weight_file(selection.weight.path, "VAE") != selection.weight
        except (OSError, ValueError) as error:
            raise RuntimeError("The selected VAE changed while loading or generating.") from error
    if weight_changed or config_changed:
        raise RuntimeError("The selected VAE changed while loading or generating.")


def resolve_preset_vae(preset, args):
    if args.vae_file is not None:
        return None, "explicit"
    if pipeline_vae_family(preset.pipeline_class) is None:
        return None, "not-configured-for-pipeline"
    source = args.config_selection or args.model_selection
    folder = Path(source.source)
    index_path = folder / "model_index.json"
    # Preset requests already supply a fixed architecture; full model validation
    # remains at the loading boundary, including for print-config requests.
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"_class_name": preset.pipeline_class}
    request = SimpleNamespace(vae=None, pipeline_class=preset.pipeline_class,
        model=args.model_selection.source,
        source_kind="single-file" if args.model_selection.single_file else "directory",
        generation_resources=getattr(args, "generation_resources", None))
    selection, status = resolve_vae_selection(request, index, folder)
    if (request.source_kind == "single-file" and selection is None and status == "model"
            and not single_file_has_vae(request.model)):
        family = pipeline_vae_family(preset.pipeline_class)
        return _select_directory(folder / "vae", VAE_CLASSES[family], family), "model-config"
    return selection, status


@contextmanager
def _vae_loading_directory(selection, cache_dir, runtime):
    if selection.source_kind == 'directory':
        yield selection.directory, None
        return
    if cache_dir is None:
        raise ValueError('A standalone VAE requires a cache directory for temporary loading resources.')
    from directory_checkpoint_conversion import _link_or_copy
    overlays = (Path(cache_dir).expanduser().resolve() / 'vae-overlays').resolve()
    config_dir = Path(selection.config_path).parent.resolve()
    conversions = (Path(cache_dir).expanduser().resolve() / 'vae-checkpoint-conversions').resolve()
    workspaces = (overlays, conversions) if selection.container_kind in ('pytorch', 'pytorch-index') else (overlays,)
    if any(path == config_dir or config_dir in path.parents for path in workspaces):
        raise ValueError(f'The cache for VAE loading must be outside the VAE configuration directory: {config_dir}')
    provider.mkdir(overlays, parents=True, exist_ok=True)
    verify_vae_selection(selection)
    conversion = None
    staged_selection = selection
    if selection.container_kind in ('pytorch', 'pytorch-index'):
        from checkpoint_conversion import materialize_safetensors, verify_conversion_source
        conversion = materialize_safetensors(selection.weight.path, conversions, quantized_tensors='dequantize')
        if not conversion.get('converted'):
            raise RuntimeError('PyTorch VAE conversion did not produce Safetensors weights.')
        verify_conversion_source(conversion)
        verify_vae_selection(selection)
        staged_selection = replace(selection, weight=resolve_weight_file(conversion['converted_path'], 'converted VAE'),
                                   container_kind='safetensors')
    with provider.TemporaryDirectory(prefix='vae-', dir=overlays) as temporary:
        root = Path(temporary)
        _link_or_copy(Path(selection.config_path), root / 'config.json')
        from vae_legacy_conversion import stage_legacy_vae
        if not stage_legacy_vae(staged_selection, root, runtime):
            _link_or_copy(Path(staged_selection.weight.path), root / 'diffusion_pytorch_model.safetensors')
        if conversion is not None:
            verify_conversion_source(conversion)
        verify_vae_selection(selection)
        yield str(root), conversion
        if conversion is not None:
            verify_conversion_source(conversion)
        verify_vae_selection(selection)


def load_selected_vae(selection, diffusers, dtype, cache_dir=None):
    verify_vae_selection(selection)
    component_class = getattr(diffusers, selection.class_name)
    with _vae_loading_directory(selection, cache_dir, diffusers) as (directory, conversion):
        component, info = component_class.from_pretrained(directory, dtype=dtype,
            local_files_only=True, use_safetensors=True, trust_remote_code=False, output_loading_info=True)
    if not isinstance(info, dict) or any(info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"VAE weights do not exactly match their configuration: {info}")
    from model_loading import require_materialized_component
    require_materialized_component(component, "vae")
    verify_vae_selection(selection)
    if conversion is not None and isinstance(conversion.get('manifest'), dict):
        manifest = conversion['manifest']
        component._iild_vae_conversion = {
            'source_format': selection.container_kind,
            'cache_hit': conversion['cache_hit'],
            'manifest_path': conversion['manifest_path'],
            'policy': manifest['conversion_policy'],
            'quantized_tensors': manifest['quantized_tensors'],
        }
    return component


def vae_metadata(selection, status):
    return {"status": status, "selection": asdict(selection) if selection else None}
