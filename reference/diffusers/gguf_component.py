"""Load a local GGUF model component into a built-in Diffusers pipeline."""

from dataclasses import dataclass
import argparse
import inspect
from pathlib import Path

from weight_files import LocalWeightFile, checked_weight_path, file_signature


@dataclass(frozen=True)
class Selection:
    name: str
    model_class: type


def _is_gguf(path: Path) -> bool:
    with path.open('rb') as stream:
        return stream.read(4) == b'GGUF'


def _candidate(path: Path) -> bool:
    if path.suffix.casefold() == '.gguf':
        return True
    try:
        return _is_gguf(path)
    except OSError:
        return False


def _conventional(folder: Path, variant: str | None):
    from directory_checkpoint_conversion import _legacy_names, _variant_name
    safe = 'diffusion_pytorch_model.safetensors'
    names = [safe, safe + '.index.json']
    for name in _legacy_names('diffusion_pytorch_model.bin'):
        names.extend((name, name + '.index.json'))
    for name in names:
        path = folder / _variant_name(name, variant)
        if path.is_file():
            return path
    return None


def _identity(path: Path) -> LocalWeightFile:
    before = file_signature(path)
    if before[3] < 4 or not _is_gguf(path) or file_signature(path) != before:
        raise ValueError(f'GGUF component has invalid or changed content: {path}')
    return LocalWeightFile(str(path), str(path.resolve()), None, before[3], before)


def nearby_config(source: Path) -> Path | None:
    """Recognize a GGUF file inside a pipeline's declared component folder."""
    source = Path(source).expanduser().absolute()
    role = source.parent.name
    if role not in ('transformer', 'unet') or not source.is_file():
        return None
    before = file_signature(source)
    if not _is_gguf(source):
        return None
    folder = source.parent.parent
    index, config = folder / 'model_index.json', source.parent / 'config.json'
    if not index.is_file() or not config.is_file() or index.stat().st_size > 1024 * 1024 or config.stat().st_size > 1024 * 1024:
        return None
    from generation_config import json_object
    try:
        index_before = file_signature(index)
        contents = json_object(index.read_text(encoding='utf-8'))
        declaration = contents.get(role)
        pipeline = contents.get('_class_name')
        if (not isinstance(pipeline, str) or not pipeline.isidentifier() or not pipeline.endswith('Pipeline')
                or not isinstance(declaration, (list, tuple)) or len(declaration) != 2
                or declaration[0] != 'diffusers' or not isinstance(declaration[1], str)
                or not declaration[1].isidentifier()):
            return None
        if file_signature(index) != index_before or file_signature(source) != before:
            raise RuntimeError('GGUF pipeline configuration changed while identifying its source.')
        return folder
    except (OSError, UnicodeError, argparse.ArgumentTypeError):
        return None


def discover(folder: Path, index: dict, variant: str | None, excluded=()) -> dict[str, LocalWeightFile]:
    """Find unambiguous GGUF weights inside declared Diffusers component folders."""
    result = {}
    for name in ('transformer', 'unet'):
        declaration = index.get(name)
        if (name in excluded or not isinstance(declaration, (tuple, list)) or len(declaration) != 2
                or declaration[0] != 'diffusers' or not isinstance(declaration[1], str)):
            continue
        base = Path(folder) / name
        if not base.is_dir():
            continue
        variant_standard = _conventional(base, variant) if variant else None
        if variant_standard is not None:
            if _is_gguf(variant_standard):
                result[name] = _identity(variant_standard)
            continue
        default = _conventional(base, None)
        if not variant and default is not None:
            if _is_gguf(default):
                result[name] = _identity(default)
            continue
        files = []
        for path in base.iterdir():
            if path.is_file() and not path.name.startswith('.'):
                files.append(path)
                if len(files) > 10000:
                    raise ValueError(f'Too many files in GGUF component directory: {base}')
        candidates = sorted(p for p in files if _candidate(p))
        matching = ([p for p in candidates if p.stem.casefold().endswith(('.' + variant.casefold(),
                                                                          '-' + variant.casefold()))]
                    if variant else [])
        if len(matching) > 1:
            raise ValueError(f'Ambiguous GGUF variant for {name}: ' + ', '.join(p.name for p in matching))
        if matching:
            result[name] = _identity(matching[0])
            continue
        if default is not None:
            if _is_gguf(default):
                result[name] = _identity(default)
            continue
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous GGUF weights for {name}: ' + ', '.join(p.name for p in candidates))
        if candidates:
            result[name] = _identity(candidates[0])
    return result


def select(index, pipeline_class, diffusers, folder: Path, component: str | None = None) -> Selection:
    """Select only an unambiguous built-in checkpoint-aware model component."""
    model_base = getattr(diffusers, 'ModelMixin', None)
    original_base = getattr(diffusers, 'FromOriginalModelMixin', None)
    if not isinstance(model_base, type) or not isinstance(original_base, type):
        raise ValueError('The installed Diffusers runtime does not provide GGUF-compatible model loaders.')
    parameters = inspect.signature(pipeline_class.__init__).parameters
    candidates = []
    if component is not None and component not in ('transformer', 'unet'):
        raise ValueError(f'Unsupported GGUF component: {component}')
    for name in ((component,) if component else ('transformer', 'unet')):
        declaration = index.get(name)
        if (name not in parameters or not isinstance(declaration, (tuple, list)) or len(declaration) != 2
                or declaration[0] != 'diffusers' or not isinstance(declaration[1], str)):
            continue
        model_class = getattr(diffusers, declaration[1], None)
        if (isinstance(model_class, type) and issubclass(model_class, model_base)
                and issubclass(model_class, original_base) and callable(getattr(model_class, 'from_single_file', None))):
            candidates.append(Selection(name, model_class))
    if len(candidates) != 1:
        raise ValueError('A GGUF model requires exactly one built-in transformer or UNet component in the local pipeline configuration.')
    selected = candidates[0]
    config = Path(folder) / selected.name / 'config.json'
    if not config.is_file() or config.stat().st_size > 1024 * 1024:
        raise ValueError(f'The GGUF {selected.name} requires a bounded local component config.json.')
    return selected


def load(selection: Selection, weights: LocalWeightFile, folder: Path, cache: Path, dtype, diffusers):
    """Use Diffusers' maintained GGUF loader without untrusted Python code."""
    quantization = getattr(diffusers, 'GGUFQuantizationConfig', None)
    if not callable(quantization):
        raise ValueError('The installed Diffusers runtime does not expose GGUFQuantizationConfig.')
    with checked_weight_path(weights, cache / 'single-file-aliases', selection.name, '.gguf') as path:
        model = selection.model_class.from_single_file(
            str(path), config=str(folder), subfolder=selection.name,
            quantization_config=quantization(compute_dtype=dtype), dtype=dtype,
            low_cpu_mem_usage=True, local_files_only=True)
    if not isinstance(model, selection.model_class):
        raise RuntimeError(f'The GGUF loader returned an incompatible {selection.name} component.')
    from model_loading import require_materialized_component
    require_materialized_component(model, 'GGUF.' + selection.name)
    return model, {'format': 'gguf', 'component': selection.name,
                   'class_name': selection.model_class.__name__, 'validation': 'component-loaded; output-not-verified'}
