"""Prepare restricted legacy component weights for local Diffusers directories."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil

from _file_provider import provider

_COMPONENT = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')
_WEIGHT_INDEX = re.compile(r'\.index(?:\.[A-Za-z0-9_-]{1,64})?\.json\Z', re.IGNORECASE)
_LEGACY_INDEX = re.compile(r'\.(?:bin|pt|pth|ckpt|pkl|pickle)\.index(?:\.[A-Za-z0-9_-]{1,64})?\.json\Z', re.IGNORECASE)
_LEGACY_SUFFIXES = frozenset({'.bin', '.pt', '.pth', '.ckpt', '.pkl', '.pickle'})


@dataclass(frozen=True)
class PreparedDirectory:
    path: Path
    conversions: list[dict]
    sources: dict | None = None


def _variant_name(name: str, variant: str | None) -> str:
    if variant is None:
        return name
    stem, extension = name.rsplit('.', 1)
    return f'{stem}.{variant}.{extension}'


def component_has_weights(folder: Path) -> bool:
    """Detect local weights, including sharded variant indexes, before fallback."""
    return any(path.is_file() and (path.suffix.casefold() in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt')
                                  or _WEIGHT_INDEX.search(path.name)) for path in Path(folder).glob('*'))


def _legacy_names(name: str) -> tuple[str, ...]:
    return (name, *(name[:-4] + suffix for suffix in ('.pt', '.pth', '.ckpt')))


def _components(folder: Path, index: dict, excluded=()):
    for component, declaration in index.items():
        if (component in excluded or not _COMPONENT.fullmatch(component) or not isinstance(declaration, (list, tuple))
                or len(declaration) != 2):
            continue
        library = declaration[0]
        if index.get('_class_name') == 'MOSSImagePipeline' and component == 'transformer':
            library = 'diffusers'
        if library not in ('diffusers', 'transformers'):
            continue
        base = folder / component
        if not base.is_dir():
            continue
        old = 'diffusion_pytorch_model.bin' if library == 'diffusers' else 'pytorch_model.bin'
        safe = 'diffusion_pytorch_model.safetensors' if library == 'diffusers' else 'model.safetensors'
        yield component, base, old, safe


def automatic_variant(folder: Path, index: dict, excluded=()) -> str | None:
    """Choose a variant only when every variant-only component agrees on one."""
    requirements = []
    for component, base, old, safe in _components(folder, index, excluded):
        names = (*_legacy_names(old), safe)
        if any((base / candidate).is_file() for name in names for candidate in (name, name + '.index.json')):
            continue
        patterns = [re.compile(re.escape(name.rsplit('.', 1)[0]) + r'\.([A-Za-z0-9_-]{1,64})\.'
                               + re.escape(name.rsplit('.', 1)[1]) + r'\Z') for name in names]
        patterns.extend(re.compile(re.escape(name + '.index') + r'\.([A-Za-z0-9_-]{1,64})\.json\Z')
                        for name in names)
        variants = {match.group(1) for path in base.iterdir() if path.is_file()
                    for pattern in patterns if (match := pattern.fullmatch(path.name))
                    and not re.search(r'-\d+-of-\d+\Z', match.group(1))}
        if variants:
            requirements.append((component, variants))
    if not requirements:
        return None
    common = set.intersection(*(variants for _, variants in requirements))
    if len(common) != 1:
        details = ', '.join(f"{name}: {', '.join(sorted(variants))}" for name, variants in requirements)
        raise ValueError(f'Choose an explicit variant; component choices are ambiguous or incompatible: {details}')
    return next(iter(common))


def _plans(folder: Path, index: dict, variant: str | None, excluded=(),
           merge_diffusers_components=()) -> list[tuple[Path, Path]]:
    plans = []
    for component, base, old, safe in _components(folder, index, excluded):
        # Diffusers selects a component-specific variant when present, and can
        # retain its unvariant default for components without that variant.
        variants = (variant, None) if variant else (None,)
        for selected in variants:
            safe_name = _variant_name(safe, selected)
            safe_index = _variant_name(safe + '.index.json', selected)
            existing_safe = next((base / name for name in (safe_name, safe_index) if (base / name).is_file()), None)
            if existing_safe is not None:
                from downloaded_model import model_container_format
                container = model_container_format(existing_safe)
                if container in ('pytorch', 'pytorch-index'):
                    plans.append((existing_safe, Path(component) / safe_name))
                elif (container == 'safetensors-index' and component in merge_diffusers_components
                      and index[component][0] == 'diffusers'):
                    plans.append((existing_safe, Path(component) / safe_name))
                elif container not in ('safetensors', 'safetensors-index'):
                    raise ValueError(f'Unsupported Diffusers component container {container}: {existing_safe}')
                break
            candidates = []
            for name in _legacy_names(old):
                direct, indexed = _variant_name(name, selected), _variant_name(name + '.index.json', selected)
                source = next((base / candidate for candidate in (direct, indexed) if (base / candidate).is_file()), None)
                if source is not None:
                    candidates.append(source)
            if candidates:
                if len(candidates) > 1 and candidates[0].name not in (
                        _variant_name(old, selected), _variant_name(old + '.index.json', selected)):
                    raise ValueError(f'ambiguous component weights for {component}: ' + ', '.join(path.name for path in candidates))
                plans.append((candidates[0], Path(component) / safe_name))
                break
    return plans


def safe_component_variant(folder: Path, component: str, library: str, variant: str | None) -> str | None:
    """Find the selected safe variant for one already prepared component."""
    if variant is None:
        return None
    safe = 'diffusion_pytorch_model.safetensors' if library == 'diffusers' else 'model.safetensors'
    base = Path(folder) / component
    return variant if any((base / _variant_name(name, variant)).is_file()
                          for name in (safe, safe + '.index.json')) else None


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.absolute(), target)
    except OSError:
        shutil.copyfile(source, target)
        from weight_files import file_sha256
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(f'Copied pipeline resource differs from its source: {source}')


def _directory_identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not resolved.is_dir():
        raise ValueError(f'Pipeline source is not a directory: {path}')
    return {'path': str(path), 'resolved': str(resolved),
            'signature': [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]}


def _snapshot(folder: Path, cache: Path | None, output: Path | None) -> dict:
    from weight_files import file_signature
    directories, files = [], []

    def visit(path: Path, relative: Path, ancestors: frozenset[Path]) -> None:
        if len(ancestors) >= 128 or len(directories) + len(files) >= 200000:
            raise ValueError('Pipeline directory exceeds the supported traversal limit.')
        identity = _directory_identity(path)
        resolved = Path(identity['resolved'])
        if resolved in ancestors:
            raise ValueError(f'Cyclic pipeline directory link: {path}')
        if cache is not None and (cache == resolved or resolved in cache.parents):
            raise ValueError('The conversion cache must be outside the source model directory.')
        if output is not None and (output == resolved or resolved in output.parents or output in resolved.parents):
            raise ValueError('The output directory must not overlap the source model directory.')
        directories.append({**identity, 'relative': str(relative)})
        for child in sorted(path.iterdir()):
            if len(directories) + len(files) >= 200000:
                raise ValueError('Pipeline directory exceeds the supported traversal limit.')
            if child.name.startswith('.') or child.name == '__pycache__':
                continue
            child_relative = relative / child.name
            if child.is_dir():
                visit(child, child_relative, ancestors | {resolved})
            elif child.is_file() and child.suffix.casefold() not in ('.py', '.pyc'):
                files.append({'path': str(child), 'relative': str(child_relative),
                              'signature': list(file_signature(child))})
            elif child.is_symlink() and not child.exists():
                raise ValueError(f'Broken pipeline source link: {child}')
        if _directory_identity(path) != identity:
            raise RuntimeError(f'Pipeline directory changed during inspection: {path}')

    visit(folder, Path('.'), frozenset())
    return {'directories': directories, 'files': files}


def verify_source_snapshot(snapshot: dict | None) -> None:
    if not snapshot:
        return
    from weight_files import file_signature
    try:
        for entry in snapshot['directories']:
            if _directory_identity(Path(entry['path'])) != {key: entry[key] for key in ('path', 'resolved', 'signature')}:
                raise RuntimeError(f"Pipeline directory changed after inspection: {entry['path']}")
        for entry in snapshot['files']:
            if file_signature(Path(entry['path'])) != tuple(entry['signature']):
                raise RuntimeError(f"Pipeline source changed after inspection: {entry['path']}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Pipeline directory changed after inspection: {error}') from error


def _legacy(path: Path) -> bool:
    return path.suffix.casefold() in _LEGACY_SUFFIXES or bool(_LEGACY_INDEX.search(path.name))


def _populate(snapshot: dict, destination: Path, *, strip_legacy: bool = False, replaced=()) -> None:
    for entry in snapshot['directories']:
        (destination / entry['relative']).mkdir(parents=True, exist_ok=True)
    for entry in snapshot['files']:
        if Path(entry['relative']) in replaced or (strip_legacy and _legacy(Path(entry['path']))):
            continue
        _link_or_copy(Path(entry['path']), destination / entry['relative'])


def _transformer_aliases(folder: Path, index: dict, variant: str | None, excluded, plans):
    """Bridge the single-file loader's Diffusers-only weight-existence check."""
    from image_package import inspect_sharded_index

    planned = {relative for _, relative in plans}
    aliases, replaced = [], set()
    for component, base, _, safe in _components(folder, index, excluded):
        if index[component][0] != 'transformers':
            continue
        selected = None
        for version in ((variant, None) if variant else (None,)):
            for name in (safe, safe + '.index.json'):
                relative = Path(component) / _variant_name(name, version)
                if relative in planned or (folder / relative).is_file():
                    selected = relative
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        indexed = selected.name.endswith('.json')
        standard = Path(component) / (safe + '.index.json' if indexed else safe)
        if selected != standard:
            # from_single_file does not forward the requested variant to the
            # fallback Transformers from_pretrained call.
            replaced.update((Path(component) / safe, Path(component) / (safe + '.index.json')))
            aliases.append((standard, selected))
        if indexed:
            shard_report = inspect_sharded_index(folder / selected)
            shards = shard_report.get('shards', ())
            if not shards:
                raise ValueError(f'Transformers component has no safe tensor shards: {base}')
            shard = Path(shards[0]['relative_path'])
            if shard.is_absolute() or '..' in shard.parts:
                raise ValueError(f'Invalid Transformers shard path: {shard}')
            marker_source = Path(component) / shard
        else:
            marker_source = selected
        marker = Path(component) / 'diffusion_pytorch_model.safetensors'
        if not (folder / marker).is_file():
            aliases.append((marker, marker_source))
    return aliases, replaced


def _diffusers_aliases(folder: Path, index: dict, variant: str | None, excluded, plans, components):
    """Expose the requested safe variant to single-file fallback loaders."""
    if not variant:
        return [], set()
    planned = {relative for _, relative in plans}
    aliases, replaced = [], set()
    for component, _, _, safe in _components(folder, index, excluded):
        if component not in components or index[component][0] != 'diffusers':
            continue
        selected = Path(component) / _variant_name(safe, variant)
        if selected in planned or (folder / selected).is_file():
            target = Path(component) / safe
            aliases.append((target, selected))
            replaced.update((target, Path(component) / (safe + '.index.json')))
    return aliases, replaced


@contextmanager
def prepare(folder: Path, index: dict, cache_dir: Path, output_dir: Path, variant: str | None, excluded=(),
            *, strip_legacy: bool = False, primary_sources=(), transformer_markers: bool = False,
            merge_diffusers_components=(), quantized_tensors: str = 'reject'):
    """Yield an ephemeral safe-weight overlay while retaining source provenance."""
    if quantized_tensors not in ('reject', 'dequantize'):
        raise ValueError("quantized_tensors must be 'reject' or 'dequantize'.")
    folder = Path(folder).expanduser().absolute()
    plans = _plans(folder, index, variant, excluded, merge_diffusers_components)
    if not plans and not strip_legacy and not transformer_markers and not merge_diffusers_components:
        yield PreparedDirectory(folder, [])
        return
    cache, output = Path(cache_dir).resolve(), Path(output_dir).resolve()
    snapshot = _snapshot(folder, cache if plans else None, output if plans else None)
    verify_source_snapshot(snapshot)
    aliases, alias_replaced = (_transformer_aliases(folder, index, variant, excluded, plans)
                               if transformer_markers else ([], set()))
    extra_aliases, extra_replaced = _diffusers_aliases(folder, index, variant, excluded, plans,
                                                       merge_diffusers_components)
    aliases.extend(extra_aliases)
    alias_replaced.update(extra_replaced)
    if strip_legacy:
        declared = {name for name, value in index.items() if _COMPONENT.fullmatch(name)
                    and isinstance(value, (list, tuple)) and len(value) == 2 and value[0] is not None}
        primary = {Path(path).resolve() for path in primary_sources if path}
        legacy = [entry for entry in snapshot['files'] if _legacy(Path(entry['path']))]
        unknown = [entry['relative'] for entry in legacy
                   if (len(Path(entry['relative']).parts) < 2 or Path(entry['relative']).parts[0] not in declared)
                   and Path(entry['path']).resolve() not in primary]
        if unknown:
            raise ValueError('Single-file configuration contains unrecognized pickle weights: ' + ', '.join(unknown[:5]))
        if not plans and not legacy and not aliases:
            yield PreparedDirectory(folder, [])
            return
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError('The output directory must not overlap the conversion cache.')
    for entry in snapshot['directories']:
        source = Path(entry['resolved'])
        if cache == source or source in cache.parents:
            raise ValueError('The conversion cache must be outside the source model directory.')
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('The output directory must not overlap the source model directory.')
    from checkpoint_conversion import materialize_safetensors, verify_conversion_source
    conversions = []
    for source, _ in plans:
        options = {'quantized_tensors': 'dequantize'} if quantized_tensors == 'dequantize' else {}
        record = materialize_safetensors(source, cache / 'checkpoint-conversions', **options)
        verify_conversion_source(record)
        conversions.append(record)
    verify_source_snapshot(snapshot)
    overlays = cache / 'pipeline-overlays'
    provider.mkdir(overlays, parents=True, exist_ok=True)
    with provider.TemporaryDirectory(prefix='pipeline-', dir=overlays) as temporary:
        destination = Path(temporary)
        replaced = ({source.relative_to(folder) for source, _ in plans}
                    | {relative for _, relative in plans} | alias_replaced)
        _populate(snapshot, destination, strip_legacy=strip_legacy, replaced=replaced)
        for (_, relative), conversion in zip(plans, conversions):
            _link_or_copy(Path(conversion['converted_path']), destination / relative)
        for target, source in aliases:
            _link_or_copy(destination / source, destination / target)
        for conversion in conversions:
            verify_conversion_source(conversion)
        verify_source_snapshot(snapshot)
        yield PreparedDirectory(destination, conversions, snapshot)
        for conversion in conversions:
            verify_conversion_source(conversion)
        verify_source_snapshot(snapshot)
