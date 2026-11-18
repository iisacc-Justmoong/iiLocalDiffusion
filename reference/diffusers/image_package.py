"""Use the native SDK to inspect and prepare embedded image components."""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes as C
from functools import lru_cache
import json
import os
from pathlib import Path
import struct
import tempfile
import time

from weight_files import file_signature

_current = ContextVar('iild_image_package', default=None)
_progress_type = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_uint64, C.c_uint64, C.c_void_p)
_schema = 'iild-safetensors-package-v1'


class _SparseProbeUnavailable(Exception):
    pass


@lru_cache(maxsize=4)
def _load(path):
    from native_image import load_library
    library = load_library(path)
    try:
        library.iild_image_package_inspect_v1.argtypes = [C.c_char_p]
        library.iild_image_package_inspect_v1.restype = C.c_void_p
        library.iild_image_package_materialize_v1.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p,
                                                               _progress_type, C.c_void_p]
        library.iild_image_package_materialize_v1.restype = C.c_void_p
        library.iild_image_package_json_v1.argtypes = [C.c_void_p]
        library.iild_image_package_json_v1.restype = C.c_char_p
        library.iild_image_package_free_v1.argtypes = [C.c_void_p]
        library.iild_image_package_free_v1.restype = None
    except AttributeError as error:
        raise RuntimeError('The native SDK does not provide image package routing; update iiLocalDiffusion.') from error
    return library


def _library():
    from native_image import library_path
    return _load(str(library_path()))


def _result(library, pointer):
    if not pointer:
        raise RuntimeError('The native image package operation could not allocate a result.')
    try:
        payload = library.iild_image_package_json_v1(pointer)
        result = json.loads(payload.decode('utf-8')) if payload else None
    finally:
        library.iild_image_package_free_v1(pointer)
    if not isinstance(result, dict) or result.get('schema') != 'iild-image-package-bridge-v1':
        raise RuntimeError('The native image package operation returned an invalid report.')
    if result.get('ok') is not True:
        raise ValueError(result.get('error') or 'The image package operation failed.')
    return result


def inspect_if_package(model, selected_model=None):
    """Return a validated native route for SDK packages, otherwise None.

    The bounded header is probed without importing a tensor runtime or reading
    weights. The native inspector owns all component and index semantics.
    """
    if not model:
        return None
    path = Path(model).expanduser().absolute()
    if not path.is_file() or path.suffix.casefold() == '.onnx':
        return None
    from downloaded_model import MAX_HEADER_BYTES, _json_object, model_container_format
    try:
        try:
            container = model_container_format(path)
        except ValueError:
            # Leave unfamiliar containers to their existing backend.
            return None
        if container != 'safetensors':
            return None
        signature = file_signature(path)
        with path.open('rb') as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                return None
            length, = struct.unpack('<Q', prefix)
            if not 2 <= length <= MAX_HEADER_BYTES or length > path.stat().st_size - 8:
                return None
            raw = source.read(length)
        if len(raw) != length or b'"iild_package_schema"' not in raw:
            return None
        header = _json_object(raw, 'image package header')
        metadata = header.get('__metadata__', {})
        if not isinstance(metadata, dict) or metadata.get('iild_package_schema') is None:
            return None
        if metadata['iild_package_schema'] != _schema:
            raise ValueError('Unsupported embedded image package schema.')
        library = _library()
        report = _result(library, library.iild_image_package_inspect_v1(str(path).encode('utf-8')))
        if report.get('packaged') is not True:
            raise ValueError('The image package manifest is missing or invalid.')
        # The SDK validates source paths and shard groups; classify each source
        # against its original names, never against the flattened package.
        # A header signature identifies an architecture, not usable weights.
        groups = report.pop('groups', None)
        if not isinstance(groups, list):
            raise RuntimeError('The native SDK lacks image package component catalog support; update iiLocalDiffusion.')
        from downloaded_model import _tensor_identity
        candidates = []
        component_paths = []
        vae_candidates = {}
        encoder_candidates = {'clip': {}, 't5xxl': {}}
        companions = {'anima': {'llm': [], 'vae': []}, 'moss-image': {'llm': [], 'vae': []}}
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get('path'), str) or not isinstance(group.get('tensors'), list):
                raise RuntimeError('The native SDK returned an invalid image package catalog.')
            user_main = group.get('user_selected_main', False)
            if type(user_main) is not bool:
                raise RuntimeError('The native SDK returned an invalid image package selection.')
            component_paths.append(group['path'])
            mapping = {}
            shapes = {}
            for entry in group['tensors']:
                if (not isinstance(entry, list) or len(entry) != 2
                        or not all(isinstance(value, str) and value for value in entry)):
                    raise RuntimeError('The native SDK returned an invalid image package tensor mapping.')
                original, output = entry
                descriptor = header.get(output)
                if original in mapping or not isinstance(descriptor, dict) or not isinstance(descriptor.get('shape'), list):
                    raise RuntimeError('The native SDK returned an invalid image package tensor mapping.')
                mapping[original] = output
                shapes[original] = tuple(descriptor['shape'])
            keys = set(shapes)
            if any({prefix + 'encoder.conv_in.weight', prefix + 'decoder.conv_out.weight'} <= keys
                   for prefix in ('', 'vae.', 'first_stage_model.')):
                vae_candidates[group['path']] = mapping
            clip_tokens = any(name.endswith(('text_model.embeddings.token_embedding.weight', 'model.token_embedding.weight'))
                              for name in keys)
            clip_positions = any(name.endswith(('text_model.embeddings.position_embedding.weight', 'model.positional_embedding'))
                                 for name in keys)
            if clip_tokens and clip_positions:
                encoder_candidates['clip'][group['path']] = mapping
            if (any(name.endswith('shared.weight') for name in keys)
                    and any('encoder.block.0.layer.0.SelfAttention.' in name for name in keys)):
                encoder_candidates['t5xxl'][group['path']] = mapping
            if ('model.embed_tokens.weight' in keys and 'model.norm.weight' in keys
                    and any(name.startswith('model.layers.') for name in keys)):
                companions['anima']['llm'].append(group['path'])
            if {'encoder.conv1.weight', 'encoder.head.2.weight', 'decoder.conv1.weight', 'decoder.head.2.weight'} <= keys:
                companions['anima']['vae'].append(group['path'])
            if ('model.language_model.embed_tokens.weight' in keys
                    and any(name.startswith('model.language_model.layers.') for name in keys)):
                companions['moss-image']['llm'].append(group['path'])
            if {'encoder.conv_in.weight', 'decoder.conv_out.weight'} <= keys:
                companions['moss-image']['vae'].append(group['path'])
            architecture, role, evidence, task = _tensor_identity(shapes)
            if architecture == 'anima' and not any(
                    prefix + 'llm_adapter.blocks.0.cross_attn.q_proj.weight' in keys
                    and prefix + 'blocks.0.self_attn.q_proj.weight' in keys
                    for prefix in ('model.diffusion_model.net.', 'model.diffusion_model.', 'diffusion_model.net.',
                                   'diffusion_model.', 'transformer.net.', 'transformer.', 'net.', '')):
                # Require both native Anima markers in the same namespace.
                continue
            if architecture and role == 'checkpoint':
                candidates.append((group['path'], architecture, evidence, task, mapping, user_main))
        selection_reason = 'unique-candidate'
        if selected_model is not None:
            if not isinstance(selected_model, str) or not selected_model:
                raise ValueError('--package-model requires an original component or shard-index path.')
            candidates = [candidate for candidate in candidates if candidate[0] == selected_model]
            if not candidates:
                raise ValueError('The requested --package-model is not a recognized image checkpoint in this package.')
            selection_reason = 'explicit-request'
        elif len(candidates) > 1:
            requested = [candidate for candidate in candidates if candidate[-1]]
            if len(requested) == 1:
                candidates = requested
                selection_reason = 'user-assigned-main'
        if len(candidates) > 1:
            choices = ', '.join(json.dumps(candidate[0][:256], ensure_ascii=True) for candidate in candidates[:8])
            raise ValueError(f'Multiple image checkpoints in the package; select an original path with --package-model. '
                             f'First {min(len(candidates), 8)} of {len(candidates)}: {choices}')
        if candidates:
            selected, architecture, evidence, task, mapping, _ = candidates[0]
            native_architecture = report.get('architecture')
            if native_architecture != 'unknown' and report.get('transformer') == selected and native_architecture != architecture:
                raise ValueError('Native and component image package inspections disagree.')
            if native_architecture != 'unknown' and report.get('transformer') != selected:
                # The native specialized route refers to a different model.
                # Do not carry its encoder, VAE or encoding onto this selection.
                report.update(text_encoder='', vae='', encoding='unknown')
            if architecture in companions:
                ambiguous = {}
                for slot, paths in companions[architecture].items():
                    matches = [path for path in paths if path != selected]
                    report['text_encoder' if slot == 'llm' else slot] = matches[0] if len(matches) == 1 else ''
                    if len(matches) > 1:
                        ambiguous[slot] = matches
                report['ambiguous_components'] = ambiguous
            report.update(architecture=architecture, role='checkpoint', transformer=selected,
                          task=task, selection_reason=selection_reason,
                          _tensor_mapping=mapping, _tensor_evidence=evidence)
        report['component_paths'] = sorted(component_paths)
        report['_vae_candidates'] = vae_candidates
        report['_encoder_candidates'] = encoder_candidates
        if file_signature(path) != signature:
            raise RuntimeError('The image package changed while inspecting its components.')
        report.update(source=str(path), signature=signature)
        return report
    except OSError as error:
        raise ValueError(f'Cannot read the selected model package: {error}') from error


def _metadata_api(name, *paths):
    library = _library()
    try:
        operation = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError('The native SDK lacks component metadata validation; update iiLocalDiffusion.') from error
    operation.argtypes = [C.c_char_p] * len(paths)
    operation.restype = C.c_void_p
    return _result(library, operation(*(str(path).encode('utf-8') for path in paths)))


def inspect_sharded_index(index):
    report = _metadata_api('iild_image_shard_index_inspect_v1', index).get('index')
    if not isinstance(report, dict) or report.get('schema') != 'iild-safetensors-index-v1':
        raise RuntimeError('The native SDK returned an invalid Safetensors shard report.')
    return report


def inspect_pytorch_index(index):
    report = _metadata_api('iild_image_pytorch_index_inspect_v1', index).get('index')
    if not isinstance(report, dict) or report.get('schema') != 'iild-pytorch-index-v1':
        raise RuntimeError('The native SDK returned an invalid PyTorch shard report.')
    return report


def inspect_wan_directory(directory):
    """Resolve declared Wan components through the native metadata inspector."""
    report = _metadata_api('iild_video_wan_directory_inspect_v1', directory).get('directory')
    if not isinstance(report, dict) or report.get('schema') != 'iild-wan-directory-v1':
        raise RuntimeError('The native SDK returned an invalid Wan directory report.')
    required = ('directory', 'manifest', 'scheduler', 'model', 'text_encoder', 'video_vae')
    if (any(not isinstance(report.get(key), str) or not report[key] for key in required)
            or not isinstance(report.get('dependencies'), list)
            or any(not isinstance(path, str) or not path for path in report['dependencies'])
            or (report.get('high_noise_model') is not None and not isinstance(report['high_noise_model'], str))
            or (report.get('moe_boundary') is not None and type(report['moe_boundary']) not in (int, float))):
        raise RuntimeError('The native SDK returned invalid Wan directory components.')
    return report


def sharded_index_paths(index):
    paths = _metadata_api('iild_image_shard_index_paths_v1', index).get('paths')
    if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
        raise RuntimeError('The native SDK returned invalid Safetensors shard paths.')
    return paths


def probe_model_container(path):
    """Share the native bounded container probe without loading tensor data."""
    result = _metadata_api('iild_image_container_probe_v1', path)
    value = result.get('format')
    if value not in ('safetensors', 'safetensors-index', 'gguf', 'pytorch', 'pytorch-index', 'unknown'):
        raise RuntimeError('The native SDK returned an invalid model container format.')
    return value


def inspect_native_vae(model):
    return _metadata_api('iild_image_vae_inspect_v1', model)


def validate_native_vae(model, candidate):
    result = _metadata_api('iild_image_vae_validate_v1', model, candidate)
    if type(result.get('valid')) is not bool:
        raise RuntimeError('The native SDK returned an invalid VAE validation result.')
    return result['valid']


def inspect_native_text_encoder(model, slot):
    return _metadata_api('iild_image_text_encoder_inspect_v1', model, slot)


def validate_native_text_encoder(model, slot, candidate):
    result = _metadata_api('iild_image_text_encoder_validate_v1', model, slot, candidate)
    if type(result.get('valid')) is not bool:
        raise RuntimeError('The native SDK returned an invalid text encoder validation result.')
    return result['valid']


def _prepare_sparse_file(file):
    if os.name != 'nt':
        return
    # NTFS requires the sparse flag before extending a file. Refuse an
    # automatic probe when the filesystem cannot provide it: a metadata
    # check must not allocate gigabytes of zero-filled model weights.
    import msvcrt
    from ctypes import wintypes
    control = C.WinDLL('kernel32', use_last_error=True).DeviceIoControl
    control.argtypes = [wintypes.HANDLE, wintypes.DWORD, C.c_void_p, wintypes.DWORD,
                        C.c_void_p, wintypes.DWORD, C.POINTER(wintypes.DWORD), C.c_void_p]
    control.restype = wintypes.BOOL
    returned = wintypes.DWORD()
    if not control(msvcrt.get_osfhandle(file.fileno()), 0x000900C4, None, 0, None, 0, C.byref(returned), None):
        raise _SparseProbeUnavailable('This filesystem cannot create a sparse component metadata probe.')


def _read_probe_header(package):
    from downloaded_model import MAX_HEADER_BYTES, _json_object
    verify_source(package)
    source = Path(package['source'])
    with source.open('rb') as original:
        prefix = original.read(8)
        if len(prefix) != 8:
            raise ValueError('The image package header is truncated.')
        length, = struct.unpack('<Q', prefix)
        size = source.stat().st_size
        if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
            raise ValueError('The image package header is invalid.')
        header = _json_object(original.read(length), 'image package header')
    verify_source(package)
    return length, size - 8 - length, header


@contextmanager
def _component_metadata_probe(package, mapping, cache, source_header=None):
    """Build a temporary sparse, metadata-only view of one original source.

    Only small Comfy quantization descriptors are copied, since the native
    metadata reader needs them. This file is never used for inference.
    """
    from downloaded_model import MAX_HEADER_BYTES
    length, data_size, header = source_header or _read_probe_header(package)
    verify_source(package)
    with Path(package['source']).open('rb') as original:
        descriptors, quantization, offset = {}, [], 0
        for name, output in mapping.items():
            item = header.get(output)
            if not isinstance(item, dict):
                raise ValueError('The component tensor mapping is invalid.')
            bounds = item.get('data_offsets')
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(value) is not int for value in bounds)
                    or not 0 <= bounds[0] <= bounds[1] <= data_size):
                raise ValueError('The component tensor offsets are invalid.')
            count = bounds[1] - bounds[0]
            if offset + count > max(data_size * 4, data_size + 1024):
                raise ValueError('The component metadata exceeds the package bounds.')
            descriptors[name] = {**item, 'data_offsets': [offset, offset + count]}
            if name.endswith('.comfy_quant'):
                if count > 1024 * 1024 or sum(item[2] for item in quantization) + count > 1024 * 1024:
                    raise ValueError('The component quantization metadata exceeds the inspection limit.')
                quantization.append((bounds[0], offset, count))
            offset += count
        raw = json.dumps(descriptors, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        raw += b' ' * (-len(raw) % 8)
        if len(raw) > MAX_HEADER_BYTES or offset > (1 << 63) - len(raw) - 8:
            raise ValueError('The component metadata exceeds the inspection limit.')
        cache.mkdir(parents=True, exist_ok=True)
        fd, probe_path = tempfile.mkstemp(prefix='.iild-component-metadata-', suffix='.safetensors', dir=cache)
        try:
            # Close before the native reader reopens it, including on Windows.
            with os.fdopen(fd, 'w+b') as probe:
                _prepare_sparse_file(probe)
                probe.write(struct.pack('<Q', len(raw)))
                probe.write(raw)
                probe.truncate(8 + len(raw) + offset)
                for source_offset, target_offset, count in quantization:
                    original.seek(8 + length + source_offset)
                    payload = original.read(count)
                    if len(payload) != count:
                        raise ValueError('The image package quantization metadata is truncated.')
                    probe.seek(8 + len(raw) + target_offset)
                    probe.write(payload)
            verify_source(package)
            yield probe_path
        finally:
            os.unlink(probe_path)
    verify_source(package)


def choose_native_vae(package, model, cache, progress=None):
    """Return the uniquely compatible package source, or no automatic choice."""
    candidates = {path: mapping for path, mapping in package.get('_vae_candidates', {}).items()
                  if path != package.get('transformer')}
    if not candidates:
        return None
    state = inspect_native_vae(model).get('state')
    if state in ('embedded', 'unsupported'):
        return None
    if state not in ('missing', 'incompatible'):
        raise RuntimeError('The native SDK returned an invalid VAE inspection state.')
    matches = []
    try:
        source_header = _read_probe_header(package)
        for index, (path, mapping) in enumerate(candidates.items()):
            if progress and progress('Inspecting packaged VAE metadata', index, len(candidates)):
                raise RuntimeError('VAE component inspection was canceled.')
            with _component_metadata_probe(package, mapping, Path(cache), source_header) as probe:
                if validate_native_vae(model, probe):
                    matches.append(path)
        if progress and progress('Inspecting packaged VAE metadata', len(candidates), len(candidates)):
            raise RuntimeError('VAE component inspection was canceled.')
    except _SparseProbeUnavailable:
        return None
    if len(matches) > 1:
        choices = ', '.join(json.dumps(path[:256], ensure_ascii=True) for path in matches[:8])
        raise ValueError(f'Multiple compatible VAE components in the package; choose one with --package-components. '
                         f'First {min(len(matches), 8)} of {len(matches)}: {choices}')
    return matches[0] if matches else None


def choose_native_text_encoders(package, model, cache, slots, progress=None):
    """Select only unambiguous, engine-compatible original encoder sources."""
    candidates = package.get('_encoder_candidates', {})
    pending = []
    for slot in slots:
        source_kind = 'clip' if slot in ('clip_l', 'clip_g') else slot
        options = {path: mapping for path, mapping in candidates.get(source_kind, {}).items()
                   if path != package.get('transformer')}
        if not options:
            continue
        state = inspect_native_text_encoder(model, slot).get('state')
        if state in ('embedded', 'unsupported'):
            continue
        if state not in ('missing', 'incompatible'):
            raise RuntimeError('The native SDK returned an invalid text encoder inspection state.')
        pending.append((slot, options))
    if not pending:
        return {}
    selected = {}
    try:
        source_header = _read_probe_header(package)
        total = sum(len(options) for _, options in pending)
        count = 0
        for slot, options in pending:
            matches = []
            for path, mapping in options.items():
                if progress and progress('Inspecting packaged text encoder metadata', count, total):
                    raise RuntimeError('Text encoder component inspection was canceled.')
                with _component_metadata_probe(package, mapping, Path(cache), source_header) as probe:
                    if validate_native_text_encoder(model, slot, probe):
                        matches.append(path)
                count += 1
            if len(matches) > 1:
                choices = ', '.join(json.dumps(path[:256], ensure_ascii=True) for path in matches[:8])
                raise ValueError(f'Multiple compatible {slot} components in the package; choose one with --package-components. '
                                 f'First {min(len(matches), 8)} of {len(matches)}: {choices}')
            if matches:
                selected[slot] = matches[0]
        if progress and progress('Inspecting packaged text encoder metadata', total, total):
            raise RuntimeError('Text encoder component inspection was canceled.')
    except _SparseProbeUnavailable:
        return {}
    return selected


def materialize(package, cache, component, progress=None):
    """Verify and reconstruct one original component or indexed shard group."""
    verify_source(package)
    library = _library()
    errors = []
    def report(detail, done, total, _):
        if progress is not None:
            try:
                return int(bool(progress(detail.decode('utf-8', errors='replace') if detail else '', done, total)))
            except BaseException as error:
                errors.append(error)
                return 1
        return 0
    callback = _progress_type(report)
    pointer = library.iild_image_package_materialize_v1(package['source'].encode('utf-8'),
        str(Path(cache).expanduser().absolute()).encode('utf-8'), component.encode('utf-8'), callback, None)
    try:
        result = _result(library, pointer)
    except (ValueError, RuntimeError):
        if errors:
            raise RuntimeError('Image package preparation was interrupted by its progress callback.') from errors[0]
        raise
    verify_source(package)
    path = result.get('path')
    if not isinstance(path, str) or not Path(path).is_file():
        raise RuntimeError('The image package returned an invalid prepared component.')
    return path


def progress_reporter():
    """Emit the worker's existing bounded model-preparation progress format."""
    previous = [0.0]
    def report(detail, done, total):
        now = time.monotonic()
        if not previous[0] or done == total or now - previous[0] >= 0.2:
            previous[0] = now
            print('IILD_NATIVE_PROGRESS ' + json.dumps({'schema': 'iild-native-progress-v1',
                  'stage': 'preparing-model', 'step': done, 'total': total, 'detail': detail}), flush=True)
        return False
    return report


def verify_source(package=None):
    package = _current.get() if package is None else package
    if package and file_signature(Path(package['source'])) != tuple(package['signature']):
        raise RuntimeError('The image package changed while preparing or generating.')


def current():
    return _current.get()


@contextmanager
def use(package):
    token = _current.set(package)
    try:
        yield
    finally:
        _current.reset(token)
