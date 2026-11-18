"""Convert explicitly selected local tensor checkpoints without unsafe pickle fallback.

Importing this module does not import Torch or safetensors. Callers provide a
cache directory outside the source model. A conversion checks serialization and
source identity only; it does not prove a checkpoint's model family or quality.
"""

from __future__ import annotations

from _file_provider import provider

from collections import Counter
from collections.abc import Mapping
import errno
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import zipfile
from typing import BinaryIO

from weight_files import file_sha256, model_content_sha256, file_signature


LEGACY_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")
SAFETENSORS_SUFFIXES = (".safetensors", ".safetensor")
CONVERSION_VERSION = 1
MINIMUM_TORCH_VERSION = (2, 10, 0)
CONVERSION_POLICY = {
    "loader": "torch.load",
    "map_location": "cpu",
    "weights_only": True,
    "tensor_layout": "dense-contiguous",
    "tensor_storage": "independent",
    "minimum_torch_version": "2.10.0",
}
_DTYPE_NAMES = (
    "bool", "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64",
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
    "float8_e8m0fnu",
)


def _identity(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Checkpoint must be a nonempty regular file: {path}")
    resolved = path.resolve()
    before = resolved.stat()
    signature = file_signature(resolved)
    digest = model_content_sha256(resolved)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or path.resolve() != resolved:
        raise RuntimeError(f"Checkpoint changed while computing its identity: {path}")
    return {
        "path": str(path),
        "resolved_file": str(resolved),
        "sha256": digest,
        "size_bytes": after.st_size,
        **({"validation": "metadata", "signature": list(signature)} if digest is None else {}),
    }


def _verify_original(original: dict) -> None:
    try:
        matches = _identity(Path(original["path"])) == original
    except (OSError, ValueError):
        matches = False
    if not matches:
        raise RuntimeError(f"Source checkpoint changed after inspection or during conversion: {original['path']}")


def verify_conversion_source(conversion: dict | None) -> None:
    """Ensure both the legacy input and converted execution file remain unchanged."""
    if conversion is not None:
        _verify_original(conversion["original"])
        for shard in conversion.get("original_shards", ()):
            _verify_original(shard)
        try:
            unchanged = _identity(Path(conversion["converted_path"])) == conversion["output"]
        except (OSError, ValueError):
            unchanged = False
        if not unchanged:
            raise RuntimeError(f"Converted checkpoint changed after conversion: {conversion['converted_path']}")


def _stream_sha256(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _can_map_zip(stream: BinaryIO) -> bool:
    """Map only ordinary, uncompressed Torch storage records on POSIX.

    PyTorch's mmap loader reads storage records by their file offsets and does
    not decompress them. Unknown or compressed containers retain the ordinary
    weights-only loader instead of being interpreted through that path.
    """
    if os.name == "nt":
        return False
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                return False
            # Do not assume storage keys are numeric: the pickle can refer to
            # any archive path under data/, including nested or custom names.
            storages = [entry for entry in entries
                        if not entry.is_dir() and re.search(r"(?:^|/)data/.+", entry.filename)]
            return bool(storages) and all(entry.compress_type == zipfile.ZIP_STORED
                                          and not (entry.flag_bits & 1) for entry in storages)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False
    finally:
        stream.seek(0)


def _copy_snapshot(stream: BinaryIO, destination: Path, original: dict) -> None:
    """Copy from the pinned source descriptor to a private, stable mapping file."""
    stream.seek(0)
    size = 0
    digest = hashlib.sha256() if original["sha256"] is not None else None
    with provider.open(destination, "xb") as output:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            output.write(block)
            size += len(block)
            if digest is not None:
                digest.update(block)
    stream.seek(0)
    if size != original["size_bytes"] or (digest is not None and digest.hexdigest() != original["sha256"]):
        raise RuntimeError("Source checkpoint changed while copying the mapping snapshot.")


def ensure_safe_torch_version(torch) -> None:
    """Reject releases with known weights-only unpickler vulnerabilities."""
    version = str(getattr(torch, "__version__", ""))
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9._-]+)?", version)
    if match is None or tuple(int(part) for part in match.groups()) < MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"Legacy checkpoint conversion requires stable Torch >=2.10.0; found {version!r}. "
            "Older weights-only unpicklers are affected by CVE-2026-24747."
        )


def _conversion_policy(quantized_tensors):
    if quantized_tensors == 'reject':
        return dict(CONVERSION_POLICY)
    if quantized_tensors == 'dequantize':
        return {**CONVERSION_POLICY, 'quantized_tensors': 'dequantize-float32-v1'}
    raise ValueError("quantized_tensors must be 'reject' or 'dequantize'.")


def _dequantize_tensor(value, torch, name):
    supported_dtypes = {getattr(torch, key) for key in ('qint8', 'quint8', 'qint32') if hasattr(torch, key)}
    per_tensor = {getattr(torch, key) for key in ('per_tensor_affine', 'per_tensor_symmetric') if hasattr(torch, key)}
    per_channel = {getattr(torch, key) for key in ('per_channel_affine', 'per_channel_symmetric',
                   'per_channel_affine_float_qparams') if hasattr(torch, key)}
    try:
        scheme = value.qscheme()
        if value.dtype not in supported_dtypes or scheme not in per_tensor | per_channel:
            raise ValueError('unsupported dtype or scheme')
        if scheme in per_tensor:
            scale = value.q_scale()
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('invalid scale')
        else:
            scales, zeros = value.q_per_channel_scales(), value.q_per_channel_zero_points()
            axis = value.q_per_channel_axis()
            if (type(axis) is not int or not 0 <= axis < len(value.shape)
                    or scales.numel() != value.shape[axis] or zeros.numel() != value.shape[axis]
                    or not bool(torch.isfinite(scales).all().item()) or not bool((scales > 0).all().item())
                    or not bool(torch.isfinite(zeros).all().item())):
                raise ValueError('invalid per-channel quantization parameters')
        restored = value.detach().cpu().dequantize()
        if (type(restored) is not torch.Tensor or restored.is_quantized or restored.is_meta
                or restored.layout != torch.strided or restored.dtype != torch.float32
                or tuple(restored.shape) != tuple(value.shape)):
            raise ValueError('invalid dequantized tensor')
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"Checkpoint entry {name!r} has unsupported or invalid quantization: {error}") from error
    return restored, str(value.dtype), str(scheme)


def _tensor_state(loaded, torch, quantized_tensors='reject') -> tuple[dict, bool, dict]:
    if not isinstance(loaded, Mapping):
        raise ValueError("Checkpoint must contain a tensor state dictionary, not a serialized model.")
    wrapped = isinstance(loaded.get("state_dict"), Mapping)
    state = loaded["state_dict"] if wrapped else loaded
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict must be a nonempty mapping of names to tensors.")
    accepted_types = [torch.Tensor]
    parameter = getattr(getattr(torch, "nn", None), "Parameter", None)
    if parameter is not None:
        accepted_types.append(parameter)
    dtypes = {getattr(torch, name) for name in _DTYPE_NAMES if hasattr(torch, name)}
    result = {}
    quantized_dtypes, quantized_schemes = Counter(), Counter()
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint tensor names must be nonempty strings.")
        if type(value) not in accepted_types:
            raise ValueError(f"Checkpoint entry {name!r} is not a plain Torch tensor.")
        if value.is_quantized and quantized_tensors == 'dequantize' and not value.is_meta and value.layout == torch.strided:
            value, dtype, scheme = _dequantize_tensor(value, torch, name)
            quantized_dtypes[dtype] += 1
            quantized_schemes[scheme] += 1
        if (value.layout != torch.strided or value.is_quantized or value.is_meta or value.dtype not in dtypes):
            raise ValueError(f"Checkpoint entry {name!r} requires a dense, non-quantized numeric tensor.")
        result[name] = value.detach().cpu().contiguous()
    # safetensors requires non-overlapping inputs. Clone only overlapping views
    # rather than duplicating every weight, while retaining all tied key names.
    spans = []
    for name, value in result.items():
        count = value.numel()
        if count:
            start = value.data_ptr()
            if not start:
                raise ValueError(f"Checkpoint entry {name!r} has no accessible CPU storage.")
            spans.append((start, start + count * value.element_size(), name))
    last_end = 0
    for start, end, name in sorted(spans):
        if start < last_end:
            result[name] = result[name].clone()
        last_end = max(last_end, end)
    return result, wrapped, {'count': sum(quantized_dtypes.values()),
        'dtypes': dict(sorted(quantized_dtypes.items())), 'schemes': dict(sorted(quantized_schemes.items()))}


def _cached_result(directory: Path, original: dict, *, cache_hit: bool, policy: dict) -> dict:
    manifest_path = directory / "conversion.json"
    output_path = directory / "model.safetensors"
    try:
        if directory.is_symlink() or manifest_path.is_symlink() or output_path.is_symlink():
            raise ValueError("Conversion cache entries must not be symlinks.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["schema"] != "iild-checkpoint-conversion"
                or manifest["schema_version"] != CONVERSION_VERSION
                or manifest["conversion_policy"] != policy
                or manifest["original"]["sha256"] != original["sha256"]
                or manifest["original"]["size_bytes"] != original["size_bytes"]
                or manifest["original"].get("signature") != original.get("signature")
                or manifest["output"]["filename"] != output_path.name):
            raise ValueError("Conversion manifest does not match this source or conversion policy.")
        output = _identity(output_path)
        if (output["sha256"] != manifest["output"]["sha256"]
                or output["size_bytes"] != manifest["output"]["size_bytes"]
                or (output["sha256"] is None
                    and output.get("signature", [])[1:] != manifest["output"].get("signature"))):
            raise ValueError("Converted tensor file no longer matches its manifest.")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid checkpoint conversion cache at {directory}: {error}") from error
    _verify_original(original)
    return {
        "converted_path": str(output_path),
        "converted": True,
        "cache_hit": cache_hit,
        "original": original,
        "output": output,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _cached_sharded_result(directory: Path, original: dict, shards: list[dict], *, cache_hit: bool,
                           source_kind: str, policy: dict) -> dict:
    result = _cached_result(directory, original, cache_hit=cache_hit, policy=policy)
    if result['manifest'].get('source_kind') != source_kind or result['manifest'].get('original_shards') != shards:
        raise RuntimeError(f'Invalid sharded checkpoint conversion cache at {directory}')
    for shard in shards:
        _verify_original(shard)
    result['original_shards'] = shards
    return result


def _materialize_sharded_index(source: Path, cache_dir: str | Path, before: tuple, source_kind: str,
                               quantized_tensors: str, policy: dict) -> dict:
    """Convert if necessary and stream validated shards into one checkpoint."""
    from downloaded_model import _json_object, _safetensors
    from image_package import inspect_pytorch_index, inspect_sharded_index
    inspect_index = inspect_pytorch_index if source_kind == 'pytorch-index' else inspect_sharded_index

    if not str(cache_dir).strip():
        raise ValueError('Conversion cache directory must not be empty.')
    index = inspect_index(source)
    paths = [source.parent / item['relative_path'] for item in index['shards']]
    expected = [set() for _ in paths]
    for item in index['tensors']:
        name, shard = item.get('name'), item.get('shard')
        if not isinstance(name, str) or not name or type(shard) is not int or not 0 <= shard < len(paths):
            raise ValueError('Invalid checkpoint index tensor mapping.')
        if name in expected[shard]:
            raise ValueError('Duplicate checkpoint index tensor mapping.')
        expected[shard].add(name)
    original = _identity(source)
    originals = [_identity(path) for path in paths]
    if file_signature(source) != before or inspect_index(source) != index:
        raise RuntimeError('Checkpoint index changed while identifying its shards.')
    for shard in originals:
        _verify_original(shard)
    cache = Path(cache_dir).expanduser().resolve()
    key = hashlib.sha256(json.dumps({'index': original, 'shards': originals}, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
    prefix = 'sharded' if source_kind == 'pytorch-index' else 'safetensors-sharded'
    quantized_tag = '-dequant-f32' if quantized_tensors == 'dequantize' else ''
    directory = cache / f'{prefix}-v{CONVERSION_VERSION}{quantized_tag}-{key}'
    if directory.exists() or directory.is_symlink():
        return _cached_sharded_result(directory, original, originals, cache_hit=True, source_kind=source_kind, policy=policy)

    temporary = None
    try:
        provider.mkdir(cache, parents=True, exist_ok=True)
        temporary = Path(provider.mkdtemp(prefix=f'.{directory.name}-', dir=cache))
        descriptors, converted, offset, dtypes = {}, [], 0, Counter()
        quantized_dtypes, quantized_schemes = Counter(), Counter()
        for path, names in zip(paths, expected):
            part = materialize_safetensors(path, temporary / 'parts', quantized_tensors=quantized_tensors)
            verify_conversion_source(part)
            weight = Path(part['converted_path'])
            shapes, _, _ = _safetensors(weight)
            if set(shapes) != names or set(descriptors).intersection(shapes):
                raise ValueError('Checkpoint index does not exactly cover its shard tensors.')
            with weight.open('rb') as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError('Converted shard has a truncated Safetensors header.')
                length, = struct.unpack('<Q', prefix)
                if not 2 <= length <= 64 * 1024 * 1024 or length > weight.stat().st_size - 8:
                    raise ValueError('Converted shard has an invalid Safetensors header length.')
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError('Converted shard has a truncated Safetensors header.')
                header = _json_object(raw, 'converted shard header')
            verify_conversion_source(part)
            header.pop('__metadata__', None)
            if set(header) != names:
                raise ValueError('Converted shard tensor names changed during inspection.')
            data_size = weight.stat().st_size - 8 - length
            for name, descriptor in header.items():
                start, end = descriptor['data_offsets']
                descriptors[name] = {**descriptor, 'data_offsets': [offset + start, offset + end]}
            if part['manifest'] is None:
                dtypes.update(descriptor['dtype'] for descriptor in header.values())
            else:
                dtypes.update(part['manifest']['tensor_dtypes'])
                quantized = part['manifest'].get('quantized_tensors', {})
                quantized_dtypes.update(quantized.get('dtypes', {}))
                quantized_schemes.update(quantized.get('schemes', {}))
            offset += data_size
            if offset > (1 << 63) - 1:
                raise ValueError('Merged checkpoint exceeds the supported file size.')
            converted.append((part, 8 + length, data_size))
        label = 'pytorch' if source_kind == 'pytorch-index' else 'safetensors'
        header = {'__metadata__': {'format': 'pt', 'conversion': f'iild-{label}-sharded-v1'}, **descriptors}
        encoded = json.dumps(header, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        encoded += b' ' * (-len(encoded) % 8)
        if len(encoded) > 64 * 1024 * 1024:
            raise ValueError('Merged checkpoint metadata exceeds 64 MiB.')
        output = temporary / 'model.safetensors'
        with provider.open(output, 'xb') as target:
            target.write(struct.pack('<Q', len(encoded)));target.write(encoded)
            for part, base, size in converted:
                verify_conversion_source(part)
                with Path(part['converted_path']).open('rb') as stream:
                    stream.seek(base);remaining = size
                    while remaining:
                        block = stream.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError('Converted shard changed while merging.')
                        target.write(block);remaining -= len(block)
                verify_conversion_source(part)
        shapes, _, _ = _safetensors(output)
        if set(shapes) != set(descriptors):
            raise RuntimeError('Merged checkpoint validation failed.')
        # The per-shard conversions are private intermediates, not additional
        # persistent model copies. The merged output is verified independently.
        if (temporary / 'parts').exists():
            shutil.rmtree(temporary / 'parts')
        output_record = {'filename': 'model.safetensors', 'sha256': model_content_sha256(output),
                         'size_bytes': output.stat().st_size}
        if output_record['sha256'] is None:
            output_record['signature'] = list(file_signature(output)[1:])
        manifest = {'schema': 'iild-checkpoint-conversion', 'schema_version': CONVERSION_VERSION,
                    'source_kind': source_kind, 'original': original, 'original_shards': originals,
                    'output': output_record, 'conversion_policy': dict(policy),
                    'tensor_count': len(descriptors), 'tensor_dtypes': dict(sorted(dtypes.items())),
                    'generation_verified': False}
        if quantized_tensors == 'dequantize':
            manifest['quantized_tensors'] = {'count': sum(quantized_dtypes.values()),
                'dtypes': dict(sorted(quantized_dtypes.items())), 'schemes': dict(sorted(quantized_schemes.items()))}
        provider.write_text(temporary / 'conversion.json', json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        _verify_original(original)
        for shard in originals:
            _verify_original(shard)
        try:
            provider.rename(temporary, directory)
        except OSError:
            if not directory.exists():
                raise
            return _cached_sharded_result(directory, original, originals, cache_hit=True, source_kind=source_kind, policy=policy)
        return _cached_sharded_result(directory, original, originals, cache_hit=False, source_kind=source_kind, policy=policy)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def materialize_safetensors(path: str | Path, cache_dir: str | Path, *, quantized_tensors: str = 'reject') -> dict:
    """Return a verified local safetensors path and original-file provenance.

    Shard indexes are merged with bounded memory and verified provenance.
    Legacy files use only ``torch.load(map_location='cpu', weights_only=True)``.
    Eligible ZIPs use a private, memory-mapped snapshot; other inputs use the
    ordinary CPU loader. The independent output is published in an atomic,
    content-addressed directory. Existing valid conversions need no ML imports.
    ``.gguf`` requires a native loader and is never converted by this function.
    """
    policy = _conversion_policy(quantized_tensors)
    if not str(path).strip():
        raise ValueError("Checkpoint path must not be empty.")
    source = Path(path).expanduser().absolute()
    if not source.is_file() or not source.stat().st_size:
        raise ValueError(f"Checkpoint must be a nonempty regular file: {source}")
    if source.suffix.casefold() == ".onnx":
        raise ValueError("Unsupported checkpoint: ONNX is not supported by this conversion path.")
    before = file_signature(source)
    from downloaded_model import model_container_format
    try:
        container = model_container_format(source)
    except ValueError as error:
        raise ValueError(f"Unsupported checkpoint container: {source}") from error
    if container == "gguf":
        raise ValueError("GGUF is not converted to safetensors; select a native GGUF loader/workflow.")
    if container in ('pytorch-index', 'safetensors-index'):
        return _materialize_sharded_index(source, cache_dir, before, container, quantized_tensors, policy)
    if container not in ('pytorch', 'safetensors'):
        raise ValueError('Checkpoint conversion requires a supported PyTorch or Safetensors file or shard index.')
    original = _identity(source)
    if file_signature(source) != before:
        raise RuntimeError(f"Source checkpoint changed while identifying its container: {source}")
    if container == "safetensors":
        return {
            "converted_path": str(source),
            "converted": False,
            "cache_hit": False,
            "original": original,
            "output": dict(original),
            "manifest_path": None,
            "manifest": None,
        }
    if not str(cache_dir).strip():
        raise ValueError("Conversion cache directory must not be empty.")
    cache = Path(cache_dir).expanduser().resolve()
    # Older metadata-only entries did not record output file identity. Keep
    # them untouched but use a separate cache key for the stronger contract.
    cache_key = original["sha256"] or "metadata2-" + hashlib.sha256(json.dumps(original["signature"]).encode()).hexdigest()
    quantized_tag = '-dequant-f32' if quantized_tensors == 'dequantize' else ''
    directory = cache / f"v{CONVERSION_VERSION}{quantized_tag}-{cache_key}"
    if directory.exists() or directory.is_symlink():
        return _cached_result(directory, original, cache_hit=True, policy=policy)
    try:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")
        safetensors = importlib.import_module("safetensors")
    except ImportError as error:
        raise RuntimeError("Legacy checkpoint conversion requires installed Torch and safetensors.") from error
    ensure_safe_torch_version(torch)
    temporary = None
    try:
        # The pinned descriptor supplies either the regular weights-only loader
        # or a private snapshot. Mapping the original directly could SIGBUS if
        # another process truncated it while the weights were being serialized.
        with Path(original["resolved_file"]).open("rb") as stream:
            if original["sha256"] is not None and _stream_sha256(stream) != original["sha256"]:
                raise RuntimeError("Source checkpoint changed before weights-only loading.")
            mapped = _can_map_zip(stream)
            if mapped:
                provider.mkdir(cache, parents=True, exist_ok=True)
                try:
                    # Leave room for both the snapshot and a similarly sized
                    # output. When disk is tight, retain the original loader.
                    mapped = shutil.disk_usage(cache).free >= 2 * original["size_bytes"]
                except OSError:
                    mapped = False
            if mapped:
                temporary = Path(provider.mkdtemp(prefix=f".{directory.name}-", dir=cache))
                snapshot = temporary / "source.pt"
                try:
                    _copy_snapshot(stream, snapshot, original)
                except OSError as error:
                    if error.errno not in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                        raise
                    if snapshot.exists():
                        snapshot.unlink()
                    stream.seek(0)
                    mapped = False
                if mapped:
                    _verify_original(original)
            load_source = str(snapshot) if mapped else stream
            try:
                options = {"map_location": "cpu", "weights_only": True}
                if mapped:
                    options["mmap"] = True
                loaded = torch.load(load_source, **options)
            except Exception as error:
                raise ValueError(
                    "Weights-only checkpoint loading failed; unsafe pickle retry is disabled. "
                    "Use a tensor-only checkpoint or obtain safetensors from its publisher."
                ) from error
            if original["sha256"] is not None and _stream_sha256(stream) != original["sha256"]:
                raise RuntimeError("Source checkpoint changed during weights-only loading.")
        tensors, wrapped, quantized = _tensor_state(loaded, torch, quantized_tensors)
        del loaded
        _verify_original(original)
        if temporary is None:
            provider.mkdir(cache, parents=True, exist_ok=True)
            temporary = Path(provider.mkdtemp(prefix=f".{directory.name}-", dir=cache))
        converted_path = temporary / "model.safetensors"
        provider.encode_at(1, safetensors_torch.save_file, tensors, str(converted_path), metadata={
            **({"original_sha256": original["sha256"]} if original["sha256"] is not None else {"source_validation": "metadata"}),
            "conversion": "iild-weights-only-v1",
        })
        if not converted_path.is_file() or converted_path.stat().st_size == 0:
            raise RuntimeError("Safetensors conversion did not produce a nonempty tensor file.")
        output = {
            "filename": "model.safetensors",
            "sha256": model_content_sha256(converted_path),
            "size_bytes": converted_path.stat().st_size,
        }
        if output["sha256"] is None:
            # The enclosing directory is renamed atomically. Store the stable
            # file identity without its temporary absolute path.
            output["signature"] = list(file_signature(converted_path)[1:])
        tensor_count = len(tensors)
        tensor_dtypes = dict(sorted(Counter(str(tensor.dtype) for tensor in tensors.values()).items()))
        del tensors
        if mapped:
            snapshot.unlink()
        manifest = {
            "schema": "iild-checkpoint-conversion",
            "schema_version": CONVERSION_VERSION,
            "original": original,
            "output": output,
            "conversion_policy": dict(policy),
            "runtime": {"torch": getattr(torch, "__version__", None),
                        "safetensors": getattr(safetensors, "__version__", None)},
            "state_dict_wrapped": wrapped,
            "source_loading": "mmap-private-snapshot" if mapped else "weights-only-cpu",
            "tensor_count": tensor_count,
            "tensor_dtypes": tensor_dtypes,
            "generation_verified": False,
        }
        if quantized_tensors == 'dequantize':
            manifest['quantized_tensors'] = quantized
        provider.write_text(temporary / "conversion.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify_original(original)
        try:
            provider.rename(temporary, directory)
        except OSError:
            # Another conversion of the same content may have won the rename.
            # Reuse it only after checking both its manifest and output hash.
            if not directory.exists():
                raise
            return _cached_result(directory, original, cache_hit=True, policy=policy)
        return _cached_result(directory, original, cache_hit=False, policy=policy)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
