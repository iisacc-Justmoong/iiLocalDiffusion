#!/usr/bin/env python3
"""Conversion policy, provenance and atomic cache tests without ML dependencies."""

from __future__ import annotations
import resource_fixture

import builtins
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import zipfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import checkpoint_conversion as conversion
import downloaded_model


class Tensor:
    def __init__(self, values=(1, 2), dtype="float32", layout="strided", *, quantized=False, meta=False,
                 quant_scheme='per_tensor_affine', quant_scale=0.25):
        self.values = values
        self.shape = (len(values),)
        self.dtype = dtype
        self.layout = layout
        self.is_quantized = quantized
        self.is_meta = meta
        self.quant_scheme = quant_scheme
        self.quant_scale = quant_scale
        self.operations = []

    def detach(self):
        self.operations.append("detach")
        return self

    def cpu(self):
        self.operations.append("cpu")
        return self

    def contiguous(self):
        self.operations.append("contiguous")
        return self

    def clone(self):
        self.operations.append("clone")
        return Tensor(self.values, self.dtype)

    def qscheme(self):
        return self.quant_scheme

    def q_scale(self):
        return self.quant_scale

    def dequantize(self):
        self.operations.append('dequantize')
        return Tensor(self.values, dtype='float32')

    def data_ptr(self):
        return id(self) * 1024

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 4


class Parameter(Tensor):
    pass


class CheckpointConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = resource_fixture.TemporaryDirectory(prefix="checkpoint-conversion-tests-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = self.directory / "cache"
        self.tensor = Tensor()
        self.torch = SimpleNamespace(
            Tensor=Tensor, nn=SimpleNamespace(Parameter=Parameter), strided="strided",
            __version__="2.13.0", load=Mock(return_value={"weight": self.tensor}),
            qint8='qint8', quint8='quint8', qint32='qint32',
            per_tensor_affine='per_tensor_affine', per_channel_affine='per_channel_affine',
            **{name: name for name in conversion._DTYPE_NAMES},
        )
        self.save_file = Mock(side_effect=self.save)
        self.safetensors = SimpleNamespace(__version__="test-safetensors")
        self.importer = Mock(side_effect=lambda name: {
            "torch": self.torch,
            "safetensors.torch": SimpleNamespace(save_file=self.save_file),
            "safetensors": self.safetensors,
        }[name])
        imported = patch.object(conversion.importlib, "import_module", self.importer)
        imported.start()
        self.addCleanup(imported.stop)

    def test_safe_shards_are_streamed_and_verified_without_torch(self):
        def write_shard(path, name, payload):
            header = json.dumps({name: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}},
                                separators=(',', ':')).encode()
            header += b' ' * (-len(header) % 8)
            path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
        first, second = self.directory / 'a.safetensors', self.directory / 'b.safetensors'
        write_shard(first, 'first', b'1234')
        write_shard(second, 'second', b'5678')
        index = self.directory / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'first': first.name, 'second': second.name}}))
        original = {path: path.read_bytes() for path in (first, second, index)}
        result = conversion.materialize_safetensors(index, self.cache)
        self.assertEqual(result['manifest']['source_kind'], 'safetensors-index')
        self.assertEqual(len(result['original_shards']), 2)
        self.assertFalse(result['cache_hit'])
        self.assertEqual(set(downloaded_model._safetensors(Path(result['converted_path']))[0]), {'first', 'second'})
        self.assertTrue(conversion.materialize_safetensors(index, self.cache)['cache_hit'])
        self.importer.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        first.write_bytes(first.read_bytes()[:-1])
        with self.assertRaises(RuntimeError):
            conversion.verify_conversion_source(result)

    def save(self, tensors, filename, metadata):
        Path(filename).write_text(json.dumps({
            "tensors": {name: {"values": list(value.values), "dtype": value.dtype}
                        for name, value in tensors.items()},
            "metadata": metadata,
        }, sort_keys=True), encoding="utf-8")

    def checkpoint(self, name="model.ckpt", data=b"mock checkpoint source"):
        path = self.directory / name
        path.write_bytes(data)
        return path

    def test_module_import_and_safetensors_passthrough_do_not_import_ml_runtimes(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "safetensors"):
                raise AssertionError("Implicit ML dependency import")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded):
            importlib.reload(conversion)
        for suffix in (".safetensors", ".safetensor"):
            source = self.checkpoint("existing" + suffix)
            result = conversion.materialize_safetensors(source, self.cache)
            self.assertFalse(result["converted"])
            self.assertEqual(result["converted_path"], str(source))
            self.assertEqual(result["original"], result["output"])
            self.assertIsNone(result["manifest_path"])
        self.importer.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_metadata_mode_converts_and_reuses_without_checksum_passes(self):
        source = self.checkpoint("model.ckpt", b"tensor fixture")
        with patch.dict(os.environ, {"IILD_MODEL_VALIDATION": "metadata"}), \
                patch.object(conversion, "_stream_sha256", side_effect=AssertionError("Full source hash")), \
                patch.object(conversion, "file_sha256", side_effect=AssertionError("Full file hash")):
            first = conversion.materialize_safetensors(source, self.cache)
            second = conversion.materialize_safetensors(source, self.cache)
            self.assertIsNone(first["original"]["sha256"])
            self.assertIsNone(first["output"]["sha256"])
            self.assertTrue(second["cache_hit"])
            self.torch.load.assert_called_once()
            self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])
            source.write_bytes(b"changed tensor fixture")
            third = conversion.materialize_safetensors(source, self.cache)
            self.assertNotEqual(first["converted_path"], third["converted_path"])

    def test_metadata_cache_rejects_same_size_output_replacement(self):
        source = self.checkpoint()
        with patch.dict(os.environ, {"IILD_MODEL_VALIDATION": "metadata"}):
            result = conversion.materialize_safetensors(source, self.cache)
            output = Path(result['converted_path'])
            original_size = output.stat().st_size
            output.write_bytes(b'X' * original_size)
            with self.assertRaisesRegex(RuntimeError, 'no longer matches its manifest'):
                conversion.materialize_safetensors(source, self.cache)
            self.assertEqual(output.read_bytes(), b'X' * original_size)

    def test_older_metadata_cache_is_preserved_without_reuse(self):
        source = self.checkpoint()
        with patch.dict(os.environ, {"IILD_MODEL_VALIDATION": "metadata"}):
            identity = conversion._identity(source)
            digest = hashlib.sha256(json.dumps(identity['signature']).encode()).hexdigest()
            old = self.cache / f'v1-metadata-{digest}'
            old.mkdir(parents=True)
            (old / 'sentinel').write_text('keep')
            result = conversion.materialize_safetensors(source, self.cache)
            self.assertNotEqual(Path(result['converted_path']).parent, old)
            self.assertEqual((old / 'sentinel').read_text(), 'keep')
            self.assertTrue(conversion.materialize_safetensors(source, self.cache)['cache_hit'])

    def test_all_legacy_suffixes_load_cpu_weights_only_and_keep_source_unchanged(self):
        for suffix in conversion.LEGACY_SUFFIXES:
            with self.subTest(suffix=suffix):
                source = self.checkpoint("model" + suffix, suffix.encode())
                before = source.read_bytes()
                result = conversion.materialize_safetensors(source, self.cache)
                self.assertTrue(result["converted"])
                self.assertFalse(result["cache_hit"])
                self.assertEqual(source.read_bytes(), before)
                args, kwargs = self.torch.load.call_args
                self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertTrue(args[0].closed)
                self.assertEqual(result["original"]["sha256"], conversion.file_sha256(source))
                self.assertEqual(result["output"]["sha256"], conversion.file_sha256(Path(result["converted_path"])))
                manifest = json.loads(Path(result["manifest_path"]).read_text())
                self.assertEqual(manifest, result["manifest"])
                self.assertEqual(manifest["conversion_policy"]["weights_only"], True)
                self.assertFalse(manifest["generation_verified"])
                self.assertEqual(manifest["tensor_count"], 1)
                self.assertEqual(manifest["runtime"], {"torch": "2.13.0", "safetensors": "test-safetensors"})
        self.assertEqual(self.torch.load.call_count, len(conversion.LEGACY_SUFFIXES))

    def test_wrapped_state_dict_ignores_training_metadata_and_preserves_tied_keys(self):
        shared = Tensor(values=(9, 7), dtype="float16")
        self.torch.load.return_value = {"epoch": 24, "state_dict": {
            "weight": shared, "tied.weight": shared, "bias": Parameter(dtype="int64"),
            "mask": Tensor(dtype="bool"),
        }}
        result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        saved = self.save_file.call_args.args[0]
        self.assertEqual(set(saved), {"weight", "tied.weight", "bias", "mask"})
        self.assertIsNot(saved["weight"], saved["tied.weight"])
        self.assertEqual(saved["weight"].values, saved["tied.weight"].values)
        self.assertEqual(shared.operations, ["detach", "cpu", "contiguous"] * 2 + ["clone"])
        self.assertTrue(result["manifest"]["state_dict_wrapped"])
        self.assertEqual(result["manifest"]["tensor_dtypes"], {"bool": 1, "float16": 2, "int64": 1})

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_zip_is_loaded_from_private_mapped_snapshot_without_copying_disjoint_tensors(self):
        source = self.directory / 'model.safetensors'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'fake tensor')
        before = source.read_bytes()
        snapshot_paths = []
        def load(snapshot, **kwargs):
            self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': True, 'mmap': True})
            self.assertEqual(Path(snapshot).read_bytes(), before)
            self.assertNotEqual(Path(snapshot), source)
            snapshot_paths.append(Path(snapshot))
            return {'weight': self.tensor}
        self.torch.load.side_effect = load
        result = conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(result['manifest']['source_loading'], 'mmap-private-snapshot')
        self.assertEqual(self.tensor.operations, ['detach', 'cpu', 'contiguous'])
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(len(snapshot_paths), 1)
        self.assertFalse(snapshot_paths[0].exists())

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_zip_with_insufficient_snapshot_space_keeps_regular_loader_available(self):
        source = self.directory / 'model.ckpt'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'fake tensor')
        with patch.object(conversion.shutil, 'disk_usage', return_value=SimpleNamespace(free=source.stat().st_size)):
            result = conversion.materialize_safetensors(source, self.cache)
        args, kwargs = self.torch.load.call_args
        self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': True})
        self.assertTrue(args[0].closed)
        self.assertEqual(result['manifest']['source_loading'], 'weights-only-cpu')
        self.assertEqual(list(Path(result['manifest_path']).parent.glob('source.pt')), [])

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_snapshot_disk_full_falls_back_without_retaining_partial_snapshot(self):
        source = self.directory / 'model.ckpt'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'fake tensor')
        def fill_disk(stream, snapshot, original):
            snapshot.write_bytes(b'partial')
            raise OSError(errno.ENOSPC, 'disk full')
        with patch.object(conversion, '_copy_snapshot', side_effect=fill_disk):
            result = conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(result['manifest']['source_loading'], 'weights-only-cpu')
        self.assertNotIn('mmap', self.torch.load.call_args.kwargs)
        self.assertFalse((Path(result['manifest_path']).parent / 'source.pt').exists())

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_zip_with_any_compressed_storage_does_not_use_memory_mapping(self):
        source = self.directory / 'mixed.ckpt'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'uncompressed', compress_type=zipfile.ZIP_STORED)
            archive.writestr('model/data/custom-name', b'compressed', compress_type=zipfile.ZIP_DEFLATED)
        result = conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(result['manifest']['source_loading'], 'weights-only-cpu')
        self.assertNotIn('mmap', self.torch.load.call_args.kwargs)

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_mapped_snapshot_failure_preserves_source_and_cleans_partial_cache(self):
        source = self.directory / 'model.ckpt'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'fake tensor')
        before = source.read_bytes()
        with patch.object(conversion, '_copy_snapshot', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                conversion.materialize_safetensors(source, self.cache)
        self.torch.load.assert_not_called()
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(list(conversion.provider.iterdir(self.cache)), [])

    @unittest.skipIf(os.name == 'nt', 'Memory mapping is currently enabled on POSIX')
    def test_changed_source_after_snapshot_is_rejected_before_loading(self):
        source = self.directory / 'model.ckpt'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('model/data.pkl', b'fake pickle')
            archive.writestr('model/data/0', b'fake tensor')
        copy_snapshot = conversion._copy_snapshot
        def change_source(stream, destination, original):
            copy_snapshot(stream, destination, original)
            source.write_bytes(b'changed checkpoint')
        with patch.object(conversion, '_copy_snapshot', side_effect=change_source):
            with self.assertRaisesRegex(RuntimeError, 'Source checkpoint changed'):
                conversion.materialize_safetensors(source, self.cache)
        self.torch.load.assert_not_called()
        self.assertEqual(list(conversion.provider.iterdir(self.cache)), [])

    def test_content_addressed_cache_reuse_checks_hashes_without_reimporting_torch(self):
        source = self.checkpoint()
        first = conversion.materialize_safetensors(source, self.cache)
        same_content = self.checkpoint("same-content.pt", source.read_bytes())
        self.importer.reset_mock()
        second = conversion.materialize_safetensors(same_content, self.cache)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["converted_path"], second["converted_path"])
        self.assertEqual(second["original"]["path"], str(same_content))
        self.assertEqual(second["manifest"]["original"]["path"], str(source))
        self.importer.assert_not_called()
        self.torch.load.assert_called_once()

    def test_gguf_and_unrecognized_suffixes_reject_before_loading(self):
        for name, message in (("model.gguf", "native GGUF loader"), ("model.pkl", "Unsupported checkpoint")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                conversion.materialize_safetensors(self.checkpoint(name), self.cache)
        self.importer.assert_not_called()

    def test_conversion_uses_container_content_when_extension_is_misleading(self):
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        safe = self.checkpoint('safe.ckpt', struct.pack('<Q', len(header)) + header + bytes(4))
        result = conversion.materialize_safetensors(safe, self.cache)
        self.assertFalse(result['converted'])
        self.assertEqual(result['converted_path'], str(safe))
        self.importer.assert_not_called()
        for name in ('legacy.safetensors', 'legacy.data'):
            with self.subTest(name=name):
                source = self.checkpoint(name, b'PK\x03\x04' + bytes(16))
                result = conversion.materialize_safetensors(source, self.cache)
                self.assertTrue(result['converted'])
                self.assertTrue(self.torch.load.call_args.kwargs['weights_only'])
        for name, data in (('quantized.safetensors', b'GGUF' + bytes(16)),
                           ('renamed.onnx', struct.pack('<Q', len(header)) + header + bytes(4))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                conversion.materialize_safetensors(self.checkpoint(name, data), self.cache)

    def test_source_cannot_change_during_container_detection(self):
        source = self.checkpoint('changed.safetensors', b'first fixture')
        def change(_source):
            source.write_bytes(b'other fixture')
            return 'safetensors'
        with patch.object(downloaded_model, 'model_container_format', side_effect=change):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                conversion.materialize_safetensors(source, self.cache)

    def test_conversion_reverification_detects_changed_converted_file(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        conversion.verify_conversion_source(result)
        Path(result['converted_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Converted checkpoint changed'):
            conversion.verify_conversion_source(result)

    def test_missing_empty_and_directory_sources_are_rejected(self):
        directory = self.directory / "directory.ckpt"
        directory.mkdir()
        for source in ("", self.directory / "missing.ckpt", self.checkpoint("empty.pt", b""), directory):
            with self.subTest(source=source), self.assertRaises(ValueError):
                conversion.materialize_safetensors(source, self.cache)
        self.importer.assert_not_called()

    def test_weights_only_failures_never_retry_unsafe_loading(self):
        for error in (RuntimeError("unpickling global is not allowed"), TypeError("weights_only unsupported")):
            with self.subTest(error=error):
                self.torch.load.reset_mock()
                self.torch.load.side_effect = error
                with self.assertRaisesRegex(ValueError, "unsafe pickle retry is disabled"):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
                self.torch.load.assert_called_once()
                self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])
        self.save_file.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_missing_runtime_dependencies_fail_without_cache_publication(self):
        self.importer.side_effect = ModuleNotFoundError("torch missing")
        with self.assertRaisesRegex(RuntimeError, "requires installed Torch and safetensors"):
            conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertFalse(self.cache.exists())

    def test_known_vulnerable_unrecognized_and_prerelease_torch_versions_reject_before_load(self):
        for version in ("2.5.1", "2.9.1", "2.10.0rc1", "unknown", None):
            with self.subTest(version=version):
                self.torch.__version__ = version
                with self.assertRaisesRegex(RuntimeError, "requires stable Torch >=2.10.0"):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.torch.load.assert_not_called()
        self.assertFalse(self.cache.exists())
        for version in ("2.10.0", "2.10.1+cu128", "2.13.0", "2.10.0.post1"):
            with self.subTest(version=version):
                self.torch.__version__ = version
                conversion.ensure_safe_torch_version(self.torch)

    def test_a_flat_tensor_named_state_dict_is_preserved(self):
        self.torch.load.return_value = {"state_dict": Tensor()}
        result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertFalse(result["manifest"]["state_dict_wrapped"])
        self.assertEqual(set(self.save_file.call_args.args[0]), {"state_dict"})

    def test_invalid_state_dictionary_entries_are_rejected(self):
        class TensorSubclass(Tensor):
            pass
        invalid = [None, [], {}, {"state_dict": {}}, {"state_dict": []},
                   {1: Tensor()}, {"": Tensor()}, {"weight": 1}, {"weight": object()},
                   {"weight": Tensor(layout="sparse_coo")}, {"weight": Tensor(quantized=True)},
                   {"weight": Tensor(meta=True)}, {"weight": Tensor(dtype="complex64")},
                   {"weight": TensorSubclass()}]
        for state in invalid:
            with self.subTest(state=state):
                self.torch.load.return_value = state
                with self.assertRaises(ValueError):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.save_file.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_quantized_tensor_conversion_is_opt_in_and_has_separate_cache(self):
        source = self.checkpoint()
        quantized = Tensor(dtype='qint8', quantized=True)
        self.torch.load.return_value = {'weight': quantized, 'bias': Tensor(dtype='float16')}
        with self.assertRaisesRegex(ValueError, 'quantized'):
            conversion.materialize_safetensors(source, self.cache)
        result = conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')
        self.assertEqual(self.save_file.call_args.args[0]['weight'].dtype, 'float32')
        self.assertEqual(self.save_file.call_args.args[0]['bias'].dtype, 'float16')
        self.assertEqual(result['manifest']['conversion_policy']['quantized_tensors'], 'dequantize-float32-v1')
        self.assertEqual(result['manifest']['quantized_tensors']['count'], 1)
        self.assertTrue(conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')['cache_hit'])
        self.assertEqual(source.read_bytes(), b'mock checkpoint source')

    def test_quantized_mode_keeps_dense_conversion_cache_separate(self):
        source = self.checkpoint()
        strict = conversion.materialize_safetensors(source, self.cache)
        widened = conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')
        self.assertNotEqual(strict['converted_path'], widened['converted_path'])
        self.assertEqual(widened['manifest']['quantized_tensors']['count'], 0)
        self.assertTrue(conversion.materialize_safetensors(source, self.cache)['cache_hit'])
        self.assertTrue(conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')['cache_hit'])

    def test_quantized_cache_rejects_a_changed_conversion_policy(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')
        manifest_path = Path(result['manifest_path'])
        manifest = json.loads(manifest_path.read_text())
        manifest['conversion_policy']['quantized_tensors'] = 'unexpected'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'conversion policy'):
            conversion.materialize_safetensors(source, self.cache, quantized_tensors='dequantize')

    def test_quantized_mode_rejects_unsupported_types_schemes_and_scales(self):
        for tensor in (Tensor(dtype='qint8', quantized=True, quant_scheme='unknown'),
                       Tensor(dtype='unknown', quantized=True),
                       Tensor(dtype='qint8', quantized=True, quant_scale=0.0),
                       Tensor(dtype='qint8', quantized=True, quant_scale=float('nan'))):
            with self.subTest(tensor=tensor):
                self.torch.load.return_value = {'weight': tensor}
                with self.assertRaisesRegex(ValueError, 'quantiz'):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache, quantized_tensors='dequantize')
        with self.assertRaisesRegex(ValueError, 'quantized_tensors'):
            conversion.materialize_safetensors(self.checkpoint(), self.cache, quantized_tensors='unknown')
        self.save_file.assert_not_called()

    def test_quantized_mode_still_rejects_custom_and_packed_objects(self):
        class CustomTensor(Tensor):
            pass
        for state in ({'weight': CustomTensor(dtype='qint8', quantized=True)},
                      {'weight': (Tensor(dtype='qint8', quantized=True), Tensor())},
                      {'weight': object()}):
            with self.subTest(state=state):
                self.torch.load.return_value = state
                with self.assertRaisesRegex(ValueError, 'plain Torch tensor'):
                    conversion.materialize_safetensors(self.checkpoint(), self.cache, quantized_tensors='dequantize')
        self.save_file.assert_not_called()

    def test_source_mutation_during_load_rejects_even_same_size_content(self):
        source = self.checkpoint(data=b"before")
        def mutate(*args, **kwargs):
            source.write_bytes(b"AFTER!")
            return {"weight": Tensor()}
        self.torch.load.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "changed during weights-only loading"):
            conversion.materialize_safetensors(source, self.cache)
        self.save_file.assert_not_called()

    def test_source_mutation_during_save_rejects_and_removes_temporary_output(self):
        source = self.checkpoint(data=b"before")
        def mutate(*args, **kwargs):
            self.save(*args, **kwargs)
            source.write_bytes(b"AFTER!")
        self.save_file.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "Source checkpoint changed"):
            conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(list(conversion.provider.iterdir(self.cache)), [])

    def test_source_symlink_retargeting_is_detected(self):
        source = self.checkpoint()
        link = self.directory / "linked.ckpt"
        link.symlink_to(source)
        other = self.checkpoint("same.pt", source.read_bytes())
        def retarget(*args, **kwargs):
            self.save(*args, **kwargs)
            link.unlink()
            link.symlink_to(other)
        self.save_file.side_effect = retarget
        with self.assertRaisesRegex(RuntimeError, "Source checkpoint changed"):
            conversion.materialize_safetensors(link, self.cache)
        self.assertEqual(list(conversion.provider.iterdir(self.cache)), [])

    def test_corrupted_output_is_rejected_and_not_overwritten(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        output = Path(result["converted_path"])
        output.write_bytes(b"corrupted converted output")
        self.importer.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "no longer matches its manifest"):
            conversion.materialize_safetensors(source, self.cache)
        self.assertEqual(output.read_bytes(), b"corrupted converted output")
        self.importer.assert_not_called()

    def test_invalid_or_missing_conversion_policy_is_rejected(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        manifest_path = Path(result["manifest_path"])
        for change in (None, {"weights_only": False}, "invalid"):
            with self.subTest(change=change):
                manifest = dict(result["manifest"])
                if change is None:
                    manifest.pop("conversion_policy")
                else:
                    manifest["conversion_policy"] = change
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "Invalid checkpoint conversion cache"):
                    conversion.materialize_safetensors(source, self.cache)
        self.torch.load.assert_called_once()

    def test_cached_tensor_file_cannot_be_replaced_by_symlink(self):
        source = self.checkpoint()
        result = conversion.materialize_safetensors(source, self.cache)
        output = Path(result["converted_path"])
        copy = self.directory / "other.safetensors"
        shutil.copyfile(output, copy)
        output.unlink()
        output.symlink_to(copy)
        with self.assertRaisesRegex(RuntimeError, "must not be symlinks"):
            conversion.materialize_safetensors(source, self.cache)

    def test_save_failure_never_publishes_partial_cache(self):
        def fail(tensors, filename, metadata):
            Path(filename).write_bytes(b"partial output")
            raise RuntimeError("save failed")
        self.save_file.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertEqual(list(conversion.provider.iterdir(self.cache)), [])

    def test_manifest_and_tensor_file_are_published_together(self):
        rename = conversion.provider.rename
        def checked_rename(source, destination):
            self.assertTrue((source / "model.safetensors").is_file())
            self.assertTrue((source / "conversion.json").is_file())
            self.assertFalse(destination.exists())
            rename(source, destination)
        with patch.object(conversion.provider, "rename", side_effect=checked_rename) as renamed:
            result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        renamed.assert_called_once()
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertEqual(len(list(conversion.provider.iterdir(self.cache))), 1)

    def test_concurrent_cache_winner_is_validated_and_temporary_entry_removed(self):
        def competing_rename(source, destination):
            shutil.copytree(source, destination)
            raise FileExistsError("A concurrent conversion won publication")
        with patch.object(conversion.provider, "rename", side_effect=competing_rename):
            result = conversion.materialize_safetensors(self.checkpoint(), self.cache)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(len(list(conversion.provider.iterdir(self.cache))), 1)


@unittest.skipUnless(os.environ.get("IILD_CHECKPOINT_REAL_RUNTIME_TESTS") == "1",
                     "Explicit local Torch/safetensors smoke is disabled")
class RealCheckpointConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.safetensors = importlib.import_module("safetensors.torch")
        cls.temporary = resource_fixture.TemporaryDirectory(prefix="checkpoint-conversion-smoke-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.smoke_run = Path(cls.temporary.name)

    def test_quantized_torch_tensors_are_restored_in_single_and_sharded_checkpoints(self):
        torch = self.torch
        torch.set_num_threads(2)
        dense = torch.tensor([[-1.5, -.5, .5], [1., 2., 3.]], dtype=torch.float32)
        per_tensor = torch.quantize_per_tensor(dense, scale=.125, zero_point=1, dtype=torch.qint8)
        per_channel = torch.quantize_per_channel(dense, scales=torch.tensor([.125, .25]),
            zero_points=torch.tensor([3, 4]), axis=0, dtype=torch.quint8)
        per_tensor_int32 = torch.quantize_per_tensor(dense, scale=.125, zero_point=0, dtype=torch.qint32)
        values = {'first.weight': per_tensor, 'second.weight': per_channel,
                  'third.weight': per_tensor_int32, 'bias': torch.tensor([.5, -1.], dtype=torch.float16)}
        root = self.smoke_run / 'quantized-conversion';root.mkdir()
        source = root / 'single.pt';torch.save({'state_dict': values}, source)
        a, b = root / 'a.bin', root / 'b.bin'
        torch.save({'first.weight': per_tensor}, a)
        torch.save({'second.weight': per_channel, 'third.weight': per_tensor_int32, 'bias': values['bias']}, b)
        index = root / 'model.index.json'
        index.write_text(json.dumps({'weight_map': {'first.weight': a.name,
            'second.weight': b.name, 'third.weight': b.name, 'bias': b.name}}))
        originals = {path: path.read_bytes() for path in (source, index, a, b)}
        for path in (source, index):
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(ValueError, 'quantized'):
                    conversion.materialize_safetensors(path, root / 'cache')
                result = conversion.materialize_safetensors(path, root / 'cache', quantized_tensors='dequantize')
                loaded = self.safetensors.load_file(result['converted_path'])
                self.assertEqual(set(loaded), set(values))
                for name, tensor in values.items():
                    torch.testing.assert_close(loaded[name], tensor.dequantize() if tensor.is_quantized else tensor,
                                               rtol=0, atol=0, msg=name)
                self.assertEqual(result['manifest']['quantized_tensors']['count'], 3)
                self.assertEqual(set(result['manifest']['quantized_tensors']['dtypes']),
                                 {'torch.qint8', 'torch.quint8', 'torch.qint32'})
                self.assertEqual(result['manifest']['conversion_policy']['quantized_tensors'], 'dequantize-float32-v1')
                self.assertTrue(conversion.materialize_safetensors(path, root / 'cache',
                                quantized_tensors='dequantize')['cache_hit'])
                conversion.verify_conversion_source(result)
        import generate_any
        config = root / 'pipeline-config';config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'FakePipeline'}))
        tokens = ['--model', str(source), '--model-config', str(config),
                  '--cache-dir', str(root / 'cli-cache'), '--output-dir', str(root / 'output'), '--no-default-modifiers']
        with self.assertRaisesRegex(ValueError, 'quantized'):
            generate_any.resolve_arguments(generate_any.build_parser().parse_args(tokens))
        args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([*tokens, '--dequantize-pytorch']))
        self.assertTrue(generate_any.configuration(args)['dequantize_pytorch'])
        self.assertEqual(args.model_conversion['manifest']['quantized_tensors']['count'], 3)
        torch.testing.assert_close(self.safetensors.load_file(args.model)['first.weight'],
                                   per_tensor.dequantize(), rtol=0, atol=0)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_quantized_per_channel_float_zero_points_are_restored_when_supported(self):
        torch = self.torch
        if not hasattr(torch, '_make_per_channel_quantized_tensor'):
            self.skipTest('This Torch version cannot construct float-qparams fixtures')
        storage = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.uint8)
        try:
            value = torch._make_per_channel_quantized_tensor(storage, torch.tensor([.125, .25]),
                torch.tensor([1.5, 2.5]), 0)
            expected = value.dequantize()
        except (RuntimeError, NotImplementedError) as error:
            self.skipTest(f'This Torch build does not support float-qparams dequantization: {error}')
        self.assertEqual(value.qscheme(), torch.per_channel_affine_float_qparams)
        source = self.smoke_run / 'float-qparams.pt';torch.save({'weight': value}, source)
        result = conversion.materialize_safetensors(source, self.smoke_run / 'float-qparams-cache',
                                                     quantized_tensors='dequantize')
        loaded = self.safetensors.load_file(result['converted_path'])['weight']
        torch.testing.assert_close(loaded, expected, rtol=0, atol=0)
        self.assertEqual(result['manifest']['quantized_tensors']['count'], 1)

    def test_real_zip_and_legacy_serialization_preserve_values_dtypes_and_tied_keys(self):
        torch = self.torch
        from image_package import probe_model_container
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        shared = torch.arange(16, dtype=torch.float16)
        state = {"weight": weight, "tied.weight": weight,
                 "mask": torch.tensor([True, False]), "step": torch.tensor(5, dtype=torch.int64),
                 "shared": shared, "shared_alias": shared, "shared_slice": shared[2:8],
                 "empty": torch.empty(0, dtype=torch.bfloat16),
                 "bf16": torch.arange(8, dtype=torch.float32).to(torch.bfloat16),
                 "int8": torch.tensor([-128, -1, 0, 127], dtype=torch.int8),
                 "scale": torch.tensor(0.125, dtype=torch.float32)}
        results = []
        for name, zip_serialization in (("tiny.ckpt", True), ("tiny-disguised.safetensors", True),
                                        ("tiny-legacy.data", False)):
            source = self.smoke_run / name
            torch.save({"state_dict": state, "epoch": 1}, source,
                       _use_new_zipfile_serialization=zip_serialization)
            self.assertEqual(probe_model_container(source), 'pytorch')
            before = conversion.file_sha256(source)
            result = conversion.materialize_safetensors(source, self.smoke_run / "cache")
            expected_mode = 'mmap-private-snapshot' if zip_serialization and os.name != 'nt' else 'weights-only-cpu'
            self.assertEqual(result['manifest']['source_loading'], expected_mode)
            restored = self.safetensors.load_file(result["converted_path"], device="cpu")
            self.assertEqual(set(restored), set(state))
            for key in state:
                self.assertTrue(torch.equal(restored[key], state[key]), key)
                self.assertEqual(restored[key].dtype, state[key].dtype)
                self.assertTrue(restored[key].is_contiguous())
            self.assertEqual(conversion.file_sha256(source), before)
            results.append({"source": result["original"], "output": result["output"],
                            "manifest_path": result["manifest_path"],
                            "zip_serialization": zip_serialization,
                            "tensor_values_equal": True, "source_unchanged": True})
        print(json.dumps({
            "torch_version": torch.__version__, "cases": results, "generation_verified": False,
        }, indent=2))

    def test_compressed_zip_uses_regular_weights_only_loader(self):
        torch = self.torch
        uncompressed = self.smoke_run / 'zip-original.pt'
        source = self.smoke_run / 'zip-compressed.pt'
        state = {'weight': torch.arange(64, dtype=torch.float32)}
        torch.save(state, uncompressed)
        with zipfile.ZipFile(uncompressed) as incoming, zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as outgoing:
            for entry in incoming.infolist():
                outgoing.writestr(entry.filename, incoming.read(entry.filename))
        before = conversion.file_sha256(source)
        result = conversion.materialize_safetensors(source, self.smoke_run / 'compressed-cache')
        self.assertEqual(result['manifest']['source_loading'], 'weights-only-cpu')
        self.assertTrue(torch.equal(self.safetensors.load_file(result['converted_path'])['weight'], state['weight']))
        self.assertEqual(conversion.file_sha256(source), before)

    def test_real_legacy_sd1_routes_to_native_conversion_without_loading_a_model(self):
        import standalone_image
        import native_image
        from image_package import probe_model_container
        torch = self.torch
        source = self.smoke_run / 'sd1-disguised.safetensors'
        state = {
            'model.diffusion_model.input_blocks.0.0.weight': torch.zeros(1, 4, 1, 1),
            'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': torch.zeros(1, 768),
            'first_stage_model.encoder.conv_in.weight': torch.zeros(1),
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': torch.zeros(1),
        }
        torch.save({'state_dict': state}, source)
        self.assertEqual(probe_model_container(source), 'pytorch')
        before = conversion.file_sha256(source)
        args = standalone_image.build_parser().parse_args(['--model', str(source), '--engine', 'native',
            '--cache-dir', str(self.smoke_run / 'sd1-cache'), '--no-default-modifiers', '--no-hires-fix'])
        _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(resolved.architecture, 'sd1')
        self.assertEqual(resolved.engine, 'native')
        self.assertEqual(native_image.configuration_values(resolved)['model'], str(source))
        self.assertTrue(resolved.model_conversion['converted'])
        self.assertEqual(set(self.safetensors.load_file(resolved.model)), set(state))
        self.assertEqual(conversion.file_sha256(source), before)
        conversion.verify_conversion_source(resolved.model_conversion)

    def test_real_sharded_pytorch_converts_and_rejects_changed_sources(self):
        from image_package import probe_model_container
        torch = self.torch
        first = self.smoke_run / 'split-first.bin'
        second = self.smoke_run / 'split-second.data'
        parts = ({'first.weight': torch.arange(12, dtype=torch.float32).reshape(3, 4)},
                 {'second.weight': torch.tensor([-8, 2, 127], dtype=torch.int8),
                  'other.weight': torch.arange(3, dtype=torch.float16)})
        torch.save(parts[0], first); torch.save(parts[1], second)
        index = self.smoke_run / 'split-checkpoint.json'
        index.write_text(json.dumps({'weight_map': {key: path.name for path, part in ((first, parts[0]), (second, parts[1])) for key in part}}))
        self.assertEqual(probe_model_container(index), 'pytorch-index')
        original = {path: path.read_bytes() for path in (index, first, second)}
        result = conversion.materialize_safetensors(index, self.smoke_run / 'split-cache')
        self.assertFalse((Path(result['converted_path']).parent / 'parts').exists(),
                         'Intermediate shards must not persist beside the merged checkpoint')
        restored = self.safetensors.load_file(result['converted_path'], device='cpu')
        expected = {**parts[0], **parts[1]}
        self.assertEqual(set(restored), set(expected))
        for key, value in expected.items(): self.assertTrue(torch.equal(restored[key], value), key)
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertTrue(conversion.materialize_safetensors(index, self.smoke_run / 'split-cache')['cache_hit'])
        conversion.verify_conversion_source(result)
        output = Path(result['converted_path'])
        output.write_bytes(b'corrupted cached output')
        with self.assertRaisesRegex(RuntimeError, 'Invalid checkpoint conversion cache'):
            conversion.materialize_safetensors(index, self.smoke_run / 'split-cache')
        self.assertEqual(output.read_bytes(), b'corrupted cached output')
        first.write_bytes(first.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            conversion.verify_conversion_source(result)

    def test_real_sharded_pytorch_rejects_wrong_or_incomplete_mapping(self):
        torch = self.torch
        first, second = self.smoke_run / 'mapping-first.bin', self.smoke_run / 'mapping-second.bin'
        torch.save({'a': torch.zeros(1), 'extra': torch.ones(1)}, first)
        torch.save({'b': torch.zeros(1)}, second)
        index = self.smoke_run / 'mapping.json'
        for mapping in ({'a': first.name, 'b': second.name},
                        {'a': second.name, 'extra': first.name, 'b': second.name}):
            with self.subTest(mapping=mapping):
                index.write_text(json.dumps({'weight_map': mapping}))
                with self.assertRaisesRegex(ValueError, 'exactly'):
                    conversion.materialize_safetensors(index, self.smoke_run / 'mapping-cache')

    def test_real_sharded_sd1_routes_to_native_without_inference(self):
        import native_image
        import standalone_image
        torch = self.torch
        first, second = self.smoke_run / 'sd1-first.bin', self.smoke_run / 'sd1-second.bin'
        parts = ({'model.diffusion_model.input_blocks.0.0.weight': torch.zeros(1, 4, 1, 1),
                  'first_stage_model.encoder.conv_in.weight': torch.zeros(1)},
                 {'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': torch.zeros(1, 768),
                  'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': torch.zeros(1)})
        torch.save(parts[0], first); torch.save(parts[1], second)
        index = self.smoke_run / 'sd1-split.json'
        index.write_text(json.dumps({'weight_map': {key: path.name for path, part in ((first, parts[0]), (second, parts[1])) for key in part}}))
        args = standalone_image.build_parser().parse_args(['--model', str(index), '--engine', 'native',
            '--cache-dir', str(self.smoke_run / 'sd1-split-cache'), '--no-default-modifiers', '--no-hires-fix'])
        _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual((resolved.architecture, resolved.engine), ('sd1', 'native'))
        self.assertEqual(native_image.configuration_values(resolved)['model'], str(index))
        self.assertEqual(set(self.safetensors.load_file(resolved.model)), set(parts[0]) | set(parts[1]))
        conversion.verify_conversion_source(resolved.model_conversion)

    def test_real_sharded_moss_routes_to_generic_without_inference(self):
        import importlib.util
        import generate_any as generic
        import moss_runtime
        torch = self.torch
        first, second = self.smoke_run / 'moss-first.bin', self.smoke_run / 'moss-second.bin'
        parts = ({'video_embedder.weight': torch.zeros(4, 2)},
                 {'final_linear_video.weight': torch.zeros(2, 4), 'layers.0.proj.qkv.weight': torch.zeros(12, 4)})
        torch.save(parts[0], first); torch.save(parts[1], second)
        index = self.smoke_run / 'moss-index.json'
        index.write_text(json.dumps({'weight_map': {key: path.name for path, part in ((first, parts[0]), (second, parts[1])) for key in part}}))
        config = self.smoke_run / 'moss-sharded-config';config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline', **moss_runtime.COMPONENTS}))
        spec = importlib.util.spec_from_file_location('sharded_moss_router', ROOT / 'reference/generate.py')
        router = importlib.util.module_from_spec(spec);spec.loader.exec_module(router)
        self.assertTrue(router.defer_unsupported_foreground(['--model', str(index)]))
        observed = []
        def preflight(tokens):
            observed.append(generic.resolve_arguments(generic.build_parser().parse_args(tokens)))
            return 0
        with patch.object(generic, 'main', side_effect=preflight), \
             patch.object(generic.model_config_defaults, 'resolve', return_value=config), \
             patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
             patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
             patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')):
            self.assertEqual(router.main(['--model', str(index), '--prompt', 'fixture', '--width', '64', '--height', '64',
                '--steps', '1', '--cache-dir', str(self.smoke_run / 'moss-sharded-cache'),
                '--output-dir', str(self.smoke_run / 'moss-sharded-output'),
                '--no-default-modifiers', '--no-hires-fix']), 0)
        self.assertEqual(len(observed), 1)
        request = observed[0]
        self.assertEqual(request.model_config_preset, 'moss-image-sft')
        self.assertEqual(generic.configuration(request)['model'], str(index))
        self.assertEqual(set(self.safetensors.load_file(request.model)), set(parts[0]) | set(parts[1]))
        self.assertFalse((self.smoke_run / 'moss-sharded-output').exists())

    def test_real_sharded_video_and_encoder_prepare_without_inference(self):
        import video_checkpoint_conversion
        from video_options import build_parser, configuration, resolve_options
        torch = self.torch
        first, second = self.smoke_run / 'video-first.bin', self.smoke_run / 'video-second.bin'
        encoder = self.smoke_run / 'video-encoder.pt'
        parts = ({'video.first': torch.ones(1)}, {'video.second': torch.arange(3, dtype=torch.float16)})
        torch.save(parts[0], first);torch.save(parts[1], second);torch.save({'encoder.weight': torch.ones(2)}, encoder)
        index = self.smoke_run / 'video-index.json'
        index.write_text(json.dumps({'weight_map': {key: path.name for path, part in ((first, parts[0]), (second, parts[1])) for key in part}}))
        args = resolve_options(build_parser().parse_args(['--model', str(index), '--text-encoder', str(encoder),
            '--cache-dir', str(self.smoke_run / 'video-cache'), '--output', str(self.smoke_run / 'output.mp4'),
            '--width', '32', '--height', '32', '--frames', '9', '--fps', '12']))
        original = configuration(args)
        video_checkpoint_conversion.prepare(args)
        self.assertEqual(args._native_original_configuration, original)
        self.assertEqual(set(self.safetensors.load_file(args.model)), set(parts[0]) | set(parts[1]))
        self.assertEqual(set(self.safetensors.load_file(str(args.text_encoder))), {'encoder.weight'})
        video_checkpoint_conversion.verify(args)
        second.write_bytes(second.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            video_checkpoint_conversion.verify(args)
        self.assertFalse((self.smoke_run / 'output.mp4').exists())

    def test_real_diffusers_directory_loads_dequantized_weights_without_inference(self):
        import diffusers
        import generate_any
        torch = self.torch
        torch.set_num_threads(2)
        root = self.smoke_run / 'quantized-ddpm-pipeline'
        unet = diffusers.UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
            layers_per_block=1, block_out_channels=(8,), down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',), norm_num_groups=4)
        pipeline = diffusers.DDPMPipeline(unet=unet, scheduler=diffusers.DDPMScheduler(num_train_timesteps=10))
        pipeline.save_pretrained(str(root), safe_serialization=False)
        source = root / 'unet/diffusion_pytorch_model.bin'
        state = dict(unet.state_dict())
        name = 'conv_in.weight'
        state[name] = torch.quantize_per_channel(state[name],
            scales=torch.full((state[name].shape[0],), .01),
            zero_points=torch.zeros(state[name].shape[0], dtype=torch.int64), axis=0, dtype=torch.qint8)
        torch.save(state, source)
        before = source.read_bytes()
        args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([
            '--model', str(root), '--cache-dir', str(self.smoke_run / 'quantized-ddpm-cache'),
            '--output-dir', str(self.smoke_run / 'quantized-ddpm-output'),
            '--no-default-modifiers', '--dequantize-pytorch']))
        loaded, report = generate_any.load_pipeline(args, diffusers, torch.float32)
        for key, expected in state.items():
            torch.testing.assert_close(loaded.unet.state_dict()[key],
                expected.dequantize() if expected.is_quantized else expected, rtol=0, atol=0, msg=key)
        self.assertEqual(source.read_bytes(), before)
        conversion = report['identity']['directory_conversions'][0]
        self.assertEqual(conversion['manifest']['quantized_tensors']['count'], 1)
        self.assertEqual(conversion['manifest']['conversion_policy']['quantized_tensors'], 'dequantize-float32-v1')
        self.assertFalse((self.smoke_run / 'quantized-ddpm-output').exists())
        generate_any.verify_identity(report['identity'])

    def test_real_diffusers_directory_loads_converted_shards_without_inference(self):
        import diffusers
        import generate_any
        torch = self.torch
        root = self.smoke_run / 'legacy-ddpm-pipeline'
        unet = diffusers.UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
            layers_per_block=1, block_out_channels=(8,), down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',), norm_num_groups=4)
        pipeline = diffusers.DDPMPipeline(unet=unet, scheduler=diffusers.DDPMScheduler(num_train_timesteps=10))
        pipeline.save_pretrained(str(root), safe_serialization=False)
        component = root / 'unet'
        single = component / 'diffusion_pytorch_model.bin'
        state = torch.load(single, map_location='cpu', weights_only=True)
        names = sorted(state)
        first, second = component / 'shard-one.bin', component / 'shard-two.bin'
        halves = (names[:len(names)//2], names[len(names)//2:])
        for path, keys in zip((first, second), halves):
            torch.save({key: state[key] for key in keys}, path)
        index = component / 'diffusion_pytorch_model.bin.index.json'
        index.write_text(json.dumps({'weight_map': {key: path.name
            for path, keys in zip((first, second), halves) for key in keys}}))
        single.unlink()  # This source is part of this test's temporary fixture.
        before = {path: path.read_bytes() for path in (index, first, second)}
        args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([
            '--model', str(root), '--cache-dir', str(self.smoke_run / 'legacy-ddpm-cache'),
            '--output-dir', str(self.smoke_run / 'legacy-ddpm-output'), '--no-default-modifiers']))
        loaded, report = generate_any.load_pipeline(args, diffusers, torch.float32)
        actual = loaded.unet.state_dict()
        self.assertEqual(set(actual), set(state))
        for key in state:
            self.assertTrue(torch.equal(actual[key], state[key]), key)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(len(report['identity']['directory_conversions']), 1)
        generate_any.verify_identity(report['identity'])
        self.assertEqual(generate_any.configuration(args)['model'], str(root))
        self.assertFalse((self.smoke_run / 'legacy-ddpm-output').exists())
        self.assertEqual(list((self.smoke_run / 'legacy-ddpm-cache' / 'pipeline-overlays').iterdir()), [])

        # An unambiguous variant-only directory is selected without any UI or
        # command-line override, then still loaded through safe tensors.
        variant_root = self.smoke_run / 'legacy-ddpm-variant'
        pipeline.save_pretrained(str(variant_root), safe_serialization=False)
        default_weights = variant_root / 'unet' / 'diffusion_pytorch_model.bin'
        variant_weights = default_weights.with_name('diffusion_pytorch_model.fp16.bin')
        default_weights.rename(variant_weights)
        component_target = self.smoke_run / 'legacy-variant-shared-unet'
        (variant_root / 'unet').rename(component_target)
        try:
            (variant_root / 'unet').symlink_to(component_target, target_is_directory=True)
        except OSError:
            component_target.rename(variant_root / 'unet')
        variant_before = variant_weights.read_bytes()
        variant_args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([
            '--model', str(variant_root), '--cache-dir', str(self.smoke_run / 'legacy-variant-cache'),
            '--output-dir', str(self.smoke_run / 'legacy-variant-output'), '--no-default-modifiers']))
        self.assertEqual(variant_args.variant, 'fp16')
        variant_pipeline, variant_report = generate_any.load_pipeline(variant_args, diffusers, torch.float32)
        for key, value in unet.state_dict().items():
            self.assertTrue(torch.equal(variant_pipeline.unet.state_dict()[key], value), key)
        self.assertEqual(variant_weights.read_bytes(), variant_before)
        generate_any.verify_identity(variant_report['identity'])

        # Transformers uses a different conventional filename from Diffusers.
        from transformers import CLIPTextConfig, CLIPTextModel
        import directory_checkpoint_conversion
        clip_root = self.smoke_run / 'legacy-clip-directory'
        clip_dir = clip_root / 'text_encoder';clip_dir.mkdir(parents=True)
        clip = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
            bos_token_id=0, eos_token_id=2, pad_token_id=1))
        clip.config.save_pretrained(str(clip_dir))
        clip_source = clip_dir / 'pytorch_model.bin';torch.save(clip.state_dict(), clip_source)
        clip_before = clip_source.read_bytes()
        component_index = {'_class_name': 'FakePipeline', 'text_encoder': ['transformers', 'CLIPTextModel']}
        with directory_checkpoint_conversion.prepare(clip_root, component_index,
                self.smoke_run / 'legacy-clip-cache', self.smoke_run / 'legacy-clip-output', None) as prepared:
            loaded_clip = CLIPTextModel.from_pretrained(str(prepared.path / 'text_encoder'),
                local_files_only=True, use_safetensors=True, trust_remote_code=False)
            for key, value in clip.state_dict().items():
                self.assertTrue(torch.equal(loaded_clip.state_dict()[key], value), key)
            clip_conversion = prepared.conversions[0]
        self.assertEqual(clip_source.read_bytes(), clip_before)
        conversion.verify_conversion_source(clip_conversion)

    def test_real_diffusers_single_file_auxiliary_loader_uses_sanitized_shards(self):
        import diffusers
        import directory_checkpoint_conversion
        from diffusers.loaders.single_file import load_single_file_sub_model
        torch = self.torch
        root = self.smoke_run / 'single-file-config'
        unet = diffusers.UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
            layers_per_block=1, block_out_channels=(8,), down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',), norm_num_groups=4)
        unet.save_pretrained(root / 'unet', safe_serialization=False, max_shard_size='32KB')
        index = {'_class_name': 'FakePipeline', 'unet': ['diffusers', 'UNet2DModel']}
        (root / 'model_index.json').write_text(json.dumps(index))
        original = {path: path.read_bytes() for path in (root / 'unet').glob('*.bin')}
        shard_index = root / 'unet/diffusion_pytorch_model.bin.index.json'
        self.assertTrue(shard_index.is_file())
        original[shard_index] = shard_index.read_bytes()
        with directory_checkpoint_conversion.prepare(root, index, self.smoke_run / 'single-file-cache',
                self.smoke_run / 'single-file-output', None, strip_legacy=True) as prepared:
            self.assertFalse(list((prepared.path / 'unet').glob('*.bin')))
            self.assertFalse((prepared.path / 'unet/diffusion_pytorch_model.bin.index.json').exists())
            self.assertTrue((prepared.path / 'unet/diffusion_pytorch_model.safetensors').is_file())
            loaded = load_single_file_sub_model(library_name='diffusers', class_name='UNet2DModel', name='unet',
                checkpoint={}, pipelines=diffusers.pipelines, is_pipeline_module=False,
                cached_model_config_path=str(prepared.path), local_files_only=True, torch_dtype=torch.float32)
            for name, value in unet.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
            self.assertEqual(len(prepared.conversions), 1)
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertFalse((self.smoke_run / 'single-file-output').exists())

        # Content, not the extension of the component or its index, selects the loader.
        renamed_root = self.smoke_run / 'single-file-renamed-index'
        shutil.copytree(root, renamed_root)
        old_index = renamed_root / 'unet/diffusion_pytorch_model.bin.index.json'
        disguised_index = renamed_root / 'unet/diffusion_pytorch_model.safetensors.index.json'
        old_index.rename(disguised_index)
        with directory_checkpoint_conversion.prepare(renamed_root, index, self.smoke_run / 'renamed-index-cache',
                self.smoke_run / 'renamed-index-output', None, strip_legacy=True) as prepared:
            self.assertFalse((prepared.path / 'unet/diffusion_pytorch_model.safetensors.index.json').exists())
            loaded = diffusers.UNet2DModel.from_pretrained(str(prepared.path / 'unet'),
                local_files_only=True, use_safetensors=True)
            for name, value in unet.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)

        renamed_file_root = self.smoke_run / 'directory-renamed-weight'
        renamed_unet = renamed_file_root / 'unet';renamed_unet.mkdir(parents=True)
        unet.save_config(renamed_unet)
        disguised_file = renamed_unet / 'diffusion_pytorch_model.safetensors'
        torch.save(unet.state_dict(), disguised_file)
        before = disguised_file.read_bytes()
        with directory_checkpoint_conversion.prepare(renamed_file_root, index,
                self.smoke_run / 'renamed-file-cache', self.smoke_run / 'renamed-file-output', None) as prepared:
            loaded = diffusers.UNet2DModel.from_pretrained(str(prepared.path / 'unet'),
                local_files_only=True, use_safetensors=True)
            for name, value in unet.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
        self.assertEqual(disguised_file.read_bytes(), before)

        alternate_root = self.smoke_run / 'directory-pth-weight'
        alternate_unet = alternate_root / 'unet';alternate_unet.mkdir(parents=True)
        unet.save_config(alternate_unet)
        alternate = alternate_unet / 'diffusion_pytorch_model.pth'
        torch.save(unet.state_dict(), alternate)
        before = alternate.read_bytes()
        with directory_checkpoint_conversion.prepare(alternate_root, index,
                self.smoke_run / 'alternate-file-cache', self.smoke_run / 'alternate-file-output', None) as prepared:
            loaded = diffusers.UNet2DModel.from_pretrained(str(prepared.path / 'unet'),
                local_files_only=True, use_safetensors=True)
            for name, value in unet.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
        self.assertEqual(alternate.read_bytes(), before)

        safe_shard_root = self.smoke_run / 'single-file-safe-sharded-unet'
        safe_unet = safe_shard_root / 'unet'
        unet.save_pretrained(safe_unet, safe_serialization=True, max_shard_size='32KB')
        import generate_any as generic
        self.assertIn('unet', generic.single_file_diffusers_fallback_components(diffusers, index))
        self.assertTrue((safe_unet / 'diffusion_pytorch_model.safetensors.index.json').is_file())
        original = {path: path.read_bytes() for path in safe_unet.iterdir() if path.is_file()}
        with directory_checkpoint_conversion.prepare(safe_shard_root, index,
                self.smoke_run / 'safe-shard-cache', self.smoke_run / 'safe-shard-output', None,
                strip_legacy=True, merge_diffusers_components={'unet'}) as prepared:
            self.assertTrue((prepared.path / 'unet/diffusion_pytorch_model.safetensors').is_file())
            loaded = load_single_file_sub_model(library_name='diffusers', class_name='UNet2DModel', name='unet',
                checkpoint={}, pipelines=diffusers.pipelines, is_pipeline_module=False,
                cached_model_config_path=str(prepared.path), local_files_only=True, torch_dtype=torch.float32)
            for name, value in unet.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
            self.assertEqual(prepared.conversions[0]['manifest']['source_kind'], 'safetensors-index')
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        (safe_shard_root / 'model_index.json').write_text(json.dumps(index))
        args = generic.resolve_arguments(generic.build_parser().parse_args([
            '--model', str(safe_unet / 'diffusion_pytorch_model.safetensors.index.json'),
            '--model-config', str(safe_shard_root), '--cache-dir', str(self.smoke_run / 'safe-main-cache'),
            '--output-dir', str(self.smoke_run / 'safe-main-output'), '--no-default-modifiers']))
        self.assertEqual(args.model_conversion['manifest']['source_kind'], 'safetensors-index')
        restored = self.safetensors.load_file(args.model)
        self.assertEqual(set(restored), set(unet.state_dict()))
        for name, value in unet.state_dict().items():
            self.assertTrue(torch.equal(restored[name], value), name)
        self.assertEqual({path: path.read_bytes() for path in original}, original)

        # Exercise Diffusers' actual single-file fallback for a Transformers
        # encoder. Its upstream existence check expects a Diffusers filename.
        from transformers import CLIPTextConfig, CLIPTextModel
        clip = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
            bos_token_id=0, eos_token_id=2, pad_token_id=1))
        encoder_index = {'_class_name': 'FakePipeline', 'text_encoder': ['transformers', 'CLIPTextModel']}
        for kind in ('legacy', 'safe', 'variant-shards'):
            encoder_root = self.smoke_run / ('single-file-encoder-' + kind)
            encoder = encoder_root / 'text_encoder';encoder.mkdir(parents=True)
            if kind == 'legacy':
                clip.config.save_pretrained(encoder)
                torch.save(clip.state_dict(), encoder / 'pytorch_model.bin')
            elif kind == 'safe':
                clip.save_pretrained(encoder, safe_serialization=True)
            else:
                clip.save_pretrained(encoder, safe_serialization=True, variant='fp16', max_shard_size='1KB')
                self.assertTrue((encoder / 'model.safetensors.index.fp16.json').is_file())
                self.safetensors.save_file({name: value.contiguous() + 1 for name, value in clip.state_dict().items()},
                                          str(encoder / 'model.safetensors'))
            original = {path: path.read_bytes() for path in encoder.iterdir() if path.is_file()}
            selected_variant = 'fp16' if kind == 'variant-shards' else None
            with directory_checkpoint_conversion.prepare(encoder_root, encoder_index,
                    self.smoke_run / ('encoder-cache-' + kind), self.smoke_run / ('encoder-output-' + kind),
                    selected_variant, strip_legacy=True, transformer_markers=True) as prepared:
                self.assertNotEqual(prepared.path, encoder_root)
                self.assertTrue((prepared.path / 'text_encoder/diffusion_pytorch_model.safetensors').is_file())
                self.assertFalse((prepared.path / 'text_encoder/pytorch_model.bin').exists())
                loaded = load_single_file_sub_model(library_name='transformers', class_name='CLIPTextModel',
                    name='text_encoder', checkpoint={}, pipelines=diffusers.pipelines, is_pipeline_module=False,
                    cached_model_config_path=str(prepared.path), local_files_only=True, torch_dtype=torch.float32)
                for name, value in clip.state_dict().items():
                    self.assertTrue(torch.equal(loaded.state_dict()[name], value), (kind, name))
            self.assertEqual({path: path.read_bytes() for path in original}, original)
            self.assertFalse((encoder / 'diffusion_pytorch_model.safetensors').exists())

    def test_real_pytorch_moss_reaches_generic_runner_without_model_inference(self):
        import importlib.util
        import generate_any as generic
        import moss_runtime
        torch = self.torch
        source = self.smoke_run / 'moss-disguised.safetensors'
        state = {'video_embedder.weight': torch.zeros(4, 2),
                 'final_linear_video.weight': torch.zeros(2, 4),
                 'layers.0.proj.qkv.weight': torch.zeros(12, 4)}
        torch.save({'state_dict': state}, source)
        before = conversion.file_sha256(source)
        config = self.smoke_run / 'moss-config'
        config.mkdir()
        (config / 'model_index.json').write_text(json.dumps({'_class_name': 'MOSSImagePipeline',
                                                              **moss_runtime.COMPONENTS}))
        spec = importlib.util.spec_from_file_location('real_moss_router', ROOT / 'reference/generate.py')
        router = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(router)
        observed = []
        def preflight(tokens):
            observed.append(generic.resolve_arguments(generic.build_parser().parse_args(tokens)))
            return 0
        with patch.object(generic, 'main', side_effect=preflight), \
             patch.object(generic.model_config_defaults, 'resolve', return_value=config), \
             patch.object(generic.moss_text_encoder, 'resolve_fallback', return_value=(None, 'model')), \
             patch.object(generic.moss_tokenizer, 'resolve', return_value=(None, 'model')), \
             patch.object(generic, 'resolve_vae_selection', return_value=(None, 'model')):
            self.assertEqual(router.main(['--model', str(source), '--prompt', 'fixture',
                '--width', '64', '--height', '64', '--steps', '1', '--num-images', '2',
                '--cache-dir', str(self.smoke_run / 'moss-cache'),
                '--output-dir', str(self.smoke_run / 'moss-output'),
                '--no-default-modifiers', '--no-hires-fix']), 0)
        self.assertEqual(len(observed), 1)
        request = observed[0]
        self.assertEqual(request.model_config_preset, 'moss-image-sft')
        self.assertTrue(request.model_conversion['converted'])
        self.assertEqual(generic.configuration(request)['model'], str(source))
        self.assertEqual(request.inputs['num_images_per_prompt'], 2)
        self.assertEqual(set(self.safetensors.load_file(request.model)), set(state))
        self.assertEqual(conversion.file_sha256(source), before)
        self.assertFalse((self.smoke_run / 'moss-output').exists())

    def test_malicious_reduce_is_rejected_without_executing_its_marker(self):
        marker = self.smoke_run / "unexpected-code-execution.txt"
        class MaliciousReduce:
            def __reduce__(self):
                return (eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",))
        source = self.smoke_run / "malicious.ckpt"
        self.torch.save({"state_dict": {"weight": MaliciousReduce()}}, source)
        self.assertFalse(marker.exists())
        before = conversion.file_sha256(source)
        with self.assertRaisesRegex(ValueError, "unsafe pickle retry is disabled") as caught:
            conversion.materialize_safetensors(source, self.smoke_run / "malicious-cache")
        self.assertFalse(marker.exists())
        self.assertEqual(conversion.file_sha256(source), before)
        self.assertFalse((self.smoke_run / "malicious-cache").exists())
        print(json.dumps({
            "torch_version": self.torch.__version__, "source": str(source),
            "source_sha256": before, "source_unchanged": True,
            "marker": str(marker), "marker_exists": marker.exists(),
            "error": str(caught.exception), "cause_type": type(caught.exception.__cause__).__name__,
            "unsafe_retry": False, "generation_verified": False,
        }, indent=2))


if __name__ == "__main__":
    unittest.main()
