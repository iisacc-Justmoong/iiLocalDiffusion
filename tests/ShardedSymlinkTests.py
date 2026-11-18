"""Synthetic Hugging Face-style index links across SDK loading boundaries."""
from contextlib import contextmanager
import json
import os
import pickle
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
from _file_provider import provider
from weight_files import verify_weight_file
import native_image


class ShardedSymlinkTests(unittest.TestCase):
    def setUp(self):
        temporary = provider.TemporaryDirectory(prefix='sharded-links-', dir=ROOT / 'resources')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.blobs = self.root / 'blobs'
        self.snapshot = self.root / 'snapshots/revision'
        self.blobs.mkdir()
        self.snapshot.mkdir(parents=True)
        self.shard_blob = self.blobs / 'opaque-shard'
        header = json.dumps({'fixture.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
        self.shard_blob.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        self.index_blob = self.blobs / 'opaque-index'
        self.index_blob.write_text(json.dumps({'weight_map': {'fixture.weight': 'part.safetensors'}}))
        self.index = self.snapshot / 'model.safetensors.index.json'
        self.shard = self.snapshot / 'part.safetensors'
        try:
            self.index.symlink_to(self.index_blob)
            self.shard.symlink_to(self.shard_blob)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')

    def test_native_weight_tracks_logical_index_and_resolved_dependencies(self):
        weight = native_image.native_weight(self.index, 'model')
        self.assertEqual(weight.path, str(self.index))
        self.assertEqual(weight.resolved_file, str(self.index_blob.resolve()))
        self.assertEqual(weight.execution_path, str(self.index))
        self.assertEqual([item.path for item in weight.shards], [str(self.shard)])
        verify_weight_file(weight, 'model')
        replacement = self.blobs / 'replacement-shard'
        replacement.write_bytes(self.shard_blob.read_bytes())
        self.shard.unlink()
        self.shard.symlink_to(replacement)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_weight_file(weight, 'model')

    def test_inspection_report_uses_the_logical_shard_base(self):
        from downloaded_model import inspect_downloaded_model
        report = inspect_downloaded_model(self.index)
        self.assertEqual(report['format'], 'safetensors-index')
        self.assertEqual(report['sharded_index']['shards'], [str(self.shard)])

    def test_indexed_moss_reader_uses_linked_shards(self):
        import moss_single_file
        @contextmanager
        def fake_open(path, **unused):
            data = Path(path).read_bytes()
            length, = struct.unpack('<Q', data[:8])
            descriptors = json.loads(data[8:8 + length])
            def tensor(name):
                begin, end = descriptors[name]['data_offsets']
                return data[8 + length + begin:8 + length + end]
            yield SimpleNamespace(get_tensor=tensor)
        with patch.dict(sys.modules, {'safetensors': SimpleNamespace(safe_open=fake_open)}):
            with moss_single_file.open_checkpoint(self.index) as source:
                self.assertEqual(source.get_slice('fixture.weight').get_shape(), [1])
                self.assertEqual(source.get_tensor('fixture.weight'), b'x')

    def test_renamed_linked_index_retains_its_logical_shard_base(self):
        from model_sources import local_model_path
        from downloaded_model import inspect_downloaded_model
        renamed = self.snapshot / 'weights.data'
        renamed.symlink_to(self.index_blob)
        self.assertEqual(local_model_path(renamed), renamed)
        report = inspect_downloaded_model(renamed)
        self.assertEqual(report['format'], 'safetensors-index')
        self.assertEqual(report['sharded_index']['shards'], [str(self.shard)])
        weight = native_image.native_weight(renamed, 'model')
        self.assertEqual(weight.execution_path, str(renamed))
        self.assertEqual([item.path for item in weight.shards], [str(self.shard)])
        verify_weight_file(weight, 'model')
        from video_options import build_parser, resolve_options
        args = resolve_options(build_parser().parse_args(['--model', str(renamed), '--text-encoder', str(renamed),
            '--video-vae', str(renamed), '--frames', '9', '--fps', '12', '--output', str(self.root / 'movie.mp4')]))
        self.assertEqual(args.model, str(renamed))
        self.assertEqual(args.text_encoder, renamed)
        self.assertEqual(args.video_vae, renamed)

    def test_model_input_and_video_component_paths_preserve_index_base(self):
        from model_sources import resolve_model_input
        from video_options import build_parser, resolve_options
        args = build_parser().parse_args(['--model', str(self.index), '--text-encoder', str(self.index),
            '--video-vae', str(self.index), '--frames', '9', '--fps', '12', '--output', str(self.root / 'movie.mp4')])
        self.assertEqual(resolve_model_input(args).location, str(self.index))
        result = resolve_options(args)
        self.assertEqual(result.model, str(self.index))
        self.assertEqual(result.text_encoder, self.index)
        self.assertEqual(result.video_vae, self.index)

    def test_native_image_request_and_components_preserve_index_base(self):
        import standalone_image
        inspection = {'format': 'safetensors-index', 'architecture': 'sd1', 'role': 'checkpoint',
                      'task': 'text-to-image', 'component_slots': ['clip_l']}
        args = standalone_image.build_parser().parse_args(['--model', str(self.index), '--engine', 'native',
            '--components', json.dumps({'vae': str(self.index)}), '--width', '64', '--height', '64'])
        with patch('standalone_image.inspect_downloaded_model', return_value=inspection):
            _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(resolved.model, str(self.index))
        self.assertEqual(resolved.components['vae'], str(self.index))
        self.assertEqual(resolved.model_selection.single_file.execution_path, str(self.index))
        self.assertEqual(resolved.native_component_files['vae'].execution_path, str(self.index))

    def test_safetensors_content_takes_precedence_over_index_suffix(self):
        from model_sources import resolve_model_input
        from video_options import build_parser
        misleading = self.snapshot / 'single.safetensors.index.json'
        misleading.symlink_to(self.shard_blob)
        args = build_parser().parse_args(['--model', str(misleading)])
        self.assertEqual(resolve_model_input(args).location, str(self.shard_blob.resolve()))
        self.assertEqual(native_image.native_weight(misleading, 'model').execution_path, str(self.shard_blob.resolve()))

    def test_index_retarget_is_detected_even_when_bytes_match(self):
        weight = native_image.native_weight(self.index, 'model')
        replacement = self.blobs / 'new-opaque-index'
        replacement.write_bytes(self.index_blob.read_bytes())
        self.index.unlink()
        self.index.symlink_to(replacement)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_weight_file(weight, 'model')

    def test_pytorch_index_link_retains_its_logical_shard_base(self):
        from downloaded_model import inspect_downloaded_model
        from image_package import inspect_pytorch_index
        from model_sources import local_model_path
        torch_blob = self.blobs / 'torch-blob'
        torch_blob.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture')
        index_blob = self.blobs / 'torch-index'
        index_blob.write_text(json.dumps({'weight_map': {'fixture.weight': 'part.bin'}}))
        index = self.snapshot / 'renamed-torch.data'
        shard = self.snapshot / 'part.bin'
        index.symlink_to(index_blob); shard.symlink_to(torch_blob)
        self.assertEqual(local_model_path(index), index)
        snapshot = inspect_pytorch_index(index)
        self.assertEqual(snapshot['path'], str(index))
        self.assertEqual(snapshot['shards'][0]['resolved_path'], str(torch_blob.resolve()))
        with patch('downloaded_model._torch_checkpoint', return_value=({'fixture.weight': (1,)}, {}, [])):
            report = inspect_downloaded_model(index)
        self.assertEqual(report['format'], 'pytorch-index')
        self.assertEqual(report['sharded_index']['shards'], [str(shard)])
        replacement = self.blobs / 'different-torch-blob'
        replacement.write_bytes(torch_blob.read_bytes())
        shard.unlink(); shard.symlink_to(replacement)
        self.assertNotEqual(inspect_pytorch_index(index), snapshot)


if __name__ == '__main__':
    unittest.main()
