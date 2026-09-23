"""Generation must reach the loader without a full model checksum pass."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import weight_files as weights
import unified_image

class MetadataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='metadata-validation-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.safetensors'
        self.model.write_bytes(b'bad model contents left for the actual loader')
        self.environment = patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_first_resolution_and_verification_do_not_hash(self):
        with patch.object(weights, 'cached_model_sha256', side_effect=AssertionError('Full model read')):
            model = weights.resolve_weight_file(str(self.model), '--model')
            weights.verify_weight_file(model, 'model')
            self.assertIsNone(model.sha256)
            self.assertEqual(weights.weight_file_metadata(model)['validation'], 'metadata')

    def test_conversion_entry_does_not_hash_safetensors(self):
        from checkpoint_conversion import materialize_safetensors
        with patch.object(weights, 'cached_model_sha256', side_effect=AssertionError('Full model read')):
            result = materialize_safetensors(self.model, self.root / 'conversion')
            self.assertFalse(result['converted'])
            self.assertIsNone(result['original']['sha256'])

    def test_missing_empty_and_changed_paths_are_still_reported(self):
        model = weights.resolve_weight_file(str(self.model), '--model')
        self.model.unlink()
        with self.assertRaises(RuntimeError):
            weights.verify_weight_file(model, 'model')
        self.model.touch()
        with self.assertRaises(ValueError):
            weights.resolve_weight_file(str(self.model), '--model')

    def test_bad_published_checksum_reaches_loader_and_surfaces_its_error(self):
        package = self.root / 'bad.iildmodel'
        package.mkdir()
        member = package / 'model.safetensors'
        member.write_bytes(b'bad weights')
        (package / 'model_index.json').write_text(json.dumps({
            'schema': 'iild-unified-model-v1', '_class_name': 'IILDUnifiedCascade',
            'composition': 'ordered-image-refinement', 'stages': [{
                'model': member.name, 'strength': 1, 'size_bytes': member.stat().st_size,
                'sha256': '0' * 64}]}))
        with patch.object(weights, 'cached_model_sha256', side_effect=AssertionError('Full model read')), \
                patch.object(unified_image, 'NativeEngine', side_effect=RuntimeError('Cannot load model tensors')) as engine:
            with self.assertRaisesRegex(RuntimeError, 'Cannot load model tensors'):
                unified_image.main(['--model-path', str(package), '--output-dir', str(self.root / 'images')])
            engine.assert_called_once()

    def test_directory_identity_does_not_hash_model_bytes(self):
        import generate_any
        with patch.object(weights, 'cached_model_sha256', side_effect=AssertionError('Full model read')):
            identity = generate_any.model_identity(self.root)
            generate_any.verify_identity(identity)
            self.assertIsNone(identity['files'][0]['sha256'])
            self.model.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                generate_any.verify_identity(identity)

if __name__ == '__main__':
    unittest.main()
