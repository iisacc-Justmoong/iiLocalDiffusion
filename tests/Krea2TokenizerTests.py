"""Krea 2 uses the verified existing tokenizer when a package omits it."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_tokenizer as tokenizer


class Krea2TokenizerTests(unittest.TestCase):
    def setUp(self):
        temporary = resource_fixture.TemporaryDirectory(prefix='krea2-tokenizer-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.resources = self.root / 'resources'
        source = ROOT / 'resources/text_encoders/qwen3vl-4b'
        target = self.resources / 'text_encoders/qwen3vl-4b'
        target.mkdir(parents=True)
        files = []
        for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
            path = target / name
            shutil.copyfile(source / name, path)
            files.append({'file': str(path.relative_to(self.resources)), 'size': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {'version': 1, 'fallback_tokenizers': [
            {'families': ['krea2'], 'class_name': 'Qwen2Tokenizer', 'files': files}]}
        (self.resources / 'generation-defaults.json').write_text(json.dumps(manifest))
        self.args = SimpleNamespace(pipeline_class=None, generation_resources=str(self.resources))
        self.index = {'_class_name': 'Krea2Pipeline', 'tokenizer': ['transformers', 'Qwen2Tokenizer']}

    def test_missing_tokenizer_selects_verified_existing_resources(self):
        selected, status = tokenizer.resolve(self.args, self.index, self.model, 'model')
        self.assertEqual(status, 'fallback')
        self.assertEqual(selected.directory, str(self.resources / 'text_encoders/qwen3vl-4b'))
        tokenizer.verify_selection(selected)
        self.assertEqual(len(selected.files), 3)

    def test_existing_or_partial_tokenizer_does_not_silently_fall_back(self):
        (self.model / 'tokenizer').mkdir()
        (self.model / 'tokenizer/tokenizer_config.json').write_text('{}')
        self.args.generation_resources = str(self.root / 'missing')
        self.assertEqual(tokenizer.resolve(self.args, self.index, self.model, 'model'), (None, 'model'))

    def test_precomputed_embeddings_need_no_tokenizer(self):
        self.args.generation_resources = str(self.root / 'missing')
        self.assertEqual(tokenizer.resolve(self.args, self.index, self.model, 'precomputed-embeddings'),
                         (None, 'precomputed-embeddings'))

    def test_incompatible_declaration_and_changed_resource_are_rejected(self):
        self.index['tokenizer'] = ['transformers', 'CLIPTokenizer']
        with self.assertRaisesRegex(ValueError, 'Qwen2Tokenizer'):
            tokenizer.resolve(self.args, self.index, self.model, 'model')
        self.index['tokenizer'] = ['transformers', 'Qwen2Tokenizer']
        selected, _ = tokenizer.resolve(self.args, self.index, self.model, 'model')
        (Path(selected.directory) / 'merges.txt').write_text('modified')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            tokenizer.verify_selection(selected)

    def test_untracked_tokenizer_override_is_rejected(self):
        selected, _ = tokenizer.resolve(self.args, self.index, self.model, 'model')
        (Path(selected.directory) / 'tokenizer.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            tokenizer.verify_selection(selected)

    def test_other_families_do_not_use_krea_tokenizer(self):
        self.index['_class_name'] = 'FluxPipeline'
        self.assertEqual(tokenizer.resolve(self.args, self.index, self.model, 'model'),
                         (None, 'not-configured-for-pipeline'))


if __name__ == '__main__':
    unittest.main()
