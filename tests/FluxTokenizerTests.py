"""FLUX.1 CLIP token limits follow the actual encoder's position capacity."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import flux_tokenizer


class FluxTokenizerTests(unittest.TestCase):
    def test_missing_declared_components_reuse_the_existing_tokenizers(self):
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix='flux-tokenizers-') as temporary:
            root = Path(temporary)
            args = SimpleNamespace(pipeline_class=None, inputs={}, generation_resources=None)
            index = {'_class_name': 'FluxPipeline', 'tokenizer': ['transformers', 'CLIPTokenizer'],
                     'tokenizer_2': ['transformers', 'T5TokenizerFast']}
            selected, status = flux_tokenizer.resolve(args, index, root)
            self.assertEqual(status, {'tokenizer': 'fallback', 'tokenizer_2': 'fallback'})
            self.assertEqual(set(selected), {'tokenizer', 'tokenizer_2'})
            flux_tokenizer.verify_selections(selected)
            (root / 'tokenizer').mkdir()
            (root / 'tokenizer' / 'partial.json').write_text('{}')
            selected, status = flux_tokenizer.resolve(args, index, root)
            self.assertEqual(status['tokenizer'], 'model')
            self.assertNotIn('tokenizer', selected)
            self.assertEqual(status['tokenizer_2'], 'fallback')
            args.inputs = {'prompt_embeds': {}, 'pooled_prompt_embeds': {}}
            selected, status = flux_tokenizer.resolve(args, index, root)
            self.assertEqual(status['tokenizer_2'], 'precomputed-embeddings')
            self.assertNotIn('tokenizer_2', selected)
            args.inputs['negative_prompt'] = 'avoid blur'
            selected, status = flux_tokenizer.resolve(args, index, root)
            self.assertEqual(status['tokenizer_2'], 'fallback')
            self.assertIn('tokenizer_2', selected)

    def test_fallback_rejects_modified_or_untracked_resources(self):
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix='flux-tokenizer-integrity-') as temporary:
            root = Path(temporary)
            (root / 'text_encoders').mkdir()
            shutil.copytree(ROOT / 'resources/text_encoders/flux1-clip', root / 'text_encoders/flux1-clip')
            shutil.copytree(ROOT / 'resources/text_encoders/flux1-t5', root / 'text_encoders/flux1-t5')
            (root / 'generation-defaults.json').write_text((ROOT / 'resources/generation-defaults.json').read_text())
            args = SimpleNamespace(pipeline_class=None, inputs={}, generation_resources=root)
            index = {'_class_name': 'FluxPipeline', 'tokenizer': ['transformers', 'CLIPTokenizer'],
                     'tokenizer_2': ['transformers', 'T5TokenizerFast']}
            selections, _ = flux_tokenizer.resolve(args, index, root / 'model')
            (root / 'text_encoders/flux1-clip/added_tokens.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'untracked'):
                flux_tokenizer.verify_selections(selections)
            (root / 'text_encoders/flux1-clip/added_tokens.json').unlink()
            (root / 'text_encoders/flux1-t5/tokenizer_config.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                flux_tokenizer.verify_selections(selections)

    def pipeline(self, name, token_limit, positions):
        pipeline = type(name, (), {})()
        pipeline.tokenizer = SimpleNamespace(model_max_length=token_limit)
        pipeline.text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=positions))
        pipeline.tokenizer_max_length = token_limit
        return pipeline

    def test_comfy_clip_maximum_is_bounded_by_encoder(self):
        pipeline = self.pipeline('FluxPipeline', 8192, 77)
        self.assertEqual(flux_tokenizer.bound_clip_length(pipeline), {'original': 8192, 'effective': 77})
        self.assertEqual(pipeline.tokenizer.model_max_length, 77)
        self.assertEqual(pipeline.tokenizer_max_length, 77)

    def test_shorter_tokenizer_and_other_families_are_unchanged(self):
        for name, length, capacity in (('FluxPipeline', 64, 77), ('Flux2Pipeline', 8192, 77),
                                      ('FluxPriorReduxPipeline', 8192, 77),
                                      ('Krea2Pipeline', 8192, 77)):
            with self.subTest(name=name):
                pipeline = self.pipeline(name, length, capacity)
                self.assertIsNone(flux_tokenizer.bound_clip_length(pipeline))
                self.assertEqual(pipeline.tokenizer.model_max_length, length)
                self.assertEqual(pipeline.tokenizer_max_length, length)

    def test_precomputed_embeddings_need_no_tokenizer(self):
        pipeline = type('FluxPipeline', (), {'tokenizer': None, 'text_encoder': None})()
        self.assertIsNone(flux_tokenizer.bound_clip_length(pipeline))

    def test_invalid_encoder_position_capacity_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'position'):
                flux_tokenizer.bound_clip_length(self.pipeline('FluxPipeline', 8192, value))


if __name__ == '__main__':
    unittest.main()
