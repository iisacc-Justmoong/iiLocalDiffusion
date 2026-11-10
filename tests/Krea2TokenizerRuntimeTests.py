"""Qwen3-VL tokenizer snapshots obtained from the pinned upstream tokenizer."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_tokenizer

AVAILABLE = all(importlib.util.find_spec(name) for name in ('transformers', 'tokenizers'))


@unittest.skipUnless(AVAILABLE, 'Install the reference Transformers runtime for tokenizer tests')
class Krea2TokenizerRuntimeTests(unittest.TestCase):
    def test_existing_tokenizer_matches_official_krea2_prompt_tokens(self):
        args = SimpleNamespace(pipeline_class=None, generation_resources=str(ROOT / 'resources'))
        selection, status = krea2_tokenizer.resolve(args, {'_class_name': 'Krea2Pipeline'}, ROOT / 'tests', 'model')
        self.assertEqual(status, 'fallback')
        tokenizer = krea2_tokenizer.load(selection)
        prefix = ('<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, '
                  'quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n')
        suffix = '<|im_end|>\n<|im_start|>assistant\n'
        expected_prefix = [151644, 8948, 198, 74785, 279, 2168, 553, 44193, 279, 1894, 11,
            6083, 11, 1379, 11, 10434, 11, 12194, 11, 1467, 11, 27979, 11871, 315, 279,
            6171, 323, 4004, 25, 151645, 198, 151644, 872, 198]
        expected_suffix = [151645, 198, 151644, 77091, 198]
        examples = {
            'a red fox in the snow': [64, 2518, 38835, 304, 279, 11794],
            '한글과 日本語, café 🌌': [23573, 83291, 53680, 75402, 21894, 102819, 11, 51950, 11162, 234, 234],
            '': [],
            '<|im_start|> literal <|im_end|>': [151644, 23141, 220, 151645],
        }
        self.assertEqual(tokenizer(prefix)['input_ids'], expected_prefix)
        for prompt, expected in examples.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(tokenizer(prefix + prompt + suffix)['input_ids'],
                                 expected_prefix + expected + expected_suffix)


if __name__ == '__main__':
    unittest.main()
