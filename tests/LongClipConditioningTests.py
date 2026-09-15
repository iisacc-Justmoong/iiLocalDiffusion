#!/usr/bin/env python3
"""CLIP chunk boundaries preserve every token, including a 75-vector embedding."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/diffusers"))
from long_clip_conditioning import token_chunks, token_attention_masks, long_clip_prompt_context


class LongClipConditioningTests(unittest.TestCase):
    tokenizer = SimpleNamespace(model_max_length=77, bos_token_id=1, eos_token_id=2, pad_token_id=0)

    def test_all_128_default_vectors_survive_chunk_boundaries(self):
        ids = list(range(1000, 1128))
        chunks = token_chunks(ids, self.tokenizer, 2)
        self.assertEqual([len(row) for row in chunks], [77, 77])
        restored = [value for row in chunks for value in row if value >= 1000]
        self.assertEqual(restored, ids)
        self.assertEqual(chunks[1][53], 1127)

    def test_short_positive_is_padded_to_the_same_chunk_count(self):
        first, second = token_chunks([100], self.tokenizer, 2)
        self.assertEqual(first[:3], [1, 100, 2])
        self.assertEqual(second[:2], [1, 2])
        self.assertEqual(len(second), 77)

    def test_eos_remains_visible_when_eos_and_padding_share_an_id(self):
        tokenizer = SimpleNamespace(model_max_length=77, bos_token_id=1, eos_token_id=2, pad_token_id=2)
        chunks = token_chunks([100], tokenizer, 2)
        masks = token_attention_masks([100], tokenizer, 2)
        self.assertEqual(chunks[0][2:4], [2, 2])
        self.assertEqual(masks[0][2:4], [1, 0])
        self.assertEqual(sum(masks[1]), 2)

    def test_context_restores_the_original_method_on_failure(self):
        def original(prompt=None):
            return prompt
        pipeline = SimpleNamespace(encode_prompt=original)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with long_clip_prompt_context(pipeline, SimpleNamespace(family="sd15")):
                self.assertIsNot(pipeline.encode_prompt, original)
                raise RuntimeError("stop")
        self.assertIs(pipeline.encode_prompt, original)


if __name__ == "__main__":
    unittest.main()
