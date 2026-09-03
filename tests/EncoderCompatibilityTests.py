#!/usr/bin/env python3
"""Scoped compatibility with flat and nested Transformers CLIP normalization."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from encoder_compatibility import clip_skip_compatibility
import presets


class EncoderCompatibilityTests(unittest.TestCase):
    def test_flat_clip_exposes_only_the_existing_norm_during_the_call(self):
        norm = Mock()
        encoder = SimpleNamespace(final_layer_norm=norm)
        with clip_skip_compatibility(SimpleNamespace(text_encoder=encoder), presets.SD15_PRESET, 1) as applied:
            self.assertTrue(applied)
            self.assertIs(encoder.text_model.final_layer_norm, norm)
        self.assertFalse(hasattr(encoder, "text_model"))

    def test_alias_is_removed_even_when_encoding_raises(self):
        encoder = SimpleNamespace(final_layer_norm=Mock())
        with self.assertRaisesRegex(RuntimeError, "encoding"):
            with clip_skip_compatibility(SimpleNamespace(text_encoder=encoder), presets.SD15_PRESET, 2):
                raise RuntimeError("encoding failed")
        self.assertFalse(hasattr(encoder, "text_model"))

    def test_nested_clip_and_other_families_are_not_modified(self):
        original = SimpleNamespace(final_layer_norm=Mock())
        encoder = SimpleNamespace(text_model=original)
        with clip_skip_compatibility(SimpleNamespace(text_encoder=encoder), presets.SD15_PRESET, 1) as applied:
            self.assertFalse(applied)
        self.assertIs(encoder.text_model, original)
        for preset, skip in ((presets.SD15_PRESET, None), (presets.SDXL_BASE_PRESET, 1)):
            with clip_skip_compatibility(SimpleNamespace(), preset, skip) as applied:
                self.assertFalse(applied)

    def test_unknown_norm_layout_fails_instead_of_skipping_normalization(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            with clip_skip_compatibility(SimpleNamespace(text_encoder=SimpleNamespace()), presets.SD15_PRESET, 1):
                pass


if __name__ == "__main__":
    unittest.main()
