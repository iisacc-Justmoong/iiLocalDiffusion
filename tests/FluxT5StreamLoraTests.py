"""Guard streamed T5 from LoRA formats that can target unsupported modules."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import flux_t5_stream
from lora import resolve_lora_selection
import resource_fixture


class FluxT5StreamLoraTests(unittest.TestCase):
    def test_only_explicit_transformer_and_clip_keys_are_accepted(self):
        with resource_fixture.TemporaryDirectory(prefix='flux-stream-lora-') as temporary:
            path = Path(temporary) / 'adapter.safetensors'
            def write(keys):
                header = json.dumps({key: {} for key in keys}).encode()
                path.write_bytes(struct.pack('<Q', len(header)) + header)
                return resolve_lora_selection(SimpleNamespace(lora=str(path), lora_revision=None,
                                                               lora_weight_name=None, lora_scale=1.0))
            transformer = 'transformer.transformer_blocks.0.attn.to_q.lora_A.weight'
            clip = 'text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight'
            flux_t5_stream.validate_lora(write([transformer, clip]))
            for keys in ([transformer, 'text_encoder_2.encoder.block.0.q.lora_A.weight'],
                         ['lora_unet_double_blocks_0_q.lora_down.weight'], []):
                with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, 'T5 and ambiguous'):
                    flux_t5_stream.validate_lora(write(keys))
            selection = write([transformer])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                flux_t5_stream.validate_lora(selection)

    def test_pipeline_must_not_advertise_t5_as_a_lora_target(self):
        flux_t5_stream.validate_lora_pipeline(SimpleNamespace(_lora_loadable_modules=['transformer', 'text_encoder']))
        for modules in (None, [], ['transformer', 'text_encoder_2'], ['transformer', 'unknown']):
            with self.subTest(modules=modules), self.assertRaisesRegex(ValueError, 'may target'):
                flux_t5_stream.validate_lora_pipeline(SimpleNamespace(_lora_loadable_modules=modules))


if __name__ == '__main__':
    unittest.main()
