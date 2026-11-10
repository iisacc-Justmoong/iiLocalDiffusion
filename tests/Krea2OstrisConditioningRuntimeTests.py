"""Numerical contracts for Krea 2 AI Toolkit visual conditioning."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import krea2_ostris_conditioning as conditioning

AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'PIL', 'numpy'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime')
class Krea2OstrisConditioningRuntimeTests(unittest.TestCase):
    def test_reference_latents_use_qwen_mean_normalization_and_reference_positions(self):
        import numpy as np
        import torch
        import diffusers as d
        from PIL import Image
        torch.set_num_threads(2)
        torch.manual_seed(81)
        vae = d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1).eval()
        pipe = d.Krea2Pipeline(transformer=None, scheduler=None, vae=vae, tokenizer=None, text_encoder=None)
        image = Image.new('RGB', (64, 48), (20, 130, 220))
        with torch.no_grad():
            actual, positions = conditioning.reference_latents(pipe, image, torch, np)
            pixels = torch.tensor([20., 130., 220.]).view(1, 3, 1, 1, 1).expand(1, 3, 1, 48, 64).div(127.5).sub(1.)
            raw = vae.encode(pixels).latent_dist.mode()
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            expected = ((raw.float() - mean) / std).squeeze(2)
            expected = pipe._pack_latents(expected, 1, 16, 6, 8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (1, 12, 64))
        self.assertEqual(positions[0].tolist(), [1, 0, 0])
        self.assertEqual(positions[-1].tolist(), [1, 2, 3])

    def test_patch_preprocessing_preserves_constant_rgb_and_expected_grid(self):
        import numpy as np
        import torch
        from PIL import Image
        image = Image.new('RGB', (32, 32), (0, 128, 255))
        patches, grid = conditioning.vision_patches(image, torch, np)
        self.assertEqual(grid.tolist(), [[1, 4, 4]])
        self.assertEqual(tuple(patches.shape), (16, 1536))
        channels = patches.reshape(16, 3, 2, 16, 16)
        torch.testing.assert_close(channels[:, 0], torch.full_like(channels[:, 0], -1.))
        torch.testing.assert_close(channels[:, 1], torch.full_like(channels[:, 1], 128 / 127.5 - 1.))
        torch.testing.assert_close(channels[:, 2], torch.ones_like(channels[:, 2]))

    def test_patch_order_matches_two_by_two_spatial_merger_cells(self):
        import numpy as np
        import torch
        from PIL import Image
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:16, :16] = (255, 0, 0)
        pixels[:16, 16:32] = (0, 255, 0)
        pixels[16:32, :16] = (0, 0, 255)
        pixels[16:32, 16:32] = (255, 255, 255)
        patches, grid = conditioning.vision_patches(Image.fromarray(pixels), torch, np)
        self.assertEqual(grid.tolist(), [[1, 4, 4]])
        colors = patches.reshape(16, 3, 2, 16, 16)[:4, :, 0, 0, 0]
        torch.testing.assert_close(colors, torch.tensor([[1., -1., -1.], [-1., 1., -1.],
                                                         [-1., -1., 1.], [1., 1., 1.]]))

    def test_existing_tokenizer_and_tiny_qwen_encode_image_and_prompt(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel
        torch.set_num_threads(2)
        torch.manual_seed(71)
        tokenizer = Qwen2Tokenizer.from_pretrained(str(ROOT / 'resources/text_encoders/qwen3vl-4b'), local_files_only=True)
        config = Qwen3VLConfig(image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
            text_config={'vocab_size': 151936, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'max_position_embeddings': 256,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                             'mrope_interleaved': True}},
            vision_config={'depth': 1, 'hidden_size': 16, 'num_heads': 2, 'intermediate_size': 32,
                           'out_hidden_size': 32, 'num_position_embeddings': 16, 'deepstack_visual_indexes': []})
        model = Qwen3VLModel(config).eval()
        pipeline = SimpleNamespace(text_encoder=model, tokenizer=tokenizer, text_encoder_select_layers=(0, 1),
            prompt_template_encode_prefix='<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n',
            prompt_template_encode_suffix='<|im_end|>\n<|im_start|>assistant\n', prompt_template_encode_start_idx=34)
        with torch.no_grad():
            first, mask = conditioning.encode_prompt(pipeline, 'a person', Image.new('RGB', (32, 32), 'black'), torch, np)
            repeat, _ = conditioning.encode_prompt(pipeline, 'a person', Image.new('RGB', (32, 32), 'black'), torch, np)
            different, _ = conditioning.encode_prompt(pipeline, 'a person', Image.new('RGB', (32, 32), 'white'), torch, np)
        self.assertEqual(first.ndim, 4)
        self.assertEqual(tuple(first.shape[:2]), tuple(mask.shape))
        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, different))
        self.assertTrue(bool(torch.isfinite(first).all()))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            conditioning.encode_prompt(pipeline, '<|image_pad|>', Image.new('RGB', (32, 32)), torch, np)


if __name__ == '__main__':
    unittest.main()
