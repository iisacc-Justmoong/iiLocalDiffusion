"""Tiny real SD3 ControlNet component loading and conditioning."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import generic_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical SD3 control tests')
class SD3ControlRuntimeTests(unittest.TestCase):
    def test_control_vae_uses_a_separate_stream_from_generation_noise(self):
        import torch
        class Vae:
            def encode(self, value):
                return SimpleNamespace(latent_dist=SimpleNamespace(
                    sample=lambda generator: torch.randn((1, 16, 8, 8), generator=generator)))
        vae = Vae()
        selection = SimpleNamespace(family='sd3')
        def control_sample():
            with control.sd3_control_seed(SimpleNamespace(vae=vae), selection, 17, torch):
                return vae.encode(None).latent_dist.sample()
        state = torch.random.get_rng_state().clone()
        first = control_sample()
        self.assertTrue(torch.equal(first, control_sample()))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        generation = torch.randn((1, 16, 8, 8), generator=torch.Generator('cpu').manual_seed(17))
        self.assertFalse(torch.equal(first, generation))
        self.assertNotIn('encode', vars(vae))

    def test_prepared_control_changes_output_and_replays_independently_of_global_rng(self):
        import torch
        import diffusers as d
        from PIL import Image
        import resource_fixture
        torch.manual_seed(19)
        torch.set_num_threads(2)
        config = dict(sample_size=8, patch_size=2, in_channels=16, attention_head_dim=16,
                      num_attention_heads=2, joint_attention_dim=32, caption_projection_dim=32,
                      pooled_projection_dim=16, pos_embed_max_size=8)
        transformer = d.SD3Transformer2DModel(**config, num_layers=2).eval()
        net = d.SD3ControlNetModel(**config, num_layers=1).eval()
        with torch.no_grad():
            for layer in (net.pos_embed_input.proj, *net.controlnet_blocks):
                torch.nn.init.constant_(layer.weight, .01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(8, 8, 8, 8),
            norm_num_groups=4, layers_per_block=1, latent_channels=16,
            sample_size=64, scaling_factor=1.5, shift_factor=.06).eval()
        pipeline = d.StableDiffusion3Pipeline(transformer=transformer, vae=vae,
            scheduler=d.FlowMatchEulerDiscreteScheduler(), text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None, text_encoder_3=None, tokenizer_3=None)
        with resource_fixture.TemporaryDirectory(prefix='sd3-control-') as temporary:
            folder = Path(temporary)
            net.save_pretrained(folder, safe_serialization=True)
            selected = control.Selection(str(folder), 'StableDiffusion3Pipeline', 'StableDiffusion3ControlNetPipeline',
                                         'SD3ControlNetModel', 'sd3', 'control_image', None)
            wrapped = control.attach(pipeline, selected, SimpleNamespace(control_mode=None), d, torch.float32)
            wrapped.set_progress_bar_config(disable=True)
            prompt, pooled = torch.randn(1, 4, 32), torch.randn(1, 16)
            def run(color, global_seed=31):
                torch.manual_seed(global_seed)
                with torch.no_grad(), control.sd3_control_seed(pipeline=wrapped, selection=selected, seed=17, torch=torch):
                    return wrapped(prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                        control_image=Image.new('RGB', (64, 64), (color,) * 3), width=64, height=64,
                        num_inference_steps=2, guidance_scale=1.,
                        generator=torch.Generator('cpu').manual_seed(17), output_type='latent').images
            first = run(40)
            self.assertTrue(torch.equal(first, run(40)))
            self.assertFalse(torch.equal(first, run(210)))
            self.assertTrue(torch.equal(first, run(40, global_seed=32)))


if __name__ == '__main__':
    unittest.main()
