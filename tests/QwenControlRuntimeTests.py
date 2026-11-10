"""Tiny real Qwen Image ControlNet composition and reproducible sampling."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import generic_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL'))


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical Qwen control tests')
class QwenControlRuntimeTests(unittest.TestCase):
    def test_prepared_control_changes_output_and_replays(self):
        import torch
        import diffusers as d
        from PIL import Image
        import resource_fixture
        torch.manual_seed(19)
        torch.set_num_threads(2)
        transformer = d.QwenImageTransformer2DModel(in_channels=64, out_channels=16, num_layers=2,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32, axes_dims_rope=(4, 6, 6)).eval()
        net = d.QwenImageControlNetModel.from_transformer(transformer, num_layers=1,
            attention_head_dim=16, num_attention_heads=2)
        with torch.no_grad():
            for layer in (net.controlnet_x_embedder, *net.controlnet_blocks):
                torch.nn.init.constant_(layer.weight, .01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        vae = d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1).eval()
        pipeline = d.QwenImagePipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(),
                                       vae=vae, text_encoder=None, tokenizer=None)
        pipeline.set_progress_bar_config(disable=True)
        with resource_fixture.TemporaryDirectory(prefix='qwen-control-') as temporary:
            folder = Path(temporary)
            net.save_pretrained(folder, safe_serialization=True)
            selection = control.Selection(str(folder), 'QwenImagePipeline', 'QwenImageControlNetPipeline',
                                          'QwenImageControlNetModel', 'qwen-image', 'control_image', None)
            wrapped = control.attach(pipeline, selection, SimpleNamespace(control_mode=None), d, torch.float32)
            wrapped.set_progress_bar_config(disable=True)
            prompt = torch.randn(1, 4, 32)
            mask = torch.ones(1, 4, dtype=torch.bool)
            def run(color):
                with torch.no_grad():
                    return wrapped(prompt_embeds=prompt, prompt_embeds_mask=mask, true_cfg_scale=1.,
                        control_image=Image.new('RGB', (64, 64), (color, color, color)),
                        width=64, height=64, num_inference_steps=2, generator=torch.Generator('cpu').manual_seed(17),
                        output_type='latent').images
            first, again, other = run(40), run(40), run(210)
            self.assertTrue(torch.equal(first, again))
            self.assertFalse(torch.equal(first, other))


if __name__ == '__main__':
    unittest.main()
