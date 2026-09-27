"""Offline numerical tests with real tiny Krea2 components, no pretrained downloads."""
import argparse
import importlib.util
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import krea2_contract as k

AVAILABLE = all(importlib.util.find_spec(v) for v in ('torch', 'diffusers', 'transformers'))

@unittest.skipUnless(AVAILABLE, 'Install reference runtime for real Krea2 numerical tests')
class Krea2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusers as d
        cls.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(42)
        transformer = d.Krea2Transformer2DModel(in_channels=64, num_layers=1, attention_head_dim=16,
            num_attention_heads=2, num_key_value_heads=2, intermediate_size=64, timestep_embed_dim=32,
            text_hidden_dim=32, num_text_layers=1, text_num_attention_heads=2, text_num_key_value_heads=2,
            text_intermediate_size=64, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
            axes_dims_rope=(4, 6, 6))
        cls.pipe = d.Krea2Pipeline(transformer=transformer,
            vae=d.AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1),
            scheduler=d.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=.5,
                max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400),
            text_encoder=None, tokenizer=None, text_encoder_select_layers=[0], is_distilled=True)
        cls.pipe.set_progress_bar_config(disable=True)
        cls.embeds = torch.randn(1, 16, 1, 32)

    def request(self, variant='turbo', mu=None):
        inputs = dict(width=64, height=64, num_inference_steps=2, prompt_embeds=self.embeds.clone(),
            prompt_embeds_mask=self.torch.ones(1, 16, dtype=self.torch.bool), output_type='latent',
            generator=self.torch.Generator('cpu').manual_seed(7))
        if variant == 'raw':
            inputs.update(negative_prompt_embeds=self.embeds * 0,
                          negative_prompt_embeds_mask=self.torch.ones(1, 16, dtype=self.torch.bool))
        args = argparse.Namespace(inputs=inputs, dtype='float32', krea2_variant=variant, krea2_mu=mu)
        return inputs, k.resolve(args, {'_class_name': 'Krea2Pipeline'})

    def run_pipeline(self, variant='turbo', mu=None):
        inputs, contract = self.request(variant, mu)
        with self.torch.no_grad(), k.execution(self.pipe, inputs, contract, self.torch) as trace:
            result = self.pipe(**inputs).images
        self.assertNotIn('set_timesteps', vars(self.pipe.scheduler))
        self.assertNotIn('callback_on_step_end', inputs)
        self.assertEqual(trace['executed_steps'], 2)
        self.assertTrue(self.torch.isfinite(result).all())
        return result, trace

    def test_seed_repeat_is_exact_and_mu_changes_real_schedule(self):
        first, trace = self.run_pipeline()
        second, repeated = self.run_pipeline()
        self.assertTrue(self.torch.equal(first, second))
        shifted, other = self.run_pipeline(mu=0)
        self.assertNotEqual(trace['sigmas'], other['sigmas'])
        self.assertFalse(self.torch.equal(first, shifted))

    def test_raw_guidance_executes_with_negative_conditioning(self):
        raw, trace = self.run_pipeline('raw')
        turbo, _ = self.run_pipeline()
        self.assertFalse(self.torch.equal(raw, turbo))
        self.assertEqual(trace['component_dtypes']['transformer'], 'torch.float32')

    def test_nonfinite_inputs_and_masks_fail_before_inference(self):
        for failure in ('nan', 'mask', 'latent'):
            inputs, contract = self.request()
            if failure == 'nan': inputs['prompt_embeds'][0, 0, 0, 0] = float('nan')
            elif failure == 'mask': inputs['prompt_embeds_mask'] = inputs['prompt_embeds_mask'].float()
            else: inputs['latents'] = self.torch.zeros(1, 16, 8, 8)
            with self.assertRaises(ValueError), k.execution(self.pipe, inputs, contract, self.torch):
                self.fail('invalid inputs entered inference')

    def test_step_nonfinite_rejected_and_hooks_restored_after_failure(self):
        inputs, contract = self.request()
        with self.assertRaisesRegex(RuntimeError, 'non-finite'), k.execution(self.pipe, inputs, contract, self.torch):
            inputs['callback_on_step_end'](self.pipe, 0, None, {'latents': self.torch.tensor([float('nan')])})
        self.assertNotIn('set_timesteps', vars(self.pipe.scheduler))
        self.assertNotIn('callback_on_step_end', inputs)

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'PEFT required for adapter numerical test')
    def test_krea_lora_scale_controls_real_transformer(self):
        from peft import LoraConfig
        from lora import validate_lora_support
        validate_lora_support(self.pipe)
        baseline, _ = self.run_pipeline()
        self.pipe.transformer.add_adapter(LoraConfig(r=2, lora_alpha=2,
            target_modules=['to_q'], init_lora_weights=False), adapter_name='krea_test')
        try:
            self.pipe.set_adapters('krea_test', adapter_weights=0.0)
            zero, _ = self.run_pipeline()
            self.pipe.set_adapters('krea_test', adapter_weights=0.25)
            quarter, _ = self.run_pipeline()
            self.assertTrue(self.torch.equal(baseline, zero))
            self.assertFalse(self.torch.equal(zero, quarter))
        finally:
            self.pipe.unload_lora_weights()

    def test_incompatible_vae_and_scheduler_refused(self):
        _, contract = self.request()
        old = self.pipe.vae
        try:
            self.pipe.vae = self.torch.nn.Identity()
            with self.assertRaisesRegex(ValueError, 'VAE'): k.validate_pipeline(self.pipe, contract)
        finally: self.pipe.vae = old
        config = self.pipe.scheduler.config
        self.pipe.scheduler.register_to_config(use_dynamic_shifting=False)
        try:
            with self.assertRaisesRegex(ValueError, 'dynamic'): k.validate_pipeline(self.pipe, contract)
        finally: self.pipe.scheduler.register_to_config(**dict(config))

if __name__ == '__main__': unittest.main()
