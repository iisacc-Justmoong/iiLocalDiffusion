"""Offline numerical tests with real tiny Krea2 components, no pretrained downloads."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import krea2_contract as k
import krea2_controlnet as control
import krea2_color_control as color_control

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

    def test_published_depth_header_matches_real_meta_transformer(self):
        import diffusers as d
        torch = self.torch
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'resources/test-fixtures/krea2-depth-header.json').read_text())
        shapes = {name: tuple(item['shape']) for name, item in fixture['header'].items() if name != '__metadata__'}
        with torch.device('meta'):
            model = d.Krea2Transformer2DModel()
        modules = dict(model.named_modules())
        targets = {name: tuple(module.weight.shape) for name, module in modules.items()
                   if getattr(module, 'weight', None) is not None and module.weight.ndim == 2}
        plan = control.plan_weights(shapes, targets, projection_shape=tuple(model.img_in.weight.shape),
                                    block_count=len(model.transformer_blocks))
        self.assertEqual(len(plan), 224)

    def test_published_ostris_pose_header_matches_real_meta_transformer(self):
        import diffusers as d
        import krea2_ostris_lora
        torch = self.torch
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'resources/test-fixtures/krea2-ostris-pose-header.json').read_text())
        shapes = {name: tuple(item['shape']) for name, item in fixture['header'].items() if name != '__metadata__'}
        with torch.device('meta'):
            model = d.Krea2Transformer2DModel()
        modules = {name: tuple(module.weight.shape) for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)}
        plan = krea2_ostris_lora.plan_weights(shapes, modules)
        self.assertEqual(len(plan), 256)

    def test_color_depth_card_contract_matches_full_meta_transformer(self):
        """Validate published layer count/shapes; this does not inspect gated weights."""
        import diffusers as d
        torch = self.torch
        with torch.device('meta'):
            model = d.Krea2Transformer2DModel()
        modules = {name: tuple(module.weight.shape) for name, module in model.named_modules()
                   if name != 'img_in' and isinstance(module, torch.nn.Linear)}
        self.assertEqual(tuple(model.img_in.weight.shape), (6144, 64))
        self.assertEqual(len(modules), 263)
        shapes = {'img_in.weight': (6144, 192), 'img_in.bias': (6144,)}
        for name, (outputs, inputs) in modules.items():
            shapes[name + '.lora_A.weight'] = (64, inputs)
            shapes[name + '.lora_B.weight'] = (outputs, 64)
        self.assertEqual(len(color_control.plan_weights(shapes, modules, tuple(model.img_in.weight.shape))), 263)

    def test_ostris_lora_changes_transformer_and_restores_on_failure(self):
        from safetensors.torch import save_file
        from weight_files import resolve_weight_file
        import resource_fixture
        import krea2_ostris_lora
        torch = self.torch
        model = self.pipe.transformer
        modules = dict(model.named_modules())
        weights = {}
        for source in ('diffusion_model.blocks.0.attn.wq',
                       'diffusion_model.txtfusion.refiner_blocks.0.mlp.down'):
            target = krea2_ostris_lora.target_name(source)
            shape = modules[target].weight.shape
            weights[source + '.lora_A.weight'] = torch.ones(2, shape[1]) * .03
            weights[source + '.lora_B.weight'] = torch.ones(shape[0], 2) * .03
        with resource_fixture.TemporaryDirectory(prefix='krea-ostris-lora-') as temporary:
            path = Path(temporary) / 'pose.safetensors'
            save_file(weights, str(path))
            selected = resolve_weight_file(str(path), 'pose')
            baseline, _ = self.run_pipeline()
            def run():
                inputs, contract = self.request()
                with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch), \
                     krea2_ostris_lora.execution(model, selected, 1., torch) as trace:
                    result = self.pipe(**inputs).images
                self.assertEqual(trace['targets'], 2)
                return result
            first = run()
            self.assertTrue(torch.equal(first, run()))
            self.assertFalse(torch.equal(first, baseline))
            with self.assertRaisesRegex(RuntimeError, 'forced failure'), \
                 krea2_ostris_lora.execution(model, selected, 1., torch):
                raise RuntimeError('forced failure')
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
            restored, _ = self.run_pipeline()
            self.assertTrue(torch.equal(restored, baseline))

    def test_ostris_reference_tokens_use_zero_timestep_and_restore_cfg(self):
        import krea2_ostris_reference
        torch = self.torch
        model = self.pipe.transformer
        reference = torch.randn(1, 4, 64, generator=torch.Generator('cpu').manual_seed(81))
        positions = torch.tensor([[1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
        def run(ref, variant, capture=False):
            inputs, contract = self.request(variant=variant)
            outputs, image_lengths = [], []
            with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch), \
                 krea2_ostris_reference.execution(self.pipe, ref, positions, variant == 'raw', torch) as trace:
                handle = model.time_mod_proj.register_forward_hook(lambda module, args, output: outputs.append(output.clone())) if capture else None
                image_handle = model.img_in.register_forward_hook(lambda module, args, output: image_lengths.append(args[0].shape[1]))
                try:
                    result = self.pipe(**inputs).images
                finally:
                    if handle is not None: handle.remove()
                    image_handle.remove()
            self.assertEqual(trace['forward_activity'], {'conditional': 2, 'unconditional': 2 if variant == 'raw' else 0})
            self.assertEqual(image_lengths, [20] * (4 if variant == 'raw' else 2))
            self.assertNotIn('step', vars(self.pipe.scheduler))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in model.modules()))
            return result, outputs
        for variant in ('turbo', 'raw'):
            baseline, _ = self.run_pipeline(variant=variant)
            first, outputs = run(reference, variant, capture=True)
            self.assertTrue(torch.equal(first, run(reference, variant)[0]))
            self.assertFalse(torch.equal(first, run(reference * 2, variant)[0]))
            self.assertFalse(torch.equal(first, baseline))
            self.assertTrue(torch.equal(baseline, self.run_pipeline(variant=variant)[0]))
            expanded = next(value for value in outputs if value.shape[1] > 1)
            normal_length = 16 + 16  # text tokens + target image tokens
            with torch.no_grad():
                zero = model.time_embed(torch.zeros(1), dtype=torch.float32)
                zero = model.time_mod_proj(torch.nn.functional.gelu(zero, approximate='tanh'))
            torch.testing.assert_close(expanded[:, normal_length:], zero.expand(-1, 4, -1), rtol=0, atol=0)
            self.assertFalse(torch.equal(expanded[:, :1], expanded[:, normal_length:normal_length + 1]))

    def test_ostris_isolated_reference_is_independent_of_target_tokens(self):
        import copy
        import diffusers as d
        import krea2_ostris_reference
        torch = self.torch
        model = copy.deepcopy(self.pipe.transformer)
        model.transformer_blocks.append(copy.deepcopy(model.transformer_blocks[0]))
        pipe = d.Krea2Pipeline(transformer=model, vae=self.pipe.vae,
            scheduler=d.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
            text_encoder=None, tokenizer=None, text_encoder_select_layers=[0], is_distilled=True)
        reference = torch.randn(1, 4, 64, generator=torch.Generator('cpu').manual_seed(81))
        positions = torch.tensor([[1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
        ids = pipe.prepare_position_ids(16, 4, 4, torch.device('cpu'))
        text = self.embeds.clone()
        target = torch.randn(1, 16, 64, generator=torch.Generator('cpu').manual_seed(18))
        def ref_output(value, mode):
            captured = []
            with torch.no_grad(), krea2_ostris_reference.execution(pipe, reference, positions, False, torch,
                                                                  attention=mode):
                hook = model.transformer_blocks[-1].register_forward_hook(lambda module, args, output: captured.append(output[:, -4:].clone()))
                try:
                    model(hidden_states=value, encoder_hidden_states=text, timestep=torch.tensor([.5]),
                          position_ids=ids, encoder_attention_mask=torch.ones(1, 16, dtype=torch.bool))
                finally:
                    hook.remove()
            return captured[0]
        isolated = ref_output(target, 'isolated')
        torch.testing.assert_close(isolated, ref_output(target * 2, 'isolated'), rtol=0, atol=0)
        self.assertFalse(torch.equal(ref_output(target, 'joint'), ref_output(target * 2, 'joint')))
        with torch.no_grad():
            ref_only = model.img_in(reference)
            time = model.time_embed(torch.zeros(1), dtype=reference.dtype)
            modulation = model.time_mod_proj(torch.nn.functional.gelu(time, approximate='tanh'))
            rope = model.rotary_emb(positions)
            for block in model.transformer_blocks:
                ref_only = block(ref_only, modulation, rope)
        torch.testing.assert_close(isolated, ref_only, rtol=1e-5, atol=1e-6)

    def test_ostris_control_composes_visual_encoder_reference_and_lora(self):
        import diffusers as d
        import numpy as np
        from PIL import Image
        from safetensors.torch import save_file
        from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel
        from weight_files import resolve_weight_file
        import krea2_ostris_control
        import krea2_ostris_lora
        import resource_fixture
        torch = self.torch
        config = Qwen3VLConfig(image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
            text_config={'vocab_size': 151936, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'max_position_embeddings': 512,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                             'mrope_interleaved': True}},
            vision_config={'depth': 1, 'hidden_size': 16, 'num_heads': 2, 'intermediate_size': 32,
                           'out_hidden_size': 32, 'num_position_embeddings': 16, 'deepstack_visual_indexes': []})
        encoder = Qwen3VLModel(config).eval()
        root = Path(__file__).resolve().parents[1]
        tokenizer = Qwen2Tokenizer.from_pretrained(str(root / 'resources/text_encoders/qwen3vl-4b'), local_files_only=True)
        pipe = d.Krea2Pipeline(transformer=self.pipe.transformer, vae=self.pipe.vae,
            scheduler=d.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=.5,
                max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400),
            text_encoder=encoder, tokenizer=tokenizer, text_encoder_select_layers=[0], is_distilled=True)
        pipe.set_progress_bar_config(disable=True)
        modules = dict(pipe.transformer.named_modules())
        source = 'diffusion_model.blocks.0.attn.wq'
        shape = modules[krea2_ostris_lora.target_name(source)].weight.shape
        weights = {source + '.lora_A.weight': torch.ones(2, shape[1]) * .03,
                   source + '.lora_B.weight': torch.ones(shape[0], 2) * .03}
        with resource_fixture.TemporaryDirectory(prefix='krea-ostris-integration-') as temporary:
            path = Path(temporary) / 'pose.safetensors'
            save_file(weights, str(path))
            weight = resolve_weight_file(str(path), 'pose')
            def run(color, variant, scale=1.):
                inputs, contract = self.request(variant=variant)
                for key in ('prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask'):
                    inputs.pop(key, None)
                inputs['prompt'] = 'a dancer'
                if variant == 'raw': inputs['negative_prompt'] = 'blurred'
                selection = krea2_ostris_control.Selection(weight, 'fixture', scale)
                image = Image.new('RGB', (64, 64), color)
                with torch.no_grad(), k.execution(pipe, inputs, contract, torch), \
                     krea2_ostris_control.execution(pipe, image, inputs, selection, torch, np) as trace:
                    result = pipe(**inputs).images
                self.assertEqual(trace['reference']['reference_tokens'], 16)
                self.assertEqual(trace['reference']['forward_activity'], {'conditional': 2,
                    'unconditional': 2 if variant == 'raw' else 0})
                self.assertEqual(inputs['prompt'], 'a dancer')
                self.assertNotIn('step', vars(pipe.scheduler))
                self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in pipe.transformer.modules()))
                return result
            for variant in ('turbo', 'raw'):
                first = run('black', variant)
                self.assertTrue(torch.equal(first, run('black', variant)))
                self.assertFalse(torch.equal(first, run('white', variant)))
                self.assertFalse(torch.equal(first, run('black', variant, scale=0.)))

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

    def test_control_lora_changes_real_transformer_and_restores_hooks(self):
        from PIL import Image
        import numpy as np
        from safetensors.torch import save_file
        import resource_fixture
        torch = self.torch
        model = self.pipe.transformer
        weights = {'first.weight': torch.cat((model.img_in.weight.detach().clone(),
                                             torch.ones_like(model.img_in.weight) * .05), dim=1),
                   'first.bias': model.img_in.bias.detach().clone()}
        modules = dict(model.named_modules())
        for source, target in control.BLOCK_TARGETS.items():
            weight = modules['transformer_blocks.0.' + target].weight
            weights['blocks.0.' + source + '.A'] = torch.ones(2, weight.shape[1]) * .005
            weights['blocks.0.' + source + '.B'] = torch.ones(weight.shape[0], 2) * .005
        with resource_fixture.TemporaryDirectory(prefix='krea-control-') as temporary:
            path = Path(temporary) / 'control.safetensors'
            save_file(weights, str(path))
            from weight_files import resolve_weight_file
            selection = control.Selection(resolve_weight_file(str(path), 'control'), '', 1.0)
            image = Image.new('RGB', (64, 64), (160, 160, 160))
            def run(hint=image):
                inputs, contract = self.request()
                with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch), \
                        control.execution(self.pipe, hint, inputs, selection, torch, np) as trace:
                    result = self.pipe(**inputs).images
                self.assertEqual(trace['blocks'], 1)
                self.assertEqual(trace['vae_input_hints']['depth']['size'], [64, 64])
                return result
            baseline, _ = self.run_pipeline()
            first = run()
            repeated = run()
            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(baseline, first))
            from native_canny_fixture import hints
            with self.subTest(preprocessor='native-canny'):
                blank, edges = hints()
                conditioned = run(edges)
                self.assertTrue(torch.equal(conditioned, run(edges)))
                self.assertFalse(torch.equal(conditioned, run(blank)))
            for precision in ('float32', 'float16'):
                with self.subTest(preprocessor='depth-anything', precision=precision):
                    from depth_hint_fixture import hints as depth_hints
                    first_hint, second_hint = depth_hints(preserve_float=True, precision=precision)
                    conditioned = run(first_hint)
                    self.assertTrue(torch.equal(conditioned, run(first_hint)))
                    self.assertFalse(torch.equal(conditioned, run(second_hint)))
            restored, _ = self.run_pipeline()
            self.assertTrue(torch.equal(baseline, restored))

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'PEFT required for adapter composition test')
    def test_depth_control_composes_with_style_lora_and_restores_pipeline(self):
        import numpy as np
        from PIL import Image
        from safetensors.torch import save_file
        from types import SimpleNamespace
        import resource_fixture
        from lora import apply_lora, resolve_lora_selection, validate_lora_activation
        from weight_files import resolve_weight_file

        torch = self.torch
        model = self.pipe.transformer
        modules = dict(model.named_modules())
        weights = {'first.weight': torch.cat((model.img_in.weight.detach().clone(),
                                             torch.ones_like(model.img_in.weight) * .05), dim=1),
                   'first.bias': model.img_in.bias.detach().clone()}
        for source, target in control.BLOCK_TARGETS.items():
            weight = modules['transformer_blocks.0.' + target].weight
            weights['blocks.0.' + source + '.A'] = torch.ones(2, weight.shape[1]) * .005
            weights['blocks.0.' + source + '.B'] = torch.ones(weight.shape[0], 2) * .005
        with resource_fixture.TemporaryDirectory(prefix='krea-style-control-') as temporary:
            root = Path(temporary)
            path = root / 'control.safetensors'
            save_file(weights, str(path))
            selection = control.Selection(resolve_weight_file(str(path), 'control'), '', 1.)
            hint = Image.new('RGB', (64, 64), (120, 120, 120))
            def run(controlled):
                inputs, contract = self.request()
                with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch):
                    if controlled:
                        with control.execution(self.pipe, hint, inputs, selection, torch, np):
                            return self.pipe(**inputs).images
                    return self.pipe(**inputs).images
            baseline, control_only = run(False), run(True)
            for targets in (('img_in',), ('transformer_blocks.0.attn.to_q',),
                            ('img_in', 'transformer_blocks.0.attn.to_q')):
                with self.subTest(style_targets=targets):
                    style = root / 'style.safetensors'
                    style_weights = {}
                    for target in targets:
                        layer = dict(model.named_modules())[target]
                        prefix = 'transformer.' + target
                        style_weights[prefix + '.lora_A.weight'] = torch.full((2, layer.weight.shape[1]), .15)
                        style_weights[prefix + '.lora_B.weight'] = torch.full((layer.weight.shape[0], 2), .15)
                    save_file(style_weights, str(style))
                    style_selection = resolve_lora_selection(SimpleNamespace(lora=str(style), lora_revision=None,
                        lora_weight_name=None, lora_scale=1.))
                    activation = apply_lora(self.pipe, style_selection, root / 'cache', True)
                    try:
                        style_only, both = run(False), run(True)
                        self.assertFalse(torch.equal(baseline, style_only))
                        self.assertFalse(torch.equal(control_only, both))
                        self.assertFalse(torch.equal(style_only, both))
                        self.assertTrue(torch.equal(both, run(True)))
                        validate_lora_activation(self.pipe, activation)
                        inputs, contract = self.request()
                        with self.assertRaisesRegex(RuntimeError, 'forced failure'), torch.no_grad(), \
                             k.execution(self.pipe, inputs, contract, torch), \
                             control.execution(self.pipe, hint, inputs, selection, torch, np):
                            raise RuntimeError('forced failure')
                        validate_lora_activation(self.pipe, activation)
                        self.assertTrue(torch.equal(style_only, run(False)))
                    finally:
                        self.pipe.unload_lora_weights()
                    self.assertTrue(torch.equal(baseline, run(False)))
                    self.assertTrue(torch.equal(control_only, run(True)))

    def test_color_depth_control_uses_real_vae_and_restores_transformer(self):
        from PIL import Image
        import numpy as np
        from safetensors.torch import save_file
        import resource_fixture
        from weight_files import resolve_weight_file
        torch = self.torch
        model = self.pipe.transformer
        weights = {'img_in.weight': torch.cat((model.img_in.weight.detach().clone(),
                    torch.ones_like(model.img_in.weight) * .05, torch.ones_like(model.img_in.weight) * .04), dim=1),
                   'img_in.bias': model.img_in.bias.detach().clone()}
        for name, module in model.named_modules():
            if name != 'img_in' and isinstance(module, torch.nn.Linear):
                weights[name + '.lora_A.weight'] = torch.ones(2, module.weight.shape[1]) * .005
                weights[name + '.lora_B.weight'] = torch.ones(module.weight.shape[0], 2) * .005
        with resource_fixture.TemporaryDirectory(prefix='krea-color-control-') as temporary:
            path = Path(temporary) / 'control.safetensors'
            save_file(weights, str(path))
            selection = color_control.Selection(resolve_weight_file(str(path), 'control'), None, None, 1., 0., 1., .5)
            color = Image.new('RGB', (64, 64), (180, 30, 50))
            depth = Image.new('RGB', (64, 64), (128, 128, 128))
            def run(color_image=color, depth_image=depth):
                inputs, contract = self.request()
                with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch), \
                     color_control.execution(self.pipe, {'color': color_image, 'depth': depth_image}, inputs, selection, torch, np) as trace:
                    output = self.pipe(**inputs).images
                self.assertEqual(trace['control_activity'], {'color_steps': [0] if color_image else [],
                                                             'depth_steps': [0, 1] if depth_image else []})
                self.assertEqual(set(trace['vae_input_hints']),
                                 ({'color'} if color_image else set()) | ({'depth'} if depth_image else set()))
                self.assertNotIn('step', vars(self.pipe.scheduler))
                return output
            baseline, _ = self.run_pipeline()
            first = run()
            self.assertTrue(torch.equal(first, run()))
            self.assertFalse(torch.equal(first, baseline))
            self.assertFalse(torch.equal(first, run(color_image=None)))
            self.assertFalse(torch.equal(first, run(depth_image=None)))
            restored, _ = self.run_pipeline()
            self.assertTrue(torch.equal(restored, baseline))

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'PEFT required for adapter composition test')
    def test_color_depth_control_composes_with_style_lora_and_restores_pipeline(self):
        import numpy as np
        from PIL import Image
        from safetensors.torch import save_file
        from types import SimpleNamespace
        import resource_fixture
        from lora import apply_lora, resolve_lora_selection, validate_lora_activation
        from weight_files import resolve_weight_file

        torch = self.torch
        model = self.pipe.transformer
        weights = {'img_in.weight': torch.cat((model.img_in.weight.detach().clone(),
                    torch.ones_like(model.img_in.weight) * .05, torch.ones_like(model.img_in.weight) * .04), dim=1),
                   'img_in.bias': model.img_in.bias.detach().clone()}
        for name, module in model.named_modules():
            if name != 'img_in' and isinstance(module, torch.nn.Linear):
                weights[name + '.lora_A.weight'] = torch.ones(2, module.weight.shape[1]) * .005
                weights[name + '.lora_B.weight'] = torch.ones(module.weight.shape[0], 2) * .005
        with resource_fixture.TemporaryDirectory(prefix='krea-color-style-') as temporary:
            root = Path(temporary)
            path = root / 'control.safetensors'
            save_file(weights, str(path))
            selection = color_control.Selection(resolve_weight_file(str(path), 'control'), None, None, 1., 0., 1., 1.)
            images = {'color': Image.new('RGB', (64, 64), (180, 30, 50)),
                      'depth': Image.new('RGB', (64, 64), (128, 128, 128))}
            def run(controlled):
                inputs, contract = self.request()
                with torch.no_grad(), k.execution(self.pipe, inputs, contract, torch):
                    if controlled:
                        with color_control.execution(self.pipe, images, inputs, selection, torch, np):
                            return self.pipe(**inputs).images
                    return self.pipe(**inputs).images
            baseline, control_only = run(False), run(True)
            for targets in (('img_in',), ('transformer_blocks.0.attn.to_q',),
                            ('img_in', 'transformer_blocks.0.attn.to_q')):
                with self.subTest(style_targets=targets):
                    style = root / 'style.safetensors'
                    style_weights = {}
                    for target in targets:
                        layer = dict(model.named_modules())[target]
                        prefix = 'transformer.' + target
                        style_weights[prefix + '.lora_A.weight'] = torch.full((2, layer.weight.shape[1]), .15)
                        style_weights[prefix + '.lora_B.weight'] = torch.full((layer.weight.shape[0], 2), .15)
                    save_file(style_weights, str(style))
                    style_selection = resolve_lora_selection(SimpleNamespace(lora=str(style), lora_revision=None,
                        lora_weight_name=None, lora_scale=1.))
                    activation = apply_lora(self.pipe, style_selection, root / 'cache', True)
                    try:
                        style_only, both = run(False), run(True)
                        self.assertFalse(torch.equal(baseline, style_only))
                        self.assertFalse(torch.equal(control_only, both))
                        self.assertFalse(torch.equal(style_only, both))
                        self.assertTrue(torch.equal(both, run(True)))
                        inputs, contract = self.request()
                        with self.assertRaisesRegex(RuntimeError, 'forced failure'), torch.no_grad(), \
                             k.execution(self.pipe, inputs, contract, torch), \
                             color_control.execution(self.pipe, images, inputs, selection, torch, np):
                            raise RuntimeError('forced failure')
                        validate_lora_activation(self.pipe, activation)
                        self.assertTrue(torch.equal(style_only, run(False)))
                    finally:
                        self.pipe.unload_lora_weights()
                    self.assertTrue(torch.equal(baseline, run(False)))
                    self.assertTrue(torch.equal(control_only, run(True)))

    def test_regions_with_color_depth_control_and_cfg_restore_pipeline(self):
        from PIL import Image
        import numpy as np
        from safetensors.torch import save_file
        from types import SimpleNamespace
        from unittest.mock import patch
        import resource_fixture
        import krea2_regions
        from weight_files import resolve_weight_file
        torch = self.torch
        model = self.pipe.transformer
        weights = {'img_in.weight': torch.cat((model.img_in.weight.detach().clone(),
                    torch.ones_like(model.img_in.weight) * .05, torch.ones_like(model.img_in.weight) * .04), dim=1),
                   'img_in.bias': model.img_in.bias.detach().clone()}
        for name, module in model.named_modules():
            if name != 'img_in' and isinstance(module, torch.nn.Linear):
                weights[name + '.lora_A.weight'] = torch.ones(2, module.weight.shape[1]) * .005
                weights[name + '.lora_B.weight'] = torch.ones(module.weight.shape[0], 2) * .005
        def encode(prompts, maximum, device):
            prompts = [prompts] if isinstance(prompts, str) else prompts
            length = 12
            values = torch.zeros(len(prompts), length, 1, 32, device=device)
            mask = torch.zeros(len(prompts), length, dtype=torch.bool, device=device)
            for i, prompt in enumerate(prompts):
                n, scale = {'scene': (2, .1), 'first': (2, .3), 'second': (3, .8), 'negative': (11, -.3)}[prompt]
                values[i, :n, 0] = torch.arange(32, device=device) * scale
                mask[i, :n] = True
            return values, mask
        with resource_fixture.TemporaryDirectory(prefix='krea-regions-runtime-') as temporary:
            root = Path(temporary)
            path = root / 'control.safetensors'
            save_file(weights, str(path))
            ctrl = color_control.Selection(resolve_weight_file(str(path), 'control'), None, None, 1., 0., 1., .5)
            image = Image.new('RGB', (64, 64), (255, 0, 0))
            image.paste((0, 255, 0), (32, 0, 64, 64))
            image.save(root / 'map.png')
            manifest = root / 'regions.json'
            manifest.write_text(json.dumps({'map': 'map.png', 'regions': [
                {'color': '#ff0000', 'prompt': 'first'}, {'color': '#00ff00', 'prompt': 'second'}],
                'isolate_until': .5}))
            request = SimpleNamespace(krea2_regions=str(manifest), controlnet=None, control_color_image=None,
                control_image=None, controlnet_scale=None, control_color_end=None, control_depth_end=None,
                controlnet_format=None, inputs={'prompt': 'scene'}, offload='none')
            region = krea2_regions.resolve(request, 'Krea2Pipeline')
            def run(mapped, variant, count=1):
                inputs, contract = self.request(variant=variant)
                for key in ('prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask'):
                    inputs.pop(key, None)
                inputs['prompt'] = 'scene'
                inputs['num_images_per_prompt'] = count
                if variant == 'raw': inputs['negative_prompt'] = 'negative'
                with torch.no_grad(), patch.object(self.pipe, 'get_text_hidden_states', side_effect=encode):
                    with k.execution(self.pipe, inputs, contract, torch), \
                         color_control.execution(self.pipe, {'depth': image, 'color': image}, inputs, ctrl, torch, np), \
                         krea2_regions.execution(self.pipe, inputs, region, mapped, torch, np) as trace:
                        result = self.pipe(**inputs).images
                self.assertEqual(inputs['prompt'], 'scene')
                self.assertNotIn('prompt_embeds', inputs)
                self.assertNotIn('step', vars(self.pipe.scheduler))
                self.assertFalse(any(module._forward_pre_hooks for module in model.modules()))
                self.assertEqual(trace['forward_activity'], {'positive_forwards': 2,
                    'negative_forwards': 2 if variant == 'raw' else 0})
                if variant == 'raw':
                    self.assertGreater(trace['negative_tokens'], sum(trace['token_counts']))
                return result
            for variant in ('turbo', 'raw'):
                first = run(image, variant)
                self.assertTrue(torch.equal(first, run(image, variant)))
                self.assertFalse(torch.equal(first, run(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), variant)))
            self.assertEqual(run(image, 'raw', count=2).shape[0], 2)
            latent = torch.randn(1, 16, 64, generator=torch.Generator('cpu').manual_seed(88))
            def isolated(change):
                inputs, _ = self.request()
                inputs.pop('prompt_embeds')
                inputs.pop('prompt_embeds_mask')
                inputs['prompt'] = 'scene'
                with torch.no_grad(), patch.object(self.pipe, 'get_text_hidden_states', side_effect=encode), \
                     krea2_regions.execution(self.pipe, inputs, region, image, torch, np):
                    embeds = inputs['prompt_embeds'].clone()
                    if change:
                        embeds[:, 4:7] += torch.arange(32).reshape(1, 1, 1, 32) * .3
                    ids = self.pipe.prepare_position_ids(embeds.shape[1], 4, 4, torch.device('cpu'))
                    return model(hidden_states=latent, encoder_hidden_states=embeds, timestep=torch.tensor([.5]),
                                 position_ids=ids, encoder_attention_mask=inputs['prompt_embeds_mask']).sample
            left, changed = isolated(False), isolated(True)
            own = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13])
            other = torch.tensor([2, 3, 6, 7, 10, 11, 14, 15])
            torch.testing.assert_close(left[:, own], changed[:, own], rtol=0, atol=1e-6)
            self.assertFalse(torch.allclose(left[:, other], changed[:, other]))
            inputs, _ = self.request()
            inputs.pop('prompt_embeds')
            inputs.pop('prompt_embeds_mask')
            inputs['prompt'] = 'scene'
            with self.assertRaisesRegex(RuntimeError, 'forced failure'), torch.no_grad(), \
                 patch.object(self.pipe, 'get_text_hidden_states', side_effect=encode), \
                 krea2_regions.execution(self.pipe, inputs, region, image, torch, np):
                raise RuntimeError('forced failure')
            self.assertEqual(inputs['prompt'], 'scene')
            self.assertNotIn('prompt_embeds', inputs)
            self.assertNotIn('step', vars(self.pipe.scheduler))
            self.assertFalse(any(module._forward_pre_hooks for module in model.modules()))

    def test_control_vae_normalization_uses_float32_and_preserves_generator(self):
        import hashlib
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        torch = self.torch
        sample = torch.tensor([[[[[.296875, -.625], [1.125, .125]]]]], dtype=torch.bfloat16)
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
                self.config = SimpleNamespace(latents_mean=[.1234567], latents_std=[1.234567])
                self.pixels = None
            def encode(self, pixels):
                self.pixels = pixels.clone()
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: sample))
        vae = Vae()
        pipe = SimpleNamespace(vae=vae, _pack_latents=lambda latent, *args: latent)
        image = Image.new('RGB', (16, 16), (65, 130, 195))
        generator = torch.Generator('cpu').manual_seed(9)
        state = generator.get_state().clone()
        identity = {}
        result = control._control_latents(pipe, image, {'width': 16, 'height': 16, 'generator': generator},
                                          torch, np, output_identity=identity)
        self.assertEqual(identity, {'encoding': 'rgb8-row-major', 'size': [16, 16],
                                    'sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        pixels = torch.tensor([65, 130, 195], dtype=torch.float32).div(127.5).sub(1).to(torch.bfloat16)
        torch.testing.assert_close(vae.pixels[0, :, 0, 0, 0], pixels, rtol=0, atol=0)
        expected = ((sample.float() - .1234567) / 1.234567).squeeze(2)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(generator.get_state(), state))

    def test_control_vae_noise_is_seeded_independently_of_generation_noise(self):
        import hashlib
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        torch = self.torch
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(latents_mean=[0.], latents_std=[1.])
            def encode(self, pixels):
                return SimpleNamespace(latent_dist=SimpleNamespace(
                    sample=lambda generator: torch.randn((1, 1, 1, 8, 8), generator=generator)))
        pipe = SimpleNamespace(vae=Vae(), _pack_latents=lambda latent, *args: latent)
        image = Image.new('RGB', (16, 16), 'black')
        generator = torch.Generator('cpu').manual_seed(19)
        state = generator.get_state().clone()
        global_state = torch.random.get_rng_state().clone()
        def encode():
            return control._control_latents(pipe, image, {'width': 16, 'height': 16,
                                            'generator': generator}, torch, np)
        first = encode()
        seed = int.from_bytes(hashlib.sha256(b'iild-krea2-control-vae-v1\0' + bytes(state.tolist())).digest()[:8], 'big')
        expected = torch.randn((1, 1, 8, 8), generator=torch.Generator('cpu').manual_seed(seed))
        self.assertTrue(torch.equal(first, expected), 'Existing Krea 2 v1 seeded outputs must remain replayable')
        self.assertTrue(torch.equal(first, encode()))
        self.assertTrue(torch.equal(generator.get_state(), state))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_state))
        noise = torch.randn((1, 1, 8, 8), generator=generator)
        self.assertFalse(torch.equal(first, noise))
        self.assertFalse(torch.equal(first, encode()))

    def test_existing_tokenizer_runs_loaded_krea_control_pipeline(self):
        """Exercise fallback tokenizer through the actual Diffusers loader and control hook."""
        import json
        import numpy as np
        import resource_fixture
        import diffusers as d
        import generate_any
        from PIL import Image
        from safetensors.torch import save_file
        from transformers import Qwen3VLConfig, Qwen3VLTextModel

        torch = self.torch
        config = Qwen3VLConfig(text_config={'vocab_size': 151936, 'hidden_size': 32,
            'intermediate_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'head_dim': 8, 'max_position_embeddings': 1024,
            'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                'mrope_interleaved': True}})
        text = Qwen3VLTextModel(config.text_config).eval()
        source = d.Krea2Pipeline(transformer=self.pipe.transformer, vae=self.pipe.vae,
            scheduler=self.pipe.scheduler, text_encoder=text, tokenizer=None,
            text_encoder_select_layers=[0], is_distilled=True)
        source.set_progress_bar_config(disable=True)
        transformer = source.transformer
        weights = {'first.weight': torch.cat((transformer.img_in.weight.detach().clone(),
                                             torch.ones_like(transformer.img_in.weight) * .05), dim=1),
                   'first.bias': transformer.img_in.bias.detach().clone()}
        modules = dict(transformer.named_modules())
        for origin, target in control.BLOCK_TARGETS.items():
            weight = modules['transformer_blocks.0.' + target].weight
            weights['blocks.0.' + origin + '.A'] = torch.ones(2, weight.shape[1]) * .005
            weights['blocks.0.' + origin + '.B'] = torch.ones(weight.shape[0], 2) * .005
        with resource_fixture.TemporaryDirectory(prefix='krea-token-control-') as temporary:
            root = Path(temporary)
            model = root / 'model'
            source.save_pretrained(str(model), safe_serialization=True)
            adapter, image_path = root / 'control.safetensors', root / 'hint.png'
            save_file(weights, str(adapter))
            Image.new('RGB', (64, 64), (160, 160, 160)).save(image_path)
            args = generate_any.resolve_arguments(generate_any.build_parser().parse_args([
                '--model', str(model), '--controlnet', str(adapter), '--control-image', str(image_path),
                '--seed', '7', '--prompt', 'red fox', '--width', '64', '--height', '64', '--steps', '2',
                '--pipeline-inputs', json.dumps({'max_sequence_length': 8, 'output_type': 'latent'}),
                '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BSC',
                                                           'representation_space': 'test-krea-packed'}}),
                '--no-default-modifiers', '--output-dir', str(root / 'output')]))
            self.assertEqual(args.tokenizer_status, 'fallback')
            pipeline, identity = generate_any.load_pipeline(args, d, torch.float32)
            pipeline.set_progress_bar_config(disable=True)
            inputs = dict(args.inputs, generator=torch.Generator('cpu').manual_seed(args.seed))
            with torch.no_grad(), k.execution(pipeline, inputs, args.krea2, torch), \
                    control.execution(pipeline, Image.open(image_path).convert('RGB'), inputs,
                                      args.generic_controlnet, torch, np) as trace:
                output = pipeline(**inputs).images
            self.assertEqual(trace['blocks'], 1)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(pipeline.tokenizer.convert_tokens_to_ids('<|im_start|>'), 151644)
            generate_any.verify_identity(identity['identity'])

            # Exercise the real CLI dispatch, native preprocessor, tokenizer
            # fallback, model load, control hook and output publication together.
            import contextlib
            import io
            import native_control_preprocessing
            from PIL import ImageDraw
            from safetensors.torch import load_file
            try:
                native_library = native_control_preprocessing.library_path()
            except ValueError as error:
                self.skipTest(f'Build the native Canny library for the full runner test: {error}')
            blank = Image.new('RGB', (64, 64), 'black')
            subject = blank.copy()
            ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
            blank_path, subject_path = root / 'blank.png', root / 'subject.png'
            blank.save(blank_path)
            subject.save(subject_path)
            def run_main(image, name, method='canny', depth_model=None):
                output = root / name
                preprocessing = (['--control-preprocess', 'canny', '--control-preprocessor-library', str(native_library)]
                                 if method == 'canny' else
                                 ['--control-preprocess', 'depth', '--control-depth-model', str(depth_model)])
                with contextlib.redirect_stdout(io.StringIO()):
                    status = generate_any.main([
                        '--model', str(model), '--controlnet', str(adapter), '--control-image', str(image),
                        *preprocessing,
                        '--device', 'cpu', '--seed', '7', '--prompt', 'red fox', '--width', '64', '--height', '64',
                        '--steps', '2', '--pipeline-inputs', json.dumps({'max_sequence_length': 8, 'output_type': 'latent'}),
                        '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BSC',
                            'representation_space': 'test-krea-packed'}}), '--no-default-modifiers',
                        '--cache-dir', str(root / 'cache'), '--output-dir', str(output)])
                self.assertEqual(status, 0)
                report = json.loads((output / 'generation.json').read_text())
                self.assertEqual(report['control_preprocessor']['name'],
                                 'native-canny' if method == 'canny' else 'depth-anything-relative')
                self.assertEqual(report['request']['tokenizer']['status'], 'fallback')
                self.assertEqual(report['control_execution']['blocks'], 1)
                return load_file(str(output / 'images.safetensors'))['images'], report
            empty_latent, empty_report = run_main(blank_path, 'blank-output')
            edge_latent, edge_report = run_main(subject_path, 'edge-output')
            repeat_latent, repeat_report = run_main(subject_path, 'repeat-output')
            self.assertFalse(torch.equal(empty_latent, edge_latent))
            self.assertTrue(torch.equal(edge_latent, repeat_latent))
            self.assertNotEqual(empty_report['control_preprocessor']['output_identity'],
                                edge_report['control_preprocessor']['output_identity'])
            self.assertEqual(edge_report['control_preprocessor']['output_identity'],
                             repeat_report['control_preprocessor']['output_identity'])
            from depth_hint_fixture import save_model
            depth_model = root / 'depth-model'
            save_model(depth_model)
            depth_blank, blank_depth_report = run_main(blank_path, 'depth-blank', 'depth', depth_model)
            depth_subject, subject_depth_report = run_main(subject_path, 'depth-subject', 'depth', depth_model)
            depth_repeat, _ = run_main(subject_path, 'depth-repeat', 'depth', depth_model)
            self.assertFalse(torch.equal(depth_blank, depth_subject))
            self.assertTrue(torch.equal(depth_subject, depth_repeat))
            self.assertEqual(subject_depth_report['control_preprocessor']['output'], 'relative-depth-float32')
            self.assertNotEqual(blank_depth_report['control_preprocessor']['output_identity'],
                                subject_depth_report['control_preprocessor']['output_identity'])
            # With no packaged VAE, exercise the actual catalogued Qwen VAE
            # through both preprocessing modes and the Control-LoRA path.
            bundled_vae = Path(__file__).resolve().parents[1] / 'resources/vae/qwen-image/vae/diffusion_pytorch_model.safetensors'
            if bundled_vae.is_file():
                (model / 'vae').rename(root / 'tiny-vae-backup')
                fallback_blank, fallback_report = run_main(blank_path, 'fallback-blank')
                fallback_edge, _ = run_main(subject_path, 'fallback-edge')
                fallback_repeat, _ = run_main(subject_path, 'fallback-repeat')
                fallback_depth_blank, _ = run_main(blank_path, 'fallback-depth-blank', 'depth', depth_model)
                fallback_depth_edge, _ = run_main(subject_path, 'fallback-depth-edge', 'depth', depth_model)
                fallback_depth_repeat, _ = run_main(subject_path, 'fallback-depth-repeat', 'depth', depth_model)
                self.assertEqual(fallback_report['request']['vae']['status'], 'fallback')
                self.assertEqual(Path(fallback_report['request']['vae']['selection']['weight']['path']), bundled_vae)
                self.assertFalse(torch.equal(fallback_blank, fallback_edge))
                self.assertTrue(torch.equal(fallback_edge, fallback_repeat))
                self.assertFalse(torch.equal(fallback_depth_blank, fallback_depth_edge))
                self.assertTrue(torch.equal(fallback_depth_edge, fallback_depth_repeat))

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

    def test_incompatible_schedule_modes_are_refused(self):
        _, contract = self.request()
        scheduler = self.pipe.scheduler
        original = dict(scheduler.config)
        for option, value in (('invert_sigmas', True), ('shift_terminal', 0.2),
                              ('use_karras_sigmas', True), ('use_exponential_sigmas', True),
                              ('use_beta_sigmas', True), ('stochastic_sampling', True)):
            with self.subTest(option=option):
                scheduler.register_to_config(**{option: value})
                try:
                    with self.assertRaisesRegex(ValueError, option):
                        k.validate_pipeline(self.pipe, contract)
                finally:
                    scheduler.register_to_config(**original)

    def test_default_schedule_matches_published_krea_formula(self):
        import math
        import numpy as np
        scheduler = self.pipe.scheduler
        for variant, width, height, steps in (('turbo', 1024, 1024, 8),
                                               ('raw', 896, 1152, 28),
                                               ('raw', 1024, 1024, 52)):
            with self.subTest(variant=variant, width=width, height=height, steps=steps):
                mu = k.default_mu(variant, width, height)
                reference_mu = (1.15 if variant == 'turbo' else
                                .5 + ((width // 16) * (height // 16) - 256) * (.65 / 6144))
                self.assertAlmostEqual(mu, reference_mu)
                scheduler.set_timesteps(steps, sigmas=np.linspace(1., 1. / steps, steps), mu=mu)
                ts = self.torch.linspace(1., 0., steps + 1)
                shift = math.exp(reference_mu)
                expected = shift / (shift + (1. / ts - 1.))
                self.torch.testing.assert_close(scheduler.sigmas.cpu(), expected, rtol=1e-6, atol=1e-7)

if __name__ == '__main__': unittest.main()
