"""Krea 2 variant, precision and sampling contracts (without large weights)."""
import argparse
import math
import json
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import krea2_contract as k
import krea2_controlnet as kc
import krea2_color_control as color_control
import krea2_single_file as single_file


class Krea2ContractTests(unittest.TestCase):
    def test_fp8_matmul_policy_counts_explicit_and_default_flags(self):
        shapes = {f'{name}.{suffix}': shape for name in ('first', 'second', 'third')
                  for suffix, shape in (('weight', (2, 2)), ('weight_scale', ()))}
        layers = {'first': {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True},
                  'second': {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': False},
                  'third': {'format': 'float8_e4m3fn'}}
        metadata = {'_quantization_metadata': json.dumps({'layers': layers})}
        policy = {}
        self.assertEqual(len(single_file.quantized_layers(shapes, metadata, matmul_policy=policy)), 3)
        self.assertEqual(policy, {'required': 1, 'not_required': 2})
        layers['second']['full_precision_matrix_mult'] = 1
        with self.assertRaisesRegex(ValueError, 'unsupported layer format'):
            single_file.quantized_layers(shapes, {'_quantization_metadata': json.dumps({'layers': layers})})

    def test_bundled_single_file_configs_match_krea2_sampling_contract(self):
        import generate_any
        root = Path(__file__).resolve().parents[1] / 'resources/model-configs'
        transformer_configs = []
        scheduler_configs = []
        for variant in ('raw', 'turbo'):
            folder = root / ('krea2-' + variant)
            index = generate_any.validate_model_index(json.loads((folder / 'model_index.json').read_text()))
            self.assertEqual(index['is_distilled'], variant == 'turbo')
            self.assertEqual(index['text_encoder_select_layers'], list(range(2, 36, 3)))
            self.assertEqual(index['transformer'], ['diffusers', 'Krea2Transformer2DModel'])
            self.assertEqual(index['text_encoder'], ['transformers', 'Qwen3VLModel'])
            self.assertEqual(index['vae'], ['diffusers', 'AutoencoderKLQwenImage'])
            transformer = json.loads((folder / 'transformer/config.json').read_text())
            scheduler = json.loads((folder / 'scheduler/scheduler_config.json').read_text())
            self.assertEqual((transformer['in_channels'], transformer['num_layers'],
                              transformer['text_hidden_dim'], transformer['num_text_layers']), (64, 28, 2560, 12))
            self.assertEqual((scheduler['use_dynamic_shifting'], scheduler['time_shift_type'],
                              scheduler['base_shift'], scheduler['max_shift'],
                              scheduler['base_image_seq_len'], scheduler['max_image_seq_len']),
                             (True, 'exponential', .5, 1.15, 256, 6400))
            args, _ = self.request(dtype='float32')
            self.assertEqual(k.resolve(args, index)['variant'], variant)
            transformer_configs.append(transformer)
            scheduler_configs.append(scheduler)
        self.assertEqual(transformer_configs[0], transformer_configs[1])
        self.assertEqual(scheduler_configs[0], scheduler_configs[1])

    def test_published_krea_depth_header_matches_contract(self):
        root = Path(__file__).resolve().parents[1]
        fixture = json.loads((root / 'resources/test-fixtures/krea2-depth-header.json').read_text())
        digest = hashlib.sha256(json.dumps(fixture['header'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '392acfac4c0f33db1201884a919cb8d72b710d2d66d86b705988ef173281bd8d')
        shapes = {name: value['shape'] for name, value in fixture['header'].items() if name != '__metadata__'}
        dimensions = {'attn.wq': (6144, 6144), 'attn.wk': (1536, 6144),
                      'attn.wv': (1536, 6144), 'attn.wo': (6144, 6144),
                      'attn.gate': (6144, 6144), 'mlp.gate': (16384, 6144),
                      'mlp.up': (16384, 6144), 'mlp.down': (6144, 16384)}
        modules = {f'transformer_blocks.{i}.{target}': dimensions[source]
                   for i in range(28) for source, target in kc.BLOCK_TARGETS.items()}
        self.assertEqual(len(kc.plan_weights(shapes, modules, projection_shape=(6144, 64), block_count=28)), 224)
        self.assertEqual(fixture['weight_sha256'], 'fb80547ed79b47c1e3fea7bb9d36297e3917b2115fab6700ca1501350f9f483c')

    def test_color_depth_adapter_requires_complete_matching_diffusers_targets(self):
        modules = {'transformer_blocks.0.attn.to_q': (8, 8), 'text_fusion.projector': (1, 3)}
        shapes = {'img_in.weight': (8, 12), 'img_in.bias': (8,)}
        for name, (out_features, in_features) in modules.items():
            shapes[name + '.lora_A.weight'] = (2, in_features)
            shapes[name + '.lora_B.weight'] = (out_features, 2)
        self.assertEqual(set(color_control.plan_weights(shapes, modules, (8, 4))), set(modules))
        for problem in ('missing', 'extra', 'projection', 'rank', 'target'):
            supplied, targets = dict(shapes), dict(modules)
            if problem == 'missing': supplied.pop('text_fusion.projector.lora_B.weight')
            if problem == 'extra': supplied['unexpected'] = (1,)
            if problem == 'projection': supplied['img_in.weight'] = (8, 8)
            if problem == 'rank': supplied['text_fusion.projector.lora_B.weight'] = (1, 3)
            if problem == 'target': targets.pop('text_fusion.projector')
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                color_control.plan_weights(supplied, targets, (8, 4))

    def test_color_depth_published_target_count_in_diffusers_layout(self):
        targets = tuple(kc.BLOCK_TARGETS.values())
        names = [f'transformer_blocks.{i}.{name}' for i in range(28) for name in targets]
        names += [f'text_fusion.{group}.{i}.{name}' for group in ('layerwise_blocks', 'refiner_blocks')
                  for i in range(2) for name in targets]
        names += ['time_embed.linear_1', 'time_embed.linear_2', 'time_mod_proj',
                  'text_fusion.projector', 'txt_in.linear_1', 'txt_in.linear_2', 'final_layer.linear']
        modules = {name: ((1, 12) if name == 'text_fusion.projector' else (8, 8)) for name in names}
        shapes = {'img_in.weight': (8, 12), 'img_in.bias': (8,)}
        for name, (outputs, inputs) in modules.items():
            shapes[name + '.lora_A.weight'] = (64, inputs)
            shapes[name + '.lora_B.weight'] = (outputs, 64)
        self.assertEqual(len(color_control.plan_weights(shapes, modules, (8, 4))), 263)
    def test_published_control_resolution_buckets_follow_input_aspect(self):
        for size, expected in (((900, 1600), (768, 1344)), ((1600, 900), (1344, 768)),
                               ((500, 500), (1024, 1024)), ((100, 1000), (704, 1472))):
            with self.subTest(size=size):
                self.assertEqual(kc.pick_bucket(*size), expected)
        with self.assertRaises(ValueError):
            kc.pick_bucket(0, 100)

    def test_control_bucket_updates_inputs_and_sampling_contract(self):
        for override in (None, 0.75):
            args = argparse.Namespace(inputs={'width': 1024, 'height': 1024},
                                      krea2={'variant': 'raw', 'mu_override': override, 'resolved_mu': None})
            inputs = dict(args.inputs)
            kc.configure_resolution(args, inputs, argparse.Namespace(size=(900, 1600)))
            self.assertEqual((inputs['width'], inputs['height']), (768, 1344))
            self.assertEqual(args.inputs, inputs)
            self.assertEqual(args.krea2['resolved_mu'], override if override is not None else k.default_mu('raw', 768, 1344))
            self.assertEqual(args.krea2['control_resolution']['output_size'], [768, 1344])

    def control_shapes(self):
        weights = {'first.weight': (8, 8), 'first.bias': (8,)}
        modules = {}
        for source, target in kc.BLOCK_TARGETS.items():
            modules[f'transformer_blocks.0.{target}'] = (8, 8)
            weights[f'blocks.0.{source}.A'] = (2, 8)
            weights[f'blocks.0.{source}.B'] = (8, 2)
        return weights, modules

    def test_published_control_weights_map_exactly_to_diffusers_modules(self):
        weights, modules = self.control_shapes()
        plan = kc.plan_weights(weights, modules, projection_shape=(8, 4), block_count=1)
        self.assertEqual(len(plan), len(kc.BLOCK_TARGETS))
        self.assertEqual(plan['blocks.0.attn.wq'], 'transformer_blocks.0.attn.to_q')
        self.assertEqual(plan['blocks.0.attn.wo'], 'transformer_blocks.0.attn.to_out.0')
        self.assertEqual(plan['blocks.0.mlp.gate'], 'transformer_blocks.0.ff.gate')

    def test_control_weight_plan_refuses_missing_extra_and_wrong_shapes(self):
        weights, modules = self.control_shapes()
        for mutation in ('missing', 'extra', 'projection', 'rank', 'target'):
            supplied, targets = dict(weights), dict(modules)
            if mutation == 'missing': supplied.pop('blocks.0.attn.wq.B')
            elif mutation == 'extra': supplied['blocks.0.attn.wq.other'] = (2, 2)
            elif mutation == 'projection': supplied['first.weight'] = (8, 4)
            elif mutation == 'rank': supplied['blocks.0.attn.wq.B'] = (8, 3)
            else: targets.pop('transformer_blocks.0.attn.to_q')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                kc.plan_weights(supplied, targets, projection_shape=(8, 4), block_count=1)

    def test_published_krea_depth_header_dimensions(self):
        dimensions = {'attn.wq': (6144, 6144), 'attn.wk': (1536, 6144),
                      'attn.wv': (1536, 6144), 'attn.wo': (6144, 6144),
                      'attn.gate': (6144, 6144), 'mlp.gate': (16384, 6144),
                      'mlp.up': (16384, 6144), 'mlp.down': (6144, 16384)}
        weights = {'first.weight': (6144, 128), 'first.bias': (6144,)}
        modules = {}
        for index in range(28):
            for source, target in kc.BLOCK_TARGETS.items():
                outputs, inputs = dimensions[source]
                modules[f'transformer_blocks.{index}.{target}'] = (outputs, inputs)
                weights[f'blocks.{index}.{source}.A'] = (64, inputs)
                weights[f'blocks.{index}.{source}.B'] = (outputs, 64)
        self.assertEqual(len(kc.plan_weights(weights, modules, projection_shape=(6144, 64), block_count=28)), 224)

    def request(self, distilled=False, **values):
        args = argparse.Namespace(inputs={}, dtype='float32', krea2_variant='auto', krea2_mu=None)
        for key, value in values.items():
            setattr(args, key, value)
        return args, {'_class_name': 'Krea2Pipeline', 'is_distilled': distilled}

    def test_official_raw_and_turbo_defaults_are_distinct(self):
        for distilled, steps, guidance in [(False, 52, 3.5), (True, 8, 0.0)]:
            args, index = self.request(distilled)
            contract = k.resolve(args, index)
            self.assertEqual(args.inputs['num_inference_steps'], steps)
            self.assertEqual(args.inputs['guidance_scale'], guidance)
            self.assertEqual(contract['variant'], 'turbo' if distilled else 'raw')

    def test_explicit_controls_survive_defaults(self):
        args, index = self.request(True, inputs={'num_inference_steps': 12, 'guidance_scale': 1.25}, krea2_mu=0.9)
        contract = k.resolve(args, index)
        self.assertEqual(args.inputs['num_inference_steps'], 12)
        self.assertEqual(args.inputs['guidance_scale'], 1.25)
        self.assertEqual(contract['mu_override'], 0.9)

    def test_variant_must_not_be_guessed_or_contradicted(self):
        args, index = self.request(True, krea2_variant='raw')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            k.resolve(args, index)
        args, index = self.request()
        del index['is_distilled']
        with self.assertRaisesRegex(ValueError, 'variant'):
            k.resolve(args, index)

    def test_fp16_is_not_a_high_precision_substitute(self):
        args, index = self.request(dtype='float16')
        with self.assertRaisesRegex(ValueError, 'float16'):
            k.resolve(args, index)

    def test_non_krea_pipeline_does_not_receive_krea_options(self):
        args, _ = self.request()
        self.assertIsNone(k.resolve(args, {'_class_name': 'FluxPipeline'}))
        args.krea2_mu = 1.15
        with self.assertRaisesRegex(ValueError, 'Krea2Pipeline'):
            k.resolve(args, {'_class_name': 'FluxPipeline'})

    def test_custom_schedule_sets_step_count(self):
        args, index = self.request(inputs={'sigmas': [1.0, 0.7, 0.2]})
        k.resolve(args, index)
        self.assertEqual(args.inputs['num_inference_steps'], 3)

    def test_bad_schedule_and_step_disagreement_fail(self):
        for schedule in [[], [1, 1], [0.2, 0.8], [1, 0], [1, math.nan], [1.1, 0.2]]:
            args, index = self.request(inputs={'sigmas': schedule})
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                k.resolve(args, index)
        args, index = self.request(inputs={'sigmas': [1, .5], 'num_inference_steps': 3})
        with self.assertRaisesRegex(ValueError, 'steps'):
            k.resolve(args, index)

    def test_shape_and_numeric_controls_are_checked_before_loading(self):
        for inputs in [{'width': 1025}, {'height': 0}, {'num_inference_steps': True},
                       {'guidance_scale': -1}, {'guidance_scale': math.inf},
                       {'max_sequence_length': 0}, {'num_images_per_prompt': 0}]:
            args, index = self.request(inputs=inputs)
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                k.resolve(args, index)

    def test_resolution_shift_matches_official_linear_formula(self):
        self.assertAlmostEqual(k.default_mu('raw', 256, 256), .5)
        self.assertAlmostEqual(k.default_mu('raw', 1280, 1280), 1.15)
        self.assertAlmostEqual(k.default_mu('turbo', 1024, 2048), 1.15)


if __name__ == '__main__':
    unittest.main()
