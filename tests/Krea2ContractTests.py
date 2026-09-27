"""Krea 2 variant, precision and sampling contracts (without large weights)."""
import argparse
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import krea2_contract as k


class Krea2ContractTests(unittest.TestCase):
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
