#!/usr/bin/env python3
"""Scheduler overrides use compatible runtime classes and reject ignored values."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import generation_scheduler
import presets


class FakeScheduler:
    def __init__(self, num_train_timesteps=1000, prediction_type="epsilon"):
        self.config = dict(num_train_timesteps=num_train_timesteps, prediction_type=prediction_type)

    @property
    def compatibles(self):
        return [FakeScheduler, AlternativeScheduler]

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(**{**config, **overrides})

    def set_timesteps(self, num_inference_steps=None, device=None):
        pass

    def step(self, model_output, timestep, sample):
        pass


class AlternativeScheduler(FakeScheduler):
    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        pass

    def step(self, model_output, timestep, sample, eta=0.0):
        pass


class GenerationSchedulerTests(unittest.TestCase):
    def args(self, *tokens):
        _, args = generate.resolve_arguments(generate.build_parser().parse_args(list(tokens)))
        return args

    def test_default_retains_loaded_scheduler_identity(self):
        scheduler = FakeScheduler()
        pipeline = SimpleNamespace(scheduler=scheduler)
        metadata = generation_scheduler.configure_scheduler(pipeline, self.args())
        self.assertIs(pipeline.scheduler, scheduler)
        self.assertEqual(metadata["class"], "FakeScheduler")
        self.assertEqual(metadata["config"]["num_train_timesteps"], 1000)

    def test_explicit_class_uses_from_config_and_records_overrides(self):
        pipeline = SimpleNamespace(scheduler=FakeScheduler())
        metadata = generation_scheduler.configure_scheduler(pipeline, self.args(
            "--scheduler", "AlternativeScheduler",
            "--scheduler-config", '{"prediction_type":"v_prediction"}',
        ))
        self.assertIsInstance(pipeline.scheduler, AlternativeScheduler)
        self.assertEqual(pipeline.scheduler.config["num_train_timesteps"], 1000)
        self.assertEqual(pipeline.scheduler.config["prediction_type"], "v_prediction")
        self.assertEqual(metadata["overrides"], {"prediction_type": "v_prediction"})

    def test_config_only_reconstructs_same_class(self):
        original = FakeScheduler()
        pipeline = SimpleNamespace(scheduler=original)
        generation_scheduler.configure_scheduler(pipeline, self.args(
            "--scheduler-config", '{"num_train_timesteps":100}',
        ))
        self.assertIsNot(pipeline.scheduler, original)
        self.assertIsInstance(pipeline.scheduler, FakeScheduler)
        self.assertEqual(pipeline.scheduler.config["num_train_timesteps"], 100)

    def test_unknown_class_and_unknown_or_private_keys_are_rejected(self):
        for tokens in (
            ("--scheduler", "ArbitraryUntrustedClass"),
            ("--scheduler-config", '{"unknown":1}'),
            ("--scheduler-config", '{"_class_name":"Other"}'),
        ):
            with self.subTest(tokens=tokens):
                pipeline = SimpleNamespace(scheduler=FakeScheduler())
                with self.assertRaises(ValueError):
                    generation_scheduler.configure_scheduler(pipeline, self.args(*tokens))

    def test_unsupported_schedule_or_eta_is_not_silently_ignored(self):
        for tokens in (("--timesteps", "900", "0"), ("--sigmas", "1", "0"), ("--eta", "0.1")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                generation_scheduler.validate_scheduler_values(
                    SimpleNamespace(scheduler=FakeScheduler()), self.args(*tokens))

    def test_supported_custom_schedule_and_eta_reach_validation(self):
        pipeline = SimpleNamespace(scheduler=AlternativeScheduler())
        for tokens in (("--timesteps", "900", "0"), ("--sigmas", "1", "0"), ("--eta", "0.1")):
            generation_scheduler.validate_scheduler_values(pipeline, self.args(*tokens))

    def test_timesteps_cannot_exceed_training_domain(self):
        with self.assertRaisesRegex(ValueError, "num_train_timesteps"):
            generation_scheduler.validate_scheduler_values(
                SimpleNamespace(scheduler=AlternativeScheduler()), self.args("--timesteps", "1000", "0"))

    def test_clip_skip_checks_both_sdxl_encoders(self):
        pipeline = SimpleNamespace(
            text_encoder=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=12)),
            text_encoder_2=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32)),
        )
        args = self.args("--preset", "sdxl-base", "--clip-skip", "11")
        generation_scheduler.validate_clip_skip(pipeline, presets.SDXL_BASE_PRESET, args)
        args.clip_skip = 12
        with self.assertRaisesRegex(ValueError, "clip-skip"):
            generation_scheduler.validate_clip_skip(pipeline, presets.SDXL_BASE_PRESET, args)


if __name__ == "__main__":
    unittest.main()
