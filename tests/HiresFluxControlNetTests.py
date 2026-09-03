#!/usr/bin/env python3
"""FLUX ControlNet refinement preserves the complete img2img noise schedule."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))

import hires_flux_controlnet


class Values(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Values(value) if isinstance(key, slice) else value

    def repeat(self, count):
        return Values(self * count)

    def tolist(self):
        return list(self)


class Scheduler:
    order = 1
    config = {"base_image_seq_len": 256, "max_image_seq_len": 4096,
              "base_shift": 0.5, "max_shift": 1.15}

    def __init__(self):
        self.calls = []
        self.begin_index = None

    def set_timesteps(self, *, sigmas, device, mu):
        self.calls.append((list(sigmas), device, mu))
        # The nonlinear transform deliberately makes a second application wrong.
        self.sigmas = Values([sigma ** 2 for sigma in sigmas] + [0.0])
        self.timesteps = Values([sigma * 1000 for sigma in self.sigmas[:-1]])
        self.begin_index = None


class ImagePipeline:
    def __init__(self, scheduler, vae, transformer, text_encoder=None):
        self.scheduler = scheduler
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae_scale_factor = 8
        self.image_processor = SimpleNamespace(preprocess=lambda images, **unused: images)

    def get_timesteps(self, steps, strength, device):
        self.scheduler.begin_index = int(steps - min(steps * strength, steps))
        return self.scheduler.timesteps[self.scheduler.begin_index:], steps - self.scheduler.begin_index

    def prepare_latents(self, image, timestep, batch_size, channels, height, width,
                        dtype, device, generator):
        self.vae.encoded.append({
            "image": image, "timestep": list(timestep), "batch": batch_size,
            "shape": (channels, height, width), "dtype": dtype, "device": device,
            "generator": generator, "sigmas": list(self.scheduler.sigmas),
            "begin_index": self.scheduler.begin_index,
        })
        return "image-derived-noisy-latents", "ids"


class Pipeline:
    def __init__(self):
        self.scheduler = Scheduler()
        self.vae = SimpleNamespace(encoded=[])
        self.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=64))
        self.components = {"scheduler": self.scheduler, "vae": self.vae,
                           "transformer": self.transformer, "text_encoder": object(),
                           "controlnet": object()}
        self.calls = []
        self.failure = None

    def __call__(self, **values):
        self.calls.append(values)
        before = self.scheduler.timesteps
        self.scheduler.set_timesteps(sigmas=values["sigmas"], device="cpu", mu=0.7)
        if self.scheduler.timesteps is not before:
            raise AssertionError("The selected img2img timesteps were overwritten")
        self.observed = {"timesteps": list(self.scheduler.timesteps),
                         "sigmas": list(self.scheduler.sigmas),
                         "begin_index": self.scheduler.begin_index}
        if self.failure:
            raise self.failure
        return SimpleNamespace(images=["result1", "result2"])


class Numpy:
    @staticmethod
    def linspace(start, end, count):
        return Values([start + index * (end - start) / (count - 1)
                       for index in range(count)]) if count > 1 else Values([start])


class HiresFluxControlNetTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.request = SimpleNamespace(hires_denoising_strength=0.5)
        self.images = ["upscaled1", "upscaled2"]
        self.generators = [object(), object()]
        self.values = {"width": 64, "height": 64, "num_inference_steps": 4,
                       "num_images_per_prompt": 1, "true_cfg_scale": 2.0,
                       "prompt_embeds": "positive", "pooled_prompt_embeds": "pooled",
                       "negative_prompt_embeds": "negative",
                       "negative_pooled_prompt_embeds": "negative pooled",
                       "control_image": "control", "controlnet_conditioning_scale": 0.8,
                       "control_guidance_start": 0.2, "control_guidance_end": 0.9}

    def refine(self):
        with patch.object(hires_flux_controlnet, "_dependencies",
                          return_value=(ImagePipeline, lambda *args: 0.7, Numpy)):
            return hires_flux_controlnet.refine_flux_controlnet(
                self.pipeline, self.request, self.images, self.values,
                SimpleNamespace(inference_mode=nullcontext), self.generators, "cpu", "float32")

    def test_refinement_uses_image_noise_and_full_transformed_schedule_once(self):
        result, metadata = self.refine()
        self.assertEqual(result.images, ["result1", "result2"])
        self.assertEqual(len(self.pipeline.scheduler.calls), 1)
        self.assertEqual(self.pipeline.observed, {
            "timesteps": [250.0, 62.5], "sigmas": [1.0, 0.5625, 0.25, 0.0625, 0.0],
            "begin_index": 2,
        })
        self.assertEqual(self.pipeline.vae.encoded[0]["timestep"], [250.0, 250.0])
        self.assertEqual(self.pipeline.vae.encoded[0]["image"], self.images)
        self.assertEqual(self.pipeline.vae.encoded[0]["batch"], 2)
        self.assertIs(self.pipeline.vae.encoded[0]["generator"], self.generators)
        self.assertEqual(metadata["effective_steps"], 2)
        self.assertEqual(metadata["denoising_timesteps"], [250.0, 62.5])
        self.assertEqual(metadata["full_sigmas"], [1.0, 0.5625, 0.25, 0.0625, 0.0])

    def test_negative_conditioning_control_and_generators_are_preserved(self):
        self.refine()
        called = self.pipeline.calls[0]
        for key, value in self.values.items():
            self.assertEqual(called[key], value)
        self.assertIs(called["generator"], self.generators)
        self.assertEqual(called["latents"], "image-derived-noisy-latents")
        self.assertNotIn("latents", self.values)

    def test_fractional_strength_uses_diffusers_flux_rounding(self):
        self.request.hires_denoising_strength = 0.3
        _, metadata = self.refine()
        self.assertEqual(metadata["effective_steps"], 2)
        self.assertEqual(self.pipeline.observed["begin_index"], 2)

    def test_schedule_and_class_method_are_restored_after_success(self):
        self.refine()
        scheduler = self.pipeline.scheduler
        self.assertNotIn("set_timesteps", scheduler.__dict__)
        self.assertEqual(scheduler.timesteps, [1000.0, 562.5, 250.0, 62.5])
        scheduler.set_timesteps(sigmas=[1.0], device="cpu", mu=0.7)
        self.assertEqual(len(scheduler.calls), 2)

    def test_existing_instance_method_is_restored_after_failure(self):
        scheduler = self.pipeline.scheduler
        original = scheduler.set_timesteps
        scheduler.set_timesteps = original
        self.pipeline.failure = RuntimeError("model failure")
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            self.refine()
        self.assertIs(scheduler.__dict__["set_timesteps"], original)
        self.assertEqual(scheduler.timesteps, [1000.0, 562.5, 250.0, 62.5])

    def test_empty_batch_and_invalid_strength_fail_before_model_execution(self):
        for value in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(strength=value):
                self.request.hires_denoising_strength = value
                with self.assertRaises(ValueError):
                    self.refine()
        self.request.hires_denoising_strength = 0.5
        self.images = []
        with self.assertRaises(ValueError):
            self.refine()
        self.assertFalse(self.pipeline.calls)
        self.assertFalse(self.pipeline.scheduler.calls)


if __name__ == "__main__":
    unittest.main()
