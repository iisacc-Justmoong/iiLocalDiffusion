#!/usr/bin/env python3
"""Real tensor interpolation with lightweight recording pipelines, no model download."""

from contextlib import redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from local_model_fixture import local_request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
from cpu_conditioning import CpuConditioning
import interpolator_runtime as runtime
from HiresRuntimeTests import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch runtime is not installed")
class InterpolatorRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.preset, self.args = local_request({"animation_mode": "Interpolator", "max_frames": 3,
                                                           "width": 32, "height": 32, "device": "cpu",
                                                           "seed": 42, "end_seed": 43, "end_prompt": "a forest"})

    def conditioning(self, value, *, pooled=False):
        tensors = {"prompt_embeds": self.torch.full((1, 3, 4), value, dtype=self.torch.float32),
                   "negative_prompt_embeds": self.torch.full((1, 3, 4), -value, dtype=self.torch.float32)}
        if pooled:
            tensors.update({"pooled_prompt_embeds": self.torch.full((1, 4), value),
                            "negative_pooled_prompt_embeds": self.torch.full((1, 4), -value)})
        return CpuConditioning(tensors, {"enabled": True, "execution_device": "cpu"})

    def test_all_conditioning_tensors_and_exact_endpoints_are_interpolated(self):
        a, b = self.conditioning(2., pooled=True), self.conditioning(6., pooled=True)
        for fraction in (0, 0.25, 1):
            result = runtime.interpolate_conditioning(a, b, fraction, self.torch)
            for name, tensor in result.tensors.items():
                expected = (2 + fraction * 4) * (-1 if name.startswith("negative") else 1)
                self.assertTrue(self.torch.equal(tensor, self.torch.full_like(tensor, expected)))
            # Caller mutation cannot corrupt a cached endpoint.
            result.tensors["prompt_embeds"].zero_()
            self.assertEqual(float(a.tensors["prompt_embeds"].mean()), 2)

    def test_incompatible_or_nonfinite_conditioning_is_rejected(self):
        a = self.conditioning(1.)
        for b in (CpuConditioning({"prompt_embeds": self.torch.ones((2, 3, 4))}, {}),
                  self.conditioning(float("nan")),
                  CpuConditioning({**a.tensors, "negative_prompt_embeds": None}, {})):
            with self.subTest(b=b.metadata), self.assertRaises(ValueError):
                runtime.interpolate_conditioning(a, b, 0.5, self.torch)

    def test_noise_blend_preserves_variance_and_equal_seed_noise(self):
        a, b = self.torch.tensor([1., 0.]), self.torch.tensor([0., 1.])
        middle = runtime.interpolate_noise(a, b, 0.5, self.torch)
        self.assertTrue(self.torch.allclose(middle, self.torch.tensor([math.sqrt(.5)] * 2)))
        self.assertAlmostEqual(float(middle.square().sum()), 1, places=6)
        self.assertTrue(self.torch.equal(runtime.interpolate_noise(a, a, .5, self.torch), a))
        for fraction, expected in ((0, a), (1, b)):
            self.assertTrue(self.torch.equal(runtime.interpolate_noise(a, b, fraction, self.torch), expected))
        for fraction in (-1, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                runtime.interpolate_noise(a, b, fraction, self.torch)

    def test_latent_seeds_are_repeatable_without_global_rng_changes(self):
        pipeline = SimpleNamespace(vae_scale_factor=8, unet=SimpleNamespace(config=SimpleNamespace(in_channels=4)))
        state = self.torch.get_rng_state().clone()
        a, b, info = runtime.prepare_noise(pipeline, self.preset, self.args, self.torch, "cpu", self.torch.float32)
        again = runtime.prepare_noise(pipeline, self.preset, self.args, self.torch, "cpu", self.torch.float32)
        self.assertEqual(tuple(a.shape), (1, 4, 4, 4))
        self.assertTrue(self.torch.equal(a, again[0]))
        self.assertFalse(self.torch.equal(a, b))
        self.assertTrue(self.torch.equal(state, self.torch.get_rng_state()))
        self.assertEqual(info["seeds"], [42, 43])

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Diffusers runtime is not installed")
    def test_flux_noise_uses_the_pipeline_packing_contract(self):
        from diffusers import FluxPipeline
        preset, args = local_request({"preset": "flux1-schnell", "animation_mode": "Interpolator",
                                                 "width": 32, "height": 32, "max_frames": 3})
        pipeline = SimpleNamespace(vae_scale_factor=8, transformer=SimpleNamespace(config=SimpleNamespace(in_channels=64)),
                                   _pack_latents=FluxPipeline._pack_latents)
        a, b, _ = runtime.prepare_noise(pipeline, preset, args, self.torch, "cpu", self.torch.float32)
        self.assertEqual(tuple(a.shape), (1, 4, 64))
        self.assertTrue(self.torch.equal(a, b))

    def test_end_prompt_is_encoded_once_before_execution_preparation(self):
        a, b = self.conditioning(1.), self.conditioning(2.)
        self.args.text_embedding_activation = None
        with patch.object(runtime, "encode_cpu_prompt", return_value=b) as encode:
            endpoints = runtime.prepare_endpoints(SimpleNamespace(), self.preset, self.args, self.torch, a)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(encode.call_args.args[2].prompt, "a forest")
        self.assertEqual(len(endpoints), 2)

    def test_adapter_preparation_encodes_both_endpoints_before_gpu_hooks(self):
        events = []
        a, b = self.conditioning(1.), self.conditioning(2.)
        pipeline = SimpleNamespace()
        with (patch.object(generate, "validate_pipeline_contract"),
              patch.object(generate, "validate_clip_skip"),
              patch.object(generate, "apply_text_embeddings", return_value=None),
              patch.object(generate, "validate_text_embeddings"),
              patch.object(generate, "apply_lora", side_effect=lambda *a, **kw: events.append("lora")),
              patch.object(generate, "encode_cpu_prompt", side_effect=lambda *unused: events.append("start") or a),
              patch.object(runtime, "prepare_endpoints", side_effect=lambda *unused: events.append("end") or (a, b)),
              patch.object(generate, "prepare_pipeline_for_execution",
                           side_effect=lambda *a, **kw: (events.append("hooks") or pipeline, {}))):
            generate.prepare_pipeline_with_adapters(pipeline, self.preset, self.args, "cpu", False, self.torch)
        self.assertEqual(events, ["lora", "start", "end", "hooks"])

    def render(self, *, finite=True, mutate=False):
        torch = self.torch
        class Scheduler:
            config = {}
            @classmethod
            def from_config(cls, config):
                return cls()
        calls, schedulers = [], []
        class Pipeline:
            vae_scale_factor = 8
            unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))
            scheduler = Scheduler()
            def set_progress_bar_config(self, **kwargs):
                pass
            def __call__(self, **call):
                calls.append(call)
                schedulers.append(self.scheduler)
                latents = call["latents"].clone()
                if not finite:
                    latents.fill_(float("nan"))
                call["callback_on_step_end"](self, 0, torch.tensor(999), {"latents": latents})
                if mutate:
                    call["latents"].zero_()
                    call["prompt_embeds"].zero_()
                return SimpleNamespace(images=[Image((32, 32), label=str(len(calls)))])
        pipeline = Pipeline()
        self.args.interpolator_conditioning = (self.conditioning(2.), self.conditioning(6.))
        self.saved = []
        def write(index, image):
            self.saved.append(index)
            return {"path": f"frame-{index:06d}.png"}
        with redirect_stdout(io.StringIO()):
            frames, info = runtime.render_interpolator_frames(
                pipeline, self.preset, self.args, torch, "cpu", torch.float32,
                build_call=generate.build_pipeline_call_arguments, write_frame=write)
        return calls, schedulers, frames, info

    def test_every_frame_diffuses_interpolated_conditions_with_fresh_scheduler(self):
        calls, schedulers, frames, info = self.render()
        self.assertEqual([float(call["prompt_embeds"].mean()) for call in calls], [2, 4, 6])
        self.assertTrue(all("prompt" not in call and "image" not in call for call in calls))
        self.assertEqual(len({id(value) for value in schedulers}), 3)
        self.assertTrue(all(frame["sampling"]["executed_steps"] == 1 for frame in frames))
        self.assertEqual([frame["fraction"] for frame in frames], [0, .5, 1])
        self.assertEqual(frames[0]["latents"]["sha256"], info["noise"]["start"]["sha256"])
        self.assertEqual(frames[-1]["latents"]["sha256"], info["noise"]["end"]["sha256"])
        self.assertEqual([frame["sampler_seed"] for frame in frames], [42, 42, 42])

    def test_pipeline_mutation_does_not_change_cached_endpoints(self):
        self.render(mutate=True)
        self.assertEqual(float(self.args.interpolator_conditioning[0].tensors["prompt_embeds"].mean()), 2)
        self.assertEqual(float(self.args.interpolator_conditioning[1].tensors["prompt_embeds"].mean()), 6)

    def test_frame_loop_does_not_read_offloaded_text_encoder_weights(self):
        # Endpoint preparation already verified this activation. The recording
        # pipeline deliberately has no text encoder once frames start.
        self.args.text_embedding_activation = object()
        _, _, frames, _ = self.render()
        self.assertEqual(len(frames), 3)

    def test_nonfinite_diffusion_cannot_write_a_frame(self):
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            self.render(finite=False)
        self.assertEqual(self.saved, [])


if __name__ == "__main__":
    unittest.main()
