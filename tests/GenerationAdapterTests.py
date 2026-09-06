#!/usr/bin/env python3
"""Real offline tiny-runtime coverage, with no downloaded model weights."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/diffusers"))
from generation_composition import Port, Source, Stage, GenerationPipeline
from generation_adapters import DiffusersAdapter, TransformersAdapter, TokenDecodeAdapter

RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "diffusers", "transformers"))


class AdapterContractTests(unittest.TestCase):
    def test_beam_logits_require_an_explicit_beam_conversion(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(num_beams=3))
        with self.assertRaisesRegex(ValueError, "beam"):
            TransformersAdapter(model, {"logits": "logits"})
        with self.assertRaisesRegex(ValueError, "beam"):
            TransformersAdapter(object(), {"logits": "logits"}, {"num_beams": 2})
        # Sequences remain the backend's final beam-selected token IDs.
        TransformersAdapter(model, {"tokens": "sequences"})
        TransformersAdapter(model, {"logits": "logits"}, {"num_beams": 1})
        TransformersAdapter(SimpleNamespace(generation_config=SimpleNamespace(num_beams=None)), {"logits": "logits"})

    def test_opaque_cache_and_disabled_structured_output_rejected(self):
        with self.assertRaises(ValueError):
            TransformersAdapter(object(), {"cache": "past_key_values"})
        with self.assertRaises(ValueError):
            TransformersAdapter(object(), {"tokens": "sequences"}, {"return_dict_in_generate": False})
        with self.assertRaises(ValueError):
            DiffusersAdapter(object(), {"images": "images"}, {"return_dict": False})


@unittest.skipUnless(RUNTIME, "optional pinned generation runtime is not installed in this interpreter")
class AdapterRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def tokens(self):
        return Port.tensor("int64", (1, None), "BS", "token_ids", "tiny-vocab-v1")

    def test_tiny_transformers_generation_preserves_tokens_and_step_logits(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=16, n_embd=8,
            n_layer=1, n_head=1, bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
        executor = TransformersAdapter(model, {"tokens": "sequences", "logits": "logits"},
            {"max_new_tokens": 2, "do_sample": False, "output_logits": True})
        stage = Stage("ar", "autoregressive", {"input_ids": self.tokens()},
            {"tokens": self.tokens(), "logits": Port.tensor("float32", (1, 2, 16), "BSV", "logits", "tiny-vocab-v1")},
            {"input_ids": Source.input("tokens")}, executor)
        result = GenerationPipeline({"tokens": self.tokens()}, [stage],
            {"tokens": Source("ar", "tokens"), "logits": Source("ar", "logits")}).run({"tokens": torch.tensor([[1, 2]])})
        self.assertEqual(tuple(result.outputs["tokens"].shape), (1, 4))
        self.assertEqual(result.outputs["tokens"].dtype, torch.int64)
        self.assertEqual(tuple(result.outputs["logits"].shape), (1, 2, 16))

    def test_tiny_diffusion_pipeline_output(self):
        from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
        unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
            block_out_channels=(8,), down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",), norm_num_groups=4)
        pipe = DDPMPipeline(unet, DDPMScheduler(num_train_timesteps=4))
        pipe.set_progress_bar_config(disable=True)
        stage = Stage("ddpm", "diffusion", {},
            {"pixels": Port.tensor("float32", (1, 8, 8, 3), "BHWC", "pixels", "rgb-0-1")}, {},
            DiffusersAdapter(pipe, {"pixels": "images"}, {"num_inference_steps": 2, "output_type": "np"}))
        output = GenerationPipeline({}, [stage], {"pixels": Source("ddpm", "pixels")}).run({}).outputs["pixels"]
        self.assertEqual(output.shape, (1, 8, 8, 3))

    def test_explicit_decode_bridge_connects_autoregressive_and_flow_prompt(self):
        import torch
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
            {"[UNK]": 0, "red": 1, "cube": 2}, unk_token="[UNK]")), unk_token="[UNK]")
        token_stage = Stage("tokens", "autoregressive", {"ids": self.tokens()}, {"ids": self.tokens()},
            {"ids": Source.input("ids")}, lambda values: dict(values))
        bridge = Stage("decode", "adapter", {"tokens": self.tokens()}, {"text": Port.text(batch=True)},
            {"tokens": Source("tokens", "ids")}, TokenDecodeAdapter(tokenizer))
        class PromptPipeline:
            def __call__(self, prompt, return_dict=True):
                return {"text": [value + " generated" for value in prompt]}
        flow = Stage("flow", "rectified-flow", {"prompt": Port.text(batch=True)}, {"text": Port.text(batch=True)},
            {"prompt": Source("decode", "text")}, DiffusersAdapter(PromptPipeline(), {"text": "text"}))
        result = GenerationPipeline({"ids": self.tokens()}, [token_stage, bridge, flow],
            {"text": Source("flow", "text")}).run({"ids": torch.tensor([[1, 2]])})
        self.assertEqual(result.outputs["text"], ["red cube generated"])
        self.assertTrue(result.hybrid)

    def test_flow_matching_and_rectified_flow_step_io_keep_velocity_distinct(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        port = Port.tensor("float32", (1, 2), "BC", "sample", "flow-state-v1")
        for architecture in ("flow-matching", "rectified-flow"):
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=4)
            scheduler.set_timesteps(2)
            stage = Stage("step", architecture, {"sample": port}, {"sample": port},
                {"sample": Source.input("sample")},
                lambda values: {"sample": scheduler.step(torch.ones_like(values["sample"]),
                    scheduler.timesteps[0], values["sample"]).prev_sample})
            result = GenerationPipeline({"sample": port}, [stage],
                {"sample": Source("step", "sample")}).run({"sample": torch.zeros(1, 2)})
            self.assertTrue(torch.isfinite(result.outputs["sample"]).all())
            self.assertFalse(torch.equal(result.outputs["sample"], torch.zeros(1, 2)))

    def test_tensor_validation_actual_dynamic_shape_nonfinite_tokens_and_dtype(self):
        import numpy as np
        port = Port.tensor("float32", (None, 4), "BC", "latent", "vae-v1")
        port.validate(np.ones((2, 4), dtype=np.float32), "good")
        for value in (np.ones((2, 5), dtype=np.float32), np.ones((2, 4), dtype=np.float64),
                      np.full((2, 4), np.nan, dtype=np.float32), np.empty((0, 4), dtype=np.float32)):
            with self.assertRaises(ValueError):
                port.validate(value, "bad")
        with self.assertRaises(ValueError):
            self.tokens().validate(np.array([[-1]], dtype=np.int64), "tokens")
        downstream_called = []
        stage = Stage("next", "diffusion", {"value": Port.tensor("float32", (1, 4), "BC", "latent", "vae-v1")},
            {"value": port}, {"value": Source.input("value")}, lambda x: downstream_called.append(1))
        pipeline = GenerationPipeline({"value": port}, [stage], {"value": Source("next", "value")})
        with self.assertRaises(ValueError):
            pipeline.run({"value": np.ones((2, 4), dtype=np.float32)})
        self.assertEqual(downstream_called, [])

    def test_duplicate_parameter_bindings_are_rejected_without_execution(self):
        adapter = DiffusersAdapter(lambda **_: None, {"pixels": "images"}, {"prompt": "fixed"})
        with self.assertRaisesRegex(ValueError, "both"):
            adapter({"prompt": "bound"})

    def test_callback_cannot_corrupt_external_or_retained_intermediate_values(self):
        import numpy as np
        port = Port.tensor("float32", (1, 2), "BC", "latent", "vae-v1")
        original = np.ones((1, 2), dtype=np.float32)
        retained = []
        def first(values):
            retained.append(values["value"])
            return dict(values)
        def mutate(values):
            values["value"][0, 0] = np.nan
            retained[0][0, 1] = np.nan
            return {"value": np.zeros((1, 2), dtype=np.float32)}
        stages = [Stage("first", "diffusion", {"value": port}, {"value": port},
                        {"value": Source.input("value")}, first),
                  Stage("second", "flow-matching", {"value": port}, {"value": port},
                        {"value": Source("first", "value")}, mutate)]
        result = GenerationPipeline({"value": port}, stages,
            {"first": Source("first", "value"), "second": Source("second", "value")}).run({"value": original})
        np.testing.assert_array_equal(original, np.ones((1, 2), dtype=np.float32))
        np.testing.assert_array_equal(result.outputs["first"], original)
        np.testing.assert_array_equal(result.outputs["second"], np.zeros((1, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
