#!/usr/bin/env python3
"""External latent/conditioning values without depending on model packages."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import generation_tensor_inputs as inputs
import presets


class Tensor:
    def __init__(self, shape, *, dtype="float32", finite=True, floating=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.finite, self.floating = finite, floating
        self.device = SimpleNamespace(type="cpu")

    def is_floating_point(self):
        return self.floating

    def to(self, *, device, dtype, copy=False):
        result = Tensor(self.shape, dtype=dtype, finite=self.finite, floating=self.floating)
        result.device = SimpleNamespace(type=device)
        return result

    def repeat_interleave(self, count, dim):
        assert dim == 0
        return Tensor((self.shape[0] * count, *self.shape[1:]), dtype=self.dtype,
                      finite=self.finite, floating=self.floating)


TORCH = SimpleNamespace(isfinite=lambda tensor: SimpleNamespace(
    all=lambda: SimpleNamespace(item=lambda: tensor.finite)))


def pipeline():
    return SimpleNamespace(
        vae_scale_factor=8,
        unet=SimpleNamespace(config=SimpleNamespace(in_channels=4, cross_attention_dim=768)),
        transformer=SimpleNamespace(config=SimpleNamespace(
            in_channels=64, joint_attention_dim=4096, pooled_projection_dim=768)),
        text_encoder_2=SimpleNamespace(config=SimpleNamespace(projection_dim=1280)),
    )


class GenerationTensorInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.file = Path(self.temporary.name) / "inputs.safetensor"
        self.file.write_bytes(b"simulated tensor data")

    def request(self, *arguments):
        return generate.resolve_arguments(generate.build_parser().parse_args([
            "--width", "64", "--height", "64", *arguments
        ]))

    def test_omitted_files_use_default_noise_and_text_without_reading_tensors(self):
        preset, args = self.request()
        with patch.object(inputs, "read_selected_tensors", side_effect=AssertionError("unexpected load")):
            result = inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")
        self.assertIsNone(result.latents)
        self.assertIsNone(result.conditioning)
        self.assertEqual(result.metadata, {})

    def test_latent_file_key_shape_and_identity_are_preserved(self):
        preset, args = self.request("--latents", str(self.file), "--latents-key", "noise")
        with patch.object(inputs, "read_selected_tensors", return_value={"latents": Tensor((1, 4, 8, 8))}) as read:
            result = inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")
        self.assertEqual(read.call_args.args[1], {"latents": "noise"})
        self.assertEqual(result.latents.shape, (1, 4, 8, 8))
        self.assertEqual(result.metadata["latents"]["file"]["sha256"], args.latents_file.sha256)
        args.tensor_inputs = result
        values = generate.build_pipeline_call_arguments(preset, args, "generator", None, "cuda", "float16")
        self.assertEqual(values["latents"].device.type, "cuda")

    def test_flux_requires_packed_latent_shape(self):
        preset, args = self.request("--preset", "flux1-schnell", "--num-images", "2", "--latents", str(self.file))
        for shape, valid in (((2, 16, 64), True), ((2, 16, 8, 8), False), ((1, 16, 64), False)):
            with self.subTest(shape=shape), patch.object(
                inputs, "read_selected_tensors", return_value={"latents": Tensor(shape)}
            ):
                if valid:
                    inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "bfloat16")
                else:
                    with self.assertRaisesRegex(ValueError, "shape"):
                        inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "bfloat16")

    def test_invalid_tensor_dtype_or_nonfinite_data_cannot_fall_back_to_noise(self):
        preset, args = self.request("--latents", str(self.file))
        for tensor in (Tensor((1, 4, 8, 8), floating=False), Tensor((1, 4, 8, 8), finite=False)):
            with patch.object(inputs, "read_selected_tensors", return_value={"latents": tensor}):
                with self.assertRaises(ValueError):
                    inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")

    def test_sd_embeddings_keep_one_prompt_and_pipeline_expands_batch(self):
        preset, args = self.request("--num-images", "3", "--embeddings", str(self.file))
        tensors = {"prompt_embeds": Tensor((1, 77, 768)), "negative_prompt_embeds": Tensor((1, 77, 768))}
        with patch.object(inputs, "read_selected_tensors", return_value=tensors):
            result = inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")
        self.assertFalse(result.conditioning.metadata["enabled"])
        self.assertTrue(result.conditioning.metadata["provided_embeddings"])
        values = generate.build_pipeline_call_arguments(preset, args, "generator", result.conditioning, "cuda", "float16")
        self.assertNotIn("prompt", values)
        self.assertEqual(values["num_images_per_prompt"], 3)

    def test_flux_embeddings_expand_once_and_include_true_cfg_negatives(self):
        preset, args = self.request("--preset", "flux1-schnell", "--num-images", "3",
                                    "--true-cfg-scale", "2", "--embeddings", str(self.file))
        tensors = {
            "prompt_embeds": Tensor((1, 128, 4096)), "pooled_prompt_embeds": Tensor((1, 768)),
            "negative_prompt_embeds": Tensor((1, 128, 4096)), "negative_pooled_prompt_embeds": Tensor((1, 768)),
        }
        with patch.object(inputs, "read_selected_tensors", return_value=tensors):
            result = inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "bfloat16")
        values = generate.build_pipeline_call_arguments(preset, args, "generator", result.conditioning, "cuda", "bfloat16")
        self.assertEqual(values["num_images_per_prompt"], 1)
        self.assertEqual(values["prompt_embeds"].shape, (3, 128, 4096))
        self.assertEqual(values["negative_prompt_embeds"].shape, (3, 128, 4096))

    def test_preexpanded_sd_embedding_batch_is_not_duplicated(self):
        preset, args = self.request("--num-images", "3", "--embeddings", str(self.file))
        tensors = {"prompt_embeds": Tensor((3, 77, 768)), "negative_prompt_embeds": Tensor((3, 77, 768))}
        with patch.object(inputs, "read_selected_tensors", return_value=tensors):
            result = inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")
        values = generate.build_pipeline_call_arguments(preset, args, "generator", result.conditioning, "cuda", "float16")
        self.assertEqual(values["num_images_per_prompt"], 1)

    def test_missing_cfg_embeddings_and_wrong_dimensions_are_explicit_errors(self):
        preset, args = self.request("--embeddings", str(self.file))
        for tensors in (
            {"prompt_embeds": Tensor((1, 77, 768))},
            {"prompt_embeds": Tensor((1, 77, 2048)), "negative_prompt_embeds": Tensor((1, 77, 2048))},
            {"prompt_embeds": Tensor((1, 77, 768)), "negative_prompt_embeds": Tensor((1, 12, 768))},
        ):
            with patch.object(inputs, "read_selected_tensors", return_value=tensors):
                with self.assertRaises(ValueError):
                    inputs.load_tensor_inputs(pipeline(), preset, args, TORCH, "float16")

    def test_embedding_file_and_cpu_text_encoding_are_mutually_exclusive(self):
        preset, args = self.request("--embeddings", str(self.file), "--cpu-text-encoding")
        with self.assertRaisesRegex(SystemExit, "embeddings"):
            generate.validate_generation_arguments(preset, args)

    def test_tensor_key_overrides_and_file_paths_round_trip(self):
        preset, args = self.request("--embeddings", str(self.file),
                                    "--embedding-keys", '{"prompt_embeds":"positive","negative_prompt_embeds":"negative"}')
        values = generate.configuration_values(args)
        _, replayed = generate.resolve_request(values)
        self.assertEqual(replayed.embedding_keys, args.embedding_keys)
        self.assertEqual(replayed.embeddings_file.sha256, args.embeddings_file.sha256)


if __name__ == "__main__":
    unittest.main()
