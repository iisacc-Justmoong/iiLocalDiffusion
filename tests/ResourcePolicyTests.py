#!/usr/bin/env python3
"""CPU conditioning and RAM offload contracts without large model dependencies."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "diffusers"
sys.path.insert(0, str(REFERENCE))
import cpu_conditioning
import generate
import presets


class FakeTensor:
    def __init__(self, device="cpu", dtype="float32"):
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.shape = (1, 77, 8)

    def to(self, *, device, dtype=None):
        return FakeTensor(device, self.dtype if dtype is None else dtype)

    def detach(self):
        return self


class FakeEncoder:
    def __init__(self):
        self.device = SimpleNamespace(type="cpu")
        self.dtype = "float16"
        self.moves = []

    def to(self, *, device, dtype):
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.moves.append((device, dtype))
        return self

    def parameters(self):
        return iter([FakeTensor(self.device.type, self.dtype)])

    def buffers(self):
        return iter([])

    def named_parameters(self):
        return iter([("weight", FakeTensor(self.device.type, self.dtype))])

    def named_buffers(self):
        return iter([])


def fake_torch():
    return SimpleNamespace(
        float32="float32", bfloat16="bfloat16", inference_mode=nullcontext,
        isfinite=lambda tensor: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
    )


class ResourcePolicyTests(unittest.TestCase):
    def test_cli_exposes_cpu_and_ram_options(self):
        args = generate.build_parser().parse_args([
            "--cpu-text-encoding", "--cpu-threads", "4", "--offload", "model"
        ])
        self.assertTrue(args.cpu_text_encoding)
        self.assertEqual(args.cpu_threads, 4)
        self.assertEqual(args.offload, "model")

    def test_cpu_threads_must_be_positive(self):
        preset, args = generate.resolve_arguments(generate.build_parser().parse_args([
            "--cpu-threads", "0"
        ]))
        with self.assertRaisesRegex(SystemExit, "CPU threads"):
            generate.validate_generation_arguments(preset, args)

    def test_offload_overrides_use_diffusers_ram_hooks(self):
        for device in ("mps", "cuda"):
            for requested, method, expected in (
                ("model", "enable_model_cpu_offload", "model-cpu"),
                ("sequential", "enable_sequential_cpu_offload", "sequential-cpu"),
            ):
                with self.subTest(device=device, requested=requested):
                    pipeline = Mock()
                    returned, metadata = generate.prepare_pipeline_for_execution(
                        pipeline, presets.SD15_PRESET, device, False, offload=requested)
                    self.assertIs(returned, pipeline)
                    getattr(pipeline, method).assert_called_once_with(device=device)
                    pipeline.to.assert_not_called()
                    self.assertEqual(metadata["offload_policy"], expected)
                    self.assertEqual(metadata["weight_storage"], "ram")

    def test_cpu_has_no_gpu_offload(self):
        with self.assertRaisesRegex(ValueError, "requires a GPU"):
            generate.prepare_pipeline_for_execution(Mock(), presets.SD15_PRESET, "cpu", False,
                                                    offload="model")

    def test_unknown_offload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "offload"):
            generate.prepare_pipeline_for_execution(Mock(), presets.SD15_PRESET, "mps", False,
                                                    offload="disk")

    def test_cpu_prompt_shapes_are_preserved_for_each_family(self):
        for preset, count, expected in (
            (presets.SD15_PRESET, 2, {"prompt_embeds", "negative_prompt_embeds"}),
            (presets.SDXL_BASE_PRESET, 4, {"prompt_embeds", "negative_prompt_embeds",
                                        "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}),
            (presets.FLUX1_SCHNELL_PRESET, 3, {"prompt_embeds", "pooled_prompt_embeds"}),
        ):
            with self.subTest(preset=preset.name):
                pipeline = SimpleNamespace(text_encoder=FakeEncoder(), text_encoder_2=FakeEncoder())
                pipeline.encode_prompt = Mock(return_value=tuple(FakeTensor() for _ in range(count)))
                args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=preset.guidance_scale)
                result = cpu_conditioning.encode_cpu_prompt(pipeline, preset, args, fake_torch())
                self.assertEqual(set(result.tensors), expected)
                self.assertTrue(result.metadata["enabled"])
                self.assertEqual(result.metadata["execution_device"], "cpu")
                self.assertEqual(pipeline.encode_prompt.call_args.kwargs["device"], "cpu")
                self.assertEqual(pipeline.text_encoder.dtype, "float16")
                self.assertEqual(pipeline.text_encoder.device.type, "cpu")
                transferred = result.for_device("mps", "float16")
                self.assertTrue(all(t.device.type == "mps" for t in transferred.values()))
                if preset.name == "flux1-schnell":
                    self.assertNotIn("negative_prompt", pipeline.encode_prompt.call_args.kwargs)
                    self.assertEqual(pipeline.encode_prompt.call_args.kwargs["max_sequence_length"], 256)

    def test_cpu_conditioning_restores_dtype_when_encoding_fails(self):
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(), encode_prompt=Mock(side_effect=RuntimeError("encoder")))
        args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=7.5)
        with self.assertRaisesRegex(RuntimeError, "encoder"):
            cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, fake_torch())
        self.assertEqual(pipeline.text_encoder.dtype, "float16")

    def test_cpu_conditioning_preserves_mixed_adapter_storage_dtype(self):
        class MixedEncoder(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.adapter = SimpleNamespace(data=FakeTensor(dtype="float32"))

            def named_parameters(self):
                return iter([("adapter", self.adapter.data)])

            def get_parameter(self, name):
                if name != "adapter":
                    raise KeyError(name)
                return self.adapter

            def to(self, *, device, dtype):
                super().to(device=device, dtype=dtype)
                self.adapter.data = self.adapter.data.to(device=device, dtype=dtype)
                return self

        for fails in (False, True):
            with self.subTest(encoding_fails=fails):
                encoder = MixedEncoder()
                pipeline = SimpleNamespace(text_encoder=encoder,
                    encode_prompt=Mock(return_value=(FakeTensor(), FakeTensor()),
                                       side_effect=RuntimeError("encoder") if fails else None))
                args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=7.5)
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "encoder"):
                        cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, fake_torch())
                else:
                    cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, fake_torch())
                self.assertEqual(encoder.dtype, "float16")
                self.assertEqual(encoder.adapter.data.dtype, "float32")

    def test_meta_weights_cannot_be_encoded_on_cpu(self):
        encoder = FakeEncoder()
        encoder.device.type = "meta"
        pipeline = SimpleNamespace(text_encoder=encoder)
        with self.assertRaisesRegex(RuntimeError, "before.*offload"):
            cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, SimpleNamespace(), fake_torch())

    def test_non_cpu_embedding_is_rejected(self):
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(),
            encode_prompt=Mock(return_value=(FakeTensor("mps"), FakeTensor())))
        args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=7.5)
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, fake_torch())

    def test_none_negative_embeddings_are_retained(self):
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(),
            encode_prompt=Mock(return_value=(FakeTensor(), None)))
        args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=1.0)
        result = cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, fake_torch())
        self.assertIsNone(result.for_device("mps", "float16")["negative_prompt_embeds"])

    def test_lora_precedes_cpu_encoding_and_offload_follows_it(self):
        for preset, expected_offload in (
            (presets.SD15_PRESET, "model"),
            (presets.SDXL_BASE_PRESET, "model"),
            (presets.FLUX1_SCHNELL_PRESET, "auto"),
        ):
            with self.subTest(preset=preset.name):
                events = []
                pipeline = object()
                conditioning = object()
                args = SimpleNamespace(lora_selection=None, cache_dir=Path("cache"),
                    local_files_only=True, offload="auto", cpu_text_encoding=True)
                def cpu(*arguments):
                    events.append("cpu")
                    return conditioning
                def place(*arguments, offload):
                    events.append("offload")
                    self.assertEqual(offload, expected_offload)
                    return pipeline, {}
                with (
                    patch.object(generate, "validate_pipeline_contract",
                                 side_effect=lambda *a: events.append("validate")),
                    patch.object(generate, "apply_lora",
                                 side_effect=lambda *a: events.append("lora")),
                    patch.object(generate, "encode_cpu_prompt", side_effect=cpu),
                    patch.object(generate, "prepare_pipeline_for_execution", side_effect=place),
                ):
                    result = generate.prepare_pipeline_with_adapters(
                        pipeline, preset, args, "mps", False, fake_torch())
                self.assertEqual(events, ["validate", "lora", "cpu", "offload"])
                self.assertIs(result[3], conditioning)
                self.assertEqual(result[1]["requested_offload"], "auto")

    def test_generation_transfers_embeddings_without_reencoding_prompt(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                _, args = generate.resolve_arguments(generate.build_parser().parse_args([
                    "--preset", preset.name
                ]))
                conditioning = cpu_conditioning.CpuConditioning({"prompt_embeds": FakeTensor()}, {})
                arguments = generate.build_pipeline_call_arguments(
                    preset, args, "generator", conditioning, "cuda", "bfloat16")
                self.assertNotIn("prompt", arguments)
                self.assertNotIn("negative_prompt", arguments)
                self.assertEqual(arguments["prompt_embeds"].device.type, "cuda")
                self.assertEqual(arguments["prompt_embeds"].dtype, "bfloat16")
                with self.assertRaisesRegex(ValueError, "target device"):
                    generate.build_pipeline_call_arguments(preset, args, "generator", conditioning)

    def test_nonfinite_cpu_embedding_is_rejected_and_dtype_restored(self):
        torch = fake_torch()
        torch.isfinite = lambda t: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False))
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(),
            encode_prompt=Mock(return_value=(FakeTensor(), FakeTensor())))
        args = SimpleNamespace(prompt="test", negative_prompt="", guidance_scale=7.5)
        with self.assertRaisesRegex(RuntimeError, "finite CPU data"):
            cpu_conditioning.encode_cpu_prompt(pipeline, presets.SD15_PRESET, args, torch)
        self.assertEqual(pipeline.text_encoder.dtype, "float16")

    def test_cpu_encoding_respects_explicit_residency(self):
        for device, requested in (("cpu", "auto"), ("mps", "none"), ("cuda", "sequential")):
            args = SimpleNamespace(lora_selection=None, cache_dir=Path("cache"),
                local_files_only=True, offload=requested, cpu_text_encoding=True)
            with (
                patch.object(generate, "validate_pipeline_contract"),
                patch.object(generate, "apply_lora"),
                patch.object(generate, "encode_cpu_prompt"),
                patch.object(generate, "prepare_pipeline_for_execution", return_value=(None, {})) as place,
            ):
                generate.prepare_pipeline_with_adapters(
                    object(), presets.SD15_PRESET, args, device, False, fake_torch())
            self.assertEqual(place.call_args.kwargs["offload"], requested)

    def test_cpu_encoding_receives_secondary_prompts_clip_skip_and_dtype(self):
        preset, args = generate.resolve_arguments(generate.build_parser().parse_args([
            "--preset", "sdxl-base", "--prompt-2", "second", "--negative-prompt-2", "bad second",
            "--clip-skip", "2", "--num-images", "3", "--cpu-text-dtype", "bfloat16"
        ]))
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(), text_encoder_2=FakeEncoder(),
            encode_prompt=Mock(return_value=tuple(FakeTensor() for _ in range(4))))
        result = cpu_conditioning.encode_cpu_prompt(pipeline, preset, args, fake_torch())
        values = pipeline.encode_prompt.call_args.kwargs
        self.assertEqual(values["prompt_2"], "second")
        self.assertEqual(values["negative_prompt_2"], "bad second")
        self.assertEqual(values["clip_skip"], 2)
        self.assertEqual(values["num_images_per_prompt"], 1)
        self.assertEqual(result.metadata["dtype"], "bfloat16")
        self.assertIn(("cpu", "bfloat16"), pipeline.text_encoder.moves)
        # SDXL expands each supplied embedding once during the pipeline call.
        call = generate.build_pipeline_call_arguments(preset, args, "generator", result, "mps", "float16")
        self.assertEqual(call["num_images_per_prompt"], 3)
        self.assertNotIn("prompt_2", call)
        self.assertNotIn("negative_prompt_2", call)

    def test_flux_cpu_encoding_preexpands_batch_and_encodes_true_cfg_negatives(self):
        preset, args = generate.resolve_arguments(generate.build_parser().parse_args([
            "--preset", "flux1-schnell", "--num-images", "3", "--max-sequence-length", "128",
            "--true-cfg-scale", "2", "--prompt-2", "second", "--negative-prompt", "bad",
            "--negative-prompt-2", "bad second",
        ]))
        pipeline = SimpleNamespace(text_encoder=FakeEncoder(), text_encoder_2=FakeEncoder(),
            encode_prompt=Mock(return_value=tuple(FakeTensor() for _ in range(3))))
        result = cpu_conditioning.encode_cpu_prompt(pipeline, preset, args, fake_torch())
        self.assertEqual(pipeline.encode_prompt.call_count, 2)
        positive, negative = [call.kwargs for call in pipeline.encode_prompt.call_args_list]
        self.assertEqual((positive["num_images_per_prompt"], positive["max_sequence_length"]), (3, 128))
        self.assertEqual((negative["prompt"], negative["prompt_2"]), ("bad", "bad second"))
        self.assertIn("negative_prompt_embeds", result.tensors)
        self.assertIn("negative_pooled_prompt_embeds", result.tensors)
        call = generate.build_pipeline_call_arguments(preset, args, "generator", result, "mps", "bfloat16")
        self.assertEqual(call["num_images_per_prompt"], 1)
        self.assertNotIn("negative_prompt", call)


if __name__ == "__main__":
    unittest.main()
