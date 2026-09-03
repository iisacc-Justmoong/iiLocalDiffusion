#!/usr/bin/env python3
"""Complete generation values, defaults, and configuration precedence."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import presets


def resolved(*arguments):
    return generate.resolve_arguments(generate.build_parser().parse_args(list(arguments)))


class GenerationOptionsTests(unittest.TestCase):
    def test_omitted_values_are_valid_for_every_family(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                _, args = resolved("--preset", preset.name)
                generate.validate_generation_arguments(preset, args)
                self.assertEqual(args.num_images, 1)
                self.assertEqual(args.seed_stride, 1)
                self.assertEqual(args.dtype, "auto")
                self.assertEqual(args.scheduler, "auto")
                self.assertEqual(args.scheduler_config, {})
                self.assertEqual(args.eta, 0.0)
                self.assertEqual(args.guidance_rescale, 0.0)
                self.assertEqual(args.true_cfg_scale, 1.0)
                self.assertEqual(args.generator_device, "cpu")
                self.assertEqual(args.device_index, 0)
                self.assertIsNone(args.lora_selection)
                self.assertEqual(args.prompt, generate.DEFAULT_PROMPT)

    def test_zero_values_are_not_replaced_by_defaults(self):
        preset, args = resolved("--seed", "0", "--seed-stride", "0", "--guidance-scale", "0",
                                "--eta", "0", "--guidance-rescale", "0", "--clip-skip", "0")
        generate.validate_generation_arguments(preset, args)
        self.assertEqual((args.seed, args.seed_stride, args.guidance_scale, args.clip_skip),
                         (0, 0, 0.0, 0))

    def test_generation_values_reach_sd15_pipeline(self):
        preset, args = resolved("--num-images", "3", "--eta", "0.4", "--guidance-rescale", "0.2",
                                "--clip-skip", "2", "--timesteps", "900", "300", "0")
        values = generate.build_pipeline_call_arguments(preset, args, "generator")
        self.assertEqual(values["num_images_per_prompt"], 3)
        self.assertEqual(values["eta"], 0.4)
        self.assertEqual(values["guidance_rescale"], 0.2)
        self.assertEqual(values["clip_skip"], 2)
        self.assertEqual(values["timesteps"], [900, 300, 0])
        self.assertNotIn("prompt_2", values)
        self.assertNotIn("max_sequence_length", values)

    def test_sdxl_secondary_prompts_and_micro_conditioning(self):
        preset, args = resolved("--preset", "sdxl-base", "--prompt", "primary", "--prompt-2", "secondary",
                                "--negative-prompt-2", "secondary negative", "--original-size", "768", "1024",
                                "--crops-coords-top-left", "8", "16", "--target-size", "1024", "1024",
                                "--negative-original-size", "512", "512", "--denoising-end", "0.8")
        generate.validate_generation_arguments(preset, args)
        values = generate.build_pipeline_call_arguments(preset, args, "generator")
        self.assertEqual(values["prompt_2"], "secondary")
        self.assertEqual(values["negative_prompt_2"], "secondary negative")
        self.assertEqual(values["original_size"], (768, 1024))
        self.assertEqual(values["crops_coords_top_left"], (8, 16))
        self.assertEqual(values["denoising_end"], 0.8)

    def test_secondary_prompt_defaults_follow_primary_without_inventing_content(self):
        for name in ("sdxl-base", "flux1-schnell"):
            preset, args = resolved("--preset", name, "--prompt", "custom subject")
            values = generate.build_pipeline_call_arguments(preset, args, "generator")
            self.assertEqual(values["prompt_2"], "custom subject")

    def test_flux_true_cfg_and_sequence_length_are_explicit(self):
        preset, args = resolved("--preset", "flux1-schnell", "--true-cfg-scale", "2",
                                "--negative-prompt", "blur", "--max-sequence-length", "128",
                                "--sigmas", "1", "0.5")
        generate.validate_generation_arguments(preset, args)
        values = generate.build_pipeline_call_arguments(preset, args, "generator")
        self.assertEqual(values["true_cfg_scale"], 2.0)
        self.assertEqual(values["negative_prompt"], "blur")
        self.assertEqual(values["negative_prompt_2"], "blur")
        self.assertEqual(values["max_sequence_length"], 128)
        self.assertEqual(values["sigmas"], [1.0, 0.5])
        self.assertNotIn("eta", values)

    def test_invalid_numeric_values_fail_before_runtime(self):
        cases = [
            ("--guidance-scale", "nan"), ("--guidance-scale", "-1"),
            ("--eta", "inf"), ("--eta", "-0.1"), ("--guidance-rescale", "1.1"),
            ("--num-images", "0"), ("--device-index", "-1"),
            ("--seed", str(2**64)), ("--seed", str(-(2**63) - 1)),
            ("--clip-skip", "-1"), ("--png-compress-level", "10"),
            ("--timesteps", "5", "7"), ("--sigmas", "1", "nan"),
            ("--timesteps", "5", "5"), ("--sigmas", "-1"),
            ("--timesteps", "9", "1", "--sigmas", "1", "0.1"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preset, args = resolved(*arguments)
                    generate.validate_generation_arguments(preset, args)

    def test_family_specific_values_cannot_be_silently_ignored(self):
        cases = [
            ("--prompt-2", "other"), ("--original-size", "512", "512"),
            ("--max-sequence-length", "128"), ("--true-cfg-scale", "2"),
            ("--preset", "flux1-schnell", "--eta", "0.2"),
            ("--preset", "flux1-schnell", "--clip-skip", "1"),
            ("--preset", "flux1-schnell", "--timesteps", "9", "1"),
            ("--preset", "sdxl-base", "--denoising-end", "1"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preset, args = resolved(*arguments)
                    generate.validate_generation_arguments(preset, args)

    def test_vae_and_attention_memory_values_are_forwarded(self):
        pipeline = Mock()
        pipeline.to.return_value = pipeline
        _, values = generate.prepare_pipeline_for_execution(
            pipeline, presets.SD15_PRESET, "cuda", True,
            vae_slicing=True, vae_tiling=True, attention_slice_size="max")
        pipeline.vae.enable_slicing.assert_called_once_with()
        pipeline.vae.enable_tiling.assert_called_once_with()
        pipeline.enable_attention_slicing.assert_called_once_with("max")
        self.assertTrue(values["vae_slicing_enabled"])
        self.assertEqual(values["attention_slice_size"], "max")

    def test_explicit_vae_disable_overrides_flux_defaults(self):
        pipeline = Mock()
        _, values = generate.prepare_pipeline_for_execution(
            pipeline, presets.FLUX1_SCHNELL_PRESET, "cuda", False,
            vae_slicing=False, vae_tiling=False)
        pipeline.vae.enable_slicing.assert_not_called()
        pipeline.vae.enable_tiling.assert_not_called()
        self.assertFalse(values["vae_slicing_enabled"])

    def test_weight_variant_and_watermark_values_reach_loader(self):
        preset, args = resolved("--preset", "sdxl-base", "--weight-variant", "none", "--watermark")
        values = generate.build_load_arguments(preset, args, "torch.float32", True)
        self.assertNotIn("variant", values)
        self.assertTrue(values["add_watermarker"])

    def test_config_file_values_cli_precedence_and_null_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "generation.json"
            path.write_text(json.dumps({
                "preset": "sdxl-base", "width": 768, "height": None, "seed": 0,
                "guidance_scale": 0, "prompt": "", "num_images": 2,
                "vae_tiling": True, "cpu_text_encoding": True,
                "scheduler_config": {"timestep_spacing": "trailing"},
            }))
            preset, args = resolved("--config", str(path), "--width", "1024",
                                    "--no-vae-tiling", "--no-cpu-text-encoding")
            generate.validate_generation_arguments(preset, args)
            self.assertEqual((args.width, args.height), (1024, 1024))
            self.assertEqual((args.seed, args.guidance_scale, args.prompt), (0, 0.0, ""))
            self.assertFalse(args.vae_tiling)
            self.assertFalse(args.cpu_text_encoding)
            self.assertEqual(args.scheduler_config, {"timestep_spacing": "trailing"})

    def test_config_paths_are_relative_to_the_configuration_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            (root / "model").mkdir()
            path = root / "options.json"
            path.write_text(json.dumps({"model": "./model", "output": "images/result.png"}))
            _, args = resolved("--config", str(path))
            self.assertEqual(args.model_selection.source, str((root / "model").resolve()))
            self.assertEqual(args.output, root / "images" / "result.png")

    def test_config_rejects_unknown_duplicate_nonfinite_and_wrong_types(self):
        invalid = [
            '{"steps":2,"steps":3}', '{"stepz":2}', '{"steps":true}',
            '{"steps":"20"}', '{"guidance_scale":NaN}', '{"prompt":12}',
            '{"vae_tiling":"false"}', '{"config":"other.json"}', '[]',
            '{"scheduler_config":{"beta_start":Infinity}}',
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "options.json"
            for text in invalid:
                with self.subTest(text=text), redirect_stderr(io.StringIO()):
                    path.write_text(text)
                    with self.assertRaises(SystemExit):
                        resolved("--config", str(path))

    def test_reusing_parser_does_not_leak_previous_configuration(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "options.json"
            path.write_text('{"seed":99}')
            parser = generate.build_parser()
            self.assertEqual(parser.parse_args(["--config", str(path)]).seed, 99)
            self.assertEqual(parser.parse_args([]).seed, generate.DEFAULT_SEED)

    def test_print_config_needs_no_models_packages_or_hardware(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["generate.py", "--print-config"]), redirect_stdout(output), \
                patch.object(generate, "package_versions", side_effect=AssertionError("packages")), \
                patch.object(generate, "load_dependencies", side_effect=AssertionError("runtime")):
            self.assertEqual(generate.main(), 0)
        values = json.loads(output.getvalue())
        self.assertEqual(values["width"], 512)
        self.assertEqual(values["num_images"], 1)
        self.assertEqual(values["model"], presets.SD15_PRESET.model_id)
        self.assertEqual(values["revision"], presets.SD15_PRESET.revision)
        self.assertNotIn("model_selection", values)

    def test_programmatic_request_uses_the_same_defaults_and_type_checks(self):
        preset, args = generate.resolve_request()
        self.assertEqual((preset.name, args.num_images, args.seed), ("sd15", 1, 42))
        _, args = generate.resolve_request({"seed": 0, "height": None, "vae_tiling": False})
        self.assertEqual((args.seed, args.height, args.vae_tiling), (0, 512, False))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            generate.resolve_request({"num_images": True})

    def test_old_programmatic_callers_inherit_new_optional_values(self):
        args = SimpleNamespace(prompt="test", negative_prompt="", width=64, height=64,
                               steps=1, guidance_scale=0.0)
        values = generate.build_pipeline_call_arguments(presets.FLUX1_SCHNELL_PRESET, args, "generator")
        self.assertEqual(values["num_images_per_prompt"], 1)
        self.assertEqual(values["max_sequence_length"], 256)
        self.assertEqual(values["prompt_2"], "test")

    def test_printed_configuration_round_trips_for_each_family(self):
        for preset in presets.PRESETS.values():
            _, original = resolved("--preset", preset.name)
            values = generate.configuration_values(original)
            _, replayed = generate.resolve_request(values)
            self.assertEqual(generate.configuration_values(replayed), values)

    def test_loader_memory_policy_is_explicit_and_defaults_to_low_memory(self):
        preset, args = resolved("--no-low-cpu-mem-usage")
        self.assertFalse(generate.build_load_arguments(preset, args, "dtype", True)["low_cpu_mem_usage"])
        _, args = resolved()
        self.assertTrue(generate.build_load_arguments(preset, args, "dtype", True)["low_cpu_mem_usage"])

    def test_portable_attention_scale_is_forwarded_and_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            lora = Path(temporary) / "adapter.safetensors"
            lora.write_bytes(b"simulated adapter")
            preset, args = resolved("--lora", str(lora), "--cross-attention-kwargs", '{"scale":0.5}')
            generate.validate_generation_arguments(preset, args)
            values = generate.build_pipeline_call_arguments(preset, args, "generator")
            self.assertEqual(values["cross_attention_kwargs"], {"scale": 0.5})
            args.cross_attention_kwargs = {"ignored_typo": 0.5}
            with self.assertRaises(SystemExit):
                generate.validate_generation_arguments(preset, args)

    def test_partial_example_keeps_valid_omission_defaults(self):
        preset, args = resolved("--config", str(ROOT / "reference/diffusers/generation.example.json"))
        generate.validate_generation_arguments(preset, args)
        self.assertEqual((args.width, args.height, args.steps, args.num_images), (512, 512, 20, 1))
        self.assertIsNone(args.lora_selection)


if __name__ == "__main__":
    unittest.main()
