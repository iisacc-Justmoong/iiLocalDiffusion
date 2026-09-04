#!/usr/bin/env python3
"""Dependency-free HiRes Fix option and second-pass request contracts."""

import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
from generation_config import ConfigurationArgumentParser, configuration_values
from generation_options import SDXL_CROPS, SDXL_SIZES
from hires_options import add_hires_options, hires_request, resolve_hires_options, resolve_hires_stage_sizes
import generate
import presets


def parser():
    result = ConfigurationArgumentParser(allow_abbrev=False)
    add_hires_options(result)
    return result


def base_arguments(preset):
    return argparse.Namespace(
        width=preset.width, height=preset.height, steps=preset.steps,
        seed=42, seed_stride=1, num_images=1,
        guidance_scale=preset.guidance_scale, true_cfg_scale=1.0,
        scheduler="auto", scheduler_config={},
        timesteps=None, sigmas=None, denoising_end=None,
        latents=None, latents_key="latents", latents_file=None, tensor_inputs=None,
        embeddings=None, embeddings_file=None,
        **{name: None for name in (*SDXL_SIZES, *SDXL_CROPS)},
    )


def resolved(*arguments, preset=presets.SD15_PRESET, base=None):
    args = parser().parse_args(list(arguments), namespace=base or base_arguments(preset))
    resolve_hires_options(preset, args)
    return args


class HiresOptionsTests(unittest.TestCase):
    def test_omitted_options_keep_hires_disabled_and_optional_values_null(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                args = resolved(preset=preset)
                self.assertFalse(args.hires_fix)
                for name in args._argument_names:
                    if name != "hires_fix":
                        self.assertIsNone(getattr(args, name), name)
                self.assertIsNone(args.hires_target_width)
                self.assertIsNone(args.hires_target_height)
                self.assertIsNone(args.hires_passes)
                self.assertIsNone(args.hires_stage_sizes)

    def test_selected_defaults_are_valid_for_every_family(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                args = resolved("--hires-fix", preset=preset)
                self.assertTrue(args.hires_fix)
                self.assertEqual(args.hires_passes, 1)
                self.assertEqual(args.hires_scale, 2.0)
                self.assertEqual(args.hires_upscaler, "lanczos")
                self.assertEqual(args.hires_denoising_strength, 0.35)
                self.assertEqual(args.hires_steps, preset.steps)
                self.assertEqual(args.hires_seed, 42)
                self.assertEqual(args.hires_guidance_scale, preset.guidance_scale)
                self.assertEqual(args.hires_true_cfg_scale, 1.0)
                self.assertEqual(args.hires_scheduler, "auto")
                self.assertEqual(args.hires_scheduler_config, {})
                self.assertFalse(args.hires_save_base)
                self.assertEqual((args.hires_target_width, args.hires_target_height),
                                 (preset.width * 2, preset.height * 2))
                self.assertEqual(args.hires_stage_sizes, [[preset.width * 2, preset.height * 2]])
                self.assertIsNone(args.hires_width)
                self.assertIsNone(args.hires_height)

    def test_all_dependent_options_require_hires_fix_including_explicit_false(self):
        cases = (
            ("--hires-passes", "1"), ("--hires-scale", "2"), ("--hires-width", "1024"),
            ("--hires-height", "1024"), ("--hires-upscaler", "nearest"),
            ("--hires-strength", "0.35"), ("--hires-steps", "20"),
            ("--hires-seed", "0"), ("--hires-guidance-scale", "0"),
            ("--hires-true-cfg-scale", "1"), ("--hires-scheduler", "auto"),
            ("--hires-scheduler-config", "{}"), ("--hires-save-base",),
            ("--no-hires-save-base",),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(SystemExit, "--hires-fix"):
                resolved(*arguments)

    def test_explicit_dimensions_preserve_the_original_request(self):
        args = resolved("--hires-fix", "--hires-width", "768", "--hires-height", "1024")
        self.assertEqual((args.hires_target_width, args.hires_target_height), (768, 1024))
        self.assertEqual((args.hires_width, args.hires_height), (768, 1024))
        self.assertIsNone(args.hires_scale)

    def test_one_dimension_preserves_aspect_ratio_with_nearest_model_multiple(self):
        base = base_arguments(presets.SD15_PRESET)
        base.width, base.height = 512, 768
        args = resolved("--hires-fix", "--hires-width", "776", base=base)
        self.assertEqual((args.hires_target_width, args.hires_target_height), (776, 1168))
        self.assertEqual(args.hires_width, 776)
        self.assertIsNone(args.hires_height)
        self.assertIsNone(args.hires_scale)
        args = resolved("--hires-fix", "--hires-height", "1168", base=base_arguments(presets.SD15_PRESET))
        self.assertEqual((args.hires_target_width, args.hires_target_height), (1168, 1168))

    def test_scaled_dimensions_round_to_each_familys_required_multiple(self):
        for preset in presets.PRESETS.values():
            expected = {"sd15": 664, "sdxl-base": 1328, "flux1-schnell": 1328}[preset.family]
            with self.subTest(preset=preset.name):
                args = resolved("--hires-fix", "--hires-scale", "1.3", preset=preset)
                self.assertEqual((args.hires_target_width, args.hires_target_height), (expected, expected))

    def test_repeated_refinement_uses_previous_rounded_size_and_preserves_base(self):
        args = resolved("--hires-fix", "--hires-passes", "3", "--hires-scale", "1.3")
        self.assertEqual(args.hires_stage_sizes, [[664, 664], [864, 864], [1120, 1120]])
        self.assertEqual((args.hires_target_width, args.hires_target_height), (1120, 1120))
        self.assertEqual((args.width, args.height), (512, 512))

    def test_explicit_first_target_repeats_each_axis_ratio(self):
        args = resolved("--hires-fix", "--hires-passes", "3", "--hires-width", "768", "--hires-height", "1024")
        self.assertEqual(args.hires_stage_sizes, [[768, 1024], [1152, 2048], [1728, 4096]])
        self.assertEqual((args.hires_width, args.hires_height), (768, 1024))
        self.assertIsNone(args.hires_scale)

    def test_single_explicit_axis_repeats_ratio_of_first_rounded_target(self):
        base = base_arguments(presets.SD15_PRESET)
        base.width, base.height = 512, 768
        args = resolved("--hires-fix", "--hires-passes", "3", "--hires-width", "776", base=base)
        self.assertEqual(args.hires_stage_sizes, [[776, 1168], [1176, 1776], [1784, 2704]])

    def test_stage_size_helper_needs_no_preset_and_rounds_for_model_multiple(self):
        self.assertEqual(resolve_hires_stage_sizes(1024, 1024, 3, scale=1.3, dimension_multiple=16),
                         [[1328, 1328], [1728, 1728], [2240, 2240]])
        self.assertEqual(resolve_hires_stage_sizes(512, 512, 2, target_width=512, target_height=1024),
                         [[512, 1024], [512, 2048]])

    def test_refinement_pass_count_is_a_positive_integer(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-passes", value)
        for value in (True, 2.0, None, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_hires_stage_sizes(512, 512, value)

    def test_pass_count_cannot_bypass_validation_from_json_or_python(self):
        for value in (True, 1.5, "3", 0, -1):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                generate.resolve_request({"hires_fix": True, "hires_passes": value})

    def test_programmatic_namespace_cannot_bypass_pass_count_validation(self):
        for value in (True, 2.0, "3", 0, -1):
            args = parser().parse_args(["--hires-fix"], namespace=base_arguments(presets.SD15_PRESET))
            args.hires_passes = value
            with self.subTest(value=value), self.assertRaises(SystemExit):
                resolve_hires_options(presets.SD15_PRESET, args)

    def test_full_python_request_exports_and_replays_repeated_refinement(self):
        preset, args = generate.resolve_request({"preset": "sd15", "hires_fix": True,
                                                "hires_passes": 3, "hires_scale": 1.3})
        exported = json.loads(json.dumps(configuration_values(args)))
        self.assertEqual(exported["hires_passes"], 3)
        replay_preset, replay = generate.resolve_request(exported)
        self.assertEqual(replay_preset, preset)
        self.assertEqual(replay.hires_stage_sizes, [[664, 664], [864, 864], [1120, 1120]])
        self.assertEqual((replay.width, replay.height), (512, 512))

    def test_cli_pass_count_overrides_json_configuration(self):
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hires-repeat-config-", dir=ROOT / "build") as directory:
            config = Path(directory) / "generation.json"
            config.write_text(json.dumps({"preset": "sd15", "hires_fix": True,
                                          "hires_passes": 3, "hires_scale": 1.3}))
            cli = generate.build_parser().parse_args(["--config", str(config), "--hires-passes", "2"])
            _, args = generate.resolve_arguments(cli)
            self.assertEqual(args.hires_passes, 2)
            self.assertEqual(args.hires_stage_sizes, [[664, 664], [864, 864]])
            self.assertEqual(args.hires_scale, 1.3)
            self.assertEqual((args.width, args.height), (512, 512))

    def test_refinement_growth_overflow_is_rejected_before_pipeline_loading(self):
        for values in ({"passes": 30}, {"passes": 10**100},
                       {"passes": 2, "target_width": 2**31}, {"passes": 2, "scale": 1e308}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "dimensions|image size"):
                resolve_hires_stage_sizes(512, 512, **values)

    def test_every_stage_requires_actual_growth_after_rounding(self):
        with self.assertRaisesRegex(ValueError, "enlarge"):
            resolve_hires_stage_sizes(512, 512, 5, scale=1.0001)
        with self.assertRaisesRegex(ValueError, "enlarge"):
            resolve_hires_stage_sizes(512, 512, 3, target_width=256, target_height=1024)

    def test_scale_and_explicit_size_are_mutually_exclusive(self):
        for dimension in ("--hires-width", "--hires-height"):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(SystemExit, "--hires-scale"):
                resolved("--hires-fix", "--hires-scale", "2", dimension, "1024")

    def test_final_image_must_not_shrink_and_must_enlarge_at_least_one_axis(self):
        for width, height in ((512, 512), (256, 1024), (1024, 256)):
            with self.subTest(width=width, height=height), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-width", str(width), "--hires-height", str(height))
        args = resolved("--hires-fix", "--hires-width", "512", "--hires-height", "1024")
        self.assertEqual((args.hires_target_width, args.hires_target_height), (512, 1024))

    def test_rounding_that_does_not_enlarge_is_rejected(self):
        with self.assertRaises(SystemExit):
            resolved("--hires-fix", "--hires-scale", "1.0001")

    def test_explicit_sizes_must_be_positive_model_multiples(self):
        for preset in presets.PRESETS.values():
            for dimension in ("--hires-width", "--hires-height"):
                for value in ("0", "-8", "1025"):
                    with self.subTest(preset=preset.name, dimension=dimension, value=value), self.assertRaises(SystemExit):
                        resolved("--hires-fix", dimension, value, preset=preset)

    def test_invalid_base_dimensions_fail_before_aspect_ratio_arithmetic(self):
        for name in ("width", "height"):
            base = base_arguments(presets.SD15_PRESET)
            setattr(base, name, 0)
            with self.subTest(name=name), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-height", "1024", base=base)

    def test_nonfinite_out_of_range_and_noop_strength_options_are_rejected(self):
        cases = (
            ("--hires-scale", "nan"), ("--hires-scale", "inf"),
            ("--hires-scale", "1"), ("--hires-scale", "0"),
            ("--hires-scale", "1e308"),
            ("--hires-strength", "0"), ("--hires-strength", "-0.1"),
            ("--hires-strength", "1.01"), ("--hires-strength", "nan"),
            ("--hires-strength", "inf"), ("--hires-steps", "0"),
            ("--hires-steps", "-1"), ("--hires-guidance-scale", "-1"),
            ("--hires-guidance-scale", "nan"), ("--hires-guidance-scale", "inf"),
            ("--hires-true-cfg-scale", "nan"), ("--hires-true-cfg-scale", "0.9"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                resolved("--hires-fix", *arguments)

    def test_strength_alias_and_one_denoising_step_boundary(self):
        for flag in ("--hires-strength", "--hires-denoising-strength"):
            args = resolved("--hires-fix", flag, "0.25", "--hires-steps", "4")
            self.assertEqual(args.hires_denoising_strength, 0.25)
            with self.assertRaisesRegex(SystemExit, "denoising step"):
                resolved("--hires-fix", flag, "0.249", "--hires-steps", "4")
        args = resolved("--hires-fix", "--hires-strength", "1", "--hires-steps", "1")
        self.assertEqual(args.hires_steps, 1)

    def test_flux_strength_uses_upstream_start_offset_rounding(self):
        for preset in presets.PRESETS.values():
            if preset.family == "flux1-schnell":
                for steps, strength in ((4, "0.1"), (4, "0.249"), (4, "0.25"), (4, "0.35"), (1, "0.1")):
                    with self.subTest(preset=preset.name, steps=steps, strength=strength):
                        args = resolved("--hires-fix", "--hires-strength", strength,
                                        "--hires-steps", str(steps), preset=preset)
                        self.assertEqual(args.hires_denoising_strength, float(strength))
                        self.assertEqual(args.hires_steps, steps)
            else:
                with self.subTest(preset=preset.name), self.assertRaisesRegex(SystemExit, "denoising step"):
                    resolved("--hires-fix", "--hires-strength", "0.1", "--hires-steps", "4", preset=preset)

    def test_flux_strength_that_rounds_to_an_empty_schedule_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "denoising step"):
            resolved("--hires-fix", "--hires-strength", "1e-100", "--hires-steps", "4",
                     preset=presets.FLUX1_SCHNELL_PRESET)

    def test_batch_seed_stride_is_checked_for_both_range_edges(self):
        for seed, stride in ((2**64 - 1, 1), (-(2**63), -1)):
            base = base_arguments(presets.SD15_PRESET)
            base.num_images, base.seed_stride = 2, stride
            with self.subTest(seed=seed, stride=stride), self.assertRaisesRegex(SystemExit, "seed"):
                resolved("--hires-fix", "--hires-seed", str(seed), base=base)
        for seed in (2**64, -(2**63) - 1):
            with self.subTest(seed=seed), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-seed", str(seed))

    def test_valid_seed_range_and_zero_values_are_preserved(self):
        for seed in (0, 2**64 - 1, -(2**63)):
            args = resolved("--hires-fix", "--hires-seed", str(seed), "--hires-guidance-scale", "0")
            self.assertEqual(args.hires_seed, seed)
            self.assertEqual(args.hires_guidance_scale, 0.0)

    def test_guidance_restrictions_distinguish_schnell_from_guidance_distilled_flux(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                if preset.family == "flux1-schnell" and not preset.requires_guidance_embeds:
                    with self.assertRaisesRegex(SystemExit, "FLUX"):
                        resolved("--hires-fix", "--hires-guidance-scale", "3.5", preset=preset)
                else:
                    args = resolved("--hires-fix", "--hires-guidance-scale", "3.5", preset=preset)
                    self.assertEqual(args.hires_guidance_scale, 3.5)
                if preset.family == "flux1-schnell":
                    args = resolved("--hires-fix", "--hires-true-cfg-scale", "2", preset=preset)
                    self.assertEqual(args.hires_true_cfg_scale, 2.0)
                else:
                    with self.assertRaisesRegex(SystemExit, "FLUX"):
                        resolved("--hires-fix", "--hires-true-cfg-scale", "2", preset=preset)

    def test_defaults_inherit_base_sampling_seed_and_guidance(self):
        base = base_arguments(presets.FLUX1_SCHNELL_PRESET)
        base.steps, base.seed, base.seed_stride, base.true_cfg_scale = 12, 7, -1, 3.0
        base.scheduler, base.scheduler_config = "FlowMatchEulerDiscreteScheduler", {"shift": 2.0}
        args = resolved("--hires-fix", preset=presets.FLUX1_SCHNELL_PRESET, base=base)
        self.assertEqual((args.hires_steps, args.hires_seed, args.hires_true_cfg_scale), (12, 7, 3.0))
        self.assertEqual(args.hires_scheduler, "auto")
        self.assertEqual(args.hires_scheduler_config, {})

    def test_upscaler_choices_scheduler_overrides_and_base_retention_are_exposed(self):
        for upscaler in ("nearest", "bilinear", "bicubic", "lanczos"):
            args = resolved("--hires-fix", "--hires-upscaler", upscaler,
                            "--hires-scheduler", "DDIMScheduler",
                            "--hires-scheduler-config", '{"timestep_spacing":"trailing"}',
                            "--hires-save-base")
            self.assertEqual(args.hires_upscaler, upscaler)
            self.assertEqual(args.hires_scheduler, "DDIMScheduler")
            self.assertEqual(args.hires_scheduler_config, {"timestep_spacing": "trailing"})
            self.assertTrue(args.hires_save_base)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            resolved("--hires-fix", "--hires-upscaler", "unknown")

    def test_scheduler_name_and_strict_json_object_are_validated(self):
        for scheduler in ("", "_private"):
            with self.subTest(scheduler=scheduler), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-scheduler", scheduler)
        for value in ('[]', '{"shift":NaN}', '{"shift":1,"shift":2}'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resolved("--hires-fix", "--hires-scheduler-config", value)

    def test_programmatic_scheduler_object_cannot_bypass_json_validation(self):
        for value in ([], {"shift": float("inf")}, {"shift": object()}):
            args = parser().parse_args(["--hires-fix"], namespace=base_arguments(presets.SD15_PRESET))
            args.hires_scheduler_config = value
            with self.subTest(value=value), self.assertRaises(SystemExit):
                resolve_hires_options(presets.SD15_PRESET, args)

    def test_exported_scaled_explicit_and_disabled_requests_replay(self):
        for arguments in ((), ("--hires-fix",), ("--hires-fix", "--hires-width", "768"),
                          ("--hires-fix", "--hires-width", "768", "--hires-height", "1024"),
                          ("--hires-fix", "--hires-passes", "3", "--hires-scale", "1.3"),
                          ("--hires-fix", "--hires-passes", "4", "--hires-width", "768")):
            with self.subTest(arguments=arguments):
                args = resolved(*arguments)
                values = {name: getattr(args, name) for name in args._argument_names}
                replay = parser().parse_values(json.loads(json.dumps(values)))
                replay = argparse.Namespace(**(vars(base_arguments(presets.SD15_PRESET)) | vars(replay)))
                resolve_hires_options(presets.SD15_PRESET, replay)
                self.assertEqual((replay.hires_target_width, replay.hires_target_height),
                                 (args.hires_target_width, args.hires_target_height))
                self.assertEqual({name: getattr(replay, name) for name in args._argument_names}, values)
                self.assertEqual(replay.hires_stage_sizes, args.hires_stage_sizes)

    def test_resolving_twice_is_idempotent(self):
        for arguments in (("--hires-fix",), ("--hires-fix", "--hires-width", "1024"),
                          ("--hires-fix", "--hires-passes", "3", "--hires-scale", "1.3"), ()):
            args = resolved(*arguments)
            before = dict(vars(args))
            resolve_hires_options(presets.SD15_PRESET, args)
            self.assertEqual(vars(args), before)

    def test_second_pass_copies_settings_and_resets_first_pass_only_inputs(self):
        base = base_arguments(presets.SD15_PRESET)
        base.timesteps, base.sigmas, base.denoising_end = [900, 500], [1.0, 0.5], 0.8
        base.latents, base.latents_file, base.tensor_inputs = "initial.safetensors", object(), object()
        base.embeddings, base.embeddings_file = "conditioning.safetensors", object()
        base.guidance_rescale = 0.3
        base.prompt, base.lora_selection, base.controlnet_selection = "subject", object(), object()
        args = resolved("--hires-fix", "--hires-steps", "30", "--hires-seed", "12",
                        "--hires-guidance-scale", "4", "--hires-scheduler", "DDIMScheduler", base=base)
        second = hires_request(presets.SD15_PRESET, args)
        self.assertIsNot(second, args)
        self.assertEqual((second.width, second.height, second.steps, second.seed, second.guidance_scale),
                         (1024, 1024, 30, 12, 4.0))
        self.assertEqual(second.scheduler, "DDIMScheduler")
        self.assertEqual(second.scheduler_config, {})
        self.assertEqual(second.guidance_rescale, 0.0)
        self.assertEqual(args.guidance_rescale, 0.3)
        for name in ("timesteps", "sigmas", "denoising_end", "latents_file", "tensor_inputs"):
            self.assertIsNone(getattr(second, name), name)
        for name in ("embeddings", "embeddings_file", "prompt", "lora_selection", "controlnet_selection"):
            self.assertEqual(getattr(second, name), getattr(args, name), name)
        self.assertEqual((args.width, args.height, args.seed, args.steps), (512, 512, 42, 20))
        self.assertIsNotNone(args.latents_file)
        self.assertIsNotNone(args.tensor_inputs)

    def test_second_pass_scheduler_overrides_do_not_alias_original(self):
        args = resolved("--hires-fix", "--hires-scheduler-config", '{"shift":1.5}')
        second = hires_request(presets.SD15_PRESET, args)
        second.scheduler_config["shift"] = 2.0
        self.assertEqual(args.hires_scheduler_config, {"shift": 1.5})

    def test_stage_requests_use_indexed_sizes_and_repeat_sampling_settings(self):
        args = resolved("--hires-fix", "--hires-passes", "3", "--hires-width", "768",
                        "--hires-steps", "30", "--hires-seed", "12", "--hires-guidance-scale", "4")
        before = (args.width, args.height, args.steps, args.seed, args.guidance_scale)
        for index, expected in enumerate(args.hires_stage_sizes):
            request = hires_request(presets.SD15_PRESET, args, pass_index=index)
            self.assertEqual([request.width, request.height], expected)
            self.assertEqual((request.steps, request.seed, request.guidance_scale), (30, 12, 4.0))
        self.assertEqual((args.width, args.height, args.steps, args.seed, args.guidance_scale), before)

    def test_stage_request_rejects_out_of_range_or_noninteger_index(self):
        args = resolved("--hires-fix", "--hires-passes", "3")
        for index in (-1, 3, True, 1.0, "1"):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "pass_index"):
                hires_request(presets.SD15_PRESET, args, pass_index=index)

    def test_stage_request_does_not_alias_original_stage_sizes(self):
        args = resolved("--hires-fix", "--hires-passes", "3")
        request = hires_request(presets.SD15_PRESET, args, pass_index=1)
        request.hires_stage_sizes[0][0] = 8
        self.assertEqual(args.hires_stage_sizes[0], [1024, 1024])

    def test_sdxl_each_stage_updates_size_conditioning(self):
        preset = presets.SDXL_BASE_PRESET
        args = resolved("--hires-fix", "--hires-passes", "3", preset=preset)
        for index, (width, height) in enumerate(args.hires_stage_sizes):
            request = hires_request(preset, args, pass_index=index)
            for name in SDXL_SIZES:
                self.assertEqual(getattr(request, name), (height, width))

    def test_sdxl_second_pass_updates_size_conditioning_but_retains_crops(self):
        preset = presets.SDXL_BASE_PRESET
        base = base_arguments(preset)
        for name in SDXL_SIZES:
            setattr(base, name, (512, 512))
        for name in SDXL_CROPS:
            setattr(base, name, (8, 16))
        args = resolved("--hires-fix", "--hires-width", "1536", "--hires-height", "2048",
                        preset=preset, base=base)
        second = hires_request(preset, args)
        for name in SDXL_SIZES:
            self.assertEqual(getattr(second, name), (2048, 1536))
            self.assertEqual(getattr(args, name), (512, 512))
        for name in SDXL_CROPS:
            self.assertEqual(getattr(second, name), (8, 16))

    def test_disabled_hires_cannot_produce_a_second_pass_request(self):
        with self.assertRaisesRegex(ValueError, "--hires-fix"):
            hires_request(presets.SD15_PRESET, resolved())

    def test_flux_negative_prompts_can_be_reserved_for_second_pass_cfg(self):
        for negatives in ({"negative_prompt": "blurry"},
                          {"negative_prompt_2": "secondary artifacts"},
                          {"negative_prompt": "blurry", "negative_prompt_2": "secondary artifacts"}):
            with self.subTest(negatives=negatives):
                preset, args = generate.resolve_request({
                    "preset": "flux1-schnell", "true_cfg_scale": 1,
                    "hires_fix": True, "hires_true_cfg_scale": 2, **negatives,
                })
                original_negatives = (args.negative_prompt, args.negative_prompt_2)
                first = generate.build_pipeline_call_arguments(preset, args, "generator")
                second_request = hires_request(preset, args)
                second = generate.build_pipeline_call_arguments(preset, second_request, "generator")
                self.assertNotIn("negative_prompt", first)
                self.assertNotIn("negative_prompt_2", first)
                self.assertEqual(second["negative_prompt"], original_negatives[0])
                self.assertEqual(second["negative_prompt_2"], original_negatives[1])
                self.assertEqual(second["true_cfg_scale"], 2)
                self.assertEqual((args.negative_prompt, args.negative_prompt_2), original_negatives)
                self.assertEqual(args.true_cfg_scale, 1)

    def test_flux_unused_negative_prompts_remain_invalid_with_one_or_two_stages(self):
        for hires in ({}, {"hires_fix": True, "hires_true_cfg_scale": 1}):
            for negatives in ({"negative_prompt": "blurry"}, {"negative_prompt_2": "artifacts"}):
                with self.subTest(hires=hires, negatives=negatives), self.assertRaisesRegex(
                        SystemExit, "negative-prompt"):
                    generate.resolve_request({"preset": "flux1-schnell", "true_cfg_scale": 1,
                                              **hires, **negatives})

    def test_flux_base_cfg_can_use_negative_prompts_when_refinement_cfg_is_disabled(self):
        preset, args = generate.resolve_request({
            "preset": "flux1-schnell", "true_cfg_scale": 2, "negative_prompt": "blurry",
            "hires_fix": True, "hires_true_cfg_scale": 1,
        })
        first = generate.build_pipeline_call_arguments(preset, args, "generator")
        second = generate.build_pipeline_call_arguments(preset, hires_request(preset, args), "generator")
        self.assertEqual(first["negative_prompt"], "blurry")
        self.assertNotIn("negative_prompt", second)
        self.assertNotIn("negative_prompt_2", second)


if __name__ == "__main__":
    unittest.main()
