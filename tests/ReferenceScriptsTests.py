#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "diffusers"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import presets
import pipeline_loading


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class ReferenceScriptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generate = load_module("iild_reference_generate", REFERENCE / "generate.py")
        cls.inspect = load_module("iild_reference_inspect", REFERENCE / "inspect_pipeline.py")

    def test_model_revision_is_shared_and_immutable(self) -> None:
        self.assertEqual(self.generate.MODEL_ID, self.inspect.MODEL_ID)
        self.assertEqual(self.generate.MODEL_REVISION, self.inspect.MODEL_REVISION)
        self.assertRegex(self.generate.MODEL_REVISION, r"^[0-9a-f]{40}$")

        for preset in presets.PRESETS.values():
            self.assertRegex(preset.revision, r"^[0-9a-f]{40}$")

    def test_invalid_runtime_policy_is_rejected(self) -> None:
        invalid_runtime = replace(
            presets.FLUX1_SCHNELL_PRESET.runtime,
            accelerator_execution="misspelled-policy",
        )
        invalid_preset = replace(
            presets.FLUX1_SCHNELL_PRESET,
            runtime=invalid_runtime,
        )
        with self.assertRaisesRegex(RuntimeError, "accelerator execution policy"):
            presets.validate_preset_definition(invalid_preset)

    def test_fixed_fixture_contract(self) -> None:
        self.assertEqual(self.generate.DEFAULT_PROMPT, "a red cube on a white table")
        self.assertEqual(self.generate.DEFAULT_NEGATIVE_PROMPT, "")
        self.assertEqual(self.generate.DEFAULT_SEED, 42)
        self.assertEqual((self.generate.DEFAULT_WIDTH, self.generate.DEFAULT_HEIGHT), (512, 512))
        self.assertEqual(self.generate.DEFAULT_STEPS, 20)
        self.assertEqual(self.generate.DEFAULT_GUIDANCE_SCALE, 7.5)

    def test_reference_dependencies_are_pinned(self) -> None:
        requirements = (REFERENCE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("-r requirements-common.txt", requirements)
        self.assertIn("torch==2.13.0", requirements)
        requirements += (REFERENCE / "requirements-common.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("hf-xet==1.6.0", requirements)
        self.assertIn("peft==0.20.0", requirements)

    def test_peft_is_preflighted_only_when_lora_is_requested(self) -> None:
        def package_version(name: str) -> str:
            if name == "peft":
                raise self.generate.importlib.metadata.PackageNotFoundError
            return f"{name}-version"

        with patch.object(
            self.generate.importlib.metadata,
            "version",
            side_effect=package_version,
        ):
            versions = self.generate.package_versions(require_lora=False)
            self.assertNotIn("peft", versions)
            with self.assertRaisesRegex(SystemExit, "LoRA generation requires PEFT"):
                self.generate.package_versions(require_lora=True)

    def test_generated_files_and_cache_stay_under_build(self) -> None:
        build_directory = ROOT / "build"
        self.assertTrue(self.generate.DEFAULT_OUTPUT.is_relative_to(build_directory))
        self.assertTrue(self.generate.DEFAULT_CACHE_DIRECTORY.is_relative_to(build_directory))
        self.assertTrue(self.generate.DEFAULT_XET_CACHE_DIRECTORY.is_relative_to(build_directory))
        self.assertTrue(self.inspect.DEFAULT_OUTPUT.is_relative_to(build_directory))
        self.assertTrue(self.inspect.DEFAULT_CACHE_DIRECTORY.is_relative_to(build_directory))
        self.assertTrue(self.inspect.DEFAULT_XET_CACHE_DIRECTORY.is_relative_to(build_directory))

        for preset in presets.PRESETS.values():
            _, generated = self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(["--preset", preset.name])
            )
            _, inspected = self.inspect.resolve_arguments(
                self.inspect.build_parser().parse_args(["--preset", preset.name])
            )
            self.assertTrue(generated.output.is_relative_to(build_directory))
            self.assertTrue(inspected.output.is_relative_to(build_directory))

    def test_default_preset_preserves_sd15_contract(self) -> None:
        preset, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args([])
        )

        self.assertEqual(preset.name, "sd15")
        self.assertEqual(arguments.model_selection.source, self.generate.MODEL_ID)
        self.assertEqual(
            arguments.model_selection.requested_revision,
            self.generate.MODEL_REVISION,
        )
        self.assertEqual((arguments.width, arguments.height), (512, 512))
        self.assertEqual(arguments.steps, 20)
        self.assertEqual(arguments.guidance_scale, 7.5)
        self.assertEqual(arguments.output, self.generate.DEFAULT_OUTPUT)

    def test_sdxl_preset_uses_dual_text_contract(self) -> None:
        preset, generated = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(["--preset", "sdxl-base"])
        )
        inspected_preset, inspected = self.inspect.resolve_arguments(
            self.inspect.build_parser().parse_args(["--preset", "sdxl-base"])
        )

        self.assertIs(preset, inspected_preset)
        self.assertEqual(preset.pipeline_class, "StableDiffusionXLPipeline")
        self.assertEqual((generated.width, generated.height), (1024, 1024))
        self.assertEqual(generated.guidance_scale, 5.0)
        self.assertEqual(generated.model_selection.requested_revision, preset.revision)
        self.assertEqual(generated.output.name, "sdxl-base-red-cube.png")
        self.assertEqual(inspected.output.name, "pipeline-inspection-sdxl-base.json")
        self.assertEqual(
            dict(preset.expected_components),
            {
                "tokenizer": "CLIPTokenizer",
                "tokenizer_2": "CLIPTokenizer",
                "text_encoder": "CLIPTextModel",
                "text_encoder_2": "CLIPTextModelWithProjection",
                "unet": "UNet2DConditionModel",
                "vae": "AutoencoderKL",
                "scheduler": "EulerDiscreteScheduler",
            },
        )

    def test_flux1_schnell_preset_uses_transformer_contract(self) -> None:
        preset, generated = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(["--preset", "flux1-schnell"])
        )
        inspected_preset, inspected = self.inspect.resolve_arguments(
            self.inspect.build_parser().parse_args(["--preset", "flux1-schnell"])
        )

        self.assertIs(preset, inspected_preset)
        self.assertEqual(preset.pipeline_class, "FluxPipeline")
        self.assertEqual((generated.width, generated.height), (1024, 1024))
        self.assertEqual(generated.steps, 4)
        self.assertEqual(generated.guidance_scale, 0.0)
        self.assertEqual(generated.model_selection.requested_revision, preset.revision)
        self.assertEqual(generated.output.name, "flux1-schnell-red-cube.png")
        self.assertEqual(inspected.output.name, "pipeline-inspection-flux1-schnell.json")
        self.assertEqual(preset.runtime.accelerator_dtype, "bfloat16")
        self.assertIsNone(preset.runtime.weight_variant)
        self.assertEqual(preset.runtime.dimension_multiple, 16)
        self.assertEqual(preset.runtime.max_sequence_length, 256)
        self.assertFalse(preset.runtime.passes_negative_prompt)
        self.assertEqual(
            dict(preset.expected_components),
            {
                "tokenizer": "CLIPTokenizer",
                "tokenizer_2": "T5TokenizerFast",
                "text_encoder": "CLIPTextModel",
                "text_encoder_2": "T5EncoderModel",
                "transformer": "FluxTransformer2DModel",
                "vae": "AutoencoderKL",
                "scheduler": "FlowMatchEulerDiscreteScheduler",
            },
        )

    def test_flux_load_and_generation_arguments_are_canonical(self) -> None:
        preset, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(["--preset", "flux1-schnell"])
        )
        load_arguments = self.generate.build_load_arguments(
            preset, arguments, "torch.bfloat16", use_weight_variant=True
        )
        self.assertEqual(load_arguments["dtype"], "torch.bfloat16")
        self.assertNotIn("variant", load_arguments)
        self.assertNotIn("add_watermarker", load_arguments)

        call_arguments = self.generate.build_pipeline_call_arguments(
            preset, arguments, "generator"
        )
        self.assertNotIn("negative_prompt", call_arguments)
        self.assertEqual(call_arguments["max_sequence_length"], 256)
        self.assertEqual(call_arguments["num_inference_steps"], 4)
        self.assertEqual(call_arguments["guidance_scale"], 0.0)
        self.assertEqual(call_arguments["output_type"], "pil")

    def test_flux_generation_argument_guards(self) -> None:
        preset, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                ["--preset", "flux1-schnell", "--width", "1000"]
            )
        )
        with self.assertRaisesRegex(SystemExit, "multiples of 16"):
            self.generate.validate_generation_arguments(preset, arguments)

        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                ["--preset", "flux1-schnell", "--steps", "5"]
            )
        )
        self.generate.validate_generation_arguments(preset, arguments)

        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                ["--preset", "flux1-schnell", "--guidance-scale", "1.0"]
            )
        )
        with self.assertRaisesRegex(SystemExit, "must be 0.0"):
            self.generate.validate_generation_arguments(preset, arguments)

        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                ["--preset", "flux1-schnell", "--negative-prompt", "blur"]
            )
        )
        with self.assertRaisesRegex(SystemExit, "does not use --negative-prompt"):
            self.generate.validate_generation_arguments(preset, arguments)

    def test_flux_mps_policy_uses_sequential_offload(self) -> None:
        events: list[str] = []

        class Vae:
            def enable_slicing(self):
                events.append("vae-slicing")

            def enable_tiling(self):
                events.append("vae-tiling")

        class Pipeline:
            vae = Vae()

            def enable_sequential_cpu_offload(self, *, device):
                events.append(f"offload:{device}")

            def to(self, device):
                raise AssertionError(f"FLUX MPS policy must not call to({device})")

        pipeline = Pipeline()
        returned, optimization = self.generate.prepare_pipeline_for_execution(
            pipeline, presets.FLUX1_SCHNELL_PRESET, "mps", False
        )
        self.assertIs(returned, pipeline)
        self.assertEqual(events, ["vae-slicing", "vae-tiling", "offload:mps"])
        self.assertEqual(optimization["offload_policy"], "sequential-cpu")
        self.assertTrue(optimization["vae_slicing_enabled"])
        self.assertTrue(optimization["vae_tiling_enabled"])

        events.clear()
        returned, optimization = self.generate.prepare_pipeline_for_execution(
            pipeline, presets.FLUX1_SCHNELL_PRESET, "cuda", False
        )
        self.assertIs(returned, pipeline)
        self.assertEqual(events, ["vae-slicing", "vae-tiling", "offload:cuda"])
        self.assertEqual(optimization["offload_policy"], "sequential-cpu")

        self.assertFalse(
            self.generate.resolve_attention_slicing(
                presets.FLUX1_SCHNELL_PRESET, "mps", None
            )
        )
        with self.assertRaisesRegex(SystemExit, "not supported"):
            self.generate.resolve_attention_slicing(
                presets.FLUX1_SCHNELL_PRESET, "mps", True
            )

    def test_explicit_output_and_local_model_override(self) -> None:
        build_directory = ROOT / "build"
        build_directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_directory) as temporary:
            local_model = Path(temporary)
            output = build_directory / "reference" / "custom.png"
            _, arguments = self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    ["--model", str(local_model), "--output", str(output)]
                )
            )

        self.assertTrue(arguments.model_selection.is_local)
        self.assertIsNone(arguments.model_selection.requested_revision)
        self.assertEqual(arguments.output, output)

    def test_local_lora_file_is_resolved_and_hashed(self) -> None:
        build_directory = ROOT / "build"
        build_directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_directory) as temporary:
            lora_file = Path(temporary) / "style.safetensors"
            lora_file.write_bytes(b"adapter")
            _, arguments = self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    ["--lora", str(lora_file), "--lora-scale", "0.65"]
                )
            )

            selection = arguments.lora_selection
            self.assertIsNotNone(selection)
            self.assertTrue(selection.is_local)
            self.assertEqual(selection.source, str(lora_file.parent.resolve()))
            self.assertEqual(selection.weight_name, "style.safetensors")
            self.assertIsNone(selection.requested_revision)
            self.assertEqual(selection.scale, 0.65)
            self.assertEqual(
                selection.sha256,
                hashlib.sha256(b"adapter").hexdigest(),
            )
            self.assertEqual(selection.size_bytes, 7)
            self.assertEqual(arguments.output.name, "sd15-red-cube-lora.png")
            activation = self.generate.LoraActivation(
                (self.generate.LORA_ADAPTER_NAME,),
                ("transformer",),
            )
            metadata = self.generate.lora_metadata(selection, activation)
            self.assertEqual(metadata["resolved_file"], str(lora_file.resolve()))
            self.assertEqual(metadata["sha256"], selection.sha256)
            self.assertEqual(metadata["size_bytes"], 7)

    def test_local_lora_directory_requires_exact_safetensors_name(self) -> None:
        build_directory = ROOT / "build"
        build_directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_directory) as temporary:
            directory = Path(temporary)
            weight = directory / "adapter.safetensors"
            weight.write_bytes(b"adapter")
            with self.assertRaisesRegex(SystemExit, "--lora-weight-name"):
                self.generate.resolve_arguments(
                    self.generate.build_parser().parse_args(
                        ["--lora", str(directory)]
                    )
                )

            _, arguments = self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    [
                        "--lora",
                        str(directory),
                        "--lora-weight-name",
                        weight.name,
                    ]
                )
            )
            self.assertEqual(arguments.lora_selection.weight_name, weight.name)

    def test_local_lora_changed_during_load_is_rejected(self) -> None:
        build_directory = ROOT / "build"
        build_directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_directory) as temporary:
            lora_file = Path(temporary) / "style.safetensors"
            lora_file.write_bytes(b"adapter")
            _, arguments = self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    ["--lora", str(lora_file)]
                )
            )

            class MutatingPipeline:
                def load_lora_weights(self, source, **kwargs):
                    lora_file.write_bytes(b"changed")

                def set_adapters(self, adapter_name, *, adapter_weights):
                    raise AssertionError("A changed LoRA must not be activated")

            with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
                self.generate.apply_lora(
                    MutatingPipeline(),
                    arguments.lora_selection,
                    arguments.cache_dir,
                    False,
                )

    def test_remote_lora_requires_immutable_revision_and_weight_name(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--lora-revision"):
            self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    ["--lora", "vendor/style-lora"]
                )
            )

        with self.assertRaisesRegex(SystemExit, "--lora-weight-name"):
            self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(
                    [
                        "--lora",
                        "vendor/style-lora",
                        "--lora-revision",
                        "b" * 40,
                    ]
                )
            )

        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                [
                    "--lora",
                    "vendor/style-lora",
                    "--lora-revision",
                    "b" * 40,
                    "--lora-weight-name",
                    "style.safetensors",
                ]
            )
        )
        selection = arguments.lora_selection
        self.assertFalse(selection.is_local)
        self.assertEqual(selection.requested_revision, "b" * 40)
        self.assertIsNone(selection.sha256)

    def test_lora_input_rejects_unsafe_or_ambiguous_values(self) -> None:
        parser = self.generate.build_parser()
        for invalid_scale in ("nan", "inf", "-inf"):
            with self.subTest(scale=invalid_scale), self.assertRaisesRegex(
                SystemExit, "finite"
            ):
                self.generate.resolve_arguments(
                    parser.parse_args(
                        [
                            "--lora",
                            "vendor/style-lora",
                            "--lora-revision",
                            "b" * 40,
                            "--lora-weight-name",
                            "style.safetensors",
                            f"--lora-scale={invalid_scale}",
                        ]
                    )
                )
        with self.assertRaisesRegex(SystemExit, "safetensors filename"):
            self.generate.resolve_arguments(
                parser.parse_args(
                    [
                        "--lora",
                        "vendor/style-lora",
                        "--lora-revision",
                        "b" * 40,
                        "--lora-weight-name",
                        "../style.safetensors",
                    ]
                )
            )
        with self.assertRaisesRegex(SystemExit, "requires --lora"):
            self.generate.resolve_arguments(
                parser.parse_args(["--lora-weight-name", "style.safetensors"])
            )
        with self.assertRaisesRegex(SystemExit, "requires --lora"):
            self.generate.resolve_arguments(
                parser.parse_args(["--lora-scale", "1.0"])
            )
        with self.assertRaisesRegex(SystemExit, "must not be empty"):
            self.generate.resolve_arguments(parser.parse_args(["--lora", ""]))
        with self.assertRaisesRegex(SystemExit, "40-character lowercase"):
            self.generate.resolve_arguments(
                parser.parse_args(
                    [
                        "--lora",
                        "vendor/style-lora",
                        "--lora-revision",
                        "main",
                        "--lora-weight-name",
                        "style.safetensors",
                    ]
                )
            )

        build_directory = ROOT / "build"
        build_directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_directory) as temporary:
            directory = Path(temporary)
            unsafe_file = directory / "adapter.bin"
            unsafe_file.write_bytes(b"pickle")
            with self.assertRaisesRegex(SystemExit, "safetensors filename"):
                self.generate.resolve_arguments(
                    parser.parse_args(["--lora", str(unsafe_file)])
                )

            uppercase_file = directory / "adapter.SAFETENSORS"
            uppercase_file.write_bytes(b"adapter")
            with self.assertRaisesRegex(SystemExit, "safetensors filename"):
                self.generate.resolve_arguments(
                    parser.parse_args(["--lora", str(uppercase_file)])
                )

            empty_file = directory / "empty.safetensors"
            empty_file.touch()
            with self.assertRaisesRegex(SystemExit, "missing or empty"):
                self.generate.resolve_arguments(
                    parser.parse_args(["--lora", str(empty_file)])
                )

            missing_file = directory / "missing.safetensors"
            with self.assertRaisesRegex(SystemExit, "does not exist"):
                self.generate.resolve_arguments(
                    parser.parse_args(["--lora", str(missing_file)])
                )

        for finite_scale in ("0", "-0.5", "1.5"):
            with self.subTest(finite_scale=finite_scale):
                _, arguments = self.generate.resolve_arguments(
                    parser.parse_args(
                        [
                            "--lora",
                            "vendor/style-lora",
                            "--lora-revision",
                            "b" * 40,
                            "--lora-weight-name",
                            "style.safetensors",
                            "--lora-scale",
                            finite_scale,
                        ]
                    )
                )
                self.assertEqual(
                    arguments.lora_selection.scale,
                    float(finite_scale),
                )

    def test_lora_is_loaded_safely_and_activated_before_execution(self) -> None:
        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                [
                    "--lora",
                    "vendor/style-lora",
                    "--lora-revision",
                    "b" * 40,
                    "--lora-weight-name",
                    "style.safetensors",
                    "--lora-scale",
                    "0.7",
                    "--local-files-only",
                ]
            )
        )
        events = []

        class Transformer:
            def active_adapters(self):
                return ["iild_lora"]

        class Pipeline:
            transformer = Transformer()

            def load_lora_weights(self, source, **kwargs):
                events.append(("load", source, kwargs))

            def set_adapters(self, adapter_name, *, adapter_weights):
                events.append(("activate", adapter_name, adapter_weights))

            def get_list_adapters(self):
                return {"transformer": ["iild_lora"], "text_encoder": []}

        pipeline = Pipeline()
        activation = self.generate.apply_lora(
            pipeline,
            arguments.lora_selection,
            arguments.cache_dir,
            arguments.local_files_only,
        )
        self.assertEqual(
            activation.active_adapters,
            (self.generate.LORA_ADAPTER_NAME,),
        )
        self.assertEqual(activation.registered_components, ("transformer",))
        self.assertEqual(events[0][0:2], ("load", "vendor/style-lora"))
        self.assertEqual(
            events[0][2],
            {
                "adapter_name": self.generate.LORA_ADAPTER_NAME,
                "cache_dir": arguments.cache_dir,
                "local_files_only": True,
                "low_cpu_mem_usage": True,
                "revision": "b" * 40,
                "use_safetensors": True,
                "weight_name": "style.safetensors",
            },
        )
        self.assertEqual(
            events[1],
            ("activate", self.generate.LORA_ADAPTER_NAME, 0.7),
        )

        metadata = self.generate.lora_metadata(arguments.lora_selection, activation)
        self.assertEqual(metadata["active_adapters"], [self.generate.LORA_ADAPTER_NAME])
        self.assertEqual(metadata["scale"], 0.7)
        self.assertEqual(metadata["source"], "vendor/style-lora")
        self.assertEqual(metadata["registered_components"], ["transformer"])
        self.assertFalse(metadata["fused"])
        self.assertEqual(metadata["type"], "lora")
        self.assertIsNone(metadata["resolved_file"])

    def test_pipeline_contract_lora_and_offload_order_is_fixed(self) -> None:
        events: list[str] = []
        pipeline = object()
        activation = self.generate.LoraActivation(("iild_lora",), ("unet",))
        arguments = SimpleNamespace(
            lora_selection="selection",
            cache_dir=Path("cache"),
            local_files_only=True,
        )

        def validate(candidate, preset):
            self.assertIs(candidate, pipeline)
            events.append("validate")

        def apply(candidate, selection, cache_directory, local_files_only):
            self.assertIs(candidate, pipeline)
            self.assertEqual(selection, "selection")
            self.assertTrue(local_files_only)
            events.append("lora")
            return activation

        def prepare(candidate, preset, device, attention_slicing, *, offload="auto"):
            self.assertIs(candidate, pipeline)
            self.assertEqual(device, "mps")
            self.assertFalse(attention_slicing)
            events.append("offload")
            return candidate, {"offload_policy": "sequential-cpu"}

        with (
            patch.object(self.generate, "validate_pipeline_contract", side_effect=validate),
            patch.object(self.generate, "apply_lora", side_effect=apply),
            patch.object(
                self.generate,
                "prepare_pipeline_for_execution",
                side_effect=prepare,
            ),
        ):
            returned, optimization, returned_activation, conditioning = (
                self.generate.prepare_pipeline_with_adapters(
                    pipeline,
                    presets.SD15_PRESET,
                    arguments,
                    "mps",
                    False,
                )
            )

        self.assertIs(returned, pipeline)
        self.assertEqual(events, ["validate", "lora", "offload"])
        self.assertEqual(optimization["offload_policy"], "sequential-cpu")
        self.assertIs(returned_activation, activation)
        self.assertIsNone(conditioning)

    def test_lora_must_be_registered_and_active(self) -> None:
        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                [
                    "--lora",
                    "vendor/style-lora",
                    "--lora-revision",
                    "b" * 40,
                    "--lora-weight-name",
                    "style.safetensors",
                ]
            )
        )

        class InactivePipeline:
            class Transformer:
                @staticmethod
                def active_adapters():
                    return []

            transformer = Transformer()

            def load_lora_weights(self, source, **kwargs):
                pass

            def set_adapters(self, adapter_name, *, adapter_weights):
                pass

            def get_list_adapters(self):
                return {"transformer": ["iild_lora"]}

        with self.assertRaisesRegex(RuntimeError, "not active on transformer"):
            self.generate.apply_lora(
                InactivePipeline(),
                arguments.lora_selection,
                arguments.cache_dir,
                False,
            )

    def test_text_encoder_only_lora_uses_component_active_state(self) -> None:
        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                [
                    "--lora",
                    "vendor/text-style-lora",
                    "--lora-revision",
                    "b" * 40,
                    "--lora-weight-name",
                    "style.safetensors",
                ]
            )
        )

        class TextEncoder:
            @staticmethod
            def active_adapters():
                return ["iild_lora"]

        class TextOnlyPipeline:
            text_encoder = TextEncoder()

            def load_lora_weights(self, source, **kwargs):
                pass

            def set_adapters(self, adapter_name, *, adapter_weights):
                pass

            def get_list_adapters(self):
                return {"text_encoder": ["iild_lora"]}

            def get_active_adapters(self):
                return []

        activation = self.generate.apply_lora(
            TextOnlyPipeline(),
            arguments.lora_selection,
            arguments.cache_dir,
            False,
        )
        self.assertEqual(activation.active_adapters, ("iild_lora",))
        self.assertEqual(activation.registered_components, ("text_encoder",))

    def test_gated_lora_error_is_reported_without_a_traceback(self) -> None:
        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                [
                    "--lora",
                    "vendor/gated-lora",
                    "--lora-revision",
                    "b" * 40,
                    "--lora-weight-name",
                    "style.safetensors",
                ]
            )
        )

        class GatedRepoError(Exception):
            pass

        class GatedPipeline:
            def load_lora_weights(self, source, **kwargs):
                raise GatedRepoError(source)

        with self.assertRaisesRegex(SystemExit, "gated LoRA"):
            self.generate.apply_lora(
                GatedPipeline(),
                arguments.lora_selection,
                arguments.cache_dir,
                False,
            )

    def test_generation_without_lora_preserves_base_behavior(self) -> None:
        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args([])
        )
        self.assertIsNone(arguments.lora_selection)
        self.assertEqual(arguments.output, self.generate.DEFAULT_OUTPUT)
        self.assertIsNone(self.generate.apply_lora(object(), None, arguments.cache_dir, False))
        self.assertIsNone(self.generate.lora_metadata(None, None))

    def test_remote_override_requires_immutable_revision(self) -> None:
        with self.assertRaisesRegex(SystemExit, "commit --revision"):
            self.generate.resolve_arguments(
                self.generate.build_parser().parse_args(["--model", "vendor/compatible-model"])
            )

        _, arguments = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args(
                ["--model", "vendor/compatible-model", "--revision", "a" * 40]
            )
        )
        self.assertEqual(arguments.model_selection.requested_revision, "a" * 40)

    def test_pipeline_loader_reports_gated_repository_without_traceback(self) -> None:
        class GatedRepoError(Exception):
            pass

        class GatedPipeline:
            @staticmethod
            def from_pretrained(source, **arguments):
                try:
                    raise GatedRepoError(source)
                except GatedRepoError as error:
                    raise RuntimeError("wrapped load failure") from error

        with self.assertRaisesRegex(SystemExit, "hf auth login"):
            pipeline_loading.load_pipeline(
                GatedPipeline,
                "black-forest-labs/FLUX.1-schnell",
                {"revision": "a" * 40},
            )

    def test_sdxl_without_safety_checker_produces_metadata(self) -> None:
        metadata = self.generate.safety_metadata(SimpleNamespace(), SimpleNamespace())
        self.assertEqual(
            metadata,
            {
                "checker_present": False,
                "nsfw_content_detected": None,
                "watermarker_present": False,
            },
        )

    def test_uniform_image_is_rejected(self) -> None:
        image = SimpleNamespace(
            size=(1024, 1024),
            mode="RGB",
            getextrema=lambda: ((0, 0), (0, 0), (0, 0)),
        )
        with self.assertRaisesRegex(RuntimeError, "uniform"):
            self.generate.validate_rendered_image(image, 1024, 1024)

        image.getextrema = lambda: ((1, 255), (2, 250), (3, 249))
        self.generate.validate_rendered_image(image, 1024, 1024)

    def test_sdxl_runtime_contract_validator(self) -> None:
        def component(class_name: str, **configuration):
            value = type(class_name, (), {})()
            value.config = SimpleNamespace(**configuration)
            return value

        components = {
            "tokenizer": component("CLIPTokenizer", model_max_length=77),
            "tokenizer_2": component("CLIPTokenizer", model_max_length=77),
            "text_encoder": component(
                "CLIPTextModel",
                hidden_size=768,
                max_position_embeddings=77,
                projection_dim=768,
            ),
            "text_encoder_2": component(
                "CLIPTextModelWithProjection",
                hidden_size=1280,
                max_position_embeddings=77,
                projection_dim=1280,
            ),
            "unet": component(
                "UNet2DConditionModel",
                in_channels=4,
                out_channels=4,
                sample_size=128,
                cross_attention_dim=2048,
                addition_embed_type="text_time",
                addition_time_embed_dim=256,
                projection_class_embeddings_input_dim=2816,
                use_linear_projection=True,
            ),
            "vae": component(
                "AutoencoderKL",
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                sample_size=1024,
                scaling_factor=0.13025,
                force_upcast=True,
                block_out_channels=[128, 256, 512, 512],
            ),
            "scheduler": component(
                "EulerDiscreteScheduler",
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
                timestep_spacing="leading",
            ),
        }
        pipeline = type("StableDiffusionXLPipeline", (), {})()
        pipeline.components = components
        pipeline.config = SimpleNamespace(force_zeros_for_empty_prompt=True)
        for name, value in components.items():
            setattr(pipeline, name, value)

        presets.validate_pipeline_contract(pipeline, presets.SDXL_BASE_PRESET)
        pipeline.scheduler.config.beta_end = 0.02
        with self.assertRaisesRegex(RuntimeError, "beta_end"):
            presets.validate_pipeline_contract(pipeline, presets.SDXL_BASE_PRESET)

    def test_flux_runtime_contract_validator(self) -> None:
        def component(class_name: str, **configuration):
            value = type(class_name, (), {})()
            value.config = SimpleNamespace(**configuration)
            return value

        components = {
            "tokenizer": component("CLIPTokenizer", model_max_length=77),
            "tokenizer_2": component("T5Tokenizer", model_max_length=512),
            "text_encoder": component(
                "CLIPTextModel",
                hidden_size=768,
                max_position_embeddings=77,
                projection_dim=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=12,
                vocab_size=49408,
            ),
            "text_encoder_2": component(
                "T5EncoderModel",
                d_model=4096,
                d_ff=10240,
                d_kv=64,
                num_heads=64,
                num_layers=24,
                vocab_size=32128,
                feed_forward_proj="gated-gelu",
            ),
            "transformer": component(
                "FluxTransformer2DModel",
                patch_size=1,
                in_channels=64,
                num_layers=19,
                num_single_layers=38,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=False,
                out_channels=None,
                axes_dims_rope=(16, 56, 56),
            ),
            "vae": component(
                "AutoencoderKL",
                in_channels=3,
                out_channels=3,
                latent_channels=16,
                sample_size=1024,
                layers_per_block=2,
                norm_num_groups=32,
                scaling_factor=0.3611,
                shift_factor=0.1159,
                force_upcast=True,
                use_quant_conv=False,
                use_post_quant_conv=False,
                block_out_channels=[128, 256, 512, 512],
                down_block_types=["DownEncoderBlock2D"] * 4,
                up_block_types=["UpDecoderBlock2D"] * 4,
                mid_block_add_attention=True,
            ),
            "scheduler": component(
                "FlowMatchEulerDiscreteScheduler",
                num_train_timesteps=1000,
                base_image_seq_len=256,
                max_image_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
                shift=1.0,
                use_dynamic_shifting=False,
            ),
        }
        pipeline = type("FluxPipeline", (), {})()
        pipeline.components = components
        for name, value in components.items():
            setattr(pipeline, name, value)

        presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)
        canonical_fast_tokenizer = component("T5TokenizerFast", model_max_length=512)
        pipeline.components["tokenizer_2"] = canonical_fast_tokenizer
        pipeline.tokenizer_2 = canonical_fast_tokenizer
        pipeline.transformer.config.axes_dims_rope = [16, 56, 56]
        pipeline.transformer.config.out_channels = 64
        presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)

        pipeline.transformer.config.out_channels = 32
        with self.assertRaisesRegex(RuntimeError, "out_channels"):
            presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)
        pipeline.transformer.config.out_channels = None

        pipeline.vae.config.block_out_channels = [128, 256, 256, 512]
        with self.assertRaisesRegex(RuntimeError, "block_out_channels"):
            presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)
        pipeline.vae.config.block_out_channels = [128, 256, 512, 512]

        pipeline.transformer.config.guidance_embeds = True
        with self.assertRaisesRegex(RuntimeError, "guidance_embeds"):
            presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)

        pipeline.transformer.config.guidance_embeds = False
        pipeline.text_encoder.config.intermediate_size = 2048
        with self.assertRaisesRegex(RuntimeError, "intermediate_size"):
            presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)
        pipeline.text_encoder.config.intermediate_size = 3072

        pipeline.text_encoder_2.config.feed_forward_proj = "relu"
        with self.assertRaisesRegex(RuntimeError, "feed_forward_proj"):
            presets.validate_pipeline_contract(pipeline, presets.FLUX1_SCHNELL_PRESET)


if __name__ == "__main__":
    unittest.main()
