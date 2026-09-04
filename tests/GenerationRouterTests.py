#!/usr/bin/env python3
"""User-facing Civitai routing and derived-preset argument contracts."""

import argparse
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iild_generation_router_tests", ROOT / "reference" / "generate.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)
import generate
import presets


class GenerationRouterTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="generation-router-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def backend(self, base=None, backend="auto", *flags):
        return router.select_backend(argparse.Namespace(base_model=base, backend=backend), list(flags))

    def test_named_derivatives_choose_presets(self):
        for name in ("Illustrious", "NoobAI", "Pony", "Flux.1 Krea"):
            with self.subTest(name=name):
                self.assertEqual(self.backend(name), "preset")

    def test_other_architectures_and_explicit_workflows_route(self):
        self.assertEqual(self.backend("SD 3.5", "diffusers"), "diffusers")
        self.assertEqual(self.backend("Other", "auto", "--workflow=x.json"), "comfyui")
        self.assertEqual(self.backend("Illustrious", "auto", "--pipeline-inputs", "{}"), "diffusers")
        with self.assertRaises(ValueError):
            self.backend("Other")

    def test_unknown_hosted_and_cross_family_presets_are_rejected(self):
        for name in ("new-model-typo", "OpenAI"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.backend(name, "comfyui", "--workflow", "x.json")
        with self.assertRaisesRegex(SystemExit, "incompatible"):
            generate.resolve_arguments(generate.build_parser().parse_args(
                ["--base-model", "Illustrious", "--preset", "flux1-dev"]))

    def test_base_model_selects_matching_default_and_retains_identity(self):
        for name, expected in (("Illustrious", "illustrious"), ("NoobAI", "noobai"),
                               ("Flux.1 Krea", "flux1-krea-dev")):
            with self.subTest(name=name):
                preset, args = generate.resolve_arguments(generate.build_parser().parse_args(["--base-model", name]))
                self.assertEqual(preset.name, expected)
                self.assertEqual(args.base_model, name)

    def test_community_base_categories_use_compatible_contracts(self):
        for name, expected in (("SD 1.5", "sd15-compatible"), ("SDXL 1.0", "sdxl"),
                               ("Flux.1 S", "flux1-schnell-compatible")):
            preset, args = generate.resolve_arguments(generate.build_parser().parse_args(["--base-model", name]))
            self.assertEqual(preset.name, expected)
            self.assertFalse(preset.strict_contract)

    def test_noobai_vpred_explicit_variant_is_retained(self):
        preset, args = generate.resolve_arguments(generate.build_parser().parse_args(
            ["--base-model", "NoobAI", "--preset", "noobai-v-pred"]))
        self.assertEqual(preset.name, "noobai-v-pred")
        self.assertIn(("prediction_type", "v_prediction"), preset.scheduler_defaults)

    def test_same_family_cannot_silently_substitute_a_different_base(self):
        for base, preset in (("Flux.1 Krea", "flux1-schnell"),
                             ("Illustrious", "noobai"), ("NoobAI", "illustrious")):
            with self.subTest(base=base), self.assertRaisesRegex(SystemExit, "identities"):
                generate.resolve_arguments(generate.build_parser().parse_args(
                    ["--base-model", base, "--preset", preset]))

    def test_guidance_zero_is_required_only_for_schnell(self):
        for name in ("flux1-dev", "flux1-krea-dev"):
            preset, args = generate.resolve_arguments(generate.build_parser().parse_args(["--preset", name]))
            generate.validate_generation_arguments(preset, args)
            call = generate.build_pipeline_call_arguments(preset, args, "generator")
            self.assertEqual(call["guidance_scale"], preset.guidance_scale)
            self.assertEqual(call["max_sequence_length"], 512)
        with self.assertRaisesRegex(SystemExit, "guidance scale"):
            preset, args = generate.resolve_arguments(generate.build_parser().parse_args(
                ["--preset", "flux1-schnell", "--guidance-scale", "3"]))
            generate.validate_generation_arguments(preset, args)

    def test_pony_requires_explicit_weights(self):
        with self.assertRaisesRegex(SystemExit, "explicit --model"):
            generate.resolve_arguments(generate.build_parser().parse_args(["--base-model", "Pony"]))

    def test_frontdoor_print_config_runs_offline(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"),
                                 "--base-model", "Illustrious", "--print-config"],
                                capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertEqual(output["preset"], "illustrious")
        self.assertEqual(output["base_model"], "Illustrious")

    def test_catalog_command_includes_every_snapshot_entry(self):
        result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--list-base-models"],
                                capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertEqual(len(output["base_models"]), 105)
        self.assertEqual(len(output["source"]["commit"]), 40)

    def test_existing_downloaded_weight_files_route_to_local_runtime(self):
        for suffix in (".safetensors", ".ckpt", ".gguf"):
            path = self.directory / ("download" + suffix)
            path.write_bytes(b"model")
            with self.subTest(suffix=suffix):
                self.assertEqual(self.backend(None, "auto", "--model", str(path)), "local")
                self.assertEqual(self.backend("Illustrious", "auto", "--model=" + str(path)), "local")

    def test_explicit_runtime_choices_keep_precedence_over_existing_file(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        for extra, expected in ((["--preset", "sdxl"], "preset"),
                                (["--pipeline-class", "StableDiffusionXLPipeline"], "diffusers"),
                                (["--workflow", "flow.json"], "comfyui")):
            with self.subTest(extra=extra):
                self.assertEqual(self.backend(None, "auto", "--model", str(path), *extra), expected)
        self.assertEqual(self.backend(None, "diffusers", "--model", str(path)), "diffusers")
        self.assertEqual(self.backend(None, "local", "--model", str(path)), "local")

    def test_local_component_and_runtime_options_route_before_loading_model(self):
        for flag in ("--model-info", "--components", "--text-encoder", "--text-encoder-2", "--runtime-source", "--runtime-python", "--model-type", "--decoder", "--model-negative", "--embedded-guidance", "--sampling-shift"):
            with self.subTest(flag=flag):
                self.assertEqual(self.backend(None, "auto", flag, "value"), "local")

    def test_local_model_config_selects_generic_loader_before_local_file(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        self.assertEqual(self.backend("Kolors", "auto", "--model", str(path), "--model-config", str(self.directory)), "diffusers")
        self.assertEqual(self.backend(None, "auto", "--model=" + str(path), "--model-config=" + str(self.directory)), "diffusers")
        self.assertEqual(self.backend("Illustrious", "auto", "--model", str(path), "--model-config", str(self.directory), "--preset", "illustrious"), "preset")
        self.assertEqual(self.backend(None, "local", "--model", str(path), "--model-config", str(self.directory)), "local")

    def test_generic_media_options_select_generic_loader(self):
        for flag in ("--audio-sample-rate", "--video-layout"):
            with self.subTest(flag=flag):
                self.assertEqual(self.backend(None, "auto", flag, "value"), "diffusers")

    def test_editing_pipeline_and_inputs_override_downloaded_file_auto_route(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        for pipeline in ("StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline", "KolorsImg2ImgPipeline"):
            with self.subTest(pipeline=pipeline):
                self.assertEqual(self.backend(None, "auto", "--model", str(path), "--pipeline-class", pipeline,
                                              "--pipeline-inputs", '{"image":{"image_path":"/input.png"}}'), "diffusers")

    def test_top_level_model_config_dispatches_generic_print_config_offline(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"model")
        config = self.directory / "config"
        config.mkdir()
        (config / "model_index.json").write_text('{"_class_name":"StableDiffusionXLPipeline"}')
        result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/generate.py"),
                                 "--model", str(path), "--model-config", str(config), "--print-config"],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["runner"], "generic-diffusers")
        self.assertEqual(report["model_config"], str(config))

    def test_diffusers_directory_routes_by_model_index(self):
        directory = self.directory / "pipeline"
        directory.mkdir()
        self.assertEqual(self.backend(None, "auto", "--model", str(directory)), "preset")
        (directory / "model_index.json").write_text("{}")
        self.assertEqual(self.backend(None, "auto", "--model", str(directory)), "diffusers")

    def test_nonexistent_model_print_config_keeps_original_preset_contract(self):
        self.assertEqual(self.backend("Illustrious", "auto", "--model", str(self.directory / "not-downloaded.safetensors"), "--print-config"), "preset")
        self.assertEqual(self.backend("NoobAI", "auto", "--prediction-type", "v_prediction", "--clip-skip", "2"), "preset")

    def test_local_backend_dispatch_preserves_model_and_base_arguments(self):
        called = []
        backend = SimpleNamespace(main=lambda arguments: called.append(arguments) or 0)
        with patch.object(router.importlib, "import_module", return_value=backend) as imported:
            result = router.main(["--backend", "local", "--base-model", "Illustrious", "--model", "download.safetensors", "--print-config"])
        self.assertEqual(result, 0)
        imported.assert_called_once_with("local_image")
        self.assertEqual(called, [["--base-model", "Illustrious", "--model", "download.safetensors", "--print-config"]])

    def test_inspect_model_outputs_tensor_identity_without_generation(self):
        path = self.directory / "not-a-name-guess.safetensors"
        header = {
            "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F32", "shape": [1, 4, 1, 1], "data_offsets": [0, 16]},
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": {"dtype": "F32", "shape": [1, 768], "data_offsets": [16, 3088]},
        }
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(3088))
        output = io.StringIO()
        with redirect_stdout(output):
            result = router.main(["--inspect-model", "--model", str(path)])
        found = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(found["architecture"], "sd1")
        self.assertEqual(found["preset"], "sd15-compatible")

    def test_inspect_model_requires_a_real_model(self):
        for tokens in (["--inspect-model"], ["--inspect-model", "--model"],
                       ["--inspect-model", "--model", str(self.directory / "missing.safetensors")]):
            with self.subTest(tokens=tokens), self.assertRaises(SystemExit) as error:
                router.main(tokens)
            self.assertEqual(error.exception.code, 2)

    def test_legacy_inspection_reuses_managed_python_and_preserves_options(self):
        interpreter = self.directory / "managed-python"
        tokens = ["--inspect-model", "--model", "download.ckpt", "--model-info", "saved.json"]
        with patch.object(router, "legacy_inspection_python", return_value=interpreter), \
                patch.object(router.subprocess, "call", return_value=7) as called:
            self.assertEqual(router.main(tokens), 7)
        called.assert_called_once_with([str(interpreter), str(ROOT / "reference/generate.py"), *tokens])

    def test_legacy_inspection_same_interpreter_does_not_reexecute(self):
        self.assertIsNone(router.legacy_inspection_python("download.ckpt", ["--runtime-python", sys.executable]))

    def test_safetensors_and_gguf_inspection_stay_dependency_free(self):
        for model in ("download.safetensors", "download.gguf"):
            with self.subTest(model=model):
                self.assertIsNone(router.legacy_inspection_python(model, ["--runtime-python", "missing-python"]))

    def test_explicit_missing_legacy_inspection_python_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime Python does not exist"):
            router.legacy_inspection_python("download.ckpt", ["--runtime-python", str(self.directory / "missing-python")])


if __name__ == "__main__":
    unittest.main()
