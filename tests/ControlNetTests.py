#!/usr/bin/env python3
"""Dependency-free regressions for explicit ControlNet generation requests."""

from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
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
import controlnet
import presets


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNwaDgAAAKEAYEml6crAAAAAElFTkSuQmCC"
)
REVISION = "a" * 40
CONFIG_REVISION = "b" * 40


def resolved(*arguments):
    preset, args = generate.resolve_arguments(generate.build_parser().parse_args(list(arguments)))
    generate.validate_generation_arguments(preset, args)
    return preset, args


class ControlNetTests(unittest.TestCase):
    def setUp(self):
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="controlnet-tests-", dir=build)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model_directory = self.directory / "local-checkpoint"
        self.model_directory.mkdir()
        self.controlnet = self.directory / "controlnet"
        self.controlnet.mkdir()
        (self.controlnet / "config.json").write_text(
            json.dumps({"_class_name": "ControlNetModel"}), encoding="utf-8"
        )
        self.image = self.directory / "condition.png"
        self.image.write_bytes(PNG)

    def request(self, **overrides):
        values = {"controlnet": str(self.controlnet), "control_image": str(self.image)}
        values.update(overrides)
        preset = presets.PRESETS[values.get("preset", "sd15")]
        if preset.requires_model_override:
            values.setdefault("model", str(self.model_directory))
        return generate.resolve_request(values)

    def weight(self, name="controlnet.safetensors", contents=b"controlnet weight fixture"):
        path = self.directory / name
        path.write_bytes(contents)
        return path

    def component(self, preset=presets.SD15_PRESET, *, meta=False):
        if preset.family == "flux1-schnell":
            config = dict(in_channels=64, patch_size=1, attention_head_dim=128,
                          num_attention_heads=24, joint_attention_dim=4096,
                          pooled_projection_dim=768, axes_dims_rope=(16, 56, 56))
        else:
            config = dict(in_channels=4, cross_attention_dim=768,
                          block_out_channels=(320, 640, 1280, 1280),
                          down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                            "CrossAttnDownBlock2D", "DownBlock2D"),
                          layers_per_block=2, addition_embed_type=None,
                          addition_time_embed_dim=None, projection_class_embeddings_input_dim=None,
                          conditioning_channels=3)
            if preset.family == "sdxl-base":
                config.update(cross_attention_dim=2048, block_out_channels=(320, 640, 1280),
                              down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                              addition_embed_type="text_time", addition_time_embed_dim=256,
                              projection_class_embeddings_input_dim=2816)
        component = type(controlnet.PIPELINES[preset.family][1], (), {})()
        component.config = SimpleNamespace(**config)
        component.named_parameters = lambda: iter((("weight", SimpleNamespace(is_meta=meta)),))
        component.named_buffers = lambda: iter(())
        component.union = False
        return component

    def loader_fixture(self, preset=presets.SD15_PRESET, **values):
        _, args = self.request(preset=preset.name, cache_dir=str(self.directory / "cache"), **values)
        args.cache_dir.mkdir(exist_ok=True)
        args.controlnet_image = SimpleNamespace(size=(1, 1))
        component = self.component(preset)
        base_component = SimpleNamespace(config=SimpleNamespace(**vars(component.config)))
        pipeline = SimpleNamespace(unet=base_component, transformer=base_component)
        replacement = object()
        loader = SimpleNamespace(from_pretrained=Mock(return_value=(component, {})),
                                 from_single_file=Mock(return_value=component))
        wrapper = SimpleNamespace(from_pipe=Mock(return_value=replacement))
        classes = {controlnet.PIPELINES[preset.family][0]: wrapper,
                   controlnet.PIPELINES[preset.family][1]: loader}
        return args, pipeline, component, loader, wrapper, classes

    def test_omitting_controlnet_preserves_every_base_pipeline(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                request = {"preset": preset.name}
                if preset.requires_model_override:
                    request["model"] = str(self.model_directory)
                _, args = generate.resolve_request(request)
                self.assertIsNone(args.controlnet_selection)
                values = generate.build_pipeline_call_arguments(preset, args, "generator")
                for name in ("image", "control_image", "controlnet_conditioning_scale",
                             "control_guidance_start", "control_guidance_end", "guess_mode"):
                    self.assertNotIn(name, values)
                configuration = generate.configuration_values(args)
                _, replayed = generate.resolve_request(configuration)
                self.assertEqual(generate.configuration_values(replayed), configuration)

    def test_local_directory_request_has_neutral_defaults_and_image_identity(self):
        _, args = self.request()
        selected = args.controlnet_selection
        self.assertTrue(selected.is_local)
        self.assertEqual(selected.source, str(self.controlnet.resolve()))
        self.assertIsNone(selected.requested_revision)
        self.assertIsNone(selected.single_file)
        self.assertEqual(args.controlnet_scale, 1.0)
        self.assertEqual(args.control_guidance_start, 0.0)
        self.assertEqual(args.control_guidance_end, 1.0)
        self.assertFalse(args.guess_mode)
        self.assertEqual(args.control_image_file.path, str(self.image.absolute()))
        self.assertEqual(args.control_image_file.resolved_file, str(self.image.resolve()))
        self.assertEqual(args.control_image_file.sha256, hashlib.sha256(PNG).hexdigest())
        self.assertEqual(args.control_image_file.size_bytes, len(PNG))

    def test_remote_controlnet_preserves_immutable_revision(self):
        _, args = self.request(controlnet="vendor/controlnet", controlnet_revision=REVISION)
        self.assertEqual(args.controlnet_selection.source, "vendor/controlnet")
        self.assertEqual(args.controlnet_selection.requested_revision, REVISION)
        self.assertFalse(args.controlnet_selection.is_local)

    def test_remote_controlnet_requires_a_lowercase_40_character_commit(self):
        for revision in (None, "", "main", "v1.0", "A" * 40, "a" * 39, "a" * 41, "g" * 40):
            with self.subTest(revision=revision), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.request(controlnet="vendor/controlnet", controlnet_revision=revision)

    def test_local_controlnet_rejects_remote_revision(self):
        for source in (self.controlnet, self.weight()):
            with self.subTest(source=source), self.assertRaises(SystemExit):
                self.request(controlnet=str(source), controlnet_revision=REVISION)

    def test_controlnet_requires_a_local_nonempty_condition_image(self):
        empty = self.directory / "empty.png"
        empty.touch()
        for image in (None, "", str(self.directory), str(empty),
                      str(self.directory / "missing.png"), "https://example.com/condition.png"):
            with self.subTest(image=image), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.request(control_image=image)

    def test_controlnet_rejects_missing_empty_and_unsafe_weight_sources(self):
        empty = self.weight("empty.safetensors", b"")
        unsafe = self.weight("controlnet.ckpt")
        for source in ("", str(empty), str(unsafe), str(self.directory / "missing.safetensors")):
            with self.subTest(source=source), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.request(controlnet=source)

    def test_dependent_arguments_require_controlnet_even_for_explicit_defaults(self):
        cases = (
            ("--control-image", str(self.image)),
            ("--controlnet-revision", REVISION),
            ("--controlnet-config", str(self.controlnet)),
            ("--controlnet-config-revision", CONFIG_REVISION),
            ("--controlnet-scale", "1"),
            ("--control-guidance-start", "0"),
            ("--control-guidance-end", "1"),
            ("--guess-mode",),
            ("--no-guess-mode",),
            ("--controlnet-variant", "fp16"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    resolved(*arguments)

    def test_single_file_requires_component_config_when_no_sibling_exists(self):
        weight = self.weight()
        with self.assertRaises(SystemExit):
            self.request(controlnet=str(weight))
        _, args = self.request(controlnet=str(weight), controlnet_config=str(self.controlnet))
        self.assertEqual(args.controlnet_selection.single_file.path, str(weight))
        self.assertEqual(args.controlnet_config_selection.source, str(self.controlnet))
        self.assertTrue(args.controlnet_config_selection.is_local)

    def test_single_file_uses_sibling_config_without_network_defaults(self):
        weight = self.weight()
        (weight.parent / "config.json").write_text('{"_class_name":"ControlNetModel"}', encoding="utf-8")
        _, args = self.request(controlnet=str(weight))
        self.assertEqual(args.controlnet_config_selection.source, str(weight.parent))
        self.assertTrue(args.controlnet_config_selection.is_local)
        self.assertIsNone(args.controlnet_config_selection.requested_revision)

    def test_single_file_accepts_a_pinned_remote_component_config(self):
        _, args = self.request(controlnet=str(self.weight()), controlnet_config="vendor/config",
                               controlnet_config_revision=CONFIG_REVISION)
        self.assertEqual(args.controlnet_config_selection.source, "vendor/config")
        self.assertEqual(args.controlnet_config_selection.requested_revision, CONFIG_REVISION)
        self.assertFalse(args.controlnet_config_selection.is_local)

    def test_component_configuration_must_be_pinned_or_local_without_revision(self):
        weight = self.weight()
        cases = (
            {"controlnet_config": "vendor/config"},
            {"controlnet_config": "vendor/config", "controlnet_config_revision": "main"},
            {"controlnet_config": "vendor/config", "controlnet_config_revision": "B" * 40},
            {"controlnet_config": str(self.controlnet), "controlnet_config_revision": CONFIG_REVISION},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(SystemExit):
                self.request(controlnet=str(weight), **values)

    def test_component_config_cannot_be_silently_ignored_for_directory_source(self):
        for values in ({"controlnet_config": str(self.controlnet)},
                       {"controlnet_config_revision": CONFIG_REVISION}):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                self.request(**values)

    def test_valid_strength_guidance_interval_and_alias_reach_requests(self):
        preset, args = resolved("--controlnet", str(self.controlnet), "--control-image", str(self.image),
                                "--controlnet-conditioning-scale", "0", "--control-guidance-start", "0.2",
                                "--control-guidance-end", "0.8", "--guess-mode")
        self.assertEqual((args.controlnet_scale, args.control_guidance_start,
                          args.control_guidance_end, args.guess_mode), (0.0, 0.2, 0.8, True))
        self.assertEqual(preset.name, "sd15")

    def test_strength_and_guidance_reject_nonfinite_or_invalid_ranges(self):
        invalid = (
            ("--controlnet-scale", "-0.1"), ("--controlnet-scale", "nan"),
            ("--controlnet-scale", "inf"), ("--control-guidance-start", "nan"),
            ("--control-guidance-start", "-0.1"), ("--control-guidance-start", "1"),
            ("--control-guidance-end", "0"), ("--control-guidance-end", "1.1"),
            ("--control-guidance-end", "inf"),
            ("--control-guidance-start", "0.5", "--control-guidance-end", "0.5"),
            ("--control-guidance-start", "0.8", "--control-guidance-end", "0.2"),
        )
        for extra in invalid:
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    resolved("--controlnet", str(self.controlnet), "--control-image", str(self.image), *extra)

    def test_json_configuration_uses_relative_paths_and_explicit_cli_precedence(self):
        path = self.directory / "generation.json"
        path.write_text(json.dumps({"controlnet": "./controlnet", "control_image": "condition.png",
                                    "controlnet_scale": 0.5, "control_guidance_start": 0.2,
                                    "control_guidance_end": 0.9, "guess_mode": True}), encoding="utf-8")
        _, args = resolved("--config", str(path), "--controlnet-scale", "0", "--no-guess-mode")
        self.assertEqual(args.controlnet_selection.source, str(self.controlnet.resolve()))
        self.assertEqual(args.control_image_file.path, str(self.image.resolve()))
        self.assertEqual(args.controlnet_scale, 0.0)
        self.assertFalse(args.guess_mode)
        self.assertEqual((args.control_guidance_start, args.control_guidance_end), (0.2, 0.9))

    def test_json_single_file_config_paths_are_relative_to_the_json_file(self):
        weight = self.weight()
        path = self.directory / "generation.json"
        path.write_text(json.dumps({"controlnet": weight.name, "controlnet_config": "./controlnet",
                                    "control_image": self.image.name}), encoding="utf-8")
        _, args = resolved("--config", str(path))
        self.assertEqual(args.controlnet_selection.single_file.path, str(weight))
        self.assertEqual(args.controlnet_config_selection.source, str(self.controlnet))

    def test_json_rejects_wrong_types_without_loading_the_runtime(self):
        for name, value in (("controlnet", True), ("control_image", 12), ("controlnet_scale", True),
                            ("controlnet_scale", "0.5"), ("control_guidance_start", False),
                            ("control_guidance_end", "1"), ("guess_mode", "false")):
            with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.request(**{name: value})

    def test_local_remote_and_single_file_configurations_round_trip(self):
        cases = (
            {},
            {"controlnet": "vendor/controlnet", "controlnet_revision": REVISION},
            {"controlnet": str(self.weight()), "controlnet_config": str(self.controlnet)},
        )
        for values in cases:
            with self.subTest(values=values):
                _, original = self.request(**values)
                configuration = generate.configuration_values(original)
                self.assertNotIn("controlnet_selection", configuration)
                self.assertNotIn("control_image_file", configuration)
                self.assertNotIn("controlnet_image", configuration)
                _, replayed = generate.resolve_request(configuration)
                self.assertEqual(generate.configuration_values(replayed), configuration)

    def test_controlnet_uses_a_separate_default_output_for_each_family(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                _, args = self.request(preset=preset.name)
                custom_suffix = "-custom" if preset.requires_model_override else ""
                self.assertEqual(args.output.name,
                                 Path(preset.generation_filename).stem + custom_suffix + "-controlnet.png")
                explicit = self.directory / "chosen.png"
                _, explicit_args = self.request(preset=preset.name, output=str(explicit))
                self.assertEqual(explicit_args.output, explicit)

    def test_lora_and_controlnet_output_modifiers_do_not_overwrite_the_base(self):
        _, args = self.request(lora=str(self.weight("lora.safetensors")))
        self.assertTrue(args.output.name.endswith("-lora-controlnet.png"))

    def test_condition_image_and_controls_reach_every_supported_pipeline(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                _, args = self.request(preset=preset.name, controlnet_scale=0.75,
                                       control_guidance_start=0.1, control_guidance_end=0.9)
                decoded_image = object()
                args.controlnet_image = decoded_image
                values = generate.build_pipeline_call_arguments(preset, args, "generator")
                image_name = "control_image" if preset.family == "flux1-schnell" else "image"
                self.assertIs(values[image_name], decoded_image)
                self.assertEqual(values["controlnet_conditioning_scale"], 0.75)
                self.assertEqual(values["control_guidance_start"], 0.1)
                self.assertEqual(values["control_guidance_end"], 0.9)
                self.assertNotIn("guidance_rescale", values)
                if preset.family == "flux1-schnell":
                    self.assertNotIn("guess_mode", values)
                    self.assertNotIn("image", values)
                else:
                    self.assertFalse(values["guess_mode"])
                    self.assertNotIn("control_image", values)

    def test_unsupported_pipeline_controls_are_not_silently_discarded(self):
        cases = (
            {"guidance_rescale": 0.25},
            {"preset": "sdxl-base", "guidance_rescale": 0.25},
            {"preset": "flux1-schnell", "guess_mode": True},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(SystemExit):
                self.request(**values)

    def test_print_config_requires_no_model_or_image_runtime_packages(self):
        output = io.StringIO()
        arguments = ["generate.py", "--controlnet", str(self.controlnet),
                     "--control-image", str(self.image), "--print-config"]
        with patch.object(sys, "argv", arguments), redirect_stdout(output), \
                patch.object(generate, "package_versions", side_effect=AssertionError("packages")), \
                patch.object(generate, "load_dependencies", side_effect=AssertionError("runtime")):
            self.assertEqual(generate.main(), 0)
        values = json.loads(output.getvalue())
        self.assertEqual(values["controlnet"], str(self.controlnet))
        self.assertEqual(values["control_image"], str(self.image))
        self.assertEqual(values["controlnet_scale"], 1.0)

    def test_omitting_controlnet_never_loads_or_rewraps_the_base(self):
        _, args = generate.resolve_request()
        pipeline = object()
        with patch.object(controlnet, "validate_pipeline_contract") as validate:
            result, metadata = controlnet.attach_controlnet(pipeline, presets.SD15_PRESET, args, {}, "dtype")
        self.assertIs(result, pipeline)
        self.assertIsNone(metadata)
        validate.assert_not_called()

    def test_directory_loader_is_safe_offline_and_records_weight_and_image_identity(self):
        weight = self.controlnet / "diffusion_pytorch_model.safetensors"
        weight.write_bytes(b"local ControlNet weights")
        args, base, component, loader, wrapper, classes = self.loader_fixture()
        with patch.object(controlnet, "validate_pipeline_contract"):
            result, metadata = controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float16")
        self.assertIs(result, wrapper.from_pipe.return_value)
        wrapper.from_pipe.assert_called_once_with(base, dtype="float16", controlnet=component)
        load_args = loader.from_pretrained.call_args.kwargs
        self.assertTrue(load_args["use_safetensors"])
        self.assertTrue(load_args["local_files_only"])
        self.assertTrue(load_args["low_cpu_mem_usage"])
        self.assertEqual(load_args["dtype"], "float16")
        self.assertNotIn("revision", load_args)
        self.assertEqual(metadata["image"]["sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertEqual(metadata["image"]["generation_size"], [512, 512])
        self.assertEqual(metadata["scale"], 1.0)
        recorded = {item["path"]: item for item in metadata["files"]}
        self.assertEqual(recorded[str(weight)]["sha256"], hashlib.sha256(weight.read_bytes()).hexdigest())

    def test_remote_download_uses_the_requested_commit_and_offline_policy(self):
        (self.controlnet / "diffusion_pytorch_model.safetensors").write_bytes(b"cached safe weights")
        args, base, _, _, _, classes = self.loader_fixture(
            controlnet="vendor/controlnet", controlnet_revision=REVISION, local_files_only=True)
        download = Mock(return_value=str(self.controlnet))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                patch.object(controlnet, "validate_pipeline_contract"):
            _, metadata = controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            self.assertEqual(call.args, ("vendor/controlnet",))
            self.assertEqual(call.kwargs["revision"], REVISION)
            self.assertTrue(call.kwargs["local_files_only"])
            self.assertEqual(call.kwargs["cache_dir"], args.cache_dir)
        self.assertEqual(download.call_args_list[0].kwargs["allow_patterns"],
                         ["config.json", "diffusion_pytorch_model.safetensors.index.json"])
        self.assertEqual(download.call_args_list[1].kwargs["allow_patterns"],
                         ["diffusion_pytorch_model.safetensors"])
        self.assertEqual(metadata["requested_revision"], REVISION)

    def test_controlnet_and_config_never_inherit_the_base_presets_remote_revision(self):
        with self.assertRaises(SystemExit):
            self.request(controlnet=presets.SD15_PRESET.model_id)
        with self.assertRaises(SystemExit):
            self.request(controlnet=str(self.weight()), controlnet_config=presets.SD15_PRESET.model_id)

    def test_directory_provenance_hashes_only_the_selected_weight_variant(self):
        names = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors",
                 "diffusion_pytorch_model.ema.safetensors")
        for name in names:
            (self.controlnet / name).write_bytes(name.encode())
        for variant, expected in ((None, names[0]), ("fp16", names[1])):
            args, base, _, loader, _, classes = self.loader_fixture(controlnet_variant=variant)
            with self.subTest(variant=variant), patch.object(controlnet, "validate_pipeline_contract"):
                _, metadata = controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
            self.assertEqual({Path(item["path"]).name for item in metadata["files"]}, {"config.json", expected})
            self.assertEqual(loader.from_pretrained.call_args.kwargs.get("variant"), variant)

    def test_modern_variant_index_fetches_exact_legacy_shard_names(self):
        modern = "diffusion_pytorch_model.safetensors.index.fp16.json"
        shards = ["diffusion_pytorch_model-00001-of-00002.fp16.safetensors",
                  "diffusion_pytorch_model-00002-of-00002.fp16.safetensors"]
        (self.controlnet / modern).write_text(json.dumps({"weight_map": {"first": shards[0], "second": shards[1]}}))
        for name in (*shards, "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.ema.safetensors"):
            (self.controlnet / name).write_bytes(name.encode())
        args, base, _, _, _, classes = self.loader_fixture(
            controlnet="vendor/controlnet", controlnet_revision=REVISION, controlnet_variant="fp16")
        download = Mock(return_value=str(self.controlnet))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                patch.object(controlnet, "validate_pipeline_contract"):
            _, metadata = controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0].kwargs["allow_patterns"], [
            "config.json", modern, "diffusion_pytorch_model.safetensors.fp16.index.json"])
        self.assertEqual(download.call_args_list[1].kwargs["allow_patterns"], shards)
        self.assertEqual({Path(item["path"]).name for item in metadata["files"]}, {"config.json", modern, *shards})

    def test_legacy_variant_index_and_duplicate_shard_references_are_supported(self):
        index = self.controlnet / "diffusion_pytorch_model.safetensors.fp16.index.json"
        shard = self.controlnet / "diffusion_pytorch_model-00001-of-00001.fp16.safetensors"
        index.write_text(json.dumps({"weight_map": {"first": shard.name, "second": shard.name}}))
        shard.write_bytes(b"one shard")
        self.assertEqual(controlnet._package_files(self.controlnet, "fp16"), [index, shard])

    def test_modern_index_takes_precedence_over_legacy_and_single_file_weights(self):
        modern = self.controlnet / "diffusion_pytorch_model.safetensors.index.fp16.json"
        legacy = self.controlnet / "diffusion_pytorch_model.safetensors.fp16.index.json"
        modern.write_text('{"weight_map":{"first":"chosen.safetensors"}}')
        legacy.write_text('{"weight_map":{"first":"ignored.safetensors"}}')
        (self.controlnet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"also ignored")
        self.assertEqual(controlnet._package_files(self.controlnet, "fp16"),
                         [modern, self.controlnet / "chosen.safetensors"])

    def test_index_rejects_invalid_maps_and_unsafe_shard_paths(self):
        index = self.controlnet / "diffusion_pytorch_model.safetensors.index.json"
        invalid = [[], {}, {"weight_map": None}, {"weight_map": []}, {"weight_map": {}},
                   {"weight_map": {"": "part.safetensors"}}, {"weight_map": {"weight": None}},
                   {"weight_map": {"weight": ["part.safetensors"]}}]
        invalid += [{"weight_map": {"weight": name}} for name in (
            "", "../part.safetensors", "/part.safetensors", "child/../../part.safetensors",
            "child/../part.safetensors", "./part.safetensors", "child//part.safetensors",
            "child\\part.safetensors", "C:\\part.safetensors", "part.bin", "part.safetensor",
            "part.SAFETENSORS", "*.safetensors", "part?.safetensors", "part[0].safetensors",
        )]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                index.write_text(json.dumps(value))
                controlnet._package_files(self.controlnet, None)

    def test_bad_remote_index_is_rejected_before_any_weight_download(self):
        (self.controlnet / "diffusion_pytorch_model.safetensors.index.json").write_text(
            '{"weight_map":{"weight":"../escape.safetensors"}}')
        args, _, _, _, _, _ = self.loader_fixture(controlnet="vendor/controlnet", controlnet_revision=REVISION)
        download = Mock(return_value=str(self.controlnet))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                self.assertRaises(ValueError):
            controlnet._snapshot(args.controlnet_selection, args)
        download.assert_called_once()

    def test_package_file_selection_reports_missing_directory_and_missing_selected_weights(self):
        missing = self.directory / "missing-package"
        with self.assertRaisesRegex(ValueError, "directory.*missing|missing.*directory"):
            controlnet._package_files(missing, None)
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            controlnet._identities(controlnet._package_files(self.controlnet, "fp16"))

    def test_remote_single_file_configuration_does_not_fetch_component_weights(self):
        _, args = self.request(controlnet=str(self.weight()), controlnet_config="vendor/config",
                               controlnet_config_revision=CONFIG_REVISION)
        download = Mock(return_value=str(self.controlnet))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            path = controlnet._snapshot(args.controlnet_config_selection, args, config_only=True)
        self.assertEqual(path, self.controlnet)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["allow_patterns"], ["config.json"])
        self.assertEqual(download.call_args.kwargs["revision"], CONFIG_REVISION)

    def test_loader_rejects_incomplete_unexpected_and_mismatched_tensor_sets(self):
        (self.controlnet / "diffusion_pytorch_model.safetensors").write_bytes(b"safe weights")
        for issue in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            args, base, component, loader, wrapper, classes = self.loader_fixture()
            loader.from_pretrained.return_value = (component, {issue: ["broken tensor"]})
            with self.subTest(issue=issue), patch.object(controlnet, "validate_pipeline_contract"):
                with self.assertRaisesRegex(RuntimeError, "weights do not match"):
                    controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
            wrapper.from_pipe.assert_not_called()

    def test_package_weights_changed_during_loading_are_rejected(self):
        weight = self.controlnet / "diffusion_pytorch_model.safetensors"
        weight.write_bytes(b"before!")
        args, base, component, loader, wrapper, classes = self.loader_fixture()

        def replace_weight(*unused, **ignored):
            weight.write_bytes(b"changed")
            return component, {}

        loader.from_pretrained.side_effect = replace_weight
        with patch.object(controlnet, "validate_pipeline_contract"), \
                self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        wrapper.from_pipe.assert_not_called()

    def test_non_safetensors_package_does_not_fall_back_to_pickle(self):
        (self.controlnet / "diffusion_pytorch_model.bin").write_bytes(b"unsafe pickle")
        args, base, _, loader, _, classes = self.loader_fixture()
        with patch.object(controlnet, "validate_pipeline_contract"), \
                self.assertRaisesRegex(RuntimeError, "missing or empty.*safetensors"):
            controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        loader.from_pretrained.assert_not_called()

    def test_native_single_file_is_loaded_safely_and_temporary_package_is_removed(self):
        weight = self.weight("controlnet.safetensor")
        args, base, _, loader, _, classes = self.loader_fixture(
            controlnet=str(weight), controlnet_config=str(self.controlnet))
        with patch.object(controlnet, "validate_pipeline_contract"), \
                patch.object(controlnet, "read_weight_keys", return_value={"controlnet_down_blocks.0.weight"}):
            _, metadata = controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        staging = Path(loader.from_pretrained.call_args.args[0])
        self.assertFalse(staging.exists())
        self.assertTrue(weight.exists())
        loader.from_single_file.assert_not_called()
        self.assertEqual(metadata["format"], "safetensors")
        self.assertEqual(metadata["sha256"], hashlib.sha256(weight.read_bytes()).hexdigest())

    def test_original_checkpoint_uses_the_explicit_component_config(self):
        args, base, component, loader, _, classes = self.loader_fixture(
            controlnet=str(self.weight()), controlnet_config=str(self.controlnet))
        native = {"weight": object()}
        component.state_dict = lambda: {"weight": object()}
        raw = {"control_model.input_blocks.0.weight": object()}
        read = Mock(return_value=raw)
        convert = Mock(return_value=native)
        with patch.object(controlnet, "validate_pipeline_contract"), \
                patch.object(controlnet, "read_weight_keys", return_value=set(raw)), \
                patch.dict(sys.modules, {
                    "safetensors.torch": SimpleNamespace(load_file=read),
                    "diffusers.loaders.single_file_utils": SimpleNamespace(convert_controlnet_checkpoint=convert),
                }):
            controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        loader.from_pretrained.assert_not_called()
        self.assertIs(loader.from_single_file.call_args.args[0], native)
        convert.assert_called_once_with(raw, config={"_class_name": "ControlNetModel"})
        self.assertEqual(loader.from_single_file.call_args.kwargs["config"], str(self.controlnet))
        self.assertTrue(loader.from_single_file.call_args.kwargs["local_files_only"])

    def test_original_checkpoint_rejects_missing_or_extra_converted_keys_in_both_memory_modes(self):
        for low_memory in (False, True):
            for keys, error in (({"weight"}, "missing"), ({"weight", "bias", "extra"}, "unexpected")):
                with self.subTest(low_memory=low_memory, keys=keys):
                    args, base, component, _, wrapper, classes = self.loader_fixture(
                        controlnet=str(self.weight()), controlnet_config=str(self.controlnet),
                        low_cpu_mem_usage=low_memory)
                    component.state_dict = lambda: {"weight": object(), "bias": object()}
                    native = {key: object() for key in keys}
                    with patch.object(controlnet, "validate_pipeline_contract"), \
                            patch.object(controlnet, "read_weight_keys", return_value={"control_model.input_blocks.0.weight"}), \
                            patch.dict(sys.modules, {
                                "safetensors.torch": SimpleNamespace(load_file=Mock(return_value={})),
                                "diffusers.loaders.single_file_utils": SimpleNamespace(
                                    convert_controlnet_checkpoint=Mock(return_value=native)),
                            }), self.assertRaisesRegex(RuntimeError, error):
                        controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
                    wrapper.from_pipe.assert_not_called()

    def test_meta_weights_are_rejected_before_pipeline_conversion(self):
        (self.controlnet / "diffusion_pytorch_model.safetensors").write_bytes(b"safe weights")
        args, base, _, loader, wrapper, classes = self.loader_fixture()
        loader.from_pretrained.return_value = (self.component(meta=True), {})
        with patch.object(controlnet, "validate_pipeline_contract"), \
                self.assertRaisesRegex(RuntimeError, "meta device"):
            controlnet.attach_controlnet(base, presets.SD15_PRESET, args, classes, "float32")
        wrapper.from_pipe.assert_not_called()

    def test_family_contract_rejects_incompatible_latent_and_attention_dimensions(self):
        for preset in presets.PRESETS.values():
            for field in ("in_channels", "joint_attention_dim" if preset.family == "flux1-schnell" else "cross_attention_dim"):
                args, base, component, _, _, _ = self.loader_fixture(preset)
                controlnet.validate_controlnet_contract(component, base, preset, args)
                setattr(component.config, field, getattr(component.config, field) + 1)
                with self.subTest(preset=preset.name, field=field), self.assertRaisesRegex(RuntimeError, field):
                    controlnet.validate_controlnet_contract(component, base, preset, args)

    def test_sd_conditioning_requires_three_rgb_channels(self):
        args, base, component, _, _, _ = self.loader_fixture()
        component.config.conditioning_channels = 1
        with self.assertRaisesRegex(RuntimeError, "three-channel RGB"):
            controlnet.validate_controlnet_contract(component, base, presets.SD15_PRESET, args)

    def test_flux_union_mode_is_required_and_checked_against_model_configuration(self):
        preset = presets.FLUX1_SCHNELL_PRESET
        args, base, component, _, _, _ = self.loader_fixture(preset)
        component.union = True
        component.config.num_mode = 3
        for mode in (None, -1, 3):
            args.control_mode = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                controlnet.validate_controlnet_contract(component, base, preset, args)
        args.control_mode = 2
        controlnet.validate_controlnet_contract(component, base, preset, args)
        component.union = False
        with self.assertRaisesRegex(ValueError, "Union"):
            controlnet.validate_controlnet_contract(component, base, preset, args)

    def test_image_changed_after_argument_resolution_is_rejected_before_decode(self):
        _, args = self.request()
        self.image.write_bytes(bytes([PNG[0] ^ 1]) + PNG[1:])
        image_api = SimpleNamespace(open=Mock())
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=image_api, ImageOps=Mock())}), \
                self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            controlnet.load_control_image(args)
        image_api.open.assert_not_called()

    def test_image_changed_during_decode_is_rejected(self):
        _, args = self.request()
        opened = Mock()
        opened.n_frames = 1
        image_api = SimpleNamespace(open=Mock())
        image_api.open.return_value.__enter__ = Mock(return_value=opened)
        image_api.open.return_value.__exit__ = Mock(return_value=False)
        decoded = Mock()
        decoded.load.side_effect = lambda: self.image.write_bytes(bytes([PNG[0] ^ 1]) + PNG[1:])
        image_ops = SimpleNamespace(exif_transpose=Mock())
        image_ops.exif_transpose.return_value.convert.return_value = decoded
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=image_api, ImageOps=image_ops)}), \
                self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            controlnet.load_control_image(args)

    def test_animated_image_is_rejected_instead_of_selecting_an_arbitrary_frame(self):
        _, args = self.request()
        opened = SimpleNamespace(n_frames=2)
        context = Mock()
        context.__enter__ = Mock(return_value=opened)
        context.__exit__ = Mock(return_value=False)
        image_api = SimpleNamespace(open=Mock(return_value=context))
        image_ops = Mock()
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=image_api, ImageOps=image_ops)}), \
                self.assertRaisesRegex(ValueError, "Animated control images"):
            controlnet.load_control_image(args)
        image_ops.exif_transpose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
