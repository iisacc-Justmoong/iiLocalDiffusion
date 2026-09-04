#!/usr/bin/env python3
"""Dependency-free regression coverage for explicit model/VAE weight composition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "diffusers"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import model_loading
import presets
import weight_files


SD15_CLIP_KEY = "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
SDXL_CLIP_KEY = (
    "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight"
)
SDXL_OPEN_CLIP_KEY = "conditioner.embedders.1.model.positional_embedding"


def load_script(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def component(*, meta: bool = False, **configuration):
    return SimpleNamespace(
        config=SimpleNamespace(**configuration),
        named_parameters=lambda: iter((("weight", SimpleNamespace(is_meta=meta)),)),
        named_buffers=lambda: iter(()),
    )


def compatible_vae(preset):
    latent, scale, shift = {
        "sd15": (4, 0.18215, None),
        "sdxl-base": (4, 0.13025, 0.0),
        "flux1-schnell": (16, 0.3611, 0.1159),
    }[preset.family]
    return component(
        in_channels=3,
        out_channels=3,
        latent_channels=latent,
        scaling_factor=scale,
        shift_factor=shift,
        block_out_channels=(128, 256, 512, 512),
    )


class ModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generate = load_script("iild_model_generate_tests", "generate.py")
        cls.inspect = load_script("iild_model_inspect_tests", "inspect_pipeline.py")

    def setUp(self) -> None:
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="model-loading-tests-", dir=build)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def weight(self, name="model.safetensors", content=b"safe tensor fixture"):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return weight_files.resolve_weight_file(str(path), "--model")

    def configuration(self, preset, **extra):
        root = self.directory / f"config-{preset.name}"
        root.mkdir(exist_ok=True)
        index = {"_class_name": preset.pipeline_class}
        for name, class_name in preset.expected_components:
            library = (
                "transformers"
                if name.startswith(("text_encoder", "tokenizer"))
                else "diffusers"
            )
            index[name] = [library, class_name]
        index.update(extra)
        (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        return root

    def arguments(self, revision="a" * 40):
        return {
            "cache_dir": self.directory / "cache",
            "dtype": "test-float16",
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "trust_remote_code": False,
            "revision": revision,
        }

    def selection(self, weight):
        return presets.ModelSelection(weight.path, None, True, weight)

    def test_local_weights_record_absolute_identity_and_metadata(self) -> None:
        for suffix in (".safetensors", ".safetensor"):
            with self.subTest(suffix=suffix):
                weight = self.weight(f"identity{suffix}", b"adapter")
                self.assertTrue(Path(weight.path).is_absolute())
                self.assertEqual(weight.resolved_file, str(Path(weight.path).resolve()))
                self.assertEqual(weight.sha256, hashlib.sha256(b"adapter").hexdigest())
                self.assertEqual(weight.size_bytes, 7)
                weight_files.verify_weight_file(weight, "model")
                metadata = weight_files.weight_file_metadata(weight)
                self.assertEqual(metadata["format"], "safetensors")
                self.assertEqual(metadata["sha256"], weight.sha256)
                self.assertEqual(metadata["resolved_file"], weight.resolved_file)

    def test_weight_selection_rejects_empty_missing_directory_and_unsafe_suffix(self) -> None:
        invalid = ["", str(self.directory / "missing.safetensors"), str(self.directory)]
        for filename, contents in (
            ("empty.safetensors", b""),
            ("model.ckpt", b"pickle"),
            ("model.bin", b"pickle"),
            ("model.pt", b"pickle"),
            ("model.SAFETENSORS", b"ambiguous"),
        ):
            path = self.directory / filename
            path.write_bytes(contents)
            invalid.append(str(path))
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ValueError):
                weight_files.resolve_weight_file(source, "--vae")

    def test_identity_rejects_same_size_mutation_and_deletion(self) -> None:
        weight = self.weight(content=b"before!")
        path = Path(weight.path)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            weight_files.verify_weight_file(weight, "model")
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            weight_files.verify_weight_file(weight, "model")

    def test_symlink_target_identity_cannot_be_retargeted(self) -> None:
        original = self.weight("target.safetensors", b"same content")
        replacement = self.weight("replacement.safetensors", b"same content")
        alias = self.directory / "selected.safetensors"
        alias.symlink_to(original.path)
        selected = weight_files.resolve_weight_file(str(alias), "--vae")
        self.assertEqual(selected.path, str(alias))
        self.assertEqual(selected.resolved_file, original.resolved_file)
        weight_files.verify_weight_file(selected, "VAE")
        alias.unlink()
        alias.symlink_to(replacement.path)
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            weight_files.verify_weight_file(selected, "VAE")

    def test_singular_suffix_uses_temporary_safetensors_alias_and_cleans_it(self) -> None:
        weight = self.weight("model.safetensor")
        with weight_files.checked_safetensors_path(
            weight, self.directory / "aliases", "model"
        ) as path:
            alias = path
            self.assertEqual(path.suffix, ".safetensors")
            self.assertTrue(path.is_symlink())
            self.assertEqual(str(path.resolve()), weight.resolved_file)
        self.assertFalse(alias.exists())
        self.assertTrue(Path(weight.path).is_file())

    def test_standard_suffix_preserves_path_and_checks_changes_after_load(self) -> None:
        weight = self.weight(content=b"before!")
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            with weight_files.checked_safetensors_path(
                weight, self.directory / "aliases", "model"
            ) as path:
                self.assertEqual(path, Path(weight.path))
                path.write_bytes(b"changed")

    def test_singular_alias_cleans_up_when_loader_raises(self) -> None:
        weight = self.weight("model.safetensor")
        with self.assertRaisesRegex(ValueError, "loader failure"):
            with weight_files.checked_safetensors_path(
                weight, self.directory / "aliases", "model"
            ) as path:
                alias = path
                raise ValueError("loader failure")
        self.assertFalse(alias.exists())

    def test_default_selection_and_inspector_keep_diffusers_contract(self) -> None:
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                selection = presets.resolve_model_selection(preset, None, None)
                self.assertEqual(selection.source, preset.model_id)
                self.assertEqual(selection.requested_revision, preset.revision)
                self.assertIsNone(selection.single_file)
        weight = self.weight()
        with self.assertRaisesRegex(ValueError, "not a directory"):
            presets.resolve_model_selection(presets.SD15_PRESET, weight.path, None)
        with self.assertRaisesRegex(SystemExit, "not a directory"):
            self.inspect.resolve_arguments(
                self.inspect.build_parser().parse_args(["--model", weight.path])
            )
        selection = presets.resolve_model_selection(
            presets.SD15_PRESET, weight.path, None, allow_single_file=True
        )
        self.assertEqual(selection.single_file, weight)

    def test_model_resolver_rejects_empty_missing_and_local_revision(self) -> None:
        preset = presets.SD15_PRESET
        invalid = ("", str(self.directory / "missing.safetensors"), "missing.ckpt")
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ValueError):
                presets.resolve_model_selection(
                    preset, source, "a" * 40, allow_single_file=True
                )
        weight = self.weight()
        with self.assertRaisesRegex(ValueError, "local model.*revision"):
            presets.resolve_model_selection(
                preset, weight.path, "a" * 40, allow_single_file=True
            )
        with self.assertRaisesRegex(ValueError, "commit --revision"):
            presets.resolve_model_selection(preset, "vendor/model", None)
        for revision in ("main", "a" * 39, "A" * 40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                presets.resolve_model_selection(preset, "vendor/model", revision)

    def test_remote_configuration_download_is_pinned_filtered_and_offline(self) -> None:
        selection = presets.ModelSelection("vendor/config", "b" * 40, False)
        downloader = Mock(return_value=str(self.directory))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(
            snapshot_download=downloader
        )}):
            actual = model_loading.download_configuration(
                selection, self.directory / "cache", True
            )
        self.assertEqual(actual, str(self.directory))
        arguments = downloader.call_args.kwargs
        self.assertEqual(downloader.call_args.args, ("vendor/config",))
        self.assertEqual(arguments["revision"], "b" * 40)
        self.assertTrue(arguments["local_files_only"])
        self.assertEqual(arguments["cache_dir"], self.directory / "cache")
        self.assertFalse(any("safetensors" in item for item in arguments["allow_patterns"]))
        self.assertFalse(any("bin" in item for item in arguments["allow_patterns"]))

    def test_configuration_resolves_remote_source_to_validated_local_directory(self) -> None:
        preset = presets.FLUX1_SCHNELL_PRESET
        directory = self.configuration(preset)
        selection = presets.ModelSelection("vendor/config", "c" * 40, False)
        with patch.object(
            model_loading, "download_configuration", return_value=str(directory)
        ) as download:
            result = model_loading.resolve_configuration_directory(
                selection, preset, self.directory / "cache", True
            )
        self.assertEqual(result, directory.resolve())
        download.assert_called_once_with(selection, self.directory / "cache", True)

    def test_configuration_rejects_missing_wrong_family_and_custom_code(self) -> None:
        preset = presets.SD15_PRESET
        selection = presets.ModelSelection(str(self.directory), None, True)
        with self.assertRaisesRegex(ValueError, "model_index.json"):
            model_loading.resolve_configuration_directory(
                selection, preset, self.directory, True
            )
        directory = self.configuration(presets.SDXL_BASE_PRESET)
        selection = presets.ModelSelection(str(directory), None, True)
        with self.assertRaisesRegex(ValueError, "StableDiffusionPipeline"):
            model_loading.resolve_configuration_directory(
                selection, preset, self.directory, True
            )
        directory = self.configuration(preset, custom_encoder=["evil_code", "Encoder"])
        selection = presets.ModelSelection(str(directory), None, True)
        with self.assertRaisesRegex(ValueError, "Unsupported component declaration"):
            model_loading.resolve_configuration_directory(
                selection, preset, self.directory, True
            )

    def test_configuration_does_not_retry_offline_download_failure(self) -> None:
        selection = presets.ModelSelection("vendor/config", "a" * 40, False)
        with patch.object(
            model_loading, "download_configuration", side_effect=OSError("not cached")
        ) as download:
            with self.assertRaisesRegex(OSError, "not cached"):
                model_loading.resolve_configuration_directory(
                    selection, presets.SD15_PRESET, self.directory, True
                )
        download.assert_called_once_with(selection, self.directory, True)

    def test_configuration_rejects_non_null_image_encoder_but_accepts_null(self) -> None:
        preset = presets.SD15_PRESET
        directory = self.configuration(
            preset, image_encoder=["transformers", "CLIPVisionModelWithProjection"]
        )
        selection = presets.ModelSelection(str(directory), None, True)
        with self.assertRaisesRegex(
            ValueError, "Unsupported component declaration.*image_encoder"
        ):
            model_loading.resolve_configuration_directory(
                selection, preset, self.directory, True
            )
        self.configuration(preset, image_encoder=[None, None])
        self.assertEqual(
            model_loading.resolve_configuration_directory(
                selection, preset, self.directory, True
            ),
            directory.resolve(),
        )

    def test_checkpoint_and_native_denoiser_classification_is_explicit(self) -> None:
        examples = (
            (presets.SD15_PRESET, {"model.diffusion_model.x", "cond_stage_model.x"},
             "checkpoint"),
            (presets.SDXL_BASE_PRESET,
             {"model.diffusion_model.x", "conditioner.embedders.0.x"}, "checkpoint"),
            (presets.SD15_PRESET, {"conv_in.weight", "down_blocks.0.weight"}, "unet"),
            (presets.SDXL_BASE_PRESET, {"model.diffusion_model.x"}, "unet"),
            (presets.FLUX1_SCHNELL_PRESET, {"img_in.weight"}, "transformer"),
            (presets.FLUX1_SCHNELL_PRESET, {"x_embedder.weight"}, "transformer"),
        )
        for preset, keys, expected in examples:
            with self.subTest(preset=preset.name, keys=keys):
                self.assertEqual(model_loading.classify_model_weights(preset, keys), expected)
        for keys in ({"encoder.conv_in.weight"}, {"lora_up.weight"}, set()):
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, "--vae and --lora"):
                model_loading.classify_model_weights(presets.SD15_PRESET, keys)

    def test_full_checkpoint_requires_text_encoders_and_vae_unless_overridden(self) -> None:
        sd15 = {SD15_CLIP_KEY, "first_stage_model.encoder.weight"}
        model_loading.require_checkpoint_components(presets.SD15_PRESET, sd15, False)
        model_loading.require_checkpoint_components(
            presets.SD15_PRESET, {SD15_CLIP_KEY}, True
        )
        with self.assertRaisesRegex(ValueError, "missing VAE"):
            model_loading.require_checkpoint_components(
                presets.SD15_PRESET, {SD15_CLIP_KEY}, False
            )
        with self.assertRaisesRegex(ValueError, "missing text encoder"):
            model_loading.require_checkpoint_components(presets.SD15_PRESET, set(), True)
        with self.assertRaisesRegex(ValueError, "conditioner.embedders.1"):
            model_loading.require_checkpoint_components(
                presets.SDXL_BASE_PRESET, {SDXL_CLIP_KEY}, True
            )

    def test_checkpoint_prefixes_without_exact_clip_sentinels_are_rejected(self) -> None:
        cases = (
            (presets.SD15_PRESET, {"cond_stage_model.transformer.weight"}),
            (presets.SDXL_BASE_PRESET,
             {"conditioner.embedders.0.weight", "conditioner.embedders.1.weight"}),
        )
        for preset, keys in cases:
            with self.subTest(preset=preset.name), self.assertRaisesRegex(
                ValueError, "missing text encoder"
            ):
                model_loading.require_checkpoint_components(preset, keys, True)

    def test_sdxl_requires_both_exact_clip_sentinels(self) -> None:
        keys = {SDXL_CLIP_KEY, SDXL_OPEN_CLIP_KEY}
        model_loading.require_checkpoint_components(presets.SDXL_BASE_PRESET, keys, True)
        for missing in keys:
            with self.subTest(missing=missing), self.assertRaises(ValueError) as error:
                model_loading.require_checkpoint_components(
                    presets.SDXL_BASE_PRESET, keys - {missing}, True
                )
            self.assertIn(missing, str(error.exception))

    def test_meta_parameters_and_buffers_are_rejected(self) -> None:
        model_loading.require_materialized_component(component(), "unet")
        with self.assertRaisesRegex(RuntimeError, "unet.weight.*meta device"):
            model_loading.require_materialized_component(component(meta=True), "unet")
        with_buffer = component()
        with_buffer.named_buffers = lambda: iter((("running", SimpleNamespace(is_meta=True)),))
        with self.assertRaisesRegex(RuntimeError, "vae.running.*meta device"):
            model_loading.require_materialized_component(with_buffer, "vae")

    def test_single_component_uses_local_config_safe_path_dtype_and_no_revision(self) -> None:
        weight = self.weight("vae.safetensor")
        loader = SimpleNamespace(from_single_file=Mock(return_value=component()))
        config = self.configuration(presets.SD15_PRESET)
        result = model_loading.load_single_component(
            loader, weight, config, "vae", self.arguments()
        )
        self.assertIs(result, loader.from_single_file.return_value)
        source = Path(loader.from_single_file.call_args.args[0])
        options = loader.from_single_file.call_args.kwargs
        self.assertEqual(source.suffix, ".safetensors")
        self.assertEqual(options["config"], str(config))
        self.assertTrue(Path(options["config"]).is_dir())
        self.assertEqual(options["subfolder"], "vae")
        self.assertEqual(options["dtype"], "test-float16")
        self.assertTrue(options["local_files_only"])
        self.assertTrue(options["low_cpu_mem_usage"])
        self.assertNotIn("revision", options)
        self.assertFalse(source.exists())

    def test_single_component_rejects_mutation_before_and_during_load(self) -> None:
        config = self.configuration(presets.SD15_PRESET)
        weight = self.weight(content=b"before!")
        loader = SimpleNamespace(from_single_file=Mock(return_value=component()))
        Path(weight.path).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            model_loading.load_single_component(loader, weight, config, "vae", self.arguments())
        loader.from_single_file.assert_not_called()

        weight = self.weight(content=b"before!")

        def mutate(*args, **kwargs):
            Path(weight.path).write_bytes(b"changed")
            return component()

        loader.from_single_file = Mock(side_effect=mutate)
        with self.assertRaisesRegex(RuntimeError, "changed after argument resolution"):
            model_loading.load_single_component(loader, weight, config, "vae", self.arguments())

    def test_same_family_vae_accepts_compatible_weights_and_rejects_latent_contract_drift(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                vae = compatible_vae(preset)
                presets.validate_vae_contract(vae, preset)
                for member, value in (
                    ("in_channels", 1), ("out_channels", 1), ("latent_channels", 8),
                    ("scaling_factor", 0.5), ("shift_factor", 1.0),
                    ("block_out_channels", (128, 256, 512)),
                ):
                    with self.subTest(member=member):
                        invalid = compatible_vae(preset)
                        setattr(invalid.config, member, value)
                        with self.assertRaises(RuntimeError):
                            presets.validate_vae_contract(invalid, preset)
        with self.assertRaisesRegex(RuntimeError, "scaling_factor"):
            presets.validate_vae_contract(compatible_vae(presets.SD15_PRESET),
                                          presets.SDXL_BASE_PRESET)
        with self.assertRaisesRegex(RuntimeError, "latent_channels"):
            presets.validate_vae_contract(compatible_vae(presets.SD15_PRESET),
                                          presets.FLUX1_SCHNELL_PRESET)

    def test_diffusers_base_path_does_not_load_single_file_or_configuration(self) -> None:
        selection = presets.ModelSelection("vendor/model", "b" * 40, False)
        pipeline = SimpleNamespace(components={"unet": component(), "vae": component()})
        with (
            patch.object(model_loading, "load_pipeline", return_value=pipeline) as load,
            patch.object(model_loading, "load_single_component") as single,
            patch.object(model_loading, "resolve_configuration_directory") as config,
        ):
            result, metadata = model_loading.load_generation_pipeline(
                object(), presets.SD15_PRESET, selection, None, None, self.arguments(), {}
            )
        self.assertIs(result, pipeline)
        self.assertEqual(load.call_args.args[1], "vendor/model")
        self.assertEqual(load.call_args.args[2]["revision"], "b" * 40)
        single.assert_not_called()
        config.assert_not_called()
        self.assertEqual(metadata["weights_role"], "diffusers")
        self.assertIsNone(metadata["configuration"])
        self.assertEqual(metadata["component_sources"], {"unet": "model", "vae": "model"})

    def test_vae_override_is_validated_and_injected_before_pipeline_construction(self) -> None:
        preset = presets.SD15_PRESET
        directory = self.configuration(preset)
        selection = presets.ModelSelection(str(directory), None, True)
        weight = self.weight("vae.safetensors")
        vae = compatible_vae(preset)
        events = []

        def load_vae(*args):
            events.append("vae")
            return vae

        def load_pipeline(cls, source, arguments):
            events.append("pipeline")
            self.assertIs(arguments["vae"], vae)
            self.assertNotIn("revision", arguments)
            return SimpleNamespace(components={"vae": vae, "unet": component()})

        with (
            patch.object(model_loading, "load_single_component", side_effect=load_vae),
            patch.object(model_loading, "load_pipeline", side_effect=load_pipeline),
        ):
            _, metadata = model_loading.load_generation_pipeline(
                object(), preset, selection, None, weight, self.arguments(),
                {"AutoencoderKL": object()},
            )
        self.assertEqual(events, ["vae", "pipeline"])
        self.assertEqual(metadata["component_sources"]["vae"], "vae_override")
        self.assertEqual(metadata["vae_override"]["sha256"], weight.sha256)
        self.assertEqual(metadata["configuration"]["id_or_path"], str(directory))

    def test_incompatible_vae_fails_before_pipeline_construction(self) -> None:
        preset = presets.SDXL_BASE_PRESET
        directory = self.configuration(preset)
        selection = presets.ModelSelection(str(directory), None, True)
        with (
            patch.object(model_loading, "load_single_component",
                         return_value=compatible_vae(presets.SD15_PRESET)),
            patch.object(model_loading, "load_pipeline") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "scaling_factor"):
                model_loading.load_generation_pipeline(
                    object(), preset, selection, None, self.weight("vae.safetensors"),
                    self.arguments(), {"AutoencoderKL": object()},
                )
        load.assert_not_called()

    def test_native_unet_and_flux_transformer_use_config_backbone_and_origin_roles(self):
        cases = (
            (presets.SD15_PRESET, {"conv_in.weight", "down_blocks.0.weight"}, "unet",
             "UNet2DConditionModel"),
            (presets.FLUX1_SCHNELL_PRESET, {"img_in.weight"}, "transformer",
             "FluxTransformer2DModel"),
        )
        for preset, keys, role, class_name in cases:
            with self.subTest(preset=preset.name):
                directory = self.configuration(preset)
                config = presets.ModelSelection("vendor/backbone", "d" * 40, False)
                weight = self.weight(f"{role}.safetensors")
                model = component()
                vae = compatible_vae(preset)
                events = []

                def load_component(cls, file, folder, name, arguments):
                    events.append(name)
                    self.assertEqual(folder, directory)
                    return vae if name == "vae" else model

                def load_pipeline(cls, source, arguments):
                    events.append("pipeline")
                    self.assertEqual(source, config.source)
                    self.assertEqual(arguments["revision"], config.requested_revision)
                    self.assertIs(arguments[role], model)
                    self.assertIs(arguments["vae"], vae)
                    return SimpleNamespace(components={role: model, "vae": vae,
                                                       "text_encoder": component()})

                with (
                    patch.object(model_loading, "read_weight_keys", return_value=keys),
                    patch.object(model_loading, "resolve_configuration_directory",
                                 return_value=directory),
                    patch.object(model_loading, "load_single_component",
                                 side_effect=load_component),
                    patch.object(model_loading, "load_pipeline", side_effect=load_pipeline),
                ):
                    _, metadata = model_loading.load_generation_pipeline(
                        object(), preset, self.selection(weight), config,
                        self.weight(f"{role}-vae.safetensors"), self.arguments(),
                        {class_name: object(), "AutoencoderKL": object()},
                    )
                self.assertEqual(events, ["vae", role, "pipeline"])
                self.assertEqual(metadata["weights_role"], role)
                self.assertEqual(metadata["component_sources"][role], "model")
                self.assertEqual(metadata["component_sources"]["text_encoder"], "model_config")
                self.assertEqual(metadata["component_sources"]["vae"], "vae_override")
                self.assertEqual(metadata["configuration"]["requested_revision"], "d" * 40)

    def test_sd_checkpoint_uses_single_file_with_local_config_and_preloaded_vae(self) -> None:
        preset = presets.SD15_PRESET
        directory = self.configuration(preset)
        config = presets.ModelSelection(str(directory), None, True)
        weight = self.weight("checkpoint.safetensor")
        vae = compatible_vae(preset)
        events = []

        def load_vae(*args):
            events.append("vae")
            return vae

        def from_single_file(path, **arguments):
            events.append("pipeline")
            self.assertEqual(Path(path).suffix, ".safetensors")
            self.assertTrue(Path(path).is_file())
            self.assertEqual(arguments["config"], str(directory))
            self.assertTrue(arguments["local_files_only"])
            self.assertNotIn("revision", arguments)
            self.assertIs(arguments["vae"], vae)
            return SimpleNamespace(components={"unet": component(), "vae": vae,
                                               "text_encoder": component(),
                                               "tokenizer": object(), "scheduler": object()})

        pipeline_class = SimpleNamespace(from_single_file=Mock(side_effect=from_single_file))
        keys = {"model.diffusion_model.weight", SD15_CLIP_KEY}
        with (
            patch.object(model_loading, "read_weight_keys", return_value=keys),
            patch.object(model_loading, "load_single_component", side_effect=load_vae),
            patch.object(model_loading, "load_pipeline") as pretrained,
        ):
            _, metadata = model_loading.load_generation_pipeline(
                pipeline_class, preset, self.selection(weight), config,
                self.weight("vae.safetensors"), self.arguments(), {"AutoencoderKL": object()},
            )
        self.assertEqual(events, ["vae", "pipeline"])
        pretrained.assert_not_called()
        self.assertEqual(metadata["weights_role"], "checkpoint")
        self.assertEqual(metadata["component_sources"]["unet"], "model")
        self.assertEqual(metadata["component_sources"]["text_encoder"], "model")
        self.assertEqual(metadata["component_sources"]["tokenizer"], "model_config")
        self.assertEqual(metadata["component_sources"]["scheduler"], "model_config")
        self.assertEqual(metadata["component_sources"]["vae"], "vae_override")

    def test_single_model_metadata_distinguishes_file_format_and_identity(self) -> None:
        weight = self.weight()
        metadata = model_loading.selection_metadata(self.selection(weight))
        self.assertEqual(metadata["format"], "safetensors")
        self.assertTrue(metadata["is_local"])
        self.assertIsNone(metadata["requested_revision"])
        self.assertEqual(metadata["id_or_path"], weight.path)
        self.assertEqual(metadata["sha256"], weight.sha256)

    def test_sd15_checkpoint_loads_declared_safety_checker_from_explicit_config_source(self):
        preset = presets.SD15_PRESET
        directory = self.configuration(
            preset, safety_checker=["stable_diffusion", "StableDiffusionSafetyChecker"]
        )
        config = presets.ModelSelection("vendor/checked-config", "c" * 40, False)
        weight = self.weight("checkpoint.safetensors")
        checker = component()
        events = []

        def load_checker(source, **arguments):
            events.append("safety_checker")
            self.assertEqual(source, config.source)
            self.assertEqual(arguments["revision"], config.requested_revision)
            self.assertEqual(arguments["subfolder"], "safety_checker")
            self.assertEqual(arguments["cache_dir"], self.directory / "cache")
            self.assertTrue(arguments["local_files_only"])
            self.assertTrue(arguments["use_safetensors"])
            self.assertFalse(arguments["trust_remote_code"])
            self.assertEqual(arguments["dtype"], "test-float16")
            self.assertNotIn("add_watermarker", arguments)
            return checker

        def from_single_file(source, **arguments):
            events.append("pipeline")
            self.assertIs(arguments["safety_checker"], checker)
            self.assertEqual(arguments["config"], str(directory))
            self.assertTrue(arguments["local_files_only"])
            return SimpleNamespace(components={
                "unet": component(), "vae": compatible_vae(preset),
                "text_encoder": component(), "safety_checker": checker,
            })

        safety_class = SimpleNamespace(from_pretrained=Mock(side_effect=load_checker))
        pipeline_class = SimpleNamespace(from_single_file=Mock(side_effect=from_single_file))
        keys = {"model.diffusion_model.weight", "first_stage_model.encoder.weight", SD15_CLIP_KEY}
        arguments = {**self.arguments(), "add_watermarker": False}
        with (
            patch.object(model_loading, "read_weight_keys", return_value=keys),
            patch.object(model_loading, "resolve_configuration_directory",
                         return_value=directory),
        ):
            pipeline, metadata = model_loading.load_generation_pipeline(
                pipeline_class, preset, self.selection(weight), config, None, arguments,
                {"StableDiffusionSafetyChecker": safety_class},
            )
        self.assertEqual(events, ["safety_checker", "pipeline"])
        safety_class.from_pretrained.assert_called_once()
        pipeline_class.from_single_file.assert_called_once()
        self.assertIs(pipeline.components["safety_checker"], checker)
        self.assertEqual(metadata["component_sources"]["safety_checker"], "model_config")
        self.assertEqual(metadata["configuration"]["id_or_path"], config.source)
        self.assertEqual(
            metadata["configuration"]["requested_revision"], config.requested_revision
        )

    def test_incomplete_clip_checkpoint_fails_before_config_or_pipeline_loading(self):
        weight = self.weight("incomplete.safetensors")
        cases = (
            (presets.SD15_PRESET, {"cond_stage_model.transformer.weight"}),
            (presets.SDXL_BASE_PRESET,
             {"conditioner.embedders.0.weight", "conditioner.embedders.1.weight"}),
        )
        for preset, clip_keys in cases:
            with self.subTest(preset=preset.name):
                config = presets.ModelSelection(preset.model_id, preset.revision, False)
                loader = SimpleNamespace(from_single_file=Mock())
                keys = clip_keys | {
                    "model.diffusion_model.weight", "first_stage_model.encoder.weight",
                }
                with (
                    patch.object(model_loading, "read_weight_keys", return_value=keys),
                    patch.object(model_loading, "resolve_configuration_directory") as resolve,
                    patch.object(model_loading, "load_single_component") as component_load,
                    patch.object(model_loading, "load_pipeline") as pipeline_load,
                ):
                    with self.assertRaisesRegex(RuntimeError, "missing text encoder"):
                        model_loading.load_generation_pipeline(
                            loader, preset, self.selection(weight), config, None,
                            self.arguments(), {},
                        )
                resolve.assert_not_called()
                component_load.assert_not_called()
                pipeline_load.assert_not_called()
                loader.from_single_file.assert_not_called()

    def test_local_safety_checker_variant_follows_its_own_weights_not_unet_variant(self):
        preset = presets.SD15_PRESET
        directory = self.configuration(
            preset, safety_checker=["stable_diffusion", "StableDiffusionSafetyChecker"]
        )
        unet_directory = directory / "unet"
        unet_directory.mkdir()
        (unet_directory / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"unet")
        safety_directory = directory / "safety_checker"
        safety_directory.mkdir()
        (safety_directory / "model.safetensors").write_bytes(b"safety checker")
        config = presets.ModelSelection(str(directory), None, True)
        weight = self.weight("checkpoint.safetensors")
        keys = {"model.diffusion_model.weight", "first_stage_model.encoder.weight", SD15_CLIP_KEY}

        for has_safety_variant in (False, True):
            with self.subTest(has_safety_variant=has_safety_variant):
                if has_safety_variant:
                    (safety_directory / "model.fp16.safetensors").write_bytes(b"fp16 checker")
                checker = component()
                safety_class = SimpleNamespace(from_pretrained=Mock(return_value=checker))
                pipeline_class = SimpleNamespace(from_single_file=Mock(return_value=(
                    SimpleNamespace(components={"safety_checker": checker})
                )))
                with patch.object(model_loading, "read_weight_keys", return_value=keys):
                    model_loading.load_generation_pipeline(
                        pipeline_class, preset, self.selection(weight), config, None,
                        {**self.arguments(), "variant": "fp16"},
                        {"StableDiffusionSafetyChecker": safety_class},
                    )
                safety_arguments = safety_class.from_pretrained.call_args.kwargs
                if has_safety_variant:
                    self.assertEqual(safety_arguments["variant"], "fp16")
                else:
                    self.assertNotIn("variant", safety_arguments)
                self.assertNotIn("revision", safety_arguments)
                self.assertEqual(safety_class.from_pretrained.call_args.args, (str(directory),))
                self.assertIs(
                    pipeline_class.from_single_file.call_args.kwargs["safety_checker"], checker
                )

    def test_sdxl_checkpoint_uses_two_embedded_encoders_and_disables_watermarker(self):
        preset = presets.SDXL_BASE_PRESET
        directory = self.configuration(preset)
        config = presets.ModelSelection(str(directory), None, True)
        weight = self.weight("xl.safetensors")
        pipeline = SimpleNamespace(components={
            "unet": component(), "vae": compatible_vae(preset),
            "text_encoder": component(), "text_encoder_2": component(),
            "tokenizer": object(), "tokenizer_2": object(),
        })
        loader = SimpleNamespace(from_single_file=Mock(return_value=pipeline))
        keys = {
            "model.diffusion_model.weight", "first_stage_model.encoder.weight",
            SDXL_CLIP_KEY, SDXL_OPEN_CLIP_KEY,
        }
        with patch.object(model_loading, "read_weight_keys", return_value=keys):
            _, metadata = model_loading.load_generation_pipeline(
                loader, preset, self.selection(weight), config, None, self.arguments(), {}
            )
        self.assertFalse(loader.from_single_file.call_args.kwargs["add_watermarker"])
        self.assertEqual(metadata["weights_role"], "checkpoint")
        self.assertEqual(metadata["component_sources"]["text_encoder_2"], "model")
        self.assertEqual(metadata["component_sources"]["tokenizer_2"], "model_config")

    def test_checkpoint_meta_tensor_is_rejected_before_returning_pipeline(self) -> None:
        preset = presets.SD15_PRESET
        directory = self.configuration(preset)
        config = presets.ModelSelection(str(directory), None, True)
        weight = self.weight("checkpoint.safetensors")
        pipeline = SimpleNamespace(components={"unet": component(meta=True)})
        loader = SimpleNamespace(from_single_file=Mock(return_value=pipeline))
        keys = {
            "model.diffusion_model.weight", "first_stage_model.encoder.weight",
            SD15_CLIP_KEY,
        }
        with patch.object(model_loading, "read_weight_keys", return_value=keys):
            with self.assertRaisesRegex(RuntimeError, "unet.weight.*meta device"):
                model_loading.load_generation_pipeline(
                    loader, preset, self.selection(weight), config, None, self.arguments(), {}
                )

    def test_singular_lora_is_loaded_only_through_standard_safetensors_alias(self) -> None:
        weight = self.weight("style.safetensor")
        _, args = self.generate.resolve_arguments(self.generate.build_parser().parse_args([
            "--lora", weight.path, "--lora-scale", "0.5",
        ]))
        selected = args.lora_selection
        self.assertEqual(selected.local_file, weight)
        observed = {}

        def load_weights(source, **arguments):
            path = Path(source) / arguments["weight_name"]
            observed["path"] = path
            self.assertEqual(path.suffix, ".safetensors")
            self.assertEqual(str(path.resolve()), weight.resolved_file)
            self.assertTrue(arguments["use_safetensors"])

        pipeline = SimpleNamespace(
            unet=SimpleNamespace(active_adapters=lambda: ["iild_lora"]),
            load_lora_weights=Mock(side_effect=load_weights),
            set_adapters=Mock(),
            get_list_adapters=lambda: {"unet": ["iild_lora"]},
        )
        activation = self.generate.apply_lora(pipeline, selected, self.directory, True)
        pipeline.set_adapters.assert_called_once_with("iild_lora", adapter_weights=0.5)
        self.assertFalse(observed["path"].exists())
        metadata = self.generate.lora_metadata(selected, activation)
        self.assertEqual(metadata["resolved_file"], weight.resolved_file)
        self.assertEqual(metadata["sha256"], weight.sha256)

    def test_existing_png_or_json_blocks_overwrite_before_dependency_loading(self) -> None:
        for suffix in (".png", ".json"):
            with self.subTest(suffix=suffix):
                output = self.directory / f"existing-{suffix[1:]}.png"
                output.with_suffix(suffix).write_bytes(b"keep existing")
                with (
                    patch.object(sys, "argv", ["generate.py", "--output", str(output)]),
                    patch.object(self.generate, "load_dependencies") as dependencies,
                    patch.object(self.generate, "package_versions") as packages,
                ):
                    with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                        self.generate.main()
                dependencies.assert_not_called()
                packages.assert_not_called()
                self.assertEqual(output.with_suffix(suffix).read_bytes(), b"keep existing")

    def test_generation_parser_combines_explicit_model_vae_lora_without_base_overwrite(self):
        model = self.weight()
        vae = self.weight("vae.safetensor")
        lora = self.weight("style.safetensors")
        directory = self.configuration(presets.SDXL_BASE_PRESET)
        preset, args = self.generate.resolve_arguments(
            self.generate.build_parser().parse_args([
                "--preset", "sdxl-base", "--model", model.path,
                "--model-config", str(directory), "--vae", vae.path,
                "--lora", lora.path, "--lora-scale", "0.75",
            ])
        )
        self.assertEqual(preset, presets.SDXL_BASE_PRESET)
        self.assertEqual(args.model_selection.single_file, model)
        self.assertEqual(args.vae_file, vae)
        self.assertEqual(args.config_selection.source, str(directory))
        self.assertEqual(args.lora_selection.scale, 0.75)
        self.assertNotEqual(args.output.name, preset.generation_filename)
        self.assertTrue(args.output.is_relative_to(ROOT / "build"))

    def test_generation_parser_default_config_is_pinned_and_override_requires_sha(self):
        model = self.weight()
        parser = self.generate.build_parser()
        preset, args = self.generate.resolve_arguments(parser.parse_args(["--model", model.path]))
        self.assertEqual(args.config_selection.source, preset.model_id)
        self.assertEqual(args.config_selection.requested_revision, preset.revision)
        with self.assertRaises(SystemExit):
            self.generate.resolve_arguments(parser.parse_args([
                "--model", model.path, "--model-config", "vendor/config",
            ]))
        _, args = self.generate.resolve_arguments(parser.parse_args([
            "--model", model.path, "--model-config", "vendor/config",
            "--model-config-revision", "e" * 40,
        ]))
        self.assertEqual(args.config_selection.source, "vendor/config")
        self.assertEqual(args.config_selection.requested_revision, "e" * 40)

    def test_generation_parser_rejects_config_options_without_single_file(self) -> None:
        parser = self.generate.build_parser()
        directory = self.configuration(presets.SD15_PRESET)
        for arguments in (
            ["--model-config", str(directory)],
            ["--model-config-revision", "f" * 40],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.generate.resolve_arguments(parser.parse_args(arguments))

    def test_generation_without_replacements_keeps_default_filename_and_model(self) -> None:
        preset, args = self.generate.resolve_arguments(self.generate.build_parser().parse_args([]))
        self.assertIsNone(args.vae_file)
        self.assertIsNone(args.config_selection)
        self.assertIsNone(args.model_selection.single_file)
        self.assertEqual(args.model_selection.source, preset.model_id)
        self.assertEqual(args.output.name, preset.generation_filename)

    def test_custom_nonvariant_configuration_does_not_inherit_preset_fp16_variant(self):
        parser = self.generate.build_parser()
        model = self.weight()
        directory = self.configuration(presets.SDXL_BASE_PRESET)
        weights = directory / "text_encoder" / "model.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"nonvariant fixture")
        cases = (
            ["--model", model.path, "--model-config", str(directory)],
            ["--model", model.path, "--model-config", "vendor/config",
             "--model-config-revision", "a" * 40],
            ["--model", str(directory)],
            ["--model", "vendor/model", "--revision", "b" * 40],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                preset, args = self.generate.resolve_arguments(parser.parse_args([
                    "--preset", "sdxl-base", *arguments,
                ]))
                load = self.generate.build_load_arguments(
                    preset, args, "test-float16", use_weight_variant=True
                )
                self.assertNotIn("variant", load)
                self.assertEqual(load["dtype"], "test-float16")

    def test_local_backbone_variant_is_used_only_when_matching_weights_exist(self):
        directory = self.configuration(presets.SD15_PRESET)
        preset, args = self.generate.resolve_arguments(self.generate.build_parser().parse_args([
            "--model", str(directory),
        ]))
        load = self.generate.build_load_arguments(preset, args, "test-fp16", True)
        self.assertNotIn("variant", load)

        encoder = directory / "text_encoder"
        encoder.mkdir()
        (encoder / "model.fp16.safetensors").write_bytes(b"fp16 variant fixture")
        load = self.generate.build_load_arguments(preset, args, "test-fp16", True)
        self.assertEqual(load["variant"], "fp16")
        self.assertNotIn(
            "variant", self.generate.build_load_arguments(preset, args, "test-fp32", False)
        )

    def test_default_model_and_default_single_file_config_keep_pinned_variant(self):
        parser = self.generate.build_parser()
        model = self.weight()
        for arguments in ([], ["--model", model.path]):
            with self.subTest(arguments=arguments):
                preset, args = self.generate.resolve_arguments(parser.parse_args(arguments))
                load = self.generate.build_load_arguments(preset, args, "test-fp16", True)
                self.assertEqual(load["variant"], "fp16")


if __name__ == "__main__":
    unittest.main()
