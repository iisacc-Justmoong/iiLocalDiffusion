#!/usr/bin/env python3
"""Standalone checkpoint routing, offline resources and artifact publication."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import DownloadedModelTests as fixtures
import standalone_image
import checkpoint_config
import generate


class StandaloneImageTests(unittest.TestCase):
    setUp = fixtures.DownloadedModelTests.setUp
    safetensors = fixtures.DownloadedModelTests.safetensors
    sd = fixtures.DownloadedModelTests.sd

    def checkpoint(self, context=2048, **kwargs):
        return self.safetensors({
            "model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, context],
            "first_stage_model.encoder.conv_in.weight": [1],
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": [1],
        }, **kwargs)

    def test_checkpoint_uses_bundled_config_without_comfy_or_network(self):
        for context, preset, family in ((768, "sd15-compatible", "sd1"), (2048, "sdxl", "sdxl")):
            model = self.checkpoint(context)
            args = standalone_image.build_parser().parse_args(["--model-path", str(model)])
            with patch("subprocess.Popen", side_effect=AssertionError("No server")):
                selected, resolved = standalone_image.resolve_arguments(args)
            self.assertEqual(selected.name, preset)
            self.assertEqual(Path(resolved.config_selection.source), checkpoint_config.CONFIGS / family)
            self.assertTrue(resolved.local_files_only)

    def test_adapter_split_and_cross_family_are_rejected_before_loading(self):
        cases = [(self.sd(2048, name="split.safetensors"), [], "components"),
                 (self.checkpoint(name="full.safetensors"), ["--base-model", "SD 1.5"], "conflicts"),
                 (self.safetensors({"lora_unet.foo.lora_up.weight": [1, 1]}, name="lora.safetensors"), [], "lora")]
        for model, extra, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((ValueError, SystemExit), message):
                standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
                    ["--model", str(model), *extra]))

    def test_sdxl_checkpoint_without_vae_reaches_pipeline_with_bundled_override(self):
        from model_loading import load_generation_pipeline
        from unittest.mock import Mock
        shapes = {
            "model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 2048],
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": [1],
            "conditioner.embedders.1.model.positional_embedding": [1],
        }
        model = self.safetensors(shapes)
        preset, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
            ["--model", str(model), "--no-default-modifiers"]))
        self.assertIsNone(args.vae_file)
        self.assertEqual(args.vae_status, "fallback")
        vae = SimpleNamespace(config=SimpleNamespace(in_channels=3, out_channels=3, latent_channels=4,
            block_out_channels=(128, 256, 512, 512), scaling_factor=0.13025),
            named_parameters=lambda: [], named_buffers=lambda: [])
        result = SimpleNamespace(components={"vae": vae})
        with patch("model_loading.read_weight_keys", return_value=set(shapes)), \
                patch("model_loading._load_checkpoint_pipeline", return_value=result) as assemble:
            pipeline, metadata = load_generation_pipeline(object, preset, args.model_selection, args.config_selection,
                None, {"dtype": "float32", "cache_dir": self.directory},
                {"AutoencoderKL": SimpleNamespace(from_pretrained=Mock(return_value=(vae, {})))},
                vae_selection=args.vae_selection, vae_status=args.vae_status)
        self.assertIs(pipeline, result)
        self.assertIs(assemble.call_args.args[5]["vae"], vae)
        self.assertEqual(metadata["component_sources"]["vae"], "vae_fallback")

    def test_missing_text_encoder_is_not_hidden_by_vae_fallback(self):
        model = self.sd(2048, name="denoiser-only.safetensors")
        with self.assertRaisesRegex((ValueError, SystemExit), "components"):
            standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(["--model", str(model)]))

    def test_explicit_config_and_prediction_metadata_are_preserved(self):
        model = self.checkpoint(metadata={"prediction_type": "v_prediction"})
        config = checkpoint_config.CONFIGS / "sdxl"
        _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
            ["--model", str(model), "--model-config", str(config)]))
        self.assertEqual(args.config_selection.source, str(config))
        self.assertEqual(args.prediction_type, "v_prediction")

    def test_marker_only_vpred_checkpoint_configures_both_prediction_and_noise_schedule(self):
        model = self.safetensors({
            "model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 2048],
            "first_stage_model.encoder.conv_in.weight": [1],
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": [1],
            "v_pred": [0], "ztsnr": [0],
        })
        _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
            ["--model", str(model)]))
        self.assertEqual(args.prediction_type, "v_prediction")
        self.assertEqual(args.scheduler_config["rescale_betas_zero_snr"], True)
        self.assertEqual(args.scheduler_config["timestep_spacing"], "trailing")
        _, explicit = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(
            ["--model", str(model), "--scheduler-config", '{"timestep_spacing":"linspace"}']))
        self.assertEqual(explicit.scheduler_config["timestep_spacing"], "linspace")

    def test_bundled_resources_match_manifest_and_contain_no_weights(self):
        manifest = json.loads((checkpoint_config.CONFIGS / "manifest.json").read_text())
        for family, record in manifest["families"].items():
            self.assertEqual(len(record["revision"]), 40)
            for name, identity in record["files"].items():
                data = (checkpoint_config.CONFIGS / family / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"], f"{family}/{name}: SHA-256")
                self.assertEqual(len(data), identity["size_bytes"], f"{family}/{name}: size")
                self.assertNotIn(Path(name).suffix, (".safetensors", ".bin", ".ckpt"))
            self.assertTrue((checkpoint_config.CONFIGS / family / "tokenizer/merges.txt").is_file())

    def test_animation_single_file_uses_same_offline_configuration(self):
        for mode in ("2D", "Interpolator"):
            _, args = generate.resolve_request({"model": str(self.checkpoint()),
                                               "animation_mode": mode})
            self.assertEqual(Path(args.config_selection.source), checkpoint_config.CONFIGS / "sdxl")

    def test_uppercase_singular_suffix_preserves_original_and_safe_alias(self):
        from weight_files import checked_safetensors_path
        model = self.checkpoint(name="MODEL.SAFETENSOR")
        _, args = standalone_image.resolve_arguments(standalone_image.build_parser().parse_args(["--model", str(model)]))
        self.assertEqual(args.model_selection.source, str(model))
        with checked_safetensors_path(args.model_selection.single_file, self.directory / "aliases", "model") as path:
            self.assertEqual(path.suffix, ".safetensors")
            self.assertEqual(path.resolve(), model)

    def test_output_failure_preserves_existing_and_never_publishes_completion(self):
        output = self.directory / "result"
        output.mkdir()
        with patch("generate.run", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(SystemExit, "inference failed"):
                standalone_image.main(["--model", str(self.checkpoint()), "--output-dir", str(output)])
        self.assertEqual(list(output.iterdir()), [])
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(SystemExit, "empty"):
            standalone_image.main(["--model", str(self.checkpoint()), "--output-dir", str(output)])
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for real PNG validation")
    def test_success_publishes_batch_and_final_paths_together(self):
        from PIL import Image
        output = self.directory / "result"
        def render(preset, args):
            self.assertFalse(output.exists(), "The public batch must remain absent until publication")
            for index in range(args.num_images):
                path = args.output.with_stem(f"image-{index:04d}")
                Image.new("RGB", (64, 64), "red").save(path)
                path.with_suffix(".json").write_text(json.dumps({"fixture": {"seed": args.seed + index}, "output": {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}))
            return 0
        with patch("generate.run", side_effect=render), patch("secrets.randbits", return_value=456) as random:
            self.assertEqual(standalone_image.main(["--model", str(self.checkpoint()),
                "--num-images", "2", "--output-dir", str(output)]), 0)
        manifest = json.loads((output / "generation.json").read_text())
        random.assert_called_once_with(32)
        self.assertEqual([record["fixture"]["seed"] for record in manifest["images"]], [456, 457])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["outputs"]), 2)
        for record in manifest["outputs"]:
            self.assertEqual(Path(record["path"]).parent, output)
            self.assertTrue(Path(record["path"]).is_file())
        self.assertEqual(list(self.directory.glob(".iild-generation-*")), [])

    def test_model_changed_during_inference_cannot_publish_success(self):
        model = self.checkpoint()
        output = self.directory / "changed-result"
        def mutate(preset, args):
            model.write_bytes(b"changed")
            return 0
        with patch("generate.run", side_effect=mutate), self.assertRaisesRegex(SystemExit, "changed"):
            standalone_image.main(["--model", str(model), "--output-dir", str(output)])
        self.assertFalse(output.exists(), "A changed model must not publish an output directory")

    def test_router_local_is_standalone_and_comfy_is_explicit(self):
        spec = importlib.util.spec_from_file_location("standalone_router", ROOT / "reference/generate.py")
        router = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(router)
        with patch.object(standalone_image, "main", return_value=0) as run:
            router.main(["--model-path", str(self.checkpoint()), "--work-dir", str(self.directory / "work")])
        run.assert_called_once()

    def test_legacy_checkpoint_is_converted_before_native_planning_and_replays_original(self):
        import checkpoint_conversion
        import native_image
        legacy = self.directory / 'legacy.safetensors'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.sd(1024, name='converted.safetensors')
        inspection = {'architecture': 'sd2', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch', 'component_slots': ['vae', 'clip_l'], 'base_model': None}
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(legacy),
                      'output': checkpoint_conversion._identity(converted)}
        args = standalone_image.build_parser().parse_args(['--model', str(legacy), '--engine', 'native',
            '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers', '--no-hires-fix'])
        with patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as materialize:
            _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(resolved.engine, 'native')
        self.assertEqual(resolved.model, str(converted.resolve()))
        self.assertEqual(resolved.model_conversion, conversion)
        self.assertEqual(native_image.configuration_values(resolved)['model'], str(legacy))
        self.assertEqual(materialize.call_args.args, (str(legacy), self.directory / 'cache' / 'checkpoint-conversions'))

    def test_source_change_between_inspection_and_conversion_is_rejected(self):
        import checkpoint_conversion
        legacy = self.directory / 'legacy.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.sd(1024, name='converted.safetensors')
        inspection = {'architecture': 'sd2', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch', 'component_slots': ['vae', 'clip_l'], 'base_model': None}
        def changed(*_args):
            legacy.write_bytes(b'PK\x03\x04' + b'x' * 20)
            return {'converted': True, 'converted_path': str(converted),
                    'original': checkpoint_conversion._identity(legacy),
                    'output': checkpoint_conversion._identity(converted)}
        args = standalone_image.build_parser().parse_args(['--model', str(legacy), '--engine', 'native',
            '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers'])
        with patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                standalone_image.resolve_arguments(args)

    def test_shard_change_between_inspection_and_conversion_is_rejected(self):
        import checkpoint_conversion
        from weight_files import file_signature
        index = self.directory / 'model-index.json'
        shard = self.directory / 'part.bin'
        index.write_text('{"weight_map":{"weight":"part.bin"}}')
        shard.write_bytes(b'original shard')
        converted = self.sd(1024, name='converted-sharded.safetensors')
        inspection = {'architecture': 'sd2', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch-index', 'component_slots': ['vae', 'clip_l'], 'base_model': None,
                      'sharded_index': {'shards': [str(shard)],
                          'source_signatures': [list(file_signature(index)), list(file_signature(shard))]}}
        def changed(*_args):
            shard.write_bytes(b'changed shard')
            return {'converted': True, 'converted_path': str(converted),
                    'original': checkpoint_conversion._identity(index),
                    'output': checkpoint_conversion._identity(converted)}
        args = standalone_image.build_parser().parse_args(['--model', str(index), '--engine', 'native',
            '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers'])
        with patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'shard changed'):
                standalone_image.resolve_arguments(args)

    def test_unsupported_legacy_architecture_is_rejected_before_conversion(self):
        import checkpoint_conversion
        legacy = self.directory / 'unknown.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        inspection = {'architecture': 'unknown', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch', 'component_slots': [], 'base_model': None}
        for engine in ('native', 'auto'):
            args = standalone_image.build_parser().parse_args(['--model', str(legacy), '--engine', engine])
            with self.subTest(engine=engine), \
                 patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
                 patch.object(standalone_image, 'inspect_checkpoint', side_effect=ValueError('unsupported architecture')), \
                 patch.object(checkpoint_conversion, 'materialize_safetensors') as convert:
                with self.assertRaisesRegex(ValueError, 'architecture|native image backend'):
                    standalone_image.resolve_arguments(args)
                convert.assert_not_called()

    def test_legacy_preflight_and_changed_source_do_not_publish(self):
        import checkpoint_conversion
        import native_image
        legacy = self.directory / 'legacy.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.sd(1024, name='converted.safetensors')
        inspection = {'architecture': 'sd2', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch', 'component_slots': ['vae', 'clip_l'], 'base_model': None}
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(legacy),
                      'output': checkpoint_conversion._identity(converted)}
        output = self.directory / 'result'
        arguments = ['--model', str(legacy), '--engine', 'native', '--output-dir', str(output),
                     '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers', '--no-hires-fix']
        with patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion) as materialize:
            with self.assertRaisesRegex(SystemExit, 'inspect-model'):
                standalone_image.main([*arguments, '--validate-only'])
            materialize.assert_not_called()
            def mutate(_preset, _args):
                legacy.write_bytes(b'changed after conversion')
                return 0
            with patch.object(native_image, 'run', side_effect=mutate), \
                 self.assertRaisesRegex(SystemExit, 'Source checkpoint changed'):
                standalone_image.main(arguments)
        self.assertFalse(output.exists())

    def test_legacy_sdxl_uses_preset_and_retains_original_for_replay(self):
        import checkpoint_conversion
        import downloaded_model
        import generation_config
        legacy = self.directory / 'original.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.checkpoint(name='converted.safetensors')
        current_inspect = downloaded_model.inspect_downloaded_model
        inspection = dict(current_inspect(converted), format='pytorch', path=str(legacy))
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(legacy),
                      'output': checkpoint_conversion._identity(converted)}
        args = standalone_image.build_parser().parse_args(['--model', str(legacy),
            '--cache-dir', str(self.directory / 'cache'), '--no-default-modifiers', '--no-hires-fix'])
        def inspect(path, info=None):
            return inspection if str(path) == str(legacy) else current_inspect(path, info)
        with patch.object(standalone_image, 'inspect_downloaded_model', side_effect=inspect), \
             patch.object(checkpoint_config, 'inspect_downloaded_model', side_effect=inspect), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion):
            preset, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(preset.name, 'sdxl')
        self.assertEqual(resolved.model_selection.single_file.path, str(converted))
        self.assertEqual(generation_config.configuration_values(resolved)['model'], str(legacy))

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow is required for PNG validation')
    def test_legacy_native_output_retains_conversion_provenance(self):
        from PIL import Image
        import checkpoint_conversion
        import native_image
        legacy = self.directory / 'original.ckpt'
        legacy.write_bytes(b'PK\x03\x04' + bytes(20))
        converted = self.sd(1024, name='converted.safetensors')
        inspection = {'architecture': 'sd2', 'role': 'checkpoint', 'task': 'text-to-image',
                      'format': 'pytorch', 'component_slots': ['vae', 'clip_l'], 'base_model': None}
        conversion = {'converted': True, 'converted_path': str(converted),
                      'original': checkpoint_conversion._identity(legacy),
                      'output': checkpoint_conversion._identity(converted)}
        class Engine:
            def image(self, args, seed, *, prepare=False, **kwargs):
                return (None, {'weight_storage': 'anonymous'}) if prepare else (Image.new('RGB', (64, 64), 'red'), {})
        output = self.directory / 'native-result'
        with patch.object(standalone_image, 'inspect_downloaded_model', return_value=inspection), \
             patch.object(checkpoint_conversion, 'materialize_safetensors', return_value=conversion), \
             patch.object(native_image, 'cached_pipeline', return_value=(Engine(), False)), \
             patch.object(native_image, 'record_execution'):
            self.assertEqual(standalone_image.main(['--model', str(legacy), '--engine', 'native',
                '--output-dir', str(output), '--cache-dir', str(self.directory / 'cache'),
                '--width', '64', '--height', '64', '--steps', '1',
                '--no-default-modifiers', '--no-hires-fix']), 0)
        report = json.loads((output / 'image.json').read_text())
        self.assertEqual(report['model_conversion']['original']['path'], str(legacy))
        self.assertEqual(report['model']['path'], str(converted))
        self.assertEqual(json.loads((output / 'generation.json').read_text())['status'], 'complete')


if __name__ == "__main__":
    unittest.main()
