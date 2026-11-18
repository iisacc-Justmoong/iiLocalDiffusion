#!/usr/bin/env python3
"""VAE omission, embedded precedence and latent-family safety contracts."""
import resource_fixture
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from vae_defaults import resolve_vae_selection, verify_vae_selection, load_selected_vae


class VaeDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = resource_fixture.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(vae=None, pipeline_class=None, model=str(self.root),
                                    source_kind="directory", generation_resources=None)
        self.index = {"_class_name": "QwenImagePipeline", "vae": ["diffusers", "AutoencoderKLQwenImage"]}

    def resolve(self):
        return resolve_vae_selection(self.args, self.index, self.root)

    @staticmethod
    def synthetic_weights(path, name):
        header = json.dumps({name: {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
        header += b' ' * (-len(header) % 8)
        path.write_bytes(struct.pack('<Q', len(header)) + header + b'x')

    def setup_single_file_vae(self):
        self.index = {'_class_name': 'FakePipeline', 'vae': ['diffusers', 'AutoencoderKL']}
        folder = self.root / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL'}))
        source_dir = self.root / 'external';source_dir.mkdir()
        return source_dir

    def test_explicit_vae_file_uses_local_config_and_checked_temporary_directory(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'artist.vae';self.synthetic_weights(source, 'decoder.weight')
        before = source.read_bytes()
        self.args.vae = str(source)
        selection, status = self.resolve()
        self.assertEqual((status, selection.source_kind), ('explicit', 'file'))
        self.assertEqual(selection.weight.path, str(source))
        observed = []
        model = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        def load(directory, **kwargs):
            path = Path(directory);observed.append(path)
            self.assertEqual((path / 'diffusion_pytorch_model.safetensors').read_bytes(), before)
            self.assertEqual(json.loads((path / 'config.json').read_text())['_class_name'], 'AutoencoderKL')
            self.assertTrue(kwargs['output_loading_info'])
            return model, {}
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock(side_effect=load)))
        self.assertIs(load_selected_vae(selection, runtime, 'dtype', cache_dir=self.root / 'cache'), model)
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(observed[0].exists())
        verify_vae_selection(selection)

    def test_pytorch_vae_selection_does_not_deserialize_and_tracks_identity(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'mislabelled.safetensors'
        source.write_bytes(b'PK\x03\x04synthetic weights')
        self.args.vae = str(source)
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            selected, status = self.resolve()
            self.assertEqual((status, selected.source_kind, selected.container_kind), ('explicit', 'file', 'pytorch'))
            self.assertEqual(selected.weight.path, str(source))
            verify_vae_selection(selected)
            convert.assert_not_called()
        source.write_bytes(b'PK\x03\x04changed weights')
        with self.assertRaisesRegex(RuntimeError, 'VAE changed'):
            verify_vae_selection(selected)

    def test_pytorch_vae_source_change_during_loading_is_rejected(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'weights.pt';source.write_bytes(b'PK\x03\x04fixture')
        safe = self.root / 'converted.safetensors';self.synthetic_weights(safe, 'decoder.weight')
        self.args.vae = str(source)
        selection, _ = self.resolve()
        observed = []
        model = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        def load(directory, **_kwargs):
            observed.append(Path(directory))
            source.write_bytes(b'PK\x03\x04changed')
            return model, {}
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock(side_effect=load)))
        conversion = {'converted': True, 'converted_path': str(safe)}
        with patch('checkpoint_conversion.materialize_safetensors', return_value=conversion), \
             patch('checkpoint_conversion.verify_conversion_source'):
            with self.assertRaisesRegex(RuntimeError, 'VAE changed'):
                load_selected_vae(selection, runtime, 'dtype', cache_dir=self.root / 'cache')
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())

    def test_explicit_vae_directory_selects_pytorch_or_safe_index(self):
        self.setup_single_file_vae()
        folder = self.root / 'vae'
        source = folder / 'diffusion_pytorch_model.bin'
        source.write_bytes(b'PK\x03\x04synthetic weights')
        self.args.vae = str(folder)
        selected, status = self.resolve()
        self.assertEqual((status, selected.source_kind, selected.container_kind), ('explicit', 'file', 'pytorch'))
        self.assertEqual(Path(selected.weight.path), source)
        shard = folder / 'part.safetensors';self.synthetic_weights(shard, 'decoder.weight')
        index = folder / 'diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'decoder.weight': shard.name}}))
        selected, _ = self.resolve()
        self.assertEqual(Path(selected.weight.path), index)
        self.assertEqual(len(selected.weight.shards), 1)

    def test_vae_directory_selects_unambiguous_or_explicit_weight_variant(self):
        self.setup_single_file_vae()
        folder = self.root / 'vae'
        fp16 = folder / 'diffusion_pytorch_model.fp16.safetensors'
        bf16 = folder / 'diffusion_pytorch_model.bf16.safetensors'
        self.synthetic_weights(fp16, 'decoder.weight')
        self.args.vae = str(folder)
        selected, _ = self.resolve()
        self.assertEqual((selected.source_kind, Path(selected.weight.path)), ('file', fp16))
        self.synthetic_weights(bf16, 'decoder.weight')
        with self.assertRaisesRegex(ValueError, '[Aa]mbiguous'):
            self.resolve()
        self.args.variant = 'bf16'
        selected, _ = self.resolve()
        self.assertEqual(Path(selected.weight.path), bf16)
        self.args.variant = 'absent'
        with self.assertRaisesRegex(ValueError, 'variant'):
            self.resolve()
        default = folder / 'diffusion_pytorch_model.safetensors'
        self.synthetic_weights(default, 'decoder.weight')
        selected, _ = self.resolve()
        self.assertEqual((selected.source_kind, Path(selected.weight.path)), ('directory', default))

    def test_vae_directory_finds_variant_shard_indexes(self):
        self.setup_single_file_vae()
        folder = self.root / 'vae'
        shard = folder / 'part.safetensors';self.synthetic_weights(shard, 'decoder.weight')
        self.args.vae = str(folder)
        for name in ('diffusion_pytorch_model.fp16.safetensors.index.json',
                     'diffusion_pytorch_model.safetensors.index.fp16.json'):
            with self.subTest(name=name):
                index = folder / name
                index.write_text(json.dumps({'weight_map': {'decoder.weight': shard.name}}))
                selected, _ = self.resolve()
                self.assertEqual(Path(selected.weight.path), index)
                self.assertEqual(len(selected.weight.shards), 1)
                index.unlink()

    def test_pytorch_vae_index_tracks_all_shards_without_deserialization(self):
        source_dir = self.setup_single_file_vae()
        first, second = source_dir / 'one.bin', source_dir / 'two.pth'
        payload = pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture'
        first.write_bytes(payload);second.write_bytes(payload)
        index = source_dir / 'model.index.json'
        index.write_text(json.dumps({'weight_map': {'a.weight': first.name, 'b.weight': second.name}}))
        self.args.vae = str(index)
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            selected, _ = self.resolve()
            self.assertEqual(selected.container_kind, 'pytorch-index')
            self.assertEqual({Path(shard.path) for shard in selected.weight.shards}, {first, second})
            verify_vae_selection(selected)
            convert.assert_not_called()
        second.write_bytes(b'PK\x03\x04changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_vae_selection(selected)

    def test_moss_pytorch_vae_is_selected_without_deserializing(self):
        self.index = {'_class_name': 'MOSSImagePipeline', 'vae': ['diffusers', 'AutoencoderKLFlux2']}
        folder = self.root / 'vae';folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'_class_name': 'AutoencoderKLFlux2',
            'latent_channels': 32, 'in_channels': 3, 'out_channels': 3,
            'block_out_channels': [8, 8, 8, 8], 'patch_size': [2, 2]}))
        source = self.root / 'moss-vae.pt';source.write_bytes(b'PK\x03\x04synthetic weights')
        self.args.vae = str(source)
        with patch('checkpoint_conversion.materialize_safetensors') as convert:
            selected, status = self.resolve()
            self.assertEqual((status, selected.source_kind, selected.container_kind), ('explicit', 'file', 'pytorch'))
            self.assertEqual(selected.class_name, 'AutoencoderKLFlux2')
            convert.assert_not_called()

    def test_explicit_vae_index_stages_all_shards_and_detects_source_mutation(self):
        source_dir = self.setup_single_file_vae()
        first, second = source_dir / 'part-a.safetensors', source_dir / 'part-b.safetensors'
        self.synthetic_weights(first, 'a.weight');self.synthetic_weights(second, 'b.weight')
        index = source_dir / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a.weight': first.name, 'b.weight': second.name}}))
        self.args.vae = str(index)
        selection, status = self.resolve()
        self.assertEqual((status, selection.source_kind, len(selection.weight.shards)), ('explicit', 'file', 2))
        original = {p: p.read_bytes() for p in (first, second, index)}
        model = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        def load(directory, **_kwargs):
            folder = Path(directory)
            staged = json.loads((folder / 'diffusion_pytorch_model.safetensors.index.json').read_text())
            self.assertEqual(staged['metadata'], {})
            self.assertEqual((folder / staged['weight_map']['a.weight']).read_bytes(), original[first])
            self.assertEqual((folder / staged['weight_map']['b.weight']).read_bytes(), original[second])
            return model, {}
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock(side_effect=load)))
        self.assertIs(load_selected_vae(selection, runtime, 'dtype', cache_dir=self.root / 'cache'), model)
        self.assertEqual({p: p.read_bytes() for p in original}, original)
        second.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_vae_selection(selection)

    def test_explicit_vae_index_flattens_shards_that_conflict_with_overlay_names(self):
        source_dir = self.setup_single_file_vae()
        reserved = source_dir / 'config.json'
        reserved.mkdir()
        shard = reserved / 'part.safetensors'
        self.synthetic_weights(shard, 'a.weight')
        index = source_dir / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a.weight': 'config.json/part.safetensors'}}))
        before = shard.read_bytes()
        self.args.vae = str(index)
        selection, _ = self.resolve()
        model = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        def load(directory, **_kwargs):
            folder = Path(directory)
            self.assertEqual(json.loads((folder / 'config.json').read_text())['_class_name'], 'AutoencoderKL')
            staged = json.loads((folder / 'diffusion_pytorch_model.safetensors.index.json').read_text())
            self.assertNotIn('/', staged['weight_map']['a.weight'])
            self.assertEqual((folder / staged['weight_map']['a.weight']).read_bytes(), before)
            return model, {}
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock(side_effect=load)))
        self.assertIs(load_selected_vae(selection, runtime, 'dtype', cache_dir=self.root / 'cache'), model)
        self.assertEqual(shard.read_bytes(), before)

    def test_explicit_vae_file_requires_a_matching_local_configuration(self):
        self.setup_single_file_vae()
        source = self.root / 'external/model.safetensors';self.synthetic_weights(source, 'weight')
        self.args.vae = str(source)
        (self.root / 'vae/config.json').write_text(json.dumps({'_class_name': 'AutoencoderKLFlux2'}))
        with self.assertRaisesRegex(ValueError, 'AutoencoderKL'):
            self.resolve()

    def test_vae_weights_named_config_json_use_pipeline_configuration(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'config.json';self.synthetic_weights(source, 'weight')
        self.args.vae = str(source)
        selection, status = self.resolve()
        self.assertEqual(status, 'explicit')
        self.assertEqual(selection.weight.path, str(source))
        self.assertEqual(selection.config_path, str(self.root / 'vae/config.json'))

    def test_deleted_vae_file_is_reported_as_a_changed_selection(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'vae.safetensors';self.synthetic_weights(source, 'weight')
        self.args.vae = str(source)
        selection, _ = self.resolve()
        source.unlink()
        with self.assertRaisesRegex(RuntimeError, 'VAE changed'):
            verify_vae_selection(selection)

    def test_explicit_vae_uses_verified_fallback_config_without_fallback_weights(self):
        self.index = {'_class_name': 'FluxPipeline', 'vae': ['diffusers', 'AutoencoderKL']}
        source = self.root / 'explicit.safetensors';self.synthetic_weights(source, 'weight')
        self.args.vae = str(source)
        resources = self.root / 'resources';(resources / 'vae').mkdir(parents=True)
        config = resources / 'vae/config.json'
        config.write_text(json.dumps({'_class_name': 'AutoencoderKL', 'latent_channels': 16,
            'in_channels': 3, 'out_channels': 3, 'block_out_channels': [8, 8, 8, 8],
            'scaling_factor': .3611, 'shift_factor': .1159}))
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        entry = {'families': ['flux1'], 'class_name': 'AutoencoderKL', 'file': 'missing.safetensors',
                 'size': 1, 'sha256': '0' * 64,
                 'config': {'file': 'vae/config.json', 'size': config.stat().st_size, 'sha256': digest}}
        (resources / 'generation-defaults.json').write_text(json.dumps({'version': 1, 'fallback_vaes': [entry]}))
        self.args.generation_resources = str(resources)
        selection, status = self.resolve()
        self.assertEqual((status, selection.config_path), ('explicit', str(config)))
        self.assertFalse((resources / 'missing.safetensors').exists())
        verify_vae_selection(selection)
        config.write_text(config.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.resolve()

    def test_explicit_vae_overlay_cannot_be_created_inside_config_directory(self):
        source_dir = self.setup_single_file_vae()
        source = source_dir / 'model.safetensors';self.synthetic_weights(source, 'weight')
        self.args.vae = str(source)
        selection, _ = self.resolve()
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock()))
        with self.assertRaisesRegex(ValueError, 'cache.*VAE'):
            load_selected_vae(selection, runtime, 'dtype', cache_dir=self.root / 'vae/cache')
        runtime.AutoencoderKL.from_pretrained.assert_not_called()
        self.assertFalse((self.root / 'vae/cache').exists())

    def test_vae_cache_symlinks_cannot_write_inside_configuration_directory(self):
        source_dir = self.setup_single_file_vae()
        cache = self.root / 'cache';cache.mkdir()
        config_dir = self.root / 'vae'
        runtime = SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=Mock()))
        for child, extension, data in (('vae-overlays', 'safetensors', None),
                                       ('vae-checkpoint-conversions', 'pt', b'PK\x03\x04fixture')):
            with self.subTest(child=child):
                source = source_dir / ('weights.' + extension)
                if data is None:
                    self.synthetic_weights(source, 'decoder.weight')
                else:
                    source.write_bytes(data)
                destination = config_dir / child;destination.mkdir()
                link = cache / child
                try:
                    link.symlink_to(destination, target_is_directory=True)
                except OSError:
                    self.skipTest('Directory symlinks are unavailable')
                self.args.vae = str(source)
                selection, _ = self.resolve()
                with patch('checkpoint_conversion.materialize_safetensors') as convert:
                    with self.assertRaisesRegex(ValueError, 'cache.*VAE'):
                        load_selected_vae(selection, runtime, 'dtype', cache_dir=cache)
                    convert.assert_not_called()
                self.assertEqual(list(destination.iterdir()), [])
                link.unlink()
        runtime.AutoencoderKL.from_pretrained.assert_not_called()

    def test_metadata_mode_loads_fallback_without_weight_checksum_scan(self):
        with patch.dict(os.environ, {"IILD_MODEL_VALIDATION": "metadata"}), \
                patch("weight_files.cached_model_sha256", side_effect=AssertionError("Full model hash")):
            for name in ("QwenImagePipeline", "StableDiffusionXLPipeline", "FluxPipeline"):
                self.index = {"_class_name": name}
                selection, status = self.resolve()
                self.assertEqual(status, "fallback")
                self.assertIsNone(selection.weight.sha256)
                verify_vae_selection(selection)

    def test_omission_loads_verified_official_qwen_vae(self):
        selection, status = self.resolve()
        self.assertEqual(status, "fallback")
        self.assertEqual(selection.weight.sha256, "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344")
        self.assertEqual(selection.class_name, "AutoencoderKLQwenImage")
        verify_vae_selection(selection)

    def test_explicitly_null_or_omitted_component_gets_fallback(self):
        for value in ([None, None], None):
            self.index["vae"] = value
            self.assertEqual(self.resolve()[1], "fallback")

    def test_fallback_is_independent_of_optional_style_modifiers(self):
        self.args.default_modifiers = False
        self.assertEqual(self.resolve()[1], "fallback")

    def test_embedded_vae_is_preserved_even_if_defaults_are_unavailable(self):
        vae = self.root / "vae"
        vae.mkdir()
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"present")
        self.args.generation_resources = self.root / "absent"
        self.assertEqual(self.resolve(), (None, "model"))

    def test_model_owned_alternative_vae_architecture_stays_with_its_loader(self):
        self.index = {"_class_name": "StableDiffusionXLPipeline", "vae": ["diffusers", "AutoencoderTiny"]}
        (self.root / "vae").mkdir()
        (self.root / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"embedded")
        self.args.generation_resources = self.root / "absent"
        self.assertEqual(self.resolve(), (None, "model"))

    def test_embedded_single_file_vae_is_preserved(self):
        model = self.root / "checkpoint.safetensors"
        header = json.dumps({"first_stage_model.decoder.conv_in.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        model.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.args.source_kind, self.args.model = "single-file", str(model)
        self.assertEqual(self.resolve(), (None, "model"))

    def test_embedded_vae_is_found_across_standalone_shards(self):
        mapping = {}
        for name, filename in (('model.weight', 'model-1.safetensors'),
                               ('first_stage_model.decoder.conv_in.weight', 'model-2.safetensors')):
            header = json.dumps({name: {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
            (self.root / filename).write_bytes(struct.pack('<Q', len(header)) + header + b'x')
            mapping[name] = filename
        model = self.root / 'model.safetensors.index.json'
        model.write_text(json.dumps({'weight_map': mapping}))
        self.args.source_kind, self.args.model = 'single-file', str(model)
        self.assertEqual(self.resolve(), (None, 'model'))

    def test_other_latent_families_and_rgba_never_receive_qwen_rgb_weights(self):
        for name in ("StableDiffusionPipeline", "StableDiffusion3Pipeline",
                     "WanPipeline", "QwenImageLayeredPipeline", "QwenImage2Pipeline"):
            self.index["_class_name"] = name
            self.assertEqual(self.resolve(), (None, "not-configured-for-pipeline"))

    def test_sdxl_and_flux_select_their_own_latent_space_without_vae_argument(self):
        for name, cls, family, channels in (
                ("StableDiffusionXLPipeline", "AutoencoderKL", "sdxl", 4),
                ("FluxPipeline", "AutoencoderKL", "flux1", 16),
                ("Flux2Pipeline", "AutoencoderKLFlux2", "flux2", 32),
                ("Flux2KleinPipeline", "AutoencoderKLFlux2", "flux2", 32),
                ("MOSSImagePipeline", "AutoencoderKLFlux2", "flux2", 32)):
            for component in (None, [None, None], ["diffusers", cls]):
                with self.subTest(pipeline=name, component=component):
                    self.index = {"_class_name": name, "vae": component}
                    selected, status = self.resolve()
                    self.assertEqual(status, "fallback")
                    self.assertEqual(selected.class_name, cls)
                    self.assertIn("/vae/" + family + "/", Path(selected.directory).as_posix() + "/")
                    self.assertEqual(json.loads(Path(selected.config_path).read_text())["latent_channels"], channels)

    def test_flux1_vae_is_not_accepted_by_sdxl_or_flux2(self):
        self.index = {"_class_name": "FluxPipeline"}
        self.args.vae = self.resolve()[0].directory
        for name in ("StableDiffusionXLPipeline", "Flux2Pipeline"):
            self.index = {"_class_name": name}
            with self.assertRaisesRegex(ValueError, "VAE"):
                self.resolve()

    def test_duplicate_family_defaults_fail(self):
        from generation_defaults import read_defaults
        root, manifest = read_defaults()
        manifest["fallback_vaes"].append(manifest["fallback_vaes"][0])
        self.index = {"_class_name": "StableDiffusionXLPipeline"}
        with patch("vae_defaults.read_defaults", return_value=(root, manifest)):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.resolve()

    def test_native_video_resource_does_not_break_supported_rgb_fallbacks(self):
        from generation_defaults import read_defaults
        from vae_defaults import select_fallback_vae
        root, manifest = read_defaults()
        manifest["fallback_vaes"] = [item for item in manifest["fallback_vaes"] if "wan" not in item["families"]]
        manifest["fallback_vaes"].append({"families": ["wan"], "class_name": "AutoencoderKLWan"})
        with patch("vae_defaults.read_defaults", return_value=(root, manifest)):
            self.assertEqual(self.resolve()[1], "fallback")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                select_fallback_vae("wan")

    def test_missing_fallback_fails_with_an_actionable_error(self):
        self.args.generation_resources = self.root
        (self.root / "generation-defaults.json").write_text('{"version":1}')
        with self.assertRaisesRegex(ValueError, "fallback_vae"):
            self.resolve()

    def test_explicit_vae_directory_precedes_embedded_and_fallback(self):
        bundled, _ = self.resolve()
        self.args.vae = bundled.directory
        self.args.generation_resources = self.root / "absent"
        selected, status = self.resolve()
        self.assertEqual(status, "explicit")
        self.assertEqual(selected.weight, bundled.weight)

    def test_moss_embedded_vae_uses_its_weights_with_verified_flux2_configuration(self):
        self.index = {"_class_name": "MOSSImagePipeline", "vae": ["diffusers", "AutoencoderKLFlux2"]}
        embedded = self.root / "embedded.safetensors"
        self.synthetic_weights(embedded, 'encoder.weight')
        self.args.vae = str(embedded)
        self.args.cache_dir = self.root / "cache"
        self.args.cache_dir.mkdir()
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}), \
                patch('vae_defaults.stage_moss_vae_weights', side_effect=lambda source, target, reference: (shutil.copyfile(source, target), False)[1]):
            selected, status = self.resolve()
            verify_vae_selection(selected)
        self.assertEqual(status, "explicit")
        self.assertEqual(Path(selected.weight.path).read_bytes(), embedded.read_bytes())
        self.assertEqual(selected.class_name, "AutoencoderKLFlux2")
        self.assertEqual(json.loads(Path(selected.config_path).read_text())["latent_channels"], 32)
        self.index = {"_class_name": "Flux2Pipeline", "vae": ["diffusers", "AutoencoderKLFlux2"]}
        selected, status = self.resolve()
        self.assertEqual((status, selected.source_kind), ("explicit", "file"))
        self.assertEqual(Path(selected.weight.path), embedded)
        self.assertEqual(json.loads(Path(selected.config_path).read_text())["latent_channels"], 32)

    def test_moss_vae_legacy_names_are_converted_without_changing_tensor_bytes(self):
        from moss_vae_conversion import _diffusers_name, stage_moss_vae_weights
        for original, expected in {
            'encoder.norm_out.weight': 'encoder.conv_norm_out.weight',
            'encoder.down.2.block.1.nin_shortcut.weight': 'encoder.down_blocks.2.resnets.1.conv_shortcut.weight',
            'encoder.down.2.downsample.conv.weight': 'encoder.down_blocks.2.downsamplers.0.conv.weight',
            'decoder.up.3.block.1.conv1.weight': 'decoder.up_blocks.0.resnets.1.conv1.weight',
            'decoder.up.1.upsample.conv.bias': 'decoder.up_blocks.2.upsamplers.0.conv.bias',
            'decoder.mid.block_2.conv1.weight': 'decoder.mid_block.resnets.1.conv1.weight',
            'decoder.mid.attn_1.proj_out.weight': 'decoder.mid_block.attentions.0.to_out.0.weight',
            'decoder.post_quant_conv.weight': 'post_quant_conv.weight',
        }.items():
            self.assertEqual(_diffusers_name(original), expected)
        def fixture(path, descriptors, data):
            header = json.dumps(descriptors, separators=(',', ':')).encode()
            header += b' ' * (-len(header) % 8)
            path.write_bytes(struct.pack('<Q', len(header)) + header + data)
        legacy = self.root / 'legacy.safetensors'
        reference = self.root / 'reference.safetensors'
        output = self.root / 'staged.safetensors'
        legacy_header = {
            'encoder.mid.attn_1.q.weight': {'dtype': 'F32', 'shape': [2, 2, 1, 1], 'data_offsets': [0, 16]},
            'encoder.quant_conv.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [16, 20]},
        }
        reference_header = {
            'encoder.mid_block.attentions.0.to_q.weight': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]},
            'quant_conv.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [16, 20]},
        }
        payload = bytes(range(20))
        fixture(legacy, legacy_header, payload); fixture(reference, reference_header, payload)
        self.assertTrue(stage_moss_vae_weights(legacy, output, reference))
        raw = output.read_bytes(); size, = struct.unpack('<Q', raw[:8]); header = json.loads(raw[8:8+size])
        self.assertEqual(header, reference_header); self.assertEqual(raw[8+size:], payload)
        self.assertEqual(legacy.read_bytes()[-20:], payload)
        already_normal = self.root / 'already-normal.safetensors'
        from weight_files import file_signature
        source_signature = file_signature(reference)
        self.assertFalse(stage_moss_vae_weights(reference, already_normal, reference))
        self.assertEqual(already_normal.read_bytes(), reference.read_bytes())
        self.assertEqual(file_signature(reference), source_signature)
        legacy_header['encoder.mid.attn_1.q.weight']['shape'] = [2, 2, 2, 1]
        fixture(legacy, legacy_header, payload)
        with self.assertRaisesRegex(ValueError, 'VAE'):
            stage_moss_vae_weights(legacy, self.root / 'bad.safetensors', reference)

    def test_wrong_vae_class_or_latent_contract_is_rejected(self):
        vae = self.root / "override"
        vae.mkdir()
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
        self.args.vae = str(vae)
        for config in ({"_class_name": "AutoencoderKL", "latent_channels": 4},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 32},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 16, "input_channels": 4},
                       {"_class_name": "AutoencoderKLQwenImage", "z_dim": 16, "out_channels": 4}):
            (vae / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "VAE"):
                self.resolve()

    def test_loader_passes_the_component_before_pipeline_construction(self):
        selected, _ = self.resolve()
        vae = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        cls = SimpleNamespace(from_pretrained=Mock(return_value=(vae, {})))
        runtime = SimpleNamespace(AutoencoderKLQwenImage=cls)
        self.assertIs(load_selected_vae(selected, runtime, "float32"), vae)
        self.assertTrue(cls.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(cls.from_pretrained.call_args.kwargs["use_safetensors"])
        self.assertTrue(cls.from_pretrained.call_args.kwargs["output_loading_info"])

    def test_loader_rejects_incomplete_or_unexpected_vae_weights(self):
        selected, _ = self.resolve()
        vae = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
        cls = SimpleNamespace(from_pretrained=Mock())
        runtime = SimpleNamespace(AutoencoderKLQwenImage=cls)
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            with self.subTest(key=key):
                cls.from_pretrained.return_value = (vae, {key: ['fixture']})
                with self.assertRaisesRegex(RuntimeError, 'VAE.*weights'):
                    load_selected_vae(selected, runtime, 'float32')

    def test_metadata_mode_detects_same_size_config_change(self):
        vae = self.root / 'explicit-vae'
        vae.mkdir()
        config = vae / 'config.json'
        config.write_text('{"_class_name":"AutoencoderKLQwenImage","z_dim":16,"marker":1}')
        (vae / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        self.args.vae = str(vae)
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            selected, _ = self.resolve()
            self.assertIsNotNone(selected.config_sha256)
            config.write_text(config.read_text().replace('"marker":1', '"marker":2'))
            with self.assertRaisesRegex(RuntimeError, 'VAE.*changed'):
                verify_vae_selection(selected)

    def test_metadata_mode_still_checks_fallback_config_manifest(self):
        from generation_defaults import read_defaults
        root, original = read_defaults()
        manifest = json.loads(json.dumps(original))
        manifest['fallback_vae']['config']['sha256'] = '0' * 64
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}), \
             patch('vae_defaults.read_defaults', return_value=(root, manifest)):
            with self.assertRaisesRegex(ValueError, 'VAE configuration'):
                self.resolve()

    def test_changed_vae_config_invalidates_selection(self):
        import shutil
        selected, _ = self.resolve()
        destination = self.root / "copy"
        shutil.copytree(selected.directory, destination)
        self.args.vae = str(destination)
        selected, _ = self.resolve()
        config = destination / "config.json"
        config.write_text(config.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "VAE changed"):
            verify_vae_selection(selected)

    def test_config_only_model_uses_fallback_and_partial_weights_do_not(self):
        directory = self.root / "vae"
        directory.mkdir()
        (directory / "config.json").write_text('{"_class_name":"AutoencoderKLQwenImage"}')
        self.assertEqual(self.resolve()[1], "fallback")
        (directory / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
        self.assertEqual(self.resolve(), (None, "model"))

    def test_single_file_missing_vae_uses_the_same_fallback(self):
        model = self.root / "checkpoint.safetensors"
        header = json.dumps({"transformer.img_in.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        model.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.args.source_kind, self.args.model = "single-file", str(model)
        self.assertEqual(self.resolve()[1], "fallback")

    def test_manifest_config_path_and_hash_are_enforced(self):
        from generation_defaults import read_defaults
        root, original = read_defaults()
        for field, value in (("file", str(root / original["fallback_vae"]["config"]["file"])),
                             ("file", "../config.json"), ("sha256", "0" * 64)):
            manifest = json.loads(json.dumps(original))
            manifest["fallback_vae"]["config"][field] = value
            with patch("vae_defaults.read_defaults", return_value=(root, manifest)):
                with self.assertRaisesRegex(ValueError, "VAE configuration"):
                    self.resolve()


if __name__ == "__main__":
    unittest.main()
