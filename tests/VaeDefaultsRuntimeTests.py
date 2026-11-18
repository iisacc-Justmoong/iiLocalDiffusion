"""Strict local VAE loading with real, small Diffusers components."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
from vae_defaults import _select_directory, load_selected_vae, resolve_vae_selection, select_fallback_vae

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors'))


def legacy_vae_name(name):
    name = name.replace('.conv_norm_out.', '.norm_out.').replace('.conv_shortcut.', '.nin_shortcut.')
    name = re.sub(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.', r'encoder.down.\1.block.\2.', name)
    name = re.sub(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.', r'encoder.down.\1.downsample.conv.', name)
    name = re.sub(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.',
                  lambda m: f'decoder.up.{3-int(m[1])}.block.{m[2]}.', name)
    name = re.sub(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.',
                  lambda m: f'decoder.up.{3-int(m[1])}.upsample.conv.', name)
    name = re.sub(r'(encoder|decoder)\.mid_block\.resnets\.(\d+)\.',
                  lambda m: f'{m[1]}.mid.block_{int(m[2])+1}.', name)
    for old, new in (('group_norm', 'norm'), ('to_q', 'q'), ('to_k', 'k'),
                     ('to_v', 'v'), ('to_out.0', 'proj_out')):
        name = name.replace(f'.mid_block.attentions.0.{old}.', f'.mid.attn_1.{new}.')
    return name


@unittest.skipUnless(AVAILABLE, 'Install the reference runtime for numerical VAE loader tests')
class VaeDefaultsRuntimeTests(unittest.TestCase):
    def test_quantized_pytorch_vae_loads_reconstructed_weights(self):
        import diffusers as d
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(37)
        original = d.AutoencoderKL(latent_channels=16, norm_num_groups=4,
            block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
            scaling_factor=.3611, shift_factor=.1159).eval()
        state = dict(original.state_dict())
        name = 'encoder.conv_in.weight'
        state[name] = torch.quantize_per_channel(state[name],
            scales=torch.full((state[name].shape[0],), .01),
            zero_points=torch.zeros(state[name].shape[0], dtype=torch.int64), axis=0, dtype=torch.qint8)
        bias = 'decoder.conv_out.bias'
        state[bias] = torch.quantize_per_tensor(state[bias], scale=.01, zero_point=0, dtype=torch.qint8)
        with resource_fixture.TemporaryDirectory(prefix='quantized-vae-runtime-') as temporary:
            root = Path(temporary);pipeline = root / 'pipeline';(pipeline / 'vae').mkdir(parents=True)
            original.save_config(pipeline / 'vae')
            source = root / 'vae.pt';torch.save(state, source)
            before = source.read_bytes()
            request = SimpleNamespace(vae=str(source), pipeline_class=None, model=str(pipeline),
                                      source_kind='directory', generation_resources=None)
            declaration = {'_class_name': 'FluxPipeline', 'vae': ['diffusers', 'AutoencoderKL']}
            selected, _ = resolve_vae_selection(request, declaration, pipeline)
            loaded = load_selected_vae(selected, d, torch.float32, cache_dir=root / 'cache')
            for key, expected in state.items():
                torch.testing.assert_close(loaded.state_dict()[key],
                    expected.dequantize() if expected.is_quantized else expected, rtol=0, atol=0, msg=key)
            self.assertEqual(loaded._iild_vae_conversion['quantized_tensors']['count'], 2)
            self.assertEqual(source.read_bytes(), before)

    def test_pytorch_vae_files_and_indexes_load_with_weights_only(self):
        import diffusers as d
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(31)
        original = d.AutoencoderKL(latent_channels=16, norm_num_groups=4,
            block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
            scaling_factor=.3611, shift_factor=.1159).eval()
        with resource_fixture.TemporaryDirectory(prefix='pytorch-vae-runtime-') as temporary:
            root = Path(temporary)
            pipeline = root / 'pipeline';(pipeline / 'vae').mkdir(parents=True)
            original.save_config(pipeline / 'vae')
            state = original.state_dict()
            single_dir = root / 'single';single_dir.mkdir()
            single = single_dir / 'vae.safetensors'
            torch.save({'state_dict': state}, single)
            split = root / 'split';split.mkdir()
            names = list(state); halves = [names[::2], names[1::2]]
            shards = [split / 'a.bin', split / 'b.safetensors']
            for shard, keys in zip(shards, halves):
                torch.save({key: state[key] for key in keys}, shard)
            index = split / 'model.index.json'
            index.write_text(json.dumps({'weight_map': {key: shard.name for shard, keys in zip(shards, halves) for key in keys}}))
            directory = root / 'directory';directory.mkdir()
            original.save_config(directory)
            direct = directory / 'diffusion_pytorch_model.fp16.bin';torch.save(state, direct)
            sources = [single, index, *shards, direct]
            before = {path: path.read_bytes() for path in sources}
            declaration = {'_class_name': 'FluxPipeline', 'vae': ['diffusers', 'AutoencoderKL']}
            for selected_file in (single, index, directory):
                with self.subTest(source=selected_file.name):
                    request = SimpleNamespace(vae=str(selected_file), pipeline_class=None, model=str(pipeline),
                        source_kind='directory', generation_resources=None)
                    with patch('checkpoint_conversion.materialize_safetensors') as convert:
                        selected, status = resolve_vae_selection(request, declaration, pipeline)
                        self.assertEqual(status, 'explicit')
                        convert.assert_not_called()
                    with patch.object(torch, 'load', wraps=torch.load) as load:
                        loaded = load_selected_vae(selected, d, torch.float32, cache_dir=root / 'cache')
                        self.assertGreater(load.call_count, 0)
                        self.assertTrue(all(call.kwargs.get('weights_only') is True for call in load.call_args_list))
                    self.assertEqual(set(loaded.state_dict()), set(state))
                    for key, value in state.items():
                        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0, msg=key)
            self.assertEqual({path: path.read_bytes() for path in sources}, before)

    def test_legacy_single_and_sharded_vae_load_exact_synthetic_weights(self):
        import diffusers as d
        import torch
        from safetensors.torch import save_file

        torch.set_num_threads(2)
        torch.manual_seed(23)
        original = d.AutoencoderKL(latent_channels=16, norm_num_groups=4,
            block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
            scaling_factor=.3611, shift_factor=.1159).eval()

        with resource_fixture.TemporaryDirectory(prefix='legacy-vae-runtime-') as temporary:
            root = Path(temporary)
            pipeline = root / 'pipeline';(pipeline / 'vae').mkdir(parents=True)
            original.save_config(pipeline / 'vae')
            declaration = {'_class_name': 'FluxPipeline', 'vae': ['diffusers', 'AutoencoderKL']}
            for label, prefix, old_names, sharded in (
                    ('unprefixed', '', True, False), ('ldm-prefix', 'first_stage_model.', True, False),
                    ('sharded-prefix', 'vae.', True, True), ('diffusers-prefix', 'vae.', False, False)):
                with self.subTest(layout=label):
                    folder = root / label;folder.mkdir()
                    state = {}
                    for name, tensor in original.state_dict().items():
                        key = prefix + (legacy_vae_name(name) if old_names else name)
                        value = tensor.detach().contiguous()
                        if old_names and '.mid.attn_1.' in key and value.ndim == 2 and key.endswith('.weight'):
                            value = value[:, :, None, None].contiguous()
                        self.assertNotIn(key, state)
                        state[key] = value
                    if sharded:
                        parts = [dict(), dict()]
                        for offset, (key, tensor) in enumerate(state.items()):
                            parts[offset % 2][key] = tensor
                        names = ['nested/part-a.safetensors', 'part-b.safetensors']
                        for name, part in zip(names, parts):
                            (folder / name).parent.mkdir(parents=True, exist_ok=True)
                            save_file(part, str(folder / name), metadata={'format': 'pt'})
                        selected = folder / 'vae.safetensors.index.json'
                        selected.write_text(json.dumps({'weight_map': {key: name for name, part in zip(names, parts) for key in part}}))
                        sources = [selected, *(folder / name for name in names)]
                    else:
                        selected = folder / 'vae.safetensors'
                        save_file(state, str(selected), metadata={'format': 'pt'})
                        sources = [selected]
                    before = {path: path.read_bytes() for path in sources}
                    request = SimpleNamespace(vae=str(selected), pipeline_class=None, model=str(pipeline),
                        source_kind='directory', generation_resources=None)
                    selection, _ = resolve_vae_selection(request, declaration, pipeline)
                    loaded = load_selected_vae(selection, d, torch.float32, cache_dir=root / 'cache')
                    self.assertEqual(set(loaded.state_dict()), set(original.state_dict()))
                    for name, tensor in original.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
                    self.assertEqual({path: path.read_bytes() for path in sources}, before)
            bad = root / 'bad';bad.mkdir()
            state = {legacy_vae_name(name): tensor.detach().contiguous() for name, tensor in original.state_dict().items()}
            missing = dict(state);missing.pop(next(iter(missing)))
            extra = dict(state);extra['unexpected.weight'] = torch.zeros(1)
            wrong_shape = dict(state);wrong_shape['encoder.conv_in.weight'] = torch.zeros(1)
            mixed = dict(state);key = next(iter(mixed));mixed['vae.' + key] = mixed.pop(key)
            for label, invalid in (('missing', missing), ('extra', extra),
                                   ('wrong-shape', wrong_shape), ('mixed', mixed)):
                with self.subTest(invalid=label):
                    selected = bad / f'{label}.safetensors'
                    save_file(invalid, str(selected), metadata={'format': 'pt'})
                    request.vae = str(selected)
                    selection, _ = resolve_vae_selection(request, declaration, pipeline)
                    with self.assertRaisesRegex((RuntimeError, ValueError), 'VAE'):
                        load_selected_vae(selection, d, torch.float32, cache_dir=root / 'cache')

    def test_separate_file_and_sharded_vae_load_exact_synthetic_weights(self):
        import diffusers as d
        import torch
        from safetensors.torch import save_file

        torch.set_num_threads(2)
        torch.manual_seed(13)
        cases = (
            ('FluxPipeline', d.AutoencoderKL, dict(latent_channels=16, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
                scaling_factor=.3611, shift_factor=.1159)),
            ('Flux2Pipeline', d.AutoencoderKLFlux2, dict(latent_channels=32, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), layers_per_block=1)),
            ('QwenImagePipeline', d.AutoencoderKLQwenImage, dict(base_dim=8, z_dim=16,
                dim_mult=[1, 2, 2, 2], num_res_blocks=1)),
        )
        with resource_fixture.TemporaryDirectory(prefix='separate-vae-runtime-') as temporary:
            root = Path(temporary)
            for name, cls, options in cases:
                with self.subTest(pipeline=name):
                    original = cls(**options).eval()
                    model_folder = root / name / 'pipeline';(model_folder / 'vae').mkdir(parents=True)
                    original.save_config(model_folder / 'vae')
                    external = root / name / 'external';external.mkdir()
                    flat = external / 'standalone.weights'
                    save_file({key: value.contiguous() for key, value in original.state_dict().items()},
                              str(flat), metadata={'format': 'pt'})
                    prefixed = external / 'prefixed.safetensors'
                    save_file({'vae.' + key: value.contiguous() for key, value in original.state_dict().items()},
                              str(prefixed), metadata={'format': 'pt'})
                    source = root / name / 'saved-shards'
                    original.save_pretrained(source, safe_serialization=True, max_shard_size='32KB')
                    index = source / 'diffusion_pytorch_model.safetensors.index.json'
                    self.assertTrue(index.is_file())
                    (source / 'config.json').unlink()
                    for selected_file in (flat, prefixed, index):
                        request = SimpleNamespace(vae=str(selected_file), pipeline_class=None, model=str(model_folder),
                            source_kind='directory', generation_resources=None)
                        declaration = {'_class_name': name, 'vae': ['diffusers', cls.__name__]}
                        selection, status = resolve_vae_selection(request, declaration, model_folder)
                        self.assertEqual((selection.source_kind, status), ('file', 'explicit'))
                        loaded = load_selected_vae(selection, d, torch.float32, cache_dir=root / 'cache')
                        self.assertEqual(set(loaded.state_dict()), set(original.state_dict()))
                        for key, value in original.state_dict().items():
                            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0, msg=key)
                    incomplete = external / 'incomplete.safetensors'
                    state = dict(original.state_dict());state.pop(next(iter(state)))
                    save_file({key: value.contiguous() for key, value in state.items()}, str(incomplete),
                              metadata={'format': 'pt'})
                    request.vae = str(incomplete)
                    selection, _ = resolve_vae_selection(request, declaration, model_folder)
                    with self.assertRaisesRegex(RuntimeError, 'VAE weights'):
                        load_selected_vae(selection, d, torch.float32, cache_dir=root / 'cache')

    @unittest.skipUnless(os.environ.get('IILD_VERIFY_BUNDLED_VAE_PAYLOAD') == '1',
                         'Set IILD_VERIFY_BUNDLED_VAE_PAYLOAD=1 for full resource hashes')
    def test_existing_local_vae_payloads_match_the_resource_manifest(self):
        from generation_defaults import read_defaults
        from weight_files import file_sha256, file_signature

        root, manifest = read_defaults(str(ROOT / 'resources'))
        entries = [manifest['fallback_vae'], *manifest.get('fallback_vaes', [])]
        for family in ('qwen-image', 'flux1', 'flux2'):
            with self.subTest(family=family):
                item = next(entry for entry in entries if family in entry['families'])
                path = root / item['file']
                if not path.is_file():
                    self.skipTest(f'Provision the local {family} VAE resource for its payload check')
                before = file_signature(path)
                self.assertEqual(path.stat().st_size, item['size'])
                self.assertEqual(file_sha256(path), item['sha256'])
                self.assertEqual(file_signature(path), before)

    def test_existing_local_vae_headers_match_real_diffusers_architectures(self):
        import diffusers as d
        import torch
        from safetensors import safe_open

        checked = 0
        with patch.dict('os.environ', {'IILD_MODEL_VALIDATION': 'metadata'}):
            for family, cls, relative in (
                    ('qwen-image', d.AutoencoderKLQwenImage, 'vae/qwen-image/vae/diffusion_pytorch_model.safetensors'),
                    ('flux1', d.AutoencoderKL, 'vae/flux1/diffusion_pytorch_model.safetensors'),
                    ('flux2', d.AutoencoderKLFlux2, 'vae/flux2/vae/diffusion_pytorch_model.safetensors')):
                with self.subTest(family=family):
                    if not (ROOT / 'resources' / relative).is_file():
                        self.skipTest(f'Provision the local {family} VAE resource for its header check')
                    selection = select_fallback_vae(family, str(ROOT / 'resources'))
                    config = cls.load_config(selection.directory, local_files_only=True)
                    with torch.device('meta'):
                        model = cls.from_config(config)
                    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
                    with safe_open(selection.weight.path, framework='pt', device='cpu') as source:
                        actual = {key: tuple(source.get_slice(key).get_shape()) for key in source.keys()}
                    self.assertEqual(set(actual), set(expected))
                    for key, shape in expected.items():
                        self.assertEqual(actual[key], shape, key)
                    checked += 1
        if not checked:
            self.skipTest('Provision the local fallback VAE resources for header checks')

    def test_local_flux_qwen_and_flux2_vaes_require_complete_weights(self):
        import diffusers as d
        import torch
        from safetensors.torch import load_file, save_file

        torch.set_num_threads(2)
        torch.manual_seed(7)
        cases = (
            ('flux1', d.AutoencoderKL, dict(latent_channels=16, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
                scaling_factor=.3611, shift_factor=.1159)),
            ('flux2', d.AutoencoderKLFlux2, dict(latent_channels=32, norm_num_groups=4,
                block_out_channels=(8, 8, 8, 8), layers_per_block=1)),
            ('qwen-image', d.AutoencoderKLQwenImage, dict(base_dim=8, z_dim=16,
                dim_mult=[1, 2, 2, 2], num_res_blocks=1)),
        )
        with resource_fixture.TemporaryDirectory(prefix='strict-vae-runtime-') as directory:
            for family, cls, kwargs in cases:
                with self.subTest(family=family):
                    folder = Path(directory) / family
                    original = cls(**kwargs).eval()
                    original.save_pretrained(folder, safe_serialization=True)
                    selection = _select_directory(folder, cls.__name__, family)
                    loaded = load_selected_vae(selection, d, torch.float32)
                    for key, expected in original.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)
                    weights = folder / 'diffusion_pytorch_model.safetensors'
                    state = load_file(str(weights))
                    missing = next(key for key in state if key.endswith('.weight'))
                    del state[missing]
                    save_file(state, str(weights), metadata={'format': 'pt'})
                    selection = _select_directory(folder, cls.__name__, family)
                    with self.assertRaisesRegex(RuntimeError, 'VAE weights'):
                        load_selected_vae(selection, d, torch.float32)


if __name__ == '__main__':
    unittest.main()
