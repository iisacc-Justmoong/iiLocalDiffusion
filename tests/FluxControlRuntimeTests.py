"""Tiny real FLUX.1 ControlNet component loading and generation."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import generic_controlnet as control

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'transformers', 'safetensors', 'PIL'))


@unittest.skipUnless(AVAILABLE, 'Install reference runtime for numerical FLUX tests')
class FluxControlRuntimeTests(unittest.TestCase):
    def test_provisioned_instantx_canny_loads_actual_controlnet_weights(self):
        if os.environ.get('IILD_VERIFY_FLUX1_CANNY_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_FLUX1_CANNY_LOAD=1 to load the real FLUX Canny adapter')
        import json
        import torch
        import diffusers as d
        torch.set_num_threads(2)
        folder = ROOT / 'resources/models/flux1-instantx-canny'
        weight = folder / 'diffusion_pytorch_model.safetensors'
        before = weight.stat()
        adapter, info = d.FluxControlNetModel.from_pretrained(
            folder, local_files_only=True, use_safetensors=True, output_loading_info=True,
            low_cpu_mem_usage=True, dtype=torch.bfloat16)
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            self.assertFalse(info.get(key), (key, info.get(key)))
        self.assertEqual(len(adapter.state_dict()), 188)
        self.assertTrue(all(not p.is_meta and bool(torch.isfinite(p).all()) for p in adapter.parameters()))
        config = json.loads((ROOT / 'resources/model-configs/flux1-schnell/transformer/config.json').read_text())
        with torch.device('meta'):
            base = d.FluxTransformer2DModel.from_config(config)
        selection = control.Selection(str(folder), 'FluxPipeline', 'FluxControlNetPipeline',
                                      'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
        control.validate_component(adapter, SimpleNamespace(transformer=base), selection,
                                   SimpleNamespace(control_mode=None))
        self.assertTrue(adapter.config.guidance_embeds)
        self.assertFalse(base.config.guidance_embeds)
        after = weight.stat()
        self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_provisioned_instantx_canny_header_and_sample_values(self):
        mode = os.environ.get('IILD_VERIFY_FLUX1_CANNY_PAYLOAD')
        if mode not in ('metadata', '1'):
            self.skipTest('Set IILD_VERIFY_FLUX1_CANNY_PAYLOAD=metadata or 1 to audit the actual adapter')
        import hashlib
        import json
        import torch
        from safetensors import safe_open
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
        files = manifest['flux1_instantx_canny_resource']['files']
        item = next(entry for entry in files if entry['source_file'] == 'diffusion_pytorch_model.safetensors')
        fixture = json.loads((ROOT / 'resources/test-fixtures/flux1-instantx-canny-header.json').read_text())['header']
        path = ROOT / 'resources' / item['file']
        before = path.stat()
        self.assertEqual(before.st_size, item['size'])
        if mode == '1':
            with path.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), item['sha256'])
        expected = {name: value for name, value in fixture.items() if name != '__metadata__'}
        with safe_open(str(path), framework='pt', device='cpu') as source:
            self.assertEqual(set(source.keys()), set(expected))
            self.assertEqual(source.metadata(), fixture.get('__metadata__'))
            for name, value in expected.items():
                view = source.get_slice(name)
                self.assertEqual(view.get_shape(), value['shape'], name)
                self.assertEqual(view.get_dtype(), value['dtype'], name)
            for name in ('context_embedder.weight', 'controlnet_blocks.0.weight', 'x_embedder.weight'):
                self.assertTrue(bool(torch.isfinite(source.get_slice(name)[:1].float()).all()), name)
        after = path.stat()
        self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_published_instantx_canny_header_matches_flux_control_contract(self):
        import json
        import torch
        import diffusers as d
        fixture = json.loads((ROOT / 'resources/test-fixtures/flux1-instantx-canny-header.json').read_text())
        with torch.device('meta'):
            adapter = d.FluxControlNetModel.from_config(fixture['config'])
            base_config = json.loads((ROOT / 'resources/model-configs/flux1-schnell/transformer/config.json').read_text())
            base = d.FluxTransformer2DModel.from_config(base_config)
        expected = {name: tuple(value.shape) for name, value in adapter.state_dict().items()}
        actual = {name: tuple(value['shape']) for name, value in fixture['header'].items() if name != '__metadata__'}
        self.assertEqual(actual, expected)
        self.assertTrue(adapter.config.guidance_embeds)
        self.assertFalse(base.config.guidance_embeds)
        selection = control.Selection('unused', 'FluxPipeline', 'FluxControlNetPipeline',
                                      'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
        control.validate_component(adapter, SimpleNamespace(transformer=base), selection,
                                   SimpleNamespace(control_mode=None))

    def test_schnell_with_guidance_distilled_controlnet_keeps_guidance_separate(self):
        """The public Dev Canny needs guidance even when Schnell's base does not."""
        import torch
        import diffusers as d
        import resource_fixture
        from native_canny_fixture import hints
        torch.set_num_threads(2)
        torch.manual_seed(512)
        transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6)).eval()
        net = d.FluxControlNetModel(in_channels=64, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
            pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(4, 6, 6)).eval()
        with torch.no_grad():
            for module in (net.controlnet_x_embedder, *net.controlnet_blocks, *net.controlnet_single_blocks):
                torch.nn.init.constant_(module.weight, .01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(8, 8, 8, 8),
            norm_num_groups=4, layers_per_block=1, latent_channels=16, sample_size=64,
            scaling_factor=.3611, shift_factor=.1159).eval()
        pipeline = d.FluxPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(), vae=vae,
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None)
        with resource_fixture.TemporaryDirectory(prefix='flux-distilled-control-') as temporary:
            folder = Path(temporary)
            net.save_pretrained(folder, safe_serialization=True)
            selection = control.Selection(str(folder), 'FluxPipeline', 'FluxControlNetPipeline',
                                          'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
            wrapped = control.attach(pipeline, selection, SimpleNamespace(control_mode=None), d, torch.float32)
            wrapped.set_progress_bar_config(disable=True)
            observed_control, observed_base = [], []
            handles = [wrapped.controlnet.register_forward_pre_hook(
                lambda module, args, kwargs: observed_control.append(kwargs.get('guidance')), with_kwargs=True),
                wrapped.transformer.register_forward_pre_hook(
                lambda module, args, kwargs: observed_base.append(kwargs.get('guidance')), with_kwargs=True)]
            self.addCleanup(lambda: [handle.remove() for handle in handles])
            prompt, pooled = torch.randn(1, 4, 32), torch.randn(1, 16)
            blank, edges = hints()
            def run(image):
                with torch.no_grad():
                    return wrapped(prompt_embeds=prompt, pooled_prompt_embeds=pooled, control_image=image,
                        width=64, height=64, num_inference_steps=2, guidance_scale=3.5,
                        generator=torch.Generator('cpu').manual_seed(17), output_type='latent').images
            first = run(edges)
            self.assertTrue(torch.equal(first, run(edges)))
            self.assertFalse(torch.equal(first, run(blank)))
            self.assertTrue(observed_control and observed_base)
            for value in observed_control:
                torch.testing.assert_close(value, torch.full_like(value, 3.5))
            self.assertTrue(all(value is None for value in observed_base))

    def test_generic_runner_uses_existing_flux_vae_with_preprocessed_hints(self):
        import contextlib
        import io
        import json
        import torch
        import diffusers as d
        from PIL import Image, ImageDraw
        from safetensors.torch import load_file, save_file
        import resource_fixture
        import generate_any
        import native_control_preprocessing
        from depth_hint_fixture import save_model

        bundled_vae = ROOT / 'resources/vae/flux1/diffusion_pytorch_model.safetensors'
        if not bundled_vae.is_file():
            self.skipTest('Provision the existing FLUX.1 VAE resource to test its integration')
        try:
            native = native_control_preprocessing.library_path()
        except ValueError:
            self.skipTest('Build the native Canny preprocessor to test its integration')
        torch.set_num_threads(2)
        torch.manual_seed(515)
        transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6)).eval()
        net = d.FluxControlNetModel.from_transformer(transformer, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2)
        with torch.no_grad():
            for module in (net.controlnet_x_embedder, *net.controlnet_blocks, *net.controlnet_single_blocks):
                torch.nn.init.constant_(module.weight, .01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(8, 8, 8, 8),
            norm_num_groups=4, layers_per_block=1, latent_channels=16, sample_size=64,
            scaling_factor=.3611, shift_factor=.1159).eval()
        pipeline = d.FluxPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(), vae=vae,
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None)
        with resource_fixture.TemporaryDirectory(prefix='flux1-fallback-control-') as temporary:
            root = Path(temporary)
            package = root / 'model'
            pipeline.save_pretrained(package, safe_serialization=True)
            (package / 'vae').rename(root / 'unused-tiny-vae')
            adapter = root / 'control'
            net.save_pretrained(adapter, safe_serialization=True)
            embed_file = root / 'prompt.safetensors'
            save_file({'prompt': torch.randn(1, 4, 32), 'pooled': torch.randn(1, 16)}, str(embed_file))
            def descriptor(key, layout):
                return {'tensor_path': str(embed_file), 'key': key, 'semantic': 'embeddings',
                        'layout': layout, 'representation_space': 'test-flux1-text'}
            inputs = {'prompt_embeds': descriptor('prompt', 'BSC'),
                      'pooled_prompt_embeds': descriptor('pooled', 'BC'), 'output_type': 'latent'}
            blank = Image.new('RGB', (64, 64), 'black')
            subject = blank.copy()
            ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
            blank_path, subject_path = root / 'blank.png', root / 'subject.png'
            blank.save(blank_path)
            subject.save(subject_path)
            depth_model = root / 'depth-model'
            save_model(depth_model)
            def run(name, image, preprocessing, prompt=None, encoder_options=(), device='cpu'):
                output = root / name
                selected_inputs = (inputs if prompt is None else
                                   {'prompt': prompt, 'max_sequence_length': 32, 'output_type': 'latent'})
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(generate_any.main(['--model', str(package), *encoder_options, '--controlnet', str(adapter),
                        '--control-image', str(image), *preprocessing, '--device', device, '--dtype', 'float32',
                        '--seed', '17', '--width', '64', '--height', '64', '--steps', '2', '--guidance-scale', '1',
                        '--pipeline-inputs', json.dumps(selected_inputs), '--tensor-outputs', json.dumps({'images': {
                            'semantic': 'latents', 'layout': 'BSC', 'representation_space': 'test-flux1-packed'}}),
                        '--no-default-modifiers', '--cache-dir', str(root / 'cache'), '--output-dir', str(output)]), 0)
                return load_file(str(output / 'images.safetensors'))['images'], json.loads((output / 'generation.json').read_text())
            for mode, options in (('canny', ('--control-preprocess', 'canny', '--control-preprocessor-library', str(native))),
                                  ('depth', ('--control-preprocess', 'depth', '--control-depth-model', str(depth_model)))):
                with self.subTest(preprocessor=mode):
                    empty, _ = run(mode + '-blank', blank_path, options)
                    controlled, report = run(mode + '-subject', subject_path, options)
                    repeated, _ = run(mode + '-repeat', subject_path, options)
                    self.assertEqual(report['request']['vae']['status'], 'fallback')
                    self.assertEqual(Path(report['request']['vae']['selection']['weight']['path']), bundled_vae)
                    self.assertFalse(torch.equal(empty, controlled))
                    self.assertTrue(torch.equal(controlled, repeated))
            # Exercise real text encoding with the existing ComfyUI tokenizer data;
            # the small CLIP and T5 model weights remain deliberately synthetic.
            from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast
            if not (ROOT / 'resources/dependencies/ComfyUI/comfy/sd1_tokenizer').is_dir() or not (
                    ROOT / 'resources/dependencies/ComfyUI/comfy/text_encoders/t5_tokenizer').is_dir():
                self.skipTest('Provision the existing ComfyUI tokenizers to test text encoding')
            tokenizer = CLIPTokenizer.from_pretrained(str(ROOT / 'resources/dependencies/ComfyUI/comfy/sd1_tokenizer'),
                                                       local_files_only=True)
            tokenizer_2 = T5TokenizerFast.from_pretrained(str(ROOT / 'resources/dependencies/ComfyUI/comfy/text_encoders/t5_tokenizer'),
                                                          local_files_only=True)
            clip = CLIPTextModel(CLIPTextConfig(vocab_size=49408, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77)).eval()
            t5 = T5EncoderModel(T5Config(vocab_size=32128, d_model=32, d_kv=16, d_ff=64,
                num_layers=1, num_heads=2, dropout_rate=0.)).eval()
            text_pipeline = d.FluxPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(), vae=vae,
                text_encoder=clip, tokenizer=tokenizer, text_encoder_2=t5, tokenizer_2=tokenizer_2)
            package = root / 'model-with-text'
            text_pipeline.save_pretrained(package, safe_serialization=True)
            (package / 'vae').rename(root / 'unused-text-tiny-vae')
            options = ('--control-preprocess', 'canny', '--control-preprocessor-library', str(native))
            fox, report = run('text-fox', subject_path, options, 'a red fox')
            repeat, _ = run('text-fox-repeat', subject_path, options, 'a red fox')
            city, _ = run('text-city', subject_path, options, 'a blue city')
            self.assertEqual(report['request']['vae']['status'], 'fallback')
            self.assertEqual(report['model']['flux_clip_token_limit'], {'original': 8192, 'effective': 77})
            self.assertTrue(torch.equal(fox, repeat))
            self.assertFalse(torch.equal(fox, city))
            (package / 'tokenizer').rename(root / 'unused-clip-tokenizer')
            (package / 'tokenizer_2').rename(root / 'unused-t5-tokenizer')
            fallback, fallback_report = run('text-fallback', subject_path, options, 'a red fox')
            self.assertTrue(torch.equal(fox, fallback))
            self.assertEqual(fallback_report['request']['flux_tokenizers']['tokenizer']['status'], 'fallback')
            self.assertEqual(fallback_report['request']['flux_tokenizers']['tokenizer_2']['status'], 'fallback')
            clip_file, t5_file = root / 'clip.safetensors', root / 't5.safetensors'
            clip_weights = {('text_model.' + key if not key.startswith('text_model.') else key): value.detach().clone()
                            for key, value in clip.state_dict().items()}
            save_file(clip_weights, str(clip_file))
            save_file({key: value.detach().clone() for key, value in t5.state_dict().items()}, str(t5_file))
            (package / 'text_encoder/model.safetensors').rename(root / 'unused-clip-encoder.safetensors')
            (package / 'text_encoder_2/model.safetensors').rename(root / 'unused-t5-encoder.safetensors')
            encoder_options = ('--flux-clip-encoder', str(clip_file), '--flux-t5-encoder', str(t5_file))
            split, split_report = run('text-split-encoders', subject_path, options, 'a red fox', encoder_options)
            self.assertTrue(torch.equal(fox, split))
            self.assertEqual(split_report['request']['flux_text_encoders']['text_encoder']['mode'], 'single-file')
            self.assertEqual(split_report['request']['flux_text_encoders']['text_encoder_2']['mode'], 'single-file')
            import hashlib
            import shutil
            resources = root / 'fallback-resources'
            catalog = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
            for name in ('flux1-clip', 'flux1-t5'):
                shutil.copytree(ROOT / 'resources/text_encoders' / name, resources / 'text_encoders' / name)
            for item in catalog['fallback_flux_text_weights']:
                source_file = clip_file if item['component'] == 'text_encoder' else t5_file
                target_file = resources / item['file']
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target_file)
                item['size'] = target_file.stat().st_size
                item['sha256'] = hashlib.sha256(target_file.read_bytes()).hexdigest()
            (resources / 'generation-defaults.json').write_text(json.dumps(catalog))
            auto_options = ('--generation-resources', str(resources), '--vae', str(bundled_vae.parent))
            automatic, auto_report = run('text-auto-encoders', subject_path, options, 'a red fox', auto_options)
            self.assertTrue(torch.equal(split, automatic))
            self.assertEqual(auto_report['request']['flux_text_encoders']['text_encoder']['status'], 'fallback')
            self.assertEqual(auto_report['request']['flux_text_encoders']['text_encoder_2']['status'], 'fallback')
            # Compare the scaled FP8 path against the exact dequantized dense T5 weights.
            scaled_t5 = {key: value.detach().clone() for key, value in t5.state_dict().items()}
            dense_t5 = dict(scaled_t5)
            for index, key in enumerate([name for name in scaled_t5 if name.endswith(('SelfAttention.q.weight', 'SelfAttention.k.weight'))]):
                scale = torch.tensor(.125 * (index + 1), dtype=torch.float32)
                fp8 = (scaled_t5[key] / scale).to(torch.float8_e4m3fn)
                scaled_t5[key] = fp8
                scaled_t5[key[:-len('.weight')] + '.scale_weight'] = scale
                dense_t5[key] = fp8.float() * scale
            scaled_t5['scaled_fp8'] = torch.empty(0, dtype=torch.float8_e4m3fn)
            scaled_file, dense_file = root / 't5-scaled.safetensors', root / 't5-dequantized.safetensors'
            save_file(scaled_t5, str(scaled_file))
            save_file(dense_t5, str(dense_file))
            scaled_args = ('--flux-clip-encoder', str(clip_file), '--flux-t5-encoder', str(scaled_file))
            dense_args = ('--flux-clip-encoder', str(clip_file), '--flux-t5-encoder', str(dense_file))
            scaled_out, scaled_report = run('text-scaled-fp8', subject_path, options, 'a red fox', scaled_args)
            dense_out, _ = run('text-dequantized', subject_path, options, 'a red fox', dense_args)
            self.assertTrue(torch.equal(scaled_out, dense_out))
            self.assertEqual(scaled_report['model']['flux_text_encoder_loading']['text_encoder_2']['format'],
                             'comfy-scalar-fp8-e4m3fn')
            self.assertEqual(scaled_report['model']['flux_text_encoder_loading']['text_encoder_2']['scaled_layers'], 2)
            streamed_out, stream_report = run('text-streamed-fp8', subject_path, options, 'a red fox',
                                               (*scaled_args, '--flux-t5-fp8-storage', 'cpu-stream'))
            self.assertTrue(torch.equal(scaled_out, streamed_out))
            self.assertEqual(stream_report['model']['flux_text_encoder_loading']['text_encoder_2']['runtime_weights'],
                             'cpu-stream-fp8')
            # Build original-format transformer tensors from the public Schnell
            # shape contract, with dimensions reduced for this numerical test.
            from unittest.mock import patch
            from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES
            from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
            source_shapes = json.loads((ROOT / 'resources/model-configs/flux1-schnell/transformer-source-shapes.json').read_text())
            dimensions = {3072: 32, 4096: 32, 768: 16, 128: 16, 6144: 64,
                          9216: 96, 12288: 128, 15360: 160, 18432: 192, 21504: 224}
            original_transformer = {}
            for key, shape in source_shapes.items():
                if key.startswith(('double_blocks.', 'single_blocks.')) and not key.startswith(
                        ('double_blocks.0.', 'single_blocks.0.')):
                    continue
                shape = tuple(dimensions.get(size, size) for size in shape)
                original_transformer['model.diffusion_model.' + key] = (
                    torch.randn(shape) * .1).to(torch.float8_e4m3fn)

            # The upstream converter hardcodes the production inner width of
            # 3072. Adapt only its split sizes to this test's width of 32;
            # retain the actual converter and all single-file loader code.
            original_split = torch.split
            split_calls = []
            def tiny_split(tensor, sizes, dim=0):
                if sizes == (3072, 3072, 3072, 12288):
                    split_calls.append(True)
                    sizes = (32, 32, 32, 128)
                return original_split(tensor, sizes, dim=dim)
            def tiny_converter(checkpoint, **kwargs):
                with patch.object(torch, 'split', side_effect=tiny_split):
                    return convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs)
            expected_transformer = d.FluxTransformer2DModel.from_config(transformer.config).eval()
            expected_transformer.load_state_dict(tiny_converter(dict(original_transformer)))
            embedded_prefix = 'text_encoders.t5xxl.transformer.'
            embedded_state = {embedded_prefix + key: value.detach().to(torch.float8_e4m3fn).clone()
                              for key, value in t5.state_dict().items() if key != 'encoder.embed_tokens.weight'}
            embedded_state.update({'text_encoders.clip_l.transformer.' + key: value.detach().clone()
                                   for key, value in clip_weights.items()})
            embedded_state.update(original_transformer)
            embedded_file = root / 'embedded-flux.safetensors'
            save_file(embedded_state, str(embedded_file))
            import flux1_transformer_stream
            streamed_model = flux1_transformer_stream.load(embedded_file, package, torch.float32)
            streamed_layer = streamed_model.transformer_blocks[0].attn.to_q
            self.assertEqual(streamed_layer._iild_streamed_weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(streamed_layer._iild_streamed_weight.device.type, 'cpu')
            self.assertEqual(streamed_layer.weight.dtype, torch.float32)
            with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
                streamed_model.state_dict()
            del streamed_model
            original_single_file = d.FluxPipeline.from_single_file
            def reference_single_file(path, config, **kwargs):
                return original_single_file(path, config=config, transformer=expected_transformer, **kwargs)
            def embedded_run(name, image, preprocessing, stream, checkpoint=embedded_file,
                             transformer_stream=False, device='cpu', precision='float32'):
                output = root / name
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(generate_any.main(['--model', str(checkpoint), '--model-config', str(package),
                        *(['--flux-t5-fp8-storage', 'cpu-stream'] if stream else []), '--controlnet', str(adapter),
                        *(['--flux-transformer-storage', 'cpu-stream'] if transformer_stream else []),
                        '--control-image', str(image), *preprocessing, '--device', device, '--dtype', precision,
                        '--seed', '17', '--width', '64', '--height', '64', '--steps', '2', '--guidance-scale', '1',
                        '--pipeline-inputs', json.dumps({'prompt': 'a red fox', 'max_sequence_length': 32,
                                                         'output_type': 'latent'}),
                        '--tensor-outputs', json.dumps({'images': {'semantic': 'latents', 'layout': 'BSC',
                            'representation_space': 'test-flux1-packed'}}), '--no-default-modifiers',
                        '--cache-dir', str(root / 'cache'), '--output-dir', str(output)]), 0)
                return load_file(str(output / 'images.safetensors'))['images'], json.loads((output / 'generation.json').read_text())
            mapping = SINGLE_FILE_LOADABLE_CLASSES['FluxTransformer2DModel']
            with patch.dict(mapping, {'checkpoint_mapping_fn': tiny_converter}):
                for mode, preprocessing in (
                    ('canny', options),
                    ('depth', ('--control-preprocess', 'depth', '--control-depth-model', str(depth_model))),
                ):
                    stream_result, embedded_report = embedded_run('embedded-' + mode, subject_path, preprocessing, True)
                    dense_result, _ = embedded_run('embedded-dense-' + mode, subject_path, preprocessing, False)
                    blank_result, _ = embedded_run('embedded-blank-' + mode, blank_path, preprocessing, True)
                    torch.testing.assert_close(stream_result, dense_result, rtol=0, atol=0)
                    self.assertFalse(torch.equal(stream_result, blank_result))
                    self.assertEqual(embedded_report['request']['flux_text_encoder_status']['text_encoder_2'], 'embedded')
                    self.assertEqual(embedded_report['model']['flux_text_encoder_loading']['text_encoder_2']['format'],
                                     'unscaled-fp8-e4m3fn')
                    streamed_transformer, stream_report = embedded_run('transformer-stream-' + mode,
                        subject_path, preprocessing, True, transformer_stream=True)
                    torch.testing.assert_close(streamed_transformer, stream_result, rtol=0, atol=0)
                    if mode == 'canny':
                        canny_stream_result = stream_result
                    self.assertEqual(stream_report['request']['flux_transformer_storage'], 'cpu-stream')
                    self.assertEqual(stream_report['model']['single_file_loading']['runtime_weights'], 'cpu-stream')
                    self.assertEqual(stream_report['model']['single_file_loading']['streamed_layers'],
                                     sum(isinstance(layer, torch.nn.Linear) for layer in transformer.modules()))
                    with patch.object(d.FluxPipeline, 'from_single_file', side_effect=reference_single_file):
                        reference, _ = embedded_run('embedded-reference-' + mode, subject_path, preprocessing, True)
                    torch.testing.assert_close(stream_result, reference, rtol=0, atol=0)
            self.assertGreaterEqual(len(split_calls), 2 + 2 * 2 * 3)
            with patch.dict(mapping, {'checkpoint_mapping_fn': tiny_converter}):
                bf16_dense, _ = embedded_run('transformer-dense-bf16', subject_path, options, True, precision='bfloat16')
                bf16_stream, _ = embedded_run('transformer-stream-bf16', subject_path, options, True,
                                               transformer_stream=True, precision='bfloat16')
            torch.testing.assert_close(bf16_stream, bf16_dense, rtol=0, atol=0)
            bf16_checkpoint = root / 'embedded-flux-bf16-transformer.safetensors'
            save_file({**embedded_state, **{key: value.to(torch.bfloat16)
                                             for key, value in original_transformer.items()}}, str(bf16_checkpoint))
            bf16_weights, _ = embedded_run('transformer-source-bf16', subject_path, options, True,
                                            bf16_checkpoint, transformer_stream=True)
            torch.testing.assert_close(bf16_weights, canny_stream_result, rtol=0, atol=0)
            if torch.backends.mps.is_available():
                with patch.dict(mapping, {'checkpoint_mapping_fn': tiny_converter}):
                    mps_dense, _ = embedded_run('transformer-dense-mps', subject_path, options, True, device='mps')
                    mps_stream, _ = embedded_run('transformer-stream-mps', subject_path, options, True,
                                                 transformer_stream=True, device='mps')
                torch.testing.assert_close(mps_stream, mps_dense, rtol=0, atol=0)

            # Public full checkpoints also contain the original FLUX VAE.
            # Keep its original key layout and reduce only hidden channels;
            # the generic runner must choose it over the fallback resource.
            from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
            full_header = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())['header']
            vae_dimensions = {128: 8, 256: 16, 512: 32}
            original_vae = {key: torch.randn(tuple(vae_dimensions.get(dimension, dimension)
                                                        for dimension in item['shape'])) * .1
                            for key, item in full_header.items() if key.startswith('vae.')}
            expected_vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(8, 16, 32, 32),
                norm_num_groups=4, layers_per_block=2, latent_channels=16, sample_size=64,
                scaling_factor=.3611, shift_factor=.1159, use_quant_conv=False, use_post_quant_conv=False).eval()
            expected_vae.load_state_dict(convert_ldm_vae_checkpoint(original_vae, expected_vae.config), strict=True)
            expected_vae.save_config(package / 'vae')
            full_file = root / 'embedded-flux-with-vae.safetensors'
            save_file({**embedded_state, **original_vae}, str(full_file))
            import flux1_single_file
            bad_shape = dict(original_vae)
            bad_shape['vae.encoder.conv_in.bias'] = bad_shape['vae.encoder.conv_in.bias'][:-1].clone()
            malformed_file = root / 'embedded-vae-bad-shape.safetensors'
            save_file(bad_shape, str(malformed_file))
            with self.assertRaisesRegex(ValueError, r'incompatible converted shape: encoder\.conv_in\.bias'):
                flux1_single_file.validate_embedded_vae(malformed_file, package, d)
            mixed_file = root / 'embedded-vae-mixed.safetensors'
            save_file({**original_vae, 'first_stage_model.encoder.conv_in.bias': torch.zeros(8)}, str(mixed_file))
            with self.assertRaisesRegex(ValueError, 'mixes original tensor namespaces'):
                flux1_single_file.validate_embedded_vae(mixed_file, package, d)
            def reference_full_file(path, config, **kwargs):
                return original_single_file(path, config=config, transformer=expected_transformer,
                                            vae=expected_vae, **kwargs)
            with patch.dict(mapping, {'checkpoint_mapping_fn': tiny_converter}):
                for mode, preprocessing in (
                    ('canny', options),
                    ('depth', ('--control-preprocess', 'depth', '--control-depth-model', str(depth_model))),
                ):
                    actual, report = embedded_run('full-' + mode, subject_path, preprocessing, True, full_file)
                    with patch.object(d.FluxPipeline, 'from_single_file', side_effect=reference_full_file):
                        reference, _ = embedded_run('full-reference-' + mode, subject_path, preprocessing, True, full_file)
                    self.assertEqual(report['request']['vae']['status'], 'model')
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    stream_full, _ = embedded_run('full-stream-' + mode, subject_path, preprocessing, True,
                                                  full_file, transformer_stream=True)
                    torch.testing.assert_close(stream_full, actual, rtol=0, atol=0)
                incomplete_file = root / 'embedded-flux-incomplete-vae.safetensors'
                incomplete = {**embedded_state, **original_vae}
                del incomplete['vae.encoder.conv_in.bias']
                save_file(incomplete, str(incomplete_file))
                with self.assertRaisesRegex(SystemExit, r'FLUX embedded VAE is missing converted weights:.*encoder\.conv_in\.bias'):
                    embedded_run('incomplete-vae', subject_path, options, True, incomplete_file)
            if torch.backends.mps.is_available():
                dense_mps, _ = run('text-scaled-mps', subject_path, options, 'a red fox', scaled_args, device='mps')
                stream_mps, _ = run('text-streamed-mps', subject_path, options, 'a red fox',
                                    (*scaled_args, '--flux-t5-fp8-storage', 'cpu-stream'), device='mps')
                torch.testing.assert_close(stream_mps, dense_mps, rtol=0, atol=0)
            if importlib.util.find_spec('peft'):
                style = root / 'style.safetensors'
                transform_key = 'transformer.transformer_blocks.0.attn.to_q'
                clip_key = 'text_encoder.text_model.encoder.layers.0.self_attn.q_proj'
                save_file({transform_key + '.lora_A.weight': torch.full((2, 32), .15),
                           transform_key + '.lora_B.weight': torch.full((32, 2), .15),
                           clip_key + '.lora_A.weight': torch.full((2, 16), .1),
                           clip_key + '.lora_B.weight': torch.full((16, 2), .1)}, str(style))
                style_options = (*scaled_args, '--lora', str(style), '--lora-scale', '.8')
                styled_dense, _ = run('text-styled-dense', subject_path, options, 'a red fox', style_options)
                styled_stream, styled_report = run('text-styled-stream', subject_path, options, 'a red fox',
                    (*style_options, '--flux-t5-fp8-storage', 'cpu-stream'))
                self.assertTrue(torch.equal(styled_dense, styled_stream))
                self.assertFalse(torch.equal(scaled_out, styled_stream))
                self.assertEqual(set(styled_report['adapters']['lora']['registered_components']), {'transformer', 'text_encoder'})
                if torch.backends.mps.is_available():
                    styled_dense_mps, _ = run('text-styled-dense-mps', subject_path, options, 'a red fox',
                                              style_options, device='mps')
                    styled_stream_mps, _ = run('text-styled-stream-mps', subject_path, options, 'a red fox',
                                               (*style_options, '--flux-t5-fp8-storage', 'cpu-stream'), device='mps')
                    torch.testing.assert_close(styled_stream_mps, styled_dense_mps, rtol=0, atol=0)

    def test_pixel_hint_controlnet_uses_its_own_patching(self):
        import torch
        import diffusers as d
        from PIL import Image
        import resource_fixture
        torch.manual_seed(123)
        torch.set_num_threads(2)
        transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6)).eval()
        net = d.FluxControlNetModel(patch_size=2, in_channels=64, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
            pooled_projection_dim=16, guidance_embeds=False, axes_dims_rope=(4, 6, 6),
            conditioning_embedding_channels=16).eval()
        with torch.no_grad():
            for layer in net.input_hint_block.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.constant_(layer.weight, .02)
                    if layer.bias is not None: torch.nn.init.zeros_(layer.bias)
            for layer in (*net.controlnet_blocks, *net.controlnet_single_blocks):
                torch.nn.init.constant_(layer.weight, .02)
                if layer.bias is not None: torch.nn.init.zeros_(layer.bias)
        vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(8, 8, 8, 8),
            norm_num_groups=4, layers_per_block=1, latent_channels=16,
            sample_size=64, scaling_factor=.3611, shift_factor=.1159).eval()
        pipeline = d.FluxPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(),
            vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None)
        with resource_fixture.TemporaryDirectory(prefix='flux-pixel-hint-') as temporary:
            folder = Path(temporary)
            net.save_pretrained(folder, safe_serialization=True)
            selected = control.Selection(str(folder), 'FluxPipeline', 'FluxControlNetPipeline',
                'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
            wrapped = control.attach(pipeline, selected, SimpleNamespace(control_mode=None), d, torch.float32)
            wrapped.set_progress_bar_config(disable=True)
            prompt, pooled = torch.randn(1, 4, 32), torch.randn(1, 16)
            def run(color):
                with torch.no_grad():
                    return wrapped(prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                        control_image=Image.new('RGB', (64, 64), (color,) * 3),
                        width=64, height=64, num_inference_steps=2, guidance_scale=1.,
                        generator=torch.Generator('cpu').manual_seed(17), output_type='latent').images
            first = run(40)
            self.assertTrue(torch.equal(first, run(40)))
            self.assertFalse(torch.equal(first, run(210)))

    def test_component_attach_and_prepared_hint_affect_real_flux_sampling(self):
        import torch
        import diffusers as d
        from PIL import Image
        import resource_fixture
        torch.manual_seed(123)
        torch.set_num_threads(2)
        transformer = d.FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=False,
            axes_dims_rope=(4, 6, 6)).eval()
        net = d.FluxControlNetModel.from_transformer(transformer, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2)
        with torch.no_grad():
            for module in (net.controlnet_x_embedder, *net.controlnet_blocks, *net.controlnet_single_blocks):
                torch.nn.init.constant_(module.weight, .01)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
        vae = d.AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4,
                              up_block_types=('UpDecoderBlock2D',) * 4,
                              block_out_channels=(8, 8, 8, 8), norm_num_groups=4, layers_per_block=1,
                              latent_channels=16, sample_size=64, scaling_factor=.3611, shift_factor=.1159).eval()
        pipeline = d.FluxPipeline(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler(), vae=vae,
                                  text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None)
        pipeline.set_progress_bar_config(disable=True)
        with resource_fixture.TemporaryDirectory(prefix='flux-control-') as temporary:
            folder = Path(temporary)
            net.save_pretrained(folder, safe_serialization=True)
            selection = control.Selection(str(folder), 'FluxPipeline', 'FluxControlNetPipeline',
                                          'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
            wrapped = control.attach(pipeline, selection, SimpleNamespace(control_mode=None), d, torch.float32)
            wrapped.set_progress_bar_config(disable=True)
            prompt = torch.randn(1, 4, 32)
            pooled = torch.randn(1, 16)
            def run(active, color, hint=None):
                with torch.no_grad():
                    return active(prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                        control_image=hint if hint is not None else Image.new('RGB', (64, 64), (color, color, color)),
                        width=64, height=64, num_inference_steps=2, guidance_scale=1.,
                        generator=torch.Generator('cpu').manual_seed(17), output_type='latent').images
            first, again, different = run(wrapped, 40), run(wrapped, 40), run(wrapped, 210)
            self.assertTrue(torch.equal(first, again))
            self.assertFalse(torch.equal(first, different))
            from native_canny_fixture import hints
            with self.subTest(preprocessor='native-canny'):
                blank, edges = hints()
                conditioned = run(wrapped, 40, edges)
                self.assertTrue(torch.equal(conditioned, run(wrapped, 40, edges)))
                self.assertFalse(torch.equal(conditioned, run(wrapped, 40, blank)))
            for precision in ('float32', 'float16'):
                with self.subTest(preprocessor='depth-anything', precision=precision):
                    from depth_hint_fixture import hints as depth_hints
                    first_hint, second_hint = depth_hints(precision=precision)
                    conditioned = run(wrapped, 40, first_hint)
                    self.assertTrue(torch.equal(conditioned, run(wrapped, 40, first_hint)))
                    self.assertFalse(torch.equal(conditioned, run(wrapped, 40, second_hint)))
            # Exercise the existing singular-extension file layout, including
            # the temporary Diffusers package and its cleanup, with real weights.
            weight = folder / 'control.safetensor'
            (folder / 'diffusion_pytorch_model.safetensors').rename(weight)
            hint = folder / 'hint.png'
            Image.new('RGB', (64, 64)).save(hint)
            args = SimpleNamespace(pipeline_class=None, controlnet=str(weight), controlnet_config=None,
                controlnet_variant=None, control_image=str(hint), controlnet_scale=None,
                control_guidance_start=None, control_guidance_end=None, guess_mode=None, control_mode=None,
                control_preprocess=None, control_preprocessor_library=None, control_color_image=None,
                control_color_end=None, control_depth_end=None, inputs={}, cache_dir=folder / '.cache')
            selected = control.resolve(args, {'_class_name': 'FluxPipeline'})
            single = control.attach(pipeline, selected, args, d, torch.float32)
            single.set_progress_bar_config(disable=True)
            self.assertTrue(torch.equal(first, run(single, 40)))
            self.assertFalse(list(args.cache_dir.glob('generic-controlnet-*')))


if __name__ == '__main__':
    unittest.main()
