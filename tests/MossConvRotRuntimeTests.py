"""Numerical tests for portable MOSS Qwen3.5 INT8 ConvRot inference."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import moss_convrot
import moss_text_encoder

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'safetensors'))


def comfy_eager_reference():
    """Execute only the pinned upstream CPU reference functions, offline."""
    import ast
    import hashlib
    import math
    import torch
    fixtures = ROOT / 'resources/test-fixtures'
    sources = (
        ('comfy-kitchen-cdab-int8-utils.py', '4776532e7394c56262cc02c48217a20b520d123c2bf9539eaf77cf0ff32ff1b8',
         {'_build_hadamard', '_rotate_activation', '_rotate_weight'}),
        ('comfy-kitchen-cdab-activations.py', 'ed51bf8955ebe505cfd6d5e3fe9090e61aa054cc2ba9c8ae990101fd6636c0f1',
         {'apply_input_act', 'apply_residual'}),
        ('comfy-kitchen-cdab-eager-quantization.py', '4412eb5bbd12cb2c9a09c5c9e076e744e16bf39632fe624e4df739b84681ef12',
         {'_cuda_device_is_turing', '_int8_mm_n_alignment', '_round_up', '_int8_matmul_accumulate',
          '_int8_stochastic_rng', '_int8_scale_for_math', '_round_int8', 'quantize_int8_rowwise', 'int8_linear'}),
    )
    namespace = {'torch': torch, 'math': math, '_HADAMARD_CACHE': {}, '_turing_device_cache': {}}
    for name, digest, requested in sources:
        raw = (fixtures / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Upstream Comfy Kitchen reference has changed: {name}.')
        tree = ast.parse(raw, filename=name)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in requested]
        if {node.name for node in functions} != requested:
            raise ValueError(f'Upstream Comfy Kitchen reference is incomplete: {name}.')
        module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        exec(compile(module, name, 'exec'), namespace)
    namespace['_apply_input_act'] = namespace['apply_input_act']
    namespace['_apply_residual'] = namespace['apply_residual']
    return namespace['int8_linear']


@unittest.skipUnless(AVAILABLE, 'Install MOSS reference dependencies')
class MossConvRotRuntimeTests(unittest.TestCase):
    def test_mps_int8_accumulation_matches_cpu_for_large_and_extreme_inputs(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('MPS is unavailable')
        torch.manual_seed(190)
        for width in (1024, 2051, 3072, 4096):
            with self.subTest(width=width):
                activations = torch.randint(-128, 128, (8, width), dtype=torch.int8)
                weights = torch.randint(-128, 128, (9, width), dtype=torch.int8)
                activations[0].fill_(127)
                activations[1].fill_(-128)
                weights[0].fill_(127)
                weights[1].fill_(-128)
                expected = activations.to(torch.int64) @ weights.to(torch.int64).T
                actual = moss_convrot.mps_int8_mm(activations.to('mps'), weights).to('cpu').to(torch.int64)
                self.assertTrue(torch.equal(actual, expected),
                                f'width {width}, maximum integer error {(actual - expected).abs().max().item()}')

    def test_mps_linear_matches_cpu_fallback_for_same_quantized_inputs(self):
        import torch
        import warnings
        if not torch.backends.mps.is_available():
            self.skipTest('MPS is unavailable')
        torch.manual_seed(291)
        inputs = torch.randn(2, 3, 2052, dtype=torch.float32).to('mps')
        weight = torch.randint(-128, 128, (7, 2052), dtype=torch.int8)
        scale = torch.rand(7, 1, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with patch.object(moss_convrot, 'mps_int8_mm', side_effect=RuntimeError('injected fallback')):
                expected = moss_convrot.linear(inputs, weight, scale, None, 4)
        with patch.object(moss_convrot, 'mps_int8_mm', wraps=moss_convrot.mps_int8_mm) as accelerated:
            actual = moss_convrot.linear(inputs, weight, scale, None, 4)
        accelerated.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_provisioned_int8_encoder_loads_actual_language_weights(self):
        if os.environ.get('IILD_VERIFY_MOSS_ENCODER_LOAD') != '1':
            self.skipTest('Set IILD_VERIFY_MOSS_ENCODER_LOAD=1 to load the real INT8 language encoder')
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        path = ROOT / 'resources/text_encoders/moss_qwen35_int8_convrot.safetensors'
        config = ROOT / 'resources/text_encoders/qwen35-9b/config.json'
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            selection = moss_text_encoder.resolve(str(path), 'MOSSImagePipeline')
            model = moss_text_encoder.load(selection, config, torch.bfloat16)
        linear_type, embedding_type = moss_convrot.module_types()
        self.assertTrue(model._iild_cpu_int8_convrot)
        self.assertEqual(sum(isinstance(module, linear_type) for module in model.modules()), 200)
        self.assertEqual(sum(isinstance(module, embedding_type) for module in model.modules()), 1)
        self.assertTrue(all(not parameter.is_meta for parameter in model.parameters()))
        layer = model.get_submodule('layers.0.mlp.gate_proj')
        self.assertIsInstance(layer, linear_type)
        self.assertEqual(layer._iild_int8.device.type, 'cpu')
        self.assertEqual(layer._iild_int8.dtype, torch.int8)
        with safe_open(str(path), framework='pt', device='cpu') as source:
            prefix = 'model.language_model.layers.0.mlp.gate_proj'
            self.assertTrue(torch.equal(layer._iild_int8[:1], source.get_slice(prefix + '.weight')[:1]))
            torch.testing.assert_close(layer._iild_scale[:1], source.get_slice(prefix + '.weight_scale')[:1],
                                       rtol=0, atol=0)

    def test_provisioned_public_encoder_metadata_scales_and_embedding(self):
        mode = os.environ.get('IILD_VERIFY_MOSS_ENCODER_PAYLOAD')
        if mode not in ('1', 'metadata'):
            self.skipTest('Set IILD_VERIFY_MOSS_ENCODER_PAYLOAD=1 to audit the provisioned 9.7 GB file')
        import hashlib
        import torch
        from safetensors import safe_open
        import moss_int8_format
        path = Path(os.environ.get('IILD_MOSS_ENCODER_AUDIT_PATH',
                                   ROOT / 'resources/text_encoders/moss_qwen35_int8_convrot.safetensors')).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('MOSS test input must stay inside the project workspace.')
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())['fallback_moss_text_weights']
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-qwen35-encoder-header.json').read_text())
        self.assertTrue(path.is_file(), 'Provision the public MOSS encoder first')
        before = path.stat()
        if mode == '1':
            with path.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), manifest['sha256'])
        self.assertEqual(path.stat().st_size, manifest['size'])
        expected = {name[len(moss_int8_format.PREFIX):]: value['shape'] for name, value in fixture['header'].items()
                    if name.startswith(moss_int8_format.PREFIX)}
        with safe_open(str(path), framework='pt', device='cpu') as source:
            header = {name: {'shape': source.get_slice(name).get_shape(), 'dtype': source.get_slice(name).get_dtype()}
                      for name in source.keys()}
            metadata = {name: json.loads(bytes(source.get_tensor(name).tolist())) for name in header
                        if name.startswith(moss_int8_format.PREFIX) and name.endswith('.comfy_quant')}
            _, quantized = moss_int8_format.plan(header, expected, metadata)
            self.assertEqual(len(quantized), 201)
            for item in quantized.values():
                scale = source.get_tensor(item.scale)
                self.assertTrue(bool(torch.isfinite(scale).all()), item.scale)
                self.assertFalse(bool((scale < 0).any()), item.scale)
            item = quantized['model.language_model.embed_tokens.weight']
            sample = source.get_slice(item.weight)[:4]
            scales = source.get_slice(item.scale)[:4]
            ids = torch.tensor([[3, 0, 2]])
            h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                               [1., -1., 1., 1.], [-1., 1., 1., 1.]])
            h = h4
            for _ in range(3):
                h = torch.kron(h, h4)
            reference = ((sample[ids].float() * scales[ids]).reshape(-1, 256) @ (h / 16).T).reshape(1, 3, -1)
            torch.testing.assert_close(moss_convrot.embedding(sample, scales, ids, 256, torch.float32),
                                       reference, rtol=0, atol=0)
            # Use actual stored rows and scales from a Qwen3.5 MLP projection.
            # The pinned upstream eager functions are the independent oracle;
            # no pretrained model or full image generation is run here.
            projection = quantized['model.language_model.layers.0.mlp.gate_proj.weight']
            source_weight = source.get_slice(projection.weight)[:11].contiguous()
            source_scale = source.get_slice(projection.scale)[:11].contiguous()
            oracle = comfy_eager_reference()
            generator = torch.Generator().manual_seed(639)
            x = torch.randn((2, 3, source_weight.shape[1]), generator=generator)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    inputs = x.to(dtype)
                    bias = torch.linspace(-.5, .5, source_weight.shape[0], dtype=dtype)
                    actual = moss_convrot.linear(inputs, source_weight, source_scale, bias, projection.group_size)
                    expected = oracle(inputs, source_weight, source_scale, bias, out_dtype=dtype,
                                      convrot=True, convrot_groupsize=projection.group_size)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        after = path.stat()
        self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_bfloat16_linear_and_embedding(self):
        import torch
        h = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                          [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
        weight = torch.tensor([[1, -3, 5, 7, -1, 6, -2, 4],
                               [2, 3, -4, 1, 5, -7, 1, 2]], dtype=torch.int8)
        scale = torch.tensor([[0.25], [0.125]])
        oracle = comfy_eager_reference()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = torch.tensor([[0.5, -0.75, 1, 2, -1, 0.125, 0.25, 0.4]], dtype=dtype)
                rotated = (inputs.reshape(-1, 4) @ h.to(dtype)).reshape(inputs.shape)
                row_scale = (rotated.abs().amax(-1, keepdim=True).float() / 127).clamp(min=1e-30)
                q = (rotated / row_scale.to(dtype)).round().clamp(-128, 127).to(torch.int64)
                expected = ((q @ weight.to(torch.int64).T).float() * (row_scale * scale.T)).to(dtype)
                torch.testing.assert_close(moss_convrot.linear(inputs, weight, scale, None, 4), expected, rtol=0, atol=0)
                torch.testing.assert_close(moss_convrot.linear(inputs, weight, scale, None, 4),
                                           oracle(inputs, weight, scale, out_dtype=dtype,
                                                  convrot=True, convrot_groupsize=4), rtol=0, atol=0)
                ids = torch.tensor([[1, 0, 1]])
                expected_embedding = ((weight[ids].float() * scale[ids]).reshape(-1, 4) @ h.T).reshape(1, 3, 8)
                torch.testing.assert_close(moss_convrot.embedding(weight, scale, ids, 4, dtype),
                                           expected_embedding.to(dtype), rtol=0, atol=0)

    def test_quantized_qwen_matches_independent_reference(self):
        import torch
        from safetensors.torch import save_file
        from transformers import Qwen3_5Config, Qwen3_5TextModel
        import resource_fixture
        from weight_files import resolve_weight_file

        torch.manual_seed(932)
        torch.set_num_threads(2)
        config = Qwen3_5Config(text_config={'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32,
            'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,
            'head_dim': 8, 'layer_types': ['linear_attention', 'full_attention'],
            'linear_key_head_dim': 8, 'linear_value_head_dim': 8,
            'linear_num_key_heads': 2, 'linear_num_value_heads': 2},
            vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                           'out_hidden_size': 16, 'num_position_embeddings': 16})
        reference = Qwen3_5TextModel(config.text_config).eval()
        h = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                          [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
        weights = {'model.language_model.' + name: tensor.detach().clone()
                   for name, tensor in reference.state_dict().items()}
        quantized = {}
        for name, module in list(reference.named_modules()):
            if type(module) not in (torch.nn.Linear, torch.nn.Embedding):
                continue
            key = 'model.language_model.' + name + '.weight'
            original = weights[key]
            rotated = (original.reshape(-1, 4) @ h.T).reshape(original.shape)
            scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-30)
            encoded = (rotated / scale).round().clamp(-128, 127).to(torch.int8)
            weights[key] = encoded
            weights[key[:-7] + '.weight_scale'] = scale
            meta = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4}).encode()
            weights[key[:-7] + '.comfy_quant'] = torch.tensor(list(meta), dtype=torch.uint8)
            quantized[name] = (encoded, scale)

        class ReferenceLinear(torch.nn.Module):
            def __init__(self, module, encoded, scale):
                super().__init__()
                self.encoded, self.scale, self.bias = encoded, scale, module.bias

            def forward(self, x):
                shape = x.shape
                x = (x.reshape(-1, 4) @ h.to(x.dtype)).reshape(-1, shape[-1])
                scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127).clamp(min=1e-30)
                denominator = scale.to(x.dtype)
                denominator = torch.where(denominator == 0,
                    torch.full_like(denominator, torch.finfo(x.dtype).tiny), denominator)
                q = (x / denominator).round().clamp(-128, 127).to(torch.int64)
                total = q @ self.encoded.to(torch.int64).T
                result = (total.float() * (scale * self.scale.T)).to(x.dtype)
                if self.bias is not None:
                    result = result + self.bias
                return result.reshape(*shape[:-1], self.encoded.shape[0])

        class ReferenceEmbedding(torch.nn.Module):
            def __init__(self, encoded, scale):
                super().__init__()
                self.encoded, self.scale = encoded, scale

            def forward(self, ids):
                x = self.encoded[ids].float() * self.scale[ids]
                return (x.reshape(-1, 4) @ h.T).reshape(*ids.shape, -1)

        for name, (encoded, scale) in quantized.items():
            module = reference.get_submodule(name)
            replacement = (ReferenceEmbedding(encoded, scale) if type(module) is torch.nn.Embedding
                           else ReferenceLinear(module, encoded, scale))
            reference.set_submodule(name, replacement)
        with resource_fixture.TemporaryDirectory(prefix='moss-int8-') as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps(config.to_dict()))
            checkpoint = root / 'encoder.safetensors'
            save_file(weights, str(checkpoint))
            selected = resolve_weight_file(str(checkpoint), 'MOSS text encoder')
            events = []
            actual = moss_text_encoder.load(selected, root, torch.float32,
                                            progress=lambda *event: events.append(event))
            self.assertTrue(events)
            self.assertTrue(all(event[0] == 'text_encoder' for event in events))
            self.assertEqual(events[-1][1], events[-1][2])
            self.assertGreater(events[-1][2], 0)
            self.assertEqual([event[1] for event in events], sorted(event[1] for event in events))
            self.assertTrue(actual._iild_cpu_int8_convrot)
            ids = torch.tensor([[1, 4, 12, 3], [2, 8, 5, 6]])
            loading_event_count = len(events)
            with torch.no_grad():
                expected = reference(input_ids=ids, use_cache=False).last_hidden_state
                output = actual(input_ids=ids, use_cache=False).last_hidden_state
            compute_events = [event for event in events[loading_event_count:] if event[0] == 'text_encoder_compute']
            self.assertTrue(compute_events)
            self.assertEqual(compute_events[-1][1], compute_events[-1][2])
            torch.testing.assert_close(output, expected, rtol=3e-6, atol=3e-7)
            with self.assertRaisesRegex(RuntimeError, 'cannot be serialized'):
                actual.state_dict()
            encoded = actual.embed_tokens._iild_int8
            actual.to(dtype=torch.bfloat16)
            self.assertIs(actual.embed_tokens._iild_int8, encoded)
            self.assertEqual(actual.embed_tokens.weight.dtype, torch.bfloat16)
            if torch.backends.mps.is_available():
                actual.to(device='mps', dtype=torch.float32)
                with torch.no_grad():
                    mps_output = actual(input_ids=ids.to('mps'), use_cache=False).last_hidden_state.cpu()
                torch.testing.assert_close(mps_output, expected, rtol=0.005, atol=0.005)
                self.assertEqual(actual.embed_tokens._iild_int8.device.type, 'cpu')
            broken = dict(weights)
            key = next(name for name in broken if name.endswith('.weight_scale'))
            broken[key] = torch.full_like(broken[key], float('nan'))
            save_file(broken, str(root / 'bad.safetensors'))
            with self.assertRaisesRegex(ValueError, 'scale values'):
                moss_text_encoder.load(resolve_weight_file(str(root / 'bad.safetensors'), 'bad'), root, torch.float32)


if __name__ == '__main__':
    unittest.main()
