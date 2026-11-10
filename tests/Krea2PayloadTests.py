"""Optional payload audits for the provisioned Krea 2 resources."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))


def audit(path, item, fixture, *, hash_file, all_finite, sampled_names=(), scale_names=()):
    """Validate the on-disk identity, descriptors and requested payload values."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    before = path.stat()
    if before.st_size != item['size']:
        raise ValueError('Krea 2 resource size does not match the catalog')
    if hash_file:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('Krea 2 resource hash does not match the catalog')
    expected = {name: value for name, value in fixture.items() if name != '__metadata__'}
    with safe_open(str(path), framework='pt', device='cpu') as source:
        if set(source.keys()) != set(expected) or source.metadata() != fixture.get('__metadata__'):
            raise ValueError('Krea 2 resource keys or metadata do not match the fixture')
        for name, info in expected.items():
            view = source.get_slice(name)
            if view.get_shape() != info['shape'] or view.get_dtype() != info['dtype']:
                raise ValueError(f'Krea 2 resource descriptor does not match: {name}')
            if all_finite:
                if not bool(torch.isfinite(source.get_tensor(name).float()).all()):
                    raise ValueError(f'Krea 2 resource has nonfinite values: {name}')
        for name in sampled_names:
            if not bool(torch.isfinite(source.get_slice(name)[:1].float()).all()):
                raise ValueError(f'Krea 2 resource sample has nonfinite values: {name}')
        for name in scale_names:
            scale = source.get_tensor(name)
            if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
                raise ValueError(f'Krea 2 resource has invalid FP8 scale: {name}')
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('Krea 2 resource changed during the audit')


@unittest.skipUnless(AVAILABLE, 'Install torch and safetensors for payload auditing')
class Krea2PayloadTests(unittest.TestCase):
    def test_synthetic_audit_detects_nonfinite_values_bad_scales_and_hashes(self):
        import torch
        from safetensors.torch import save_file
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix='krea2-payload-audit-') as temporary:
            path = Path(temporary) / 'model.safetensors'
            tensors = {'layer.weight': torch.tensor([[1., 2.]]), 'layer.weight_scale': torch.tensor(0.5)}
            def save():
                save_file(tensors, str(path), metadata={'format': 'pt'})
                return {'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            fixture = {'__metadata__': {'format': 'pt'}, 'layer.weight': {'shape': [1, 2], 'dtype': 'F32'},
                       'layer.weight_scale': {'shape': [], 'dtype': 'F32'}}
            item = save()
            audit(path, item, fixture, hash_file=True, all_finite=True,
                  sampled_names=('layer.weight',), scale_names=('layer.weight_scale',))
            wrong_shape = {**fixture, 'layer.weight': {'shape': [2, 1], 'dtype': 'F32'}}
            with self.assertRaisesRegex(ValueError, 'descriptor does not match'):
                audit(path, item, wrong_shape, hash_file=False, all_finite=False)
            with self.assertRaisesRegex(ValueError, 'keys or metadata'):
                audit(path, item, {**fixture, '__metadata__': {'format': 'different'}},
                      hash_file=False, all_finite=False)
            tensors['layer.weight_scale'] = torch.tensor(-1.)
            item = save()
            with self.assertRaisesRegex(ValueError, 'invalid FP8 scale'):
                audit(path, item, fixture, hash_file=False, all_finite=False, scale_names=('layer.weight_scale',))
            tensors['layer.weight'] = torch.tensor([[float('nan'), 2.]])
            item = save()
            with self.assertRaisesRegex(ValueError, 'nonfinite values'):
                audit(path, item, fixture, hash_file=False, all_finite=True)
            item['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash does not match'):
                audit(path, item, fixture, hash_file=True, all_finite=False)

    def test_provisioned_depth_adapter(self):
        if os.environ.get('IILD_VERIFY_KREA2_DEPTH_PAYLOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_DEPTH_PAYLOAD=1 for the complete depth adapter audit')
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
        item = manifest['krea2_provisioned_resources']['depth']
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-depth-header.json').read_text())['header']
        audit(ROOT / 'resources' / item['file'], item, fixture, hash_file=True, all_finite=True)

    def test_provisioned_pose_adapter(self):
        if os.environ.get('IILD_VERIFY_KREA2_POSE_PAYLOAD') != '1':
            self.skipTest('Set IILD_VERIFY_KREA2_POSE_PAYLOAD=1 for the complete pose adapter audit')
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
        item = manifest['krea2_provisioned_resources']['pose']
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-ostris-pose-header.json').read_text())['header']
        audit(ROOT / 'resources' / item['file'], item, fixture, hash_file=True, all_finite=True)

    def test_provisioned_turbo_checkpoint(self):
        mode = os.environ.get('IILD_VERIFY_KREA2_TURBO_PAYLOAD')
        if mode not in ('1', 'metadata'):
            self.skipTest('Set IILD_VERIFY_KREA2_TURBO_PAYLOAD=metadata or 1 to audit Turbo')
        import krea2_single_file
        manifest = json.loads((ROOT / 'resources/generation-defaults.json').read_text())
        item = manifest['krea2_provisioned_resources']['turbo']
        fixture = json.loads((ROOT / 'resources/test-fixtures/krea2-turbo-fp8-header.json').read_text())['header']
        shapes = {name: info['shape'] for name, info in fixture.items() if name != '__metadata__'}
        scales = krea2_single_file.quantized_layers(shapes, fixture['__metadata__']).values()
        audit(ROOT / 'resources' / item['file'], item, fixture, hash_file=mode == '1', all_finite=False,
              sampled_names=('first.weight', 'blocks.0.attn.wq.weight', 'last.linear.weight'), scale_names=scales)


if __name__ == '__main__':
    unittest.main(verbosity=2)
