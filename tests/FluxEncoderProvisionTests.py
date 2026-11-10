"""Validate resumable, identity-checked FLUX encoder provisioning."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_flux1_text_encoders as provisioner
import provision_flux1_schnell as schnell
import provision_flux1_canny as canny
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, body, start, total, status=206):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class FluxEncoderProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='flux-provision-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'synthetic-encoder'
        entry = {'component': 'text_encoder', 'repo_id': provisioner.REPO, 'revision': provisioner.REVISION,
                 'source_file': provisioner.SOURCES['clip'], 'file': provisioner.TARGETS['clip'],
                 'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        (self.root / 'generation-defaults.json').write_text(json.dumps({'fallback_flux_text_weights': [entry]}))

    def test_resume_then_verify_and_never_replace(self):
        partial = self.root / '.state/flux1-clip-download.part'
        partial.parent.mkdir()
        partial.write_bytes(self.data[:4])
        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=4-')
            return Response(self.data[4:], 4, len(self.data))
        self.assertEqual(provisioner.provision(self.root, 'clip', opener=open_range), 'provisioned')
        target = self.root / provisioner.TARGETS['clip']
        self.assertEqual(target.read_bytes(), self.data)
        self.assertFalse(partial.exists())
        self.assertEqual(provisioner.provision(self.root, 'clip', check=True), 'verified')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root, 'clip', opener=open_range)
        self.assertEqual(target.read_bytes(), b'changed')

    def test_ignoring_range_and_bad_checksum_do_not_publish(self):
        def wrong_range(request, timeout):
            return Response(self.data, 0, len(self.data), status=200)
        with self.assertRaisesRegex(RuntimeError, 'range'):
            provisioner.provision(self.root, 'clip', opener=wrong_range)
        self.assertFalse((self.root / provisioner.TARGETS['clip']).exists())
        def wrong_content(request, timeout):
            return Response(b'x' * len(self.data), 0, len(self.data))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root, 'clip', opener=wrong_content)
        self.assertFalse((self.root / provisioner.TARGETS['clip']).exists())
        self.assertTrue((self.root / '.state/flux1-clip-download.part').exists())
        self.assertFalse((self.root / '.state/flux1-clip-download.lock').exists())

    def test_schnell_catalog_matches_public_checkpoint_header(self):
        item = schnell.validate_catalog(ROOT / 'resources')
        fixture = json.loads((ROOT / 'resources/test-fixtures/flux1-schnell-fp8-full-header.json').read_text())
        self.assertEqual((item['repo_id'], item['revision'], item['source_file'], item['size'], item['sha256']),
                         (fixture['repository'], fixture['revision'], fixture['filename'],
                          fixture['file_size'], fixture['file_sha256']))

    def test_schnell_resume_and_source_identity(self):
        item = {'repo_id': schnell.REPO, 'revision': schnell.REVISION, 'source_file': schnell.SOURCE,
                'file': schnell.TARGET, 'format': schnell.FORMAT, 'license': 'apache-2.0',
                'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        catalog = self.root / 'generation-defaults.json'
        catalog.write_text(json.dumps({'flux1_schnell_checkpoint_resource': item}))
        item['revision'] = 'different'
        catalog.write_text(json.dumps({'flux1_schnell_checkpoint_resource': item}))
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            schnell.provision(self.root)
        item['revision'] = schnell.REVISION
        catalog.write_text(json.dumps({'flux1_schnell_checkpoint_resource': item}))
        partial = self.root / '.state/flux1-schnell-download.part'
        partial.parent.mkdir()
        partial.write_bytes(self.data[:4])
        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=4-')
            return Response(self.data[4:], 4, len(self.data))
        self.assertEqual(schnell.provision(self.root, opener=open_range), 'provisioned')
        self.assertEqual(schnell.provision(self.root, check=True), 'verified')
        target = self.root / schnell.TARGET
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            schnell.provision(self.root)
        self.assertEqual(target.read_bytes(), b'changed')

    def test_published_canny_catalog_matches_header_snapshot(self):
        entries = canny.validate_catalog(ROOT / 'resources')
        fixture = json.loads((ROOT / 'resources/test-fixtures/flux1-instantx-canny-header.json').read_text())
        self.assertEqual((fixture['repository'], fixture['revision']), (canny.REPO, canny.REVISION))
        self.assertEqual(entries['config.json']['sha256'], fixture['config_sha256'])
        weights = entries['diffusion_pytorch_model.safetensors']
        self.assertEqual((weights['size'], weights['sha256']), (fixture['weight_size_bytes'], fixture['weight_sha256']))

    def test_canny_resumes_both_files_and_refuses_changed_target_or_source(self):
        content = {name: ('synthetic-' + name).encode() for name in canny.FILES}
        files = [{'source_file': name, 'file': f'models/flux1-instantx-canny/{name}',
                  'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                 for name, data in content.items()]
        item = {'repo_id': canny.REPO, 'revision': canny.REVISION,
                'license': 'flux-1-dev-non-commercial-license', 'files': files}
        catalog = self.root / 'generation-defaults.json'
        def save_catalog():
            catalog.write_text(json.dumps({'flux1_instantx_canny_resource': item}))
        save_catalog()
        partial = self.root / '.state/flux1-instantx-canny-diffusion_pytorch_model.safetensors-download.part'
        partial.parent.mkdir()
        partial.write_bytes(content['diffusion_pytorch_model.safetensors'][:4])
        def open_range(request, timeout):
            name = request.full_url.split('?')[0].rsplit('/', 1)[1]
            start = 4 if name.endswith('.safetensors') else 0
            self.assertEqual(request.get_header('Range'), f'bytes={start}-')
            return Response(content[name][start:], start, len(content[name]))
        self.assertEqual(set(canny.provision(self.root, opener=open_range).values()), {'provisioned'})
        self.assertEqual(set(canny.provision(self.root, check=True).values()), {'verified'})
        target = self.root / files[0]['file']
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            canny.provision(self.root)
        self.assertEqual(target.read_bytes(), b'changed')
        item['revision'] = 'changed'
        save_catalog()
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            canny.provision(self.root)


if __name__ == '__main__':
    unittest.main()
