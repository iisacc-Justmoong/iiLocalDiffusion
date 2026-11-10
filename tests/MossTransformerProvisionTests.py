"""MOSS SFT transformer provisioning identity and safe resume contracts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_moss_transformer as provisioner
import provision_moss_control as control_provisioner
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, data, start, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class MossTransformerProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = resource_fixture.TemporaryDirectory(prefix='moss-transformer-provision-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b'original-moss-transformer'
        self.item = {'repo_id': provisioner.REPO, 'revision': provisioner.REVISION,
                     'source_file': provisioner.SOURCE, 'file': provisioner.TARGET, 'format': provisioner.FORMAT,
                     'license': 'apache-2.0', 'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.catalog = self.root / 'generation-defaults.json'
        self.catalog.write_text(json.dumps({'moss_sft_transformer_resource': self.item}))

    def test_pinned_catalog_matches_public_header_identity(self):
        item = provisioner.validate_catalog(ROOT / 'resources')
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-sft-base-header.json').read_text())
        self.assertEqual(item['size'], fixture['weight_size_bytes'])
        self.assertEqual(item['sha256'], fixture['weight_sha256'])
        self.assertEqual(item['revision'], fixture['revision'])

    def test_resume_verify_and_do_not_replace_existing(self):
        partial = self.root / '.state/moss-sft-transformer-download.part'
        partial.parent.mkdir()
        partial.write_bytes(self.data[:6])
        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=6-')
            return Response(self.data[6:], 6, len(self.data))
        self.assertEqual(provisioner.provision(self.root, opener=open_range), 'provisioned')
        target = self.root / provisioner.TARGET
        self.assertEqual(target.read_bytes(), self.data)
        self.assertFalse(partial.exists())
        self.assertEqual(provisioner.provision(self.root, check=True), 'verified')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root)
        self.assertEqual(target.read_bytes(), b'changed')

    def test_bad_source_or_download_is_not_published(self):
        self.item['revision'] = 'changed'
        self.catalog.write_text(json.dumps({'moss_sft_transformer_resource': self.item}))
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            provisioner.provision(self.root)
        self.item['revision'] = provisioner.REVISION
        self.catalog.write_text(json.dumps({'moss_sft_transformer_resource': self.item}))
        def wrong_range(request, timeout):
            return Response(self.data, 0, len(self.data), status=200)
        with self.assertRaisesRegex(RuntimeError, 'range'):
            provisioner.provision(self.root, opener=wrong_range)
        def wrong_content(request, timeout):
            return Response(b'x' * len(self.data), 0, len(self.data))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root, opener=wrong_content)
        self.assertFalse((self.root / provisioner.TARGET).exists())
        self.assertFalse((self.root / '.state/moss-sft-transformer-download.lock').exists())

    def test_control_adapter_catalog_and_safe_provisioning(self):
        fixture = json.loads((ROOT / 'resources/test-fixtures/moss-color-depth-header.json').read_text())
        item = control_provisioner.validate_catalog(ROOT / 'resources')
        self.assertEqual((item['size'], item['sha256'], item['revision']),
                         (fixture['weight_size_bytes'], fixture['weight_sha256'], fixture['revision']))
        control = {'repo_id': control_provisioner.REPO, 'revision': control_provisioner.REVISION,
                   'source_file': control_provisioner.SOURCE, 'file': control_provisioner.TARGET,
                   'format': control_provisioner.FORMAT, 'license': 'apache-2.0',
                   'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.catalog.write_text(json.dumps({'moss_color_depth_control_resource': control}))
        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=0-')
            return Response(self.data, 0, len(self.data))
        self.assertEqual(control_provisioner.provision(self.root, opener=open_range), 'provisioned')
        target = self.root / control_provisioner.TARGET
        self.assertEqual(target.read_bytes(), self.data)
        self.assertEqual(control_provisioner.provision(self.root, check=True), 'verified')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            control_provisioner.provision(self.root)
        self.assertEqual(target.read_bytes(), b'changed')


if __name__ == '__main__':
    unittest.main()
