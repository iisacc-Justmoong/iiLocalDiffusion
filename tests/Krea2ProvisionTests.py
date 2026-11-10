"""Pinned Krea 2 checkpoint and depth adapter provisioning contracts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_krea2 as provisioner
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, body, start, total):
        super().__init__(body)
        self.status = 206
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class Krea2ProvisionTests(unittest.TestCase):
    def test_catalog_matches_both_public_header_identities(self):
        for role, fixture_name in [('turbo', 'krea2-turbo-fp8-header.json'), ('depth', 'krea2-depth-header.json'),
                                   ('pose', 'krea2-ostris-pose-header.json')]:
            with self.subTest(role=role):
                item = provisioner.validate_catalog(ROOT / 'resources', role)
                fixture = json.loads((ROOT / 'resources/test-fixtures' / fixture_name).read_text())
                self.assertEqual((item['revision'], item['size'], item['sha256']),
                                 (fixture['revision'], fixture['weight_size_bytes'], fixture['weight_sha256']))

    def test_all_resources_resume_check_and_reject_changed_targets(self):
        with resource_fixture.TemporaryDirectory(prefix='krea2-provision-') as temporary:
            root = Path(temporary)
            data = b'fixture-checkpoint'
            entries = {}
            for role, (repo, revision, source, target, format_name, license_name) in provisioner.SOURCES.items():
                entries[role] = {'repo_id': repo, 'revision': revision, 'source_file': source, 'file': target,
                                 'format': format_name, 'license': license_name,
                                 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            catalog = root / 'generation-defaults.json'
            catalog.write_text(json.dumps({'krea2_provisioned_resources': None}))
            with self.assertRaisesRegex(ValueError, 'source or identity'):
                provisioner.provision(root, 'depth')
            catalog.write_text(json.dumps({'krea2_provisioned_resources': entries}))
            for role in provisioner.SOURCES:
                with self.subTest(role=role):
                    partial = root / f'.state/krea2-{role}-download.part'
                    partial.parent.mkdir(exist_ok=True)
                    partial.write_bytes(data[:4])
                    def open_range(request, timeout):
                        self.assertEqual(request.get_header('Range'), 'bytes=4-')
                        return Response(data[4:], 4, len(data))
                    self.assertEqual(provisioner.provision(root, role, opener=open_range), 'provisioned')
                    self.assertEqual(provisioner.provision(root, role, check=True), 'verified')
                    target = root / provisioner.SOURCES[role][3]
                    target.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'does not match'):
                        provisioner.provision(root, role)
                    self.assertEqual(target.read_bytes(), b'changed')
            entries['depth']['revision'] = 'changed'
            catalog.write_text(json.dumps({'krea2_provisioned_resources': entries}))
            with self.assertRaisesRegex(ValueError, 'source or identity'):
                provisioner.provision(root, 'depth')


if __name__ == '__main__':
    unittest.main()
