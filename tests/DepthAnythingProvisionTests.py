"""Pinned depth preprocessor resources and safe multi-file provisioning."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_depth_anything_v2 as provisioner
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, body, start, total):
        super().__init__(body)
        self.status = 206
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class DepthAnythingProvisionTests(unittest.TestCase):
    def test_bundled_catalog_validates_both_variants(self):
        for variant in provisioner.SOURCES:
            with self.subTest(variant=variant):
                self.assertEqual(set(provisioner.validate_catalog(ROOT / 'resources', variant)), set(provisioner.FILES))

    def test_resume_check_and_refuse_changed_files_or_catalog(self):
        with resource_fixture.TemporaryDirectory(prefix='depth-v2-provision-') as temporary:
            root = Path(temporary)
            repo, revision, license_name = provisioner.SOURCES['small']
            content = {name: ('payload-' + name).encode() for name in provisioner.FILES}
            entries = [{'source_file': name, 'file': f'depth-estimators/depth-anything-v2-small/{name}',
                        'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                       for name, data in content.items()]
            catalog = {'depth_anything_v2_resources': {'small': {'repo_id': repo, 'revision': revision,
                        'license': license_name, 'files': entries}}}
            manifest = root / 'generation-defaults.json'
            def save_catalog():
                manifest.write_text(json.dumps(catalog))
            save_catalog()
            partial = root / '.state/depth-v2-small-model.safetensors-download.part'
            partial.parent.mkdir()
            partial.write_bytes(content['model.safetensors'][:4])
            def open_range(request, timeout):
                name = request.full_url.split('?')[0].rsplit('/', 1)[1]
                start = 4 if name == 'model.safetensors' else 0
                self.assertEqual(request.get_header('Range'), f'bytes={start}-')
                return Response(content[name][start:], start, len(content[name]))
            self.assertEqual(set(provisioner.provision(root, 'small', opener=open_range).values()), {'provisioned'})
            self.assertEqual(set(provisioner.provision(root, 'small', check=True).values()), {'verified'})
            target = root / entries[0]['file']
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                provisioner.provision(root, 'small')
            self.assertEqual(target.read_bytes(), b'changed')
            entries[0]['file'] = '../escape'
            save_catalog()
            with self.assertRaisesRegex(ValueError, 'file identity'):
                provisioner.provision(root, 'small')
            entries[0]['source_file'] = entries[1]['source_file']
            save_catalog()
            with self.assertRaisesRegex(ValueError, 'file catalog'):
                provisioner.provision(root, 'small')


if __name__ == '__main__':
    unittest.main()
