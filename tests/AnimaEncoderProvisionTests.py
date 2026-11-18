"""Anima encoder provisioning: pinned source, resume and safe publication."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_anima_text_encoder as provisioner
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, data, start, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class AnimaEncoderProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='anima-provision-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'synthetic-qwen3-encoder'
        self.item = {'repo_id': provisioner.REPO, 'revision': provisioner.REVISION,
                     'source_file': provisioner.SOURCE, 'file': provisioner.TARGET,
                     'format': 'safetensors', 'class_name': 'Qwen3ForCausalLM',
                     'families': ['anima'], 'optional': True, 'license': 'apache-2.0',
                     'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.catalog = self.root / 'generation-defaults.json'
        self.save()
        mocked = patch.multiple(provisioner, SIZE=self.item['size'], SHA256=self.item['sha256'])
        mocked.start()
        self.addCleanup(mocked.stop)

    def save(self):
        self.catalog.write_text(json.dumps({'fallback_llms': [self.item]}))

    def test_resume_verify_and_preserve_existing(self):
        partial = self.root / '.state/anima-qwen3-download.part'
        partial.parent.mkdir()
        partial.write_bytes(self.data[:5])
        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=5-')
            return Response(self.data[5:], 5, len(self.data))
        self.assertEqual(provisioner.provision(self.root, opener=open_range), 'provisioned')
        target = self.root / provisioner.TARGET
        self.assertEqual(target.read_bytes(), self.data)
        self.assertFalse(partial.exists())
        self.assertEqual(provisioner.provision(self.root, check=True), 'verified')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root)
        self.assertEqual(target.read_bytes(), b'changed')

    def test_invalid_source_and_download_never_published(self):
        self.item['revision'] = 'changed'
        self.save()
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            provisioner.provision(self.root)
        self.item['revision'] = provisioner.REVISION
        self.save()
        def bad_range(request, timeout):
            return Response(self.data, 0, len(self.data), status=200)
        with self.assertRaisesRegex(RuntimeError, 'range'):
            provisioner.provision(self.root, opener=bad_range)
        def bad_content(request, timeout):
            return Response(b'x' * len(self.data), 0, len(self.data))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root, opener=bad_content)
        self.assertFalse((self.root / provisioner.TARGET).exists())
        self.assertFalse((self.root / '.state/anima-qwen3-download.lock').exists())

    def test_catalog_rejects_ambiguous_family_or_unpinned_digest(self):
        self.item['sha256'] = '0' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            provisioner.provision(self.root)
        self.item['sha256'] = hashlib.sha256(self.data).hexdigest()
        self.catalog.write_text(json.dumps({'fallback_llms': [self.item, self.item]}))
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            provisioner.provision(self.root)


class AnimaEncoderCatalogTests(unittest.TestCase):
    def test_workspace_catalog_pins_the_official_encoder(self):
        item = provisioner.validate_catalog(ROOT / 'resources')
        self.assertEqual(item['size'], 1192135096)
        self.assertEqual(item['sha256'], 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba')


if __name__ == '__main__':
    unittest.main()
