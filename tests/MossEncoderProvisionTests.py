"""MOSS resource resume, integrity, and source identity contracts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import provision_moss_text_encoder as provisioner
import resource_fixture


class Response(io.BytesIO):
    def __init__(self, data, start, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{total - 1}/{total}'}


class MossEncoderProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = resource_fixture.TemporaryDirectory(prefix='moss-provision-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'original-int8-encoder'
        self.item = {'repo_id': provisioner.REPO, 'revision': provisioner.REVISION,
                     'source_file': provisioner.SOURCE, 'file': provisioner.TARGET, 'format': provisioner.FORMAT,
                     'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.catalog = self.root / 'generation-defaults.json'
        self.catalog.write_text(json.dumps({'fallback_moss_text_weights': self.item}))

    def test_resume_verify_and_preserve_existing(self):
        partial = self.root / '.state/moss-qwen35-download.part'
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

    def test_invalid_source_or_download_never_published(self):
        self.item['revision'] = 'changed'
        self.catalog.write_text(json.dumps({'fallback_moss_text_weights': self.item}))
        with self.assertRaisesRegex(ValueError, 'source or identity'):
            provisioner.provision(self.root)
        self.item['revision'] = provisioner.REVISION
        self.catalog.write_text(json.dumps({'fallback_moss_text_weights': self.item}))
        def wrong_range(request, timeout):
            return Response(self.data, 0, len(self.data), status=200)
        with self.assertRaisesRegex(RuntimeError, 'range'):
            provisioner.provision(self.root, opener=wrong_range)
        def wrong_content(request, timeout):
            return Response(b'x' * len(self.data), 0, len(self.data))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provisioner.provision(self.root, opener=wrong_content)
        self.assertFalse((self.root / provisioner.TARGET).exists())
        self.assertFalse((self.root / '.state/moss-qwen35-download.lock').exists())


if __name__ == '__main__':
    unittest.main()
