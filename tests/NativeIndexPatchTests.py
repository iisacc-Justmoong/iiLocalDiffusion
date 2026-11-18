"""Check the native index probe and Wan patch can be applied repeatedly."""
import resource_fixture
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv.pop(1)).resolve()
GIT = sys.argv.pop(1)
FILES = ('src/model_loader.cpp', 'src/stable-diffusion.cpp')


class NativeIndexPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = resource_fixture.TemporaryDirectory(prefix='native-index-patches-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        for name in FILES:
            target = self.work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, target)

    def apply(self, name, *arguments):
        return subprocess.run([GIT, 'apply', *arguments, str(ROOT / 'cmake/patches' / (name + '.patch'))],
                              cwd=self.work, text=True, capture_output=True)

    def test_unapply_and_reapply_preserves_backend(self):
        before = {name: (self.work / name).read_bytes() for name in FILES}
        for patch, arguments in (('native-index-content-format', ('--reverse',)),
                                 ('native-wan-expert-namespace', ('--reverse',)),
                                 ('native-wan-expert-namespace', ()),
                                 ('native-index-content-format', ())):
            result = self.apply(patch, *arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {name: (self.work / name).read_bytes() for name in FILES})

    def test_relaxed_context_still_detects_a_modified_wan_patch(self):
        self.assertEqual(self.apply('native-index-content-format', '--reverse', '--check').returncode, 0)
        target = self.work / 'src/model_loader.cpp'
        data = target.read_text()
        marker = '// Wan experts may use bare, prefixed, or Diffusers tensor names.'
        self.assertEqual(data.count(marker), 1)
        target.write_text(data.replace(marker, '// Modified Wan expert patch.'))
        self.assertNotEqual(self.apply('native-wan-expert-namespace', '--reverse', '--check', '-C0').returncode, 0)


if __name__ == '__main__':
    unittest.main()
