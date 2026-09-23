"""Installed launcher selection without model loading or merge execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PythonLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="launcher-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "reference").mkdir()
        (self.root / "reference/merge.py").write_text("import sys; print('ENTRY', sys.executable)\n")
        self.launcher = self.root / "iild-merge"
        self.launcher.write_text((ROOT / "cmake/iild-python.py.in").read_text()
            .replace("@IILD_REFERENCE_FROM_BINDIR@", "reference").replace("@IILD_PYTHON_ENTRY@", "merge.py"))
        self.env = dict(os.environ)
        self.env.pop("IILD_PYTHON_EXECUTABLE", None)

    def run_launcher(self):
        return subprocess.run([sys.executable, str(self.launcher)], env=self.env, text=True, capture_output=True)

    def test_configured_environment_is_used_without_user_override(self):
        (self.root / "reference/runtime-python.json").write_text(json.dumps({"python": sys.executable}))
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ENTRY", result.stdout)

    def test_explicit_override_has_priority(self):
        (self.root / "reference/runtime-python.json").write_text('{"python":"/missing/python"}')
        self.env["IILD_PYTHON_EXECUTABLE"] = sys.executable
        self.assertEqual(self.run_launcher().returncode, 0)

    def test_old_interpreter_fails_before_entry_import(self):
        old = self.root / "old-python"
        old.write_text("#!/bin/sh\nexit 1\n")
        old.chmod(0o755)
        self.env["IILD_PYTHON_EXECUTABLE"] = str(old)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.10 or newer", result.stderr)
        self.assertNotIn("ENTRY", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_broken_configuration_is_actionable(self):
        (self.root / "reference/runtime-python.json").write_text('{"python":"/missing/python"}')
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot start SDK Python", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

if __name__ == "__main__":
    unittest.main()
