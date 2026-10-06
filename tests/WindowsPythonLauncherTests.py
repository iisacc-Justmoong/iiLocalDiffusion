"""Exercise native launcher quoting, redirected output, and child exit status."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import ctypes
import time

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(sys.argv.pop(1))

class WindowsPythonLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native launcher 한글 ", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.executable = self.directory / "iild-merge.exe"
        shutil.copy2(LAUNCHER, self.executable)
        self.script = self.directory / "iild-merge.py"
        self.script.write_text("import json, sys; print(json.dumps(sys.argv[1:], ensure_ascii=False)); sys.exit(7)\n", encoding="utf-8")
        self.env = dict(os.environ, IILD_PYTHON_EXECUTABLE=sys.executable, PYTHONUTF8="1")

    def test_unicode_quotes_and_trailing_backslashes_round_trip(self):
        arguments = ['모델 입력', 'quoted "tensor"', 'C:\\a path\\', '']
        result = subprocess.run([self.executable, *arguments], env=self.env, capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(json.loads(result.stdout), arguments)

    def test_missing_interpreter_returns_actionable_failure(self):
        self.env['IILD_PYTHON_EXECUTABLE'] = str(self.directory / 'missing.exe')
        result = subprocess.run([self.executable], env=self.env, capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 127)
        self.assertIn('IILD_PYTHON_EXECUTABLE', result.stderr)

    def test_terminating_launcher_also_stops_its_python_children(self):
        marker = self.directory / 'child.pid'
        self.script.write_text("import subprocess, sys, time\nfrom pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"Path({str(marker)!r}).write_text(str(child.pid))\ntime.sleep(60)\n", encoding='utf-8')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        process = subprocess.Popen([self.executable], env=self.env)
        handle = None
        try:
            deadline = time.monotonic() + 15
            while not marker.is_file() and process.poll() is None and time.monotonic() < deadline: time.sleep(.05)
            self.assertTrue(marker.is_file(), 'Python child did not start')
            handle = kernel.OpenProcess(0x100000, False, int(marker.read_text()))
            self.assertTrue(handle)
            process.terminate(); process.wait(timeout=5)
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0, 'Launcher left its Python child alive')
        finally:
            if process.poll() is None: process.kill(); process.wait()
            if handle: kernel.CloseHandle(handle)

if __name__ == '__main__': unittest.main()
