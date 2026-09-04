#!/usr/bin/env python3
"""Pinned runtime installation preserves existing sources and Python environments."""

import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_comfyui", ROOT / "reference/setup_comfyui.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupComfyUITests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_foreign_or_modified_checkout_is_never_reset(self):
        for values in (("wrong", setup.COMFY_COMMIT), (setup.COMFY_URL, "wrong"),
                       (setup.COMFY_URL, setup.COMFY_COMMIT, "main.py")):
            with self.subTest(values=values), patch.object(setup.subprocess, "check_output", side_effect=values), patch.object(setup.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    setup.checkout(self.directory, setup.COMFY_URL, setup.COMFY_COMMIT)
                run.assert_not_called()

    def test_fresh_checkout_fetches_immutable_revision(self):
        with patch.object(setup.subprocess, "run") as run:
            setup.checkout(self.directory / "source", setup.COMFY_URL, setup.COMFY_COMMIT)
        self.assertIn(setup.COMFY_COMMIT, run.call_args_list[2].args[0])
        self.assertIn("--detach", run.call_args_list[3].args[0])

    def test_failed_fetch_leaves_no_partial_checkout_and_can_be_retried(self):
        destination = self.directory / "source"
        with patch.object(setup.subprocess, "run", side_effect=[None, None, subprocess.CalledProcessError(1, "git fetch")]):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.checkout(destination, setup.COMFY_URL, setup.COMFY_COMMIT)
        self.assertFalse(destination.exists())
        with patch.object(setup.subprocess, "run"):
            setup.checkout(destination, setup.COMFY_URL, setup.COMFY_COMMIT)
        self.assertTrue(destination.exists())

    def test_runtime_and_cache_stay_under_build_and_freeze_is_saved(self):
        with patch.object(setup, "ROOT", self.directory), patch.object(setup, "checkout"), patch.object(setup.shutil, "which", return_value="uv"), patch.object(setup.subprocess, "run") as run, patch.object(setup.subprocess, "check_output", return_value="torch==2.13.0\n"):
            (self.directory / "build/reference").mkdir(parents=True)
            result = setup.install(sys.executable)
        self.assertTrue(Path(result["python"]).is_relative_to(self.directory / "build"))
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[1], "--cache-dir")
            self.assertIn(str(self.directory / "build/reference/uv-cache"), command)
        self.assertEqual((self.directory / "build/reference/comfyui-installed.lock").read_text(), "torch==2.13.0\n")


if __name__ == "__main__":
    unittest.main()
