"""Contract tests use unique, disposable resource workspaces."""
import importlib.util
import errno
import os
from pathlib import Path
import subprocess
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import ii_file_provider as provider

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resource_runner", ROOT / "tests/resource_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class WorkspaceRunnerTest(unittest.TestCase):
    def test_owned_cleanup_retries_only_transient_nonempty_directories(self):
        nonempty = OSError(errno.ENOTEMPTY, "preview still being written")
        with patch.object(runner.shutil, "rmtree", side_effect=[nonempty, None]) as remove, \
             patch.object(runner.time, "sleep"):
            runner._remove_owned_tree(Path("owned"))
        self.assertEqual(remove.call_count, 2)
        with patch.object(runner.shutil, "rmtree", side_effect=OSError(errno.EACCES, "denied")) as remove:
            with self.assertRaises(PermissionError):
                runner._remove_owned_tree(Path("owned"))
        self.assertEqual(remove.call_count, 1)
        with patch.object(runner.shutil, "rmtree", side_effect=nonempty) as remove, \
             patch.object(runner.time, "sleep"):
            with self.assertRaises(OSError):
                runner._remove_owned_tree(Path("owned"))
        self.assertEqual(remove.call_count, 3)
    def test_direct_installer_fixture_uses_resource_runner(self):
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix="installer-route-") as temporary:
            owner = Path(temporary)
            fake_python = owner / "python"
            arguments = owner / "arguments.txt"
            provider.write_text(fake_python, '#!/bin/sh\nprintf "%s\\n" "$@" > "$RECORDED_ARGS"\n')
            fake_python.chmod(0o755)
            environment = dict(os.environ, IILD_PYTHON_EXECUTABLE=str(fake_python),
                               RECORDED_ARGS=str(arguments), TMPDIR=str(owner))
            environment.pop("IILD_INSTALL_SCRIPT_TEST_DIRECTORY", None)
            result = subprocess.run(["bash", str(ROOT / "tests/InstallScriptTest.sh")],
                                    env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = arguments.read_text().splitlines()
            self.assertIn(str(ROOT / "tests/resource_runner.py"), args)
            self.assertIn("IILD_INSTALL_SCRIPT_TEST_DIRECTORY=@WORK@/install-script-test", args)

    def test_installer_fixture_rejects_build_directory_before_running(self):
        environment = dict(os.environ, IILD_PYTHON_EXECUTABLE=sys.executable,
                           IILD_INSTALL_SCRIPT_TEST_DIRECTORY=str(ROOT / "build"))
        result = subprocess.run(["bash", str(ROOT / "tests/InstallScriptTest.sh")],
                                env=environment, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be under project resources", result.stderr)

    def test_installer_fixture_preserves_an_existing_nonempty_workspace(self):
        import resource_fixture
        with resource_fixture.TemporaryDirectory(prefix="installer-existing-") as temporary:
            owner = Path(temporary)
            provider.write_text(owner / "user.txt", "keep")
            environment = dict(os.environ, IILD_PYTHON_EXECUTABLE=sys.executable,
                               IILD_INSTALL_SCRIPT_TEST_DIRECTORY=str(owner))
            result = subprocess.run(["bash", str(ROOT / "tests/InstallScriptTest.sh")],
                                    env=environment, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((owner / "user.txt").read_text(), "keep")
            self.assertFalse((owner / "bin").exists())

    def test_compiled_fixture_build_and_resource_workspace_are_owned_and_cleaned(self):
        seen = []
        def execute(args, **kwargs):
            resource = Path(kwargs["cwd"])
            compiled = Path(args[2])
            self.assertEqual(Path(args[1]), resource)
            self.assertTrue(resource.is_relative_to(ROOT / "resources/test-workspaces"))
            self.assertTrue(compiled.is_relative_to(ROOT / "build/test-workspaces"))
            provider.write_text(resource / "input.txt", "fixture")
            provider.write_text(compiled / "output.o", "fixture")
            seen.append((resource, compiled))
            return subprocess.CompletedProcess(args, 7)
        self.assertEqual(runner.run(ROOT / "resources", ["test", "@WORK@", "@BUILD_WORK@"], execute=execute), 7)
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0][0].exists())
        self.assertFalse(seen[0][1].exists())

    def test_compiled_workspace_collision_preserves_existing_files(self):
        token = runner.uuid.uuid4().hex[:4]
        while token in runner._issued or (ROOT / "resources/test-workspaces" / token).exists():
            token = runner.uuid.uuid4().hex[:4]
        previous = ROOT / "build/test-workspaces" / token
        provider.mkdir(previous, parents=True, exist_ok=False)
        provider.write_text(previous / "user.txt", "keep")
        try:
            with patch.object(runner.uuid, "uuid4", return_value=SimpleNamespace(hex=token)):
                with self.assertRaises(FileExistsError):
                    runner.run(ROOT / "resources", ["test", "@BUILD_WORK@"],
                               execute=lambda *args, **kwargs: self.fail("Must not run"))
            self.assertEqual((previous / "user.txt").read_text(), "keep")
        finally:
            shutil.rmtree(previous)
            try:
                previous.parent.rmdir()
            except OSError:
                pass

    def test_compiled_workspace_is_removed_when_command_raises(self):
        seen = []
        def execute(args, **kwargs):
            seen.append((Path(kwargs["cwd"]), Path(args[1])))
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            runner.run(ROOT / "resources", ["test", "@BUILD_WORK@"], execute=execute)
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0][0].exists())
        self.assertFalse(seen[0][1].exists())

    def test_workspace_context_removes_resources_after_success_or_failure(self):
        resources = ROOT / "resources"
        with runner.workspace(resources) as first:
            self.assertTrue(first.is_dir())
            self.assertTrue(first.is_relative_to(resources / "test-workspaces"))
            provider.write_text(first / "fixture.txt", "temporary")
        self.assertFalse(first.exists())

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with runner.workspace(resources) as second:
                self.assertNotEqual(first, second)
                provider.write_text(second / "fixture.txt", "temporary")
                raise RuntimeError("intentional")
        self.assertFalse(second.exists())

    def test_legacy_build_temporary_directory_uses_resources_and_cleans_up(self):
        import resource_fixture
        for inherited in (True, False):
            with patch.dict(os.environ):
                if not inherited:
                    os.environ.pop("IIFILEPROVIDER_TEST_DIRECTORY", None)
                for requested in (None, ROOT / "build"):
                    with resource_fixture.TemporaryDirectory(prefix="migration-", dir=requested) as path:
                        owner = Path(path)
                        self.assertTrue(owner.is_relative_to(ROOT / "resources/test-workspaces"))
                        provider.write_text(owner / "input", "temporary")
                    self.assertFalse(owner.exists())

    def test_test_workspace_environment_cannot_redirect_outside_resources(self):
        import resource_fixture
        with patch.dict("os.environ", {"IIFILEPROVIDER_TEST_DIRECTORY": str(ROOT / "build")}):
            with self.assertRaises(ValueError):
                resource_fixture.TemporaryDirectory(dir=ROOT / "build")
        for options in ({"prefix": "../escape-"}, {"suffix": "/escape"}):
            with self.assertRaises(ValueError):
                resource_fixture.TemporaryDirectory(**options)

    def test_manual_smoke_commands_use_and_clean_resource_workspaces(self):
        test_dir = str(ROOT / "tests")
        sys.path.insert(0, test_dir)
        try:
            for name in ("ModelMergeDiffusersSmoke", "ModelMergeLoraSmoke"):
                with self.subTest(command=name):
                    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
                    smoke = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(smoke)
                    seen = []

                    def verify(args, work):
                        self.assertTrue(work.is_relative_to(ROOT / "resources/test-workspaces"))
                        provider.write_text(work / "fixture.txt", "temporary")
                        seen.append(work)

                    with patch.object(smoke, "run_smoke", side_effect=verify), \
                         patch.object(sys, "argv", [name]):
                        smoke.main()
                    self.assertEqual(len(seen), 1)
                    self.assertFalse(seen[0].exists())

                    def fail(args, work):
                        seen.append(work)
                        provider.write_text(work / "fixture.txt", "temporary")
                        raise RuntimeError("intentional")

                    with patch.object(smoke, "run_smoke", side_effect=fail), \
                         patch.object(sys, "argv", [name]), \
                         self.assertRaisesRegex(RuntimeError, "intentional"):
                        smoke.main()
                    self.assertFalse(seen[-1].exists())
        finally:
            sys.path.remove(test_dir)

    def test_manual_checkpoint_verifiers_clean_their_resource_workspace(self):
        test_dir = str(ROOT / "tests")
        sys.path.insert(0, test_dir)
        try:
            commands = (
                ("verify_model_merge_samples", ["base.safetensors", "material.safetensors"]),
                ("verify_model_merge_nonfinite", ["--base", "base.safetensors", "--material", "material.safetensors"]),
            )
            for name, arguments in commands:
                with self.subTest(command=name):
                    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
                    verifier = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(verifier)
                    seen = []
                    def verify(args, work):
                        self.assertTrue(work.is_relative_to(ROOT / "resources/test-workspaces"))
                        provider.write_text(work / "sample", "temporary")
                        seen.append(work)
                    with patch.object(verifier, "run_verification", side_effect=verify), \
                         patch.object(sys, "argv", [name, *arguments]):
                        verifier.main()
                    self.assertEqual(len(seen), 1)
                    self.assertFalse(seen[0].exists())
        finally:
            sys.path.remove(test_dir)

    def test_symlink_resource_root_is_rejected(self):
        state = ROOT / "resources/test-workspaces"
        provider.mkdir(state, parents=True, exist_ok=True)
        owner = state / runner.uuid.uuid4().hex
        provider.mkdir(owner, exist_ok=False)
        link = owner / "resources"
        try:
            try:
                link.symlink_to(ROOT / "resources", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Cannot create a symbolic link on this platform: {error}")
            with self.assertRaises(ValueError):
                runner.run(link, ["test"])
            self.assertTrue(link.is_symlink())
        finally:
            shutil.rmtree(owner)

    def test_unique_workspaces_are_confined_and_removed(self):
        seen = []
        def execute(args, **kwargs):
            path = Path(kwargs["cwd"])
            self.assertTrue(path.is_relative_to(ROOT / "resources/test-workspaces"))
            self.assertFalse(any(part.startswith(".") for part in path.parts if part not in (".", "..")))
            self.assertEqual(args[-1], str(path / "input"))
            self.assertEqual(kwargs["env"]["TMPDIR"], str(path))
            self.assertEqual(kwargs["env"]["IIFILEPROVIDER_TEST_DIRECTORY"], str(path))
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            seen.append(path)
            return subprocess.CompletedProcess(args, 0)
        for _ in range(2):
            self.assertEqual(runner.run(ROOT / "resources", ["test", "@WORK@/input"], execute=execute), 0)
            self.assertFalse(seen[-1].exists())
        self.assertNotEqual(seen[0], seen[1])

    def test_failure_cleans_workspace_and_preserves_status(self):
        seen = []
        def execute(args, **kwargs):
            seen.append(Path(kwargs["cwd"]))
            return subprocess.CompletedProcess(args, 7)
        self.assertEqual(runner.run(ROOT / "resources", ["test"], execute=execute), 7)
        self.assertFalse(seen[0].exists())

    def test_exception_also_cleans_workspace(self):
        seen = []
        def execute(args, **kwargs):
            seen.append(Path(kwargs["cwd"]))
            raise RuntimeError("expected test failure")
        with self.assertRaises(RuntimeError):
            runner.run(ROOT / "resources", ["test"], execute=execute)
        self.assertFalse(seen[0].exists())

    def test_collision_preserves_the_existing_workspace(self):
        state = ROOT / "resources/test-workspaces"
        provider.mkdir(state, parents=True, exist_ok=True)
        token = runner.uuid.uuid4().hex[:4]
        while token in runner._issued or (state / token).exists():
            token = runner.uuid.uuid4().hex[:4]
        other = runner.uuid.uuid4().hex[:4]
        while other == token or other in runner._issued or (state / other).exists():
            other = runner.uuid.uuid4().hex[:4]
        blocked = state / token
        provider.mkdir(blocked, exist_ok=False)
        def execute(args, **kwargs):
            self.assertEqual(Path(kwargs["cwd"]).name, other)
            self.assertTrue(blocked.is_dir())
            return subprocess.CompletedProcess(args, 0)
        try:
            with patch.object(runner.uuid, "uuid4", side_effect=[SimpleNamespace(hex=token), SimpleNamespace(hex=other)]):
                self.assertEqual(runner.run(ROOT / "resources", ["test"], execute=execute), 0)
            self.assertTrue(blocked.is_dir())
            self.assertFalse((state / other).exists())
        finally:
            blocked.rmdir()

if __name__ == "__main__":
    unittest.main()
