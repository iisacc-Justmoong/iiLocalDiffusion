"""Run one contract test in a disposable project resource workspace."""
from contextlib import contextmanager, ExitStack
import errno
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import uuid

_issued = set()


def _remove_owned_tree(path):
    for attempt in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno != errno.ENOTEMPTY or attempt == 2:
                raise
            time.sleep(0.05)


def _provider():
    try:
        import ii_file_provider as provider
    except ModuleNotFoundError as error:
        if error.name != "ii_file_provider":
            raise
        provider_dir = os.environ.get("IIFILEPROVIDER_PYTHON_DIRECTORY") or str(
            Path.home() / ".local/SDK/iiFileProvider/share/iiFileProvider/python")
        sys.path.insert(0, provider_dir)
        import ii_file_provider as provider
    return provider


@contextmanager
def workspace(resources):
    provider = _provider()
    resources = Path(resources)
    if resources.is_symlink():
        raise ValueError("Resource root cannot be a symlink")
    resources = resources.resolve(strict=True)
    # Short, isolated names avoid collisions with concurrently running tests.
    # Presentation deliberately ignores private path components, so test
    # workspaces must be visible for normal metadata/preview behavior.
    state = resources / "test-workspaces"
    if state.is_symlink():
        raise ValueError("Test state cannot be a symlink")
    provider.mkdir(state, parents=True, exist_ok=True)
    for _ in range(32):
        token = uuid.uuid4().hex[:4]
        if token in _issued:
            continue
        owner = state / token
        try:
            provider.mkdir(owner, exist_ok=False)
            _issued.add(token)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Cannot reserve a test workspace")
    try:
        yield owner
    finally:
        _remove_owned_tree(owner)


@contextmanager
def compiled_workspace(resources, owner):
    provider = _provider()
    build = Path(resources).resolve().parent / "build"
    if build.is_symlink() or not build.is_dir():
        raise ValueError("Compiled test fixtures require the project build directory")
    parent = build / "test-workspaces"
    if parent.is_symlink():
        raise ValueError("Compiled test workspace parent cannot be a symlink")
    provider.mkdir(parent, exist_ok=True)
    directory = parent / owner.name
    provider.mkdir(directory, exist_ok=False)
    try:
        yield directory
    finally:
        _remove_owned_tree(directory)
        try:
            parent.rmdir()
        except OSError:
            pass


def run(resources, command, *, execute=subprocess.run):
    with workspace(resources) as owner, ExitStack() as stack:
        compiled = None
        if any("@BUILD_WORK@" in part for part in command):
            compiled = stack.enter_context(compiled_workspace(resources, owner))
        expanded = [part.replace("@WORK@", str(owner)).replace("@BUILD_WORK@", str(compiled))
                    for part in command]
        env = dict(os.environ, TMPDIR=str(owner), TMP=str(owner), TEMP=str(owner),
                   IIFILEPROVIDER_TEST_DIRECTORY=str(owner),
                   PYTHONDONTWRITEBYTECODE="1")
        return execute(expanded, cwd=owner, env=env, check=False).returncode


def main(argv):
    if len(argv) < 3:
        raise ValueError("Expected provider directory, resources and command")
    provider_dir = argv[0]
    if provider_dir in ("", "auto"):
        provider_dir = os.environ.get("IIFILEPROVIDER_PYTHON_DIRECTORY") or str(
            Path.home() / ".local/SDK/iiFileProvider/share/iiFileProvider/python")
    sys.path.insert(0, provider_dir)
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = provider_dir + (os.pathsep + inherited if inherited else "")
    return run(argv[1], argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
