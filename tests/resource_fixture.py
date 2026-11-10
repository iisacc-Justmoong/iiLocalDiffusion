"""Reserve isolated test fixtures through iiFileProvider, outside build/."""
import os
from pathlib import Path
import resource_runner

ROOT = Path(__file__).resolve().parents[1]


def TemporaryDirectory(suffix="", prefix="tmp", dir=None, ignore_cleanup_errors=False):
    if any(separator in part for part in (prefix, suffix) for separator in ("/", "\\")):
        raise ValueError("Fixture prefix and suffix cannot contain path separators")
    provider = resource_runner._provider()
    build = ROOT / "build"
    requested = Path(dir).resolve() if dir is not None else None
    if requested is None or requested.is_relative_to(build.resolve()):
        resources = ROOT / "resources"
        state = resources / "test-workspaces"
        if resources.is_symlink() or state.is_symlink():
            raise ValueError("Test resource directories cannot be symlinks")
        provider.mkdir(state, parents=True, exist_ok=True)
        inherited = os.environ.get("IIFILEPROVIDER_TEST_DIRECTORY")
        if inherited:
            directory = Path(inherited)
            if directory.is_symlink() or not directory.resolve().is_relative_to(state.resolve()):
                raise ValueError("Test workspace must be inside project resources")
            if not directory.is_dir():
                raise ValueError("Test workspace does not exist")
        else:
            directory = state
        dir = directory
    return provider.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=dir,
                                       ignore_cleanup_errors=ignore_cleanup_errors)
