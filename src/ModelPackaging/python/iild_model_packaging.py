"""Python access to the native ModelPackaging CLI, without a tensor runtime."""

import json
import os
import shutil
import subprocess


class ModelPackagingError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def _run(operation, root, selected, output, excluded, executable, *, repair_nonfinite=False, assignments=None,
         asset_role=None, asset_path=None, tensor_names=()):
    paths = [os.fspath(path) for path in selected]
    if not paths and operation in ("scan", "create"):
        raise ValueError("At least one model must be selected")
    program = os.fspath(executable) if executable is not None else shutil.which("iild-model-package")
    if not program:
        raise FileNotFoundError("iild-model-package is not on PATH")
    command = [program, operation, "--input", os.fspath(root)]
    for path in paths:
        command.extend(("--select", path))
    for path in excluded:
        command.extend(("--exclude", os.fspath(path)))
    for name in tensor_names:
        command.extend(("--tensor", name))
    for path, choice in (assignments or {}).items():
        if not isinstance(choice, dict) or any(key not in ("role", "position") for key in choice):
            raise ValueError("Assignments must map each path to role and/or position")
        for field in ("role", "position"):
            if choice.get(field):
                command.extend(("--assign-" + field, os.fspath(path) + "=" + choice[field]))
    if output is not None:
        command.extend(("--output", os.fspath(output)))
    if repair_nonfinite:
        command.extend(("--repair-nonfinite", "true"))
    if asset_role is not None:
        command.extend(("--role", asset_role))
    if asset_path is not None:
        command.extend(("--asset-path", os.fspath(asset_path)))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    except (ValueError, TypeError) as error:
        raise ModelPackagingError("The native packaging tool returned invalid JSON") from error
    report = records[-1] if records else None
    if result.returncode != 0 or not isinstance(report, dict):
        message = report.get("error") if isinstance(report, dict) else result.stderr.strip()
        raise ModelPackagingError(message or "The native packaging tool failed", report)
    return report


def scan_selection(root, selected, *, excluded=(), assignments=None, executable=None):
    """Inspect explicit root-relative model paths and return the native report."""
    return _run("scan", root, selected, None, excluded, executable, assignments=assignments)


def create_selection(root, selected, output, *, excluded=(), assignments=None, executable=None):
    """Create and verify a package from explicit root-relative model paths."""
    return _run("create", root, selected, output, excluded, executable, assignments=assignments)


def calibrate_package(model, output, *, repair_nonfinite=False, executable=None):
    """Rebuild a verified package; optional numeric repair changes weight values."""
    return _run("calibrate", model, (), output, (), executable, repair_nonfinite=repair_nonfinite)


def materialize_package_asset(model, cache, *, role=None, relative_source=None, executable=None):
    """Verify and reconstruct one role or one exact original package path."""
    if role is not None and relative_source is not None:
        raise ValueError("Choose either a role or an original relative source path")
    return _run("materialize", model, (), cache, (), executable, asset_role=role, asset_path=relative_source)


def detect_container(model, *, executable=None):
    """Probe a container prefix; this does not validate or execute weights."""
    return _run("probe", model, (), None, (), executable)


def inspect_weights(model, *, executable=None):
    """Count non-finite floating point weights without modifying the model."""
    return _run("weight-health", model, (), None, (), executable)


def inspect_sharded_index(model, *, executable=None):
    """Validate an ordinary Safetensors index and its exact shard mappings."""
    return _run("inspect-index", model, (), None, (), executable)


def inspect_pytorch_index(model, *, executable=None):
    """Inspect a PyTorch index and shard identities without loading pickle."""
    return _run("inspect-pytorch-index", model, (), None, (), executable)


def inspect_wan_directory(directory, *, executable=None):
    """Resolve unambiguous local Wan text-to-video components without loading weights."""
    return _run("inspect-wan-directory", directory, (), None, (), executable)


def contains_tensors(model, names, *, executable=None):
    """Check exact tensor names without reading weights or claiming compatibility."""
    if isinstance(names, (str, bytes)):
        raise ValueError("Tensor names must be an iterable of names")
    names = tuple(names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Supply one or more nonempty tensor names")
    return _run("contains-tensors", model, (), None, (), executable, tensor_names=names)


def find_tensor_names(model, names, *, executable=None):
    """Return the requested tensor names found in a validated model header."""
    return contains_tensors(model, names, executable=executable)['matches']


def repair_standalone_weights(model, output, *, executable=None):
    """Create a separate standalone file with non-finite values replaced by zero."""
    return _run("calibrate-file", model, (), output, (), executable, repair_nonfinite=True)
