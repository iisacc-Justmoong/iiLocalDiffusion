"""Reuse the native SDK's Canny implementation for a Diffusers control hint."""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
import os
from pathlib import Path
import sys


class NativeImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("channel", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_uint8))]


def library_path(explicit=None):
    override = explicit or os.environ.get("IILD_NATIVE_LIBRARY")
    if override:
        candidates = [Path(override).expanduser()]
    else:
        filename = "iiLocalDiffusion.dll" if sys.platform == "win32" else (
            "libiiLocalDiffusion.dylib" if sys.platform == "darwin" else "libiiLocalDiffusion.so")
        root = Path(__file__).resolve().parents[2]
        candidates = [root / "build" / filename]
        # The installed reference and its native library belong to the same
        # prefix, which may be different from the user's ~/.local SDK.
        if root.name == "iiLocalDiffusion" and root.parent.name == "share":
            prefix = root.parent.parent
            folders = ("bin",) if sys.platform == "win32" else ("lib", "lib64")
            candidates.extend(prefix / folder / filename for folder in folders)
        candidates.append(Path.home() / ".local" / ("bin" if sys.platform == "win32" else "lib") / filename)
        if sys.platform == "win32":
            candidates = [variant for path in candidates for variant in (path, path.with_name("libiiLocalDiffusion.dll"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Native Canny needs the iiLocalDiffusion shared library; set --control-preprocessor-library or IILD_NATIVE_LIBRARY.")


def load_library(path):
    path = Path(path).resolve()
    with ExitStack() as directories:
        if sys.platform == "win32":
            seen = set()
            search = [path.parent, *(Path(part) for part in os.environ.get("PATH", "").split(os.pathsep) if part)]
            for directory in search:
                directory = directory.resolve()
                key = os.path.normcase(str(directory))
                if key not in seen and directory.is_dir():
                    seen.add(key)
                    directories.enter_context(os.add_dll_directory(str(directory)))
        library = ctypes.CDLL(str(path))
    try:
        function = library.preprocess_canny
    except AttributeError as error:
        raise ValueError("The selected iiLocalDiffusion library does not expose the native Canny backend.") from error
    function.argtypes = [NativeImage, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
    function.restype = ctypes.c_bool
    return library


def apply_canny(image, image_module, library):
    if image.mode != "RGB" or image.size[0] < 1 or image.size[1] < 1 or max(image.size) > 2048:
        raise ValueError("Native Canny requires an RGB image with each dimension in 1..2048.")
    pixels = image.tobytes()
    if len(pixels) != image.size[0] * image.size[1] * 3:
        raise ValueError("Native Canny received an invalid RGB image buffer.")
    owned = (ctypes.c_uint8 * len(pixels)).from_buffer_copy(pixels)
    native = NativeImage(image.size[0], image.size[1], 3, owned)
    if not library.preprocess_canny(native, .08, .08, .8, 1., False):
        raise RuntimeError("Native Canny preprocessing failed.")
    return image_module.frombytes("RGB", image.size, bytes(owned))
