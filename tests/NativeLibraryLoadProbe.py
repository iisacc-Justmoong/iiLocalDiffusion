"""Load an installed native worker library without CMake consumer RPATHs."""
import argparse
import ctypes
from pathlib import Path
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("library", type=Path)
args = parser.parse_args()
library_path = args.library.resolve(strict=True)
library = ctypes.CDLL(str(library_path))
for symbol in ("iild_native_video_generate_v1", "iild_native_video_metadata_v1",
               "iild_native_video_free_v1"):
    getattr(library, symbol)

if sys.platform == "darwin":
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    images = [Path(dyld._dyld_get_image_name(index).decode())
              for index in range(dyld._dyld_image_count())]
    if any("onnx" in path.name.lower() for path in images):
        raise RuntimeError("Retired ONNX runtime entered the native video dependency closure")

print("Installed native video library loaded successfully")
