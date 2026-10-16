"""Locate the common provider in source checkouts and installed SDK packages."""
from pathlib import Path
import os
import sys

for _parent in Path(__file__).resolve().parents:
    for _root in (_parent / "iiFileProvider" / "python", _parent / "python",
                  _parent / "share" / "iiFileProvider" / "python"):
        if (_root / "ii_file_provider" / "__init__.py").is_file():
            sys.path.insert(0, str(_root))
_installed = Path(os.environ.get("IIFILEPROVIDER_PREFIX", Path.home() / ".local/SDK/iiFileProvider"))
sys.path.insert(0, str(_installed / "share/iiFileProvider/python"))
_adjacent = Path(__file__).resolve().parent
if (_adjacent / "ii_file_provider" / "__init__.py").is_file():
    # A distributed runtime carries the provider adapter beside its workers.
    sys.path.insert(0, str(_adjacent))
import ii_file_provider as provider
