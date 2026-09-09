#!/usr/bin/env python3
"""Merge local checkpoints through the SDK's Python runtime."""

from pathlib import Path
import sys

REFERENCE = Path(__file__).resolve().parent / "diffusers"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from model_merge import main


if __name__ == "__main__":
    raise SystemExit(main())
