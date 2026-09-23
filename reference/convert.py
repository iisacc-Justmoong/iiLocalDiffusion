#!/usr/bin/env python3
"""Convert local diffusion checkpoints to the SDK DiT merge standard."""

from pathlib import Path
import sys

REFERENCE = Path(__file__).resolve().parent / "diffusers"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from dit_conversion import main


if __name__ == "__main__":
    raise SystemExit(main())
