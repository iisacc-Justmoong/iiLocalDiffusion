"""Default locations for reference outputs and model download caches."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = REPOSITORY_ROOT / "resources"
OUTPUT_DIRECTORY = RESOURCE_ROOT / "outputs" / "reference"
CACHE_DIRECTORY = RESOURCE_ROOT / ".state" / "reference-cache"
RUNTIME_DIRECTORY = RESOURCE_ROOT / ".state" / "reference-runtime"
DEPENDENCY_DIRECTORY = RESOURCE_ROOT / "dependencies"
