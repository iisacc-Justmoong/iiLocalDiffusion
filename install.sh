#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_directory="${project_root}/build"
install_prefix="${IILD_INSTALL_PREFIX:-${HOME}/.local/SDK/iiLocalDiffusion}"
build_jobs="${IILD_BUILD_JOBS:-4}"

# Reuse the configured native backends and SDK paths; only the build mode changes.
cmake -S "${project_root}" -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
    -DBUILD_TESTING=ON -DIILD_BUILD_TOOLS=ON -DIILD_INSTALL_PYTHON_REFERENCE=ON "$@"
cmake --build "${build_directory}" --config Release --parallel "${build_jobs}"
ctest --test-dir "${build_directory}" --build-config Release --output-on-failure --parallel "${build_jobs}"
cmake --install "${build_directory}" --config Release --prefix "${install_prefix}"

# A local development installation shares its existing environments and model
# caches. Never copy virtual environments or replace a caller-owned directory.
python3 - "${project_root}" "${install_prefix}" <<'PY'
from pathlib import Path
import sys

source, prefix = (Path(value).expanduser().resolve() for value in sys.argv[1:])
installed = prefix / "share/iiLocalDiffusion"
for original, link in (
    (source / "build", installed / "build"),
    (source / "reference/diffusers/.venv", installed / "reference/diffusers/.venv"),
):
    if not original.is_dir():
        continue
    if link.is_symlink() and link.resolve() == original:
        continue
    if link.exists() or link.is_symlink():
        sys.exit(f"Cannot link the existing runtime: destination already exists: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(original, target_is_directory=True)
    print(f"Runtime: {link} -> {original}")
PY

"${install_prefix}/bin/iild-run" --help
"${install_prefix}/bin/iild-generate" --backend preset --hires-fix --print-config
printf 'Installed iiLocalDiffusion: %s\n' "${install_prefix}"
