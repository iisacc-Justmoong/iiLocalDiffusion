#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${source_root}/build/install-script-test"
mkdir -p "${work}/bin" "${work}/home/.local/SDK/iiPaintEngine/lib/cmake/iiPaintEngine"
touch "${work}/home/.local/SDK/iiPaintEngine/lib/cmake/iiPaintEngine/iiPaintEngineConfig.cmake"
cat > "${work}/bin/cmake" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${INSTALL_TEST_ARGUMENTS}"
[[ "${INSTALL_TEST_SMOKE:-0}" == 1 ]] && exit 0
exit 91
MOCK
chmod +x "${work}/bin/cmake"
for mode in default override; do
    expected="${work}/home/.local/SDK/iiLocalDiffusion"
    options=("PATH=${work}/bin:${PATH}")
    if [[ "${mode}" == override ]]; then
        expected="${work}/custom prefix"
        options+=("IILD_INSTALL_PREFIX=${expected}")
    fi
    set +e
    env -u IILD_INSTALL_PREFIX -u IISHAREDCANVAS_IIPAINTENGINE_PREFIX \
        HOME="${work}/home" PATH="${work}/bin:${PATH}" \
        INSTALL_TEST_ARGUMENTS="${work}/arguments" "${options[@]}" \
        bash "${source_root}/install.sh" > "${work}/output" 2>&1
    result=$?
    set -e
    [[ ${result} == 91 ]] || { cat "${work}/output"; exit 1; }
    grep -Fx -- "-DCMAKE_INSTALL_PREFIX=${expected}" "${work}/arguments"
    grep -Fx -- "${source_root}/build" "${work}/arguments"
done

# Exercise the final installer commands with the real relocated Python launcher.
# Native compilation is stubbed here; the normal CTest suite covers the library.
smoke_prefix="${work}/smoke prefix"
mkdir -p "${smoke_prefix}/bin"
cat > "${work}/bin/ctest" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK
cat > "${smoke_prefix}/bin/iild-run" <<'MOCK'
#!/usr/bin/env bash
[[ "$#" == 1 && "$1" == --help ]]
MOCK
chmod +x "${work}/bin/ctest" "${smoke_prefix}/bin/iild-run"
python3 - "${source_root}" "${smoke_prefix}" <<'PY'
from pathlib import Path
import os
import shutil
import sys

source, prefix = map(Path, sys.argv[1:])
reference = prefix / "share/iiLocalDiffusion/reference"
reference.mkdir(parents=True, exist_ok=True)
for entry in ("generate.py", "merge.py"):
    shutil.copy2(source / "reference" / entry, reference / entry)
diffusers = source / "reference/diffusers"
for directory, children, files in os.walk(diffusers):
    children[:] = [name for name in children if name not in {".venv", "__pycache__", ".git"}]
    for name in files:
        original = Path(directory) / name
        if original.suffix not in {".py", ".json"}:
            continue
        relative = original.relative_to(diffusers)
        destination = reference / "diffusers" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
for command in ("generate", "merge"):
    launcher = prefix / f"bin/iild-{command}"
    launcher.write_text((source / "cmake/iild-python.py.in").read_text().replace(
        "@IILD_REFERENCE_FROM_BINDIR@", "../share/iiLocalDiffusion/reference").replace(
        "@IILD_PYTHON_ENTRY@", f"{command}.py"))
    launcher.chmod(0o755)
PY
env PATH="${work}/bin:${PATH}" IILD_INSTALL_PREFIX="${smoke_prefix}" \
    INSTALL_TEST_ARGUMENTS="${work}/arguments" INSTALL_TEST_SMOKE=1 \
    bash "${source_root}/install.sh" > "${work}/smoke-output" 2>&1 || {
        cat "${work}/smoke-output"
        exit 1
    }
