#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${IILD_INSTALL_SCRIPT_TEST_DIRECTORY:-}" ]]; then
    exec "${IILD_PYTHON_EXECUTABLE:-python3}" -B "${source_root}/tests/resource_runner.py" auto \
        "${source_root}/resources" env "IILD_INSTALL_SCRIPT_TEST_DIRECTORY=@WORK@/install-script-test" \
        bash "${BASH_SOURCE[0]}"
fi
work="${IILD_INSTALL_SCRIPT_TEST_DIRECTORY}"
"${IILD_PYTHON_EXECUTABLE:-python3}" -B -c \
    'from pathlib import Path; import os,sys; root=Path(sys.argv[1]); path=Path(sys.argv[2]); work=path.resolve(); resources=root / "resources"; state=resources / "test-workspaces"; inherited=os.environ.get("IIFILEPROVIDER_TEST_DIRECTORY"); valid=not resources.is_symlink() and not state.is_symlink() and not path.is_symlink() and work != state.resolve() and work.is_relative_to(state.resolve()) and (not inherited or work.is_relative_to(Path(inherited).resolve())); sys.exit("Installer test workspace must be under project resources/test-workspaces") if not valid else None; sys.exit("Installer test workspace already contains files") if work.exists() and (not work.is_dir() or any(work.iterdir())) else None' \
    "${source_root}" "${work}"
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
"${IILD_PYTHON_EXECUTABLE:-python3}" - "${source_root}" "${smoke_prefix}" <<'PY'
from pathlib import Path
import os
import shutil
import sys

source, prefix = map(Path, sys.argv[1:])
reference = prefix / "share/iiLocalDiffusion/reference"
reference.mkdir(parents=True, exist_ok=True)
# Exercise the installed launcher with its small embedding/configuration assets.
# Backend model weights are not needed for --print-config and must not be copied.
def large_models(directory, names):
    private = {".cache", ".state", ".recovery", "dependencies", "outputs", "test-workspaces", "__pycache__"}
    return [name for name in names if name in private or ((Path(directory) / name).is_file()
            and (Path(directory) / name).stat().st_size > 64 * 1024 * 1024)]
shutil.copytree(source / "resources", reference.parent / "resources", dirs_exist_ok=True,
                ignore=large_models)
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
