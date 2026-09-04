#!/usr/bin/env python3
"""Install the pinned, isolated local checkpoint runtime under build/."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMFY_COMMIT = "e80c1570b6b44a2557d5d8e341e05782d18c9bbb"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
COMFY_URL = "https://github.com/Comfy-Org/ComfyUI.git"
GGUF_URL = "https://github.com/city96/ComfyUI-GGUF.git"


def checkout(path: Path, url: str, commit: str):
    if path.exists():
        origin = subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip()
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if origin != url or head != commit:
            raise ValueError(f"Existing runtime checkout is not the pinned source: {path}")
        if subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--name-only"], text=True).strip():
            raise ValueError(f"Existing runtime source has local changes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".iild-runtime-") as temporary:
        staged = Path(temporary) / "source"
        staged.mkdir()
        subprocess.run(["git", "init", str(staged)], check=True)
        subprocess.run(["git", "-C", str(staged), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(staged), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(staged), "checkout", "--detach", "FETCH_HEAD"], check=True)
        staged.rename(path)


def install(python: str):
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv first; the runtime installer does not alter system Python.")
    source = ROOT / "build/reference/ComfyUI"
    environment = ROOT / "build/reference/comfyui-venv"
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    checkout(source, COMFY_URL, COMFY_COMMIT)
    checkout(source / "custom_nodes/ComfyUI-GGUF", GGUF_URL, GGUF_COMMIT)
    prefix = [uv, "--cache-dir", str(ROOT / "build/reference/uv-cache")]
    if not executable.exists():
        subprocess.run([*prefix, "venv", "--python", python, str(environment)], check=True)
    subprocess.run([*prefix, "pip", "install", "--python", str(executable),
                    "-r", str(source / "requirements.txt"),
                    "-r", str(source / "custom_nodes/ComfyUI-GGUF/requirements.txt"),
                    "-c", str(ROOT / "reference/diffusers/requirements-common.txt"),
                    "torch==2.13.0"], check=True)
    frozen = subprocess.check_output([*prefix, "pip", "freeze", "--python", str(executable)], text=True)
    (ROOT / "build/reference/comfyui-installed.lock").write_text(frozen)
    result = {"comfyui_commit": COMFY_COMMIT, "gguf_commit": GGUF_COMMIT,
              "source": str(source), "python": str(executable)}
    (ROOT / "build/reference/comfyui-installation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.python), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
