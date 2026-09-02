# iiLocalDiffusion

iiLocalDiffusion is a C++ runtime for assembling local generative-model
components and orchestrating inference. It does not implement tensor storage,
matrix multiplication, convolution, attention kernels, or device command
execution.

The current `0.3.0` milestone is intentionally smaller than C++ image
generation. It recognizes three pinned Diffusers reference contracts: Stable
Diffusion v1, Stable Diffusion XL Base 1.0, and FLUX.1-schnell. The executable
validates package metadata and required artifact paths without loading tensor
contents.

## Build and test

Requirements:

- CMake 3.31 or newer
- a C++20 compiler
- pkg-config
- json-c 0.18 or newer

The build prefers json-c's CMake package and falls back to pkg-config.

Use only the repository's `build/` directory:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The test suite also installs the library into `build/`, configures a clean
consumer with `find_package(iiLocalDiffusion)`, links it, and runs it.

Inspect a local Diffusers model package:

```bash
./build/iild-run inspect /path/to/diffusers-model-package
```

The repository includes a metadata-only fixture for a quick CLI smoke test:

```bash
./build/iild-run inspect tests/fixtures/sd-v1-manifest
./build/iild-run inspect tests/fixtures/sdxl-base-manifest
./build/iild-run inspect tests/fixtures/flux1-schnell-manifest
```

Those fixtures contain placeholder files, not model tensors. Likewise, a
successful `inspect` result means that metadata and required artifact paths are
consistent; it does not mean the weights were parsed or inference succeeded.

## Python reference oracle

The Python tools under `reference/diffusers/` are a validation oracle, not a
runtime dependency of the C++ library. They pin both the package versions and
the Hugging Face model revision used by the project's fixed red-cube fixture.

```bash
uv venv reference/diffusers/.venv --python 3.14
uv pip install \
  --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements.txt

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset flux1-schnell

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset flux1-schnell
```

Apply one explicit local LoRA during generation:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora /absolute/path/to/adapter.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-lora.png
```

No adapter is bundled or selected automatically. Remote adapters require an
immutable commit SHA and exact safetensors filename. See
[`docs/lora.md`](docs/lora.md) for the complete input and provenance contract.

Downloads and generated outputs default to `build/reference/`, keeping the
source tree and the system disk free of multi-gigabyte model caches. See
[`reference/diffusers/README.md`](reference/diffusers/README.md) for the exact
commands and limitations.

## Project boundaries

The ownership boundary, dependency decisions, tensor contracts, and explicit
non-goals are recorded in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/stable-diffusion-v1.md`](docs/stable-diffusion-v1.md)
- [`docs/stable-diffusion-xl.md`](docs/stable-diffusion-xl.md)
- [`docs/flux1-schnell.md`](docs/flux1-schnell.md)
- [`docs/lora.md`](docs/lora.md)
- [`docs/dependencies.md`](docs/dependencies.md)

The project does not yet declare a distribution license. The SD 1.5 reference
uses CreativeML OpenRAIL-M, SDXL Base 1.0 uses CreativeML Open RAIL++-M, and
FLUX.1-schnell uses Apache-2.0; no model weights are stored in this repository.
