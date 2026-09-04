# Local installation

`install.sh` installs the library and generation entry point into
`$HOME/.local/SDK/iiLocalDiffusion`. Run it from an existing checkout with the
[native build requirements](../README.md#build-and-test) available:

```bash
./install.sh
```

The script configures a Release build in the repository's `build/` directory,
builds it, runs CTest, and then installs the package. Existing CMake cache
settings for MLX, Core ML and optional LibTorch are retained. A configured
backend therefore remains enabled after installation; installation does not
add a backend that was disabled in that build. No new Python environment or
model download is performed.

The prefix and build concurrency can be supplied as environment variables:

```bash
IILD_INSTALL_PREFIX="$HOME/.local/SDK/iiLocalDiffusion" \
IILD_BUILD_JOBS=4 \
  ./install.sh
```

These are also the defaults. Repeat the command after changing the library or
Python sources to update the installed copies.

## Checkout relocation

The checkout lives under `Workspace/SDK/iiLocalDiffusion`; the default CMake
prefix is `~/.local/SDK/iiLocalDiffusion`, including when configuring directly.
Installer arguments are passed to CMake. After moving a configured checkout,
use `./install.sh --fresh` with any optional backend and SDK settings supplied
again, such as `-DIILD_ENABLE_LIBTORCH=ON -DTorch_DIR=/new/sdk/cmake/Torch`.
Keep downloaded models and Python environments, but regenerate stale build
metadata and update moved environment entry-point paths before running them.
The installer contract test checks both the SDK default and a custom prefix.

## Installed files and retained runtimes

The prefix contains the following public entry points and resources:

| Path relative to the prefix | Contents |
| --- | --- |
| `bin/iild-run` | Native component computation and metadata inspection CLI |
| `bin/iild-generate` | Unified Python image-generation launcher |
| `lib/` | Shared library and enabled bundled native runtime resources |
| `lib/cmake/iiLocalDiffusion/` | CMake package configuration and exported target |
| `include/` | Public C++ headers |
| `share/licenses/iiLocalDiffusion/` | Installed dependency license notices |
| `share/doc/iiLocalDiffusion/` | README and documentation |
| `share/iiLocalDiffusion/reference/` | Python sources, JSON configurations and requirements files |

The native executable uses a relative runtime search path to locate the
installed library. MLX resources are installed when enabled. System libraries
and an optional LibTorch SDK remain external dependencies; retain the SDK used
to configure that backend.

This is a developer installation that reuses existing local runtimes. When
present, the installer connects these source paths to the installed reference
tree with symbolic links:

| Installed link | Existing source path |
| --- | --- |
| `share/iiLocalDiffusion/reference/diffusers/.venv` | `reference/diffusers/.venv` in the checkout |
| `share/iiLocalDiffusion/build` | `build/` in the checkout |

The second link makes the existing managed ComfyUI environment and cached
models available to the installed Python scripts. Python source files are
copied into the prefix. Retain the checkout's linked virtual environment and
`build/` directory to keep those runtimes and caches usable. Moving or deleting
them breaks the corresponding links. If relocating the checkout, update the
links to its new location after restoring those dependencies. The installer
refuses to replace an existing runtime destination that points elsewhere.

## Use from CMake

An application links the exported target:

```cmake
cmake_minimum_required(VERSION 3.31)
project(LocalDiffusionConsumer LANGUAGES CXX)

find_package(iiLocalDiffusion 0.3 REQUIRED CONFIG)
add_executable(consumer main.cpp)
target_compile_features(consumer PRIVATE cxx_std_20)
target_link_libraries(consumer PRIVATE iiLocalDiffusion::iiLocalDiffusion)
```

For example, `main.cpp` can inspect a local Diffusers package:

```cpp
#include <ModelManifest/DiffusionModelManifest.hpp>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const auto manifest = iild::loadModelManifest(argv[1]);
    (void)manifest;
}
```

From the consumer project's directory, configure and build with the installed
prefix:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/.local/SDK/iiLocalDiffusion"
cmake --build build --parallel 4
./build/consumer /absolute/path/to/diffusers-package
```

On macOS, the exported target also supplies its runtime search path when the
package's `lib/` directory is present in `LIBRARY_PATH`. CMake treats that
directory as an implicit linker location, but dyld still needs an `LC_RPATH` for
the library's `@rpath` install name. Applications need only the target link shown
above; they do not need `BUILD_RPATH` or `DYLD_LIBRARY_PATH` adjustments. The
relocated installed-consumer test covers this inherited environment, including
an installation path containing spaces.

The C++ API performs component computation and metadata inspection. Complete
image generation uses the separate Python entry point below.

## Use the installed commands

Use the full command paths without changing the shell's `PATH`:

```bash
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" devices
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" compute
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" inspect \
  /absolute/path/to/diffusers-package

"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" --list-base-models
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" \
  --base-model Illustrious --print-config
```

To generate with a downloaded local checkpoint and four successive HiRes
refinements:

```bash
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" \
  --model /absolute/path/to/illustrious.safetensors \
  --base-model Illustrious \
  --prompt "a red cube" --device mps \
  --hires-fix --hires-passes 4 --hires-scale 1.5 \
  --output-dir /absolute/path/to/output
```

This requires the model's companion weights and a configured local generation
runtime. See [download-to-image generation](local-image-generation.md) for
model inputs and runtime setup. Installation alone does not establish
compatibility for every catalog entry or supply missing model components.

When HiRes Fix is enabled, an omitted repeat count defaults to **1**.
`--hires-passes` accepts a positive integer, and each pass refines the preceding
pass's output with the same settings. HiRes Fix remains disabled unless
requested. See [HiRes Fix](hires-fix.md) for supported options and per-stage
provenance.

The launcher can use the linked oracle virtual environment. To select a
different existing Python interpreter, supply its executable explicitly:

```bash
IILD_PYTHON_EXECUTABLE=/absolute/path/to/venv/bin/python \
  "$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" --check-runtime
```

That interpreter must contain the dependencies for the selected Python backend.
The managed local engine also has its own environment, documented in the local
generation guide.

## Manual CMake installation

For an already configured and tested build, CMake also supports installing
directly:

```bash
cmake --install build --config Release \
  --prefix "$HOME/.local/SDK/iiLocalDiffusion"
```

This installs the native package, tools and Python sources, but does not create
the developer runtime links or install Python dependencies. Set
`IILD_PYTHON_EXECUTABLE` to an existing environment and configure the selected
generation backend's runtime paths when using this route.
