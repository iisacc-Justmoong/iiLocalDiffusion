# Build and resource layout

The native diffusion module, its command-line interfaces and selected tests
compile in `build/`. Fetched MLX, stable-diffusion.cpp and other dependency
source trees live under `resources/dependencies/`; compiled dependency files
and CMake metadata remain in `build/dependencies/`. A caller can still supply a
different checkout through the corresponding `FETCHCONTENT_SOURCE_DIR_*`
override. Existing models, virtual environments and other user data recovered
from earlier builds remain under project resources. Compatibility links can be
retained temporarily when another active project still points to an older build
path.
The CUDA CCCL license is a pinned, checksum-verified input under
`resources/dependencies/`; an existing mismatched file or symlink is rejected.
Cached source directories are reused as local checkouts: archive checksums are
verified on initial acquisition, not again on every configure. Preserve local
edits and refresh the relevant source when changing a pinned dependency.

Manual inference probes, cache integration programs and the weight-read
benchmark are excluded from the default build and remain explicit targets.
Python test registrations disable bytecode generation instead of writing a
`build/pycache` tree.

Native contract, model-manifest, cache and result tests reserve distinct
`resources/test-workspaces/` directories through iiFileProvider and remove
their temporary files after each run. Cross-compilation keeps the existing
direct test commands and uses resource fallback paths, which require cleanup
after a cross-platform run.
Installer, CLI and packaging fixtures also use these disposable resource
workspaces. A consumer's compiled test files use a separate owned directory
under `build/test-workspaces/`; the runner removes both areas on success or
failure and refuses to replace an existing directory. Installer fixture copies
exclude private state, recovery, dependencies and test workspaces, so the
fixture cannot recursively copy its own work area.
Python policy and model-contract fixtures use the same resource area, both when
invoked directly and when nested inside a test-runner workspace. A requested
workspace must remain inside that area; fixture names cannot contain path
separators. The provider is found through `IIFILEPROVIDER_PYTHON_DIRECTORY` or
the normal installed iiFileProvider Python package.

The two manual model-merge smoke commands use the same disposable resource
workspace, disable Python bytecode caches, and print their verification report
as JSON. Image and model paths in that report are temporary and are removed
when the command returns. These inference commands are not run by the workspace
layout contract tests.
The manual sampled-checkpoint and nonfinite verifiers also use disposable
resource workspaces and print their full evidence as JSON. Their tests exercise
only workspace orchestration; actual checkpoints and the installed merge CLI
are not run by those tests.

Diffusers reference generation, inspection and ComfyUI runs default to
`resources/outputs/reference/`; their download caches default to
`resources/.state/reference-cache/`. Generated output remains user data, while
an idle cache may be discarded. Explicit output and cache arguments retain
their original behavior. Reference and checkpoint-conversion test fixtures use
disposable iiFileProvider workspaces; optional real-conversion evidence is
printed instead of writing a report to the build tree.
Standalone, unified, native, remote, generic and video generation, model
merging and managed image jobs use the same output root. The ComfyUI source is
kept under `resources/dependencies/ComfyUI` and its isolated environment under
`resources/.state/reference-runtime/`. MLX parity evidence must be placed in
project resources; its compiled native test remains in `build/`.
The runtime resource installer excludes private state, recovery, dependency,
test and generated-output directories even when they contain model or metadata
filenames. A small fixture installation tests the real shared install rule
without packaging user models or executing an application.
The `Runtime` install component includes the iiLocalDiffusion shared library
alongside its Python tools and resources. The generated installer is checked for
that component without copying the full, potentially multi-gigabyte resource set.

The managed local-image process places Python bytecode in the selected resource
cache, including when no cache override is given. Core ML conversion evidence is
restricted to `resources/`; its compiled native test stays under `build/`.
Manual model-inference smoke scripts default to `resources/outputs/verification/`
and retain their generated evidence there for the caller to inspect or remove.
They do not execute during the layout policy test. Local IDE Debug profiles use
the project's existing `build/` rather than creating `cmake-build-debug/`.

Tests that start application-facing tools or require model inference are not
part of clean-layout verification. Historical documents can name the former
`build/` locations where earlier evidence was originally recorded. A successful
build or unit test does not imply that model inference or an application was run.
