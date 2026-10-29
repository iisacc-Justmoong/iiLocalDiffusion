# Native model packaging

`iiLocalDiffusion::ModelPackaging` is a Qt-free C++23 static SDK module. It
turns a local model folder into one valid `.safetensors` container without
loading weights into an inference backend or changing their precision.
It depends on json-c 0.18+ and OpenSSL Crypto for streaming SHA-256.

```cpp
#include <ModelPackaging/ModelPackaging.hpp>
auto inspection = iild::scanModelFolder(folder, {}, stopToken, progress);
auto saved = iild::createModelPackage(folder, output, excludedFiles, stopToken, progress);
auto verified = iild::verifyModelPackage(output, stopToken, progress);
auto restored = iild::extractModelPackage(output, newFolder, stopToken, progress);
```

The functions return JSON using `iild-model-package-report-v1`; I/O and
cancellation throw `std::runtime_error`. Scans expose `ready`, `errors`,
`files`, `components`, counts and estimated bytes. Optional selections use
source-relative file IDs. Creation scans again, verifies the written file,
fsyncs it, and atomically commits without replacing an existing destination.
A `verified: true` creation report is emitted only after the final file exists.
Progress phases are `scan`, `deduplicate`, `write`, `verify`, `save`, and
`extract`; byte totals are local to each phase. Duplicate checks are displayed
as indeterminate because their ranges have independent totals.

## Container contract

Metadata identifies `iild-safetensors-package-v1` in `iild_package_schema` and
stores a JSON manifest in `iild_package_manifest`. Main safetensors tensor keys
and original metadata remain intact. Separate safetensors components receive
role and source namespaces. GGUF components and tokenizer/configuration assets
are retained as U8 assets under `__iild_assets__/`; extraction restores their
original bytes and extension. Legacy `.safetensor` files and previously supported
supporting files remain readable. ONNX files and manifests declaring embedded
ONNX models are rejected during scanning, verification and materialization;
existing input files are not modified. This is
component bundling, not tensor averaging or LoRA fusion. A manifest-aware
consumer is required to load all components. Existing `.iildmodel` ZIP packages
are a separate contract.

The manifest preserves each source path, role, format, raw Safetensors header,
metadata, per-tensor mapping, SHA-256 and original-file SHA-256. Independently
verified duplicate components refer to the existing main tensors, so extraction
can reconstruct even their original headers and filenames byte for byte.
Excluded, invalid and ignored files are documented but not restored.

## Validation and resource use

Safetensors headers, dtypes, shapes, JSON keys, contiguous extents and exact
file sizes are checked. GGUF headers, supported tensor encodings and byte extents
are checked without loading the tensors. Shard indexes are
resolved within their source directory and every indexed tensor must exist.
LTX 2.x requires its model, video/audio decoders, projections and text encoder.
Diffusers model indexes require their declared non-null component directories.
Conflicting embedded VAE/projection tensors block packaging. File symlinks
retain their logical names; directory symlinks are not traversed.

An 8 MiB streaming buffer bounds tensor I/O; JSON header size is limited to
64 MiB. Disk space is checked before writing. Source snapshots are checked
before and after copying; duplicate hashes are rechecked during creation.
Output tensor hashes and layout are validated by re-reading the output.
Cancellation removes temporary output. Extraction validates the package and
original hashes, rejects unsafe paths, stages into a new directory and preserves
the source package. A selected safetensors or GGUF text encoder can also be
reconstructed to the asset cache. Its format structure, mapping and whole-file
SHA-256 are checked before publication; safetensors tensor hashes are checked
separately. Other weights are not read. No network,
Python, Torch, GPU or tensor cast is required. This packaging check does not
establish that any particular inference backend supports the model architecture.
Materialization requires exactly one matching encoder component; a package with
multiple encoder files is rejected rather than silently choosing a shard.

## CLI and build

```sh
cmake --build build --target iild-model-package
ctest --test-dir build -R '^iiLocalDiffusion.ModelPackaging$' --output-on-failure
cmake --install build --component ModelPackaging
iild-model-package scan --input /path/to/folder
iild-model-package create --input /path/to/folder --output /path/to/model.safetensors
iild-model-package create --input /path/to/folder --output /path/to/core.safetensors --exclude spatial_upscaler.safetensors
iild-model-package verify --input /path/to/model.safetensors
iild-model-package extract --input /path/to/model.safetensors --output /path/to/new-folder
```

The CLI emits NDJSON progress and a final report. SIGINT/SIGTERM request
cooperative cancellation. Exit codes: 0 success, 2 error, 130 cancellation.
On macOS the `ModelPackaging` install component also carries its selected
OpenSSL Crypto dynamic library so its CLI can start without a developer-machine
dynamic-library search path. CMake consumers still need an OpenSSL development
SDK to resolve the exported static library's dependency.
The integration suite uses tiny Safetensors and GGUF files, sharded indexes,
byte-identical reconstruction, duplicate conflicts, missing components, ONNX
rejection, output/cache tampering, no-overwrite behavior and cancellation.
The controlled native video adapter test checks that either encoder format
reaches the engine; it is not a real-model inference or hardware performance test.
The canonical native-enabled `build/` also passed the image, controlled video
adapter and video options tests; the packaging suite passed after its CLI target
was rebuilt.

On 2026-10-27, the macOS arm64 source and separately staged CLI each passed all 23 packaging cases; the installed C++ consumer and controlled native video adapter tests also passed.
