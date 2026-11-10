# Native model packaging

`iiLocalDiffusion::ModelPackaging` is a Qt-free C++23 static SDK module. It
turns a local model folder into one valid `.safetensors` container without
loading weights into an inference backend or changing their precision.
It depends on json-c 0.18+ and OpenSSL Crypto for streaming SHA-256.

```cpp
#include <ModelPackaging/ModelPackaging.hpp>
auto inspection = iild::scanModelFolder(folder, {}, stopToken, progress);
auto saved = iild::createModelPackage(folder, output, excludedFiles, stopToken, progress);
std::vector<std::filesystem::path> models{"Checkpoint/base.safetensors", "LLM/encoder.gguf"};
auto selection = iild::scanModelSelection(modelsRoot, models, {}, stopToken, progress);
auto packaged = iild::createModelPackageFromSelection(modelsRoot, models, output, excludedFiles, stopToken, progress);
std::vector<iild::ModelPackagingAssignment> assignments{
    {"Checkpoint/base.safetensors", "model", "main"},
    {"LLM/encoder.gguf", "text_encoder", ""}};
auto manualInspection = iild::scanModelSelectionWithAssignments(modelsRoot, models, assignments, {}, stopToken, progress);
auto manualPackage = iild::createModelPackageFromSelectionWithAssignments(modelsRoot, models, manualOutput, assignments, excludedFiles, stopToken, progress);
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

The selection API reads only the explicitly selected files and model directories
below its root. IDs and extraction paths remain root-relative, so same-named files
in different categories stay distinct. Duplicate and overlapping selections are
included once. It rejects traversal, missing inputs and redirected selections;
provider-owned `.iisacc.json` and hidden control files are never packaged. Folder
scans also omit provider-owned `.iisacc.json` files. Source files are not copied
into a staging directory before inspection or altered. LLM and Text Encoder
category paths identify text encoder components when filenames are inconclusive.

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

## Manual component assignments

Each selected Safetensors or GGUF source can have an explicitly assigned role
and a position of main model or component. Leaving either choice automatic
retains the detected role or the automatic main-model selection. A main position
requires the model role; only one model or sharded model group may be main.
Shard members identified by an included shard index share the group's assignment.
If automatically detected shard roles disagree, inspection requires one explicit
group role rather than creating a partially classified model.
Invalid paths, roles, positions and
conflicting assignments are rejected before a package can be written. Assignments
are applied before validation, duplicate detection and output key construction,
and the report and manifest identify the effective role, position and whether each
was chosen by the user. Creation repeats the inspection with the same choices.
Supported role values are `model`, `text_encoder`, `video_vae`, `audio_vae`,
`projections`, `upscaler`, `lora` and `other`; `video_vae` remains the existing
wire value for the VAE/decoder role. Empty values keep automatic detection.
When an indexed optional sharded component is excluded, all its identified shards
and its shard index are excluded together. An individually selected shard without
its index cannot be identified as a group; include its index or model directory.
Calibration retains explicitly assigned choices when repackaging an existing
package; older packages without explicit choices continue to be re-detected.
Assigning a role does not establish architectural or inference compatibility.

## Validation and resource use

Safetensors headers, dtypes, shapes, JSON keys, contiguous extents and exact
file sizes are checked. GGUF headers, supported tensor encodings and byte extents
are checked without loading the tensors. Shard indexes are
resolved within their source directory and every indexed tensor must exist.
LTX 2.x requires its model, video/audio decoders, projections and text encoder.
Diffusers model indexes require their declared non-null component directories.
Indexes inside selected library directories are checked relative to their own
directory as well; an incomplete selected pipeline cannot be created. Legacy
`iild-unified-model-v1` indexes retain their separate format.
Conflicting embedded VAE/projection tensors block packaging. In folder mode file
symlinks retain their logical names and directory symlinks are not traversed.
Explicit library selections reject symlinks, including within selected directories.

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
iild-model-package create --input /path/to/Models --select Checkpoint/base.safetensors --select LLM/encoder.gguf --output /path/to/model.safetensors
iild-model-package create --input /path/to/Models --select Other/base.safetensors --select Other/encoder.safetensors --assign-position Other/base.safetensors=main --assign-role Other/encoder.safetensors=text_encoder --output /path/to/model.safetensors
iild-model-package verify --input /path/to/model.safetensors
iild-model-package extract --input /path/to/model.safetensors --output /path/to/new-folder
```

The CLI emits NDJSON progress and a final report. SIGINT/SIGTERM request
cooperative cancellation. Exit codes: 0 success, 2 error, 130 cancellation.
The `ModelPackaging` component also installs `iild_model_packaging.py` under
`share/iiLocalDiffusion/python`. `scan_selection(root, paths, executable=...)`
and `create_selection(root, paths, output, executable=...)` call the native CLI
and return its final JSON report; paths are relative to the supplied root.
The executable defaults to `iild-model-package` on `PATH`, and failures raise
`ModelPackagingError`. Empty selections are rejected rather than scanning the
entire root. The wrapper does not depend on Torch or a GUI toolkit.
Both selection functions accept `assignments={"relative/path": {"role": "model",
"position": "main"}}`; either field may be omitted for automatic selection.
For legacy packages with ambiguous roles, `iild-model-package materialize --input
PACKAGE --output CACHE --asset-path RELATIVE_SOURCE` and the Python
`materialize_package_asset(..., relative_source=...)` select the exact original
manifest path. The role and path options are mutually exclusive. Unsafe or
duplicate source paths, incomplete mappings and checksum mismatches are errors;
the input package is not modified. This reconstructs a component, not a proof
that an engine can load it or generate an image.
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
