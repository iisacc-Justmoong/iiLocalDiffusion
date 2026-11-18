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
original bytes and extension. Scans identify Safetensors and GGUF from bounded content headers even when file extensions differ. Legacy `.safetensor` files and previously supported
supporting files remain readable. ONNX files and manifests declaring embedded
ONNX models are rejected during scanning, verification and materialization;
existing input files are not modified. This is
component bundling, not tensor averaging or LoRA fusion. A manifest-aware
consumer is required to load all components. Existing `.iildmodel` ZIP packages
are a separate contract. Package inputs for verification, extraction, materialization and calibration are identified by content; new package and repaired-model outputs still use `.safetensors`.

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
`projections`, `upscaler`, `lora`, `high_noise_model`, `vision_encoder` and `other`; `video_vae` remains the existing
wire value for the VAE/decoder role. Empty values keep automatic detection.
When an indexed optional sharded component is excluded, all its identified shards
and its shard index are excluded together. An individually selected shard without
its index cannot be identified as a group; include its index or model directory.
Calibration retains explicitly assigned choices when repackaging an existing
package; older packages without explicit choices continue to be re-detected.
Known GGUF diffusion architectures are detected as model roles before automatic main selection; explicit role assignments retain precedence. Assigning a role does not establish architectural or inference compatibility.
The GGUF architecture tags `wan`, `wan2.1` and `wan2.2` identify a diffusion-model role, but do not by themselves establish a video output modality: other models also reuse those tags.

## Validation and resource use

Safetensors headers, dtypes, shapes, JSON keys, contiguous extents and exact
file sizes are checked. GGUF headers, supported tensor encodings and byte extents
are checked without loading the tensors. Shard indexes are
resolved within their source directory and every indexed tensor must exist.
FP8 FNUZ and C64 descriptors are preserved byte for byte; recognizing their
storage does not establish that an inference engine supports their arithmetic.
LTX 2.x requires its model, video/audio decoders, projections and text encoder.
Diffusers model indexes require their declared non-null component directories.
For a WanPipeline or WanImageToVideoPipeline index declaring both `transformer` and `transformer_2` as
`WanTransformer3DModel` and a numeric `boundary_ratio`, the first component is the high-noise expert and the
second is the main low-noise expert. Each directory must contain exactly one
weight group, including a validated sharded group; ambiguous or conflicting
assignments are rejected. Explicit component roles `high_noise_model` and
`vision_encoder` are also available for models without that index. These roles
do not by themselves establish runtime or model compatibility. A declared single image-to-video transformer is main; its declared CLIP image encoder is a vision_encoder component.
For a primary Safetensors model with a complete base Diffusers Wan tensor
signature, the primary tensor names map into the native Wan namespace, including
single-model packages without a dual-expert pipeline index.
The manifest retains the original names and extraction reconstructs the source
files; unrelated pipelines and incomplete signatures are not rewritten.
The validated boundary is recorded in the package for native video requests;
indexes without a valid boundary are rejected for this automatic dual-expert
route rather than silently switching models at an assumed time.
Indexes inside selected library directories are checked relative to their own
directory as well; an incomplete selected pipeline cannot be created. Legacy
`iild-unified-model-v1` indexes retain their separate format.

For standalone `*.safetensors.index.json` files, C++ callers can use
`inspectSafetensorsIndex`, `verifySafetensorsIndex`, and
`safetensorsIndexShardPaths`; the CLI exposes `iild-model-package inspect-index`
and Python exposes `inspect_sharded_index`. The inspector checks every named
tensor against exactly one referenced shard, including shapes, dtypes, extents,
and safe relative paths. It rejects missing, extra, duplicate, truncated, or
misassigned tensors and returns file identities for later change detection.
The index suffix is matched without ASCII case sensitivity; the recorded shard
paths and tensor names retain their original spelling and are resolved exactly.
Indexes and individual headers are limited to 64 MiB, aggregate metadata to
256 MiB, mappings to one million, and shards to 4,096. File symlinks are
identified by their targets. A linked index retains its selected path as the
base for relative shard references, while the index and each shard target are
snapshotted and rechecked. Image and video callers pass that same selected index
path to the native loader; they do not copy or rewrite weights. Catalogs may apply stricter link policies.
The C bridge returns the same metadata without reading tensor values; this does not establish generation compatibility.

`probeModelTensorNames`, `modelContainsTensorNames` and `findModelTensorNames` check exact source tensor names in Safetensors,
standalone shard indexes and GGUF without loading weights. It validates the
container metadata and checks source identity before and after the probe.
CLI `contains-tensors --input MODEL --tensor NAME` accepts repeated `--tensor`
arguments, and Python `contains_tensors` and `find_tensor_names` provide the same result. Presence is
not proof of tensor shapes, numerical validity or backend compatibility. The probe
also reports whether Safetensors metadata claims an SDK package; package integrity
still requires the separate manifest and checksum verification path.
For a standalone Safetensors file, exact-name probing permits an unknown dtype
while still validating header structure and data extents. It does not claim the
dtype's shape-to-byte size or native-engine support. Packaging and indexed-shard
inspection continue to require a known dtype.
The component-install test also exercises both Python operations against a
relocated CLI and a synthetic Safetensors file.

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
separately. Weights outside the selected component are not read. No network,
Python, Torch, GPU or tensor cast is required. This packaging check does not
establish that any particular inference backend supports the model architecture.
Materialization accepts one unambiguous indexed Safetensors shard group as a
single component and streams its tensors to a deterministic cached Safetensors
file. The embedded index must cover every tensor exactly once and all referenced
shards, original headers, shapes and checksums must agree. Role-based selection
requires exactly one matching group; an explicit shard-index path selects a group
when a package has several. Unindexed or ambiguous encoder files remain an error.
The source package is not modified; this conversion preserves tensor bytes and
dtypes and per-tensor descriptor fields, but does not prove that an inference
backend supports the architecture. New manifests record the shard index for each
component. For older manifests, native image inspection verifies the embedded
index checksum and recovers its mapping; individual tensors and original shard
checksums are verified during reconstruction. If recorded roles disagree, role
selection rejects the group; an explicit index path can select it. A shard
already deduplicated against the main model remains part of its indexed group,
and an explicit original path can restore that shard. Cached output is revalidated
before reuse. A per-call tensor lookup avoids repeated scans of the entire package
header. Synthetic tests exercise 2,048 mapped tensors and deduplicated shards.
The runtime's `Generation/ImagePackageBridge.hpp` exposes versioned C calls for
bounded image-package inspection and exact component materialization, including
progress and cooperative cancellation; each JSON result must be freed by the
same library. The runtime Python `image_package` adapter uses this interface to
route packaged image models without reimplementing tensor grouping.
`iiLocalDiffusion::inspectEmbeddedImagePackage` returns the inspection, selected
specialized components, and a catalog of original source/index paths and tensor
name mappings in one bounded read. `inspectEmbeddedImageCatalog` exposes the
catalog separately; GGUF assets have no tensor mappings. The equivalent C ABI
uses `iild_image_package_inspect_v1` and `iild_image_package_catalog_v1` and
frees reports with `iild_image_package_free_v1`. Catalog entries are metadata,
not proof that tensor payloads or their combinations are compatible. Before
execution, the exact selected source is materialized with index and checksum
verification; multiple recognizable image checkpoints are not selected silently.
The Python router accepts `--package-model` with the original checkpoint or
shard-index path to choose among recognized image checkpoints. When several
candidates exist, a single matching model explicitly assigned `main` by the
package creator is selected automatically; automatically inferred positions
never resolve ambiguity. The catalog exposes this user assignment for entire
indexed groups, and the error for an unspecified ambiguous selection lists up
to eight candidate paths.
The Python generation router accepts `--package-components` to map native slots
or MOSS `llm` and `vae` slots to original paths inside a package. This is distinct from `--components`, which
selects already available local files, and conflicts are rejected.
Inspection alone does not read model weights, and neither this bridge nor a
synthetic worker test establishes successful pretrained inference.

For a recognized native image package with no complete embedded or explicitly
chosen VAE, the router can inspect separate VAE candidates against the actual
native engine's tensor contract. `Generation/NativeVaeInspection.hpp` exposes
metadata-only inspection and candidate validation to C++; the versioned
`iild_image_vae_inspect_v1` and `iild_image_vae_validate_v1` C functions return
owned reports using the same free function as the package bridge. The Python
adapter inspects a temporary sparse metadata view, copying only bounded Comfy
quantization descriptors when required. It verifies and reconstructs only the
unique compatible component. Multiple compatible candidates require an
explicit `--package-components` choice; unsupported contracts, no match, and
filesystems unable to create the sparse probe leave existing backend behavior
unchanged. An explicit VAE always takes precedence. These contracts validate
tensor names and dimensions, not weight values, text encoders, or inference.
The same package router can resolve CLIP-L, CLIP-G and T5-XXL for supported
SD1/2, SDXL, SD3 and FLUX.1 slots. `Generation/NativeTextEncoderInspection.hpp`
and the matching versioned C calls inspect native encoder contracts without
executing tensors. The engine's optional-tensor rules are retained; T5-XXL
auto-selection additionally requires the standard 24-layer 4096-dimensional
contract. Standard CLIP/T5 safetensors signatures are candidate filters only:
all required engine tensors must match before a unique source is selected.
GGUF assets without a package tensor mapping and other encoder architectures
remain available through explicit component paths. An explicit component or a
complete embedded component takes precedence, and multiple matching sources
are reported rather than guessed. Native-runtime-disabled builds report the
inspection contract as unsupported without changing their existing route.

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
manifest path or its embedded Safetensors shard-index path. The role and path
options are mutually exclusive. Unsafe or
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

## Renamed shard indexes

A complete Safetensors shard group is preserved even when its index or shard
filenames use nonstandard extensions. New package manifests identify that index
explicitly; existing manifests with conventional index names remain readable.
Reconstruction still validates the original index, exact tensor coverage and
checksums, and never rewrites the source files. Same-directory links to one index
are treated as one group, preferring a conventional index name; distinct indexes
claiming the same shard are rejected.
Native index-probe and Wan loader patches are checked together during repeat
configuration; tests also reapply them in an isolated resource workspace.
