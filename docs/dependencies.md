# Dependency decisions

## Interpolator animation

The Interpolator reuses pinned Diffusers/PyTorch, including the existing text
encoding, pipeline, adapter and latent-packing APIs. Tensor interpolation uses
PyTorch operations; PNG and video output reuse Pillow and the shared external
FFmpeg/FFprobe layer described below. There is no new package, model download
or native dependency. Existing Diffusers is Apache-2.0 and PyTorch BSD-3-Clause;
their pinned versions and distributed notices remain in force.

The official [DiffusionBee Interpolator source](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui/blob/master/backends/stable_diffusion/applets/frame_interpolator.py)
defines prompt/seed interpolation. Its application/plugin/model-container
coupling would duplicate this SDK's loader and execution policies; no source
is copied or imported. The small two-endpoint coordinator is domain code.
Maintained numerical, neural inference and media libraries perform the heavy
operations. This standalone prompt/seed mode does not need a frame-insertion
model. Post-LTX video uses the separate frame-interpolation decision below.

## Post-LTX frame Interpolator

The second video stage reuses the installed FFmpeg `minterpolate` filter with
motion compensation, plus its timestamp/padding filters and Pillow PNG checks.
No inference package or second model is introduced. The maintained upstream
filter supplies motion estimation; SDK code only plans source positions,
preserves anchors/cuts and verifies/publishes artifacts. This avoids adding a
second neural runtime or weight-distribution requirement for basic interpolation.
The existing libx264-enabled external FFmpeg build remains GPL and is not bundled.
Filter availability and the actual executable version are verified locally.
Sources: [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#minterpolate),
[FFmpeg license](https://ffmpeg.org/legal.html).

## Deforum 2D animation

The animation runner reuses pinned Diffusers 0.40.0 (Apache-2.0), PyTorch,
NumPy and Pillow for inference, numeric operations and PNGs. Its optional
`requirements-deforum.txt` pins `opencv-python-headless==4.13.0.92` for maintained
affine camera transforms and explicit edge modes. The reviewed macOS arm64
wheel is about 46 MB and adds only a NumPy dependency, already present. The
headless build avoids Qt/X11 GUI dependencies. OpenCV is Apache-2.0; its wheel
packaging and included third-party components carry their own notices. Sources:
[package release](https://pypi.org/project/opencv-python-headless/4.13.0.92/),
[upstream affine API](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).

The maintained FFmpeg/FFprobe command-line tools handle H.264 encoding and
decoded stream verification. They are external executables and are not bundled;
libx264-enabled FFmpeg builds are GPL. The selected executables/versions are
recorded in every animation report. See [FFmpeg licensing](https://ffmpeg.org/legal.html)
and [image sequence input](https://ffmpeg.org/ffmpeg-formats.html#image2-1).

The upstream Deforum notebook (MIT plus component notices) and AUTOMATIC1111
extension (AGPL-3.0) were reviewed. Their notebook/WebUI execution coupling and
additional model/runtime stack would duplicate this repository's existing
loading, hardware and adapter policies. No upstream source is vendored or
imported. The small schedule/feedback coordinator is application-domain code;
image transforms, neural inference and encoding use maintained libraries.
References: [notebook](https://github.com/deforum/deforum-stable-diffusion),
[extension](https://github.com/deforum/sd-webui-deforum),
[animation semantics](https://github.com/deforum/sd-webui-deforum/wiki/Animation-Settings).

## Broad local model generation

The [three model source inputs](model-sources.md) add no package dependency.
Direct remote endpoints use Python's standard HTTP client. Cloud model IDs use
the existing pinned Apache-2.0 `huggingface-hub==1.29.0` InferenceClient and its
maintained provider adapters. Pillow/FFmpeg validate and postprocess returned
media. Providers remain separate services; model-ID parsing and mock dispatch
tests do not establish live availability or completed paid inference.

The Civitai compatibility work reuses pinned Diffusers 0.40.0 (Apache-2.0)
instead of implementing additional sampling algorithms. The generic runner
loads only built-in pipelines and disables custom remote code. Optional
architecture dependencies remain explicit errors. Model licenses are separate.

SentencePiece 0.2.2 (Apache-2.0) is pinned for Kolors/ChatGLM tokenization and
other SentencePiece-based encoders. The official PyPI release provides a CPython
3.14 Apple Silicon wheel (about 1.35 MB) and adds no mandatory Python package
dependencies. This reuses Google's maintained tokenizer implementation instead
of implementing model-specific tokenization. The local audit detected that
Diffusers otherwise exposes a dummy Kolors class. Sources:
[official package](https://pypi.org/project/sentencepiece/0.2.2/),
[upstream](https://github.com/google/sentencepiece), and the installed Diffusers
`is_sentencepiece_available()` dependency gate.

ComfyUI is an optional, separately installed local process accessed through its
documented HTTP API. Its maintained native/custom node ecosystem provides model
architectures and weight layouts absent from pinned Diffusers. ComfyUI is
GPL-3.0; custom nodes and weights have separate licenses. No ComfyUI code, nodes
or weights are vendored by this change. The standard-library HTTP client adds
no Python package dependency. This process boundary keeps the substantial
Torch/node environment out of C++ and the pinned Diffusers environment. See
[workflow setup and verification](comfyui-generation.md).

The optional `reference/setup_comfyui.py` installer now pins the ComfyUI source
and ComfyUI-GGUF extension in a separate `build/reference/comfyui-venv`, retaining
an installed-package lock. The managed local-image backend starts and stops this
engine per invocation. ComfyUI-GGUF is Apache-2.0 and reuses the maintained GGUF
reader/quantized loaders instead of adding a tensor decoder to iiLocalDiffusion.
See [installation and exact revisions](local-image-generation.md).

Kolors raw-checkpoint alternatives were also reviewed. The GPL-3.0
[ComfyUI-Kolors-MZ source](https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ/tree/43ec2701a1390259a17ef3bea6244a3134aa5153)
contains checkpoint/UNet loaders and a ChatGLM conditioning path, but its model
hooks date from 2024, its latest reviewed change is a March 2025 registry update,
and it has no dependency manifest. The four-bit encoder code also contains an
implicit package installation path. It was not installed or advertised as
compatible with this 2026 engine. The Apache-2.0
[KwaiKolorsWrapper](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/tree/6fc1cd9d20bb7537facf180e5494b486b9710e24)
uses Diffusers directories and does not close the raw-checkpoint gap. Complete
Kolors Diffusers packages retain the existing generic local execution path.

## Current production dependencies

### json-c

- Required version: 0.18 or newer
- Version verified in this workspace: 0.19
- License: MIT
- Linkage: private implementation dependency; CMake package preferred with a pkg-config fallback
- Purpose: strict parsing of Diffusers JSON metadata

The C++ standard library has no JSON parser. Using the maintained
[`json-c`](https://github.com/json-c/json-c) implementation avoids creating a
partial parser and keeps JSON types out of the public API. Version 0.19 was
released in 2026 and is the current documented release. The dependency is
small relative to an inference runtime and can be replaced without changing
the manifest interface.

### MLX native computation

MLX 0.32.2 is a private shared-library dependency, enabled by default. The
source is pinned to commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85` and
archive SHA-256 `cb988a5bdc38c798918d042b9b1c6edda3ccc5f23a2155138d3aa5c1b2acc301`.
The reviewed C++ API supplies tensors, safetensors reading, device discovery,
streams, matrix operations, and synchronization without embedding Python.
This is considerably larger than json-c but avoids maintaining custom GPU
kernels or a tensor system. Public headers expose no MLX types.

Metal is built on Apple Silicon; the CUDA source configuration targets
Linux/NVIDIA. MLX's explicit CPU backend supports diagnostics, host readback,
and opt-in concurrent CPU/GPU linear partitioning; automatic selection still
requires a GPU. RAM staging uses MLX copies and streams, with standard C++
thread/future coordination and no additional runtime dependency. The first native
component is `LinearLayer`, verified against a real SD 1.5 CLIP layer and
PyTorch. Whole diffusion-pipeline assembly still belongs to the independent
Python oracle. See [hardware-compute.md](hardware-compute.md).

`cmake/IildMlx.cmake` pins fetched dependencies by version and SHA-256:

| Dependency | Version | License | Use |
|---|---|---|---|
| MLX, including bundled JACCL code | 0.32.2 | MIT | Native computation/runtime |
| Apple metal-cpp | 26 | Apache-2.0 | Metal C++ bindings, Metal builds only |
| nlohmann/json | 3.11.3 | MIT | Private MLX safetensors metadata and bundled code |
| fmt | 12.1.0 | MIT | Private MLX formatting |
| NVIDIA CCCL | 3.1.3 | Apache-2.0 with upstream exceptions | CUDA device/JIT headers |
| NVIDIA NVTX | 3.1.1 | Apache-2.0 with LLVM exceptions | CUDA profiling annotations |
| NVIDIA cuDNN frontend | 1.16.0 | MIT | CUDA neural operation descriptors |
| NVIDIA CUTLASS headers | 4.4.2 | BSD-3-Clause | CUDA matrix kernels/JIT headers |

Unused backend dependencies are declared but not fetched. MLX upstream's
tests, examples, benchmarks, Python bindings, and GGUF loading are disabled.
Metal JIT remains enabled. The shared runtime and `mlx.metallib` are installed
with the library; CUDA builds also install their required JIT headers. Full
upstream notices are installed under `share/licenses/iiLocalDiffusion/`.
The CCCL header release omits its full license text, so that exact-version
document is fetched separately with a pinned SHA-256. CUTLASS's Python DSL is
not built, used, or distributed here.

The Metal compiler/frameworks and CUDA toolkit, driver, cuDNN libraries, and
BLAS/LAPACK remain platform dependencies. The project does not imply that
permissive source-header licenses cover NVIDIA's binary SDKs or model weights.
Physical Metal operation and relocation have been verified on the M1 Max
host; CUDA compilation and execution have not been run on NVIDIA hardware.

### Apple Core ML components

Apple builds additionally use the system Foundation and Core ML frameworks
as private implementation dependencies, independently controlled by
`IILD_ENABLE_COREML`. Public C++ headers contain no Objective-C/framework
types. Public compute-device and compute-plan APIs provide ANE discovery and
anticipated placement; no private `_ANE` APIs, custom neural kernels, or
third-party full diffusion runtime are used. The reviewed SDK exposes macOS
14.4 APIs. Frameworks remain operating-system dependencies under Apple's SDK
terms and are not copied into the install.

For offline conversion only, `reference/coreml/requirements.txt` pins
coremltools 9.0 (BSD-3-Clause), NumPy 2.3.5 (BSD-3-Clause), safetensors 0.6.2
(Apache-2.0), and ml_dtypes 0.5.3 (Apache-2.0). Apple's
[coremltools](https://github.com/apple/coremltools) is maintained and supplies
the conversion/compilation and plan APIs. Stable 9.0 was selected over the
reviewed 9.1 development release. Its CPython 3.13 macOS wheel and small
conversion-only dependency set avoid adding Torch/TensorFlow to this
environment or changing the Python 3.14 Diffusers installation. The separate
environment stays under `build/reference/coreml-venv/`; neither it nor NumPy
is a C++ runtime dependency. Predictions and numerical validation use the
C++ Core ML bridge, with an independent NumPy oracle, rather than the Python
prediction buffer bridge. See [conversion notes](../reference/coreml/README.md)
and [hardware limitations](neural-accelerators.md).

Tensor Core detection and precision selection reuse MLX/cuBLAS and the
already-pinned PyTorch APIs. No additional NVIDIA SDK or custom CUDA code is
introduced.

### Optional AMD ROCm / LibTorch

`IILD_ENABLE_LIBTORCH=ON` uses an externally installed LibTorch 2.x SDK
(>=2.9); 2.13.0 was compiled and executed on the current host. PyTorch is
actively maintained, uses a BSD-style license with bundled third-party
notices, and already supplies the independent oracle's tensor operations.
Its C++ API avoids introducing a custom HIP tensor allocator or matrix
kernel. The dependency is large, so it is optional and never automatically
downloaded or bundled. Actual Radeon execution requires a matching vendor
HIP build, supported GPU/OS and AMD driver; a CPU/NVIDIA distribution is not
reported as ROCm. Legacy C++ ABI=0 distributions are rejected.

The bridge uses `at::Context::hasROCM()` and Torch's GPU count, then the same
CUDA-namespaced ATen operations used by HIP PyTorch. Only component-level
host values and metadata are public; no Torch types or Python interpreter
are exposed. An external SDK library path remains necessary after install;
relocating iiLocalDiffusion does not relocate LibTorch/ROCm. Kernel libraries,
drivers and SDK components retain their vendor licenses and distribution
requirements. [Upstream C++ setup](https://docs.pytorch.org/cppdocs/installing.html)
and [HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) were reviewed.

MLX does not provide the required AMD backend and LibTorch has no equivalent
native safetensors loader. The optional path therefore uses the small,
dependency-free C++ reader from
[safetensors-cpp](https://github.com/syoyo/safetensors-cpp), pinned to commit
`af90b6c3006cdcecf8b7d7254f5f32d301728acc` (2025-12-27), archive SHA-256
`f978132be070d6e0ae0be097c6cd5b65edeedf19f78c57158b2c43ffa412323d`.
It is a smaller community project than PyTorch; its documented incomplete
shape validation is supplemented by bounded shape products and byte-range
checks, including full payload coverage and rejection of overlapping ranges,
plus malformed-file regression tests. Its float conversion helpers
are not used: LibTorch owns FP16/BF16 conversion. It is not a security sandbox.
The reader is MIT; embedded notices cover minijson/nlohmann-json, Grisu2,
memory mapping and AMD-derived code under MIT, and an unused FP16 helper
under CC0. The full pinned header and license accompany the native install.
The upstream README also lists historical Apache-2.0 parsing code; retain
upstream notices when changing the pin. See [Radeon setup](radeon-rocm.md).

## Reference-only Python dependencies

`reference/diffusers/requirements.txt` pins the direct oracle dependencies.
They are Apache-2.0 or similarly permissive libraries, but they are not linked
into or invoked by iiLocalDiffusion. Installation is opt-in and remains under
the workspace's `reference/diffusers/.venv/` directory.
The unchanged non-Torch pins now live in `requirements-common.txt`.
`requirements-rocm.txt` includes those pins without forcing the macOS Torch
version onto AMD's vendor wheel. Install the matching HIP wheel first and
validate it again after resolving Python dependencies; the runtime probe
rejects an incompatible CPU/NVIDIA replacement before model loading.

The fixed Python version for this verified environment is CPython 3.14 on
macOS arm64. Package versions and hardware are recorded beside every generated
reference image.

The oracle also pins Hugging Face Xet 1.6.0 (Apache-2.0), which
`huggingface_hub` uses to transfer the reference model's large files. Its chunk
cache is redirected to `build/reference/huggingface-xet` so model downloads do
not consume the system-volume cache.

PEFT 0.20.0 (Apache-2.0) is pinned for runtime LoRA adapter injection and
activation through Diffusers. It is required only by the Python oracle and is
not linked into the C++ library. Model/VAE single-file loading uses the already
pinned Diffusers and safetensors packages; no additional third-party runtime
dependency is introduced. Diffusers owns checkpoint-layout conversion and
neural component construction, while project code owns argument resolution,
composition, file identity, and provenance.

Optional ControlNet generation reuses Diffusers 0.40.0 (Apache-2.0) and the
existing PyTorch, Accelerate, safetensors, huggingface_hub, and Pillow pins.
The maintained upstream ControlNet model/pipeline implementations own neural
execution and image resizing; Pillow handles static-image decoding, EXIF
orientation, and RGB conversion. Project code owns explicit selection,
family compatibility, composition, hashes, and provenance. Native single
files are staged as a temporary Diffusers component package, while supported
original SD/SDXL files use its single-file converter. This adds no runtime
dependency, C++ binding, or independently maintained neural implementation.
No automatic condition detector, OpenCV, or depth/pose model is introduced;
the caller supplies the prepared image. See [ControlNet inputs](controlnet.md).

Optional Hires Fix uses those same maintained Diffusers, PyTorch,
Accelerate and Pillow dependencies. Pillow supplies nearest, bilinear,
bicubic and Lanczos RGB resizing; Diffusers provides image-to-image VAE
preparation, scheduling and denoising with the selected base-model, VAE,
LoRA and optional ControlNet components. Project code coordinates the two
stages and records their provenance. No learned upscaler package, additional
model download, tensor kernel, or C++ binding is introduced. The existing
dependency licenses and weight-specific terms continue to apply. See
[Hires Fix](hires-fix.md).

The pinned FLUX ControlNet img2img call lacks the negative-conditioning
arguments supported by the existing generation interface. A scoped
compatibility adapter uses the upstream FLUX img2img latent/schedule
helpers with the upstream ControlNet denoising loop, preserving true CFG
and negative embeddings. It does not maintain a duplicate neural loop;
this API compatibility point requires regression checks when upgrading
Diffusers.

Textual Inversion reuses the pinned Diffusers, Transformers, PyTorch and
safetensors dependencies. Their tokenizer and embedding-table APIs handle
learned-token registration and tensor storage, while project code checks
local file identity, encoder compatibility and token collisions and records
provenance. The feature adds no training runtime, new package, bundled
embedding weights or automatic download. Its learned vectors have their
own terms, independent of the runtime licenses. See
[learned text embeddings](text-embeddings.md).

CPU prompt encoding and RAM offload reuse the already-pinned PyTorch,
Diffusers, and Accelerate packages. Accelerate owns the offload hook lifecycle;
no custom hooks, disk-swapping layer, external scheduler, or extra dependency
is introduced. Upstream memory guidance is linked in
[hardware-compute.md](hardware-compute.md#cpugpu-cooperation-and-ram-storage).

Local model, VAE, LoRA, and ControlNet inputs accept `.safetensors` and `.safetensor`.
The singular spelling uses a temporary canonical `.safetensors` symlink,
keeping Diffusers on its safetensors branch; pickle-weight formats are
rejected. Remote adapter filenames still require `.safetensors`. Configuration
snapshots are resolved explicitly at immutable revisions and passed to
single-file loaders as local directories so offline operation does not depend
on Diffusers' automatic config-download fallback. Details are in
[model-inputs.md](model-inputs.md).

## Reviewed future inference backends

| Candidate | Reviewed release | License | Decision |
|---|---:|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.29.0 | MIT | Preferred later cross-platform backend; requires a versioned ONNX export pipeline |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | rolling commit releases | MIT | Comparison/spike only because it owns pipeline semantics |
| [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) | 1.1.1 | MIT | Full Core ML/Swift diffusion deployment reference; separate from the implemented native component bridge |

These alternatives are not linked. MLX's current implementation does not
freeze a general tensor/backend interface or transfer pipeline ownership to
an external full Stable Diffusion implementation.

## Model and project licensing

Inference-library licenses do not cover model weights. The pinned Stable
Diffusion 1.5 mirror declares CreativeML OpenRAIL-M. The pinned SDXL Base 1.0
repository declares CreativeML Open RAIL++-M; this must not be confused with
the separate SDXL 0.9 research terms. The pinned FLUX.1-schnell repository
declares Apache-2.0; it must not be confused with FLUX.1-dev's separate
non-commercial license. Model identity, revision, license metadata,
safety-checker status, and watermarker status must be carried into eventual
generation results.

The original code and documentation in iiLocalDiffusion are distributed under
the GNU Affero General Public License version 3.0 only
(`SPDX-License-Identifier: AGPL-3.0-only`). See [LICENSE](../LICENSE) for the full
license text. Third-party code, libraries, tools, and model weights retain
their respective licenses; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Temporal video dependency decision

The video backend reuses Diffusers 0.40.0 `LTXConditionPipeline` and its maintained
LTX transformer, temporal VAE, T5 and flow-matching scheduler. The existing
PyTorch/Transformers/Accelerate stack owns neural inference and offload; existing
Pillow/FFmpeg utilities own image preparation and verified MP4 publication.
`requirements-video.txt` adds `protobuf==7.36.1` for Transformers' conversion of
the published T5 SentencePiece tokenizer. This maintained BSD-3-Clause package
provides a small native wheel and avoids a custom tokenizer implementation;
see [the pinned release](https://pypi.org/project/protobuf/7.36.1/).
Tokenizer loading is checked before reading multi-GB model weights.
Previously validated LTX Video 2B 0.9.5 weights
have separate Open RAIL-M terms; callers must supply compatible local weights. Newer model versions have
different terms. See [the video contract](temporal-video.md) for the reference
workflow, selection rationale and scope.
