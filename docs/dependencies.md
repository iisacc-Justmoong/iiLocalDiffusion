# Dependency decisions

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

iiLocalDiffusion itself has no declared distribution license at this milestone.
A license decision is required before public source or binary distribution.
