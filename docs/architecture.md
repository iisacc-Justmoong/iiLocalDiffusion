# Architecture

## Purpose

iiLocalDiffusion is a runtime that loads, represents, connects, and executes
the components of local generative models. It is not a tensor-operation
implementation. This boundary keeps the library focused on model and pipeline
semantics while allowing the low-level inference runtime to be replaced.

## Responsibility boundary

iiLocalDiffusion owns:

- model-package structure and metadata
- pipeline composition and component lifecycle
- generation requests and validation
- seed and reproducibility policy
- scheduler selection and denoising orchestration
- conditioning flow and model adapters
- result metadata, provenance, and diagnostics
- the boundary through which an inference backend is selected

An inference backend owns:

- tensor allocation, dtype, strides, layout, and device memory
- matrix multiplication, convolution, and attention primitives
- graph execution and synchronization
- CPU, Metal, CUDA, and other accelerator commands and kernels

Backend tensor types must stay behind implementation boundaries. They must not
appear in iiLocalDiffusion's public request or result types.

## Current dependency direction

Metadata inspection and native component computation are independent paths:

```text
iild-run -> loadModelManifest
loadModelManifest -> StableDiffusionModelManifest
loadModelManifest -> FluxModelManifest
StableDiffusionModelManifest -> private ModelManifestParser
FluxModelManifest -> private ModelManifestParser
ModelManifestParser -> filesystem and json-c
iild-run compute -> LinearLayer -> private MLX runtime -> Metal/CUDA
iild-run compute -> LinearLayer -> private LibTorch runtime -> AMD HIP/ROCm
LibTorch weight loading -> private safetensors-cpp reader -> owned LibTorch storage
LinearLayer -> ComputeRuntime (explicit device policy)
iild-run neural-compute -> CoreMLModel -> private system Core ML -> CPU/ANE/(optional GPU)
```

Python Diffusers is an independent reference oracle. The C++ library neither
embeds Python nor invokes the reference scripts. The Python generation oracle
can assemble explicitly selected model/checkpoint and VAE files, attach one
compatible ControlNet, then apply one selected LoRA after validating the
pipeline. `presets.py` owns family and
configuration contracts, `model_loading.py` owns Diffusers assembly, and
`weight_files.py` owns local file identity and canonical safe-loader paths.
`controlnet.py` owns optional ControlNet selection, prepared-image loading,
component compatibility, Diffusers pipeline attachment, and provenance.
It delegates neural execution, single-file conversion, and image resizing
to the existing runtime dependencies; it does not implement a condition
detector. The [ControlNet contract](controlnet.md) applies only to generation.
Optional [Hires Fix](hires-fix.md) orchestrates a base pass, Pillow RGB
resizing and an img2img refinement pass across all three families and their
ControlNet variants. It reuses the selected neural components and retains
the existing inference backend and offload policy. Project code owns stage
parameters, scheduler lifecycle, output reservation and provenance; the
existing dependencies own image resampling, VAE encoding, noise and neural
execution. Stage-specific schedules and output validation make the second
pass observable without asserting universal visual quality.
`hires_options.py` owns second-stage argument resolution;
`hires.py` owns image resizing, refinement assembly and stage validation.
`generation_output.py` reserves the final and optional base output paths.
`hires_flux_controlnet.py` combines Diffusers' public FLUX img2img
latent/schedule preparation with the existing FLUX ControlNet denoising
loop. This preserves negative conditioning and true CFG that are absent
from the pinned upstream ControlNet img2img call, without copying the
transformer or ControlNet computation. Its scheduler adjustment is scoped
to one sequential call and restored afterward.
The [model-input contract](model-inputs.md) separates embedded checkpoint
weights from the configuration source's auxiliary components. The C++
manifest does not load weight tensors or inspect/execute adapters. The
separate native `LinearLayer` can select named tensors from a safetensors file
but does not assemble a complete checkpoint or apply adapters.
`hardware.py` owns the oracle's GPU-required selection, hardware preflight,
and actual execution-device checks, including NVIDIA Tensor Core eligibility
and TF32 precision policy. It distinguishes HIP's shared CUDA namespace from
NVIDIA execution, records AMD properties, and does not apply NVIDIA TF32
controls on ROCm. It does not delegate generation to MLX or Core ML.
`cpu_conditioning.py` owns explicit CPU prompt encoding and conditioning
transfers. CPU encoding follows LoRA activation and precedes placement;
Diffusers/Accelerate owns model/sequential RAM offload hooks. This CPU stage
feeds GPU denoising and is separate from native concurrent linear partitioning.

Optional [Textual Inversion](text-embeddings.md) adds learned tokenizer
entries and matching encoder input vectors before prompt encoding and
offload placement. It composes with the existing model, LoRA, ControlNet
and Hires Fix paths. Existing tokenizer/embedding-table APIs own storage
and execution; project code owns safe input selection, dimensions, token
collision checks, multi-vector prompt handling and provenance. Completed
prompt tensors from `--embeddings` remain a separate, mutually exclusive
input path that bypasses encoding.
`text_embedding_options.py` owns ordered file/token/encoder selections;
`text_embeddings.py` validates vectors, invokes the existing Diffusers
Textual Inversion loader and verifies registered IDs and vector values.
Its scoped prompt handling expands each encoder's multi-vector tokens once
and restores normal pipeline prompt conversion after the call.

Headers and implementations live together under `src/`, as required by this
workspace's library convention. A separate source `include/` tree is not used.
Installed consumers receive the same header path under their prefix. While the
project is pre-1.0, binary and CMake package compatibility is promised only
within the same `0.x` minor release line.

## Current milestone

`loadModelManifest()` dispatches by pipeline class. The existing
`StableDiffusionModelManifest::load()` and new `FluxModelManifest::load()`
remain model-family-specific entry points. Together they validate:

- an accessible model root and a valid object-valued `model_index.json`
- the canonical SD v1, SDXL Base, or FLUX.1-schnell component set
- required component directories and configuration files
- the metadata contracts documented in `stable-diffusion-v1.md`,
  `stable-diffusion-xl.md`, and `flux1-schnell.md`
- non-empty `.safetensors` paths for every declared text encoder, denoiser,
  and VAE

It does not parse safetensors headers or tensor bodies, authenticate a model,
distinguish Stable Diffusion 1.4 from 1.5 weights, instantiate neural modules,
or execute inference. CLI wording deliberately reports `valid-metadata` and
never reports components as loaded.

The inspector is a package-compatibility diagnostic, not a security sandbox.
It follows regular-file symlinks because Hugging Face snapshots link into a
content-addressed blob cache, and it assumes the local package is not being
mutated concurrently. Runtime code must separately authenticate revisions and
apply stronger file-opening policy before loading tensors.

`safety_checker` and `feature_extractor` are present in the pinned SD v1
reference but remain outside its core-five contract. SDXL Base and
FLUX.1-schnell do not provide a safety checker. The Python oracle records
checker and watermarker presence; before image generation becomes a product
feature, product safety policy must remain separate from package compatibility.

## Backend decision

MLX C++ 0.32.2 is now a private production dependency, enabled by default.
It owns tensor storage, device transfers, matrix multiplication, bias
addition, and synchronization. `LinearLayer` is the first implemented neural
component, numerically compared with real SD 1.5 CLIP weights and the PyTorch
oracle. Metal is selected on Apple Silicon; CUDA is the Linux/NVIDIA backend.
Automatic runtime selection requires a GPU. Explicit CPU operation and
opt-in CPU/GPU cooperation are available; unavailable or failing GPU work is not retried on
CPU. See [hardware-compute.md](hardware-compute.md) for the exact scope.

The implemented AMD requirement adds an optional LibTorch HIP backend behind
the same `LinearLayer` API. A small private `LinearBackend` component boundary
separates the two concrete runtimes; it is not installed or generalized into
a tensor/graph registry. LibTorch owns AMD tensors, dtype conversion,
matrix arithmetic, CPU/GPU transfers and scheduling. The pinned safetensors-cpp
reader supplies file metadata/bytes; project code bounds its shape/offset
contract before copying selected weights into owned runtime storage.
`rocmCapabilities()` and an additive selection overload leave existing
capability layouts and enum values unchanged. MLX remains preferred for
explicit CPU calls when available; otherwise enabled LibTorch supplies CPU.
Radeon availability never implies NVIDIA Tensor Cores or AMD NPU support.
See [radeon-rocm.md](radeon-rocm.md) for deployment and verification boundaries.

The public component accepts shapes and ordinary host values or a file path
plus exact weight keys. MLX arrays and streams stay inside its PIMPL. It
retains weights between calls and uses instance-owned streams without
changing MLX's global default device. `LinearResourceOptions` can divide
output features between CPU and GPU workers, retain weights in RAM, and bound
the GPU weight block staged per forward iteration. MLX still owns tensor
storage, scheduling, copies, and math; resource counts are logical rather
than RSS/VRAM measurements. File I/O and result readback are explicit host
boundaries. No custom tensor type or kernel is introduced.
`LinearMathOptions` additionally selects GPU FP32/FP16/BF16 while retaining
FP32 CPU-shard arithmetic and ordinary float32 host outputs. CUDA kernel
selection remains MLX/cuBLAS's responsibility, not a custom Tensor Core kernel.

`CoreMLModel` is an independent native component runtime, enabled by default
on Apple platforms. Its C++ PIMPL owns an Objective-C Core ML model; callers
provide a compiled `.mlmodelc` path and named host feature vectors. Core ML
owns storage, casts, convolution, and CPU/GPU/ANE scheduling. Plan introspection
is diagnostic, not a new graph executor or a hardware counter. By default,
at least one operation must prefer ANE and only CPU/ANE are permitted.
An offline coremltools converter emits a named linear component and NumPy
oracle; it is never invoked by the library. This adds no runtime LoRA merging,
automatic checkpoint conversion, or full C++ diffusion pipeline. See
[neural-accelerators.md](neural-accelerators.md).

Full C++ diffusion generation remains unimplemented.

`stable-diffusion.cpp` is useful as a comparison implementation but already
owns most tokenizer, scheduler, pipeline, and model semantics. Making it the
primary dependency would turn iiLocalDiffusion into a wrapper and requires a
separate architecture decision.

The implemented components need only small compute options/capability types
and a private linear-component backend boundary, not a speculative general
tensor, plugin registry, or graph abstraction. Additional runtimes and model
implementations remain separate architecture decisions.

## Explicit non-goals for the manifest milestone

- GUI, Qt, QML, LIMBO, or Dreamscapes integration
- image generation in C++
- model downloading UI or remote inference
- custom tensors, kernels, or graph executor
- LoRA or ControlNet package inspection or execution in C++
- training, external img2img inputs, inpainting, Multi-ControlNet, SDXL Refiner, FLUX.1-dev, or
  arbitrary FLUX derivatives
- `.ckpt`, pickle-based `.bin`, GGUF, or arbitrary checkpoint compatibility;
  whole-checkpoint pipeline assembly is limited to the independent Python
  generation oracle; native safetensors loading selects one linear component
- `.iicharacter` support

These are scope exclusions, not permanent product decisions.
