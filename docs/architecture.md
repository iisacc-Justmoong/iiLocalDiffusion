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

The manifest milestone has one executable path with pipeline-specific leaves:

```text
iild-run -> loadModelManifest
loadModelManifest -> StableDiffusionModelManifest
loadModelManifest -> FluxModelManifest
StableDiffusionModelManifest -> private ModelManifestParser
FluxModelManifest -> private ModelManifestParser
ModelManifestParser -> filesystem and json-c
```

Python Diffusers is an independent reference oracle. The C++ library neither
embeds Python nor invokes the reference scripts. The Python generation oracle
may apply one explicitly selected LoRA after validating the base pipeline; the
C++ manifest and runtime do not yet inspect or execute adapters.

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

No neural inference backend is linked in the manifest milestone.

The first implementation candidate is MLX C++, pinned to a reviewed release,
because its C++ API and Apple unified-memory execution fit the intended
division of responsibilities. MLX model examples are not treated as the
iiLocalDiffusion architecture. If verified cross-platform demand appears,
ONNX Runtime is the leading second backend, with exported ONNX artifacts
tracked separately from their Diffusers source revision.

`stable-diffusion.cpp` is useful as a comparison implementation but already
owns most tokenizer, scheduler, pipeline, and model semantics. Making it the
primary dependency would turn iiLocalDiffusion into a wrapper and requires a
separate architecture decision.

No backend interface is added before the first component implementation needs
one. This avoids freezing a speculative abstraction.

## Explicit non-goals for the manifest milestone

- GUI, Qt, QML, LIMBO, or Dreamscapes integration
- image generation in C++
- model downloading UI or remote inference
- custom tensors, kernels, or graph executor
- LoRA package inspection or execution in C++
- training, img2img, inpainting, ControlNet, SDXL Refiner, FLUX.1-dev, or
  arbitrary FLUX derivatives
- `.ckpt`, pickle-based `.bin`, GGUF, or arbitrary checkpoint compatibility
- `.iicharacter` support

These are scope exclusions, not permanent product decisions.
