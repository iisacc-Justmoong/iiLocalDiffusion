# Dependency decisions

## Current production dependency

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

## Reference-only Python dependencies

`reference/diffusers/requirements.txt` pins the direct oracle dependencies.
They are Apache-2.0 or similarly permissive libraries, but they are not linked
into or invoked by iiLocalDiffusion. Installation is opt-in and remains under
the workspace's `reference/diffusers/.venv/` directory.

The fixed Python version for this verified environment is CPython 3.14 on
macOS arm64. Package versions and hardware are recorded beside every generated
reference image.

The oracle also pins Hugging Face Xet 1.6.0 (Apache-2.0), which
`huggingface_hub` uses to transfer the reference model's large files. Its chunk
cache is redirected to `build/reference/huggingface-xet` so model downloads do
not consume the system-volume cache.

PEFT 0.20.0 (Apache-2.0) is pinned for runtime LoRA adapter injection and
activation through Diffusers. It is required only by the Python oracle and is
not linked into the C++ library. The adapter loader explicitly disables
pickle-weight fallback by accepting `.safetensors` files only.

## Reviewed future inference backends

| Candidate | Reviewed release | License | Decision |
|---|---:|---|---|
| [MLX](https://github.com/ml-explore/mlx) | 0.32.2 | MIT | Preferred first Apple Silicon backend; not yet added |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.29.0 | MIT | Preferred later cross-platform backend; requires a versioned ONNX export pipeline |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | rolling commit releases | MIT | Comparison/spike only because it owns pipeline semantics |
| [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) | 1.1.1 | MIT | Core ML/Swift deployment reference, not the first C++ backend |

MLX remains a future dependency until one neural component is implemented and
verified numerically against the Python oracle. Its exact release, source
hash, transitive dependencies, and `mlx.metallib` packaging must be fixed in
the change that introduces it.

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
