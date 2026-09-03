# iiLocalDiffusion

iiLocalDiffusion is a C++ runtime for assembling local generative-model
components and orchestrating inference. It does not implement tensor storage,
matrix multiplication, convolution, attention kernels, or device command
execution.

The current `0.3.0` milestone recognizes three pinned Diffusers reference
contracts: Stable Diffusion v1, Stable Diffusion XL Base 1.0, and
FLUX.1-schnell. Its inspector validates metadata and artifact paths without
loading tensors. Native component computation uses MLX 0.32.2 with Metal
or CUDA, or optional LibTorch with AMD ROCm: `LinearLayer` executes batched neural operations from caller-provided
or named safetensors weights. `CoreMLModel` executes explicitly supplied
compiled Core ML components with Neural Engine plan checks. CUDA Tensor Core
eligibility and FP16/BF16/TF32 policies are exposed separately from measured
hardware utilization. Complete image generation remains in the
independent Python oracle, not the C++ library.

## Build and test

Requirements:

- CMake 3.31 or newer
- a C++20 compiler
- pkg-config
- json-c 0.18 or newer
- for the default native runtime: Apple Silicon with the Xcode Metal compiler,
  or Linux with a CUDA toolkit, cuDNN, and BLAS/LAPACK development libraries
- for Apple Core ML support: a macOS SDK exposing Core ML's macOS 14.4 APIs
- for optional Radeon execution: a compatible HIP-enabled LibTorch SDK (2.x, >=2.9), AMD driver and supported GPU/OS

The build prefers json-c's CMake package and falls back to pkg-config. MLX and
its source dependencies are fetched at pinned versions and archive hashes.
On Apple Silicon, `xcrun metal --version` must succeed; merely installing the
command-line shim is insufficient. See [hardware setup](docs/hardware-compute.md).

Use only the repository's `build/` directory:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The test suite also installs the library under `build/`, relocates the prefix,
configures a clean consumer with `find_package(iiLocalDiffusion)`, and runs
both metadata inspection and native computation. For a metadata-only build,
configure with `-DIILD_ENABLE_MLX=OFF -DIILD_ENABLE_COREML=OFF -DIILD_ENABLE_LIBTORCH=OFF`.

Discover hardware and execute a verified native batched linear operation:

```bash
./build/iild-run devices
./build/iild-run compute
```

Native computation and Python generation default to GPU-required automatic
selection: an available CUDA/ROCm GPU, otherwise Metal. CPU is used only when
explicitly selected, either alone or as an opt-in cooperating device. An
unavailable accelerator or failed operation is not
silently retried on CPU. Metal/CUDA/ROCm are backends; MLX and LibTorch are runtimes,
not device-name aliases. See [hardware compute](docs/hardware-compute.md)
for the C++ API, platform options, verification, and remaining limitations.

Radeon uses `--device rocm`. The Python path requires an AMD HIP PyTorch
distribution; native C++ additionally enables `IILD_ENABLE_LIBTORCH=ON` with
that SDK's CMake package. Model/VAE/LoRA/ControlNet arguments, CPU prompt encoding, and
RAM offload remain available. See [Radeon setup and hardware checks](docs/radeon-rocm.md).

Use CPU and GPU together, with weights retained in RAM and bounded GPU staging:

```bash
./build/iild-run compute --cpu-share 0.25 --weight-storage ram --gpu-weight-mib 64
```

The CPU computes a share of the linear layer's output features, concurrently
with the GPU's remaining share. RAM stores weights; it is not an arithmetic
device. The budget limits a staged weight/bias block, not total process memory.

Use eligible matrix precision or an explicitly compiled Neural Engine component:

```bash
./build/iild-run compute --precision fp16
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc
```

Core ML defaults to CPU + Neural Engine, with at least one ANE-preferred
operation required in its anticipated plan. This is independent of MLX and
does not move the Python diffusion pipeline onto ANE. See
[Neural Engine / Tensor Cores](docs/neural-accelerators.md) and
[named-weight conversion](reference/coreml/README.md).

Inspect a local Diffusers model package:

```bash
./build/iild-run inspect /path/to/diffusers-model-package
```

The repository includes a metadata-only fixture for a quick CLI smoke test:

```bash
./build/iild-run inspect tests/fixtures/sd-v1-manifest
./build/iild-run inspect tests/fixtures/sdxl-base-manifest
./build/iild-run inspect tests/fixtures/flux1-schnell-manifest
```

Those fixtures contain placeholder files, not model tensors. Likewise, a
successful `inspect` result means that metadata and required artifact paths are
consistent; it does not mean the weights were parsed or inference succeeded.

## Python reference oracle

The Python tools under `reference/diffusers/` are a validation oracle, not a
runtime dependency of the C++ library. They pin both the package versions and
the Hugging Face model revision used by the project's fixed red-cube fixture.

All supported generation values are exposed as CLI options, a typed JSON
`--config`, and Python `resolve_request()`, with shared/preset defaults
when omitted. `generate.py --print-config` shows the complete resolved values
without model downloads or a Torch installation. Sampling, scheduler settings,
secondary prompts, batching/seeds, dtypes, memory policies and optional
latent/embedding files are covered in [generation parameters](docs/generation-parameters.md).

```bash
uv venv reference/diffusers/.venv --python 3.14
uv pip install \
  --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements.txt

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset flux1-schnell

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset flux1-schnell
```

The generation default is `--device auto`; explicit choices include
`--device metal` (alias `mps`), `--device cuda`, `--device rocm`, and diagnostic `--device cpu`.
GPU preflight and the actual pipeline execution device are recorded in the
image sidecar. The Python oracle still uses PyTorch, independently of MLX.
CUDA defaults also permit TF32 on eligible Ampere-or-newer NVIDIA GPUs;
`--no-cuda-tf32` selects IEEE FP32 math without disabling FP16/BF16 Tensor
Core kernels. Eligibility and applied policy are recorded, not inferred
instruction utilization.
HIP uses Torch's `cuda` namespace internally but records the actual backend
as `rocm`, never NVIDIA. TF32 switches are rejected on AMD; use
`requirements-rocm.txt` after installing the correct vendor Torch wheel.

To compute text embeddings on CPU and store inactive weights in RAM while
keeping image denoising/decoding on the selected GPU:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base --cpu-text-encoding --cpu-threads 4 --offload model \
  --output build/reference/sdxl-cpu-ram.png
```

`--offload auto|none|model|sequential` controls weight residency separately from
CPU arithmetic. Existing model/VAE/LoRA/ControlNet inputs compose with these options.
Full CPU-only generation remains available through `--device cpu`.

Replace model, VAE, and LoRA weights independently during generation:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset sdxl-base \
  --model /absolute/path/to/sdxl-checkpoint.safetensors \
  --vae /absolute/path/to/sdxl-vae.safetensors \
  --lora /absolute/path/to/sdxl-style.safetensors \
  --lora-scale 0.75
```

The generation oracle accepts local single-file weights as well as the
existing Diffusers-directory and pinned Hub-model inputs. `--model-config`
selects the configuration and auxiliary-component source for a single-file
model; it defaults to the selected preset's pinned repository. SD 1.5 and
SDXL accept original-format checkpoints or denoiser-only files, while FLUX
accepts a transformer-only file. Each file must match the selected model
family. Both `.safetensors` and the local `.safetensor` spelling are accepted;
pickle checkpoints are not. This does not add single-file loading or image
generation to the C++ inspector. See
[`docs/model-inputs.md`](docs/model-inputs.md) for the full contract.

Apply only a local LoRA while retaining the preset's base weights:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora /absolute/path/to/adapter.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-lora.png
```

No adapter is bundled or selected automatically. Remote adapters require an
immutable commit SHA and exact safetensors filename. See
[`docs/lora.md`](docs/lora.md) for the complete input and provenance contract.

Load learned Textual Inversion tokens and reference them in prompt text:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 \
  --text-embedding /absolute/path/to/style.safetensors \
  --text-embedding-token '<style>' \
  --prompt 'a ceramic cup in <style> style'
```

`--text-embedding` accepts one or more local learned-token files and works
with SD 1.5, SDXL and FLUX, including ControlNet, CPU prompt encoding and
Hires Fix. Token names and target encoders can be selected explicitly.
The existing `--embeddings` option instead supplies completed prompt
tensors; the two input modes cannot be combined. See
[learned text embeddings](docs/text-embeddings.md) for file formats,
multi-vector tokens and encoder selection.

Add one ControlNet with a prepared conditioning image:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 \
  --controlnet /absolute/path/to/sd15-controlnet-package \
  --control-image /absolute/path/to/prepared-canny.png \
  --controlnet-scale 0.8
```

All three presets support a compatible single ControlNet from a local
Diffusers component package, pinned Hub repository, or local safetensors
file with its component configuration. Omitting `--controlnet` disables it;
when selected, strength defaults to 1.0 for the full denoising interval.
Inputs must already be the model's expected edges, depth, pose, or other
condition; no detector runs automatically. The default filename adds
`-controlnet`, and sidecars record weight/configuration/image hashes and
control settings. Model, VAE, LoRA, CPU prompt encoding, and RAM offload
remain composable. See [ControlNet inputs and limits](docs/controlnet.md).

Add Hires Fix for two-stage generation with every preset, including runs
with ControlNet, LoRA, or a replacement VAE:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 --width 512 --height 512 \
  --hires-fix --hires-scale 2 --hires-upscaler lanczos \
  --hires-denoising-strength 0.35 --hires-steps 30 \
  --hires-save-base
```

This generates at 512 × 512, resizes in RGB, then performs img2img diffusion
at 1024 × 1024 with the selected components. Final dimensions, resize
method, denoising strength, steps, seed, guidance, and scheduler are exposed
as parameters. Hires Fix is disabled by default; enabling it adds `-hires`
to default filenames, and `--hires-save-base` also preserves the first-stage
image. Strength selects the active portion of the second-stage schedule,
so `--hires-steps` is not the number of active refinement steps. See the
[Hires Fix contract](docs/hires-fix.md) for defaults, composition, metadata,
and quality-validation limits.

Downloads and generated outputs default to `build/reference/`, keeping the
source tree and the system disk free of multi-gigabyte model caches. See
[`reference/diffusers/README.md`](reference/diffusers/README.md) for the exact
commands and limitations.

## Project boundaries

The ownership boundary, dependency decisions, tensor contracts, and explicit
non-goals are recorded in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/stable-diffusion-v1.md`](docs/stable-diffusion-v1.md)
- [`docs/stable-diffusion-xl.md`](docs/stable-diffusion-xl.md)
- [`docs/flux1-schnell.md`](docs/flux1-schnell.md)
- [`docs/model-inputs.md`](docs/model-inputs.md)
- [`docs/lora.md`](docs/lora.md)
- [`docs/text-embeddings.md`](docs/text-embeddings.md)
- [`docs/controlnet.md`](docs/controlnet.md)
- [`docs/hires-fix.md`](docs/hires-fix.md)
- [`docs/hardware-compute.md`](docs/hardware-compute.md)
- [`docs/neural-accelerators.md`](docs/neural-accelerators.md)
- [`docs/dependencies.md`](docs/dependencies.md)

The project does not yet declare a distribution license. The SD 1.5 reference
uses CreativeML OpenRAIL-M, SDXL Base 1.0 uses CreativeML Open RAIL++-M, and
FLUX.1-schnell uses Apache-2.0; no model weights are stored in this repository.
