# Diffusers reference oracle

These scripts preserve a known-good implementation against which future C++
components can be compared. They are not called by the iiLocalDiffusion
library.

## Reproducible environment

The verified host uses macOS arm64 and CPython 3.14. Create the environment on
the external workspace volume:

```bash
uv venv reference/diffusers/.venv --python 3.14
uv pip install \
  --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements.txt
```

For AMD Radeon, use a separate environment under `build/`, install the
GPU/OS-compatible HIP PyTorch wheel first, then install
`requirements-rocm.txt`. It reuses the common direct pins without imposing
the standard Torch version. The [Radeon guide](../../docs/radeon-rocm.md)
includes native C++ setup and platform limitations. A model-free check is:

```bash
python reference/diffusers/hardware.py --device rocm --dtype float16
```

Both scripts use the shared immutable presets in `presets.py`:

| Preset | Model revision | Resolution | Guidance |
|---|---|---:|---:|
| `sd15` (default) | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | 512 x 512 | 7.5 |
| `sdxl-base` | `462165984030d82259a11f4367a4eed129e94a7b` | 1024 x 1024 | 5.0 |
| `flux1-schnell` | `741f7c3ce8b383c54771c7003378a50191e9efe9` | 1024 x 1024 | 0.0 |

They use only safetensors and disable remote code. The default cache is
`build/reference/huggingface`, and the Xet chunk cache defaults to
`build/reference/huggingface-xet`, not the system-volume Hugging Face cache.

Inspect and persist component configs without running denoising:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py
```

Generate the fixed 512 x 512 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py
```

Inspect SDXL Base and generate its fixed 1,024 x 1,024 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset sdxl-base
```

Inspect FLUX.1-schnell and generate its fixed 1,024 x 1,024 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset flux1-schnell

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset flux1-schnell
```

The official FLUX.1-schnell repository is gated even though its model license
is Apache-2.0. Accept its terms while signed in on Hugging Face, then make the
same account available to `huggingface_hub` before running these commands.

## Hardware policy

Generation defaults to `--device auto`, which requires an available CUDA,
ROCm, or Metal device. CUDA/HIP takes precedence; on Apple Silicon the PyTorch device is
`mps`. `--device metal` is an alias for `mps`. A CPU-only host must explicitly
select `--device cpu`; no automatic CPU fallback is permitted. An explicitly
unavailable accelerator fails instead of selecting another device.

Before loading large weights, a matrix-product preflight verifies the selected
device and dtype. The pipeline's actual `_execution_device` is checked after
placement/offload setup. `runtime.hardware` in the JSON sidecar records the
requested and selected device, backend, runtime, GPU flag, preflight result,
and no-fallback policy. `PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected for Metal
runs. Out-of-memory or unsupported-kernel errors are not retried on CPU.

`--device rocm` requires an actual HIP build and usable AMD GPU. HIP uses
the Torch `cuda` device namespace, including for offload and CPU conditioning
transfers, but the sidecar records `backend: rocm` with AMD GPU/HIP properties.
Both TF32 CLI switches are NVIDIA-only and fail on ROCm; auto does not touch
NVIDIA precision setters on AMD. All existing model/VAE/LoRA, CPU text and
RAM offload options remain applicable.

This path uses PyTorch, not MLX. The separate C++ MLX component API and its
Metal/CUDA build options are documented in
[hardware-compute.md](../../docs/hardware-compute.md). A complete MLX diffusion
generator is not supplied by these scripts.

### NVIDIA Tensor Core policy

CUDA availability is not treated as proof of Tensor Cores. The oracle checks
device family and compute capability, excludes GTX 16-series and ROCm from
affirmative NVIDIA claims, and records FP16/BF16/TF32 eligibility at
`runtime.hardware.tensor_cores`. Kernel dispatch remains PyTorch's choice;
`usage_verified` stays false without an external kernel profile.

TF32 matrix/convolution math is enabled by default on eligible Ampere-or-newer
NVIDIA GPUs. Use `--no-cuda-tf32` for IEEE FP32 multiplication or
`--cuda-tf32` to explicitly require TF32 capability. These flags require CUDA
and do not disable FP16/BF16 Tensor Core kernels. They apply before GPU
preflight/model loading and preserve LoRA/CPU-encoding/offload order. Numeric
results may differ with TF32; the sidecar records the actual applied policy.

Apple Neural Engine execution is a separate compiled-model C++ Core ML path,
not an MPS alias or an automatic transformation of this Python image pipeline.
See [Neural Engine / Tensor Cores](../../docs/neural-accelerators.md).

### CPU computation and RAM offload

`--cpu-text-encoding` computes prompt embeddings on CPU while keeping
denoising and VAE decoding on the GPU. It runs after base validation and LoRA
activation, before offload hooks. `--cpu-threads N` controls PyTorch CPU
intra-op parallelism (positive integer; default retains PyTorch's setting).

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base --device auto --cpu-text-encoding --cpu-threads 4 \
  --offload model --output build/reference/sdxl-cpu-ram.png
```

`--offload` independently selects `auto`, `none`, `model`, or `sequential`.
Model offload retains inactive whole models in RAM, while sequential offload
transfers submodules as called. Both execute those models on the GPU; RAM
storage alone is not CPU arithmetic. `auto` preserves the original presets
unless CPU text encoding is requested: then SD/SDXL use model offload and
FLUX retains sequential offload. Explicit `none` moves the entire pipeline
to the GPU, including already-used text encoders. `--device cpu` still runs
the whole pipeline on CPU/RAM and rejects model/sequential GPU offload.

CPU encoding uses float32 for SD/SDXL and bfloat16 for FLUX, restores the
encoder storage dtype, and transfers all embeddings to the chosen GPU/dtype.
The sidecar records CPU conditioning devices/shapes, CPU thread count,
participating devices, requested/effective offload, and weight storage.
CPU text encoding is a stage before GPU denoising, not concurrent denoiser
sharding. Neither CPU cooperation nor offload guarantees a speedup. Apple
unified RAM remains the same finite physical memory; offload does not make
oversized models fit automatically. See the
[resource contracts](../../docs/hardware-compute.md#cpugpu-cooperation-and-ram-storage).

## Replace model components

`generate.py --model` accepts a Diffusers directory, a pinned Hub repository,
or an explicit local `.safetensors`/`.safetensor` file. `--vae` independently
replaces the VAE with a local single file, and `--lora` adds an optional
adapter. For example, use an SDXL checkpoint and separately selected VAE:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset sdxl-base \
  --model /absolute/path/to/sdxl-checkpoint.safetensors \
  --vae /absolute/path/to/sdxl-vae.safetensors \
  --lora /absolute/path/to/sdxl-style.safetensors \
  --lora-scale 0.8
```

SD 1.5 and SDXL accept original-format checkpoints containing their neural
components or a standalone UNet. FLUX accepts a standalone transformer. The
loader classifies the role from tensor keys; a VAE or LoRA passed as the model
is rejected. Native Diffusers component state dictionaries and supported
original-format component weights are delegated to Diffusers' single-file
loaders. This is weight replacement within the chosen architecture, not
cross-family conversion.

Single-file models still need configs, tokenizers, scheduler settings, and
any neural components not embedded in the file. `--model-config` selects that
Diffusers-format source, defaulting to the preset's pinned repository:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --model /absolute/path/to/flux-schnell.safetensor \
  --model-config /absolute/path/to/flux-schnell-diffusers-package \
  --vae /absolute/path/to/flux-vae.safetensors \
  --local-files-only
```

A remote `--model-config` override requires its immutable
`--model-config-revision`, independently of `--revision` for a remote
`--model`. With a directory/Hub model and `--vae`, the VAE config comes from
that model source instead; `--model-config` is only for a single-file model.
An original SD checkpoint uses its embedded UNet and text encoders and its
embedded VAE unless overridden. A denoiser-only file instead uses the config
source's text encoders and default VAE. Neither case is self-contained merely
because the denoiser weights are a local file.

VAE injection happens while constructing the pipeline, before compatibility
validation, LoRA activation, or device/offload setup. Model and VAE file paths,
resolved targets, SHA-256, byte sizes, configuration source, and component
origins are recorded in the result metadata. The exact accepted inputs and
failure boundaries are documented in
[`model-inputs.md`](../../docs/model-inputs.md).

## Optional LoRA

Apply one local LoRA safetensors file after the selected base pipeline has
passed its contract:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora /absolute/path/to/style.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-style.png
```

For a remote LoRA, pin both its commit and exact file:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora owner/adapter-repository \
  --lora-revision 0123456789abcdef0123456789abcdef01234567 \
  --lora-weight-name adapter.safetensors \
  --lora-scale 1.0
```

Only an explicitly provided adapter is loaded. The loader forces safetensors,
uses the fixed internal name `iild_lora`, activates the finite scale with
`set_adapters()`, verifies registration, and applies device/offload policy
afterward. Optional CPU text encoding occurs between activation and offload,
so encoder LoRA weights affect the computed embeddings. Local adapters are
identified by SHA-256 and byte size; remote
adapters require an immutable revision. A default LoRA output uses a separate
`*-lora.png` name; custom-model and VAE suffixes compose with it. See the
[`LoRA generation contract`](../../docs/lora.md) for compatibility and
licensing limits.

Use `--local-files-only` after the pinned snapshot is cached. Existing outputs
are not overwritten unless `--overwrite` is explicit. A remote `--model`
override must also provide a 40-character commit `--revision`; a local model
directory or file rejects `--revision`. A local `.safetensor` input is
temporarily exposed to Diffusers through a `.safetensors` symlink so it cannot
select Diffusers' pickle branch. The target's identity is checked before and
after loading, and no weight bytes are copied or rewritten for this alias.

Without `--output`, an explicit `--model` or `--revision` adds `-custom`,
`--vae` adds `-vae`, and `--lora` adds `-lora`, in that order. For example, an
SDXL run with all three inputs defaults to
`build/reference/sdxl-base-red-cube-custom-vae-lora.png`. Existing output and
sidecar files still require `--overwrite` before replacement.

## Interpretation

`inspect_pipeline.py` loads actual component weights through Diffusers and
writes normalized metadata to `build/reference/pipeline-inspection.json` for
SD 1.5, `build/reference/pipeline-inspection-sdxl-base.json` for SDXL Base, or
`build/reference/pipeline-inspection-flux1-schnell.json` for FLUX.1-schnell.
It verifies the selected pipeline's required component classes; for SDXL it
also checks the critical 2,048/2,816 conditioning widths and VAE contract. For
FLUX.1-schnell it checks both tokenizers and text encoders, the 64-channel
packed-latent transformer contract, 16-channel shifted VAE, and flow-matching
scheduler.
Its `--model` remains a Diffusers directory or pinned repository; the new
single-file and VAE inputs belong to `generate.py` only. C++ `iild-run inspect`
continues to inspect package metadata without loading weights.

`generate.py` writes an RGB PNG and a JSON sidecar containing the model preset
and revision, all fixture parameters, direct package versions, device, dtype,
attention-slicing state, safety-checker and watermarker presence, and output
SHA-256. Custom runs also identify the model's weight role, configuration
source, VAE override, and each component's source. Its `adapters` array is empty
for a base run or records the exact LoRA
source, identity, scale, registered components, and active state. For presets
that support it, attention slicing is enabled
automatically on MPS to reduce peak memory use and can be changed explicitly.
For SDXL on fp16 accelerators, the pinned Diffusers pipeline's verified
`force_upcast` path performs VAE decode in float32. Loading only the VAE as
float32 up front is intentionally avoided: Diffusers 0.40's current MPS branch
casts it back to the latent dtype and can produce NaNs. The oracle also rejects
uniform RGB output so that this failure cannot pass merely because a PNG was
written.

FLUX.1-schnell uses BF16 without an `fp16` weight variant, passes no negative
prompt, and caps `max_sequence_length` at 256. The fixed fixture uses four
steps, while positive step overrides remain available for diagnostic runs.
Guidance is fixed at 0.0, and width and height must be multiples of 16. On MPS
and CUDA the default `--offload auto` keeps the full pipeline off the
accelerator, enables VAE
slicing and tiling, and uses sequential CPU offload. Attention slicing is
deliberately rejected for this preset. The optimization sidecar records that
these switches were enabled, not that a particular VAE branch executed; at
batch one and exactly 1,024 square, slicing or tiling may not activate.
CPU offload describes where inactive weights are stored: each neural component
executes on the selected GPU when called. Tokenization, seed generation, file
I/O, and output readback remain explicit host work, not GPU fallback.
With `--cpu-text-encoding`, text encoders instead run explicitly on CPU before
the offloaded pipeline receives their precomputed embeddings.

The full 12B official snapshot has not been loaded on the verified 32 GiB M1
Max host. Its BF16 weights leave too little headroom for Python, activations,
and the operating system, so swap pressure or process termination remains
possible even with sequential offload. A tiny FLUX pipeline verifies the API
and MPS path only; it is not a production-model memory or quality result.

The CPU generator reduces one source of platform drift but does not make PNGs
pixel-identical across PyTorch versions or accelerators. Future numerical
parity fixtures should export initial latents and intermediate tensors as
safetensors or NPY plus explicit shape, dtype, layout, hashes, and tolerances.

The SD 1.5 repository's CreativeML OpenRAIL-M terms, the SDXL Base repository's
CreativeML Open RAIL++-M terms, and FLUX.1-schnell's Apache-2.0 terms apply
independently of the Python package licenses. SDXL Base and FLUX.1-schnell
contain no safety checker. The fixed SDXL oracle explicitly disables its
optional invisible watermarker; FLUX.1-schnell has no watermarker component.
Those facts are not a product safety policy.
