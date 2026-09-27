# Architecture-aware generation

The public `iild-generate` launcher chooses an execution path from the model
container and request. Local checkpoint tensors are inspected without loading
their values; a Diffusers directory is inspected through `model_index.json`.
Filenames are not architecture evidence. Civitai metadata may name a variant,
but must agree with recognized tensors. Header inspection is not a complete
weight compatibility check and is never reported as a successful generation.

## Local image backends

`--backend local --engine auto` retains the established Diffusers SD1/SDXL
presets, and selects the in-process native engine for the other contracts below
when no Diffusers configuration was requested. `--engine native` explicitly
uses native inference for any supported contract, including SD1 and SDXL.
An explicit Diffusers configuration/package continues to use Diffusers.

| Architecture | Native text component slots | VAE contract |
| --- | --- | --- |
| SD 1.x | `clip_l` | SD1 |
| SD 2.x | `clip_l` (the matching OpenCLIP weights) | SD2 |
| SDXL, Illustrious, NoobAI, Pony | `clip_l`, `clip_g` | SDXL |
| SD3 / SD3.5 | `clip_l`, `clip_g`, `t5xxl` | SD3 |
| FLUX.1 dev / schnell / Krea | `clip_l`, `t5xxl` | FLUX.1 |
| FLUX.2 dev | `llm` (matching Mistral) | FLUX.2 |
| FLUX.2 Klein | `llm` (matching Qwen3 size) | FLUX.2 |
| Z-Image Base / Turbo | `llm` (Qwen3) | FLUX.1 |
| Qwen Image | `llm` (matching Qwen VL) | Qwen Image RGB |
| Chroma | `t5xxl` | FLUX.1 |
| Krea 2 | `llm` (Qwen3-VL) | Qwen Image RGB, native Wan tensor representation |
| Anima | `llm` | Anima / Qwen Image RGB |

These adapters use the actual architecture implementations in the pinned
stable-diffusion.cpp dependency. They do not resize incompatible tensors or
convert every model into SDXL. Complete checkpoints can use embedded encoders
and VAEs. Split safetensors/GGUF files accept explicit companion files. GGUF
quantization is decoded by the native engine; legacy pickle containers must
first use the existing safe conversion path.

```sh
iild-generate --list-backends
iild-generate --inspect-model --model /models/z-image.safetensors
iild-generate --model /models/z-image.safetensors \
  --base-model ZImageTurbo \
  --components '{"llm":"/models/qwen3-4b.safetensors","vae":"/models/ae.safetensors"}' \
  --prompt 'A cabin beside a lake' --output-dir /outputs/cabin
```

`--inspect-model --components '{...}'` reports the selected native backend,
architecture, embedded/missing component slots and defaults without starting
inference. `--validate-only` resolves all local file identities without loading
the model. A foreground worker preparation loads and validates the real native
context. Generation validates and publishes the resulting RGB images.

The same tensor layout can represent Base/Turbo or distilled variants. An
architecture signature alone cannot establish its training recipe. Supply
matching `--base-model` metadata where available, or explicit `--steps`,
`--guidance-scale` and `--embedded-guidance`. Native `guidance-scale` is standard
CFG; embedded guidance is a separate flow-model input. Krea 2's published
Diffusers guidance uses `cond + scale*(cond-uncond)`, so native standard CFG is
that value plus one. The explicit `--krea2-variant raw|turbo` contract resolves
Raw to 52 steps / native CFG 4.5 and Turbo to 8 steps / native CFG 1.
Unknown variants are not inferred from filenames. See [Krea 2 controls](krea2.md)
for precision, flow shift, sigma schedules and native V3.

SD1/2/XL support `--prediction-type epsilon|v_prediction`; recognized prediction
metadata is propagated to the native context. Changing prediction type rebuilds
the context. Flow architectures retain their own native prediction contract.
Krea 2's `text_encoder_select_layers` pipeline configuration is validated as
layer indices rather than misinterpreted as an importable model component.

New native contracts use a single denoising pass at the requested output size
(rounded to the engine canvas grid, then cropped to the exact requested size).
The existing complete-Anima V1 route preserves its earlier Hires behavior.
The C++ V2 component API explicitly controls Hires. Unsupported editing,
inpainting and RGBA/layered requests require their matching Diffusers pipeline
with image/mask inputs; they are not silently interpreted as text-to-image.

## Pipeline packages and output publication

A local directory with `model_index.json` selects the installed built-in
Diffusers class. This covers additional image, video, audio and tensor pipelines
through the existing typed input/output system. `--inspect-model` also accepts
these directories. Package inspection reports configuration evidence; actual
class availability, required inputs, companion tensors and output types are
validated by the loader. Custom downloaded Python code is not executed.

The native worker publishes PNG files, per-image provenance and
`iild-standalone-image-v1` `generation.json`. Provenance includes architecture,
backend plan, component identities, seed, CFG, distilled guidance, Hires mode,
timing, dimensions and output SHA-256. The final directory is published only
after the entire batch passes output checks. Existing Diffusers image/video/
audio/tensor manifests keep their existing schema and media contracts.

## C++ and C interfaces

`NativeModelComponents` and `generateNativeImageWithComponents` add explicit
CLIP-L/OpenCLIP, CLIP-G, T5, LLM and VAE files, CFG/distilled guidance, CPU/auto
placement and preparation-only execution. Existing C++ structure layouts and
entry points are unchanged. `iild_native_request_v2` embeds the V1 image request
and adds these options. `iild_native_generate_v2` uses the same owned result,
metadata, RGB, progress, preview, cancellation and free functions as V1.

All supplied files participate in residency identity and post-generation
mutation checks. Changing a companion binding invalidates the old context.
Known VAE contracts are validated before loading; newer families without an
auto-mount contract require an explicit VAE and the full native context loader
validates its tensors. No companion is replaced with arbitrary values.

## Validation

`BackendRegistryTests` uses bounded real safetensors fixtures for original and
Diffusers tensor names, component completeness, metadata conflicts, replay and
the Python/C V2 boundary. `NativeResultTests` and `NativeMobileResultTests`
compile the real C++ adapter against a controlled engine fixture to verify
component forwarding, sampling parameters, cache invalidation, dimensions,
cancellation, ownership and V1/V2 ABI rejection. These tests do not measure
pretrained-model quality. Real runtime smoke artifacts are kept in `build/`;
record the specific family and weights used rather than claiming that all
possible checkpoints have been exercised.

`tests/BackendRuntimeSmoke.py` runs SD3, FLUX.1, FLUX.2 Klein, Z-Image, Qwen
Image and Krea 2 through the public router using real tiny randomly initialized
Diffusers networks and explicit conditioning tensors. It checks tensor-family
inspection, actual denoising/decoding, nonconstant 64x64 RGB PNGs and output
manifests without downloading weights. Its `build/backend-runtime-smoke/results.json`
is computation/export evidence, not a pretrained quality or native-engine benchmark.

References: [pinned native backend](https://github.com/leejet/stable-diffusion.cpp/tree/d04e8950c1ec8d30248cbe996682b3182fb1adf6),
[Z-Image pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/z_image),
[Krea 2 pipeline and CFG convention](https://huggingface.co/docs/diffusers/main/api/pipelines/krea2),
[FLUX.2 pipelines](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2).
