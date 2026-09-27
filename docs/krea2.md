# iisacc Krea 2 model ecosystem

Krea 2 Raw and Turbo are the standard ecosystem for new iisacc image workflows.
This is an architecture and execution contract; it does not relabel existing
SDXL/Illustrious merges as Krea 2 or project incompatible tensors into Krea weights.
Existing model families remain available for existing callers.

## Variant and precision policy

The published Raw quality example uses 52 steps and Krea guidance 3.5. Turbo
uses 8 steps and Krea guidance 0. Raw is the upstream recommendation for LoRA
training and Turbo for inference. These are defaults, not locked parameters.
See the [official implementation](https://github.com/krea-ai/krea-2) and
[Diffusers contract](https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2).

| Control | Diffusers package | Native checkpoint |
| --- | --- | --- |
| Variant | `model_index.json` boolean `is_distilled`, or explicit `--krea2-variant` when absent | explicit `--krea2-variant raw\|turbo`; auto uses the Raw quality sampling preset |
| Raw defaults | 52 steps, guidance 3.5 | 52 steps, standard CFG 4.5 |
| Turbo defaults | 8 steps, guidance 0 | 8 steps, standard CFG 1 |
| Precision | `--dtype float32` default, or `bfloat16`; FP16 refused | source checkpoint precision, including supported quantization |
| Resolution | multiples of 16, 16–8192 contract limit; device capacity still applies | 8px output grid, 64–2048; internal canvas rounded up to 64px |
| Sampler | FlowMatchEulerDiscreteScheduler | `--native-sampler euler\|heun`; auto resolves to Euler |
| Extra denoising pass | no implicit HiRes pass | no implicit HiRes pass |

Krea guidance is `cond + g * (cond - uncond)`; native standard CFG is `g + 1`.
Turbo at zero Krea guidance does not use negative conditioning. Set a nonzero
Krea guidance explicitly when exploring negative prompts; this departs from the
published Turbo default. Filename substrings are not evidence of distillation.
Conflicting explicit variant and package metadata fail before inference.

FP32 describes runtime arithmetic, not recovery of information already lost in
INT8/FP8/GGUF weights. BF16 retains a wider exponent range than FP16 but fewer
mantissa bits than FP32. High fidelity comparisons should use original unquantized
weights, identical seed, conditioning, VAE, dimensions and sampling schedule.
Neither numerical checks nor tiny fixtures guarantee pretrained image quality.

## Conditioning and VAE

The model uses Qwen3-VL text conditioning and a 16-channel Qwen Image RGB VAE,
factor 8, patch size 2 (64 packed transformer input channels). The official
encoder selection uses 12 tapped layers; fine-tunes must preserve their trained
selection and encoder. The loader checks selected layer count against the
transformer and indices against encoder depth. Explicit embeddings are BSLC
(batch, sequence, selected layer, channel), with a matching boolean BS mask.

An SDXL/FLUX VAE cannot replace this VAE. The native engine names its shared
Qwen/Wan tensor representation `wan`; this is not permission to use an arbitrary
Wan VAE. The native engine validates companion tensor compatibility at loading.
Native split weights use `--components` with `llm` and `vae`. Diffusers accepts
its existing `--vae` selection; that path uses the Qwen Image RGB validator.
No SDXL aesthetic LoRA is injected into Krea 2. Compatible explicit adapters use
the existing `--lora` / `--lora-scale` loader with activation checks. Arbitrary
SDXL, FLUX.1 Krea or FLUX.2 adapters are not Krea 2 adapters. Unsupported edit,
ControlNet and inpainting inputs require an actual compatible pipeline and are
not treated as text-to-image controls.

## Flow shift and schedules

`--krea2-mu` overrides the exponential shift (finite 0–4). Without it, Turbo uses
1.15 and Raw computes `0.5 + (image_tokens - 256) * 0.65 / (6400 - 256)`, with
`image_tokens = (width / 16) * (height / 16)`. Diffusers requires the published
dynamic exponential scheduler configuration; unrelated scheduler overrides fail.

`--sigmas '[1.0,0.75,0.4,0.1]'` specifies strictly descending, finite **unshifted**
sigmas in (0,1]. Their count sets the steps unless an identical step count is
explicitly supplied. Native CLI applies `exp(mu)*s/(1+(exp(mu)-1)*s)` and appends
terminal zero. Diffusers performs that transformation in its scheduler. The native Krea 2
default also supplies the explicit shifted `1, …, 1/steps, 0` schedule so
upstream discrete-scheduler defaults cannot change it silently. JSON
configuration replay preserves variant, mu, sigma list and native sampler.

The additive C++ `NativeSamplingControls` / `generateNativeImageWithSampling`
and C `iild_native_request_v3` / `iild_native_generate_v3` preserve V1/V2 ABI.
V3 sampler values are 0=automatic, 1=Euler, 2=Heun; `flow_shift=+infinity` retains
the upstream default. **Direct V3 custom sigmas are already shifted**, have
`steps + 1` entries and end at zero. Invalid size, pointer, enum, NaN, ordering
or step count is rejected. Schedules are per request and do not reload weights.

## Examples

Complete local Diffusers package:

```sh
iild-generate --model /absolute/local/krea2-turbo-package \
  --krea2-variant turbo --dtype float32 --device cpu \
  --prompt 'a vivid red flower in daylight' --seed 42 \
  --width 1024 --height 1024 --output-dir /absolute/empty/output
```

Native compatible split checkpoint (choose actual matching companion files):

```sh
iild-generate --model /absolute/local/krea2-turbo.safetensors \
  --engine native --krea2-variant turbo --native-sampler euler \
  --components '{"llm":"/absolute/local/qwen3-vl.safetensors","vae":"/absolute/local/qwen-image-vae.safetensors"}' \
  --prompt 'a vivid red flower in daylight' --seed 42 \
  --width 1024 --height 1024 --output-dir /absolute/empty/output
```

## Evidence and limits

Diffusers checks finite input embeddings/latents, loaded precision, masks and
latent packing, then checks every denoising callback for NaN/Inf. Failure aborts
publication; temporary scheduler/callback hooks are restored even on failure.
`generation.json` records variant, requested precision, actual component dtypes,
resolved mu, actual scheduler sigmas and executed step count. Native records the
resolved sampling request and source precision policy; its upstream interface
does not expose all latent tensors, so it does **not** claim per-step finite
validation (`finite_latents_required=false`).

`Krea2ContractTests.py` checks preflight/default contracts; `Krea2RuntimeTests.py`
uses real tiny random Krea2 transformer/VAE/scheduler components to check seed
repeatability, Raw negative conditioning, effective mu changes, invalid inputs,
NaN rejection and hook restoration. `BackendRegistryTests.py` checks native
resolution and replay; `NativeResultTests.cpp` checks actual adapter-to-engine
V3 parameter delivery and V1/V2 behavior. `BackendRuntimeSmoke.py --families krea2`
checks the public router through a real PNG using tiny random weights.
These tests do not validate the visual quality of the full pretrained model.

## Embedded ComfyUI encoder compatibility

Complete Krea2 checkpoints may use `text_encoders.qwen3vl_4b.transformer.*`.
The preflight recognizes this exact namespace as an embedded LLM for Krea2,
and the native loader maps it to `text_encoders.llm.*` before ordinary text,
vision, and quantization-scale conversion. Existing files are not rewritten.
`BackendRegistryTests` and `NativeKrea2NamesTests` cover both boundaries; similar
but unrelated namespace names are not accepted as this embedded component.

Bare native checkpoint auto mode uses the Raw quality sampling preset so desktop
requests do not require a CLI-only flag. This is not training-variant detection:
`variant_source=native-quality-default` records the decision. Explicit Turbo
remains available, filenames do not select it, and Diffusers packages retain
the strict `is_distilled` metadata contract.

ComfyUI Qwen encoders can contain scaled FP8 E4M3/E5M2 weights. Metal supports
storage and conversion for them but has no direct FP8 matrix kernel in the
pinned engine. The capability query now rejects direct FP8 multiplication,
allowing Linear to cast to BF16 and apply its stored scale.
`NativeFp8MetalTests` executes both FP8 formats through the real Linear graph
and checks scaled output values on the actual Metal backend.

## QuickGenerate aspect ratios

Native Krea 2 accepts Dreamscapes QuickGenerate's existing output sizes:
1024×1024, 1368×1024, 1024×1368, 1824×1024, and 1024×1824.
The native engine rounds each internal canvas axis up to 64 pixels and
center-crops the decoded RGB to the requested output, without resampling.
For 9:16 this is a 1024×1856 canvas cropped to 1024×1824 (16 pixels from
each vertical edge). The resolution-dependent Raw mu uses the internal
canvas token count. Reports record `output_size`, `canvas_size`, and
`output_transform`; replay retains the original output dimensions.
Diffusers package dimensions retain their separate 16px contract.
