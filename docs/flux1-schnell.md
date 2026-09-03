# FLUX.1-schnell reference contract

## Canonical reference

The FLUX oracle is Black Forest Labs' official Hugging Face repository
[`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
at immutable revision:

```text
741f7c3ce8b383c54771c7003378a50191e9efe9
```

The corresponding
[`741f7c3` snapshot tree](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/741f7c3ce8b383c54771c7003378a50191e9efe9)
is the metadata source for this profile.

The repository declares Apache-2.0. It is nevertheless gated on Hugging Face,
so users must accept the repository terms and authenticate before downloading
the official weights. This profile does not accept FLUX.1-dev: Dev embeds
guidance in the transformer and has different licensing and scheduler
semantics.

The values below follow the official
[`FluxPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/flux) and
[`FluxTransformer2DModel`](https://huggingface.co/docs/diffusers/api/models/flux_transformer)
contracts as well as the pinned repository metadata.

The accepted Diffusers layout is:

```text
model-root/
├── model_index.json
├── tokenizer/
├── tokenizer_2/
├── text_encoder/
│   ├── config.json
│   └── *.safetensors
├── text_encoder_2/
│   ├── config.json
│   └── *.safetensors
├── transformer/
│   ├── config.json
│   └── *.safetensors
├── vae/
│   ├── config.json
│   └── *.safetensors
└── scheduler/
    └── scheduler_config.json
```

Tokenizer 1 requires `tokenizer_config.json`, `vocab.json`, and `merges.txt`.
Tokenizer 2 requires `tokenizer_config.json`, `spiece.model`, and
`tokenizer.json`. Sharded safetensors require the canonical index JSON, whose
`weight_map` must reference a complete, consecutive shard set. Every required
neural component must have at least one non-empty regular artifact.

The package index names tokenizer 2 `T5TokenizerFast`. Transformers 5.x aliases
that implementation to the runtime class name `T5Tokenizer`, while 4.x reports
`T5TokenizerFast`; the Python oracle accepts either runtime spelling.

## Component metadata

The C++ `Flux1Schnell` profile validates this prompt-to-image contract:

| Component | Class | Contract used by the inspector |
|---|---|---|
| Tokenizer 1 | `CLIPTokenizer` | maximum sequence length 77 |
| Tokenizer 2 | `T5TokenizerFast` in the package index | T5 runtime tokenizer; maximum sequence length 512 |
| Text encoder 1 | `CLIPTextModel` | hidden/projection width 768; intermediate width 3,072; 12 heads; 12 layers; vocabulary 49,408 |
| Text encoder 2 | `T5EncoderModel` | model width 4,096; feed-forward width 10,240 with `gated-gelu`; 64 heads; 24 layers; vocabulary 32,128 |
| Denoiser | `FluxTransformer2DModel` | packed channels 64; 19 dual-stream and 38 single-stream blocks; 24 heads of width 128; joint width 4,096; no guidance embeddings |
| Image codec | `AutoencoderKL` | RGB input/output; latent channels 16; sample size 1,024; scale 0.3611; shift 0.1159; four encoder/decoder levels |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | 1,000 training steps; sequence range 256 to 4,096; base/max shift 0.5/1.15; fixed shift 1.0 |

The canonical transformer's rotary axes are `[16, 56, 56]`. Diffusers may
materialize this constructor default even when the source JSON omits it, so the
C++ package inspector validates the value when present while the loaded Python
oracle validates the effective runtime value. Transformer output channels may
likewise be absent or null in JSON; the effective value must be 64.

## Tensor flow

For prompt batch `B`, image batch `E`, and image dimensions `H` and `W`:

```text
CLIP token IDs                                  [B, 77]
T5 token IDs                                    [B, <=256]
pooled CLIP conditioning                        [E, 768]
T5 token conditioning                           [E, <=256, 4096]
VAE latent                                      [E, 16, H/8, W/8]
2 x 2 packed latent sequence                    [E, (H/16)*(W/16), 64]
transformer noise prediction                    [E, (H/16)*(W/16), 64]
unpacked latent                                 [E, 16, H/8, W/8]
VAE-decoded image                               [E, 3, H, W]
```

The transformer consumes 64 channels because each token packs a 2 by 2 region
of the 16-channel VAE latent. Consequently, supported image width and height
must be positive multiples of 16.

## Fixed reference fixture

```text
Prompt:                 a red cube on a white table
Negative prompt:        not passed
Seed:                   42
Resolution:             1024 x 1024
Inference steps:        4
Guidance scale:         0.0
Maximum prompt length:  256
Scheduler:              FlowMatchEulerDiscreteScheduler from the pinned repository
```

Schnell is optimized for high-quality generation in one to four denoising
steps; four is the reproducible fixture choice rather than a hard pipeline
limit. The oracle uses BF16 for both CPU and accelerator loading and does not
request a nonexistent `fp16` repository variant. On MPS and CUDA it applies
sequential CPU offload plus VAE slicing and tiling; it does not move the
complete pipeline to the accelerator as one resident allocation.
These are defaults, not hidden constants: dtype, token limit, memory policy,
seed/batch and scheduler values can be supplied through CLI/JSON. Optional
`--true-cfg-scale > 1` enables a separate negative-prompt CFG pass without
changing the Schnell `guidance_scale=0` contract. See
[generation parameters](generation-parameters.md) for values and fallback rules.

## Compatibility boundary

`Flux1Schnell` means that the canonical Schnell package metadata and required
artifact paths match. It does not mean safetensors bodies were parsed, the
authenticated official weights were downloaded, or inference ran. GGUF,
single-file checkpoints, quantized derivatives, ControlNet, FLUX.1-dev, and
arbitrary third-party FLUX layouts are outside this C++ package profile.
LoRA likewise remains outside the C++ package profile.

The independent Python generation oracle accepts a compatible Schnell
transformer file through `--model`, a separate `AutoencoderKL` file through
`--vae`, and one optional `--lora`. Its `--model-config` source supplies the
CLIP/T5 encoders, tokenizers, scheduler, component configs, and default VAE if
not replaced. A single transformer file is not a self-contained FLUX pipeline.
Native Diffusers and supported original-format transformer weights use the
pinned Diffusers single-file loader; `.safetensors` and local `.safetensor`
filenames are accepted. All assembled components must still satisfy the
Schnell contract before LoRA activation and generation. This does not admit
Dev guidance embeddings, quantized weights, or alternate latent contracts.
See [model-inputs.md](model-inputs.md) for source selection and provenance.

The Python oracle performs the stronger check by loading actual weights,
verifying effective component classes and critical configuration, and
generating an RGB PNG. FLUX.1-schnell includes neither a safety checker nor a
watermarker; package compatibility must not be presented as a product safety
policy.

The current 32 GiB M1 Max verification host has run the same API path with a
pinned tiny FLUX fixture, but has not loaded the official 12B weights because
the repository requires authenticated gate access. The tiny run proves
framework and MPS call compatibility only; it does not prove official-model
memory fit, numerical correctness, image quality, or production readiness.
