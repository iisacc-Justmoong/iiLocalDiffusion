# Stable Diffusion XL Base reference contract

## Canonical reference

The SDXL oracle is the official Hugging Face repository
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
at immutable revision:

```text
462165984030d82259a11f4367a4eed129e94a7b
```

The repository declares the CreativeML Open RAIL++-M license. The revision pin
makes the oracle reproducible; it does not transfer or broaden the model
license. SDXL 0.9 has different access and license terms and is not this
profile.

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
├── unet/
│   ├── config.json
│   └── *.safetensors
├── vae/
│   ├── config.json
│   └── *.safetensors
└── scheduler/
    └── scheduler_config.json
```

Each tokenizer directory must retain its own `tokenizer_config.json`,
`vocab.json`, and `merges.txt`. The two tokenizers share vocabulary content in
the pinned repository but have different padding configuration, so an
inference implementation must not collapse them into one configured object.

## Component metadata

The C++ `StableDiffusionXLBase` profile validates the official Base 1.0
prompt-to-image package contract:

| Component | Class | Contract used by the inspector |
|---|---|---|
| Tokenizer 1 | `CLIPTokenizer` | maximum sequence length 77 |
| Tokenizer 2 | `CLIPTokenizer` | maximum sequence length 77 |
| Text encoder 1 | `CLIPTextModel` | hidden width 768; projection width 768; maximum positions 77 |
| Text encoder 2 | `CLIPTextModelWithProjection` | hidden and projection width 1,280; maximum positions 77 |
| Denoiser | `UNet2DConditionModel` | latent channels 4; sample size 128; cross-attention width 2,048; `text_time` added conditioning |
| Image codec | `AutoencoderKL` | RGB input/output; latent channels 4; sample size 1,024; scale 0.13025; forced upcast |
| Scheduler | `EulerDiscreteScheduler` | 1,000 training steps; beta 0.00085 to 0.012; scaled-linear schedule; epsilon prediction; leading timestep spacing |

The UNet's `addition_time_embed_dim` is 256 and its
`projection_class_embeddings_input_dim` is 2,816. These values encode two
cross-component invariants:

```text
768 + 1280 = 2048
6 * 256 + 1280 = 2816
```

The first is the concatenated token-conditioning width. The second combines
six micro-conditioning values with the pooled projection from text encoder 2.
The inspector also requires `force_zeros_for_empty_prompt=true` from the
pipeline index and `use_linear_projection=true` from the UNet.

## Tensor flow

For prompt batch `B`, effective image batch `E`, and classifier-free guidance
factor `G` (`2` when enabled, otherwise `1`):

```text
token IDs from each tokenizer                    [B, 77]
text encoder 1 penultimate hidden state          [B, 77, 768]
text encoder 2 penultimate hidden state          [B, 77, 1280]
concatenated prompt embeddings                   [G*E, 77, 2048]
pooled embeddings from text encoder 2            [G*E, 1280]
micro-conditioning IDs                           [G*E, 6]
initial latent                                   [E, 4, H/8, W/8]
UNet latent input and noise prediction           [G*E, 4, H/8, W/8]
guided latent                                    [E, 4, H/8, W/8]
VAE-decoded image                                [E, 3, H, W]
```

The six micro-conditioning values are original height, original width, top
crop, left crop, target height, and target width. At the canonical 1,024 by
1,024 resolution, the latent spatial size is 128 by 128. VAE decode divides
the latent by 0.13025 and uses float32 because `force_upcast` is true.

## Fixed reference fixture

```text
Prompt:             a red cube on a white table
Negative prompt:    empty string
Seed:               42
Resolution:         1024 x 1024
Inference steps:    20
Guidance scale:     5.0
Scheduler:          EulerDiscreteScheduler from the pinned repository
Watermarker:        explicitly disabled
```

Twenty steps are an iiLocalDiffusion smoke/reference choice, not the
Diffusers call default. An empty negative prompt is also intentional: in SDXL
it is not equivalent to `None`, because the pinned pipeline zeros negative
embeddings only when the prompt is absent.

## Compatibility boundary

`StableDiffusionXLBase` means that the canonical Base 1.0 package metadata and
required artifact paths match. It does not mean that safetensors bodies were
parsed, a coherent fp16/fp32 variant was selected, or inference ran. The
single-file checkpoint, ONNX, OpenVINO, Flax, auxiliary VAE, SDXL Refiner,
img2img, and arbitrary derivative contracts are outside this profile.

The Python oracle performs the stronger next check: Diffusers loads actual
weights, verifies component classes and critical tensor configuration, and
can generate the fixed 1,024 by 1,024 MPS fixture. It records that SDXL Base
has no safety checker and whether any watermarker was present rather than
claiming an unperformed safety step.
