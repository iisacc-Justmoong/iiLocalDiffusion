# Stable Diffusion v1 reference contract

## Canonical reference

The initial oracle is the Hugging Face repository
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
at immutable revision:

```text
451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

The repository identifies itself as a mirror of the deprecated RunwayML
repository and distributes weights under CreativeML OpenRAIL-M. The revision
pin makes the oracle reproducible; it does not transfer or broaden the model
license.

The first accepted package layout is:

```text
model-root/
├── model_index.json
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── text_encoder/
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

Other metadata and optional components may be present. Unknown keys are
preserved by the package and ignored by the current inspector.

## Component metadata

The pinned files declare:

| Component | Class | Contract used by the inspector |
|---|---|---|
| Tokenizer | `CLIPTokenizer` | maximum sequence length 77 |
| Text encoder | `CLIPTextModel` | hidden width 768; maximum positions 77 |
| Denoiser | `UNet2DConditionModel` | input/output channels 4; sample size 64; cross-attention width 768 |
| Image codec | `AutoencoderKL` | RGB input/output; latent channels 4; sample size 512 |
| Scheduler | `PNDMScheduler` | 1,000 training steps; scaled-linear beta schedule; PRK steps skipped |

The pinned VAE configuration does not contain `scaling_factor`. Diffusers
`AutoencoderKL` supplies the runtime default `0.18215`. This is a reference
runtime default, not a value proven solely from the model package, and a future
C++ VAE implementation must test it explicitly.

## Tensor flow

For batch size `B`, height `H`, and width `W` divisible by eight:

```text
positive prompt -> token ids                 [B, 77]
negative prompt -> token ids                 [B, 77]
CLIP outputs, each                           [B, 77, 768]
classifier-free guidance context             [2B, 77, 768]
initial latent                               [B, 4, H/8, W/8]
UNet latent input under guidance             [2B, 4, H/8, W/8]
UNet noise prediction                        [2B, 4, H/8, W/8]
guided latent after scheduler steps          [B, 4, H/8, W/8]
VAE-decoded image                            [B, 3, H, W]
```

At the canonical 512 x 512 resolution, the latent spatial shape is 64 x 64.
Layout notation above follows PyTorch's logical NCHW convention. A future
backend may store tensors differently internally, but its component boundary
must preserve the logical contract.

## Fixed reference fixture

```text
Prompt:             a red cube on a white table
Negative prompt:    empty string
Seed:               42
Resolution:         512 x 512
Inference steps:    20
Guidance scale:     7.5
Scheduler:          PNDMScheduler from the pinned repository
```

A seed is not a cross-backend noise specification. PyTorch, MLX, and other
runtimes may use different random algorithms, and GPU results can vary by
hardware and release. Component parity tests must eventually exchange the
actual initial latent in safetensors or NPY form with shape, dtype, layout,
hash, and numeric tolerances; a final PNG is not a sufficient diagnostic.

## Inspector guarantee

The C++ inspector validates metadata and locates non-empty safetensors files.
It does not validate safetensors serialization, tensor names, shapes, hashes,
or numeric values. `StableDiffusionV1` therefore means metadata compatibility,
not proof that the weights are Stable Diffusion 1.5 or executable.
