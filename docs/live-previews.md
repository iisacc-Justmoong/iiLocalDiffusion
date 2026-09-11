# Live denoising previews

Pass `--preview-dir /absolute/empty/temporary-directory` to the local checkpoint,
preset, or generic Diffusers image runner. No additional dependency or model is
needed: the existing Diffusers callback, Torch, VAE and Pillow produce a preview
after every actual denoising step. The original sampler latents and RNG state are
preserved. This performs extra VAE decoding and therefore adds generation time.

SDXL accelerator defaults enable the existing Diffusers VAE tiling path for
both previews and final decoding. The bundled VAE tiles dimensions above
1024px to limit decoder activation memory; this does not rescale the sampler's
latents or the requested output. Explicit `--no-vae-tiling` remains supported.

The directory must be new or empty, outside the final output directory. A frame
is atomically published as `step-000001.png` before its newline-delimited stdout
event is flushed:

```text
IILD_PREVIEW {"schema": "iild-preview-v1", "step": 1, "total_steps": 20, "image": "step-000001.png"}
```

Steps are one-based and `total_steps` counts the actual scheduler timesteps,
which can differ from the requested sampling steps. Each RGB preview preserves
its aspect ratio with a maximum dimension of 512 pixels. A batch previews its
first image. Events are mixed with ordinary logs; consumers must buffer partial
lines and only accept the versioned event prefix. Every filename is unique within
the request so image caches cannot hide successive frames. The caller owns
preview cleanup; Dreamscapes uses a per-job temporary directory and removes it
after success, failure or cancellation. Previews never constitute successful
generation or appear in the final asset manifest.

The decoder handles spatial AutoencoderKL latents (including SD/SDXL scaling,
mean/std normalization and VAE upcasting) and FLUX.1 packed latents with its
pipeline unpacker and VAE shift. The pipeline must expose
`callback_on_step_end` with `latents` and a VAE decoder. Unsupported pipelines
report an explicit error. Remote requests, animation and multi-pass HiRes are
not supported by this single-pass image preview option.

`GenerationPreviewTests` checks real tensors, immutable per-step PNGs, latent/RNG
isolation, normalization, FP16 restoration, FLUX unpacking and invalid paths.
Run it with the installed Torch/Diffusers Python environment. The consumer must
still verify the final process exit, image and generation manifest separately.

`Cannot preview non-finite denoising latents` means the sampler already produced
NaN or infinity before preview decoding. It remains a hard failure; skipping
the preview or replacing invalid values would conceal a failed generation.
For SDXL on MPS, the default processor is PyTorch SDPA because FP16 sliced QK
scores can overflow. Explicit slicing upcasts score accumulation through
Diffusers while retaining FP16 weights. `GenerationPreviewTests` reproduces
the overflow with finite FP16 inputs on CPU and, when available, MPS, and checks
finite probabilities after the precision policy is applied.
The same SDXL/MPS slicing failure is documented in
[Diffusers issue 11229](https://github.com/huggingface/diffusers/issues/11229);
the SDK policy is covered by the local tensor regression and real-checkpoint
verification in `docs/standalone-validation.md`.

The callback contract follows the official
[Diffusers pipeline callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback)
documentation; final VAE scaling follows the installed Diffusers pipeline code.
