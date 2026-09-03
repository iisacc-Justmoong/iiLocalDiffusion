# Generation values and neutral defaults

The independent Python generator exposes the supported SD 1.5, SDXL Base and
FLUX.1-schnell text-to-image values through CLI arguments, a data-only JSON
file, and the same Python request resolver. No extra dependency is introduced:
argument parsing and JSON use the standard library; loading, scheduling,
encoding and tensor operations stay in the existing Diffusers/PyTorch APIs.
This does not add a complete diffusion pipeline to the native C++ library.

## Omission is valid; an invalid explicit value is not replaced

No generation argument is required. Omitted values and top-level JSON
`null` use the documented defaults. Explicit `0`, `false` and empty text
are retained, not mistaken for missing input. Unknown keys, wrong JSON types,
duplicate keys, non-finite numbers and incompatible combinations fail before
model loading where the necessary information is available. Architecture,
scheduler and tensor-shape checks that depend on the loaded model remain
runtime checks.

Precedence is **explicit CLI > JSON configuration > preset/shared defaults**.
Boolean controls support both `--name` and `--no-name`, including
`--no-cpu-text-encoding`, `--no-local-files-only` and `--no-overwrite`.
Configuration keys use the CLI destination in snake_case, for example
`num_images`, `guidance_scale`, `vae_tiling`. File-relative paths are
resolved against the JSON file's directory; CLI paths use the working
directory. Quoted `~` paths are expanded before output/cache creation, so
the exported path and the actual filesystem target agree. Config files are
limited to 1 MiB and cannot recursively load
other config files.

Inspect or export the complete resolved configuration without importing
Torch, checking GPUs, downloading a model, creating caches or generating:

```bash
python reference/diffusers/generate.py --print-config
python reference/diffusers/generate.py --preset sdxl-base --print-config
python reference/diffusers/generate.py --preset flux1-schnell --print-config
```

The printed flat JSON can be saved and supplied to `--config`. Dynamic
`auto` hardware policies remain `auto` until a real execution; the sidecar
also records their actual resolved runtime values. Explicit local tensor
files are still checked and hashed during request resolution.

For example, a partial configuration is sufficient:

```json
{
  "preset": "sdxl-base",
  "prompt": "a ceramic cup on a plain table",
  "negative_prompt": "",
  "seed": 0,
  "num_images": 2,
  "width": null,
  "height": null,
  "guidance_rescale": 0.0,
  "vae_tiling": false
}
```

```bash
python reference/diffusers/generate.py --config build/generation.json \
  --steps 24 --device rocm --cpu-text-encoding --offload model
```

A minimal reusable file is supplied at
[`generation.example.json`](../reference/diffusers/generation.example.json).
The Python entry point resolves exactly the same schema:

```python
# With reference/diffusers on sys.path:
from generate import resolve_request, configuration_values

preset, request = resolve_request({
    "preset": "sdxl-base",
    "seed": 0,
    "width": None,
    "vae_tiling": False,
})
resolved_values = configuration_values(request)
# resolve_request() with no mapping also returns a valid default request.
```

## Model-specific defaults

| Value | SD 1.5 | SDXL Base | FLUX.1-schnell |
|---|---|---|---|
| `--preset` | `sd15` (global default) | `sdxl-base` | `flux1-schnell` |
| `--width`, `--height` | 512 × 512 | 1024 × 1024 | 1024 × 1024 |
| `--steps` | 20 | 20 | 4 |
| `--guidance-scale` | 7.5 | 5.0 | 0.0; Schnell has no Dev guidance embedding |
| Automatic GPU dtype | float16 | float16 | bfloat16 |
| Automatic CPU/text dtype | float32 | float32 | bfloat16 |
| `--max-sequence-length` | Not applicable | Not applicable | 256; explicit range 1–512 |
| Automatic weight variant | fp16 when available on GPU | fp16 when available on GPU | None |
| Automatic GPU offload | Resident, or model offload with CPU encoding | Same | Sequential |
| Automatic GPU VAE slicing/tiling | Off / off | Off / off | On / on |

These are existing compatible reference defaults, not universal optimum
settings. The simple fallback prompt is `a red cube on a white table`,
without a style/persona/artist modifier. Negative text is empty, LoRA is
absent, one image is produced, and additional rescaling/early stopping is
disabled. A fixed seed of 42 provides a reproducible starting point rather
than imposing a visual style.

## Prompt, sampling and batch values

| Argument | Default / behavior |
|---|---|
| `--prompt`, `--negative-prompt` | Simple cube prompt / empty string |
| `--prompt-2`, `--negative-prompt-2` | SDXL/FLUX inherit their respective primary text; explicit values are forwarded |
| `--seed` | 42; PyTorch's range [-2^63, 2^64−1] |
| `--num-images` / `--num-images-per-prompt` | 1; positive images for this prompt, all saved |
| `--seed-stride` | 1; image i receives seed + i × stride; 0 intentionally repeats |
| `--generator-device` | `cpu`; `execution` requires CUDA/ROCm |
| `--guidance-rescale` | 0; SD/SDXL range [0,1] |
| `--eta` | 0; nonzero requires a scheduler with an eta parameter, such as DDIM |
| `--clip-skip` | None; native CLIP selection, checked against loaded encoder depths |
| `--true-cfg-scale` | 1; FLUX values >1 enable explicit two-pass CFG and negative prompts |
| `--scheduler` | `auto`; otherwise an installed compatible Diffusers scheduler class name |
| `--scheduler-config` | `{}`; strict JSON constructor overrides |
| `--timesteps` | None; descending non-negative SD/SDXL integer schedule |
| `--sigmas` | None; descending finite noise schedule, mutually exclusive with timesteps |
| `--denoising-end` | None; SDXL early-stop fraction strictly between 0 and 1 |
| `--cross-attention-kwargs` | `{}`; SD/SDXL portable call-time LoRA `scale` |
| `--joint-attention-kwargs` | `{}`; FLUX portable call-time LoRA `scale` |

The scheduler is replaced only after the loaded base contract passes.
Replacement uses its runtime `compatibles` classes and `from_config`;
configuration keys must belong to that constructor, not arbitrary Python
objects or internal metadata. A misspelled key or unsupported schedule/eta
is rejected rather than silently ignored. Example:

```bash
python reference/diffusers/generate.py --scheduler DDIMScheduler --eta 0.2 \
  --scheduler-config '{"timestep_spacing":"trailing"}' --steps 20
```

Custom schedules take precedence over the step-count request. The sidecar
retains both the request and the pipeline-reported timestep count, plus the
scheduler's timetable. Early stopping may use only part of that timetable.
The meanings of sigma endpoints are scheduler-specific; the generator does
not manufacture or resample an explicitly provided schedule.

Attention JSON currently accepts only the portable finite `scale` field
supported by these pipelines, and requires a selected LoRA. It is a
call-time scale distinct from `--lora-scale`; both are recorded. CPU text
encoding forwards the same scale. Processor-specific tensors, GLIGEN and
IP-Adapter setup are not introduced by accepting this JSON.

FLUX's `guidance_scale=0` contract is unchanged; `true_cfg_scale` is a
separate optional second denoiser pass, not Dev guidance embeddings.
Sequence lengths above Schnell's default 256 are an explicit experiment,
not a newly verified full-model quality setting.

The installed Transformers 5.16.1 CLIP exposes its final normalization at the
root, while Diffusers 0.40's SD 1.5 CLIP-skip branch still looks under
`text_model`. A scoped non-Module view references that same normalization
during encoding and is removed even on failure. It neither registers a
duplicate neural module nor changes weight keys/math; the applied
compatibility path is recorded. Nested CLIP layouts and SDXL are untouched.

For SDXL, the following size values use `HEIGHT WIDTH` order and default
to the output dimensions:
`--original-size`, `--target-size`, `--negative-original-size`,
`--negative-target-size`. Both `--crops-coords-top-left` and
`--negative-crops-coords-top-left` use `TOP LEFT` and default to `0 0`.
These condition the model; they do not crop the saved PNG.

## Model, hardware, memory and output values

Existing `--model`, `--revision`, `--model-config`,
`--model-config-revision`, `--vae`, `--lora`, `--lora-revision`,
`--lora-weight-name`, and `--lora-scale` remain available in all three
input forms. Omitted model/config selections use the preset's pinned source;
VAE comes from that source, and no LoRA is loaded. A selected LoRA defaults
to scale 1.0. See [model inputs](model-inputs.md) for file layouts and identity.

| Argument | Default / behavior |
|---|---|
| `--device` | `auto`: compatible CUDA/ROCm first, then Metal; no silent CPU fallback |
| `--device-index` | 0; bounds-checked CUDA/ROCm index, CPU/Metal require 0 |
| `--dtype` | `auto`, or explicit float32/float16/bfloat16 |
| `--weight-variant` | `auto`, `none`, or an explicit filename variant; independent of conversion dtype |
| `--low-cpu-mem-usage` | True for model, VAE and LoRA loaders |
| `--cpu-text-encoding` | False |
| `--cpu-text-dtype` | `auto`; an explicit dtype requires CPU text encoding |
| `--cpu-threads` | Retain the installed runtime default; explicit count must be positive |
| `--offload` | `auto`; overrides are none/model/sequential |
| `--cuda-tf32` | Automatic eligible NVIDIA policy; explicit TF32 switches are rejected on AMD |
| `--attention-slicing` | Preset/device policy; incompatible FLUX requests fail |
| `--attention-slice-size` | `auto`; also accepts max or positive integer |
| `--vae-slicing`, `--vae-tiling` | Preset/device policy, with explicit enable/disable |
| `--watermark` | SDXL false; explicit true needs its optional installed dependency |
| `--progress` | True; false hides generation progress |
| `--cache-dir`, `--xet-cache-dir` | Repository build/reference/huggingface and huggingface-xet |
| `--local-files-only` | False |
| `--output` | Preset filename under build/reference; custom/vae/lora suffixes compose |
| `--overwrite` | False |
| `--png-compress-level` | 6, range 0–9; lossless |
| `--png-optimize` | False; lossless storage optimization |

CPU text encoding remains after LoRA activation and before GPU/offload
placement. SD/SDXL expand one prompt's embeddings during the pipeline call;
FLUX expands CPU embeddings first and receives a call-time multiplier of
one. This prevents dropped images or accidental squared batch sizes.
VAE slicing/tiling metadata records the enabled policy, not proof that a
size-dependent branch ran. Explicit memory/dtype changes can still exhaust
RAM/VRAM; defaults are not a guarantee that a large model fits.

Without an explicit output path, collisions receive `-run-0002`,
`-run-0003`, etc., preserving previous evidence. Batches receive
`-0001.png`, `-0002.png`, etc. Every PNG has its own JSON sidecar, image
seed and hash, complete batch membership, resolved `parameters`, scheduler
configuration and tensor input identities. Explicit output collisions fail
before dependency/model loading unless overwrite was requested.
Atomic non-overwrite publication uses same-directory hard links; an
unsupported filesystem or an I/O failure is an explicit error, not permission
to overwrite. A multi-file batch is not a filesystem transaction.

## Optional initial latents and precomputed text

`--latents FILE` accepts a local safetensors file. `--latents-key`
defaults to `latents`. Omission samples fresh noise from the configured
generator. SD/SDXL files contain unscaled initial noise with shape
`[num_images, unet_channels, height/vae_factor, width/vae_factor]`.
FLUX files contain already packed noise with shape
`[num_images, (height/(2*vae_factor))*(width/(2*vae_factor)), transformer_channels]`.
The dimensions come from the loaded model; a wrong shape is not resized.

`--embeddings FILE` accepts `prompt_embeds` and, where applicable,
`pooled_prompt_embeds`, `negative_prompt_embeds`,
`negative_pooled_prompt_embeds`. `--embedding-keys` defaults to `{}`,
meaning those canonical keys, and can map them to other file keys:

```bash
python reference/diffusers/generate.py --embeddings /absolute/path/text.safetensors \
  --embedding-keys '{"prompt_embeds":"positive","negative_prompt_embeds":"negative"}' \
  --latents /absolute/path/noise.safetensors --latents-key noise
```

Prompt tensors have shape `[1 or num_images, tokens, embedding_dimension]`;
pooled tensors have shape `[same_batch, projection_dimension]`. Guidance
requires the matching negative tensors. A pre-expanded batch is not expanded
again. FLUX single-prompt tensors are repeated explicitly. External
embeddings replace text encoding and cannot be combined with CPU text
encoding or CLIP skip; textual prompt fields are recorded but not executed.
Omitting the embedding file uses the normal text encoder with its defaults.

Only selected keys are materialized, and shape/dtype/finite-value checks
precede inference. Safe file identity is verified before and after reading;
tensors are copied before the mapped file is released. Both
`.safetensors` and `.safetensor` are accepted. Malformed files never
fall back to fabricated zero embeddings or random noise.

## Deliberately preserved boundaries

Defaults prevent **missing-parameter** errors, not missing hardware, model
files, authentication or licenses. A custom remote repository still needs
its immutable revision; a LoRA directory/repository still needs an exact
weight name. The program does not guess an unrelated checkpoint or silently
download a replacement after an explicit file fails.

Safetensors-only loading, disabled remote code, trained architecture checks,
RGB PNG output and inference-only execution remain contracts, not unsafe
JSON switches. Model structural constants come from the validated model
configuration. Callbacks/code objects, arbitrary attention processors,
IP-Adapter, ControlNet, image-to-image and SDXL refiner assembly are not added.

The API routing was checked against the installed Diffusers 0.40.0 source
and its official [SD pipeline documentation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img),
[SDXL documentation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl),
[FLUX documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux) and
[scheduler guide](https://huggingface.co/docs/diffusers/using-diffusers/schedulers).
Unit tests use simulated tensors/devices; actual cached model smoke runs
provide separate execution evidence, not image-quality or all-hardware proof.
