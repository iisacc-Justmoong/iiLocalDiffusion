# Hires Fix: two-stage image generation

The independent Python generator supports optional Hires Fix for every
existing generation family: `sd15`, `sdxl-base`, and `flux1-schnell`, each
with or without a compatible ControlNet. Model/checkpoint selection, VAE,
LoRA, learned Textual Inversion tokens, batch generation, precomputed
embeddings, CPU prompt encoding, and
device/offload policies compose with the two-stage path. Omitting
`--hires-fix` retains single-stage generation.

Hires Fix first generates at `--width` / `--height`, resizes the resulting
RGB images, and then performs image-to-image diffusion at the final size.
The second stage encodes the resized images with the selected VAE and adds
noise according to its denoising strength before refinement. This is an
executed second diffusion stage; choosing strength zero is rejected.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 --width 512 --height 512 \
  --hires-fix --hires-scale 2 --hires-upscaler lanczos \
  --hires-denoising-strength 0.35 --hires-steps 30 \
  --hires-save-base
```

This requests a 512 × 512 base image and a 1024 × 1024 final image.
The second-stage step setting is the full schedule length before img2img
strength selects its active portion. With the example's ordinary first-order
schedule, `int(30 * 0.35)` gives 10 refinement steps; it does not request 30
active refinement steps. Runtime scheduler metadata records what ran.

## Parameters and neutral defaults

All options are available through the CLI, JSON `--config`, and Python
`resolve_request()`, with the same precedence and validation as other
[generation parameters](generation-parameters.md). JSON keys use snake_case.
When Hires Fix is disabled, dependent options remain inactive/null; they do
not silently enable an additional pass. Explicit dependent settings require
`--hires-fix`.

For a disabled JSON starter, use `"hires_fix": false` and `null` for every
dependent value, including `hires_save_base`. An explicit
`"hires_save_base": false` or `--no-hires-save-base` is a selected dependent
setting and therefore still requires Hires Fix.

| Argument | Default when enabled / behavior |
|---|---|
| `--hires-fix` / `--no-hires-fix` | Disabled; enable both stages explicitly |
| `--hires-scale` | 2 when no final dimension is given; finite upscale factor greater than 1 |
| `--hires-width`, `--hires-height` | Final dimensions; one omitted axis is inferred from the base aspect ratio |
| `--hires-upscaler` | `lanczos`; choices `nearest`, `bilinear`, `bicubic`, `lanczos` |
| `--hires-denoising-strength` / `--hires-strength` | 0.35; finite value with `0 < strength <= 1` |
| `--hires-steps` | Inherit `--steps`; positive full schedule length, producing at least one active step under the family's img2img rule |
| `--hires-seed` | Inherit `--seed`; the existing `--seed-stride` applies to the second-stage batch |
| `--hires-guidance-scale` | Inherit `--guidance-scale`; the selected model's existing restrictions still apply |
| `--hires-true-cfg-scale` | Inherit `--true-cfg-scale`; finite value at least 1, with values above 1 available only for FLUX |
| `--hires-scheduler` | `auto`: inherit the first stage's actual scheduler class and configuration, with fresh runtime state |
| `--hires-scheduler-config` | `{}`; validated constructor overrides for the second-stage scheduler |
| `--hires-save-base` / `--no-hires-save-base` | False; additionally save the first-stage PNG and its sidecar |

Do not specify `--hires-scale` together with either final dimension. Explicit
dimensions must be positive multiples of 8 for SD/SDXL or 16 for FLUX.
Calculated dimensions use the nearest required multiple, with exact halfway
values rounded upward. Both
final axes must be at least their corresponding base axes, and at least one
must increase. Supplying both dimensions selects that exact shape; choosing
a different aspect ratio resizes the image to that shape.

For example, preserve the aspect ratio while choosing a final width:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base --width 1024 --height 768 \
  --hires-fix --hires-width 1536 \
  --hires-strength 0.4 --hires-steps 30 \
  --hires-guidance-scale 5 --hires-seed 1234
```

The inferred final height is 1152. The default upscale multiplier does not
apply when a final width or height is explicitly selected.

The same request can be supplied through JSON or `resolve_request()`:

```json
{
  "preset": "sdxl-base",
  "width": 1024,
  "height": 768,
  "hires_fix": true,
  "hires_width": 1536,
  "hires_denoising_strength": 0.4,
  "hires_steps": 30,
  "hires_guidance_scale": 5.0,
  "hires_seed": 1234
}
```

`hires_denoising_strength` is the canonical JSON/Python key;
`--hires-strength` is a CLI alias.

Larger denoising strength adds more noise and allows more change from the
base image; strength 1 uses the complete second-stage schedule. The choices
of strength, sampling steps, and resolution remain model- and task-dependent.
The active count is family- and scheduler-dependent. SD/SDXL require
`int(steps * strength) >= 1`. The pinned FLUX img2img implementation uses
`start = int(max(steps - min(steps * strength, steps), 0))` and selects the
schedule tail from `start * scheduler.order`; at least one resulting step
must remain. FLUX's defaults of 4 steps and strength 0.35 therefore select
two active refinement steps. The actual executed timesteps are recorded;
the schedule length alone is not an execution count. An explicit
`--hires-steps` can materially change the amount of refinement.

## Composition across both stages

The second stage reuses the selected neural components, including an
overridden VAE, active LoRA and its scales, and optional ControlNet. It does
not select or download a separate refinement checkpoint. The positive and
negative prompt inputs and supported attention values remain in effect;
stage-specific guidance overrides change only their documented guidance
values. Precomputed embeddings must satisfy the guidance requirements of
both stages. CPU prompt encoding and external embeddings retain the existing
batch policy, producing one refined image for each base image.

[Learned text embeddings](text-embeddings.md) loaded with
`--text-embedding` retain their registered tokenizer entries and encoder
vectors across both stages. Multi-vector tokens are expanded for each
encoder's prompt processing, including CPU text encoding. These learned
tokens use the text-encoding path and cannot be combined with the completed
prompt tensors supplied by `--embeddings`.

For FLUX, `--true-cfg-scale 1 --hires-true-cfg-scale 2 --negative-prompt blurry`
enables negative conditioning only during refinement; `--negative-prompt-2`
can be supplied as well. The shared negative text is passed only to a stage
whose true CFG is above 1. Explicit negative text remains invalid when both
stages have true CFG 1. Each saved stage's `fixture.negative_prompt` records
whether that stage used negative conditioning.

ControlNet receives the same prepared conditioning image at each stage's
resolution. Its scale, start/end interval, and supported guess/Union mode
remain active in both passes; the interval is applied to the active
denoising schedule of each stage. The generated base image becomes the
second-stage img2img input, while the conditioning image remains a separate
ControlNet input.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset flux1-schnell \
  --controlnet /absolute/path/to/schnell-controlnet-package \
  --control-image /absolute/path/to/prepared-condition.png \
  --controlnet-scale 0.8 \
  --hires-fix --hires-scale 1.5 --hires-upscaler bicubic \
  --hires-steps 8 --hires-strength 0.5 \
  --cpu-text-encoding --offload sequential
```

The existing `--latents` input initializes only the first stage. The second
stage derives its image latents from the resized first-stage output.
Likewise, `--timesteps`, `--sigmas`, and SDXL `--denoising-end` apply only to
the first stage. The second stage starts a fresh schedule selected by
`--hires-scheduler`, `--hires-scheduler-config`, `--hires-steps`, and strength.
`--guidance-rescale` also remains a first-stage option; the refinement pass
uses zero guidance rescaling. Its prompt guidance can instead be selected
with `--hires-guidance-scale` and, for FLUX, `--hires-true-cfg-scale`.
SDXL's second-stage positive and negative size conditioning uses the final
height/width, rather than retaining the base resolution.

Each stage starts fresh per-image generators. Image `i` uses
`seed + i * seed_stride` in the first stage and
`hires_seed + i * seed_stride` in the second. Equal stage seeds do not mean
equal noise tensors: img2img VAE sampling, image sizes, and ControlNet can
consume random values differently. Reproducibility remains dependent on the
runtime versions and hardware, as with single-stage generation.

## Output identity and execution evidence

Without an explicit output path, `-hires` follows any `-custom`, `-vae`,
`-lora`, `-embedding`, and `-controlnet` modifiers. An SD 1.5 base-model Hires Fix run
therefore writes `build/reference/sd15-red-cube-hires.png`. `--output` names
the final result. When `--hires-save-base` is enabled, the first-stage file
adds `-base` to the final stem: `result.png` has `result-base.png`; batched
`result-0001.png` has `result-0001-base.png`.

All final/base PNGs and their JSON sidecars participate in output collision
checks before generation. Default-path collisions select an unused run
suffix; explicit-path collisions fail unless `--overwrite` is enabled.
Publishing multiple output files retains the existing per-file atomic write
policy and is not a filesystem transaction.

Sidecars retain the resolved request in `parameters`. `hires_fix.base`
records first-stage size, seeds, pipeline/scheduler, actual denoising steps,
timesteps, finite-latent checks, component dtypes, safety state and pixel
hashes. `hires_fix.upscale` records the interpolation method, requested and
actual scaling, sizes and resized-image pixel hashes.
`hires_fix.refinement` records the second stage's settings, actual denoising
steps/timesteps, finite-latent checks and scheduler configuration. Final
PNG identity is in `output`; final safety and component dtypes are in
`safety` and `runtime.component_dtypes`. Optional base sidecars use
`artifact_role: "hires_base"`, describe the base image in those top-level
fields, and link to the completed image through `final_output`.
Both stages must produce the requested dimensions and pass the non-uniform
RGB output checks. Runtime latent callbacks establish that denoising ran
and reject non-finite latent values. Failure in the
second stage is an error; it does not silently return an upscale-only image
as a successful Hires Fix result.

Tests and smoke generation establish argument routing, both diffusion
stages, image dimensions, composition, and rejection of invalid output.
They cannot guarantee perceptual quality for every prompt, checkpoint, or
setting. Hires Fix provides the repeatable two-stage procedure and its
execution evidence; visual evaluation of the intended model and task remains
necessary.

## Dependencies and scope

Pillow performs the four supported RGB resize methods. Existing Diffusers
and PyTorch provide VAE encoding, noise, img2img scheduling and neural
execution, with Accelerate retaining ownership of offload hooks. No new
package, learned upscaler weights, or custom tensor/kernel implementation is
introduced. This interface does not expose arbitrary external img2img
inputs, latent-space upscaling, SDXL Refiner assembly, or a full native C++
generation pipeline.

The upstream contracts are described by the
[Diffusers image-to-image guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/img2img.md)
and [Pillow resize API](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize).
