# Hires Fix: repeated image refinement

The independent Python preset generator supports optional Hires Fix for
the SD1/SDXL/FLUX families and their named derivatives, each
with or without a compatible ControlNet. Model/checkpoint selection, VAE,
LoRA, learned Textual Inversion tokens, batch generation, precomputed
embeddings, CPU prompt encoding, and
device/offload policies compose across the base and refinement passes. Omitting
`--hires-fix` retains single-stage generation.

Hires Fix first generates at `--width` / `--height`, resizes the resulting
RGB images, and then performs image-to-image diffusion at the resized size.
`--hires-passes N` selects the number of additional refinement passes after
the base image. It defaults to 1 when enabled, preserving the original
base-plus-one-refinement behavior. Each pass takes the preceding pass's
decoded image, resizes it, encodes it with the selected VAE and adds noise
before refinement. Every pass uses the same configured resize and img2img
settings with fresh scheduler and generator state. Strength zero is rejected.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 --width 512 --height 512 \
  --hires-fix --hires-passes 2 --hires-scale 2 --hires-upscaler lanczos \
  --hires-denoising-strength 0.35 --hires-steps 30 \
  --hires-save-base
```

This requests 512 × 512 base generation, a 1024 × 1024 first refinement,
and a 2048 × 2048 second refinement. The scale applies at every pass, so
additional passes increase both the work and the final image size.
The refinement step setting is the full schedule length before img2img
strength selects its active portion. With the example's ordinary first-order
schedule, `int(30 * 0.35)` gives 10 refinement steps; it does not request 30
active refinement steps per pass. Runtime scheduler metadata records what ran.

## Parameters and neutral defaults

The preset options below are available through the CLI, JSON `--config`, and Python
`resolve_request()`, with the same precedence and validation as other
[generation parameters](generation-parameters.md). JSON keys use snake_case.
When Hires Fix is disabled, dependent options remain inactive/null; they do
not silently enable an additional pass. Explicit dependent settings require
`--hires-fix`.

For a disabled JSON starter, use `"hires_fix": false` and `null` for every
dependent value, including `hires_passes` and `hires_save_base`. An explicit
`"hires_save_base": false` or `--no-hires-save-base` is a selected dependent
setting and therefore still requires Hires Fix.

| Argument | Default when enabled / behavior |
|---|---|
| `--hires-fix` / `--no-hires-fix` | Disabled; enable base generation followed by refinement explicitly |
| `--hires-passes` | 1; positive integer number of refinement passes after the base image; null when disabled |
| `--hires-scale` | 2 when no explicit first-refinement dimension is given; finite factor greater than 1 applied at every pass |
| `--hires-width`, `--hires-height` | First-refinement dimensions; infer an omitted axis from the base aspect ratio, then repeat the first pass's per-axis factors |
| `--hires-upscaler` | `lanczos`; choices `nearest`, `bilinear`, `bicubic`, `lanczos` |
| `--hires-denoising-strength` / `--hires-strength` | 0.35; finite value with `0 < strength <= 1` |
| `--hires-steps` | Inherit `--steps`; positive full schedule length, producing at least one active step under the family's img2img rule |
| `--hires-seed` | Inherit `--seed`; each refinement starts fresh per-image generators with the same seed and existing `--seed-stride` |
| `--hires-guidance-scale` | Inherit `--guidance-scale`; the selected model's existing restrictions still apply |
| `--hires-true-cfg-scale` | Inherit `--true-cfg-scale`; finite value at least 1, with values above 1 available only for FLUX |
| `--hires-scheduler` | `auto`: inherit the first stage's actual scheduler class and configuration, with fresh runtime state |
| `--hires-scheduler-config` | `{}`; validated constructor overrides shared by all refinement schedulers |
| `--hires-save-base` / `--no-hires-save-base` | False; additionally save the first-stage PNG and its sidecar |

Scheduler inheritance uses the class and configuration that actually ran,
including explicit base overrides. Preset scheduler defaults and the original
`--prediction-type` are not reapplied at refinement; an explicit
`--hires-scheduler-config` may override that inherited prediction configuration.

Do not specify `--hires-scale` together with either first-refinement dimension. Explicit
dimensions must be positive multiples of 8 for SD/SDXL or 16 for FLUX.
Calculated dimensions use the nearest required multiple, with exact halfway
values rounded upward. Both
axes at each pass must retain or enlarge their input axes, and at least one
must increase. Supplying both dimensions selects the first refinement's exact
shape. Its width/base-width and height/base-height factors are then repeated
independently, with family rounding at each pass. For example, 512 × 512 with
`--hires-width 1024 --hires-height 768 --hires-passes 2` produces 1024 × 768,
then 2048 × 1152. Explicit dimensions therefore name the final output size
only when there is one refinement pass.

For example, preserve the aspect ratio while choosing the first refinement width:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base --width 1024 --height 768 \
  --hires-fix --hires-passes 2 --hires-width 1536 \
  --hires-strength 0.4 --hires-steps 30 \
  --hires-guidance-scale 5 --hires-seed 1234
```

The first refinement is 1536 × 1152. Repeating its 1.5 factors produces a
2304 × 1728 final image. The default upscale multiplier does not apply
when a first-refinement width or height is explicitly selected.

The same request can be supplied through JSON or `resolve_request()`:

```json
{
  "preset": "sdxl-base",
  "width": 1024,
  "height": 768,
  "hires_fix": true,
  "hires_passes": 2,
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
input image; strength 1 uses the complete refinement schedule. The choices
of strength, sampling steps, and resolution remain model- and task-dependent.
The active count is family- and scheduler-dependent. SD/SDXL require
`int(steps * strength) >= 1`. The pinned FLUX img2img implementation uses
`start = int(max(steps - min(steps * strength, steps), 0))` and selects the
schedule tail from `start * scheduler.order`; at least one resulting step
must remain. FLUX's defaults of 4 steps and strength 0.35 therefore select
two active refinement steps. The actual executed timesteps are recorded;
the schedule length alone is not an execution count. An explicit
`--hires-steps` can materially change the amount of refinement.

## Composition across all preset stages

Every refinement pass reuses the selected neural components, including an
overridden VAE, active LoRA and its scales, and optional ControlNet. It does
not select or download a separate refinement checkpoint. The positive and
negative prompt inputs and supported attention values remain in effect;
stage-specific guidance overrides change only their documented guidance
values. Precomputed embeddings must satisfy the guidance requirements of
the base and refinement stages. CPU prompt encoding and external embeddings retain the existing
batch policy, producing one refined image for each base image.

[Learned text embeddings](text-embeddings.md) loaded with
`--text-embedding` retain their registered tokenizer entries and encoder
vectors across all stages. Multi-vector tokens are expanded for each
encoder's prompt processing, including CPU text encoding. These learned
tokens use the text-encoding path and cannot be combined with the completed
prompt tensors supplied by `--embeddings`.

For FLUX, `--true-cfg-scale 1 --hires-true-cfg-scale 2 --negative-prompt blurry`
enables negative conditioning only during refinement; `--negative-prompt-2`
can be supplied as well. The shared negative text is passed only to a stage
whose true CFG is above 1. Explicit negative text remains invalid when both
the base and refinement settings have true CFG 1. Each saved stage's `fixture.negative_prompt` records
whether that stage used negative conditioning.

ControlNet receives the same prepared conditioning image at each stage's
resolution. Its scale, start/end interval, and supported guess/Union mode
remain active in every pass; the interval is applied to the active
denoising schedule of each stage. Each refinement takes the previous
generated image as its img2img input, while the conditioning image remains
a separate ControlNet input.

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

The existing `--latents` input initializes only the base stage. Each refinement
derives its image latents from the resized output of the preceding stage.
Likewise, `--timesteps`, `--sigmas`, and SDXL `--denoising-end` apply only to
the base stage. Every refinement starts a fresh schedule selected by
`--hires-scheduler`, `--hires-scheduler-config`, `--hires-steps`, and strength.
`--guidance-rescale` also remains a first-stage option; each refinement pass
uses zero guidance rescaling. Its prompt guidance can instead be selected
with `--hires-guidance-scale` and, for FLUX, `--hires-true-cfg-scale`.
SDXL's refinement positive and negative size conditioning uses that pass's
height/width, rather than retaining the base or first-refinement resolution.

Each stage starts fresh per-image generators. Image `i` uses
`seed + i * seed_stride` in the first stage and
`hires_seed + i * seed_stride` in every refinement. The pass index does not
automatically change this seed. Equal stage seeds do not mean
equal noise tensors: img2img VAE sampling, image sizes, and ControlNet can
consume random values differently. Reproducibility remains dependent on the
runtime versions and hardware, as with single-stage generation.

## Managed local checkpoint backend

The public downloaded-file route also supports repeated HiRes refinement
through its managed ComfyUI image workflow:

```bash
python3 reference/generate.py \
  --model /absolute/path/to/illustrious.safetensors --base-model Illustrious \
  --width 1024 --height 1024 --steps 25 \
  --hires-fix --hires-passes 2 --hires-scale 1.5 \
  --hires-upscaler lanczos --hires-strength 0.35 --hires-steps 30 \
  --output-dir build/reference/illustrious-hires
```

This backend accepts `--hires-fix` / `--no-hires-fix`, `--hires-passes`,
`--hires-scale`, `--hires-upscaler`, `--hires-steps`, and
`--hires-denoising-strength` / `--hires-strength`. Enabled defaults are one
refinement, scale 2, Lanczos resizing and strength 0.35; refinement steps
inherit the base step count. Each pass resizes the preceding decoded image,
encodes it through the selected VAE and runs another refinement sampler.
Sizes are rounded to the image family's required multiples at every pass.
The native `SplitSigmasDenoise` node retains `round(steps * strength)`
refinement steps; at least one must remain. This is the managed scheduler's
rule, separate from the preset SD/SDXL and FLUX rules above. The managed
route validates steps in `[1, 4096]` and rejects any planned image axis above
the native `ImageScale` limit of 16384 before starting the server.

The managed backend does not accept the preset's `--hires-width`,
`--hires-height`, `--hires-seed`, guidance/true-CFG/scheduler overrides,
`--hires-scheduler-config`, or `--hires-save-base`. Such unsupported options
fail explicitly. It saves the final images and does not publish intermediate
refinements or an additional base image. Its supported base model, component,
device and output-directory rules remain those of the managed image route.

`request.hires` records the requested passes, scale, upscaler, strength,
steps and ordered target-size plan. The saved ComfyUI graph and its complete
execution result establish that the refinement chain ran, and final-image
validation checks the delivered artifact. Planned stage entries do not
claim per-stage pixel hashes or Diffusers denoising callback evidence.

## Preset output identity and execution evidence

Without an explicit output path, `-hires` follows any `-custom`, `-vae`,
`-lora`, `-embedding`, and `-controlnet` modifiers. An SD 1.5 base-model Hires Fix run
therefore writes `build/reference/sd15-red-cube-hires.png`. `--output` names
the final result. When `--hires-save-base` is enabled, the first-stage file
adds `-base` to the final stem: `result.png` has `result-base.png`; batched
`result-0001.png` has `result-0001-base.png`.
Only the final refinement is saved by default. `--hires-save-base` also
saves the original base image; intermediate refinement images are consumed
in memory and are not published as separate files.

All final/base PNGs and their JSON sidecars participate in output collision
checks before generation. Default-path collisions select an unused run
suffix; explicit-path collisions fail unless `--overwrite` is enabled.
Publishing multiple output files retains the existing per-file atomic write
policy and is not a filesystem transaction.

Sidecars retain the resolved request in `parameters`. `hires_fix.base`
records first-stage size, seeds, pipeline/scheduler, actual denoising steps,
timesteps, finite-latent checks, component dtypes, safety state and pixel
hashes. `hires_fix.requested_passes` and `hires_fix.completed_passes` distinguish
the requested refinement count from completed refinement work. The ordered
`hires_fix.stages` list records every refinement's `input`, `upscale`,
`refinement` and `output`. These carry image identity and sizes, interpolation
and scaling, actual denoising steps/timesteps, finite-latent checks and
scheduler configuration. Each entry's input is the preceding output, starting
with the base image. Existing `hires_fix.upscale` and `hires_fix.refinement`
fields remain available and describe the last refinement stage. Final
PNG identity is in `output`; final safety and component dtypes are in
`safety` and `runtime.component_dtypes`. Optional base sidecars use
`artifact_role: "hires_base"`, describe the base image in those top-level
fields, and link to the completed image through `final_output`.
Every stage must produce its requested dimensions and pass the non-uniform
RGB output checks. Runtime latent callbacks establish that denoising ran
and reject non-finite latent values. Failure in the
refinement sequence is an error; it does not silently return an earlier or upscale-only image
as a successful Hires Fix result.

Tests and smoke generation establish argument routing, the exercised diffusion
stages, image dimensions, composition, and rejection of invalid output.
They cannot guarantee perceptual quality for every prompt, checkpoint, or
setting. Hires Fix provides the repeatable refinement procedure and its
execution evidence; visual evaluation of the intended model and task remains
necessary.

## Dependencies and scope

Pillow performs the four supported RGB resize methods. Existing Diffusers
and PyTorch provide VAE encoding, noise, img2img scheduling and neural
execution, with Accelerate retaining ownership of offload hooks. No new
package, learned upscaler weights, or custom tensor/kernel implementation is
introduced for repeated passes. The managed backend reuses its existing
ComfyUI image resize, VAE and sampler nodes. This interface does not expose arbitrary external img2img
inputs, latent-space upscaling, SDXL Refiner assembly, or a full native C++
generation pipeline.

The upstream contracts are described by the
[Diffusers image-to-image guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/img2img.md)
and [Pillow resize API](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize).

## Repeat validation

The repeated path was executed with three refinements for SD1.5, SDXL and
FLUX, each with and without ControlNet. The ControlNet cases also retain LoRA,
CPU prompt encoding and two-image batches. These synthetic/tiny models verify
execution and composition, not the visual quality of arbitrary checkpoints.

- SD1.5/SDXL: `64 → 96 → 144 → 216` pixels per axis.
- FLUX: `64 → 96 → 144 → 224`, including its 16-pixel rounding requirement.
- Every stage had nonempty denoising, finite latents, non-uniform decoded images
  and input hashes equal to the preceding stage's output hashes.
- A later-stage failure propagates without returning a partial chain as success.
- The installed ComfyUI node schemas accepted 152 repeated workflow variants;
  fixture inventory names in that check do not prove model-weight inference.

Evidence is retained in `build/reference/repeated-hires/six-paths/verification.json`
and `build/workflow-source-review/hires-repeat-live-schema-validation.json`.
The public managed CLI also generated a non-uniform 432×432 PNG from the existing
original SDXL checkpoint through `128 → 192 → 288 → 432`, using three separate
refinement samplers. Original/staged file identities and output SHA256 were
verified, and the owned engine process terminated successfully. This is recorded
in `build/reference/repeated-hires/managed-sdxl/local-image.json`; the declared
Illustrious category in that fixture is not evidence of Illustrious-trained weights.
The CMake build and 53 CTest cases passed. Python discovery ran 572 cases:
570 passed and the two pre-existing opt-in checkpoint-conversion cases were skipped.
`build/hires-repeat-ctest.log` and `build/hires-repeat-tests.log` retain the logs.
