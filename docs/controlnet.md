# ControlNet generation contract

The independent Python generator accepts one optional ControlNet and one
prepared local conditioning image with `sd15`, `sdxl-base`, or
`flux1-schnell`. Omitting `--controlnet` keeps the original generation path.
No ControlNet model or image is selected by default. This feature does not
add ControlNet inspection or image generation to C++.

## Generate with a conditioning image

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 \
  --controlnet /absolute/path/to/sd15-controlnet-package \
  --control-image /absolute/path/to/prepared-canny.png \
  --controlnet-scale 0.8
```

The image must already represent the input expected by that model, such as
Canny edges, a depth map, or pose guidance. The generator decodes one static
image, applies EXIF orientation, converts it to RGB, and lets Diffusers resize
it to `--width` and `--height`. It does not calculate edges, depth, or pose
from a photograph. Animated images and failed decoding are rejected.
The same conditioning image is used for every image in `--num-images`.

| Preset | Required ControlNet component | Pipeline | Image argument used internally |
|---|---|---|---|
| `sd15` | `ControlNetModel` | `StableDiffusionControlNetPipeline` | `image` |
| `sdxl-base` | `ControlNetModel` | `StableDiffusionXLControlNetPipeline` | `image` |
| `flux1-schnell` | `FluxControlNetModel` | `FluxControlNetPipeline` | `control_image` |

Class identity and tensor-interface configuration are checked against the
selected base before attachment. An SDXL ControlNet cannot be used with SD
1.5 merely because both use `ControlNetModel`. A structurally compatible
FLUX ControlNet still needs weights suitable for Schnell; accepting its
architecture does not establish training compatibility or image quality.

## Arguments and defaults

| Argument | Omitted value / accepted input |
|---|---|
| `--controlnet` | Disabled; otherwise local Diffusers component directory, pinned Hub repository, or local safetensors file |
| `--controlnet-revision` | Required for a remote ControlNet: 40-character lowercase commit SHA; invalid for a local source |
| `--control-image` | Required when ControlNet is selected; non-empty local static image |
| `--controlnet-config` | Single-file component configuration: sibling `config.json` by default, otherwise an explicit local directory or pinned Hub repository |
| `--controlnet-config-revision` | Immutable commit for a remote component configuration; independent of base and ControlNet revisions |
| `--controlnet-variant` | No package filename variant; explicit values such as `fp16` are independent of `--weight-variant` |
| `--controlnet-scale` / `--controlnet-conditioning-scale` | 1.0 when selected; finite and non-negative, 0 gives zero conditioning strength |
| `--control-guidance-start` | 0.0 when selected |
| `--control-guidance-end` | 1.0 when selected; requires `0 <= start < end <= 1` |
| `--guess-mode` / `--no-guess-mode` | False when selected; enabling it requires SD/SDXL |
| `--control-mode` | None; required for a FLUX Union model, using that model's non-negative mode index |

Guidance start/end specify fractions of denoising steps. Scale zero still
loads and executes the selected ControlNet; omit `--controlnet` to disable
the feature. Dependent controls require `--controlnet`, even when explicitly
set to a default value. FLUX Union mode IDs are model-specific and checked
against `num_mode`; a non-Union model rejects `--control-mode`.

## Weight and configuration sources

A component package has a root `config.json` declaring the expected
ControlNet class and standard Diffusers safetensors weights. Base-pipeline
packages and ControlNet component packages are separate selections. Remote
weights use an immutable snapshot; replace the example ID and commit with
the actual compatible model and its revision:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base \
  --controlnet owner/sdxl-controlnet-repository \
  --controlnet-revision 0123456789abcdef0123456789abcdef01234567 \
  --control-image /absolute/path/to/prepared-depth.png
```

Local single files accept `.safetensors` and `.safetensor`. A sibling
`config.json` supplies the component configuration when present. Otherwise,
pass `--controlnet-config`; unlike `--model-config`, it does not default to
the base preset because that repository does not define the selected
ControlNet. The configuration source must contain its component
`config.json` at the root, not a base pipeline's `model_index.json`.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset flux1-schnell \
  --controlnet /absolute/path/to/flux-controlnet.safetensor \
  --controlnet-config /absolute/path/to/flux-controlnet-config \
  --control-image /absolute/path/to/prepared-canny.png \
  --local-files-only
```

Native Diffusers tensor names are accepted for all three families. They are
presented to Diffusers as a temporary component package under the configured
cache, retaining its normal low-memory loader and configuration checks.
Supported original-format SD/SDXL ControlNet safetensors are delegated to
`ControlNetModel.from_single_file`. Original-format FLUX ControlNet
conversion is not provided. Pickle `.ckpt`, `.bin`, and `.pt` files are
rejected; the singular safe-file extension uses a temporary canonical alias.
Configuration options apply only to single files; package variants apply
only to directories or repositories.

For packages, the selected variant's index determines exactly which
safetensors shards are downloaded and hashed, including legacy shard names.
Without an index, only that variant's standard single weight file is used.
Other variants in the same repository or local directory are not loaded.

Original-format conversion uses the pinned upstream converter. Its native
tensor keys must exactly match the loaded component's state dictionary,
including when `--no-low-cpu-mem-usage` is selected. Missing weights cannot
be replaced with random initialization, and unexpected converted tensors
are rejected.

`--local-files-only` applies to both weights and component configuration.
Missing files, invalid architecture, incompatible weights, or a changed
file identity fail explicitly. A failed selection is not replaced with a
different model. The base model, ControlNet, and configuration revisions are
independent inputs.

## JSON, Python, and composition

The CLI, `--config`, `--print-config`, and Python `resolve_request()` share
the same schema. JSON uses `controlnet_scale` as the canonical scale key;
the longer spelling is a CLI alias. Paths in a JSON file are resolved from
that file's directory. Omitted values and JSON null preserve neutral
defaults. For example, a configuration placed alongside a `models/` and
`inputs/` directory can contain:

```json
{
  "preset": "sd15",
  "controlnet": "./models/sd15-controlnet",
  "control_image": "./inputs/canny.png",
  "controlnet_scale": 0.8,
  "control_guidance_start": 0.0,
  "control_guidance_end": 1.0
}
```

Python callers can pass the same keys to `resolve_request(values)`.
ControlNet composes with the existing base-model, VAE, LoRA, scheduler,
initial-latent, and embedding inputs. Base validation precedes ControlNet
attachment, and optional LoRA activation and CPU prompt encoding happen
before GPU placement or model/sequential offload. ControlNet uses the same
selected device and dtype. Diffusers/Accelerate owns offload hooks.
The pipeline conversion explicitly retains the requested dtype rather than
using Diffusers' `from_pipe` default of float32.

The pipelines do not implement nonzero `--guidance-rescale`, so it is
rejected rather than silently ignored. Existing family-specific restrictions
still apply. `--guess-mode` is an SD/SDXL option; FLUX Union uses
`--control-mode` instead. Multi-ControlNet, automatic condition detectors,
external image-to-image inputs, and inpainting are outside this interface.

Optional [Hires Fix](hires-fix.md) applies to all three ControlNet paths.
The same selected ControlNet and prepared conditioning image remain active
during the base pass and the second img2img diffusion pass. Diffusers
prepares that condition at each stage's resolution, with the selected
scale, guidance interval, and family-specific mode. The upscaled base image
is the img2img input; it does not replace the separate ControlNet condition.
LoRA, replacement VAE, CPU prompt encoding and batching remain composable.

FLUX ControlNets without an input hint block sample VAE latents from the
conditioning image before sampling initial denoising noise. This consumes
the configured per-image generators too; equal seeds do not imply the same
noise sequence as a generation without ControlNet. CPU precomputed prompt
embeddings retain the existing batch-expansion policy.

## Output identity and verification

The default output stem adds `-controlnet` after any `-custom`, `-vae`, and
`-lora` modifiers. An SD 1.5 ControlNet-only run therefore starts at
`build/reference/sd15-red-cube-controlnet.png`. Collision handling and batch
numbering follow the existing output rules, preserving previous evidence.
Hires Fix adds `-hires` after `-controlnet`; its optional saved base image
adds `-base` to the final image stem. Both stages' output paths and sidecars
participate in the collision checks.

The sidecar records the ControlNet source and immutable revision, model
class, configuration source, weight/configuration file paths, SHA-256 hashes
and byte sizes, package variant, scale, guidance interval, guess/union mode,
and the conditioning image's identity, oriented dimensions, RGB conversion,
and requested output size. LoRA activation remains separately recorded.
The complete resolved request is retained in `parameters` for replay.

Regression coverage addresses request defaults and errors, pinned revisions,
single-file configuration, JSON/CLI precedence, family-specific call
arguments, filenames, loading identity, and compatibility. Small real
Diffusers smoke runs exercise conditioning and generation separately from
these request tests. Neither synthetic weights nor tiny pipelines establish
full-model visual quality, suitability of arbitrary ControlNets, or operation
on every accelerator.

The implementation reuses the pinned Diffusers 0.40.0, PyTorch, Accelerate,
safetensors, Hugging Face Hub, and Pillow dependencies; no detector runtime
or new dependency is introduced. Model weights retain their own terms,
independently of Diffusers' Apache-2.0 license. No weights are bundled.
The API contracts are documented by the upstream
[SD ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet),
[SDXL ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet_sdxl),
[FLUX ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet_flux),
and [single-file loaders](https://huggingface.co/docs/diffusers/v0.40.0/en/api/loaders/single_file).
