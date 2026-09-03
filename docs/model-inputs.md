# Model and VAE input contract

## Scope

The Python generation oracle accepts independently selected model, VAE, and
LoRA weights. Its presets remain `sd15`, `sdxl-base`, and `flux1-schnell`.
Selecting different weights does not change the pipeline family or imply
compatibility between SD 1.5, SDXL, and FLUX.

This interface belongs to `reference/diffusers/generate.py`. The Python
inspection command still accepts Diffusers packages, and the C++ inspector
still validates package metadata without parsing or executing tensor weights.

## Arguments

| Argument | Accepted input | Purpose |
|---|---|---|
| `--model` | Local Diffusers directory, pinned Hub repository, or local single file | Select the complete model package, SD checkpoint, or denoiser weights |
| `--revision` | 40-character lowercase commit SHA | Pin a remote model package; invalid for local paths |
| `--model-config` | Local Diffusers directory or pinned Hub repository | Supply configuration and auxiliary components for a single-file model |
| `--model-config-revision` | 40-character lowercase commit SHA | Pin a remote configuration source independently of the model |
| `--vae` | Local `.safetensors` or `.safetensor` file | Replace only the `AutoencoderKL` weights |
| `--lora` | Local file/directory or pinned adapter repository | Apply one optional adapter after pipeline validation |

Omitting `--model` retains the selected preset's pinned repository and
revision. Omitting `--vae` retains the package's or checkpoint's VAE. Omitting
`--lora` does not load an adapter. Adapter-specific filename, revision, and
scale arguments are described in [the LoRA contract](lora.md).

Local paths must exist. Local weight files must be non-empty regular files,
possibly reached through a regular-file symlink, with exactly the lowercase
`.safetensors` or `.safetensor` extension. `.ckpt`, `.bin`, `.pt`, GGUF, URLs to
single files, and implicit selection among files in a directory are not
accepted as single-file model or VAE inputs. A Hub model override must supply
`--revision`; a Hub config override must supply `--model-config-revision`.
The preset repository may retain its built-in pinned revision when no
revision override is provided.

## Model-file roles

The loader inspects safetensors keys before assembling the pipeline:

| Preset and file role | Weights used from `--model` | Weights and metadata used from `--model-config` |
|---|---|---|
| SD 1.5 / SDXL original-format checkpoint | UNet, text encoder(s), and embedded VAE unless overridden | Component configs, tokenizers, scheduler, and selected auxiliary components |
| SD 1.5 / SDXL standalone UNet | UNet only | Configs, tokenizers, text encoder(s), scheduler, and VAE unless overridden |
| FLUX.1-schnell standalone transformer | Transformer only | Configs, tokenizers, CLIP, T5, scheduler, and VAE unless overridden |

SD/SDXL full-checkpoint classification requires original denoiser keys and
the exact CLIP sentinel keys that Diffusers uses to recognize every required
embedded text encoder. Required text encoders must be present; they never fall
back to similarly named weights from `--model-config` for a file classified as
a complete checkpoint. This prevents a partially embedded checkpoint from
being reported as the sole neural-weight source. A missing embedded VAE
requires an explicit `--vae`. Standalone native Diffusers UNet state
dictionaries and original-format UNet-only files are supported by the
corresponding Diffusers component loader. FLUX uses its transformer component
loader for native or supported original-format weights. Model files cannot
be substituted with adapter-only or VAE-only files.

`--model-config` is accepted only with a single-file `--model`. It defaults
to the selected preset's fixed repository, not to a model guessed from the
filename. The source must describe the matching Diffusers pipeline in
`model_index.json`, with permitted component declarations and the required
component configs. A standalone JSON/YAML config file is not this interface.
If non-embedded neural components are needed, the source must also provide
their weights. A config-only directory is therefore insufficient for a
denoiser-only model unless all required auxiliary weights are available there.
Non-null IP-Adapter or `image_encoder` declarations are rejected because those
auxiliary models are outside this generation interface. A configuration source
cannot silently expand the selected preset with additional model components.

On accelerator runs, the canonical SD 1.5 and SDXL repositories retain their
pinned `fp16` weight variant. A local Diffusers source uses that variant only
when its actual component filenames declare `fp16`; a custom Hub source is not
assumed to publish the preset repository's variant. The separately loaded
SD 1.5 safety checker checks its own local subdirectory, so an unrelated
component's `fp16` file does not require an unavailable safety-checker variant.
Single-file weights are loaded directly at the selected runtime dtype and do
not use a Hub variant.

For example, replace all three SDXL weight inputs while retaining the preset
configuration source:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset sdxl-base \
  --model /absolute/path/to/sdxl-checkpoint.safetensors \
  --vae /absolute/path/to/sdxl-vae.safetensors \
  --lora /absolute/path/to/sdxl-style.safetensors \
  --lora-scale 0.75
```

Use an explicit local package as the source of FLUX auxiliary components:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --model /absolute/path/to/flux-schnell.safetensor \
  --model-config /absolute/path/to/flux-schnell-package \
  --vae /absolute/path/to/flux-vae.safetensors \
  --local-files-only
```

For a remote configuration override, use
`--model-config owner/config-repository --model-config-revision COMMIT_SHA`,
replacing `COMMIT_SHA` with the exact 40-character lowercase commit. This
revision is independent of any LoRA repository revision.

## VAE replacement and ordering

The VAE is loaded using `AutoencoderKL.from_single_file` and the `vae/`
configuration from the model package, or from `--model-config` for a
single-file model. The tensor file does not independently determine scaling,
shift, or architecture settings. Native Diffusers and supported original
VAE layouts are delegated to Diffusers; configuration and tensor shapes must
agree. Quantized and alternate VAE architectures are not added by this API.

The VAE is passed into the pipeline constructor, so the superseded VAE need
not be loaded first. The assembled pipeline must satisfy its preset contract
before LoRA loading and activation, followed by device placement/offload and
generation. All three profiles require RGB input/output and four VAE levels
for 8x spatial downsampling. SD 1.5 uses four latent channels and scale
`0.18215`, SDXL uses four and `0.13025`, and FLUX uses sixteen, scale `0.3611`,
and shift `0.1159`. SDXL and FLUX retain their additional profile checks.

## Safe loading, offline operation, and provenance

Remote configuration snapshots are first resolved at the selected revision,
respecting `--local-files-only`, before their real local directory is passed
to `from_single_file`. This avoids Diffusers' fallback behavior that can
otherwise download missing configs even with its local-only flag set. The
flag also covers auxiliary component and LoRA loading. Missing cached
configuration, tokenizers, or weights fail instead of silently fetching a
replacement. A local model file alone does not guarantee an offline run.

Each local model/VAE/LoRA file is identified by the supplied absolute path,
resolved target, SHA-256, and byte size. Identity is checked before and after
loading. The singular `.safetensor` spelling uses a temporary symlink ending
in `.safetensors` under the configured cache, keeping Diffusers on its safe
loader branch without copying or rewriting the selected bytes. The alias is
removed afterward; filesystems without symlink support must use the standard
extension. These checks are not a general sandbox against malicious
concurrent filesystem changes.

Diffusers' low-memory single-file loading is used, and remaining meta tensors
are rejected. Component configurations and shapes are still validated; a
successful safetensors parse is not a compatibility guarantee. No new
third-party dependency is introduced for this feature.

Result metadata records the model selection, with detected `weights_role`,
configuration selection and local directory, optional VAE file identity, and
`component_sources` under `model.loading`. Component provenance labels are
`model`, `model_config`, and `vae_override`. The top-level `vae` object also
records whether it was overridden, its source, file identity, latent channels,
scaling/shift factors, and spatial downsampling factor. LoRA identity and
activation remain in the separate `adapters` array. An
explicit `--model` or `--revision` adds `-custom` to the default output stem,
`--vae` adds `-vae`, and `--lora` adds `-lora`, in that order. The combined SDXL
default is `sdxl-base-red-cube-custom-vae-lora.png`. This separates custom runs
from the canonical fixture; existing files still require `--overwrite`.

## Boundaries

This does not add SDXL Refiner, FLUX.1-dev, arbitrary FLUX derivatives,
quantized checkpoints, cross-family conversion, or C++ inference. File
provenance does not establish training provenance, image quality, commercial
rights, or safety. Each selected checkpoint, auxiliary model, VAE, and LoRA
has independent licensing terms.

The underlying format adapters are the pinned Diffusers APIs documented in
[single-file loading](https://huggingface.co/docs/diffusers/v0.40.0/api/loaders/single_file).
Project selection/composition lives in `presets.py` and `model_loading.py`;
local identity and temporary canonical paths live in `weight_files.py`.
