# LoRA generation contract

To save a full checkpoint with one or more LoRA changes fused into its weights,
use [model merging](model-merging.md). `iild-merge --additional-model` accepts LoRAs
alongside full checkpoints for weighted sums or direct weighted subtraction.
The following contract describes applying an adapter during generation.

## Scope

The preset and standalone runners support SD1, SDXL and FLUX.1 LoRAs. The
generic `--backend diffusers` runner shares the same loader, strength controls
and component-level activation checks with all installed pipelines exposing
Diffusers' `load_lora_weights`, `set_adapters` and `get_list_adapters` APIs.
This includes SD2/SD3 and compatible FLUX2, Qwen Image and other image pipelines;
an unsupported loader fails before base weights are allocated. No custom remote
pipeline code or new inference dependency is introduced.

Omitting `--lora` selects the matching family entry from the shared
[generation defaults manifest](generation-defaults.md). The supplied
`addDetailAesthetic_v20_32` is an SDXL adapter with default strength 1.0.
Other families need their own compatible adapter; an SDXL LoRA does not become
an SD2, SD3 or FLUX LoRA by changing its family label, keys or tensor dimensions.
An explicit adapter replaces the fallback. `--no-default-modifiers` provides
an explicit baseline without automatic weights.

The native C++ generator also passes explicit LoRAs to its engine for any
supported model family and resolves family defaults after the engine identifies
the loaded model. The metadata-only C++ manifest inspector does not run LoRAs.
The base model must first pass its pipeline contract;
the adapter is then loaded and activated before device placement or sequential
CPU offload hooks are installed.
When `--cpu-text-encoding` is enabled, CPU prompt encoding runs after verified
LoRA activation and before those hooks. Thus text-encoder adapters are active
on CPU too; only the resulting embeddings move to the GPU for generation.
Model and sequential RAM offload remain independent of adapter selection.
The CPU stage restores original encoder storage precision afterward, including
mixed-precision adapter tensors/buffers, even if encoding fails. It does not
fuse, deactivate, or change the selected adapter scale.
The base weights and VAE can also be selected explicitly through `--model`
and `--vae`; see [the model-input contract](model-inputs.md). An adapter must
match the resulting pipeline, not merely share its file extension.
LoRA can also compose with one selected [ControlNet](controlnet.md). The
ControlNet component is attached before LoRA activation; `--lora` continues
to target the family's existing denoiser/text-encoder adapter interface and
does not select or train a ControlNet weight adapter.

## Local adapter

Pass an exact safetensors file:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset flux1-schnell \
  --lora /absolute/path/to/style.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-style.png
```

A directory is also accepted only when its exact file is named explicitly:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sdxl-base \
  --lora /absolute/path/to/adapter-directory \
  --lora-weight-name style.safetensors \
  --lora-scale 0.8
```

SD2, SD3 and other built-in image pipelines use their complete local Diffusers
directory (or an explicit local single-file configuration):

Explicit `[null, null]` components in its local `model_index.json` are forwarded
as `None` to matching constructor parameters, including an SD3 model saved
without T5. Omitted components are not invented or fetched.

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend diffusers --model /absolute/path/sd3-diffusers \
  --lora /absolute/path/sd3-style.safetensors --lora-scale 0.75 \
  --prompt 'a red cube' --output-dir build/reference/sd3-style
```

The same `--lora` and `--lora-scale` inputs work with `--backend deforum`,
including SD, SDXL and FLUX.1 text-to-image followed by image-to-image frames.
Every generated frame validates adapters before prompt encoding, after device
placement and after inference. Conversion with `from_pipe` must retain them.
A zero-denoising-strength frame only warps the previous image and records
`frames[].lora.applied: false`; it does not claim new LoRA inference.

The selected local file must exist, be non-empty, and end in `.safetensors`
or `.safetensor`. The singular spelling is exposed through a temporary
canonical `.safetensors` symlink under the cache so Diffusers never selects
its pickle loader for that spelling. The original path and resolved target,
SHA-256, and byte size are recorded; identity is checked before and after
loading. The alias does not rewrite or copy the original weights and is
removed after loading.

When `--output` is omitted, `-lora` is appended to the default output stem,
after any `-custom` model and `-vae` suffixes. Thus combined model/VAE/LoRA
inputs cannot silently use the canonical base fixture's filename.

## Local-only adapter inputs

Adapters must be supplied as existing local files or directories. A directory
requires `--lora-weight-name`; a direct file does not. Hub IDs and non-null
`--lora-revision` values are rejected. Model and adapter loading always use
`local_files_only=True`; no adapter is downloaded during generation.

## Runtime semantics

The fixed internal adapter name is `iild_lora`. Diffusers loads it with
`use_safetensors=True` and `low_cpu_mem_usage=True`, after which
`set_adapters("iild_lora", adapter_weights=scale)` activates the requested
finite scale. The adapter is not fused into the base weights. Negative, zero,
and greater-than-one finite scales remain available because Diffusers supports
them and some adapters depend on values outside zero to one.

The loader verifies that the adapter is both registered on at least one model
component and listed as active before inference begins. SD adapters can target
the UNet and CLIP encoder, SDXL adapters can additionally target its second
CLIP encoder, and FLUX adapters can target the transformer and first CLIP
encoder. The FLUX T5 encoder is not a LoRA-loadable component in Diffusers
0.40.0.

An adapter must have been trained for the selected base architecture. A
successful download does not establish shape compatibility, intended trigger
words, output quality, safety, ownership, or commercial rights. Trigger words
are not inferred or inserted into the prompt. A Diffusers-native safetensors
LoRA is runtime-verified. Other safetensors layouts are delegated to Diffusers
0.40's format conversion; Kohya, DoRA, LyCORIS variants, Control LoRA, fusion,
and multiple simultaneous adapters are neither independently verified nor
explicitly rejected by this interface.

## Provenance

Preset/Deforum JSON contains an `adapters` array; generic `generation.json`
contains `adapters.lora` (null for a base-only run). A LoRA run records its source,
exact file, requested revision, local file
hash and size when applicable, scale, safetensors format, fixed adapter name,
actual registered components, active adapter list, and `fused: false`.
Generic reports also record PEFT's installed version and default selection
status. Cached image pipelines include the adapter identity and strength in
their construction key: changing either reloads the pipeline; an unchanged
request reuses its registered adapter. Files are rechecked after inference.

`UniversalLoraTests`, `ModelPreparationCacheTests`, `DeforumRuntimeTests` and
`NativeResultTests` cover routing, default selection, precedence, cache changes
and adapter loss. `tests/UniversalLoraDiffusersSmoke.py` is an opt-in real
Diffusers/PEFT test using small locally initialized SD1, SD2, SDXL, SD3 and FLUX
weights. It compares no adapter, zero strength, default strength and repeated
generation, then checks SD1/SDXL/FLUX Deforum frame effects and video decoding.
It does not download models or establish trained-model visual quality.

The implementation uses the existing Diffusers 0.40 / PEFT 0.20 stack and its
[official LoRA loader interfaces](https://huggingface.co/docs/diffusers/en/api/loaders/lora).

Adapter licensing is independent of the base model and Diffusers/PEFT library
licenses. The operator must review the selected adapter repository or local
artifact terms before product use.
