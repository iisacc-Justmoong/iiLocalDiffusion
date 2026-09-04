# Civitai base-model compatibility

The current [validation report](civitai-validation.md) separates catalog and
installed-class coverage from actual generated artifacts and remaining gaps.

The entry point is `reference/generate.py`. It routes an explicit Civitai base
model to a named preset, a local Diffusers pipeline, or a local ComfyUI API
workflow. The catalog contains **all 105 base-model names** in the pinned Civitai
snapshot, including hidden and disabled categories. Its 79 `local` entries have
an upstream local-runtime route; 24 `hosted` entries and 2 `unknown` categories
do not acquire an automatic local route.

Existing local weight files now default to the [managed local image runtime](local-image-generation.md),
which automatically assembles its supported image workflows. An explicit preset,
pipeline or workflow keeps its corresponding execution path. The
[download inspector](downloaded-models.md) identifies bundled/split weights and
component roles; [legacy conversion](checkpoint-formats.md) preserves source hashes.

This is a **runtime-dependent compatibility catalog**, not evidence that every
model weight has generated a valid result. No new full-sized model weights were
downloaded or generation-tested while adding this catalog. Unit tests verify
complete name coverage, architecture distinctions, task metadata and routing
contracts. A particular model still needs compatible local components, supported
runtime versions, enough memory and validation with its actual weights. The C++
manifest/metadata boundary is unchanged.

## Commands

List the exact names and route metadata without importing a GPU runtime:

```bash
python reference/generate.py --list-base-models
```

Inspect whether the selected Python environment can import each catalogued
Diffusers pipeline class without loading model weights or contacting a server:

```bash
reference/diffusers/.venv/bin/python reference/generate.py --check-runtime
reference/diffusers/.venv/bin/python reference/generate.py \
  --check-runtime --base-model 'Krea 2'
```

The audit records the pinned catalog source, installed Diffusers/Torch package
versions and a status for every requested row. `pipeline_available` means only
that the suggested class is an installed built-in `DiffusionPipeline` subclass
and its lazy imports succeeded. `runtime_missing` includes a missing package,
an absent class, a dummy/non-pipeline export or an optional-dependency import
failure. Each failure carries its diagnostic without aborting other rows.
`workflow_required` leaves ComfyUI installation, server, nodes and model files
unchecked; `hosted` and `unknown` remain explicit. These three statuses do not
import ML runtimes when they are the only selected rows.

Every audit result and row sets `generation_verified=false` and
`weights_verified=false`. The audit never calls a model constructor, downloads
weights, runs denoising or checks the GPU. Use the actual model-generation path
for weight compatibility and output verification, or ComfyUI `--validate-only`
for live workflow/node/model availability. Python callers can use
`runtime_compatibility.inspect_runtime(base_name=None)` for the same report.
Importing that module alone does not import Torch or Diffusers.

Use a matching local Illustrious checkpoint with the SDXL-derived preset:

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --base-model Illustrious \
  --model /absolute/path/Illustrious.safetensors
```

NoobAI is not enough information to decide the prediction parameterization.
Use the v-pred preset explicitly when the actual checkpoint is a v-pred model:

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --base-model NoobAI --preset noobai-v-pred \
  --model /absolute/path/NoobAI-XL-Vpred.safetensors
```

Use the exact FLUX.1 Krea name for the FLUX.1-derived weights; `Krea 2` is a
separate architecture:

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --base-model 'Flux.1 Krea' \
  --model /absolute/path/flux1-krea-dev.safetensors
```

Complete local Diffusers directories preserve the configuration in their
`model_index.json` and component folders:

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend diffusers --base-model 'SD 3.5' \
  --model /absolute/path/sd35-diffusers \
  --pipeline-inputs '{"prompt":"a ceramic teapot on a wooden table","num_inference_steps":28}' \
  --output-dir build/reference/sd35
```

For ComfyUI, export an **API-format** workflow with the matching local models,
loaders and output nodes. The workflow is the authority for model components and
task-specific inputs. Node IDs in `--workflow-inputs` must exist in that workflow;
`6` below is only an example. Validate the binding before queuing:

```bash
python reference/generate.py \
  --backend comfyui --base-model Anima \
  --workflow /absolute/path/anima-api.json \
  --workflow-inputs '{"6":{"text":"a ceramic teapot on a wooden table"}}' \
  --output-dir build/reference/anima --validate-only
```

Run the same command without `--validate-only` when the local ComfyUI server,
models and output nodes are ready. A category name never supplies missing
weights, never changes a hosted provider into a local model, and never grants
model access or usage rights.

## Architecture and task distinctions

- **Illustrious, NoobAI and Pony** derive from SDXL. Preserve the named model's
  identity for adapters even when the denoiser architecture matches. The NoobAI
  v-pred author explicitly requires `prediction_type="v_prediction"` and
  `rescale_betas_zero_snr=True`; its published scheduler JSON can still say
  `epsilon`, so the category or raw scheduler JSON alone is insufficient.
- **Pony V7** derives from AuraFlow, not SDXL. **FLUX.1 Krea** uses `FluxPipeline`;
  **Krea 2** uses `Krea2Pipeline`. Raw/base and distilled/turbo variants need
  different guidance and timestep settings.
- **SD 2.x** can use epsilon or v-prediction. LCM, Hyper, Lightning and Turbo are
  not ordinary base-model scheduler defaults. SDXL Distilled can also change
  denoiser dimensions. Use the actual local config and checkpoint instructions.
- **Anima and MiniMax H3** have Modular Diffusers support, which is different from
  a conventional `DiffusionPipeline`. Lens, MageFlow and HiDream-O1 also require
  their own code/contracts. The catalog prefers explicit local ComfyUI workflows
  for these families rather than assuming a generic image pipeline exists.
- **Wan 2.2 A14B** is a two-expert architecture; a high-noise-only or low-noise-only
  file is not a complete pipeline. Wan image conditioning, resolution and VAE
  versions must match. LTX-2.5 has different weights from LTX-2/2.3.
- **SVD/SVD XT** generate video from an image, despite Civitai's legacy
  `type="image"` metadata. **ACE Audio/MiniMax Music 3** generate audio.
  **Hunyuan3D/Pixal3D/Trellis.2** produce 3D assets. Their workflow outputs must be
  exported in the appropriate media/mesh format.
- A **Civitai base-model name is not a file format**. SafeTensor, PickleTensor,
  GGUF, Diffusers directories, Core ML and ONNX need different loaders. A LoRA,
  VAE, text encoder, UNet, workflow or upscaler is not necessarily a complete
  checkpoint. The generic local Diffusers route does not silently convert GGUF,
  execute pickle, or import arbitrary custom remote code; use a matching local
  workflow/runtime for formats outside that route.

## Catalog API and update contract

`reference/diffusers/civitai_catalog.py` uses only the Python standard library.
`lookup_base_model(name)` returns an independent dictionary and rejects unknown
names with `ValueError`; only letter case and outer whitespace are normalized.
`list_base_models()` returns independent records in upstream order.
`CATALOG_SOURCE` and `UPSTREAM_SOURCES` expose pinned source information.

Each record includes `name`, `family`, `local_status`, `preferred_backend`,
`preset`, `pipeline_class`, `task`, `notes`, `sources`, `civitai_hidden` and
`civitai_disabled`. `pipeline_class` is a suggested task-specific class, not an
instruction to override an incompatible `model_index.json`. A comma-separated
`task` value names alternatives requiring their corresponding pipelines or
workflow branches; it does not imply that one pipeline supports every task.
`preset` is supplied only for named presets with a defined model contract.

Update the catalog JSON and its independent name fixture together when adopting
a newer Civitai snapshot. Review architecture, local-weight availability and
runtime source evidence for each new name. Do not infer local availability from
Civitai's `hidden`, `disabled`, `generation` or `selfHosted` flag: these describe
Civitai's service, not the user's computer. Unknown categories remain explicit.
No network access occurs when reading the checked-in catalog.

## Source snapshot

- Civitai: [ec49115e55d7c85722ec6223ec342c36df59c9d4](https://github.com/civitai/civitai/blob/ec49115e55d7c85722ec6223ec342c36df59c9d4/packages/civitai-shared/src/basemodel.constants.ts), committed 2026-09-04T02:33:02Z, retrieved 2026-09-04. File SHA-256: `032c3f038b5c4abf1e4a47a4622e9eeb8db084080e38c9f2830dcb8fa9f5fea3`.
- diffusers: [7643c4826609c47755e3da0e5b768e8070468f49](https://github.com/huggingface/diffusers/blob/7643c4826609c47755e3da0e5b768e8070468f49/src/diffusers/pipelines/__init__.py).
- comfyui: [e80c1570b6b44a2557d5d8e341e05782d18c9bbb](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/supported_models.py).
- [Illustrious XL model card](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0), [NoobAI v-pred instructions](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md), [FLUX.1 Krea model card](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev).
- [Krea 2 pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2), [MiniMax H3 modular pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3), [Lens model card](https://huggingface.co/microsoft/Lens), [MageFlow reference](https://github.com/microsoft/Mage/tree/main/mage_flow).

## Complete snapshot matrix

`local` means an upstream local implementation exists, subject to the runtime
requirements above. `hosted` describes the Civitai provider/category with no
verified local route in this catalog; `unknown` requires explicit architecture
identification. Source evidence and variant caveats are retained per row in the
JSON catalog.

| Civitai base-model name | Status | Preferred route | Task |
|---|---|---|---|
| Anima | local | `comfyui` | text-to-image |
| AuraFlow | local | `AuraFlowPipeline` | text-to-image |
| Chroma | local | `ChromaPipeline` | text-to-image |
| CogVideoX | local | `CogVideoXPipeline` | text-to-video,image-to-video,video-to-video |
| Ernie | local | `ErnieImagePipeline` | text-to-image |
| Flux.1 S | local | `flux1-schnell-compatible` | text-to-image |
| Flux.1 D | local | `flux1-dev` | text-to-image |
| Flux.1 Krea | local | `flux1-krea-dev` | text-to-image |
| Flux.1 Kontext | local | `FluxKontextPipeline` | image-to-image |
| Flux.2 D | local | `Flux2Pipeline` | text-to-image,image-to-image |
| Flux.2 Klein 9B | local | `Flux2KleinPipeline` | text-to-image,image-to-image |
| Flux.2 Klein 9B-base | local | `Flux2KleinPipeline` | text-to-image,image-to-image |
| Flux.2 Klein 4B | local | `Flux2KleinPipeline` | text-to-image,image-to-image |
| Flux.2 Klein 4B-base | local | `Flux2KleinPipeline` | text-to-image,image-to-image |
| Flux 3 Video | hosted | `unavailable` | text-to-video |
| Grok | hosted | `unavailable` | text-to-image,text-to-video,image-to-video |
| HappyHorse | hosted | `unavailable` | text-to-video |
| HiDream | local | `HiDreamImagePipeline` | text-to-image |
| HiDream-O1 | local | `comfyui` | text-to-image,image-to-image |
| Hunyuan 1 | local | `HunyuanDiTPipeline` | text-to-image |
| Hunyuan Video | local | `HunyuanVideoPipeline` | text-to-video,image-to-video |
| Ideogram 4.0 | local | `Ideogram4Pipeline` | text-to-image |
| Boogu | local | `comfyui` | text-to-image,image-to-image |
| Illustrious | local | `illustrious` | text-to-image |
| Imagen4 | hosted | `unavailable` | text-to-image |
| Kolors | local | `KolorsPipeline` | text-to-image |
| Krea 2 | local | `Krea2Pipeline` | text-to-image |
| LTXV | local | `LTXPipeline` | text-to-video,image-to-video |
| LTXV2 | local | `comfyui` | text-to-video,image-to-video |
| LTXV 2.3 | local | `comfyui` | text-to-video,image-to-video |
| LTXV 2.5 | local | `comfyui` | text-to-video,image-to-video |
| Lens | local | `comfyui` | text-to-image |
| Lumina | local | `Lumina2Pipeline` | text-to-image |
| MageFlow | local | `comfyui` | text-to-image,image-to-image |
| MAI | hosted | `unavailable` | text-to-image,image-to-image |
| Mochi | local | `MochiPipeline` | text-to-video |
| Nano Banana | hosted | `unavailable` | text-to-image,image-to-image |
| NoobAI | local | `noobai` | text-to-image |
| ODOR | unknown | `unavailable` | unknown |
| OpenAI | hosted | `unavailable` | text-to-image |
| Upscaler | local | `comfyui` | image-upscale |
| Other | unknown | `unavailable` | unknown |
| PixArt a | local | `PixArtAlphaPipeline` | text-to-image |
| PixArt E | local | `PixArtSigmaPipeline` | text-to-image |
| Playground v2 | local | `StableDiffusionXLPipeline` | text-to-image |
| Pony | local | `pony` | text-to-image |
| Pony V7 | local | `AuraFlowPipeline` | text-to-image |
| Qwen | local | `QwenImagePipeline` | text-to-image |
| Qwen 2 | hosted | `unavailable` | text-to-image,image-to-image |
| Qwen 3 | hosted | `unavailable` | text-to-image,image-to-image |
| Stable Cascade | local | `StableCascadeCombinedPipeline` | text-to-image |
| SD 1.4 | local | `StableDiffusionPipeline` | text-to-image |
| SD 1.5 | local | `sd15-compatible` | text-to-image |
| SD 1.5 LCM | local | `LatentConsistencyModelPipeline` | text-to-image |
| SD 1.5 Hyper | local | `StableDiffusionPipeline` | text-to-image |
| SD 2.0 | local | `StableDiffusionPipeline` | text-to-image |
| SD 2.0 768 | local | `StableDiffusionPipeline` | text-to-image |
| SD 2.1 | local | `StableDiffusionPipeline` | text-to-image |
| SD 2.1 768 | local | `StableDiffusionPipeline` | text-to-image |
| SD 2.1 Unclip | local | `StableUnCLIPImg2ImgPipeline` | image-to-image |
| SD 3 | local | `StableDiffusion3Pipeline` | text-to-image |
| SD 3.5 | local | `StableDiffusion3Pipeline` | text-to-image |
| SD 3.5 Large | local | `StableDiffusion3Pipeline` | text-to-image |
| SD 3.5 Large Turbo | local | `StableDiffusion3Pipeline` | text-to-image |
| SD 3.5 Medium | local | `StableDiffusion3Pipeline` | text-to-image |
| SDXL 0.9 | local | `StableDiffusionXLPipeline` | text-to-image |
| SDXL 1.0 | local | `sdxl` | text-to-image |
| SDXL 1.0 LCM | local | `StableDiffusionXLPipeline` | text-to-image |
| SDXL Lightning | local | `StableDiffusionXLPipeline` | text-to-image |
| SDXL Hyper | local | `StableDiffusionXLPipeline` | text-to-image |
| SDXL Turbo | local | `StableDiffusionXLPipeline` | text-to-image |
| SDXL Distilled | local | `StableDiffusionXLPipeline` | text-to-image |
| Reve | hosted | `unavailable` | text-to-image,image-to-image |
| Muse Image | hosted | `unavailable` | text-to-image,image-to-image |
| Seedream | hosted | `unavailable` | text-to-image |
| SVD | local | `StableVideoDiffusionPipeline` | image-to-video |
| SVD XT | local | `StableVideoDiffusionPipeline` | image-to-video |
| Sora 2 | hosted | `unavailable` | text-to-video |
| Veo 3 | hosted | `unavailable` | text-to-video |
| Wan Video | local | `WanPipeline` | text-to-video |
| Wan Video 1.3B t2v | local | `WanPipeline` | text-to-video |
| Wan Video 14B t2v | local | `WanPipeline` | text-to-video |
| Wan Video 14B i2v 480p | local | `WanImageToVideoPipeline` | image-to-video |
| Wan Video 14B i2v 720p | local | `WanImageToVideoPipeline` | image-to-video |
| Wan Video 2.2 TI2V-5B | local | `WanPipeline` | text-to-video,image-to-video |
| Wan Video 2.2 I2V-A14B | local | `WanImageToVideoPipeline` | image-to-video |
| Wan Video 2.2 T2V-A14B | local | `WanPipeline` | text-to-video |
| Wan Video 2.5 T2V | hosted | `unavailable` | text-to-video |
| Wan Video 2.5 I2V | hosted | `unavailable` | image-to-video |
| Wan Image 2.7 | hosted | `unavailable` | text-to-image |
| Wan Video 2.7 | hosted | `unavailable` | text-to-video |
| Wan Video 3.0 | hosted | `unavailable` | text-to-video |
| ZImageTurbo | local | `ZImagePipeline` | text-to-image |
| ZImageBase | local | `ZImagePipeline` | text-to-image |
| Vidu Q1 | hosted | `unavailable` | text-to-video |
| MiniMax H3 | local | `comfyui` | text-to-video,image-to-video,video-to-video |
| Kling | hosted | `unavailable` | text-to-video |
| Seedance | hosted | `unavailable` | text-to-video |
| ACE Audio | local | `comfyui` | text-to-audio |
| MiniMax Music 3 | local | `comfyui` | text-to-audio |
| PolyGen | hosted | `unavailable` | text-to-3d,image-to-3d |
| Tripo | hosted | `unavailable` | image-to-3d |
| Hunyuan3D | local | `comfyui` | image-to-3d |
| Pixal3D | local | `comfyui` | image-to-3d |
| Trellis.2 | local | `comfyui` | image-to-3d |
