# Generic Diffusers generation

`reference/diffusers/generate_any.py` runs a public pipeline class built into the installed Diffusers runtime. It removes the three-preset restriction for models distributed as complete Diffusers directories, while retaining the existing preset oracle for its more specific component validation and generation extensions. The current dependency pin is Diffusers 0.40.0.

A Civitai base-model label identifies a model family, not a universal file format. For example, an SDXL-derived family can share the SDXL pipeline architecture while requiring a different scheduler, prediction type, or conditioning configuration. The generic runner uses the supplied model configuration and actual pipeline call signature. `--base-model` checks the Civitai catalog label against the selected pipeline architecture. SD/SDXL/SD3/Flux/Flux2/Qwen task siblings are grouped explicitly so editing, inpainting, and ControlNet variants remain usable while cross-family mismatches are rejected before weight loading. Other conventional pipelines use their installed architecture namespace. Catalog entries without a conventional Diffusers pipeline class are marked `declared-only`; unknown names and hosted-only labels are rejected. This establishes pipeline architecture only: it cannot distinguish SD1 from SD2 or identify a specific fine-tune within an architecture. It never changes tensor architecture or silently converts model weights.

## Model loading

A local Diffusers directory must contain `model_index.json`, component configurations, tokenizers, and all required model weights. The declared pipeline class is selected automatically; `--pipeline-class` can explicitly select another installed compatible task pipeline, such as an image-editing or inpainting pipeline. Only public built-in Diffusers pipeline classes and Diffusers/Transformers components are accepted. Custom pipeline scripts, arbitrary import paths, and dynamic remote code are excluded.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/sd35-medium \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 28 --guidance-scale 4.5 \
  --seed 42 --dtype bfloat16 --device cuda --offload model \
  --output-dir build/reference/sd35-run
```

A Hub model requires `--revision` with an immutable lowercase 40-character commit SHA. The runner first reads and checks that revision's model index, downloads through Diffusers with `use_safetensors=True` and `trust_remote_code=False`, then loads the resulting local snapshot with `local_files_only=True`. `--local-files-only` also prevents the initial download, permitting cached pinned snapshots only. Downloads and the Xet cache default to `build/reference/`.

A local `.safetensors` checkpoint requires `--model-config` pointing to a **local** Diffusers configuration directory with `model_index.json`, tokenizers, and any components missing from the checkpoint. The selected class must implement Diffusers' `from_single_file`. Loading all single-file formats is not implied by supporting the pipeline's directory format. In particular, a transformer-only file does not become a complete pipeline without the necessary VAE and text encoders. If the class has no single-file loader, provide a complete Diffusers directory or use the workflow backend.

The top-level `reference/generate.py` selects this generic runner automatically
when `--model-config` is present, including when `--model` names an existing
downloaded checkpoint. Explicit `--backend` and `--preset` choices retain their
precedence. `--pipeline-class` and `--pipeline-inputs` also select the generic
runner before the automatic local-file route, so image-to-image, inpainting and
refiner inputs reach the requested task pipeline.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/model.safetensors \
  --model-config /Volumes/Storage/Workspace/Models/model-config \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a small architectural model in daylight' \
  --device mps --dtype float32 \
  --output-dir build/reference/single-file-run
```

Checkpoint `.ckpt`, pickle `.pt`/`.pth`/`.bin`, GGUF, and custom Python model packages are not read by this runner. Single-file configuration extras must not contain pickle-format weights: Diffusers 0.40's internal single-file extras loader does not forward `use_safetensors` to every component, so those files are rejected explicitly. For quantized GGUF models and families not implemented by installed Diffusers, use the top-level workflow backend with a matching local workflow/runtime.

## Kolors and task-specific checkpoints

Kolors is supported here as a complete local Diffusers directory with its Kolors
UNet, VAE, ChatGLM text encoder, tokenizer and scheduler. The official distribution
identifies the built-in `KolorsPipeline`; the built-in `KolorsImg2ImgPipeline`
can reuse those components for image-to-image generation.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --model /Volumes/Storage/Workspace/Models/Kolors-diffusers \
  --base-model Kolors --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 50 --guidance-scale 5 \
  --device cuda --dtype float16 --offload model --local-files-only \
  --output-dir build/reference/kolors-image
```

**A downloaded Kolors single-file checkpoint is not established as loadable by
this generic runner.** Inspection of installed Diffusers 0.40.0 and the official
pipeline source confirms that neither Kolors pipeline has `from_single_file`.
`--model-config` cannot add that missing loader. Although
`UNet2DConditionModel` exposes a component-level single-file loader, that alone
does not establish conversion of a Kolors checkpoint's projection weights,
ChatGLM conditioning, VAE and tokenizer. Such files require a verified conversion
to a complete Diffusers directory or a compatible explicit workflow. The runner
raises a specific error instead of loading the configuration directory's base
weights while ignoring the selected checkpoint. No full Kolors model was
downloaded or generated during this inspection.

For pipelines that do implement single-file loading, supply the task's matching
configuration and its required inputs. This example uses an SDXL inpainting
checkpoint and a local mask:

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --model /Volumes/Storage/Workspace/Models/sdxl-inpaint.safetensors \
  --model-config /Volumes/Storage/Workspace/Models/sdxl-inpaint-config \
  --pipeline-class StableDiffusionXLInpaintPipeline \
  --prompt 'a copper dome above the observatory' \
  --pipeline-inputs '{"image":{"image_path":"/Volumes/Storage/Workspace/Assets/source.png"},"mask_image":{"image_path":"/Volumes/Storage/Workspace/Assets/mask.png","mode":"L"},"strength":0.8}' \
  --device mps --dtype float32 --local-files-only \
  --output-dir build/reference/inpaint-image
```

For SDXL image-to-image or a refiner checkpoint, use
`StableDiffusionXLImg2ImgPipeline`, the corresponding checkpoint configuration,
and an `image` input with `strength`; a mask is not required by that class.
For Kolors image-to-image, select `KolorsImg2ImgPipeline` with the complete Kolors
directory and the same typed image-input syntax. These routing examples do not
certify arbitrary downloaded weights or task-specific visual quality.

Sources: [official Kolors Diffusers usage](https://github.com/Kwai-Kolors/Kolors#using-with-diffusers),
[Kolors pipeline implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kolors/pipeline_kolors.py),
[Kolors image-to-image implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kolors/pipeline_kolors_img2img.py),
[single-file model conversion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py).

## Pipeline inputs and defaults

`--pipeline-inputs` accepts a JSON object or `@/absolute/path/inputs.json`. The object is passed as explicit arguments to the selected pipeline's `__call__`. Every supplied key must appear by name in that signature; an unrestricted `**kwargs` parameter does not permit silent acceptance of unknown values. Required inputs are checked before model weights are loaded. Duplicate JSON keys, non-finite JSON numbers, JSON generator objects, latent output requests, and `return_dict=false` are rejected.

`--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, and `--guidance-scale` are optional. When supplied they override the corresponding JSON values; `--steps` maps to `num_inference_steps`. When omitted, the runner leaves each pipeline's own default untouched. Explicit empty prompts and guidance `0` are retained. `--seed` constructs a CPU `torch.Generator`; omission preserves the pipeline's stochastic default. `--print-config` resolves local paths and JSON without importing Torch, Diffusers, Pillow, or NumPy and without generating or downloading weights. It does not prove pipeline compatibility.

Video, audio, editing, and model-specific options belong in the JSON object. For example, video pipelines commonly accept `num_frames`, and an editing pipeline may accept `strength`. These are examples of pipeline-specific names, not promises that every class accepts them.

```json
{
  "prompt": "a slow orbit around a ceramic sculpture",
  "num_frames": 17,
  "image": {"image_path": "/Volumes/Storage/Workspace/Assets/sculpture.png"}
}
```

Image-bearing arguments support local typed objects of the form `{"image_path":"/absolute/file.png","mode":"RGB"}`. Modes are `RGB` (default), `RGBA`, or `L`; masks can request `L`. Supported argument names are `image`, `images`, `mask_image`, `control_image`, `control_images`, `reference_image`, `reference_images`, `ip_adapter_image`, `conditioning_image`, `video`, `frames`, `last_image`, `start_image`, `end_image`, and `image_2` through `image_4`. Lists can represent image batches or explicit video frames. Input files are hashed before decoding and checked again afterward. Animated image files require an explicit frame list. Arbitrary file-to-tensor conversion, Python callbacks, serialized generators, and user code are not accepted through JSON.

## Hardware and artifacts

`--device auto` uses the shared accelerator policy: CUDA/ROCm or Apple MPS must be available. It never silently falls back to CPU. `--device cpu` explicitly enables CPU generation. `--device metal` aliases MPS, and `--device rocm` requires a HIP-enabled PyTorch runtime. The runner executes the shared arithmetic preflight and checks the pipeline execution device before and after generation.

`--dtype` defaults to `float32`; `float16` and `bfloat16` are explicit alternatives whose availability and numeric behavior depend on the hardware and model. `--offload` defaults to `none`; `model` and `sequential` call Diffusers' built-in offload methods with the selected execution device. CPU offloading is refused for explicit CPU generation.

The output directory contains every generated item, rather than just the first batch element:

| Pipeline output | Artifact |
| --- | --- |
| `images` | `image-0001.png`, `image-0002.png`, ... |
| `frames` | `video-0001/frame-000001.png`, ... |
| `audios` or `audio` | `audio-0001.wav`, ... as PCM16 |

Pillow images, decoded uint8 arrays, and finite decoded floating pixels in `[0, 1]` are supported. Torch outputs are moved to CPU for export. Videos are frame sequences; no codec or guessed playback frame rate is imposed. NumPy video output defaults to `[batch, frames, height, width, channels]` (BFHWC), and Torch output defaults to Diffusers VideoProcessor's `[batch, frames, channels, height, width]` (BFCHW). `--video-layout bfhwc|bfchw|bcfhw` explicitly selects another layout. The runner does not guess frame/channel axes from a small frame count. Audio sample rates are read from the actual pipeline's vocoder/VAE/config, or supplied by `--audio-sample-rate`; the runner does not guess an absent sample rate. Audio floating amplitudes are clipped to `[-1, 1]` when encoded as PCM16. Meshes, embeddings, latent tensors, and other return types fail explicitly instead of being represented as generated media.

`generation.json` records requested inputs, identity label, selected and actual class, component-file SHA-256 hashes and byte sizes, single-checkpoint identity, decoded input identities, all output hashes, dependency versions, device preflight, dtype, offload mode, architecture-label validation, and actual safety-checker result fields when provided. Absent safety results remain absent; checker absence is not represented as successful screening. Image-level NSFW flags must match the image batch length. Model identities are rechecked after loading and after generation. Each output file is published atomically. An existing non-empty output directory is rejected unless `--overwrite` is explicit; even then unrelated files are retained. Immediately before replacing any media, an existing `generation.json` is atomically moved to a unique `generation.previous-*.json` history file. A new current report appears only after every artifact and result-metadata check succeeds. A failure during export can leave completed media without a current report, so consumers must require a valid `generation.json` before treating the run as complete. Archived reports preserve old provenance, not previous media bytes; their hashes identify the prior files even if those paths have since been replaced.

## Validation scope

Run the dependency-free contract suite with:

```sh
python3 tests/GenericDiffusersTests.py
```

The 39 dependency-free tests cover neutral omission, explicit overrides, pinned remote identity, safe component selection, single-file/config boundaries, rejection when a pipeline lacks a single-file loader, Kolors task-family namespace validation, unknown-argument rejection, input media decoding boundaries, file-change detection, complete image/video batches, sample-rate resolution, hardware/offload checks, non-clobbering publication, partial-overwrite failure injection, retained safety flags, architecture-label rejection, and short-video frame/channel axis handling.

The validated tiny FLUX CPU smoke used the already cached `hf-internal-testing/tiny-flux-pipe` snapshot at commit `a98bc4ec6a80c47c477ed22d7e81c79872d9c723`, 64×64, one step, seed 42, guidance 0, and `max_sequence_length=64`. Its report at `build/reference/generic-diffusers-smoke/generation.json` identifies the exact tested files and the non-uniform 64×64 RGB output. A second actual CPU generation used a locally initialized tiny `DDPMPipeline`, without any prompt, and exported a two-image NumPy batch; its report is `build/reference/generic-media-smoke/ddpm-generation/generation.json`. Synthetic media export separately verified Torch uint8 images, channel-first video tensors, and stereo PCM16 WAV headers in `build/reference/generic-media-smoke/encoding-validation.json`; this is encoding evidence, not video/audio model generation evidence. A third actual CPU generation exercised a locally initialized tiny `StableDiffusion3Pipeline` with `SD3Transformer2DModel`, two small CLIP projection encoders, and cached tiny T5/VAE/tokenizers, producing two 32×32 images. Its builder is `build/reference/generic-sd3-smoke/run.py` and its report is `build/reference/generic-sd3-smoke/generation/generation.json`; no large model was downloaded. Tiny models establish runner execution only; they do not establish full-sized model quality, all-family compatibility, licensing, or GPU performance. No large model download is necessary for the contract tests.

The installed pipeline determines executable model support. Consult the official [Diffusers pipeline overview](https://huggingface.co/docs/diffusers/api/pipelines/overview) and [single-file loader documentation](https://huggingface.co/docs/diffusers/api/loaders/single_file). A Civitai catalog route plus a valid CLI configuration is a compatibility path, not evidence that every checkpoint in that family has been generated successfully.
