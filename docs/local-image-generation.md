# Standalone local image generation

`iild-generate --model-path /models/checkpoint.safetensors` executes Diffusers and
PyTorch in the SDK Python process. No ComfyUI source, interpreter, HTTP server,
workflow, custom nodes or ComfyUI cache is needed. Dreamscapes uses this same
`--backend local` route.

For app queues, use the SDK's [resident inference worker](inference-worker.md).
It retains Python initialization and compatible loaded weights between requests,
and reuses full model hashes while filesystem identity is unchanged.
Model composition, parsed configuration and device placement remain in memory.
Sampling options can change without rebuilding the model or repeating placement;
changing only execution placement reuses the already loaded components.

## Offline checkpoint configuration

The SDK ships approximately 4.8 MB of SD 1.x and SDXL configuration/tokenizer
resources under `reference/diffusers/configs/`. Exact upstream revisions, file
hashes and local modifications are recorded in `configs/manifest.json`.
These resources contain no model weights. SDXL derivatives such as Illustrious
use the SDXL tensor architecture; an optional Civitai sidecar or `--base-model`
retains a more specific identity. Filenames never select an architecture.

A complete SD 1.x or SDXL checkpoint supplies the denoiser, VAE and text encoders.
The bundled configuration supplies component definitions, the scheduler and CLIP
vocabulary/merges. `--model-config` can override this with an explicit local
configuration. Model/configuration loading stays offline. Missing companion
weights are reported before inference rather than downloaded or substituted.

```bash
iild-generate --model-path /models/illustration.safetensors \
  --prompt 'a red ceramic teapot on a wooden table' \
  --width 512 --height 512 --steps 20 --device mps \
  --output-dir /existing/parent/new-result
```

`--backend local` also accepts the existing preset options for LoRA, ControlNet,
textual inversion, HiRes, scheduler, dtype and offload. Prediction metadata is
retained unless explicitly overridden. `.safetensor` and uppercase suffixes use
a checked temporary `.safetensors` alias without renaming or copying the original.

Configured FLUX.1 single files still require local companion weights and
`--model-config`. Other supported Diffusers pipelines use `--backend diffusers`
with a complete local model directory or a compatible single-file configuration.
SD2, SD3, editing/refiner and quantized models are not falsely treated as SDXL.
The optional legacy conversion/GGUF/graph backend is explicitly selected with
`--backend comfyui-local`; see [managed ComfyUI](managed-comfyui-image.md).
There is no automatic fallback to that backend.

Omitting `--seed` chooses a fresh random base seed per request. Explicit seeds,
including `--seed 0`, are preserved. The resolved request and per-image metadata
record the actual seed; image batches derive subsequent seeds with `--seed-stride`.
Deforum/Interpolator and LTX likewise randomize their base seed per request while
retaining the configured frame/shot seed policies.

## Consumer paths and publication

`--preview-dir` enables [live denoising previews](live-previews.md) for single-pass
image requests. Each actual sampling step publishes a VAE-decoded PNG and a
flushed `IILD_PREVIEW` JSON event, independently of final output publication.

`--output-dir` accepts a new or empty directory. Images and their provenance are
prepared in an adjacent temporary directory and published together after
successful inference, decoding and hash verification. `generation.json` uses
schema `iild-standalone-image-v1`, backend `diffusers`, status `complete`, final
output paths and full per-image metadata. A failure leaves the output directory
empty and preserves existing results. `--output` retains the preset's direct
image/sidecar output contract and is mutually exclusive with `--output-dir`.

`--work-dir` optionally stores the resolved request in a new or empty real
directory. Dreamscapes supplies an app-owned temporary job directory outside Society and removes it at job/session exit.
`--cache-dir` holds runtime/weight-alias caches; Dreamscapes places it inside the
same temporary job directory and does not persist it in Society. Redirected or nonempty output/work
directories are rejected. The old `--startup-timeout` is accepted for previous
callers, but no server is started and Dreamscapes no longer supplies it.
Dreamscapes keeps its queue and job state only in app memory, without persistence
or restart recovery. It stages engine output and provenance in the temporary job directory, then
publishes only validated images directly into Society `Generation History/`.
History has no app/job subfolders, request JSON, runtime cache or previews.
Generated images are not automatically registered in `Asset Library/`.

`--print-config` and `--validate-only` validate the request without loading or
sampling a pipeline; they are not inference evidence.

## Standalone video

LTX video already executes directly through Diffusers/PyTorch and FFmpeg/FFprobe.
A complete local LTX model directory is required. Deforum/Interpolator use the
same image model loader and now automatically select the bundled SD1/SDXL
configuration for compatible checkpoint files.

```bash
iild-generate --backend video --model-path /models/local-ltx \
  --prompt 'a slow camera move through a forest' --frames 9 --fps 24 \
  --output /existing/parent/forest.mp4

iild-generate --backend deforum --model-path /models/illustration.safetensors \
  --prompt 'a red ceramic teapot' --width 512 --height 512 --dtype float32 \
  --frames 4 --fps 8 \
  --output /existing/parent/teapot.mp4
```

The optional Deforum dependency remains OpenCV. No new inference package or
paid service was introduced. Diffusers (Apache-2.0) and PyTorch remain the
maintained upstream inference implementations; the small bundled resources
retain their upstream model licenses and notices. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and the official
[Diffusers single-file loader](https://huggingface.co/docs/diffusers/api/loaders/single_file).

## Verification

`StandaloneImageTests` covers routing without server creation, bundled resource
integrity, model-family and missing-component rejection, uppercase suffixes,
image-animation configuration reuse and failed-output preservation. Build and
installed-runtime inference evidence for this change is recorded in
`build/standalone-validation/` and described in `standalone-validation.md`.
