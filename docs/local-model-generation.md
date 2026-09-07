# Generate from local model arguments

This page describes the **local-path** source. The other model locations are
[direct API and cloud model ID](model-sources.md), both evaluated remotely.
iiLocalDiffusion's local route generates with caller-supplied files and packages.
`--model-path` (legacy `--model`) is required for local image, Deforum, Interpolator and temporal video requests,
including `--print-config`. A preset selects architecture and sampling defaults;
it does not select or download model weights. Hub repository IDs, URLs and missing
paths fail before model loading, even if an immutable revision is supplied.

The existing Diffusers/PyTorch implementation and managed local ComfyUI runtime
remain the inference dependencies. No new inference library or paid service is
introduced. Runtime installation is a separate preparation step. Generation uses
local-only model loading; missing configurations, tokenizers, encoders and weights
must be supplied by the caller instead of downloaded during inference.

## Model inputs

| Generation route | Required local input |
|---|---|
| Automatic image route | Downloaded checkpoint file, or complete Diffusers model directory |
| Preset image, Deforum, Interpolator | Compatible SD/SDXL/FLUX.1 Diffusers directory; a safetensors model file additionally requires a local `--model-config` directory |
| Generic Diffusers | Complete model directory; supported single-file pipelines require a local `--model-config` directory |
| Temporal video | Compatible LTX Diffusers directory containing the transformer, temporal VAE, text encoder, tokenizer and scheduler |
| Explicit ComfyUI workflow | Local API workflow referring to models already installed in the loopback runtime |

Optional `--vae`, `--lora`, `--controlnet`, component configurations and text
embeddings also refer to local files or directories. A standalone denoiser may
still need separately supplied VAE/text-encoder weights. Local availability does
not imply architecture compatibility or visual quality.

The old revision arguments have no generation route. Non-null revisions are
rejected. `--local-files-only` remains accepted for existing local commands and is
always enabled; requests to disable it are rejected. JSON paths resolve relative
to their configuration file. CLI paths resolve relative to the working directory.

## Images and animations from the same image model

The unified launcher's video default is **local LTX at 24 FPS**. It recognizes
video requests from `.mp4`/`.gif` output, FPS, frame count or duration. In automatic
selection, **FPS <= 12 OR GIF output** uses Deforum, or Interpolator when end
prompt/seed inputs are present. Explicit Deforum/Interpolator requests default
to 12 FPS and are rejected above 12 FPS unless the output extension is `.gif`.
The policy also applies to the direct preset runner and Python requests.

| Request | Automatic backend | Required local model |
|---|---|---|
| MP4, FPS omitted or greater than 12 | LTX → frame Interpolator (`video`), final 24 FPS when omitted | LTX directory only |
| MP4, FPS at most 12 | Deforum | Compatible image model |
| GIF, including FPS greater than 12 | Deforum | Compatible image model |
| Eligible animation with end prompt/seed | Interpolator | Compatible image model |

Use `--backend interpolator` to select interpolation explicitly. Explicit
`--backend video` and LTX-specific keyframe/camera/storyboard requests retain
LTX, including at low FPS. A forbidden image-animation request fails instead of
silently substituting a different model or discarding its controls. `--model`
always names the caller's compatible local weights; the launcher never fetches
or replaces weights when selecting a backend. Image requests keep their existing
routing, and explicit generic pipelines/workflows keep their own contracts.

`--frames` and `--max-frames` are aliases, including in JSON. `--duration` is
exclusive with a frame count and rounds `seconds * fps` to output frames. CLI
FPS and output paths override JSON before routing; output format is determined
by the extension, case-insensitively.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend preset --preset sdxl \
  --model /absolute/path/sdxl-diffusers --prompt 'A glass bottle on a table.' \
  --output build/bottle.png

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend deforum --preset sdxl \
  --model /absolute/path/sdxl-diffusers --prompt 'A glass bottle on a table.' \
  --max-frames 48 --fps 12 --zoom 1.01 --output build/bottle-camera.mp4

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend interpolator --preset sdxl \
  --model /absolute/path/sdxl-diffusers \
  --prompt 'A blue glass bottle.' --end-prompt 'An amber glass bottle.' \
  --max-frames 48 --fps 12 --output build/bottle-transition.mp4
```

These modes reuse the same image weights. Deforum uses previous-frame feedback;
Interpolator blends prompt/seed conditions. They do not turn an image model into
a model trained on temporal motion.

For a 24 FPS GIF, use the same image weights with `--fps 24 --output build/bottle.gif`.
GIF output uses FFmpeg palette generation/encoding and Pillow decoding, without
a new dependency. It loops indefinitely. GIF delays have 10 ms resolution:
encoded duration is rounded accordingly and the report includes both requested
FPS and effective `encoded_fps`. The supported GIF timing range is
`100/65535 <= fps <= 100`. H.264 CRF/preset options apply only to MP4. Both formats
retain the PNG frames, provenance report and transactional publication checks.

## Temporal video from a local video model

Ordinary LTX video uses **LTX generation followed by frame interpolation**.
The second stage needs no additional model. `--fps` is the final rate and
`--interpolation-factor` defaults to 2 (allowed 2–8). LTX generates source
frames with endpoints/keyframes preserved, then the motion Interpolator fills
the remaining output positions without changing video length or crossing cuts.
See [the two-stage contract](temporal-video.md#ltx-followed-by-frame-interpolation)
for the source-frame schedule, CPU postprocessing and provenance. Explicit LTX
at FPS <= 12 skips the postprocess; standalone GIF/low-FPS animation is unchanged.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend video --model /absolute/path/ltx-diffusers \
  --prompt 'A red boat moves across blue water.' \
  --camera dolly-in --duration 5 --fps 24 --interpolation-factor 2 --output build/boat.mp4
```

First/last image conditions, camera descriptions, shot planning and previous-shot
continuity remain available. The model's declared local architecture must match
the selected backend; arbitrary model filenames are not interpreted as LTX.

Python callers use the same explicit model field:

```python
from generate import resolve_request

preset, request = resolve_request({
    "model": "/absolute/path/sdxl-diffusers",
    "preset": "sdxl",
    "prompt": "A glass bottle on a table.",
    "animation_mode": "2D",
    "max_frames": 48,
    "output": "build/bottle.mp4",
})
```

This resolves a request; the selected runtime executes inference. Help, catalog,
camera-list and runtime-inspection commands do not require generation weights.
