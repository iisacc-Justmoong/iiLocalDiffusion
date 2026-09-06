# Deforum 2D video generation

`iild-generate --backend deforum` runs a sequential 2D diffusion animation and
writes an H.264 MP4, lossless PNG frames and a JSON provenance report. The
equivalent preset-runner option is `--animation-mode 2D`. Frame zero uses
text-to-image, or an optional local init image. Every subsequent frame warps
the **preceding generated frame**, then uses it as image-to-image conditioning.
Model weights are loaded once; the img2img pipeline shares those components.

This implements the [Deforum 2D feedback and keyframe method](https://github.com/deforum/sd-webui-deforum/wiki/Animation-Settings)
through the existing Diffusers runtime. It is not an installation of the
AUTOMATIC1111 extension and does not import arbitrary Deforum settings files.
3D depth warping, optical flow, cadence/tween frames, video input, audio
synchronization and resume are not implemented. Prompt-embedding blending is
available in the separate [Interpolator mode](interpolator-video.md). Deforum prompt
keyframes change text at their specified frames; numeric anchors interpolate.
The C++ library still provides its existing contracts and compute components;
complete video inference runs in Python, as complete image inference does.

## Runtime and usage

Use the existing Diffusers environment plus the optional maintained camera
dependency. FFmpeg **and FFprobe** must be available on PATH, with FFmpeg's
`libx264` encoder. The generator preflights these before loading model weights.

```sh
uv pip install --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements-deforum.txt

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend deforum --preset sd15-compatible \
  --model /absolute/path/to/diffusers-model --local-files-only \
  --animation-prompts '{"0":"a city at sunrise","60":"a forest at sunrise"}' \
  --max-frames 120 --fps 24 --steps 20 \
  --strength-schedule '0:(0.35), 119:(0.45)' \
  --zoom '0:(1.01)' --angle '0:(0.2*sin(2*pi*t/fps))' \
  --output build/deforum.mp4
```

Use `--preset sdxl`, `illustrious`, `noobai`, `noobai-v-pred`, `pony`,
`flux1-dev`, `flux1-krea-dev`, or an existing compatible preset with the
corresponding weights. `--base-model` retains the catalog's exact family
routing. A local checkpoint can use the existing `--model-config` companion
directory. Deforum uses the preset loader, so raw GGUF/ComfyUI workflows and
unrelated generic architectures do not become supported animation models.

Model/VAE replacement, LoRA, textual inversion, a static ControlNet image,
CPU prompt encoding, CPU/Metal/CUDA/ROCm execution and existing offload policies
remain available. The existing FLUX ControlNet refinement adapter preserves
its negative conditioning path. Every frame recreates its scheduler state and
generator; CPU prompt embeddings are refreshed for the current frame's text.

The same options work in a typed JSON `--config` and Python
`generate.resolve_request({...})`. Explicit CLI values override JSON values;
`--print-config` validates and exports replayable parameters without importing
Torch/OpenCV, downloading weights or running FFmpeg. See
[`deforum.example.json`](../reference/diffusers/deforum.example.json).
`deforum_runtime.render_deforum_frames()` also accepts an already prepared
pipeline and an output callback for embedding this frame loop in Python.

```sh
python3 reference/generate.py --backend deforum \
  --config reference/diffusers/deforum.example.json --print-config
```

## Options and schedule semantics

| Option | Default in 2D mode | Meaning |
|---|---|---|
| `max_frames` | 120 | Number of output frames, 1 to 1,000,000 |
| `fps` | 24 | Finite positive frame rate, at most 240 |
| `init_image` | none | Static local image, EXIF-corrected and resized to the requested dimensions |
| `animation_prompts` | `{"0": prompt}` | Positive text keyed by canonical zero-based frame numbers |
| `animation_negative_prompts` | `{"0": negative_prompt}` | Negative text keyframes |
| `animation_prompts_2`, `animation_negative_prompts_2` | Primary schedules | Secondary encoder schedules for SDXL/FLUX; explicit secondary text stays fixed unless scheduled |
| `zoom` | `0:(1)` | Per-frame scale; above 1 zooms in, in `(0,100]` |
| `angle` | `0:(0)` | Per-frame counterclockwise rotation in degrees |
| `translation_x`, `translation_y` | `0:(0)` | Per-frame pixel translation; positive moves image content right/down |
| `strength_schedule` | `0:(0.35)` | Diffusers denoising strength in `[0,1]`; higher changes more of the preceding image |
| `cfg_scale_schedule` | `0:(guidance_scale)` | Per-frame guidance scale; FLUX schnell requires 0 |
| `noise_schedule` | `0:(0)` | Independent Gaussian pixel noise standard deviation relative to 255, in `[0,1]` |
| `contrast_schedule` | `0:(1)` | Nonnegative pixel multiplier before denoising |
| `seed_behavior` | `fixed` | `fixed`, `iter` (uses `seed_stride`), or reproducible hash-based `random` |
| `border` | `replicate` | OpenCV `replicate`, `reflect` or `wrap` image borders |
| `color_coherence` | `none` | `RGB` matches the first output frame's per-channel histograms |
| `video_crf`, `video_preset` | 18, `medium` | FFmpeg libx264 quality `[0,51]` and encoding speed preset |
| `ffmpeg`, `ffprobe` | Executable names on PATH | Override either executable by path |
| `encoding_timeout` | 300 seconds | Timeout for encoding and decoding verification |

Numeric schedules accept a constant (`"1.02"`) or `"0:(1), 48:(1.02)"`.
Numeric anchors interpolate linearly, with held endpoint values. A value
expression using `t` evaluates at the current frame until the next keyframe.
`max_f` is the **last frame index**, `max_frames - 1`; `fps`, `pi` and `e` are
also available. Expressions support `+ - * / % **`, unary signs, parentheses,
and `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `floor`, `ceil`, `abs`, `min`,
`max`. They use a bounded arithmetic interpreter, never Python `eval`.
Expressions are limited to 1024 characters/128 syntax nodes, exponents to
`[-32,32]`, schedules to 64 KiB and intermediate values to finite `+/-1e12`.
Duplicate, fractional, negative or out-of-range frame keys are rejected.

Prompt schedules must include `"0"`; each prompt is held until the next key.
Inline Deforum `--neg`/weighted prompt expressions are not parsed: use the
explicit negative schedule. FLUX negative prompts require `true_cfg_scale > 1`.
The two prompt streams and all defaults are retained in the exported config,
so replay preserves secondary-encoder fallback behavior.

`strength_schedule=0` explicitly skips diffusion when an input image exists;
the report labels that frame `warp-only` with zero sampling steps. Without
an init image, the first frame always uses a complete text-to-image pass.
Positive strengths must produce at least one denoising step. The entire
timeline is checked before weights load, including later expression failures.

Animation requires `num_images=1` and an `.mp4` output. HiRes Fix, supplied
latents/embeddings, custom timestep/sigma arrays and SDXL early stopping cannot
be combined with animation. SD 1.5 and ControlNet img2img do not accept a
nonzero `guidance_rescale`. These combinations fail explicitly. Resolution,
LoRA, scheduler, sampling steps and static ControlNet settings remain shared
by the sequence; they are not per-frame schedules.

## Artifacts and verification

For `build/deforum.mp4`, the bundle contains `deforum.mp4`, `deforum.json` and
`deforum-frames/frame-000000.png` onward. Camera transforms and pixel noise run
on CPU; denoising runs on the selected device. Only the previous/reference
images stay in memory; PNGs are streamed to a temporary directory. FFprobe
decodes the encoded video and verifies H.264/yuv420p, resolution, FPS, duration
and exact frame count before publication. The report records model/loading
identity, adapter provenance, initial-image identity, each frame's prompts,
seed, source/output pixel hashes, PNG hashes, actual sampling timesteps,
finite latents and MP4 hash. A seed does not guarantee identical pixels across
hardware or runtime versions.

Existing explicit output paths fail unless `--overwrite` is supplied. Default
paths receive a fresh numbered suffix on collision. Overwrite only replaces
a frames directory marked as an iiLocalDiffusion animation bundle. Sampling
or encoding failure leaves the previous bundle intact; publication rollback
restores previous artifacts if a normal filesystem error occurs. The completion
report is published last. A process lock prevents simultaneous writers to the
same output. A forced process kill/power loss can leave its lock or staging
directory behind; remove those only after confirming the process has ended.
This is not a resumable rendering format or a cross-filesystem ACID transaction.

`DeforumOptionsTests`, `DeforumRuntimeTests` and `DeforumMediaTests` are registered
with CTest. The media tests skip on interpreters without optional OpenCV/Pillow/
NumPy or FFmpeg. Run them explicitly with the generation environment:

```sh
reference/diffusers/.venv/bin/python tests/DeforumMediaTests.py
reference/diffusers/.venv/bin/python tests/DeforumDiffusersSmoke.py --device cpu
reference/diffusers/.venv/bin/python tests/DeforumDiffusersSmoke.py --device mps
```

The smoke creates small random SD/SDXL-compatible safetensors locally and uses
the public CLI with and without ControlNet. It checks real sampling, feedback
hashes and video decoding. It verifies execution, not trained-model visual
quality or CUDA/ROCm hardware support.
