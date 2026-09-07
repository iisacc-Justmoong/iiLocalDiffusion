# Interpolator video generation

`iild-generate --backend interpolator` generates an H.264 MP4 or animated GIF by interpolating
the text conditioning and initial noise between two endpoints. The equivalent
preset-runner/Python configuration is `animation_mode="Interpolator"`.
Every frame, including both endpoints, receives a full text-to-image diffusion
pass. This is prompt/seed interpolation, not optical-flow frame insertion or
an image crossfade. Deforum's camera/previous-frame feedback remains a separate
[2D mode](deforum-video.md).

Interpolator is restricted to **FPS <= 12 or GIF output**; its default is 12 FPS.
Other automatic video requests use local LTX at 24 FPS. See the
[shared video selection and GIF rules](local-model-generation.md#images-and-animations-from-the-same-image-model).

This guide covers standalone **image-model prompt/seed animation**. Ordinary
LTX video instead uses the [post-LTX frame Interpolator](temporal-video.md#ltx-followed-by-frame-interpolation)
to insert frames into an already generated temporal sequence. That second stage
uses FFmpeg motion compensation, needs only the LTX source frames, and runs
for final FPS above 12. It does not load this guide's image-model pipeline.

The method follows the prompt/seed semantics of the official
[DiffusionBee Interpolator](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui/blob/master/backends/stable_diffusion/applets/frame_interpolator.py).
For frame `i` of `N`, `t = i / (N - 1)`. All active positive, negative and pooled
prompt tensors use `(1-t)*A + t*B`. Initial noise uses
`sqrt(1-t)*A + sqrt(t)*B`, preserving unit expected variance for independent
Gaussian endpoints. Identical noise is held unchanged, avoiding an amplitude
bump when both seeds are the same. Frame zero and frame `N-1` retain the exact
endpoint inputs. These are independently rendered images; seed changes can
change composition considerably and do not guarantee motion continuity.

## Usage

Use the existing pinned Diffusers Python environment with FFmpeg and FFprobe
on PATH. FFmpeg must expose `libx264` for MP4, or its GIF encoder and palette
filters for GIF. Interpolator requires no new Python
dependency, no OpenCV and no interpolation model download.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend interpolator --preset sd15-compatible \
  --model /absolute/path/to/diffusers-model --local-files-only \
  --prompt 'a city at sunrise' --end-prompt 'a forest at sunrise' \
  --negative-prompt 'blur, low quality' \
  --seed 42 --end-seed 43 --max-frames 120 --fps 12 --steps 20 \
  --output build/interpolator.mp4
```

Use the same seed at both ends (or omit `--end-seed`) for prompt-only changes.
Use the same prompt at both ends (or omit `--end-prompt`) for seed-only changes.
Omitting both end values deliberately yields a constant sequence with a
deterministic scheduler. `--max-frames` includes both endpoints and must be
between 2 and 1,000,000. Duration is `max_frames / fps`; the last frame starts
at `(max_frames - 1) / fps`.

| Option | Default | Meaning |
|---|---|---|
| `end_prompt`, `end_negative_prompt` | Starting text | Final positive and negative text |
| `end_prompt_2`, `end_negative_prompt_2` | Final primary text, or explicitly supplied starting secondary text | Final SDXL/FLUX secondary-encoder text |
| `end_seed` | `seed` | Final initial-noise seed, in `[-2^63, 2^64-1]` |
| `max_frames` / `frames`, `fps` | 120, 12 | MP4: positive FPS at most 12; GIF: `100/65535` to 100 |
| `duration` | none | Seconds rounded to frames; exclusive with a frame count |
| `video_crf`, `video_preset` | 18, `medium` | H.264 quality and encoding speed |
| `ffmpeg`, `ffprobe` | Executable names | Override media tool paths |
| `encoding_timeout` | 300 | Encoding/decode verification timeout in seconds |

An explicit empty primary negative prompt remains empty. Empty secondary
strings retain the existing Diffusers fallback to primary text. JSON and
Python values use the same strict schema as CLI arguments; CLI overrides JSON.
`--print-config` validates and exports all resolved endpoint values without
importing Torch or media packages, loading weights or running FFmpeg.

```sh
python3 reference/generate.py --backend interpolator --model /absolute/path/image-diffusers \
  --config reference/diffusers/interpolator.example.json --print-config
```

In Python, use `generate.resolve_request({...})` to validate/resolve settings.
`prepare_pipeline_with_adapters()` computes and attaches the two conditioning
endpoints before execution hooks. `interpolator_runtime.render_interpolator_frames()`
then accepts that pipeline/request and a frame-writing callback. The regular
CLI performs this lifecycle and publishes the video bundle automatically.

## Runtime and compatibility

The existing SD 1.x, SDXL/Illustrious/NoobAI/Pony and FLUX.1 preset families
use their existing model loader. Local safetensors checkpoints still need the
appropriate model configuration/extras. Model/VAE overrides, LoRA, textual
inversion, static ControlNet, scheduler configuration and CPU/Metal/CUDA/ROCm
execution retain their existing contracts. Generic video models, raw GGUF and
ComfyUI workflows are not converted into Interpolator pipelines.

Both endpoints are encoded once on CPU after text embeddings and LoRA are
installed, before accelerator/offload hooks. `cpu_text_encoding` resolves to
true; explicit `--no-cpu-text-encoding` is rejected in this mode. Denoising and
decoding use the selected execution device. Negative and pooled tensors are
blended alongside positive embeddings. FLUX negative prompts need
`true_cfg_scale > 1`, and FLUX noise is packed through its pipeline's own
2x2 latent layout. Prompt/noise interpolation uses float32 intermediates and
casts to the selected inference dtype with finiteness checks.

Schedulers are recreated for every frame. Additional stochastic sampler
noise uses an independent generator reset to the starting seed for every
frame, preventing generation order from advancing a shared random stream.
Consequently, an end frame from a stochastic sampler need not be pixel-identical
to a standalone image generated with the end seed, even though its initial
latent and conditioning are exact. Hardware/runtime changes can also change
pixels. Deterministic samplers are preferable for smoother transitions.

The mode requires `num_images=1`, complete denoising and `.mp4` or `.gif` output. HiRes Fix,
external latent/embedding files, SDXL early stopping and Deforum-specific
camera/prompt schedules/seed policies are rejected. Nonzero guidance rescale
with ControlNet is rejected. All other generation settings stay fixed across
the two endpoints. This mode provides two endpoints, not a multi-keyframe
timeline, image inversion, audio synchronization or resumable rendering.
Complete inference runs in Python; the C++ API keeps its existing contract
and compute-component responsibilities.

## Artifacts and validation

`build/interpolator.mp4` is accompanied by `interpolator.json` and
`interpolator-frames/frame-000000.png` onward. Each frame is streamed to disk;
only endpoint tensors and the current frame remain in memory, while compact
per-frame provenance accumulates for the JSON report. The report records
endpoint configuration, model/adapter identity, tensor shapes and canonical
float32 hashes, interpolation fraction, actual denoising timesteps, finite
latent status, PNG hashes and decoded MP4/GIF properties.

The shared `animation_video.py` verifies H.264/yuv420p, dimensions, frame count,
FPS and duration by decoding with FFprobe. GIF decoding uses Pillow to check
frames, dimensions, loop and duration with 10 ms timing resolution.
It publishes the completion report
last and restores previous output on ordinary sampling/encoding/publication
failure. Default output names receive a numbered suffix on collision; explicit
paths require `--overwrite` for replacement. Unmanaged frame directories and
symlinks cannot be overwritten. Both modes share an exclusive output lock;
an existing frame bundle must belong to the same animation mode. A forced
process kill/power loss can leave staging/lock files, as described in the
[Deforum artifact contract](deforum-video.md#artifacts-and-verification).

`InterpolatorOptionsTests`, `InterpolatorRuntimeTests` and
`InterpolatorMediaTests` are registered with CTest. Tensor tests need the
generation runtime; media tests need Pillow/FFmpeg. Run the real-inference
smoke on installed hardware:

```sh
reference/diffusers/.venv/bin/python -m unittest discover -s tests -p 'Interpolator*Tests.py'
reference/diffusers/.venv/bin/python tests/InterpolatorDiffusersSmoke.py --device cpu
reference/diffusers/.venv/bin/python tests/InterpolatorDiffusersSmoke.py --device mps
reference/diffusers/.venv/bin/python tests/InterpolatorAdapterSmoke.py --device mps
reference/diffusers/.venv/bin/python tests/InterpolatorAdapterSmoke.py --device mps --dtype float16 --offload model
```

The smoke uses locally created small random SD/SDXL-compatible safetensors,
with and without ControlNet. It verifies changing embeddings/noise/pixels,
endpoint hashes, actual finite diffusion steps and video decoding. It does not
establish trained-model visual quality, FLUX inference or CUDA/ROCm hardware
execution. Artifacts and logs are under `build/interpolator-smoke`.
The adapter smoke adds synthetic nonzero LoRA, a two-vector learned token,
static ControlNet and selectable precision/offload. Textual-inversion vectors
are validated before endpoint encoding; the frame loop uses cached conditions
without reading text-encoder weights after offload converts them to meta tensors.

On the tested M1 Max runtime (PyTorch 2.13.0, Diffusers 0.40.0, Accelerate 1.14.0),
the combined adapter fixture produced non-finite denoising values with FP16
and sequential offload, including after refreshing hooks. FP32 with sequential
offload and FP16 with model offload or resident weights completed. This is a measured numerical
limitation of that combination, not a trained-model compatibility claim.
Use `--dtype float32` or another offload policy for this case. The finite-value
audit rejects a failed frame and leaves existing output intact; it never
silently substitutes precision, repairs NaNs or publishes an invalid video.
