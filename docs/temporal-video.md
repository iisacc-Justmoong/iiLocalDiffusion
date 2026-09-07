# Temporal video generation

`iild-generate --backend video` generates a shot jointly over space and time
with Diffusers' `LTXConditionPipeline`, a video diffusion transformer and a
temporally compressed video VAE. Text, image keyframes and camera descriptions
condition the model. Each shot samples a video latent sequence in one pipeline
call. Ordinary video then passes through a **second-stage frame Interpolator**
before shots are assembled into a verified H.264 MP4.

## LTX followed by frame interpolation

For final FPS above 12, generation always runs in this order:

1. The caller's local LTX model generates a temporal sequence of source frames.
2. The frame Interpolator inserts the missing output frames within each shot.
3. The completed sequence is encoded and decoded for MP4 verification.

`--fps`, `--frames` and `--duration` describe the **final output**, including
interpolated frames. `--interpolation-factor` defaults to **2** and accepts
integers 2–8. It sets the usual spacing of LTX source frames on that output
timeline; it does not multiply the requested final FPS or duration. Start/end
frames and every timed image condition are always included as LTX source
anchors, even when they fall between regular samples. The source sampling FPS
preserves the shot's first-to-last time span; added anchors can make it differ
slightly from `final_fps / factor`. The interpolator uses the recorded output
timestamps when placing those source frames.

For example, a 5-second, 24 FPS output contains 120 frames. The default plan
keeps 61 LTX source frames, pads temporal inference to 65 frames for the VAE,
and inserts 59 frames in the second stage. All retained source PNGs, including
the last frame, are copied unchanged into the final PNG sequence. Very short
or fully keyframed shots can have zero missing frames; the report records this
instead of claiming that additional frames were synthesized.

The **FPS <= 12 and GIF exceptions retain the existing standalone Deforum or
prompt/seed Interpolator routes**. Explicit low-FPS LTX requests remain a single
LTX stage. The high-FPS restriction on `--backend interpolator` concerns that
standalone image-model animation, not the video backend's postprocessing stage.

The postprocessor uses the existing FFmpeg
[`minterpolate` motion-compensated filter](https://ffmpeg.org/ffmpeg-filters.html#minterpolate)
on CPU. It requires no image model, interpolation model, download, OpenCV or new
Python dependency. Its filter availability is checked before LTX weights load.
This is frame interpolation over LTX's lossless PNG sequence, separate from
the prompt/noise interpolation described in [Interpolator animation](interpolator-video.md).
FFmpeg estimates motion; occlusions and complex motion may still produce artifacts.

Each shot is processed independently, so no synthetic transition is introduced
across a storyboard cut. Boundary padding supplies interpolation lookahead and
is discarded. LTX pipeline references are released before the CPU postprocess.
Failures in the second stage abort the entire new bundle rather than publishing
the unfinished first stage.

## Local model and dependencies

Local generation requires `--model-path /absolute/path/to/local-ltx-model` (legacy `--model`), containing
`model_index.json`, the transformer, temporal VAE, text encoder, tokenizer and
scheduler. LTX is the default video architecture for automatic requests above
12 FPS or with FPS omitted (default 24), unless the output is GIF. There is no
default model path or automatic model download. Remote execution is selected
explicitly through `--model-api` or `--model-cloud` plus `--model-provider`,
with `--model-family ltx`; see [model sources](model-sources.md).
For the local source, CLI, JSON `model_path`/`model` and the Python request use the
same local-directory contract. See [local model generation](local-model-generation.md).

The existing Diffusers 0.40.0 / PyTorch / Transformers / Accelerate packages
provide the neural model, temporal attention, VAE, scheduler and RAM offload.
No additional Python inference stack or paid API is required. FFmpeg and
FFprobe, already used by the animation backends, encode and verify the result.
The selected FFmpeg must also provide `minterpolate` and `tpad` for ordinary
two-stage video. Its existing external-executable licensing remains unchanged.

The previously validated model is
[`Lightricks/LTX-Video-0.9.5`](https://huggingface.co/Lightricks/LTX-Video-0.9.5),
at `e58e28c39631af4d1468ee57a853764e11c1d37e`. Its compatible local snapshot can
be supplied as `--model`; it is never selected implicitly. Its version-specific
Open RAIL-M license permits commercial use subject to its restrictions. The
weights are downloaded separately and are not redistributed in the SDK.
Newer LTX versions have different licenses and may require substantially more
memory; see the [upstream model and license catalog](https://huggingface.co/Lightricks/LTX-Video).

## Generate a video

Install the video tokenizer dependency alongside the existing managed Python
environment (see [environment setup](../reference/diffusers/README.md)):

```sh
uv pip install --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements-video.txt
```

This pins `protobuf`, required to convert the model's SentencePiece tokenizer.
Tokenizer validation runs before the large model weights are read. Run with
the managed Python environment, or use the installed
`iild-generate` launcher with `IILD_PYTHON_EXECUTABLE` selecting that environment:

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video \
  --prompt 'A glass bottle on a stone table, warm sunlight glinting through amber liquid.' \
  --camera dolly-in --duration 5 --fps 24 --interpolation-factor 2 \
  --output build/reference/bottle.mp4
```

For image-to-video, add `--first-frame /absolute/path/keyframe.png`. An optional
`--last-frame /absolute/path/end.png` conditions the final output frame. Inputs
are decoded as static local images, EXIF-oriented, converted to RGB and center
cropped to the output aspect ratio. Keyframes influence model generation;
they are not a promise of pixel-exact endpoints or identity preservation.

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video --first-frame /absolute/path/keyframe.png \
  --prompt 'The subject turns slowly toward the window in warm afternoon light.' \
  --camera pan-left zoom-in --duration 3 --seed 7 \
  --output build/reference/keyframe-video.mp4
```

`--camera` accepts up to three compatible motions. List the editable prompt
descriptions with `--backend video --list-camera-motions`. Supported choices
include dolly, pan, tilt, orbit, crane, zoom, handheld, tracking and dolly zoom.
`none` preserves the scene prompt; `static` requests a locked tripod. Neutral,
static, duplicate or directly opposing motions cannot be mixed. Camera control
is learned text conditioning, not a measured or guaranteed 3D camera path.
The motion description precedes the scene caption. Requests exceeding the
selected tokenizer limit fail with an actionable error instead of dropping
camera or scene instructions through silent truncation.

## Shot plans and reference continuity

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video --storyboard reference/diffusers/video-storyboard.example.json \
  --output build/reference/story.mp4
```

A storyboard is a JSON object with a `shots` array. Shot fields override the
global defaults: `prompt`, `negative_prompt`, `camera` (array), `frames` or
`duration`, `seed`, `first_frame`, `last_frame`, `continue_previous`, and
`conditions`. Paths inside a shot are relative to the storyboard file.
For a timed image condition use:

```json
{"image": "detail.png", "frame": 24, "strength": 0.8}
```

The `conditions` array accepts up to 32 entries with unique frame indices.
Strength is in `[0.001,1]`. `continue_previous: true` conditions frame zero on the
previous shot's generated last frame and cannot conflict with another frame-zero
reference. Each shot retains its prompt, camera, seed and reference metadata.
Shots meet at cuts; this is sequential shot generation, not one native
multi-shot model call. Semantic continuity still depends on prompts and weights.

## Configuration and execution

JSON uses the same argument names with underscores. An explicit CLI value
overrides its JSON value. `--config reference/diffusers/video.example.json`
selects the video backend through `backend: "video"`. `--print-config` resolves
and validates configuration without importing Torch or downloading weights.

| Setting | Default and contract |
| --- | --- |
| `width`, `height` | 704 × 480; multiples of 32, from 32 to 4096 |
| `duration` / `frames` | 5 seconds or an explicit frame count; mutually exclusive |
| `fps` | 24; final output rate after interpolation |
| `interpolation_factor` | 2, integer 2–8; source-frame spacing for FPS above 12 |
| `steps`, `guidance_scale` | 30, 3 |
| `seed` | 42; subsequent shots increment it unless specified |
| `max_sequence_length` | 256; T5 prompt limit, configurable up to 512 |
| `device` | GPU-required auto, or explicit cpu/mps/metal/cuda/rocm |
| `dtype` | auto uses float32 on CPU and bfloat16 on accelerators |
| `offload` | auto uses model RAM offload on GPU and none on CPU |
| `cpu_text_encoding` | false; opt in to encode all captions on CPU and release T5 before sampling |
| `vae_tiling` | true |
| `decode_timestep`, `decode_noise_scale` | 0.05, 0.025 |
| `image_cond_noise_scale` | 0 |
| `video_crf`, `video_preset` | 18, medium |
| `encoding_timeout` | 300 seconds |

Duration is rounded to the nearest output frame. The sampler pads each shot's
**LTX source count** to `8k+1` frames, then trims only the extra tail frames.
The Interpolator fills the remaining output positions; requested duration,
FPS and frame count remain explicit. Each shot accepts 2–4097 output frames,
and a storyboard accepts up to 256 shots. Large requests require more memory.
Inference errors are reported without silent CPU or image-animation fallback.

For local execution, `--model-path`/`--model` requires a complete LTX Diffusers directory. Non-null revision
arguments are rejected. Only built-in components and safetensors are loaded;
missing resources fail without a download. `--cache-dir` controls temporary
working storage, not model selection. The model identity
record includes component-file hashes and sizes; loading and inference check
for source-file changes.

## Outputs and verification

Every completed run publishes `name.mp4`, `name.json`, and `name-frames/`.
The JSON records model files/revision, requested settings, actual captions,
camera controls, keyframe hashes, cut positions, denoising tensor shapes and
devices, individual PNG hashes, and the decoded MP4's count, duration, FPS,
resolution and SHA-256. Non-finite latents or decoded pixels are rejected.

`stages` records LTX and Interpolator separately. `source_frames` identifies the
LTX PNGs under `name-frames/ltx/shot-NNNN/`, with hashes and output positions;
`frames` identifies the complete output sequence. Each inserted frame records
its neighboring source indices and interpolation fraction. `interpolation`
records the FFmpeg filter, CPU execution, factor, actual inserted count,
per-shot verification and timings. Source anchor hashes must survive unchanged.

The existing bundle remains intact if loading, sampling, interpolation, encoding or
publication fails. A common exclusive lock also prevents animation and video
jobs from publishing to the same destination concurrently. Existing explicit
outputs require `--overwrite`; generated default names receive a new run suffix.

`VideoOptionsTests.py` covers planning and routing, `VideoRuntimeTests.py` covers
tensor/media/publication contracts. `VideoInterpolationTests.py` checks timing,
motion insertion, source hashes, off-grid anchors, cut boundaries and failures.
`VideoDiffusersSmoke.py` runs real tiny
LTX pipelines for text, endpoint-keyframe and chained-shot requests on CPU or
MPS with both stages enabled by default; `--fps 8` checks the low-FPS exception.
Tiny random fixtures prove execution, not trained-model visual quality.
The native C++ SDK's existing manifest/computation boundary remains unchanged;
this feature executes through the installed Python generation entry point.
