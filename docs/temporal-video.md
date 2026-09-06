# Temporal video generation

`iild-generate --backend video` generates a shot jointly over space and time
with Diffusers' `LTXConditionPipeline`, a video diffusion transformer and a
temporally compressed video VAE. Text, image keyframes and camera descriptions
condition the model. Each shot samples a video latent sequence in one pipeline
call. Multiple shots are assembled into a verified H.264 MP4.

## Reference workflow and dependency choice

[Seedance](https://seed.bytedance.com/en/seedance) documents text/image input,
motion and multi-shot video diffusion. [Higgsfield DoP](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-dop)
documents a keyframe, scene prompt, camera preset or preset mix, duration, seed
and sampling steps. Its model is proprietary. This backend implements that
directed local generation workflow using available model weights; it does not
run either company's proprietary model or claim their output quality.

The existing Diffusers 0.40.0 / PyTorch / Transformers / Accelerate packages
provide the neural model, temporal attention, VAE, scheduler and RAM offload.
No additional Python inference stack or paid API is required. FFmpeg and
FFprobe, already used by the animation backends, encode and verify the result.

The default weights are
[`Lightricks/LTX-Video-0.9.5`](https://huggingface.co/Lightricks/LTX-Video-0.9.5),
pinned to `e58e28c39631af4d1468ee57a853764e11c1d37e`. The 2B model is a practical
local baseline with text and multiple image conditions. Its version-specific
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
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend video \
  --prompt 'A glass bottle on a stone table, warm sunlight glinting through amber liquid.' \
  --camera dolly-in --duration 5 --fps 24 \
  --output build/reference/bottle.mp4
```

For image-to-video, add `--first-frame /absolute/path/keyframe.png`. An optional
`--last-frame /absolute/path/end.png` conditions the final output frame. Inputs
are decoded as static local images, EXIF-oriented, converted to RGB and center
cropped to the output aspect ratio. Keyframes influence model generation;
they are not a promise of pixel-exact endpoints or identity preservation.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
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
reference/diffusers/.venv/bin/python reference/generate.py \
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
| `fps` | 24; also conditions the model's temporal positions |
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

Duration is rounded to the nearest output frame. The sampler pads each shot
to `8k+1` frames, then trims only the extra tail frames; requested duration,
FPS and frame count remain explicit. Each shot accepts 2–4097 output frames,
and a storyboard accepts up to 256 shots. Large requests require more memory.
Inference errors are reported without silent CPU or image-animation fallback.

`--model` accepts a complete local LTX Diffusers directory or a Hub model ID.
A non-default Hub model requires a full immutable `--revision`. Only built-in
LTX transformer/VAE/T5/scheduler classes and safetensors weights are loaded.
Use `--cache-dir` to control storage and `--local-files-only` after download.
Defaults store models under `build/reference/huggingface`. The model identity
record includes component-file hashes and sizes; loading and inference check
for source-file changes.

## Outputs and verification

Every completed run publishes `name.mp4`, `name.json`, and `name-frames/`.
The JSON records model files/revision, requested settings, actual captions,
camera controls, keyframe hashes, cut positions, denoising tensor shapes and
devices, individual PNG hashes, and the decoded MP4's count, duration, FPS,
resolution and SHA-256. Non-finite latents or decoded pixels are rejected.

The existing bundle remains intact if loading, sampling, encoding or
publication fails. A common exclusive lock also prevents animation and video
jobs from publishing to the same destination concurrently. Existing explicit
outputs require `--overwrite`; generated default names receive a new run suffix.

`VideoOptionsTests.py` covers planning and routing, `VideoRuntimeTests.py` covers
tensor/media/publication contracts, and `VideoDiffusersSmoke.py` runs real tiny
LTX pipelines for text, endpoint-keyframe and chained-shot requests on CPU or
MPS. Tiny random fixtures prove execution, not trained-model visual quality.
The native C++ SDK's existing manifest/computation boundary remains unchanged;
this feature executes through the installed Python generation entry point.
