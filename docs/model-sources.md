# Three model input locations

The Python generation entry point (`reference/generate.py`, installed as
`iild-generate`) accepts exactly one model source. Model location and generation
method are separate choices. A cloud model is evaluated remotely by its provider;
its weights are not downloaded for local execution.

| Input | CLI | JSON / Python values | Execution |
|---|---|---|---|
| Local path | `--model-path PATH` | `model_path` (canonical `model`) | Existing local checkpoint/Diffusers runtime |
| Direct API | `--model-api URL` | `model_api` | POST to the supplied inference endpoint |
| Cloud model | `--model-cloud OWNER/MODEL --model-provider PROVIDER` | `model_cloud`, `model_provider` | Hugging Face Inference Provider selected explicitly by name |

`--model` remains an alias for **local** `--model-path`. It never guesses that a
missing path is a cloud model. The two remote inputs accept no local model path.
Mixed source types, including mixtures between a config and CLI, are rejected;
same-field CLI values can override the config. Local config paths resolve against
the configuration directory; API URLs and cloud IDs remain opaque identifiers.

`model_sources.ModelInput` records `kind` (`local`, `api`, `cloud`), `location`,
`provider`, `token_env`, `family`, and `timeout`. The C++ metadata inspector and
component executor retain their existing contracts; this feature does not add
native C++ HTTP inference or move Python model execution into the library.

## Commands

```sh
# Existing locally installed image model.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-path /absolute/path/image-diffusers --prompt 'A glass bottle.'

# A deployed endpoint implementing the contract below.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-api https://inference.example/image --model-token-env IMAGE_API_TOKEN \
  --prompt 'A glass bottle.' --width 512 --height 512 --output build/api-image.png

# OWNER/MODEL must have a live mapping for the chosen provider and image task.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-cloud OWNER/MODEL --model-provider fal-ai \
  --prompt 'A glass bottle.' --output build/cloud-image.png

# LTX first, then the existing local frame Interpolator, with final 24 FPS.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-api https://inference.example/ltx --model-token-env VIDEO_API_TOKEN \
  --model-family ltx --prompt 'A slow camera orbit around a glass bottle.' \
  --frames 49 --fps 24 --width 704 --height 480 --output build/api-video.mp4
```

The domains and model IDs above are placeholders, not provisioned services.
For cloud LTX, replace `--model-api` with `--model-cloud OWNER/LTX-MODEL
--model-provider PROVIDER`; retain `--model-family ltx`. A provider must actually
serve that model for `text-to-video`. Neither a catalog entry nor an accepted
model ID establishes a live deployment.

Tokens are read at execution from `--model-token-env NAME`. Cloud input defaults
to `HF_TOKEN`; direct API input is unauthenticated unless a token variable is
specified. A specified but absent/empty variable fails before submission. Configs
and sidecars store the variable name, never its value. Use HTTPS; loopback HTTP
is supported for local servers and tests. Credentials/query strings in endpoint
URLs are rejected. `--model-timeout` is a finite positive number up to 86400
seconds (default 300). `--print-config` validates without contacting a service,
reading tokens, loading tensors or downloading weights.

Equivalent image JSON, also accepted by `generate.resolve_request(values)`:

```json
{
  "model_cloud": "OWNER/MODEL",
  "model_provider": "fal-ai",
  "model_token_env": "HF_TOKEN",
  "prompt": "A glass bottle.",
  "width": 512,
  "height": 512,
  "output": "./cloud-image.png"
}
```

`generate.resolve_request` returns `(preset, args)` for local images and
`(None, args)` for remote images; the latter execute through
`remote_generation.generate_image(args)`. Video values use
`video_options.build_parser().parse_values(values)` and `resolve_options`;
`generate_video.py` selects the local or remote evaluator. Resolved configurations
can be replayed using the same schema.

## API and provider adapters

Direct API input implements a synchronous Hugging Face-style inference endpoint
contract, not arbitrary vendor REST APIs. It sends one JSON POST per image/shot:

```json
{
  "inputs": "the prompt, including camera text for video",
  "parameters": {
    "width": 704,
    "height": 480,
    "num_inference_steps": 30,
    "guidance_scale": 3,
    "negative_prompt": "blurry",
    "seed": 42,
    "num_frames": 25,
    "frame_rate": 12
  }
}
```

`num_frames`/`frame_rate` are included only for video and describe the **LTX source**
request, not final interpolated frame count/FPS. The endpoint must return HTTP 200
and raw `image/png` or `video/mp4` bytes, up to 512 MiB. Job-ID polling, redirects,
URL result envelopes and OpenAI-style image APIs are not this protocol. Failed
or timed-out direct requests are not resubmitted automatically.

Cloud input uses the existing pinned `huggingface-hub==1.29.0`
[`InferenceClient`](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client),
its provider mapping and `text_to_image`/`text_to_video` methods. The provider and
Hub model ID are supplied separately; `auto` provider selection is not used.
Available tasks, native resolution, frame-count limits and scheduling depend on
the deployment. The standardized video client has no width/height/frame-rate
parameters: choose a deployment whose native resolution matches the request.
Returned source frames are placed on iiLocalDiffusion's requested timeline;
the provider's container playback rate is not the final output rate. Responses
with the wrong dimensions or source frame count fail, without silently resizing,
duplicating frames or publishing a completed result. Provider billing is outside
iiLocalDiffusion; automated tests do not submit paid cloud generations.

No new library is added: direct HTTP uses Python's standard library, cloud
inference reuses the maintained Apache-2.0 Hugging Face client already pinned in
`requirements-common.txt`, and decoding/encoding reuses Pillow and FFmpeg.
Provider-specific errors are summarized without echoing credentials or response
bodies. No provider coverage beyond the installed client's adapters is implied.

## Generation rules and verified boundaries

Images support prompt, negative prompt, resolution, steps, guidance, seeded
batches and PNG output. Remote calls do not expose local scheduler, adapter,
latent, text-encoder or hardware controls; explicitly supplied unsupported
options fail before inference. A remote model is selected by its source argument,
so `--base-model` and local preset identities do not replace it.

The default video route remains LTX then frame Interpolator at final 24 FPS.
Remote video requires `--model-family ltx`: this is the caller's declaration,
not local verification of the provider's weights. Every shot requests LTX's
8n+1 source length, decodes the result, retains the planned source frames, and
applies the same FFmpeg motion interpolation and output verification as local
LTX. Independent text storyboard shots and camera prompt controls are supported.
Keyframes, previous-shot image continuity and latent controls require the local
LTX evaluator until a provider adapter can honor those exact contracts.

FPS <= 12 or GIF still routes automatically to local Deforum (or prompt/seed
Interpolator). Supplying a remote source for these image-animation modes fails
before submission because a text-to-image API does not expose their latent
controls. Explicit `--backend video` retains LTX at low FPS and skips the second
stage. GIF is not an LTX output. No remote provider/model is substituted to get
around these rules; Higgsfield and removed hosted Seedance routes remain removed.

Output sidecars identify the source and actual decoded media, and clearly mark
remote weights as not verified locally. Video bundles preserve received source
clips, decoded PNG hashes, anchors, shot boundaries and final encoding checks.
Malformed responses or failures preserve existing outputs. Tests cover real
loopback HTTP image/video transfers, authentication headers, failures, atomic
publication and real FFmpeg interpolation. Cloud tests verify client dispatch
with simulated provider responses; they do not establish live provider/model
availability, image quality or successful billed inference.
