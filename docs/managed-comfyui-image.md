# Optional managed ComfyUI checkpoint backend

Dreamscapes 등 소비 앱은 `--work-dir <앱 임시 위치>/<job>/runtime`을 지정하여 작업 로그·워크플로·모델 링크·서버 중간 결과를 Society 밖에서 잠시 사용하고 정리한다. 큐와 실행 상태는 앱 메모리에만 둔다. 없거나 비어 있는 실제 디렉터리만 받으며 기존 내용과 리디렉션된 경로는 거부한다. 최종 실행기 결과는 기존 `--output-dir`로 따로 지정한다. Dreamscapes는 앱 전용 임시 위치에서 결과를 검증하고 이미지 파일만 `Generation History/` 바로 아래에 저장한다. 앱별 하위 폴더나 Asset Library 등록은 만들지 않는다. 옵션을 생략하면 기존 SDK `build/reference/local-image-jobs/` 배치를 사용한다. 원본 모델은 복사하지 않고 작업용 링크를 사용하며, `.safetensor`와 대문자 확장자는 이 링크에서 `.safetensors`로 정규화한다. 모델 원본 이름과 해시 검증은 유지한다. `LocalImageTests`가 Society 작업 경로·기존 내용 보존·링크 거부·확장자 정규화를 검사한다.

`reference/generate.py --backend comfyui-local --model /absolute/path/download.safetensors` explicitly
inspects an existing local weight file and starts an isolated ComfyUI process.
It assembles an image workflow from the model family, executes it, saves decoded
images and records the original file hashes. No workflow editing or separately
started server is required for supported automatic recipes.

## Install once

`--cache-dir`는 실행기 캐시 경로이다. Dreamscapes는 앱 전용 임시 작업 위치의 `cache/`를 전달한다. Python bytecode·런타임 캐시·작업 로그·중간 결과는 작업 종료와 함께 정리하며 Society에 영구 저장하지 않는다. 지정하지 않으면 기존 SDK 캐시 경로를 사용한다.

```bash
reference/diffusers/.venv/bin/python reference/setup_comfyui.py \
  --python reference/diffusers/.venv/bin/python
```

The installer requires Git and `uv`. It checks out ComfyUI
`e80c1570b6b44a2557d5d8e341e05782d18c9bbb` and ComfyUI-GGUF
`6ea2651e7df66d7585f6ffee804b20e92fb38b8a` into `build/reference/`, installs
their dependencies in `build/reference/comfyui-venv`, and records the resolved
packages in `build/reference/comfyui-installed.lock`. Source revisions are
pinned; the lock records the platform-specific dependency resolution. Existing
foreign or edited runtime checkouts are rejected rather than overwritten.
New checkouts are published only after a successful fetch, so failed downloads
can be retried without leaving an incomplete runtime source directory.
The original Diffusers environment and C++ library are independent.

The full engine has substantially more dependencies than the HTTP client,
including Torch, vision/audio packages, tokenizers and the official frontend.
Weights are supplied locally; this command downloads software packages only.
ComfyUI is GPL-3.0 and ComfyUI-GGUF is Apache-2.0. Their source and dependencies
remain outside the library package.

## Bundled checkpoint

```bash
reference/diffusers/.venv/bin/python reference/generate.py --backend comfyui-local \
  --model /models/illustration.safetensors --base-model Illustrious \
  --prompt "a lighthouse above a calm ocean" --device mps \
  --output-dir build/reference/lighthouse
```

SD1/SD2/SDXL/SD3/FLUX tensor signatures can infer the architecture without
`--base-model`. A sidecar `download.civitai.info` can retain the exact Civitai
identity, optionally verified with its SHA256. Filenames do not prove identity.
Supply the category for architectures whose tensor layout is not recognized.
An explicit category contradicting the detected tensors is rejected.

For NoobAI v-prediction checkpoints, embedded/sidecar prediction metadata is
used automatically. Otherwise pass `--prediction-type v_prediction`; pass
`--zsnr` when the model's sampling recipe requires zero-terminal SNR.
`--guidance-scale` controls classifier-free guidance. FLUX dev/Krea's separate
embedded guidance is `--embedded-guidance` (default 3.5). `--clip-skip` follows
the existing convention: zero selects the last layer, one the penultimate.

The legacy `.ckpt`, `.pt`, `.pth`, and `.bin` containers are safely converted to
cached safetensors before loading. See [conversion restrictions](checkpoint-formats.md).

Omitting `--seed` selects a fresh random base seed for each request. Explicit
seeds, including zero, are preserved. `local-image.json` records the selected
seed in `request.seed`, and the workflow uses the same seed for HiRes passes.

## Split or quantized model

A denoiser alone needs the matching text encoders and VAE. GGUF selects the
installed native GGUF loader, preserving quantization. For example:

```bash
reference/diffusers/.venv/bin/python reference/generate.py --backend comfyui-local \
  --model /models/krea.gguf --base-model 'Flux.1 Krea' \
  --vae /models/ae.safetensors \
  --text-encoder /models/clip_l.safetensors \
  --text-encoder-2 /models/t5xxl.gguf \
  --prompt "a lighthouse above a calm ocean" --device cuda \
  --output-dir build/reference/krea-lighthouse
```

`--components '{"vae":"...","text_encoder":"..."}'` also accepts explicit
local paths. Encoder inputs are consecutive through `text_encoder_4`.
`decoder` supplies Stable Cascade's stage B denoiser; `model_negative` supplies
Ideogram4's unconditional denoiser. Paths are registered through symlinks in
each job's private model inventory. No component download or filename matching
is performed. See the [architecture recipe table](comfyui-image-workflows.md).

`--model-type checkpoint|diffusion_model` overrides the inspection of bundled
versus split files. Explicit `--backend preset` retains the original image
oracle's LoRA, ControlNet, textual inversion and HiRes inputs. Explicit
`--pipeline-class`/`--pipeline-inputs` selects generic Diffusers; `--workflow`
selects an existing local API workflow. Unsupported arguments are errors.

## Evidence and execution limits

`--print-config` inspects only the download. `--validate-only` starts the engine
and validates node/input availability without sampling. Neither proves inference.
Successful sampling writes the images, `generation.json` and `local-image.json`
to a new or empty output directory. The latter records the original model and
component hashes, conversion manifests, detected architecture, complete graph,
runtime device information and image dimensions. Hashes are checked again after
execution, along with the staged symlink targets actually passed to the engine.
The public command delegates to the managed interpreter when necessary; the
host Python does not need Torch or Pillow. Logs, node schemas and staged paths remain under
`build/reference/local-image-jobs/`.

The managed process listens on loopback only, disables hosted API nodes and
loads only the selected GGUF extension. It is stopped on completion or failure.
`--device auto` requires an accelerator; CPU must be selected explicitly.
`--runtime-source` and `--runtime-python` permit an independently installed
compatible runtime. `--timeout` and `--startup-timeout` bound execution/startup.

The automatic builder covers its documented text-to-image architectures.
An image-conditioned model requires an input image and its appropriate pipeline
or workflow. LoRA, VAE, embeddings, ControlNet and upscalers are components;
audio and 3D weights are not standalone text-to-image generators. A category,
valid container or existing node does not establish correct sampling with every
downloaded weight. See [the actual validation evidence](civitai-validation.md).

Sources: [ComfyUI source](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb),
[GGUF loader source](https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a),
[official installation](https://docs.comfy.org/installation/manual_install).
