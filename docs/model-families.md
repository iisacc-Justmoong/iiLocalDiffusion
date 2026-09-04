# Model families and local checkpoint generation

`reference/diffusers/generate.py` separates a model's name from its inference
architecture. Illustrious, NoobAI and Pony XL use the SDXL pipeline; FLUX.1 dev
and Krea use guidance-distilled FLUX. The original `sd15`, `sdxl-base` and
`flux1-schnell` presets retain their strict reference-snapshot contracts.

| Preset | Architecture | Default model or required input | Training settings |
| --- | --- | --- | --- |
| `sd15-compatible` | SD1.x | Pinned SD1.5; accepts a selected SD1.x derivative | Keeps the loaded compatible scheduler |
| `flux1-schnell-compatible` | FLUX.1 schnell | Pinned FLUX.1 schnell; accepts a selected schnell derivative | Keeps the loaded flow scheduler; no guidance embedding tensors |
| `sdxl` | SDXL | Pinned SDXL base; accepts a selected SDXL derivative | Keeps the loaded scheduler |
| `illustrious` | SDXL | Pinned OnomaAI Illustrious XL early release | Keeps the loaded scheduler |
| `noobai` | SDXL | Pinned Laxhar NoobAI XL 1.0 EPS | Keeps the loaded scheduler |
| `noobai-v-pred` | SDXL | Pinned Laxhar NoobAI XL V-Pred 1.0 | Euler, `v_prediction`, zero-terminal-SNR beta rescaling |
| `pony` | SDXL | Explicit local checkpoint or selected Diffusers directory required | SDXL configuration is only the conversion fallback |
| `flux1-dev` | FLUX.1 | Pinned Black Forest Labs FLUX.1 dev | Guidance embeddings, default guidance 3.5, 512 T5 tokens |
| `flux1-krea-dev` | FLUX.1 | Pinned Black Forest Labs FLUX.1 Krea dev | Guidance embeddings, default guidance 4.5, 512 T5 tokens |

Presets identify architectures and defaults, not all checkpoints bearing a
marketing name. For example, Pony versions based on SD1.x require the SD1.x
architecture, and NoobAI's EPS and V-pred weights require distinct prediction
settings. Models do not become compatible merely by renaming their files.

## Local files and directories

A complete Diffusers directory is loaded using its stored components and
scheduler. Remote overrides still require an immutable 40-character commit SHA.
Single-file models must be local `.safetensors` (or `.safetensor`) files. Pickle
checkpoints are not enabled. Checkpoint/denoiser files and VAE/LoRA files retain
separate input roles, file-size and SHA-256 provenance, and before/after identity
checks.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset illustrious --model /absolute/path/illustrious.safetensors \
  --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/illustrious.png

reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset noobai-v-pred --model /absolute/path/noobai-vpred.safetensors \
  --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/noobai-vpred.png

reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset flux1-krea-dev --model /absolute/path/flux1-krea-dev.safetensors \
  --model-config /absolute/path/complete-flux-dev-diffusers-package \
  --local-files-only --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/krea.png
```

Single-file conversion needs the matching tokenizer/component configuration.
Full SDXL checkpoints must contain both recognized CLIP encoders and a VAE,
unless `--vae` supplies it. Denoiser-only files use the selected configuration
package's encoders and VAE, so that package must include those weights.
`--local-files-only` requires all of this material to be present locally; it
never obtains missing components from an unpinned fallback. Gated remote model
repositories still require the user's existing access and authentication.

For V-pred derivatives without a dedicated preset, use
`--preset sdxl --prediction-type v_prediction`. Use
`--scheduler-config '{"rescale_betas_zero_snr":true}'` when the model's training
recipe requires zero-SNR rescaling. `--prediction-type auto` preserves the
loaded value, except for a preset's documented training defaults. Conflicting
explicit `--prediction-type` and scheduler JSON values fail instead of being
silently reordered. Scheduler configuration and the applied preset defaults
are recorded in output metadata.

The NoobAI V-pred model card explicitly prescribes Euler, V-prediction and
zero-SNR rescaling, although its checked-in scheduler configuration declares
`epsilon`. Its dedicated preset applies the documented recipe after loading.
A user-supplied scheduler or scheduler configuration can deliberately override
the preset defaults; this does not establish quality for that recipe.

## Compatibility boundaries and verification

SDXL family validation accepts built-in compatible diffusion schedulers and varying
sample-size, prompt-zeroing and VAE upcast policies. It still rejects incorrect
UNet input/output channels, SDXL refiner/inpainting shapes, text-encoder widths,
latent dimensions/scales and custom component declarations. Flow schedulers
cannot be used as SDXL diffusion schedulers. FLUX dev/Krea requires guidance
embedding tensors; applying a schnell configuration to those weights is rejected
before a loader can discard the extra tensors.

`tests/ModelFamilyTests.py` covers the family routing, scheduler training defaults,
configuration allowlist, cross-family rejection and safe selection boundaries.
ControlNet pipeline conversion preserves the underlying family, guidance and
scheduler training settings; augmented pipeline validation also rejects a
ControlNet from a different architecture.
`ModelLoadingTests.py` and `GenerationSchedulerTests.py` retain the original
loader and scheduler regression coverage. Synthetic tiny models can verify
actual loading, denoising and guidance plumbing, but are not proof of the visual
quality or completeness of a full-size community model.

The SD1.x compatibility preset keeps the 4-channel UNet and 768-wide text/cross-attention
contracts. It rejects SD2.x and inpainting architectures even when their scheduler
is otherwise accepted. The schnell compatibility preset accepts flow scheduler
configuration changes and still rejects guidance-distilled dev/Krea weights.

The 2026-09-04 local integration run also used the public
`reference/generate.py` entry point with cached original SDXL 1.0 fp16 weights:
`illustrious` routing produced a 128x128 EPS image, and `noobai-v-pred` routing
with a synthetic ControlNet produced a 192x192 HiRes image. Both V-pred stages
recorded zero-SNR rescaling and finite latents (two base steps and one refinement
step). The local report is
`build/reference/model-families/sdxl-fixture-cli-smoke.json`. This verifies
routing/composition with an SDXL fixture, not trained Illustrious/NoobAI/Pony
weights or their image quality; the NoobAI EPS and Pony routes were configuration
checks only in that run.

These native adapters do not claim every Civitai entry, task type, quantization
container or custom node workflow. Quantized variants and other architectures
need their own verified runtime/container support. Inference compatibility also
does not grant model redistribution or commercial-use rights; consult the
selected checkpoint's license independently.

## Primary references

Verified against these upstream model cards/configurations and installed
Diffusers 0.40.0 on 2026-09-04:

- [Illustrious XL model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [NoobAI V-pred model card and Diffusers recipe](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/README.md)
- [NoobAI V-pred scheduler configuration](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/scheduler/scheduler_config.json)
- [Pony XL author checkpoint release](https://huggingface.co/AstraliteHeart/pony-diffusion-v6/commit/5ec9c05863255568f1b59753e3838107befaa712)
- [FLUX.1 Krea dev model card](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
- [Black Forest Labs model specifications](https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py)
- [Diffusers FLUX pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_flux.py)
