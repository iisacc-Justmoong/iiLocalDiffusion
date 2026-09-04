# Civitai compatibility validation — 2026-09-04

The implementation exposes downloaded-file inspection, managed local image
generation, named image presets, generic built-in Diffusers pipelines and
explicit local ComfyUI workflows through `reference/generate.py`.
See [usage and the full catalog](civitai-models.md).

## Verified environment and checks

- CMake configure and `cmake --build build --parallel 4`: passed.
- `ctest --test-dir build --output-on-failure`: **52/52 passed**.
- Python unittest discovery: **539 cases, 537 passed and 2 opt-in cases skipped**.
  The opt-in real Torch conversion suite was also run in the installed ML
  environment: **22/22 passed**, including those two cases.
- `git diff --check`: passed.
- Diffusers 0.40.0, PyTorch 2.13.0, SentencePiece 0.2.2.
- Managed ComfyUI 0.34.0 and the pinned ComfyUI-GGUF extension are installed in
  a separate environment. Actual MPS image inference and clean process shutdown
  were verified through the public CLI, including a plain system Python caller.
- Live node-schema validation: **152 graphs** for **52 Civitai image labels**
  across **25 architectures**. Checkpoint, split and GGUF loader schemas are
  included; inventory filenames are fixtures, so this does not prove tensor loading.
- Offline installed-runtime audit of all 105 catalog rows:
  **64 pipeline classes available, 15 workflow-required, 24 hosted, 2 unknown**.
  No Diffusers route remained missing after installing pinned SentencePiece
  for Kolors. Class availability does not verify model weights or inference.

## Actual generation evidence

| Execution | Evidence and scope |
| --- | --- |
| Illustrious named route, 128×128 EPS | Generated through the unified CLI using cached **original SDXL 1.0** weights. This validates routing and the compatible contract, not Illustrious-trained weights. |
| NoobAI v-pred route, ControlNet + HiRes 128→192 | Original SDXL weights plus a synthetic ControlNet. Both stages used v-prediction/zero-SNR and finite latents. This is composition evidence, not NoobAI image-quality validation. |
| FLUX dev/Krea guidance path | Tiny synthetic guidance-bearing transformer, saved and reloaded from safetensors; changing guidance changed generated pixels. Not a full Krea 12B test. |
| Generic FLUX, DDPM and SD3 | Actual CPU inference with tiny models, including two-image output batches and component/output hashes. |
| Video/audio export | Synthetic decoded frames and stereo WAV encoding verified. No full video/audio model inference was performed. |
| Managed ComfyUI, Illustrious label | Existing 6.94 GB original SDXL checkpoint → automatic inspection/graph → MPS inference → one 128×128 PNG. Source/output SHA256, decoded pixels and process termination checked. |
| Managed ComfyUI, NoobAI label | Plain `python3` → managed Python → original SDXL checkpoint with v-prediction + zero-SNR → two 128×128 PNGs. This validates execution and batching, not NoobAI-trained weights or visual quality. |
| Legacy checkpoint conversion | Actual zip and old non-zip Torch checkpoints converted to safetensors with tensor/dtype/key preservation. Malicious reduce fixture was rejected without executing its marker. Torch >=2.10 is required. |
| Local HTTP contract | Fixtures cover queue errors, node/type/input failures, artifact downloads, redirects, loopback boundaries and publication failures. |

Machine-readable evidence and detailed logs remain under `build/`:

- `build/reference/civitai-validation.json`
- `build/reference/civitai-runtime-audit.json`
- `build/reference/model-families/sdxl-fixture-cli-smoke.json`
- `build/reference/model-families/tiny-flux-dev-smoke.json`
- `build/reference/generic-sd3-smoke/generation/generation.json`
- `build/reference/generic-diffusers-smoke/generation.json`
- `build/reference/generic-media-smoke/ddpm-generation/generation.json`
- `build/reference/generic-media-smoke/encoding-validation.json`
- `build/reference/local-image-smoke/verification.json`
- `build/reference/local-image-smoke/illustrious-sdxl-fixture/local-image.json`
- `build/reference/local-image-smoke/noobai-sdxl-fixture/local-image.json`
- `build/workflow-source-review/live-schema-validation.json`
- `build/reference/checkpoint-conversion-smoke/tiny-conversion.json`
- `build/reference/checkpoint-conversion-smoke/malicious-checkpoint.json`
- `build/civitai-ctest.log`, `build/civitai-python-tests-final.log`

The implementation does **not** establish successful generation for all 79
locally available categories, every checkpoint format, custom node, adapter or
hardware combination. The remaining per-model contract requires real weights,
matching configurations/components, an available runtime or workflow and a
successful generated artifact. The runtime audit and saved provenance make
those checks repeatable without presenting catalog coverage as a guarantee.

Known file-level gap: the installed Kolors pipeline loads complete Diffusers
directories but has no `from_single_file` interface for raw Civitai checkpoints.
HunyuanDiT and HiDream-O1 automatic graphs currently require bundled checkpoints.
Image-conditioned, audio, video and 3D tasks require their appropriate inputs
and execution paths; component weights alone cannot serve as an image generator.
