# Standalone generation verification — 2026-09-08

The previous Dreamscapes request failed with `Local image runtime is missing.
Run reference/setup_comfyui.py once.` Automatic single-file dispatch now calls
`standalone_image`, which uses the SDK's Diffusers/PyTorch process and bundled
SD1/SDXL configuration/tokenizers. ComfyUI is an explicitly selected option.

## Verified results

- SDK native build and CTest: **72/72 passed** in `build/standalone-validation/build`.
- Python runtime suite: **744 run, 742 passed, 2 environment skips**, no failures.
- Dreamscapes rebuilt in `build/`; GUI and generation suites: **2/2 passed**.
- Installed CMake consumer and relocated launcher/resource validation passed.
- Runtime-only installation updated `~/.local/SDK/iiLocalDiffusion`; the three
  existing native library hashes remained unchanged.
- Actual Dreamscapes Generate action with `redLilyIllu_v10.safetensors` completed
  job `4e4f253e-33e2-4ee8-8c0e-89c176883dc8`: **512×512, 20 steps, MPS FP16**.
  The native result screen displayed the generated teapot image. Its PNG SHA-256
  is `cb354b8a781c3de796edf0e4235ab78ed119e001e5cd5fa00daed3745035c161`.
- Installed Deforum backend used the same real SDXL checkpoint and bundled
  configuration to generate a **512×512, 3-frame GIF in MPS FP32 (8 FPS requested; GIF delays use centiseconds)**.
- Installed LTX backend generated and interpolated a **64×64, 9-frame, 24 FPS
  H.264 MP4** from a small locally initialized LTX test model. FFprobe verified
  frame count/rate and the generator verified decoding. This is pipeline and
  media encoding evidence, not a full-sized LTX quality benchmark.

## Random seed defaults

Image/checkpoint, Deforum/Interpolator, LTX, remote image and managed workflow
requests now resolve an omitted seed to a fresh random 32-bit base seed.
Explicit seeds, including zero, remain unchanged. Batch strides, animation
frame/shot policies and HiRes seed inheritance are preserved. Resolved
configuration and generation metadata retain the actual seed for replay.
Generic Diffusers continues using its pipeline's existing stochastic default.

The installed runtime passed actual CPU inference with the existing small local
SD15 and LTX fixtures. Two image requests had different seeds and pixel hashes;
replaying the first recorded seed reproduced its exact pixel hash. Two LTX
requests had different seeds and frame hashes, with verified 9-frame, 24 FPS
H.264 decoding. These fixtures verify seed handling and execution, not model
quality. Evidence is in `build/standalone-validation/random-seed/verification.json`.
The same installed standalone checkpoint route was also resolved against the
real Society `redLilyIllu_v10.safetensors`: two omitted seeds differed, and an
explicit zero stayed zero. Those configuration checks did not sample the large
model again. Runtime installation preserved all three existing native libraries.

The structured report and copied Dreamscapes PNG/provenance are in
`build/standalone-validation/verification.json` and
`build/standalone-validation/dreamscapes-*`. Native build, CTest, Python tests,
installed runtime, image and video logs are kept beside them.

## Boundaries

The host's missing Metal compiler prevents configuring the existing optional
native MLX Metal backend. The separate validation build disables optional
MLX/LibTorch/CoreML; only the Runtime installation component was applied to the
active SDK. Python MPS inference ran on the actual GPU.

An explicit 256×256 FP16 Deforum probe reported non-finite denoising latents and
published no completed video. A 512×512 FP32 probe completed all three frames.
Use `--dtype float32` for that model's validated animation settings. The image
path's default MPS FP16 succeeded at 512×512. This does not establish support
for every checkpoint/precision/dimension combination.

No model weights were downloaded or bundled. iOS/Android native inference and
arbitrary GGUF/custom-node models remain outside the standalone desktop path.
