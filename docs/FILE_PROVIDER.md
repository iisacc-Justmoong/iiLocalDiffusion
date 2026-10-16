# File creation ownership in iiLocalDiffusion

Native model/cache/telemetry writers and Python image, video, PCM, checkpoint and model-package outputs use provider streams or staging. PNG generation remains PNG; editor IISC persistence is a separate domain operation. Worker scans exclude provider control files and temporary cleanup tolerates moved stages.

The common [ownership contract](../../iiFileProvider/docs/FILE_OWNERSHIP.md) defines stream, staging, control-file and
installed-package rules. Source regression tests remain with this module; the
workspace creation audit also checks these boundaries. Build/deployment and target
runtime validation are separate evidence.

Runtime installation includes the provider's Python worker adapter, shared native
library and license notices. Relocated workers resolve the adjacent SDK library;
macOS app workers also resolve the versioned library in `Contents/Frameworks`.
An installed product does not require a separate SDK in the user's home directory.

Changed boundaries:


- `reference/diffusers/animation_video.py`
- `reference/diffusers/checkpoint_conversion.py`
- `reference/diffusers/comfyui_runtime.py`
- `reference/diffusers/controlnet.py`
- `reference/diffusers/dit_conversion.py`
- `reference/diffusers/generate.py`
- `reference/diffusers/generate_any.py`
- `reference/diffusers/generation_output.py`
- `reference/diffusers/generation_preview.py`
- `reference/diffusers/iild_package.py`
- `reference/diffusers/inference_session.py`
- `reference/diffusers/inspect_pipeline.py`
- `reference/diffusers/local_image.py`
- `reference/diffusers/model_merge.py`
- `reference/diffusers/model_merge_compatibility.py`
- `reference/diffusers/model_merge_unified.py`
- `reference/diffusers/native_image.py`
- `reference/diffusers/remote_generation.py`
- `reference/diffusers/standalone_image.py`
- `reference/diffusers/unified_image.py`
- `reference/diffusers/video_interpolator.py`
- `reference/diffusers/video_runtime.py`
- `reference/diffusers/weight_files.py`
- `src/Generation/NativeDiskCache.hpp`
- `src/Generation/NativeTelemetry.hpp`
- `src/Generation/UnifiedModel.hpp`
- `src/ModelPackaging/ModelPackaging.cpp`

Generation History remains PNG; IISC conversion belongs to editor persistence.
Installer smoke fixtures copy bounded embedding/configuration assets and exclude
backend weights larger than 64 MiB. They never copy multi-gigabyte developer models.
