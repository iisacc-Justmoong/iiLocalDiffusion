# Resident inference session

`iild-generate --worker` keeps one SDK Python process alive for sequential
requests. Python imports and dependency discovery are initialized once on first
use. The SDK also retains one compatible prepared image pipeline in memory.
Applications own their queue and send the next request after completion; the
worker writes no queue, cache database or Society state. No additional library
dependency is introduced: this uses Python's standard library and the existing
Diffusers/PyTorch installation.

## Foreground preparation

The SDK advertises `foreground-residency` in `IILD_READY.capabilities`. A client
must check this capability before sending lifecycle controls; older workers
could otherwise interpret their arguments as a generation request.

```json
{"schema":"iild-worker-request-v1","id":"focus-1","action":"foreground","foreground":true,"arguments":["--backend","local","--model-path","/models/model.safetensors","--device","auto","--cache-dir","/app-temp/session/cache"]}
```

The foreground command resolves and verifies the selected local model, loads
Python dependencies, constructs the supported image pipeline and applies its
execution placement. It does not run denoising, encode a prompt, create an image,
emit a preview or write a generation/request record. Its cache directory must
belong to the app's worker session, outside Society. Normal generation later
uses the same prepared model and device placement even when its output paths,
prompt, seed or image size differ. No second model/weight cache belongs to the app.

Every `IILD_RESULT` includes `action` and `residency`: `foreground`, `ready`,
`state`, and, for a prepared model, `device`, `offload`, `model`, `gpu_resident`.
The initial `IILD_READY` only means the transport has started; wait for the
foreground command's successful result with `residency.ready` before declaring
the model ready. `gpu_resident` is true for a resident CUDA/ROCm or MPS placement;
an explicit CPU choice or CPU offload is reported accurately rather than being
labelled GPU residency. The existing device/offload/memory policy is respected.

Once prepared, the SDK blocks waiting on stdin without generating dummy frames,
polling the GPU or releasing the model when the app queue becomes empty. The app
forwards foreground state and model selection; all preparation and readiness
decisions are owned by `InferenceSession` and the SDK backends. A Python caller
can use `InferenceSession.set_foreground(True, callback)` with the same generator
entry point; `is_preparing()` selects the preparation-only execution path.

```json
{"schema":"iild-worker-request-v1","id":"focus-2","action":"foreground","foreground":false}
```

Background notification does not cancel a running generation or discard the
existing session caches. It does not initiate a new preparation. Foreground with
no arguments means no model is selected and releases the previous prepared model.
Foreground with a different model/configuration uses the normal invalidation
rules. Failed preparation clears readiness and returns an error; it never
silently creates a generation job. Unsupported remote/media/adapter paths are
rejected for foreground preparation before generation can run. Applications
coalesce lifecycle/model updates and send at most one outstanding command.

## Transport

The worker prints `IILD_READY {"schema":"iild-worker-v1","pid":123}` to stdout.
Send UTF-8 JSON, one object per line, on stdin (maximum 1 MiB per request):

```json
{"schema":"iild-worker-request-v1","id":"job-1","arguments":["--backend","local","--model-path","/models/model.safetensors","--prompt","a red cube","--output-dir","/app-temp/job-1/output","--cache-dir","/app-temp/job-1/cache"]}
```

`arguments` uses the existing generation CLI contract without a shell. Each job
must have its own output, work and preview paths. Existing `IILD_PREVIEW` events
belong to the single active request. Logs/progress may also appear on stdout.
Completion is a flushed `IILD_RESULT` JSON object with schema
`iild-worker-result-v1`, the matching `id`, `ok`, `exit_code`, `error`, `pid`,
`elapsed_seconds`, and per-request `cache` counters: `model_hashes`,
`model_hash_hits`, `model_bytes_hashed`, `pipeline_loads`, `pipeline_hits`,
`configuration_reads`, `configuration_hits`, `device_placements`,
`device_placement_hits`. Reads/loads/placements count initialization calls;
hits count requests reusing each corresponding stage.
An ordinary inference error evicts the prepared pipeline and leaves the worker
available for the next request. Recursive `--worker` requests are rejected.

Closing stdin releases the session after its current request. To cancel running
inference, terminate the worker (and its process group when applicable), wait for
exit, then start a new worker for subsequent requests. Cancellation/crashes lose
memory caches. They never restore or resume a job from disk. Send only one
outstanding request so cancellation cannot accidentally run a queued request.

## Cache identity and lifetime

Full model SHA-256 values are cached in process memory, with at most 1,024 current
file identities. Every lookup checks the resolved path, device, inode/file ID,
size, nanosecond write time and metadata-change time. Windows uses
[`FILE_BASIC_INFO.ChangeTime`](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info)
because Python's Windows `st_ctime` describes creation time. Access time is not
part of identity. Changed files are hashed again; hashing itself checks identity
before/after reading and never caches a changing file. Verification of a captured
request still rejects a changed digest, missing file or redirected symlink.
The same cache serves checkpoint/VAE/LoRA/ControlNet identities, Civitai sidecar
hash matching and generic Diffusers directory weights. Output images and external
generation inputs retain their own verification.

This optimization relies on the filesystem reporting file identity and change
metadata correctly; it is not tamper-proof attestation against forged metadata.
Process restart or bounded-cache eviction also requires a fresh hash.

Model composition and execution placement are separate cache stages for ordinary
SD 1.x, SDXL and FLUX image requests, including complete Diffusers directories:

- The constructed pipeline retains its denoiser, VAE, encoders, tokenizers and
  initial scheduler configuration. Construction identity uses actual loader
  arguments and model/configuration/VAE file identities. A new prompt, sampler,
  prediction setting, clip-skip, seed, size or step count does not reload weights.
- Placement retains the device/index, effective offload policy, attention slicing
  and VAE slicing/tiling configuration. An unchanged placement does not call the
  SDK's device-placement initialization again. Changing only placement uses the
  existing model components; obsolete offload hooks and slicing settings are
  removed before applying the new placement. A dtype or weight-variant change
  reconstructs weights to preserve their precision and source identity.
- Model inspection and model-index parsing use a separate bounded memory cache
  of 16 configurations. Every lookup checks source identity, including optional
  sidecar absence, addition, replacement and removal. Results are copied so
  callers cannot mutate the cached configuration. No configuration cache is
  written to disk.

Model/configuration/VAE files are checked on every request, including directory
additions/removals. Changing those files releases the old composition and
placement before reconstruction. A fresh scheduler is constructed from the
model's original cached configuration for each request so neither sampling
history nor the previous request's scheduler overrides can leak into the next
request. Runtime metadata reports `model_configuration_cache_hit`,
`device_placement_cache_hit`, and their conjunction `pipeline_cache_hit`.

CPU offload layouts also retain their models and preparation policy between
requests. Offload still moves weights during inference, according to the selected
Diffusers/Accelerate policy; it does not promise permanent GPU residency. Changing
from offload to resident execution materializes/removes the old hooks using
Diffusers' `remove_all_hooks` before device placement. Placement failure evicts
the partially moved pipeline.
Generation and preview decoding use `torch.no_grad` so weights replaced by
offload/VAE hooks retain tensor version counters. `torch.inference_mode` would
leave some restored parameters unusable during a later placement change or
MPS convolution. Gradient recording remains disabled without copying the model.
See [PyTorch inference-mode constraints](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).

Adapter, ControlNet, textual-inversion, external-tensor, CPU text-encoding,
and multi-stage animation/HiRes requests retain their existing pipeline
initialization paths; they still reuse Python imports and unchanged model hashes.
Other generic media pipelines likewise do not reuse a prepared pipeline yet.
This prevents retaining task-specific mutable conditioning or hooks across jobs.
The ordinary one-shot CLI is unchanged and shares hashes within that process.

The caller must keep the worker's working directory and TMPDIR/TEMP/TMP valid
for the entire session: imported libraries may retain JIT/temp paths. Per-job
files can be removed after `IILD_RESULT`; session runtime temporary files are
removed after worker exit. Keep both outside Society. Use
`PYTHONDONTWRITEBYTECODE=1` to avoid new bytecode files while reading installed
`__pycache__` files. Do not point `PYTHONPYCACHEPREFIX` at a new empty directory:
[Python ignores the existing cache tree when that prefix is set](https://docs.python.org/3/library/sys.html#sys.pycache_prefix).

## Verification

`InferenceCacheTests` covers repeated verification with one full hash, equal-size
edits with restored write time, atomic replacement, symlink retargeting, mutation
during hashing, shared sidecar/ControlNet digests, directory changes, pipeline
reuse/eviction, scheduler reset, protocol errors and EOF. `ModelLoadingTests`,
`StandaloneImageTests` and `GenericDiffusersTests` retain output and mutation
checks. Real-model warm-request timing is a separate integration measurement.
`ModelPreparationCacheTests` verifies independent composition/placement reuse,
sampler changes and restoration, device changes, offload-to-resident transitions,
placement failure eviction, configuration/dtype invalidation, cached model-index
parsing and sidecar changes.
`ForegroundInferenceTests` covers preparation before a queue exists, retained GPU
placement for the next request, background/foreground transitions, removing a
model selection, malformed controls, unsupported preparation and absence of
generated images/request files. Direct Python session calls also clear prior
readiness when a replacement preparation raises or returns a failure code.
The opt-in `ModelPreparationCacheTests` regression generates previews while moving the same
FP32 model through CPU, MPS, model offload, sequential offload and resident
execution, and verifies that the CPU roundtrip preserves output pixels:

```sh
IILD_PLACEMENT_MODEL=/path/to/local/tiny-sd1-diffusers-model \
  reference/diffusers/.venv/bin/python -B tests/ModelPreparationCacheTests.py
```

This test requires MPS and an already available local SD1 Diffusers model; it
never downloads weights. Without that environment variable, the deterministic
cache tests still run and the real-model regression is skipped.

On 2026-09-08, the installed worker generated three 512×512, 3-step SDXL images
from `redLilyIllu_v10.safetensors` (6,938,042,184 bytes) on an M1 Max with PyTorch
MPS FP16. Time to first preview was 31.749 s for the initial request, then
1.326 s and 1.263 s using the same process/model. Model hash counts were 1, 0, 0;
pipeline load counts were 1, 0, 0. Repeating the same prompt/seed produced the
same PNG bytes; changing both produced a different image. EOF and temporary
cleanup passed. These are SDK-request timings, not a quality benchmark or a
guarantee for other hardware, models or request options. Raw measurements are
in the checkout's `build/inference-worker-validation/`.

The subsequent composition/placement extension was verified with six SDXL
requests in one installed worker, changing the scheduler, restoring it, changing
prompt/seed, and changing then reusing VAE slicing. Total model construction and
full checkpoint hashing were each one. Configuration reads were 2 initially and
0 thereafter. Placement initialization occurred once initially and once for the
explicit layout change. Restoring the scheduler reproduced the original PNG.
Final first-preview times were 87.997 s initially and 1.254–1.483 s subsequently;
the preceding run of the same implementation had an 85.334 s first request.
Stage timestamps put 77.993 s before dependency/device preflight completed,
6.982 s in model construction/placement, then 3.022 s to the first preview.
Repeated requests reached completed placement in 11–64 ms. The initial startup
cost remains separate from the verified reuse and is not promised to disappear.

Real CPU/MPS/model-offload/sequential-offload/resident transitions retained one
FP32 SD1 pipeline across nine installed-runtime requests and preserved CPU
roundtrip pixels. A regression with previews also passed for the same transitions.
SDK CTest 75/75, the opt-in model preparation tests 8/8, actual Torch preview
tests 6/6, Dreamscapes CTest 4/4 and its two-request installed-SDK consumer passed.
Evidence and limitations: `build/model-preparation-validation/REPORT.md`.

The foreground extension was then exercised against the installed worker using
the same SDXL checkpoint. Before submitting any image request, preparation took
114.100 s and reported MPS resident execution. An idle interval and two later
foreground notifications retained the same PID and pipeline. Across preparation,
two 512×512, 3-step image requests and background/foreground transitions, the
checkpoint was fully hashed once, the model constructed once and the device
placement initialized once. Preparation emitted no previews, images or request
records. The first generated preview arrived in 2.473 s, then 1.367 s for the
second request; identical prompts/seeds reproduced identical PNG bytes. Initial
preparation is moved before the queue, not eliminated, and a request submitted
before it completes still waits for it. These timings were measured while a
separate tiny-model consumer verification was running on the same machine.

The Dreamscapes consumer independently prepared an actual SD1 Diffusers model on
MPS before creating a job, then reused its worker, composition and placement for
the first generation. A native macOS Qt Quick window verified foreground event
delivery separately. Reproduction logs and assertions are in
`build/foreground-validation/REPORT.md`.
