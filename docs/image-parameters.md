# Editable image-generation parameters

`Generation/ImageParameters.hpp` is a header-only C++23 contract. It has no Qt,
filesystem, model-loading, or inference dependency and does not change the native
request ABI. `ImageParameters::defaults()` creates a detached draft;
`imageParameterSpecs()` provides scalar types, ranges, choices, and defaults.
`validateImageParameters(draft)` accepts incomplete editing (including an empty
prompt). Pass `true` to require a nonblank prompt for submission.

The contract covers Dreamscapes Figma `bn8O4AHKr1X9DWnhR1TgEy`, node `59:229`:

- Essentials: prompt, negative prompt, model ID, width, height, output count.
- Composition: signed random-seed sentinel `-1`, tiling, transparent background.
- References: up to 20 image sources, image strength, ordered ControlNet drafts
  with stable IDs, process, image/model/mask sources, weight, IP-Adapter, regional
  mask, and explicit applied state.
- Sampling: sampler, scheduler, steps, CFG, CLIP skip, eta.
- Fine-tuning: ordered LoRAs with independent weights, textual embeddings, VAE,
  prompt weighting, FreeU.
- Enhancement: refiner/switch, denoise strength, Hires, upscaler, face restoration,
  detailer.
- Output: color profile, metadata retention, watermark, safety filter.

UI labels are not wire values: `DPM++ 2M` is `dpmpp_2m`, `Karras` is `karras`,
`4× Ultra` is `4x-ultra`, and model-default VAE is an empty string. Figma examples
are not declarations of installed model availability or inference support.
Defaults use automatic sampler/scheduler, model-default CLIP skip (`0`), disabled optional enhancements, and
`safetyFilter=off` because no safety classifier is connected. No filtering is
implied. Width/height default to 1024, advanced count to 4, steps to 30, CFG to 7.
These do not alter the separate QuickGenerate defaults.

All fields are strongly typed. Nonfinite numbers, unknown fields/selections,
duplicate collection IDs, invalid ranges, and seed-sequence overflow are errors.
Dimensions use an 8-pixel grid in [64,4096]; output count is [1,1000]; seed is
`-1` or uint32 with room for the batch. Text limits are UTF-8 byte counts. Controls
and LoRAs are bounded to 64 entries. Applied controls require their input image,
model and process, plus a mask when regional masking is enabled. These structural
checks do not load, download, or authenticate models.

## Execution boundary

`nativeParameterIssues(draft, desktopWorker, poseAvailable=false)` is a conservative capability check,
not a substitute for structural validation or the engine's architecture checks.
Applied Pose controls require `poseDetector` and `poseModel`; callers explicitly
pass native Pose availability. The legacy worker does not support Pose.
Current native routes accept explicit dimensions up to 2048, steps, seed,
negative prompt, CFG, automatic/Euler/Heun sampling, VAE and LoRAs. The desktop
worker exposes one LoRA; the in-process API accepts multiple. The application
implements color-profile conversion and metadata policy after inference.

The in-process `generateNativeAdvancedImage` entry point additionally connects
seamless X/Y tiling, all catalog samplers and schedulers, CLIP skip and Eta directly to the native C
API. `dpmpp_sde` selects DPM++ 2M SDE; `ddim` selects DDIM trailing. `normal`
selects the discrete schedule. Auto resolves against the loaded model. Hires
uses half-size base diffusion, nearest/bilinear/bicubic/Lanczos pixel upscaling, and a second
diffusion pass with the chosen denoise strength (must be > 0). Final requested
dimensions still use alignment followed by center crop, never output resizing.
The separate `NativeAdvancedControls` structure preserves existing ABI layouts.
The entry point always uses runtime-resident anonymous model sources.
The process worker's narrower capabilities remain unchanged.
The managed `native-pixel-upscalers` patch adds the two pixel-space interpolation
modes without renumbering existing engine enum values. These decode, interpolate,
clamp RGB, encode and refine; they are not latent-space substitutes or learned
4x-Ultra models. `NativePixelUpscalerTests` checks the actual backend numerical
operations; `NativeResultTests` and its mobile variant check all four C API modes.

Learned `4x-ultra` Hires additionally requires `upscalerModel`: canonical local
4x ESRGAN/RRDB weights (safetensors or GGUF in the product). The mode name does
not assert a specific vendor checkpoint. The managed `native-learned-upscaler`
patch eagerly loads its complete source into anonymous memory and prepares CPU
parameter storage before base image generation. Compute follows the selected
native backend; Metal remains available. The prepared runner is retained by the
cached diffusion context, and source buffers belong to the existing explicit-
release runtime residency pool. No model download, disk-backed fallback or
interpolation substitution occurs on loading/inference failure. A wrong scale
or incompatible tensor set fails before base sampling. Source identity joins
cache invalidation and the post-generation source-change check.

Hires keeps the common half-size base and requested final dimensions: ESRGAN
performs genuine 4x inference, then its intermediate result is normalized to the
requested Hires canvas before re-encoding and denoising. It does not multiply the
user's output dimensions by four. `NativeLearnedUpscalerTests` runs a real
one-block RRDB graph with two synthetic weight sets and known outputs, twice
after removing each source file, on CPU and (on Apple) an explicitly verified
Metal backend. This proves weight consumption and warm
in-memory inference, not production checkpoint quality or performance.

Native reference inputs are caller-owned RGB buffers in submission order (up to
20). The first is the img2img initial image; `imageStrength` is its denoising
amount in [0, 1], not an attention weight. Editing architectures additionally
receive the complete ordered reference list. The managed engine capacity query
rejects multiple references on ordinary img2img models rather than silently
ignoring them; capacity depends on the loaded architecture, not its filename.
All dimensions and byte counts are validated before inference. The legacy
process worker still rejects reference inputs.

`promptWeighting=false` switches the conditioner parser to literal text (including
brackets, escapes, explicit weight syntax and BREAK). It applies to positive and
negative conditioning without rewriting the prompt or reloading model weights.
Each native request reapplies its policy, so a cached literal-mode context cannot
leak into legacy weighted generation. The managed `native-text-conditioning`
patch routes all conditioner attention-parser callsites through that context policy.

Explicit textual embeddings register unique lowercase `[a-z][a-z0-9_]{0,127}`
tokens and canonical local weight paths through `NativeAdvancedControls.embeddings`.
They are eagerly loaded/validated against the active conditioner before sampling;
unsupported architectures or invalid vectors fail instead of falling back to
ordinary words. Their identities participate in the native context cache key.
If a token is not already present in the positive or negative prompt, the native
adapter appends it to positive conditioning. Explicit registration overrides a
same-named default embedding and removes that token's automatic negative insertion.
Resident native calls load embedding sources anonymously, and repeat token loads
use cached vectors before reopening a file. Context release follows the existing
runtime policy; the app does not force a disk-backed embedding reload between jobs.
The managed `native-resident-alignment` patch aligns pristine and writable-copy
anonymous buffers to the backend's queried alignment. Plain malloc was insufficient
for small files and could abort the process at CPU buffer registration. Bulk-read
tests cover tiny/odd source sizes and private copies; CLIP embedding loading remains
part of the regression suite.
The legacy process worker still rejects these two controls.
`NativeTextConditioningTests` exercises the real parser and CLIP embedding loader
with tiny generated F32 safetensors: exact vector values, mismatched dimensions,
cached reuse while the fixture source is temporarily moved, and token weights in
enabled/literal modes. It creates only CLIP tensor metadata and does not run a
pretrained text encoder or diffusion model; full-model quality is separate evidence.

`freeU=true` enables the native UNet decoder's FreeU operations, not a prompt
modifier or output filter. It follows the [Diffusers v0.40.0 tensor contract](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/utils/torch_utils.py):
at each block of the first two decoder resolution stages, amplify the first
half of backbone channels and apply the threshold-1 Fourier mask to the skip
features before concatenation. ControlNet residuals are incorporated before
FreeU, and the same policy applies in the Hires pass. It uses the constant
backbone scaling variant, not the original repository's adaptive hidden-mean variant.

The boolean UI uses the [published Diffusers model profiles](https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu):

| Native profile | b1 | b2 | s1 | s2 |
| --- | --- | --- | --- | --- |
| SD 1.5 | 1.2 | 1.4 | 0.9 | 0.2 |
| SD 2 | 1.1 | 1.2 | 0.9 | 0.2 |
| SDXL | 1.1 | 1.2 | 0.6 | 0.4 |

The managed `native-freeu` patch owns the profile table and numerical graph helper
in a separate `model/diffusion/freeu.hpp`; UNet only calls that helper. The skip
filter evaluates the four modified Fourier bins directly, including the real
projection of the reference's asymmetric mask. This is the same linear transform,
not a blur approximation. Backbone operations remain ordinary backend graph ops;
the frequency kernel uses the CPU fallback and parallelizes across channels/batches.
This may add CPU work/device transitions; no zero-overhead or quality guarantee is
claimed. Enabling does not download models, train weights or rebuild the model
context. Every request reapplies the flag, including legacy requests disabling it.
Non-UNet, tiny UNet and animated models reject enabled FreeU before sampling.
The legacy process worker remains unsupported. Numerical tests compare with an
independently written full DFT/inverse-DFT oracle and run real graph computation;
full-model output quality remains a separate verification requirement.

Up to 64 ordered applied Canny/Tile ControlNets are connected for SD 1.5 and SDXL base models.
`NativeAdvancedControls.controls` supplies a canonical local model, owned RGB hint
(at most 2048 per side), process and finite weight in [0,2]. Canny uses the engine's
real CPU edge detector (high/low 0.08, weak 0.8, strong 1, not inverted) on a private
copy; Tile passes RGB unchanged. The hint is resized by the engine for each base/
Hires canvas. The model's architecture/tensor validation runs during context
assembly; unsupported families and a missing ControlNet fail before inference.
Weights join the anonymous resident model loader and the context identity;
changing image/process/weight reuses the context, replacing/removing the model
does not. This does not promise identical output for distinct control models.
The managed `native-controlnet-fail-closed` patch propagates ControlNet compute
failure out of sampling instead of continuing without the requested conditioning.
`native-controlnet-hires-hint` rebuilds the hint at the target Hires canvas size
instead of reusing a base-resolution hint with the enlarged latent.
`NativeControlPreprocessingTests` runs actual Canny on synthetic pixels;
the `native-canny-preprocessing` patch corrects zero-gradient thresholding,
gradient-direction range comparisons and complete eight-neighbor weak-edge
hysteresis. Tests cover a black image, rectangle edges, all direction bins and
a reverse-diagonal weak chain disconnected from an isolated weak edge.
Adapter fixture tests cover forwarding, cache identity, isolation and rejection.
They are not full-model ControlNet output/quality verification.

The managed `native-multi-controlnet` patch prepares every selected model before
sampling. Each has its own model manager and resident parameter storage; a disk
parameter backend is rejected. In each Base UNet evaluation, models execute in
submitted order. Every residual is independently masked and multiplied by that
model's weight, then **summed without averaging**. The aggregate reaches the UNet
with strength 1, so weights are not applied twice. A zero-weight selected model
still executes and validates; it is not silently removed from the request.
Failed model preparation, hint processing, computation, mismatched residual
shapes, non-finite values or cancellation abort the entire request. No successful
subset is published. Each hint is resized to the current latent canvas, including
Hires; Base control residuals are not passed into the Refiner branch.
Seamless convolution axes are applied/reset on every selected model. Registered
inputs activate the public image generation API directly, without also requiring
a duplicate legacy `control_image` argument; the first owned hint supplies that
compatibility image internally.

The additive C API `sd_prepare_control_nets(ctx, paths, count)` atomically prepares
models. Repeating the same ordered paths reuses resident models. Standalone C API
callers replacing a file at the same path must explicitly clear/reprepare; the SDK
already fingerprints every selected model and invalidates the context if any
changes. `sd_set_control_net_inputs` copies independently owned RGB/mask inputs.
Count zero disables inputs without releasing weights; prepare count zero explicitly
releases the models. End-of-pass cleanup releases calculation scratch, not weights.
Changing only hints, masks, processes or weights reuses the SDK's model context.
`NativeMultiControlTests` verifies numerical weighted sums, independent masks,
source ownership, resize, failure, cancellation and shape validation.
`NativeRefinerSamplingTests` exercises the real production sampler with controlled
neural compute to verify aggregate residual injection, no double scaling, later
layer failure, seamless axes/reset, cleanup and changed canvas size. It also runs
the public generation API through batches, Hires and Refiner switches with the
registered inputs alone. Desktop/mobile adapter fixtures
verify all inputs and every model identity. These are not real-checkpoint quality
or throughput measurements.

Regional masks for each connected Canny/Tile ControlNet use the optional
`NativeAdvancedControls::ControlNet::mask` RGB coverage image. An empty image
disables masking. The SDK validates dimensions and byte count before sampling,
copies coverage through `sd_set_control_net_mask` (single) or the per-layer input
setter (multiple), and clears it for every later
unmasked/legacy request even when the model context is cached. Changing mask
pixels does not reload model weights. The old process-worker route still rejects
regional masks.

The managed `native-controlnet-region-mask` patch multiplies every ControlNet
output residual spatially before it reaches the UNet. This applies to conditional
and unconditional evaluations and every base/Hires residual resolution, not to
the input hint or final RGB. White means full influence, black means zero and
gray means proportional influence. It is not an inpainting mask and does not
guarantee that final pixels outside the mask are unchanged: the UNet remains a
spatially coupled model. Refiner and Detailer crop passes without ControlNet do
not acquire this mask.

Coverage is RGB luminance `(77R + 150G + 29B) / (256 * 255)`; the public C API
also accepts one-channel or RGBA pixels (alpha multiplies coverage). Pixel-center
bilinear interpolation stretches normalized mask coordinates to each residual
plane. Per-resolution coverage is cached within the request and repeated across
feature channels and batch. All shapes are validated before residual mutation.
The product decoder already combines alpha into grayscale RGB coverage.
`NativeControlMaskTests` checks numerical interpolation, channel/batch broadcast,
source ownership, alpha, black/white endpoints, cache replacement and malformed
shape atomicity. The actual setter is covered in `NativeRefinerSamplingTests`;
adapter fixtures check forwarding, failure and reset. These are not full-model
regional generation quality tests.

`detailer=true` requires `detailerModel`, a canonical local converted YOLOv8
detection checkpoint. This follows the native [ADetailer contract](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/docs/adetailer.md):
fused convolution/BatchNorm weights, not a raw Ultralytics pickle, segmentation
checkpoint or MediaPipe model. No model is downloaded or automatically converted.
The managed `native-resident-detailer` patch loads anonymous source bytes and CPU
parameter storage eagerly before base generation. The detector computes on the
selected backend and shares explicit cache-release ownership with diffusion.
Source identity participates in cache invalidation and the final consistency check.

After generation/Hires, actual YOLO detection and NMS produce masks, each cropped
region is regenerated at 512x512 with the main prompt, negative prompt, sampler,
steps, CFG and LoRAs, then feathered back into the original canvas. The submitted
`denoiseStrength` controls crop regeneration and must be positive when enabled.
Native detector defaults remain confidence 0.3, NMS 0.45, at most 100 detections,
four-pixel dilation/feathering and 32-pixel crop padding. Crops do not inherit
whole-image references, ControlNet hints or Hires. Output dimensions remain fixed.
Crop-only previews are suppressed; completion publishes the composited canvas.

Zero detections is a valid unchanged image. Detector graph failure/malformed or
nonfinite output is distinct and fails the job. Masked generation failure or
cancellation also fails/cancels instead of publishing the unmodified base image.
Abort/pause callbacks are observed before detection and between detected regions.
Both base and refinement image allocations are owned and released on failures.
`NativeDetailerTests` uses real CPU/Metal YOLO graphs and production mask/composite
code with only the diffusion sampler substituted. Controlled weights verify
detection/no-detection, source removal, repeat execution, feathered composition,
input isolation, failure and cancellation. It is not real-checkpoint quality proof.

Other control preprocessors, alpha output,
face restoration and safety filters are currently reported as unsupported. Draft values
for inactive dependent controls are retained without enabling their parent feature.
Do not discard unsupported active fields and report a successful generation.

### Final-image watermark compositing

`Generation/ImageWatermark.hpp` exports the Qt-free C++23
`compositeImageWatermark` operation. The caller supplies borrowed straight-alpha
RGBA8 source/destination buffers in one color space, independent row strides,
a fully contained target rectangle and opacity in [0,1]. It resizes using
premultiplied-alpha bilinear interpolation and performs source-over compositing.
Transparent source colors cannot bleed into edges; destination alpha and row
padding are preserved correctly. It changes only covered pixels, allocates no
image-sized temporary buffer, and retains no pointers after returning.

Validation precedes all writes: malformed/truncated buffers, overflowing row
layout, invalid geometry, nonfinite opacity and aliased storage fail with an
error and unchanged destination. Zero opacity is a validated no-op. The SDK
does not know Dreamscapes branding, placement defaults, file codecs or profiles.

`watermark` is a publication effect, like color profile and metadata retention;
it is now accepted for either supported product-native route. Raw inference
APIs do not automatically brand their returned RGB. Dreamscapes invokes the
compositor after final inference/refinement/cropping, before atomic history
publication. It supplies its existing common brand PNG, applies the selected
color profile to both images, and uses a bottom-right square at 1/16 of the
short side (1..128 pixels), a 1/64 margin (fitted to small images), and 55%
opacity. These defaults are product policy, independent of the reusable SDK.
It is a visible brand mark, not invisible provenance or a tamper-proof signature.

Watermark is off by default. It does not change output size, previews, metadata
policy or the raw engine output file. Queued snapshots own the switch value;
later draft edits cannot change already queued outputs. Missing assets and
compositing/encoding failures prevent publication instead of silently omitting
the requested mark. Real inference quality is orthogonal to these pixel tests.

Verification on 2026-09-29: the library and new public compositor test rebuilt;
all 23 selected SDK suites passed in 27.20 seconds
(`build/watermark-final-sdk-tests.log`). Exact tests cover opaque/translucent
source-over, premultiplied edge interpolation, transparent/zero-opacity no-ops,
independent strides/padding, aliased/truncated/overflowing buffers, invalid
geometry and atomic failure. The staged dylib has the same Mach-O UUID as the
built library. Product tests additionally decode the real bundled brand PNG,
check final pixels/profiles/metadata and byte-for-byte engine source retention,
and verify both in-process and worker queue publication with later draft edits.

### Independent IP-Adapter attention and regional coverage

The managed `native-multi-ip-attention` follow-up adds a private native UNet
input vector. Each entry names a distinct slot (0–63), projected image tokens,
finite strength in [0,2], and optional spatial coverage. Its K/V parameters use
`<unet-attn2>.ip_adapters.<slot>.to_k.weight` and `to_v.weight`; widths and token
counts may differ across adapters. Legacy single-adapter names remain supported,
but mixing legacy tokens and indexed inputs in one request is rejected.

Each adapter computes its own image softmax from the shared query, applies its
own regional coverage and strength, and adds its contribution to text attention
before the common output projection. There is no token concatenation, averaging,
text masking or second application on self-attention. A single image-token batch
broadcasts explicitly to the latent batch; an incompatible batch is rejected.
SpatialTransformer supplies actual width/height and restores prior state on
success or error, so rectangular and multi-resolution attention do not infer a
square from token count. Per-slot/per-resolution mask graph inputs are reused.

ControlNet residuals and image attention share `ControlRegionMask::coverage`'s
pixel-center bilinear interpolation. The request owns the source mask and cached
float arrays; these borrowed arrays must remain alive and unchanged until graph
execution finishes. No mask callback means full coverage; a callback returning
null, wrong-sized, non-finite or out-of-range coverage is a failed request.
Missing K/V pairs, duplicate/unbound slots and malformed/non-finite CPU tokens
also fail, including at zero strength. UNet catches typed attention-input errors
and returns an empty result through normal graph cleanup, rather than propagating
an exception or silently generating without the requested condition.

`NativeIPAdapterTests` verifies independent K/V/softmax against numerical oracles,
unequal token widths/counts, sparse/reordered slots, zero and nonzero strengths,
rectangular regional masks, shared/per-image batches, and convolutional/linear
SpatialTransformer paths on CPU and Metal. A reduced-width production UNet graph
uses controlled resident weights to verify `DiffusionParams` forwarding through
four spatial resolutions, mask-input reuse, fail-closed cleanup and a successful
retry. Only pretrained-weight residency is substituted in that integration test.
This is not checkpoint-quality or installed-product proof.

This graph-level foundation alone did not enable the product capability.
Independent adapter/vision resource loading, request preparation and SDK/product
forwarding are now connected by the follow-ups below. The installed application
has not been replaced by these source/test changes.

Verification on 2026-09-29 rebuilt the native library and selected targets;
21 SDK suites passed (130.19 seconds), including CPU and MTL0 execution. Exact
patch reversal checks and repeated CMake configuration passed. After staging only
to `build/install` and relinking product consumers, both product suites passed
an unchanged rerun (39.19 seconds). The first product run's existing CPU-progress
watchdog failure is retained in `multi-ip-product-tests.log`; the passing rerun
is `multi-ip-final-product-tests.log`. This change does not claim to resolve that
timing sensitivity. Optional real-model fixture cases remain skipped.

### SDK and product IP-Adapter forwarding

`NativeAdvancedControls::ipAdapters` carries independent `{model, vision, image,
weight, mask}` entries, up to 64. This is separate from `controls`: IP receives
unprocessed reference RGB while a concurrent Canny ControlNet uses its own copy.
Each entry requires canonical local weights, an owned RGB image at most 2048
per side, finite [0,2] strength and optional same-bounds RGB coverage. The SDK
uses the additive resident factory, checks the SD 1.5/SDXL Base family before
loading, and resets copied native inputs on every request, including legacy
requests that reuse the context. Base-only conditioning is already excluded
from Refiner steps in the native sampler.

Before Detailer crop regeneration, the SDK clears indexed IP/ControlNet inputs
and regional coverage, not their loaded resources. Whole-image mask/hint
coordinates cannot be reused on a detected crop. The next Base request sets
its inputs again. A regression first reproduced the leak with IP plus two
ControlNets and Detailer, then verifies crop isolation and warm-request recovery.

Model/vision identities, order and count invalidate IP resource reuse; image,
mask and strength changes do not. A same-Base request with no IP inputs retains
the loaded context and disables conditioning, so re-enabling unchanged resources
does not reload them. A different Base/other resource configuration can rebuild
the context through existing cache rules. Both source identities are rechecked
after inference; replacement fails the request and discards RGB output. Failures
use the existing cache cleanup path; explicit runtime release retains its meaning.

The parameter document adds `ipAdapterModel` and `ipAdapterVision` to each control.
Missing fields in older presets default to empty; active submission reports the
missing resources instead of corrupting or dropping old data. Process `None` or
`IP-Adapter` with the toggle on uses only IP; `Canny`/`Tile` can use both branches.
Unsupported detectors still fail even with IP enabled. The product independently
canonicalizes weight sources and atomically decodes both input lists. Native
resource loading, not a file extension, determines actual checkpoint compatibility.

SDK adapter tests substitute only the upstream C API and verify path/input
forwarding, warm reuse, disable/re-enable and legacy reset, CPU placement,
resource replacement, and failures before sampling. Separate native numerical
and sampler tests exercise production attention and orchestration. Product tests
exercise immutable queued snapshots and real bounded image decoding; GUI tests
exercise the existing LVRS toggle and two-stage file selection. None substitutes
for a real-checkpoint final artifact or installed-app verification.

Forwarding verification on 2026-09-29 rebuilt the SDK library and passed all
22 selected suites in 32.42 seconds (`build/ip-forward-final-sdk-tests.log`).
The staged `build/install` library was relinked into product consumers:
AdvancedParameters and Generation passed 2/2 in 44.01 seconds, with 12 and 61
individual cases passing respectively; two optional real-model cases skipped.
The unchanged product rerun passed after the earlier timing-sensitive watchdog
failure; that failure is retained in `ip-forward-product-tests.log` and is not
claimed fixed. Two focused GUI cases also passed (4 including setup/cleanup),
covering independent inputs, two-stage cancellation/commit and workspace scroll
retention. No global SDK or installed app was replaced.

### Resident IP-Adapter resources and request preparation

The managed `native-ip-adapter-resources` patch adds
`new_sd_ctx_with_ip_adapters(params, models, count)` without changing the existing
`sd_ctx_params_t` ABI. Each of 1–64 descriptors supplies a local adapter and CLIP
vision checkpoint. Only SD 1.5 and SDXL Base are accepted. Projection parameters
and attention K/V weights use independent slot namespaces; an adapter cannot
replace unrelated Base weights. The loader imports converted metadata atomically,
preserves source offsets, rebases file indices and deduplicates repeated paths.
Identical vision paths share one encoder. Incompatible/missing projection,
vision or attention parameters abort context creation, never return a partial
resource set. Legacy and indexed IP model configuration cannot be mixed.

The factory forces eager anonymous-memory model residency and disables file
mapping; disk-backed diffusion/vision parameter placement is rejected. Inputs
and transient runner scratch have separate lifetimes from context-owned model
resources. Disabling inputs does not unload the models. Memory pressure remains
subject to the OS, including its normal swap policy; this does not reserve RAM
or guarantee successful allocation on an undersized machine.

CLIP vision L/14, H/14 and bigG/14 are selected from checked weight dimensions,
including the checkpoint's projection width. The vision encoder explicitly uses
QuickGELU for L and GELU for H/bigG; the existing text-encoder activation behavior
is unchanged. Unsupported shapes fail rather than guessing another architecture.

`sd_set_ip_adapter_inputs(ctx, inputs, count)` copies slot, RGB image, optional
regional mask and finite [0,2] strength between generations. Duplicate/absent
slots, malformed images and masks fail atomically, preserving the prior input
set; callers must check its return value. Count zero clears inputs and prepared
tokens, retaining resources. All requested Classic/Plus positive and unconditional
tokens are prepared before sampling. Failure or cancellation cannot publish a
partial token set. Classic uses zero pooled unconditional embeddings; Plus uses
the zero-normalized-pixel vision branch. Base, batch and Hires sampling reuse
the prepared views; Base image conditioning is not applied to Refiner steps.

`NativeIPResourceTests` loads real small safetensors fixtures, imports multiple
slots, deletes only its own fixture weight files, and twice reads exact values
through the production loader from resident memory. It also checks collision
rollback, repeated-path reuse, malformed public descriptors and actual vision
MLP graph activation selection. `NativeRefinerSamplingTests` covers copied-input
ownership, independent Classic/Plus preparation, cancellation/retry, and
batch/Hires/Refiner orchestration. These tests do not establish real-checkpoint
image quality, throughput, or installed-product functionality. The product
IP-Adapter capability is now accepted by the in-process product route with
explicit model and vision sources, as documented below.

Verification on 2026-09-29 rebuilt the SDK library and 22 selected test targets;
all 22 suites passed in 89.82 seconds (`build/ip-resources-sdk-tests.log`).
Repeated configuration and the exact managed-patch reverse-check passed. After
staging only to `build/install` and rebuilding product consumers, Dreamscapes
AdvancedParameters and Generation passed 2/2 in 54.81 seconds. The additional
headless product contract covers independent per-control toggles, unsupported
effect preset retention and nonmutating field-specific capability errors.
Optional real-model tests are skipped; no global SDK or app reinstall occurred.

### IP-Adapter native execution foundation

The managed `native-ip-adapter-execution` patch makes the existing native engine's
IP-Adapter image preparation fail closed. Requested images require both adapter
and CLIP-Vision resources, RGB dimensions in [1,2048] and finite strength in [0,2].
Encoding/projection failures, non-finite embeddings, incompatible token shapes
and cancellation stop public image generation before sampling; no unconditioned
image is published as a successful IP-Adapter result. Conditional/unconditional
tokens commit together, and all previous request tokens clear before preparation.
An empty image disables conditioning without releasing the loaded model objects.
Runner scratch ends on both success and failure.
Context creation also checks that every UNet cross-attention layer has compatible
image K/V tensors. A projection-only or partially attached checkpoint fails instead
of preparing image tokens that some or all layers would ignore.

Classic uses projected pooled CLIP embeddings and a zero unconditional embedding.
Plus uses penultimate CLIP hidden states; its unconditional branch encodes **zero
normalized pixel values** through CLIP before projection, not zero embeddings or
black RGB pixels normalized afterward. This follows the
[official IP-Adapter implementation](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py)
inspected on 2026-09-29. The existing SD/SDXL decoupled image attention remains the
native graph path; image strength scales that attention contribution before its
shared output projection, not the image tokens or text conditioning.

`NativeIPAdapterTests` runs actual Classic linear/layer-norm, one-layer Plus
Resampler and decoupled attention graphs on CPU and Metal. Controlled weights are
compared with independent numerical formulas, including repeated inputs, zero
embeddings, separate text/image softmax and strengths 0/0.5/1/2. Production-engine
preparation tests substitute only neural compute to inspect Classic/Plus branch
inputs, token atomicity, invalid data, runner cleanup and public failure propagation.
Sampler integration tests combine IP-Adapter, multiple ControlNets, batches, Hires
and Refiner endpoints; each Base UNet evaluation must receive the correct positive
or negative image tokens and strength, while Refiner receives neither.
This is a backend foundation, not real-checkpoint quality/performance proof or
the completed Dreamscapes feature. The graph follow-up described above provides
indexed K/V attention and regional masks; independent model/vision resource
preparation, SDK forwarding and product selection have since been connected by
the resource and forwarding follow-ups. Real-checkpoint output remains separately
unverified.
The 2026-09-29 verification rebuilt the native library and related test targets:
21 SDK suites and both product regression suites passed. SDK staging was limited
to `build/install`; no global dependency or installed application was replaced.

### SDXL Refiner architecture preparation

The managed `native-sdxl-refiner` patch adds architecture-based recognition,
not filename detection, for the official 384-channel, four-stage SDXL Refiner
UNet with 1280-wide cross-attention and a 2560-wide ADM input. Original checkpoint
and Diffusers UNet names map to the same native parameter layout. A single bigG
text encoder replaces the base model's CLIP-L plus bigG pair. Refiner conditioning
uses pooled bigG, original height/width, crop coordinates, and aesthetic score
(positive 6.0, negative 2.5); it does not append the base model's target size.
Negative conditioning is explicit, including an empty negative prompt, rather
than inferred from the base model's empty-prompt zeroing policy.

Architecture and conditioning follow Stability-AI's
[Refiner config](https://github.com/Stability-AI/generative-models/blob/main/configs/inference/sd_xl_refiner.yaml)
and [reference sampling demo](https://github.com/Stability-AI/generative-models/blob/main/scripts/demo/sampling.py).
`NativeRefinerTests` checks tensor-shape detection, name conversion, the actual
UNet/CLIP parameter metadata, and numerical ADM construction. It does not allocate
or infer the full pretrained weights. This is a prerequisite only: `refiner=true`
remains rejected until resident second-context ownership and product model
selection are connected and verified. No RGB decode/re-encode pass is substituted
for a Refiner.

### Native same-latent Refiner sampling

The managed `native-refiner-sampling` patch adds `sd_ctx_can_refine` and
`generate_image_with_refiner` without changing existing request-structure layouts
or the legacy `generate_image` signature. Both contexts are borrowed for the call;
their caller must own the resident model resources for the application runtime.
SDXL Base/SSD-1B/Vega with epsilon or CompVis v-prediction can pair with an epsilon
SDXL Refiner. Other architectures, EDM noise contracts, inpaint/Pix2Pix variants
and aliased contexts fail preflight. Refiner is reported as `sdxl-refiner` while
its VAE family remains `sdxl-base`.

One original sampler invocation retains the same latent, sigma schedule, RNG,
multistep solver history and initial-latent mask anchor. The first
`ceil(step_count * switch_at)` steps use Base, the remainder use Refiner;
float rounding tolerance preserves exact decimal boundaries. Zero selects
Refiner for every step; one never selects it. Predictor and corrector evaluations
use the same model (the backend signals them with negative/positive step numbers).
Each selected model supplies its own text conditions, timestep mapping and
prediction scaling. CFG/eta and the selected sampler remain unchanged.

Base-specific ControlNet, IP-Adapter and generation extensions are not passed to
the differently shaped Refiner. Base LoRAs remain attached to the Base runner.
Cross-step model-output caches are rejected for this API, so cached Base
predictions cannot cross the model boundary. Ordinary model-resource residency
is unaffected. If Hires is enabled, its initial pass remains entirely Base and
the switch applies to the final Hires pass, with final-size Refiner conditions.
There is no intermediate VAE decode, new noise injection or sampler restart at
the switch. A failed/cancelled Refiner pass cannot publish the completed Base as
a successful refinement.

`NativeRefinerSamplingTests` compiles the production orchestration with controlled
network runners and checks actual Euler, Euler-A, Heun, DPM++ 2M, DPM++ 2M SDE and
DDIM solvers against an independent denoiser oracle for epsilon and v-prediction
bases, masked and unmasked latents and both switch endpoints. It also exercises
the public generation API with controlled VAE/conditioner/network compute to check
batch output, final-pass Hires selection, conditioning sizes, decode/encode counts,
Refiner failure and cancellation on the last compute before publication.
These are execution-contract tests, not pretrained-checkpoint quality or
performance proof.

### Refiner request adapter and runtime ownership

`NativeAdvancedControls.refiner`, `refinerSwitch` and `refinerModel` connect
the parameter document to this API. The source must be a canonical local complete
SDXL Refiner checkpoint including its bigG encoder. Its VAE can be embedded or
use the selected/shared SDXL fallback VAE. Architecture, compatible noise contract,
source identity and explicit embedding compatibility are checked before sampling.
The legacy desktop process worker still rejects Refiner requests.

The second model context is owned by the native runtime cache and prepared with
the same anonymous-memory and compute placement policy as Base. Changing only
the switch, temporarily disabling Refiner, or using switch=1 retains the warm
second context without executing it. Replacing weights changes its identity.
Explicit cache release releases both contexts; failure cleanup follows the existing
native context policy. Anonymous source-buffer ownership remains governed by the
runtime resident pool, not an external-disk fallback.

The new C API's final optional `negative_prompt_override` argument keeps
Base-specific automatic negative embeddings out of Refiner conditioning. A null
pointer inherits the request's negative prompt; a non-null empty string means an
explicit empty negative. The SDK supplies the user's negative text. Explicit
embeddings are registered on both models and must contain compatible Refiner
bigG vectors. FreeU and prompt-weighting policy are reapplied on both contexts
for every active request; Base LoRAs and ControlNet remain Base-only.

`NativeResultTests` and its mobile-policy variant exercise the production
adapter with controlled C API fixtures: forwarding, warm reuse, switch endpoints,
per-request conditioning, failure/cancellation, source replacement, incompatible
embeddings and explicit release. They do not prove full-checkpoint inference.

This module does not implement those missing inference algorithms. Callers can
edit/store all fields now and connect additional engines later without changing
the view-facing parameter document. Anonymous model residency and explicit
release remain governed by the existing native runtime.

Validation: `cmake --build build --target ImageParametersTests` followed by
`ctest --test-dir build -R '^ImageParametersTests$' --output-on-failure`.
