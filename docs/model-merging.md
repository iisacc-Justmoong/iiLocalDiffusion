# Local model merging

## Preflight and saved-output verification

Inspection returns `base_profile`, per-resource `profile`, and `preflight`.
Profiles identify ecosystems from header evidence and separately enumerate UNet,
DiT, VAE, text-encoder and unclassified tensors (counts, shapes, dtypes and bounded
examples). Unclassified tensors are not silently called a denoiser. Resources have
`compatible`, `conditional` or `incompatible` status: an ecosystem match alone is
not an exact structural match. `lora_targets` lists the base tensor and row range
for every accepted projection; rejected adapters retain the actual target or
configuration failure as their exclusion reason.

Preflight explains retained/excluded resources, effective coefficients, projected
tensor counts and numeric/quality risks. Weighted arithmetic preserves the base
tensor contract, fitting eligible coordinates under `common-layer`. Unified is
an independent sequential image-refinement package, **not** single-network
conversion. Neither fitting nor successful validation certifies generated images.

Before publishing a weighted output, every saved tensor is reopened and compared
exactly with its computed value, shape and dtype. Floating outputs must be finite;
the independently observed changed-tensor count must match the calculation.
`output_verification` reports full coverage, not a sample. Unified output is
reopened and all packaged checkpoint bytes are hashed against the planned members;
LoRA-fused members also receive tensor verification. Copied Unified members remain
byte-preserved and are not numerically repaired. Verification adds full-output
read I/O and never modifies source models.

Best-effort arithmetic preserves empty base tensors and omits contributions with
empty/invalid quantization scales. Identical FP8 LoRA aliases are compared through
FP32, avoiding unsupported storage-dtype comparison operators. An entirely
unusable material set yields a clearly reported preserved or repaired base, not a
claim that material weights were blended. Unreadable base structure, insufficient
storage, process termination and failed output-integrity checks still stop
publication; they cannot honestly be labeled successful merges.

Heterogeneous SD 1.5, SDXL, DiT and editing checkpoints can first be normalized
to one Tsubaki DiT tensor contract with `iild-convert`; see
[DiT-standard model conversion](model-conversion.md). Converted outputs that use
the same target template are ordinary compatible DiT checkpoints and can use the
weight modes below. This is distinct from `unified`, which preserves independent
architectures as an ordered image-space cascade.

## Base-scoped compatibility selection

Every merge now treats the selected base model as the compatibility boundary.
Before resolving coefficients or writing output, the SDK identifies ecosystem
evidence from model metadata and tensor structure. Checkpoints are retained when
they share a base ecosystem (for example FLUX.2 and Krea 2, or SDXL and
Illustrious), or when an unclassified checkpoint has an exact base tensor layout.
LoRAs are retained only when their complete target set resolves against the base
checkpoint under strict target and shape rules.

Incompatible materials are omitted individually instead of aborting a partially
valid request. A request with 20 materials can therefore merge 11 and report the
other nine in `excluded_sources`, including the detected ecosystems and reason.
Explicit weights follow their materials: excluded coefficients are removed and
the weighted-sum base coefficient is recalculated from the retained checkpoints.
In default common-layer arithmetic, unusable additional files are also omitted.
If nothing can contribute, a base-only output is created and explicitly reported
as `result_kind: base-fallback`, `no_effect: true`, not as an effective blend.
Strict arithmetic and unified packaging still require an included material.
Both inspection and completed reports expose `resource_compatibility`,
`included_material_count`, and `excluded_material_count`.

## Single Safetensors output

The default `weighted-sum` and explicit `weighted-difference` modes write a new
Safetensors file when the base is a single-file checkpoint. Inputs are retained
at their original paths, and existing outputs are never replaced. Society also supports the unified `.iildmodel` cascade format and requires a
user-entered output model name.

Equivalent Anima checkpoint backbone namespaces (`model.diffusion_model.*`,
`diffusion_model.*`, `net.*`, their optional nested `net.*`, and unwrapped names)
are aligned in memory. Both inputs must contain the Anima LLM-adapter signature,
with identical, unambiguous complete tensor inventories after prefix alignment.
Shapes, dtypes, architecture metadata and prediction markers must still agree.
Embedded encoder/VAE names remain exact; missing or extra components are errors.
The output retains every base tensor name and dtype, with source hashes and the
key policy recorded in Safetensors merge metadata. This does not convert different
architectures or move, rename, rewrite or delete source files.

Strict-policy regression coverage includes both arithmetic modes with LoRA,
namespace variants, base-name and source-byte preservation, and
missing/extra/duplicate/shape rejection.

### Cross-architecture common layer

Strict weighted merging requires identical tensor inventories, shapes, prediction
markers, runtime assets and architecture metadata. Flux2 and Krea2 do not satisfy
that contract: Krea2 Turbo INT8 uses `blocks.*.attn.wq/wk/wv/wo` tensors plus
`weight_scale`, while Flux checkpoints use a different transformer namespace and
often different dimensions. This mismatch, rather than the arithmetic operation,
was the reason they were rejected before output creation.

After base-scoped ecosystem selection, `--checkpoint-policy common-layer` (the
default) deliberately relaxes the tensor-layout contract *within that ecosystem*.
The base
checkpoint remains the output format and executable tensor inventory. For each
learned base tensor, the SDK uses an exact address, a unique normalized name,
then a same-component/same-role/same-kind source. Known wrapper prefixes
(`module`, `_orig_mod`, `model.diffusion_model`, `diffusion_model`, `model`)
and underscore separators are normalized. Block depths use relative positions
within a role instead of raw block numbers. Package stages, text encoders,
VAEs, denoisers, Q/K/V roles and bias/weight kinds have separate matching pools.

Policy `base-layout-normalize-project-flatten-v3` fits coordinates in this order:
unchanged shape; reversed 2-D shape transposition; equal element-count reshape;
rank-changing flatten/resample; otherwise axis-wise nearest sampling on
normalized coordinates. Scalar targets select the first source coordinate;
scalar sources expand. Shrinking axes precede expanding axes, and flattened
indexes are built in chunks of at most 1,048,576 coordinates. Empty tensors and
missing roles preserve the base rather than padding invented values.

INT8/UINT8 and FP8 with an adjacent `weight_scale` or module `scale_weight` are
dequantized before arithmetic. Optional `weight_zero_point` or
`weight.zero_point` is subtracted first. Scales can be scalar, per-output-row,
broadcastable, per-axis divisible blocks, or divisible contiguous flat blocks.
A 1-D scale matching the leading dimension means an output-row scale, including
square matrices. A quantized base is decoded, merged, then encoded with its own
parameters. Integer storage is rounded and saturated to its range. FP8 indexing
and finite checks use supported FP32 arithmetic; FP64 sources retain precision.
Base scale/zero-point tensors are not averaged.

Scheduler coordinates such as `denoiser.sigmas`, prediction markers, running
statistics, ordinary integer/bool buffers and unmapped tensors retain base
values even when the source also contains them. In a sum, a missing material's
share returns to the base **for that tensor**; in a difference it contributes
zero. Previously, a missing tensor's base substitute incorrectly subtracted
part of the base. Zero-contribution tensors are counted as preserved rather
than merged. Foreign-only tensors are omitted; output names, shapes and storage
dtypes always follow the base.

This policy is deterministic and designed to finish a user-requested experiment
between ecosystem-compatible implementations such as FLUX.2 and Krea 2;
it is not architecture conversion, distillation, or evidence of semantic layer
equivalence. Image quality may be poor or nonsensical. Reports and embedded merge
metadata set `checkpoint_policy: common-layer`, `semantic_equivalence: false`, and
record exact/projected/preserved counts, `transform_counts`, bounded examples,
dequantization counts and runtime-preservation reasons. Invalid quantizers
(empty/nonfinite/nonpositive scales or indivisible blocks) preserve only the
affected base tensor and attach a reason. `numeric_normalization` records base
quantizer failures and requantized tensors. Header inspection cannot guarantee
the runtime validity of scale values, which it does not load. Use
`--checkpoint-policy strict` when any layout mismatch must stop the operation.
For a common-layer checkpoint material, NaN and positive/negative infinity are
treated as missing **coordinates**, not a reason to abort the whole merge.
After dequantization and shape projection, an invalid coordinate uses the base
value in a weighted sum and zero in a weighted difference. Finite coordinates
and all other materials still contribute normally. If the whole material tensor
is invalid and no other contribution applies, the base tensor is copied exactly
and is not counted as merged. Original files are never sanitized in place.

Reports and embedded metadata declare `nonfinite_material_policy` and record
`numeric_normalization.nonfinite_material_values`, `nonfinite_material_tensors`,
and up to 64 `nonfinite_material_examples`. Each example identifies the recipe's
source index, source/target tensor, NaN/+Inf/-Inf counts, replacement rule, and
up to eight coordinates in the **projected base** shape. Counts therefore refer
to coordinates repaired for arithmetic, not necessarily raw source elements.
Inspection remains header-only and does not certify finite tensor values.

### Best-effort numerical recovery

Default common-layer weighted arithmetic declares `repair_policy: best-effort-v1`.
Nonfinite base values, including preserved floating state, become zero before
arithmetic. Nonfinite LoRA factors become zero before contraction; any remaining
nonfinite delta coordinates become zero. A failed LoRA contraction is omitted
with a reason, keeping the other deltas and checkpoints. Additional checkpoint
coordinates still use the sanitized base for sums and zero for differences.

Output NaNs fall back to the sanitized base coordinate. Infinite or out-of-range
results saturate to the finite range of the base storage dtype before casting,
including FP8 and FP16. This is deterministic loss containment, not recovery of
the original learned values. Inputs are never rewritten.

Empty, absent, malformed or incomplete **additional** resources are excluded;
unreadable additional tensors have neutral contributions. All-excluded requests
produce the base layout. If only base numerical repairs changed values, the
report uses `result_kind: repaired-base`; if no values changed it uses
`base-fallback` and `no_effect: true`. `numeric_normalization` records counts such
as `base_nonfinite_values`, `lora_nonfinite_values`,
`lora_delta_nonfinite_values`, `output_nonfinite_values`, `output_clipped_values`,
`skipped_lora_deltas` and `unreadable_material_tensors`, plus at most 64
`repair_events` with addresses, source indexes where applicable, and reasons.

The base must still expose a readable, nonempty tensor inventory: a missing or
truncated base cannot define an output contract. Source mutation, output
collisions, write failures, unavailable dependencies and allocation failures
remain real failures. Strict policy retains numerical rejection and does not
enable these repairs. Unified members copied without arithmetic retain their
original bytes; their contents are not globally sanitized by this policy.

### Predictable problems and regression coverage

| Condition | Common-layer treatment |
| --- | --- |
| Missing key or different export prefix | Normalize names, then same-role mapping; preserve if unavailable |
| Different block counts | Relative block depth with deterministic tie-breaking |
| Transposed matrix / equal element counts | Transpose / reshape before approximate resampling |
| Different rank, kernel, width or scalar layout | Flattened or axis-normalized nearest resampling |
| Empty tensor or prediction marker | Preserve base bytes and shape |
| Missing scheduler or differing state values | Preserve base state in both arithmetic modes |
| Mixed FP8/FP16/BF16/FP32/FP64 | Supported accumulation dtype; retain base storage dtype |
| Scaled INT8/UINT8/FP8 and zero points | Dequantize, merge, re-encode in the base quantizer |
| Invalid quantizer | Preserve that tensor and attach a runtime reason |
| Missing material in a difference | Zero contribution rather than subtracting the base |
| Projection memory spike | Chunk flat indexes; shrink axes before expanding |
| Unrelated components or package stages | No cross-component fallback |
| NaN/Inf checkpoint material coordinates | Sum: use base; difference: use zero; retain finite coordinates and report repairs |
| Nonfinite base or LoRA factors/deltas | Zero-fill bad coordinates; omit failed LoRA contractions with reasons |
| Floating output overflow / output NaN | Saturate to finite dtype range / use sanitized base coordinate |
| Empty, absent, corrupt or incomplete additional resource | Exclude that resource; keep other contributions |
| No effective material remains | Publish explicit base-fallback or repaired-base output, without claiming an effective blend |
| Checkpoint sum coefficients above one | Preserve their ratios and normalize to total one; leave LoRA coefficients unchanged |
| Unreadable base, source mutation, concurrent output, storage/allocation failure | Retain integrity checks and no-clobber publication; fail honestly |

Regression coverage includes a 64-pair scalar/1-D/2-D/3-D/4-D shape matrix,
chunk boundaries, quantized bases/materials, invalid scales, FP64 precision,
missing-key differences, and exact expected numerical outputs.
`tests/verify_model_merge_samples.py BASE MATERIAL` writes full-header planning
and bounded real-tensor evidence into `build/real-merge-*`. It compares up to 64
learned tensors against independent weighted-sum arithmetic and checks the
reported missing scheduler without rewriting original files.

`tests/verify_model_merge_nonfinite.py --base BASE --material MATERIAL` replays
the reported CLIP layer-11 `mlp.fc1.weight` failure against the installed CLI.
Repeat `--material` to cover several real exports. It tests both arithmetic
modes with the original nonfinite tensor and a finite layer-10 probe, checks
that the repaired output is finite and the probe still changes, and writes
reports/sample artifacts into `build/nonfinite-real-*`. Unit coverage also
includes partial NaN/+Inf/-Inf masks, wholly-invalid tensors, multiple materials,
shape projection, FP8 storage, and bounded coordinate diagnostics.
Use `--repair-base` with a real nonfinite base and a finite material to replay
base zero-filling through the same installed CLI and independent output oracle.

Limits: reshaping is not learned neuron alignment. Fused QKV packing and opaque
or packed INT4/architecture-specific quantizers need dedicated decoders; they are
not generally reconstructed. Unknown components may be preserved. Explicit
coefficient normalization is reported separately from coordinate projection.
Whole output shards still
occupy memory during serialization, so chunked projection is not a full
out-of-core serializer. Retaining the base inventory does not prove that an
inference backend can load the output or that generated images are useful.

## Unified model objects

`--mode unified --output NAME.iildmodel` builds one portable ZIP64 stored package
file containing `model_index.json`, source provenance, and independent checkpoint
members. Compression is deliberately disabled so large tensor members can be
streamed and addressed without an additional decompression representation.
It is an **ordered image-refinement cascade**, not a single-network weight
average or distillation. Base-scoped selection still applies: only checkpoints
in the base ecosystem and LoRAs targeting the base enter the new package.
Actual inference still requires each member to be supported by the installed
native backend and to contain its required text encoders and compatible VAE.
Unknown families may be packaged; this is not proof of runtime support.
Architecture identification is advisory: unrecognized hybrid signatures and
marker storage conventions remain intact in independently copied members.

The first checkpoint generates an image. Each subsequent checkpoint encodes
that RGB image with its own VAE and refines it. Models run one at a time using
the existing native context cache. Checkpoint stages retain independent latent
spaces and tensor layouts; optional LoRA adaptation is described below. Order matters, execution costs add up, and visual quality must be
evaluated on actual generations. Later stages retain the native half-size/Hires
policy. Checkpoint weights in this explicit mode are refinement strengths in
`[0,1]`, default `0.35`; zero skips that stage. LoRA strengths default to `1`.
This mode has no weighted-subtraction interpretation.

Under the strict policy, LoRAs are matched by **all** target names, shapes, rank and alpha contracts.
Each adapter is fused into the nearest preceding compatible checkpoint, or the
only later compatible checkpoint. Put an adapter immediately after its intended
model. Equivalent Anima namespaces include `diffusion_model.*`, `net.*`,
`model.diffusion_model.*`, and the complete export's `model.diffusion_model.net.*`.
Under the default strict policy, ambiguous aliases and incompatible shapes remain errors.

### Legacy LoRA compatibility objects

The bridge API remains available for direct inspection, but the merge entry
point no longer discovers or inserts a foreign checkpoint to rescue an unmatched
LoRA. Base-scoped selection excludes that whole adapter first. The details below
describe the retained legacy API, not current merge routing.

When none of the requested checkpoints accepts a LoRA, unified mode now looks for
a **real compatibility checkpoint** among direct sibling safetensors files in the
requested checkpoints' folders. It checks every adapter target, shape, rank and
alpha. It does not recurse into packages, scan the machine, download models or
use filenames as evidence of compatibility. Invalid unrelated files are ignored.

`LoraCompatibility.inspect(checkpoint, adapter)` exposes a reusable structural
result (`compatible`, `architecture`, `target_count`, `embedded_components`,
`reason`). `LoraCompatibilityBridge.discover(checkpoints)` lists local candidates;
`LoraCompatibilityBridge.resolve(adapter, candidates)` selects one and returns an
immutable bridge with `checkpoint`, `refinement_strength`, and `as_dict()`.
These methods live in `model_merge_compatibility.py`; they do not load whole
checkpoint tensors or compute checkpoint hashes. Legacy input inspection still
uses the existing safe conversion when necessary.

For Anima, a matching checkpoint with embedded text encoder and VAE takes
precedence over matching denoiser-only exports. Component presence is structural
evidence, not an inference certificate. If several equally suitable checkpoints
remain, selection fails with their paths; add the desired checkpoint as an
ordinary material, or pass `--compatibility-model PATH`. Repeat that option to
provide a candidate pool in other folders. `--no-auto-compatibility` disables
fallback. The corresponding Python argument is `compatibility_models`: `None`
searches siblings, a sequence restricts candidates, and `[]` disables fallback.

The selected checkpoint is fused with the LoRA and inserted at that adapter's
position in the cascade. Further compatible adapters can reuse the same stage.
The existing RGB handoff re-encodes the previous image with that checkpoint's own
VAE; each network retains its own latent dimensions and prediction settings.
Bridge refinement defaults to `0.35`; `--compatibility-strength` /
`compatibility_strength` sets it in `[0,1]` independently of LoRA delta strength.
Requested material indexes and weights remain unchanged; appended bridge sources
receive separate provenance indexes and hashes. Inspection reports include
`resolved_sources`; stages include `compatibility_bridge`, and completed reports
include `compatibility_bridge_count`. The total remains limited to 64 stages.

For example, SDXL + Noob + an Anima LoRA can automatically add a local compatible
complete Anima checkpoint. The resulting object contains three independent
networks. It is not a conversion of Anima weights into SDXL. A LoRA cannot recreate
a missing base network: without a matching checkpoint the operation still fails,
and weight-sum/difference modes remain strict. No arbitrary tensor resizing,
partial adapter dropping, cross-family LoRA projection or retraining is claimed.

```python
from model_merge import merge_models
from model_merge_compatibility import LoraCompatibility, LoraCompatibilityBridge

compatibility = LoraCompatibility.inspect("/models/anima-complete.safetensors", "/models/style.safetensors")
bridge = LoraCompatibilityBridge.resolve("/models/style.safetensors", ["/models/anima-complete.safetensors"])
report = merge_models("/models/sdxl.safetensors", "/models/style.safetensors",
                      mode="unified", compatibility_models=[bridge.checkpoint],
                      compatibility_strength=0.35, output="/models/compatible.iildmodel")
```

```bash
iild-merge --base-model /models/sdxl.safetensors \
  --additional-model /models/noob-vpred.safetensors \
  --additional-model /models/anima.safetensors \
  --additional-model /models/anima-lora.safetensors \
  --mode unified --weights 0.35 0.35 1 --output /models/combined.iildmodel
iild-generate --model-path /models/combined.iildmodel --prompt 'a mountain lake' \
  --output-dir /images/combined
```

The native C++ image entry points accept the `.iildmodel` package file as
`modelPath`; the CLI auto-selects the `unified` backend. The loader rejects
compressed, encrypted, duplicate, redirected and oversized entries, verifies the
manifest contract, CRC, member sizes and published SHA-256 identities, and safely
materializes a source-identity-scoped local cache. Packages contain independent copies
and remain usable after the original paths move.
The current cascade accepts single-file checkpoints and standard supported
LoRAs; Diffusers checkpoint directories must first be exported. It does not
flatten a multi-stage unified object into one checkpoint; a valid single-stage
`.iildmodel` without pending LoRAs is accepted whole as a weighted-merge base or
material. Legacy directory-shaped `.iildmodel` objects remain readable for
generation, but new unified merges always publish a single file. Safetensors/legacy conversion and
adapter restrictions below still apply. Cancellation, errors or a changed member
discard intermediate images. The native loader confines member paths, validates
sizes and verifies file identities throughout the run; the CLI also checks the
recorded SHA-256 hashes with the SDK's unchanged-file hash cache.

Members without adapters are preserved byte for byte, including any nonfinite
values already present in their source. Packaging them does not certify their
numerical health or replace values in layers that a runtime may not use. The
report states this validation scope explicitly. Checkpoints with adapters pass
through the normal finite-input/finite-output arithmetic checks before publication.

`--inspect` performs structural checks and reports resolved strengths, architecture
evidence and adapter routing before whole-checkpoint hashing or output creation.
Legacy pickle inputs still require the existing safe conversion. Normal merges
also preflight before hashing. `--print-config` retains its argument-only contract.
Empty `v_pred`/`ztsnr` tensors are prediction markers, not trainable weights:
matching markers are preserved, while differing markers require unified mode.
Simply dropping them would change the interpretation of the resulting model.

Validation: `ModelMergeCompatibilityTests.py` checks nested export names, public
compatibility objects, automatic/explicit/disabled selection, ambiguity, shape
rejection, source preservation, ordering, shared bridges and real delta arithmetic.
`UnifiedModelMergeTests.py` checks real tensors, marker preservation,
early failures, adapter routing, finite arithmetic, byte preservation and publication.
`NativeResultTests` and `NativeMobileResultTests` execute the production cascade
adapter against controlled engine results, checking packaged-file loading, RGB
handoff, zero-strength skips, path confinement and cancellation.
`UnifiedImageTests.py` additionally verifies stored-archive materialization,
SHA-256 validation, automatic routing, and rejection of compression and traversal.
These fixtures do not certify image
quality for every named model family.

`iild-merge` and Python `merge_models()` combine local checkpoints and LoRAs.
The **base model and first additional model are required**. Weights, further
additional models, mode, output and conversion cache are optional. The base is a
full checkpoint; the first and subsequent materials may each be a checkpoint or
LoRA. Merging uses
the existing Python PyTorch/safetensors runtime, alongside `iild-generate`;
it is not a C++ tensor operation or an inference-worker request.

## Arithmetic and defaults

Let `A` be the base model and `B_i` the additional full checkpoints. Arithmetic
applies to every matching floating-point tensor, with no sequential reblending.

| Mode | Formula | Omitted weights |
| --- | --- | --- |
| `weighted-sum` (default) | `(1 - sum(w_i)) A + sum(w_i B_i)` | Every model gets `1 / (N + 1)` for `N` additional models |
| `weighted-difference` | `A - sum(w_i B_i)` | Subtract `0.5` of each additional model; base coefficient stays `1` |

For two models the defaults are `(A + B) / 2` and `A - 0.5 B`.
Three models with sum weights `[0.2, 0.3]` produce `0.5 A + 0.2 B + 0.3 C`.
The same difference weights produce `A - 0.2 B - 0.3 C`. Difference means direct
weighted subtraction; there is no implicit reference model or `A + w(B - C)`.

LoRAs contribute changes to those checkpoint results. Let `D_j` be each adapter's
delta, including its own alpha/rank scaling, and `s_j` its requested strength:

| Mode with LoRAs | Formula |
| --- | --- |
| `weighted-sum` | `(1 - sum(w_i)) A + sum(w_i B_i) + sum(s_j D_j)` |
| `weighted-difference` | `A - sum(w_i B_i) - sum(s_j D_j)` |

LoRAs do not consume the base coefficient. Each omitted LoRA strength is **1**
in either mode; checkpoint defaults count only the base and full checkpoints.
Thus base + one LoRA defaults to `A + D` or `A - D`, and base + checkpoint + LoRA
defaults to `0.5 A + 0.5 B + D` or `A - 0.5 B - D`. Multiple adapters may have
different ranks. Their deltas are accumulated on the same checkpoint blend.

Supply one weight to broadcast to all additional materials, or exactly one per
additional model in the same order. Values must be finite nonnegative numbers.
In sum mode, the remainder of full-checkpoint weights belongs to the base.
Common-layer totals above `1` are proportionally normalized to `1`, with base
weight zero; `weight_normalization` records requested and effective coefficients.
This calculation remains stable even if adding the requested weights overflows.
Strict sum mode requires the checkpoint total to be at most `1`.
LoRA strengths and difference weights may exceed `1` and are not normalized.
Boolean, negative, NaN/infinite and mismatched weight lists fail before tensor
loading; the checkpoint total is checked after material classification.
A zero weight still inspects the input structure; common-layer checkpoint
arithmetic skips reading zero-weight material values.

## CLI

After the normal [installation](installation.md), the commands are:

```bash
iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --output /models/merged.safetensors

iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --additional-model /models/detail.safetensors \
  --weights 0.2 0.3 --output /models/blended.safetensors

iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --mode weighted-difference --weight 0.25 \
  --output /models/subtracted.safetensors

# The required additional model can itself be a LoRA.
iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style-lora.safetensors \
  --weight 0.8 --output /models/fused.safetensors

# Checkpoints and LoRAs may be interleaved in the ordered material list.
iild-merge --base-model /models/base-diffusers \
  --additional-model /models/style-lora.safetensors \
  --additional-model /models/other-diffusers \
  --additional-model /models/peft-adapter \
  --weights 0.8 0.25 0.3 --mode weighted-difference \
  --output /models/subtracted-diffusers
```

From a checkout use `reference/diffusers/.venv/bin/python reference/merge.py`
instead of `iild-merge`. The installed command shares `IILD_PYTHON_EXECUTABLE`
and the linked Diffusers environment with the generation launcher.

Repeat `--additional-model` for every extra model. `--weights` and `--weight`
are aliases. `--print-config` checks paths and requested weights without Torch,
tensor reads, hashing, output creation or downloads. It reports omitted weights
and `base_weight` as `null`, with `coefficient_resolution: after-input-inspection`:
effective defaults depend on whether each material contains checkpoint or LoRA
keys. The completed merge report contains the resolved coefficients and kinds.
The configuration preview does not establish model compatibility. Omitted output defaults to
`build/reference/merged/<base-name>-<mode>.safetensors`, or an extensionless
directory name for a Diffusers package. Existing destinations are rejected;
choose a different `--output` for another result.

## Python API

Add the checkout's `reference/diffusers` directory, or the installed
`share/iiLocalDiffusion/reference/diffusers` directory, to Python's module path.

```python
from model_merge import merge_models

# Only these two model arguments are required.
report = merge_models("/models/base.safetensors", "/models/style.safetensors")

# A LoRA is accepted in the same required position; its default strength is 1.
report = merge_models("/models/base.safetensors", "/models/style-lora.safetensors")

report = merge_models(
    "/models/base.safetensors",
    "/models/style.safetensors",
    additional_models=["/models/detail.safetensors"],
    weights=[0.2, 0.3],
    mode="weighted-difference",
    output="/models/subtracted.safetensors",
)

# Map a compatible ecosystem variant onto the base layout.
report = merge_models(
    "/models/flux.safetensors",
    "/models/krea-int8.safetensors",
    checkpoint_policy="common-layer",
    output="/models/flux-krea-experimental.safetensors",
)
```

`resolve_merge_request()` in `model_merge_options` exposes the same lightweight
validation and immutable resolved request. `merge_models()` returns a JSON-ready
report containing ordered source SHA-256 identities, material kinds, coefficients, runtime
versions, output hashes and tensor/buffer counts. The CLI prints this report.

## Formats and compatibility

For the two weight-arithmetic modes, full checkpoints must all use the base's storage format: single files or
Diffusers directories. LoRA files/directories can be mixed with either format:

- `.safetensors` and `.safetensor` files are read with the existing safe loader.
  Tensor names and shapes must match exactly across inputs.
- `.ckpt`, `.pt`, `.pth` and `.bin` single checkpoints reuse the existing
  [weights-only conversion](checkpoint-formats.md). Its Torch version gate and
  refusal to retry unsafe pickle loading remain in force. Converted files use
  `build/reference/model-merge-cache/`, overridable with `--cache-dir`.
- Diffusers directories require `model_index.json` and safetensors weights.
  Tensor names are matched within their component directory, so inputs may use
  different shard layouts. Shard indexes are validated against actual keys and
  filenames. The output retains the base's filenames, shards, configuration,
  tokenizer assets and other visible auxiliary files. Files may be symlinks;
  nested directory symlinks and hidden cache/version-control directories are
  excluded from the supported package layout.
- Directory runtime configurations/tokenizer assets must match. JSON comparison
  ignores only `_name_or_path`, `_diffusers_version`, `transformers_version`,
  `_commit_hash`, `_use_default_values` and `torch_dtype`. Shard index layout and
  the previous `merge.json` do not need to match. Different architectures,
  prediction settings or tokenizers require compatible inputs before merging.
  Packages with pickle, GGUF, ONNX or other weight formats must first be prepared
  as a single safetensors variant. Duplicate tensor keys/variants are rejected.
- A legacy directory-shaped `.iildmodel` remains a package during weighted
  arithmetic and therefore requires an `.iildmodel` output path. Every member
  is rewritten with the requested arithmetic, stage SHA-256 and byte sizes are
  refreshed in `model_index.json`, and the package is published atomically as a
  directory. Equal or nearly equal disk size is expected because base tensor
  names, shapes and dtypes are preserved; `changed_tensor_count` and output
  hashes, rather than file size, prove whether numeric values changed.
- Float8 checkpoint tensors are promoted to the normal Float32 accumulation
  dtype for finite-value checks and weighted arithmetic, then converted back to
  the base checkpoint's Float8 storage dtype. This permits mixed FP8/INT8
  common-layer projection without invoking PyTorch operations that are not
  implemented directly for Float8 tensors.

Under `--checkpoint-policy strict`, integer/bool buffers must have identical
dtype, shape and values and are copied from the base. Floating inputs support
FP8, FP16, BF16, FP32 and FP64, including mixed precision; output keeps the base
dtype. FP8/FP16/BF16 accumulate in FP32; FP64 inputs retain FP64. Strict rejects
nonfinite materials and does not dequantize scaled integer storage. The default
common-layer policy instead uses the reported mapping, quantization and
nonfinite-coordinate fallback described above. Both policies retain source
integrity checks and reject complex weights. Strict rejects floating output
overflow; common-layer saturates it and records the repair.

Compatibility checks establish structural/configuration consistency, not visual
quality or that independently trained checkpoints share a useful weight space.
Explicit `modelspec.architecture`, `modelspec.implementation` and
`modelspec.prediction_type` header hints must agree within matching components
when present under the strict policy.
Single files do not carry complete tokenizer/scheduler configuration; callers
must supply checkpoints from a compatible family. The result can be passed to
the existing `iild-generate --model-path` workflow with its usual family/config
arguments. Model licenses continue to apply to the inputs and resulting weights.

## LoRA materials

LoRA inputs are identified by their tensor keys, not the filename. Supported
inputs include safe tensor files, legacy tensor checkpoints through the same
safe converter, PEFT adapter directories (`adapter_config.json` plus safetensors),
and directories containing a Diffusers/Kohya safetensors adapter export. Each
material must contain one adapter, with complete down/up pairs and no unrelated
trained tensors. For an explicitly selected `adapter_model.safetensors` file,
its sibling `adapter_config.json` is also read, hashed and revalidated.

Supported parameterizations are standard linear LoRA and ungrouped Conv1d/2d/3d
LoRA with a spatial down projection and a 1x1 up projection. For linear targets,
`D = (alpha / rank) * (B @ A)`. Per-layer `.alpha` tensors take precedence over
PEFT `lora_alpha`/`alpha_pattern`; absent alpha defaults to the actual rank.
PEFT `r`/`rank_pattern` must agree with actual tensor ranks. `use_rslora` uses
`alpha / sqrt(rank)` and `fan_in_fan_out` is supported for linear targets.
Tensor-only PEFT exports must retain their config when alpha differs from rank.

Recognized names include PEFT `lora_A`/`lora_B` (with an optional single adapter
name and `base_model.model.` prefix), Diffusers `lora.down`/`lora.up`, legacy
attention-processor/`lora_linear_layer` pairs, and Kohya `lora_down`/`lora_up` with
`lora_unet_`, `lora_te_`, `lora_te1_` or `lora_te2_` module names. Canonical
Diffusers component/module names can also target transformer weights.

Targets are resolved against actual base keys; unqualified names must be unique
across components. SD 1.x/2.x/XL original UNet names use the pinned Diffusers LDM
key mapping (the standard two-residual-layers-per-block layout). SD 1.x/XL's first
CLIP and SD 2.x/XL OpenCLIP names are supported, including packed Q/K/V row updates
and transposed OpenCLIP text projections. The base layout is retained in output.
Ambiguous flattened names, unknown targets, incompatible shapes, incomplete
pairs and unsupported tensors exclude that whole adapter during compatibility
selection (or fail if strict selection has no usable material). Numeric LoRA
NaN/Inf values use the common-layer recovery above; strict arithmetic rejects them.
SGM/Kohya block names can also target Diffusers UNet packages, using their
`unet/config.json` `layers_per_block` and the pinned adapter name converters.

DoRA, LyCORIS/LoHa/LoKr, embedding LoRA, grouped convolutions, bias/module-save
updates and adapters with modified base initialization are not supported by this
offline merger. Architecture-specific packed transformer adapter formats need a
standard Diffusers/PEFT export whose targets match the base. This does not change
the separate [generation-time LoRA loader](lora.md) and its backend capabilities.

The saved result is a full checkpoint with deltas fused into existing tensors.
It needs no LoRA input at generation time. Provenance records each adapter's
source hash, resolved target and alpha/rank scale, plus the requested strength.

## Storage, memory and verification

Merging is a CPU weight-preparation operation and does not load an inference
pipeline or reserve a GPU. Input safetensors are opened lazily; one output shard
is retained while additional tensors are read individually. Memory includes the
output shard, working tensors, mapped input pages and serializer buffers. A
single-file model is one shard, so enough RAM for that output is still required.

Original files are never written. Cached model hashes are reused only while the
existing file identity checks remain unchanged. Every source is revalidated
before publication; changed files or a changed package inventory abort the merge.
Temporary output is created beside the destination and removed on failure. A
single-file result uses atomic, no-clobber hard-link publication; a complete
package uses a directory rename. File output therefore needs filesystem hard-link
support. Existing outputs and destinations inside input packages are rejected.

Each safetensors header stores `iild_merge` provenance and nests original header
metadata under `iild_merge_base_metadata`. It does not reuse an old model hash as
the merged file's identity. Diffusers packages also include `merge.json` with
output hashes. No image-quality claim is embedded in a successful merge report.

Run the small real tensor tests with the existing runtime, then the native suite:

```bash
reference/diffusers/.venv/bin/python tests/ModelMergeTests.py
reference/diffusers/.venv/bin/python tests/ModelMergeLoraTests.py
reference/diffusers/.venv/bin/python tests/ModelMergeDiffusersSmoke.py
reference/diffusers/.venv/bin/python tests/ModelMergeLoraSmoke.py
cmake -S . -B build
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

`ModelMergeTests` always checks arguments and dependency-free help/configuration.
Its real tensor cases skip when the selected interpreter lacks Torch/safetensors;
the explicit first command runs them in the existing generation environment.
`ModelMergeLoraTests` checks adapter arithmetic, mixed material defaults, mapping,
rank/alpha, convolution, precision, failures and source preservation with real
small tensors. It also skips runtime cases if dependencies are unavailable.
`InstallConsumerTests` checks the relocated `iild-merge` command and configuration.
`ModelMergeDiffusersSmoke.py` initializes two tiny DDPM models locally, verifies
both arithmetic modes against every reloaded parameter, then runs each merged
model through the SDK generator. It writes images and `verification.json` below
`build/reference/model-merge-smoke/`. Pass `--merge-entry` and `--generate-entry`
to verify relocated installed commands as well. It makes no network requests and
does not establish full-size model quality.

`ModelMergeLoraSmoke.py` creates two nonzero PEFT adapters of different ranks with
both attention and convolution targets, and combines them with two tiny DDPM
checkpoints. Both sum and subtraction are compared to PEFT's unfused forward
pass and official `merge_and_unload()` result for every model parameter, then
run through SDK image generation. Reports and images are retained below
`build/reference/model-merge-lora-smoke/`. It accepts the same installed entry-point
overrides and runs entirely offline.

### Foreground preparation of a packaged checkpoint

A single-stage unified package can embed a complete checkpoint (including its
original denoiser, text encoder and VAE) as one `model.safetensors` member. The
manifest records its relative path, byte size and SHA-256; the package is one
catalog object. The loader identifies `model_index.json`, not the directory suffix.

Unified foreground preparation retains its NativeEngine in InferenceSession and
records successful preparation, as the single-file backend does. The cache tracks
the manifest and every member; changed members invalidate readiness. Repeated
preparation reuses the engine and creates no output images. UnifiedImageTests covers
readiness, reuse and same-size member replacement alongside package validation.

## Installed Python and legacy compatibility registry

Installed launchers validate Python >= 3.10 before importing model modules. `IILD_PYTHON_EXECUTABLE` overrides `reference/runtime-python.json`, whose shape is `{"python":"/absolute/venv/bin/python"}`. Otherwise the bundled `.venv` or launcher interpreter is considered. Invalid or old environments produce actionable diagnostics instead of an annotation traceback.

Unified bridge discovery also reads installation-local `reference/merge-compatibility.json`: `{"checkpoints":["/absolute/full-checkpoint.safetensors"]}`. Relative paths resolve from that registry directory. The registry augments nearby checkpoint discovery, including the category containing single-checkpoint wrappers. Explicit compatibility candidates override discovery. All targets/shapes are validated; missing, ambiguous or incompatible bases still fail. Inspection never downloads models or executes weight arithmetic. Different networks compose as sequential image refinement; normalization does not make arbitrary network tensors interchangeable.

PythonLauncherTests verifies environment selection, explicit override, old-version rejection before entry import, and broken configuration. ModelMergeCompatibilityTests covers registered bridge discovery from wrapped inputs and output-free input inspection.

### LoRA alias normalization

LoRA projection pairs are normalized to the checkpoint's actual tensor address after shape, rank, alpha and orientation checks. Some exports include both SGM and Diffusers names for the same address. Identical projection tensors with identical dtype, scale and orientation are applied once. Distinct projection pairs are retained as additive deltas on that address, including each pair's own alpha/rank scale. No target is silently dropped or arbitrarily reshaped. Reports expose `lora_alias_policy` as `identical-projections-once; distinct-projections-additive`. Input inspection may read the duplicate LoRA projections to establish equality, but does not materialize full checkpoint tensors or produce a merged model.

The LoRA regression suite verifies duplicate aliases apply once, distinct projections sum correctly, and different alpha values preserve their individual scaling, using tiny fixture tensors.

## Legacy synthetic cross-family LoRA adaptation

The option is accepted for command/API compatibility, but merge preflight now
requires strict base targets and excludes an unmatched LoRA before this legacy
adapter can run. It cannot override base-scoped ecosystem selection. The
algorithm below remains documented for direct legacy-module consumers.

`--lora-policy synthetic` (Python `lora_policy="synthetic"`) enables a deliberately
untrained adapter conversion. Default `strict` retains existing compatibility checks.
Exact compatible targets retain their normal calculation. An unmatched module is
mapped deterministically to a floating matrix/convolution weight by component
(denoiser/text/VAE), projection role, block/layer index, flattened shape distance,
and finally lexical tensor address. If no same-component/role target exists, the
next closest candidate is used and the remapping is reported. No new network
layers are invented and every base tensor name, shape and dtype remains unchanged.

After ordinary rank/alpha scaling, the source delta is flattened to output rows
and input columns. Its top-left intersection is retained, excess rows/columns
are cropped, and missing values are filled with **zero** before reshaping to the
target tensor. This is deterministic, requires no downloads or training, and
allocates only the current source/target delta, not a complete second model.
Multiple distinct deltas mapped to one target are added with their requested
strengths. This can magnify an update; begin with small strengths when evaluating
visual quality. Malformed/unpaired factors, unsupported adapter variants and
nonfinite arithmetic are still errors.

In unified mode, exact compatible supplied checkpoints take precedence. An
otherwise unmatched adapter is synthetically fused into the nearest preceding
supplied checkpoint instead of inserting a compatibility checkpoint. The policy
is forwarded to the member merge. Independent checkpoint architectures still use
the existing RGB refinement cascade; this option does not convert whole networks.

Inspection and output provenance record policy `role-depth-zero-pad-crop-v1`,
source module, target tensor, original/target shapes, retained/zero-filled/cropped
value counts and `semantic_equivalence: false`. Inspection is structural only;
execution validates source hashes and finite values. Artificial zeros supply
missing dimensions, not learned features, so successful arithmetic does not
certify preservation of the original LoRA style or generated image quality.

```sh
iild-merge --base-model anima.safetensors --additional-model sdxl-style.safetensors \
  --lora-policy synthetic --weights 0.1 --output adapted.safetensors --inspect
# Remove --inspect to write a new output; existing files are never overwritten.
```

`ModelMergeSyntheticTests` verifies padding, cropping, subtraction, alpha/strength
arithmetic, exact-name dimension mismatch, source preservation, deterministic
inspection, malformed/nonfinite rejection, and actual unified member fusion.

### Package-local VAE components

An ordered cascade stage may specify `"vae": "vae/anima/model.safetensors"`.
This optional relative path explicitly overrides the member checkpoint's embedded
VAE; old packages retain embedded/fallback selection. Shared decoders are stored
once and referenced by multiple stages. The package reader only checks containment,
nonempty files and stat identity. Tensor compatibility is checked by the native
engine when preparing/generating; failures identify the affected cascade stage.
No decoder is averaged with another latent family. The decoded RGB image remains
the boundary between stages. Source revisions, download hashes, licenses and
family aliases should accompany collected components in `vae/catalog.json`.
Adding a reserve decoder does not add a denoiser or make an unsupported model
architecture executable. Civitai ecosystem names alone do not establish latent
compatibility (for example, Illustrious XL and Illustrious Lumina are distinct).

Decoder verification can run without a denoiser or GPU using
`build/NativeVaeFallbackTests --decode-component <sd15|sd2|sd3|sdxl|anima|flux1|flux2> <weights.safetensors> <build/output-directory>`.
It checks the real tensor mounting contract and executes a 64x64 RGB native CPU
decode, rejecting missing tensors and nonfinite output. This is a decoder smoke
test, not an image-quality assessment of the complete cascade.

The DiT conversion regression explicitly selects `checkpoint_policy="strict"`
when asserting that different template hashes are rejected. The default
common-layer policy is an experimental projection and does not impose that
strict metadata identity contract.
