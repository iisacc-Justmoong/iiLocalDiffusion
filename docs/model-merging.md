# Local model merging

Heterogeneous SD 1.5, SDXL, DiT and editing checkpoints can first be normalized
to one Tsubaki DiT tensor contract with `iild-convert`; see
[DiT-standard model conversion](model-conversion.md). Converted outputs that use
the same target template are ordinary compatible DiT checkpoints and can use the
weight modes below. This is distinct from `unified`, which preserves independent
architectures as an ordered image-space cascade.

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

`--checkpoint-policy common-layer` deliberately relaxes that contract. The base
checkpoint remains the output format and executable tensor inventory. For each
floating base tensor, the SDK first uses an exact source address and otherwise
selects the foreign tensor with the closest component role, block depth, tensor
kind and shape. INT8 values are converted to float and use an adjacent
`weight_scale` when it can be broadcast. The selected values are flattened,
cropped or zero-padded to the base shape before ordinary weighted arithmetic.
Nonfloating buffers and base tensors with no source of the same kind preserve the
base value. Foreign-only tensors are omitted because the output must remain a
loadable base-format checkpoint.

This policy is deterministic and designed to finish a user-requested experiment;
it is not architecture conversion, distillation, or evidence of semantic layer
equivalence. Image quality may be poor or nonsensical. Reports and embedded merge
metadata set `checkpoint_policy: common-layer`, `semantic_equivalence: false`, and
record exact/projected/preserved counts plus bounded mapping examples. Use
`--checkpoint-policy strict` when incompatibility must stop the operation.
Malformed or incomplete Safetensors, unreadable tensors, nonfinite inputs and
arithmetic overflow still fail: the permissive policy does not fabricate bytes
for a damaged file.

## Different architectures: unified model objects

`--mode unified --output NAME.iildmodel` builds one portable ZIP64 stored package
file containing `model_index.json`, source provenance, and independent checkpoint
members. Compression is deliberately disabled so large tensor members can be
streamed and addressed without an additional decompression representation.
It is an **ordered image-refinement cascade**, not a single-network weight
average or distillation. SD1/SD2/SDXL derivatives (including Illustrious, Pony,
NAI and Noob), FLUX/Krea and Anima can share an object without pretending that
their differently shaped weights, latent spaces or prediction conventions match.
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

### LoRA compatibility objects

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
In sum mode, only full-checkpoint weights must total at most `1`; their remainder
belongs to the base. LoRA strengths and difference weights may exceed `1`.
Boolean, negative, NaN/infinite and mismatched weight lists fail before tensor
loading; the checkpoint total is checked after material classification.
A zero weight still validates the input
model's structure and data.

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

# Explicitly run an unstable cross-architecture projection.
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

No checkpoint tensor key is silently skipped or remapped. Integer and boolean buffers must
have identical dtype, shape and values and are copied from the base. Floating
inputs support FP16, BF16, FP32 and FP64, including mixed input precision; output
keeps each base tensor's dtype. FP16/BF16 accumulate in FP32; FP64 inputs use FP64.
Input NaN/infinity and output overflow fail the merge. Quantized, complex and
FP8 tensor arithmetic is not supported.

Compatibility checks establish structural/configuration consistency, not visual
quality or that independently trained checkpoints share a useful weight space.
Explicit `modelspec.architecture`, `modelspec.implementation` and
`modelspec.prediction_type` header hints must agree within matching components
when present.
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
pairs, nonfinite values and unsupported tensors fail; no adapter key is ignored.
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

## Installed Python and compatibility registry

Installed launchers validate Python >= 3.10 before importing model modules. `IILD_PYTHON_EXECUTABLE` overrides `reference/runtime-python.json`, whose shape is `{"python":"/absolute/venv/bin/python"}`. Otherwise the bundled `.venv` or launcher interpreter is considered. Invalid or old environments produce actionable diagnostics instead of an annotation traceback.

Unified bridge discovery also reads installation-local `reference/merge-compatibility.json`: `{"checkpoints":["/absolute/full-checkpoint.safetensors"]}`. Relative paths resolve from that registry directory. The registry augments nearby checkpoint discovery, including the category containing single-checkpoint wrappers. Explicit compatibility candidates override discovery. All targets/shapes are validated; missing, ambiguous or incompatible bases still fail. Inspection never downloads models or executes weight arithmetic. Different networks compose as sequential image refinement; normalization does not make arbitrary network tensors interchangeable.

PythonLauncherTests verifies environment selection, explicit override, old-version rejection before entry import, and broken configuration. ModelMergeCompatibilityTests covers registered bridge discovery from wrapped inputs and output-free input inspection.

### LoRA alias normalization

LoRA projection pairs are normalized to the checkpoint's actual tensor address after shape, rank, alpha and orientation checks. Some exports include both SGM and Diffusers names for the same address. Identical projection tensors with identical dtype, scale and orientation are applied once. Distinct projection pairs are retained as additive deltas on that address, including each pair's own alpha/rank scale. No target is silently dropped or arbitrarily reshaped. Reports expose `lora_alias_policy` as `identical-projections-once; distinct-projections-additive`. Input inspection may read the duplicate LoRA projections to establish equality, but does not materialize full checkpoint tensors or produce a merged model.

The LoRA regression suite verifies duplicate aliases apply once, distinct projections sum correctly, and different alpha values preserve their individual scaling, using tiny fixture tensors.

## Synthetic cross-family LoRA adaptation

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
