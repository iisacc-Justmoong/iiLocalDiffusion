# Local model merging

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
```

`resolve_merge_request()` in `model_merge_options` exposes the same lightweight
validation and immutable resolved request. `merge_models()` returns a JSON-ready
report containing ordered source SHA-256 identities, material kinds, coefficients, runtime
versions, output hashes and tensor/buffer counts. The CLI prints this report.

## Formats and compatibility

Full checkpoints must all use the base's storage format: single files or
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
