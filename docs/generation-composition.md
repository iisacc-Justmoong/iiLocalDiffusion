# Generation architecture I/O and composition

iiLocalDiffusion provides three complementary interfaces:

* [Native C++ host tensor contracts](generation-io-native.md) validate and execute
  named stages through a caller-supplied backend callback.
* `reference/diffusers/generation_composition.py` connects already-loaded Python
  runtimes using typed host tensor and text ports. `generation_adapters.py`
  calls Diffusers pipelines, Transformers `generate()`, and tokenizer decoding.
* [Generic Diffusers file I/O](generic-diffusers.md#diffusion-flow-autoregressive-and-hybrid-interchange)
  saves/reloads safetensors descriptors alongside images, video, audio and text.

Diffusion, rectified flow, flow matching, and autoregressive generation are
explicit stage declarations. Multiple kinds form a hybrid pipeline. A model
with internal mixed generation can also declare `hybrid` in the Python API or
generic CLI; C++ represents its component stages. An architecture declaration
does not select a sampler, inspect training, convert weights, or certify a model.
Rectified flow and flow matching may share a flow scheduler while using different
training objectives. Diffusion `v_prediction` and a flow `velocity` are distinct
port semantics and cannot be wired together without an explicit conversion.

Existing model loading, device selection, and preset defaults are unchanged.
These APIs add no package dependency. Maintained, already-pinned Diffusers and
Transformers supply inference and generation; safetensors supplies typed storage;
NumPy/Torch supply validation and host copies. They avoid implementing model math,
tokenization or unsafe tensor deserialization in this project. Package versions,
licenses and deployment costs remain in [dependency decisions](dependencies.md).

## Python contract

`Port.tensor(dtype, shape, layout, semantic, representation_space)` describes one
value. `None` is an independent wildcard dimension; all concrete Python dimensions
must be positive. `Port.text()` accepts one string, and `Port.text(batch=True)` a
nonempty string batch. The native API additionally permits zero-size tensors and
uses `-1` wildcards; it has no Python object or JSON ABI dependency.

Tensor payloads must be ordinary NumPy arrays or CPU Torch tensors. Dtypes are
checked exactly, including BF16 and integer token IDs. No automatic floating
conversion of token IDs, latent normalization, transpose or reshape occurs.
Ports require matching dtype, rank, layout, semantic and representation space.
Dynamic producer dimensions are checked again against concrete consumer shapes.
Floating values must be finite and token IDs nonnegative. Vocabulary bounds and
cross-input relationships, such as attention-mask lengths, remain model checks.

Use a space identity that includes the actual tokenizer/encoder/VAE revision and
normalization convention. `auto`, `unspecified` and `unknown` are reserved and
cannot establish composition compatibility. The library cannot derive a valid
coordinate-system identity from a tensor shape or a caller-provided label.

`Stage` has a name, architecture, input/output port maps, bindings and executor.
`Source.input(name)` binds an external input; `Source(stage, port)` binds a named
output of an earlier stage. Every input needs exactly one binding. Missing,
extra, duplicate or forward connections fail before execution. Export maps can
retain multiple final or intermediate outputs. Callbacks must return exactly
their declared output map. `architecture="adapter"` identifies an explicit
decoder/encoder/conversion bridge and is excluded from the result's architecture
list. Bridge outputs receive the same value validation as model outputs.

The executor boundary owns snapshots of external inputs, callback inputs,
callback outputs and exported results. A callback's in-place writes cannot
modify another stage's retained values or the caller's original input. This
costs host memory/copies and is not a zero-copy device graph. Execution is
sequential; a failure stops later stages and does not undo backend side effects.

## Autoregressive to flow through an explicit text bridge

The following integration fragment assumes `language_model`, `tokenizer` and
`flow_pipeline` are already loaded, validated and placed by the application.
The caller must substitute the actual tokenizer space identity and correct flow
output shape/dtype for its selected model. No weights are downloaded by the
adapters. The language model must be in `eval()` mode.

```python
from generation_composition import Port, Source, Stage, GenerationPipeline
from generation_adapters import TransformersAdapter, TokenDecodeAdapter, DiffusersAdapter

tokens = Port.tensor("int64", (1, None), "BS", "token_ids", "my-tokenizer:revision")
texts = Port.text(batch=True)
pixels = Port.tensor("float32", (1, None, None, 3), "BHWC", "pixels", "rgb:0-1")

stages = [
    Stage("language", "autoregressive", {"input_ids": tokens}, {"tokens": tokens},
          {"input_ids": Source.input("tokens")},
          TransformersAdapter(language_model, {"tokens": "sequences"},
                              {"max_new_tokens": 32, "do_sample": False}, device="cpu")),
    Stage("decode", "adapter", {"tokens": tokens}, {"text": texts},
          {"tokens": Source("language", "tokens")}, TokenDecodeAdapter(tokenizer)),
    Stage("image", "rectified-flow", {"prompt": texts}, {"pixels": pixels},
          {"prompt": Source("decode", "text")},
          DiffusersAdapter(flow_pipeline, {"pixels": "images"},
                           {"output_type": "np", "num_inference_steps": 4}, device="mps")),
]
pipeline = GenerationPipeline({"tokens": tokens}, stages,
    {"tokens": Source("language", "tokens"), "pixels": Source("image", "pixels")})
result = pipeline.run({"tokens": input_ids_on_cpu})
assert result.hybrid
```

The same API connects diffusion→flow, flow→autoregressive, or interleaved blocks.
Supply real model-specific bridges where token, embedding or latent spaces differ.
An identity callback cannot make incompatible representations equivalent.

`DiffusersAdapter` requires explicit pipeline call parameters and a structured
result. Use `output_type="np"` for typed pixels or the pipeline's documented latent
output mode with a matching output port. Its `outputs` map selects exact result
fields. `TransformersAdapter` exports `sequences` without casting integer IDs and
can export raw per-step `logits`, stacking `[B,V]` steps into `[B,S,V]`. Logits are
not sampled token IDs. Beam-search sequences are supported; raw logits with
`num_beams != 1` are rejected because beam ancestry needs a separate adapter.
Model-specific KV caches and processed scores are not portable output ports.
Other multimodal fields need an explicit application adapter and contract.

Adapter `device` controls input placement only. It does not move the loaded model
or retry on CPU. Stage-bound arguments cannot duplicate fixed adapter parameters.
RNG, schedules, generation length and sampling defaults belong to the caller's
runtime configuration; adapters do not silently replace them.

## Reusing saved tensors

A generic output's `generation.json.outputs[].tensor_input` contains a path, exact
key, SHA-256, dtype, shape, semantic, layout and representation-space label.
`Port.from_tensor_descriptor(descriptor)` maps that concrete file descriptor into
a composition port. Load and authenticate the actual data separately with
`generic_io.load_tensor_input(descriptor, generate_any.file_identity)`; then pass
it to `GenerationPipeline.run()`. The loader checks exported file metadata against
the descriptor, so relabeling a VAE space cannot convert it. For automatic token
exports with `unspecified` space, declare the true tokenizer identity at export
using `--tensor-outputs` before attempting composition.

Generic `latents`, `embeddings`, `token-ids`, `v-prediction`, `continuous-state`
and `attention-mask` map to Python `latent`, `embedding`, `token_ids`,
`v_prediction`, `sample` and `mask`. The generic `tensor` semantic stays `tensor`.
C++ uses its documented enums and owned vectors; it does not parse this file
descriptor or expose backend tensors. Application adapters perform that boundary.

## Verification and upstream contracts

Run the lightweight contracts through CTest. For real, offline runtime checks:

```sh
PYTHONPYCACHEPREFIX=build/pycache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  reference/diffusers/.venv/bin/python tests/GenerationAdapterTests.py
PYTHONPYCACHEPREFIX=build/pycache \
  reference/diffusers/.venv/bin/python tests/GenericIOTests.py
```

Tests construct a tiny GPT-2 and DDPM from local configuration, exercise the
installed flow scheduler, verify explicit bridges and stop on invalid outputs.
They also cover mutation isolation, dynamic shape mismatches, raw-logit/beam
boundaries, BF16 and large integer safetensors round trips. These establish I/O
and small runtime execution, not large model quality or universal checkpoint
support. See also the recorded tiny FLUX latent round trip in the generic guide.

Official references: [Transformers generation output shapes](https://huggingface.co/docs/transformers/internal/generation_utils),
[FLUX pipeline input/output options](https://huggingface.co/docs/diffusers/api/pipelines/flux),
and [FlowMatchEuler scheduler contract](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete).
