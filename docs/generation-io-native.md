# Native generation I/O contracts

`Generation/GenerationIO.hpp` provides a C++20 host data boundary for diffusion,
rectified flow, flow matching, autoregressive generation, and ordered combinations
of those stages. It validates and routes **actual typed tensor data** through a
caller-supplied `GenerationExecutor`. It is installed with the shared library and
works without MLX, LibTorch, Python, or a downloaded model.

The API does not implement a model, tokenizer, numerical sampler, decoder, or a
backend tensor runtime. Existing Python generation remains a separate runtime.
Use an existing backend for those operations and map its input/output tensors to
the declared host ports. No additional third-party dependency is introduced for
this boundary: containers, checked dimensions, and execution callbacks use the
C++ standard library.

## Architecture and prediction semantics

Every `model` stage must explicitly declare one `GenerationArchitecture`:
`diffusion`, `rectifiedFlow`, `flowMatching`, or `autoregressive`. A `bridge` stage
has no architecture declaration. These declarations describe the caller's model
contract; the API does not infer a training objective from a scheduler, class
name, file extension, or tensor shape. Declaring an architecture is not proof that
a backend implements it. A complete hybrid model can instead be represented as
its explicitly ordered component stages.

Ports use `GenerationSemantic` to retain the meaning of the payload:

| Semantic | Payload meaning |
| --- | --- |
| `sample` | Continuous sample/state in the declared representation space |
| `epsilon` | Noise prediction |
| `vPrediction` | Diffusion v-parameterization prediction |
| `velocity` | Flow vector-field prediction |
| `logits` | Floating scores for the backend's declared vocabulary space |
| `tokenIds` | Nonnegative integer identifiers in the declared tokenizer space |
| `embeddings` | Floating vectors in the declared encoder space |
| `pixels` | Pixel tensor with declared layout and value-space identity |
| `timestep` | Backend-specific time or timestep values |
| `attentionMask` | Backend-specific mask values |

`vPrediction` and `velocity` remain distinct. A prediction is never silently
treated as a sample, and logits are never silently converted to tokens. The
backend owns normalization, token vocabulary bounds, masks, and scheduling.

## Tensor contract

Each `GenerationTensorSpec` requires exact dtype, rank/shape, layout, semantic,
and a nonempty `representationSpace`. The last field must identify the actual
VAE/tokenizer/encoder coordinate system, including such details as latent
scaling/shift or token vocabulary revision. For example, equal `[1,4,64,64]`
latent shapes from different VAEs do not permit direct wiring. Applications are
responsible for trustworthy space identifiers; the library cannot verify model
identity from the tensor bytes.

The exact, case-sensitive values `auto`, `unspecified`, and `unknown` are reserved
sentinels and are rejected as representation identities. Matching unknown spaces
cannot establish compatibility; supply the actual shared coordinate-system
identity before connecting a port or executing a plan.

`-1` is an independent wildcard dimension in a port contract. A concrete payload
must replace every wildcard with a nonnegative size. Zero dimensions permit empty
token prefixes and empty batches; an empty shape means one scalar. Layout is an
opaque, exact identifier such as `NCHW`, `BS`, `BSC`, or `scalar`; the library
does not transpose or infer axis meanings. Wildcards do not create symbolic
equalities across separate axes or ports. Backends must validate any additional
relationships, such as matching token and mask sequence lengths.

The supported dtypes and storage alternatives are:

| Dtype | `GenerationTensorData` storage |
| --- | --- |
| `float16`, `bfloat16` | `std::vector<std::uint16_t>` containing native 16-bit bit patterns |
| `float32` | `std::vector<float>` |
| `float64` | `std::vector<double>` |
| `int32` | `std::vector<std::int32_t>` |
| `int64` | `std::vector<std::int64_t>` |
| `uint8` | `std::vector<std::uint8_t>` |

The validator checks element count, element-count and byte-size overflow, storage
type, concrete shape, and finite floating values, including half-precision NaN
and infinity bit patterns. Token identifiers require `int32` or `int64` storage
and cannot be negative. Continuous states, predictions, logits, and embeddings
require floating dtype. Other dtypes require an explicit backend conversion
before entering this host boundary. No implicit cast, reshape, scaling, or
precision conversion occurs.

## Named composition and execution

`GenerationPlan.inputs` declares every external input. Each stage input has
exactly one `GenerationBinding`; an empty source stage references an external
input, and another source stage must appear earlier in the plan. Duplicate
names, missing inputs, missing source ports, forward references, and cycles fail
before any executor call. Sources can fan out to later consumers. Named plan
outputs can expose final or intermediate tensors.

Direct bindings require compatible shape, dtype, layout, semantic, and
representation space. To connect tokens to a flow model, declare an
autoregressive stage, a bridge with token inputs and latent/embedding outputs,
then the flow stage. The bridge executor must run the real decoder/encoder or
other conversion and return the promised tensor. Merely labeling a bridge does
not disable validation of its inputs or outputs. Model stage contracts may
naturally differ between their own inputs and outputs, such as a denoiser's
sample input and epsilon output.

`executeGenerationPlan` validates the full static plan and external payloads,
then validates each stage's concrete inputs before execution and every returned
output before forwarding it. A wildcard source may be statically compatible
with a fixed consumer; the actual produced shape must satisfy that consumer at
runtime. Extra callback outputs are rejected just like missing outputs. A
callback exception stops the plan and is reported with the stage identifier;
previously executed external backend side effects are not rolled back.

The host API owns vectors and copies payloads when routing named inputs and
exporting results. It is a portable correctness boundary, not a zero-copy device
graph or a replacement for backend-native tensor storage.

```cpp
#include <Generation/GenerationIO.hpp>

const iild::GenerationTensorSpec state{
    iild::TensorDType::float32, {1, 2}, "BC",
    iild::GenerationSemantic::sample, "vae:example-v1"};
auto prediction = state;
prediction.semantic = iild::GenerationSemantic::epsilon;

const iild::GenerationPlan plan{
    {{"state", state}},
    {{"denoiser", iild::GenerationStageRole::model,
      iild::GenerationArchitecture::diffusion,
      {{"sample", state}}, {{"prediction", prediction}},
      {{"sample", {"", "state"}}}}},
    {{"prediction", {"denoiser", "prediction"}, prediction}}};

// Implement this using your actual inference backend. The callback must return
// a map with exactly the stage's declared output names and concrete tensor specs.
iild::GenerationExecutor backend = myBackendExecutor;
const auto result = iild::executeGenerationPlan(
    plan, {{"state", {state, std::vector<float>{0.1F, 0.2F}}}}, backend);
```

`GenerationIOError` exposes `invalidContract`, `incompatibleBinding`,
`invalidTensor`, and `executionFailed` error codes. `GenerationIOTests` exercises
all four architecture declarations, an actual token-to-latent bridge followed by
flow matching, dynamic and empty inputs, prediction distinctions, dtype/storage
errors, finite-value checks, overflow, invalid graphs, and executor failures.
`InstallConsumerTests` separately compiles, links, and runs the public API from a
relocated installation prefix.
