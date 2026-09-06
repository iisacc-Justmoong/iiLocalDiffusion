#pragma once

#include "Export.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace iild
{

// Explicit declarations from the model/backend contract, never scheduler guesses.
enum class GenerationArchitecture { diffusion, rectifiedFlow, flowMatching, autoregressive };
enum class GenerationStageRole { model, bridge };
enum class TensorDType { float16, bfloat16, float32, float64, int32, int64, uint8 };
enum class GenerationSemantic
{
    sample, epsilon, vPrediction, velocity, logits, tokenIds,
    embeddings, pixels, timestep, attentionMask
};

enum class GenerationIOErrorCode { invalidContract, incompatibleBinding, invalidTensor, executionFailed };

class IILD_EXPORT GenerationIOError final : public std::runtime_error
{
public:
    GenerationIOError(GenerationIOErrorCode code, std::string message);
    [[nodiscard]] GenerationIOErrorCode code() const noexcept;

private:
    GenerationIOErrorCode code_;
};

[[nodiscard]] IILD_EXPORT std::string_view generationArchitectureName(GenerationArchitecture architecture);
[[nodiscard]] IILD_EXPORT std::string_view generationSemanticName(GenerationSemantic semantic);
[[nodiscard]] IILD_EXPORT std::string_view tensorDTypeName(TensorDType dtype);

struct GenerationTensorSpec
{
    TensorDType dtype = TensorDType::float32;
    // -1 is a wildcard in a port contract only. Concrete dimensions may be zero.
    // An empty shape denotes one scalar, not an empty tensor.
    std::vector<std::int64_t> shape;
    // Exact, opaque identifiers: e.g. NCHW, BSC, BS, scalar.
    std::string layout;
    GenerationSemantic semantic = GenerationSemantic::sample;
    // Identifies the VAE/tokenizer/embedding coordinate system, including scaling.
    // Every port must declare it; exact auto/unspecified/unknown sentinels are invalid.
    // Equal shape alone does not imply compatibility.
    std::string representationSpace;
};

// float16 and bfloat16 contain native uint16_t bit patterns, not integer samples.
using GenerationTensorData = std::variant<std::vector<float>, std::vector<double>,
    std::vector<std::int32_t>, std::vector<std::int64_t>, std::vector<std::uint8_t>,
    std::vector<std::uint16_t>>;

struct GenerationTensor
{
    GenerationTensorSpec spec;
    GenerationTensorData data;
};

using GenerationValues = std::map<std::string, GenerationTensor, std::less<>>;

struct GenerationPort
{
    std::string name;
    GenerationTensorSpec tensor;
};

struct GenerationSource
{
    // Empty stage refers to the plan's external inputs; otherwise an earlier stage.
    std::string stage;
    std::string port;
};

struct GenerationBinding
{
    std::string input;
    GenerationSource source;
};

struct GenerationStage
{
    std::string id;
    GenerationStageRole role = GenerationStageRole::model;
    // Required for model stages; absent for bridges (decoders, tokenizers, samplers).
    std::optional<GenerationArchitecture> architecture;
    std::vector<GenerationPort> inputs;
    std::vector<GenerationPort> outputs;
    std::vector<GenerationBinding> bindings;
};

struct GenerationOutput
{
    std::string name;
    GenerationSource source;
    GenerationTensorSpec tensor;
};

struct GenerationPlan
{
    std::vector<GenerationPort> inputs;
    // Ordered execution with explicit named dependencies, without implicit conversion.
    std::vector<GenerationStage> stages;
    std::vector<GenerationOutput> outputs;
};

// The supplied backend owns model evaluation, sampling, tokenization and bridges.
using GenerationExecutor = std::function<GenerationValues(const GenerationStage &, const GenerationValues &)>;

IILD_EXPORT void validateGenerationTensor(const GenerationTensorSpec &contract,
                                          const GenerationTensor &tensor);
IILD_EXPORT void validateGenerationPlan(const GenerationPlan &plan);
[[nodiscard]] IILD_EXPORT GenerationValues executeGenerationPlan(
    const GenerationPlan &plan, const GenerationValues &inputs, const GenerationExecutor &executor);

} // namespace iild
