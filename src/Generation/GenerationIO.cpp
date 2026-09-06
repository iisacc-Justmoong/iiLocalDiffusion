#include "GenerationIO.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

namespace iild
{

GenerationIOError::GenerationIOError(const GenerationIOErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code)
{
}

GenerationIOErrorCode GenerationIOError::code() const noexcept { return code_; }

namespace
{

[[noreturn]] void fail(const GenerationIOErrorCode code, const std::string &message)
{
    throw GenerationIOError(code, message);
}

void identifier(const std::string &value, const std::string &field)
{
    if (value.empty() || value.find('\0') != std::string::npos ||
        value.find_first_not_of(" \t\r\n") == std::string::npos)
        fail(GenerationIOErrorCode::invalidContract, field + " must be a nonempty identifier without NUL");
}

bool floating(const TensorDType dtype)
{
    return dtype == TensorDType::float16 || dtype == TensorDType::bfloat16 ||
           dtype == TensorDType::float32 || dtype == TensorDType::float64;
}

std::size_t byteWidth(const TensorDType dtype)
{
    switch (dtype)
    {
    case TensorDType::float16: case TensorDType::bfloat16: return 2;
    case TensorDType::float32: case TensorDType::int32: return 4;
    case TensorDType::float64: case TensorDType::int64: return 8;
    case TensorDType::uint8: return 1;
    }
    fail(GenerationIOErrorCode::invalidContract, "unsupported dtype");
}

std::size_t elementCount(const GenerationTensorSpec &spec, const bool concrete,
                         const GenerationIOErrorCode code)
{
    std::size_t count = 1;
    for (const auto dimension : spec.shape)
    {
        if (dimension < 0)
        {
            if (dimension != -1) fail(code, "shape dimension must be nonnegative or -1 in a contract");
            if (concrete) fail(code, "payload shape must be concrete; -1 is only a contract wildcard");
        }
    }
    // Empty prefixes/batches are valid, but still require all dimensions to be legal.
    if (std::find(spec.shape.begin(), spec.shape.end(), 0) != spec.shape.end()) return 0;
    for (const auto dimension : spec.shape)
    {
        if (dimension == -1) continue;
        const auto extent = static_cast<std::uint64_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() ||
            count > std::numeric_limits<std::size_t>::max() / extent)
            fail(code, "tensor element count overflow");
        count *= static_cast<std::size_t>(extent);
    }
    if (count > std::numeric_limits<std::size_t>::max() / byteWidth(spec.dtype))
        fail(code, "tensor byte size overflow");
    return count;
}

void validateSpec(const GenerationTensorSpec &spec, const bool concrete = false)
{
    static_cast<void>(tensorDTypeName(spec.dtype));
    static_cast<void>(generationSemanticName(spec.semantic));
    identifier(spec.layout, "layout");
    identifier(spec.representationSpace, "representationSpace");
    if (spec.representationSpace == "auto" || spec.representationSpace == "unspecified" ||
        spec.representationSpace == "unknown")
        fail(GenerationIOErrorCode::invalidContract,
             "representationSpace requires an explicit identity; auto, unspecified and unknown are reserved");
    if (spec.semantic == GenerationSemantic::tokenIds &&
        spec.dtype != TensorDType::int32 && spec.dtype != TensorDType::int64)
        fail(GenerationIOErrorCode::invalidContract, "tokenIds require int32 or int64 dtype");
    switch (spec.semantic)
    {
    case GenerationSemantic::sample: case GenerationSemantic::epsilon:
    case GenerationSemantic::vPrediction: case GenerationSemantic::velocity:
    case GenerationSemantic::logits: case GenerationSemantic::embeddings:
        if (!floating(spec.dtype))
            fail(GenerationIOErrorCode::invalidContract,
                 std::string(generationSemanticName(spec.semantic)) + " requires a floating dtype");
        break;
    case GenerationSemantic::tokenIds: case GenerationSemantic::pixels:
    case GenerationSemantic::timestep: case GenerationSemantic::attentionMask: break;
    }
    static_cast<void>(elementCount(spec, concrete,
        concrete ? GenerationIOErrorCode::invalidTensor : GenerationIOErrorCode::invalidContract));
}

void compatible(const GenerationTensorSpec &expected, const GenerationTensorSpec &actual,
                const GenerationIOErrorCode code, const std::string &context)
{
    const auto mismatch = [&](const char *field) { fail(code, context + ": " + field + " mismatch"); };
    if (expected.dtype != actual.dtype) mismatch("dtype");
    if (expected.layout != actual.layout) mismatch("layout");
    if (expected.semantic != actual.semantic) mismatch("semantic");
    if (expected.representationSpace != actual.representationSpace) mismatch("representationSpace");
    if (expected.shape.size() != actual.shape.size()) mismatch("shape rank");
    for (std::size_t index = 0; index < expected.shape.size(); ++index)
    {
        if (expected.shape[index] != -1 && actual.shape[index] != -1 &&
            expected.shape[index] != actual.shape[index]) mismatch("shape");
    }
}

void validatePayload(const GenerationTensor &tensor)
{
    const auto expected = elementCount(tensor.spec, true, GenerationIOErrorCode::invalidTensor);
    std::visit([&](const auto &values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        bool matches = false;
        if constexpr (std::is_same_v<Value, float>) matches = tensor.spec.dtype == TensorDType::float32;
        else if constexpr (std::is_same_v<Value, double>) matches = tensor.spec.dtype == TensorDType::float64;
        else if constexpr (std::is_same_v<Value, std::int32_t>) matches = tensor.spec.dtype == TensorDType::int32;
        else if constexpr (std::is_same_v<Value, std::int64_t>) matches = tensor.spec.dtype == TensorDType::int64;
        else if constexpr (std::is_same_v<Value, std::uint8_t>) matches = tensor.spec.dtype == TensorDType::uint8;
        else if constexpr (std::is_same_v<Value, std::uint16_t>)
            matches = tensor.spec.dtype == TensorDType::float16 || tensor.spec.dtype == TensorDType::bfloat16;
        if (!matches) fail(GenerationIOErrorCode::invalidTensor, "tensor storage does not match declared dtype");
        if (values.size() != expected)
            fail(GenerationIOErrorCode::invalidTensor, "tensor element count does not match concrete shape");
        for (const auto element : values)
        {
            if constexpr (std::is_floating_point_v<Value>)
            {
                if (!std::isfinite(element))
                    fail(GenerationIOErrorCode::invalidTensor, "tensor values must be finite");
            }
            else if constexpr (std::is_same_v<Value, std::uint16_t>)
            {
                const auto exponentMask = tensor.spec.dtype == TensorDType::float16 ? 0x7c00U : 0x7f80U;
                if ((element & exponentMask) == exponentMask)
                    fail(GenerationIOErrorCode::invalidTensor, "half precision tensor values must be finite");
            }
            else if constexpr (std::is_signed_v<Value>)
            {
                if (tensor.spec.semantic == GenerationSemantic::tokenIds && element < 0)
                    fail(GenerationIOErrorCode::invalidTensor, "tokenIds must be nonnegative integers");
            }
        }
    }, tensor.data);
}

using Ports = std::map<std::string, const GenerationTensorSpec *, std::less<>>;
using StagePorts = std::map<std::string, Ports, std::less<>>;

Ports validatePorts(const std::vector<GenerationPort> &ports, const std::string &context)
{
    Ports result;
    for (const auto &port : ports)
    {
        identifier(port.name, context + " port name");
        validateSpec(port.tensor);
        if (!result.emplace(port.name, &port.tensor).second)
            fail(GenerationIOErrorCode::invalidContract, context + ": duplicate port " + port.name);
    }
    return result;
}

const GenerationTensorSpec &sourceSpec(const GenerationSource &source, const Ports &external,
                                       const StagePorts &stages)
{
    identifier(source.port, "source port");
    const Ports *ports = &external;
    if (!source.stage.empty())
    {
        const auto stage = stages.find(source.stage);
        if (stage == stages.end())
            fail(GenerationIOErrorCode::invalidContract, "source must reference an earlier stage: " + source.stage);
        ports = &stage->second;
    }
    const auto port = ports->find(source.port);
    if (port == ports->end())
        fail(GenerationIOErrorCode::invalidContract, "source port does not exist: " + source.stage + "/" + source.port);
    return *port->second;
}

void validateValues(const std::vector<GenerationPort> &ports, const GenerationValues &values,
                    const std::string &context)
{
    for (const auto &[name, tensor] : values)
    {
        static_cast<void>(tensor);
        if (std::none_of(ports.begin(), ports.end(), [&](const auto &port) { return port.name == name; }))
            fail(GenerationIOErrorCode::invalidTensor, context + ": unexpected port " + name);
    }
    for (const auto &port : ports)
    {
        const auto value = values.find(port.name);
        if (value == values.end())
            fail(GenerationIOErrorCode::invalidTensor, context + ": missing port " + port.name);
        try { validateGenerationTensor(port.tensor, value->second); }
        catch (const GenerationIOError &error)
        {
            fail(error.code(), context + "/" + port.name + ": " + error.what());
        }
    }
}

} // namespace

std::string_view generationArchitectureName(const GenerationArchitecture architecture)
{
    switch (architecture)
    {
    case GenerationArchitecture::diffusion: return "diffusion";
    case GenerationArchitecture::rectifiedFlow: return "rectified-flow";
    case GenerationArchitecture::flowMatching: return "flow-matching";
    case GenerationArchitecture::autoregressive: return "autoregressive";
    }
    fail(GenerationIOErrorCode::invalidContract, "unsupported generation architecture");
}

std::string_view generationSemanticName(const GenerationSemantic semantic)
{
    switch (semantic)
    {
    case GenerationSemantic::sample: return "sample";
    case GenerationSemantic::epsilon: return "epsilon";
    case GenerationSemantic::vPrediction: return "v-prediction";
    case GenerationSemantic::velocity: return "velocity";
    case GenerationSemantic::logits: return "logits";
    case GenerationSemantic::tokenIds: return "token-ids";
    case GenerationSemantic::embeddings: return "embeddings";
    case GenerationSemantic::pixels: return "pixels";
    case GenerationSemantic::timestep: return "timestep";
    case GenerationSemantic::attentionMask: return "attention-mask";
    }
    fail(GenerationIOErrorCode::invalidContract, "unsupported generation semantic");
}

std::string_view tensorDTypeName(const TensorDType dtype)
{
    switch (dtype)
    {
    case TensorDType::float16: return "float16";
    case TensorDType::bfloat16: return "bfloat16";
    case TensorDType::float32: return "float32";
    case TensorDType::float64: return "float64";
    case TensorDType::int32: return "int32";
    case TensorDType::int64: return "int64";
    case TensorDType::uint8: return "uint8";
    }
    fail(GenerationIOErrorCode::invalidContract, "unsupported dtype");
}

void validateGenerationTensor(const GenerationTensorSpec &contract, const GenerationTensor &tensor)
{
    validateSpec(contract);
    validateSpec(tensor.spec, true);
    compatible(contract, tensor.spec, GenerationIOErrorCode::invalidTensor, "tensor");
    validatePayload(tensor);
}

void validateGenerationPlan(const GenerationPlan &plan)
{
    const auto external = validatePorts(plan.inputs, "plan input");
    if (plan.stages.empty()) fail(GenerationIOErrorCode::invalidContract, "plan requires at least one stage");
    if (plan.outputs.empty()) fail(GenerationIOErrorCode::invalidContract, "plan requires at least one output");
    StagePorts completed;
    for (const auto &stage : plan.stages)
    {
        identifier(stage.id, "stage id");
        if (completed.contains(stage.id))
            fail(GenerationIOErrorCode::invalidContract, "duplicate stage id: " + stage.id);
        switch (stage.role)
        {
        case GenerationStageRole::model:
            if (!stage.architecture)
                fail(GenerationIOErrorCode::invalidContract, "model stage requires explicit architecture: " + stage.id);
            static_cast<void>(generationArchitectureName(*stage.architecture));
            break;
        case GenerationStageRole::bridge:
            if (stage.architecture)
                fail(GenerationIOErrorCode::invalidContract, "bridge stage must not declare a model architecture: " + stage.id);
            break;
        default: fail(GenerationIOErrorCode::invalidContract, "unsupported generation stage role");
        }
        const auto inputPorts = validatePorts(stage.inputs, stage.id + " input");
        const auto outputPorts = validatePorts(stage.outputs, stage.id + " output");
        if (outputPorts.empty())
            fail(GenerationIOErrorCode::invalidContract, "stage requires at least one output: " + stage.id);
        std::set<std::string, std::less<>> bound;
        for (const auto &binding : stage.bindings)
        {
            const auto input = inputPorts.find(binding.input);
            if (input == inputPorts.end())
                fail(GenerationIOErrorCode::invalidContract, stage.id + ": binding targets unknown input " + binding.input);
            if (!bound.insert(binding.input).second)
                fail(GenerationIOErrorCode::invalidContract, stage.id + ": duplicate binding " + binding.input);
            compatible(*input->second, sourceSpec(binding.source, external, completed),
                       GenerationIOErrorCode::incompatibleBinding, stage.id + "/" + binding.input);
        }
        if (bound.size() != inputPorts.size())
            fail(GenerationIOErrorCode::invalidContract, stage.id + ": every input requires exactly one binding");
        completed.emplace(stage.id, outputPorts);
    }
    std::set<std::string, std::less<>> outputNames;
    for (const auto &output : plan.outputs)
    {
        identifier(output.name, "plan output name");
        if (!outputNames.insert(output.name).second)
            fail(GenerationIOErrorCode::invalidContract, "duplicate plan output: " + output.name);
        validateSpec(output.tensor);
        compatible(output.tensor, sourceSpec(output.source, external, completed),
                   GenerationIOErrorCode::incompatibleBinding, "plan output/" + output.name);
    }
}

GenerationValues executeGenerationPlan(const GenerationPlan &plan, const GenerationValues &inputs,
                                       const GenerationExecutor &executor)
{
    validateGenerationPlan(plan);
    validateValues(plan.inputs, inputs, "plan input");
    if (!executor) fail(GenerationIOErrorCode::invalidContract, "generation executor is required");
    std::map<std::string, GenerationValues, std::less<>> completed;
    const auto resolve = [&](const GenerationSource &source) -> const GenerationTensor & {
        return source.stage.empty() ? inputs.at(source.port) : completed.at(source.stage).at(source.port);
    };
    for (const auto &stage : plan.stages)
    {
        GenerationValues stageInputs;
        for (const auto &binding : stage.bindings) stageInputs.emplace(binding.input, resolve(binding.source));
        // Dynamic source dimensions can satisfy the static plan but fail a concrete consumer.
        validateValues(stage.inputs, stageInputs, stage.id + " input");
        GenerationValues stageOutputs;
        try { stageOutputs = executor(stage, stageInputs); }
        catch (const std::exception &error)
        {
            fail(GenerationIOErrorCode::executionFailed, "stage " + stage.id + " execution failed: " + error.what());
        }
        catch (...)
        {
            fail(GenerationIOErrorCode::executionFailed, "stage " + stage.id + " execution failed with a nonstandard exception");
        }
        validateValues(stage.outputs, stageOutputs, stage.id + " output");
        completed.emplace(stage.id, std::move(stageOutputs));
    }
    GenerationValues result;
    for (const auto &output : plan.outputs)
    {
        const auto &tensor = resolve(output.source);
        try { validateGenerationTensor(output.tensor, tensor); }
        catch (const GenerationIOError &error)
        { fail(error.code(), "plan output/" + output.name + ": " + error.what()); }
        result.emplace(output.name, tensor);
    }
    return result;
}

} // namespace iild
