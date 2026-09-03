#pragma once

#include "Export.hpp"

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace iild
{

struct CoreMLCapabilities
{
    bool runtime = false;
    bool deviceDiscovery = false;
    bool computePlan = false;
    bool neuralEngine = false;
    std::size_t neuralEngineCores = 0;
};

enum class CoreMLComputeUnits
{
    cpuAndNeuralEngine,
    all,
    cpuOnly
};

struct CoreMLOptions
{
    CoreMLComputeUnits computeUnits = CoreMLComputeUnits::cpuAndNeuralEngine;
    // A scheduling estimate, NOT a hardware execution counter.
    bool requireNeuralEnginePlan = true;
};

enum class CoreMLScalarType
{
    float32,
    float16
};

struct CoreMLFeature
{
    std::string name;
    std::vector<std::size_t> shape;
    std::size_t elementCount = 0;
    CoreMLScalarType scalarType = CoreMLScalarType::float32;
};

struct CoreMLPlanOperation
{
    std::string name;
    std::string operation;
    std::string preferredDevice;
    std::vector<std::string> supportedDevices;
};

struct CoreMLModelInfo
{
    CoreMLComputeUnits computeUnits = CoreMLComputeUnits::cpuAndNeuralEngine;
    std::vector<CoreMLFeature> inputs;
    std::vector<CoreMLFeature> outputs;
    std::vector<CoreMLPlanOperation> plan;
    std::size_t neuralEnginePreferredOperations = 0;
    bool planAvailable = false;
    // Remains false: Core ML's public compute plan is anticipated dispatch.
    bool hardwareUsageVerified = false;
};

[[nodiscard]] IILD_EXPORT CoreMLCapabilities coreMLCapabilities();
[[nodiscard]] IILD_EXPORT std::string_view coreMLComputeUnitsName(CoreMLComputeUnits units);

// Executes a caller-provided compiled Core ML component. Core ML owns its model,
// buffers, kernels, and scheduling; no backend types cross the public boundary.
// Inputs/outputs use the model's declared shape in row-major float32 host storage.
// Core ML converts storage for components whose interface uses float16.
class IILD_EXPORT CoreMLModel final
{
public:
    using Features = std::map<std::string, std::vector<float>>;

    [[nodiscard]] static CoreMLModel load(
        const std::filesystem::path &compiledModel, CoreMLOptions options = {});

    ~CoreMLModel();
    CoreMLModel(CoreMLModel &&) noexcept;
    CoreMLModel &operator=(CoreMLModel &&) noexcept;
    CoreMLModel(const CoreMLModel &) = delete;
    CoreMLModel &operator=(const CoreMLModel &) = delete;

    [[nodiscard]] Features predict(const Features &inputs) const;
    [[nodiscard]] const CoreMLModelInfo &info() const;

private:
    struct Impl;
    explicit CoreMLModel(std::unique_ptr<Impl> implementation);
    [[nodiscard]] Impl &implementation() const;
    std::unique_ptr<Impl> implementation_;
};

} // namespace iild
